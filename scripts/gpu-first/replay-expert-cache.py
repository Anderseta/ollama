#!/usr/bin/env python3
"""Replay optional WildLlama expert traces without loading model weights.

Models a unified cache of complete gate/up/down expert bundles. Native layout
banks, compute, CUDA synchronization and disk latency are deliberately not
simulated. The result is a byte-traffic estimate, never a tokens/s prediction.
"""
from collections import Counter, OrderedDict
import argparse
import json
from pathlib import Path
import sys


def integer(value, name, minimum=0, maximum=(1 << 64) - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")
    return value


def read_trace(path, allow_incomplete=False):
    """Validate and stream records; malformed JSON is never silently skipped."""
    header = end = None
    count = sequence = 0
    with Path(path).open(encoding="utf8") as stream:
        for number, line in enumerate(iter(lambda: stream.readline(1024 * 1024 + 1), ""), 1):
            if len(line) > 1024 * 1024:
                raise ValueError(f"Oversized trace record at line {number}")
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Malformed trace record at line {number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at line {number}")
            if end is not None:
                raise ValueError("Data follows the end record")
            if header is None:
                if record != {"kind": "header", "version": 1, "unit": "expert_bundle", "scope": "runner_shared"}:
                    raise ValueError("Unsupported trace header")
                header = record
                continue
            if record.get("kind") == "end":
                end = record
                integer(end.get("routes"), "end route count")
                if end["routes"] != count:
                    raise ValueError("End record disagrees with recorded route count")
                if not isinstance(end.get("truncated"), bool):
                    raise ValueError("Invalid truncation flag")
                if (end["truncated"] or end.get("io_error")) and not allow_incomplete:
                    raise ValueError("Trace was truncated; use --allow-incomplete for a partial estimate")
                if not end["truncated"] and sequence != count:
                    raise ValueError("Complete trace has missing sequence records")
                continue
            if record.get("kind") != "route":
                raise ValueError(f"Unknown record at line {number}")
            current = integer(record.get("seq"), "sequence", 1)
            if current <= sequence:
                raise ValueError("Trace sequence is not strictly increasing")
            sequence = current
            for key in ("model", "eval"):
                integer(record.get(key), key)
            integer(record.get("layer"), "layer", maximum=(1 << 31) - 1)
            integer(record.get("bytes"), "expert bytes", 1)
            tokens = integer(record.get("tokens"), "tokens", 1, 65536)
            k = integer(record.get("k"), "k", 1, 65536)
            experts = record.get("experts")
            if not isinstance(experts, list) or len(experts) > 65536 or len(experts) != tokens * k:
                raise ValueError("Expert count differs from tokens times k")
            for expert in experts:
                integer(expert, "expert", maximum=(1 << 31) - 1)
            count += 1
            yield record
    if header is None:
        raise ValueError("Empty trace")
    if end is None and not allow_incomplete:
        raise ValueError("Trace has no end record; use --allow-incomplete after an interrupted runner")


class ExpertCache:
    """Byte-bounded LRU/SLRU with optional decaying frequency admission."""
    def __init__(self, capacity, policy):
        if policy not in {"lru", "slru", "admission"}:
            raise ValueError("Unknown cache policy")
        integer(capacity, "cache capacity")
        self.capacity, self.policy = capacity, policy
        self.probation, self.protected = OrderedDict(), OrderedDict()
        self.used = self.protected_bytes = 0
        self.frequency = Counter()
        self.accesses = self.hits = self.misses = self.hit_bytes = self.miss_bytes = 0
        self.bypasses = self.evictions = 0
        self.frequency_observations = 0

    def __contains__(self, key):
        return key in self.probation or key in self.protected

    def remove(self, key):
        if key in self.protected:
            size = self.protected.pop(key)
            self.protected_bytes -= size
        else:
            size = self.probation.pop(key, 0)
        self.used -= size
        return size

    def touch(self, key):
        if self.policy == "lru":
            self.probation.move_to_end(key)
            return
        if key in self.protected:
            self.protected.move_to_end(key)
        else:
            size = self.probation.pop(key)
            self.protected[key] = size
            self.protected_bytes += size
        while self.protected_bytes > self.capacity // 2 and self.protected:
            old, size = self.protected.popitem(last=False)
            self.protected_bytes -= size
            self.probation[old] = size

    def observe(self, requests):
        if self.policy != "admission":
            return
        self.frequency.update(requests)
        self.frequency_observations += sum(requests.values())
        if self.frequency_observations >= 65536:
            self.frequency = Counter({key: count // 2 for key, count in self.frequency.items() if count >= 2})
            self.frequency_observations = 0

    def admit(self, key, size, protected_keys):
        if size > self.capacity:
            self.bypasses += 1
            return False
        candidates, available = [], self.capacity - self.used
        for segment in (self.probation, self.protected):
            for victim, victim_size in segment.items():
                if available >= size:
                    break
                if victim in protected_keys:
                    continue
                candidates.append(victim)
                available += victim_size
            if available >= size:
                break
        if available < size:
            self.bypasses += 1
            return False
        if self.policy == "admission" and candidates:
            # Weights have a byte-proportional miss cost, so reuse frequency is
            # the saved-transfer benefit per resident byte. Ties bypass scans.
            candidate_score = self.frequency[key]
            victim_score = sum(self.frequency[victim] * self.size(victim) for victim in candidates) / sum(self.size(victim) for victim in candidates)
            if candidate_score <= victim_score:
                self.bypasses += 1
                return False
        for victim in candidates:
            self.remove(victim)
            self.evictions += 1
        self.probation[key] = size
        self.used += size
        return True

    def size(self, key):
        return self.protected.get(key, self.probation.get(key, 0))

    def access_batch(self, requests, size):
        """One demand per unique expert; repeated prompt rows add reuse weight."""
        self.observe(requests)
        already_present = {key for key in requests if key in self}
        missing = []
        for key in requests:
            self.accesses += 1
            if key in already_present:
                if self.size(key) != size:
                    raise ValueError("A cached expert changed its byte size")
                self.hits += 1
                self.hit_bytes += size
                self.touch(key)
            else:
                self.misses += 1
                self.miss_bytes += size
                missing.append(key)
        protected_keys = set(already_present)
        for key in missing:
            if self.admit(key, size, protected_keys):
                protected_keys.add(key)
        assert 0 <= self.used <= self.capacity
        return missing

    def summary(self):
        demand = self.hit_bytes + self.miss_bytes
        return {"capacity_bytes": self.capacity, "resident_bytes": self.used,
                "requests": self.accesses, "hits": self.hits, "misses": self.misses,
                "hit_bytes": self.hit_bytes, "miss_bytes": self.miss_bytes,
                "byte_hit_ratio": self.hit_bytes / demand if demand else 0,
                "evictions": self.evictions, "bypasses": self.bypasses}


class Replay:
    def __init__(self, vram_bytes, ram_bytes, policy):
        self.gpu = ExpertCache(vram_bytes, policy)
        self.ram = ExpertCache(ram_bytes, policy)
        self.policy = policy
        self.routes = self.ssd_bytes = self.h2d_bytes = 0
        self.layer_cost = {}
        self.layer_stats = {}

    def route(self, record):
        model, layer, size = record["model"], record["layer"], record["bytes"]
        identity = (model, layer)
        if self.layer_cost.setdefault(identity, size) != size:
            raise ValueError("Layer expert byte size changed within one model identity")
        requests = Counter((model, layer, expert) for expert in record["experts"])
        missing_gpu = self.gpu.access_batch(requests, size)
        missing_ram = self.ram.access_batch(Counter({key: requests[key] for key in missing_gpu}), size)
        self.h2d_bytes += len(missing_gpu) * size
        self.ssd_bytes += len(missing_ram) * size
        layer_stats = self.layer_stats.setdefault(identity, {"routes": 0, "h2d_bytes": 0, "ssd_bytes": 0})
        layer_stats["routes"] += 1
        layer_stats["h2d_bytes"] += len(missing_gpu) * size
        layer_stats["ssd_bytes"] += len(missing_ram) * size
        self.routes += 1

    def summary(self):
        return {"policy": self.policy, "routes": self.routes,
                "h2d_bytes": self.h2d_bytes, "ssd_bytes": self.ssd_bytes,
                "gpu": self.gpu.summary(), "ram": self.ram.summary(),
                "layers": [{"model": model, "layer": layer, **value}
                           for (model, layer), value in sorted(self.layer_stats.items())]}


def replay(path, vram_mib, ram_mib, policies, allow_incomplete=False):
    simulations = [Replay(vram * 1024**2, ram_mib * 1024**2, policy)
                   for vram in vram_mib for policy in policies]
    for record in read_trace(path, allow_incomplete):
        for simulation in simulations:
            simulation.route(record)
    return {"version": 1, "partial_allowed": allow_incomplete,
            "scope": "unified_quantized_expert_bundles_inclusive_ram_gpu",
            "limitations": ["No estimate of tokens/s, disk latency, compute or synchronization.",
                            "Native per-layout banks and row tiles are not represented.",
                            "Repeated experts within one route are transferred once.",
                            "Trace evaluations are not necessarily generated tokens.",
                            "Results use an initially empty cache and the recorded shared-runner order."],
            "results": [simulation.summary() for simulation in simulations]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--vram-mib", type=int, nargs="+", default=[1024, 2048, 4096, 6144])
    parser.add_argument("--ram-mib", type=int, default=6144)
    parser.add_argument("--policies", nargs="+", choices=["lru", "slru", "admission"], default=["lru", "slru", "admission"])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ram_mib < 0 or any(value < 0 for value in args.vram_mib):
        parser.error("Memory budgets must be nonnegative")
    if args.allow_incomplete:
        print("Warning: partial traces can bias cache estimates; no throughput inference is valid.", file=sys.stderr)
    try:
        report = replay(args.trace, args.vram_mib, args.ram_mib, args.policies, args.allow_incomplete)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Trace replay failed: {error}\n")
    encoded = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
