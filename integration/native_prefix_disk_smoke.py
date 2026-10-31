"""Compare synchronous/async cold prefix restores on one tiny CPU-only payload.

Uses an existing local GGUF (<=64 MiB), never downloads a model, and proves an
actual backing-file restore rather than accepting a RAM-only cache hit.
"""
import sys
sys.dont_write_bytecode = True
if not __debug__:
    raise RuntimeError("Run without -O; all fixture and equivalence checks are required")

import argparse
import json
import math
import os
from pathlib import Path
import re
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_execution_features_smoke import NativeServer, comparator, require, sha


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_arg(command, name, value):
    command[command.index(name) + 1] = str(value)


def state_files(directory):
    files = list(directory.rglob("*.state"))
    require(all(p.is_file() and p.stem.isdecimal() and int(p.stem) > 0 for p in files),
            "Unexpected state-file layout")
    return sorted(files, key=lambda p: (str(p.parent), int(p.stem)))


def stable(response, expected_prompt_tokens, eos=False):
    mandatory = {"content", "tokens", "stop", "stop_type", "stopping_word", "truncated",
                 "tokens_predicted", "tokens_evaluated", "generation_settings", "completion_probabilities"}
    require(type(response) is dict and mandatory <= response.keys(), "Incomplete completion schema")
    require(type(response["tokens"]) is list and response["tokens"] and
            all(type(t) is int and t >= 0 for t in response["tokens"]), "Missing generated token IDs")
    require(type(response["content"]) is str and response["stop"] is True and
            response["truncated"] is False and response["stopping_word"] == "", "Unexpected stop/truncation")
    require(response["tokens_evaluated"] == expected_prompt_tokens, "Logical prompt length changed")
    require(response["stop_type"] == ("eos" if eos else "limit"), "Unexpected EOS/limit behavior")
    if eos:
        require(response["content"] == "OK", "Forced EOS grammar changed content")
    else:
        require(len(response["tokens"]) == 8 and response["tokens_predicted"] == 8, "Output limit changed")
    settings = response["generation_settings"]
    require(settings.get("temperature") == 0 and settings.get("n_probs") == 5 and
            settings.get("backend_sampling") is False and settings.get("ignore_eos") is (not eos) and
            settings.get("n_predict") == 8 and
            settings.get("post_sampling_probs") is False, "Logical sampling settings changed")
    probabilities = response["completion_probabilities"]
    require(type(probabilities) is list and probabilities, "Missing pre-sampling probabilities")
    count = 0
    for item in probabilities:
        require(type(item) is dict and type(item.get("top_logprobs")) is list and item["top_logprobs"],
                "Missing top-logprob candidates")
        for candidate in [item] + item["top_logprobs"]:
            value = candidate.get("logprob")
            require(type(value) in (int, float) and math.isfinite(value) and value <= 0,
                    "Invalid logprob value")
            require(type(candidate.get("id")) is int and candidate["id"] >= 0, "Invalid probability token ID")
            count += 1
    # Only physical cache/timing counters differ; all logical fields remain exact.
    return {k: v for k, v in response.items() if k not in ("timings", "tokens_cached")}, count


def run_mode(args, label, asynchronous):
    directory = args.output_dir / label
    directory.mkdir()
    disk = directory / "prefix-state-cache"
    disk.mkdir()
    server = NativeServer(args.after, args.model, directory, False)
    for name, value in (("--ctx-size", 8192), ("--parallel", 1), ("--batch-size", 512),
                        ("--ubatch-size", 512), ("--cache-ram", 5)):
        set_arg(server.command, name, value)
    server.env.update(WILDLLAMA_DEVICE_MASK="0", WILDLLAMA_PREFIX_DISK_CACHE_DIR=str(disk),
                      WILDLLAMA_PREFIX_DISK_CACHE_MIB="128", WILDLLAMA_PREFIX_DISK_ASYNC=str(int(asynchronous)))
    raw, outputs, logprob_count, proof = [], [], 0, {}
    prefixes = [[20 + j] + [40 + ((i * 61 + j * 17) % 190) for i in range(511)] for j in range(6)]
    require(len({p[0] for p in prefixes}) == 6 and all(len(p) == 512 for p in prefixes),
            "Fixture must have six disjoint 512-token prefixes")
    first_seen = None
    try:
        with server:
            props = server.request("/props")
            require(props.get("total_slots") == 1 and
                    props.get("default_generation_settings", {}).get("n_ctx") == 8192,
                    "Native runner changed logical context or parallelism")
            write_json(directory / "props.json", props)

            def complete(name, prompt, eos=False):
                nonlocal logprob_count
                body = dict(prompt=prompt, n_predict=8, temperature=0, seed=1234, top_k=40,
                            top_p=0.95, min_p=0, backend_sampling=False, cache_prompt=True,
                            n_probs=5, post_sampling_probs=False, return_tokens=True,
                            ignore_eos=not eos, stream=False)
                if eos:
                    body["grammar"] = 'root ::= "OK"'
                event = dict(name=name, request=body)
                raw.append(event)
                response = server.request("/completion", body)
                event["response"] = response
                result, count = stable(response, len(prompt), eos)
                outputs.append(result)
                logprob_count += count
                event["state_files_after"] = [str(p.relative_to(disk)) for p in state_files(disk)]
                write_json(directory / "responses.json", raw)
                return response

            for index, prefix in enumerate(prefixes):
                complete(chr(65 + index), prefix)
                files = state_files(disk)
                if index == 0:
                    require(not files, "A was not the first active prefix in a fresh cache")
                if index == 2:
                    deadline = time.monotonic() + 3
                    while asynchronous and not files and time.monotonic() < deadline:
                        time.sleep(.02)
                        files = state_files(disk)
                    require(bool(files) is asynchronous,
                            "Expected async prewrite before pressure, with no synchronous-control files yet")
                    proof["prepressure_files_after_c"] = [str(p.relative_to(disk)) for p in files]
                    proof["actual_async_prewrite"] = asynchronous
                if files and first_seen is None:
                    first_seen = dict(after_request=chr(65 + index), path=str(files[0].relative_to(disk)))
                    require(files[0].name == "1.state", "First backing entry was not ordinal 1")
            files = state_files(disk)
            require(first_seen is not None and len(files) >= 2, "No real cold-cache files were written")
            require(len({p.parent for p in files}) == 1, "Expected one isolated pager session")
            a_file = disk / first_seen["path"]
            require(a_file.is_file() and a_file.name == "1.state" and a_file.stat().st_size > 1024 * 1024,
                    "Oldest A backing file was missing before its return")
            proof.update(first_seen=first_seen, a_file_bytes=a_file.stat().st_size,
                         a_file_sha256=sha(a_file), files_before_a=[str(p.relative_to(disk)) for p in files],
                         mapping="A is the first captured prefix; unique first tokens prevent containment; FIFO spill/prewrite selects A first")
            # An identical prompt re-evaluates its final token for logits. One
            # probe token preserves 512 shared rows and makes cache_n>=512 proof
            # possible without counting a full recomputation as a cache hit.
            response = complete("A-return-probe", prefixes[0] + [249])
            require(not a_file.exists(), "A backing file was not consumed by the restore")
            cached = response.get("timings", {}).get("cache_n")
            require(type(cached) is int and cached >= 512, "A recomputed instead of restoring its cold prefix")
            proof.update(a_file_consumed=True, a_cached_tokens=cached, actual_cold_restore=True)
            for index in (1, 2, 0):
                complete(chr(65 + index) + "-return-probe", prefixes[index] + [249 + index])
            complete("forced-eos", [239, 240], eos=True)
            require(server.process.poll() is None, "CPU server exited during cache validation")
            text = server.log_text()
            require(re.search(r"\bCPU(?:_[A-Za-z0-9]+)?\s+compute buffer size", text),
                    "No CPU execution allocation evidence")
            require("cold prompt cache entry unavailable" not in text and
                    "failed to restore complete prompt state" not in text, "Cold restore silently fell back")
    finally:
        write_json(directory / "responses.json", raw)
    return dict(async_enabled=asynchronous, command=server.command, backend=str(server.backend),
                backend_sha256=sha(server.backend), log_sha256=sha(server.logfile),
                raw_responses_sha256=sha(directory / "responses.json"), cases=len(outputs),
                logprob_values=logprob_count, cold_restore=proof), outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", "--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.after, args.model, args.output_dir = (p.resolve() for p in (args.after, args.model, args.output_dir))
    require(args.output_dir.drive.upper() != "C:", "All outputs/caches must be outside C:")
    require(args.after.is_file() and 0 < args.model.stat().st_size <= 64 * 1024 * 1024,
            "Use an existing native payload and a local model no larger than 64 MiB")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare, comparator_path = comparator()
    report = dict(status="running", cpu_only=True, throughput_measured=False,
                  executable=str(args.after), executable_sha256=sha(args.after), model=str(args.model),
                  model_bytes=args.model.stat().st_size, model_sha256=sha(args.model),
                  harness_sha256=sha(Path(__file__)), launcher_sha256=sha(Path(__file__).with_name("native_execution_features_smoke.py")),
                  comparator_sha256=sha(comparator_path), context_tokens=8192, parallel=1,
                  cache_ram_mib=5, disk_quota_mib=128, probability_tolerance=0.0)
    try:
        before, left = run_mode(args, "sync", False)
        after, right = run_mode(args, "async", True)
        report.update(sync=before, asynchronous=after, comparison=compare(left, right, 0.0), status="passed")
        require(before["cases"] == after["cases"] == 11 and before["logprob_values"] > 0,
                "Required completion/probability checks were missing")
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
