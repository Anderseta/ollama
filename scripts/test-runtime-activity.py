#!/usr/bin/env python3
"""Exercise live activity on an isolated runner through the standard Ollama API.

Uses an already prepared GPU-first laboratory and payload. The benchmark owns
and stops its server; no production model store or listener is changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=11448)
    parser.add_argument("--label", default="live-activity")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / (args.label + "-activity.json")
    if report_path.exists():
        raise ValueError("Refusing to overwrite activity observations")
    samples, errors, stop = [], [], threading.Event()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{args.port}"

    def request(route, body=None, timeout=2):
        req = urllib.request.Request(base + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        return opener.open(req, timeout=timeout)

    try:
        with request("/api/version"):
            raise ValueError("Laboratory port is already occupied")
    except urllib.error.URLError:
        pass

    def observe():
        while not stop.is_set():
            start = time.monotonic()
            try:
                with request("/api/activity") as response:
                    data = json.load(response)
                    samples.append({"elapsed_ms": (time.monotonic() - start) * 1000,
                                    "received_at": time.time_ns() // 1_000_000, "snapshot": data})
            except (OSError, ValueError) as exc:
                # Connection refusal is expected only during startup/shutdown.
                if isinstance(exc, urllib.error.HTTPError):
                    errors.append({"status": exc.code, "error": str(exc)})
                elif samples and not isinstance(exc, urllib.error.URLError):
                    errors.append({"error": str(exc)})
            stop.wait(.5)

    command = [sys.executable, str(Path(__file__).with_name("benchmark-gpu-first.py")),
               "--payload", str(args.payload.resolve()), "--lab-root", str(args.lab_root.resolve()),
               "--output", str(output), "--port", str(args.port), "--label", args.label,
               "--rounds", "2", "--predict", "4", "--gpu-exclusive",
               "--gpu-vram-limit-mib", "12288", "--hold-seconds", "30"]
    report = {"samples": samples, "poll_errors": errors, "checks": {}}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    try:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if "observation_hold_s" in event:
                # Preserve the tested context and execution policy. Only the
                # protocol changes from /api/chat to /api/generate here.
                options = {"num_ctx": 131072, "num_predict": 1, "temperature": 0,
                           "seed": 42, "gpu_exclusive": True, "gpu_vram_limit_mib": 12288,
                           "moe_ram_limit_mib": 6144, "num_batch": 0}
                with request("/api/generate", {"model": "qwen3.8-flash-next-uncensored:q4_k_m",
                             "prompt": "Responda apenas OK.", "stream": False,
                             "think": False, "options": options}, timeout=300) as response:
                    generated = json.load(response)
                report["checks"]["standard_generate"] = generated.get("done") is True and not generated.get("error")
                report["generate_eval_count"] = generated.get("eval_count")
                with request("/api/generate", {"model": "qwen3.8-flash-next-uncensored:q4_k_m",
                             "keep_alive": 0, "stream": False}, timeout=30) as response:
                    unloaded = json.load(response)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    with request("/api/activity") as response:
                        state = json.load(response)
                    if not state["runners"]:
                        report["checks"]["unloaded_runner_removed"] = unloaded.get("done") is True
                        break
                    time.sleep(.2)
        report["benchmark_exit_code"] = process.wait(timeout=30)
        benchmark = json.loads((output / (args.label + "-results.json")).read_text(encoding="utf8"))
        report["checks"]["standard_chat"] = benchmark.get("ok") is True and len(benchmark["results"]) == 2
        report["checks"]["isolated_server_stopped"] = benchmark.get("stopped") is True
        report["checks"]["capability_advertised"] = "live_activity" in benchmark.get("version", {}).get("capabilities", [])
        phases, operations, memories = set(), set(), []
        for sample in samples:
            snapshot = sample["snapshot"]
            assert snapshot["version"] == 1 and snapshot["scope"] == "shared_runner"
            for runner in snapshot["runners"]:
                assert runner["context_length"] == 131072 and runner["gpu_exclusive"] is True
                state = runner["activity"]
                phases.add(state["phase"])
                phases.update(event["phase"] for event in state["history"])
                assert len(state["history"]) <= 24
                if "operation" in state:
                    operations.add(state["operation"]["name"])
                if "memory" in state:
                    memories.append(state["memory"])
                assert 0 <= state.get("progress", 0) <= 1
        report["phases"], report["operations"] = sorted(phases), sorted(operations)
        report["checks"]["measured_lifecycle"] = {"loading", "context", "prefill", "generating", "ready"} <= phases
        report["checks"]["memory_observed"] = any(m["ram_bytes"] > 0 and m["vram_bytes"] > 0 for m in memories)
        report["checks"]["weight_ram_quota"] = bool(memories) and all(m["ram_bytes"] <= m["ram_limit_bytes"] for m in memories)
        report["checks"]["managed_cuda_quota"] = bool(memories) and all(m["vram_bytes"] <= m["vram_limit_bytes"] for m in memories)
        ordered = sorted(sample["elapsed_ms"] for sample in samples)
        report["poll_latency_ms"] = {"count": len(ordered), "max": max(ordered, default=0),
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * .95))] if ordered else 0}
        report["checks"]["responsive_polling"] = bool(ordered) and report["poll_latency_ms"]["p95"] < 1000
        report["checks"]["no_poll_http_errors"] = not errors
        encoded = json.dumps(samples)
        report["checks"]["no_prompt_or_local_paths"] = all(value not in encoded for value in
            ("Responda apenas OK.", str(args.lab_root.resolve()), str(args.payload.resolve())))
        report["ok"] = report["benchmark_exit_code"] == 0 and all(report["checks"].values())
        print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2), flush=True)
    finally:
        stop.set()
        monitor.join(timeout=5)
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.terminate()
            process.wait(timeout=30)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
