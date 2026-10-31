"""Compare dense decode on two explicit native payloads and a local small GGUF.

Only diagnostic models up to 64 MiB are accepted. No download or installation is
performed. The result measures this fixture; it is not a trained-model TPS claim.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

if not __debug__:
    raise RuntimeError("Run without Python optimization; benchmark correctness checks are required")
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_arg(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend((name, str(value)))


def stable(response):
    # Cache accounting and elapsed time are physical observations. Every other
    # field, including candidates, settings, token counts and stop metadata, is exact.
    return {key: value for key, value in response.items()
            if key not in ("timings", "tokens_cached")}


def checked_response(response, body, wall_ms):
    require(type(response) is dict, "Completion response must be an object")
    required = {"content", "tokens", "stop", "stop_type", "stopping_word", "truncated",
                "tokens_predicted", "tokens_evaluated", "generation_settings", "timings"}
    require(required <= response.keys(), "Completion omitted required semantic/timing fields")
    tokens = response["tokens"]
    require(type(tokens) is list and len(tokens) == body["n_predict"] and
            all(type(token) is int and token >= 0 for token in tokens), "Missing generated token IDs")
    require(type(response["content"]) is str and response["stop"] is True and
            response["stop_type"] == "limit" and response["stopping_word"] == "" and
            response["truncated"] is False, "Generation did not end at the unchanged requested limit")
    require(response["tokens_predicted"] == body["n_predict"], "Generated-token count changed")
    settings = response["generation_settings"]
    require(type(settings) is dict, "Effective generation settings are missing")
    for key in ("backend_sampling", "repeat_last_n", "repeat_penalty", "frequency_penalty",
                "presence_penalty", "n_predict", "n_probs", "temperature", "seed", "ignore_eos",
                "post_sampling_probs"):
        require(key in settings, f"Effective setting missing: {key}")
        actual, expected = settings[key], body[key]
        if type(expected) is bool:
            require(type(actual) is bool and actual is expected, f"Setting was ignored: {key}")
        elif type(expected) is float:
            require(type(actual) in (int, float) and math.isfinite(actual) and
                    math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6),
                    f"Setting was ignored: {key}")
        else:
            require(type(actual) is type(expected) and actual == expected, f"Setting was ignored: {key}")
    if body["n_probs"]:
        require(type(response.get("completion_probabilities")) is list and
                response["completion_probabilities"], "Requested probability output is missing")
    else:
        require("completion_probabilities" not in response, "Unrequested probability output was generated")
    timings = response["timings"]
    require(type(timings) is dict and timings.get("cache_n") == 0 and
            timings.get("prompt_n") == len(body["prompt"]) and
            timings.get("predicted_n") == body["n_predict"], "Prompt reuse or token accounting changed")
    metrics = {"wall_ms": wall_ms, "wall_tokens_per_second": len(tokens) * 1000.0 / wall_ms}
    for key in ("prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second"):
        value = timings.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"Invalid native timing: {key}")
        metrics[key] = value
    return metrics


def summarize(windows):
    result = {}
    for metric in ("wall_ms", "prompt_ms", "predicted_ms", "wall_tokens_per_second",
                   "prompt_per_second", "predicted_per_second"):
        before = [row["metrics"][metric] for row in windows if row["version"] == "before"]
        after = [row["metrics"][metric] for row in windows if row["version"] == "after"]
        b, a = statistics.median(before), statistics.median(after)
        higher_is_better = metric.endswith("per_second")
        pairs = []
        for index in range(0, len(windows), 4):
            block = windows[index:index + 4]
            require([row["version"] for row in block] == ["before", "after", "after", "before"],
                    "ABBA window order was not preserved")
            b_pair = statistics.mean([block[0]["metrics"][metric], block[3]["metrics"][metric]])
            a_pair = statistics.mean([block[1]["metrics"][metric], block[2]["metrics"][metric]])
            pairs.append({"cycle": index // 4, "before": b_pair, "after": a_pair,
                          "after_minus_before": a_pair - b_pair,
                          "speedup": a_pair / b_pair if higher_is_better else b_pair / a_pair,
                          "improved": a_pair > b_pair if higher_is_better else a_pair < b_pair})
        result[metric] = {
            "before_median": b, "after_median": a,
            "after_change_percent": (a / b - 1.0) * 100.0,
            "speedup": a / b if higher_is_better else b / a,
            "before_cv": statistics.pstdev(before) / statistics.mean(before),
            "after_cv": statistics.pstdev(after) / statistics.mean(after),
            "paired_improved": sum(row["improved"] for row in pairs),
            "paired_worsened": sum(not row["improved"] and row["after_minus_before"] != 0 for row in pairs),
            "paired_equal": sum(row["after_minus_before"] == 0 for row in pairs),
            "paired_cycles": pairs,
        }
    return result


def payload_identity(server):
    extension = ".dll" if os.name == "nt" else ".so"
    prefix = "" if os.name == "nt" else "lib"
    paths = [server.executable, server.backend]
    for name in ("llama", "ggml", "ggml-base", "ggml-cpu"):
        path = server.executable.parent / (prefix + name + extension)
        require(path.is_file(), f"Required native library missing: {path.name}")
        paths.append(path)
    return {str(path.resolve()): sha(path) for path in dict.fromkeys(paths)}


def placement_proof(server):
    text = server.log_text()
    layers = [(int(left), int(right)) for left, right in
              re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)]
    cuda_compute = bool(re.search(r"CUDA\d+\s+compute buffer size", text))
    cpu_compute = bool(re.search(r"CPU\s+compute buffer size", text))
    if server.gpu:
        require(any(left > 0 and left == right for left, right in layers) and cuda_compute,
                "GPU run lacks complete offload/CUDA compute proof")
    else:
        require(server.command[server.command.index("--n-gpu-layers") + 1] == "0" and
                server.command[server.command.index("--device") + 1] == "none" and
                "--no-op-offload" in server.command and server.env.get("CUDA_VISIBLE_DEVICES") == "" and
                cpu_compute and not cuda_compute and not any(left > 0 for left, _ in layers),
                "CPU control admitted GPU compute or lacks actual CPU compute proof")
    return {"offloaded_layers": layers, "cuda_compute_buffer": cuda_compute,
            "cpu_compute_buffer": cpu_compute, "requested_gpu": server.gpu,
            "explicit_backend": str(server.backend), "backend_sha256": sha(server.backend)}


def run(args, launcher, comparator, raw, reports):
    servers = {}
    identities = {}
    with ExitStack() as stack:
        for label in ("before", "after"):
            directory = args.output_dir / label
            directory.mkdir()
            server = launcher.NativeServer(getattr(args, label), args.model, directory, args.gpu)
            set_arg(server.command, "--ctx-size", 4096)
            set_arg(server.command, "--batch-size", 256)
            set_arg(server.command, "--ubatch-size", 256)
            # Native core INFO is verbosity 4. Debug token logging is excluded.
            set_arg(server.command, "--log-verbosity", 4)
            if args.cpu_mask:
                for name, value in (("--cpu-mask", args.cpu_mask), ("--cpu-mask-batch", args.cpu_mask),
                                    ("--cpu-strict", 1), ("--cpu-strict-batch", 1)):
                    set_arg(server.command, name, value)
            identities[label] = payload_identity(server)
            stack.enter_context(server)
            servers[label] = server
        processes = {label: {"pid": server.process.pid, "command": server.command,
                             "backend_path": str(server.backend),
                             "placement": placement_proof(server),
                             "optional_feature_environment": sorted(key for key in server.env
                                 if key.startswith(("OLLAMA_", "LLAMA_", "WILDLLAMA_")))}
                     for label, server in servers.items()}

        def request(label, case, stage, body, cycle=None, order=None):
            start = time.perf_counter()
            response = servers[label].request("/completion", body)
            wall_ms = (time.perf_counter() - start) * 1000.0
            row = {"version": label, "case": case, "stage": stage, "cycle": cycle,
                   "order": order, "request": body, "response": response, "wall_ms": wall_ms}
            raw.append(row)
            row["metrics"] = checked_response(response, body, wall_ms)
            return row

        cases = [("cpu-sampler-history-64", False, 64, True),
                 ("cpu-sampler-history-512", False, 512, True),
                 ("disabled-penalties-control", False, 64, False)]
        if args.backend_control:
            cases.append(("backend-sampling-control", True, 64, True))
        prompt = [35 + (index * 37) % 95 for index in range(256)]
        for name, backend, history, penalties in cases:
            body = dict(prompt=prompt, n_predict=args.n_predict, temperature=0.0, seed=1234,
                        top_k=40, top_p=0.95, min_p=0.0, cache_prompt=False, stream=False,
                        return_tokens=True, ignore_eos=True, backend_sampling=backend,
                        repeat_last_n=history, repeat_penalty=1.13 if penalties else 1.0,
                        frequency_penalty=0.25 if penalties else 0.0,
                        presence_penalty=-0.125 if penalties else 0.0,
                        n_probs=5, post_sampling_probs=True)
            correctness = {label: request(label, name, "correctness", body) for label in servers}
            comparison = comparator.compare_outputs(
                [stable(correctness["before"]["response"])],
                [stable(correctness["after"]["response"])], 0.0)
            timed_body = dict(body, n_probs=0)
            for warmup in range(2):
                for label in servers:
                    request(label, name, "warmup", timed_body, order=warmup)
            windows = []
            reference = None
            for cycle in range(args.cycles):
                for order, label in enumerate(("before", "after", "after", "before")):
                    row = request(label, name, "timed", timed_body, cycle, order)
                    observed = stable(row["response"])
                    if reference is None:
                        reference = observed
                    comparator.compare_outputs([reference], [observed], 0.0)
                    windows.append(row)
            case = {"case": name, "backend_sampling": backend, "penalties_enabled": penalties,
                    "history": history, "untimed_probability_comparison": comparison,
                    "timed_repeats_exact": True, "warmups_per_version": 2,
                    "timed_requests_per_version": args.cycles * 2,
                    "summary": summarize(windows)}
            reports.append(case)
            print(json.dumps({"case": name, "decode_speedup": case["summary"]["predicted_ms"]["speedup"],
                              "wall_speedup": case["summary"]["wall_ms"]["speedup"],
                              "decode_pairs_improved": case["summary"]["predicted_ms"]["paired_improved"],
                              "cycles": args.cycles}), flush=True)
        for label, server in servers.items():
            require(server.process.poll() is None, "Native server exited during the benchmark")
            require(payload_identity(server) == identities[label], "Native payload changed during the benchmark")
    return {"processes": processes, "payload_files_sha256": identities,
            "log_sha256": {label: sha(server.logfile) for label, server in servers.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--backend-control", action="store_true")
    parser.add_argument("--n-predict", type=int, choices=(64, 128), default=128)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--cpu-mask", help="Optional native strict worker CPU mask in hexadecimal; identical for both payloads")
    args = parser.parse_args()
    for field in ("before", "after", "model", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    require(args.before.is_file() and args.after.is_file(), "Explicit native executables are required")
    require(args.model.is_file() and 0 < args.model.stat().st_size <= 64 * 1024 * 1024,
            "Use an existing local diagnostic GGUF of at most 64 MiB; no downloads")
    require(args.model.suffix.lower() == ".gguf", "An explicit local GGUF fixture is required")
    require(6 <= args.cycles <= 24, "Use between 6 and 24 complete ABBA cycles")
    if args.cpu_mask:
        require(0 < int(args.cpu_mask, 16) < (1 << 256), "Invalid bounded CPU mask")
    require(os.name != "nt" or args.output_dir.drive.upper() in ("G:", "F:"),
            "All Windows benchmark output/temp/cache files must be on G: or F:")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    integration = Path(__file__).resolve().parent
    launcher_path = integration / "native_execution_features_smoke.py"
    comparator_path = integration / "native_sglang_smoke.py"
    launcher = load_local("dense_decode_native_launcher", launcher_path)
    comparator = load_local("dense_decode_exact_comparator", comparator_path)
    raw, cases = [], []
    raw_path = args.output_dir / "responses.json"
    report = {"status": "running", "gpu": args.gpu, "fixture_bytes": args.model.stat().st_size,
              "fixture_sha256": sha(args.model), "harness_sha256": sha(Path(__file__)),
              "launcher_sha256": sha(launcher_path), "comparator_sha256": sha(comparator_path),
              "context_tokens": 4096, "prompt_tokens": 256, "n_predict": args.n_predict,
              "requested_strict_native_cpu_mask": args.cpu_mask,
              "warmups_per_case_per_version": 2, "abba_cycles_per_case": args.cycles,
              "case_reports": cases, "probability_tolerance": 0.0,
              "scope": "Same-setting small synthetic/diagnostic dense fixture; no trained-model quality or TPS claim"}
    try:
        report.update(run(args, launcher, comparator, raw, cases))
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        raise
    finally:
        raw_path.write_text(json.dumps({"requests": raw}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["responses_sha256"] = sha(raw_path)
        report["raw_request_count"] = len(raw)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
