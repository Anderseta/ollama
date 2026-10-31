"""Check meaningful pre-sampling logprobs on small local native GGUF fixtures.

Compare explicit before/after payloads on one selected device. Optional CUDA
controls compare the final payload's mask-disabled and automatic-mask paths.
This checks exact numerical/API equivalence, not language quality or throughput.
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
import sys
import time

if not __debug__:
    raise RuntimeError("Run without Python optimization; meaningful logprob checks are required")
sys.dont_write_bytecode = True


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
    return {key: value for key, value in response.items()
            if key not in ("timings", "tokens_cached")}


def checked_response(response, body):
    required = {"content", "tokens", "stop", "stop_type", "stopping_word", "truncated",
                "tokens_predicted", "tokens_evaluated", "generation_settings", "completion_probabilities"}
    require(type(response) is dict and required <= response.keys(), "Completion schema is incomplete")
    require(type(response["content"]) is str and type(response["tokens"]) is list and
            len(response["tokens"]) == body["n_predict"] and
            all(type(token) is int and token >= 0 for token in response["tokens"]), "Generated IDs/content are missing")
    require(response["stop"] is True and response["stop_type"] == "limit" and
            response["stopping_word"] == "" and response["truncated"] is False and
            response["tokens_predicted"] == body["n_predict"], "Logical generation/stop boundary changed")
    settings = response["generation_settings"]
    require(type(settings) is dict, "Effective settings are missing")
    for key in ("backend_sampling", "repeat_last_n", "repeat_penalty", "frequency_penalty",
                "presence_penalty", "n_predict", "n_probs", "temperature", "seed", "ignore_eos",
                "post_sampling_probs"):
        actual, expected = settings.get(key), body[key]
        if type(expected) is bool:
            require(type(actual) is bool and actual is expected, f"Setting was ignored: {key}")
        elif type(expected) is float:
            require(type(actual) in (int, float) and math.isfinite(actual) and
                    math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6), f"Setting was ignored: {key}")
        else:
            require(type(actual) is type(expected) and actual == expected, f"Setting was ignored: {key}")
    timings = response.get("timings")
    require(type(timings) is dict and timings.get("cache_n") == 0 and
            timings.get("prompt_n") == len(body["prompt"]), "Prompt caching or input length changed")
    rows = response["completion_probabilities"]
    require(type(rows) is list and rows, "Pre-sampling logprobs are missing")
    scalars, informative, distinct_min = 0, 0, None

    def logprob(item):
        nonlocal scalars
        require(type(item) is dict and type(item.get("id")) is int and item["id"] >= 0,
                "Invalid logprob candidate ID")
        value = item.get("logprob")
        require(type(value) in (int, float) and math.isfinite(value) and value <= 0,
                "Logprob must be a finite nonpositive number")
        require("prob" not in item, "Post-sampling probabilities replaced raw logprobs")
        scalars += 1
        return value

    for row in rows:
        logprob(row)
        require(row["id"] in response["tokens"], "Logprob row refers to an ungenerated token")
        top = row.get("top_logprobs")
        require(type(top) is list and 2 <= len(top) <= body["n_probs"],
                "At least two top-logprob candidates are required")
        values = [logprob(candidate) for candidate in top]
        unique = len({candidate["id"] for candidate in top})
        require(unique == len(top) and unique >= 2, "Top-logprob candidates are not distinct")
        # Reject singleton/one-hot output and clamped log(0) sentinels. Genuine
        # alternatives must have representable probability strictly between 0 and 1.
        genuine = sum(value < 0 and 0 < math.exp(value) < 1 for value in values)
        require(genuine >= 2, "Top logprobs contain no informative nonzero alternatives")
        informative += genuine
        distinct_min = unique if distinct_min is None else min(distinct_min, unique)
    return {"token_probability_rows": len(rows), "logprob_scalars": scalars,
            "informative_negative_alternatives": informative, "minimum_distinct_candidates": distinct_min}


def payload_identity(server):
    suffix, prefix = (".dll", "") if os.name == "nt" else (".so", "lib")
    files = [server.executable, server.backend]
    for name in ("llama", "ggml", "ggml-base", "ggml-cpu"):
        path = server.executable.parent / (prefix + name + suffix)
        require(path.is_file(), f"Required native library missing: {path.name}")
        files.append(path)
    return {str(path.resolve()): sha(path) for path in dict.fromkeys(files)}


def placement(server):
    text = server.log_text()
    layers = [(int(left), int(right)) for left, right in
              re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)]
    cuda = bool(re.search(r"CUDA\d+\s+compute buffer size", text))
    cpu = bool(re.search(r"CPU\s+compute buffer size", text))
    if server.gpu:
        require(any(left > 0 and left == right for left, right in layers) and cuda,
                "CUDA run lacks full model offload/compute proof")
    else:
        require(cpu and not cuda and not any(left > 0 for left, _ in layers) and
                server.command[server.command.index("--n-gpu-layers") + 1] == "0" and
                server.command[server.command.index("--device") + 1] == "none" and
                "--no-op-offload" in server.command and server.env.get("CUDA_VISIBLE_DEVICES") == "",
                "CPU control admitted GPU computation")
    return {"offloaded_layers": layers, "cuda_compute": cuda, "cpu_compute": cpu}


def make_server(args, launcher, executable, model, directory, mask=None, mask_test=False):
    directory.mkdir()
    server = launcher.NativeServer(executable, model, directory, args.gpu)
    for name, value in (("--ctx-size", 4096), ("--batch-size", 512 if mask_test else 256),
                        ("--ubatch-size", 512 if mask_test else 256), ("--parallel", 1),
                        ("--threads", 1), ("--threads-batch", 1), ("--log-verbosity", 5)):
        set_arg(server.command, name, value)
    if mask_test:
        set_arg(server.command, "--flash-attn", "on")
    if mask is None:
        server.env.pop("WILDLLAMA_DEVICE_MASK", None)
    else:
        server.env["WILDLLAMA_DEVICE_MASK"] = mask
    return server


def request_body(length, count, backend=False, history=64, penalties=True):
    return dict(prompt=[35 + (index * 37) % 95 for index in range(length)],
                n_predict=count, temperature=0.0, seed=1234, top_k=40, top_p=0.95, min_p=0.0,
                cache_prompt=False, stream=False, return_tokens=True, ignore_eos=True,
                backend_sampling=backend, repeat_last_n=history,
                repeat_penalty=1.13 if penalties else 1.0,
                frequency_penalty=0.25 if penalties else 0.0,
                presence_penalty=-0.125 if penalties else 0.0,
                n_probs=5, post_sampling_probs=False)


def request(server, label, case, body, raw):
    start = server.logfile.stat().st_size
    response = server.request("/completion", body)
    row = {"server": label, "case": case, "request": body, "response": response}
    raw.append(row)
    row["logprob_proof"] = checked_response(response, body)
    return row, start


def exact(comparator, left, right):
    result = comparator.compare_outputs([stable(left["response"])], [stable(right["response"])], 0.0)
    require(left["logprob_proof"] == right["logprob_proof"], "Informative probability structure changed")
    result.update(logprob_scalars_compared=left["logprob_proof"]["logprob_scalars"],
                  logprob_value_differences=0, max_logprob_abs_error=0.0)
    return result


def group_identity(servers):
    return {label: {"pid": server.process.pid, "command": server.command,
                    "payload_files_sha256": payload_identity(server), "placement": placement(server),
                    "device_mask_environment": server.env.get("WILDLLAMA_DEVICE_MASK", "unset"),
                    "backend_path": str(server.backend)} for label, server in servers.items()}


def dense_cases(args, launcher, comparator, raw, reports):
    servers = {}
    with ExitStack() as stack:
        for label in ("before", "after"):
            server = make_server(args, launcher, getattr(args, label), args.model,
                                 args.output_dir / label)
            stack.enter_context(server)
            servers[label] = server
        identities = group_identity(servers)
        for name, backend, history, penalties in (
                ("history-64", False, 64, True), ("history-512", False, 512, True),
                ("disabled-penalties", False, 64, False), ("requested-backend-prelogprobs", True, 64, True)):
            body = request_body(256, args.n_predict, backend, history, penalties)
            reference = None
            for repeat in range(2):
                observed = {}
                for label, server in servers.items():
                    row, _ = request(server, label, f"{name}-repeat-{repeat}", body, raw)
                    observed[label] = row
                    if reference is None:
                        reference = row
                    exact(comparator, reference, row)
                comparison = exact(comparator, observed["before"], observed["after"])
                reports.append({"case": name, "repeat": repeat, "comparison": comparison,
                                "informative_logprob_proof": observed["after"]["logprob_proof"]})
    for label, server in servers.items():
        require(payload_identity(server) == identities[label]["payload_files_sha256"], "Payload changed during the check")
        identities[label]["log_sha256"] = sha(server.logfile)
    return identities


def mask_marker(server, start, expected):
    deadline = time.monotonic() + 3
    while True:
        count = server.log_text(start).count("device-mask inputs:")
        if count or not expected:
            require(bool(count) is expected, "Compact-mask execution did not match the requested control")
            return count
        require(server.process.poll() is None, "Native server exited while checking compact-mask execution")
        if time.monotonic() >= deadline:
            raise AssertionError("Automatic compact-mask inputs were not populated for actual execution")
        time.sleep(0.02)


def expected_automatic_mask(role, length):
    # This matrix uses 512-token batches and default recurrent checkpoints.
    # Hybrid tails split off four tokens: 2048 never combines 512 queries with
    # the extended-layout 2 MiB floor. At 3072 an earlier full batch qualifies.
    cases = {"ordinary": {512: True, 2048: True},
             "hybrid": {512: False, 2048: False, 3072: True}}
    require(role in cases and length in cases[role], "Unsupported compact-mask matrix case")
    return cases[role][length]


def mask_cases(args, launcher, comparator, role, model, raw, reports):
    servers = {}
    with ExitStack() as stack:
        for label, mask in (("disabled", "0"), ("automatic", None)):
            server = make_server(args, launcher, args.after, model,
                                 args.output_dir / f"{role}-{label}", mask, True)
            stack.enter_context(server)
            servers[label] = server
        identities = group_identity(servers)
        for length in ((512, 2048, 3072) if role == "hybrid" else (512, 2048)):
            reference = None
            for repeat in range(2):
                observed = {}
                markers = {}
                for label, server in servers.items():
                    body = request_body(length, args.n_predict)
                    row, start = request(server, f"{role}-{label}", f"{length}-repeat-{repeat}", body, raw)
                    expected_mask = label == "automatic" and expected_automatic_mask(role, length)
                    markers[label] = mask_marker(server, start, expected_mask)
                    observed[label] = row
                    if reference is None:
                        reference = row
                    exact(comparator, reference, row)
                reports.append({"case": f"{role}-mask-{length}", "repeat": repeat,
                                "comparison": exact(comparator, observed["disabled"], observed["automatic"]),
                                "expected_compact_mask_inputs": {
                                    "disabled": False, "automatic": expected_automatic_mask(role, length)},
                                "actual_compact_mask_markers": markers,
                                "informative_logprob_proof": observed["automatic"]["logprob_proof"]})
        require("device-mask inputs:" not in servers["disabled"].log_text(),
                "Disabled compact-mask control populated compact inputs")
    for label, server in servers.items():
        require(payload_identity(server) == identities[label]["payload_files_sha256"], "Payload changed during the check")
        identities[label]["log_sha256"] = sha(server.logfile)
    return {"model": str(model), "model_bytes": model.stat().st_size, "model_sha256": sha(model),
            "servers": identities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "model", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--n-predict", type=int, choices=(8, 16), default=16)
    parser.add_argument("--ordinary-mask-model", type=Path)
    parser.add_argument("--hybrid-mask-model", type=Path)
    args = parser.parse_args()
    for name in ("before", "after", "model", "output_dir", "ordinary_mask_model", "hybrid_mask_model"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    require(args.before.is_file() and args.after.is_file(), "Explicit native payload executables are required")
    for model in (args.model, args.ordinary_mask_model, args.hybrid_mask_model):
        if model is not None:
            require(model.is_file() and model.suffix.lower() == ".gguf" and
                    0 < model.stat().st_size <= 64 * 1024 * 1024, "Use only existing local GGUF fixtures up to 64 MiB")
    require(args.gpu or not (args.ordinary_mask_model or args.hybrid_mask_model),
            "Automatic-mask controls require explicit --gpu")
    require(args.output_dir.drive.upper() != "C:", "All test output/temp/cache files must be outside C:")
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    integration = Path(__file__).resolve().parent
    launcher_path = integration / "native_execution_features_smoke.py"
    comparator_path = integration / "native_sglang_smoke.py"
    launcher = load_local("dense_logprob_native_launcher", launcher_path)
    comparator = load_local("dense_logprob_exact_comparator", comparator_path)
    raw, cases = [], []
    report = {"status": "running", "gpu": args.gpu, "throughput_measured": False,
              "language_quality_measured": False, "fixture_bytes": args.model.stat().st_size,
              "fixture_sha256": sha(args.model), "harness_sha256": sha(Path(__file__)),
              "launcher_sha256": sha(launcher_path), "comparator_sha256": sha(comparator_path),
              "n_predict": args.n_predict, "post_sampling_probs": False, "n_probs": 5,
              "logprob_absolute_tolerance": 0.0, "case_reports": cases,
              "backend_request_scope": "Pre-sampling probabilities require host logits; backend:true is an API request/fallback control, not proof of a device selector"}
    try:
        report["dense_servers"] = dense_cases(args, launcher, comparator, raw, cases)
        report["mask_controls"] = {}
        for role, model in (("ordinary", args.ordinary_mask_model), ("hybrid", args.hybrid_mask_model)):
            if model is not None:
                report["mask_controls"][role] = mask_cases(args, launcher, comparator, role, model, raw, cases)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        raw_path = args.output_dir / "responses.json"
        raw_path.write_text(json.dumps({"requests": raw}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["responses_sha256"] = sha(raw_path)
        report["raw_request_count"] = len(raw)
        report["logprob_scalars_compared"] = sum(case["comparison"]["logprob_scalars_compared"] for case in cases)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(cases),
                      "logprob_scalars_compared": report["logprob_scalars_compared"], "gpu": args.gpu}), flush=True)


if __name__ == "__main__":
    main()
