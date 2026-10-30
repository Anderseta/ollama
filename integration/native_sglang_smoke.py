"""Compare native payloads with a tiny, local GGUF; never download a model.

This checks response/state equivalence, not model throughput or language quality.
Pass --gpu to exercise CUDA. Both payloads use the same device and settings.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

if not __debug__:
    raise RuntimeError("Run without Python optimization; fixture and equivalence checks are required")


def compare_outputs(before, after, probability_tolerance=0.0):
    """Keep tokens/metadata exact; bound only explicitly requested probability scalars."""
    metrics = dict(probability_abs_tolerance=probability_tolerance,
                   probability_values_compared=0, probability_value_differences=0,
                   max_probability_abs_error=0.0,
                   tokens_content_candidate_ids_exact=True)

    def check(a, b, path, probability=False, allow_probabilities=False):
        if type(a) is not type(b):
            raise AssertionError(f"Response type mismatch: {path}")
        if probability:
            if not isinstance(a, (int, float)) or isinstance(a, bool):
                raise AssertionError(f"Invalid probability type: {path}")
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= 1 and 0 <= b <= 1):
                raise AssertionError(f"Invalid probability value: {path}")
            error = abs(a - b)
            metrics["probability_values_compared"] += 1
            metrics["max_probability_abs_error"] = max(metrics["max_probability_abs_error"], error)
            metrics["probability_value_differences"] += a != b
            if error > probability_tolerance:
                raise AssertionError(f"Probability mismatch: {path}, error={error}")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise AssertionError(f"Response keys mismatch: {path}")
            for key in a:
                check(a[key], b[key], f"{path}/{key}",
                      allow_probabilities and key == "prob", allow_probabilities)
        elif isinstance(a, list):
            if len(a) != len(b):
                raise AssertionError(f"Response length mismatch: {path}")
            for index, (left, right) in enumerate(zip(a, b)):
                check(left, right, f"{path}/{index}", allow_probabilities=allow_probabilities)
        elif a != b:
            raise AssertionError(f"Response mismatch: {path}")

    if len(before) != len(after):
        raise AssertionError("Response case count mismatch")
    for index, (left, right) in enumerate(zip(before, after)):
        if left.keys() != right.keys():
            raise AssertionError(f"Response keys mismatch: case {index}")
        for key in left:
            check(left[key], right[key], f"{index}/{key}",
                  allow_probabilities=key == "completion_probabilities")
    return metrics


def run_payload(executable, model, output_dir, label, gpu, speculative, resident_caches=False):
    (output_dir / "cuda-cache").mkdir(parents=True, exist_ok=True)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(route, body=None):
        req = urllib.request.Request(
            base_url + route,
            None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=45) as response:
            return json.load(response)

    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OLLAMA_", "LLAMA_", "WILDLLAMA_", "GGML_", "CUDA_"))}
    env.update(TEMP=str(output_dir), TMP=str(output_dir),
               CUDA_CACHE_PATH=str(output_dir / "cuda-cache"))
    if resident_caches:
        # Unknown additive settings are ignored by the reference payload. Both
        # payloads keep exactly the same context, weights and sampling inputs.
        env.update(WILDLLAMA_GRAPH_VARIANTS_MIB="32",
                   WILDLLAMA_PREFIX_RETAIN_TOKENS="256")
    if gpu:
        suffix = "ggml-cuda.dll" if os.name == "nt" else "libggml-cuda.so"
        candidates = [executable.parent / "cuda_v12" / suffix,
                      executable.parent / "cuda_v13" / suffix,
                      executable.parent / suffix]
        backend = next((path for path in candidates if path.is_file()), None)
        if backend is None:
            raise FileNotFoundError("CUDA backend missing; refusing a CPU fallback")
        env["GGML_BACKEND_PATH"] = str(backend)
        env["CUDA_MODULE_LOADING"] = "EAGER"
        library_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[library_path] = os.pathsep.join((str(executable.parent), str(backend.parent),
                                            env.get(library_path, "")))
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    args = [str(executable), "--model", str(model), "--ctx-size", "1024",
            "--batch-size", "128", "--ubatch-size", "128", "--parallel", "2",
            "--threads", "1", "--threads-batch", "1", "--cache-ram", "32",
            "--no-warmup", "--no-webui", "--host", "127.0.0.1", "--port", str(port),
            "--n-gpu-layers", "99" if gpu else "0"]
    args += ["--device", "CUDA0" if gpu else "none"]
    if resident_caches:
        args += ["--kv-unified"]
    if speculative:
        args += ["--spec-type", "draft-simple", "--model-draft", str(model),
                 "--spec-draft-n-max", "3", "--spec-draft-n-min", "1",
                 "--spec-draft-p-min", "0", "--n-gpu-layers-draft", "99" if gpu else "0",
                 "--spec-draft-device", "CUDA0" if gpu else "none"]
    logfile = output_dir / f"{label}-{'gpu' if gpu else 'cpu'}-{'draft' if speculative else 'plain'}.log"
    with logfile.open("wb") as log:
        process = subprocess.Popen(
            args, cwd=executable.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Native server exited: {logfile}")
                try:
                    if request("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Native server not ready: {logfile}")
                time.sleep(0.1)

            def complete(prompt, **overrides):
                body = dict(prompt=prompt, n_predict=16, temperature=0, seed=1234,
                            top_k=20, top_p=0.95, min_p=0, cache_prompt=True,
                            stream=False, return_tokens=True, ignore_eos=True)
                body.update(overrides)
                return request("/completion", body)

            def stable(response):
                assert response.get("tokens"), "Missing generated token IDs"
                return {key: response.get(key) for key in
                        ("content", "tokens", "stop_type", "completion_probabilities")}

            results = []
            # Repeated and branching prefixes exercise cache replacement/restore.
            prefix = list(range(20, 84))
            for suffix in ([100], [101], [102], [100], [103], [101]):
                results.append(stable(complete(prefix + suffix)))
            # Preserve both sampling and final candidate probability responses.
            for temperature in (0, 0.6):
                for backend in (False, True):
                    for probs in (0, 5):
                        results.append(stable(complete(
                            [20, 21, 22, 23], temperature=temperature,
                            backend_sampling=backend, n_probs=probs,
                            post_sampling_probs=True, cache_prompt=False)))
            # Grammar must remain valid through prepared/fallback admission.
            for backend in (False, True):
                response = complete([20, 21, 22], grammar='root ::= "OK"',
                                    backend_sampling=backend, ignore_eos=False)
                assert response["content"] == "OK", response
                results.append(stable(response))
            # Fixed completion demands no arbitrary acceptance of invalid grammar.
            try:
                complete([20, 21], grammar="root ::= [")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400, exc
                results.append({"invalid_grammar_status": exc.code})
            else:
                raise AssertionError("Invalid grammar unexpectedly succeeded")
            # Multi-request isolation: compare contents rather than timing-dependent
            # probabilities, which can vary with the native GPU batch shape.
            prompts = [[20 + i, 40 + i, 70 + i] for i in range(8)]

            def burst(item):
                i, prompt = item
                options = dict(grammar='root ::= "OK"', backend_sampling=True,
                               ignore_eos=False) if i % 3 == 0 else {}
                return complete(prompt, **options)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(burst, enumerate(prompts)))
            for i, response in enumerate(responses):
                serial = burst((i, prompts[i]))
                assert response["tokens"] == serial["tokens"], "Concurrent sequence state leaked"
                results.append({key: response.get(key) for key in ("content", "tokens", "stop_type")})
            return results
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--resident-caches", action="store_true",
                        help="Exercise bounded graph variants and retained unified KV prefixes")
    args = parser.parse_args()
    assert 0 < args.model.stat().st_size <= 64 * 1024 * 1024, "Use a tiny fixture, at most 64 MiB"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = run_payload(args.before.resolve(), args.model.resolve(), args.output_dir.resolve(),
                         "before", args.gpu, args.speculative, args.resident_caches)
    after = run_payload(args.after.resolve(), args.model.resolve(), args.output_dir.resolve(),
                        "after", args.gpu, args.speculative, args.resident_caches)
    # Shared physical KV rows can change CUDA attention reduction order. Apply
    # a conservative 1e-5 absolute probability-metadata bound only in this
    # opt-in GPU test. Native fixtures check logits separately. Tokens, candidate
    # IDs, schemas and all other fields remain exact; ordinary tests are exact.
    tolerance = 1e-5 if args.gpu and args.resident_caches else 0.0
    try:
        comparison = compare_outputs(before, after, tolerance)
    except AssertionError:
        (args.output_dir / "mismatched-responses.json").write_text(
            json.dumps(dict(before=before, after=after), indent=2) + "\n", encoding="utf-8")
        raise
    report = dict(status="passed", cases_per_payload=len(after), gpu=args.gpu,
                  speculative=args.speculative, fixture_bytes=args.model.stat().st_size,
                  before_outputs_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                  outputs_sha256=hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
                  throughput_measured=False, resident_caches=args.resident_caches,
                  comparison=comparison)
    cache_suffix = "-caches" if args.resident_caches else ""
    name = f"smoke-{'gpu' if args.gpu else 'cpu'}-{'draft' if args.speculative else 'plain'}{cache_suffix}.json"
    (args.output_dir / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
