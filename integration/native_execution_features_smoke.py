"""Check grammar-first HTTP semantics on ONE final native payload and tiny model.

Reference and backend requests use the same grammar ordering and binary. This
does not compare grammar-first to grammar-last or claim model throughput.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

if not __debug__:
    raise RuntimeError("Run without Python optimization; feature and equivalence checks are required")
sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparator():
    for root in Path(__file__).resolve().parents:
        path = root / "integration/native_sglang_smoke.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("wildllama_native_equivalence", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            def compare(left, right, tolerance=0.0):
                if type(left) is dict and type(right) is dict:
                    return module.compare_outputs([left], [right], tolerance)
                return module.compare_outputs(left, right, tolerance)
            return compare, path
    raise FileNotFoundError("The existing strict native HTTP comparator was not found")


def literal(text):
    return "root ::= " + json.dumps(text, ensure_ascii=False) + "\n"


GRAMMARS = (
    ("forced-ok", literal("OK"), "OK"),
    ("bounded-json", "root ::= " + json.dumps('{"ok":') + ' [0-3] "}"\n', None),
    ("utf8-literal", literal("ação"), "ação"),
)
MASK_CHAIN = re.compile(r"sampler chain:.*(?:^|\s|->)\+token-mask(?:\s|$|->)", re.MULTILINE)
SELECTED = "Backend sampler selected token"
LEGACY_FALLBACK = "backend sampling is not compatible with grammar, disabling"


def stable_response(response, first, backend, expected_literal, n_probs):
    require(type(response) is dict, "Completion response must be an object")
    mandatory = {"content", "tokens", "stop", "stop_type", "stopping_word", "truncated",
                 "tokens_predicted", "tokens_evaluated", "generation_settings"}
    require(mandatory <= response.keys(), "Completion schema omitted required semantic fields")
    settings = response["generation_settings"]
    require(type(settings) is dict and settings.get("grammar_first") is first,
            "The native request schema ignored or changed grammar_first")
    effective_backend = backend if first else False
    require(settings.get("backend_sampling") is effective_backend,
            "The native schema ignored the backend false override, or legacy fallback was not reflected")
    require(type(response["content"]) is str and type(response["tokens"]) is list and response["tokens"] and
            all(type(token) is int and token >= 0 for token in response["tokens"]), "Missing generated token IDs/content")
    require(response["stop"] is True and response["stop_type"] == "eos" and response["truncated"] is False,
            "The bounded grammar did not finish through EOS")
    require(response["stopping_word"] == "", "An unrelated stopping word ended the grammar test")
    require("\ufffd" not in response["content"], "UTF-8 was replaced instead of preserved")
    if expected_literal is not None:
        require(response["content"] == expected_literal, "Forced literal grammar produced different text")
    else:
        parsed = json.loads(response["content"])
        require(type(parsed) is dict and set(parsed) == {"ok"} and type(parsed["ok"]) is int and
                0 <= parsed["ok"] <= 3, "Bounded JSON grammar produced invalid content")
    if n_probs:
        require(type(response.get("completion_probabilities")) is list and response["completion_probabilities"],
                "Probability response was missing")
    else:
        require("completion_probabilities" not in response, "Unrequested probabilities were fabricated")
    # Timing and physical cache counters are not semantic outputs. Both modes
    # intentionally echo a different backend_sampling flag; every other field,
    # nested schema/type, token/candidate ID and stop value stays comparable.
    result = {key: value for key, value in response.items() if key not in ("timings", "tokens_cached")}
    result["generation_settings"] = {key: value for key, value in settings.items() if key != "backend_sampling"}
    return result


class NativeServer:
    def __init__(self, executable, model, directory, gpu):
        self.executable, self.directory, self.gpu = executable, directory, gpu
        self.process = None
        self.logfile = directory / ("gpu-server.log" if gpu else "cpu-server.log")
        self.log = None
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("OLLAMA_", "LLAMA_", "WILDLLAMA_", "GGML_", "CUDA_"))}
        for folder in (directory / "temp", directory / "cuda-cache"):
            folder.mkdir(parents=True, exist_ok=True)
        self.env.update(TEMP=str(directory / "temp"), TMP=str(directory / "temp"),
                        TMPDIR=str(directory / "temp"), CUDA_CACHE_PATH=str(directory / "cuda-cache"),
                        PYTHONDONTWRITEBYTECODE="1")
        suffix = ".dll" if os.name == "nt" else ".so"
        prefix = "" if os.name == "nt" else "lib"
        if gpu:
            name = prefix + "ggml-cuda" + suffix
            candidates = [executable.parent / "cuda_v12" / name, executable.parent / "cuda_v13" / name,
                          executable.parent / name]
            self.backend = next((path for path in candidates if path.is_file()), None)
            require(self.backend is not None, "Explicit CUDA backend missing; refusing CPU fallback")
            self.env["CUDA_MODULE_LOADING"] = "EAGER"
        else:
            self.backend = executable.parent / (prefix + "ggml-cpu" + suffix)
            require(self.backend.is_file(), "Explicit CPU backend missing")
            self.env["CUDA_VISIBLE_DEVICES"] = ""
        self.env["GGML_BACKEND_PATH"] = str(self.backend)
        library = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        self.env[library] = os.pathsep.join((str(executable.parent), str(self.backend.parent), self.env.get(library, "")))
        self.command = [str(executable), "--model", str(model), "--offline", "--ctx-size", "512",
            "--batch-size", "64", "--ubatch-size", "64", "--parallel", "1", "--threads", "1",
            "--threads-batch", "1", "--cache-ram", "32", "--no-warmup", "--no-webui", "--spec-type", "none",
            "--host", "127.0.0.1", "--port", str(port), "--n-gpu-layers", "99" if gpu else "0",
            "--device", "CUDA0" if gpu else "none", "--log-verbosity", "5"]
        if not gpu:
            self.command.append("--no-op-offload")

    def request(self, route, body=None):
        request = urllib.request.Request(self.url + route,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=45) as response:
            return json.load(response)

    def log_text(self, start=0):
        with self.logfile.open("rb") as stream:
            stream.seek(start)
            return stream.read().decode("utf-8", errors="replace")

    def wait_log(self, start, backend, first):
        deadline = time.monotonic() + 3
        while True:
            text = self.log_text(start)
            chain_seen = "sampler chain:" in text
            mask_seen = bool(MASK_CHAIN.search(text))
            selected = SELECTED in text
            if chain_seen and (not (backend and first) or (mask_seen and selected)):
                if backend and first:
                    return {"token_mask_chain_admitted": True, "backend_selector_executed": True,
                            "selected_token_log_count": text.count(SELECTED)}
                require(not mask_seen, "A reference/legacy request unexpectedly admitted the backend grammar mask")
                if backend and not first:
                    require(LEGACY_FALLBACK in text, "Legacy grammar-last request did not demonstrate its required fallback")
                return {"token_mask_chain_admitted": False, "backend_selector_executed": False,
                        "legacy_fallback": bool(backend and not first)}
            require(self.process.poll() is None, "Native server exited during HTTP feature validation")
            if time.monotonic() >= deadline:
                raise AssertionError("Required actual grammar-mask/selector log markers were not observed")
            time.sleep(.02)

    def __enter__(self):
        self.log = self.logfile.open("wb")
        try:
            self.process = subprocess.Popen(self.command, cwd=self.executable.parent, env=self.env,
                stdout=self.log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 60
            while True:
                require(self.process.poll() is None, f"Native server exited; inspect {self.logfile}")
                try:
                    if self.request("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Native server was not ready; inspect {self.logfile}")
                time.sleep(.05)
            if self.gpu:
                text = self.log_text()
                layers = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
                require(any(int(a) > 0 and a == b for a, b in layers) and
                        re.search(r"CUDA\d+\s+compute buffer size", text),
                        "GPU test did not demonstrate full model offload and CUDA compute allocation")
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()

    def __exit__(self, *unused):
        self.close()


def run(args, compare):
    before, after, cases, raw = [], [], [], []
    positive_cases = legacy_cases = negative_cases = 0
    # The common sampler and GGML softmax use different arithmetic reductions,
    # including on CPU. This bound applies only to probability metadata in this
    # same-binary, same-grammar-order algorithm comparison. Tokens, IDs, types,
    # schema and content remain exact; repeats/default grammar-last stay exact.
    tolerance = 1e-5
    name = "gpu" if args.gpu else "cpu"
    raw_path = args.output_dir / (name + "-responses.json")

    def dump():
        raw_path.write_text(json.dumps(dict(before=before, after=after, raw_requests=raw),
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def checked_compare(left, right, bound):
        try:
            return compare(left, right, bound)
        except BaseException:
            dump()
            raise

    with NativeServer(args.after, args.model, args.output_dir, args.gpu) as server:
        def completion(body, first, backend, expected_literal):
            start = server.logfile.stat().st_size
            response = server.request("/completion", body)
            entry = dict(request=body, response=response)
            raw.append(entry)
            try:
                stable = stable_response(response, first, backend, expected_literal, body["n_probs"])
                proof = server.wait_log(start, backend, first)
                entry["proof"] = proof
            except BaseException:
                dump()
                raise
            return stable, proof

        def body(grammar, temperature):
            return dict(prompt=[20, 21, 22, 23], n_predict=48, temperature=temperature, seed=424242,
                samplers=["top_k", "top_p", "temperature"], top_k=20, top_p=.9, min_p=0,
                repeat_penalty=1, frequency_penalty=0, presence_penalty=0, grammar=grammar,
                stream=False, cache_prompt=False, return_tokens=True, ignore_eos=False,
                n_probs=0 if temperature == 0 else 5, post_sampling_probs=True)

        previous = {}
        for temperature in (0.0, .6):
            for name, grammar, text in GRAMMARS:
                for repeat in range(2):
                    common = body(grammar, temperature)
                    left, _ = completion(dict(common, grammar_first=True, backend_sampling=False), True, False, text)
                    right, proof = completion(dict(common, grammar_first=True, backend_sampling=True), True, True, text)
                    metrics = checked_compare(left, right, tolerance)
                    key = (name, temperature)
                    if repeat:
                        checked_compare(previous[key][0], left, 0.0)
                        checked_compare(previous[key][1], right, 0.0)
                    previous[key] = (left, right)
                    before.append(left); after.append(right)
                    cases.append(dict(name=f"first-{name}-{temperature}-repeat{repeat}", grammar_first=True,
                        reference_backend_sampling=False, candidate_backend_sampling=True,
                        comparison=metrics, actual_feature=proof))
                    positive_cases += 1
                # Default grammar-last must match explicit false under the same
                # ordering. Never compare it with the grammar-first algorithm.
                common = dict(body(grammar, temperature), backend_sampling=True)
                left, _ = completion(common, False, True, text)
                right, proof = completion(dict(common, grammar_first=False), False, True, text)
                metrics = checked_compare(left, right, 0.0)
                before.append(left); after.append(right)
                cases.append(dict(name=f"default-last-{name}-{temperature}", grammar_first=False,
                    comparison=metrics, actual_feature=proof))
                legacy_cases += 1

        for name, invalid in (("invalid-grammar", dict(grammar="root ::= [", grammar_first=True)),
                              ("invalid-order-type", dict(grammar=literal("OK"), grammar_first="invalid"))):
            statuses = []
            for backend in (False, True):
                request = dict(body(literal("OK"), 0.0), **invalid, backend_sampling=backend)
                try:
                    server.request("/completion", request)
                except urllib.error.HTTPError as error:
                    require(error.code == 400, "Invalid grammar/order field must return HTTP400")
                    statuses.append({"invalid_request_status": error.code})
                else:
                    raise AssertionError("Invalid grammar/order type was silently accepted")
            before.append(statuses[0]); after.append(statuses[1])
            cases.append(dict(name=name, comparison=checked_compare(statuses[0], statuses[1], 0.0), negative=True))
            negative_cases += 1

    dump()
    comparison = compare(before, after, tolerance)
    require(positive_cases == 12 and legacy_cases == 6 and negative_cases == 2 and len(cases) == 20,
            "Feature case matrix was incomplete")
    return dict(status="passed", comparison_subject="Same final after binary; explicit same grammar-first contract, CPU sampling versus complete backend sampling",
        gpu=args.gpu, backend_sampler_device="CUDA" if args.gpu else "CPU", speculative=False,
        cases_per_mode=len(cases), positive_feature_cases=positive_cases, legacy_grammar_last_cases=legacy_cases,
        invalid_request_cases=negative_cases, actual_positive_mask_and_selector_proof=True,
        fixture_bytes=args.model.stat().st_size, fixture_sha256=sha(args.model), executable_sha256=sha(args.after),
        loaded_backend_sha256=sha(server.backend), server_log_sha256=sha(server.logfile),
        before_outputs_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        outputs_sha256=hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
        response_dump_sha256=sha(raw_path), grammar_first_default=False, repeated_requests_exact=True,
        default_vs_explicit_grammar_last_exact=True, throughput_measured=False,
        probability_comparison_scope="Same-binary common CPU sampler versus backend sampler only; finite probability metadata absolute1e-5 on CPU/CUDA, all tokens/content/IDs/types/schema exact; repeats and default grammar-last exact",
        comparison=comparison, cases=cases,
        scope="Tiny model semantic/API proof only. No cross-order comparison, skipped/fallback-only positive case or model-speed claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    require(args.output_dir.is_absolute(), "An explicit absolute output directory is required")
    args.after, args.model, args.output_dir = args.after.resolve(), args.model.resolve(), args.output_dir.resolve()
    require(args.after.is_file(), "Final native server executable is missing")
    require(args.model.is_file() and 0 < args.model.stat().st_size <= 4 * 1024 * 1024,
            "Use a local synthetic GGUF at most4MiB; this harness never downloads a model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / ("gpu.json" if args.gpu else "cpu.json")
    report_path.write_text(json.dumps(dict(status="running", gpu=args.gpu)) + "\n", encoding="utf-8")
    compare, source = comparator()
    try:
        report = run(args, compare)
        report["strict_comparator_sha256"] = sha(source)
        report["harness_sha256"] = sha(Path(__file__).resolve())
    except BaseException as error:
        report_path.write_text(json.dumps(dict(status="failed", gpu=args.gpu, error=str(error)), indent=2) + "\n", encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "gpu", "cases_per_mode", "positive_feature_cases",
        "legacy_grammar_last_cases", "invalid_request_cases", "before_outputs_sha256", "outputs_sha256", "comparison")}))


if __name__ == "__main__":
    main()
