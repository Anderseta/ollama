#!/usr/bin/env python3
"""Isolated runner-cold/warm Flash benchmark; never changes the production store.

Preparation copies only the executable/native payload and small metadata. Model
blobs are hard links on the same volume, never copies. Run as a normal Python
process: it owns and stops its server tree, including on a failed memory guard.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


def physical_memory():
    if os.name != "nt":
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return {"total": int(values["MemTotal"].split()[0]) * 1024,
                "available": int(values["MemAvailable"].split()[0]) * 1024}
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "available", "pagefile_total",
             "pagefile_available", "virtual_total", "virtual_available", "extended")]
    state = Memory()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return {"total": state.total, "available": state.available}


class WindowsTreeMemory:
    """Toolhelp descendants + current working set/private commit, without WMI."""
    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows process memory sampler is unavailable on this platform")
        from ctypes import wintypes
        class Entry(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                        ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                        ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                        ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                        ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]
        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_ws", "working_set", "peak_paged_pool",
                "paged_pool", "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private")]
        self.Entry, self.Counters = Entry, Counters
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
            "Process32FirstW": ([wintypes.HANDLE, ctypes.POINTER(Entry)], wintypes.BOOL),
            "Process32NextW": ([wintypes.HANDLE, ctypes.POINTER(Entry)], wintypes.BOOL),
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "K32GetProcessMemoryInfo": ([wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    def sample(self, root_pid):
        snapshot = self.kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        entries = {}
        entry = self.Entry()
        entry.size = ctypes.sizeof(entry)
        try:
            found = self.kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while found:
                entries[entry.pid] = {"pid": entry.pid, "parent_pid": entry.parent, "name": entry.name}
                found = self.kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            self.kernel.CloseHandle(snapshot)
        pids = {root_pid}
        while True:
            children = {pid for pid, value in entries.items() if value["parent_pid"] in pids}
            if children <= pids:
                break
            pids.update(children)
        processes, skipped = [], []
        for pid in sorted(pids):
            handle = self.kernel.OpenProcess(0x1000 | 0x10, False, pid)
            if not handle:
                skipped.append(pid)
                continue
            counters = self.Counters()
            counters.size = ctypes.sizeof(counters)
            try:
                if self.kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.size):
                    processes.append({**entries.get(pid, {"pid": pid}),
                                      "working_set_bytes": counters.working_set,
                                      "private_usage_commit_bytes": counters.private})
                else:
                    skipped.append(pid)
            finally:
                self.kernel.CloseHandle(handle)
        return {"working_set_sum_bytes": sum(p["working_set_bytes"] for p in processes),
                "private_usage_commit_sum_bytes": sum(p["private_usage_commit_bytes"] for p in processes),
                "processes": processes, "unreadable_or_exited_pids": skipped}


class NVMLDeviceMemory:
    """Device-wide totals, including display/other processes, never runner quota."""
    def __init__(self):
        self.library = None
        self.error = None
        self.devices = []
        class Memory(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]
        self.Memory = Memory
        try:
            library = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
            library.nvmlInit_v2.argtypes, library.nvmlInit_v2.restype = [], ctypes.c_int
            library.nvmlShutdown.argtypes, library.nvmlShutdown.restype = [], ctypes.c_int
            library.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            library.nvmlDeviceGetCount_v2.restype = ctypes.c_int
            library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory)]
            library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            library.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            library.nvmlDeviceGetUUID.restype = ctypes.c_int
            status = library.nvmlInit_v2()
            if status:
                raise OSError(f"nvmlInit_v2 returned {status}")
            self.library = library
            count = ctypes.c_uint()
            if library.nvmlDeviceGetCount_v2(ctypes.byref(count)):
                raise OSError("NVML device count failed")
            for index in range(count.value):
                handle = ctypes.c_void_p()
                if library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)):
                    continue
                uuid = ctypes.create_string_buffer(96)
                status = library.nvmlDeviceGetUUID(handle, uuid, len(uuid))
                self.devices.append((index, uuid.value.decode() if not status else f"nvml-index-{index}", handle))
        except (OSError, AttributeError) as exc:
            self.error = str(exc)
            self.close()

    def sample(self):
        result = []
        if not self.library:
            return result
        for index, uuid, handle in self.devices:
            memory = self.Memory()
            status = self.library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            if status:
                result.append({"nvml_index": index, "uuid": uuid, "error_code": status})
            else:
                result.append({"nvml_index": index, "uuid": uuid, "total_bytes": memory.total,
                               "used_bytes": memory.used, "free_bytes": memory.free})
        return result

    def close(self):
        if self.library:
            self.library.nvmlShutdown()
            self.library = None


def run_output(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=15,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, payload, lab):
    if payload.exists():
        raise ValueError("Refusing to overwrite an existing payload snapshot")
    if lab.exists() and any(lab.iterdir()):
        raise ValueError("Refusing to prepare a nonempty laboratory")
    if source == lab or source in lab.parents or lab in source.parents:
        raise ValueError("Production and laboratory directories must be separate")
    if source == payload or source in payload.parents:
        raise ValueError("Refusing to write a payload snapshot into production")
    if (source / "models").stat().st_dev != lab.parent.stat().st_dev:
        raise ValueError("Model source and lab must share a filesystem for hard links")
    payload.mkdir(parents=True)
    shutil.copy2(source / "ollama.exe", payload / "ollama.exe")
    shutil.copytree(source / "lib", payload / "lib")
    models = lab / "models"
    models.mkdir(parents=True)
    for name in ("manifests", "manifests-v2", "metadata"):
        if (source / "models" / name).exists():
            shutil.copytree(source / "models" / name, models / name)
    blobs = models / "blobs"
    blobs.mkdir()
    shared_bytes = 0
    linked = 0
    for original in (source / "models" / "blobs").iterdir():
        if not original.is_file():
            raise ValueError(f"Unexpected non-file in blob directory: {original}")
        os.link(original, blobs / original.name)
        shared_bytes += original.stat().st_size
        linked += 1
    marker = {"purpose": "gpu-first-moe isolated benchmark", "source": str(source),
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "shared_blob_bytes": shared_bytes, "hardlinked_blobs": linked,
              "duplicate_weight_bytes": 0}
    (lab / "gpu-first-lab.json").write_text(json.dumps(marker, indent=2), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-from", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--port", type=int, default=11448)
    parser.add_argument("--model", default="qwen3.8-flash-next-uncensored:q4_k_m")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--predict", type=int, default=8)
    parser.add_argument("--prompt", default="Responda apenas OK.")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--moe-ram-mib", type=int, default=6144)
    parser.add_argument("--moe-gpu-cache-mib", type=int, default=1024)
    parser.add_argument("--fit-margin-mib", type=int, default=4096)
    parser.add_argument("--gpu-exclusive", action="store_true")
    parser.add_argument("--gpu-vram-limit-mib", type=int, default=0)
    parser.add_argument("--gpu-diagnostics", action="store_true")
    parser.add_argument("--moe-runtime", action="store_true")
    parser.add_argument("--moe-dynamic-prefill", action="store_true")
    parser.add_argument("--fixed-expert-cache", action="store_true")
    parser.add_argument("--expert-trace-path", type=Path)
    parser.add_argument("--graph-input-debug", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0,
                        help="Keep the successful model loaded for observation after all timed requests")
    parser.add_argument("--free-ram-guard-gib", type=float, default=4)
    args = parser.parse_args()
    payload, lab = args.payload.resolve(), args.lab_root.resolve()
    output = args.output.resolve()
    if args.port in (11434, 11446, 11447) or not 1024 <= args.port <= 65535:
        raise ValueError("Use a separate laboratory port (default 11448)")
    if not args.label or any(not (c.isalnum() or c in "-_") for c in args.label):
        raise ValueError("Invalid report label")
    if args.rounds < 1 or args.predict < 1:
        raise ValueError("Positive rounds and prediction length required")
    if not 0 <= args.hold_seconds <= 60:
        raise ValueError("Observation hold must be between 0 and 60 seconds")
    if args.prepare_from:
        prepare(args.prepare_from.resolve(), payload, lab)
    if args.prepare_only:
        return
    if not (lab / "gpu-first-lab.json").is_file():
        raise ValueError("Laboratory marker missing; prepare an isolated lab first")
    production = Path(json.loads((lab / "gpu-first-lab.json").read_text(encoding="utf8"))["source"]).resolve()
    if payload == production or production in payload.parents:
        raise ValueError("Refusing to execute the production payload")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise ValueError("Laboratory port is already occupied")
    output.mkdir(parents=True, exist_ok=True)
    report_file = output / (args.label + "-results.json")
    log_file = output / (args.label + "-server.log")
    if report_file.exists() or log_file.exists():
        raise ValueError("Refusing to overwrite benchmark results; select another label")
    binary = payload / "ollama.exe"
    if args.gpu_exclusive or args.gpu_diagnostics or args.gpu_vram_limit_mib or args.moe_runtime:
        # Older API servers can ignore unknown JSON options. Require the new
        # executable flags so a legacy run cannot be mislabeled GPU-exclusive.
        help_text = run_output([str(binary), "serve", "--help"])
        required = ["--gpu-exclusive"] if args.gpu_exclusive else []
        if args.gpu_diagnostics:
            required.append("--gpu-diagnostics")
        if args.gpu_vram_limit_mib:
            required.append("--gpu-vram-limit-mib")
        if args.moe_runtime:
            required.append("--moe-runtime")
        if any(flag not in help_text for flag in required):
            raise ValueError("This payload does not advertise the requested GPU-first options")
    report = {"label": args.label, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cold_definition": "New server/runner; OS page cache is not flushed", "results": [],
              "throughput_definition": "tokens_s = Ollama eval_count / eval_duration; very short outputs include too few decode intervals for sustained throughput claims",
              "hardware": {"memory": physical_memory(), "gpu": run_output(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"]),
                "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "")},
              "payload_sha256": {str(p.relative_to(payload)): sha256(p)
                    for p in sorted(payload.rglob("*")) if p.is_file() and p.suffix.lower() in (".exe", ".dll")}}
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            report["hardware"]["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    def save():
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OLLAMA_", "LLAMA_"))}
    paths = {"USERPROFILE": lab / "home", "HOME": lab / "home", "OLLAMA_MODELS": lab / "models",
             "LOCALAPPDATA": lab / "home/AppData/Local", "APPDATA": lab / "home/AppData/Roaming",
             "TEMP": lab / "runtime/temp", "TMP": lab / "runtime/temp", "TMPDIR": lab / "runtime/temp",
             "CUDA_CACHE_PATH": lab / "runtime/cuda"}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update({"OLLAMA_HOST": f"127.0.0.1:{args.port}", "OLLAMA_NO_CLOUD": "1", "OLLAMA_NOPRUNE": "1",
                "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "-1",
                "OLLAMA_CONTEXT_LENGTH": str(args.context), "OLLAMA_FLASH_ATTENTION": "1",
                "OLLAMA_KV_CACHE_TYPE": "q8_0", "LLAMA_ARG_CACHE_RAM": "1",
                "LLAMA_ARG_MOE_RAM_LIMIT_MIB": str(args.moe_ram_mib), "LLAMA_ARG_LOAD_MODE": "mmap",
                "LLAMA_ARG_LAZY_MODE": "on", "LLAMA_ARG_CPU_MOE": "1",
                "LLAMA_ARG_MOE_CACHE_MIB": str(args.moe_gpu_cache_mib), "LLAMA_ARG_N_GPU_LAYERS": "auto",
                "LLAMA_ARG_FIT": "on", "LLAMA_ARG_FIT_TARGET": str(args.fit_margin_mib),
                "LLAMA_ARG_BATCH": "512", "LLAMA_ARG_UBATCH": "32",
                "LLAMA_ARG_OVERRIDE_TENSOR": "per_layer_token_embd=CPU", "PYTHONDONTWRITEBYTECODE": "1"})
    if args.gpu_exclusive:
        # Strict mode rejects explicit CPU tensor placement and explicit partial offload.
        env.pop("LLAMA_ARG_OVERRIDE_TENSOR")
        env["LLAMA_ARG_N_GPU_LAYERS"] = "all"
    env["OLLAMA_MOE_RUNTIME"] = "1" if args.moe_runtime else "0"
    env["WILDLLAMA_MOE_RUNTIME"] = "1" if args.moe_runtime else "0"
    env["WILDLLAMA_MOE_DYNAMIC_PREFILL"] = "1" if args.moe_dynamic_prefill else "0"
    env["WILDLLAMA_MOE_CACHE_AUTO"] = "0" if args.fixed_expert_cache else "1"
    if args.expert_trace_path:
        trace_path = args.expert_trace_path.resolve()
        if trace_path.exists():
            raise ValueError("Refusing to overwrite expert trace")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        env["WILDLLAMA_EXPERT_TRACE"] = str(trace_path)
    else:
        env.pop("WILDLLAMA_EXPERT_TRACE", None)
    if args.graph_input_debug:
        env["LLAMA_GRAPH_INPUT_DEBUG"] = "1"
    report["runner_environment"] = {k: v for k, v in env.items() if k.startswith(("OLLAMA_", "LLAMA_", "WILDLLAMA_"))}
    report["manifest_sha256"] = {str(p.relative_to(lab / "models")): sha256(p)
        for directory in ("manifests", "manifests-v2")
        for p in sorted((lab / "models" / directory).rglob("*")) if p.is_file()}
    save()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(route, body=None, timeout=30):
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}" + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            return opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf8", errors="replace")
            error = {"route": route, "status": exc.code, "body": detail}
            report.setdefault("api_errors", []).append(error)
            save()
            print(json.dumps({"api_error": error}, ensure_ascii=False), flush=True)
            raise
    if physical_memory()["available"] < args.free_ram_guard_gib * 1024**3:
        raise RuntimeError("Insufficient free RAM before server launch")
    done = threading.Event()
    process = None
    guard_thread = None
    device_memory = NVMLDeviceMemory()
    try:
        tree_memory = WindowsTreeMemory()
    except OSError as exc:
        tree_memory = None
        report["process_memory_sampling_error"] = str(exc)
    report["process_memory_observation"] = {
        "method": "Toolhelp32Snapshot + K32GetProcessMemoryInfo",
        "scope": "Laboratory server PID and its currently observed descendants only",
        "working_set_semantics": "Sum of process working sets; shared pages can be counted more than once, so this is not unique physical RAM",
        "private_usage_semantics": "Sum of PrivateUsage commit charge, including potentially nonresident pages; not physical RAM or model-weight residency",
        "interval_s": 1, "observed_peak_working_set_sum_bytes": None,
        "observed_peak_private_usage_commit_sum_bytes": None, "samples": 0}
    report["device_memory_observation"] = {"method": "NVML nvmlDeviceGetMemoryInfo (ctypes, no extra package)",
        "scope": "All processes, display and driver allocations on each physical GPU; this is not the runner's managed allocation quota",
        "interval_s": 1, "initialization_error": device_memory.error, "observed_peak_used_bytes": {}}
    report["resource_samples"] = []
    observation_started = time.monotonic()
    def stop():
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.terminate()
            process.wait(timeout=30)
    def guard():
        while not done.wait(1):
            memory = physical_memory()
            report["min_available_ram_bytes"] = min(memory["available"], report.get("min_available_ram_bytes", memory["available"]))
            if memory["available"] < args.free_ram_guard_gib * 1024**3:
                report["guard_error"] = "Free RAM fell below the configured guard; stopped only the laboratory process tree"
                stop()
                return
            sample = {"elapsed_s": time.monotonic() - observation_started, "available_physical_ram_bytes": memory["available"]}
            sampling_started = time.monotonic()
            try:
                if tree_memory and process is not None:
                    tree = tree_memory.sample(process.pid)
                    sample["process_tree"] = tree
                    observation = report["process_memory_observation"]
                    observation["samples"] += 1
                    for metric, source in (("observed_peak_working_set_sum_bytes", "working_set_sum_bytes"),
                                           ("observed_peak_private_usage_commit_sum_bytes", "private_usage_commit_sum_bytes")):
                        observation[metric] = max(tree[source], observation[metric] or 0)
                devices = device_memory.sample()
                sample["device_memory"] = devices
                peaks = report["device_memory_observation"]["observed_peak_used_bytes"]
                for device in devices:
                    if "used_bytes" in device:
                        peaks[device["uuid"]] = max(peaks.get(device["uuid"], 0), device["used_bytes"])
            except OSError as exc:
                sample["sampling_error"] = str(exc)
            sample["sampling_elapsed_ms"] = (time.monotonic() - sampling_started) * 1000
            report["resource_samples"].append(sample)
    try:
        with log_file.open("wb") as log:
            process = subprocess.Popen([str(binary), "serve"], cwd=lab / "runtime", env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            report["pid"] = process.pid
            save()
            guard_thread = threading.Thread(target=guard, daemon=True)
            guard_thread.start()
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError("Laboratory server exited; inspect server log")
                try:
                    with api("/api/version", timeout=2) as response:
                        report["version"] = json.load(response)
                    break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError("Laboratory server did not become ready")
            print(json.dumps({"started": args.label, "pid": process.pid, "port": args.port}), flush=True)
            for run in range(args.rounds):
                options = {"num_ctx": args.context, "num_batch": 0, "moe_ram_limit_mib": args.moe_ram_mib,
                           "num_predict": args.predict, "temperature": 0, "seed": 42,
                           "moe_runtime": args.moe_runtime}
                if args.gpu_exclusive:
                    options.update(gpu_exclusive=True, gpu_vram_limit_mib=args.gpu_vram_limit_mib)
                elif args.gpu_vram_limit_mib:
                    options["gpu_vram_limit_mib"] = args.gpu_vram_limit_mib
                if args.gpu_diagnostics:
                    options["gpu_diagnostics"] = True
                body = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
                        "stream": True, "think": False, "keep_alive": -1, "options": options}
                started, first = time.monotonic(), None
                resource_start_index = len(report["resource_samples"])
                text, thinking, final = "", "", {}
                with api("/api/chat", body, timeout=900) as response:
                    for line in response:
                        item = json.loads(line)
                        if item.get("error"):
                            raise RuntimeError(item["error"])
                        message = item.get("message", {})
                        if (message.get("content") or message.get("thinking")) and first is None:
                            first = time.monotonic() - started
                            with api("/api/ps", timeout=5) as state_response:
                                loaded = json.load(state_response)
                            report["first_token_observation"] = {"run": run, "ttft_s": first,
                                "piece": message.get("content") or message.get("thinking"), "loaded_models": loaded}
                            save()
                            print(json.dumps({"first_token": report["first_token_observation"]}, ensure_ascii=False), flush=True)
                        text += message.get("content", "")
                        thinking += message.get("thinking", "")
                        if item.get("done"):
                            final = item
                duration = final.get("eval_duration", 0)
                result = {"case": "runner-cold" if run == 0 else f"warm-{run}",
                          "elapsed_s": time.monotonic() - started, "ttft_s": first,
                          "tokens_s": final.get("eval_count", 0) * 1e9 / duration if duration else None,
                          "decode_steps_s": max(0, final.get("eval_count", 0) - 1) * 1e9 / duration if duration else None,
                          "load_s": final.get("load_duration", 0) / 1e9,
                          "prompt_eval_s": final.get("prompt_eval_duration", 0) / 1e9,
                          "eval_s": duration / 1e9, "text": text, "thinking": thinking,
                          "eval_count": final.get("eval_count"), "prompt_tokens": final.get("prompt_eval_count"),
                          "resource_sample_start_index": resource_start_index,
                          "resource_sample_end_index": len(report["resource_samples"]),
                          "cached_tokens": final.get("prompt_eval_cached_count"), "request": body, "final": final}
                report["results"].append(result)
                save()
                print(json.dumps({k: v for k, v in result.items() if k not in ("request", "final", "text", "thinking")}), flush=True)
                with api("/api/ps") as response:
                    report["loaded_models"] = json.load(response)
            report["ok"] = True
            if args.hold_seconds:
                report["observation_hold_started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                save()
                print(json.dumps({"observation_hold_s": args.hold_seconds, "pid": process.pid}), flush=True)
                done.wait(args.hold_seconds)
                report["observation_hold_finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        done.set()
        stop()
        if guard_thread:
            guard_thread.join(timeout=5)
        device_memory.close()
        report["stopped"] = process is not None and process.poll() is not None
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
