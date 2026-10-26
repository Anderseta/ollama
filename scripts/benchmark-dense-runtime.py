#!/usr/bin/env python3
"""Run isolated dense Qwen cold/warm comparisons, sharing only hardlinked weights.

--dry-run validates the existing laboratory and prints the plan without writing
files, probing CUDA or starting servers. --prepare-only links missing model blobs
and copies only small manifests into a separate same-volume laboratory. Neither
operation copies model weights or changes the production store.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
HELPER_PATH = Path(__file__).with_name("benchmark-gpu-first.py")
SPEC = importlib.util.spec_from_file_location("wildllama_benchmark_helpers", HELPER_PATH)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)

SHORT_PROMPT = (
    "Escreva um parágrafo com pelo menos 180 palavras explicando como organizar "
    "cópias de segurança de documentos pessoais. Use linguagem clara, exemplos "
    "práticos e texto contínuo, sem listas."
)
LONG_PROMPT = """Uma biblioteca de bairro está digitalizando seu acervo. A equipe tem três
computadores, um servidor local e uma conexão com a internet que costuma ficar
indisponível durante algumas horas. Os documentos incluem atas, fotografias,
jornais e pequenos livros produzidos pela comunidade. Há arquivos recentes e
outros que não são modificados há muitos anos. A biblioteca quer permitir buscas
rápidas, preservar os documentos originais e continuar atendendo mesmo quando
a internet não funciona.

O computador da recepção atende consultas do público. Outro computador recebe
novos documentos e confere a qualidade das digitalizações. O terceiro fica
reservado para revisão e organização dos nomes dos arquivos. Pessoas diferentes
podem trabalhar no mesmo dia, por isso o processo precisa ser simples de seguir.
Alguns nomes de arquivos são parecidos e certos documentos aparecem em mais de
uma pasta. A equipe também precisa registrar a origem e a data aproximada dos
materiais, sem preencher dezenas de campos para cada item.

O servidor tem um disco rápido para as consultas e outro disco para cópias de
segurança. Um disco removível pode ser guardado em outro prédio toda semana.
Não existe dinheiro para comprar novos computadores neste semestre. É possível
reservar algum tempo ao final de cada dia para tarefas automáticas, desde que
elas possam continuar depois de uma interrupção e mostrem claramente o progresso.
Durante o expediente, a prioridade é responder às consultas sem bloquear os
computadores usados pelas pessoas.

Também é necessário pensar nos erros mais comuns: um arquivo pode ser apagado
por engano, uma digitalização pode ficar ilegível e uma cópia pode ser feita
antes de todos os documentos terminarem de chegar. A equipe quer verificar se
os backups podem realmente ser restaurados, em vez de apenas confiar na mensagem
de sucesso. A solução precisa explicar o que conferir diariamente, o que conferir
semanalmente e como recuperar um documento sem alterar os outros.

Com esse cenário, escreva uma recomendação prática em português, com pelo menos
220 palavras. Priorize organização, verificação e recuperação. Explique as razões
das decisões, sem inventar números de desempenho ou depender de serviços pagos.
Use parágrafos conectados e comece pela ação mais importante."""


def model_location(name):
    stem, colon, tag = name.rpartition(":")
    if not colon or "/" in tag:
        stem, tag = name, "latest"
    parts = stem.split("/")
    if len(parts) == 1:
        parts = ["ollama.com", "library", parts[0]]
    elif len(parts) == 2:
        parts = ["ollama.com", *parts]
    if len(parts) != 3 or any(not re.fullmatch(r"[A-Za-z0-9._-]+", p) or p in (".", "..") for p in (*parts, tag)):
        raise ValueError("Use a local model name with safe registry/namespace/name:tag components")
    return Path(*parts, tag)


def find_manifest(root, name):
    relative = model_location(name)
    paths = [root / "models/manifests-v2" / relative,
             root / "models/manifests" / relative]
    if relative.parts[0] == "ollama.com":
        paths.append(root / "models/manifests" / "registry.ollama.ai" / Path(*relative.parts[1:]))
    for path in paths:
        if path.is_file():
            return path
    raise ValueError(f"Selected model manifest is missing in {root}; prepare its isolated hardlinks first")


def manifest_blobs(path):
    data = json.loads(path.read_text(encoding="utf8"))
    if "layers" not in data or "config" not in data:
        raise ValueError("This benchmark requires a resolved local GGUF model manifest")
    entries = [data["config"], *data["layers"]]
    for entry in entries:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", entry.get("digest", "")):
            raise ValueError("Unsafe or unsupported model blob digest")
    weights = {entry["digest"] for entry in data["layers"]
               if entry.get("mediaType") == "application/vnd.ollama.image.model"}
    if not weights:
        raise ValueError("Model manifest contains no GGUF weight layer")
    return entries, weights


def separate_paths(source, lab, payloads, output):
    if source == lab or source in lab.parents or lab in source.parents:
        raise ValueError("Production and laboratory roots must be separate")
    for path in (*payloads, output):
        if path == source or source in path.parents:
            raise ValueError("Payload execution and reports must stay outside production")
    if os.name == "nt" and any(path.drive.upper() == "C:" for path in (lab, output)):
        raise ValueError("Laboratory and report writes must not use C:")


def inspect_store(source, lab, name):
    source_manifest = find_manifest(source, name)
    lab_manifest = find_manifest(lab, name)
    _, expected = manifest_blobs(source_manifest)
    entries, actual = manifest_blobs(lab_manifest)
    if actual != expected:
        raise ValueError("Laboratory and production select different model weights")
    weights = []
    for entry in entries:
        filename = entry["digest"].replace(":", "-")
        local = lab / "models/blobs" / filename
        if not local.is_file():
            raise ValueError(f"Missing laboratory blob: {filename}")
        if entry["digest"] in actual:
            original = source / "models/blobs" / filename
            if not original.is_file() or not os.path.samefile(original, local):
                raise ValueError("Selected laboratory weights are not hardlinks to the production store")
            weights.append({"digest": entry["digest"], "bytes": local.stat().st_size, "hardlinked": True})
    return {"source_manifest": str(source_manifest), "lab_manifest": str(lab_manifest),
            "manifest_sha256": HELPERS.sha256(lab_manifest), "weights": weights, "duplicate_weight_bytes": 0}


def prepare_store(source, lab, name):
    source_manifest = find_manifest(source, name)
    entries, _ = manifest_blobs(source_manifest)
    marker = lab / "gpu-first-lab.json"
    if marker.exists() and Path(json.loads(marker.read_text(encoding="utf8"))["source"]).resolve() != source:
        raise ValueError("Existing laboratory belongs to a different production root")
    lab.mkdir(parents=True, exist_ok=True)
    if (source / "models").stat().st_dev != lab.stat().st_dev:
        raise ValueError("Laboratory and model store must share a filesystem for hardlinks")
    blobs = lab / "models/blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        filename = entry["digest"].replace(":", "-")
        original, target = source / "models/blobs" / filename, blobs / filename
        if target.exists():
            if not os.path.samefile(original, target):
                raise ValueError("Refusing to replace an existing non-hardlinked blob")
        else:
            os.link(original, target)
    # Existing lab profiles may differ; never overwrite one silently. Requests
    # explicitly replace load-affecting options equally for both payloads.
    target = lab / "models/manifests-v2" / model_location(name)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_manifest, target)
    if not marker.exists():
        marker.write_text(json.dumps({"purpose": "dense runtime isolated benchmark", "source": str(source),
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "duplicate_weight_bytes": 0}, indent=2), encoding="utf8")
    return inspect_store(source, lab, name)


def native_observation(log_path, context):
    sizes, samples = [], []
    for line in log_path.read_text(encoding="utf8", errors="replace").splitlines():
        match = re.search(r"llama_context:\s+n_ctx\s*=\s*(\d+)", line)
        if match:
            sizes.append(int(match.group(1)))
        if line.startswith("wildllama_activity "):
            try:
                event = json.loads(line[len("wildllama_activity "):])
                if event.get("kind") == "memory" and event.get("stage") == "cuda":
                    samples.append(event["memory"])
            except (ValueError, KeyError):
                pass
    if not sizes or any(size != context for size in sizes):
        raise RuntimeError(f"Native logical context differs or is unverified: {sizes}")
    return {"native_contexts": sizes, "managed_cuda": {
        "scope": "Native allocator of the runner; excludes other processes and driver allocations",
        "samples": len(samples), "sampled_peak_bytes": max((x["vram_bytes"] for x in samples), default=None),
        "last": samples[-1] if samples else None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-payload", type=Path, required=True)
    parser.add_argument("--candidate-payload", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("F:/WildLlama"))
    parser.add_argument("--lab-root", type=Path, default=Path("F:/WildLlama-GpuFirst-Lab"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5-9B-UNCENSORED-Q8_0:latest")
    parser.add_argument("--label", default="dense-runtime")
    parser.add_argument("--port", type=int, default=11448)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--baseline-dense-opt", choices=(0, 1), type=int, default=0)
    parser.add_argument("--candidate-dense-opt", choices=(0, 1), type=int, default=0,
                        help="Enable explicitly for a kernel ablation; payload comparisons default to equal options")
    parser.add_argument("--baseline-kv-initial-tokens", type=int, default=0)
    parser.add_argument("--candidate-kv-initial-tokens", type=int, default=0,
                        help="Set explicitly for a KV-growth ablation; payload comparisons default to equal options")
    parser.add_argument("--gpu-vram-limit-mib", type=int, default=0, help="Optional managed quota, e.g. 12288 for 12 GiB")
    parser.add_argument("--gpu-exclusive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--free-ram-guard-gib", type=float, default=4)
    parser.add_argument("--prompts", choices=("short", "long", "both"), default="both")
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    source, lab, output = args.source_root.resolve(), args.lab_root.resolve(), args.output.resolve()
    payloads = [args.baseline_payload.resolve(), args.candidate_payload.resolve()]
    separate_paths(source, lab, payloads, output)
    if not args.label or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise ValueError("Report label must contain only letters, digits, hyphens or underscores")
    if args.port in (11434, 11446, 11447) or not 1024 <= args.port <= 65535:
        raise ValueError("Use an isolated laboratory port, normally 11448")
    if args.rounds < 2 or args.predict < 1 or args.context < 4 or not 1 <= args.ubatch <= args.batch:
        raise ValueError("Require rounds >=2, positive prediction/context and 1 <= ubatch <= batch")
    if args.free_ram_guard_gib < 4 or not 0 <= args.gpu_vram_limit_mib <= 2097151:
        raise ValueError("Keep at least the 4 GiB free-RAM guard and a valid optional GPU quota")
    for initial in (args.baseline_kv_initial_tokens, args.candidate_kv_initial_tokens):
        if initial != 0 and not 256 <= initial <= 2097152:
            raise ValueError("KV initial commitment must be 0 or 256..2097152")
    if args.prepare_only:
        if args.dry_run:
            raise ValueError("--prepare-only writes the isolated lab; do not combine it with --dry-run")
        print(json.dumps(prepare_store(source, lab, args.model), indent=2))
        return
    for payload in payloads:
        if not (payload / "ollama.exe").is_file():
            raise ValueError(f"Missing isolated payload executable: {payload}")
    inventory = inspect_store(source, lab, args.model)
    if not (lab / "gpu-first-lab.json").is_file():
        raise ValueError("Laboratory marker missing; use --prepare-only")
    prompts = [("short", SHORT_PROMPT), ("long", LONG_PROMPT)]
    if args.prompts != "both":
        prompts = [item for item in prompts if item[0] == args.prompts]
    variants = [("baseline", payloads[0], args.baseline_dense_opt, args.baseline_kv_initial_tokens),
                ("candidate", payloads[1], args.candidate_dense_opt, args.candidate_kv_initial_tokens)]
    if args.candidate_first:
        variants.reverse()
    plan = []
    for prompt_index, (prompt_name, prompt) in enumerate(prompts):
        # Reverse payload order for the second prompt to reduce simple order bias.
        for variant, payload, dense_opt, initial in (variants if prompt_index == 0 else reversed(variants)):
            label = f"{args.label}-{variant}-{prompt_name}"
            command = ["--payload", str(payload), "--lab-root", str(lab), "--output", str(output),
                       "--port", str(args.port), "--model", args.model, "--label", label,
                       "--model-mode", "dense", "--context", str(args.context), "--batch", str(args.batch),
                       "--ubatch", str(args.ubatch), "--predict", str(args.predict), "--rounds", str(args.rounds),
                       "--dense-opt", str(dense_opt), "--kv-cache-initial-tokens", str(initial),
                       "--gpu-vram-limit-mib", str(args.gpu_vram_limit_mib),
                       "--free-ram-guard-gib", str(args.free_ram_guard_gib), "--prompt", prompt]
            if args.gpu_exclusive:
                command.append("--gpu-exclusive")
            plan.append({"variant": variant, "prompt_case": prompt_name, "label": label, "argv": command})
    suite = {"kind": "dense-runtime-ablation", "cold_definition": "New server/runner; OS page cache remains warm",
             "model_store": inventory, "fixed_policy": {"context": args.context, "batch": args.batch,
                 "ubatch": args.ubatch, "kv_type": "q8_0", "flash_attention": True, "fit": "off",
                 "num_gpu": -1, "gpu_exclusive": args.gpu_exclusive, "gpu_vram_limit_mib": args.gpu_vram_limit_mib,
                 "temperature": 0, "seed": 42, "predict": args.predict, "think": False, "use_mmap": False},
             "prompt_lengths": "Natural prose; actual token lengths are recorded, not assumed to equal 32/512",
             "plan": plan, "runs": []}
    if args.dry_run:
        print(json.dumps(suite, indent=2, ensure_ascii=False))
        return
    output.mkdir(parents=True, exist_ok=True)
    suite_path = output / (args.label + "-suite.json")
    if suite_path.exists():
        raise ValueError("Refusing to overwrite an existing dense benchmark suite")
    try:
        for item in plan:
            HELPERS.main(item["argv"])
            report_path = output / (item["label"] + "-results.json")
            report = json.loads(report_path.read_text(encoding="utf8"))
            observation = native_observation(output / (item["label"] + "-server.log"), args.context)
            suite["runs"].append({"variant": item["variant"], "prompt_case": item["prompt_case"],
                "report": str(report_path), "ok": report.get("ok", False), **observation,
                "results": [{key: row.get(key) for key in ("case", "tokens_s", "decode_steps_s", "ttft_s", "elapsed_s",
                    "load_s", "prompt_eval_s", "eval_s", "eval_count", "prompt_tokens", "cached_tokens")} for row in report["results"]]})
            suite_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf8")
        suite["ok"] = True
    except BaseException as exc:
        suite["error"] = repr(exc)
        raise
    finally:
        suite["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        suite_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
