"""Generate a deterministic dense GGUF fixture for native decode comparisons.

The model contains synthetic weights, a byte tokenizer and unused vocabulary
rows. It is a diagnostic fixture, not a trained model or a quality evaluation.
No packages, models or other resources are downloaded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError("Run without Python optimization; diagnostic model size checks are required")
sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output_dir, native_source, vocab):
    if vocab not in (32768, 131072):
        raise ValueError("Use one of the bounded diagnostic vocabulary sizes")
    if not output_dir.is_absolute():
        raise ValueError("An absolute output directory is required")
    output_dir, native_source = output_dir.resolve(), native_source.resolve()
    if output_dir.drive.upper() == "C:":
        raise ValueError("Diagnostic output/temp/cache files must be outside C:")
    package = native_source / "gguf-py"
    if not (package / "gguf/__init__.py").is_file():
        raise FileNotFoundError("The selected native source does not contain gguf-py")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / "temp"
    temp.mkdir(exist_ok=True)
    for variable in ("TEMP", "TMP", "TMPDIR"):
        os.environ[variable] = str(temp)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(package))
    import gguf
    import numpy as np

    if package not in Path(gguf.__file__).resolve().parents:
        raise RuntimeError("gguf was imported from a different source tree")
    output = output_dir / f"dense-vocab-{vocab}.gguf"
    pending = output.with_suffix(".gguf.partial")
    if output.exists() or pending.exists():
        raise FileExistsError("Use a fresh fixture destination; existing files are preserved")
    width, ff, layers = 64, 128, 2
    writer = gguf.GGUFWriter(str(pending), "llama")
    writer.add_name("synthetic-dense-penalties-diagnostic")
    writer.add_context_length(4096)
    writer.add_embedding_length(width)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(1)
    writer.add_head_count_kv(1)
    writer.add_rope_dimension_count(width)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{value:02X}>" for value in range(256)]
    tokens.extend(f"[unused-{value}]" for value in range(259, vocab))
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([2, 3, 3] + [6] * 256 + [5] * (vocab - 259))
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(True)
    rng = np.random.default_rng(104729)
    embeddings = np.zeros((vocab, width), dtype=np.float32)
    embeddings[:259] = rng.normal(0.0, 0.12, (259, width)).astype(np.float32)
    embeddings[3 + 32:3 + 127] *= np.float32(2.0)
    writer.add_tensor("token_embd.weight", embeddings)
    writer.add_tensor("output_norm.weight", np.ones(width, dtype=np.float32))
    for layer in range(layers):
        prefix = f"blk.{layer}."
        for name in ("attn_norm.weight", "ffn_norm.weight"):
            writer.add_tensor(prefix + name, np.ones(width, dtype=np.float32))
        for name, shape in (
            ("attn_q.weight", (width, width)),
            ("attn_k.weight", (width, width)),
            ("attn_v.weight", (width, width)),
            ("attn_output.weight", (width, width)),
            ("ffn_gate.weight", (ff, width)),
            ("ffn_up.weight", (ff, width)),
            ("ffn_down.weight", (width, ff)),
        ):
            writer.add_tensor(prefix + name, rng.normal(0.0, 0.025, shape).astype(np.float32))
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    if not 0 < pending.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("Generated fixture exceeded the 64 MiB diagnostic limit")
    pending.replace(output)
    report = {"path": str(output), "bytes": output.stat().st_size, "sha256": sha(output),
              "vocab": vocab, "width": width, "layers": layers, "context_tokens": 4096,
              "trained_weights": False, "output_tied_to_embeddings": True, "seed": 104729,
              "native_source": str(native_source), "gguf_module": str(Path(gguf.__file__).resolve()),
              "numpy_version": np.__version__, "generator_sha256": sha(Path(__file__))}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, default=root / ".build/gpu-first-moe/native")
    parser.add_argument("--vocab", type=int, choices=(32768, 131072), default=131072)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.native_source, args.vocab), indent=2))


if __name__ == "__main__":
    main()
