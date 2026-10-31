"""Generate small deterministic dense GGUF fixtures without downloading weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def fixture(directory, dim, layers, ff):
    path = directory / f'dense-d{dim}-l{layers}-f32.gguf'
    if path.exists():
        raise FileExistsError(path)
    writer = gguf.GGUFWriter(path, 'llama')
    writer.add_name('wildllama-local-synthetic-dense-mask')
    writer.add_context_length(16384)
    writer.add_embedding_length(dim)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(dim // 64)
    writer.add_head_count_kv(dim // 64)
    writer.add_rope_dimension_count(64)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_tokenizer_model('llama')
    writer.add_token_list(['<unk>', '<s>', '</s>'] + [f'<0x{i:02X}>' for i in range(256)])
    writer.add_token_scores([0.0] * 259)
    writer.add_token_types([2, 3, 3] + [6] * 256)
    writer.add_uint32('tokenizer.ggml.unknown_token_id', 0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(True)

    def tensor(name, shape):
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
        rng = np.random.Generator(np.random.PCG64(seed))
        data = rng.uniform(-0.02, 0.02, size=shape).astype(np.float32)
        if 'norm.weight' in name:
            data += np.float32(1.0)
        writer.add_tensor(name, data)

    tensor('token_embd.weight', (259, dim))
    tensor('output.weight', (259, dim))
    tensor('output_norm.weight', (dim,))
    for i in range(layers):
        prefix = f'blk.{i}.'
        tensor(prefix + 'attn_norm.weight', (dim,))
        for name in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight'):
            tensor(prefix + name, (dim, dim))
        tensor(prefix + 'ffn_norm.weight', (dim,))
        for name in ('ffn_gate.weight', 'ffn_up.weight'):
            tensor(prefix + name, (ff, dim))
        tensor(prefix + 'ffn_down.weight', (dim, ff))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if not 0 < path.stat().st_size <= 64 * 1024 * 1024:
        raise AssertionError('Generated fixture exceeded the small-model limit')
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                embedding=dim, layers=layers, head_size=64, format='F32')


def main():
    global gguf, np
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, default=repo / '.build/gpu-first-moe/native')
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        raise ValueError('An explicit absolute output directory is required')
    directory = args.output_dir.resolve()
    if os.name == 'nt' and directory.drive.upper() == 'C:':
        raise ValueError('Fixture output must be outside C:')
    source = args.native_source.resolve() / 'gguf-py'
    if not (source / 'gguf/__init__.py').is_file():
        raise FileNotFoundError('Prepare the pinned native source and its gguf-py package first')
    sys.path.insert(0, str(source))
    import gguf
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    result = [fixture(directory, 64, 2, 128), fixture(directory, 256, 4, 768)]
    (directory / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
