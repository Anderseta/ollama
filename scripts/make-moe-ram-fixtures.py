"""Small deterministic GGUF fixtures for native memory-cache regression tests.

Shapes follow the loaders in the local llama.cpp b11845 checkout. These are
random test weights, not useful language models. No network downloads.
"""
from pathlib import Path
import argparse
import sys
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-source', type=Path, required=True, help='llama.cpp source tree containing gguf-py')
parser.add_argument('--output', type=Path, required=True, help='Explicit directory for small test models')
args = parser.parse_args()
sys.path.insert(0, str(args.native_source / 'gguf-py'))
import numpy as np
import gguf

DEST = args.output
DEST.mkdir(parents=True, exist_ok=True)

def make(arch, tied=False, dim=128):
    label = arch + ('-tied' if tied else '') + ('-wide' if dim == 256 else '')
    w = gguf.GGUFWriter(str(DEST / (label + '.gguf')), arch)
    w.add_name('cache-regression-' + label)
    w.add_context_length(256)
    w.add_embedding_length(dim)
    w.add_block_count(4)
    w.add_feed_forward_length(dim)
    w.add_head_count(4)
    w.add_head_count_kv(4 if arch == 'olmoe' else 2)
    w.add_key_length(dim // 4)
    w.add_value_length(dim // 4)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(dim // 4)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_expert_count(8)
    w.add_expert_used_count(2)
    if arch in ('qwen2moe', 'gpt-oss'):
        w.add_expert_feed_forward_length(dim)
    if arch == 'qwen2moe':
        w.add_expert_shared_feed_forward_length(dim)
    if arch == 'gpt-oss':
        w.add_sliding_window(64)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model('llama')
    w.add_tokenizer_pre('default')
    w.add_token_list(['<unk>', '<s>', '</s>'] + [f'<0x{i:02X}>' for i in range(256)])
    w.add_token_scores([0.0] * 259)
    w.add_token_types([2, 3, 3] + [6] * 256)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)
    w.add_add_space_prefix(False)
    rng = np.random.default_rng(123456)
    def tensor(name, *shape):
        values = rng.normal(0, 0.02, size=shape).astype(np.float32)
        if name.endswith('_norm.weight'):
            values += 1.0
        w.add_tensor(name, values)
    tensor('token_embd.weight', 259, dim)
    tensor('output_norm.weight', dim)
    if not tied:
        tensor('output.weight', 259, dim)
    for layer in range(4):
        p = f'blk.{layer}.'
        kv = dim if arch == 'olmoe' else dim // 2
        tensor(p + 'attn_norm.weight', dim)
        for name, width in [('q', dim), ('k', kv), ('v', kv)]:
            tensor(p + 'attn_' + name + '.weight', width, dim)
        tensor(p + 'attn_output.weight', dim, dim)
        if arch == 'olmoe':
            tensor(p + 'attn_q_norm.weight', dim)
            tensor(p + 'attn_k_norm.weight', dim)
        tensor(p + ('post_attention_norm.weight' if arch == 'gpt-oss' else 'ffn_norm.weight'), dim)
        tensor(p + 'ffn_gate_inp.weight', 8, dim)
        for name in ('gate', 'up', 'down'):
            tensor(p + 'ffn_' + name + '_exps.weight', 8, dim, dim)
            if arch == 'gpt-oss':
                tensor(p + 'ffn_' + name + '_exps.bias', 8, dim)
            if arch == 'qwen2moe':
                tensor(p + 'ffn_' + name + '_shexp.weight', dim, dim)
        if arch == 'qwen2moe':
            tensor(p + 'ffn_gate_inp_shexp.weight', dim)
        if arch == 'gpt-oss':
            tensor(p + 'attn_sinks.weight', 4)
            tensor(p + 'attn_output.bias', dim)
            tensor(p + 'ffn_gate_inp.bias', 8)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(label, flush=True)

for architecture in ('llama', 'qwen2moe', 'olmoe', 'gpt-oss'):
    make(architecture)
make('llama', tied=True)
make('llama', tied=True, dim=256)
make('gpt-oss', dim=256)
