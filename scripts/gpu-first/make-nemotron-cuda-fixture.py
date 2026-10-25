"""Create a separate tiny random-weight Nemotron state128 CUDA test fixture.

This is not a model conversion: only the small synthetic fixture is accepted.
Original expert bytes and all non-resized tensors are preserved verbatim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-source', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.source.resolve() == args.output.resolve():
    parser.error('output must differ from source')
if args.source.stat().st_size > 64 * 1024 * 1024:
    parser.error('only small synthetic input fixtures are accepted')
sys.pycache_prefix = str(args.output.parent / 'pycache')
sys.path.insert(0, str(args.native_source / 'gguf-py'))
import numpy as np
import gguf

reader = gguf.GGUFReader(str(args.source))
arch = reader.fields['general.architecture'].contents()
if arch != 'nemotron_h_moe' or reader.fields['general.name'].contents() != 'tiny-moe':
    parser.error('expected the original tiny-moe/nemotron_h_moe regression fixture')
def field(key):
    return reader.fields[arch + '.' + key].contents()
if field('embedding_length') != 128 or field('ssm.state_size') != 16 or field('expert_count') != 8:
    parser.error('unexpected synthetic fixture dimensions')

state, seed = 128, 20261025
embd, inner = field('embedding_length'), field('ssm.inner_size')
groups, rank, kernel = field('ssm.group_count'), field('ssm.time_step_rank'), field('ssm.conv_kernel')
conv_channels = inner + 2 * groups * state
input_width = 2 * inner + 2 * groups * state + rank
rng = np.random.default_rng(seed)
args.output.parent.mkdir(parents=True, exist_ok=True)
writer = gguf.GGUFWriter(str(args.output), arch)
for key, value in reader.fields.items():
    if key.startswith('GGUF.') or key == 'general.architecture':
        continue
    contents = value.contents()
    if key == arch + '.ssm.state_size':
        contents = state
    elif key == 'general.name':
        contents = 'synthetic-nemotron-cuda-state128'
    sub_type = value.types[-1] if value.types[0] == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(key, contents, value.types[0], sub_type=sub_type)
writer.add_string('general.description', 'Synthetic numerical fixture; resized SSM random weights, original expert bytes retained.')
changed = []
for tensor in reader.tensors:
    shape = None
    if tensor.name.endswith('.ssm_in.weight'):
        shape = (input_width, embd)
    elif tensor.name.endswith('.ssm_conv1d.weight'):
        shape = (conv_channels, kernel)
    elif tensor.name.endswith('.ssm_conv1d.bias'):
        shape = (conv_channels,)
    if shape is not None:
        values = rng.normal(0, 0.02, size=shape).astype(np.float32)
        writer.add_tensor(tensor.name, values)
        changed.append({'name': tensor.name, 'original_ggml_shape': tensor.shape.tolist(), 'new_ggml_shape': list(reversed(shape)), 'new_type': 'F32'})
    else:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
if len(changed) != 6:
    raise ValueError(f'expected six resized tensors in two recurrent blocks, got {len(changed)}')
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

copied = gguf.GGUFReader(str(args.output))
unchanged, experts = 0, 0
changed_names = {entry['name'] for entry in changed}
assert len(reader.tensors) == len(copied.tensors)
for old, new in zip(reader.tensors, copied.tensors):
    assert old.name == new.name
    if old.name not in changed_names:
        assert old.tensor_type == new.tensor_type and np.array_equal(old.shape, new.shape)
        assert np.array_equal(old.data, new.data), old.name
        unchanged += 1
        experts += int('_exps' in old.name)
report = {
    'synthetic_only': True, 'source': str(args.source), 'target': str(args.output),
    'seed': seed, 'original_state_size': 16, 'new_state_size': state,
    'conv_channels': conv_channels, 'ssm_input_width': input_width,
    'resized_tensors': changed, 'unchanged_tensors': unchanged, 'unchanged_expert_tensors': experts,
    'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
    'target_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
}
args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
