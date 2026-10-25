"""Add activation-precision hints to a tiny GGUF, preserving every weight byte."""
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
sys.pycache_prefix = str(args.output.parent / 'pycache')
sys.path.insert(0, str(args.native_source / 'gguf-py'))
import numpy as np
import gguf
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata, MetadataDetails

args.output.parent.mkdir(parents=True, exist_ok=True)
reader = gguf.GGUFReader(str(args.source))
names = [t.name for t in reader.tensors if '_exps' in t.name and t.name.endswith('weight')]
if not names:
    parser.error('source has no expert weight tensors')
writer = gguf.GGUFWriter(str(args.output), reader.fields['general.architecture'].contents())
copy_with_new_metadata(reader, writer, {
    'general.tensor_extra.name': MetadataDetails(gguf.GGUFValueType.ARRAY, names),
    'general.tensor_extra.prec_a4': MetadataDetails(gguf.GGUFValueType.ARRAY, [False] * len(names)),
}, [])
writer.close()
copied = gguf.GGUFReader(str(args.output))
assert len(reader.tensors) == len(copied.tensors)
for old, new in zip(reader.tensors, copied.tensors):
    assert old.name == new.name and old.tensor_type == new.tensor_type
    assert np.array_equal(old.data, new.data), old.name
report = {
    'source': str(args.source), 'target': str(args.output), 'hinted_expert_tensors': names,
    'weights_identical': True,
    'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
    'target_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
}
args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
