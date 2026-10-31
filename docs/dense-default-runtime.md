# Dense runtime defaults and bounded prefix prewrites

Patches [042](../llama/compat/042-async-prefix-writeback.patch),
[043](../llama/compat/043-canonical-penalties.patch),
[044](../llama/compat/044-hybrid-text-device-mask.patch), and
[045](../llama/compat/045-auto-device-mask.patch) extend the
[execution pipeline](execution-pipeline.md). They preserve model weights,
quantization, requested logical context, and saved sampling parameters.

[047](../llama/compat/047-preserve-mask-execution.patch) additionally preserves
execution flags and allocation contracts when inserting mask operations.
[048](../llama/compat/048-conservative-extended-mask-default.patch) adds the
larger automatic mask-size floor for hybrid and text M-RoPE layouts.

Dispatch uses candidate layouts, token positions, memory classes, and actual
backend placement. Dense and MoE models can benefit when those capabilities
match; model names and repository names do not select a special implementation.
An eligible ordinary attention model and a hybrid/recurrent model can take
different paths even when both fit completely in VRAM.

## What changes automatically

| Feature | Default | Override or fallback |
| --- | --- | --- |
| Canonical host penalty application | Automatic when its candidate/history guards pass | Other layouts retain the general penalty loop. |
| Compact attention metadata, ordinary one-dimensional text | Automatic for eligible batches of at least 512 queries per stream and a dense mask of at least 256 KiB | `WILDLLAMA_DEVICE_MASK=0` retains host masks. |
| Compact metadata, proven hybrid or text M-RoPE layouts | Automatic for eligible batches of at least 512 queries per stream and a dense mask of at least 2 MiB | The full memory, position, and placement proofs remain mandatory. |
| Compact metadata for smaller eligible batches | Disabled without explicit opt-in | `WILDLLAMA_DEVICE_MASK=1` permits the existing minimum of 32 queries per stream, with all other guards retained. |
| Asynchronous cold prefix prewrites | Disabled | Requires `WILDLLAMA_PREFIX_DISK_ASYNC=1` and the existing explicit disk-cache settings. |

Automatic compact masks still require every semantic, size, and placement
check below. Single-token decoding and batches below the automatic threshold do not enter
the automatic path. This is a prefill/input-preparation optimization, rather
than an automatic change to the model's token-generation kernels.

Existing graph-variant budgets, native prefix retention, physical KV growth,
experimental MTP transport, and grammar-first sampling keep their own settings.
This update does not enable those optional features.

## Canonical penalty application

The host repetition/frequency/presence penalty sampler can avoid a hash lookup
for every vocabulary row. It first proves a canonical prefix, where candidate
row `i` has token ID `i`, then updates only history IDs inside that prefix.
The original arithmetic and candidate metadata rules remain in place.

For enabled penalties, the factory selects the original general-loop interface
directly when the vocabulary size is nonpositive or the configured history
window exceeds `n_vocab - n_vocab/4` (approximately 75% of the vocabulary).
Those configurations do not pay an additional canonical-prefix probe. Other
configurations select the canonical interface, with the runtime guards below.

The canonical path requires a full-vocabulary candidate count. If the distinct
history-ID count exceeds `size - size/4`
(approximately 75% of candidates), it immediately uses the general loop.
An empty canonical prefix also selects that loop. When the prefix is only
partial, a history count greater than `size/8` selects the general loop for
the whole candidate array, avoiding an expensive sparse pass plus suffix pass.

For an admitted partial prefix, the unchanged general loop handles its suffix.
Reordered, duplicate, filtered, unknown-ID, and other noncanonical layouts
therefore retain complete candidate handling. Disabled penalties retain their
existing no-op behavior. This does not change penalty settings, filter order,
RNG consumption, model precision, or the configured history window.

Binding, snapshot, and budget checks recognize the two exact builtin interface
addresses. Both retain the existing backend callbacks and history ownership;
copied/custom interface tables do not qualify merely by copying callbacks.

This is host sampler work: it does not imply an additional benefit for a
request whose penalties already execute entirely through another backend.

## Compact attention metadata and default admission

Instead of constructing and uploading the dense attention mask on the CPU,
eligible graphs upload positions and sequence-membership metadata. CUDA
expands the same zero/negative-infinity mask. The dense GPU mask still exists;
the saved work concerns host construction, staging, and transfer bytes.

With `WILDLLAMA_DEVICE_MASK` unset, automatic admission additionally requires
token-backed, unmixed inputs with one, three, or four position coordinates and
at least 512 queries per stream. Ordinary attention and the supported plain
hybrid/text M-RoPE path share this query-count policy. Hybrid inputs or text
layouts with multiple position coordinates additionally require a dense mask
of at least 2 MiB (2,097,152 bytes) for automatic admission. Ordinary
one-dimensional text retains the 256 KiB dense-mask floor. These are mask-byte
thresholds, independent of model-file size and logical context capacity.

The exact memory and position proofs below remain mandatory. Explicit `1`
retains the existing minimum of 32 queries per stream and the 256 KiB mask
floor for both ordinary and extended text layouts; the additional 2 MiB floor
applies only to automatic admission. All other guards remain in place.
The query threshold applies to each microbatch, so a short tail of a longer
prompt retains the host-mask path under automatic policy.
Explicit `0`, an empty value, or an unrecognized value retains the host path.

The automatic size/input check runs before walking model layers and backends,
avoiding those optional admission costs for single-token decode and short batches.

Both modes require a dense mask of at least 256 KiB, at least a fourfold
reduction in staging bytes, at most 8 MiB of compact metadata, at most 32
active sequence IDs, and at most 65,535 query rows across streams.
Queries must divide evenly among streams.

Placement checks require one ordinary CUDA device/backend, all repeating
layers on that device, ordinary device-local attention K/V storage, and
compatible attention consumers on that device. CPU input embeddings can
remain on CPU. Partial offload, pipeline/multidevice execution, custom tensor
overrides, split/meta/host KV buffers, and unsupported attention consumers
retain the original mask path. SWA, ALiBi, training, mixed inputs, and
unsupported memory classes also retain the original implementation.

The hybrid path supports the exact plain hybrid wrapper and its
ordinary attention cache; recurrent layers do not need nonexistent K/V tensors.
For text M-RoPE, query coordinates must match their nonnegative ordinary
position, with a zero fourth coordinate when present. Cached keys must be
token-backed and have neutral or position-aligned coordinates. PLE, spatial
or media cells, and derived/indexer/compressed memories retain host handling.
The occupied cached prefix is checked, not the entire logical allocation.

### Preserving selected graph branches

Mask insertion retains the execution flags of every original graph node, all
original input leaves, and disconnected state-write roots. Only the new mask
operation becomes executable, and only when an active consumer needs it.
Unselected embedding/sampler branches remain inactive; their input indices
are not populated and must never be read as if they were selected inputs.

The regression fixtures exercise poisoned indices in inactive branches,
standalone input allocation, state writes, and a file-loaded hybrid model
with CPU input embeddings and output-free/single-output prefill batches.
These are distinct from loading a model entirely into GPU buffers with a
custom tensor initializer.

## Optional asynchronous prefix prewrites

These native-process environment settings are all required:

```text
WILDLLAMA_PREFIX_DISK_CACHE_DIR=G:\Goose\prefix-state-cache
WILDLLAMA_PREFIX_DISK_CACHE_MIB=1024
WILDLLAMA_PREFIX_DISK_ASYNC=1
```

There is no default disk directory. A missing, relative, invalid, or unwritable
path does not select a system-disk substitute. `WILDLLAMA_PREFIX_DISK_ASYNC=0`
or any value other than exactly `1` retains synchronous cold-cache behavior.
A finite prompt-RAM cache budget is required for proactive prewrites.

Near 75% of that RAM budget, one lazy worker can prewrite immutable serialized
text snapshots: target, draft, speculative state, and recurrent checkpoints
remain one transaction. At most two requests share a 32 MiB aggregate bound,
including vector capacities and request/descriptor accounting. Admission also
accounts for retained token/checkpoint metadata. No extra payload-RAM allowance
is granted; pending and completed-but-resident vectors remain charged until
the inference thread releases them. Full, oversized, and unsupported requests
retain the synchronous spill/eviction path.

Successful prewrites stay in RAM until actual pressure. A hot pending prefix
reuses its original vectors after writer completion instead of rereading SSD.
Failures restore complete vectors and suppress repeated proactive attempts
for that entry. Ready entries use RAM restore costs; writer waits do not
pollute the RAM or disk-read cost estimators. Entry destruction waits for its
owned work; cache/model teardown drains and joins the worker.

The existing disk quota includes headers, checksums, and payloads, with at most
64 owned entries. Prewrites can perform extra I/O for a prefix later reused
or erased, so this feature remains opt-in. Normal filesystem caching and
thread/runtime overhead are outside the payload accounting; this is not an
operating-system-wide RAM cap, persistent cross-restart cache, expert-weight
prefetcher, or per-token SSD KV streaming implementation.

## Activation and validation scope

Apply variables to the native runner's launch environment, then restart the
runner or unload/reload its model. Changing a terminal or user environment
setting does not update an already-running process. Remove
`WILDLLAMA_DEVICE_MASK` to restore automatic policy; setting it to `0` is an
explicit override of that policy. None of these controls reduces context.

Build/test results are published separately in the
[validation record](benchmarks/dense-default-runtime-20261030.json).
The final Windows build passed 36 CPU and 11 CUDA native tests, 300 HTTP
before/after cases per payload, and 2,064 meaningful pre-sampling log-probability
comparisons across the dense CPU/GPU matrix. Separate HTTP tests exercised
actual cold-prefix files, EOS handling, grammar failures, and concurrent recovery.

Same-binary CUDA measurements reduced native prompt-processing latency by
approximately 16–25% for eligible ordinary-attention cases and 10–11% for the
larger hybrid cases. Whole-request dense generation measurements retained both
server startup orders:

| Device / host penalty history | Baseline-first generation TPS change | Final-first generation TPS change |
| --- | --- | --- |
| CUDA / 64 tokens | +8.9% | +25.8% |
| CUDA / 512 tokens | +23.0% | +51.7% |
| CPU / 64 tokens | +8.4% | +2.0% |
| CPU / 512 tokens | +16.6% | +6.8% |

These are small random-weight fixtures, not trained-model speed predictions.
The no-penalty and backend controls also varied between runs; their raw results
remain in the record. The table therefore does not establish an exact isolated
sampler speedup or a guarantee of unchanged latency for every workload. The
automatic mask policy retains host masks for short or unsupported batches;
optional asynchronous prewrites remain disabled because they can add disk I/O.

The policies above do not assert a trained-Qwen tokens/s gain, universal
backend support, or a model-quality improvement. Component timings and tiny
synthetic models must be interpreted within their recorded workload limits.
