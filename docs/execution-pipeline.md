# GPU sampling, attention metadata and tiered prefix state

These native GGML/llama.cpp changes continue the
[resident execution work](resident-execution.md). Admission checks use memory
layouts, sampler capabilities and actual backend placement. There is no list of
installed model names. Unsupported layouts retain the existing execution path.

The work targets transfers, repeated grammar work and cache capacity. It does
not change weights, quantization or the requested logical context, and does not
establish performance parity with SGLang.

## Samplers and persistent graph arenas

When `WILDLLAMA_GRAPH_VARIANTS_MIB` is enabled, supported backend samplers can
now accompany retained graph arenas. Each graph owns its input bindings and
scratch references. Logical RNG and token history remain owned by the original
sampler rather than being duplicated for every arena.

Admission estimates the entire binding map before calling allocation/capture
factories. A graph that cannot fit its metadata budget falls back before making
those copies; retained snapshots reserve their exact node count.

Identity and revision checks reject replacement, removal and reattachment of
samplers. Binding allocation failure retains the primary graph path. The cache
budget includes retained binding metadata and compute storage. This is an extra
graph-cache budget, not a process-RAM limit or a replacement for the model-weight
limits.

## Compact causal attention metadata

`WILDLLAMA_DEVICE_MASK=1` opts in; the default remains disabled. Eligible prefill
batches upload key/query positions and sequence-membership bitsets, then expand
the exact existing zero/negative-infinity mask on CUDA. The dense GPU mask still
exists; the improvement removes its dense CPU construction, host reservation
and host-to-device transfer. This is not approximate or sparse attention.

Admission requires at least 32 queries per stream, a dense mask of at least
256 KiB, at least a fourfold staging-byte reduction, no more than 8 MiB of compact
metadata, at most 32 active sequence IDs and at most 65,535 query rows across
streams. Smaller batches retain the original path.

Placement is checked after the actual graph consumers are known: one regular
CUDA device, compatible attention/KV storage on that device, and supported
attention consumers without contrary CPU placement. Partial offload, pipeline,
multiple devices and unsupported custom layouts retain the host path. CPU input
embeddings can remain on CPU when the attention path itself qualifies.
Explicit tensor-buffer overrides retain the host path as well.

SWA, ALiBi, media/spatial positions and unsupported memory/input classes also
retain the original implementation. Debug logs contain `device-mask inputs:`
when compact metadata is populated for execution, rather than only reserved.
This path requires ordinary one-dimensional token positions; the pure-text
M-RoPE retention support described below does not extend mask eligibility.

## Cold serialized snapshots on an explicit disk

The optional cold tier preserves serialized prefix states that RAM-cache
pressure would otherwise evict. Both settings are required:

```text
WILDLLAMA_PREFIX_DISK_CACHE_DIR=G:\Goose\prefix-state-cache
WILDLLAMA_PREFIX_DISK_CACHE_MIB=1024
```

There is no default directory. A missing, relative, invalid or unwritable path
retains normal RAM-cache eviction. The MiB setting defaults to zero and is
bounded at 524,288 MiB. Existing RAM and token budgets continue to apply.

Target, draft, speculative state and recurrent-checkpoint payloads are stored as
one transaction in 256 KiB checksum-framed pages. Vector capacities are released
only after a complete write. Restoration stages and verifies every component
before publication, making room inside the RAM-cache accounting budget first.
Corrupt, missing or truncated cache files cause prompt recomputation.

The per-cache quota covers file headers, checksums and payloads, with at most
64 owned entries. Reservations begin before writing; failed cleanup retains
both byte and entry charges. Each cache instance creates its own session
directory and removes owned files on normal destruction. This is ephemeral,
process-local storage, not cross-restart persistence; an abnormal process exit
can leave its session directory behind.

Disk restore cost is tracked separately from RAM restore cost. Media entries
and live native resident handles keep their original eviction path. Serialized
entries must fit the RAM-cache materialization budget. These are page-framed
snapshots, not a new paged-attention allocator or per-token SSD KV streaming.

## Bounded state-file staging and failed restores

Ordinary CPU and device-default buffers use at most 4 MiB of temporary staging
per state-file tensor transfer. Meta, split, pinned and nondefault/custom buffer
types retain their existing whole-tensor callback contract. The file format and
tensor bytes are unchanged.

Read range and remaining file length are checked before the first tensor
upload. Missing files, invalid headers and token-count probes preserve live
state. Once a real restore starts, failure clears its destination sequence;
failure during a whole-context restore clears that context's sequence memory.
This prevents partially restored state from being reused as a valid prefix.

## Retained hybrid prefixes

`WILDLLAMA_PREFIX_RETAIN_TOKENS` remains opt-in with default zero. Eligible plain
hybrid memories can retain immutable attention-cell references plus a snapshot
of their recurrent state. Full KV payloads need not be read back to the host for
these retained entries. Recurrent checkpoints remain attached to the prompt so
shorter and divergent requests can restore the appropriate checkpoint.

`WILDLLAMA_PREFIX_RECURRENT_MIB` limits aggregate live retained recurrent
snapshots per context: default 32 MiB, maximum 1,024 MiB; zero disables this
hybrid retention path. The value is captured when the owner is constructed.
Existing token and server prompt-cache budgets still apply. The recurrent part
continues to consume host RAM.

Support is limited to exact plain KV/hybrid memory classes. Derived indexer,
compression and other extended memories retain complete serialized snapshots.
Pure-text M-RoPE cells require proof of contiguous, token-backed,
neutral/position-aligned metadata; embeddings, media and PLE layouts are
excluded. A failed composite restore clears both destination memories. Weak
ownership, mutation invalidation and pressure eviction protect saved handles.

## Explicit grammar-first constrained sampling

The existing grammar-last validation/resampling order remains the default.
The new path requires an explicit grammar-first contract:

```json
{"grammar_first": true, "backend_sampling": true}
```

Native CLI/environment equivalents are `--grammar-first` /
`LLAMA_ARG_GRAMMAR_FIRST=1` and `--backend-sampling` /
`LLAMA_ARG_BACKEND_SAMPLING=1`. A request's explicit `grammar_first: false`
overrides the process default. No new Ollama Go API option is introduced here;
these are native request/launch settings.

Choosing grammar-first changes the order of filtering relative to grammar-last.
Validation compares CPU and backend sampling under the **same chosen order**.
No automatic conversion of existing requests to grammar-first is performed.

The existing parser produces an owned allowed-token bitset. A bounded exact
grammar-state cache retains at most 16 entries / 1 MiB per grammar instance,
including keys and metadata. Keys include complete parser stacks and partial
UTF-8 state. Awaiting lazy triggers bypass this cache; reset and clone/copy
lifetimes do not borrow another grammar's parser state.

CPU/CUDA mask operations preserve surviving logits and exclude disallowed
tokens before the existing filters. Backend use requires a supported complete
chain and one output per sequence. Unknown grammars, reasoning controllers and
unsupported partial, custom, nested or multi-output layouts retain CPU fallback.

The constrained path validates the selected support before publishing a token.
Empty support, blocked EOS and nonfinite sampling results fail the affected
request with HTTP 400 instead of emitting an invalid token. RNG updates commit
only for accepted results. Request-local failures leave other active slots
running, and discarded sampler branches are not expanded into the graph.

## Evidence and rejected candidates

Validation uses small local/generated fixtures and same-setting comparisons.
Mask-staging and repeated-grammar benchmarks measure specific components or
synthetic workloads; they are not a Qwen checkpoint tokens/s claim.
The long-prefill HTTP fixture overrides its synthetic Llama training-context
metadata to 4,096 identically in both processes, allowing the 512-token case
past the server's fixture-only 256-token slot cap. No GGUF file is modified.

The combined Windows build passed 31 CPU and 10 CUDA CTest cases, 45 server
component cases (190 assertions), seven architecture/quantization fixtures and
300 before/after HTTP cases per payload. Additional checks cover 20 same-order
grammar cases per device, 40 request-local failure cases with recovery and
concurrent survivors, and five hybrid-prefix HTTP cases per device. Supported
CUDA attention, unsupported flash-attention fallback, mixed CPU/CUDA placement
and same-shape causal-mode transitions are exercised explicitly.

Default-path ABBA sanity timings retained identical generated tokens. Small
timing windows were noisy and did not establish a consistent regression signal
or a general model-speed improvement. Those measurements are separate from
the targeted mask/grammar component experiments.

A direct device-token feedback candidate removed the measured token-input
uploads and preserved tokens/state, but repeated end-to-end synthetic comparisons
included regressions and inconsistent gains. It was rejected and is not part of
the shipped patch stack. No autonomous next-token pipeline is claimed.

The final source, build, payload and test evidence is recorded in
[execution-pipeline-20261030.json](benchmarks/execution-pipeline-20261030.json).
Windows CPU/CUDA execution is the runtime validation scope; Linux, HIP, Metal
and MLX performance is not certified by these tests.

Related SGLang implementations:
[future input relay](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/overlap_utils.py),
[constrained decoding](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/constrained/xgrammar_backend.py),
[unified prefix components](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_radix_cache.py),
and [hierarchical cache design](https://docs.sglang.io/docs/advanced_features/hicache_design).
