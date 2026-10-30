# SGLang-inspired runtime work

This update adapts ideas from SGLang's
[overlap scheduler](https://github.com/sgl-project/sglang/blob/3b6c28b0b9aff784115bafeecc67f37e1fe43454/python/sglang/srt/managers/scheduler.py),
[prefix cache](https://github.com/sgl-project/sglang/blob/3b6c28b0b9aff784115bafeecc67f37e1fe43454/python/sglang/srt/mem_cache/unified_radix_cache.py),
and [post-verification graphs](https://github.com/sgl-project/sglang/blob/3b6c28b0b9aff784115bafeecc67f37e1fe43454/python/sglang/srt/speculative/eagle_post_verify_graphs.py)
to the existing GGML/llama.cpp runtime. The implementation preserves existing
model formats, context capacity, sampling settings and saved profiles.

## Exact prefix lookup

Patch `025` indexes immutable cached token spans using sorted entries and
adjacent longest-common-prefix lengths. A long shared prefix is compared once
instead of being scanned again for every cached snapshot. The index borrows
the existing token arrays and does not copy prompts or serialized model state.

The existing cost-based restore decision, list ordering and fallback scoring
remain unchanged. Small caches use the original scan. Multimodal prompts,
incomplete indexes and failed optional allocations retain the exact original
comparison. Entries are removed before their token owner is moved or freed.
Index entry storage is capped at 1 MiB, separately from snapshot payload limits.

Snapshots still own complete KV, recurrent, draft and speculative state. This
change adds a compact shared-prefix index; it does not introduce shared GPU KV
pages or change a checkpoint's restore boundary.

## Post-verification host work

Patch `026` avoids rebuilding intermediate host candidate arrays when the
backend has already selected tokens for a matching verification batch. Eligible
accept handlers cannot inspect the deferred candidates or throw. Unknown or
stateful handlers keep the original path. The final public candidate view is
materialized before returning, with the same selected index and values.

This path applies when target backend sampling is enabled. The default CPU
target path and single-row draft sampling retain their original behavior.
It does not remove graph-output transfers or add a device-side commit graph.

Rejection verification also uses a bounded token-probability index for large
draft distributions. It preserves the first occurrence of duplicate token IDs,
target iteration order, floating-point arithmetic and RNG consumption. Small
top-k supports use linear lookup. The reusable workspace is capped at 1 MiB per
sampler and is excluded from sampler snapshots.

## Bounded CPU preparation

Patch `027` prepares eligible queued text-request samplers on one lazy worker
while current inference can continue. The number of outstanding preparations
is bounded by the configured parallel capacity. Queued work owns its sampling
configuration; slot selection, KV changes, backend attachment and state restore
remain on the inference thread.

Cancellation releases the request's preparation ticket. A running canceled job
continues to occupy its budget until it finishes. Model unload stops and joins
the worker before releasing the vocabulary and prepared samplers. Cold requests
retain synchronous admission; unsupported request types and large grammars use
the original path.

This window requires a following request to reach the native queue. The Ollama
wrapper can serialize requests earlier for architectures with restricted
parallelism; this update retains those restrictions and the existing sequence
isolation rules.

Prompt accounting is aggregated during asynchronous decode, before the existing
output fence. Post-fence accounting uses per-slot counts instead of rescanning
the token batch twice. Timings and result publication retain their synchronization
boundaries.

## Validation and performance interpretation

Validation uses local synthetic models and small CPU/CUDA fixtures. The HTTP
comparison is available as
[`integration/native_sglang_smoke.py`](../integration/native_sglang_smoke.py).
It refuses models larger than 64 MiB and explicitly selects CUDA when requested,
so a missing CUDA backend cannot silently turn a GPU test into a CPU result.

Component measurements establish reduced comparisons, candidate materializations
and metadata scans. They do not establish a new large-model tokens/s result or
performance parity with SGLang. The underlying engine still uses its existing
GPU memory layouts and sampling interfaces.

The [validation record](benchmarks/sglang-adaptation-20261030.json) includes
23 CPU CTest cases, two standalone MSVC helper fixtures, five CUDA cases,
44 server component cases, and 100 before/after HTTP cases across CPU/CUDA and
ordinary/speculative generation. Separate legacy HTTP sampling comparisons
retain identical text, tokens and probabilities in eight cases.

For 128 cached token spans sharing 8,192 tokens, the exact lookup fixture reduces
token comparisons from 1,048,705 to 8,206. In the actual cache-selection component
benchmark with 32 entries, the medians for 1,000 queries were 153.42 ms for the
original scoring scan and 5.24 ms for indexed selection. The benchmark compares
the selection routines, without model execution, serialization or GPU transfers.
