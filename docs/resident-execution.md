# Resident execution and bounded reuse

This work continues the [SGLang-inspired runtime changes](sglang-inspired-runtime.md).
Dispatch depends on operators, memory layouts and backend capabilities, rather
than a list of installed model names. Unsupported combinations retain the
existing execution path. These are native GGML/llama.cpp changes; they do not
replace the MLX engine or establish support for every model architecture.

## MTP hidden-state transport

The experimental device transport retains hidden rows in independently owned CUDA buffers
and supplies draft inputs through device-to-device copies. It avoids sending
these rows through host memory on every draft/verification round. Adjacent rows
from the same generation are copied as a range, including during prompt catch-up.

Set `WILDLLAMA_MTP_DEVICE_STATE=1` in the server process environment to opt in.
The default retains host transport: small end-to-end synthetic comparisons did
not establish a consistent speedup despite eliminating hidden-state host copies.
MTP drafting and verifier settings are independent of this transport switch.

The initial path requires one CUDA device shared by target and draft,
one prediction head, one sequence, independent target/draft memory, and sequential
text positions. Other layouts, media inputs and unsupported devices retain the
host path. It does not depend on a model's name or repository.

The owned device banks have an additional 32 MiB limit per context and use the
existing backend allocator, so their allocations also count toward its CUDA
quota. Rejected history is discarded; predecessor and accepted-boundary rows
remain associated with their original positions. Snapshot serialization
materializes the same host state format when needed. Optional allocation failure
falls back to host transport instead of reducing the requested context.

The legacy host output reservation remains available for fallback and public
embedding getters. This change reduces transfers, rather than claiming that
every retained host allocation disappears. An unset switch or
`WILDLLAMA_MTP_DEVICE_STATE=0` retains host transport.

## Selected-token backend output

Backend sampling already exists. Ordinary server generation can now request
only its selected token when it needs no probabilities, grammar/controller
fallback or speculative candidate view. A complete eligible backend chain omits
candidate/probability/logit outputs, their device-to-host copies, and common
sampler candidate reconstruction. Its backend sampling arithmetic and RNG remain
the same.

Partial backend chains retain the logits and candidate IDs required by their
CPU suffix. Detailed probability requests, draft/rejection consumers and custom
accept observers keep their original output contract. This does not silently
enable backend sampling for CPU-target profiles, replace a sampler or fabricate
probabilities for an omitted candidate view.

When all registered backend chains use compact output, a fresh output reservation
also avoids their full-vocabulary host candidate buffers. A previously enlarged
reservation can retain capacity until context destruction; changing request type
does not promise immediate process-RAM reclamation.

## Persistent graph variants

`WILDLLAMA_GRAPH_VARIANTS_MIB` opts into bounded extra graph arenas, from 0 to
1024 MiB; the default is 0. The original worst-case scheduler remains available.
Up to four secondary arenas retain their own scheduler, graph metadata and
compute allocation. Retained metadata and compute storage share the extra-cache
budget; backend allocations remain subject to the existing allocation quota.

Only eligible short batches, up to eight tokens, enter this cache. Reuse checks
the complete existing topology/input contract, including warmup state. Layout,
adapter and scheduler changes invalidate stale arenas. Failed optional admission
uses the original scheduler.

Supported complete backend samplers now retain separate graph-owned bindings;
their logical RNG and history stay with the original sampler. See the
[binding admission rules](execution-pipeline.md#samplers-and-persistent-graph-arenas).
Unsupported samplers, training, encoder, pipeline, custom callback and
managed-weight-provider paths retain their original execution.

## Retained prefix cells

`WILDLLAMA_PREFIX_RETAIN_TOKENS` opts into retained prefix references; default 0.
Plain, unified, fully allocated attention caches can retain up to four complete
text prefixes using internal sequence aliases. Appending a branch allocates new
cells while its unchanged prefix remains shared. No additional GPU KV allocation
or context-capacity change is required.

The aggregate conservative token budget is bounded by both the configured value
and one quarter of the cache capacity. Retained entries also consume the existing
prompt-cache accounting budget. Pressure releases optional references before
retrying active work. Position shifts, state replacement and incompatible changes
invalidate retained handles; an invalid entry cannot restore stale state.

Eligible plain hybrid memories can now combine retained attention cells with a
bounded recurrent snapshot; see [retained hybrid prefixes](execution-pipeline.md#retained-hybrid-prefixes).
Unsupported recurrent layouts, MTP/draft state, sliding windows, multimodal
positions, physical KV growth and split-stream layouts retain complete serialized
snapshots. This is capability-based sharing for eligible memories, rather than a
universal paged-memory implementation. Existing KV sharing between live sequences
remains available independently.

## Validation and rejected candidates

Validation uses small synthetic models and operator fixtures. Tests must check
actual optimized-path use as well as outputs: transfer counters, omitted output
views, arena reuse and retained-handle lifecycle. Before/after HTTP comparisons
exercise ordinary and speculative generation, probability responses, grammar
fallback, branching prefixes and concurrent requests.

A compile-time short-token Gated DeltaNet specialization was investigated on
256 small shapes with two changed inputs each. Outputs and rollback snapshots
matched bitwise, but CUDA-graph measurements did not establish a consistent
speedup and included regressions. That kernel candidate is not enabled or shipped.
The existing recurrent kernels remain in use.

A second experiment to reduce MTP event synchronization also passed correctness
tests but did not demonstrate a repeatable end-to-end gain on the small fixtures.
It was rejected. The device-transport option above retains the original bounded
implementation and remains opt-in.

Separate GPU API controls isolated a small probability-metadata difference to
retained prefixes: graph variants alone were exact; prefix retention reproduced
82 differing probability scalars, with maximum absolute error
`1.430511474609375e-6`. All generated tokens, text, stop types, candidate IDs and
response structures matched. Changed physical KV residency is consistent with
different floating-point reduction layouts; these controls do not identify a
particular CUDA instruction as the cause.

The API harness keeps ordinary and CPU comparisons exact. For the opt-in GPU
resident-cache comparison only, it checks probability scalars within `1e-5`
absolute error, requiring finite values in `[0,1]`, while keeping tokens, text,
candidate IDs, types and structure exact. Reports include both response hashes,
the differing-value count and maximum error. Native fixtures check logits and
state separately. This is a numerical regression bound, not a language-quality
measurement.

The [validation record](benchmarks/resident-execution-20261030.json) contains the
tested source and payload hashes, CPU/CUDA suites, seven small architecture and
quantization fixtures, and before/after API comparisons. The final run passed
27 CPU tests, eight CUDA tests, 44 server component cases and 200 API comparisons
per payload, with the numerical distinction above recorded explicitly.
This update was executed
on Windows with CUDA; Linux, HIP and Metal were not runtime-tested.

Component work/transfer reductions do not establish a new large-model tokens/s
result or superiority over SGLang. Comparisons require the same model, numerical
format, context, sampling, hardware and concurrency. In particular, SSD-backed
expert streaming has different costs from entirely resident inference.

SGLang references: [device post-verification graphs](https://github.com/sgl-project/sglang/blob/dde1ef21a03da18fe226308fb27043013297e38e/python/sglang/srt/speculative/eagle_post_verify_graphs.py),
[future input relay](https://github.com/sgl-project/sglang/blob/dde1ef21a03da18fe226308fb27043013297e38e/python/sglang/srt/managers/overlap_utils.py),
and [adaptive graph tiers](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding).
