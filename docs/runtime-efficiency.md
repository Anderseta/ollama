# Runtime efficiency: dense and MoE models

These changes build on `0621cb66`, the dense-placement and GPU-first runtime
already used by WildGoose. Native changes are in
[`008-runtime-efficiency.patch`](../llama/compat/008-runtime-efficiency.patch),
after patches 001 through 007 on llama.cpp `b11845`.

They preserve weights, quantization, selected experts, context capacity and
saved profiles. There are no new user settings. They target avoidable host
work and data-transfer waits; they do not make an SSD-resident model run at
VRAM bandwidth or reduce its necessary weight traffic.

## Streamed tool calls

The native chat adapter now accumulates fragmented tool-call arguments, names
and IDs with request-local builders. Previously each fragment copied the whole
string accumulated so far. A generated file inside a tool argument could cause
gigabytes of cumulative allocation even though the final argument was small.
The final JSON parser and the content received by the caller are unchanged.

Finalization sorts the indices that actually arrived, rather than iterating
from zero to the largest index. Sparse indices, including the maximum signed
integer, therefore terminate normally. Negative indices fail explicitly.

On the local Ryzen 5 2600, three repetitions with 8-byte fragments produced:

| Final argument | Median time before / after | Cumulative bytes allocated before / after |
| --- | --- | --- |
| 4 KiB | 2.03 / 0.182 ms | 1.14 MB / 28.3 KB |
| 64 KiB | 255 / 2.10 ms | 289 MB / 426 KB |
| 256 KiB | 2,522 / 10.23 ms | 4,417 MB / 1.72 MB |

This benchmark includes argument accumulation and final JSON parsing. It
excludes SSE transport and inference. Allocation totals are cumulative bytes,
not simultaneous RAM usage or VRAM savings. The tests also exercise the real
SSE adapter with interleaved tools, fragmented UTF-8 and malformed arguments.

## MoE read and upload scheduling

A batched upload resolves each source region once and keeps a cursor into its
required RAM pages. When another read finishes, it resumes at the first page
not yet ready, instead of rescanning every previously checked page. Every
cursor belongs to one batch. That batch pins all source pages with the current
cache epoch, so eviction cannot invalidate the cursor while it is in use.

On Windows, a completed SSD read first copies its staging bytes into protected
RAM. The freed I/O slot is then refilled before notifying upload consumers.
The next read can progress during a RAM-to-pinned staging copy or an H2D-slot wait. The
existing four I/O slots, two upload slices and memory limits are unchanged.
The synchronous single-slot POSIX reader retains upload-before-next-read order.

Exceptions drain submitted reads before releasing their pages or staging
storage. Tests cover out-of-order completion, shared pages, renewed batches,
queue depth, failures during submission/completion/upload, partial pages,
tensor views, eviction, truncated files and recovery. This does not change the
cache admission policy or approximate/prune expert computations.

The readiness fixture reduced 35,840 page checks to 4,096 for overlapping
ranges with progressive completion. In its separate CPU microbenchmark, the
32-completion case took 101.10 / 30.46 ms before / after; two completions took
13.28 / 10.41 ms. The single-completion case took 9.76 / 10.36 ms, a small
regression in that synthetic case. These are medians of seven samples of
100,000 repetitions, not disk or model throughput results.

## Host snapshots

Saving and restoring an in-memory state groups supported tensor transfers on
the context's existing backend streams, then fences each participating backend
once. It does not allocate a new CUDA stream or add a pinned RAM cache.
Device-to-device snapshots and streaming state-file I/O are unchanged.

CPU buffers, nonstandard buffer types, unsupported async accessors and partially
committed CUDA VMM buffers retain synchronous buffer callbacks. Partial VMM
storage must keep the callbacks that ensure a range is physically committed
under its quota. This fallback is independent of the current KV restore path's
own precommit logic.

All queued transfers are drained before success or exception propagation; no
destructor submits transfers. Internal C++ callbacks retain exception handling
under MSVC `/EHsc`, which assumes ordinary `extern "C"` calls do not throw.
See [Microsoft's exception model documentation](https://learn.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model?view=msvc-170).

Fewer explicit fences do not guarantee fully asynchronous transfers from
pageable host memory. The runtime keeps the existing host allocation policy;
pinned-memory microbenchmarks are reported separately and are not attributed
to the ordinary snapshot cache. See
[CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

With 64 tensors and ordinary pageable host storage, the measured transfer/fence
stage was:

| Total bytes | Direction | Median before / after |
| --- | --- | --- |
| 256 KiB | GPU to RAM | 1.256 / 1.289 ms |
| 256 KiB | RAM to GPU | 1.047 / 0.823 ms |
| 16 MiB | GPU to RAM | 5.640 / 5.477 ms |
| 16 MiB | RAM to GPU | 5.490 / 3.035 ms |

Each entry has three warmups and sixteen measured pairs with alternating order
and exact byte checks. Timing excludes construction of the transfer helper.
The small GPU-to-RAM case did not improve. This is a component benchmark,
not end-to-end chat latency or a guarantee that every snapshot is faster.

## Validation scope

Local validation uses Windows, CUDA 12.8 and an RTX 5060 Ti. POSIX queue behavior
has a synthetic single-slot regression test; this is not a Linux runtime test.
Measurements and final payload identities are recorded in the accompanying
benchmark artifact. Model comparisons retain context 131,072 and the same
weights, temperature, seed, batch settings and memory budgets on both sides.
Small prompt benchmarks do not represent throughput at a full 131,072-token
context or prove answer quality for every model.

Passed checks include the six Go package suites, SSE/tool regression tests,
deferred multi-backend transfers with injected failures, VMM reads/writes with
no precommitted pages, and renewed buffer classification between operations.
Host snapshot baseline/copy/scatter/failure tests passed 4/4 on CPU and 4/4 on
CUDA with no skipped cases. The KV-growth regression retained bitwise replay
after quota denial and recovery. The MoE fixture's 2,331 logits remained within
the existing tolerance, with the same top selection at all nine positions.
All 42 native server component cases and 157 assertions passed.

## Full-model comparisons

The sanitized [measurement artifact](benchmarks/runtime-efficiency-20261026.json)
contains payload hashes, model digests, actual options, samples and output
hashes. The benchmark now defaults to the same optional kernel/KV settings on
both payloads; ablations require explicit flags. Both optional features were
disabled for the comparisons below.

| Case | Mean warm tokens/s, baseline / candidate | Observation |
| --- | --- | --- |
| Dense Qwen9B, 55-token prompt | 34.42 / 34.83 | Four warm repeats per payload. |
| Dense Qwen9B, 506-token prompt | 43.00 / 35.54 | Apparent regression; required a separate confirmation. |
| Same dense long prompt, confirmation | 41.17 / 41.57 | Three warm repeats; the large drop did not reproduce. |
| Qwen Flash MoE, 27-token prompt | 1.209 / 1.315 | Two warm repeats, 16 generated tokens. |
| Same MoE baseline repeated after candidate | 1.318 | The apparent MoE gain also did not reproduce against this control. |

Each dense request generates 128 tokens. Warm requests reuse the same prompt
prefix. All matched visible text and thinking fields are identical; thinking
is disabled in these runs. The original anomalous dense result is retained in
the artifact. Clocks, thermal state and external load were not continuously
controlled, so no general decode speedup or universal nonregression is claimed.
The confirmed benefit is lower overhead in the separately tested operations.

The observed managed CUDA peak is unchanged: 12,444,705,024 bytes for the dense
model and 12,658,688,384 bytes for Flash. Flash's managed host-weight cache stays
at its 6 GiB test quota. The changes do not shrink resident weights or the
logical KV cache. No model/profile setting or experimental option was enabled
in the installed configuration as part of this work.

## Further reliability and efficiency changes

The next changes build on `9b8b11ae`. Native changes are carried by
`009-bounded-copy-pipeline.patch` and `010-scheduler-exception-contract.patch`.
Validation for this round uses small synthetic models and fixtures only; no
new Qwen9B or Flash throughput comparison is claimed.

### Cancelled streams release the runner

Generate, native/rendered Chat and Pull/Push response producers now stop sending
when their consumer has disconnected or failed to write. Previously a producer
could block on an abandoned channel while holding a runner's semaphore. The
consumer can also exit while waiting for the next token. Cancelling inference
after a parser error still delivers that error through the live response channel.

HTTP tests disconnect the client or fail its writer, then require the handler
and runner lease to finish and the next request to succeed. The disconnect
regression fails against the old handlers using a Go overlay; focused race
checks pass against the correction. This prevents wasted/stuck work rather
than increasing the speed of an uninterrupted generation.

### Smaller embedding preparation retention

Embedding preparation releases token arrays after each window instead of
retaining the whole request's token cache until inference ends. It keeps the
same global stable cost sort, batch packing, deduplication and result order.
Single-item fallback accepts already prepared text and retokenizes only when
an actual 413 requires a new truncation attempt.

In the synthetic 4,096-document, 1,024-token fixture, token arrays referenced by
the cache peak at **512 KiB instead of 32 MiB**. Both versions issue exactly 64
batched tokenization calls and 586 embedding calls, with no extra individual
tokenization. This is retained token backing, not total process RSS or VRAM;
Go's collector determines when unreachable allocations are reclaimed. Total
cumulative allocations remain approximately 61 MB in both versions.

Finite embedding components can also overflow their float32 sum of squares.
That case previously produced zeroed vectors. It now recomputes only the
overflowing normalization in float64. Ordinary vectors, zero/tiny values and
the existing `1e-12` floor retain their previous bit patterns; NaN and infinity
inputs continue to be rejected.

### Bounded MoE upload fallback

When an expert-copy batch exceeds the host-weight RAM quota, the pooled GPU
fallback now reuses the existing pinned upload pipeline for individual bounded
chunks. It replaces synchronous uploads of scattered 64 KiB pages without
enlarging the 4 MiB staging buffer or the host-weight quota. CPU and unavailable
staging backends retain their fallback path. `copy()` still completes its
destination before returning and drains submitted work on failure.

The small test copies 6 MiB plus 47 bytes through an offset view, preserving
destination gaps. It records 49 uploads and two explicit backend fences under
a 128 KiB RAM quota, or 13 uploads and two fences under 512 KiB. The previous
code's loop required 97 synchronous page uploads plus 50/14 explicit fences
for those ranges (derived from the old loop, not an elapsed-time benchmark).
Tests cover renewed event slots, empty copies, truncated reads and recovery.
Fewer calls do not imply fewer required SSD bytes or a measured model TPS gain.

### Correct scheduler exception boundary on Windows

MSVC `/EHsc` could omit a caller's catch around the public scheduler compute
APIs because their `extern C` declarations were assumed not to throw. Managed
storage callbacks can throw on read/allocation failure. The two C++ declarations
and definitions now explicitly use `noexcept(false)`; C declarations, symbols
and compiler-wide exception flags are unchanged.

A real GGML CPU fixture reproduces the old bypass and requires local handling
for synchronous/asynchronous execution and preparation/evaluation failures,
followed by explicit draining and exact changed-input recovery. It does not
promise recovery from arbitrary CUDA driver failure.
Callers must be rebuilt against the corrected declaration; replacing a GGML
DLL alone cannot change exception handling already compiled into an old caller.

### Accurate, lighter activity parsing

The progress parser now discards an entire oversized log line through its
newline. Previously its suffix could be mistaken for a new memory/progress
record, affecting an observed VRAM value. Complete lines are parsed directly
from bytes; polling snapshots also copy scalar and layer pointers so a caller
cannot mutate the runner's internal state through a snapshot.

For 64 ordinary log lines, the isolated framing benchmark goes from 128
allocations/8 KiB to zero allocations; its median time goes from approximately
14.8 to 9.6 microseconds. This measures `observeActivity`, not the complete log
writer, GUI polling or inference. Fragmented/oversized framing, recovery and
snapshot independence have regression tests.

The [small-test artifact](benchmarks/runtime-reliability-20261026.json) records
the final payload hashes and this round's checks. All six Go package suites
and focused race checks passed. The rebuilt scheduler, pooled RAM CPU/CUDA,
dense placement, state transfer, MoE logits and server component fixtures passed.
Patch replay through 010 is repeatable and the source-integrity guard passed.

The real HTTP smoke uploads a 436,576-byte synthetic GGUF to a separate local
store and uses CPU only, two threads and context 256. Generate and rendered
chat pass in streaming/nonstreaming modes; closing an active stream permits the
next request to finish. No production models are used by this smoke, and the
owned server and its children are stopped afterwards. This is functional
validation, not a language-quality or large-model throughput comparison.
