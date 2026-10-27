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

## Bounded MoE batching inspired by Strata

Patch `011-moe-read-and-execution-batching.patch` adds the following changes on
top of `3e54c375`. The design was informed by a source review of Strata commit
`4b9e05a2a806a3fe24f7b2560d98626c37dade35`; this does not copy its approximate
expert skipping or its GPU spin waits. Target weights, quantization, selected
experts, context and configured memory quotas remain unchanged.

- The coordinated pooled RAM reader combines same-file spans across tensor
  reservations. It accepts overlaps and small holes only within the existing
  staging slot, with a 16 KiB maximum individual hole and at most 1/32 additional
  aligned traffic relative to useful payload. Only requested pages enter RAM.
  The 4 MiB staging allocation plus alignment, four Windows I/O slots and single
  POSIX slot remain unchanged. Set `WILDLLAMA_MOE_READ_COALESCE=0` for the original
  same-tensor grouping; the nonpooled reader retains its original grouping.
- Expert uploads merge only consecutive copy ranges contiguous at both the
  source and destination. Tiled source gaps, different tensors, different banks
  and overflow retain separate copies. Oversized merged ranges still use the
  existing bounded fallback. This does not reorder routed experts.
- Dynamic prefill retains at most four child graph contexts and independent
  host ID arrays before draining. The same child stream orders uploads, matmuls
  and output copies; bank upload events protect earlier consumers. Shared GPU
  workspace size does not grow. End-of-provider and exception paths drain before
  releasing metadata. `WILDLLAMA_MOE_EXECUTOR_BATCH=0` selects a one-work batch.
- Layer capacities are computed at cache initialization, and actual route IDs
  and exact prompt-window partitions are reused within an evaluation. This
  avoids repeated all-layer scans, route vector copies and partition passes.
- The existing `speculative_adaptive` option now refreshes its plain-decoding
  reference, observes the actual proposed span and backs off according to wasted
  cost. Failed probes retry after 32-1024 plain steps; useful drafting is checked
  against two fresh plain observations after at least 512 output tokens.
  Invalid clocks/cancelled steps are discarded, and recurrent replay cost stays
  in the observation. Target verification and sampling are unchanged. This
  neither enables a missing MTP head nor changes saved profile choices.

Only small tests were run, on Windows with CUDA 12.8 and the RTX 5060 Ti. The GPU
was shared with other applications, so these results are component observations,
not an isolated inference-speed benchmark:

| Small fixture | Before / after |
| --- | --- |
| Four nearby tensor regions | 4 / 1 SSD reads; 205,270 / 200,939 bytes; identical 327,680-byte managed RAM peak |
| 192 adjacent expert uploads | 192 / 3 actual CUDA uploads; same bytes and 1 MiB RAM quota |
| Same warm upload fixture | Median 5.0892 / 0.1914 ms, seven alternating pairs; excludes SSD reads and model inference |
| 32 queued child operations | 32 / 8 explicit drains, including shared GPU scratch/ID reuse |

Thirteen CTest components passed, including CPU/CUDA transfer and lifetime
fixtures. All 44 server component cases and 184 assertions passed. Legacy RAM,
pooled CPU and pooled CUDA fixtures passed. Tests cover truncated reads and
recovery, partial pages, shards, overlapping regions, destination guards,
overflow, quotas and invalid speculative timing.

Sixteen small synthetic-model comparisons cover Qwen3MoE, Qwen2MoE, Gemma4 and
GPT-OSS/MXFP4. They include fixed caches, multiple row tiles, output-free prefixes,
consecutive decode steps and serial/batched controls. All 2,331 logits per case
were finite and within the existing scaled tolerance of 0.0002; top-1 agreed at
all nine positions per case. Maximum observed scaled difference was 0.0000943127.
These are numerical regressions, not an assessment of real-model language
quality. No Qwen Flash or other large model was loaded or downloaded.

The portable planner is tested through Windows builds; this round does not
claim a Linux runtime test. SSD prefetch prediction, a full persistent CUDA
window and hybrid CPU/GPU expert arithmetic are not added by this patch. The
small-component improvements do not predict tokens/s for a future IQ3_S model.

## Expert reuse, packed rows and actual draft budgets

Patch `012-expert-major-and-packed-rows.patch` continues the review of the same
Strata commit, particularly its expert-major prompt execution and pinned upload
rings (`include/strata/prefill/prefill.hpp` and `moe_mmq.hpp`). The implementation
uses the existing WildLlama cache, GGML operators and reserved workspace. It does
not drop routed experts, change quantization or require a converted model.
Expert grouping and row packing require the existing opt-in coordinated MoE
runtime; grouping additionally requires its GPU-exclusive dynamic-prefill path.
This patch does not enable these modes or change saved profiles. See
[activation and memory scopes](moe-residency-pipeline.md).

### Reuse expert weights across prompt tokens

When the token-window executor repeatedly uploads the same experts, the new path
groups routed entries by expert across the prompt. Gather and scatter operations
restore each original token/rank position before the existing weighted reduction.
Gate/up shared inputs and down-projection distinct rank inputs are both covered.
Weights remain available while all bounded chunks for their group are processed.

This path requires at least 32 tokens, more globally selected experts than cache
slots, at least twice as many estimated window uploads as distinct experts and
at most twice the old graph-work count. Its gather, output and index arrays fit
the existing GPU workspace. The child stream, upload fences and four-work
metadata retention bound remain in force, including draining on exceptions.
Quantized execution queries the CUDA backend's actual MMVQ batch limit and caps
both eligibility and grouped chunks accordingly. Unsupported backends, shapes,
precision/kernel combinations or insufficient workspace retain the previous
executor. F32 weights retain the previous executor: changing its matrix batch
shape exceeded the existing full-model numerical tolerance in a stress fixture.
Set `WILDLLAMA_MOE_EXPERT_MAJOR=0` to disable this optimization. Optional
`WILDLLAMA_MOE_EXPERT_MAJOR_TRACE=1` reports only actual executions.

### Pack small host-weight rows before uploading

Selective token/PLE row copies now pack their original quantized bytes, in caller
order, into the existing two pinned upload slots. CUDA still performs the original
dequantization. Repeated rows, tensor views, partial pages and destination guards
are preserved. Source pages are admitted in groups within the existing RAM quota;
an individual oversized row uses bounded reads. Unsupported layouts/backend
queues retain the copy-list path. `WILDLLAMA_MOE_PACK_ROWS=0` disables packing.

The small Q5_0 CUDA fixture uploads 8,192 rows in one call rather than 8,192 calls,
with identical payload bytes. Larger cases cross upload-slot boundaries. Tests
also truncate the source after a packet has already been submitted, require
draining and then restore the file and retry. This reduces calls and fences;
it does not change the 64 KiB disk-page granularity or promise equal SSD savings.

### Apply speculative depth before computing drafts

MTP and EAGLE previously could compute the global draft maximum and discard the
excess after a smaller per-sequence request. They now honor each sequence's budget
before its next forward pass. Zero disables drafting, negative uses the global
maximum and positive values are capped by it. Simple drafting follows the same
rule. Early EOS and checkpoint ownership are retained; an unused final hidden
state download is omitted. Target verification and sampling remain unchanged.
This fixes wasted work only when these speculative modes are already enabled.

### Experimental RAM history stays optional

`WILDLLAMA_MOE_RAM_ADMISSION=1` opts the pooled reader into bounded frequency-aware
victim selection. Small aging counters fit existing page-record padding and the
victim scan is capped at 64 candidates. Recency aging lets old popular pages go.
The ordinary SLRU policy remains the default and performs no history updates.

Real file traces under a fixed 4 MiB quota saved 23.17% of SSD reads on a hot-set
plus scan workload, 0.24% on one skewed workload, and none on sequential/uniform
cycles. A changing hot set incurred 7.64% more reads. This tradeoff is why the
option is not enabled by default or in saved profiles.

The [validation artifact](benchmarks/moe-expert-reuse-20261027.json) records
small-test results, controls and payload hashes. Component measurements are not
large-model tokens/s measurements. No large model was loaded or downloaded.

The final small suite includes 23 CTest components, 44 server cases, legacy and
pooled RAM tests, a CPU-only HTTP smoke and 24 CUDA synthetic-model comparisons.
The latter include serial/batched controls and actual expert-major execution
under fixed cache pressure. Every approved comparison retains the existing
0.0002 scaled-logit tolerance and agrees on top-1 at all nine output positions.
In the fixed-cache Qwen3MoE test, total recorded H2D bytes decrease from 9,526,896
to 3,733,104; the Gemma4 test decreases from 6,475,344 to 2,788,944. The same
fixtures with a cache large enough for their expert union retain the old path.

An additional diagnostic forced MXFP4 into a 32 KiB cache with auto-sizing off.
It exceeds the resident-reference tolerance equally with expert-major enabled
and disabled (scaled error 0.0117443). The legacy executor changes from MMQ to
MMVQ, whose Q8 activation scales use different precision. This is recorded as an
unresolved baseline numerical limitation, not counted among passing cases and
not addressed by relaxing tolerance. Its normally auto-sized cases pass.
Linux portability was reviewed statically; this round ran Windows tests only.

### Preserve incomplete-generation signals in the Go API

A clean HTTP EOF without a generation terminal marker now returns an explicit
error in chat and completion. Partial text remains available, cancellation keeps
its original error, and this condition never restarts the runner or retries the
generation. A completed request still releases its stream before its final
callback, preserving reentrant tokenization.

Native chat maps both `length` and `max_tokens` to `done_reason: "length"`.
If that terminal event contains invalid tool arguments, the entire buffered tool
batch is discarded and the response adds `tool_calls_truncated: true`. Valid
batches retain their tools and length reason; invalid arguments under `stop`
remain errors. The flag is omitted otherwise. Clients can request complete,
smaller calls without receiving or executing part of an incomplete batch.
OpenAI conversions preserve length ahead of the tool-call finish reason.

The completion repetition guard ignores terminal and empty protocol events and
compares actual content, preserving distinct spaces, tabs and line breaks. Its
existing threshold still limits repeated nonempty content. Sampling, requested
context capacity, model weights and cache policies are unchanged.

Local HTTP fakes cover missing terminal markers, cancellation, repetition,
all-or-nothing tool batches, metrics and streaming/nonstreaming API propagation:

```sh
go test ./llm ./server ./openai -run 'TestLlamaServer(ChatRequiresTerminalMarker|CompletionRequiresTerminalMarker|MissingTerminalPreservesCancellation|ChatToolCallLimit|CompletionTerminalMarker|CompletionRepeatGuard)$|TestNativeChatToolCallTruncationFlag$|TestFinishChunk$|TestToChatCompletion_FinishReasonPrecedence$' -count=1 -timeout 90s
```

These are Go-only changes: a staged `go build -o <staging>/ollama.exe .` can reuse
the installed native payload. Set `GOCACHE`, `GOMODCACHE`, `GOPATH` and `GOTMPDIR`
to the installation's persistent cache/temp directories before testing or
building; no native payload rebuild or model download is required.
