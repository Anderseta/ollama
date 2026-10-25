# Coordinated MoE runtime validation

Validation recorded on 2026-10-25 for the experimental residency pipeline in
`llama/compat/006-moe-residency-pipeline.patch`. This report separates component
tests, tiny-model numerical regressions and full-model throughput. A component
PASS is not a claim that every model passes the numerical gate or runs faster.
The implementation and opt-in controls are described in
[moe-residency-pipeline.md](moe-residency-pipeline.md).

## Component evidence

The following logs are existing local build artifacts under
`../.build/gpu-first-moe/reports/runtime-units/`; they are not tracked test
fixtures. Expected allocation denials and the deliberately thrown CUDA provider
exception are rejection tests, followed by explicit PASS summaries.

| Test | Recorded result | Evidence |
| --- | --- | --- |
| Expert admission | PASS: cyclic retention, wide routes, rollback and changing workloads. The synthetic cyclic-48 case recorded 308 admission hits versus 0 LRU hits. This is a policy test, not a throughput measurement. | `test-moe-admission.log` |
| Expert cache budgeting | PASS: budgets, alignment, headroom and overflow boundaries. | `test-moe-budget.log` |
| Prompt partitioning | PASS: exact expert unions, maximal contiguous windows and token-order preservation. | `test-moe-partition.log` |
| Preparation barriers | PASS: explicit host barriers and deferred output readiness. | `test-prepare-async.log` |
| CUDA host provider | PASS: parent readiness, child arithmetic, capture exclusion, metadata and failure recovery. | `test-cuda-host-op.log` |
| Asynchronous CUDA upload | PASS: transfer dependencies, source-slot reuse, bounds and cleanup. | `test-cuda-upload.log` |
| Pooled host weights, CPU transport | PASS: exact reads/copies, slot reuse without recommit, cooperative eviction, shared readers, bounds and I/O recovery. The test's 131,072-byte weight budget was respected. | `pool-cpu.log` |
| Pooled host weights, CUDA transport | PASS for the same checks with GPU copies. Final run reported a 131,072-byte weight budget and a separate fixed 4 MiB pinned upload buffer. | `pool-cuda-final.log` |
| CUDA allocation quota | PASS: ownership, denials, concurrent reservations, frees after policy release, strict zero-cap case, retained compute pool/workspace and forced-pool mixed-type conversion. Conversion cleanup ended at 0 managed bytes. | `cuda-budget-final.log` |
| Native server components | 42 tests, 157 assertions, 0 failures, 0 exceptions, 0 skipped. | `server-component-final.log` |

The trace's standalone C++17 test also passed concurrent recording, the byte cap,
complete-record termination, numeric identity preservation and exclusive file
creation. The final capped artifact
`trace-final-7f9d2bdaee464c98846bbf7a1e1870d2.jsonl` is 3,723 bytes within its
4,096-byte cap. Its footer reports 33 recorded routes, truncation, one dropped
route and no I/O error. Its `.complete` companion reports one route, no
truncation and no drops. These outcomes are intentionally different; a capped
observation must not be replayed as complete.

The 10 Python replay tests passed. Their source is
[`test_replay_expert_cache.py`](../scripts/gpu-first/tests/test_replay_expert_cache.py).
They cover incomplete/malformed traces, sequence/footer validation, deduplication,
identity, RAM-hit upload costs, capacity boundaries, policy behavior and report
limitations. Replay estimates complete expert-bundle traffic; it does not
reproduce every native row-tile bank or predict tokens per second.

Targeted Go option/integration tests passed for `api`, `envconfig`, `llm`, `cmd`
and `server` in
`../.build/gpu-first-moe/go-moe-runtime-option-tests.log`. The subsequent recorded
Go checks also passed for those packages and `discover` in
`../.build/gpu-first-moe/go-moe-residency-final-tests.log`. These logs do not
establish that the entire upstream Go test suite was run. Patch-stack tests
record PASS for clean application, every partial prefix, repeated application,
index preservation and conflict preservation in
`../.build/gpu-first-moe/patch-stack-tests.log`.

## Numerical regressions with dynamic prompt windows

These tests compare the same tiny synthetic GGUF in resident CUDA and bounded
streaming CUDA modes. They retain the original weights, router selections and
context. The sequence includes a two-token prefix requesting no logits, a
32-token prefill, and eight decode steps, yielding 2,331 compared logits across
nine output steps. Requested context is 256; host-weight budget is 1 MiB.
The coordinated runtime and dynamic prompt provider are enabled for the
streaming side. Fixture provenance and the numerical harness are documented in
[fixtures.md](../scripts/gpu-first/fixtures.md).

The unchanged per-logit gate is
`abs(actual-reference) / max(1, abs(reference)) <= 2e-4`, with finite outputs
required. Matching top-1 alone does not satisfy that gate.

| Fixture | Expert-cache capacity | Maximum absolute/scaled error | Top-1 agreement | Verdict / local log |
| --- | --- | --- | --- | --- |
| qwen3moe Q4_K_M | 1 MiB | 0 | 9/9 | PASS; `runtime-units/qwen-dynamic-1m.log` |
| qwen3moe Q4_K_M | 32 KiB | 0.0000506653 | 9/9 | PASS; `runtime-units/qwen-dynamic-32k.log` |
| qwen3moe split F32, six GGUF shards | 32 KiB | 0.0000000135042 | 9/9 | PASS; `runtime-units/qwen-dynamic-split.log` |

All three logs report zero submitted CPU heavy operations and positive
GPU/cache/transfer activity. This covers quantized weight transport, a split
model, small banks requiring output-row tiles, multi-token prefill and decode.
It does not establish trained-model quality or support for every architecture,
quantization, CUDA kernel, context size or device.

### Known tied-F32 numerical failure

The tied/wide F32 llama fixture still fails the same strict resident-reference
gate. It was not converted into a pass or granted a wider tolerance. The three
recorded runs use context 256, prefill 32, a 128 KiB expert cache and a 1 MiB
host-weight budget.

| Streaming configuration | Maximum absolute/scaled error | RMSE | Outside tolerance | Top-1 agreement | Verdict / local log |
| --- | --- | --- | --- | --- | --- |
| Coordinated runtime off | 0.000648521 | 0.000156332 | 453 | 9/9 | FAIL; `runtime-units/llama-control-f32.log` |
| Coordinated runtime on, static windows | 0.000648521 | 0.000156332 | 453 | 9/9 | FAIL; `runtime-units/llama-runtime-static-f32.log` |
| Coordinated runtime on, dynamic windows | 0.000573695 | 0.000153223 | 453 | 9/9 | FAIL; `runtime-units/llama-dynamic-f32.log` |

Runtime off and runtime on with static windows have identical recorded numerical
summaries and cache/transfer summaries. That is evidence against a new regression
in this particular comparison; it is not a direct bitwise comparison of their
full logit arrays. Both already fail against resident execution. Dynamic windows
change batch dimensions and can change CUDA kernel dispatch and rounding, so
their different summary also remains a failure requiring investigation. All
three outputs are finite. The smaller dynamic maximum error does not establish
an accuracy improvement.

Earlier architecture coverage is recorded separately in
[fixtures.md](../scripts/gpu-first/fixtures.md) and
[`gpu-first-20261025.json`](benchmarks/gpu-first-20261025.json). That earlier
matrix contained 8 passes, 9 finite numerical failures and 2 unsupported cases;
two additional synthetic Nemotron state-128 cases passed. Those results predate
this coordinated-runtime validation and must not be represented as a fresh
all-architecture pass for patch 006. The earlier resident batch-variation
diagnostics also exceeded the gate in failing cases; they explain a comparison
dimension without clearing the streaming failures.

## Flash benchmark conditions

The full-model target is the existing
`qwen3.8-flash-next-uncensored:q4_k_m`. No re-quantization, pruning, top-k change
or context reduction is part of this comparison.

| Item | Recorded condition |
| --- | --- |
| Host | Windows; AMD Ryzen 5 2600; approximately 32 GiB installed RAM |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GiB; driver 621.01 |
| Storage | Existing model on a SATA SSD; laboratory artifacts on separate non-system volumes |
| Context capacity | 131,072 tokens; the short prompt does not fill that capacity |
| Managed host-weight limit | 6,144 MiB (6 GiB) |
| Managed CUDA allocation limit | 12,288 MiB (12 GiB) |
| Fixed expert GPU cache | 1,024 MiB requested for the fixed-budget comparison |
| Sampling | Temperature 0, seed 42, thinking disabled |
| Request length | At most 32 output tokens per round |
| Repetition | Two rounds: a new-runner request followed by a repeated request |
| Diagnostics | Detailed GPU diagnostics and expert trace disabled for throughput comparisons |
| Isolation | Separate laboratory payload/listener; production installation and listeners retained |

The RAM number is a managed model-weight budget, not a limit on the entire
process. The CUDA number covers allocations charged to the instrumented
allocator, including the dynamic provider's workspace; it excludes display,
driver and other allocations outside that allocator. Metadata, application
heaps, KV/compute host allocations and fixed I/O/pinned staging need separate
headroom. See [gpu-first-memory-validation.md](gpu-first-memory-validation.md)
for the exact accounting and platform limitations. Sampled OS counters do not
prove the absence of a short-lived peak or OS paging.

Each two-round run uses the same synthetic counting task. The first round is
runner-cold: a new server/runner, without flushing the OS page cache. The second
round repeats the prompt and can reuse prompt/KV state as well as weight caches.
It therefore measures that repeated-request condition, not a fully independent
warm prompt.

Keep control and coordinated runs at identical model, context, fixed cache,
quotas, sampling and output counts. Measure automatic cache sizing separately
because changing the cache allocation is a different comparison. Preserve each
report's payload/model hashes and effective runner environment with the results.
Local reports are under `../.build/gpu-first-moe/reports/`.

Report first-output latency separately from decode throughput. API tokens/s is
`eval_count / (eval_duration_ns / 1e9)`; the supplementary decode-steps rate uses
`max(eval_count - 1, 0)` in the numerator. State the token count beside either
rate. Record SSD and H2D byte deltas per output token, weight-cache peaks and
managed CUDA peaks without adding overlapping traffic counters together.
Nested child-provider matmuls/copies are not included separately in the parent
submitted-operation count, so that count cannot be compared one-for-one with
the static graph.

Two rounds of 32 tokens are a short operational check, not a statistically
repeated sustained-throughput benchmark or a language-quality evaluation.
Longer generations, unrelated prompts, repeated independent runs and broader
numerical/quality checks remain necessary before general performance or
all-model correctness claims.

## Flash measurements: short 32-token checks

The five completed reports each contain two successful responses of 32 tokens,
with `done=true` and `done_reason=length`. All ten generated texts and empty
thinking strings match exactly. Every repeated round reused 28 prompt tokens.
This checks this short task's output preservation; it is not a general quality
evaluation. A sanitized, machine-readable summary with source hashes, native
payload hashes, counters and source line numbers is available in
[`moe-residency-20261025.json`](benchmarks/moe-residency-20261025.json).

The labels below identify report basenames under
`../.build/gpu-first-moe/reports/`, each ending in `-results.json` and
`-server.log`:

- **Control:** `runtime-control-count32`, coordinated runtime off and fixed cache.
- **Rejected candidate:** `runtime-fixed-count32`, the earlier runtime with
  unrestricted cooperative eviction preference. It completed but regressed;
  it is retained as failed performance evidence.
- **No-cooperation ablation:** `runtime-no-coop-count32`, the same earlier native
  payload with `WILDLLAMA_MOE_COOPERATIVE_CACHE=0` and fixed cache.
- **Corrected fixed cache:** `runtime-fixed-bounded-count32`, bounded cooperative
  preference with the same requested 1,024 MiB cache as control.
- **Corrected automatic cache:** `runtime-auto-bounded-count32`, the corrected
  native payload with automatic expert-cache sizing. Its final bank is
  6,688.38 MiB, approximately 6.53 GiB, versus the initial aligned 1,021.39 MiB
  bank. This is a different cache-capacity comparison within the same 12 GiB
  total managed CUDA quota.

| Run | Cold API tokens/s | Warm API tokens/s | Cold first output, s | Warm first output, s |
| --- | --- | --- | --- | --- |
| Control | 0.608 | 0.678 | 71.953 | 8.953 |
| Rejected candidate | 0.311 | 0.393 | 79.844 | 8.656 |
| No-cooperation ablation | 0.689 | 0.844 | 69.516 | 7.703 |
| Corrected fixed cache | 0.686 | 0.824 | 72.391 | 7.797 |
| Corrected automatic cache | 0.982 | 1.860 | 67.938 | 5.766 |

The rejected candidate was not accepted merely because it reduced H2D traffic.
Its SSD traffic increased and generation slowed. Disabling cooperation identified
that combined component as a useful diagnostic target. The correction bounds
cooperative preference to 64 pages per queue and preserves SLRU's
probation-before-protected eviction order. The ablation does not isolate the
contribution of each subcorrection independently.

At fixed requested cache capacity, the corrected runtime's warm rate is 21.4%
higher in this single short pair. Its cold first-output latency is slightly
higher than control. The automatic-cache run's 1.860 warm tokens/s also uses a
substantially larger expert bank; it must not be described as the fixed-budget
algorithmic gain. These observations are preliminary, without a statistical
speedup claim.

### Request traffic and counter precision

SSD deltas use the final `moe_ram` snapshot immediately before each request's
`slot print_timing` prompt line. Those cumulative counters are printed to
0.01 MiB. Warm deltas subtract the preceding request boundary; cold starts at
zero and includes initialization/prefill activity, rather than isolated decode.
These are application-issued weight reads, not measurements of physical SSD
controller traffic or proof of page-cache misses.

H2D deltas use the last `wildllama_activity` RAM snapshot preceding the same
boundary. Activity emission is sampled and can lag the last upload, so these
deltas are approximate. They are not exact end-of-request transfer totals.
Dividing either request total by 32 gives request traffic per output token,
including prompt work, rather than decode-only traffic.

| Run / round | SSD read delta, MiB | SSD MiB/output token | Approx. H2D MiB/output token |
| --- | --- | --- | --- |
| Control / cold | 34,969.38 | 1,092.79 | 2,121.84 |
| Control / warm | 13,744.72 | 429.52 | 1,530.89 |
| Rejected candidate / cold | 44,720.23 | 1,397.51 | 1,782.35 |
| Rejected candidate / warm | 20,099.43 | 628.11 | 1,119.17 |
| No-cooperation / cold | 34,768.71 | 1,086.52 | 1,782.30 |
| No-cooperation / warm | 13,136.81 | 410.53 | 1,113.72 |
| Corrected fixed / cold | 34,768.71 | 1,086.52 | 1,781.99 |
| Corrected fixed / warm | 13,136.81 | 410.53 | 1,113.20 |
| Corrected automatic / cold | 30,111.24 | 940.98 | 1,117.61 |
| Corrected automatic / warm | 6,369.18 | 199.04 | 294.09 |

### Observed memory and API reporting correction

The RAM log's managed-weight peak was 6,144 MiB in every run. The activity JSON
also reached exactly 6 GiB of managed RAM in every run. The CUDA column below is
the maximum **live managed allocation snapshot observed in activity JSON**,
rather than a separately queried allocator high-water mark. NVML is sampled
once per second and covers the whole physical device, including display,
driver and unrelated processes. Neither observation proves that every
short-lived peak was sampled.

| Run | Max observed managed CUDA, MiB | Max sampled global NVML use, GiB |
| --- | --- | --- |
| Control | 6,400.65 | 9.72 |
| Rejected candidate | 6,405.27 | 8.82 |
| No-cooperation ablation | 6,405.27 | 8.90 |
| Corrected fixed cache | 6,405.27 | 7.81 |
| Corrected automatic cache | 12,072.27 | 14.49 |

All observed managed CUDA values are below the 12,288 MiB quota. The automatic
run's global NVML value exceeding 12 GiB does not measure the runner's managed
allocation budget, and the lower global values in other runs do not establish
that those changes reduced driver/display or other-process memory.

These 32-token reports predate the Go live-memory reporting correction. Their
`/api/ps` `size_vram` values remain around 3.62 GiB even when activity observes
approximately 11.79 GiB of managed CUDA allocations. That older estimate
undercounts runtime allocations/cache growth and must not be presented as actual
VRAM use. Its corresponding model-size estimate is also not a measurement of
weight RAM or process RAM. A separate Go correction uses the runner's live
managed CUDA accounting; its validation is separate from these preserved
pre-correction reports. The 128-token runs below include that Go correction.

## Flash measurements: 128-token natural explanation

`runtime-control-prose128` and `runtime-auto-prose128` each completed one
runner-cold request with the same natural explanatory prompt, model, sampling
and final payload. Both returned 128 tokens with thinking disabled and
`done_reason=length`; neither response is a completed full explanation. Context
remained 131,072 tokens, managed host weights 6 GiB and managed CUDA 12 GiB.
Control used the aligned 1,021.39 MiB expert bank; automatic sizing used
6,688.38 MiB. No unrelated-prompt warm round or statistical repetition was run.

| Run | API tokens/s | First output, s | Prompt evaluation, s | Generation evaluation, s | Total elapsed, s |
| --- | --- | --- | --- | --- | --- |
| Control | 0.514 | 119.437 | 98.379 | 248.868 | 368.297 |
| Corrected automatic cache | 0.800 | 108.609 | 85.220 | 160.007 | 268.609 |

The observed API decode rate is 55.5% higher and total elapsed time 27.1% lower
for automatic sizing in this pair. This compares the complete runtime and its
larger expert cache within unchanged total quotas; it does not isolate a gain
at equal expert-cache capacity.

The texts differ beginning with the second word, although both attempt the same
kind of explanation. Their subsequent token and routing trajectories therefore
cannot be assumed identical. Different CUDA batching/kernel rounding is a
possible explanation for divergence, not an established cause. This is an
end-to-end operational measurement on different generated continuations,
without a quality-equivalence or same-trajectory proof. Both hitting the length
limit further prevents a completeness claim.

Using the same counter-boundary method as above, SSD traffic per output token
was 805.91 MiB for control and 658.77 MiB for automatic sizing; sampled H2D was
approximately 1,725.15 and 794.91 MiB per output token respectively. These cold
request totals include initialization/prefill, and the H2D endpoint remains
approximate.

Both runs reached 6 GiB of managed host weights. Maximum observed managed CUDA
was 6,711,565,184 bytes for control and 12,658,688,384 bytes for automatic sizing,
within the 12 GiB quota. The corrected `/api/ps` `size_vram` matched the last live
managed CUDA snapshot in each run. It remains a runner-managed allocation
counter, not total physical-device or process memory. The sanitized summary's
`longer_natural_validation` section preserves the separate source/payload hashes
and scopes for these runs. More prompts and independent repetitions are required
before generalizing the observed gains.

## Larger speedups: speculative decoding remains unvalidated

Inspection of the local Flash GGUF shards found trunk blocks 0 through 47 and
no nextn/MTP-head tensors. The current target therefore does not contain the
draft head needed for self-speculation. The publisher offers a separate
approximately 4 GB MTP sidecar and reports roughly 1.3-2x acceleration on its own
setup; that claim is not a measurement or prediction for this machine.
See the [publisher's MTP section](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF#speculative-draft-mtp).

The native runtime already has draft-head execution and batched target
verification. This stage also preserves the configured external sidecar path
when selecting an explicit draft algorithm, and accepts `draft-mtp` and
`draft-eagle3` through the Go API. Those routing fixes do not certify the
target/draft combination under the memory quotas.

Joint budgeting remains necessary: reserve draft weights, KV and workspace
before automatically sizing the target's GPU expert cache. The current
managed-MoE draft path fails with a zero expert cache, while consuming the
remaining VRAM for the draft can shrink target residency and increase SSD/H2D
traffic. We have not downloaded or tested this head. Acceptance rate, end-to-end
speed, output correctness and combined memory accounting must be measured before
claiming a benefit.

### Observed expert overlap diagnostic

A separate Flash route trace contains 863 records and 15 complete single-token
decode frames spanning evaluations 4-18, each with all 48 layers. Analysis
ignored 141 multi-token records and two incomplete/mixed frames. Forced runner
termination left no completion footer; `--allow-incomplete` was explicitly used
for this valid prefix. Trace v1 does not identify request/warmup boundaries, so
the first complete frame was not silently discarded. No conflicting duplicate
evaluation/layer routes were accepted.

Across 12 overlapping four-token windows, the mean byte-weighted expert union
is 3.735 billion bytes versus 6.291 billion bytes without within-window reuse:
relative raw bundle traffic 0.59368 under perfect acceptance and perfect reuse.
With the illustrative constant conditional acceptance `p=0.67` and three draft
tokens, expected emission is `1+p+p^2+p^3 = 2.419663` tokens. Charging that same
observed four-token union gives relative raw bytes per expected token 0.98143,
before draft-head cost or existing caches.

These are true-token routes, not actual draft predictions or rejected-token
routes. The calculation excludes residency, eviction, SSD/H2D scheduling,
shared weights, draft weights, KV and arithmetic. Its overlapping windows are
dependent and its interrupted prefix is short. It is neither a TPS prediction
nor a verdict against MTP: real acceptance and cache/kernel behavior need a
joint-budget target/draft benchmark. The publisher's speed claim alone does not
establish a large SSD-traffic reduction on this machine. Source hashes and
methodology are preserved in the summary's `expert_overlap_diagnostic` field.
