# GPU-first memory validation

This describes the experimental GPU-first allocator and host-weight cache in this
branch. It distinguishes enforced backend allocation quotas from operating-system
memory totals. Nothing in this document changes the installed production runner.

## What each limit covers

| Setting or counter | Covered memory | Excluded memory |
| --- | --- | --- |
| `moe_ram_limit_mib` | Committed and locked pages of host model weights managed by the selective cache, including 64 KiB page padding | KV, compute buffers, model metadata, application heaps, Go/server memory, diagnostic maps/JSON, I/O buffers and pinned upload staging |
| `gpu_vram_limit_mib` | CUDA allocations made through the instrumented backend allocator: resident weights, KV/compute buffers, expert cache, retained legacy pool allocations, VMM physical storage and backend-owned cuBLAS workspaces | Driver/context allocations, CUDA event resources, internal library allocations outside the backend allocator, display memory and other processes |
| `managed_used_bytes` / `managed_peak_bytes` | Successful live backend allocations charged to the physical CUDA device; pending allocations already reserve quota internally | Total physical GPU residency reported by Windows or `nvidia-smi` |
| `ram_resident_bytes` / `ram_peak_bytes` | Current/peak locked weight-cache pages | Total process working set or committed RAM |

Here, **managed allocation** means an allocation accounted for by this backend.
It does not mean CUDA Unified Memory. An enabled strict allocation policy rejects
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` and uses ordinary CUDA device allocations.

An enabled VRAM policy with limit `0` has no explicit byte quota. It still tracks
allocations, rejects Unified Memory and checks physical free memory. With the
policy disabled, ordinary loading keeps the legacy allocation path. A nonzero
VRAM limit or GPU-exclusive mode activates the policy; instrumentation alone does
not impose a VRAM quota.

The RAM weight cache also reserves one fixed locked I/O buffer, outside its weight
quota: 4 MiB plus alignment padding (16 KiB on Windows, 4 KiB on Linux in this
implementation). A CUDA backend with genuine asynchronous pinned-host support can
add one 4 MiB upload buffer per device. That buffer is split into two 2 MiB slots;
it does not grow with model size or the number of uploads. Slot events prevent the
CPU from overwriting a half before its prior transfer finishes. Final backend
synchronization finishes all transfers before source cache pages can be reused.
Backends without the required pinned/event facilities retain bounded synchronous
fallbacks.

Thus, a 10 GiB weight-cache limit is **not** a 10 GiB total-RAM limit. Budget the
context, staging buffers, runtime state, metadata and the application separately.

## CUDA policy lifetime and failure behavior

The policy is shared by the physical GPU inside one runner process. Virtual
devices backed by that GPU share the same counters. It is not a separate allowance
for every model or request. The initial GPU-first integration requires exactly one
CUDA device; multidevice/allreduce scratch paths are outside this validation.

Each successful `ggml_backend_cuda_set_memory_limit(device, limit, true)` registers
an owner. Additional owners must request the identical limit. Model/draft cleanup,
including failed loading, must release one registration with `enabled=false`.
Only the final release disables enforcement. Existing tracked buffers remain
charged until their real free/unmap, even if the policy has been released.
Enabling over a live legacy, untracked CUDA allocation fails; it does not guess
the size of buffers allocated before tracking began.

Quota is reserved before calling the driver, including concurrent allocations.
Failed allocations roll back their reservations. Host metadata allocation failure
also frees the new CUDA allocation before rolling back its charge. A pool returning
an allocation to its internal cache does not return the quota: the retained memory
is still physically allocated. VMM storage remains charged after releasing its
allocation handle and is decremented only after unmapping the physical storage.
The virtual-address reservation itself is not charged as physical VRAM.

The allocator additionally checks `cudaMemGetInfo` and keeps 64 MiB of physical
free-memory headroom at each allocation check. This is a guard, not a reservation
that prevents other applications from consuming that memory. It is independent
of the explicit managed-byte quota. Internal driver/library overhead and allocation
granularity can make OS totals exceed the sum of requested backend allocation
sizes. Windows WDDM can make residency/paging decisions outside the application.
These controls therefore do not guarantee an OS-wide physical-RAM/VRAM maximum or
that Windows will never page any GPU resource.

Buffer allocation denial returns failure before an over-quota allocation. Some
compute-pool/workspace paths retain the backend's existing fatal CUDA-check
behavior: an infeasible configuration may terminate the isolated runner with an
explicit allocation error. The policy does not silently switch heavy operators
to CPU, enable Unified Memory or promise recoverable OOM for every native caller.

## Metrics and interpretation

RAM snapshots acquire the model weight-cache mutex. Code already holding
`cache.acquire()` must use `cache.stats(guard)` rather than recursively calling
`cache.stats()`. Standalone snapshots can wait for an active managed-weight graph;
they must not be requested from its own preparation/copy callback.

RAM/SSD/H2D counters are model-wide and cumulative. Request deltas may include
other contexts sharing the model. The RAM peak is a model-lifetime gauge. CUDA
allocation peaks and denial totals are physical-device allocator-lifetime counters
inside the process; they can survive model reloads and are not private request
peaks. A historical peak can exceed a later, smaller configured limit without a
current allocation violating that limit.

Detailed host timing is opt-in. `io_submit_us` measures submission work;
`io_wait_us` measures the host's blocking I/O intervals; neither is a direct SSD
controller service-time measurement. `host_copy_us`, `h2d_us` and `h2d_wait_us`
contain overlapping intervals and must not be added as independent phase times.
H2D counters cover tracked weight uploads, not every metadata or temporary copy
performed by CUDA. Cache-H2D and RAM-H2D counters can describe the same transfer
at different layers and must not be summed as distinct traffic.

Optional CUDA stream timing uses a bounded ring of 32 start/end event pairs per
backend. Events bracket backend graph execution outside graph capture; scheduler
weight-copy callbacks occur outside that interval. Collection queries completed
end events without synchronizing the host. A full ring drops measurements, not
inference work. Check `samples`, `pending` and `dropped` together with
`cuda_execution_stream_ms`; pending work is not yet included in the elapsed total.
The interval can include internal device copies, stream waits and host launch
gaps. It is **not pure kernel time** and is not a per-layer timing breakdown.
Default-off execution creates no timing events and performs no event queries.

The opt-in `layer_profile` is a separate **serialized host-wall** measurement
between tagged graph endpoints. It includes storage, transfers, computation and
the diagnostic barriers themselves; it is neither per-layer CUDA event time nor
a decomposition of `cuda_execution_stream_ms`. Untagged/shared graph segments
use layer index `-1`. This profiling can change overlap and latency, so compare
ordinary performance with diagnostics off and treat the enabled profile as an
attribution aid, not a low-overhead benchmark.

`expert_usage` reads each canonical FFN router result once and counts its selected
entries, rather than every gate/up/down weight or output-row tile access. The readback is immediate
at the router callback, before consumers can reuse its storage; top-k views are
read using their actual element/row/sequence strides. The histogram stores at
most **262,144 distinct layer/expert records**. New unknown selections beyond
capacity increment `dropped_selections`; existing keys continue to count. A
nonzero dropped count means the histogram is incomplete. The JSON exposes only
the top 32 entries alongside total/distinct/dropped counts. These diagnostic
maps, frame metadata, readback vectors and JSON allocations are outside the
managed host-weight RAM quota.

Layer/expert snapshot APIs require the context thread, with **no concurrent
decode/getter access**; they do not lock a separate diagnostic mutex. Their
cumulative values and request deltas still describe shared context activity,
not exclusive accounting for concurrently scheduled requests. Process and NVML
memory observations in the benchmark are sampled, rather than proofs that no
short-lived OS-memory peak occurred.

Event completion and elapsed-time semantics follow the
[NVIDIA CUDA Runtime event API](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__EVENT.html).
Vendor event aliases follow the official
[HIP event API](https://rocmdocs.amd.com/projects/HIP/en/develop/reference/hip_runtime_api/modules/event_management.html)
and [MUSA runtime guide](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/history_version/v5.1.0/programming_guide/api_guides/runtime_api_guide/).
HIP/MUSA builds and timing behavior have not been validated locally.

For performance comparisons, preserve the model file, context, KV types, prompt,
batch settings, output-token count and warmup choice. Report the first request
separately from later requests. A warm request with zero extra SSD reads can
still spend time on H2D, kernel execution, scheduling, cache replacement or graph
rebuild. A repeated prompt may also reuse its KV/prompt cache, so it is not by
itself evidence of a faster weight-cache implementation.

## Isolated validation procedure

Build/test output belongs in `.build/gpu-first-moe/out`, temporary fixtures in
`.build/gpu-first-moe/temp`, and logs in `.build/gpu-first-moe/reports`. Do not
replace `F:\WildLlama`, the official Ollama installation, their model stores or
their active ports to run these tests. Run GPU tests serially with no competing
model inference. All commands below are from the repository root after the native
source has been prepared and the validation wrapper has been built.

```powershell
$labRoot = Join-Path (Get-Location) '.build\gpu-first-moe'
$binRoot = Join-Path $labRoot 'out\bin'
$tempRoot = Join-Path $labRoot 'temp'
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
# This variable names a backend DLL, not a directory. Avoid inheriting an
# unrelated production plugin; the loader searches beside these executables.
Remove-Item Env:\GGML_BACKEND_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_ENABLE_UNIFIED_MEMORY -ErrorAction SilentlyContinue

& (Join-Path $binRoot 'test-cuda-memory-budget.exe')
& (Join-Path $binRoot 'test-gpu-layer-diagnostics.exe')
& (Join-Path $binRoot 'test-moe-ram-metrics.exe') $tempRoot
& (Join-Path $binRoot 'test-moe-ram-metrics.exe') $tempRoot --gpu
& (Join-Path $binRoot 'test-moe-ram.exe') --cache $tempRoot
& (Join-Path $binRoot 'test-moe-ram.exe') --cache-gpu $tempRoot
```

Do not infer success merely from the absence of a crash. Preserve each exit code
and log and require its explicit PASS message. Expected denial/error messages in
the budget/cache tests are part of the rejection cases, not a reason to ignore a nonzero
exit code. Tests create and remove only their own small temporary fixtures.
GPU-only cases return 77 when no usable GPU/backend is present; native CTest
registration marks that as skipped. A skip is not a successful CUDA validation.
The CPU metrics case still requires a usable CPU backend and does not skip for
missing CUDA.

As a separate negative case, set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` in the test
process and rerun the budget test. Require exit 1 and the explicit incompatibility
message at policy registration. This expected rejection is not a successful
positive budget-test run. Clear the variable before subsequent positive tests.

| Test | Required evidence |
| --- | --- |
| `test-cuda-memory-budget` | Exact quota accounting; preallocation denial; compatible/conflicting owners; final owner release; free after release; rejection with a live untracked allocation; concurrent reservations; strict zero-explicit-cap mode; exact real CUDA matmul; retained pool/workspace charging and cleanup; opt-in stream samples and disable behavior |
| `test-moe-ram-metrics` CPU and GPU | Exact bytes after an 8 MiB transfer; bounded 4 MiB upload staging and event-slot reuse on supported GPU; warm-cache zero extra reads; counters and timer opt-in; saturating deltas; concurrent snapshots |
| `test-moe-ram --cache` / `--cache-gpu` | Tied/strided expert selection, row/bias reads, eviction/reload, bounds and overflow rejection, oversized-batch fallback, repeated upload-buffer reuse |
| `test-gpu-layer-diagnostics` | CPU top-k views with element/row/sequence strides and nonzero offset; exclusion of unselected argsort suffixes; immediate snapshot lifetime; shared/layer endpoints; user callback ask/after forwarding and false-result preservation; bounded histogram, dropped counts and validation of dtype/IDs |
| `test-gpu-exclusive-scheduler` | CPU-only arithmetic rejection and normal-mode preservation; when CUDA is available, rejection of a heavy cast whose output view aliases a preallocated CPU destination (including explicit CUDA placement), and rejection of explicit Flash Attention with a CUDA-unsupported head dimension before any arithmetic submission |

The cache fixtures do not validate every model architecture. Full-model quality
validation additionally compares resident and streamed logits on a fixed small
MoE fixture, then representative large-model requests with identical model
weights and settings. A small exact matmul or a cache PASS is not a benchmark of
Qwen Flash throughput or proof that every MoE architecture is supported.

## Executed results

The final isolated suite ran on **2026-10-25, 03:02:18–03:02:21 UTC-03:00**, on Windows
x64 with an NVIDIA GeForce RTX 5060 Ti, driver **621.01**, CUDA 12.8 and the MSVC
19.39 build. The loaded CUDA backend reported compute capability **12.0**,
**VMM: yes**, and 16,283 MiB usable VRAM. GPU cases ran serially; no other test
agent performed GPU inference during this sequence.

| Executed case | Exit | Result |
| --- | --- | --- |
| CUDA quota, owners, concurrent allocation, forced physical pool/workspace lifetime and optional stream timing | 0 | PASS |
| Unified Memory enabled in a separate process | 1, expected | Explicit incompatibility rejection; PASS for this negative case |
| RAM metrics and bounded transfer, CPU | 0 | PASS |
| RAM metrics and bounded transfer, CUDA | 0 | PASS |
| Cache selection, eviction, bounds and reuse, CPU | 0 | PASS |
| Cache selection, eviction, bounds and reuse, CUDA | 0 | PASS |

No case was skipped. The CUDA quota test held two simultaneous 1 MiB buffers under
a 2 MiB quota and rejected additional allocations, then verified exact matmul
results and complete managed-byte cleanup after backend destruction. A separate
mixed-F32/F16 matmul forced a conversion through the VMM-capable pool: it reported
**655,360 tensor bytes**, **35,651,584 retained pool/workspace bytes (34 MiB)**
after the tensors were freed, and **0 managed bytes** after backend destruction.
Stream timing produced three completed samples with no pending/dropped samples after
synchronous graph execution; disabling it preserved the accumulated measurements.

The CUDA metrics fixture reported an **8 MiB resident/peak weight cache**, a
**4 MiB pinned upload buffer**, **32 MiB of cumulative uploads**, and exact byte
comparisons after repeated reuse. Its initial cold load ran with detailed timing
disabled deliberately. Timed warm transfers made no additional SSD reads. The
smaller cache stress test reported a **262,144-byte peak** under its 256 KiB quota
while exercising eviction, recovery and repeated upload staging.

Local artifacts are in `.build/gpu-first-moe/reports/`:

- `memory-validation-final-20261025.json` records exit codes, PASS-marker checks, hardware,
  wall durations and SHA-256 identities of the test executables and native runtime dependencies.
- `memory-validation-final-20261025-<case>.log` contains each complete native test log.
- The tested `ggml-cuda.dll` SHA-256 is
  `28BD9039CA4A50106CE03A0D86859018A7D945B7F3A593BC41BF2D24DE0B61ED`.
- The tested `llama.dll` SHA-256 is
  `5FBBAB94315C92CF42FC079EB1CAB4D00AA97B354C3CF44EDB47599AE8603D0F`.

Earlier `memory-validation` artifacts, including the `cohesive` and incomplete
fixture runs, are historical rather than the final suite result. The final cache
fixture explicitly distinguishes empty graph branches from manually constructed
nonempty operations, and verifies that empty routing does not request pages.

The CPU-only diagnostic helper regression separately passed both element strides
(4/8 bytes), padded row/sequence strides and offsets, snapshot lifetime, endpoint
attribution, external callback forwarding/false propagation, and full-histogram
drop accounting. Artifacts are `layer-diagnostics-helper-final-20261025.log` and
`.json`; the JSON records executable/native-library identities. This helper test
does not substitute for model-level diagnostics-off/on logit and exact-router
selection comparisons. The native server component suite also passed on the
final host build: **42 tests / 157 assertions / zero failures or skips**, recorded
in `server-components-final.log`.

An additional scheduler regression ran separately on 2026-10-25 with CUDA available
and exited 0. Both automatic and explicitly pinned CUDA placement rejected an
F16-to-F32 copy/cast into a CPU-backed output view at reservation and execution;
no heavy operations were submitted and destination sentinels remained intact.
Its log is `scheduler-destination-view-20261025.log` in the same reports directory.

These wall durations include backend/module initialization and assertions; they
are not model-throughput measurements. They establish this suite's Windows/CUDA
behavior, not Qwen Flash tokens/second, every MoE architecture, Linux behavior or
multidevice support. Those require separate model and platform validation.

## Isolated operator diagnostic

`test-gpu-exclusive-op` loads original expert bytes from the small GGUF fixtures
and calls public GGML `MUL_MAT_ID` directly on CUDA, without the weight cache,
scheduler copy callback or surrounding FFN. It compares eight original experts
with a compact two-slot bank mapping logical experts 1/6 to slots 1/0, including
one zeroed padding slot. GPU readback verifies all uploaded raw bytes. The tested
reduction/output dimensions were K=256/N=256, top-k=2, T=1/32, input lanes=1/2,
and output-row tiles 17/31/64/128/full. The optional scalar oracle dequantizes the
same GGUF rows and accumulates in double; its differences are reported separately.

The initial 2026-10-25 exploratory execution recorded 144 metrics in
`.build/gpu-first-moe/reports/op-isolated-20261025.json`, with the complete output
in `op-isolated-20261025.log`. No numerical acceptance tolerance was applied:
exit 0 means that the diagnostic completed and byte/integrity checks held.

| Original weight type | Full bank versus remapped full/64/128-row banks | Largest 17/31-row tile error | Largest full-T32 versus full-T1 first-token error |
| --- | --- | --- | --- |
| F32 | Bit-exact in all tested cases | 0.00123155 | 0 |
| Q4_K | Bit-exact in all tested cases | 2.07280 | 0.0115772 |
| Q6_K | Bit-exact in all tested cases | 1.27373 | 0.000292242 |
| MXFP4 | Bit-exact in all tested cases | 1.64502 | 0.180720 |

The large odd-row quantized discrepancies occurred with T=1 and two distinct
input lanes; with one shared input lane, or T=32, those quantized tile cases were
bit-exact. This isolates a shape-sensitive operator problem despite exact weight
fetching, distinct from the separately measured batch/kernel numerical variation.
It is not evidence that these large errors are acceptable or that end-to-end
model quality has passed. The diagnostic makes no changes to model weights,
context settings, runtime kernels or quality tolerances.

The operator program was then converted into a regression with explicit criteria:

- Q4_K/Q6_K/MXFP4 outputs must match the full expert operation **bitwise** for
  the same T/input lanes, including the 17/31-row shapes. Batch-size comparisons
  remain separate measurements and are not tested as identities.
- F32 full and tiled outputs are independently compared with the scalar double
  dot product. Bitwise equality is not required because the existing MMF path
  uses TF32 MMA while other shapes can select FP32 kernels. TF32 has a
  [10-bit fraction with FP32 exponent range](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html).
  For normal-or-zero operands the test uses an explicit format-error envelope,
  not a constant selected from observed discrepancies.

For each F32 dot, let `S = sum(abs(a_i*x_i))`, `m = 2*K`,
`u_t = 2^-10`, `u_f = 2^-23`, `d = 2*u_t + u_t*u_t`, and
`gamma = m*u_f/(1-m*u_f)`. The bound is
`[d + gamma*(1+d) + gamma_double]*S`, plus the measured double-to-F32 reference
rounding and an underflow term `m*float_min_normal`. `gamma_double` accounts for
the scalar accumulation. The operand terms cover TF32 truncation of both inputs;
the accumulation term allows FP32 multiply/add rounding in an unspecified order.
Every output must be finite and remain inside this envelope. A more accurate
FP32 result also satisfies it. This checks consistency with the existing
operator's precision formats; it does not widen any full-model logit tolerance
or establish end-to-end quality.

After the CUDA row-bound correction, the same fixtures and shape set ran again
and exited 0: **80 combinations, zero quantized bitwise failures, zero F32
format-bound violations**. All 60 Q4_K/Q6_K/MXFP4 combinations, including T=1,
two input lanes and 17/31-row tiles, became bit-exact against the full operator.
The earlier 2.07280 maximum Q4_K discrepancy became exactly zero.

The 24 F32 scalar checks had a maximum error/bound ratio of 0.113179. Across the
tested T/lane settings, 17/31-row tiles had at most 0.000551612 absolute error
against the scalar oracle; full/64/128-row shapes had at most 0.001019002.
For T=1 the 17/31-row scalar error was only about 1.79e-7. These measurements
demonstrate why F32 shape changes cannot be tested as bitwise identities; they
do not justify changing a model-quality threshold.

The operator suite was repeated on the final CUDA build with the same successful
criteria and results. Final artifacts are `op-regression-final-20261025.log` and
`op-regression-final-20261025.json` in the same reports directory; the JSON contains
164 numerical metrics, scalar checks, criteria and binary identities. The tested
CUDA DLL SHA-256 is
`28BD9039CA4A50106CE03A0D86859018A7D945B7F3A593BC41BF2D24DE0B61ED`.
The exploratory `op-isolated` artifacts are retained as the pre-correction
evidence. No full-model or Flash quality claim follows from this operator test.

## SSM convolution with ragged channel counts

`test-gpu-exclusive-ssm` is a GGML-only CUDA regression with a scalar double
oracle; it needs no model download or application configuration. It covers
127/128/129/192 channels, T=1/7/32, one/two sequences, convolution kernels
3/4/5/7/9/15, plain convolution, SiLU, and channel bias plus SiLU. Each case runs
with contiguous storage and again with nonzero-offset views, padded sequence
strides and padded weight rows. Temporal input rows retain the contiguous layout
required by the CUDA operator. Padding is NaN, and all input/weight bytes are
read back unchanged after execution. The 128-channel cases retain coverage of
the original optimized path alongside the new ragged-channel path.

Convolution accepts only finite outputs within
`[gamma_float(2K+1)+gamma_double(2K+1)] * sum(abs(products and bias))` of the
scalar sum. Here `gamma(n)=n*u/(1-n*u)` and `u` is the format's epsilon, which
also covers separate multiply/add rounding when FMA is not used. The fixtures
contain normal dyadic values, with no subnormal-input or overflow assumptions.
SiLU propagates that convolution bound through `z/(1+exp(-z))`, including the
[CUDA 12.8 documented intrinsic error limits](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#intrinsic-functions)
of `2+floor(abs(1.173*z))` ULP for `__expf` and 2 ULP for `__fdividef`, plus
addition and scalar-reference rounding. No threshold is fitted to measured
errors. The convolution/ADD/SiLU graphs allow the backend's existing fusion
path, without claiming that timing alone demonstrates fusion.

The 2026-10-25 execution on the RTX 5060 Ti passed **864 cases / 2,488,320
outputs / zero bound violations**, with a maximum error/bound ratio of
0.137934413. Maximum absolute scalar errors were 8.19563866e-8 for convolution,
6.36594490e-8 for SiLU and 6.96020864e-8 for bias plus SiLU. This validates these
operator shapes and views; complete Nemotron/model numerical validation remains
a separate test.

Artifacts are `.build/gpu-first-moe/reports/ssm-scalar-regression-20261025.log`
and `.json`; the latter records hardware, status and binary hashes. The tested
CUDA DLL SHA-256 is
`28BD9039CA4A50106CE03A0D86859018A7D945B7F3A593BC41BF2D24DE0B61ED`.
Run `test-gpu-exclusive-ssm` from the build's `bin` directory; exit 77 denotes
CUDA unavailable and is not positive validation. CTest marks this test serial.
