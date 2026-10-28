# Optional resident CUDA kernel calibration

WildLlama can consume an explicitly calibrated policy for the existing MMVQ and
MMQ dense matrix kernels. This feature is **off by default**. Loading a model
never starts calibration, and a missing, incompatible or rejected policy keeps
the original kernel heuristics.

This is a narrow optimization for fully resident matrix operations, not a
replacement for memory admission or the MoE expert cache. It does not change
weights, quantization, context length, sampling or model parameters.

GPU discovery can report compute capability, SM and warp counts, shared-memory
limits, memory-bus properties and virtual-memory support. The NVIDIA kernel
driver version is distinct from the CUDA compatibility version. These optional
capabilities prepare selection for other cards without treating a marketing
name or missing attribute as evidence of support. A detector does not make a
binary compiled for one CUDA architecture compatible with every other GPU.

## Eligibility and fallback

The initial implementation supports CUDA only, one physical/logical GPU in the
normal server launch, and contiguous two-dimensional quantized weights with F32
activations/output. Dimensions are bounded: input width is a multiple of 256
between 256 and 16384, output width is a multiple of 128 between 128 and 16384,
and the token batch is 1–8. Supported weight types are Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0 and Q2_K through Q6_K.

Views, split buffers, MoE routing, nondefault precision, fused operations and
explicit forced-kernel builds retain their original paths. Q8 inputs with a
shared activation-quantization plan also retain MMVQ: isolated matrix timings
must not discard savings shared by several graph consumers.

A policy can select an alternative only when the inference pool already has
enough mapped free workspace. It never grows the pool for that choice. Unknown
or legacy pool capacity causes a normal fallback. Once a CUDA graph exists,
the backend rejects policy reconfiguration; reload the backend to change it.

Other GPU vendors/backends continue using their existing dispatch. The generic
device descriptor can carry partial capabilities without treating missing
attributes as support. CUDA UUID is kept internal; missing UUID falls back to
PCI identity. A missing full NVIDIA driver version prevents policy reuse.

## Configure the trusted local policy directory

Set `WILDLLAMA_CUDA_TUNING_DIR` in the **server's** environment to an absolute
directory controlled by the local administrator. It is not a model parameter
and cannot be supplied through a model's `runner_env` or an API request. Keep
this directory and any CUDA JIT cache on the intended data drive.

The Go launcher derives a SHA-256 fingerprint from the physical GPU identity,
capabilities, full kernel-driver version and exact installed native payload.
Payload hashing covers the runner and runtime libraries, with process-local
memoization and file replacement checks. A driver or payload change requires a
new policy. When debug logging is enabled and the policy is missing, the log
includes the expected fingerprint and filename:

```text
<WILDLLAMA_CUDA_TUNING_DIR>/<fingerprint>.policy
```

Only a regular file of at most 64 KiB is passed to the native validator. The
launcher clears inherited `GGML_CUDA_TUNING_CACHE` and
`GGML_CUDA_TUNING_FINGERPRINT` values first, then supplies its validated path
and current fingerprint. The native backend independently validates the file
contents. No cache parsing or filesystem access happens in matrix dispatch.

## Build and run the explicit calibration tool

The validation CMake project provides `wildllama-cuda-calibrate` and the CPU-only
`test-cuda-tuning-policy` target. For an already prepared Windows checkout:

```powershell
.\scripts\gpu-first\build-windows.ps1 -Build -Targets @(
  'wildllama-cuda-calibrate', 'test-cuda-tuning-policy'
)
```

Use the exact CUDA module and fingerprint of the installation that will consume
the result. Do not calibrate an unrelated build and reuse its identity. The
output filename must be new; the tool never overwrites an existing cache.

```text
wildllama-cuda-calibrate
  --backend <absolute ggml-cuda.dll or .so path>
  --fingerprint <64 lowercase hexadecimal characters>
  --output <absolute directory>/<fingerprint>.policy
  --type q8_0 --k 2048 --m 2048 --n 1
  --pairs 5 --time-ms 1500 --max-mib 64 --scratch-mib 8 --device 0
```

Run this deliberately while the GPU is otherwise idle. The tool loads no model;
it generates three deterministic finite fixtures and invokes the actual native
kernels. `--no-warmup` returns a fallback **before loading CUDA**. Ctrl+C requests
cancellation between launches. Time limits prevent submission of more work;
safe cleanup drains work already submitted, so they cannot preempt a running
kernel.

`--max-mib` bounds owned host/device fixture and scratch bytes, excluding
driver/library/context overhead. `--scratch-mib` bounds a fixed quota-accounted
GPU arena plus host output copies. The API does not retain calibration storage
in the inference context. No unified-memory or SSD substitute is introduced.

Every measured pair must produce finite, **bitwise identical** outputs and the
same alternative must beat the original choice by at least 5% in every pair,
across all three fixtures. Fallback is a valid result: no policy is published
when equivalence, repeatability, cancellation or budgets fail. The tool prints
one JSON result with the reason, selected kernels, timing and workspace scope.

These checks establish agreement on the tested fixtures, not a mathematical
proof for all future inputs or an end-to-end tokens/second improvement. No
tolerance is silently relaxed to obtain a winner. Compare a small real workload
before using a calibrated policy for normal work.

## Native interface and file format

`ggml/include/ggml-cuda-tuning.h` defines ABI 1, obtained through
`ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_tuning_api")`.
The three functions are `configure`, `get_stats` and `calibrate`. Configure and
calibrate require a quiescent backend; calibration is rejected during CUDA
capture or after a compiled graph exists. Explicit callers can configure each
backend separately; the simpler server environment contract is single-GPU only.

The bounded text cache contains at most 128 unique shape records:

```text
wildllama-cuda-policy 1 <fingerprint> bitwise-finite
<type> <k> <m> <n> <original> <selected> <pairs> <minimum_speedup> <workspace_bytes>
```

`type` is the native `ggml_type` integer. Kernel IDs are 1 for MMVQ and 2 for
MMQ; an unchanged or default selection is not an approved record. The parser
rejects wrong identities/versions, malformed or duplicate records, nonfinite
timings, insufficient pairs/gain and out-of-range shapes/workspace. Partial
files fail closed. Publication uses a flushed private temporary file and an
atomic no-replacement hard link; unsupported filesystems report an error.

Dispatch performs an in-memory bounded lookup only. Hit/miss statistics count
direct evaluations/captures, not each later CUDA graph replay. They are not
inference timing or model-quality measurements.
