# GPU-first MoE inference (experimental)

This feature is opt-in on the `gpu-first-moe-inference` branch. It extends the
bounded expert RAM cache and the CUDA expert cache rather than replacing the
normal Ollama runtime. The native changes are shipped as
`llama/compat/004-gpu-first-moe.patch`, after patches 001–003, against the revision
in `LLAMA_CPP_VERSION`.

## Options

| Ollama API option | Server environment | Native runner flag | Meaning |
| --- | --- | --- | --- |
| `gpu_exclusive` | `OLLAMA_GPU_EXCLUSIVE` | `--gpu-exclusive` | Require CUDA for floating point inference operators. Unsupported placement fails instead of using CPU arithmetic. Default false. |
| `gpu_vram_limit_mib` | `OLLAMA_GPU_VRAM_LIMIT_MIB` | `--gpu-vram-limit-mib` | Limit tracked CUDA allocations of the runner on one physical device. MiB, default 0 (physical headroom check, no explicit byte ceiling in exclusive mode). |
| `gpu_diagnostics` | `OLLAMA_GPU_DIAGNOSTICS` | `--gpu-diagnostics` | Enable additional host I/O timers and a final JSON diagnostic summary. Default false. |

The native environment aliases `LLAMA_ARG_GPU_EXCLUSIVE`,
`LLAMA_ARG_GPU_VRAM_LIMIT_MIB` and `LLAMA_ARG_GPU_DIAGNOSTICS` are also accepted.
An explicit API value wins, including `false` and `0`; a per-runner environment
override wins over the server default. Effective settings participate in runner
reuse so a request does not silently inherit an incompatible already-loaded
runner. Native aliases are removed from the child environment after resolution.

Example for the isolated laboratory listener:

```json
{
  "model": "qwen3.8-flash-next-uncensored:q4_k_m",
  "messages": [{"role": "user", "content": "Responda apenas OK."}],
  "stream": true,
  "think": false,
  "options": {
    "num_ctx": 131072,
    "gpu_exclusive": true,
    "gpu_vram_limit_mib": 12288,
    "gpu_diagnostics": true,
    "moe_ram_limit_mib": 6144,
    "runner_env": {
      "LLAMA_ARG_CPU_MOE": "1",
      "LLAMA_ARG_MOE_CACHE_MIB": "1024"
    }
  }
}
```

`CPU_MOE` describes **weight storage** here. Selected quantized expert weights
are transferred to CUDA; it does not permit CPU matrix multiplication in
exclusive mode. Host-stored experts require a positive CUDA expert cache.
`num_gpu=0`, partial layer offload, CPU KV placement and incompatible non-CUDA
backends conflict with this mode. Layer placement becomes `all` and automatic
fit is disabled: the runtime must not reduce context to make a request fit.
The default warmup is skipped in exclusive mode to avoid reading unused experts
before the first request. Explicit warmup remains supported using real top-k
routing, without artificially activating every expert.

## Execution and storage

The existing RAM implementation reserves virtual addresses for GGUF expert
tensors, commits and locks bounded pages, reads missing spans with aligned
asynchronous I/O, and evicts unneeded pages using its existing SLRU policy.
The existing VRAM SLRU is reused for cache entries. These mechanisms, GGUF
compatibility translations and the prior runtime performance changes remain in
patches 001–003.

The new exclusive path changes execution in four places:

1. Before allocating graph copies, the GGML scheduler places floating point
   arithmetic on a CUDA backend that supports the operator. A second check
   immediately before submission rejects CPU arithmetic. Integer indices,
   views, layout copies, storage reads, orchestration and sampling may use CPU.
2. Contiguous 3-D expert matrices are cached by layout and output-row tile. A
   tile preserves the complete input reduction dimension and the original
   quantized bytes. CUDA computes the original routed experts; concatenation
   reconstructs output rows. There is no approximation or expert pruning.
3. Token slices bound the number of distinct routed experts needed at once.
   This avoids abandoning the cache when a prompt's union of experts is larger
   than the cache. Cache maps and reusable slots are synchronized before reuse.
4. Host token and per-layer/hash embeddings are gathered as exact quantized rows and dequantized
   by CUDA `GET_ROWS`, avoiding a full embedding-table graph copy. SSD-to-RAM
   cache reads and host-to-device transfers reuse the bounded RAM machinery.

Host-to-device uploads now use two event-protected pinned slices of 2 MiB in
the same total 4 MiB staging allocation. A slice is reused only after its prior
transfer completes, and final synchronization precedes unlocking source pages.
This pipelines uploads; it does not claim that all SSD, PCIe and CUDA work can
overlap. Reading the router's actual selected indices still has a dependency
on CUDA. Future expert choices are not known in advance.

Shared/dense weights and KV buffers currently remain resident on CUDA. If
these fixed allocations and the minimum compute working set cannot fit, the
request fails. This implementation does not stream arbitrary dense matrices or
offer unrestricted multi-GPU execution. Graph reservation can retry with a
smaller expert cache; it preserves model weights, quantization and context.

The server and multimodal command-line entrypoints also propagate the exclusive
policy to the separate projector scheduler. A projector requires CUDA and rejects
unsupported graph arithmetic rather than silently offloading it to CPU. CPU
media decoding and preprocessing remain outside that neural-network scheduler.
Full image/audio numerical validation has not been executed in this laboratory;
unsupported projector graphs can fail during reservation, including an unsupported
fused-attention probe. The operator counters in the request summary describe the
text context, not a combined total for every projector context.

## Memory contract

The RAM setting limits managed host-weight pages, including selected expert and
embedding-table rows, not total process RSS. Metadata,
tokenizer, pinned I/O buffers, application memory and the operating system's
file cache are separate. The CUDA quota counts the allocator's real live
allocations, including retained pools, VMM allocations and explicit cuBLAS
workspaces. Returning a logical tensor to a pool does not free its charge.
Multiple models sharing a device must use compatible quota settings. Enabling a
policy over an existing untracked legacy allocation fails instead of guessing.

The quota is not an operating-system-wide VRAM ceiling. Other processes and
implicit driver/library allocations are outside its accounting. The allocator
also leaves 64 MiB of physical free-device headroom and rejects managed/unified
memory in quota mode. This cannot override WDDM's own residency or paging
decisions, nor does it promise a hard cap on all system RAM or the absence of
all operating-system paging. See [memory validation](gpu-first-memory-validation.md).

## Diagnostics

With diagnostics enabled, native final timings contain `gpu_first`; Ollama
forwards it as the optional top-level `gpu_first` response field for generate
and chat, and logs `gpu_first: request_summary` in the native server. It reports
RAM and VRAM cache counters, SSD bytes/reads, H2D bytes/calls/waits, host I/O
intervals, tiles and submitted CPU/CUDA floating point operator counts. CUDA
memory figures describe managed allocations and include lifetime peaks.

Request summaries are **deltas of shared model/context counters**. Concurrent
requests can contribute to those deltas. RAM peaks are model-lifetime peaks;
CUDA peaks and allocation-denial counts have physical-device allocator lifetime.
Timing categories overlap and must not be added together. H2D wall time is not
CUDA compute time. Optional CUDA events measure execution-stream intervals,
including internal copies/waits and launch gaps, rather than pure kernel time.
The bounded event ring reports completed, pending and dropped samples; it drops
measurements instead of blocking inference when full. Operator counts are
submitted operations, not kernel counts, and are enabled only in exclusive
mode. Zero counters in normal mode do not prove an absence of CPU arithmetic.
The execution barrier and numerical regression tests provide additional evidence
that a successful exclusive run did not use CPU arithmetic.

Detailed diagnostics also report `layer_profile` and `expert_usage`. Layer
profiling deliberately synchronizes at tagged graph and router boundaries. Its
wall intervals include storage, transfers, computation and diagnostic barriers;
they are not pure CUDA kernel timings. Untagged/shared work has layer index -1.
Attribution follows graph tags and execution order; some architectures can have
multiple segments for a layer. Use diagnostics-off runs to compare performance.
FFN router popularity observes actual top-k IDs once per graph node, including
strided views, rather than counting every up/gate/down matrix or output tile.
The final summary contains the 32 most selected layer/expert pairs and total
selections. Histogram storage is bounded to 262144 distinct pairs; additional
untracked selections are explicitly reported as dropped.
Separate attention-expert routers are not included in the FFN histogram.

Full per-layer CUDA event timings and a guaranteed total-process RAM cap are
not implemented. The benchmark records client-observed TTFT, load, prefill,
generation, and memory observations separately. Native prompt timing alone is
not client TTFT.

## Validation and benchmarks

Prepare the llama.cpp revision from `LLAMA_CPP_VERSION` in
`.build/gpu-first-moe/native`, then apply the ordered compatibility patches:

```powershell
$wildLlamaSource = (Get-Location).Path
Push-Location .build/gpu-first-moe/native
cmake "-DPATCH_DIR=$wildLlamaSource/llama/compat" -P "$wildLlamaSource/cmake/apply-git-patches.cmake"
Pop-Location
./scripts/gpu-first/build-windows.ps1 -Build
```

The Windows launcher reuses installed Visual Studio, Ninja, CMake and CUDA;
it does not install tools or change user environment settings. Its temporary
files and CUDA cache stay under the chosen build root. `-NativeSource` and
`-BuildRoot` can select other isolated paths. The normal repository CMake build
also applies patch 004 automatically; the test wrapper uses a prepared source
tree and performs no downloads. The patch applier verifies overlapping applied
prefixes on private file copies, so configuring twice remains safe.

The native wrapper does not compile the Go server. On this existing Windows
development machine, the Go step can use the already-cached toolchain without
placing build caches or temporary files on C:. These are process-local settings:

```powershell
$wildLlamaSource = (Get-Location).Path
$wildLlamaTemp = Join-Path $wildLlamaSource '.build/gpu-first-moe/temp'
New-Item -ItemType Directory -Force -Path $wildLlamaTemp | Out-Null
$env:GOCACHE = 'G:\Goose\cache\go-build'
$env:GOMODCACHE = 'G:\Goose\cache\go-mod'
$env:GOPATH = 'G:\Goose\cache\go-path'
$env:GOTOOLCHAIN = 'auto'
$env:GOTMPDIR = $wildLlamaTemp
$env:TEMP = $wildLlamaTemp
$env:TMP = $wildLlamaTemp
$env:CC = 'G:\Goose\cache\toolchains\w64devkit\bin\gcc.exe'
$env:CXX = 'G:\Goose\cache\toolchains\w64devkit\bin\g++.exe'
$env:PATH = "G:\Goose\cache\toolchains\w64devkit\bin;$env:PATH"
Get-ChildItem Env: | Where-Object Name -Match '^(OLLAMA_|LLAMA_)' | Remove-Item
go test ./api ./envconfig ./llm ./server ./cmd ./discover
if ($LASTEXITCODE -ne 0) { throw 'Go tests failed' }
go build -o .build/gpu-first-moe/payload-candidate/ollama.exe .
if ($LASTEXITCODE -ne 0) { throw 'Go build failed' }
```

Existing MSVC/CUDA/Python installations are reused; these commands do not install
them. Run from the repository on G: and keep the laboratory listener on 11448.
Installed production listeners and the Wildgoose shortcut are not changed by
this workflow.

Use the isolated build and test workflow described in the memory validation
document. Tests compare resident CUDA logits against bounded streaming using
the same GGUF and inputs, including prefill and consecutive decode steps.
Scheduler tests reject CPU arithmetic and budget tests exercise actual CUDA
allocation, reuse, concurrency and matmul. Tests must run before using the
experimental payload outside the laboratory.

`scripts/benchmark-gpu-first.py` launches a separate listener (default 11448),
uses a separate model-manifest store and home, and stops only its own server
tree. The laboratory reuses model bytes with same-volume hard links; preparing
it never copies the 100+ GB weights. A free-RAM guard aborts the test if available
RAM falls below the requested safety threshold. It does not flush the OS cache,
so “runner-cold” means a new runner, not a guaranteed cold SSD.

Initial baseline on this machine used the frozen installed WildLlama payload,
context 131072, 6144 MiB expert RAM cache and 1024 MiB CUDA expert cache. Actual
hardware was **Ryzen 5 2600, 32 GiB RAM, RTX 5060 Ti 16311 MiB**, different from
the project's reference CPU. Windows identifies the model volume F: as a
**Lexar NS100 1 TB SATA SSD**; the observed GPU link is **PCIe generation 3 ×8**.
The workspace G: is a separate SATA SSD. Two-token responses are useful for TTFT/cache
checks, not sustained throughput comparisons.

| Baseline case | Client total | Client TTFT | Load | Prefill | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Runner-cold | 78.375 s | 75.203 s | 32.501 s | 42.695 s | 2 |
| Warm runner | 1.578 s | 1.031 s | 0.005 s | see raw report | 2 |

The two-token baseline minimum observed available RAM was 12.24 GiB. Its expert
cache reached 6144 MiB. Cumulative SSD reads were about 18.325 GiB, with no
additional reads on the warm request.

### Real Qwen Flash comparison

The real model is `qwen3.8-flash-next-uncensored:q4_k_m`, GGUF family `qwen4exp`,
176.9B parameters, manifest digest
`bee26cfee476e3762c23027d1d23b92acdf125eafb82ba36d1fcb11c05278242`.
The existing payload and candidate used the same weights, 131072 context,
6144 MiB host weight cache, 1024 MiB CUDA expert cache, q8_0 KV, one sequence,
512 batch and 32 microbatch. The baseline used its existing 4096 MiB fit margin;
the candidate used exclusive mode, fit disabled and a 12288 MiB tracked CUDA
quota. Neither reduced the requested context or changed model quantization.

Prompt: `Conte de 1 até 100, separando os números por vírgulas.` Temperature 0,
seed 42, thinking disabled, 32 output tokens. One runner-cold request followed
by one warm request, diagnostics off:

| Metric | Existing cold | Candidate cold | Existing warm | Candidate warm |
| --- | ---: | ---: | ---: | ---: |
| Client TTFT | 94.656 s | 79.828 s | 9.828 s | 8.953 s |
| Client total | 148.875 s | 131.203 s | 57.875 s | 54.578 s |
| API load | 27.663 s | 18.979 s | 0.005 s | 0.007 s |
| API prefill | 66.990 s | 60.842 s | 9.801 s | 8.946 s |
| API generation | 54.217 s | 51.384 s | 48.052 s | 45.620 s |
| API tokens/s | 0.5902 | 0.6228 | 0.6659 | 0.7014 |

API throughput is `eval_count / eval_duration`. Counting only 31 decode
intervals gives 0.5718 → 0.6033 cold and 0.6451 → 0.6795 warm. The observed
total-time reductions were 11.87% cold and 5.70% warm, with throughput gains
of 5.51% and 5.33%. All four responses produced byte-identical text, truncated
by the explicit 32-token test limit. This is a short equivalence check, not a
general quality/perplexity evaluation.

The comparison is **one pair of runs**, without flushing OS caches or controlling
other applications. The baseline started with less available system RAM and more
global VRAM already occupied. These numbers are observations, not a statistically
established speedup or a promise for other machines. Reports preserve exact
payload hashes. Later diagnostic-only additions and the generic SSM convolution
kernel were not retroactively benchmarked by these runs.

Sampled process-tree working-set peaks were 6.610 GiB existing and 6.615 GiB
candidate. Shared pages can be counted in multiple processes. Private commit
peaks were 15.998 and 14.680 GiB; commit is not physical RAM residency. Whole-GPU
NVML peaks were 11.062 and 8.797 GiB, including the display, driver and other
processes. These one-second samples can miss short peaks and do not replace
the allocator ledger. Minimum whole-machine available RAM was 5.533 and
12.765 GiB, illustrating the differing system conditions.

A separate two-request diagnostic run recorded zero submitted heavy CPU
operators, 11814/7868 submitted CUDA heavy operators, a 6144 MiB peak locked
weight cache and approximately 6.25 GiB peak tracked CUDA allocations, below
the 12 GiB quota. Cold SSD traffic was 11.879 GB; warm SSD traffic was zero.
Warm host-to-device weight traffic was still 4.750 GB, with no VRAM-cache hits
and 4350 misses/evictions. CUDA stream intervals were 1.333/0.188 s, rather
than the much larger total latency. The current small VRAM cache still suffers
cyclic replacement; moving arithmetic to CUDA alone does not remove that cost.
The diagnostic categories overlap and cannot be summed into a latency budget.

The first real-model failures found and corrected during development were:

- A 33569.50 MiB per-layer/hash embedding table was being copied whole into the
  graph. Exact selected-row gathering reduced graph reservation to 278.76 MiB
  for this case (the existing payload reserved 1711.48 MiB).
- Intermediate prefill batches with no output logits could concatenate an empty
  MoE tile list. The zero-token path now returns a correctly shaped empty result.
- A CUDA MMVQ tail could write beyond an odd-sized output-row tile. The actual
  row bound now guards writes and reads without changing the aligned fast path.

### Numerical and backend validation

Small-model checks cover prefix batches with no logits, prefill, consecutive
decode, contexts 256/512 and deliberately undersized expert caches. See
[fixture provenance and results](../scripts/gpu-first/fixtures.md) and
[allocation/operator evidence](gpu-first-memory-validation.md).

The original 19-case streaming matrix has eight strict passes, nine finite
numerical failures at its predeclared 2e-4 threshold, and two explicit unsupported
Nemotron state-16 scan cases. All 17 cases producing logits retained top-1 in
9/9 checked steps and submitted zero heavy CPU operations. Resident CUDA itself
also exceeds that threshold when changing microbatch dispatch in seven
corresponding tests. Those failures remain visible; they are not relabeled as
passes or proof of equivalent quality. A separate Nemotron state-128 synthetic
fixture passes with 64 KiB and 1 MiB caches; the original state-16 input remains
an unsupported negative test.

The isolated tiled expert operator suite passes 80 combinations; 60 quantized
cases are bitwise equal to resident execution. F32 dispatch differences are
checked against a double scalar oracle using bounds derived from TF32/FP32
rounding. The SSM convolution fallback passes 864 combinations and 2488320
outputs against its scalar oracle. It only handles channel counts outside the
existing aligned CUDA fast path; it does not add missing SSM scan shapes.

The detailed-diagnostic helper passes CPU tests for strided IDs, padding,
immediate tensor lifetime, shared/layer endpoints, callback forwarding/abort
and bounded histogram overflow. CUDA integration with 1 MiB and 32 KiB expert
caches counted exactly 256 logical selections (top-k 2 × 32 tokens × 4 layers),
without multiplying by tile count. All 8288 logits were bitwise equal with
diagnostics off/on when using the same externally requested node boundaries;
external callbacks were preserved. This controls segmentation in that test,
not a promise that extra diagnostic barriers never alter CUDA dispatch.
The final memory/cache/budget suite passes all six cases without skips, and
server-component tests pass 42 tests / 157 assertions. The Go API, options,
runner, server, CLI and device-discovery suites pass on the final Go source.

The final payload was also exercised with the real Flash after these additions:
both two-token requests returned `OK`, `/api/ps` confirmed context 131072,
and both summaries reported zero heavy CPU operators. There were 49 layer
records (48 layers plus shared work), 8950/2370 canonical FFN expert selections,
no dropped histogram entries and no pending/dropped CUDA event samples. The
managed RAM peak was exactly 6 GiB and the tracked CUDA peak approximately
6.25 GiB, with no denied allocations under the 12 GiB quota. The warm request
read zero SSD bytes but uploaded 4754488912 weight bytes. The lab held the model
for 45 seconds for observation, then stopped its process tree and released port
11448. This functional smoke is distinct from the 32-token performance run.

Windows CUDA execution was tested locally. Linux host-cache support is retained
in the source, but Linux, HIP/MUSA, multi-GPU, long-duration stress and full
multimodal numerical validation were not executed here. This branch remains
experimental, particularly for architectures outside the passing fixtures.

Raw JSON includes request bodies, final API responses, binary/library hashes,
hardware inventory and sampled resources. It is retained under
`.build/gpu-first-moe/reports`, including `benchmark-count32-comparison.json`,
the two `*-count32-a-results.json` files and the separate diagnostic report.
A compact extract with hashes, scopes, retained failures and final-smoke results
is versioned in [the benchmark evidence](benchmarks/gpu-first-20261025.json),
so those observations survive cleaning the ignored build directory.

## Further work justified by the measurements

The existing SLRU and four aligned asynchronous storage reads are retained.
No speculative future-layer routing is assumed. Pipelined uploads improved
this measured configuration, but automatic cache admission has not eliminated
scan thrashing with only a 1 GiB expert cache. Next steps are to replay bounded
layer/expert traces at equal memory budgets and compare SLRU with per-layer
protected hot sets plus transient demand slots. Admission must never evict
confirmed active experts or prefetch beyond its budget. Smaller tiles trade
retention capacity against more launches and transfers, so they need measurement.

Cold-start commit/lock already groups up to 1 MiB of 64 KiB pages. Grouping
multiple reads into larger contiguous commit extents may reduce Windows call
overhead, while keeping page-level quota reservation and rollback. It cannot
eliminate the cost of making memory resident. Further overlap should use actual
router dependencies and preserve transfer completion before bank/page reuse.
Dense-weight streaming, adaptive repartitioning between fixed KV and expert
memory, per-layer CUDA events and broader quality/long-run tests remain separate
engineering work; no claim of maximum possible performance is made.
