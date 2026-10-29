<p align="center">
  <img src="docs/assets/wildllama-mark.svg" alt="WildLlama" width="176" />
</p>

<h1 align="center">WildLlama</h1>

<p align="center"><strong>Local inference with tighter memory control and smarter MoE streaming.</strong></p>

<p align="center">
  <a href="#what-wildllama-adds">Features</a> ·
  <a href="#getting-started">Get started</a> ·
  <a href="#memory-controls">Memory controls</a> ·
  <a href="#measured-improvements">Results</a> ·
  <a href="https://github.com/Anderseta/wildllama/issues">Issues</a>
</p>

WildLlama is an independent fork of [Ollama](https://github.com/ollama/ollama) focused on making local inference more efficient on limited hardware. Its main work is in **bounded model-weight RAM, controlled CUDA allocations, and selective SSD → RAM → GPU streaming for large Mixture-of-Experts models**. It also improves dense-model placement, chat preparation, state transfers, and runtime visibility.

The goal is to spend less time moving the same weights, repeating host work, or computing drafts that will be discarded. WildLlama keeps Ollama's familiar CLI and HTTP API while adding controls and diagnostics for these paths. The executable, `OLLAMA_*` settings, and API names currently retain their upstream names for compatibility.

## What WildLlama adds

| Focus | WildLlama's additions to the inherited Ollama runtime |
| --- | --- |
| **Model-weight RAM control** | A bounded, locked host-weight cache for supported raw GGUF MoE models. Load selected experts and embedding rows on demand instead of making the complete host weight set resident. |
| **More efficient MoE streaming** | Coordinate RAM and GPU residency, batch nearby SSD reads, merge adjacent uploads, reuse route plans, and use event-protected upload slots. Eligible prompt batches group work by expert to avoid repeatedly transferring the same weights. |
| **Explicit GPU policy** | Optional GPU-exclusive execution and a quota for tracked native CUDA allocations. Unsupported placement fails explicitly rather than quietly moving floating-point inference to the CPU. |
| **Dense-model efficiency** | More accurate hybrid/recurrent KV placement estimates, corrected dense embedding placement in GPU-exclusive mode, optional physical KV growth with unchanged logical context capacity, and measured native MTP with compatible prediction heads. Experimental CUDA dense paths remain separately opt-in. |
| **Less runtime overhead** | A single synchronization for speculative verification batches, reuse of known CUDA graphs without rescanning their nodes, bounded schema/grammar caching, batched state transfers, and safer snapshots. Neural drafting avoids repeated history copies; streamed Qwen tool calls avoid repeated full-buffer scans. |
| **Visible progress** | An additive `/api/activity` endpoint reports sampled loading, context preparation, prefill, generation, memory-cache counters, and weight-transfer activity. WildGoose can present these observations in its interface. |

These changes target unnecessary work and transfers. They do not prune routed experts, requantize weights, or automatically shrink the requested context to make a model fit. Eligible optimized paths retain the original calculation or fall back when their supported layout or kernel conditions are not met. Numerical regression tests accompany the native changes.

Coordinated MoE execution, GPU-exclusive execution, physical KV growth, and the experimental CUDA dense kernels are **opt-in**. Ordinary runners retain their existing default behavior. See [runtime efficiency](docs/runtime-efficiency.md), [dense-model improvements](docs/dense-runtime.md), and [GPU-first MoE inference](docs/gpu-first-moe-inference.md) for implementation details and activation rules.

## Large MoE models on smaller machines

A MoE model uses only its router-selected experts for each token, but the full set of expert weights can still be much larger than VRAM or available RAM. WildLlama's selective-weight path keeps bounded working sets in memory and reads missing weights from the original GGUF files on SSD.

The GPU-first path tiles expert matrices and prompt windows so the selected work can execute on CUDA within its cache and workspace. The coordinated runtime then tries to keep useful experts resident, avoid duplicate reads and uploads, and pipeline bounded transfers. Prompt-wide expert reuse can help when a cache is too small to hold the union of experts selected across the prompt.

**MoE does not eliminate weight traffic.** A cold expert still has to reach the device. SSD bandwidth, PCIe transfers, model architecture, routing, quantization, cache size, and the available GPU working set determine practical speed. Smaller budgets can increase misses and latency; a model whose fixed weights and minimum working set cannot fit may still fail. WildLlama does not make SSD storage equivalent to VRAM bandwidth.

The optimized paths work with supported original GGUF weights; they do not require a proprietary model conversion or an additional on-disk expert index. Architecture-independent graph hooks do not establish support for every architecture, quantization, adapter, or multimodal projector.

## Getting started

Build **this repository** to get WildLlama. The upstream Ollama installers and `ollama/ollama` container image install Ollama, not the modifications in this fork.

This guide covers source builds and CLI/native runtime use. It does not provide or claim isolation for a packaged desktop installer, desktop auto-start, or updater.

Prerequisites include Go, CMake 3.24+, and a C/C++ toolchain. CUDA builds additionally require the appropriate CUDA toolkit. See the [development guide](docs/development.md) for platform requirements and backend selection.

```sh
git clone https://github.com/Anderseta/wildllama.git
cd wildllama
cmake -B build .
cmake --build build --parallel 8
```

The default build is CPU-only except for Metal on macOS arm64. Select a CUDA backend explicitly for the CUDA-specific features, for example:

```sh
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel 8
```

Use the locally built executable: `./ollama` on Linux/macOS or `.\ollama.exe` on Windows. A Go-only rebuild can reuse an existing compatible native payload, but **the new native features require the patched native runtime**; replacing only the Go executable is insufficient.

### Run alongside an existing Ollama installation

Choose a separate listener and model directory to keep the services independent. For example, from the built checkout:

**PowerShell**

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11446"
$env:OLLAMA_MODELS = Join-Path (Get-Location) "models"
.\ollama.exe serve
```

**Linux/macOS**

```sh
OLLAMA_HOST=127.0.0.1:11446 OLLAMA_MODELS="$PWD/models" ./ollama serve
```

Point your client at `http://127.0.0.1:11446`. For CLI commands in another terminal, set the same `OLLAMA_HOST` there. A separate model directory does not automatically migrate models; import or download them into the chosen store. Account files, logs, temporary directories, and build caches have their own locations—`OLLAMA_MODELS` controls the model store only.

### API compatibility

The standard `/api/chat`, `/api/generate`, model-management endpoints, and inherited OpenAI-compatible routes remain available. Existing Ollama clients can use WildLlama by selecting its base URL. Extra controls are optional; clients can inspect `/api/version` for advertised capabilities before using them.

An ordinary chat request needs no WildLlama-specific options:

```sh
curl http://127.0.0.1:11446/api/chat -d '{
  "model": "your-model",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

See the inherited [API documentation](https://docs.ollama.com/api) for standard endpoints and the local guides below for WildLlama extensions. Feature support still depends on the selected runner and model.

## Memory controls

| Setting | What it controls | What it does not control |
| --- | --- | --- |
| `moe_ram_limit_mib` | Resident pages of the managed host-weight cache for supported MoE models. `10240` means 10 GiB; `0` disables it. | Total process RAM, KV cache, activations, metadata, staging buffers, other applications, or system-wide swapping. |
| `gpu_vram_limit_mib` | Tracked native CUDA allocations on one physical device, including retained allocator pools. | Other processes, display/driver allocations, and every implicit library allocation. |
| `LLAMA_ARG_MOE_CACHE_MIB` | The GPU cache used for host-stored expert weights. | The full GPU footprint; fixed weights, KV, and workspaces still need room. |
| `kv_cache_initial_tokens` | Optional initial physical KV commitment on supported CUDA VMM layouts. Logical context capacity stays unchanged. | A maximum KV budget or permission to exceed the eventual allocation quota. Growing the context still needs memory. |

`fit-target`/fit margin is placement guidance, not a hard VRAM allocation ceiling. The managed RAM and CUDA budgets are separate scopes; neither is a total machine-memory limiter. Locked managed weight pages avoid pageable storage for that cache, but **WildLlama cannot guarantee that the operating system never pages any other allocation**.

A server-wide MoE weight budget can be set with:

```sh
./ollama serve --moe-ram-limit-mib 10240
```

The corresponding server environment variable is `OLLAMA_MOE_RAM_LIMIT_MIB`. The API option or a model's `PARAMETER moe_ram_limit_mib` takes precedence, including explicit zero. Server defaults apply to MoE models; an incompatible explicit request fails rather than silently ignoring the budget.

### Opt into coordinated CUDA MoE execution

The following illustrates the extension fields for an eligible MoE model. **These budgets are examples, not a universal model profile.** Keep enough memory for the model's fixed tensors, KV, compute workspace, and other applications.

```json
{
  "model": "your-moe-model",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "options": {
    "moe_ram_limit_mib": 10240,
    "moe_runtime": true,
    "gpu_exclusive": true,
    "gpu_vram_limit_mib": 12288,
    "runner_env": {
      "LLAMA_ARG_CPU_MOE": "1",
      "LLAMA_ARG_MOE_CACHE_MIB": "1024",
      "WILDLLAMA_MOE_DYNAMIC_PREFILL": "1"
    }
  }
}
```

Here `CPU_MOE` describes **weight storage**. In GPU-exclusive mode the selected expert arithmetic still runs on CUDA. This mode requires compatible full GPU-layer placement and a positive GPU expert cache for host-stored experts. It does not silently fall back to CPU arithmetic or reduce the requested context when an allocation cannot fit.

`moe_runtime` enables the coordinated runtime; dynamic prefill has its separate opt-in switch. Expert grouping and packed row uploads require the appropriate coordinated path. Optional frequency-aware RAM admission stays off by default because it can help stable hot sets but hurt changing ones.

Read the [RAM-budget contract](docs/moe-ram-limit.md), [GPU execution and quota contract](docs/gpu-first-moe-inference.md), [memory validation](docs/gpu-first-memory-validation.md), and [coordinated runtime guide](docs/moe-residency-pipeline.md) before tuning constrained workloads.

## Measured improvements

Small fixtures establish specific improvements without requiring a large-model download:

| Component or small-model case | Recorded result |
| --- | --- |
| Qwen3MoE under fixed GPU-cache pressure | H2D traffic: **9,526,896 → 3,733,104 bytes**, approximately 61% less. |
| Gemma4 under fixed GPU-cache pressure | H2D traffic: **6,475,344 → 2,788,944 bytes**, approximately 57% less. |
| Packed Q5_0 embedding rows | **8,192 → 1 CUDA upload calls**, with identical transferred payload bytes and row order. |
| Nearby GGUF tensor regions | **4 → 1 SSD reads**, within the same managed RAM peak and existing staging allocation. |
| Capped MTP/EAGLE draft request | **48 → 1 actual draft output rows** for a one-token request; unused drafts are no longer computed first. |

These are **component and synthetic-model observations, not equivalent tokens-per-second gains**. Cache pressure, warm state, model layout, and shared GPU load matter. The expert-grouping result is chiefly a prompt-processing optimization; it does not predict single-token decoding speed for a large Flash model.

The [expert-reuse validation artifact](docs/benchmarks/moe-expert-reuse-20261027.json) records 23 passing CTest components, 24 CUDA synthetic-model comparisons, and 44 server cases. Approved numerical comparisons retain the existing scaled-logit tolerance of `0.0002` and agree on top-1 at all nine checked output positions. This checks numerical regressions, not real-model language quality or universal model compatibility.

The record also preserves limitations: expert grouping is disabled for F32 weights after a numerical regression, and a forced 32 KiB MXFP4-cache diagnostic exceeds its resident-reference tolerance equally with grouping enabled and disabled. It is an unresolved baseline kernel-precision difference, not counted as a passing comparison. See [runtime efficiency](docs/runtime-efficiency.md) and the [batching artifact](docs/benchmarks/moe-batching-20261027.json) for controls, scopes, and additional results.

A separate trained-model comparison measured **native MTP depth two** on Qwen3.5 9B DARK ROAST Q8_0 with an RTX 5060 Ti 16 GiB. Each result is the median of three warm, 128-token greedy runs at the same 131,072-token configured context:

| Fixed Portuguese task | Same GGUF, MTP off | MTP on | Observed change |
| --- | --- | --- | --- |
| Document backups | 39.35 tokens/s | 56.90 tokens/s | +44.6% |
| Library recovery | 37.87 tokens/s | 58.09 tokens/s | +53.4% |

This controlled table used conventional full KV allocation: tracked CUDA increased by about **908 MiB**, remaining below a 13 GiB managed quota. Separate JSON, tools, thinking and conversation A→B→A checks passed at temperature 0.6. In a further comparison using the preserved normal profile, an initial physical KV allocation of 8,192 cells reduced target-plus-draft KV buffers by **2,496 MiB**, while keeping the logical context at 131,072 and showing no throughput regression in three warm runs. A final 2,512-token prefill check supported retaining the original implicit microbatch 512, with 34/34 GPU layers, over 4.5 GiB device VRAM free and no observed decode regression. Batched verification can change floating-point logits and generated text; these local results do not establish universal quality or speed gains. See [MTP verification and memory](docs/native-mtp.md) and the [final benchmark](docs/benchmarks/native-mtp-final-20261028.json); the [initial record](docs/benchmarks/native-mtp-20261028.json) is preserved.

## Compatibility and current scope

- **Windows + NVIDIA CUDA** is the primary runtime validation target for the new selective-weight and GPU-first paths. Linux portability has been reviewed, but these paths still need wider Linux runtime testing; macOS and other backends are not validated for these additions.
- Linux selective-weight caching requires supported direct I/O and a sufficient locked-memory allowance. Lock failure is reported instead of silently changing to pageable managed weights.
- Validation covers several small MoE architectures and quantizations, with the exact matrices documented in the technical guides. It does not establish that every MoE model or multimodal path works.
- Dense models benefit from their applicable placement and runtime fixes. The **MoE host-weight budget does not turn arbitrary dense models into streamed MoE models**.
- Keep context capacity, reasoning policy, quantization, and memory budgets as separate choices. Memory tuning does not automatically change them, and optional reasoning limits can affect answer quality.
- Ollama compatibility remains the interface goal. New native features require a matching patched payload, and experimental paths may reject unsupported configurations.

## Documentation and integration

- [Build and development](docs/development.md)
- [MoE host-weight RAM budget](docs/moe-ram-limit.md)
- [GPU-first execution](docs/gpu-first-moe-inference.md) and [memory validation](docs/gpu-first-memory-validation.md)
- [Coordinated expert residency and transfer pipeline](docs/moe-residency-pipeline.md)
- [Dense-model runtime](docs/dense-runtime.md)
- [Native MTP and speculative decoding](docs/native-mtp.md)
- [GPU capabilities and measured dispatch policies](docs/cuda-tuning.md)
- [Runtime efficiency and measured results](docs/runtime-efficiency.md)
- [Live activity API and WildGoose integration](docs/runtime-activity.md)
- [Brand assets and regeneration](docs/branding.md)

[WildGoose](https://github.com/Anderseta/wildgoose) can manage the local service and model profiles and display WildLlama's sampled runtime activity. Other Ollama-compatible frontends can use the usual inference endpoints; they need explicit support to expose WildLlama's additional controls and telemetry.

For bugs and proposals, use [WildLlama issues](https://github.com/Anderseta/wildllama/issues). Include the model/quantization, OS, GPU/backend, relevant budgets, and a minimal reproduction. Distinguish cold loading, prompt processing, and steady-state decoding when reporting performance. Development in this fork follows `main`; upstream Ollama updates are reviewed separately.

## License and acknowledgments

WildLlama is an independent project derived from **[Ollama](https://github.com/ollama/ollama)**, with native inference powered by **[llama.cpp](https://github.com/ggml-org/llama.cpp)**. It is not an official Ollama release. Existing upstream copyright notices and the [MIT license](LICENSE) are preserved; bundled dependencies retain their own notices.

The MoE efficiency work also draws on ideas studied in [Strata](https://github.com/Niko1221/Strata). The implementation and validation notes identify the reviewed revision and explain the differences: WildLlama preserves router-selected experts and does not adopt approximate expert skipping to obtain its reported results.
