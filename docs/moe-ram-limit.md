# Experimental MoE host-weight RAM budget

This experimental fork adds a bounded, locked host-weight cache to the llama.cpp runner.
It is a model-weight budget, not a process memory or VRAM quota. The default is off.
The original context, KV precision, and compute settings are independent of it.

## Configuration

```sh
ollama serve --moe-ram-limit-mib 10240
```

Equivalent server environment: `OLLAMA_MOE_RAM_LIMIT_MIB=10240`.
`LLAMA_ARG_MOE_RAM_LIMIT_MIB` is accepted as a fallback. Server defaults apply only
to MoE models. A per-model/API setting takes precedence, including explicit zero:

```text
PARAMETER moe_ram_limit_mib 10240
PARAMETER num_ctx 131072
```

```json
{"model":"my-moe","options":{"moe_ram_limit_mib":10240,"num_ctx":131072}}
```

Units are MiB; 10240 MiB = 10 GiB. Zero restores ordinary loading. Fractions,
negative values, invalid units, and values above 2097151 MiB are rejected.
The native runner accepts `--moe-ram-limit-mib` too.

## Mechanism and scope

* Raw host GGUF tensors receive stable virtual address reservations. Reserving
  addresses does not read or commit the complete tensor.
* A scheduler preparation hook loads weights before each active consuming op.
  `MUL_MAT_ID` loads selected experts; `GET_ROWS` loads selected rows. Other ops
  require their complete managed input spans. Inactive SELECT branches are skipped.
* Resident pages are 64 KiB, charged including padding. A segmented LRU promotes
  reuse and evicts probation pages first; all pages needed by the current operation
  are pinned during selection. A required set larger than the quota aborts cleanly.
* Windows uses unbuffered GGUF reads and VirtualLock; Linux uses O_DIRECT and mlock.
  Failure to lock is an error, never a switch to pageable model weights. Eviction
  decommits pages. A fixed aligned 4 MiB I/O staging buffer is also locked; it is
  separate from the host-weight quota.
* CPU operations, host-to-GPU expert copies, and existing GPU expert-cache misses
  all use the same model-owned cache. Device uploads complete before staging or
  host cache pages can be reused. Concurrent contexts serialize managed-weight
  execution until a separate pinning/prefetch design is implemented.
* Initial device weights load through the locked I/O staging buffer, without a
  second pageable host copy or whole-model page-cache scan.
* Host weight mapping spans are replaced by the weight-cache allowance in Ollama's
  memory accounting. Device, KV, and other buffer allocations remain counted.

KV caches, activations, runtime metadata, other processes, the driver, and GPU
allocations are outside this quota. It cannot promise that Windows will never page
other allocations. `--moe-cache-mib` controls an independent GPU expert cache.
`--fit-target` is VRAM placement guidance, not a hard VRAM allocation limit.
For one GPU, fitting preserves CPU expert overrides and explicit context length
while searching the number of GPU layers when the initial placement does not fit.

This requires a supported raw GGUF MoE, mmap, disabled host repacking, disabled
whole-model tensor checking, and no whole-model mlock. The common model-parameter
conversion selects the compatible loading settings. Transformed host layouts and
non-MoE explicit requests fail rather than silently ignoring the limit.
It is architecture-independent at the graph-op level, but universal support is not
established by the initial three-architecture test matrix. Windows/CUDA is the
local validation target; other OS and backend paths still require validation.

## Source and builds

The fork is based on Ollama commit `675150d6ad2747df1e218ef59de9194bc4a7e538`.
The native patch targets llama.cpp `b11845`, commit
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`.
`llama/compat/002-moe-ram-limit.patch` carries the native implementation and test,
and is applied by the existing CMake patch mechanism after the compatibility hook
patch. Repeated application has been checked. Build through the normal Ollama
CMake workflow; setting `OLLAMA_LLAMA_CPP_SOURCE` deliberately skips patches, as
in upstream, so prepare that developer tree manually in that case.

Native test: build `test-moe-ram` with LLAMA_BUILD_TESTS and pass a small MoE GGUF,
weight budget in MiB, optional GPU layers, and optional GPU expert-cache MiB.
It compares complete logits across prompt microbatches and generation, and checks
clean failure under an insufficient budget. Test fixture generation and local
results are recorded in the laboratory diagnostics, not production models.

This is an experimental fork. No upstream acceptance or production
readiness is implied; an upstream design discussion and wider platform testing
would be needed before proposing a contribution.
