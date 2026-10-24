# WildLlama runtime performance

This private fork keeps model files, quantization and the requested context unchanged.
The native delta is `llama/compat/003-runtime-performance.patch`, applied after the
existing compatibility and MoE RAM patches. These changes are not an upstream PR.

## Automatic improvements

- Prompt-cache restoration compares estimated saved prefill time with observed
  restoration cost once both measurements exist. Cold caches retain the previous
  heuristic. Matching remains exact, including recurrent checkpoints.
- Explicitly noncausal GGUF pooling models can batch independent text embeddings.
  Tokenization is batched, work is bounded by physical tokens and sequence count,
  duplicate texts are computed once, and results retain input order. Media and
  incompatible runners retain their existing path. A runner 413 retries only the
  affected group through the existing truncation logic. Cancellation and errors
  propagate. `OLLAMA_EMBEDDING_MAX_SEQUENCES` selects 1–32 slots (default 8 for
  eligible models); 1 restores the sequential runner behavior.
- Resident-model eviction considers measured reload cost, reuse, recency and bytes
  released among models with the same keep-alive priority. Idle models are still
  preferred. Missing observations retain the previous ordering.
- A bounded, process-local placement cache records fully GPU-resident dense loads
  with allocator headroom. Signatures include model timestamp, devices, drivers,
  context, runner options and relevant environment. Partial placements, MoE,
  projectors, auxiliary drafts and uncertain observations use existing estimates.
- Chat truncation keeps tool results with their assistant calls, preserves system
  messages and prepares cut boundaries in one pass. Rendered token counts and
  native recurrent/SWA checkpoint checks remain authoritative. No approximation of
  recurrent state or new media-token heuristic is introduced.

## Per-model runner settings

`GET /api/version` advertises `runner_env`, `num_ubatch` and `embedding_batch`.
`options.runner_env` is a validated object of native runner settings. It can also
be persisted through `/api/create` in `parameters.runner_env`. Values affect the
child runner, without changing the server process environment. Paths, credentials,
model selection and arbitrary environment variables are rejected.

For example, a reference model can retain an independent cache and physical batch:

```json
{
  "model": "example-profile",
  "from": "existing-model",
  "parameters": {
    "num_ubatch": 128,
    "runner_env": {
      "LLAMA_ARG_CACHE_RAM": "1024",
      "LLAMA_ARG_FIT_TARGET": "4096",
      "OLLAMA_KV_CACHE_TYPE": "q8_0"
    }
  },
  "stream": false
}
```

Allowed families: load/lazy/fit, GPU and CPU-MoE placement, MoE GPU cache and RAM
weights budget, prompt RAM cache, logical/physical batches, KV cache types, flash
attention, parallel sequences, tensor placement and warmup. Empty values clear an
inherited runner override. Explicit API options and command-line settings retain
their existing precedence. The RAM budget still covers MoE weights, not total
process memory or dense offload.

`num_ubatch` separates the physical microbatch from `num_batch`; zero/omitted keeps
the previous behavior. Smaller physical batches can save scratch space but may
reduce prefill throughput. Whole-prompt pooling/decision models reject a microbatch
smaller than their required physical batch rather than silently splitting it.

Wildgoose detects this capability. Applying load-affecting settings unloads only
the affected model. Sampling-only changes keep its weights resident. Truly global
settings such as maximum loaded models and keep-alive can still require a daemon
restart; official Ollama retains the legacy profile behavior.

## Optional experiments

These are exposed in Wildgoose profiles and remain disabled/omitted by default:

- `backend_sampling: true` tries native backend token selection. Grammar, reasoning
  budgets and incompatible samplers retain the CPU path. Existing filters and
  penalties are preserved. Shader startup can increase the first-token delay.
- `speculative_type`: `auto`, `none`, `ngram-simple`, `ngram-mod`, `ngram-cache`.
  N-grams require no extra model; they still use target-model verification.
- `draft_num_predict` bounds proposed tokens. `speculative_adaptive: true` measures
  ordinary generation and total proposal/verification cost, cautiously probes,
  reduces ineffective drafts and retries after a cooldown. It cannot make every
  prompt faster and does not enable missing MTP weights.
  Probes use the initialized technique's native span: tiny drafts can lose solely
  to fixed recurrent-checkpoint costs and incorrectly hide profitable long drafts.

Greedy-output equality, embedding cosine/ranking comparisons and grammar checks
are useful regressions, not proof of unchanged quality on every possible input.
Hardware and model measurements belong in a separate benchmark report; cold load,
warm generation, prompt processing and cache restoration must be reported separately.
