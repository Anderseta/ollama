# Dense-model runtime improvements

This work improves dense GGUF placement, native chat preparation and optional
CUDA execution without rewriting model weights, changing their quantization or
reducing the requested context. The native changes are carried by
[`007-dense-runtime.patch`](../llama/compat/007-dense-runtime.patch), on the pinned
llama.cpp `b11845` base and its preceding compatibility patches. A matching
native payload is required; replacing only the Go executable does not provide
the new CUDA capabilities.

The main local target is the installed Qwen3.5 9B Q8_0 dense model: 32 blocks,
24 recurrent Gated DeltaNet blocks and eight full-attention blocks. Its KV
requirements differ substantially from a conventional 32-attention-layer model.
The placement/chat corrections are automatic. Physical KV growth and the new
CUDA dense optimizations remain independently **off by default**.

## Separate reasoning budget

Native GGUF chat accepts optional `options.reasoning_budget`:

| Value | Meaning |
| --- | --- |
| Omitted | Preserve existing native/template policy. |
| `-1` | Inherit the native reasoning-budget setting; unrestricted when no setting supplies a budget. |
| `0` | Ask the native reasoning controller to close thinking immediately. |
| `1` through `2097152` | Limit the thinking-token budget to that value. |

The Go layer forwards the value as `reasoning_budget_tokens` and validates the
same range for API-decoded options and direct callers. It does not change
`think`, inject reasoning-effort instructions or modify `num_predict`. The
template must expose a supported thinking boundary for the native controller
to apply it. Closing the block can itself require delimiter tokens.

This is a user-selected reasoning/latency control. Reducing reasoning can change
answer quality; the runtime does not automatically select a smaller budget.
The total output limit still includes generated thinking and answer tokens, so
a separate budget does not guarantee a complete final answer. Check
`done_reason`, particularly `length`.

The implemented forwarding applies to the native chat route used by
`/api/chat`. Do not assume the same forwarding exists for every legacy rendered
completion or `/api/generate` path.

## Physical KV commitment with unchanged context

`options.kv_cache_initial_tokens` controls initial physical commitment, not
logical context capacity:

| Value | Meaning |
| --- | --- |
| Omitted | Use `OLLAMA_KV_CACHE_INITIAL_TOKENS`, otherwise `0`. |
| `0` | Disable growth and retain conventional allocation. An explicit API zero overrides the server setting. |
| `256` through `2097152` | Request that initial number of KV cells, clamped to the logical cache size. |

The launcher supplies the resolved child-local setting, including explicit
zero, so an inherited native-only environment setting cannot override the API.
The requested context, KV tensor dimensions/addresses and cache formats remain
unchanged.

The supported validation scope is one CUDA device with VMM, Flash Attention's
nontransposed V layout, one stream/sequence, no sliding-window cache and no
shared `other` cache. Unsupported backends/layouts retain conventional full
allocation. A CUDA device lacking the optional VMM API also uses that path.
Conventional storage is also selected when page rounding and quantized tensor
tail padding would make the sparse initial allocation as large as, or larger
than, the ordinary buffer. Small caches therefore do not pay a larger physical
allocation merely to enable growth.
This fallback can require the full ordinary allocation; it is not SSD paging or
permission to exceed a quota.

For an eligible cache, CUDA reserves the full virtual address range and maps
physical device pages as needed. An execution commits its required padded
prefix before modifying cell metadata or launching kernels. Expansion preserves
existing bytes and initializes new pages. Growth is monotonic until the buffer
is unloaded: deleting tokens or clearing a conversation does not release the
already mapped physical pages.

On the local Windows device, the VMM granularity is 2 MiB. For the Qwen3.5
configuration with Q8_0 K/V and context 131,072, the complete logical attention
KV is 2,176 MiB. An 8,192-cell initial prefix occupies approximately 160 MiB
after page rounding across its tensors. This is a page-allocation example, not
a total-process/GPU-memory figure or a throughput result. The Go `InitialKV`
estimate describes nominal row storage; native page rounding can make physical
commitment larger.

Mapped pages use the existing CUDA allocation policy. An expansion that cannot
fit the available managed quota fails explicitly; it does not reduce context,
silently move arithmetic to CPU, switch to Unified Memory or page KV to SSD.
Maintaining a large logical context is therefore not a promise that its eventual
maximum footprint fits the chosen quota. Weights, compute workspaces, recurrent
states, snapshots and other allocations still need headroom.

Managed CUDA bytes are not an operating-system-wide VRAM limit. Driver/display
allocations and other applications have separate scopes. See
[gpu-first-memory-validation.md](gpu-first-memory-validation.md) for accounting.

## Correct placement estimates

The Go estimator now uses explicit K/V head dimensions, per-layer GQA and
recurrent/full-attention layout instead of treating every layer as ordinary
attention. It respects the launch precedence of common versus separate K/V
formats and uses the selected format's block/row storage. Recurrent SSM state
and enabled speculative rollback rows are accounted for separately.

`WorstCaseKV` remains the logical maximum. `InitialKV` is a diagnostic initial
commitment estimate; it does not replace `WorstCaseKV` in admission totals.
Invalid metadata and arithmetic overflow do not turn an unknown footprint into
a fit. Native allocation/fit and actual allocator counters remain authoritative.

Measured placement observations are neither recorded nor reused as lifetime
bounds when partial initial KV commitment is enabled. A small load-time
observation must not convince the scheduler that the later expanded context
will remain that small. Existing architecture-specific parallelism restrictions
remain in place.

## Dense input embeddings in GPU-exclusive mode

GPU-exclusive execution also runs token lookup (`GET_ROWS`) on CUDA. The
traditional CPU placement of the input embedding table can then cause the
scheduler to copy the entire vocabulary table into a GPU compute buffer for
each batch, including single-token decode. For the inspected dense Qwen, that
table is approximately 1 GiB; gathering one token does not require transferring
the complete table repeatedly.

The correction places the standard token-embedding tensor with the first
decoder's CUDA buffers only for a dense model with full GPU-layer placement and
explicit GPU-exclusive execution. Other input tensors, normal execution and
MoE/expert-streaming placement retain their existing behavior. The regular
loader still applies explicit tensor overrides and its tied input/output
deduplication when both uses share a buffer context. An explicit CPU override
can therefore retain the original transfer cost.

Resident embeddings use the ordinary weight allocator and consume its CUDA
quota. This does not bypass allocation denial or silently make an infeasible
model fit. It changes placement and transfer lifetime, without changing lookup
values, weights or logical context. The focused placement regression and the
local trained-model comparison are recorded below.

## Native chat preparation and history

- Assistant `Thinking` is forwarded separately as `reasoning_content`. User,
  system and tool messages do not acquire that field. The native template still
  decides which historical reasoning to retain; visible content is unchanged.
- Native history truncation preserves a tool result together with the assistant
  call that produced it, including parallel results and legacy calls without
  IDs. An oversized final tool group keeps its valid boundary rather than
  returning an orphan result.
- Removing only preserved system messages does not trigger another identical
  sizing attempt. Generic fallback preparation retains at most one exact
  prompt/count locally, without hashing or caching prompts across requests.
- The additive `/apply-template` request field `count_tokens: true` returns
  `token_count` for eligible text-only runners. It renders and counts the entire
  prompt in one request, using generation's BOS and special-token parsing policy
  without exporting a potentially large token-ID array. Older payloads fall
  back to separate whole-prompt tokenization. Projector/media runners retain
  their separate accounting path and existing media heuristic.

Counts are never summed independently per message: Jinja control flow, tools,
schema, thinking and delimiters affect the whole rendered prompt. Final native
chat still applies its template to construct the correct parser and grammar.
This does not introduce a cross-request rendered-prompt, session or date cache,
and does not claim that Jinja runs only once for an entire request.

## Snapshot and shift safety

Buffered host/device state I/O performs tensor transfers through explicit
commits inside the caller's error handling. Destructors do not flush queued
transfers during a failed parse or unwinding. Invalid/truncated device headers
return failure; an unsuccessful replacement device snapshot is not exposed as a
valid saved state. Recoverable metadata/allocation errors can be handled without
throwing from a destructor.

KV restore precommits its required physical prefix before queuing tensor
copies. Failed-restore cleanup can zero already mapped ranges without trying to
allocate the missing pages that caused the failure. K-shift graphs and their
position inputs are bounded to the committed prefix rather than touching the
unmapped logical tail. Ordinary serialized formats and logical cache positions
remain unchanged.

These protections do not make every driver error recoverable or provide an
arbitrary concurrent state-restore contract. Existing context synchronization
and single-owner requirements still apply.

## Experimental opt-in CUDA dense paths

The experimental per-model runner setting is read before backend initialization:

```json
{
  "options": {
    "runner_env": {
      "GGML_CUDA_DENSE_OPT": "1"
    }
  }
}
```

`0` or omission retains the previous paths. This switch is independent of the
MoE runtime and does not activate speculative decoding. Leave it disabled for
the stable local Qwen3.5 9B setup: the small comparisons below do not establish
a repeatable benefit across both prompts.

Two narrow optimizations are implemented:

- Q8_0 matrix-vector paths can share the existing quantized conversion of the
  same immutable F32 activation between multiple consumers within one graph
  evaluation. Eligible inputs are contiguous, unbatched across higher dimensions
  and contain at most eight token rows. The plan rejects intervening writes and
  unsafe custom operations, uses a bounded 4 MiB reuse allowance, and does not
  share across concurrent streams/evaluations. Unsupported cases keep the
  existing conversion path.
- A guarded F32 RMS normalization, weight multiplication, SiLU gate and final
  multiplication pattern can execute as one CUDA fusion. Shape, dtype,
  contiguity, use-count and memory-range checks must pass; otherwise the original
  operations execute.

Weights and model operations are preserved. Floating-point dispatch and rounding
still require numerical regression checks; this document makes no universal
bitwise-equivalence or tokens-per-second claim.

## MTP and future work

The inspected local dense Qwen3.5 GGUF has `nextn_predict_layers=0` and no MTP
head tensors, despite MTP appearing in its descriptive model name. This work
does not train, add or activate a draft head. A future target/draft experiment
needs a compatible trained head, output/acceptance validation and joint memory
budgeting before any speed or quality claim.

## Recorded validation

The latest guarded native runs on the local RTX 5060 Ti report the following
results. These are component/fixture checks, not measurements of trained-model
answer quality or throughput. Local evidence is retained under
`../.build/gpu-first-moe/reports/`. The sanitized benchmark artifact records
settings, measurements and source/payload hashes without raw prompts or local
environment details: [dense-runtime-20261025.json](benchmarks/dense-runtime-20261025.json).

| Check | Recorded result | Local evidence |
| --- | --- | --- |
| Final CUDA dense operators, option on/off | PASS: CPU-reference comparisons and changed-input replay; five eligible gated-normalization cases require positive fused submissions with the option on, while a retained intermediate and all option-off cases require zero. | `dense-cuda-final-opt1.log`, `dense-cuda-final-opt0.log` |
| CUDA VMM API | PASS: virtual reservation without full commitment; 16/32 MiB quotas; sparse/idempotent growth; preserved bytes; views/2D; quantized padding; future-page clears; denied-growth cleanup; release to baseline. Observed page granularity: 2,097,152 bytes. Repeated on the final rebuilt payload. | `dense-vmm-final.log` |
| KV growth integration | PASS: growth off/on at unchanged 65,536-cell context, 320-token prefill plus 16 decode steps; maximum prefill/decode/K-shift differences all zero, identical greedy selections and bitwise state replays for the supplied tiny fixture. Repeated with the final rebuilt DLL. | `dense-kv-final.log` |
| Restore under quota pressure | PASS: 512 MiB quota, 32,768-token snapshot, 256-cell initial commitment; restore denial reported, cleanup recovered and subsequent continuation remained bitwise identical. One allocation denial was expected. Repeated with the final rebuilt DLL. | `dense-kv-final.log` |
| Failed device snapshot, CPU | `snapshot_failure`: one case passed, none skipped or failed. Truncated replacement save and short restore header failed as intended; the prior complete snapshot remained recoverable. | `dense-snapshot-cpu.log` |
| Failed device snapshot, CUDA | The same `snapshot_failure` case passed, with no skipped/failed cases. | `dense-snapshot-cuda.log` |
| Native server components | 42 cases, 157 assertions, zero failures, exceptions or skipped cases on the final rebuilt payload. | `dense-server-component-final.log` |
| Dense input-embedding placement | PASS: dense CUDA residency, shared tied storage, normal CPU placement, explicit CPU override, unchanged MoE placement, logits comparison and 64 MiB quota checks. Repeated on the final rebuilt payload. | `dense-embedding-final.log` |
| MoE streaming regression after placement change | PASS: 2,331 logits, zero maximum absolute/scaled error, finite outputs and identical top selections at all nine compared positions. | `dense-moe-regression.log` |
| Clean patch replay and idempotence | PASS: compatibility patches 001 through 007 apply to the pinned clean base; a second application succeeds without changes, with no file mismatches. | `dense-patch-replay.log`, `final-patch-replay.json` |

The dense-operator comparator requires finite values and
`abs(error) <= tolerance * (1 + abs(reference))`, using `1e-2` for projections
and `3e-5` for normalization. These tolerances are not bitwise guarantees. The
projection cases include one/eight token rows, incompatible/broadcast cases and
repeated capture/replay with changing inputs; the planner also checks writes,
aliases and custom side effects. Full-model numerical or language-quality
validation cannot be inferred from these operator checks.

The final normalization cases cover widths 128, 257 and 4,096, multiple tokens
and sequences, and a projected gate. Each eligible option-on case records two
fused submissions; the counter measures host launches/captures, not the number
of CUDA graph replays or individual GPU executions. An earlier matcher wrongly
required a linear chain even though the SiLU gate is an independent branch.
Its numerical comparator could pass through fallback without proving fusion.
The final matcher uses guarded subgraph fusion and the tests explicitly assert
activation as well as rejection. The old numerical pass alone is not evidence
that the fused kernel ran.

The KV integration uses the synthetic `llama-tied-wide.gguf` fixture. Its
ordinary CUDA KV allocation was 68 MiB versus 18 MiB initially with growth in
the recorded 65,536-cell case. This proves allocation behavior for that fixture;
it is separate from the production Qwen allocation example above. The forced
allocation denial and logged restore errors are required negative-test outcomes.

The embedding-placement regression generates separate small F32 dense, tied
dense and four-expert MoE fixtures. It checks the architecture classification,
embedding/decoder device agreement, tied pointer identity, unchanged logical
context, finite logits and CPU-versus-CUDA embedding results within
`1e-4 * (1 + abs(reference))`. Its CUDA allocator usage and peak remain within
the 64 MiB test quota. These fixture checks do not establish the production
Qwen Q8_0 model's throughput or every architecture's behavior.

The CPU snapshot log also contains a `CUDA_Host` compute-buffer bookkeeping
warning: 0.2704 MiB observed versus 0.1418 MiB expected. The snapshot case passed,
but this reserve/accounting diagnostic remains a known limitation requiring
follow-up; the passing case does not resolve the warning.

Go verification passed for `api`, `envconfig`, `llm`, `server`, `cmd` and
`discover`, recorded in `../.build/gpu-first-moe/dense-go-verification.log`.
Subsequent focused `api`/`llm` checks also passed in
`../.build/gpu-first-moe/dense-go-final-focused.log`. Coverage includes option
validation/defaults, safe placement estimates, initial-KV observation isolation,
assistant reasoning serialization, causal tool-history truncation, combined
text counting and compatibility fallback. This is not a claim that every
upstream package or every native model architecture was tested.

The rebuilt native payload passed the source/payload integrity guard. The
embedding, MoE, KV integration and server-component regressions above were run
after the placement correction; the earlier guarded operator checks cover
their respective components.
Clean replay verifies the patch bundle against its pinned base, separately from
the trained-model throughput comparison.

### Dense placement benchmark

The completed `dense-placement-v2-suite.json` comparison uses the installed
Qwen3.5 9B Q8_0 on the RTX 5060 Ti, before versus after the embedding-placement
correction. Both sides use context 131,072, batch 512, microbatch 128, Q8_0 K/V,
Flash Attention, all GPU layers, GPU-exclusive mode and a 12,288 MiB managed
CUDA quota. Fit is disabled, `GGML_CUDA_DENSE_OPT=0`, initial-KV growth is
disabled, mmap is disabled, temperature is zero, seed is 42, thinking is off
and the output cap is 128 tokens. This is not a comparison with unmodified
upstream Ollama.

Each of two natural-prose prompts has one new-runner request and two repeated
warm requests per payload. Here, runner-cold does not mean cold SSD/OS page
cache. The prompts contain 55 and 506 tokens; their warm repeats reuse 51 and
502 cached tokens respectively. All twelve requests generate 128 tokens.

| Prompt | Runner-cold tokens/s, before → after | Mean warm tokens/s, before → after | Warm ratio |
| --- | --- | --- | --- |
| Short, 55 tokens | 5.83 → 44.12 | 5.82 → 44.33 | 7.62× |
| Longer, 506 tokens | 5.81 → 40.52 | 5.83 → 41.51 | 7.12× |

Throughput is the API's `eval_count / eval_duration`. Measured time to first
output, including loading on new-runner requests, changes from 5.703 to
5.469 seconds for the short prompt and 8.250 to 6.531 seconds for the longer
prompt. Warm first-output times range from 0.219–0.266 seconds before and
0.062–0.125 seconds after. These are a few local sequential runs, not a
statistical latency distribution or a maximum-context throughput test.

The tradeoff is persistent weight residency: the sampled managed CUDA peak
rises from 11,859,315,200 to 12,444,705,024 bytes, an increase of 558.27 MiB,
while staying below the 12 GiB quota. The logs report weight buffers increasing
from 8,045.05 to 9,075.67 MiB, compute reservation decreasing from 1,032.63 to
560.27 MiB, disappearance of the 1,030.62 MiB host model buffer and graph
splits decreasing from three to one. Managed peaks cover this runner's native
allocator; they exclude other processes, display/driver allocations and
unsampled transients.

All six matched before/after request pairs have byte-identical visible text
and thinking fields; thinking is empty in these runs. This supports output
preservation for the tested prompts and settings. Outputs are capped at 128
tokens, so it is not a general answer-quality evaluation or proof of numerical
identity on every input.

### Optional CUDA-path comparisons

Both comparisons repeat the two-prompt, three-request procedure, using the
same payload on each side with `GGML_CUDA_DENSE_OPT=0` versus `1`. Initial-KV
growth remains disabled and the other controls match the placement comparison.

| Comparison | Short mean warm tokens/s, off → on | Longer mean warm tokens/s, off → on | Scope |
| --- | --- | --- | --- |
| D2, `dense-kernels-v2-suite.json` | 43.79 → 41.72 (−4.7%) | 44.03 → 43.55 (−1.1%) | Historical matcher: input-quantization reuse was observed; normalization fusion was not. |
| D3, `dense-kernels-v3-suite.json` | 44.06 → 44.16 (+0.2%) | 38.36 → 44.34 (+15.6%) | Final payload with the corrected subgraph matcher and activation-checked component tests. |

D2 predates the matcher correction and is not a measurement of the final
normalization-fusion path. D3's longer-prompt control is lower than earlier
controls, which were approximately 41–44 tokens/s. With only two warm repeats,
the observed difference cannot establish a repeatable causal speedup or a
universal benefit. The short-prompt difference is small. The option remains
experimental and off by default.

Each comparison has six matched option-off/on pairs with byte-identical visible
text and thinking fields. All four runs in each comparison report the same
sampled managed peak of 12,444,705,024 bytes. Passing component comparisons
establishes their bounded correctness coverage; it does not establish an
end-to-end performance gain.

### Production KV-growth check

`dense-growth-v2-validation.json` records the corrected Qwen payload with
initial commitment disabled versus 256 cells. Context remains 131,072, the
optional dense kernels remain disabled and the same 12 GiB managed quota
applies. Each side runs a 4,961-token prompt followed by 32 output tokens, then
one repeated request using 4,957 cached prompt tokens. Both matched output
pairs are identical. This exercises real growth past the requested 256-cell
prefix, rather than only loading an empty cache.

The partial cache starts with 32 MiB physically committed for its 2,176 MiB
logical KV capacity. After those requests, the live runner-managed CUDA count
reported by the loaded-model API is 10,263,666,944 bytes, versus
12,444,705,024 bytes for conventional commitment: 2,080 MiB less for this
tested prefix. These are final live managed counts, not global GPU peaks or a
promise that memory remains this low as more of the logical context is used.

Runner-cold generation is 41.42 versus 41.88 tokens/s; the single warm request
is 43.79 versus 42.27 tokens/s. First-output times are 7.766 versus 7.000
seconds runner-cold and 0.094 seconds on both warm requests. Thirty-two output
tokens and one warm repeat are insufficient to establish a throughput gain or
an absence of overhead. The demonstrated benefit is lower physical commitment
at unchanged logical capacity. The placement speedup above was measured with
growth disabled and must not be attributed to this option.

### Native preparation and dispatch diagnostics

`dense-diagnostic-v2-validation.json` records a text-only template request whose
combined `token_count` is 17, matching separate whole-prompt tokenization at
17. This is a direct endpoint regression check, not a timing result or a claim
that every multimodal/template path is covered.

The earlier v2 CUDA diagnostic contained shared-input quantization/reuse events
but no gated-normalization fusion, before the matcher correction. The final
`dense-diagnostic-v3-server.log` records both reuse and `norm-gamma-silu`
submissions on the actual Qwen graph, including shape `[128,32,2,1]`.
Diagnostics print at most four events of each kind; this is not a total
execution count or a timing attribution. Final activation evidence and
component correctness do not by themselves establish a repeatable speedup.
