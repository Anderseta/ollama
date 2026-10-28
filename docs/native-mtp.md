# Native MTP and measured speculative decoding

MTP is a decoding accelerator, not a different reasoning mode. A compatible
prediction head proposes tokens, and the target model verifies them. A proposal
is never accepted merely because its text looks plausible. The normal
sample-and-match path samples the target with the request's temperature,
filters, penalties and constraints.

The gain depends on acceptance, the cost of the head, target verification and
the GPU. An acceptance percentage is not a speedup measurement. Compare output
tokens per second, time to first token and memory under the same context and
sampling settings. Portuguese prompts work through the same tokenizer and
verification path; they do not require an English-only MTP setting.

## Compatible weights

A repository or filename containing `MTP` is insufficient. The GGUF must retain
the architecture's prediction-head metadata and tensors, or a compatible
sidecar must already be configured. WildLlama recognizes embedded Qwen heads
through `nextn_predict_layers` and supported legacy `mtp.*` tensors. It cannot
reconstruct a missing head from ordinary target weights.

When replacing a quantization, verify the fine-tune, quantization and chat
template, not just the parameter count. To preserve an existing target exactly,
compare every common tensor's name, shape, type and bytes against the original.
Check the complete download against the pinned source revision and digest
before importing it. Retain the old model until functional and memory checks
have passed.

## Options

Native GGUF chat and completion support these per-model/request options:

```json
{
  "speculative_type": "draft-mtp",
  "draft_num_predict": 2,
  "speculative_adaptive": false
}
```

`draft_num_predict` bounds the auxiliary draft length. Start with a short
measured depth; larger is not necessarily faster. `speculative_type: "none"`
explicitly disables speculation. Omitting the type preserves the engine's
existing selection policy; configure both fields explicitly when testing MTP.
A model name containing `MTP` does not activate a missing or disabled head.
These settings do not change the requested context, output limit, target
quantization or sampling parameters.

The optional adaptive controller measures drafting and verification costs and
can reduce speculative work. It is experimental and remains off by default.
WildGoose exposes these options in the model profile and persists them with
the other profile parameters.

## Memory accounting

The extra head's weight size is only part of the cost. Qwen's MTP draft context
also has a full-attention K/V cache, compute buffers and target recurrent
rollback states. The estimator separates target blocks from prediction blocks
and counts the draft attention cache independently. The native draft cache is
F16 by default; the target's Q8 cache setting does not automatically change it.

For example, one 9B Qwen Q8 head added about 247 MiB of weights, but its F16
draft cache at 131,072 tokens added 512 MiB before rollback states and scratch.
This is an inspected configuration, not a universal MTP memory requirement.
Admission estimates do not replace allocator and whole-device observations.
Those observations still determine whether the actual run fits. Do not lower a
user's context silently to make speculation fit.

In the measured configuration below, tracked CUDA allocations increased from
11,868.20 MiB without MTP to 12,776.28 MiB with depth two: about 908.08 MiB
extra. The configured quota was 13,312 MiB. This quota excludes display, driver
and other applications; the minimum sampled whole-device free memory was
1,339.60 MiB in the final controlled run. A larger context, another cache format,
concurrent models or a different architecture can require more memory.

Physical KV growth is a separate setting. In the normal-profile comparison with
`kv_cache_initial_tokens: 8192`, both this Qwen's target and MTP draft caches
were eligible for CUDA VMM: target physical KV changed from 2,176 to 160 MiB,
and draft physical KV from 512 to 32 MiB. Their logical capacities remained
2,176/512 MiB and 131,072 cells. The initial saving was 2,496 MiB (2.44 GiB).
Other layouts can fall back to conventional allocation.

The estimator's draft-head `InitialKV` remains conservative and can count the
complete logical head cache. It is not a measurement of initial physical
commitment. Admission's worst-case budget, the initial diagnostic estimate and
actual physical buffers have different scopes. Growth still requires memory;
it is not SSD/swap paging and does not reduce context or alter cache formats.

## Conversation checkpoints and carryover

The target KV/recurrent state and the draft KV are not the complete MTP state.
The head also carries a target hidden row into the next draft catch-up step.
Restoring another conversation's KV while keeping the previous conversation's
hidden row can contaminate the draft cache before the next proposal. This is a
draft-state/acceptance issue; it does not by itself establish corruption of the
target's verified output.

The cache corrections in patches `014` and `015` capture target, draft and
speculative carryover as one transaction. A replacement is published only after
capture succeeds. A failed restore clears both incomplete live sequences and
carryover and keeps the saved cache entry available for retry. Empty restores
and a full prompt restart reset carryover before prefill; resetting only in the
generation's `begin()` hook would occur after prefill has already consumed it.

For ordinary one-dimensional text, MTP checkpoints retain the pending hidden
row, verification rows with their positions, and the boundary row preceding the
verification batch. The server re-evaluates at least the final token of an
otherwise fully cached prompt. If that token is at position `p`, catch-up needs
`h(p-1)`, rather than simply the last hidden row in the snapshot. The boundary
also preserves that predecessor when the preceding verification batch had only
one row. Rows belonging to a rejected proposal suffix are removed from reusable
history; retained buffer capacity can be reused by later rounds.

Snapshot loading checks the format, embedding width, bounded row counts,
positions and the byte identity of the pending row with its verification copy
before replacing live carryover. Before submitting target decode for a cached
prefix, the server checks that its first token has a compatible predecessor.
If the required row is outside the saved history, it discards prefix reuse and
processes the full prompt. It does not substitute a hidden row from another
position or approximate the target computation.

Visual MROPE positions can repeat or jump: Qwen image patches can share the
same temporal `pos[0]` while their spatial positions differ. The positional
text-cache rules are therefore not imposed on every media batch. Such a flow
keeps the existing draft catch-up path and stops advertising a reusable
one-dimensional carryover snapshot; a later prefix-reuse attempt falls back to
full prefill. A repeated visual position zero in a nonempty sequence is not
itself treated as a fresh text start. This is a conservative cache fallback,
not a new multimodal indexing or hidden-state reconstruction algorithm.

The history buffers reuse their capacity during inference. Serialized history
still needs host memory and is bounded by the configured batch geometry; it is
included in speculative snapshot data and the prompt-cache budget. These
changes do not add a separate collection of GPU-resident conversation snapshots
or relax the existing hybrid-model parallelism restrictions.

The added CPU regressions cover positional state validation, last-token replay,
conversation A/B restores, shorter-prefix checkpoints, rejected rows, empty
reset, nonunitary positions and target/draft/speculative restore failures.
These are correctness checks. No throughput, latency, acceptance-rate or task-
quality improvement is claimed for these checkpoint changes without a separate
measurement.

## Outputs and diagnostics

With stochastic sampling, different output text is expected. Verification
preserves the target sampling procedure, but floating-point logits can differ
between single-token and batched verification. Even greedy text can therefore
change. This is not proof of a state bug, nor does it prove equal task quality.

The diagnostic in [MTP target-logit diagnostics](../scripts/support/mtp-diagnostics.md)
replays identical forced prefixes and compares ordinary decode, batched decode
and recurrent rollback. It distinguishes a batching difference from additional
rollback drift. Functional tests should also exercise the actual sampling
settings, JSON constraints, tool calls and reasoning control.

Final `/api/chat` and `/api/generate` responses can include:

```json
{
  "speculative_draft_tokens": 119,
  "speculative_accepted_tokens": 66
}
```

The counts are optional. Unknown counts are omitted; a known zero remains zero.
Acceptance is accepted draft tokens divided by proposed draft tokens. It does
not include the extra target-sampled token produced by each verification step.
The same completed-generation observation is available through `/api/activity`,
with its observation timestamp. It describes the shared engine's last
generation, not necessarily the current chat or a specific speculative
algorithm. New generations clear the previous observation. WildGoose labels
the scope explicitly and does not present acceptance as a speed measurement.

## Measured Qwen9B result, 2026-10-28

An RTX 5060 Ti 16 GiB on Windows ran the Qwen3.5 9B DARK ROAST Q8_0 fine-tune
with an embedded head from the pinned
[mradermacher conversion](https://huggingface.co/mradermacher/Qwen3.5-9B-The-Defiant-Fable-DARK-ROAST-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF/tree/46f0623297f0df3e9b09d4eb1c63e2c1756e591a).
All 427 common tensors, totaling 9,516,533,760 bytes, matched the previous
headless file byte for byte; the chat template also matched. The replacement
added 15 head tensors. Both full-file digests were verified.

Two fixed Portuguese prose prompts produced 128 counted output tokens per run.
The table reports the median of three repeated warm runs per prompt, with
context 131,072, batch/ubatch 512/128, one sequence, target Q8_0 K/V, Flash
Attention, full GPU placement and the same 13 GiB tracked-allocation quota.
Timing runs used greedy sampling to keep a controlled baseline. Depth-two MTP
used the ordinary sample-and-match verifier; adaptive control and the separate
experimental dense CUDA switches stayed off.

| Prompt | Same new GGUF, MTP off | MTP depth two | Observed change | Warm draft acceptance |
| --- | --- | --- | --- | --- |
| Document backups | 39.35 tokens/s | 56.90 tokens/s | +44.6% | 55.5% |
| Library recovery | 37.87 tokens/s | 58.09 tokens/s | +53.4% | 57.6% |

These final measurements include the checkpoint/carryover corrections above.
They compare whole MTP decoding with speculation disabled; no separate speedup
is attributed to a particular checkpoint fix. Empty-cache trimming and the
experimental dense CUDA paths remained off.

Median warm time to first token was 78 ms in both modes for the first prompt,
and 78 ms in both modes for the second. Initial loads were
recorded separately; filesystem/compiler caches were not flushed, so the
ordered first-load timings do not establish a loading-speed improvement.

The eight new-file runs without MTP all matched the old-file reference text.
Four of eight MTP runs differed in greedy text. Forced-prefix replay reproduced
the first observed difference with target-only three-token batching: the same
prefix changed argmax without invoking the head. Snapshot-only autoregressive
decode remained identical; rejected-suffix rollback produced the same reported
full-vocabulary error metrics and top-eight evidence as pure batching at all
129 compared prefixes. This isolates the observed batching effect without
claiming that every model's state handling or task quality has been proved.

The largest raw-logit difference in that replay was 0.49496, and maximum
full-vocabulary softmax total variation was 0.04412. The latter uses temperature
one and is not a comparison of the user's temperature/top-k/top-p distribution.
Independent JSON-schema, tool-call and completed-thinking checks passed with
temperature 0.6, top-p 0.95, top-k 20 and repeat penalty 1. Conversation A→B→A
returned the correct independently checked values 42, 57 and 42 on the same
MTP model. The final A reused 332 of its 336 prompt tokens. The schema allowed
any integer, so correctness was not supplied by an answer enum.

The [final sanitized record](benchmarks/native-mtp-final-20261028.json) includes
model/payload hashes, sampling, warm measurements, memory scopes and recovery
observations. The [initial record](benchmarks/native-mtp-20261028.json) remains
available with the fixed synthetic prompts and earlier diagnostic evidence.
These short prompts validate configured
capacity and this local speedup; they do not measure generation with a filled
131,072-token history or establish gains on other GPUs/models. No weight
quantization, context reduction, approximate token acceptance or expert
skipping was used to obtain the result.

## Preserved profile and physical KV comparison

A separate check preserved fit on, automatic GPU layers, the 2,048 MiB margin,
context 131,072 and the original temperature 0.6/top-p 0.95/top-k 20/repeat
penalty 1. GPU-exclusive execution and a managed CUDA quota were not forced.
All 34/34 native layers remained on GPU. Two warm runs measured 42.61 tokens/s
without MTP and 61.21 with depth two. This smaller stochastic comparison is
separate from the controlled greedy table above.

Under that same normal profile, a confined full-KV versus 8,192-initial-cell
comparison used three warm 128-token runs per mode. The medians were 55.27 and
60.54 tokens/s; warm first-token medians were 110 and 78 ms. There was no
observed throughput or warm-prefill regression. All four same-seed prose
outputs matched, and both tool checks had correct function names/arguments.
The full-allocation samples varied, so the observed ratio is not presented as
a general speedup caused by VMM. The 8,192-cell case kept at least 4,948.59 MiB
of whole-device VRAM and 12.40 GiB of system RAM free in the sampled run.

One initial target-only load in the normal-profile check reported CUDA shared
object initialization failure. The Windows startup handler retried once with
child-local `CUDA_MODULE_LOADING=EAGER`, preserving inference arguments, and the
request then completed. The following MTP load and the separate KV comparison
needed no retry. This records recovery without establishing the underlying
cause or requiring a global loader setting.

The last bounded comparison checked microbatches 128 and 512 with MTP depth two
and initial physical KV 8,192. Each runner processed one independent, uncached
2,512-token synthetic prompt with 64 output tokens, followed by two resident
128-output-token decode requests. Prompt evaluation took 1.1646 s at 128 versus
1.0185 s at 512; the separate decode medians were 59.44 and 60.43 tokens/s.
Both retained 34/34 GPU layers. Microbatch 512 kept at least 4,637.42 MiB of
whole-device VRAM and 13.94 GiB of system RAM free, exceeding the two-GiB
selection criterion without an observed decode regression. These few ordered
samples do not establish a universal gain.

The final tested profile keeps microbatch 512, matching the previous implicit
microbatch, with MTP depth two and initial physical KV 8,192. Sampling, logical
context and fit margin stay unchanged. Physical KV growth is an explicit
profile choice and remains off by default; later expansion still requires
headroom. These real-model comparisons did not exercise growth beyond the
initial cells.
