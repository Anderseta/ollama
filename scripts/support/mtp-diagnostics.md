# MTP target-logit diagnostics

These standalone tools investigate a changed token without changing the
production sampler, profiles, context size, or native runtime. A different
sentence alone does not identify a recurrent-cache bug or prove unchanged
quality.

`mtp-logprob-trace.py capture` talks to an already running **isolated native**
`llama-server`, renders its actual template, tokenizes it, and requests exact
generated token IDs with target logprobs **before sampling**. For the second
capture use `--prompt-from` to reuse the first capture's exact prompt tokens.
The `compare` command reports the first different token, shared prefix, and
top-two logit gap. JSON uses ASCII escapes, so console encoding cannot silently
change the evidence. A trace can add memory overhead; use the same live GPU/RAM
guards as the isolated benchmark. The tool does not start or stop a server.

```text
python -B scripts/support/mtp-logprob-trace.py capture --url http://127.0.0.1:NATIVE_PORT --messages messages.json --output target.json
python -B scripts/support/mtp-logprob-trace.py capture --url http://127.0.0.1:NATIVE_PORT --messages messages.json --prompt-from target.json --output mtp.json
python -B scripts/support/mtp-logprob-trace.py compare --baseline target.json --candidate mtp.json --output comparison.json
python -B scripts/support/test_mtp_logprob_trace.py
```

`mtp-logit-replay.cpp` replays the target trace's generated tokens as **forced
input**, so all compared logits see the same prefix. It loads one model and
creates/frees one target context at a time. There is no draft context, so this
diagnostic isolates target batching and recurrent rollback from draft quality.
It compares:

- Autoregressive decode without recurrent rollback snapshots.
- Autoregressive decode with two snapshots.
- Autoregressive decode with MTP hidden-output collection enabled.
- Three-token verification batches without rejection.
- Three-token batches with deliberately wrong suffixes, then one- or two-token
  rollback before continuing the original forced prefix.

The report includes the raw full-vocabulary maximum/RMS error, normalized
softmax total variation, argmax changes, and the top eight exact token/logit
pairs for every prefix. Total variation here is computed at temperature 1 on
the complete vocabulary. It is **not** the difference between the user's
temperature/top-k/top-p sampler distributions. A matching collection of these
summary metrics is evidence, not a proof that every raw bit matches.

To build, make a temporary CMake source directory outside production with:

```cmake
cmake_minimum_required(VERSION 3.20)
project(mtp_diagnostic LANGUAGES CXX)
include("/absolute/path/to/scripts/support/mtp-logit-replay.cmake")
```

Configure it with `NATIVE_SOURCE` set to the exact patched native source and
`NATIVE_BUILD` to its existing build. On Windows this links the existing import
libraries; ensure the matching native DLLs are discoverable via `PATH`, and
select the CUDA backend DLL through `GGML_BACKEND_PATH` if it lives in a separate
directory. It does not compile or replace production native libraries.

```text
mtp-logit-replay --tiny --output tiny.json
mtp-logit-replay --model exact-model.gguf --input target.json --output replay.json --context 131072 --vram-mib 13312
```

`--tiny` uses an in-memory Qwen hybrid fixture with deterministic nonzero
weights, 64 forced tokens, and CPU-only tensor placement. It downloads nothing.
Actual model replays require the owner's isolated GPU benchmark guards; the
managed VRAM quota alone does not cover display/driver allocations or RAM.

For normal sampling, sample-and-match verification still draws each accepted
token from the target's full configured sampler. It does not accept a draft
because its text is plausible. Batching can nevertheless change floating-point
target logits. To assess that effect, inspect the same-prefix numerical evidence
and exercise functional tasks with the real sampling settings; exact generated
text is not a suitable stochastic-output gate.
