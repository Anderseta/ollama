# Live runner activity

WildLlama exposes `GET /api/activity` on the same HTTP listener as `/api/chat`
and `/api/generate`. The normal inference request and streaming formats are
unchanged. `/api/version` advertises the optional `live_activity` capability.
Clients must opt into polling and presenting these additional observations.

The response has `version: 1`, `scope: "shared_runner"`, an `observed_at` Unix
timestamp in milliseconds, `scheduler_busy`, `queue_length`, and `runners`.
Each runner includes its model name, PID, context length, GPU-exclusive policy,
and an `activity` snapshot. Polling does not wait for the scheduler's model-load
lock; a busy scheduler can temporarily return no runners with `scheduler_busy`.
An empty list in that case does not mean a load finished or failed.

Activities report the last measured phase (`starting`, `loading`, `context`,
`warming`, `ready`, `prefill`, `generating`, `sleeping`, or `failed`) and up to
24 phase transitions. Very short phases can appear only in this history.
Optional fields include loader progress, processed/total prompt tokens, generated
tokens and generation rate. Percentages apply to measured loading/prefill work;
there is no estimate of how much of an answer remains to be generated.

An optional `operation` distinguishes computation, MoE routing, missing weight
reads and weight uploads. These are sampled observations: a client need not see
every brief operation. Native observations are throttled and introduce no new
CUDA events, graph barriers or tensor readbacks. The Go launcher enables the
native protocol through the child-local `WILDLLAMA_ACTIVITY=1` environment value.

Memory fields measure the selective RAM weight cache and tracked CUDA allocator.
They are not total process RAM or total physical GPU usage. SSD-read and RAM-to-GPU
byte counts accumulate over this model load, including previous requests. See
[the memory contract](gpu-first-memory-validation.md) for quota exclusions.

Snapshots are shared by all slots using a runner. Clients must not attribute them
to a particular concurrent chat without their own request tracking. Timestamps
belong to the server clock; remote clients can translate relative to `observed_at`.
The schema omits prompts, answers, token IDs, file paths and arbitrary log text.

WildGoose polls while observing model activity and presents a compact phase with
optional measured progress/rate. Expanded details show the memory scopes and
transfer counters. Servers without this endpoint retain the ordinary request and
streaming feedback. A telemetry timeout does not fail the inference request.

## Isolated live check

After preparing a laboratory with `scripts/benchmark-gpu-first.py --prepare-only`
and building a separate payload, run:

```powershell
python scripts/test-runtime-activity.py `
  --payload .build/gpu-first-moe/payload-candidate `
  --lab-root F:/WildLlama-GpuFirst-Lab `
  --output .build/gpu-first-moe/reports `
  --label live-activity-check
```

This check uses the installed Qwen Flash model metadata from the prepared lab,
context 131072, a 6144 MiB weight cache and a 12288 MiB managed CUDA quota. It
observes two `/api/chat` requests, a standard `/api/generate` request, unload,
phase history, polling latency and quota counters. The benchmark owns and stops
only its isolated server tree on port 11448. Each label must be new so failed
observations are preserved rather than overwritten. These short requests check
integration, not sustained throughput or numerical equivalence across models.

## Windows validation, 2026-10-25

The first live request exposed an invalid task access in the new prefill
observation: the slot did not own its request yet. The event now runs immediately
after ownership is assigned. A clean native rebuild and a fresh live run passed.

The run observed loading, context preparation, prefill, generation and ready;
two standard chat requests returned `OK`, a standard generate request completed,
and unloading removed the runner from activity. Context remained 131072. The
observed weight-cache peak was exactly 6144 MiB, and tracked CUDA peaked at
6711565184 bytes, below its 12288 MiB quota. These are managed counters, not total
process or physical-device memory.

There were 199 successful polls and no HTTP poll errors. The observed p95 latency
was 16 ms, with one maximum of 3485 ms while other validation/build work also ran.
This is an observation, not a per-request latency guarantee. The client retains
its timeout and ordinary conversation feedback when telemetry is unavailable.

The cold request reached its first output after 65.344 s; the repeated short
request reached it after 0.782 s with 14 prompt tokens cached. Both generated
only two counted tokens, so these timings do not establish sustained generation
speed or a speedup over another build.

Clean and repeated application of patches 001–005 passed. The native server
component executable passed 42 tests / 157 assertions. WildGoose passed 96 Python
tests, MCP/web/backend smoke checks, TypeScript checking and 205 UI tests across
37 files. Its Windows package built successfully and the packaged Python included
the catalogue fix. Browser replay of the captured API snapshots passed for load,
prefill and generation at widths 1000 and 390, with no JavaScript errors or
horizontal overflow. Replay uses the actual UI components and a fixture bridge;
it is not a complete interactive desktop-agent conversation test.

The laboratory server was stopped. Production executables, model weights and
user profiles were not replaced. Linux, multidevice and longer conversations
remain outside this integration check.
