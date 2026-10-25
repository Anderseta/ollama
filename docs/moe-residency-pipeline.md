# Experimental coordinated MoE runtime

This opt-in runtime preserves the GGUF bytes, router top-k, context capacity and
memory quotas. It changes storage admission, transfer scheduling and the way
prompt tokens share resident experts. It does not prune or requantize a model.

Enable the server default with `OLLAMA_MOE_RUNTIME=1` or `ollama serve
--moe-runtime`. A request can override it through `options.moe_runtime`, including
explicit `false`. Effective runtime mode participates in runner identity. The
launcher forces the resolved child-local `WILDLLAMA_MOE_RUNTIME` value, so an
inherited native environment cannot override an explicit API choice.

The main target is the existing single-device GPU-exclusive selective-weight
path. Normal runners retain their existing default. Keep experimental payloads,
model metadata and listeners separate from production during validation.

## Residency and transport

- The automatic cache planner first reserves the existing graph and fixed
  allocations, measures the CUDA quota and physical free memory, then optionally
  rebuilds the expert banks once during initialization. It retains a safety
  margin and never changes context or weights. An allocation failure can restore
  the requested cache instead of silently moving arithmetic to CPU.
- Frequency admission with aging, soft layer balance and transient capacity
  protects reused specialists from cyclic scans. The current route is always
  executable; transient requests can borrow persistent capacity. Failed uploads
  invalidate mappings and policy metadata together.
- RAM receives reference-counted residency hints from GPU cache entries. Under
  pressure, RAM can preferentially release pages with another resident GPU copy.
  This is a preference rather than mandatory exclusivity. The hint search is
  bounded to 64 candidates per queue, then falls back to that queue's normal
  SLRU victim. Probation is exhausted before protected pages are considered;
  hints cannot trigger a whole-cache scan for every eviction.
- A bounded RAM arena commits and locks slots gradually and reuses them after
  eviction. Managed tensor virtual addresses remain identifiers; streaming reads
  gather from arena slots. The conventional CPU path keeps its original direct
  virtual-address storage. Unsupported direct reads fail explicitly.
- Windows aligned asynchronous reads are consumed by completion. Ready source
  ranges can enter the pinned upload ring before all other reads finish.
- A dedicated CUDA upload stream fences earlier consumers and publishes a
  readiness event to compute. Two recorded source-slot events protect pinned
  buffer reuse. Cleanup drains outstanding I/O and transfers before freeing
  their buffers; a host-side read failure can abandon and recover a batch.
- Canonical route selections are reused across weight bindings in one graph
  evaluation. Immutable per-evaluation map snapshots keep asynchronous metadata
  copies alive until their consumers finish. Host preparation still keeps its
  required barrier; graph views without host evaluation can remain asynchronous.

Weight quotas continue to exclude metadata, I/O/pinned staging and application
heaps. The RAM arena's resident counter includes committed reusable slots, even
when some are temporarily free. CUDA workspace allocations remain charged to the
existing allocator quota. Neither quota is an operating-system-wide memory cap.

## Dynamic prompt windows

`WILDLLAMA_MOE_DYNAMIC_PREFILL=1` additionally enables the CUDA provider for
prompt calls with multiple tokens. It keeps the parent output shape fixed and
greedily forms maximal contiguous windows whose actual expert union fits every
bank used by that layer. IDs and token order are unchanged.

A dedicated child CUDA backend executes the original quantized matmul kernels
and strided copies into the parent's output. Bounded output/ID workspace is
reserved while constructing the graph. Detached aliases prevent a child graph
from traversing or re-executing the parent's operations. Parent CUDA capture is
disabled around the host provider, and child work completes before it returns.
CPU fallback is rejected. Different batch dimensions can select different CUDA
kernels, so numerical regressions remain required even with identical weights.
Submitted GPU-operation counters describe the parent scheduler graph; nested
provider matmuls/copies are not separately added to that counter. Weight-cache
traffic still passes through the shared accounting. Parent and child GPU counts
must not be compared one-for-one with the static graph's operation count.

The static path also recognizes that a bank able to hold every expert can accept
any token union. A larger bank alone does not imply all model experts are
simultaneously resident across layers.

## Comparison controls and tracing

The following child-local switches accept `0` to disable an individual component
while the coordinated runtime is enabled:

```text
WILDLLAMA_MOE_CACHE_AUTO
WILDLLAMA_MOE_ADMISSION
WILDLLAMA_MOE_COOPERATIVE_CACHE
WILDLLAMA_MOE_RAM_POOL
WILDLLAMA_MOE_UPLOAD_OVERLAP
WILDLLAMA_MOE_ROUTE_PLAN
```

Dynamic prompt windows require their separate explicit `1` setting. Trace output
is independently opt-in:

```text
WILDLLAMA_EXPERT_TRACE=<absolute path of a new JSONL file>
WILDLLAMA_EXPERT_TRACE_MAX_BYTES=<byte cap, default 64 MiB>
```

The bounded asynchronous trace records numeric expert IDs, layer/evaluation
identity, counts and byte costs. It contains no prompts, token IDs or file paths.
It never overwrites an existing trace. A footer distinguishes complete,
truncated and dropped observations. Forced process termination can leave a trace
incomplete; replay rejects that by default.

`scripts/gpu-first/replay-expert-cache.py` compares storage budgets and policies.
Its bundle-level model is an estimate of cache traffic, not an exact simulator of
every native per-layout tile bank and not a tokens-per-second prediction.

The isolated benchmark accepts `--moe-runtime`, `--moe-dynamic-prefill`,
`--fixed-expert-cache` and `--expert-trace-path`. Compare fixed-budget runs first,
then measure automatic budgeting separately. Record SSD bytes/output token,
H2D bytes/output token, byte hit rates, first-output latency, longer generation
throughput and quality checks. A repeated two-token answer does not establish
sustained generation performance.

Measured validation and known numerical failures are recorded in
[moe-residency-validation.md](moe-residency-validation.md). A compatible external
MTP draft can use the same target GGUF through the existing draft-model path.
Explicit `draft-mtp` and `draft-eagle3` selections preserve that configured path;
`none` and n-gram modes do not load the sidecar. An MTP selection alone cannot
add missing MTP weights. GPU-exclusive streaming of a target plus an external
draft still requires joint memory planning and sidecar validation; patch 006
does not claim that combination has passed the Flash tests.
