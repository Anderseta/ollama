# Native GPU fixture validation

Run this suite after the isolated CUDA build, with no other laboratory model
using the GPU:

```powershell
./scripts/gpu-first/test-gpu-fixtures.ps1 -FixtureRoot 'G:\Goose\ollama-moe-lab\fixtures'
```

`-BuildRoot`, `-OutputRoot`, `-PrecisionFixture`, and `-ReportRoot` select other
isolated directories. Temporary files and the CUDA compilation cache use the
build directory. The script does not download models, install software, start
Ollama, modify model files, or contact a production listener.

The executable compares the same model, fixed input tokens and requested
context in resident CUDA and bounded streaming CUDA modes. Every evaluation
starts with a two-token prefix requesting zero output logits, then a 32/64-token
prefill and eight consecutive decode steps. The prefix covers the zero-output
last-layer branch that previously exposed an empty-tile join. Context length is
checked after execution. A successful strict run must report zero CPU heavy
operations, positive CUDA heavy operations, and MoE cache/transfer activity.

The 19 streaming cases cover qwen3moe, llama/Mixtral, qwen2moe shared experts,
olmoe, gpt-oss expert biases/MXFP4, gemma4 fused experts, and nemotron_h_moe
hybrid layers. The supplied gemma4 fixture has per-layer input width zero and
does not test the per-layer embedding gather. They include tied weights, split
GGUF, 32 KiB/64 KiB expert caches, F32/Q4_K_M/MXFP4 and activation-precision
metadata. These are tiny random-weight fixtures for numerical and execution
checks; they cannot establish language-model quality.

The strict numerical threshold remains
`abs(actual-reference)/max(1,abs(reference)) <= 2e-4` for every logit. Finite
differences outside it retain `FAIL_NUMERICAL` and exit code 1 in the report;
the script continues to collect other architectures. Runtime failures and NaN/Inf
stop immediately. The final script exit code remains 1 if any streaming case
failed. The report includes maximum absolute/scaled error, RMSE, normalized
mean-square error `sum(error^2)/sum(reference^2)`, and matching top-1 counts.

For each distinct failing fixture/context/prefill combination the script also
runs resident CUDA with ubatch 1, comparing it to the original resident ubatch
32/64. These `resident-batch-variation` entries are separate diagnostics of the
existing kernel/activation-precision dispatch, not replacements for the original
reference or permission to change a failed streaming verdict. They can be
omitted with `-SkipResidentDiagnostics`.

`-CasePattern` selects fixture basenames by regular expression, allowing a
separate explicitly selected subset to run after investigating an unsupported
shape. `-CacheBytes` further selects the listed cache capacities. These filters
do not suppress a failure within the selected subset. `-ResidentOnly`
runs only the distinct resident-batch diagnostics for the selected cases and
also retains a nonzero exit code for a numerical mismatch. Keep subset results
beside the original stopped report instead of overwriting it.

Each report directory has `results.json`, all stdout/stderr logs,
`fixtures.json` with SHA-256/size for every input including GGUF shards, and
`binaries.json` identifying the executable and native DLLs. Earlier failed runs
must be retained alongside corrected runs.

## Fixture provenance

The laboratory set predates this branch and was created locally for the bounded
RAM-cache audit. No external trained model is needed. The tracked generator
`scripts/make-moe-ram-fixtures.py` creates deterministic F32 llama, tied/wide
llama, qwen2moe, olmoe and gpt-oss models (NumPy RNG seed 123456) using the pinned
llama.cpp `gguf-py` schema:

```powershell
python ./scripts/make-moe-ram-fixtures.py --native-source ./.build/gpu-first-moe/native --output <fixture-directory>
```

Quantized variants were made using the matching `llama-quantize` binary. For
example:

```powershell
./.build/gpu-first-moe/out/bin/llama-quantize.exe <fixture-directory>/llama-tied-wide.gguf <fixture-directory>/llama-tied-wide-Q4_K_M.gguf Q4_K_M
./.build/gpu-first-moe/out/bin/llama-quantize.exe <fixture-directory>/gpt-oss-wide.gguf <fixture-directory>/gpt-oss-wide-MXFP4.gguf MXFP4
```

The previous qwen3moe, gemma4 and nemotron_h_moe fixtures and their split/quantized
variants must currently be supplied in `FixtureRoot`; their complete original
generators are not tracked here. The report manifest records the exact inputs
used rather than claiming that the base generator reproduces those three
architectures. The matrix fails explicitly if any required fixture is missing.

The additional precision-policy fixture is reproducible from the supplied tiny
MXFP4 fixture. This copies metadata and every tensor without requantization,
adds `general.tensor_extra.prec_a4=false` for expert weights, and verifies all
tensor names/types/raw bytes against the source:

```powershell
python ./scripts/gpu-first/make-precision-fixture.py --native-source ./.build/gpu-first-moe/native --source <fixture-directory>/gpt-oss-wide-MXFP4.gguf --output ./.build/gpu-first-moe/fixtures/gpt-oss-wide-MXFP4-prec-q8.gguf
```

NumPy and the existing Python environment are prerequisites; these scripts do
not install them. The precision generator redirects Python cache files beside
its output. Use `PYTHONPYCACHEPREFIX` in the build directory when running the
base generator to keep generated Python cache files there too.

An additional supported-shape Nemotron architecture fixture can be generated
without changing the original state16 negative fixture:

```powershell
python ./scripts/gpu-first/make-nemotron-cuda-fixture.py --native-source ./.build/gpu-first-moe/native --source <fixture-directory>/nemotron_h_moe-q4.gguf --output ./.build/gpu-first-moe/fixtures/nemotron_h_moe-cuda-state128.gguf
./scripts/gpu-first/test-gpu-fixtures.ps1 -FixtureRoot <fixture-directory> -CudaNemotronFixture ./.build/gpu-first-moe/fixtures/nemotron_h_moe-cuda-state128.gguf -CasePattern 'cuda-state128'
```

This generator accepts only the tiny synthetic fixture, never a trained model.
It changes `ssm.state_size` from 16 to 128 and generates F32 random values with
NumPy seed 20261025 for six resized SSM tensors in recurrent blocks 0/4: the
convolution width grows from 192 to 640, and the input projection width from
324 to 772. The remaining 63 tensors, including all eight expert tensors,
retain their exact types, shapes and raw bytes. The generator emits a JSON
manifest with those checks and source/output hashes. These two optional cases
are additional architecture tests; they do not replace or repair the original
state16 unsupported-model verdict. No production model or context is altered.

## Observed results, 2026-10-25

On the RTX 5060 Ti, the final native CUDA DLL
`28BD9039CA4A50106CE03A0D86859018A7D945B7F3A593BC41BF2D24DE0B61ED`
produced these strict end-to-end results. Each finite comparison contains 2,331
logits (nine output steps), includes the zero-output prefix, and preserved the
requested context. The exact threshold above was not changed.

| Fixture | Streaming cases | Largest absolute logit difference | Largest separately measured resident-batch difference |
| --- | --- | --- | --- |
| qwen3moe Q4 | 2 PASS | 0.0000506653 | Not needed |
| qwen3moe split F32 | 1 PASS | 0.0000000135042 | Not needed |
| qwen2moe F32/shared experts | 2 PASS | 0.000161640 | Not needed |
| gemma4 Q4/fused experts | 2 PASS | 0.000137573 | Not needed |
| olmoe F32 | 1 PASS, 1 numerical FAIL | 0.000306040 | 0.000518769 |
| llama tied/wide Q4_K_M | 2 numerical FAIL | 0.0163170 | 0.0268141 |
| llama tied/wide F32 | 2 numerical FAIL | 0.000648521 | 0.000689186 |
| gpt-oss wide MXFP4 | 2 numerical FAIL | 0.223459 | 0.222882 |
| gpt-oss wide MXFP4/precision metadata | 2 numerical FAIL | 0.0135562 | 0.207997 |
| nemotron_h_moe Q4/state16 | 2 unsupported-operator FAIL | No logits | Not run |
| Separate synthetic nemotron_h_moe/state128 | 2 additional PASS | 0.0000267355 | Not needed |

All 17 cases reaching logits had finite outputs, matching top-1 in all nine
steps, zero submitted CPU heavy operations and positive CUDA/cache activity.
This is **8 passing cases, 9 finite numerical failures, and 2 unsupported
cases**, not a universally passing model-quality gate. All seven separate
resident-batch diagnostics also exceeded the unchanged threshold; they establish
pre-existing batch/kernel variation but do not turn streaming failures into
passes. Local quantized operator identity and F32 format-envelope tests are
documented separately in `docs/gpu-first-memory-validation.md`.

The separate state128 synthetic fixture then passed both additional cases:
64 KiB cache/context256/prefill32 had maximum error 0.0000267355 and NMSE
0.00000045279; 1 MiB cache/context512/prefill64 was bitwise identical for all
2,331 logits. Both had zero CPU heavy operations and top-1 agreement 9/9.
Thus the combined set has 21 cases: 10 PASS, 9 finite numerical FAIL and the
two original unsupported state16 cases. The synthetic fixture's SHA-256 is
`58e1f2435d994f4b391c7eee338c7d27897a021a241cca6571920ebed36927ce`;
`gpu-fixtures-nemotron-state128` contains its separate manifests and reports.

The original Nemotron fixture first exposed the CUDA convolution requirement
that channels be a multiple of 128. A separate bounds-checked F32 CUDA path now
supports ragged channels, preserving the optimized aligned path. Its scalar
oracle passed 864 combinations, 2,488,320 outputs, with zero violations, including
127/128/129/192 channels, bias/SiLU and padded sequence/weight strides. The full
fixture then reached the next independent limitation: its Mamba2 `d_state=16`
is unsupported by the native CUDA `SSM_SCAN` implementation, which accepts
96/128/256. Exclusive mode rejects it explicitly instead of using CPU scan.
No production-model dimensions were changed to disguise this restriction.

Final raw artifacts are under `.build/gpu-first-moe/reports`:
`gpu-fixtures-final-summary.json` combines all 19 cases and seven resident
diagnostics; `gpu-fixtures-final-ssm` contains the finite comparisons, hashes and
logs; the original Nemotron failures are retained in the explicitly named logs
and subset directory. `scheduler-final-ssm.log` records passing fail-closed
checks for CPU arithmetic, a CPU-backed output view, and explicit unsupported
Flash Attention. `ssm-scalar-regression-20261025.log/.json` records the convolution
oracle. Earlier failures before the row-bound and convolution fixes remain
available as separate evidence.

The final host diagnostic callback also passed an integration test on the
qwen3moe Q4 fixture at both 1 MiB and 32 KiB expert-cache capacities:

```powershell
./.build/gpu-first-moe/out/bin/test-gpu-exclusive.exe <fixture-directory>/qwen3moe-q4.gguf 1048576 256 32 diagnostics
./.build/gpu-first-moe/out/bin/test-gpu-exclusive.exe <fixture-directory>/qwen3moe-q4.gguf 32768 256 32 diagnostics
```

Unlike the ordinary nine-step comparison, this mode uses one 32-token prefill
with **every token requesting logits**, so no layer is pruned and the expected
histogram is exactly `top_k 2 * tokens 32 * layers 4 = 256`. Both capacities
reported exactly 256 selections, zero dropped selections, all four layer
profiles plus shared layer -1, and zero submitted CPU heavy operations. This
remained 256 despite the much larger number of expert tiles at 32 KiB.
Diagnostics off returned no layer/expert profiles or enabled CUDA timer.

An external evaluation callback requests **every graph node in both modes**.
That deliberately gives the two runs identical graph segmentation, while also
checking that diagnostic wrapping preserves the external callback. The 1 MiB
case delivered 401 asks and 401 reads in each mode; 32 KiB delivered 6,393 asks
and 6,393 reads in each mode. All 8,288 logits were finite and bitwise identical
between off/on in each case. This proves the toggle and callback composition
for these controlled inputs; it does not promise bitwise equivalence across
every architecture when diagnostics itself changes graph segmentation.

The corresponding final logs are
`.build/gpu-first-moe/reports/diagnostics-integration-1m.log` and
`diagnostics-integration-32k.log`. These integration tests use the final host
diagnostic build; the preceding broad fixture matrix predates that host-only
addition. No CUDA kernel, weight representation or quality threshold changed
in the diagnostic addition.
