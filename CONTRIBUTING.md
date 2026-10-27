# Contributing to WildLlama

WildLlama is an independent fork of [Ollama](https://github.com/ollama/ollama),
focused on efficient local inference, bounded weight memory and large MoE models.
Send WildLlama issues and pull requests to
[Anderseta/wildllama](https://github.com/Anderseta/wildllama).

## Set up

Start with the [README](README.md) and
[development documentation](docs/development.md). Keep a development runtime,
listener and model metadata separate from an existing Ollama installation. The
Go module and compatible command/API names intentionally retain their upstream
names; a repository rename does not require renaming every import or model.

Development in this fork targets `main`. Open pull requests against `main` and
preserve the upstream history. Updating from upstream Ollama is separate from
integrating WildLlama changes.

## Report a problem or propose a change

Use [WildLlama issues](https://github.com/Anderseta/wildllama/issues) for bugs,
performance reports, model compatibility and feature proposals. Follow
[SECURITY.md](SECURITY.md) for vulnerabilities rather than posting exploit details
in a public issue.

For a reproducible inference problem, include:

- The WildLlama commit or build version, operating system, GPU/driver and backend.
- The model architecture, quantization, context, batch sizes and relevant options.
- Configured RAM/VRAM limits and whether coordinated MoE mode is enabled.
- The smallest prompt or fixture that reproduces the problem, with relevant logs.

Remove access tokens, private prompts and sensitive file paths from reports.
Large GGUF files and complete model stores do not belong in an issue or patch.
If a problem also occurs on unmodified Ollama, link the upstream report and
describe which runtime was tested.

Discuss substantial changes in an issue before investing in a large patch.
Explain the concrete problem, the resulting behavior, compatibility impact and
how a reviewer can validate it.

## Performance and memory changes

Preserve target weights, selected experts and sampling behavior unless a feature
explicitly changes them and documents the tradeoff. Memory claims must identify
their scope: a weight-cache quota is not a cap on the entire process, KV cache or
operating-system memory use.

Use small, reproducible fixtures first. Compare the old and new paths under the
same model, context, precision and memory limits. Record warm-up, cache state,
hardware and competing workloads. Distinguish component counters or timings from
end-to-end tokens/s; fewer uploads alone do not prove a proportional speedup.
Check numerical outputs and exercise failure/recovery paths when changing storage
or asynchronous execution.

The [runtime efficiency notes](docs/runtime-efficiency.md),
[MoE memory design](docs/moe-ram-limit.md) and
[validation artifacts](docs/benchmarks) describe the existing test approach.
Do not run large-model benchmarks on shared hardware without coordinating access.

## Pull requests

Keep changes focused and explain the problem and final behavior in the PR
description. Include the validation performed and any known platform limits.
Tests should cover meaningful behavior; documentation-only changes usually need
link, example and rendering checks rather than an inference benchmark.

Use a concise commit title, for example:

```text
llm: preserve terminal errors in streamed responses
docs: explain MoE weight-cache limits
```

Add dependencies sparingly and explain why an existing implementation is
insufficient. Preserve license notices and credit Ollama, llama.cpp and other
upstream projects when adapting their work. Public branding, installers and
update paths must clearly identify this fork and avoid replacing an independent
Ollama installation.
