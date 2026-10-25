# AGENTS.md

## WildLlama workflow

Work directly on `main` in this fork. Keep completed changes on `main`; create
another branch only when the user explicitly requests it. Preserve the existing
Git history when integrating earlier feature branches.

`main` tracks the user's fork at `origin/main`. Use `origin` when pushing.
`upstream` points to the original Ollama repository and remains a reference;
updating from upstream is separate from integrating our own changes.

## Building

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.
