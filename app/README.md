# WildLlama desktop application sources

This directory contains the macOS and Windows desktop application inherited from
Ollama. For WildLlama's CLI/native runtime, source builds and isolated portable
payloads, start with the [project README](../README.md) and
[development documentation](../docs/development.md).

Official installers from ollama.com install upstream Ollama, not WildLlama.
The inherited desktop installer and automatic updater are not yet fully isolated
for this fork: they retain upstream application identifiers, storage/protocol
paths and update-service behavior. New icons do not make those packaging paths a
standalone WildLlama release. Do not distribute the inherited installer unchanged
or use it to update the isolated WildLlama runtime.

## Desktop development

Build the native runtime described in the project documentation first. Use a
separate application data directory, model metadata and listener when developing
alongside an existing Ollama installation.

The app sources use Go plus a React/Vite UI. From this `app` directory:

```sh
go generate ./...
go run ./cmd/app
```

### UI development

The binding generator requires
[`tscriptify`](https://github.com/tkrajina/typescriptify-golang-structs):

```sh
go install github.com/tkrajina/typescriptify-golang-structs/tscriptify@latest
```

Start the development UI from `app/ui/app`:

```sh
npm ci
npm run dev
```

In a separate terminal, from `app`, run the native shell:

```sh
go generate ./...
go run ./cmd/app -dev
```

Development mode loads the UI from `http://localhost:5173`, uses
`http://127.0.0.1:3001` for its UI API and enables development CORS behavior.

## Packaging status

The repository retains the upstream packaging sources in
[`ollama.iss`](ollama.iss),
[`build_windows.ps1`](../scripts/build_windows.ps1) and
[`build_darwin.sh`](../scripts/build_darwin.sh). They are development references,
not a claim that WildLlama provides signed desktop installers for every platform.

A standalone desktop release needs its own application/install identifiers,
update channel, signing configuration, protocol registration and safe storage and
uninstall behavior. Keep those changes coordinated with runtime compatibility
and coexistence testing. Report fork-specific desktop work in
[WildLlama issues](https://github.com/Anderseta/wildllama/issues).
