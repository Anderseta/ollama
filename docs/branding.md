# WildLlama identity

The canonical mark is [assets/wildllama-mark.svg](assets/wildllama-mark.svg):
an ivory llama in motion on a rounded deep teal tile. Mint motion accents and
a small amber ear accent connect it visually to
[WildGoose](https://github.com/Anderseta/wildgoose).

| Color | Hex |
| --- | --- |
| Deep teal | `#122d32` |
| Mint | `#57d6b2` |
| Ivory | `#fff9e9` |
| Pale mint | `#e6f5ee` |
| Amber | `#f5a64a` |

The mark is native SVG artwork. PNGs, Windows ICOs, macOS ICNS files, tray
variants, documentation wordmarks and the installer bitmap are derived exports.
For GitHub's **Settings → General → Social preview**, upload
[wildllama-social-preview.png](assets/wildllama-social-preview.png), a 1280 × 640
PNG with a solid background. Committing this file does not automatically change
the repository setting; the image must also be uploaded in GitHub's interface.
Keep edits in the canonical SVG and regenerate the exports together:

```sh
npm install --prefix scripts/branding --ignore-scripts --cache .cache/npm
node scripts/branding/render.mjs
```

The renderer uses the pinned `@resvg/resvg-js` 2.6.2 package. To reuse an already
installed copy without downloading dependencies:

```sh
node scripts/branding/render.mjs --renderer-module /absolute/path/to/@resvg/resvg-js
```

Icon artwork has no font dependency. Wordmark PNGs use the available Arial or
sans-serif fallback; the SVG wordmarks retain selectable text. Generated artwork
is committed, so building the runtime does not require the branding renderer.

Some inherited resource paths still contain `Ollama`, matching the existing
build scripts. Updating these images does not establish a separate desktop
installer identity, protocol registration or updater. See the
[desktop packaging status](../app/README.md) before using those build paths.
The CLI and API keep upstream-compatible names; official registry URLs and
upstream attribution are intentional. `docs/ollama-logo.svg` remains the
upstream reference asset rather than being relabeled as WildLlama.

The new artwork is distributed under the repository's [MIT license](../LICENSE).
