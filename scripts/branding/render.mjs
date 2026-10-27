// Render the native SVG mark. No image-generation service or runtime dependency.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const require = createRequire(import.meta.url);
const rendererIndex = process.argv.indexOf('--renderer-module');
const renderer = rendererIndex < 0 ? '@resvg/resvg-js' : process.argv[rendererIndex + 1];
if (!renderer) throw new Error('--renderer-module requires an installed module path');
const { Resvg } = require(renderer);
const source = fs.readFileSync(path.join(root, 'docs/assets/wildllama-mark.svg'), 'utf8');
const body = source.replace(/^.*?<desc[^>]*>.*?<\/desc>/s, '').replace(/<\/svg>\s*$/, '');
const glyphBody = body.replace(/<rect[^>]*\/>/, '').replace(/#[0-9a-f]{6}/gi, '#000000').replace(/[ \t]+$/gm, '');
const wrap = (content, name = 'WildLlama') => `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 256 256"><title>${name}</title>${content}</svg>\n`;
const glyph = wrap(glyphBody);
const upgrade = wrap(body + '<circle cx="206" cy="207" r="39" fill="#57d6b2" stroke="#122d32" stroke-width="8"/><path d="M206 226v-38m-14 14 14-14 14 14" fill="none" stroke="#122d32" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>', 'WildLlama update');
const cache = new Map();
function raster(svg, size, fonts = false) {
  const key = `${size}:${fonts}:${svg}`;
  if (!cache.has(key)) cache.set(key, new Resvg(svg, {
    fitTo: { mode: 'width', value: size }, font: { loadSystemFonts: fonts, sansSerifFamily: 'Arial' },
  }).render());
  return cache.get(key);
}
function save(relative, bytes) {
  const target = path.join(root, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, bytes);
}
function png(svg, size) { return raster(svg, size).asPng(); }
function ico(svg) {
  const sizes = [16, 24, 32, 48, 64, 128, 256];
  const images = sizes.map(size => png(svg, size));
  const header = Buffer.alloc(6 + 16 * sizes.length);
  header.writeUInt16LE(1, 2); header.writeUInt16LE(sizes.length, 4);
  let offset = header.length;
  images.forEach((data, i) => {
    const entry = 6 + 16 * i;
    header[entry] = sizes[i] === 256 ? 0 : sizes[i]; header[entry + 1] = header[entry];
    header.writeUInt16LE(1, entry + 4); header.writeUInt16LE(32, entry + 6);
    header.writeUInt32LE(data.length, entry + 8); header.writeUInt32LE(offset, entry + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...images]);
}
function icns(svg) {
  const chunks = [['ic07', 128], ['ic08', 256], ['ic09', 512], ['ic10', 1024]].map(([type, size]) => {
    const data = png(svg, size), header = Buffer.alloc(8);
    header.write(type); header.writeUInt32BE(data.length + 8, 4);
    return Buffer.concat([header, data]);
  });
  const header = Buffer.alloc(8);
  header.write('icns'); header.writeUInt32BE(8 + chunks.reduce((n, b) => n + b.length, 0), 4);
  return Buffer.concat([header, ...chunks]);
}
function setupBitmap() {
  const width = 138, height = 140, stride = (width * 3 + 3) & ~3;
  const bmp = Buffer.alloc(54 + stride * height, 255);
  bmp.fill(0, 0, 54); bmp.write('BM'); bmp.writeUInt32LE(bmp.length, 2); bmp.writeUInt32LE(54, 10);
  bmp.writeUInt32LE(40, 14); bmp.writeInt32LE(width, 18); bmp.writeInt32LE(height, 22);
  bmp.writeUInt16LE(1, 26); bmp.writeUInt16LE(24, 28); bmp.writeUInt32LE(stride * height, 34);
  const rgba = raster(source, 128).pixels;
  for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) {
    const from = (y * 128 + x) * 4, to = 54 + (height - 1 - y - 6) * stride + (x + 5) * 3;
    const alpha = rgba[from + 3] / 255;
    for (let c = 0; c < 3; c++) bmp[to + c] = Math.round(rgba[from + 2 - c] * alpha + 255 * (1 - alpha));
  }
  return bmp;
}
function wordmark(color) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="520" height="128" viewBox="0 0 520 128" role="img" aria-label="WildLlama"><title>WildLlama</title><g transform="scale(.5)">${body}</g><text x="150" y="87" fill="${color}" font-family="Arial, Helvetica, sans-serif" font-size="66" font-weight="700" letter-spacing="-2">WildLlama</text></svg>\n`;
}

save('docs/assets/wildllama-glyph.svg', glyph);
save('docs/assets/wildllama-mark.png', png(source, 512));
save('docs/assets/wildllama-icon-128.png', png(source, 128));
save('docs/assets/wildllama.ico', ico(source));
save('docs/assets/wildllama.icns', icns(source));
save('docs/assets/wildllama-wordmark.svg', wordmark('#122d32'));
save('docs/assets/wildllama-wordmark-dark.svg', wordmark('#fff9e9'));
save('docs/images/logo.png', raster(wordmark('#122d32'), 1040, true).asPng());
save('docs/images/logo-dark.png', raster(wordmark('#fff9e9'), 1040, true).asPng());
save('docs/images/favicon.png', png(source, 64));
for (const name of ['logo.svg', 'favicon.svg', 'favicon-dark.svg']) save(`docs/${name}`, source);
for (const [name, svg] of [['app', source], ['tray', glyph], ['tray_upgrade', upgrade]]) save(`app/assets/${name}.ico`, ico(svg));
save('app/assets/setup.bmp', setupBitmap());
const mac = 'app/darwin/Ollama.app/Contents/Resources';
save(`${mac}/icon.icns`, icns(source));
for (const [name, svg] of [['ollama', glyph], ['ollamaUpdate', wrap(glyphBody + '<circle cx="213" cy="205" r="34" fill="#000000"/>')]]) {
  for (const [suffix, size] of [['', 22], ['@2x', 44]]) {
    save(`${mac}/${name}${suffix}.png`, png(svg, size));
    save(`${mac}/${name}Dark${suffix}.png`, png(svg.replaceAll('#000000', '#ffffff'), size));
  }
}
save('app/ui/app/src/assets/wildllama-mark.svg', source);
save('app/ui/app/src/assets/wildllama-glyph.svg', glyph);
save('app/ui/app/public/wildllama.svg', source);
console.log('WildLlama SVG, PNG, ICO, ICNS, BMP and UI assets generated from docs/assets/wildllama-mark.svg.');
