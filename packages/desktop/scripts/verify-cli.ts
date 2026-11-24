import { execFileSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, win32 } from 'node:path';

import { createCliLauncher } from '../src/cli-launcher.js';

function hasDigits(value: string): boolean {
  if (value.length === 0) return false;
  for (const char of value) {
    if (char < '0' || char > '9') return false;
  }
  return true;
}

/** Accept the bare version and grok-branded banner emitted by supported agents. */
export function hasAgentVersionBanner(banner: string): boolean {
  const trimmed = banner.trim();
  const value = trimmed.startsWith('grok ') ? trimmed.slice('grok '.length) : trimmed;
  const separator = value.indexOf(' ');
  const version = separator === -1 ? value : value.slice(0, separator);
  const parts = version.split('.');
  const third = parts[2] ?? '';
  return (
    parts.length >= 3 &&
    hasDigits(parts[0]) &&
    hasDigits(parts[1]) &&
    third.length > 0 &&
    third[0] >= '0' &&
    third[0] <= '9'
  );
}

function findEnvCaseInsensitive(env: NodeJS.ProcessEnv, key: string): string | undefined {
  const upper = key.toUpperCase();
  for (const [k, value] of Object.entries(env)) {
    if (value !== undefined && k.toUpperCase() === upper) return value;
  }
  return undefined;
}

/**
 * Minimal, isolated environment for probing packaged native addons under Electron
 * running as Node (`ELECTRON_RUN_AS_NODE=1`). Explicitly omits
 * `NAPI_RS_NATIVE_LIBRARY_PATH` and `MLX_CORE_NATIVE_LIBRARY_PATH` so every
 * addon must resolve its own packaged binary rather than inheriting an override.
 */
export function packagedRuntimeProbeEnv(
  home: string,
  platform: NodeJS.Platform = process.platform,
  hostEnv: NodeJS.ProcessEnv = process.env,
): Record<string, string> {
  if (platform === 'win32') {
    const systemRoot =
      findEnvCaseInsensitive(hostEnv, 'SystemRoot') ?? findEnvCaseInsensitive(hostEnv, 'WINDIR') ?? 'C:\\Windows';
    const windir = findEnvCaseInsensitive(hostEnv, 'WINDIR') ?? systemRoot;
    return {
      ELECTRON_RUN_AS_NODE: '1',
      HOME: home,
      USERPROFILE: home,
      TEMP: home,
      TMP: home,
      APPDATA: win32.join(home, 'AppData', 'Roaming'),
      LOCALAPPDATA: win32.join(home, 'AppData', 'Local'),
      SystemRoot: systemRoot,
      WINDIR: windir,
      PATH: `${win32.join(systemRoot, 'System32')};${systemRoot}`,
    };
  }
  return {
    ELECTRON_RUN_AS_NODE: '1',
    HOME: home,
    PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
  };
}

/**
 * Build an inline ESM smoke script that verifies every packaged native addon
 * and PDF rendering runtime asset inside `appDir`:
 *   1. `@napi-rs/keyring` loads and instantiates `AsyncEntry`
 *   2. `@napi-rs/canvas` renders a 4x4 `#123456` canvas and reads back exact RGBA pixels
 *   3. `pdfjs-dist` legacy main + worker modules load and `wasm`/`cmaps`/`standard_fonts` are non-empty
 *   4. `@mlx-node/lm/dist/clairic-pdf.js` rasterizes a minimal 1-page `%PDF-1.4` to RGB24
 */
export function buildPackagedRuntimeProbeScript(appDir: string): string {
  const escapedAppDir = JSON.stringify(appDir);
  return `
import { readdirSync, realpathSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const appDir = realpathSync(resolve(${escapedAppDir}));
const appPrefix = appDir.endsWith(sep) ? appDir : appDir + sep;
const assertInsideApp = (label, filePath) => {
  const real = realpathSync(filePath);
  const inside =
    process.platform === 'win32'
      ? real.toLowerCase().startsWith(appPrefix.toLowerCase())
      : real.startsWith(appPrefix);
  if (!inside) {
    throw new Error(\`\${label} resolved outside packaged appDir (\${appDir}): \${real}\`);
  }
};
const req = createRequire(join(appDir, 'package.json'));

// 1. @napi-rs/keyring
assertInsideApp('@napi-rs/keyring', req.resolve('@napi-rs/keyring'));
const keyring = req('@napi-rs/keyring');
const entry = new keyring.AsyncEntry('mlx-package-smoke', 'unused');
if (typeof entry.getPassword !== 'function') {
  throw new Error('Keyring AsyncEntry.getPassword is not a function');
}

// 2. @napi-rs/canvas
assertInsideApp('@napi-rs/canvas', req.resolve('@napi-rs/canvas'));
const canvasMod = req('@napi-rs/canvas');
const canvas = canvasMod.createCanvas(4, 4);
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#123456';
ctx.fillRect(0, 0, 4, 4);
const pixels = ctx.getImageData(0, 0, 4, 4).data;
if (
  pixels.length !== 64 ||
  pixels[0] !== 18 ||
  pixels[1] !== 52 ||
  pixels[2] !== 86 ||
  pixels[3] !== 255
) {
  throw new Error(
    \`Canvas pixel check failed: length=\${pixels.length} pixel0=[\${pixels[0]},\${pixels[1]},\${pixels[2]},\${pixels[3]}]\`,
  );
}

// 3. pdfjs-dist legacy modules and runtime asset directories
const pdfjsPkg = req.resolve('pdfjs-dist/package.json');
assertInsideApp('pdfjs-dist', pdfjsPkg);
const pdfjsRoot = dirname(pdfjsPkg);
const pdfjs = await import(pathToFileURL(join(pdfjsRoot, 'legacy', 'build', 'pdf.mjs')).href);
const worker = await import(pathToFileURL(join(pdfjsRoot, 'legacy', 'build', 'pdf.worker.mjs')).href);
if (typeof pdfjs.getDocument !== 'function') {
  throw new Error('pdfjs-dist getDocument is not a function');
}
if (typeof worker.WorkerMessageHandler === 'undefined') {
  throw new Error('pdfjs-dist worker did not export WorkerMessageHandler');
}
function hasNonEmptyFile(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (hasNonEmptyFile(full)) return true;
    } else if (entry.isFile() && statSync(full).size > 0) {
      return true;
    }
  }
  return false;
}
for (const subdir of ['wasm', 'cmaps', 'standard_fonts']) {
  const dir = join(pdfjsRoot, subdir);
  if (!hasNonEmptyFile(dir)) {
    throw new Error(\`pdfjs-dist asset directory has no non-empty files: \${dir}\`);
  }
}

// 4. @mlx-node/lm clairic-pdf end-to-end PDF rasterization
const clairicPdfPath = join(appDir, 'node_modules', '@mlx-node', 'lm', 'dist', 'clairic-pdf.js');
assertInsideApp('clairic-pdf', clairicPdfPath);
const { renderClairicDocuments } = await import(pathToFileURL(clairicPdfPath).href);

// Verify every CJS module and .node native addon loaded by createRequire came
// from inside the packaged appDir, never an ancestor checkout's node_modules.
for (const loaded of Object.keys(req.cache)) {
  assertInsideApp('loaded CJS module', loaded);
}

const content = '0 0 0.8 rg\\n10 10 40 20 re f\\n0 0 0 rg\\nBT /F1 12 Tf 10 40 Td (OK) Tj ET';
const objects = [
  '<< /Type /Catalog /Pages 2 0 R >>',
  '<< /Type /Pages /Kids [4 0 R] /Count 1 >>',
  '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
  '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 60] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>',
  \`<< /Length \${Buffer.byteLength(content, 'ascii')} >>\\nstream\\n\${content}\\nendstream\`,
];
let pdf = '%PDF-1.4\\n';
const offsets = [0];
for (let i = 0; i < objects.length; i++) {
  offsets.push(Buffer.byteLength(pdf, 'ascii'));
  pdf += \`\${i + 1} 0 obj\\n\${objects[i]}\\nendobj\\n\`;
}
const xrefOffset = Buffer.byteLength(pdf, 'ascii');
pdf += \`xref\\n0 \${objects.length + 1}\\n0000000000 65535 f \\n\`;
for (let i = 1; i <= objects.length; i++) {
  pdf += \`\${String(offsets[i]).padStart(10, '0')} 00000 n \\n\`;
}
pdf += \`trailer\\n<< /Size \${objects.length + 1} /Root 1 0 R >>\\nstartxref\\n\${xrefOffset}\\n%%EOF\\n\`;
const pdfDataUri = \`data:application/pdf;base64,\${Buffer.from(pdf, 'ascii').toString('base64')}\`;

const pages = await renderClairicDocuments({ documents: [pdfDataUri] });
if (
  !Array.isArray(pages) ||
  pages.length !== 1 ||
  pages[0].width <= 0 ||
  pages[0].height <= 0 ||
  pages[0].rgb.length !== pages[0].width * pages[0].height * 3
) {
  throw new Error('renderClairicDocuments did not return a valid rendered RGB page');
}
process.stdout.write(\`ok:\${pages[0].width}x\${pages[0].height}\\n\`);
`;
}

export interface PackagedNativeRuntimeOptions {
  executable: string;
  appDir: string;
  home: string;
  platform?: NodeJS.Platform;
}

/** Run the native runtime smoke probe inside the packaged Electron binary as Node. */
export function verifyPackagedNativeRuntime(options: PackagedNativeRuntimeOptions): void {
  const platform = options.platform ?? process.platform;
  const script = buildPackagedRuntimeProbeScript(options.appDir);
  try {
    const output = execFileSync(options.executable, ['--input-type=module', '-e', script], {
      env: packagedRuntimeProbeEnv(options.home, platform),
      encoding: 'utf8',
      timeout: 60_000,
      maxBuffer: 1024 * 1024,
    }).trim();
    if (!output.startsWith('ok:')) {
      throw new Error(`Packaged native runtime probe returned unexpected output: ${output}`);
    }
  } catch (error) {
    const stderr =
      typeof error === 'object' && error !== null && 'stderr' in error
        ? String((error as { stderr?: string | Buffer }).stderr ?? '').trim()
        : '';
    if (stderr !== '') {
      throw new Error(
        `Packaged native runtime probe failed: ${error instanceof Error ? error.message : String(error)}\n${stderr}`,
        { cause: error },
      );
    }
    throw error;
  }
}

/** Exercise the packaged CLI through the same launcher used by coding agents. */
export async function verifyBundledCli(appPath: string): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'mlx-cli-package-check-'));
  const resources = join(appPath, 'Contents', 'Resources');
  try {
    const executable = join(appPath, 'Contents', 'MacOS', 'mlx-node');
    const launcher = createCliLauncher(
      {
        executable,
        entry: join(resources, 'app', 'node_modules', '@mlx-node', 'cli', 'dist', 'cli.js'),
        nativeAddon: join(resources, 'native', 'mlx-core.darwin-arm64.node'),
        modelsDir: join(home, 'models'),
      },
      home,
    );
    const path = await launcher.prepare();
    // --version enters the full agent runtime before exiting, so missing lazy
    // dependencies and native runtime incompatibilities cannot hide behind help.
    const version = execFileSync(path, ['agent', '--version'], {
      env: { PATH: '/usr/bin:/bin:/usr/sbin:/sbin', HOME: home, GROK_HOME: join(home, 'agent') },
      encoding: 'utf8',
      timeout: 30_000,
      maxBuffer: 256 * 1024,
    }).trim();
    if (!hasAgentVersionBanner(version)) throw new Error(`The packaged agent did not report its version: ${version}`);
    // These enter the download modules and instantiate Keyring without reading
    // credentials or downloading anything. delegate --help never imports it.
    for (const args of [['download'], ['download', 'model'], ['download', 'dataset']]) {
      const help = execFileSync(path, [...args, '--help'], {
        env: { PATH: '/usr/bin:/bin:/usr/sbin:/sbin', HOME: home },
        encoding: 'utf8',
        timeout: 30_000,
        maxBuffer: 256 * 1024,
      });
      if (!help.includes('Usage:')) throw new Error(`The packaged ${args.join(' ')} did not report its usage`);
    }
    verifyPackagedNativeRuntime({
      executable,
      appDir: join(resources, 'app'),
      home,
      platform: 'darwin',
    });
    console.log(
      `packaged CLI and native runtime verified without global Node/mlx (agent ${version}, Keyring, Canvas, pdfjs-dist, Clairic PDF)`,
    );
  } finally {
    await rm(home, { recursive: true, force: true });
  }
}
