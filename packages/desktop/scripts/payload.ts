/**
 * What has to be inside the `.app`, and where it comes from.
 *
 * Split out from `package.ts` so the part that can be wrong silently — a missing
 * or stale artifact — is a pure function with tests, rather than something you
 * discover after a 16-minute build and a notarization round trip.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import {
  assertAddonMinOs,
  assertMetallibIntegrity,
  assertMlxMetallibCarriesNax,
  assertPagedMetallibIntegrity,
} from '../../core/metallib-select.js';
import { PDFJS_WORKER, pdfjsBase } from '../../dashboard/ui/pdfjs-assets.js';

/**
 * The three native files, which MUST end up in ONE directory together.
 *
 * MLX loads `mlx.metallib` from beside the binary that loaded it, and
 * `paged_attn.metallib` is resolved `dladdr`-relative from the loaded `.node`.
 * Split them across directories and the addon loads fine and then produces
 * garbage or fails at first kernel dispatch — with no error naming a path.
 */
export const NATIVE_FILES = ['mlx-core.darwin-arm64.node', 'mlx.metallib', 'paged_attn.metallib'] as const;

/** A floor, not a checksum: catches a truncated or placeholder artifact. */
const MIN_BYTES: Record<string, number> = {
  'mlx-core.darwin-arm64.node': 40 * 1024 * 1024,
  'mlx.metallib': 100 * 1024 * 1024,
  'paged_attn.metallib': 10 * 1024 * 1024,
};

export interface PayloadSource {
  /** Absolute path per file in {@link NATIVE_FILES}. All in one source dir. */
  native: Record<string, string>;
  /** The built SPA (`packages/dashboard/web`), copied to `Contents/Resources/www`. */
  wwwRoot: string;
  /** `packages/desktop` — the app root packager copies. */
  appDir: string;
}

export class PayloadError extends Error {
  constructor(
    message: string,
    readonly remedy: string,
  ) {
    super(message);
    this.name = 'PayloadError';
  }
}

/**
 * Resolve and validate everything the bundle needs, or throw with the command
 * that produces the missing piece.
 *
 * Every one of these is gitignored and built out of band, so "absent" is the
 * normal state of a fresh checkout — not an exceptional one. The error has to say
 * what to run, or the next person guesses.
 */
export function resolvePayload(repoRoot: string): PayloadSource {
  const nativeDir = join(repoRoot, 'packages', 'core');
  const native: Record<string, string> = {};

  for (const file of NATIVE_FILES) {
    const path = join(nativeDir, file);
    if (!existsSync(path)) {
      throw new PayloadError(`Missing native artifact: ${path}`, 'yarn build:native');
    }
    const { size } = statSync(path);
    const floor = MIN_BYTES[file] ?? 0;
    if (size < floor) {
      throw new PayloadError(
        `${file} is ${size} bytes, under the ${floor}-byte floor — truncated or a placeholder`,
        'yarn build:native',
      );
    }
    native[file] = path;
  }
  checkAddonMinOs(native['mlx-core.darwin-arm64.node']!);
  checkMetallibPair(native['mlx.metallib']!, native['paged_attn.metallib']!);

  const wwwRoot = join(repoRoot, 'packages', 'dashboard', 'web');
  checkSpa(wwwRoot);

  const appDir = join(repoRoot, 'packages', 'desktop');
  if (!existsSync(join(appDir, 'dist', 'main', 'index.js'))) {
    throw new PayloadError(`Desktop main not built: ${join(appDir, 'dist', 'main', 'index.js')}`, 'yarn build:ts');
  }
  if (!existsSync(join(appDir, 'dist', 'preload', 'index.cjs'))) {
    // Its own check: `.cts` is easy to drop from tsconfig `include`, since
    // `src/**/*.ts` does NOT match `.cts`, and the symptom is a window that opens
    // with no bridge rather than a build error.
    throw new PayloadError(
      `Preload not built: ${join(appDir, 'dist', 'preload', 'index.cjs')} — is "src/**/*.cts" in tsconfig include?`,
      'yarn build:ts',
    );
  }

  return { native, wwwRoot, appDir };
}

/**
 * The built SPA holds `index.html` and the pdf.js worker. index.html specifically: the directory survives a failed
 * build, and an empty `www` produces a blank window with no error rather than a packaging failure. The worker is
 * copied after the bundle, not bundled, and without it every PDF the person adds fails to open.
 */
export function checkSpa(wwwRoot: string): void {
  for (const file of ['index.html', join(pdfjsBase(), PDFJS_WORKER)]) {
    if (!existsSync(join(wwwRoot, file))) {
      throw new PayloadError(
        `Dashboard SPA not built: ${join(wwwRoot, file)} is missing`,
        'yarn workspace @mlx-node/dashboard build:ui',
      );
    }
  }
}

/**
 * The addon's `LC_BUILD_VERSION` minos is at least the 26.2 floor, the gate
 * `yarn build:native` applies: the Info.plist floor is derived from it.
 */
export function checkAddonMinOs(addonPath: string): void {
  try {
    assertAddonMinOs(readFileSync(addonPath), addonPath);
  } catch (err) {
    throw new PayloadError((err as Error).message, 'yarn build:native');
  }
}

/**
 * Both metallibs pass the same gates `yarn build:native` applies
 * (packages/core/metallib-select.ts). `paged_attn.metallib` holds the K-quant
 * (NAX included), segmented SDPA and mixed-affine kernels, which have no JIT
 * fallback, so an older library clears any size floor and then throws on
 * first use. `mlx.metallib` must carry the kernel names of the CURRENT MLX
 * pin (`NAX_KERNEL_MARKERS`), not only some NAX kernels: a library left over
 * from the previous pin has the latter and would otherwise ship. Either file
 * shorter than its MTLB header declares fails. Size floors are applied per
 * file in `resolvePayload`.
 */
export function checkMetallibPair(mlxMetallibPath: string, pagedMetallibPath: string): void {
  try {
    const mlxMetallib = readFileSync(mlxMetallibPath);
    // First for its diagnosis (MLX_METAL_NO_NAX build vs. stale pin); the
    // integrity gate repeats it after the pin-marker check.
    assertMlxMetallibCarriesNax(mlxMetallib, mlxMetallibPath);
    assertMetallibIntegrity(mlxMetallib, { path: mlxMetallibPath, minBytes: 0 });
    assertPagedMetallibIntegrity(readFileSync(pagedMetallibPath), {
      path: pagedMetallibPath,
      minBytes: MIN_BYTES['paged_attn.metallib'],
    });
  } catch (err) {
    throw new PayloadError((err as Error).message, 'yarn build:native');
  }
}

/**
 * Where the addon lands inside the bundle. The app must hand this EXACT path to
 * `NAPI_RS_NATIVE_LIBRARY_PATH`: `packages/core/index.cjs` checks that variable
 * first in `requireNative()` and passes it straight to `require()`, so a
 * directory does not work — it has to name the file.
 */
export function bundledAddonPath(resourcesPath: string): string {
  return join(resourcesPath, 'native', 'mlx-core.darwin-arm64.node');
}

/**
 * Hardened-runtime signing arguments for one Mach-O.
 *
 * `--options runtime` is what the notary requires; `--timestamp` is what makes
 * the signature outlive the certificate. The addon additionally needs
 * `--identifier`, because it ships with `Identifier=libmlx_core.dylib` — a build
 * artifact name that would become this app's identity for that binary.
 */
export function codesignArgs(opts: {
  file: string;
  identity: string;
  entitlements?: string;
  identifier?: string;
}): string[] {
  const args = ['--sign', opts.identity, '--force', '--timestamp', '--options', 'runtime'];
  if (opts.identifier !== undefined) args.push('--identifier', opts.identifier);
  if (opts.entitlements !== undefined) args.push('--entitlements', opts.entitlements);
  args.push(opts.file);
  return args;
}

/** Required native addons and pdfjs-dist entry scripts inside a staged macOS app. */
export const DARWIN_APP_RUNTIME_FILES = [
  'node_modules/@napi-rs/canvas-darwin-arm64/skia.darwin-arm64.node',
  'node_modules/@napi-rs/keyring-darwin-arm64/keyring.darwin-arm64.node',
  'node_modules/pdfjs-dist/legacy/build/pdf.mjs',
  'node_modules/pdfjs-dist/legacy/build/pdf.worker.mjs',
] as const;

/** Required pdfjs-dist runtime asset directories inside a staged app. */
export const APP_PDFJS_RUNTIME_DIRS = [
  'node_modules/pdfjs-dist/wasm',
  'node_modules/pdfjs-dist/cmaps',
  'node_modules/pdfjs-dist/standard_fonts',
] as const;

function hasNonEmptyRegularFile(dir: string): boolean {
  const stack: string[] = [dir];
  for (let current = stack.pop(); current !== undefined; current = stack.pop()) {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const fullPath = join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile() && statSync(fullPath).size > 0) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Fail closed if the staged macOS app is missing native `.node` addons or
 * `pdfjs-dist` runtime assets needed by Keyring and Clairic PDF rendering.
 */
export function assertDarwinAppRuntimeAssets(appDir: string): void {
  const missing: string[] = [];
  for (const relPath of DARWIN_APP_RUNTIME_FILES) {
    const fullPath = join(appDir, relPath);
    if (!existsSync(fullPath)) {
      missing.push(relPath);
      continue;
    }
    const stat = statSync(fullPath);
    if (!stat.isFile() || stat.size === 0) {
      missing.push(`${relPath}${stat.isFile() ? ' (empty)' : ' (not a file)'}`);
    }
  }
  for (const relDir of APP_PDFJS_RUNTIME_DIRS) {
    const fullDir = join(appDir, relDir);
    if (!existsSync(fullDir) || !statSync(fullDir).isDirectory() || !hasNonEmptyRegularFile(fullDir)) {
      missing.push(relDir);
    }
  }
  if (missing.length > 0) {
    throw new PayloadError(
      `Staged macOS app is missing required runtime assets in ${appDir}:\n  ${missing.join('\n  ')}`,
      'vp install',
    );
  }
}

/**
 * Walk `nodeModulesDir` for regular `*.node` files, returning sorted paths and
 * skipping symlinks (both symlinked files and symlinked directories).
 */
export function findStagedNativeAddons(nodeModulesDir: string): string[] {
  if (!existsSync(nodeModulesDir)) return [];
  const addons: string[] = [];
  const stack: string[] = [nodeModulesDir];
  for (let dir = stack.pop(); dir !== undefined; dir = stack.pop()) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const fullPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.node')) {
        addons.push(fullPath);
      }
    }
  }
  return addons.sort();
}

/**
 * Extract every `LC_ID_DYLIB` install name from `otool -l` output.
 *
 * A universal Mach-O yields one entry per slice; an `MH_BUNDLE` or non-Mach-O
 * (ELF/PE32+) yields an empty array.
 */
export function parseDylibIds(otoolLoadCommands: string): string[] {
  const ids: string[] = [];
  let inIdDylib = false;
  for (const line of otoolLoadCommands.split(/\r?\n/)) {
    if (/^\s*Load command\b/.test(line)) {
      inIdDylib = false;
      continue;
    }
    const cmdMatch = /^\s*cmd\s+(\S+)\s*$/.exec(line);
    if (cmdMatch !== null) {
      inIdDylib = cmdMatch[1] === 'LC_ID_DYLIB';
      continue;
    }
    if (inIdDylib) {
      const nameMatch = /^\s*name (.+) \(offset\b/.exec(line);
      if (nameMatch !== null) {
        ids.push(nameMatch[1]);
        inIdDylib = false;
      }
    }
  }
  return ids;
}

/** Whether any `LC_ID_DYLIB` install name needs rewriting to `@rpath/<basename>`. */
export function needsDylibIdRewrite(dylibIds: readonly string[]): boolean {
  return dylibIds.some((id) => !id.startsWith('@rpath/') || id.includes('/target/'));
}
