/**
 * What has to be inside the `.app`, and where it comes from.
 *
 * Split out from `package.ts` so the part that can be wrong silently — a missing
 * or stale artifact — is a pure function with tests, rather than something you
 * discover after a 16-minute build and a notarization round trip.
 */

import { existsSync, statSync } from 'node:fs';
import { join } from 'node:path';

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

  const wwwRoot = join(repoRoot, 'packages', 'dashboard', 'web');
  // index.html specifically: the directory survives a failed build, and an empty
  // `www` produces a blank window with no error rather than a packaging failure.
  if (!existsSync(join(wwwRoot, 'index.html'))) {
    throw new PayloadError(
      `Dashboard SPA not built: ${join(wwwRoot, 'index.html')} is missing`,
      'yarn workspace @mlx-node/dashboard build:ui',
    );
  }

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
