/**
 * Build a self-contained app directory for `@electron/packager` to copy.
 *
 * Why this exists rather than pointing packager at `packages/desktop`:
 *
 * `prune: true` walks the dependency tree with flora-colossus starting from the
 * app dir, and Yarn hoists workspace dependencies to the REPO ROOT. So
 * `packages/desktop/node_modules` contains only `@types` and packager dies with
 * `Failed to locate module "@mlx-node/lm"`. Turning pruning off does not help —
 * it would then copy that same 2.8 MB of type packages and none of the runtime
 * dependencies, producing an app that packages cleanly and fails on first launch.
 * Copying the root tree instead is not an option either: it is 920 MB.
 *
 * So the closure is computed explicitly and copied. The result is deterministic,
 * needs no network, and ships exactly the versions that were tested — an
 * `npm install --omit=dev` into a staging dir would re-resolve against the
 * registry and could ship something nobody ran.
 */

import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

interface PackageJson {
  name?: string;
  version?: string;
  dependencies?: Record<string, string>;
  optionalDependencies?: Record<string, string>;
}

function readJson(path: string): PackageJson | null {
  try {
    return JSON.parse(readFileSync(path, 'utf-8')) as PackageJson;
  } catch {
    return null;
  }
}

/** Workspace packages resolve to `packages/<dir>`, everything else to the root tree. */
function workspaceDirFor(repoRoot: string, name: string): string | null {
  if (!name.startsWith('@mlx-node/')) return null;
  const dir = join(repoRoot, 'packages', name.slice('@mlx-node/'.length));
  return existsSync(join(dir, 'package.json')) ? dir : null;
}

/**
 * napi-rs publishes one prebuilt package per platform (`@mlx-node/core-darwin-arm64`,
 * `-linux-x64-gnu`, …) and declares them as optionalDependencies of `@mlx-node/core`.
 *
 * They must NOT be staged. The darwin one alone is 239 MB — the same `.node` and
 * both metallibs the bundle already ships at `Contents/Resources/native/` — so
 * including it shipped the entire native payload TWICE and took the app from
 * ~754 MB to 993 MB. Nothing loads it either: `packages/core/index.cjs` checks
 * `NAPI_RS_NATIVE_LIBRARY_PATH` first, and packaging points that at the staged
 * copy precisely so the addon lives in exactly one place.
 */
function isPrebuiltAddonPackage(name: string): boolean {
  return /^@mlx-node\/core-/.test(name);
}

/**
 * Transitive runtime closure of `roots`.
 *
 * devDependencies are excluded, which is what keeps Electron itself (a
 * devDependency, ~277 MB) from shipping a second copy inside the app.
 *
 * `optionalDependencies` are followed but **may legitimately be absent**, and the
 * distinction is load-bearing rather than pedantic. This tree reaches
 * `@mariozechner/clipboard`, which fans out to one prebuilt per platform —
 * `…-win32-x64-msvc`, `…-linux-x64-gnu` and so on. On macOS only the darwin one
 * is installed, so treating a missing optional as fatal fails the build on a
 * package that could never have been used. A missing HARD dependency stays fatal:
 * its symptom in a packaged app is a module-not-found on a code path nobody
 * exercised before shipping.
 */
export function runtimeClosure(
  repoRoot: string,
  roots: string[],
): { external: string[]; workspace: string[]; skippedOptional: string[]; excludedPrebuilt: string[] } {
  const external = new Set<string>();
  const workspace = new Set<string>();
  const skippedOptional: string[] = [];
  const excludedPrebuilt: string[] = [];
  const seen = new Set<string>();
  const queue: Array<{ name: string; optional: boolean }> = roots.map((name) => ({ name, optional: false }));

  while (queue.length > 0) {
    const { name, optional } = queue.pop() as { name: string; optional: boolean };
    if (seen.has(name)) continue;
    seen.add(name);

    if (isPrebuiltAddonPackage(name)) {
      excludedPrebuilt.push(name);
      continue;
    }

    const wsDir = workspaceDirFor(repoRoot, name);
    const pkgPath = wsDir === null ? join(repoRoot, 'node_modules', name, 'package.json') : join(wsDir, 'package.json');
    const pkg = readJson(pkgPath);
    if (pkg === null) {
      if (optional) {
        skippedOptional.push(name);
        continue;
      }
      throw new Error(`Cannot resolve "${name}" for the bundle (looked at ${pkgPath}). Run: vp install`);
    }
    if (wsDir === null) external.add(name);
    else workspace.add(name);

    for (const dep of Object.keys(pkg.dependencies ?? {})) queue.push({ name: dep, optional: false });
    for (const dep of Object.keys(pkg.optionalDependencies ?? {})) queue.push({ name: dep, optional: true });
  }
  return {
    external: [...external].sort(),
    workspace: [...workspace].sort(),
    skippedOptional: skippedOptional.sort(),
    excludedPrebuilt: excludedPrebuilt.sort(),
  };
}

/**
 * Directories inside a published package that are never imported at runtime.
 *
 * Pruning them is not a size optimisation that happens to help — it is required.
 * `@electron/osx-sign` walks the bundle and tries to sign anything that looks like
 * a binary, and `@earendil-works/pi-coding-agent` ships
 * `examples/extensions/doom-overlay/doom/build/doom.wasm`. A .wasm is not a Mach-O,
 * codesign fails on it, and the whole signing step dies on a DOOM build that has no
 * business being in an inference app in the first place.
 *
 * Deliberately conservative: only directories that are unambiguously not shipping
 * code. `src/` is NOT pruned — plenty of packages resolve their entry points into
 * it — and neither is anything that could be a runtime asset.
 */
const NON_RUNTIME_DIRS = new Set(['examples', 'example', '__tests__', '.github', 'docs']);

function pruneNonRuntime(dir: string): number {
  let removed = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (NON_RUNTIME_DIRS.has(entry.name)) {
      rmSync(join(dir, entry.name), { recursive: true, force: true });
      removed += 1;
      continue;
    }
    if (entry.name === 'node_modules') continue;
    removed += pruneNonRuntime(join(dir, entry.name));
  }
  return removed;
}

export interface StageResult {
  appDir: string;
  externalCount: number;
  workspaceCount: number;
  /** Platform-gated optionals that are not installed here — reported, never silent. */
  skippedOptional: string[];
  /** napi prebuilt packages deliberately left out; the payload ships once, in Resources/native. */
  excludedPrebuilt: string[];
  /** Count of examples/docs/test directories removed from staged packages. */
  prunedDirs: number;
}

/**
 * Materialise the staged app at `stageDir`.
 *
 * `roots` are the runtime entry dependencies — what the three entries actually
 * import, not what `packages/desktop/package.json` happens to declare.
 */
export function stageApp(opts: {
  repoRoot: string;
  desktopDir: string;
  stageDir: string;
  roots: string[];
}): StageResult {
  const { repoRoot, desktopDir, stageDir } = opts;
  rmSync(stageDir, { recursive: true, force: true });
  mkdirSync(stageDir, { recursive: true });

  cpSync(join(desktopDir, 'dist'), join(stageDir, 'dist'), { recursive: true });
  cpSync(join(desktopDir, 'build'), join(stageDir, 'build'), { recursive: true });

  const source = readJson(join(desktopDir, 'package.json'));
  // Dependencies are stripped from the staged manifest on purpose. node_modules
  // below is already the complete resolved closure, and leaving the field in
  // would invite packager (or a future reader) to try to re-resolve it in a tree
  // that has no root to hoist from.
  writeFileSync(
    join(stageDir, 'package.json'),
    `${JSON.stringify(
      {
        name: source?.name ?? '@mlx-node/desktop',
        productName: 'mlx-node',
        version: source?.version ?? '0.0.0',
        private: true,
        type: 'module',
        main: 'dist/main/index.js',
      },
      null,
      2,
    )}\n`,
  );

  const { external, workspace, skippedOptional, excludedPrebuilt } = runtimeClosure(repoRoot, opts.roots);
  const modules = join(stageDir, 'node_modules');

  for (const name of external) {
    // `dereference` matters: Yarn's tree is full of symlinks, and a symlink that
    // escapes the bundle is both a broken app and a codesign failure.
    cpSync(join(repoRoot, 'node_modules', name), join(modules, name), { recursive: true, dereference: true });
  }

  for (const name of workspace) {
    const from = workspaceDirFor(repoRoot, name);
    if (from === null) continue;
    const to = join(modules, name);
    mkdirSync(to, { recursive: true });
    cpSync(join(from, 'package.json'), join(to, 'package.json'));
    // Only what a consumer can actually import. Notably NOT the workspace's own
    // node_modules, which is where a second copy of a hoisted dependency (and a
    // second Electron) would sneak in.
    for (const dir of ['dist', 'web']) {
      if (existsSync(join(from, dir))) cpSync(join(from, dir), join(to, dir), { recursive: true, dereference: true });
    }
    for (const file of ['index.cjs', 'index.d.cts', 'index.js', 'index.d.ts']) {
      if (existsSync(join(from, file))) cpSync(join(from, file), join(to, file));
    }
  }

  const prunedDirs = pruneNonRuntime(modules);

  return {
    appDir: stageDir,
    externalCount: external.length,
    workspaceCount: workspace.length,
    skippedOptional,
    excludedPrebuilt,
    prunedDirs,
  };
}
