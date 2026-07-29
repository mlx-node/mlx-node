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

export const RUNTIME_BUILD_FILES = ['iconTemplate.png', 'iconTemplate@2x.png'] as const;

/**
 * Only the tray images are opened at runtime. `icon.icns` is installed by
 * Electron Packager, while the generator and its ignored iconset are build
 * inputs that do not belong in `Contents/Resources/app`.
 */
export function stageRuntimeBuildFiles(desktopDir: string, stageDir: string): void {
  const target = join(stageDir, 'build');
  mkdirSync(target, { recursive: true });
  for (const name of RUNTIME_BUILD_FILES) {
    cpSync(join(desktopDir, 'build', name), join(target, name));
  }
}

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
 * `@mariozechner/clipboard` and its per-platform prebuilts, which arrive
 * transitively via `@earendil-works/pi-coding-agent` and must not ship.
 *
 * They fail the release gate outright. Both darwin prebuilts bake upstream's CI
 * home into their load commands —
 * `/Users/runner/work/clipboard/clipboard/target/…/libcrosscopy_clipboard.dylib`
 * — and step [3/5] of `verify-bundle.ts` refuses a bundle that ships a build
 * path. That is not our leak to fix: it is baked into the published tarball, so
 * the only lever on this side is not to carry it.
 *
 * Dropping it is safe for three independent reasons, and it needs all three:
 *  1. pi-coding-agent declares it under `optionalDependencies`, so absence is a
 *     supported state rather than a tree we happen to get away with breaking.
 *  2. `dist/utils/clipboard-native.js` loads it inside a `try {} catch {}` that
 *     falls through to `null` — a terminal-clipboard helper degrading in a
 *     process that has no terminal.
 *  3. Nothing in this repo references it. We import pi-coding-agent for session
 *     parsing (`SessionManager`, `parseSessionEntries`, `FileEntry`) only.
 *
 * It also ships TWICE for one architecture: `-darwin-arm64` (1.4 MB) alongside
 * `-darwin-universal` (2.9 MB), whose x86_64 slices this app can never execute.
 */
function isExcludedThirdPartyBinary(name: string): boolean {
  return /^@mariozechner\/clipboard(-|$)/.test(name);
}

/**
 * The cloud-LLM provider SDKs, ~114 MB of an app whose entire premise is that
 * inference happens locally. They are `dependencies` of `@earendil-works/pi-ai`,
 * which arrives under `@earendil-works/pi-coding-agent`. Our entire value-import
 * surface on that package is four session-file parsing symbols —
 * `SessionManager`, `parseSessionEntries`, `buildContextEntries`,
 * `migrateSessionEntries` — plus erased `type` imports. Nothing else.
 *
 * Only these SEVEN names are listed. Everything else that goes with them —
 * `zod`, `protobufjs`, `web-streams-polyfill`, `@opentelemetry/*`, the rest of
 * `@aws-sdk/*` and `@smithy/*`, 81 packages in all — disappears because the walk
 * below can no longer reach it, not because it was named. That distinction is
 * the safety property: the day something in the app legitimately needs `zod`,
 * `zod` comes back on its own.
 *
 * Unreachability is a property of pi-ai's design rather than an accident of our
 * import graph, which is why this is safe against a code path nobody exercised:
 *
 *  1. Every provider SDK is behind a `*.lazy.js` shim. `api/lazy.js` calls
 *     `load()` only from inside `stream()`/`streamSimple()` — the module is
 *     fetched when a request is issued to that provider, not when the api object
 *     is constructed. `bedrock-converse-stream.lazy.js` goes further and hides
 *     the specifier in a variable so no bundler can follow it either.
 *  2. So reaching any of them requires calling `stream()` on a pi-ai model. This
 *     app never does: nothing under `packages/desktop`, `packages/dashboard` or
 *     `packages/server` constructs an `AgentSession`, a `ModelRuntime`, or spawns
 *     the `pi` binary. `@mlx-node/agent` is in the graph only through
 *     `@mlx-node/agent/catalog`, a subpath whose built module has no imports at
 *     all.
 *  3. Measured, not reasoned: importing the whole `@mlx-node/dashboard` entry
 *     under a `module.registerHooks` resolve hook loads 2330 modules and not one
 *     file from any of these packages.
 *  4. `@opentelemetry/api` is a declared dependency that pi-ai never imports —
 *     the string does not appear in any `.js` it ships.
 *
 * The failure mode if this is ever wrong is loud rather than silent: an
 * ERR_MODULE_NOT_FOUND out of a `lazyApi` load, surfaced by `lazyStream` as an
 * error event on the very request that needed it.
 */
const CLOUD_PROVIDER_SDKS = new Set([
  '@anthropic-ai/sdk',
  '@aws-sdk/client-bedrock-runtime',
  '@google/genai',
  '@mistralai/mistralai',
  '@opentelemetry/api',
  '@smithy/node-http-handler',
  'openai',
]);

function isCloudProviderSdk(name: string): boolean {
  return CLOUD_PROVIDER_SDKS.has(name);
}

function isExcludedPackage(name: string): boolean {
  return isPrebuiltAddonPackage(name) || isExcludedThirdPartyBinary(name) || isCloudProviderSdk(name);
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
): {
  external: string[];
  workspace: string[];
  skippedOptional: string[];
  excludedPrebuilt: string[];
  excludedThirdParty: string[];
  excludedProviderSdk: string[];
} {
  const external = new Set<string>();
  const workspace = new Set<string>();
  const skippedOptional: string[] = [];
  const excludedPrebuilt: string[] = [];
  const excludedThirdParty: string[] = [];
  const excludedProviderSdk: string[] = [];
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

    if (isExcludedThirdPartyBinary(name)) {
      excludedThirdParty.push(name);
      continue;
    }

    if (isCloudProviderSdk(name)) {
      excludedProviderSdk.push(name);
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
    excludedThirdParty: excludedThirdParty.sort(),
    excludedProviderSdk: excludedProviderSdk.sort(),
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

/**
 * Files a Node runtime never opens: TypeScript declarations, source maps, and
 * `tsc -b`'s incremental state. 75 MB of the staged tree, and the single largest
 * line item in it — `drizzle-orm` ships 17.8 MB of `.d.ts` + `.map` against
 * 7.6 MB of executable code.
 *
 * Narrow on purpose, in both directions:
 *
 *  - `.d.ts`/`.d.cts`/`.d.mts` only, never plain `.ts`. Node 22+ executes `.ts`
 *    directly, so a package whose entry point resolves into `src/*.ts` would stop
 *    working; a declaration file has no runtime form to resolve to. Verified: no
 *    file in the staged tree imports a `.d.ts` specifier.
 *  - `.map` matched only behind a code extension. A stray `foo.map` could be a
 *    package's own data; `foo.js.map` cannot be anything but a source map.
 *    Verified: all 6571 `.map` files in the staged tree are `*.{js,cjs,mjs,ts,
 *    cts,mts,css}.map`.
 *
 * Nothing reads them here. Neither Electron nor this app enables source maps
 * (no `--enable-source-maps`, no `setSourceMapsEnabled`), and a `//#
 * sourceMappingURL` pointing at a file that is gone is ignored rather than
 * fatal even when they are on — the cost is a stack frame that names the emitted
 * line instead of the source one. `.tsbuildinfo` is worse than dead weight: it
 * is a build-layout record with no consumer inside a shipped app.
 */
const NON_RUNTIME_FILE = /(\.d\.[cm]?ts|\.(?:[cm]?[jt]s|css)\.map|\.tsbuildinfo)$/;

/** Exported so the rule itself can be pinned, not just its effect on one tree. */
export function isNonRuntimeFile(name: string): boolean {
  return NON_RUNTIME_FILE.test(name);
}

interface PruneCount {
  dirs: number;
  files: number;
}

/**
 * Recurses through nested `node_modules` as well. Yarn hoists most of the tree,
 * but not all of it: `@earendil-works/*` alone carries 20 MB of nested copies,
 * and skipping them left the biggest single duplicate — three copies of typebox
 * — untouched.
 */
function pruneNonRuntime(dir: string, count: PruneCount = { dirs: 0, files: 0 }): PruneCount {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (NON_RUNTIME_DIRS.has(entry.name)) {
        rmSync(join(dir, entry.name), { recursive: true, force: true });
        count.dirs += 1;
        continue;
      }
      pruneNonRuntime(join(dir, entry.name), count);
      continue;
    }
    if (entry.isFile() && NON_RUNTIME_FILE.test(entry.name)) {
      rmSync(join(dir, entry.name), { force: true });
      count.files += 1;
    }
  }
  return count;
}

/** Package names directly under a `node_modules` directory, `@scope/name` included. */
function packageNamesIn(nodeModules: string): string[] {
  const names: string[] = [];
  for (const entry of readdirSync(nodeModules, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    if (!entry.name.startsWith('@')) {
      names.push(entry.name);
      continue;
    }
    for (const scoped of readdirSync(join(nodeModules, entry.name), { withFileTypes: true })) {
      if (scoped.isDirectory()) names.push(`${entry.name}/${scoped.name}`);
    }
  }
  return names;
}

/**
 * Remove excluded packages that ride in on a nested `node_modules`.
 *
 * `runtimeClosure` decides by NAME while the copy above happens by DIRECTORY,
 * and the two disagree wherever Yarn could not hoist. `@earendil-works/pi-ai`
 * carries its own `node_modules/@smithy/node-http-handler`, so the exclusion
 * list refused the package and staged it anyway — a rule that reports success
 * and does nothing.
 *
 * Today that gap is 232 KB. The reason to close it is not the 232 KB: nesting is
 * a resolution outcome, so the same version bump that moves a provider SDK down
 * one level would put the whole 114 MB back with no diff to explain it.
 */
export function pruneExcludedNested(modules: string): string[] {
  const removed: string[] = [];
  const visit = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const child = join(dir, entry.name);
      if (entry.name === 'node_modules') {
        for (const name of packageNamesIn(child)) {
          if (!isExcludedPackage(name)) continue;
          rmSync(join(child, name), { recursive: true, force: true });
          removed.push(name);
        }
      }
      visit(child);
    }
  };
  visit(modules);
  return removed.sort();
}

export interface StageResult {
  appDir: string;
  externalCount: number;
  workspaceCount: number;
  /** Platform-gated optionals that are not installed here — reported, never silent. */
  skippedOptional: string[];
  /** napi prebuilt packages deliberately left out; the payload ships once, in Resources/native. */
  excludedPrebuilt: string[];
  /** Third-party binaries dropped because they leak a build path; see isExcludedThirdPartyBinary. */
  excludedThirdParty: string[];
  /** Cloud-LLM SDKs dropped because nothing local-inference can reach them; see isCloudProviderSdk. */
  excludedProviderSdk: string[];
  /** Count of examples/docs/test directories removed from staged packages. */
  prunedDirs: number;
  /** Count of .d.ts / source-map / tsbuildinfo files removed from the staged tree. */
  prunedFiles: number;
  /** Excluded packages that arrived nested inside another package's node_modules. */
  prunedNested: string[];
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
  stageRuntimeBuildFiles(desktopDir, stageDir);

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

  const { external, workspace, skippedOptional, excludedPrebuilt, excludedThirdParty, excludedProviderSdk } =
    runtimeClosure(repoRoot, opts.roots);
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

  // The app's own `dist` is pruned too: `tsc -b` emits `.d.ts` + `.d.ts.map`
  // beside every entry, and none of it is reachable once the app is running.
  const pruned = pruneNonRuntime(modules);
  pruneNonRuntime(join(stageDir, 'dist'), pruned);
  const prunedNested = pruneExcludedNested(modules);

  return {
    appDir: stageDir,
    externalCount: external.length,
    workspaceCount: workspace.length,
    skippedOptional,
    excludedPrebuilt,
    excludedThirdParty,
    excludedProviderSdk,
    prunedDirs: pruned.dirs,
    prunedFiles: pruned.files,
    prunedNested,
  };
}
