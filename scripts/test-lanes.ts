#!/usr/bin/env oxnode
/**
 * Classify every vitest file into the `addon` or `pure-ts` lane by tracing its
 * static import graph for a path to `@mlx-node/core` (the NAPI addon loader).
 *
 * `vite.config.ts` calls `classify()` at config load to build the two vitest
 * projects: `addon` (serial) and `pure-ts` (parallel). See the comment there
 * for why the two lanes exist (Metal must not be oversubscribed).
 *
 *   oxnode scripts/test-lanes.ts   # print both lists
 *
 * What counts as "reaches the addon":
 *   - a module whose specifier is `@mlx-node/core` (the only entry that loads
 *     `mlx-core.<platform>.node`), reached through any chain of static
 *     `import`/`export ... from`, `import()` with a literal specifier, or
 *     `require()` with a literal specifier;
 *   - `@mlx-node/*` specifiers resolve exactly like the `workspaceSource` vite
 *     plugin: through each package's `exports[...]['@mlx-node/source']`;
 *   - `import type` / `export type` edges are skipped (erased at runtime);
 *   - `vi.mock('@mlx-node/core', ...)` does NOT remove a file from the addon
 *     lane: the mock factory may still touch the real module and the file is
 *     cheap enough serialised. Conservative is correct here.
 *
 * Files that spawn a child process which loads the addon on its own (desktop
 * sidecar, `mlx` CLI e2e) are listed in `SERIAL_BY_HAND`; the
 * tracer cannot see across a process boundary.
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE_CONDITION = '@mlx-node/source';
const ADDON_PACKAGE = '@mlx-node/core';

/**
 * Test files kept in the serial lane by hand, for reasons the tracer cannot
 * see: a child process that loads the addon (the child owns the GPU while it
 * runs), or a pick-a-free-port-then-bind pattern that is only race-free when
 * no other test runs at the same time.
 */
export const SERIAL_BY_HAND: string[] = [
  // Spawns the built `mlx` CLI (`packages/cli/dist/cli.js agent …`), which
  // loads a real qwen3.5 checkpoint and runs inference.
  '__test__/cli/agent-e2e.test.ts',
  // `execFileSync(node, …)` probes that `require` the addon / import the built
  // server host to see whether `mlx-core` gets mapped. No inference, but the
  // addon is loaded in a child, so keep it off the parallel lane.
  'packages/desktop/__test__/child-entries.test.ts',
  // Forks the built inference entry (`packages/desktop/dist/inference/index.js`)
  // with `NAPI_RS_NATIVE_LIBRARY_PATH` set — the addon loads in the child.
  'packages/desktop/__test__/sidecar-e2e.test.ts',
  // Its "guards the guard" probe `createRequire`s packages/core/index.cjs in a
  // `node -e` child, which dlopens the addon.
  '__test__/server/host/addon-free-subpaths.test.ts',
  // `listen(0)` → close → bind the same port later. Two parallel workers can
  // interleave free/bind into EADDRINUSE; serial was what made it safe.
  '__test__/server/host/net.test.ts',
  '__test__/server/host/logger.test.ts',
];

// Same globs as `test.include` in vite.config.ts.
const TEST_DIRS = [
  { dir: '__test__', pattern: /\.(test|spec)\.ts$/ },
  { dir: 'examples', pattern: /\.(test|spec)\.ts$/ },
];

function walk(dir: string, out: string[], pattern: RegExp): void {
  for (const entry of readdirSync(dir)) {
    if (entry === 'node_modules' || entry === 'dist' || entry.startsWith('.')) continue;
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) walk(full, out, pattern);
    else if (pattern.test(entry)) out.push(full);
  }
}

export function listTestFiles(): string[] {
  const files: string[] = [];
  for (const { dir, pattern } of TEST_DIRS) {
    const full = join(ROOT, dir);
    if (existsSync(full)) walk(full, files, pattern);
  }
  const packagesRoot = join(ROOT, 'packages');
  for (const pkg of readdirSync(packagesRoot)) {
    const testDir = join(packagesRoot, pkg, '__test__');
    if (existsSync(testDir) && statSync(testDir).isDirectory()) walk(testDir, files, /\.(test|spec)\.ts$/);
  }
  return files.map((f) => relative(ROOT, f)).sort();
}

interface PkgExports {
  dir: string;
  exports: Record<string, string | Record<string, string>>;
}

function loadWorkspacePackages(): Map<string, PkgExports> {
  const packages = new Map<string, PkgExports>();
  const packagesRoot = join(ROOT, 'packages');
  for (const entry of readdirSync(packagesRoot)) {
    const manifestPath = join(packagesRoot, entry, 'package.json');
    if (!existsSync(manifestPath)) continue;
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8')) as {
      name?: string;
      exports?: Record<string, string | Record<string, string>>;
    };
    if (manifest.name === undefined || manifest.exports === undefined) continue;
    packages.set(manifest.name, { dir: dirname(manifestPath), exports: manifest.exports });
  }
  return packages;
}

const PACKAGES = loadWorkspacePackages();

/** Strip comments so commented-out imports do not count as edges. */
function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
}

/** Collect runtime import specifiers from one TS/JS source file. */
function collectSpecifiers(src: string): string[] {
  const code = stripComments(src);
  const specs = new Set<string>();
  // import x from '...' / import { a } from '...' / import '...' — skipping `import type`.
  for (const m of code.matchAll(/^\s*import\s+(?!type\s)(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/gm)) specs.add(m[1]);
  // export { a } from '...' / export * from '...' — skipping `export type`.
  for (const m of code.matchAll(/^\s*export\s+(?!type\s)[^'"]*?\sfrom\s*['"]([^'"]+)['"]/gm)) specs.add(m[1]);
  // import('...') / require('...') with a literal.
  for (const m of code.matchAll(/\b(?:import|require)\s*\(\s*['"]([^'"]+)['"]\s*\)/g)) specs.add(m[1]);
  return [...specs];
}

const EXTS = ['.ts', '.tsx', '.mts', '.cts', '.js', '.mjs', '.cjs'];

function resolveFile(base: string): string | null {
  if (existsSync(base) && statSync(base).isFile()) return base;
  for (const ext of EXTS) if (existsSync(base + ext)) return base + ext;
  // `./foo.js` written for ESM output but living as `foo.ts` in source.
  const stripped = base.replace(/\.(js|mjs|cjs)$/, '');
  if (stripped !== base) for (const ext of EXTS) if (existsSync(stripped + ext)) return stripped + ext;
  for (const ext of EXTS) if (existsSync(join(base, `index${ext}`))) return join(base, `index${ext}`);
  return null;
}

/** Resolve a specifier to an absolute file, `ADDON_PACKAGE` sentinel, or null (external). */
function resolveSpecifier(spec: string, fromFile: string): string | null {
  if (spec === ADDON_PACKAGE || spec.startsWith(`${ADDON_PACKAGE}/`)) return ADDON_PACKAGE;
  // The addon binary itself, its loader, or a platform package, reached by a
  // relative path instead of the package name. Pure TS helpers that live in
  // packages/core (metallib-select, publish-targets) are traced normally.
  if (spec.endsWith('.node')) return ADDON_PACKAGE;
  if (spec.startsWith('.') || spec.startsWith('/')) {
    const abs = resolve(dirname(fromFile), spec);
    const core = join(ROOT, 'packages/core');
    if (abs === join(core, 'index.cjs') || abs === join(core, 'index') || abs.startsWith(join(core, 'npm') + '/')) {
      return ADDON_PACKAGE;
    }
    return resolveFile(abs);
  }
  // Dashboard SPA alias (`resolve.alias['@/']` in vite.config.ts).
  if (spec.startsWith('@/')) return resolveFile(join(ROOT, 'packages/dashboard/ui/src', spec.slice(2)));
  if (spec.startsWith('@mlx-node/')) {
    const parts = spec.split('/');
    const pkg = PACKAGES.get(`${parts[0]}/${parts[1]}`);
    if (pkg === undefined) return null;
    const key = parts.length > 2 ? `./${parts.slice(2).join('/')}` : '.';
    const entry = pkg.exports[key];
    const target = typeof entry === 'string' ? entry : entry?.[SOURCE_CONDITION];
    if (target === undefined) return null;
    return resolveFile(resolve(pkg.dir, target));
  }
  return null; // node builtin or third-party dependency
}

const reachCache = new Map<string, boolean>();

/** True when `file` (absolute) can reach `@mlx-node/core` through runtime imports. */
export function reachesAddon(file: string, stack: Set<string> = new Set()): boolean {
  const cached = reachCache.get(file);
  if (cached !== undefined) return cached;
  if (stack.has(file)) return false; // cycle: resolved by whoever entered first
  stack.add(file);
  let hit = false;
  const src = readFileSync(file, 'utf-8');
  for (const spec of collectSpecifiers(src)) {
    const target = resolveSpecifier(spec, file);
    if (target === null) continue;
    if (target === ADDON_PACKAGE || reachesAddon(target, stack)) {
      hit = true;
      break;
    }
  }
  stack.delete(file);
  reachCache.set(file, hit);
  return hit;
}

export interface Lanes {
  addon: string[];
  pureTs: string[];
}

export function classify(): Lanes {
  const addon: string[] = [];
  const pureTs: string[] = [];
  for (const rel of listTestFiles()) {
    if (SERIAL_BY_HAND.includes(rel) || reachesAddon(join(ROOT, rel))) addon.push(rel);
    else pureTs.push(rel);
  }
  return { addon, pureTs };
}

function main(): void {
  const lanes = classify();
  console.log(`// addon lane (${lanes.addon.length} files)`);
  for (const f of lanes.addon) console.log(`  '${f}',`);
  console.log(`\n// pure-ts lane (${lanes.pureTs.length} files)`);
  for (const f of lanes.pureTs) console.log(`  ${f}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
