#!/usr/bin/env oxnode
/**
 * Classify every vitest file into the `addon`, `darwin-ts`, or `pure-ts` lane
 * by tracing its AST import graph for a path to `@mlx-node/core` (the NAPI
 * addon loader).
 *
 * `vite.config.ts` calls `classify()` at config load to build the three vitest
 * projects: `pure-ts` (parallel, OS-agnostic — runnable on Linux CI),
 * `darwin-ts` (parallel, macOS-specific pure-TS tests), and `addon` (serial,
 * Metal must not be oversubscribed).
 *
 *   oxnode scripts/test-lanes.ts   # print all three lists
 *
 * What counts as "reaches the addon":
 *   - a module whose specifier is `@mlx-node/core` (the only entry that loads
 *     `mlx-core.<platform>.node`), reached through any chain of static
 *     `import`/`export ... from`, `import()` with a literal specifier, or
 *     `require()` with a literal specifier (parsed via `oxc-parser`, so string
 *     literals in test fixtures are never misread as imports);
 *   - `@mlx-node/*` specifiers resolve exactly like the `workspaceSource` vite
 *     plugin: through each package's `exports[...]['@mlx-node/source']`;
 *   - `import type` / `export type` edges are skipped (erased at runtime);
 *   - a top-level `vi.mock(spec, () => ...)` whose factory takes no parameters
 *     and never calls `importActual`/`importOriginal` replaces that resolved
 *     target for ESM edges (`import`/`export ... from`/`import()`) in that test
 *     file's module graph, pruning edges to the mocked module. Partial mocks
 *     (`importOriginal` / `importActual`) keep the real edge, and CommonJS
 *     `require()` / `createRequire()` edges are never pruned because they go
 *     through Node's CJS loader and bypass Vitest's ESM mock.
 *
 * Files that spawn a child process which loads the addon on its own (desktop
 * sidecar, `mlx` CLI e2e) are listed in `SERIAL_BY_HAND`; the tracer cannot
 * see across a process boundary.
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseSync } from 'oxc-parser';

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
  // Reads `data/openai-gsm8k/{train,test}.jsonl`, which CI prepares via
  // `yarn mlx download dataset` (uses `@mlx-node/core`'s `convertParquetToJsonl`).
  '__test__/utils/dataset.test.ts',
];

/**
 * Pure-TS test files that rely on macOS-specific behaviour (APFS/darwin
 * flock, `/usr/bin/trash`, macOS volume layout) and therefore run on macOS CI
 * (`--project darwin-ts`) while `pure-ts` stays OS-agnostic for Linux CI.
 */
export const DARWIN_TS: string[] = [
  // Inspects the real macOS `.node` and `paged_attn.metallib` artifacts from
  // `yarn build:native` (only downloaded in the macOS `ts-test` job).
  '__test__/core/metallib-select.test.ts',
  // Exercises case-insensitive APFS cache-root folding (`volumeIsCaseInsensitive`),
  // which early-returns on case-sensitive Linux ext4.
  'packages/agent/__test__/cold-tier-root.test.ts',
  'packages/dashboard/__test__/api-prefix-cache.test.ts',
  'packages/dashboard/__test__/chat-work-landing.test.ts',
  'packages/dashboard/__test__/download.test.ts',
  'packages/dashboard/__test__/prefix-cache.test.ts',
  // Inspects the real macOS `.node` and `.metallib` artifacts from `yarn build:native`.
  'packages/desktop/__test__/payload.test.ts',
  // Copies `/bin/echo` expecting a macOS Mach-O binary (`file` / `findMachOFiles`);
  // Ubuntu's `/bin/echo` is ELF.
  'packages/desktop/__test__/verify-bundle.test.ts',
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
  return files.map((f) => relative(ROOT, f).replaceAll('\\', '/')).sort();
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

type AstNode = Record<string, unknown>;

function staticString(node: unknown): string | null {
  if (typeof node !== 'object' || node === null) return null;
  const n = node as AstNode;
  if (n.type === 'Literal' && typeof n.value === 'string') return n.value;
  if (n.type === 'TemplateLiteral') {
    const exprs = n.expressions as unknown[] | undefined;
    const quasis = n.quasis as AstNode[] | undefined;
    if (exprs?.length === 0 && quasis?.length === 1) {
      const value = quasis[0]?.value as { cooked?: unknown } | undefined;
      if (typeof value?.cooked === 'string') return value.cooked;
    }
  }
  return null;
}

function visitAst(node: unknown, fn: (n: AstNode) => void): void {
  if (typeof node !== 'object' || node === null) return;
  if (Array.isArray(node)) {
    for (const item of node) visitAst(item, fn);
    return;
  }
  const obj = node as AstNode;
  if (typeof obj.type === 'string') fn(obj);
  for (const value of Object.values(obj)) {
    if (typeof value === 'object' && value !== null) visitAst(value, fn);
  }
}

interface ImportEdge {
  specifier: string;
  /** True for ESM imports/re-exports/dynamic `import()`; false for CJS `require`/`createRequire`. */
  mockable: boolean;
}

interface ParsedFile {
  edges: ImportEdge[];
  fullMockSpecs: string[];
}

const parsedCache = new Map<string, ParsedFile>();

function isViMockCallee(callee: unknown): boolean {
  if (typeof callee !== 'object' || callee === null) return false;
  const c = callee as AstNode;
  if (c.type !== 'MemberExpression' || c.computed === true) return false;
  const obj = c.object as AstNode | undefined;
  const prop = c.property as AstNode | undefined;
  return (
    obj?.type === 'Identifier' &&
    (obj.name === 'vi' || obj.name === 'vitest') &&
    prop?.type === 'Identifier' &&
    prop.name === 'mock'
  );
}

function isCreateRequireCall(node: unknown): boolean {
  if (typeof node !== 'object' || node === null) return false;
  const n = node as AstNode;
  if (n.type !== 'CallExpression') return false;
  const callee = n.callee as AstNode | undefined;
  if (callee?.type === 'Identifier' && callee.name === 'createRequire') return true;
  if (callee?.type === 'MemberExpression' && callee.computed !== true) {
    const prop = callee.property as AstNode | undefined;
    if (prop?.type === 'Identifier' && prop.name === 'createRequire') return true;
  }
  return false;
}

function isRequireCallee(callee: unknown, requireAliases: ReadonlySet<string>): boolean {
  if (typeof callee !== 'object' || callee === null) return false;
  const c = callee as AstNode;
  if (c.type === 'Identifier' && typeof c.name === 'string' && requireAliases.has(c.name)) return true;
  return isCreateRequireCall(c);
}

function isFullMockFactory(factory: unknown): boolean {
  if (typeof factory !== 'object' || factory === null) return false;
  const f = factory as AstNode;
  if (f.type !== 'ArrowFunctionExpression' && f.type !== 'FunctionExpression') return false;
  const params = f.params as unknown[] | undefined;
  // A factory taking `(importOriginal)` / `(importActual)` can load the real module.
  if (params !== undefined && params.length > 0) return false;
  let touchesReal = false;
  visitAst(f.body, (node) => {
    if (touchesReal) return;
    if (node.type === 'Identifier' && (node.name === 'importActual' || node.name === 'importOriginal')) {
      touchesReal = true;
    }
  });
  return !touchesReal;
}

/** Collect runtime import specifiers (and top-level full `vi.mock` targets) via `oxc-parser`. */
function parseFile(file: string): ParsedFile {
  const cached = parsedCache.get(file);
  if (cached !== undefined) return cached;

  const src = readFileSync(file, 'utf-8');
  const parsed = parseSync(file, src);
  // Map specifier -> mockable (false wins when the same file both imports and requires a spec).
  const edgeMap = new Map<string, boolean>();
  const addEdge = (spec: string, mockable: boolean): void => {
    if (!mockable) edgeMap.set(spec, false);
    else if (!edgeMap.has(spec)) edgeMap.set(spec, true);
  };
  const fullMockSpecs: string[] = [];

  // Static imports: skip pure type imports (`import type` / `import { type X }`), keep side-effect `import '...'`.
  for (const imp of parsed.module.staticImports) {
    if (imp.entries.length > 0 && imp.entries.every((e) => e.isType)) continue;
    addEdge(imp.moduleRequest.value, true);
  }
  // Static re-exports: skip `export type ... from '...'`.
  for (const exp of parsed.module.staticExports) {
    for (const entry of exp.entries) {
      if (!entry.isType && entry.moduleRequest) addEdge(entry.moduleRequest.value, true);
    }
  }
  // Collect local aliases created via `const req = createRequire(...)`.
  const requireAliases = new Set<string>(['require']);
  visitAst(parsed.program, (node) => {
    if (node.type === 'VariableDeclarator' && isCreateRequireCall(node.init)) {
      const id = node.id as AstNode | undefined;
      if (id?.type === 'Identifier' && typeof id.name === 'string') {
        requireAliases.add(id.name);
      }
    }
  });
  // Dynamic `import('...')` and `require('...')` / `createRequire(...)('...')` with literal specifiers.
  visitAst(parsed.program, (node) => {
    if (node.type === 'ImportExpression') {
      const lit = staticString(node.source);
      if (lit !== null) addEdge(lit, true);
    } else if (node.type === 'CallExpression' && isRequireCallee(node.callee, requireAliases)) {
      const args = node.arguments as unknown[] | undefined;
      const lit = args?.[0] !== undefined ? staticString(args[0]) : null;
      if (lit !== null) addEdge(lit, false);
    }
  });

  // Top-level hoisted `vi.mock('spec', () => ({ ... }))` full module replacements.
  const body = (parsed.program as unknown as { body?: AstNode[] }).body ?? [];
  for (const stmt of body) {
    if (stmt.type !== 'ExpressionStatement') continue;
    const expr = stmt.expression as AstNode | undefined;
    if (expr?.type !== 'CallExpression' || !isViMockCallee(expr.callee)) continue;
    const args = expr.arguments as unknown[] | undefined;
    if (args === undefined || args.length < 2) continue;
    const spec = staticString(args[0]);
    if (spec !== null && isFullMockFactory(args[1])) {
      fullMockSpecs.push(spec);
    }
  }

  const edges: ImportEdge[] = [...edgeMap.entries()].map(([specifier, mockable]) => ({ specifier, mockable }));
  const result: ParsedFile = { edges, fullMockSpecs };
  parsedCache.set(file, result);
  return result;
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

function reachesAddonWithMocks(
  file: string,
  stack: Set<string>,
  mockedTargets: ReadonlySet<string>,
  cache: Map<string, boolean>,
): boolean {
  const cached = cache.get(file);
  if (cached !== undefined) return cached;
  if (stack.has(file)) return false; // cycle: resolved by whoever entered first
  stack.add(file);
  let hit = false;
  const { edges } = parseFile(file);
  for (const { specifier, mockable } of edges) {
    const target = resolveSpecifier(specifier, file);
    if (target === null) continue;
    if (mockable && mockedTargets.has(target)) continue;
    if (target === ADDON_PACKAGE || reachesAddonWithMocks(target, stack, mockedTargets, cache)) {
      hit = true;
      break;
    }
  }
  stack.delete(file);
  cache.set(file, hit);
  return hit;
}

/** True when `file` (absolute) can reach `@mlx-node/core` through runtime imports. */
export function reachesAddon(file: string, stack: Set<string> = new Set()): boolean {
  const { fullMockSpecs } = parseFile(file);
  if (fullMockSpecs.length === 0) {
    return reachesAddonWithMocks(file, stack, new Set(), reachCache);
  }
  const mockedTargets = new Set<string>();
  for (const spec of fullMockSpecs) {
    const target = resolveSpecifier(spec, file);
    if (target !== null) mockedTargets.add(target);
  }
  if (mockedTargets.size === 0) {
    return reachesAddonWithMocks(file, stack, new Set(), reachCache);
  }
  return reachesAddonWithMocks(file, stack, mockedTargets, new Map());
}

export interface Lanes {
  addon: string[];
  darwinTs: string[];
  pureTs: string[];
}

export function classify(): Lanes {
  const addon: string[] = [];
  const darwinTs: string[] = [];
  const pureTs: string[] = [];
  const darwinSet = new Set(DARWIN_TS);
  for (const rel of listTestFiles()) {
    if (SERIAL_BY_HAND.includes(rel) || reachesAddon(join(ROOT, rel))) addon.push(rel);
    else if (darwinSet.has(rel)) darwinTs.push(rel);
    else pureTs.push(rel);
  }
  return { addon, darwinTs, pureTs };
}

function main(): void {
  const lanes = classify();
  console.log(`// addon lane (${lanes.addon.length} files)`);
  for (const f of lanes.addon) console.log(`  '${f}',`);
  console.log(`\n// darwin-ts lane (${lanes.darwinTs.length} files)`);
  for (const f of lanes.darwinTs) console.log(`  '${f}',`);
  console.log(`\n// pure-ts lane (${lanes.pureTs.length} files)`);
  for (const f of lanes.pureTs) console.log(`  ${f}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
