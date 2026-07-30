/**
 * Which of the three processes may map the native addon.
 *
 * INFERENCE must; MAIN and CONTROL PANEL must not. That split IS the three-process
 * design — `mlx-core.darwin-arm64.node` is 61 MB of mapped Mach-O and ~29 MB of
 * RSS on the import alone, and a dashboard bug that took the tray and the
 * supervisor with it is the thing crash isolation is buying.
 *
 * Nothing enforces it but this file, and the regression is one stray `import`
 * away in a package nobody was editing: `@mlx-node/dashboard` depends on
 * `@mlx-node/agent`, whose MAIN entry pulls `@mlx-node/lm`. Today the dashboard
 * only reaches `@mlx-node/agent/catalog`, a leaf whose sole non-builtin edge is
 * an `import type`. Nothing but the walk below keeps it that way.
 *
 * Same two halves as `__test__/server/host/addon-free-subpaths.test.ts` at the
 * repo root, which is the pattern this follows: a static import-graph walk that
 * catches the regression at its cause, and a fresh-process `sharedObjects` probe
 * that catches a load arriving through a path the regex cannot see.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const src = (rel: string): string => resolve(ROOT, 'packages/desktop/src', rel);

/** Packages that map the addon, directly or by re-export. */
const ADDON_PACKAGES = ['@mlx-node/core', '@mlx-node/lm', '@mlx-node/vlm', '@mlx-node/trl'];

/** Every `from '…'` and bare `import '…'` specifier, **type-only imports included**. */
function specifiersOf(file: string): string[] {
  const source = readFileSync(file, 'utf-8');
  const out: string[] = [];
  for (const match of source.matchAll(/\bfrom\s+['"]([^'"]+)['"]/g)) out.push(match[1]);
  for (const match of source.matchAll(/\bimport\s+['"]([^'"]+)['"]/g)) out.push(match[1]);
  // `await import('…')` would otherwise sneak an addon package past the walk.
  for (const match of source.matchAll(/\bimport\(\s*['"]([^'"]+)['"]\s*\)/g)) out.push(match[1]);
  return out;
}

/**
 * The subset of {@link specifiersOf} reached ONLY through `import type` /
 * `export type`, which tsc erases entirely — no `require`, no dlopen, no bytes.
 *
 * Recognized only in the fully-explicit form. `import { type X }` and a value
 * import used solely as a type are both erased too, but neither is matched here:
 * the classification is deliberately biased so it can only ever count an erased
 * edge as a real one, never the reverse. Over-reporting fails loudly; the
 * opposite would be a hole.
 *
 * Kept as its own bucket rather than dropped, because "erased" is not "allowed":
 * `@mlx-node/agent/catalog` needs `ModelType` off the model registry so its
 * family list cannot drift from the families that exist, and that edge is
 * legitimate — but it is one keyword away from mapping 61 MB into CONTROL PANEL, so the
 * test names it explicitly instead of letting the walk fall silent about it.
 */
function typeOnlySpecifiersOf(file: string): string[] {
  const source = readFileSync(file, 'utf-8');
  const valued = new Set<string>();
  for (const match of source.matchAll(/\bfrom\s+['"]([^'"]+)['"]/g)) {
    const statement = source.slice(0, match.index);
    const start = Math.max(statement.lastIndexOf('import'), statement.lastIndexOf('export'));
    if (start >= 0 && /^(?:import|export)\s+type\s/.test(source.slice(start))) continue;
    valued.add(match[1]);
  }
  const out = new Set<string>();
  for (const match of source.matchAll(/\b(?:import|export)\s+type\s+[^'";]*?\bfrom\s+['"]([^'"]+)['"]/g)) {
    if (!valued.has(match[1])) out.add(match[1]);
  }
  return [...out];
}

/**
 * Where a `@mlx-node/*` specifier's SOURCE lives.
 *
 * The walk has to cross package boundaries or it proves nothing: stopping at
 * `@mlx-node/dashboard` would leave the whole question — what does the dashboard
 * import? — unasked, which is exactly where the regression would land.
 */
function workspaceSourceFor(specifier: string): string | null {
  const match = /^@mlx-node\/([^/]+)(?:\/(.+))?$/.exec(specifier);
  if (match === null) return null;
  const [, pkg, subpath] = match;
  const base = resolve(ROOT, 'packages', pkg, 'src');
  for (const candidate of subpath === undefined
    ? [resolve(base, 'index.ts')]
    : [resolve(base, `${subpath}.ts`), resolve(base, subpath, 'index.ts')]) {
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

interface Walk {
  files: string[];
  /**
   * Every `@mlx-node/*` package name reached by an edge that SURVIVES
   * compilation, whether or not it was followed. This is the list an offender
   * has to stay off.
   */
  packages: string[];
  /** Reached only through an erased `import type` — see {@link typeOnlySpecifiersOf}. */
  typeOnlyPackages: string[];
  /**
   * `@mlx-node/*` specifiers whose source could not be found.
   *
   * Fatal, not ignorable: an unresolvable specifier silently truncates the walk,
   * and a truncated walk is green for every reason except the right one.
   */
  unresolved: string[];
}

function walk(entry: string): Walk {
  const seen = new Set<string>();
  const packages = new Set<string>();
  const typeOnlyPackages = new Set<string>();
  const unresolved = new Set<string>();
  const queue = [entry];
  while (queue.length > 0) {
    const file = queue.pop() as string;
    if (seen.has(file)) continue;
    seen.add(file);
    const typeOnly = new Set(typeOnlySpecifiersOf(file));
    for (const specifier of specifiersOf(file)) {
      if (specifier.startsWith('.')) {
        // Source is written with `.js` specifiers (tsc NodeNext); the files on
        // disk are `.ts` / `.cts`.
        const stem = resolve(dirname(file), specifier.replace(/\.js$/, ''));
        const target = [`${stem}.ts`, `${stem}.cts`, resolve(stem, 'index.ts')].find(existsSync);
        if (target !== undefined) queue.push(target);
        continue;
      }
      if (!specifier.startsWith('@mlx-node/')) continue;
      // An erased edge is still WALKED (a type-only import of a workspace leaf
      // must not truncate the graph) — it is only bucketed apart, so the offender
      // list stays the set of edges that actually load something.
      if (typeOnly.has(specifier)) typeOnlyPackages.add(specifier);
      else packages.add(specifier);
      // `@mlx-node/core` is the addon itself — a published `index.cjs`, not a
      // `src` tree — so failing to resolve it is expected and not a truncation.
      if (ADDON_PACKAGES.includes(specifier.split('/').slice(0, 2).join('/'))) continue;
      const source = workspaceSourceFor(specifier);
      if (source === null) unresolved.add(specifier);
      else queue.push(source);
    }
  }
  return {
    files: [...seen].map((f) => f.slice(ROOT.length + 1)).sort(),
    packages: [...packages],
    typeOnlyPackages: [...typeOnlyPackages],
    unresolved: [...unresolved],
  };
}

const reachesAddon = (packages: string[]): string[] =>
  packages.filter((name) => ADDON_PACKAGES.some((p) => name === p || name.startsWith(`${p}/`)));

describe('CONTROL PANEL never links the native addon', () => {
  const graph = walk(src('control-panel/index.ts'));

  it('resolves every workspace specifier it walks through', () => {
    // Guards the guard. An unresolved specifier stops the walk there, and the
    // offender check below would then be looking at half a graph.
    expect(graph.unresolved).toEqual([]);
    // …and the walk really did cross into the dashboard, rather than stopping at
    // the package boundary and calling it clean.
    expect(graph.files).toContain('packages/dashboard/src/runtime.ts');
    expect(graph.files).toContain('packages/dashboard/src/download.ts');
  });

  it('reaches no package that maps the addon', () => {
    expect({ offenders: reachesAddon(graph.packages), reachedPackages: graph.packages.sort() }).toMatchObject({
      offenders: [],
    });
  });

  /**
   * The one erased edge, pinned by name.
   *
   * `@mlx-node/agent/catalog` takes `ModelType` off the model registry so its
   * cold-tier family list cannot name a family that does not exist. tsc erases
   * that import — the fresh-process probe below confirms nothing is mapped — but
   * "erased" is a property of the KEYWORD, not of the dependency, and deleting
   * five characters would turn it into 61 MB in CONTROL PANEL. Asserting the exact set
   * means a second such edge, or this one turning real, has to be argued for
   * here rather than appearing silently.
   */
  it('reaches the addon only through an erased type-only import, and only that one', () => {
    expect(reachesAddon(graph.typeOnlyPackages)).toEqual(['@mlx-node/lm']);
  });

  it('imports nothing but node builtins, the dashboard, and its own module', () => {
    // Whatever else the entry grows, it stays a file whose whole job is plumbing.
    // `electron` is deliberately absent: `process.parentPort` is a global, and an
    // import of that specifier would make an addon-free entry look like an
    // Electron one to the walk above for a type that is erased anyway.
    const specifiers = specifiersOf(src('control-panel/index.ts')).sort();
    expect(specifiers).toEqual(['./session.js', './shutdown-timings.js', '@mlx-node/dashboard', 'node:fs']);
  });
});

describe('INFERENCE is the one that does', () => {
  const graph = walk(src('inference/index.ts'));

  it('reaches the addon through @mlx-node/server/host', () => {
    // The positive half. Without it, "CONTROL PANEL is addon-free" would also be
    // satisfied by an app in which nothing loads the addon at all — a sidecar
    // that starts, serves /health, and cannot run a model.
    expect(reachesAddon(graph.packages).length).toBeGreaterThan(0);
    expect(graph.packages).toContain('@mlx-node/server/host');
  });

  it('keeps every decision out of the file that loads it', () => {
    // A file that imports `@mlx-node/server/host` cannot be imported by any test
    // that has to run without a GPU, so it holds no rules — `sidecar.ts` does.
    const source = readFileSync(src('inference/index.ts'), 'utf-8');
    expect(source).toContain('sidecarHostOptions()');
    // An options literal here would reintroduce whatever `sidecarHostOptions`
    // decides — the `pickFreePort` race a literal port brings back, or a bind
    // with no `authToken` — and no test of `sidecar.ts` could see either.
    expect(/createInferenceHost\(\s*\{/.test(source)).toBe(false);
  });
});

describe('MAIN never links the native addon', () => {
  const graph = walk(src('main/index.ts'));

  it('reaches only the addon-free leaf of @mlx-node/server', () => {
    expect({ offenders: reachesAddon(graph.packages) }).toMatchObject({ offenders: [] });
    expect(graph.packages).toEqual(['@mlx-node/server/host/env-policy']);
  });
});

/*
 * The dynamic half. The walk proves nothing is WRITTEN; these prove the addon is
 * or is not actually mapped, in a fresh process — inside the vitest worker some
 * other test has almost certainly loaded it already, so an in-process check would
 * be green for entirely the wrong reason.
 */
describe('observed in a real process', () => {
  const dashboard = resolve(ROOT, 'packages/dashboard/dist/index.js');
  const serverHost = resolve(ROOT, 'packages/server/dist/host/index.js');

  const mapped = (specifier: string): string => {
    const probe = `
      await import(${JSON.stringify(specifier)});
      const found = process.report.getReport().sharedObjects.filter((s) => s.includes('mlx-core'));
      console.log(found.length > 0 ? 'MAPPED' : 'NOT-MAPPED');
    `;
    return execFileSync(process.execPath, ['--input-type=module', '-e', probe], { encoding: 'utf-8' }).trim();
  };

  it.skipIf(!existsSync(dashboard))('importing the dashboard runtime does not map the addon', () => {
    expect(mapped(dashboard)).toBe('NOT-MAPPED');
  });

  // Guards the guard: if the probe could not see a mapped addon, the assertion
  // above would hold no matter what CONTROL PANEL imported.
  it.skipIf(!existsSync(serverHost))('the probe really does detect one', () => {
    expect(mapped(serverHost)).toBe('MAPPED');
  });
});

describe('NAPI_RS_NATIVE_LIBRARY_PATH must name the .node file', () => {
  const core = resolve(ROOT, 'packages/core/index.cjs');
  const addon = resolve(ROOT, 'packages/core/mlx-core.darwin-arm64.node');

  /** Load `packages/core` in a fresh process with the variable set to `value`. */
  const loadWith = (value: string): string => {
    const probe = `
      try {
        const m = require(${JSON.stringify(core)});
        const fns = Object.keys(m).filter((k) => typeof m[k] === 'function').length;
        console.log('LOADED fns=' + fns + ' memoryStats=' + typeof m.memoryStats);
      } catch (err) {
        console.log('THREW ' + String(err.message).split('\\n')[0]);
      }
    `;
    return execFileSync(process.execPath, ['-e', probe], {
      encoding: 'utf-8',
      env: { ...process.env, NAPI_RS_NATIVE_LIBRARY_PATH: value },
    })
      .trim()
      .split('\n')
      .at(-1) as string;
  };

  it.skipIf(!existsSync(addon))('loads the addon when pointed at the file', () => {
    expect(loadWith(addon)).toMatch(/^LOADED fns=[1-9]\d+ memoryStats=function$/);
  });

  it.skipIf(!existsSync(addon))('SILENTLY half-loads when pointed at the containing directory', () => {
    // The reason `AppPaths.nativeAddon` is a file path and this test exists.
    // `requireNative()` passes the value straight to `require()`, and
    // `require(<dir holding index.cjs>)` resolves back to the module that is
    // being evaluated — Node hands back its half-built exports. 67 keys, and
    // every single one `undefined`: no throw, no useful warning, and a sidecar
    // that binds a port, answers /health with `ok`, and cannot run a model.
    expect(loadWith(resolve(ROOT, 'packages/core'))).toBe('LOADED fns=0 memoryStats=undefined');
  });

  it.skipIf(!existsSync(addon))('throws when pointed at any other directory', () => {
    expect(loadWith(resolve(ROOT, 'packages/desktop/build'))).toContain('THREW Cannot find native binding');
  });
});
