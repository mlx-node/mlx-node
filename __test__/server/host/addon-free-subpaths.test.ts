import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

/**
 * `@mlx-node/server/host` statically imports `@mlx-node/lm`, so importing it
 * dlopens `mlx-core.darwin-arm64.node` and costs ~30 MB RSS at module load.
 *
 * That is fine for the inference sidecar, which needs the addon anyway. It is NOT
 * fine for:
 *
 *   - `mlx download` — the command you run BEFORE you have anything to run, where
 *     a Metal-at-dlopen abort has bitten CI before;
 *   - the Electron MAIN process, which supervises the sidecar and exists to stay
 *     small and responsive.
 *
 * Both reach for one narrow thing (`resolveModelsDir`, `engineEnvFor`), so those
 * live in leaf modules published as their own subpaths. Nothing enforced that
 * split until this file: it is one stray `import` away from silently regressing,
 * and the symptom — a slower start and more memory — is invisible in a test suite
 * that only checks behaviour.
 */
const ADDON_FREE_SUBPATHS = ['host/env-policy.ts', 'host/paths.ts'];

/** Modules that pull in the native addon, directly or by re-export. */
const FORBIDDEN = ['@mlx-node/lm', '@mlx-node/core', '@mlx-node/vlm', '@mlx-node/trl'];

/**
 * Every `from '...'` specifier in a source file, **type-only imports included**.
 *
 * Deliberately stricter than the runtime property. `import type` is erased by tsc
 * and loads nothing, so flagging it is over-strict on its face — but it is one
 * character away from becoming a value import, and these two leaf modules have no
 * legitimate need for a type out of the addon packages. The dynamic probe below
 * is what pins the actual runtime behaviour; this is the early warning.
 */
function specifiersOf(file: string): string[] {
  const src = readFileSync(file, 'utf-8');
  const out: string[] = [];
  for (const m of src.matchAll(/\bfrom\s+['"]([^'"]+)['"]/g)) out.push(m[1]);
  // `import '...'` for side effects carries no `from`.
  for (const m of src.matchAll(/\bimport\s+['"]([^'"]+)['"]/g)) out.push(m[1]);
  return out;
}

/** Walk the real import graph from `entry`, following relative specifiers. */
function reachableFrom(entry: string): { files: string[]; bare: Set<string> } {
  const seen = new Set<string>();
  const bare = new Set<string>();
  const queue = [entry];
  while (queue.length > 0) {
    const file = queue.pop() as string;
    if (seen.has(file)) continue;
    seen.add(file);
    for (const spec of specifiersOf(file)) {
      if (!spec.startsWith('.')) {
        bare.add(spec);
        continue;
      }
      // Source is written with `.js` specifiers (tsc NodeNext) but the files on
      // disk are `.ts`.
      const target = resolve(dirname(file), spec.replace(/\.js$/, '.ts'));
      if (existsSync(target)) queue.push(target);
    }
  }
  return { files: [...seen], bare };
}

describe.each(ADDON_FREE_SUBPATHS)('%s stays addon-free', (rel) => {
  const entry = resolve(ROOT, 'packages/server/src', rel);

  it('exists and is published as its own subpath', () => {
    expect(existsSync(entry)).toBe(true);
    const pkg = JSON.parse(readFileSync(resolve(ROOT, 'packages/server/package.json'), 'utf-8')) as {
      exports: Record<string, unknown>;
    };
    expect(Object.keys(pkg.exports)).toContain(`./${rel.replace(/\.ts$/, '')}`);
  });

  // The static half: catches the regression at its cause — someone adding an
  // import — rather than at a symptom that no behavioural test would notice.
  it('reaches no module that loads the native addon', () => {
    const { files, bare } = reachableFrom(entry);
    const offenders = [...bare].filter((s) => FORBIDDEN.some((f) => s === f || s.startsWith(`${f}/`)));
    expect({ offenders, graph: files.map((f) => f.slice(ROOT.length + 1)) }).toMatchObject({ offenders: [] });
  });
});

// The dynamic half. The static walk proves nothing is *written*; this proves the
// addon is not actually mapped, which also covers a transitive load arriving
// through a path the regex cannot see.
describe('addon-free subpaths, observed in a real process', () => {
  const built = resolve(ROOT, 'packages/server/dist/host/env-policy.js');

  it.skipIf(!existsSync(built))('importing host/env-policy does not map the addon', () => {
    const probe = `
      await import(${JSON.stringify(built)});
      const mapped = process.report.getReport().sharedObjects.filter((s) => s.includes('mlx-core'));
      if (mapped.length > 0) { console.error('MAPPED: ' + mapped.join(', ')); process.exit(3); }
    `;
    // A fresh process is essential: inside the vitest worker some other test has
    // almost certainly loaded the addon already, so an in-process check here
    // would be green for entirely the wrong reason.
    const out = execFileSync(process.execPath, ['--input-type=module', '-e', probe], { encoding: 'utf-8' });
    expect(out).not.toContain('MAPPED');
  });

  // Guards the guard: if the probe could not detect a mapped addon, the test
  // above would pass no matter what.
  it('the probe really does detect a mapped addon', () => {
    const core = resolve(ROOT, 'packages/core/index.cjs');
    const probe = `
      const { createRequire } = await import('node:module');
      createRequire(${JSON.stringify(import.meta.url)})(${JSON.stringify(core)});
      const mapped = process.report.getReport().sharedObjects.filter((s) => s.includes('mlx-core'));
      console.log(mapped.length > 0 ? 'MAPPED' : 'NOT-MAPPED');
    `;
    const out = execFileSync(process.execPath, ['--input-type=module', '-e', probe], { encoding: 'utf-8' });
    expect(out.trim()).toBe('MAPPED');
  });
});
