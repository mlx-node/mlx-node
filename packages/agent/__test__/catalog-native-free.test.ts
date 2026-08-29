/**
 * Subprocess gate for the native-free import contract.
 *
 * `@mlx-node/agent/catalog` is the agent package's one native-free entry
 * point (dashboard viewer process, `mlx agent --help`), and it now
 * value-imports `@mlx-node/lm/family-data` — legal only while that leaf stays
 * free of runtime imports (`import type` only). Nothing enforced either
 * contract until this file: one stray value import away, the dashboard starts
 * dlopening Metal, and the symptom (slower start, more memory) is invisible
 * to every behavioural test running inside a vitest worker that already
 * loaded the addon.
 *
 * A fresh node child registers a resolve hook that throws on `@mlx-node/core`
 * or any `.node` specifier, then imports the BUILT subpaths — so the check
 * covers the real published import graph, not the source aliases. Run
 * `yarn build:ts` first; a missing dist fails the existence assertions loudly
 * instead of skipping.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

const CATALOG_DIST = resolve(ROOT, 'packages/agent/dist/catalog.js');
const FAMILY_DATA_DIST = resolve(ROOT, 'packages/lm/dist/family-data.js');
const LM_ROOT_DIST = resolve(ROOT, 'packages/lm/dist/index.js');

const RESOLVE_HOOK = `
  const { registerHooks } = await import('node:module');
  registerHooks({
    resolve(specifier, context, nextResolve) {
      if (specifier === '@mlx-node/core' || specifier.startsWith('@mlx-node/core/') || specifier.endsWith('.node')) {
        throw new Error('native import blocked: ' + specifier);
      }
      return nextResolve(specifier, context);
    },
  });
`;

function runNodeChild(script: string): { status: number; output: string } {
  try {
    const output = execFileSync(process.execPath, ['--input-type=module', '-e', script], {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    return { status: 0, output };
  } catch (error) {
    const failure = error as { status?: number; stdout?: string; stderr?: string };
    return { status: failure.status ?? 1, output: `${failure.stdout ?? ''}${failure.stderr ?? ''}` };
  }
}

describe('native-free catalog + family-data subpaths', () => {
  it('built artifacts and the published subpath exist', () => {
    expect(existsSync(CATALOG_DIST)).toBe(true);
    expect(existsSync(FAMILY_DATA_DIST)).toBe(true);
    const lmPkg = JSON.parse(readFileSync(resolve(ROOT, 'packages/lm/package.json'), 'utf-8')) as {
      exports: Record<string, unknown>;
    };
    expect(Object.keys(lmPkg.exports)).toContain('./family-data');
    const agentPkg = JSON.parse(readFileSync(resolve(ROOT, 'packages/agent/package.json'), 'utf-8')) as {
      exports: Record<string, unknown>;
    };
    expect(Object.keys(agentPkg.exports)).toContain('./catalog');
  });

  it('imports both built subpaths in a child whose resolver bans the addon', () => {
    const probe = `
      ${RESOLVE_HOOK}
      const catalog = await import(${JSON.stringify(CATALOG_DIST)});
      const familyData = await import(${JSON.stringify(FAMILY_DATA_DIST)});
      if (typeof catalog.matchFamily !== 'function') throw new Error('catalog.matchFamily missing');
      if (typeof catalog.rawModelTypeToCanonical !== 'function') throw new Error('catalog.rawModelTypeToCanonical missing');
      if (!(catalog.NON_GENERATIVE_FAMILY_IDS instanceof Set)) throw new Error('catalog.NON_GENERATIVE_FAMILY_IDS missing');
      if (!Array.isArray(catalog.CHAT_FAMILY_IDS)) throw new Error('catalog.CHAT_FAMILY_IDS missing');
      if (!(catalog.COLD_TIER_RESTORE_FAMILIES instanceof Set)) throw new Error('catalog.COLD_TIER_RESTORE_FAMILIES missing');
      if (catalog.matchFamily('/probe', { model_type: 'muse_glimmer_text' }) !== 'muse_glimmer') {
        throw new Error('matchFamily decision wrong in child');
      }
      if (familyData.matchFamily !== catalog.matchFamily) throw new Error('catalog must re-export the leaf');
      console.log('NATIVE-FREE-OK');
    `;
    const { status, output } = runNodeChild(probe);
    expect(output, output).toContain('NATIVE-FREE-OK');
    expect(status).toBe(0);
  });

  // Guards the guard: the same hook must actually fire on a module that DOES
  // reach the addon, or the success above proves nothing.
  it('the resolve hook really blocks a native import', () => {
    const probe = `
      ${RESOLVE_HOOK}
      await import(${JSON.stringify(LM_ROOT_DIST)});
      console.log('SHOULD-NOT-REACH');
    `;
    const { status, output } = runNodeChild(probe);
    expect(output).not.toContain('SHOULD-NOT-REACH');
    expect(output).toContain('native import blocked');
    expect(status).not.toBe(0);
  });
});
