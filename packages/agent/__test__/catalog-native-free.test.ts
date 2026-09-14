/**
 * Subprocess gate for the native-free import contract: `@mlx-node/agent/catalog`
 * and the `@mlx-node/lm/family-data` leaf must reach nothing that dlopens the
 * addon. A stray value import is invisible to every behavioural test, which runs
 * in a vitest worker that already loaded it.
 *
 * Imports the BUILT subpaths, so run `yarn build:ts` first; a missing dist fails
 * the existence assertions loudly instead of skipping.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

const CATALOG_DIST = resolve(ROOT, 'packages/agent/dist/catalog.js');
const PATHS_DIST = resolve(ROOT, 'packages/agent/dist/paths.js');
const FAMILY_DATA_DIST = resolve(ROOT, 'packages/lm/dist/family-data.js');
const LM_ROOT_DIST = resolve(ROOT, 'packages/lm/dist/index.js');
const MODELS_DIST = resolve(ROOT, 'packages/agent/dist/provider/models.js');
const MODEL_HANDLER_DIST = resolve(ROOT, 'packages/dashboard/dist/api/handlers/models.js');

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
    expect(Object.keys(agentPkg.exports)).toContain('./paths');
    expect(existsSync(PATHS_DIST)).toBe(true);
    expect(Object.keys(agentPkg.exports)).toContain('./models');
    expect(Object.keys(lmPkg.exports)).toContain('./model-detection');
    expect(existsSync(MODELS_DIST)).toBe(true);
  });

  it('discovers a standalone GGUF through the built dashboard handler without the addon', () => {
    const probe = `
      ${RESOLVE_HOOK}
      const fs = await import('node:fs/promises');
      const { tmpdir } = await import('node:os');
      const { join } = await import('node:path');
      const { discoverMlxModels } = await import(${JSON.stringify(MODELS_DIST)});
      const { handleCodingAgentModels } = await import(${JSON.stringify(MODEL_HANDLER_DIST)});
      const dir = await fs.mkdtemp(join(tmpdir(), 'native-free-gguf-'));
      try {
        const str = text => {
          const bytes = Buffer.from(text);
          const length = Buffer.alloc(8);
          length.writeBigUInt64LE(BigInt(bytes.length));
          return Buffer.concat([length, bytes]);
        };
        const header = Buffer.alloc(24);
        header.write('GGUF'); header.writeUInt32LE(3, 4); header.writeBigUInt64LE(1n, 16);
        await fs.writeFile(join(dir, 'Qwen3.8-Q4_K_XL.gguf'), Buffer.concat([
          header, str('general.architecture'), Buffer.from([8,0,0,0]), str('qwen35')
        ]));
        const agentNames = (await discoverMlxModels(dir)).map(model => model.discovered.name);
        const uiNames = (await handleCodingAgentModels({ modelsDir: dir })).models.map(model => model.name);
        if (JSON.stringify(agentNames) !== JSON.stringify(['Qwen3.8-Q4_K_XL'])) throw new Error('Missing GGUF');
        if (JSON.stringify(agentNames) !== JSON.stringify(uiNames)) throw new Error('Inventories differ');
        console.log('NATIVE-FREE-GGUF-OK');
      } finally {
        await fs.rm(dir, { recursive: true, force: true });
      }
    `;
    const { status, output } = runNodeChild(probe);
    expect(output, output).toContain('NATIVE-FREE-GGUF-OK');
    expect(status).toBe(0);
  });

  it('imports built catalog and path subpaths in a child whose resolver bans the addon', () => {
    const probe = `
      ${RESOLVE_HOOK}
      const catalog = await import(${JSON.stringify(CATALOG_DIST)});
      const familyData = await import(${JSON.stringify(FAMILY_DATA_DIST)});
      const paths = await import(${JSON.stringify(PATHS_DIST)});
      const { pathToFileURL } = await import('node:url');
      const dir = ${JSON.stringify(resolve(ROOT, '.cache/agent #配置'))};
      if (paths.expandPiAgentDir(pathToFileURL(dir).href) !== dir) throw new Error('File URL expansion differs');
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
