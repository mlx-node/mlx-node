import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';

import type { Qwen35LoadOptions } from '@mlx-node/core';
import type { LoadableModel, LoadModelOptions } from '@mlx-node/lm';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createInferenceHost, type InferenceHost } from '../../../packages/server/src/host/index.js';

/**
 * The SSD cache sets one DFlash2 state's staging aside in a Qwen3.5 memory
 * plan, so the load-free window of `/v1/models` must be priced with the same
 * `--max-cache-disk` the load takes (Splash `stateStagingBytes`).
 */

let scratch: string[] = [];
let hosts: InferenceHost[] = [];

afterEach(async () => {
  for (const host of hosts) await host.close({ timeoutMs: 250 }).catch(() => undefined);
  hosts = [];
  for (const dir of scratch) await rm(dir, { recursive: true, force: true }).catch(() => undefined);
  scratch = [];
});

async function modelsDir(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-host-estimate-'));
  scratch.push(root);
  for (const [name, modelType] of [
    ['qwen', 'qwen3_5'],
    ['gemma', 'gemma4'],
  ] as const) {
    await mkdir(join(root, name), { recursive: true });
    await writeFile(join(root, name, 'config.json'), JSON.stringify({ model_type: modelType }), 'utf-8');
  }
  return root;
}

describe('host cache-disk options', () => {
  // Mutation EC1: the estimate options without `cacheDisk` → red.
  it.each(['0', 'auto'])('prices the estimate with cacheDisk %s', async (cacheDisk) => {
    const asked: Array<Qwen35LoadOptions | undefined> = [];
    let settle: () => void = () => undefined;
    const estimated = new Promise<void>((resolve) => (settle = resolve));
    const host = await createInferenceHost({
      port: 0,
      disableStore: true,
      sweepOrphanTempRoots: false,
      modelsDir: await modelsDir(),
      cacheDisk,
      estimateContextWindow: async (_path, _modelType, options) => {
        asked.push(options);
        settle();
        return null;
      },
      loadModel: async () => ({}) as unknown as LoadableModel,
    });
    hosts.push(host);
    await estimated;
    expect(asked).toEqual([{ cacheDisk }]);
  });

  it('hands the cache options to a qwen3_5 load and to no other family', async () => {
    const loads: Array<{ path: string; options?: LoadModelOptions }> = [];
    const host = await createInferenceHost({
      port: 0,
      disableStore: true,
      sweepOrphanTempRoots: false,
      modelsDir: await modelsDir(),
      cacheDisk: '17179869184',
      persistentCache: true,
      cacheDir: '/var/cache/mlx',
      estimateContextWindow: async () => null,
      loadModel: async (path, options) => {
        loads.push({ path, ...(options !== undefined ? { options } : {}) });
        return {} as unknown as LoadableModel;
      },
    });
    hosts.push(host);
    await host.loadModel('qwen');
    await host.loadModel('gemma');
    expect(loads.map(({ path, ...rest }) => ({ name: basename(path), ...rest }))).toEqual([
      {
        name: 'qwen',
        options: { cacheDisk: '17179869184', persistentCache: true, cacheDir: '/var/cache/mlx' },
      },
      { name: 'gemma' },
    ]);
  });
});
