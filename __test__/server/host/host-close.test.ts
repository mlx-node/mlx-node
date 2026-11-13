import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import type { ServableModel } from '@mlx-node/server';
import { FLUSH_MS } from '@mlx-node/server/supervision';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createInferenceHost, type InferenceHost } from '../../../packages/server/src/host/index.js';

/**
 * Splash `flushRestorePoints` under `supervision::kFlush`: a clean stop
 * closes the HTTP server first, then every loaded model's SSD cache in turn
 * against ONE deadline, then drains the other families' process tier with
 * whatever is left. A crashed host's temporary caches go at the next start.
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
  const root = await mkdtemp(join(tmpdir(), 'mlx-host-close-'));
  scratch.push(root);
  await mkdir(join(root, 'qwen'), { recursive: true });
  await writeFile(join(root, 'qwen', 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }), 'utf-8');
  return root;
}

describe('host close', () => {
  // Mutation HC1: a fresh FLUSH_MS deadline per model → red (the second
  // model's deadline is 6 s away, not 2 s).
  it('closes every cache in one flush budget, after the server', async () => {
    let clock = 1_000_000;
    const events: string[] = [];
    const deadlines: Array<{ model: string; left: number }> = [];
    const drains: number[] = [];
    const lines: string[] = [];
    let sweeps = 0;
    const host = await createInferenceHost({
      port: 0,
      disableStore: true,
      modelsDir: await modelsDir(),
      estimateContextWindow: async () => null,
      loadModel: async () => ({}) as unknown as LoadableModel,
      sweepCacheLeftovers: () => {
        sweeps += 1;
        return 0;
      },
      coldCacheDrain: (timeoutMs) => {
        events.push('drain');
        drains.push(timeoutMs);
        return true;
      },
      now: () => clock,
      log: (line) => lines.push(line),
    });
    hosts.push(host);
    expect(sweeps).toBe(1);

    const cacheModel = (name: string, takesMs: number): ServableModel =>
      ({
        closeCache: async (deadlineMs: number) => {
          events.push(`closeCache ${name}`);
          deadlines.push({ model: name, left: deadlineMs - clock });
          clock += takesMs;
          return true;
        },
      }) as unknown as ServableModel;
    const first = cacheModel('a', 4000);
    host.server.registry.register('a', first);
    host.server.registry.register('a-alias', first);
    host.server.registry.register('b', cacheModel('b', 500));

    const serverClose = host.server.close.bind(host.server);
    host.server.close = async (opts) => {
      events.push('server.close');
      return serverClose(opts);
    };

    await host.close({ timeoutMs: 250 });

    expect(events).toEqual(['server.close', 'closeCache a', 'closeCache b', 'drain']);
    expect(deadlines).toEqual([
      { model: 'a', left: FLUSH_MS },
      { model: 'b', left: FLUSH_MS - 4000 },
    ]);
    expect(drains).toEqual([FLUSH_MS - 4500]);
    expect(lines).toEqual(['[mlx] SSD cache close: 2 of 2 flushed']);
  });

  // Mutation HL1: the close line counts a cache that missed its deadline as
  // flushed → red.
  it('logs how many caches flushed, missed the deadline or failed', async () => {
    let clock = 1_000_000;
    const lines: string[] = [];
    const host = await createInferenceHost({
      port: 0,
      disableStore: true,
      modelsDir: await modelsDir(),
      sweepOrphanTempRoots: false,
      estimateContextWindow: async () => null,
      loadModel: async () => ({}) as unknown as LoadableModel,
      coldCacheDrain: () => true,
      now: () => clock,
      log: (line) => lines.push(line),
    });
    hosts.push(host);
    const model = (closeCache: () => Promise<boolean>): ServableModel => ({ closeCache }) as unknown as ServableModel;
    host.server.registry.register(
      'flushed',
      model(async () => {
        clock += FLUSH_MS;
        return true;
      }),
    );
    host.server.registry.register(
      'late',
      model(async () => false),
    );
    host.server.registry.register(
      'failed',
      model(async () => {
        throw new Error('tier gone');
      }),
    );
    host.server.registry.register(
      'hung',
      model(() => new Promise<boolean>(() => undefined)),
    );

    await host.close({ timeoutMs: 250 });

    expect(lines).toEqual(['[mlx] SSD cache close: 1 of 4 flushed, 2 not flushed in time, 1 failed']);
  });

  it('logs nothing when no loaded model has an SSD cache', async () => {
    const lines: string[] = [];
    const host = await createInferenceHost({
      port: 0,
      disableStore: true,
      modelsDir: await modelsDir(),
      sweepOrphanTempRoots: false,
      estimateContextWindow: async () => null,
      loadModel: async () => ({}) as unknown as LoadableModel,
      coldCacheDrain: () => true,
      log: (line) => lines.push(line),
    });
    hosts.push(host);
    host.server.registry.register('plain', {} as unknown as ServableModel);

    await host.close({ timeoutMs: 250 });

    expect(lines).toEqual([]);
  });

  it('skips the start sweep when the host is told to leave leftovers alone', async () => {
    let sweeps = 0;
    const host = await createInferenceHost({
      port: 0,
      disableStore: true,
      modelsDir: await modelsDir(),
      sweepOrphanTempRoots: false,
      estimateContextWindow: async () => null,
      loadModel: async () => ({}) as unknown as LoadableModel,
      sweepCacheLeftovers: () => {
        sweeps += 1;
        return 0;
      },
      coldCacheDrain: () => true,
    });
    hosts.push(host);
    expect(sweeps).toBe(0);
  });
});
