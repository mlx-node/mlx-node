import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { LoadableModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { createInferenceHost, type InferenceHost } from '../../../packages/server/src/host/index.js';
import { residentSessionRegistry } from '../../../packages/server/src/inference-admission.js';
import { ModelWorkCoordinator } from '../../../packages/server/src/model-work-coordinator.js';
import type { ModelRegistry } from '../../../packages/server/src/registry.js';
import type { SessionRegistry } from '../../../packages/server/src/session-registry.js';

let hosts: InferenceHost[] = [];
let dirs: string[] = [];
afterEach(async () => {
  for (const host of hosts) await host.close({ timeoutMs: 250 }).catch(() => undefined);
  hosts = [];
  for (const dir of dirs) await rm(dir, { recursive: true, force: true }).catch(() => undefined);
  dirs = [];
});

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function modelsDir(names: string[]): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'mlx-alias-check-'));
  dirs.push(root);
  for (const name of names) {
    await mkdir(join(root, name), { recursive: true });
    await writeFile(join(root, name, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }), 'utf-8');
    await writeFile(join(root, name, 'model.safetensors'), 'not-real-weights', 'utf-8');
  }
  return root;
}

interface Turn {
  load: number;
  text: string;
}

function lanes(): {
  loads: string[];
  turns: Turn[];
  entered: (text: string) => Promise<void>;
  finishLong: () => void;
  loadModel: (path: string) => Promise<LoadableModel>;
} {
  const loads: string[] = [];
  const turns: Turn[] = [];
  const waiters = new Map<string, () => void>();
  const longGate = deferred();
  const result = {
    text: 'ok',
    toolCalls: [],
    thinkingEnabled: false,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'ok',
    cachedTokens: 0,
  };
  const run = async (load: number, messages: Array<{ content?: unknown }>): Promise<void> => {
    const text = messages.map((m) => (typeof m.content === 'string' ? m.content : '')).join(' ');
    turns.push({ load, text });
    for (const [key, wake] of waiters) if (text.includes(key)) wake();
    if (text.includes('long')) await longGate.promise;
  };
  const loadModel = async (path: string): Promise<LoadableModel> => {
    loads.push(path);
    const load = loads.length - 1;
    const start = async (messages: Array<{ content?: unknown }>) => (await run(load, messages), result);
    return {
      hasBlockPagedCache: () => true,
      maxConcurrentSequences: () => 4,
      chatSessionStart: start,
      chatSessionContinue: start,
      chatSessionContinueTool: start,
      chatStreamSessionStart: async function* (messages: Array<{ content?: unknown }>) {
        await run(load, messages);
        yield { ...result, done: true, thinking: null };
      },
      resetCaches: async () => undefined,
    } as unknown as LoadableModel;
  };
  return {
    loads,
    turns,
    entered: (text) =>
      new Promise<void>((resolve) => {
        if (turns.some((turn) => turn.text.includes(text))) resolve();
        else waiters.set(text, resolve);
      }),
    finishLong: longGate.resolve,
    loadModel,
  };
}

async function startLanes(fake: ReturnType<typeof lanes>): Promise<InferenceHost> {
  const host = await createInferenceHost({
    port: 0,
    disableStore: true,
    sweepOrphanTempRoots: false,
    estimateContextWindow: async () => null,
    modelsDir: await modelsDir(['alpha', 'beta']),
    model: 'beta',
    pagedModelTypes: [],
    loadModel: fake.loadModel,
  });
  hosts.push(host);
  return host;
}

function send(host: InferenceHost, model: string, text: string): Promise<number> {
  return fetch(`${host.url}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages: [{ role: 'user', content: text }], max_tokens: 8 }),
  }).then(async (res) => (await res.text(), res.status));
}

async function startedOrParked(host: InferenceHost, started: Promise<unknown>): Promise<'started' | 'parked'> {
  let done = false;
  const parked = (async () => {
    while (!done && host.server.modelWork.waitingWriters === 0) await new Promise((r) => setImmediate(r));
    return 'parked' as const;
  })();
  const first = await Promise.race([started.then(() => 'started' as const), parked]);
  done = true;
  return first;
}

describe('checker: alias of the loaded model', () => {
  // Mutation: the helper ignores the binder -> the alias queues a writer and the new turn parks behind it.
  it('a new turn on the loaded model is not held back by an alias first use', async () => {
    const fake = lanes();
    const host = await startLanes(fake);
    const long = send(host, 'beta', 'long turn');
    await fake.entered('long');

    const alias = send(host, 'mlx-local', 'title please');
    const next = send(host, 'beta', 'next turn');
    expect(await startedOrParked(host, Promise.all([fake.entered('title'), fake.entered('next')]))).toBe('started');

    fake.finishLong();
    expect(await Promise.all([long, alias, next])).toEqual([200, 200, 200]);
    expect(fake.loads).toHaveLength(1);
  });

  // Mutation: the helper ignores a queued writer -> the alias binds to beta, parks as a reader, answers 400.
  it('an alias behind a queued direct host.loadModel swap runs on the new model', async () => {
    const fake = lanes();
    const host = await startLanes(fake);
    const long = send(host, 'beta', 'long turn');
    await fake.entered('long');

    const swap = host.loadModel('alpha');
    while (host.server.modelWork.waitingWriters === 0) await new Promise((r) => setImmediate(r));
    const alias = send(host, 'mlx-local', 'title please');
    while (host.server.modelWork.waitingWriters < 2 && host.server.registry.get('mlx-local') === undefined)
      await new Promise((r) => setImmediate(r));

    fake.finishLong();
    await swap;
    expect(await Promise.all([long, alias])).toEqual([200, 200]);
    expect(fake.loads.map((path) => path.slice(path.lastIndexOf('/') + 1))).toEqual(['beta', 'alpha']);
    expect(fake.turns.find((turn) => turn.text.includes('title'))?.load).toBe(1);
  });

  // Mutation: no `aliases.add` in `bindLoadedAlias` -> after the swap the alias still runs on the old model.
  it('an alias bound beside a turn follows a later swap, never the swapped-out model', async () => {
    const fake = lanes();
    const host = await startLanes(fake);
    const long = send(host, 'beta', 'long turn');
    await fake.entered('long');
    const alias = send(host, 'mlx-local', 'title one');
    expect(await startedOrParked(host, fake.entered('title one'))).toBe('started');
    fake.finishLong();
    expect(await Promise.all([long, alias])).toEqual([200, 200]);

    expect(await send(host, 'alpha', 'swap turn')).toBe(200);
    expect(await send(host, 'mlx-local', 'title two')).toBe(200);
    expect(fake.turns.find((turn) => turn.text.includes('title two'))?.load).toBe(1);
    expect(host.server.registry.get('beta')).toBeUndefined();
  });
});

describe('checker: residentSessionRegistry', () => {
  const lane = {} as SessionRegistry;
  function registry(bound: Set<string>): ModelRegistry {
    return { getSessionRegistry: (name: string) => (bound.has(name) ? lane : undefined) } as unknown as ModelRegistry;
  }

  // Mutation: drop `writerActive ||` from the helper -> it binds while a writer holds the slot.
  it('does not bind while a writer holds the slot, binds once it is free', async () => {
    const bound = new Set<string>();
    const bind = vi.fn((name: string) => (bound.add(name), true));
    const coordinator = new ModelWorkCoordinator();
    const gate = deferred();
    const held = coordinator.withModelLoad(() => gate.promise);
    expect(coordinator.writerActive).toBe(true);
    expect(coordinator.waitingWriters).toBe(0);

    expect(residentSessionRegistry(registry(bound), 'mlx-local', bind, coordinator)).toBeUndefined();
    expect(bind).not.toHaveBeenCalled();

    gate.resolve();
    await held;
    expect(residentSessionRegistry(registry(bound), 'mlx-local', bind, coordinator)).toBe(lane);
    expect(bind).toHaveBeenCalledTimes(1);
  });
});
