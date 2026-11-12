import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import type { SessionCapableModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { createHandler } from '../../packages/server/src/handler.js';
import { createIdleSweeper } from '../../packages/server/src/idle-sweeper.js';
import { ModelWorkCoordinator } from '../../packages/server/src/model-work-coordinator.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

/**
 * `POST /v1/models/unload` at the handler: the token gate, the method, the busy refusal that never waits, the
 * names it returns, and the idle-sweeper bracket that clears the allocator cache on its usual delay.
 */

const TOKEN = 'unload-test-token';

let servers: Server[] = [];

afterEach(async () => {
  const open = servers;
  servers = [];
  for (const server of open) {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

async function serve(options: {
  unloadModels?: () => Promise<string[]>;
  coordinator?: ModelWorkCoordinator;
  registry?: ModelRegistry;
  onDrain?: () => void;
}): Promise<string> {
  const handler = createHandler(options.registry ?? new ModelRegistry(), {
    authToken: TOKEN,
    modelWorkCoordinator: options.coordinator ?? new ModelWorkCoordinator(),
    idleSweeper: createIdleSweeper(10, options.onDrain ?? (() => {})),
    ...(options.unloadModels === undefined ? {} : { unloadModels: options.unloadModels }),
  });
  const server = createServer((req, res) => void handler(req, res));
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1/models/unload`;
}

function answeringModel(): SessionCapableModel {
  const result = {
    text: 'Hello!',
    toolCalls: [],
    thinkingEnabled: false,
    numTokens: 2,
    promptTokens: 3,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'Hello!',
    cachedTokens: 0,
    performance: undefined,
  };
  return {
    chatSessionStart: vi.fn().mockResolvedValue(result),
    chatSessionContinue: vi.fn().mockResolvedValue(result),
    chatSessionContinueTool: vi.fn().mockResolvedValue(result),
    resetCaches: vi.fn().mockResolvedValue(undefined),
  } as unknown as SessionCapableModel;
}

const post = (url: string, token: string | null = TOKEN, method = 'POST'): Promise<Response> =>
  fetch(url, { method, headers: token === null ? {} : { 'x-api-key': token } });

describe('POST /v1/models/unload', () => {
  it('needs the token and POST', async () => {
    const unloadModels = vi.fn(async () => ['beta']);
    const url = await serve({ unloadModels });
    expect((await post(url, null)).status).toBe(401);
    expect((await post(url, 'wrong')).status).toBe(401);
    expect((await post(url, TOKEN, 'GET')).status).toBe(405);
    expect(unloadModels).not.toHaveBeenCalled();
  });

  it('idle: unloads under the writer slot, names what it unloaded, and arms the cache clear', async () => {
    const coordinator = new ModelWorkCoordinator();
    let held = false;
    const unloadModels = vi.fn(async () => {
      held = coordinator.writerActive;
      return ['beta'];
    });
    let drains = 0;
    const url = await serve({ unloadModels, coordinator, onDrain: () => (drains += 1) });
    const res = await post(url);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ unloaded: ['beta'] });
    expect(held).toBe(true);
    expect(coordinator.lastLoad).toMatchObject({ label: 'unload', ok: true });
    await vi.waitFor(() => expect(drains).toBe(1));
  });

  it('answers an empty list when nothing is in memory', async () => {
    const url = await serve({ unloadModels: async () => [] });
    expect(await (await post(url)).json()).toEqual({ unloaded: [] });
  });

  it.each(['reader', 'writer', 'queued writer'] as const)(
    'a %s on the engine is a 409 at once, and nothing is unloaded',
    async (kind) => {
      const coordinator = new ModelWorkCoordinator();
      let release = (): void => {};
      const hold = new Promise<void>((resolve) => {
        release = resolve;
      });
      const work: Array<Promise<unknown>> = [];
      if (kind === 'reader') work.push(coordinator.withInference(() => hold));
      if (kind === 'writer') work.push(coordinator.withModelLoad(() => hold));
      if (kind === 'queued writer') {
        work.push(coordinator.withInference(() => hold));
        work.push(coordinator.withModelLoad(async () => {}));
      }
      const unloadModels = vi.fn(async () => ['beta']);
      const url = await serve({ unloadModels, coordinator });
      const res = await post(url);
      expect(res.status).toBe(409);
      expect(unloadModels).not.toHaveBeenCalled();
      release();
      await Promise.all(work);
      expect((await post(url)).status).toBe(200);
    },
  );

  it('a request that holds its dispatch lease but no reader slot yet is a 409, and its answer completes', async () => {
    const registry = new ModelRegistry();
    registry.register('a', answeringModel());
    const coordinator = new ModelWorkCoordinator();
    const sessions = registry.getSessionRegistry('a')!;
    let openProbe = (): void => {};
    const probe = sessions.tryIdleProbe(
      () =>
        new Promise<void>((resolve) => {
          openProbe = resolve;
        }),
    );
    expect(probe).not.toBeNull();
    const unloadModels = vi.fn(async () => {
      registry.unregister('a');
      return ['a'];
    });
    const url = await serve({ unloadModels, coordinator, registry });
    const answer = fetch(url.replace('/v1/models/unload', '/v1/messages'), {
      method: 'POST',
      headers: { 'x-api-key': TOKEN, 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'a', max_tokens: 8, messages: [{ role: 'user', content: 'hi' }] }),
    });
    await vi.waitFor(() => expect(sessions.activeDispatches).toBe(1));
    expect(coordinator.readerCount).toBe(0);
    const refused = await post(url);
    expect(refused.status).toBe(409);
    expect(unloadModels).not.toHaveBeenCalled();
    openProbe();
    await probe;
    const res = await answer;
    expect(res.status).toBe(200);
    expect(((await res.json()) as { content: Array<{ text?: string }> }).content[0]?.text).toBe('Hello!');
    await vi.waitFor(() => expect(registry.dispatchLeaseCount).toBe(0));
    expect(await (await post(url)).json()).toEqual({ unloaded: ['a'] });
  });

  it('is not a route on a handler with no unloader', async () => {
    const url = await serve({});
    expect((await post(url)).status).toBe(404);
  });
});
