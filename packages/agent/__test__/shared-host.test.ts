import type { ChatConfig, ChatMessage, LoadableModel } from '@mlx-node/lm';
import { describe, expect, it, vi, type MockInstance } from 'vite-plus/test';

import { MlxModelHost } from '../src/provider/model-host.js';
import { SharedInferenceHost } from '../src/provider/shared-host.js';
import type { SharedProfile } from '../src/provider/shared-protocol.js';
import { SHARED_SESSION_LIMIT } from '../src/provider/shared-protocol.js';
import { resetPreservingNativeCacheForWarmReuse } from '../src/provider/warm-reuse.js';

const profile: SharedProfile = {
  discovered: { name: 'local', path: '/models/local', modelType: 'qwen3' },
  persistPagedCache: true,
  preserveEmbeddedGemmaDraft: false,
};
function gate() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function fixture(capacity = 2, paged = true, overrides: Record<string, unknown> = {}) {
  const loader = vi.fn(
    async () =>
      ({
        hasBlockPagedCache: () => paged,
        maxConcurrentSequences: () => capacity,
        resetCaches: vi.fn(),
        ...overrides,
      }) as unknown as LoadableModel,
  );
  const host = new SharedInferenceHost((config) => new MlxModelHost([config.discovered], { loadModelFn: loader }));
  const signal = new AbortController().signal;
  return { host, loader, request: (config = profile) => host.forRequest(config, signal) };
}

describe('shared delegate model ownership', () => {
  it('retains native owner state across turns and releases it on shutdown', async () => {
    const cached = new Map<string, number>();
    const releaseCacheOwner = vi.fn((owner: string) => {
      cached.delete(owner);
    });
    const chatSessionStart = vi.fn(async (messages: ChatMessage[], config: ChatConfig) => {
      const cachedTokens = cached.get(config.cacheOwnerId!) ?? 0;
      cached.set(config.cacheOwnerId!, messages.length);
      return {
        text: 'answer',
        toolCalls: [],
        thinking: null,
        thinkingEnabled: false,
        rawText: 'answer',
        finishReason: 'stop',
        numTokens: 1,
        promptTokens: messages.length,
        cachedTokens,
      };
    });
    const { host, request } = fixture(2, true, { releaseCacheOwner, chatSessionStart });
    const owner = 'client/session';
    const turn = (messages: ChatMessage[]) =>
      request().runWithResident(
        'local',
        async (session) => {
          await resetPreservingNativeCacheForWarmReuse(session);
          session.primeHistory(messages);
          return session.startFromHistory({ cacheOwnerId: owner, cacheRootOwnerId: 'client/root' });
        },
        owner,
      );
    try {
      expect((await turn([{ role: 'user', content: 'first' }])).cachedTokens).toBe(0);
      expect(
        (
          await turn([
            { role: 'user', content: 'first' },
            { role: 'assistant', content: 'answer' },
            { role: 'user', content: 'next' },
          ])
        ).cachedTokens,
      ).toBe(1);
      expect(releaseCacheOwner).not.toHaveBeenCalled();
      expect(chatSessionStart.mock.calls.map(([, config]) => config.cacheRootOwnerId)).toEqual([
        'client/root',
        'client/root',
      ]);
    } finally {
      await host.close();
    }
    expect(releaseCacheOwner).toHaveBeenCalledExactlyOnceWith(owner);
    expect(cached.size).toBe(0);
  });

  it('serializes the same owner while allowing another owner to run concurrently', async () => {
    const { host, request } = fixture();
    const entered = gate();
    const release = gate();
    const otherEntered = gate();
    const sessions: unknown[] = [];
    const first = request().runWithResident(
      'local',
      async (session) => {
        sessions.push(session);
        entered.resolve();
        await release.promise;
      },
      'same',
    );
    await entered.promise;
    const repeated = request().runWithResident(
      'local',
      async (session) => {
        sessions.push(session);
      },
      'same',
    );
    const other = request().runWithResident(
      'local',
      async () => {
        otherEntered.resolve();
      },
      'other',
    );
    try {
      await otherEntered.promise;
      expect(sessions).toHaveLength(1);
    } finally {
      release.resolve();
      await Promise.all([first, repeated, other]);
      await host.close();
    }
    expect(sessions).toHaveLength(2);
    expect(sessions[1]).toBe(sessions[0]);
  });

  it('bounds warm owners and releases evicted sessions', async () => {
    const { host, request } = fixture();
    const disposals: MockInstance<() => Promise<void>>[] = [];
    try {
      for (let i = 0; i <= SHARED_SESSION_LIMIT; i++) {
        await request().runWithResident(
          'local',
          async (session) => {
            disposals.push(vi.spyOn(session, 'dispose'));
          },
          `owner-${i}`,
        );
      }
      expect(disposals[0]).toHaveBeenCalledOnce();
      for (const dispose of disposals.slice(1)) expect(dispose).not.toHaveBeenCalled();
    } finally {
      await host.close();
    }
    for (const dispose of disposals) expect(dispose).toHaveBeenCalledOnce();
  });

  it('discards a dirty session before admitting the next turn for its owner', async () => {
    const { host, request } = fixture();
    const failed = request();
    let previous: unknown;
    let dispose: MockInstance<() => Promise<void>> | undefined;
    await failed.runWithResident(
      'local',
      async (session) => {
        previous = session;
        dispose = vi.spyOn(session, 'dispose');
        failed.markResidentDirty('local');
      },
      'owner',
    );
    expect(dispose).toHaveBeenCalledOnce();
    try {
      await request().runWithResident(
        'local',
        async (session) => {
          expect(session).not.toBe(previous);
        },
        'owner',
      );
    } finally {
      await host.close();
    }
  });

  it('waits for eviction to release an owner before admitting its next turn', async () => {
    const { host, request } = fixture();
    const evicting = gate();
    const released = gate();
    for (let i = 0; i < SHARED_SESSION_LIMIT; i++) {
      await request().runWithResident(
        'local',
        async (session) => {
          if (i === 0) {
            const dispose = session.dispose.bind(session);
            vi.spyOn(session, 'dispose').mockImplementation(async () => {
              evicting.resolve();
              await released.promise;
              await dispose();
            });
          }
        },
        `owner-${i}`,
      );
    }
    const next = request().runWithResident('local', async () => {}, 'new-owner');
    await evicting.promise;
    const resumed = vi.fn(async () => {});
    const previous = request().runWithResident('local', resumed, 'owner-0');
    try {
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(resumed).not.toHaveBeenCalled();
    } finally {
      released.resolve();
      await Promise.all([next, previous]);
      await host.close();
    }
    expect(resumed).toHaveBeenCalledOnce();
  });

  it('loads once and admits independent sessions concurrently on a supported scheduler', async () => {
    const { host, loader, request } = fixture();
    const entered = gate();
    const release = gate();
    const sessions: unknown[] = [];
    const run = () =>
      request().runWithResident('local', async (session) => {
        sessions.push(session);
        session.primeHistory([{ role: 'user', content: `task ${sessions.length}` }]);
        if (sessions.length === 2) entered.resolve();
        await release.promise;
      });
    const first = run();
    const second = run();
    try {
      await entered.promise;
      expect(sessions[0]).not.toBe(sessions[1]);
      expect(loader).toHaveBeenCalledOnce();
      expect(host.pending).toBe(2);
    } finally {
      release.resolve();
      await Promise.all([first, second]);
      await host.close();
    }
  });

  it.each([
    [1, true],
    [8, false],
  ])('serializes unsupported models (%i, paged %s)', async (capacity, paged) => {
    const { host, request } = fixture(capacity, paged);
    const entered = gate();
    const release = gate();
    const order: string[] = [];
    const first = request().runWithResident('local', async () => {
      order.push('first');
      entered.resolve();
      await release.promise;
      order.push('end');
    });
    await entered.promise;
    const second = request().runWithResident('local', async () => {
      order.push('second');
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(order).toEqual(['first']);
    release.resolve();
    await Promise.all([first, second]);
    expect(order).toEqual(['first', 'end', 'second']);
    await host.close();
  });

  it('drains active sessions before switching models or cache policy', async () => {
    const { host, loader, request } = fixture();
    const entered = gate();
    const release = gate();
    const first = request().runWithResident('local', async () => {
      entered.resolve();
      await release.promise;
    });
    await entered.promise;
    const next = request({ ...profile, persistPagedCache: false }).runWithResident('local', async () => {});
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(loader).toHaveBeenCalledOnce();
    release.resolve();
    await Promise.all([first, next]);
    expect(loader).toHaveBeenCalledTimes(2);
    await host.close();
  });

  it('skips a cancelled queued turn without loading its model or touching a session', async () => {
    const { host, loader, request } = fixture(1);
    const entered = gate();
    const release = gate();
    const first = request().runWithResident('local', async () => {
      entered.resolve();
      await release.promise;
    });
    await entered.promise;
    const controller = new AbortController();
    const fn = vi.fn();
    const queued = host.forRequest(profile, controller.signal).runWithResident('local', fn);
    const rejected = expect(queued).rejects.toThrow();
    controller.abort();
    release.resolve();
    await first;
    await rejected;
    expect(fn).not.toHaveBeenCalled();
    expect(loader).toHaveBeenCalledOnce();
    expect(host.pending).toBe(0);
    await host.close();
  });
});
