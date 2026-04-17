import type { ChatResult } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import { QueueFullError, SessionRegistry } from '@mlx-node/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

/**
 * Build a minimal `SessionCapableModel` stub whose methods return
 * predictable `ChatResult` / `ChatStreamEvent` shapes. The mock only
 * needs to satisfy the structural shape well enough for
 * `new ChatSession(mock)` to be constructable — the registry's unit
 * tests never actually drive a turn through the session.
 */
function makeMockModel(): SessionCapableModel {
  const result: ChatResult = {
    text: 'ok',
    toolCalls: [],
    thinking: undefined,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'eos',
    rawText: 'ok',
  };
  const finalEvent = {
    text: 'ok',
    done: true as const,
    finishReason: 'eos',
    toolCalls: [],
    thinking: null,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    rawText: 'ok',
  };
  return {
    chatSessionStart: async () => result,
    chatSessionContinue: async () => result,
    chatSessionContinueTool: async () => result,
    // eslint-disable-next-line @typescript-eslint/require-await
    chatStreamSessionStart: async function* () {
      yield finalEvent;
    },
    // eslint-disable-next-line @typescript-eslint/require-await
    chatStreamSessionContinue: async function* () {
      yield finalEvent;
    },
    // eslint-disable-next-line @typescript-eslint/require-await
    chatStreamSessionContinueTool: async function* () {
      yield finalEvent;
    },
    resetCaches: () => {},
  } as unknown as SessionCapableModel;
}

describe('SessionRegistry', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('getOrCreate(null) returns a fresh session without caching', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });

    const { session, hit } = reg.getOrCreate(null, null);

    expect(session).toBeInstanceOf(ChatSession);
    expect(hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('getOrCreate on a missing key returns a fresh session without caching', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });

    const { session, hit } = reg.getOrCreate('resp_missing', null);

    expect(session).toBeInstanceOf(ChatSession);
    expect(hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('adopt inserts a session and getOrCreate leases it out on hit (single-use)', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);

    expect(reg.size).toBe(1);
    // First hit returns the cached instance and flags `hit: true`.
    const hit1 = reg.getOrCreate('resp_1', null);
    expect(hit1.session).toBe(s1);
    expect(hit1.hit).toBe(true);
    // ...and the entry is now GONE (lease semantics).
    expect(reg.size).toBe(0);
    // A subsequent lookup against the same id misses and returns a
    // fresh ChatSession — cold replay responsibility is on the caller.
    const miss = reg.getOrCreate('resp_1', null);
    expect(miss.session).not.toBe(s1);
    expect(miss.session).toBeInstanceOf(ChatSession);
    expect(miss.hit).toBe(false);
  });

  it('overlapping getOrCreate against the same id cannot share a live session', () => {
    // Regression for the ChatSession single-flight race: two requests
    // referencing the same `previous_response_id` must not both receive
    // the same live ChatSession object. The first wins the lease, the
    // second misses and cold-replays.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);
    reg.adopt('resp_concurrent', s1, null);

    const a = reg.getOrCreate('resp_concurrent', null);
    const b = reg.getOrCreate('resp_concurrent', null);

    expect(a.session).toBe(s1);
    expect(a.hit).toBe(true);
    // The second caller gets an independent fresh session — it is
    // responsible for priming/cold-replaying from the ResponseStore.
    expect(b.session).not.toBe(s1);
    expect(b.session).toBeInstanceOf(ChatSession);
    expect(b.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('getOrCreate with matching instructions returns the cached session', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be brief');
    const got = reg.getOrCreate('resp_1', 'be brief');
    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('getOrCreate with mismatched instructions evicts and returns a fresh session', () => {
    // A cache hit with new `instructions` must fall through to cold replay; otherwise
    // a warmed session reuses stale system context while a cold miss replays the new one.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be brief');
    const got = reg.getOrCreate('resp_1', 'be verbose');

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    // Entry was evicted by the mismatch check.
    expect(reg.size).toBe(0);
  });

  it('null instructions match null instructions', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    const got = reg.getOrCreate('resp_1', null);
    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('adopt(..., null) then getOrCreate(..., "foo") is a mismatch', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    const got = reg.getOrCreate('resp_1', 'be brief');

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('adopt(..., "foo") then getOrCreate(..., null) is a mismatch', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be brief');
    const got = reg.getOrCreate('resp_1', null);

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('evicts entries whose TTL has expired on lookup', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    expect(reg.size).toBe(1);

    // Advance past the TTL window.
    vi.advanceTimersByTime(61 * 1000);

    const got = reg.getOrCreate('resp_1', null);
    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('adopt evicts any prior entry under the single-warm invariant', () => {
    // Native KV state for one SessionCapableModel is a single shared
    // mutable resource — at most ONE cached `ChatSession` wrapper can
    // reflect that state at a time. `adopt` therefore clears the map
    // before inserting so a later `getOrCreate` cannot hand out a
    // wrapper whose assumed state has been stomped by a turn on
    // another entry. This test pins that contract: adopting B MUST
    // drop A, and a later lookup against A MUST miss and cold-replay.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const sA = new ChatSession(model);
    const sB = new ChatSession(model);

    reg.adopt('a', sA, null);
    expect(reg.size).toBe(1);

    reg.adopt('b', sB, null);
    // B is live. A has been evicted.
    expect(reg.size).toBe(1);

    const aMiss = reg.getOrCreate('a', null);
    expect(aMiss.session).not.toBe(sA);
    expect(aMiss.session).toBeInstanceOf(ChatSession);
    expect(aMiss.hit).toBe(false);
    // Looking up 'a' cleared the map for the single-warm invariant
    // (every `getOrCreate` hand-off drops any other entry so the next
    // lookup cannot trip over a stale wrapper).
    expect(reg.size).toBe(0);
  });

  it('getOrCreate(null) clears any prior entry', () => {
    // The single-warm invariant: any `getOrCreate` call is about to
    // run a turn that overwrites the model's shared native KV cache,
    // so the registry drops whatever is left in the map.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const sA = new ChatSession(model);

    reg.adopt('a', sA, null);
    expect(reg.size).toBe(1);

    const fresh = reg.getOrCreate(null, null);
    expect(fresh.session).toBeInstanceOf(ChatSession);
    expect(fresh.session).not.toBe(sA);
    expect(fresh.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('getOrCreate on a miss clears any prior entry', () => {
    // Same invariant via the lookup-miss path: the caller is about to
    // run a turn, so any leftover entry must be dropped regardless of
    // whether the lookup hit or missed.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const sA = new ChatSession(model);

    reg.adopt('a', sA, null);
    expect(reg.size).toBe(1);

    const fresh = reg.getOrCreate('resp_unknown', null);
    expect(fresh.session).toBeInstanceOf(ChatSession);
    expect(fresh.session).not.toBe(sA);
    expect(fresh.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('adopt overwrites an existing key and refreshes expiry', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const s1 = new ChatSession(model);
    const s2 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    // Advance near expiry so we can verify the TTL got refreshed.
    vi.advanceTimersByTime(50 * 1000);
    reg.adopt('resp_1', s2, null);

    expect(reg.size).toBe(1);
    expect(reg.getOrCreate('resp_1', null).session).toBe(s2);

    // (Re-adopt so a second lookup has something to find.)
    reg.adopt('resp_1', s2, null);

    // Original expiry was 60s after t=0; refreshed expiry is 60s after
    // t=50s, so at t=80s the entry should still be live.
    vi.advanceTimersByTime(30 * 1000);
    expect(reg.getOrCreate('resp_1', null).session).toBe(s2);
  });

  it('drop removes an entry and subsequent getOrCreate misses', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    reg.drop('resp_1');

    expect(reg.size).toBe(0);
    const got = reg.getOrCreate('resp_1', null);
    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);

    // drop on an unknown key is a no-op.
    expect(() => reg.drop('nonexistent')).not.toThrow();
  });

  it('sweep drops the entry when its TTL has expired', () => {
    // Under the single-warm invariant the registry holds at most one
    // entry, so `sweep()` is effectively "check if the one entry is
    // stale and drop it if so". Scheduling sweep on an interval keeps
    // the map bounded even when no `getOrCreate` / `adopt` traffic
    // comes through after an entry goes stale.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const sA = new ChatSession(model);

    reg.adopt('a', sA, null);
    expect(reg.size).toBe(1);

    // Advance past the TTL window and sweep.
    vi.advanceTimersByTime(61 * 1000);
    reg.sweep();

    expect(reg.size).toBe(0);
    // Subsequent lookup misses and returns a fresh session.
    const got = reg.getOrCreate('a', null);
    expect(got.session).not.toBe(sA);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
  });

  it('sweep is a no-op when the entry is still fresh', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const sA = new ChatSession(model);

    reg.adopt('a', sA, null);
    // Advance partially through the TTL window.
    vi.advanceTimersByTime(30 * 1000);
    reg.sweep();

    expect(reg.size).toBe(1);
    expect(reg.getOrCreate('a', null).session).toBe(sA);
  });

  it('clear empties the registry', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    reg.adopt('a', new ChatSession(model), null);

    expect(reg.size).toBe(1);
    reg.clear();
    expect(reg.size).toBe(0);
  });

  describe('withExclusive', () => {
    beforeEach(() => {
      // The withExclusive tests drive two concurrent dispatches
      // through microtask interleaving — fake timers from the
      // outer describe would stall the native-promise chain and
      // produce false serialization. Restore real timers just for
      // this block.
      vi.useRealTimers();
    });

    it('serializes two overlapping dispatches against the same registry', async () => {
      // `/v1/responses` and `/v1/messages` can arrive in overlapping ticks for the same
      // model; both dispatches hold a `ChatSession` pointing at the SAME shared native
      // model. The mutex must serialize them so at most one dispatch owns the model at
      // a time — otherwise concurrent primeHistory/send* calls clobber KV state.
      const model = makeMockModel();
      const reg = new SessionRegistry({ model });

      const events: string[] = [];
      // The two closures both block on an externally controlled
      // promise so the test can pin the ordering. If the mutex
      // serialized correctly, the second dispatch does NOT
      // observe its own start event before the first has
      // resolved — only after `releaseA` fires.
      let releaseA!: () => void;
      const aDone = new Promise<void>((r) => {
        releaseA = r;
      });

      const dispatchA = reg.withExclusive(async () => {
        events.push('A:start');
        await aDone;
        events.push('A:end');
      });

      const dispatchB = reg.withExclusive(async () => {
        events.push('B:start');
        events.push('B:end');
      });

      // Yield to the microtask queue twice so any incorrect
      // interleaving would already have recorded both "A:start"
      // AND "B:start" here — B is still blocked on the chained
      // `prev` promise from the mutex, so only "A:start" is
      // visible.
      await Promise.resolve();
      await Promise.resolve();
      expect(events).toEqual(['A:start']);

      // Resolving A's gate lets A end, then releases B.
      releaseA();
      await dispatchA;
      await dispatchB;

      expect(events).toEqual(['A:start', 'A:end', 'B:start', 'B:end']);
    });

    it('releases the mutex when the closure throws', async () => {
      // A dispatch that errors out inside the lock must still
      // release so the next waiter is not stuck forever. The
      // `withExclusive` implementation uses a try/finally around
      // the closure specifically to cover this path.
      const model = makeMockModel();
      const reg = new SessionRegistry({ model });

      const events: string[] = [];

      const failing = reg.withExclusive(async () => {
        events.push('fail:start');
        // Dummy await so the body is genuinely async.
        await Promise.resolve();
        throw new Error('boom');
      });

      const following = reg.withExclusive(async () => {
        events.push('ok:start');
        events.push('ok:end');
      });

      await expect(failing).rejects.toThrow('boom');
      await following;

      expect(events).toEqual(['fail:start', 'ok:start', 'ok:end']);
    });

    it('preserves FIFO ordering across three waiters', async () => {
      // Sanity check: the mutex is a FIFO chain, not a
      // "first-to-await-wins" race. Three overlapping dispatches
      // must run in the exact order they called `withExclusive`.
      const model = makeMockModel();
      const reg = new SessionRegistry({ model });

      const events: string[] = [];
      const dispatches = [1, 2, 3].map((i) =>
        reg.withExclusive(async () => {
          events.push(`start:${i}`);
          // Yield twice so an incorrect implementation has a
          // chance to interleave with the other closures.
          await Promise.resolve();
          await Promise.resolve();
          events.push(`end:${i}`);
        }),
      );

      await Promise.all(dispatches);

      expect(events).toEqual(['start:1', 'end:1', 'start:2', 'end:2', 'start:3', 'end:3']);
    });

    it('queuedCount is 0 initially and tracks waiters while withExclusive calls are pending', async () => {
      // Admission-control observability: `queueDepth` must reflect the
      // count of callers WAITING for the mutex, not the one actively
      // running. A single dispatch parked behind a held-open closure
      // raises the counter by exactly one; once the holder releases,
      // the counter drops to zero (the next caller is now active, not
      // waiting).
      const model = makeMockModel();
      const reg = new SessionRegistry({ model });

      expect(reg.queueDepth).toBe(0);

      let releaseA!: () => void;
      const aDone = new Promise<void>((r) => {
        releaseA = r;
      });

      const dispatchA = reg.withExclusive(async () => {
        // The active holder does NOT contribute to `queueDepth` — it
        // already transitioned from waiting to running on entry.
        expect(reg.queueDepth).toBe(0);
        await aDone;
      });

      // Yield so A can enter the mutex and decrement past its own
      // waiting state.
      await Promise.resolve();
      await Promise.resolve();
      expect(reg.queueDepth).toBe(0);

      // Second caller queues behind A.
      const dispatchB = reg.withExclusive(async () => {
        // When B finally runs, it has also transitioned to active
        // and is no longer counted as queued.
        expect(reg.queueDepth).toBe(0);
      });

      // B is parked — it is the only waiter now.
      await Promise.resolve();
      await Promise.resolve();
      expect(reg.queueDepth).toBe(1);

      // Release A, let B run through.
      releaseA();
      await dispatchA;
      await dispatchB;

      expect(reg.queueDepth).toBe(0);
    });

    it('throws QueueFullError when queue exceeds maxQueueDepth', async () => {
      // Cap semantics: `maxQueueDepth = N` permits one running
      // dispatch plus N waiting dispatches. The (N+1)th waiter is
      // rejected synchronously with `QueueFullError`.
      const model = makeMockModel();
      const reg = new SessionRegistry({ model, maxQueueDepth: 1 });

      let releaseA!: () => void;
      const aDone = new Promise<void>((r) => {
        releaseA = r;
      });

      // A becomes the active holder.
      const dispatchA = reg.withExclusive(async () => {
        await aDone;
      });
      await Promise.resolve();
      await Promise.resolve();

      // B queues behind A — within the cap (1 waiting).
      const dispatchB = reg.withExclusive(async () => {
        // no-op
      });
      await Promise.resolve();

      // C would be the SECOND waiter (cap is 1); admission is
      // rejected synchronously before any await.
      expect(() => reg.withExclusive(async () => {})).toThrowError(QueueFullError);

      // The rejection path MUST NOT mutate the queued counter — the
      // queue depth is still exactly 1 (just B).
      expect(reg.queueDepth).toBe(1);

      releaseA();
      await dispatchA;
      await dispatchB;
    });

    it('attaches queuedCount and limit to the thrown QueueFullError', async () => {
      // The error needs to carry both numbers so endpoint handlers
      // can surface them to the client in the 429 body.
      const model = makeMockModel();
      const reg = new SessionRegistry({ model, maxQueueDepth: 1 });

      let releaseA!: () => void;
      const aDone = new Promise<void>((r) => {
        releaseA = r;
      });

      const dispatchA = reg.withExclusive(async () => {
        await aDone;
      });
      await Promise.resolve();

      const dispatchB = reg.withExclusive(async () => {});
      await Promise.resolve();

      let caught: unknown;
      try {
        await reg.withExclusive(async () => {});
      } catch (err) {
        caught = err;
      }
      expect(caught).toBeInstanceOf(QueueFullError);
      const queueErr = caught as QueueFullError;
      expect(queueErr.queuedCount).toBe(1);
      expect(queueErr.limit).toBe(1);
      expect(queueErr.message).toContain('1 waiting (limit 1)');

      releaseA();
      await dispatchA;
      await dispatchB;
    });

    it('queuedCount decrements after a rejected waiter releases', async () => {
      // A rejected waiter did not actually queue, so the counter
      // stays stable. After the current holder + legitimate waiters
      // drain, `queueDepth` returns to 0 so the next burst of traffic
      // can refill from 0 (rather than silently carrying drift from
      // the rejected attempt).
      const model = makeMockModel();
      const reg = new SessionRegistry({ model, maxQueueDepth: 1 });

      let releaseA!: () => void;
      const aDone = new Promise<void>((r) => {
        releaseA = r;
      });

      const dispatchA = reg.withExclusive(async () => {
        await aDone;
      });
      await Promise.resolve();

      const dispatchB = reg.withExclusive(async () => {});
      await Promise.resolve();
      expect(reg.queueDepth).toBe(1);

      // Rejected admission does not increment.
      expect(() => reg.withExclusive(async () => {})).toThrowError(QueueFullError);
      expect(reg.queueDepth).toBe(1);

      releaseA();
      await dispatchA;
      await dispatchB;

      // Queue fully drained.
      expect(reg.queueDepth).toBe(0);

      // A fresh waiter can now queue again — the cap's slot is free.
      let releaseC!: () => void;
      const cDone = new Promise<void>((r) => {
        releaseC = r;
      });
      const dispatchC = reg.withExclusive(async () => {
        await cDone;
      });
      await Promise.resolve();
      await Promise.resolve();
      // D now queues within cap.
      const dispatchD = reg.withExclusive(async () => {});
      await Promise.resolve();
      expect(reg.queueDepth).toBe(1);
      releaseC();
      await dispatchC;
      await dispatchD;
    });

    it('unbounded mode (maxQueueDepth === undefined) never throws', async () => {
      // The default behaviour: with no cap configured, no amount of
      // queued waiters ever triggers `QueueFullError`. This pins the
      // opt-in contract — the feature MUST stay off by default.
      const model = makeMockModel();
      const reg = new SessionRegistry({ model });

      let releaseA!: () => void;
      const aDone = new Promise<void>((r) => {
        releaseA = r;
      });
      const dispatchA = reg.withExclusive(async () => {
        await aDone;
      });
      await Promise.resolve();

      const extras: Promise<void>[] = [];
      for (let i = 0; i < 64; i += 1) {
        // Every one of these MUST queue, not throw.
        extras.push(reg.withExclusive(async () => {}));
      }
      await Promise.resolve();
      expect(reg.queueDepth).toBe(64);

      releaseA();
      await dispatchA;
      await Promise.all(extras);
      expect(reg.queueDepth).toBe(0);
    });
  });
});
