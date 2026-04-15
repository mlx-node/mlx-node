import type { ChatResult } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import { SessionRegistry } from '@mlx-node/server';
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

    const session = reg.getOrCreate(null, null);

    expect(session).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('getOrCreate on a missing key returns a fresh session without caching', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });

    const session = reg.getOrCreate('resp_missing', null);

    expect(session).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('adopt inserts a session and getOrCreate leases it out on hit (single-use)', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);

    expect(reg.size).toBe(1);
    // First hit returns the cached instance.
    expect(reg.getOrCreate('resp_1', null)).toBe(s1);
    // ...and the entry is now GONE (lease semantics).
    expect(reg.size).toBe(0);
    // A subsequent lookup against the same id misses and returns a
    // fresh ChatSession — cold replay responsibility is on the caller.
    const fresh = reg.getOrCreate('resp_1', null);
    expect(fresh).not.toBe(s1);
    expect(fresh).toBeInstanceOf(ChatSession);
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

    expect(a).toBe(s1);
    // The second caller gets an independent fresh session — it is
    // responsible for priming/cold-replaying from the ResponseStore.
    expect(b).not.toBe(s1);
    expect(b).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('getOrCreate with matching instructions returns the cached session', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be brief');
    expect(reg.getOrCreate('resp_1', 'be brief')).toBe(s1);
  });

  it('getOrCreate with mismatched instructions evicts and returns a fresh session', () => {
    // Finding 1 regression: a cache hit with new `instructions` must
    // fall through to cold replay. Returning the warmed session would
    // silently reuse the stale system context while a cold miss would
    // replay the new one — making output depend on LRU state.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be brief');
    const got = reg.getOrCreate('resp_1', 'be verbose');

    expect(got).not.toBe(s1);
    expect(got).toBeInstanceOf(ChatSession);
    // Entry was evicted by the mismatch check.
    expect(reg.size).toBe(0);
  });

  it('null instructions match null instructions', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    expect(reg.getOrCreate('resp_1', null)).toBe(s1);
  });

  it('adopt(..., null) then getOrCreate(..., "foo") is a mismatch', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    const got = reg.getOrCreate('resp_1', 'be brief');

    expect(got).not.toBe(s1);
    expect(got).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('adopt(..., "foo") then getOrCreate(..., null) is a mismatch', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be brief');
    const got = reg.getOrCreate('resp_1', null);

    expect(got).not.toBe(s1);
    expect(got).toBeInstanceOf(ChatSession);
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
    expect(got).not.toBe(s1);
    expect(got).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('evicts the least-recently-adopted entry when maxEntries overflows', () => {
    // With lease semantics, LRU ordering is driven entirely by
    // `adopt()` — there is no in-place promotion on hit. This test
    // adopts three entries in staggered order and verifies the oldest
    // is the eviction victim.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, maxEntries: 2 });
    const sA = new ChatSession(model);
    const sB = new ChatSession(model);
    const sC = new ChatSession(model);

    reg.adopt('a', sA, null);
    reg.adopt('b', sB, null);
    // 'c' displaces 'a' (the oldest adopt), leaving {b, c}.
    reg.adopt('c', sC, null);

    expect(reg.size).toBe(2);
    // 'a' was evicted — lookup returns a fresh miss.
    const aMiss = reg.getOrCreate('a', null);
    expect(aMiss).not.toBe(sA);
    expect(aMiss).toBeInstanceOf(ChatSession);
    // But only AFTER 'a' is gone — b and c are still there.
    expect(reg.getOrCreate('b', null)).toBe(sB);
    expect(reg.getOrCreate('c', null)).toBe(sC);
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
    expect(reg.getOrCreate('resp_1', null)).toBe(s2);

    // (Re-adopt so a second lookup has something to find.)
    reg.adopt('resp_1', s2, null);

    // Original expiry was 60s after t=0; refreshed expiry is 60s after
    // t=50s, so at t=80s the entry should still be live.
    vi.advanceTimersByTime(30 * 1000);
    expect(reg.getOrCreate('resp_1', null)).toBe(s2);
  });

  it('drop removes an entry and subsequent getOrCreate misses', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null);
    reg.drop('resp_1');

    expect(reg.size).toBe(0);
    const got = reg.getOrCreate('resp_1', null);
    expect(got).not.toBe(s1);
    expect(got).toBeInstanceOf(ChatSession);

    // drop on an unknown key is a no-op.
    expect(() => reg.drop('nonexistent')).not.toThrow();
  });

  it('sweep drops only the entries whose TTL has expired', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const sA = new ChatSession(model);
    const sB = new ChatSession(model);
    const sC = new ChatSession(model);

    reg.adopt('a', sA, null);
    // Stagger the adopts so 'a' expires before 'b' and 'c'.
    vi.advanceTimersByTime(30 * 1000);
    reg.adopt('b', sB, null);
    reg.adopt('c', sC, null);
    // t=30s. 'a' expires at 60s, 'b' and 'c' at 90s.
    vi.advanceTimersByTime(40 * 1000);
    // t=70s. Sweep should drop 'a' but keep 'b' and 'c'.
    reg.sweep();

    expect(reg.size).toBe(2);
    expect(reg.getOrCreate('a', null)).not.toBe(sA);
    expect(reg.getOrCreate('b', null)).toBe(sB);
    expect(reg.getOrCreate('c', null)).toBe(sC);
  });

  it('clear empties the registry', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    reg.adopt('a', new ChatSession(model), null);
    reg.adopt('b', new ChatSession(model), null);

    expect(reg.size).toBe(2);
    reg.clear();
    expect(reg.size).toBe(0);
  });
});
