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

    const session = reg.getOrCreate(null);

    expect(session).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('getOrCreate on a missing key returns a fresh session without caching', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });

    const session = reg.getOrCreate('resp_missing');

    expect(session).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('adopt inserts a session and subsequent getOrCreate returns the cached instance', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1);

    expect(reg.size).toBe(1);
    expect(reg.getOrCreate('resp_1')).toBe(s1);
    // Second lookup still returns the same instance (LRU promotion is
    // idempotent on a size-1 map).
    expect(reg.getOrCreate('resp_1')).toBe(s1);
    expect(reg.size).toBe(1);
  });

  it('evicts entries whose TTL has expired on lookup', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1);
    expect(reg.size).toBe(1);

    // Advance past the TTL window.
    vi.advanceTimersByTime(61 * 1000);

    const got = reg.getOrCreate('resp_1');
    expect(got).not.toBe(s1);
    expect(got).toBeInstanceOf(ChatSession);
    expect(reg.size).toBe(0);
  });

  it('evicts the least-recently-used entry when maxEntries overflows', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, maxEntries: 2 });
    const sA = new ChatSession(model);
    const sB = new ChatSession(model);
    const sC = new ChatSession(model);

    reg.adopt('a', sA);
    reg.adopt('b', sB);
    // Touch 'a' so 'b' becomes LRU.
    expect(reg.getOrCreate('a')).toBe(sA);
    // Now 'c' displaces 'b' (the LRU victim), not 'a'.
    reg.adopt('c', sC);

    expect(reg.size).toBe(2);
    expect(reg.getOrCreate('a')).toBe(sA);
    expect(reg.getOrCreate('c')).toBe(sC);
    // 'b' was evicted — lookup returns a fresh miss.
    const bMiss = reg.getOrCreate('b');
    expect(bMiss).not.toBe(sB);
    expect(bMiss).toBeInstanceOf(ChatSession);
  });

  it('adopt overwrites an existing key and refreshes expiry', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const s1 = new ChatSession(model);
    const s2 = new ChatSession(model);

    reg.adopt('resp_1', s1);
    // Advance near expiry so we can verify the TTL got refreshed.
    vi.advanceTimersByTime(50 * 1000);
    reg.adopt('resp_1', s2);

    expect(reg.size).toBe(1);
    expect(reg.getOrCreate('resp_1')).toBe(s2);

    // Original expiry was 60s after t=0; refreshed expiry is 60s after
    // t=50s, so at t=80s the entry should still be live.
    vi.advanceTimersByTime(30 * 1000);
    expect(reg.getOrCreate('resp_1')).toBe(s2);
  });

  it('drop removes an entry and subsequent getOrCreate misses', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1);
    reg.drop('resp_1');

    expect(reg.size).toBe(0);
    const got = reg.getOrCreate('resp_1');
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

    reg.adopt('a', sA);
    // Stagger the adopts so 'a' expires before 'b' and 'c'.
    vi.advanceTimersByTime(30 * 1000);
    reg.adopt('b', sB);
    reg.adopt('c', sC);
    // t=30s. 'a' expires at 60s, 'b' and 'c' at 90s.
    vi.advanceTimersByTime(40 * 1000);
    // t=70s. Sweep should drop 'a' but keep 'b' and 'c'.
    reg.sweep();

    expect(reg.size).toBe(2);
    expect(reg.getOrCreate('a')).not.toBe(sA);
    expect(reg.getOrCreate('b')).toBe(sB);
    expect(reg.getOrCreate('c')).toBe(sC);
  });

  it('clear empties the registry', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    reg.adopt('a', new ChatSession(model));
    reg.adopt('b', new ChatSession(model));

    expect(reg.size).toBe(2);
    reg.clear();
    expect(reg.size).toBe(0);
  });
});
