/**
 * Tier-2 `prompt_cache_key` lookup tests for {@link SessionRegistry}.
 *
 * Companion to `session-registry.test.ts` (which covers tier-1
 * `previousResponseId` lookup and the single-warm invariant). The tier-2
 * lookup only runs when `previousResponseId` is `null`, so every test
 * here threads `null` as the first argument to `getOrCreate` and asserts
 * on the `promptCacheKey` third argument.
 *
 * Precedence contract pinned by {@link SessionRegistry.getOrCreate}:
 * when `previousResponseId` is supplied the registry NEVER falls through
 * to tier 2 on miss — cold replay is the safe default because the two
 * keys could identify different conversation branches. One of the
 * precedence tests below pins that contract by setting up a scenario
 * where tier 2 would otherwise hit and asserting the miss.
 */

import type { ChatResult } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import { SessionRegistry } from '@mlx-node/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

/**
 * Build a minimal `SessionCapableModel` stub. Mirrors the mock in
 * `session-registry.test.ts` — only enough shape for
 * `new ChatSession(mock)` to be constructable; these tier-2 tests
 * never actually drive a turn through the session.
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
    cachedTokens: 0,
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
    cachedTokens: 0,
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

describe('SessionRegistry tier-2 prompt_cache_key lookup', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    // Tier-2 reuse is ON by default (see session-registry.ts); no env
    // stubbing is required here. The companion
    // `session-registry-key-scoping.test.ts` covers the
    // `MLX_DISABLE_PROMPT_CACHE_KEY` opt-out behaviour explicitly.
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
  });

  it('tier-2 hit: adopt with promptCacheKey and look up by the same key', () => {
    // Core stateless-agent use case: pi-mono (and every other stateless
    // client) never threads `previous_response_id`, so the only way
    // warm-session reuse can happen across turns is the
    // `promptCacheKey`-keyed tier-2 scan.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'be concise', 'chain-abc');
    const got = reg.getOrCreate(null, 'be concise', 'chain-abc');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
    // Single-use lease: entry is consumed on hit.
    expect(reg.size).toBe(0);
  });

  it('previousResponseId wins unconditionally on tier-1 miss — never falls through to tier 2', () => {
    // Pin the precedence rule: even when tier 2 would have hit
    // (matching promptCacheKey + instructions on the live entry),
    // a supplied `previousResponseId` that tier 1 cannot resolve
    // must fall through to fresh/cold-replay — NOT to tier 2. The
    // two keys can legitimately identify different conversation
    // branches, so routing the prev-id request to a
    // promptCacheKey-matching session risks splicing the wrong warm
    // state into the wrong chain.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    // Adopt under a responseId that WON'T be queried, with a
    // `promptCacheKey` that tier 2 would match if reached.
    reg.adopt('resp_live', s1, 'ins', 'chain-abc');
    // Request a different prev-id (tier-1 miss) with the tier-2-matching
    // cache key. Precedence: prev-id wins → miss, fresh session.
    const got = reg.getOrCreate('resp_other', 'ins', 'chain-abc');

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('tier-2 miss on unknown key returns a fresh session', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'chain-known');
    const got = reg.getOrCreate(null, null, 'chain-unknown');

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    // Single-warm invariant: the miss-path clear evicts the prior entry.
    expect(reg.size).toBe(0);
  });

  it('instructions mismatch blocks tier-2 hit even when promptCacheKey matches', () => {
    // Same prefix-state guard that protects tier 1: a warm session
    // keyed under one `instructions` value must NOT be reused for a
    // request whose instructions differ, or the model silently
    // depends on cached state instead of request contents.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'ins-A', 'chain-abc');
    const got = reg.getOrCreate(null, 'ins-B', 'chain-abc');

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('cross-key isolation: adopting a new key evicts the prior entry under the single-warm invariant', () => {
    // The registry holds at most one entry. Adopting session B under
    // a different `promptCacheKey` evicts session A, and a subsequent
    // lookup against A's key misses — exactly the same semantics as
    // the prev-id tier-1 path.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const sA = new ChatSession(model);
    const sB = new ChatSession(model);

    reg.adopt('resp_a', sA, null, 'chain-aaa');
    expect(reg.size).toBe(1);
    reg.adopt('resp_b', sB, null, 'chain-bbb');
    expect(reg.size).toBe(1);

    // A's key now misses (A was evicted by B's adopt).
    const aMiss = reg.getOrCreate(null, null, 'chain-aaa');
    expect(aMiss.session).not.toBe(sA);
    expect(aMiss.hit).toBe(false);
    expect(reg.size).toBe(0);
  });

  it('expired entries miss tier-2 lookup', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model, ttlSec: 60 });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'chain-abc');
    expect(reg.size).toBe(1);
    // Advance past TTL.
    vi.advanceTimersByTime(61 * 1000);

    const got = reg.getOrCreate(null, null, 'chain-abc');
    expect(got.session).not.toBe(s1);
    expect(got.hit).toBe(false);
    // Expired-entry path still honors the single-warm invariant.
    expect(reg.size).toBe(0);
  });

  it('null promptCacheKey and empty-string promptCacheKey are distinct — adopting with null misses a lookup with ""', () => {
    // Null-vs-empty-string distinction is load-bearing: a client that
    // forgets to thread the key (adopts with `null`) must NOT collide
    // with another client that explicitly sets an empty key.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, null);
    const got = reg.getOrCreate(null, null, '');

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
  });

  it('null promptCacheKey and empty-string promptCacheKey are distinct — adopting with "" misses a lookup with null', () => {
    // Converse of the above.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, '');
    const got = reg.getOrCreate(null, null, null);

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
  });

  it('tier-2 lookup with null promptCacheKey against a null-keyed entry misses (no key means no tier-2 hit)', () => {
    // Lookup without a key must not accidentally pick up a null-keyed
    // entry — "no key" on both sides cannot constitute a conversation-
    // chain match. Only an explicit non-null key on both the request
    // and the entry can hit tier 2.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, null);
    const got = reg.getOrCreate(null, null, null);

    expect(got.session).not.toBe(s1);
    expect(got.session).toBeInstanceOf(ChatSession);
    expect(got.hit).toBe(false);
  });

  it('tier-2 match with matching null instructions on both sides', () => {
    // `null === null` on instructions must still count as a match —
    // same semantics as tier 1. Regression check: nullable-field
    // comparisons via `!==` would otherwise compare null as a unique
    // sentinel.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'chain-abc');
    const got = reg.getOrCreate(null, null, 'chain-abc');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('empty-string instructions match empty-string instructions on tier-2 hit', () => {
    // "" is an explicit value, distinct from null; matched like any
    // other string.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, '', 'chain-abc');
    const got = reg.getOrCreate(null, '', 'chain-abc');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('null-vs-empty instructions distinction: tier-2 miss when adopted with null and looked up with ""', () => {
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'chain-abc');
    const got = reg.getOrCreate(null, '', 'chain-abc');

    expect(got.session).not.toBe(s1);
    expect(got.hit).toBe(false);
  });

  it('tier-2 hit then re-adopt chains the session across stateless turns', () => {
    // Integration shape: turn 1 adopts under `chain-abc`. Turn 2 passes
    // the same key and warm-hits. Turn 2's adopt re-inserts under the
    // same key so turn 3 can also hit. Pins the typical agent-replay
    // lifecycle the feature was designed for.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_turn_1', s1, 'sys', 'chain-abc');

    const turn2 = reg.getOrCreate(null, 'sys', 'chain-abc');
    expect(turn2.session).toBe(s1);
    expect(turn2.hit).toBe(true);

    // Simulate the endpoint's post-dispatch `adopt` call — same key,
    // new responseId.
    reg.adopt('resp_turn_2', turn2.session, 'sys', 'chain-abc');

    const turn3 = reg.getOrCreate(null, 'sys', 'chain-abc');
    expect(turn3.session).toBe(s1);
    expect(turn3.hit).toBe(true);
  });
});
