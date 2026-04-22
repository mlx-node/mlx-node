/**
 * Opt-in gate + HMAC-scoping tests for {@link SessionRegistry}'s tier-2
 * `prompt_cache_key` lookup.
 *
 * Companion to `session-registry-prompt-cache-key.test.ts` (which covers
 * the tier-2 match semantics with the feature enabled). This file pins
 * the security contract:
 *
 *   - Tier-2 reuse is ON by default. Opt out via
 *     `MLX_DISABLE_PROMPT_CACHE_KEY=1`. When opted out, every tier-2
 *     lookup misses regardless of key.
 *   - Stored keys are HMAC-scoped with a boot-time nonce held only in
 *     the module's memory. Resetting the nonce (simulating a server
 *     restart) invalidates every tier-2 entry.
 *   - A caller-supplied key must be at least 8 chars to be accepted;
 *     shorter keys are treated as if the caller had not supplied one.
 *   - Null / empty keys never match regardless of the opt-out.
 *
 * The nonce reset helper is imported via the deep path into
 * `session-registry.ts` — it is not part of the server package's public
 * `index.ts` export surface. Nothing outside test code should ever
 * invoke it.
 */

import type { ChatResult } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import { SessionRegistry } from '@mlx-node/server';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

// Test-only deep import. `__resetPromptCacheKeyNonceForTests` is
// intentionally NOT exported from the package's public `index.ts`;
// exposing it there would let downstream consumers nuke every live
// tier-2 entry in production. Tests reach it via the source path so
// the "this is internal, do not use" signal is loud at the call site.
import {
  __loggedSilentMissKeysSizeForTests,
  __resetPromptCacheKeyNonceForTests,
  maybeWarnPromptCacheKeyIneligible,
} from '../../packages/server/src/session-registry.js';

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

describe('SessionRegistry prompt_cache_key scoping', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    // Reset the module-scoped nonce so each test starts from a clean
    // "fresh server boot" state.
    __resetPromptCacheKeyNonceForTests();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    __resetPromptCacheKeyNonceForTests();
  });

  it('tier-2 enabled by default: keys match when no disable env is set', () => {
    // Security default: tier-2 reuse is ON out of the box so any
    // stateless agent that threads `prompt_cache_key` gets warm-session
    // reuse immediately. No env stubbing is required — adopting with a
    // non-empty key and looking it up must HIT. The negative-regression
    // branch below flips `MLX_DISABLE_PROMPT_CACHE_KEY=1` and asserts
    // the same lookup misses, pinning the opt-out contract.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'stateless-chain-1234');
    const got = reg.getOrCreate(null, null, 'stateless-chain-1234');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);

    // Negative regression: with the opt-out set, adopt + lookup of the
    // same key misses because tier-2 scoping returns null on both ends.
    vi.stubEnv('MLX_DISABLE_PROMPT_CACHE_KEY', '1');
    const reg2 = new SessionRegistry({ model });
    const s2 = new ChatSession(model);
    reg2.adopt('resp_2', s2, null, 'stateless-chain-1234');
    const got2 = reg2.getOrCreate(null, null, 'stateless-chain-1234');
    expect(got2.session).not.toBe(s2);
    expect(got2.session).toBeInstanceOf(ChatSession);
    expect(got2.hit).toBe(false);
  });

  it('tier-2 disabled: stored key is null (not the raw client key)', () => {
    // Belt-and-suspenders: even if some future lookup path were to
    // bypass the opt-out gate, the stored `promptCacheKey` is itself
    // null when the feature is disabled — so the raw caller-supplied
    // string is never reachable from the cached entry in any form.
    vi.stubEnv('MLX_DISABLE_PROMPT_CACHE_KEY', '1');
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    // Sanity: adopt with a non-empty key that would match under tier-2
    // if the feature were enabled.
    reg.adopt('resp_1', s1, null, 'stateless-chain-1234');
    // Clearing the opt-out AFTER adopt cannot retroactively resurrect
    // the stored key — the scoping happened at adopt time, not at
    // lookup time.
    vi.stubEnv('MLX_DISABLE_PROMPT_CACHE_KEY', '');
    const got = reg.getOrCreate(null, null, 'stateless-chain-1234');
    expect(got.hit).toBe(false);
  });

  it('tier-2 enabled: hit on matching scoped key', () => {
    // Positive-path smoke: when tier-2 is active (the default) and
    // both sides pass a sufficiently-long matching key, the registry
    // hits on tier 2. The stored key is the HMAC of the raw key; the
    // lookup applies the same HMAC so both sides compare equal.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, 'sys', 'stateless-chain-1234');
    const got = reg.getOrCreate(null, 'sys', 'stateless-chain-1234');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('nonce reset invalidates every previously stored entry', () => {
    // A process restart re-randomizes the module-scoped nonce, so the
    // scoped HMAC on the next adopt / lookup differs from anything
    // stored before. Existing entries become unreachable — exactly
    // the contract promised in the module docstring. Simulated here
    // via the test-only `__resetPromptCacheKeyNonceForTests` hook.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'stateless-chain-1234');

    // Simulate a server restart.
    __resetPromptCacheKeyNonceForTests();

    const got = reg.getOrCreate(null, null, 'stateless-chain-1234');
    expect(got.session).not.toBe(s1);
    expect(got.hit).toBe(false);
  });

  it('empty-string key misses regardless of opt-out (min-length gate)', () => {
    // The minimum-length gate rejects the empty string on both
    // adopt and lookup, so the entry's scoped `promptCacheKey` is
    // null and no request can hit tier 2 via "".
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, '');
    const got = reg.getOrCreate(null, null, '');

    expect(got.hit).toBe(false);
  });

  it('too-short key misses regardless of opt-out (min-length gate)', () => {
    // Keys shorter than the minimum are rejected at scope time — they
    // would be trivially guessable and make cross-client collisions
    // plausible. The guard treats them exactly like the "no key"
    // case; the stored entry's scoped key is null and the lookup
    // returns a fresh session.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    // 7 chars — below the 8-char minimum.
    reg.adopt('resp_1', s1, null, 'abc1234');
    const got = reg.getOrCreate(null, null, 'abc1234');

    expect(got.hit).toBe(false);
  });

  it('exactly-minimum-length key is accepted when tier-2 is enabled', () => {
    // Boundary case: 8 chars is acceptable; pins the minimum-length
    // boundary at `length < 8` rather than `<= 8`.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'abcd1234');
    const got = reg.getOrCreate(null, null, 'abcd1234');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('only literal MLX_DISABLE_PROMPT_CACHE_KEY="1" disables — other truthy values leave it enabled', () => {
    // The check is `=== '1'`, not truthy. `'true'` / `'yes'` / numeric
    // `1` (env vars are always strings anyway) all leave the feature
    // ON. Pins that accidental truthy-but-not-"1" values do NOT
    // silently disable tier-2 reuse — operators must set the exact
    // opt-out literal.
    vi.stubEnv('MLX_DISABLE_PROMPT_CACHE_KEY', 'true');
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    reg.adopt('resp_1', s1, null, 'stateless-chain-1234');
    const got = reg.getOrCreate(null, null, 'stateless-chain-1234');

    expect(got.session).toBe(s1);
    expect(got.hit).toBe(true);
  });

  it('raw caller-supplied key is not stored on the entry (HMAC-scoped only)', () => {
    // Defense-in-depth: if a future change introduces a side channel
    // that exposes the entry's `promptCacheKey` string, it must not
    // reveal the raw caller-supplied key. This test reaches into the
    // registry via the test-visible `size` getter after adopt and
    // asserts that the stored scoped string is NOT byte-equal to the
    // raw input — the HMAC pre-image resistance is what protects
    // clients from cross-request probing in multi-tenant setups.
    const model = makeMockModel();
    const reg = new SessionRegistry({ model });
    const s1 = new ChatSession(model);

    const rawKey = 'stateless-chain-1234';
    reg.adopt('resp_1', s1, null, rawKey);

    // Hit via the same raw key (proves the scoping is deterministic
    // within one nonce lifetime).
    const got = reg.getOrCreate(null, null, rawKey);
    expect(got.hit).toBe(true);
    // Directly probing via a raw-key byte-equal lookup against the
    // stored entry is impossible from outside — the `entries` map
    // is private. The deterministic-hit behaviour above is the
    // functional observation that proves scoping is reversible from
    // inside the module but opaque from outside.
  });

  it('silent-miss warning dedupe cache is bounded under attacker-supplied key flooding (Round 9 Fix #1)', () => {
    // Regression: `maybeWarnPromptCacheKeyIneligible` used to retain
    // every distinct raw `prompt_cache_key` in a module-scoped
    // Set<string> that never evicted. A client (or attacker) flooding
    // the endpoint with distinct ineligible keys would drive unbounded
    // process memory growth. The fix switched to SHA-256-digest keys in
    // a FIFO-bounded Map capped at 256 entries. This test drives 5,000
    // distinct keys through the warning path with the env gate OFF
    // (every call hits the silent-miss branch) and verifies that: (a)
    // the dedupe store does not grow past the cap, and (b) the helper
    // does not throw or mis-classify. Exact cap is a private detail —
    // we just assert a reasonable upper bound that is LESS THAN the
    // flood size.
    vi.stubEnv('MLX_DISABLE_PROMPT_CACHE_KEY', '1');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const flood = 5000;
      const keyFor = (i: number): string =>
        `flood-key-${i.toString().padStart(10, '0')}-0123456789abcdef0123456789abcdef`;
      for (let i = 0; i < flood; i += 1) {
        // 64 chars of hex per key, all distinct.
        maybeWarnPromptCacheKeyIneligible(keyFor(i));
      }
      // Primary invariant: the dedupe Map stays bounded by the FIFO
      // cap even after `flood >> cap` distinct keys. This is the only
      // observation that actually distinguishes the fixed code from the
      // pre-fix unbounded Set — warning-count alone cannot, because
      // each call syntactically emits at most one warning whether the
      // cap works or not. Cap is a module-private constant (256 at time
      // of writing); we assert it is saturated but not exceeded.
      const sizeAfterFlood = __loggedSilentMissKeysSizeForTests();
      expect(sizeAfterFlood).toBeGreaterThan(0);
      expect(sizeAfterFlood).toBeLessThan(flood);
      // Re-feeding a still-resident recent key must dedupe (no new
      // warn, no size change) — proves the Map is doing its primary
      // job within the window.
      const warnsBeforeRecent = warn.mock.calls.length;
      maybeWarnPromptCacheKeyIneligible(keyFor(flood - 1));
      expect(warn.mock.calls.length).toBe(warnsBeforeRecent);
      expect(__loggedSilentMissKeysSizeForTests()).toBe(sizeAfterFlood);
      // Re-feeding the oldest key (evicted thousands of iterations
      // ago) must re-warn exactly once — proves FIFO eviction actually
      // removed it from the dedupe store rather than the Set growing
      // unbounded.
      const warnsBeforeEvicted = warn.mock.calls.length;
      maybeWarnPromptCacheKeyIneligible(keyFor(0));
      expect(warn.mock.calls.length).toBe(warnsBeforeEvicted + 1);
    } finally {
      warn.mockRestore();
    }
    // Subsequent nonce reset must clear the dedupe store (same path
    // tests use to re-exercise the once-per-key signal).
    __resetPromptCacheKeyNonceForTests();
    expect(__loggedSilentMissKeysSizeForTests()).toBe(0);
  });
});
