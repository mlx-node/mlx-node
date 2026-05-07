/**
 * Phase 7 — `/v1/messages` warm-slot bypass on paged-active models.
 *
 * Pins the runtime decision documented in
 * `packages/server/src/endpoints/messages.ts`: when the underlying
 * native model has the block-paged KV cache adapter active
 * (`SessionCapableModel.hasBlockPagedCache?.()`), the endpoint
 * allocates a fresh `ChatSession` per request and never touches the
 * warm slot — cross-request prefix reuse is handled entirely by the
 * native `BlockAllocator`'s content-addressed prefix-hash table, so
 * the JS-side warm slot would only serialize parallel requests for
 * no benefit.
 *
 * The orthogonal case (non-paged models keeping the warm-slot path)
 * already has dedicated coverage in `messages-handler.test.ts` and is
 * NOT re-asserted here. This file pins ONLY the new bypass behaviour
 * so a future regression that re-introduces a warm-slot lease on the
 * paged path fails loudly.
 */

import type { ServerResponse } from 'node:http';

import type { ChatResult, ToolCallResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';

// ---------------------------------------------------------------------------
// Mock helpers (mirror the patterns used by `messages-handler.test.ts`)
// ---------------------------------------------------------------------------

function createMockRes(): {
  res: ServerResponse;
  getStatus: () => number;
  getBody: () => string;
  getHeaders: () => Record<string, string | string[]>;
} {
  const { Writable } = require('node:stream');
  let status = 200;
  let body = '';
  const headers: Record<string, string | string[]> = {};

  const writable = new Writable({
    write(chunk: Uint8Array | string, _encoding: string, callback: () => void) {
      body += chunk.toString();
      callback();
    },
  });
  writable.on('error', () => {});

  writable.writeHead = (s: number, h?: Record<string, string>) => {
    status = s;
    if (h) {
      for (const [k, v] of Object.entries(h)) {
        headers[k.toLowerCase()] = v;
      }
    }
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = (name: string, value: string) => {
    headers[name.toLowerCase()] = value;
  };
  writable.getHeader = (name: string) => {
    return headers[name.toLowerCase()];
  };
  writable.headersSent = false;

  const origEnd = writable.end.bind(writable);
  writable.end = (chunkArg?: unknown, encodingArg?: unknown, cbArg?: unknown) => {
    let chunk: string | Uint8Array | undefined;
    let encoding: unknown;
    let cb: ((err?: Error | null) => void) | undefined;
    if (typeof chunkArg === 'function') {
      cb = chunkArg as (err?: Error | null) => void;
    } else {
      chunk = chunkArg as string | Uint8Array | undefined;
      if (typeof encodingArg === 'function') {
        cb = encodingArg as (err?: Error | null) => void;
      } else {
        encoding = encodingArg;
        if (typeof cbArg === 'function') {
          cb = cbArg as (err?: Error | null) => void;
        }
      }
    }
    if (chunk != null) body += chunk.toString();
    writable.headersSent = true;
    origEnd(undefined, encoding, (err?: Error | null) => {
      if (cb) cb(err ?? null);
    });
    return writable;
  };

  return {
    res: writable as unknown as ServerResponse,
    getStatus: () => status,
    getBody: () => body,
    getHeaders: () => headers,
  };
}

function makeChatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'Hello!',
    toolCalls: [] as ToolCallResult[],
    numTokens: 10,
    promptTokens: 5,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: 'Hello!',
    cachedTokens: 0,
    performance: undefined,
    ...overrides,
  };
}

/**
 * Build a session-capable mock model with a configurable
 * `hasBlockPagedCache()` return. `chatSessionStart` resolves with the
 * supplied result; other entry points reject so a mistaken hot-path
 * call surfaces immediately.
 */
function createMockModel(paged: boolean, result: ChatResult = makeChatResult()): SessionCapableModel {
  async function* fallbackStream() {
    yield {
      done: true,
      text: result.text,
      finishReason: result.finishReason,
      toolCalls: result.toolCalls,
      thinking: result.thinking ?? null,
      numTokens: result.numTokens,
      promptTokens: result.promptTokens,
      reasoningTokens: result.reasoningTokens,
      rawText: result.rawText,
      cachedTokens: result.cachedTokens,
    };
  }
  return {
    chatSessionStart: vi.fn().mockResolvedValue(result),
    chatSessionContinue: vi.fn().mockRejectedValue(new Error('hot path: chatSessionContinue not expected')),
    chatSessionContinueTool: vi.fn().mockRejectedValue(new Error('hot path: chatSessionContinueTool not expected')),
    chatStreamSessionStart: vi.fn(() => fallbackStream()),
    chatStreamSessionContinue: vi.fn(() => fallbackStream()),
    chatStreamSessionContinueTool: vi.fn(() => fallbackStream()),
    resetCaches: vi.fn(),
    hasBlockPagedCache: vi.fn(() => paged),
  } as unknown as SessionCapableModel;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('handleCreateMessage — paged-active warm-slot bypass', () => {
  it('paged-active model does NOT lease the warm slot (size stays 0 after a successful turn)', async () => {
    // Contract: when the model reports `hasBlockPagedCache() === true`,
    // `/v1/messages` calls `SessionRegistry.createFreshSession()`
    // instead of `getOrCreateWarmAny`, runs the turn, and never
    // adopts a warm entry. The size invariant is the load-bearing
    // proof: 0 before AND 0 after a successful dispatch.
    const registry = new ModelRegistry();
    const mockModel = createMockModel(/* paged */ true);
    registry.register('paged-model', mockModel);
    const sessionReg = registry.getSessionRegistry('paged-model')!;
    expect(sessionReg.size).toBe(0);

    const { res, getStatus, getHeaders } = createMockRes();
    await handleCreateMessage(
      res,
      {
        model: 'paged-model',
        system: 'sysA',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
      },
      registry,
    );

    expect(getStatus()).toBe(200);
    // The load-bearing assertion: paged path NEVER adopts the warm slot.
    expect(sessionReg.size).toBe(0);
    // Paged reuse is native/content-addressed. A fresh JS ChatSession is
    // still allocated, but the endpoint must not call the public full
    // native reset because that clears MoE GDN prefix checkpoints between
    // otherwise cacheable stateless turns.
    expect(mockModel.resetCaches).not.toHaveBeenCalled();
    // Pre-dispatch header is `fresh` because `lookup.hit` is always
    // false on the paged path; no `cachedTokens > 0` promotion fired
    // because the mock returned `cachedTokens: 0`.
    expect(getHeaders()['x-session-cache']).toBe('fresh');
  });

  it('paged-active non-streaming response promotes X-Session-Cache to prefix_hit when engine reports cachedTokens > 0', async () => {
    // Contract: even though `lookup.hit` is `false` (we
    // `createFreshSession`), the post-dispatch promotion branch
    // flips the header to `prefix_hit` when the native engine
    // reports `cachedTokens > 0` — that's the authoritative signal
    // that `BlockAllocator`'s content-addressed prefix lookup
    // recovered shared SYS blocks on this turn.
    const registry = new ModelRegistry();
    const result = makeChatResult({ cachedTokens: 42 });
    registry.register('paged-model', createMockModel(/* paged */ true, result));
    const sessionReg = registry.getSessionRegistry('paged-model')!;

    const { res, getStatus, getHeaders } = createMockRes();
    await handleCreateMessage(
      res,
      {
        model: 'paged-model',
        system: 'sysA',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
      },
      registry,
    );

    expect(getStatus()).toBe(200);
    expect(sessionReg.size).toBe(0);
    expect(getHeaders()['x-session-cache']).toBe('prefix_hit');
    expect(getHeaders()['x-cached-tokens']).toBe('42');
  });

  it('paged-active path does NOT consult the warm slot even when one is pre-seeded with byte-equal instructions', async () => {
    // Adversarial: pre-seed a warm entry with `instructions === sysA`
    // so a non-paged model would have leased it. Then dispatch a
    // request whose model has `hasBlockPagedCache() === true`. The
    // pre-seeded entry MUST stay untouched because the paged path
    // uses `createFreshSession()` and never walks the warm map.
    //
    // The proxy: the pre-seeded session's `chatSessionStart` is a
    // distinct spy from the model's; if the warm slot were leased,
    // the lookup would clear the map (single-use lease) and the
    // pre-seeded entry would disappear. The paged path keeps it.
    const mockModel = createMockModel(/* paged */ true);
    const registry = new ModelRegistry();
    registry.register('paged-model', mockModel);
    const sessionReg = registry.getSessionRegistry('paged-model')!;

    // Pre-seed a warm entry under an arbitrary id with
    // `instructions === sysA`. We import `ChatSession` lazily to
    // avoid the heavyweight module-level import in the test header.
    const { ChatSession } = await import('@mlx-node/lm');
    const preseedSession = new ChatSession(mockModel);
    sessionReg.adopt('resp_pre_seed', preseedSession, 'sysA', null);
    expect(sessionReg.size).toBe(1);

    const { res, getStatus } = createMockRes();
    await handleCreateMessage(
      res,
      {
        model: 'paged-model',
        system: 'sysA',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
      },
      registry,
    );

    expect(getStatus()).toBe(200);
    // Load-bearing: the pre-seeded entry is STILL THERE because
    // the paged path never touched the warm map. A non-paged
    // request would have leased it (clearing the map) and re-adopted
    // under the `__msg_warm__` sentinel; on the paged path neither
    // happens, so the original key survives.
    expect(sessionReg.size).toBe(1);
    // Sanity: the model's `chatSessionStart` was called for THIS
    // request, not just the pre-seed sitting idle.
    // oxlint-disable-next-line @typescript-eslint/unbound-method
    const startSpy = mockModel.chatSessionStart as unknown as ReturnType<typeof vi.fn>;
    expect(startSpy).toHaveBeenCalledTimes(1);
  });

  it('non-paged model still leases + adopts the warm slot (regression guard for the conditional gate)', async () => {
    // The conditional gate must keep the non-paged path intact: when
    // `hasBlockPagedCache()` returns false (or is missing), the
    // endpoint MUST go through `getOrCreateWarmAny` + adopt under
    // the sentinel id, otherwise non-paged models lose their only
    // cross-conversation reuse mechanism.
    const registry = new ModelRegistry();
    const mockModel = createMockModel(/* paged */ false);
    registry.register('flat-model', mockModel);
    const sessionReg = registry.getSessionRegistry('flat-model')!;

    const { res, getStatus } = createMockRes();
    await handleCreateMessage(
      res,
      {
        model: 'flat-model',
        system: 'sysA',
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 100,
      },
      registry,
    );

    expect(getStatus()).toBe(200);
    // Load-bearing: non-paged path DID adopt the warm slot.
    expect(sessionReg.size).toBe(1);
    // Non-paged MISS still takes the public full reset branch. Its only
    // authorized native-cache-preserving path is a warm-slot HIT.
    expect(mockModel.resetCaches).toHaveBeenCalledTimes(1);
  });

  it('two parallel paged-active requests with the same system both get fresh sessions and BOTH commit (no warm-slot serialization)', async () => {
    // Contract: paged-active dispatches do not contend over the warm
    // slot. Two requests against the same model + same `system`
    // string should each run through their own fresh `ChatSession`
    // (no lease, no eviction, no `getOrCreateWarmAny` race).
    //
    // We can't easily verify "they ran in parallel" through this
    // mock (the `withExclusive` mutex still serializes them per the
    // single mutable native model contract — that is unchanged by
    // Phase 7), but we CAN verify both commits land cleanly and
    // neither created a warm entry. The point of the bypass is
    // semantic, not parallelism: even after the second turn the
    // map size stays 0 because neither side adopts.
    const registry = new ModelRegistry();
    registry.register('paged-model', createMockModel(/* paged */ true));
    const sessionReg = registry.getSessionRegistry('paged-model')!;

    const { res: res1, getStatus: getStatus1 } = createMockRes();
    const { res: res2, getStatus: getStatus2 } = createMockRes();
    await Promise.all([
      handleCreateMessage(
        res1,
        {
          model: 'paged-model',
          system: 'sysA',
          messages: [{ role: 'user', content: 'hi' }],
          max_tokens: 100,
        },
        registry,
      ),
      handleCreateMessage(
        res2,
        {
          model: 'paged-model',
          system: 'sysA',
          messages: [{ role: 'user', content: 'hello' }],
          max_tokens: 100,
        },
        registry,
      ),
    ]);

    expect(getStatus1()).toBe(200);
    expect(getStatus2()).toBe(200);
    // Both turns committed AND neither adopted: the size is still 0.
    expect(sessionReg.size).toBe(0);
  });
});
