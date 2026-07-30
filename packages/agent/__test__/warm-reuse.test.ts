import type { ChatConfig, ChatMessage, ChatResult, ToolDefinition } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { resetPreservingNativeCacheForWarmReuse as resetServerWarmReuse } from '../../server/src/chat-session-warm-reuse.js';
import {
  WARM_REUSE_TOUCHED_FIELDS,
  resetPreservingNativeCacheForWarmReuse as resetAgentWarmReuse,
} from '../src/provider/warm-reuse.js';

const previousTools: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'previous_conversation_tool',
      description: 'Must not leak into the next conversation',
      parameters: {
        type: 'object',
        properties: '{}',
      },
    },
  },
];

const defaultTools: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'operator_default_tool',
      description: 'Constructor default restored across resets',
      parameters: {
        type: 'object',
        properties: '{}',
      },
    },
  },
];

/**
 * Minimal `SessionCapableModel` stub: `new ChatSession(model)` only stores
 * the reference (no natives touched), and these tests never drive a turn.
 * `resetCaches` is a spy so the tests can distinguish the full `reset()`
 * path (calls it) from the warm-reuse path (must not).
 */
function makeMockModel(): SessionCapableModel & {
  resetCaches: ReturnType<typeof vi.fn>;
  startConfigs: Array<ChatConfig | null | undefined>;
} {
  const startConfigs: Array<ChatConfig | null | undefined> = [];
  const result: ChatResult = {
    text: 'ok',
    toolCalls: [],
    thinking: undefined,
    thinkingEnabled: true,
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
    thinkingEnabled: true,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    rawText: 'ok',
  };
  return {
    chatSessionStart: async (_messages: ChatMessage[], config?: ChatConfig | null) => {
      startConfigs.push(config);
      return result;
    },
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
    resetCaches: vi.fn(),
    startConfigs,
  } as unknown as SessionCapableModel & {
    resetCaches: ReturnType<typeof vi.fn>;
    startConfigs: Array<ChatConfig | null | undefined>;
  };
}

/** Runtime-mutable view of the ChatSession private fields these tests poke. */
interface SessionInternals {
  inFlight: boolean;
  history: ChatMessage[];
  lastImagesKey: string | null;
  lastAudioKey: string | null;
  turnCount: number;
  unresolvedOkToolCallCount: number | null;
  needsFullReplay: boolean;
  defaultConfig: ChatConfig;
  activeTools: ChatConfig['tools'];
}

function internalsOf(session: ChatSession): SessionInternals {
  return session as unknown as SessionInternals;
}

describe('warm-reuse drift detection', () => {
  it('every private field the port touches exists on a real ChatSession instance', () => {
    const session = new ChatSession(makeMockModel());

    // TS `private` fields are ordinary own properties at runtime; all
    // touched fields are initialized in the constructor or by class-field
    // initializers, so a rename/removal in @mlx-node/lm surfaces here.
    const missing = WARM_REUSE_TOUCHED_FIELDS.filter((field) => !Object.prototype.hasOwnProperty.call(session, field));
    expect(missing).toEqual([]);
  });

  it('touched fields carry the initial values the port relies on', () => {
    const session = new ChatSession(makeMockModel());
    const internals = internalsOf(session);

    // Value-level pin: guards against a rename-plus-shadow where the old
    // name survives with a different meaning/type.
    expect(internals.inFlight).toBe(false);
    expect(Array.isArray(internals.history)).toBe(true);
    expect(internals.history).toEqual([]);
    expect(internals.lastImagesKey).toBeNull();
    expect(internals.lastAudioKey).toBeNull();
    expect(internals.turnCount).toBe(0);
    expect(internals.unresolvedOkToolCallCount).toBeNull();
    expect(internals.needsFullReplay).toBe(false);
    expect(internals.defaultConfig).toEqual({});
    expect(internals.activeTools).toBeUndefined();
  });

  it('the touched-field list is non-empty and matches the documented contract', () => {
    expect([...WARM_REUSE_TOUCHED_FIELDS].sort()).toEqual([
      'activeTools',
      'defaultConfig',
      'history',
      'inFlight',
      'lastAudioKey',
      'lastImagesKey',
      'needsFullReplay',
      'turnCount',
      'unresolvedOkToolCallCount',
    ]);
  });
});

describe.each([
  ['agent', resetAgentWarmReuse],
  ['server', resetServerWarmReuse],
])('resetPreservingNativeCacheForWarmReuse (%s)', (_name, resetPreservingNativeCacheForWarmReuse) => {
  function makePopulatedSession(defaultConfig?: ChatConfig) {
    const model = makeMockModel();
    const session = new ChatSession(model, { system: 'sys', defaultConfig });
    const internals = internalsOf(session);
    internals.history = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' },
    ];
    internals.lastImagesKey = 'deadbeef';
    internals.lastAudioKey = 'cafe';
    internals.turnCount = 3;
    internals.unresolvedOkToolCallCount = 2;
    internals.needsFullReplay = true;
    internals.activeTools = previousTools;
    return { model, session, internals };
  }

  it('wipes the TS-side conversation state', async () => {
    const { session, internals } = makePopulatedSession();

    await resetPreservingNativeCacheForWarmReuse(session);

    expect(internals.history).toEqual([]);
    expect(internals.lastImagesKey).toBeNull();
    expect(internals.lastAudioKey).toBeNull();
    expect(internals.turnCount).toBe(0);
    expect(internals.unresolvedOkToolCallCount).toBeNull();
    expect(internals.needsFullReplay).toBe(false);
    expect(internals.activeTools).toBeUndefined();
    // Public getters read the same state — turnCount 0 is what re-arms
    // `primeHistory()` for the next warm replay.
    expect(session.turns).toBe(0);
    expect(session.pendingUnresolvedToolCallCount).toBeNull();
    expect(session.hasImages).toBe(false);
  });

  it('preserves the native cache while clearing all JS media and replay state', async () => {
    const { model, session, internals } = makePopulatedSession();

    await resetPreservingNativeCacheForWarmReuse(session);

    expect(model.resetCaches).not.toHaveBeenCalled();
    expect(internals.lastAudioKey).toBeNull();
    expect(internals.needsFullReplay).toBe(false);
  });

  it('does not leak committed tools into an unrelated warm-reused conversation', async () => {
    const { model, session, internals } = makePopulatedSession();

    await resetPreservingNativeCacheForWarmReuse(session);
    session.primeHistory([{ role: 'user', content: 'new unrelated conversation' }]);
    await session.startFromHistory();

    expect(internals.activeTools).toBeUndefined();
    expect(model.startConfigs[0]?.tools).toBeUndefined();
  });

  it('restores constructor tool defaults instead of the prior committed overlay', async () => {
    const { model, session, internals } = makePopulatedSession({ tools: defaultTools });

    await resetPreservingNativeCacheForWarmReuse(session);
    session.primeHistory([{ role: 'user', content: 'new conversation with operator defaults' }]);
    await session.startFromHistory();

    expect(internals.activeTools).toBe(defaultTools);
    expect(model.startConfigs[0]?.tools).toBe(defaultTools);
  });

  it('full reset() DOES call model.resetCaches — the spy can tell the paths apart', async () => {
    const { model, session, internals } = makePopulatedSession();

    await session.reset();

    expect(model.resetCaches).toHaveBeenCalledTimes(1);
    expect(internals.lastAudioKey).toBeNull();
  });

  it('rejects while a send() is in flight and leaves state untouched', async () => {
    const { session, internals } = makePopulatedSession();
    internals.inFlight = true;

    await expect(resetPreservingNativeCacheForWarmReuse(session)).rejects.toThrow(
      'ChatSession: cannot resetPreservingNativeCacheForWarmReuse() while a send() is in flight; await the previous call first',
    );
    expect(internals.turnCount).toBe(3);
    expect(internals.history).toHaveLength(2);
    expect(internals.lastImagesKey).toBe('deadbeef');
    expect(internals.lastAudioKey).toBe('cafe');
    expect(internals.unresolvedOkToolCallCount).toBe(2);
    expect(internals.needsFullReplay).toBe(true);
    expect(internals.activeTools).toBe(previousTools);
  });
});
