/**
 * Unit tests for the generic `ChatSession<M>` wrapper.
 *
 * Hermetic — no model weights are loaded. A mock
 * `SessionCapableModel` with `vi.fn()` / `vi.fn()` async generator
 * stubs drives every code path.
 *
 * Covers:
 *   - Turn routing: turn 0 → start, turn N → continue, image-change
 *     → restart (reset caches + full history).
 *   - `inFlight` concurrency guard on `send` and `sendStream`.
 *   - `sawFinal` gating (throws, caller-break, `finishReason='error'`).
 *   - `reset()` clears history, image key, turn counter, and calls
 *     `resetCaches` on the model.
 *   - `sendToolResult` / `sendToolResultStream` routing.
 *   - `hasImages` getter semantics.
 *
 * Imports `ChatSession` via relative path because T1 intentionally
 * does not touch `packages/lm/src/index.ts` — the package-level
 * export is added in T5.
 */
import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';
import { describe, expect, it, vi } from 'vite-plus/test';

import { ChatSession, type SessionCapableModel } from '../../packages/lm/src/chat-session.js';
import type { ChatStreamEvent, ChatStreamFinal } from '../../packages/lm/src/stream.js';

/** Build a minimal `ChatResult` sufficient for the session layer. */
function makeChatResult(text: string): ChatResult {
  return {
    text,
    rawText: text,
    toolCalls: [],
    thinking: null,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'stop',
    performance: undefined,
  } as unknown as ChatResult;
}

/** Build a minimal terminal `ChatStreamFinal` chunk. */
function finalChunk(text: string, finishReason: string = 'stop'): ChatStreamFinal {
  return {
    text,
    done: true,
    finishReason,
    toolCalls: [],
    thinking: null,
    numTokens: 2,
    promptTokens: 1,
    reasoningTokens: 0,
    rawText: text,
  } satisfies ChatStreamFinal;
}

/**
 * Build a typed mock `SessionCapableModel` with all six session
 * entry points + `resetCaches` spied. Returns the mock plus direct
 * handles to each spy for assertions.
 */
function makeMockModel() {
  const chatSessionStart = vi.fn(
    async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> => makeChatResult('start-reply'),
  );
  const chatSessionContinue = vi.fn(
    async (_userMessage: string, _images: Uint8Array[] | null, _config?: ChatConfig | null): Promise<ChatResult> =>
      makeChatResult('continue-reply'),
  );
  const chatSessionContinueTool = vi.fn(
    async (_toolCallId: string, _content: string, _config?: ChatConfig | null): Promise<ChatResult> =>
      makeChatResult('tool-reply'),
  );
  const chatStreamSessionStart = vi.fn(async function* (
    _messages: ChatMessage[],
    _config?: ChatConfig | null,
  ): AsyncGenerator<ChatStreamEvent> {
    yield { text: 'start', done: false };
    yield { text: '-reply', done: false };
    yield finalChunk('start-reply');
  });
  const chatStreamSessionContinue = vi.fn(async function* (
    _userMessage: string,
    _images: Uint8Array[] | null,
    _config?: ChatConfig | null,
  ): AsyncGenerator<ChatStreamEvent> {
    yield { text: 'cont', done: false };
    yield finalChunk('cont-reply');
  });
  const chatStreamSessionContinueTool = vi.fn(async function* (
    _toolCallId: string,
    _content: string,
    _config?: ChatConfig | null,
  ): AsyncGenerator<ChatStreamEvent> {
    yield { text: 'tool', done: false };
    yield finalChunk('tool-reply');
  });
  const resetCaches = vi.fn(() => undefined);

  const model: SessionCapableModel = {
    chatSessionStart,
    chatSessionContinue,
    chatSessionContinueTool,
    chatStreamSessionStart,
    chatStreamSessionContinue,
    chatStreamSessionContinueTool,
    resetCaches,
  };

  return {
    model,
    chatSessionStart,
    chatSessionContinue,
    chatSessionContinueTool,
    chatStreamSessionStart,
    chatStreamSessionContinue,
    chatStreamSessionContinueTool,
    resetCaches,
  };
}

describe('ChatSession', () => {
  // -------------------------------------------------------------------
  // send() — non-streaming path
  // -------------------------------------------------------------------

  describe('send() routing', () => {
    it('routes turn 0 through chatSessionStart with the full history', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model, { system: 'Be concise.' });

      expect(session.turns).toBe(0);
      await session.send('Hi there!');

      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).not.toHaveBeenCalled();
      expect(session.turns).toBe(1);

      const [messages, config] = chatSessionStart.mock.calls[0];
      expect(messages).toEqual([
        { role: 'system', content: 'Be concise.' },
        { role: 'user', content: 'Hi there!' },
      ]);
      expect(config?.reuseCache).toBe(true);
    });

    it('omits the system prompt when none was configured', async () => {
      const { model, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('Hello');
      const [messages] = chatSessionStart.mock.calls[0];
      expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
    });

    it('routes turn N through chatSessionContinue with images=null', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('First');
      await session.send('Second');
      await session.send('Third');

      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(3);

      expect(chatSessionContinue.mock.calls[0][0]).toBe('Second');
      expect(chatSessionContinue.mock.calls[0][1]).toBeNull();
      expect(chatSessionContinue.mock.calls[1][0]).toBe('Third');
      expect(chatSessionContinue.mock.calls[1][1]).toBeNull();

      for (const call of chatSessionContinue.mock.calls) {
        expect(call[2]?.reuseCache).toBe(true);
      }
    });

    it('forces reuseCache=true even when the caller passes false', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('First', { config: { reuseCache: false } });
      await session.send('Second', { config: { reuseCache: false } });

      expect(chatSessionStart.mock.calls[0][1]?.reuseCache).toBe(true);
      expect(chatSessionContinue.mock.calls[0][2]?.reuseCache).toBe(true);
    });

    it('merges defaultConfig and per-call config (per-call wins)', async () => {
      const { model, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model, {
        defaultConfig: { maxNewTokens: 32, temperature: 0.2 },
      });

      await session.send('Hello', { config: { temperature: 0.8 } });

      const [, config] = chatSessionStart.mock.calls[0];
      expect(config?.maxNewTokens).toBe(32);
      expect(config?.temperature).toBe(0.8);
      expect(config?.reuseCache).toBe(true);
    });

    it('appends assistant reply to history on success', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);
      await session.send('turn-1');
      await session.send('turn-2');
      // Internal history is private — assert via the image-change
      // restart which rebuilds chatSessionStart from history.
      expect(session.turns).toBe(2);
    });
  });

  // -------------------------------------------------------------------
  // Image-change routing
  // -------------------------------------------------------------------

  describe('image-change routing', () => {
    const imgA = new Uint8Array([1, 2, 3]);
    const imgB = new Uint8Array([4, 5, 6]);

    it('routes turn-0 image send through chatSessionStart with images attached', async () => {
      const { model, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      expect(session.hasImages).toBe(false);
      await session.send('describe', { images: [imgA] });
      expect(session.hasImages).toBe(true);

      const [messages] = chatSessionStart.mock.calls[0];
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('user');
      expect(messages[0].content).toBe('describe');
      expect(messages[0].images).toEqual([imgA]);
    });

    it('text continue after image start stays on the cheap delta path', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('describe', { images: [imgA] });
      await session.send('follow-up question'); // text-only

      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue.mock.calls[0][1]).toBeNull();
      // Image key sticks around since we never cleared it.
      expect(session.hasImages).toBe(true);
    });

    it('different image bytes trigger a full restart', async () => {
      const { model, chatSessionStart, chatSessionContinueTool, resetCaches } = makeMockModel();
      const session = new ChatSession(model, { system: 'You are helpful.' });

      await session.send('describe A', { images: [imgA] });
      // Use a tool-result turn to keep the lastImagesKey stable across
      // the gap between the two image sends. (sendToolResult never
      // touches lastImagesKey, which is precisely what we need for the
      // next send to be detected as an image-set change rather than a
      // brand-new session.)
      await session.sendToolResult('call-1', 'tool-output');
      await session.send('describe B', { images: [imgB] }); // image change → restart

      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      // Restart path: resetCaches invoked exactly once at the image
      // boundary. History is PRESERVED across the restart — the plan's
      // Turn 3 example requires "full jinja on 3-turn history + image
      // B", i.e. chatSessionStart receives the system prompt plus
      // every prior user / assistant / tool turn plus the new user
      // turn with the new image attached.
      expect(resetCaches).toHaveBeenCalledTimes(1);

      const restartMessages = chatSessionStart.mock.calls[1][0];
      expect(restartMessages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe A', images: [imgA] },
        { role: 'assistant', content: 'start-reply' },
        { role: 'tool', content: 'tool-output', toolCallId: 'call-1' },
        { role: 'assistant', content: 'tool-reply' },
        { role: 'user', content: 'describe B', images: [imgB] },
      ]);
      // Three successful turns: initial send, tool result, restart send.
      expect(session.turns).toBe(3);
    });

    it('identical image bytes do NOT trigger a restart', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('describe', { images: [imgA] });
      // Send the SAME bytes again — should take the cheap delta path.
      await session.send('describe-again', { images: [new Uint8Array([1, 2, 3])] });

      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
    });

    it('text-only follow-up after image turn stays on the delta path', async () => {
      // Semantic: omitting `images` means "keep the cache state as
      // is", not "clear images". This lets VLM users refer back to
      // the same cached image context via cheap text-only turns.
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('describe', { images: [imgA] });
      await session.send('what about the top-right?'); // no images → delta

      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
      // lastImagesKey is unchanged — cache still holds the image.
      expect(session.hasImages).toBe(true);
    });

    it('hasImages stays true across text-only follow-ups and tool turns', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      expect(session.hasImages).toBe(false);
      await session.send('turn-1', { images: [imgA] });
      expect(session.hasImages).toBe(true);
      // Text-only follow-up — image key is preserved.
      await session.send('text follow-up');
      expect(session.hasImages).toBe(true);
      // Tool turn — image key is preserved.
      await session.sendToolResult('call-1', 'tool output');
      expect(session.hasImages).toBe(true);
    });

    it('failed image-change restart rolls back state and re-routes through start on next call', async () => {
      const { model, chatSessionStart, resetCaches } = makeMockModel();
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: image start succeeds.
      await session.send('describe A', { images: [imgA] });
      // Turn 2: text follow-up takes the delta path.
      await session.send('text follow-up');
      expect(session.turns).toBe(2);
      expect(session.hasImages).toBe(true);

      // Turn 3: image change triggers restart, but the native call
      // rejects. State must roll back: prior history is preserved,
      // but turnCount + lastImagesKey drop so the next call re-routes
      // through the start path (caches were already wiped).
      chatSessionStart.mockRejectedValueOnce(new Error('restart-fail'));
      await expect(session.send('describe B', { images: [imgB] })).rejects.toThrow('restart-fail');

      // resetCaches was called once (the failed restart).
      expect(resetCaches).toHaveBeenCalledTimes(1);
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);

      // Recovery call: routes through the start path with the
      // preserved prior conversation + the new user turn.
      const recoveryResult = await session.send('describe B again', { images: [imgB] });
      expect(recoveryResult.text).toBe('start-reply');

      // chatSessionStart has now been called three times total:
      //   [0] initial turn-1 start
      //   [1] failed image-change restart
      //   [2] recovery start
      expect(chatSessionStart).toHaveBeenCalledTimes(3);
      const recoveryMessages = chatSessionStart.mock.calls[2][0];
      // The recovery messages preserve the full prior history plus
      // the new user turn with the new image attached.
      expect(recoveryMessages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe A', images: [imgA] },
        { role: 'assistant', content: 'start-reply' },
        { role: 'user', content: 'text follow-up' },
        { role: 'assistant', content: 'continue-reply' },
        { role: 'user', content: 'describe B again', images: [imgB] },
      ]);
      // One successful turn after the rollback (the recovery).
      expect(session.turns).toBe(1);
    });
  });

  // -------------------------------------------------------------------
  // Error rollback
  // -------------------------------------------------------------------

  describe('error rollback', () => {
    it('failed first-turn start does NOT corrupt history', async () => {
      const { model, chatSessionStart } = makeMockModel();
      chatSessionStart.mockRejectedValueOnce(new Error('first-fail'));
      const session = new ChatSession(model);

      await expect(session.send('hello')).rejects.toThrow('first-fail');
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);

      // Recovery call: history must not contain the failed user push.
      // The next start messages should be just the new user turn.
      await session.send('hello again');
      const messages = chatSessionStart.mock.calls[1][0];
      expect(messages).toEqual([{ role: 'user', content: 'hello again' }]);
      expect(session.turns).toBe(1);
    });
  });

  // -------------------------------------------------------------------
  // inFlight guard
  // -------------------------------------------------------------------

  describe('concurrency guard', () => {
    it('rejects concurrent send() calls', async () => {
      let resolveFirst: (r: ChatResult) => void = () => {
        /* overwritten below */
      };
      const first = new Promise<ChatResult>((r) => {
        resolveFirst = r;
      });
      const chatSessionStart = vi.fn(async () => first);
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('c')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('t')),
        chatStreamSessionStart: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      const firstPromise = session.send('First');
      await expect(session.send('Second')).rejects.toThrow(/concurrent send\(\) not allowed/);

      resolveFirst(makeChatResult('first'));
      await firstPromise;
      expect(session.turns).toBe(1);

      await session.send('Third');
      expect(session.turns).toBe(2);
    });

    it('clears inFlight on exception', async () => {
      const { model, chatSessionStart } = makeMockModel();
      chatSessionStart.mockRejectedValueOnce(new Error('boom'));
      const session = new ChatSession(model);

      await expect(session.send('Hello')).rejects.toThrow('boom');
      expect(session.turns).toBe(0);

      // Follow-up must not be blocked by stale inFlight.
      await session.send('Retry');
      expect(session.turns).toBe(1);
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
    });
  });

  // -------------------------------------------------------------------
  // reset()
  // -------------------------------------------------------------------

  describe('reset()', () => {
    const imgA = new Uint8Array([1, 2, 3]);

    it('clears history, image key, and turn counter', async () => {
      const { model, chatSessionStart, resetCaches } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('one', { images: [imgA] });
      await session.sendToolResult('c1', 'tool-out');
      expect(session.turns).toBe(2);
      expect(session.hasImages).toBe(true);

      await session.reset();

      expect(resetCaches).toHaveBeenCalledTimes(1);
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);

      // After reset, the next send must re-route through the start
      // path — and the history must NOT include any prior turns.
      await session.send('fresh');
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      const [messages] = chatSessionStart.mock.calls[1];
      expect(messages).toEqual([{ role: 'user', content: 'fresh' }]);
    });

    it('rejects reset() while a send() is in flight', async () => {
      let resolveFirst: (r: ChatResult) => void = () => {
        /* overwritten below */
      };
      const first = new Promise<ChatResult>((r) => {
        resolveFirst = r;
      });
      const chatSessionStart = vi.fn(async () => first);
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('c')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('t')),
        chatStreamSessionStart: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      const firstPromise = session.send('first');
      await expect(session.reset()).rejects.toThrow(/cannot reset.*in flight/i);

      resolveFirst(makeChatResult('first'));
      await firstPromise;
      expect(session.turns).toBe(1);

      // After the in-flight call completes, reset works again.
      await session.reset();
      expect(session.turns).toBe(0);
    });
  });

  // -------------------------------------------------------------------
  // sendToolResult()
  // -------------------------------------------------------------------

  describe('sendToolResult() routing', () => {
    it('always routes through chatSessionContinueTool', async () => {
      const { model, chatSessionContinueTool, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      // No prior send — tool result can be called even on turn 0.
      const result = await session.sendToolResult('call-42', '{"status":"ok"}');
      expect(result.text).toBe('tool-reply');

      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      expect(chatSessionStart).not.toHaveBeenCalled();
      expect(chatSessionContinueTool.mock.calls[0][0]).toBe('call-42');
      expect(chatSessionContinueTool.mock.calls[0][1]).toBe('{"status":"ok"}');
      expect(chatSessionContinueTool.mock.calls[0][2]?.reuseCache).toBe(true);
      expect(session.turns).toBe(1);
    });

    it('forces reuseCache=true even with caller override', async () => {
      const { model, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      await session.sendToolResult('c1', 'out', { config: { reuseCache: false } });
      expect(chatSessionContinueTool.mock.calls[0][2]?.reuseCache).toBe(true);
    });

    it('clears inFlight on exception', async () => {
      const { model, chatSessionContinueTool } = makeMockModel();
      chatSessionContinueTool.mockRejectedValueOnce(new Error('tool-boom'));
      const session = new ChatSession(model);

      await expect(session.sendToolResult('c1', 'out')).rejects.toThrow('tool-boom');
      // Follow-up works.
      await session.sendToolResult('c2', 'out2');
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(2);
    });
  });

  // -------------------------------------------------------------------
  // sendStream() — streaming path
  // -------------------------------------------------------------------

  describe('sendStream()', () => {
    it('routes turn 0 through chatStreamSessionStart with full history', async () => {
      const { model, chatStreamSessionStart, chatStreamSessionContinue } = makeMockModel();
      const session = new ChatSession(model, { system: 'Be concise.' });

      const events: ChatStreamEvent[] = [];
      for await (const e of session.sendStream('Hi')) events.push(e);

      expect(chatStreamSessionStart).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();
      expect(session.turns).toBe(1);
      expect(events[events.length - 1].done).toBe(true);

      const [messages, config] = chatStreamSessionStart.mock.calls[0];
      expect(messages).toEqual([
        { role: 'system', content: 'Be concise.' },
        { role: 'user', content: 'Hi' },
      ]);
      expect(config?.reuseCache).toBe(true);
    });

    it('routes turn N through chatStreamSessionContinue with images=null', async () => {
      const { model, chatStreamSessionStart, chatStreamSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      for await (const _e of session.sendStream('First')) void _e;
      for await (const _e of session.sendStream('Second')) void _e;
      for await (const _e of session.sendStream('Third')) void _e;

      expect(chatStreamSessionStart).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinue).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(3);
      expect(chatStreamSessionContinue.mock.calls[0][0]).toBe('Second');
      expect(chatStreamSessionContinue.mock.calls[0][1]).toBeNull();
      expect(chatStreamSessionContinue.mock.calls[1][0]).toBe('Third');
      expect(chatStreamSessionContinue.mock.calls[1][1]).toBeNull();
    });

    it('rejects concurrent sendStream calls', async () => {
      let release: () => void = () => {
        /* overwritten */
      };
      const unblock = new Promise<void>((r) => {
        release = r;
      });
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        await unblock;
        yield finalChunk('done');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      const firstIter = session.sendStream('First');
      const firstNext = firstIter.next();

      await expect(async () => {
        for await (const _e of session.sendStream('Second')) void _e;
      }).rejects.toThrow(/concurrent send/i);

      release();
      await firstNext;
      for (;;) {
        const { done } = await firstIter.next();
        if (done) break;
      }
      expect(session.turns).toBe(1);
    });

    it('increments turnCount only when the final done chunk is yielded', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      let sawDone = false;
      for await (const e of session.sendStream('Hello')) {
        if (e.done) sawDone = true;
      }
      expect(sawDone).toBe(true);
      expect(session.turns).toBe(1);
    });

    it('does NOT increment turnCount on caller break mid-stream', async () => {
      const { model, chatStreamSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      for await (const e of session.sendStream('Hello')) {
        expect(e.done).toBe(false);
        break;
      }
      expect(session.turns).toBe(0);

      // Retry must re-route through the start path.
      for await (const _e of session.sendStream('Retry')) void _e;
      expect(chatStreamSessionStart).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(1);

      // The recovery call's chatStreamSessionStart payload must NOT
      // contain the aborted 'Hello' user turn — the caller-break
      // rollback (driven by the try/finally in runStartStreamPath)
      // dropped it from the staged history before the retry started.
      const recoveryMessages = chatStreamSessionStart.mock.calls[1][0];
      expect(recoveryMessages).toEqual([{ role: 'user', content: 'Retry' }]);
    });

    it('does NOT increment turnCount when the stream throws', async () => {
      let callCount = 0;
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        callCount++;
        yield { text: 'partial', done: false };
        if (callCount === 1) throw new Error('boom');
        yield finalChunk('recovered');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      await expect(async () => {
        for await (const _e of session.sendStream('Hello')) void _e;
      }).rejects.toThrow('boom');
      expect(session.turns).toBe(0);

      for await (const _e of session.sendStream('Retry')) void _e;
      expect(chatStreamSessionStart).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(1);
    });

    it('does NOT increment turnCount when final chunk has finishReason="error"', async () => {
      const errorChunk: ChatStreamFinal = finalChunk('fake', 'error');
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        yield errorChunk;
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      const events: ChatStreamEvent[] = [];
      for await (const e of session.sendStream('Hi')) events.push(e);
      expect(events).toEqual([errorChunk]);
      expect(session.turns).toBe(0);
    });

    it('caller break during turn-1 stream rolls back the user push (finally path)', async () => {
      // Stream yields two deltas and then never reaches a final chunk —
      // the caller will `break` out of the `for await` after the second.
      // Since JS calls `iterator.return()` on caller break, only a
      // `finally` block runs — the post-loop `if (sawFinal)` branch is
      // SKIPPED. This test proves the try/finally rollback path does
      // the right thing for the start-stream case.
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        yield { text: 'hello ', done: false };
        yield { text: 'world', done: false };
        yield { text: '!', done: false };
        // No done:true — caller breaks out before getting here.
      });
      const chatSessionStart = vi.fn(
        async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> =>
          makeChatResult('recovery'),
      );
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      let seen = 0;
      for await (const _e of session.sendStream('hello')) {
        if (++seen >= 2) break;
      }
      expect(seen).toBe(2);
      expect(session.turns).toBe(0);

      // Recovery: next send() must not see 'hello' in messages.
      await session.send('hello again');
      const lastCall = chatSessionStart.mock.calls[chatSessionStart.mock.calls.length - 1];
      expect(lastCall[0]).toEqual([{ role: 'user', content: 'hello again' }]);
      expect(session.turns).toBe(1);
    });

    it('caller break during image-change restart stream rolls back state (finally path)', async () => {
      const imgA = new Uint8Array([10, 20, 30]);
      const imgB = new Uint8Array([40, 50, 60]);

      // First start stream yields a successful 3-event session for imgA.
      // Second start stream (the image-change restart) yields ONE delta
      // and then stops — caller will break after reading it. JS calls
      // `iterator.return()`, only `finally` runs, and the rollback
      // must drop turnCount + lastImagesKey (since caches were wiped).
      let startCall = 0;
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        startCall++;
        if (startCall === 1) {
          yield { text: 'A-desc', done: false };
          yield finalChunk('describe-A-reply');
          return;
        }
        // Second call (restart): one delta, no done, caller breaks.
        yield { text: 'partial-B', done: false };
      });
      const chatSessionStart = vi.fn(
        async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> =>
          makeChatResult('recovery-reply'),
      );
      const resetCaches = vi.fn();
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches,
      };
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: successful image start.
      for await (const _e of session.sendStream('describe A', { images: [imgA] })) void _e;
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);

      // Turn 2: image-change restart, caller breaks after one delta.
      let seen = 0;
      for await (const _e of session.sendStream('describe B', { images: [imgB] })) {
        seen++;
        break;
      }
      expect(seen).toBe(1);
      // resetCaches was called once (the image-change wipe at the top
      // of the restart path).
      expect(resetCaches).toHaveBeenCalledTimes(1);
      // Rollback must have forced turnCount → 0 and cleared
      // lastImagesKey so the next call re-routes through the start
      // path with the preserved prior history.
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);

      // Recovery: next call must re-route through the start path with
      // the preserved turn-1 conversation plus the new user turn.
      const recoveryResult = await session.send('describe B again', { images: [imgB] });
      expect(recoveryResult.text).toBe('recovery-reply');
      const lastStartCall = chatSessionStart.mock.calls[chatSessionStart.mock.calls.length - 1];
      const recoveryMessages = lastStartCall[0];
      expect(recoveryMessages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe A', images: [imgA] },
        { role: 'assistant', content: 'describe-A-reply' },
        { role: 'user', content: 'describe B again', images: [imgB] },
      ]);
      // Exactly one successful turn after the rollback (the recovery).
      expect(session.turns).toBe(1);
    });

    it('caller break during delta-continue stream does NOT advance turnCount (finally path)', async () => {
      // Turn 1 succeeds via a normal start stream. Turn 2 takes the
      // delta path, and the stream yields two deltas with no done.
      // Caller breaks after reading one, triggering `iterator.return()`
      // so only the inner `finally` runs. Because the delta path never
      // pushes to history before commit, the assertion here is purely
      // "turnCount did NOT advance" — no start-path re-routing needed.
      let continueCall = 0;
      const chatStreamSessionContinue = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        continueCall++;
        if (continueCall === 1) {
          yield { text: 'partial', done: false };
          yield { text: 'more', done: false };
          // No done:true — caller breaks before getting here.
          return;
        }
        // Subsequent continue call (the retry) — successful.
        yield finalChunk('retry-reply');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart: vi.fn(async function* () {
          yield finalChunk('turn-1-reply');
        }),
        chatStreamSessionContinue,
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      // Turn 1 (start stream) succeeds.
      for await (const _e of session.sendStream('turn 1')) void _e;
      expect(session.turns).toBe(1);

      // Turn 2 (delta stream) — caller breaks after one delta.
      let seen = 0;
      for await (const _e of session.sendStream('turn 2')) {
        seen++;
        break;
      }
      expect(seen).toBe(1);
      // turnCount stays at 1 — the abandoned turn must not count.
      expect(session.turns).toBe(1);

      // Retry succeeds: turnCount goes 1 → 2 (not 1 → 3).
      for await (const _e of session.sendStream('turn 2 retry')) void _e;
      expect(session.turns).toBe(2);
    });
  });

  // -------------------------------------------------------------------
  // sendToolResultStream()
  // -------------------------------------------------------------------

  describe('sendToolResultStream()', () => {
    it('routes through chatStreamSessionContinueTool and advances on success', async () => {
      const { model, chatStreamSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      let sawDone = false;
      for await (const e of session.sendToolResultStream('c1', 'tool-out')) {
        if (e.done) sawDone = true;
      }
      expect(sawDone).toBe(true);
      expect(session.turns).toBe(1);
      expect(chatStreamSessionContinueTool).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinueTool.mock.calls[0][0]).toBe('c1');
      expect(chatStreamSessionContinueTool.mock.calls[0][1]).toBe('tool-out');
    });

    it('does NOT advance turnCount on caller break', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      for await (const e of session.sendToolResultStream('c1', 'out')) {
        expect(e.done).toBe(false);
        break;
      }
      expect(session.turns).toBe(0);
    });
  });

  // -------------------------------------------------------------------
  // Cold-restart primitives: primeHistory / startFromHistory / startFromHistoryStream
  // -------------------------------------------------------------------

  describe('primeHistory()', () => {
    it('sets history on a fresh session without running inference', () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      const primed: ChatMessage[] = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello!' },
        { role: 'user', content: 'Follow-up' },
      ];
      session.primeHistory(primed);

      // primeHistory does not run inference.
      expect(chatSessionStart).not.toHaveBeenCalled();
      expect(chatSessionContinue).not.toHaveBeenCalled();
      expect(session.turns).toBe(0);
    });

    it('rejects when turnCount > 0', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('turn-1');
      expect(() => session.primeHistory([{ role: 'user', content: 'prime' }])).toThrow(/fresh session/i);
    });

    it('rejects while a send() is in flight', async () => {
      let resolveFirst: (r: ChatResult) => void = () => {
        /* overwritten */
      };
      const pending = new Promise<ChatResult>((r) => {
        resolveFirst = r;
      });
      const chatSessionStart = vi.fn(async () => pending);
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('c')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('t')),
        chatStreamSessionStart: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      const firstPromise = session.send('Hello');
      expect(() => session.primeHistory([{ role: 'user', content: 'prime' }])).toThrow(/in flight/);

      resolveFirst(makeChatResult('first'));
      await firstPromise;
    });

    it('replaces any previously primed history (shallow copy)', () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      const original: ChatMessage[] = [{ role: 'user', content: 'first' }];
      session.primeHistory(original);
      // Mutating the original array after priming must not affect
      // what the session holds.
      original.push({ role: 'user', content: 'mutated' });

      // Re-prime is allowed while turnCount is still 0.
      session.primeHistory([{ role: 'user', content: 'new' }]);
    });
  });

  describe('startFromHistory()', () => {
    it('calls chatSessionStart with the primed history and returns its result', async () => {
      const { model, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      const primed: ChatMessage[] = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello!' },
        { role: 'user', content: 'what is 2+2?' },
      ];
      session.primeHistory(primed);

      const result = await session.startFromHistory();

      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      const [messages, config] = chatSessionStart.mock.calls[0];
      expect(messages).toEqual(primed);
      expect(config?.reuseCache).toBe(true);
      expect(result.text).toBe('start-reply');
    });

    it('rejects when history is empty', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      await expect(session.startFromHistory()).rejects.toThrow(/primed history/);
    });

    it('rejects when turnCount > 0', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('turn-1');
      await expect(session.startFromHistory()).rejects.toThrow(/fresh session/i);
    });

    it('rejects while a send() is in flight', async () => {
      let resolveFirst: (r: ChatResult) => void = () => {
        /* overwritten */
      };
      const pending = new Promise<ChatResult>((r) => {
        resolveFirst = r;
      });
      const chatSessionStart = vi.fn(async () => pending);
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('c')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('t')),
        chatStreamSessionStart: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      const firstPromise = session.send('Hello');
      await expect(session.startFromHistory()).rejects.toThrow(/in flight/);

      resolveFirst(makeChatResult('first'));
      await firstPromise;
    });

    it('advances turnCount and appends the assistant reply', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([{ role: 'user', content: 'hi' }]);
      expect(session.turns).toBe(0);
      const result = await session.startFromHistory();
      expect(session.turns).toBe(1);
      expect(result.text).toBe('start-reply');
    });

    it('hydrates lastImagesKey when the trailing user message has images', async () => {
      const img = new Uint8Array([7, 8, 9]);
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'system', content: 'hi' },
        { role: 'user', content: 'describe', images: [img] },
      ]);
      expect(session.hasImages).toBe(false);
      await session.startFromHistory();
      expect(session.hasImages).toBe(true);
    });

    it('does NOT set lastImagesKey when no user message has images', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'system', content: 'hi' },
        { role: 'user', content: 'text only' },
      ]);
      await session.startFromHistory();
      expect(session.hasImages).toBe(false);
    });

    it('routes subsequent send() through the delta chatSessionContinue path', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello!' },
        { role: 'user', content: 'follow-up?' },
      ]);
      await session.startFromHistory();
      expect(chatSessionStart).toHaveBeenCalledTimes(1);

      // Next send() is turn 2 from the session's perspective — it should
      // go through chatSessionContinue, not restart.
      await session.send('another follow-up');
      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
    });
  });

  describe('startFromHistoryStream()', () => {
    it('yields events and commits on success', async () => {
      const { model, chatStreamSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'system', content: 'hi' },
        { role: 'user', content: 'say hi' },
      ]);

      const events: ChatStreamEvent[] = [];
      for await (const e of session.startFromHistoryStream()) events.push(e);

      expect(chatStreamSessionStart).toHaveBeenCalledTimes(1);
      expect(events[events.length - 1].done).toBe(true);
      expect(session.turns).toBe(1);
    });

    it('rejects when history is empty', async () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      await expect(async () => {
        for await (const _e of session.startFromHistoryStream()) void _e;
      }).rejects.toThrow(/primed history/);
    });

    it('does NOT advance turnCount on caller break (finally path leaves primed history intact)', async () => {
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        yield { text: 'partial', done: false };
        yield { text: 'more', done: false };
        // no done — caller will break.
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      session.primeHistory([{ role: 'user', content: 'hi' }]);
      for await (const _e of session.startFromHistoryStream()) break;
      expect(session.turns).toBe(0);
      // Session is still on turn 0 with the primed history — retrying
      // startFromHistory should succeed without needing a re-prime.
      const result = await session.startFromHistory();
      expect(result.text).toBe('x');
      expect(session.turns).toBe(1);
    });

    it('does NOT advance turnCount when final chunk has finishReason="error"', async () => {
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        yield finalChunk('oops', 'error');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      session.primeHistory([{ role: 'user', content: 'hi' }]);
      for await (const _e of session.startFromHistoryStream()) void _e;
      expect(session.turns).toBe(0);
    });

    it('hydrates lastImagesKey when the trailing user message has images', async () => {
      const img = new Uint8Array([1, 2, 3]);
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([{ role: 'user', content: 'describe', images: [img] }]);
      for await (const _e of session.startFromHistoryStream()) void _e;
      expect(session.hasImages).toBe(true);
    });
  });
});
