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
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import type { ChatStreamEvent, ChatStreamFinal } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { resetPreservingNativeCacheForWarmReuse } from '../../packages/server/src/chat-session-warm-reuse.js';

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

/**
 * Build a `ChatResult` that carries a single outstanding `ok` tool
 * call. Used to establish the pre-conditions for `sendToolResult*`
 * tests: the chat-session API requires an unresolved single-call
 * assistant turn before any tool-result dispatch is legal.
 */
function makeChatResultWithSingleToolCall(text: string, callId: string): ChatResult {
  return {
    ...makeChatResult(text),
    toolCalls: [{ id: callId, name: 'tool_fn', arguments: '{}', status: 'ok' }],
  } as unknown as ChatResult;
}

/**
 * Build a terminal stream chunk carrying a single outstanding `ok`
 * tool call. Streaming counterpart of `makeChatResultWithSingleToolCall`.
 */
function finalChunkWithSingleToolCall(text: string, callId: string): ChatStreamFinal {
  return {
    ...finalChunk(text),
    toolCalls: [{ id: callId, name: 'tool_fn', arguments: '{}', status: 'ok' }],
  } as unknown as ChatStreamFinal;
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
    cachedTokens: 0,
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
    async (
      _userMessage: string,
      _images: Uint8Array[] | null,
      _audio: Uint8Array[] | null,
      _config?: ChatConfig | null,
    ): Promise<ChatResult> => makeChatResult('continue-reply'),
  );
  const chatSessionContinueTool = vi.fn(
    async (
      _toolCallId: string,
      _content: string,
      _config?: ChatConfig | null,
      _isError?: boolean | null,
    ): Promise<ChatResult> => makeChatResult('tool-reply'),
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
    _audio: Uint8Array[] | null,
    _config?: ChatConfig | null,
  ): AsyncGenerator<ChatStreamEvent> {
    yield { text: 'cont', done: false };
    yield finalChunk('cont-reply');
  });
  const chatStreamSessionContinueTool = vi.fn(async function* (
    _toolCallId: string,
    _content: string,
    _config?: ChatConfig | null,
    _signal?: AbortSignal,
    _isError?: boolean | null,
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
      // arg[2] is the new audio param; delta continues never carry audio.
      expect(chatSessionContinue.mock.calls[0][2]).toBeNull();
      expect(chatSessionContinue.mock.calls[1][0]).toBe('Third');
      expect(chatSessionContinue.mock.calls[1][1]).toBeNull();
      expect(chatSessionContinue.mock.calls[1][2]).toBeNull();

      for (const call of chatSessionContinue.mock.calls) {
        expect(call[3]?.reuseCache).toBe(true);
      }
    });

    it('forces reuseCache=true even when the caller passes false', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('First', { config: { reuseCache: false } });
      await session.send('Second', { config: { reuseCache: false } });

      expect(chatSessionStart.mock.calls[0][1]?.reuseCache).toBe(true);
      expect(chatSessionContinue.mock.calls[0][3]?.reuseCache).toBe(true);
    });

    // Regression guard for the Phase 3b audio-surface ABI shift. The
    // shared chat surface inserts an `audio` positional argument between
    // `images` and `config`. A QianfanOCR-shaped model (handwritten NAPI
    // methods, not the `chat_napi_surface!` macro) must carry the SAME
    // positional ABI as the macro families, or the delta path would push
    // the caller's `config` into the audio slot and silently drop it. We
    // model the native binding as a positional recorder (it only knows
    // arguments by index) and assert `config` lands in slot 3, never the
    // audio slot.
    it('delivers (userMessage, images, audio, config) to a Qianfan-shaped continue without dropping config', async () => {
      const continueArgs: unknown[][] = [];
      const qianfanShaped: SessionCapableModel = {
        chatSessionStart: vi.fn(
          async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> =>
            makeChatResult('start-reply'),
        ),
        // Positional recorder: a native NAPI binding sees its arguments by
        // index only, so capturing `arguments` here detects any slot shift.
        chatSessionContinue: vi.fn(async function (this: unknown, ...args: unknown[]): Promise<ChatResult> {
          continueArgs.push(args);
          return makeChatResult('continue-reply');
        }) as unknown as SessionCapableModel['chatSessionContinue'],
        chatSessionContinueTool: vi.fn(
          async (): Promise<ChatResult> => makeChatResult('tool-reply'),
        ) as unknown as SessionCapableModel['chatSessionContinueTool'],
        chatStreamSessionStart: vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
          yield finalChunk('start-reply');
        }) as unknown as SessionCapableModel['chatStreamSessionStart'],
        chatStreamSessionContinue: vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
          yield finalChunk('cont-reply');
        }) as unknown as SessionCapableModel['chatStreamSessionContinue'],
        chatStreamSessionContinueTool: vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
          yield finalChunk('tool-reply');
        }) as unknown as SessionCapableModel['chatStreamSessionContinueTool'],
        resetCaches: vi.fn(() => undefined),
      };

      const session = new ChatSession(qianfanShaped, {
        defaultConfig: { maxNewTokens: 7 },
      });

      // Turn 0 routes through start; turn 1 takes the cheap delta path.
      await session.send('First');
      await session.send('Second');

      expect(continueArgs).toHaveLength(1);
      const [userMessage, images, audio, config] = continueArgs[0];
      expect(userMessage).toBe('Second');
      expect(images).toBeNull();
      // audio MUST be the third positional argument and null on a delta —
      // if the ABI were the pre-fix shape, `config` would land here.
      expect(audio).toBeNull();
      // config MUST land in slot 3, fully intact (not shifted into audio).
      expect(config).toBeDefined();
      expect((config as ChatConfig | undefined)?.maxNewTokens).toBe(7);
      expect((config as ChatConfig | undefined)?.reuseCache).toBe(true);
      // No stray fifth positional argument.
      expect(continueArgs[0]).toHaveLength(4);
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

    // -----------------------------------------------------------------
    // W7 (MTP): `enableMtp` auto-default
    // -----------------------------------------------------------------

    it('auto-defaults enableMtp=true when the model exposes hasMtpWeights()==true', async () => {
      const { model, chatSessionStart } = makeMockModel();
      (model as SessionCapableModel).hasMtpWeights = () => true;
      const session = new ChatSession(model);

      await session.send('Hello');

      const [, config] = chatSessionStart.mock.calls[0];
      expect(config?.enableMtp).toBe(true);
    });

    it('does not set enableMtp when the model exposes hasMtpWeights()==false', async () => {
      const { model, chatSessionStart } = makeMockModel();
      (model as SessionCapableModel).hasMtpWeights = () => false;
      const session = new ChatSession(model);

      await session.send('Hello');

      const [, config] = chatSessionStart.mock.calls[0];
      // Auto-default never fires → property stays undefined (not `false`),
      // mirroring the contract from the JSDoc on `mergeConfig`.
      expect(config?.enableMtp).toBeUndefined();
    });

    it('does not set enableMtp when the model omits hasMtpWeights() entirely', async () => {
      // Models predating W7 (Qwen3, Gemma4, LFM2, etc.) do NOT define
      // `hasMtpWeights` on their native wrapper. The duck check inside
      // `mergeConfig` must skip the auto-default cleanly.
      const { model, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('Hello');

      const [, config] = chatSessionStart.mock.calls[0];
      expect(config?.enableMtp).toBeUndefined();
    });

    it('respects an explicit enableMtp=false even when the model has MTP weights', async () => {
      // An explicit opt-out from the caller must win over the auto-
      // default — operators benchmarking MTP-vs-AR need to be able to
      // force the AR path on a checkpoint that ships an MTP head.
      const { model, chatSessionStart } = makeMockModel();
      (model as SessionCapableModel).hasMtpWeights = () => true;
      const session = new ChatSession(model);

      await session.send('Hello', { config: { enableMtp: false } });

      const [, config] = chatSessionStart.mock.calls[0];
      expect(config?.enableMtp).toBe(false);
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

      // Seed the first turn so it emits a single ok tool call — the
      // tool-result turn that follows needs a legal outstanding
      // single-call state before the image-change restart can run.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('start-reply', 'call-1'));

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
        // The assistant turn that emitted the `call-1` tool call must
        // carry `toolCalls` through to the replayed history so the
        // subsequent `role: 'tool'` message remains paired with a
        // declaring assistant turn on the jinja render. Dropping it
        // would corrupt the replayed conversation structure.
        {
          role: 'assistant',
          content: 'start-reply',
          toolCalls: [{ id: 'call-1', name: 'tool_fn', arguments: '{}' }],
        },
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

    it('large image buffers: identical bytes stay on delta path, single-byte-different bytes restart', async () => {
      // The 3-byte imgA/imgB fixtures above never exceed a single
      // hash chunk. This exercises the image-identity hash at the
      // byte range the native-crypto perf fix targets (multi-MB
      // buffers), guarding against regressions in the length-prefix
      // framing across a large `hash.update()` call.
      const bigA = new Uint8Array(2 * 1024 * 1024);
      for (let i = 0; i < bigA.length; i++) bigA[i] = i & 0xff;
      const bigACopy = Uint8Array.from(bigA);
      const bigBLastByteFlipped = Uint8Array.from(bigA);
      bigBLastByteFlipped[bigBLastByteFlipped.length - 1] ^= 0xff;

      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('describe', { images: [bigA] });
      // Same bytes via a distinct Uint8Array instance -> cheap delta path.
      await session.send('describe-again', { images: [bigACopy] });
      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);

      // Flip only the trailing byte -> must be detected as a changed
      // image set and trigger a full restart.
      await session.send('describe-changed', { images: [bigBLastByteFlipped] });
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
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

    it('an audio-change restart preserves the untouched image key and does not re-detect the same image', async () => {
      // A restart driven by a change in only ONE modality replays the
      // FULL history through the start path, so the post-restart sticky
      // keys must be derived from that history — not from the single
      // turn's literal media args (which null the untouched modality).
      const audX = new Uint8Array([9, 9, 9]);
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      // Turn 1: image start.
      await session.send('describe', { images: [imgA] });
      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(session.hasImages).toBe(true);

      // Turn 2: audio-only turn → audio change → full restart. Image A is
      // still present in the replayed history, so the image key must be
      // rehydrated from history, not nulled by the per-turn audio args.
      await session.send('transcribe', { audio: [audX] });
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      // The image is still live in the native cache — hasImages must NOT
      // lie (this assertion fails before the fix, passes after).
      expect(session.hasImages).toBe(true);

      // Turn 3: refer back to the SAME image A. Because lastImagesKey was
      // preserved as key(A), this is NOT detected as an image change, so
      // it routes through the cheap delta path — no spurious third start
      // that would duplicate image A in the prompt.
      await session.send('what was in the image?', { images: [imgA] });
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
    });

    it('hasImages stays true across text-only follow-ups and tool turns', async () => {
      const { model, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      expect(session.hasImages).toBe(false);
      await session.send('turn-1', { images: [imgA] });
      expect(session.hasImages).toBe(true);
      // Text-only follow-up — mock the continue response as a
      // single-call turn so the subsequent sendToolResult has a legal
      // outstanding-call state.
      chatSessionContinue.mockResolvedValueOnce(makeChatResultWithSingleToolCall('continue-with-call', 'call-1'));
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
      // Seed the first turn with an outstanding single-call so the
      // sendToolResult that follows has a legal pre-state.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('with-call', 'c1'));
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

    it('public reset() always full-wipes — the keepNativeCache option is not accepted (Round 5 Fix #1)', async () => {
      // Public contract: `ChatSession.reset()` is always a full wipe.
      // Round 4 accidentally exposed a `{ keepNativeCache: true }`
      // option that downstream consumers could call in a context
      // where the shared native model still held an unrelated
      // request's cache, reintroducing the cross-request cache-
      // affinity leak that Round 3 closed. Round 5 removed the
      // option from the public surface — the preserved-cache path
      // is now behind the helper
      // `resetPreservingNativeCacheForWarmReuse(session)`, which
      // lives inside `@mlx-node/server` and is called exclusively by
      // `SessionRegistry`-gated server endpoints. (Round 6 Fix #1
      // refactored this from a class method into a module-level
      // function; Round 7 Fix #2 relocated the module into the
      // server package itself so there is no `@mlx-node/lm` export
      // surface the helper could leak through.)
      const { model, resetCaches } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('one');
      expect(session.turns).toBe(1);

      // No-argument form always wipes.
      await session.reset();
      expect(resetCaches).toHaveBeenCalledTimes(1);
      expect(session.turns).toBe(0);
    });

    it('resetPreservingNativeCacheForWarmReuse() wipes JS state only — no resetCaches call (Round 5 Fix #1 / Round 6 Fix #1 internal helper)', async () => {
      // The module-level helper that replaced Round 4's
      // `reset({ keepNativeCache: true })` public option, then Round 5's
      // `_resetPreservingNativeCacheForWarmReuse()` class method. Used
      // only by the server-side SessionRegistry warm-replay path on a
      // tier-1 / tier-2 HIT, where the registry authoritatively vouches
      // for the native cache belonging to this chain. Verify:
      // (1) no resetCaches call, (2) turns/history zeroed so
      // primeHistory() will accept the session.
      const { model, resetCaches } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('one');
      await session.send('two');
      expect(session.turns).toBe(2);

      await resetPreservingNativeCacheForWarmReuse(session);
      expect(resetCaches).not.toHaveBeenCalled();
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);
    });

    it('resetPreservingNativeCacheForWarmReuse is NOT exported from @mlx-node/lm public surface (Round 6 Fix #1 / Round 7 Fix #2)', async () => {
      // Structural guard: the helper lives inside `@mlx-node/server`
      // (server-private module) and MUST NOT appear in either the
      // `@mlx-node/lm` main export surface or the `@mlx-node/server`
      // main export surface. Round 6 Fix #1 enforced the lm-public
      // absence; Round 7 Fix #2 deleted the `@mlx-node/lm/internal`
      // subpath export entirely and relocated the helper into the
      // server package so the only reachable call site is
      // `endpoints/responses.ts`, which already holds the
      // `SessionRegistry` HIT gate. Downstream consumers doing a plain
      // `import { ... } from '@mlx-node/lm'` or `from '@mlx-node/server'`
      // must therefore not be able to discover or invoke the helper.
      const lmPublicModule = (await import('../../packages/lm/src/index.js')) as Record<string, unknown>;
      expect('resetPreservingNativeCacheForWarmReuse' in lmPublicModule).toBe(false);

      const serverPublicModule = (await import('../../packages/server/src/index.js')) as Record<string, unknown>;
      expect('resetPreservingNativeCacheForWarmReuse' in serverPublicModule).toBe(false);

      // Sanity-check: the helper is still reachable from the
      // server-private module the endpoint code imports. Tests can
      // reach it via the relative path; downstream consumers cannot.
      const serverPrivateModule = (await import('../../packages/server/src/chat-session-warm-reuse.js')) as Record<
        string,
        unknown
      >;
      expect(typeof serverPrivateModule.resetPreservingNativeCacheForWarmReuse).toBe('function');
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
    it('routes an outstanding single-call turn through chatSessionContinueTool', async () => {
      const { model, chatSessionContinueTool, chatSessionStart } = makeMockModel();
      // The chat-session API requires that a tool-result dispatch be
      // preceded by an assistant turn that emitted exactly one `ok`
      // tool call — turn 0 is not a valid entry point for
      // sendToolResult because the native backends would synthesize a
      // <tool_response> delta for a call that never existed. Seed the
      // single outstanding call via `chatSessionStart` before the
      // tool-result dispatch.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'call-42'));
      // Second turn (the tool-result resolution) keeps the tool mock
      // result at its default 'tool-reply'.
      const session = new ChatSession(model);

      await session.send('fire the tool call');
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      const result = await session.sendToolResult('call-42', '{"status":"ok"}');
      expect(result.text).toBe('tool-reply');

      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      expect(chatSessionContinueTool.mock.calls[0][0]).toBe('call-42');
      expect(chatSessionContinueTool.mock.calls[0][1]).toBe('{"status":"ok"}');
      // The third native arg is the merged `config` (config precedes
      // `isError` at the NAPI boundary so the trailing-optional shape
      // accepts pre-feature 3-arg callers without InvalidArg).
      expect(chatSessionContinueTool.mock.calls[0][2]?.reuseCache).toBe(true);
      // The fourth arg is the structured `isError` signal — defaults to
      // `null` when the caller omits it (we coerce `undefined` to `null`
      // at the wrapper boundary so the NAPI binding sees a stable
      // shape).
      expect(chatSessionContinueTool.mock.calls[0][3]).toBeNull();
      expect(session.turns).toBe(2);
      // Tool-reply has no further outstanding calls.
      expect(session.pendingUnresolvedToolCallCount).toBeNull();
    });

    it('forwards isError=true through to the native binding', async () => {
      // Pin the structured tool-error plumbing: when the caller passes
      // `isError: true`, the wrapper MUST forward it as the fourth
      // positional argument to the native `chatSessionContinueTool`
      // call so the renderer can inject the wire-format
      // `[tool error]` marker. The history entry MUST also carry the
      // structured flag so cold-replay (image-change restart,
      // `startFromHistory*`, server-side `SessionRegistry`
      // cache-miss rebuild) re-renders the marker consistently. The
      // assertion is at the native binding boundary so the test
      // covers BOTH the JS-side argument plumbing and the
      // history-append shape without needing a loaded model.
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'call-err'));
      const session = new ChatSession(model);

      await session.send('fire the tool call');
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      await session.sendToolResult('call-err', '{"error":"boom"}', { isError: true });
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      expect(chatSessionContinueTool.mock.calls[0][0]).toBe('call-err');
      expect(chatSessionContinueTool.mock.calls[0][1]).toBe('{"error":"boom"}');
      // The fourth native arg is the structured `isError` signal — `true`
      // must pass through verbatim (not coerced to `null` like the omit
      // case). `config` precedes it at index [2].
      expect(chatSessionContinueTool.mock.calls[0][3]).toBe(true);
    });

    it('forwards isError=true through to the streaming native binding', async () => {
      // Streaming counterpart to the non-streaming `isError=true`
      // assertion above. Mirrors the wire-format invariant: the fifth
      // positional arg of `chatStreamSessionContinueTool` must carry
      // the structured flag so the streaming renderer injects the
      // same `[tool error]` marker as the non-streaming path. `config`
      // sits at index [2] and `signal` at [3]; `isError` is the
      // trailing-optional fifth arg.
      const { model, chatSessionStart, chatStreamSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'call-err'));
      const session = new ChatSession(model);

      await session.send('fire the tool call');
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      const events: ChatStreamEvent[] = [];
      for await (const ev of session.sendToolResultStream('call-err', '{"error":"boom"}', { isError: true })) {
        events.push(ev);
      }
      expect(chatStreamSessionContinueTool).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinueTool.mock.calls[0][0]).toBe('call-err');
      expect(chatStreamSessionContinueTool.mock.calls[0][1]).toBe('{"error":"boom"}');
      // `signal` (index [3]) precedes `isError` (index [4]) at the
      // wrapper boundary — mirrors the native NAPI ordering where
      // the trailing-optional `isError` follows the required callback
      // so pre-feature 4-arg callers still type-check.
      expect(chatStreamSessionContinueTool.mock.calls[0][4]).toBe(true);
      // The stream still terminates normally on the mock — confirms the
      // history-append commit branch ran (sawFinal === true).
      expect(events.at(-1)?.done).toBe(true);
    });

    it('defaults isError to null when the caller omits it', async () => {
      // Default semantics: `isError` is optional and the wrapper
      // coerces `undefined` to `null` at the NAPI boundary so the
      // binding sees a stable shape. This pins the "no marker"
      // pre-feature behavior against accidental flag-inference drift.
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'call-ok'));
      const session = new ChatSession(model);

      await session.send('fire the tool call');
      await session.sendToolResult('call-ok', '{"status":"ok"}');
      // `isError` omitted → coerced to `null` at index [3], not `undefined`.
      expect(chatSessionContinueTool.mock.calls[0][3]).toBeNull();
    });

    it('forwards isError=false unchanged (still no marker, but explicit)', async () => {
      // `Some(false)` is the same wire output as `None` per the
      // renderer's contract — but the wrapper must NOT silently fold
      // `false` into `null` because the structured channel is the
      // authoritative source-of-truth and callers may legitimately
      // pass an explicit `false` to disambiguate from "not set".
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'call-ok'));
      const session = new ChatSession(model);

      await session.send('fire the tool call');
      await session.sendToolResult('call-ok', '{"status":"ok"}', { isError: false });
      expect(chatSessionContinueTool.mock.calls[0][3]).toBe(false);
    });

    it('accepts the opts-bag without isError (backward compat with { config } only)', async () => {
      // Regression: the additive opts-bag signature must accept callers
      // that pass only `{ config }` without `isError`. The pre-fix
      // signature interleaved `isError` as a positional third argument
      // so any caller using `sendToolResult(id, content, { config })`
      // landed `{ config }` in the `isError` slot and crashed at the
      // native NAPI boundary (Option<bool>). This pins the opts-bag
      // contract: `{ config }` alone is a valid call shape and
      // `isError` defaults to `null` at the native boundary.
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      const session = new ChatSession(model);

      await session.send('fire');
      await session.sendToolResult('c1', 'out', { config: { reuseCache: false } });
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      // The merged config sits at index [2] and still forces reuseCache: true.
      expect(chatSessionContinueTool.mock.calls[0][2]?.reuseCache).toBe(true);
      // `isError` was omitted → coerced to `null` at the NAPI boundary,
      // landing in the trailing-optional [3] slot.
      expect(chatSessionContinueTool.mock.calls[0][3]).toBeNull();
    });

    it('rejects sendToolResult when no outstanding tool call exists', async () => {
      // Regression for the zero-outstanding gate. A fresh session has
      // never seen a tool call, so tool-result entry points must
      // throw — the native backends do not authenticate tool_call_id
      // against prior state, so the only defense against synthesizing
      // a tool response for a non-existent call is to refuse the
      // dispatch at the wrapper level.
      const { model, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      await expect(session.sendToolResult('call-phantom', '{}')).rejects.toThrow(
        /ChatSession\.sendToolResult: no outstanding ok tool call/,
      );
      expect(chatSessionContinueTool).not.toHaveBeenCalled();
    });

    it('rejects sendToolResult after a plain assistant turn with no tool calls', async () => {
      // A normal send that returns a pure-text assistant reply leaves
      // the outstanding-call flag null. A subsequent sendToolResult
      // must throw, since the model never emitted a call to resolve.
      const { model, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('say hi'); // returns 'start-reply' with 0 tool calls
      expect(session.pendingUnresolvedToolCallCount).toBeNull();

      await expect(session.sendToolResult('call-phantom', '{}')).rejects.toThrow(
        /ChatSession\.sendToolResult: no outstanding ok tool call/,
      );
      expect(chatSessionContinueTool).not.toHaveBeenCalled();
    });

    it('rejects sendToolResultStream when no outstanding tool call exists', async () => {
      const { model, chatStreamSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      await expect(async () => {
        for await (const _e of session.sendToolResultStream('call-phantom', '{}')) void _e;
      }).rejects.toThrow(/ChatSession\.sendToolResultStream: no outstanding ok tool call/);
      expect(chatStreamSessionContinueTool).not.toHaveBeenCalled();
    });

    it('forces reuseCache=true even with caller override', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      const session = new ChatSession(model);

      await session.send('fire'); // establishes outstanding call 'c1'
      await session.sendToolResult('c1', 'out', { config: { reuseCache: false } });
      // The third positional native arg is the merged `ChatConfig` —
      // `mergeConfig` always forces `reuseCache: true` regardless of the
      // caller-supplied value (the session path is a cache-reuse op by
      // construction).
      expect(chatSessionContinueTool.mock.calls[0][2]?.reuseCache).toBe(true);
    });

    it('clears inFlight on exception', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      // Seed two successive outstanding single-call turns so both
      // sendToolResult dispatches have a legal pre-state. The first
      // chatSessionContinueTool call rejects (tool-boom), which must
      // NOT consume the outstanding-call flag; the second dispatch
      // proceeds against the same outstanding state.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      chatSessionContinueTool.mockRejectedValueOnce(new Error('tool-boom'));
      chatSessionContinueTool.mockResolvedValueOnce(makeChatResult('recovered'));
      const session = new ChatSession(model);

      await session.send('fire');
      await expect(session.sendToolResult('c1', 'out')).rejects.toThrow('tool-boom');
      // Follow-up works against the same outstanding call — the
      // failed dispatch rolled back the flag, so the second call
      // still sees `unresolvedOkToolCallCount === 1`.
      expect(session.pendingUnresolvedToolCallCount).toBe(1);
      await session.sendToolResult('c1', 'out2');
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(2);
    });

    // ---------------------------------------------------------------
    // Single-tool-call-per-turn enforcement
    // ---------------------------------------------------------------

    it('rejects sendToolResult after an assistant turn with multiple ok tool calls', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      // Turn 0: simulate the model emitting TWO ok tool calls in the
      // same assistant turn — the chat-session API cannot serve this
      // pattern because each sendToolResult dispatch immediately
      // re-opens the assistant turn.
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('multi'),
        toolCalls: [
          { id: 'call-a', name: 'fa', arguments: {}, status: 'ok' },
          { id: 'call-b', name: 'fb', arguments: {}, status: 'ok' },
        ],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('fan out');
      expect(session.turns).toBe(1);

      await expect(session.sendToolResult('call-a', 'result-a')).rejects.toThrow(
        /previous assistant turn emitted 2 ok tool calls/,
      );
      expect(chatSessionContinueTool).not.toHaveBeenCalled();
    });

    it('allows sendToolResult after an assistant turn with exactly one ok tool call', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('one-call'),
        toolCalls: [{ id: 'call-a', name: 'fa', arguments: {}, status: 'ok' }],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('make one call');
      // A single outstanding ok tool call leaves the guard at 1, but
      // `sendToolResult` is the exact entry point it's meant to be
      // resolved through, so dispatch is allowed.
      expect(session.pendingUnresolvedToolCallCount).toBe(1);
      await session.sendToolResult('call-a', 'result-a');
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      // The tool-result reply emits zero ok tool calls (default mock),
      // so the flag clears and a plain text continuation is unblocked
      // again.
      expect(session.pendingUnresolvedToolCallCount).toBeNull();
      await session.send('follow up');
    });

    it('blocks send()/sendStream() after a single unresolved ok tool call', async () => {
      // The single-call case is the subtle one the earlier multi-call
      // guard missed: a plain user delta after a single outstanding
      // tool call would weave a new user turn between the assistant's
      // tool_call and any response, orphaning the call. Every plain
      // text entry point must reject until the call is resolved or the
      // session is reset.
      const { model, chatSessionStart, chatSessionContinue, chatStreamSessionContinue, chatSessionContinueTool } =
        makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('one-call'),
        toolCalls: [{ id: 'call-a', name: 'fa', arguments: {}, status: 'ok' }],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('make one call');
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      await expect(session.send('orphan the call')).rejects.toThrow(
        /ChatSession\.send: previous assistant turn has 1 unresolved ok tool call;/,
      );
      expect(chatSessionContinue).not.toHaveBeenCalled();

      await expect(async () => {
        for await (const _e of session.sendStream('orphan via stream')) void _e;
      }).rejects.toThrow(/ChatSession\.sendStream: previous assistant turn has 1 unresolved ok tool call;/);
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();

      // Resolving via sendToolResult is the correct recovery and
      // clears the flag so the subsequent plain `send` proceeds.
      await session.sendToolResult('call-a', 'result-a');
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      await session.send('follow up');
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
    });

    it('ignores parse_error / invalid_json tool calls for fan-out detection', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      // One ok + several non-ok — only the ok call counts for the
      // fan-out guard, so the session stays serviceable.
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('mixed'),
        toolCalls: [
          { id: 'call-a', name: 'fa', arguments: {}, status: 'ok' },
          { id: 'call-b', name: 'fb', arguments: 'garbage', status: 'parse_error' },
          { id: 'call-c', name: 'fc', arguments: '', status: 'invalid_json' },
        ],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('mixed');
      await session.sendToolResult('call-a', 'result-a');
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
    });

    it('blocks send() and sendStream() after a multi-call turn until the session is reset', async () => {
      // A pending multi-call fan-out must hard-stop every continuation
      // entry point, not just sendToolResult*. A plain user `send` (or
      // `sendStream`) would silently overwrite the fan-out flag with the
      // new assistant reply and effectively orphan the sibling tool
      // calls — the caller must either reset() the session or re-enter
      // through primeHistory + startFromHistory with a resolved history.
      const { model, chatSessionStart, chatSessionContinue, chatStreamSessionContinue, chatSessionContinueTool } =
        makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('multi'),
        toolCalls: [
          { id: 'call-a', name: 'fa', arguments: {}, status: 'ok' },
          { id: 'call-b', name: 'fb', arguments: {}, status: 'ok' },
        ],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('fan out');

      await expect(session.send('orphan the siblings')).rejects.toThrow(
        /ChatSession\.send: previous assistant turn has 2 unresolved ok tool calls/,
      );
      expect(chatSessionContinue).not.toHaveBeenCalled();

      await expect(async () => {
        for await (const _e of session.sendStream('orphan via stream')) void _e;
      }).rejects.toThrow(/ChatSession\.sendStream: previous assistant turn has 2 unresolved ok tool calls/);
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();

      // reset() is one valid recovery path; the next plain `send`
      // must succeed on the cleared session and route through the
      // start path (since turnCount is back to 0).
      await session.reset();
      await session.send('fresh topic');
      // chatSessionStart called twice: once for the initial fan-out
      // turn, once for the post-reset fresh start.
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      expect(chatSessionContinueTool).not.toHaveBeenCalled();
    });

    it('reset() clears the pending tool-call flag', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('multi'),
        toolCalls: [
          { id: 'call-a', name: 'fa', arguments: {}, status: 'ok' },
          { id: 'call-b', name: 'fb', arguments: {}, status: 'ok' },
        ],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('fan out');
      await expect(session.sendToolResult('call-a', 'result-a')).rejects.toThrow(/previous assistant turn emitted/);
      expect(session.pendingUnresolvedToolCallCount).toBe(2);

      await session.reset();
      // After reset, the pending flag is null — a plain send is the
      // valid next entry point and routes through the start path
      // since turnCount is back to 0.
      expect(session.pendingUnresolvedToolCallCount).toBeNull();
      await session.send('fresh topic');
      expect(chatSessionStart).toHaveBeenCalledTimes(2);
      expect(chatSessionContinue).not.toHaveBeenCalled();
    });

    it('rejects sendToolResultStream after a multi-call turn', async () => {
      const { model, chatSessionStart, chatStreamSessionContinueTool } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('multi'),
        toolCalls: [
          { id: 'call-a', name: 'fa', arguments: {}, status: 'ok' },
          { id: 'call-b', name: 'fb', arguments: {}, status: 'ok' },
        ],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('fan out');
      // Invoking the generator must throw BEFORE yielding anything —
      // the caller cannot observe any stream events on a rejected
      // dispatch.
      await expect(async () => {
        for await (const _e of session.sendToolResultStream('call-a', 'result-a')) void _e;
      }).rejects.toThrow(/previous assistant turn emitted 2 ok tool calls/);
      expect(chatStreamSessionContinueTool).not.toHaveBeenCalled();
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

    // -----------------------------------------------------------------
    // gemma4 empty-done-chunk history commit
    //
    // gemma4 is the only family whose native streaming `finish` emits a
    // terminal chunk with `text: ''` (the visible text was already
    // streamed as per-token deltas). The history commit must therefore
    // fall back to the accumulated VISIBLE deltas instead of storing an
    // empty assistant turn — otherwise any cold-restart that resends
    // `this.history` (e.g. the media→text restart) loses the reply.
    //
    // History is private, so these assert through the image-change
    // restart, which replays the full committed history into
    // `chatStreamSessionStart`.
    // -----------------------------------------------------------------

    const gemmaImgA = new Uint8Array([91, 92, 93]);
    const gemmaImgB = new Uint8Array([94, 95, 96]);

    /**
     * Read back the committed assistant content of the FIRST streamed
     * turn by triggering an image-change restart and inspecting the
     * replayed `chatStreamSessionStart` history.
     */
    async function committedAssistantViaRestart(firstTurnDeltas: ChatStreamEvent[]): Promise<string | undefined> {
      let startCall = 0;
      const chatStreamSessionStart = vi.fn(async function* (
        _messages: ChatMessage[],
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        startCall++;
        if (startCall === 1) {
          for (const ev of firstTurnDeltas) yield ev;
          return;
        }
        // Restart stream — content does not matter for the assertion.
        yield finalChunk('restart-reply');
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

      // Turn 1: gemma4-shape stream with an image attached so the next
      // (different-image) send is detected as an image change.
      for await (const _e of session.sendStream('turn 1', { images: [gemmaImgA] })) void _e;
      expect(session.turns).toBe(1);

      // Turn 2: image-change restart replays the committed history.
      for await (const _e of session.sendStream('turn 2', { images: [gemmaImgB] })) void _e;

      const restartMessages = chatStreamSessionStart.mock.calls[1][0];
      const assistant = restartMessages.find((m) => m.role === 'assistant');
      return assistant?.content;
    }

    it('commits the accumulated visible text when the done chunk is empty (gemma4)', async () => {
      // Visible deltas streamed, then a gemma4-shape empty final chunk.
      const content = await committedAssistantViaRestart([
        { text: 'Pa', done: false, isReasoning: false },
        { text: 'ris', done: false, isReasoning: false },
        finalChunk(''),
      ]);
      expect(content).toBe('Paris');
    });

    it('excludes reasoning deltas from the empty-done-chunk commit (gemma4)', async () => {
      // Reasoning-body delta must NOT leak into stored history; only the
      // visible delta survives when the final chunk is empty.
      const content = await committedAssistantViaRestart([
        { text: 'thinking…', done: false, isReasoning: true },
        { text: 'Answer', done: false, isReasoning: false },
        finalChunk(''),
      ]);
      expect(content).toBe('Answer');
    });

    it('commits the non-empty done-chunk text verbatim (non-gemma regression)', async () => {
      // For families whose final chunk carries non-empty `text`, the
      // commit must use that text unchanged — `||` ≡ `??` here. The
      // accumulated visible deltas must NOT override it.
      const content = await committedAssistantViaRestart([
        { text: 'partial-stream-text', done: false, isReasoning: false },
        finalChunk('cont-reply'),
      ]);
      expect(content).toBe('cont-reply');
    });
  });

  // -------------------------------------------------------------------
  // Media-held delta rejection → transparent cold replay
  //
  // After an image/audio turn, gemma4's native session refuses a
  // text-only delta with a typed `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`
  // rejection. The session layer must catch that exact prefix and replay
  // the full conversation through the cold start path instead of letting
  // the raw error escape. The native streaming wrapper converts the
  // worker-thread sink error into a THROWN error on the generator's first
  // iteration (verified against `packages/lm/src/stream.ts` `if (item.error)
  // throw item.error`), so both `send()` and `sendStream()` see a thrown
  // rejection before any chunk is emitted.
  // -------------------------------------------------------------------

  describe('media-held continuation → cold replay', () => {
    const RESTART_PREFIX = 'IMAGE_CHANGE_REQUIRES_SESSION_RESTART:';
    const imgA = new Uint8Array([7, 7, 7]);

    function mediaHeldError(): Error {
      return new Error(`${RESTART_PREFIX}chat_tokens_delta_sync is text-only; session currently holds image state`);
    }

    it('send(): media-held delta rejection replays through chatSessionStart', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: image start succeeds.
      await session.send('describe', { images: [imgA] });
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);

      // Turn 2: text-only follow-up routes through the delta path, but the
      // native session (media held) rejects with the typed prefix. The
      // session must catch it and replay through chatSessionStart.
      chatSessionContinue.mockRejectedValueOnce(mediaHeldError());
      const result = await session.send('what about the top-right?');

      // Resolves with the replayed reply — no throw.
      expect(result.text).toBe('start-reply');
      // The delta path was attempted exactly once before the replay.
      expect(chatSessionContinue).toHaveBeenCalledTimes(1);
      // The replay ran through the start path (turn-1 start + turn-2 replay).
      expect(chatSessionStart).toHaveBeenCalledTimes(2);

      // The replay re-rendered the FULL history, including the prior image
      // turn, plus the new text-only user turn (no media attached).
      const replayMessages = chatSessionStart.mock.calls[1][0];
      expect(replayMessages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'start-reply' },
        { role: 'user', content: 'what about the top-right?' },
      ]);

      // History appended exactly once for this turn (user + assistant), and
      // turnCount advanced by exactly one.
      expect(session.turns).toBe(2);
      // The trailing image key was rehydrated from history, so the cache
      // still reflects the held image.
      expect(session.hasImages).toBe(true);

      // inFlight cleared — a follow-up send is accepted.
      const followUp = await session.send('and the bottom-left?');
      expect(followUp.text).toBe('continue-reply');
      expect(session.turns).toBe(3);
    });

    it('send(): a non-prefix delta rejection propagates without replay', async () => {
      const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
      const session = new ChatSession(model);

      await session.send('describe', { images: [imgA] });
      expect(chatSessionStart).toHaveBeenCalledTimes(1);

      chatSessionContinue.mockRejectedValueOnce(new Error('some other native failure'));
      await expect(session.send('text follow-up')).rejects.toThrow('some other native failure');

      // No replay — chatSessionStart not called a second time.
      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      // The failed turn did not advance the counter.
      expect(session.turns).toBe(1);

      // inFlight cleared — a follow-up still routes through the delta path.
      await session.send('recover');
      expect(session.turns).toBe(2);
    });

    it('sendStream(): media-held delta rejection replays through chatStreamSessionStart', async () => {
      let continueCalls = 0;
      const chatStreamSessionContinue = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        continueCalls++;
        // First continue (the media-held delta): throw before any chunk,
        // exactly how stream.ts surfaces the native sink error.
        throw mediaHeldError();
        // eslint-disable-next-line no-unreachable
        yield finalChunk('unreachable');
      });
      let startCalls = 0;
      const startHistories: ChatMessage[][] = [];
      const chatStreamSessionStart = vi.fn(async function* (messages: ChatMessage[]): AsyncGenerator<ChatStreamEvent> {
        startCalls++;
        startHistories.push(messages);
        if (startCalls === 1) {
          // Turn-1 image start.
          yield { text: 'A', done: false };
          yield finalChunk('describe-reply');
          return;
        }
        // Turn-2 replay.
        yield { text: 'replayed', done: false };
        yield finalChunk('replayed-reply');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue,
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: streamed image start.
      for await (const _e of session.sendStream('describe', { images: [imgA] })) void _e;
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);

      // Turn 2: text follow-up, media-held rejection → transparent replay.
      const events: ChatStreamEvent[] = [];
      for await (const e of session.sendStream('what about the top-right?')) events.push(e);

      // The replayed reply was yielded; no duplicate / partial emission of
      // the failed delta (the failed continue produced no chunk).
      expect(events.map((e) => e.text)).toEqual(['replayed', 'replayed-reply']);
      expect(events[events.length - 1].done).toBe(true);
      expect((events[events.length - 1] as ChatStreamFinal).finishReason).toBe('stop');

      // Delta path attempted once; replay ran through the start stream.
      expect(continueCalls).toBe(1);
      expect(startCalls).toBe(2);

      // The replay re-rendered the FULL history incl. the prior image turn.
      expect(startHistories[1]).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'describe-reply' },
        { role: 'user', content: 'what about the top-right?' },
      ]);

      // History pushed exactly once; turnCount +1.
      expect(session.turns).toBe(2);
      expect(session.hasImages).toBe(true);

      // inFlight cleared — a follow-up streams cleanly.
      for await (const _e of session.sendStream('and the bottom-left?')) void _e;
      expect(session.turns).toBe(3);
    });

    it('sendStream(): caller break mid-replay leaves consistent state', async () => {
      const chatStreamSessionContinue = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        throw mediaHeldError();
        // eslint-disable-next-line no-unreachable
        yield finalChunk('unreachable');
      });
      // Both turn-1 (the image start stream) and the turn-2 replay run
      // through the streamed start. The first call (turn 1) completes
      // normally; the second (the replay) emits one delta and never reaches
      // done so the caller can break mid-stream, exercising the
      // runStartStreamPath rollback finally.
      let startCalls = 0;
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        startCalls++;
        if (startCalls === 1) {
          yield finalChunk('describe-reply');
          return;
        }
        // Replay stream: emit one delta, never reach done — caller breaks.
        yield { text: 'partial-replay', done: false };
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('recovery-reply')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue,
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      // Turn 1: streamed image start succeeds.
      for await (const _e of session.sendStream('describe', { images: [imgA] })) void _e;
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);

      // Turn 2: media-held rejection → replay stream, caller breaks after
      // one partial delta. The replay is dispatched as a media-change
      // restart (caches wiped up front), so the runStartStreamPath rollback
      // resets turnCount → 0 and clears the image key, forcing the next
      // call to re-route through the start path with the preserved history.
      let seen = 0;
      for await (const _e of session.sendStream('text follow-up')) {
        seen++;
        break;
      }
      expect(seen).toBe(1);
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);

      // inFlight cleared — a recovery send routes through the start path
      // with the preserved history (turn-1 conversation + the new turn).
      const recovery = await session.send('recover', { images: [imgA] });
      expect(recovery.text).toBe('recovery-reply');
      const recoveryMessages = (model.chatSessionStart as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(recoveryMessages).toEqual([
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'describe-reply' },
        { role: 'user', content: 'recover', images: [imgA] },
      ]);
      expect(session.turns).toBe(1);
    });

    it('sendStream(): a non-prefix mid-stream throw still propagates (no replay)', async () => {
      const chatStreamSessionContinue = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        yield { text: 'partial', done: false };
        throw new Error('mid-stream native failure');
        // eslint-disable-next-line no-unreachable
        yield finalChunk('unreachable');
      });
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        yield finalChunk('turn-1-reply');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue,
        chatStreamSessionContinueTool: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      // Turn 1 via streamed start.
      for await (const _e of session.sendStream('turn 1')) void _e;
      expect(session.turns).toBe(1);

      // Turn 2: a non-prefix throw AFTER a token was emitted must NOT be
      // swallowed into a replay — it propagates, and no replay start runs.
      await expect(async () => {
        for await (const _e of session.sendStream('turn 2')) void _e;
      }).rejects.toThrow('mid-stream native failure');

      // Only the turn-1 start stream ran; no replay was triggered.
      expect(chatStreamSessionStart).toHaveBeenCalledTimes(1);
      expect(session.turns).toBe(1);
    });

    it('sendToolResult(): media-held rejection replays through chatSessionStart', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: image start whose assistant reply carries exactly one ok
      // tool call — the legal pre-state for a tool-result dispatch.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      await session.send('describe', { images: [imgA] });
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // Tool result: the native session (media held) rejects the
      // tool-result delta with the typed prefix. The session must catch
      // it and replay through chatSessionStart.
      chatSessionContinueTool.mockRejectedValueOnce(mediaHeldError());
      const result = await session.sendToolResult('c1', 'tool-out');

      // Resolves with the replayed reply — no throw.
      expect(result.text).toBe('start-reply');
      // The delta path was attempted exactly once before the replay.
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      // The replay ran through the start path (turn-1 start + replay).
      expect(chatSessionStart).toHaveBeenCalledTimes(2);

      // The replay re-rendered the FULL history: the prior image turn, the
      // assistant tool-call turn, and the pending tool message (isError
      // omitted → undefined).
      const replayMessages = chatSessionStart.mock.calls[1][0];
      expect(replayMessages).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'first-call', toolCalls: [{ id: 'c1', name: 'tool_fn', arguments: '{}' }] },
        { role: 'tool', content: 'tool-out', toolCallId: 'c1', isError: undefined },
      ]);

      // History advanced by exactly one turn.
      expect(session.turns).toBe(2);
      // Trailing image key rehydrated from history — cache still reflects
      // the held image.
      expect(session.hasImages).toBe(true);

      // inFlight cleared — a follow-up send is accepted (the replay reply
      // had no outstanding tool calls, so the plain path is legal).
      const followUp = await session.send('and now?');
      expect(followUp.text).toBe('continue-reply');
      expect(session.turns).toBe(3);
    });

    it('sendToolResult(): preserves isError through the cold replay', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model, { system: 'You are helpful.' });

      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      await session.send('describe', { images: [imgA] });

      chatSessionContinueTool.mockRejectedValueOnce(mediaHeldError());
      await session.sendToolResult('c1', '{"error":"boom"}', { isError: true });

      // The pending tool message carried through the replay keeps the
      // structured error flag so the wire-format marker re-renders.
      const replayMessages = chatSessionStart.mock.calls[1][0];
      expect(replayMessages[replayMessages.length - 1]).toEqual({
        role: 'tool',
        content: '{"error":"boom"}',
        toolCallId: 'c1',
        isError: true,
      });
    });

    it('sendToolResult(): a non-prefix tool-result rejection propagates without replay', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      await session.send('describe', { images: [imgA] });
      expect(chatSessionStart).toHaveBeenCalledTimes(1);

      chatSessionContinueTool.mockRejectedValueOnce(new Error('some other native failure'));
      await expect(session.sendToolResult('c1', 'tool-out')).rejects.toThrow('some other native failure');

      // No replay — chatSessionStart not called a second time.
      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      // The failed turn did not advance the counter and left the
      // outstanding-call flag intact for a retry.
      expect(session.turns).toBe(1);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // inFlight cleared — a retry against the same outstanding call works.
      await session.sendToolResult('c1', 'retry');
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(2);
    });

    it('sendToolResultStream(): media-held rejection replays through chatStreamSessionStart', async () => {
      let toolCalls = 0;
      const chatStreamSessionContinueTool = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        toolCalls++;
        // The media-held tool delta: throw before any chunk, exactly how
        // stream.ts surfaces the native sink error on the first iteration.
        throw mediaHeldError();
        // eslint-disable-next-line no-unreachable
        yield finalChunk('unreachable');
      });
      let startCalls = 0;
      const startHistories: ChatMessage[][] = [];
      const chatStreamSessionStart = vi.fn(async function* (messages: ChatMessage[]): AsyncGenerator<ChatStreamEvent> {
        startCalls++;
        startHistories.push(messages);
        if (startCalls === 1) {
          // Turn-1 image start whose reply carries one ok tool call.
          yield { text: 'A', done: false };
          yield finalChunkWithSingleToolCall('describe-reply', 'c1');
          return;
        }
        // Tool-result replay.
        yield { text: 'replayed', done: false };
        yield finalChunk('replayed-reply');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool,
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: streamed image start emits a single ok tool call.
      for await (const _e of session.sendStream('describe', { images: [imgA] })) void _e;
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // Tool result: media-held rejection → transparent replay.
      const events: ChatStreamEvent[] = [];
      for await (const e of session.sendToolResultStream('c1', 'tool-out')) events.push(e);

      // The replayed reply was yielded; no duplicate / partial emission of
      // the failed delta (the failed tool continue produced no chunk).
      expect(events.map((e) => e.text)).toEqual(['replayed', 'replayed-reply']);
      expect(events[events.length - 1].done).toBe(true);
      expect((events[events.length - 1] as ChatStreamFinal).finishReason).toBe('stop');

      // Tool delta attempted once; replay ran through the start stream.
      expect(toolCalls).toBe(1);
      expect(startCalls).toBe(2);

      // The replay re-rendered the FULL history incl. the prior image turn,
      // the assistant tool-call turn, and the pending tool message.
      expect(startHistories[1]).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'describe-reply', toolCalls: [{ id: 'c1', name: 'tool_fn', arguments: '{}' }] },
        { role: 'tool', content: 'tool-out', toolCallId: 'c1', isError: undefined },
      ]);

      // History committed exactly once; turnCount +1.
      expect(session.turns).toBe(2);
      expect(session.hasImages).toBe(true);

      // inFlight cleared — a follow-up streams cleanly.
      for await (const _e of session.sendStream('and now?')) void _e;
      expect(session.turns).toBe(3);
    });

    it('sendToolResultStream(): a non-prefix tool throw still propagates (no replay)', async () => {
      const chatStreamSessionContinueTool = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        throw new Error('tool native failure');
        // eslint-disable-next-line no-unreachable
        yield finalChunk('unreachable');
      });
      let startCalls = 0;
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        startCalls++;
        yield finalChunkWithSingleToolCall('turn-1-reply', 'c1');
      });
      const model: SessionCapableModel = {
        chatSessionStart: vi.fn(async () => makeChatResult('x')),
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool,
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model);

      // Turn 1 via streamed start emits one ok tool call.
      for await (const _e of session.sendStream('turn 1')) void _e;
      expect(session.turns).toBe(1);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // Tool result: a non-prefix throw must NOT be swallowed into a
      // replay — it propagates, and no replay start runs.
      await expect(async () => {
        for await (const _e of session.sendToolResultStream('c1', 'tool-out')) void _e;
      }).rejects.toThrow('tool native failure');

      // Only the turn-1 start stream ran; no replay was triggered.
      expect(startCalls).toBe(1);
      expect(session.turns).toBe(1);
    });

    it('sendToolResultStream(): an interrupted media-held replay leaves a cold session that the next sendToolResult self-heals through the start path', async () => {
      const chatStreamSessionContinueTool = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        // The media-held tool delta: throw before any chunk, as
        // stream.ts surfaces the native sink error on the first
        // iteration.
        throw mediaHeldError();
        // eslint-disable-next-line no-unreachable
        yield finalChunk('unreachable');
      });
      // Streamed start drives both the turn-1 image start (one ok tool
      // call) and the interrupted turn-2 replay (one delta, never
      // reaches done → caller breaks → rollback to turnCount 0).
      let startStreamCalls = 0;
      const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
        startStreamCalls++;
        if (startStreamCalls === 1) {
          yield { text: 'A', done: false };
          yield finalChunkWithSingleToolCall('describe-reply', 'c1');
          return;
        }
        // The replay stream: emit one partial delta, never reach done.
        yield { text: 'partial-replay', done: false };
      });
      const chatSessionStart = vi.fn(
        async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> =>
          makeChatResult('recovered-reply'),
      );
      const model: SessionCapableModel = {
        chatSessionStart,
        chatSessionContinue: vi.fn(async () => makeChatResult('x')),
        chatSessionContinueTool: vi.fn(async () => makeChatResult('x')),
        chatStreamSessionStart,
        chatStreamSessionContinue: vi.fn(async function* () {
          yield finalChunk('x');
        }),
        chatStreamSessionContinueTool,
        resetCaches: vi.fn(),
      };
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: streamed image start emits a single ok tool call.
      for await (const _e of session.sendStream('describe', { images: [imgA] })) void _e;
      expect(session.turns).toBe(1);
      expect(session.hasImages).toBe(true);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // Tool result: media-held rejection → replay stream, but the
      // caller BREAKS after the first partial delta, before done:true.
      // The replay was a media-change restart (caches wiped up front),
      // so the runStartStreamPath rollback resets turnCount → 0 and
      // clears the media keys, leaving the session cold while the
      // unresolved tool-call flag is still set.
      let seen = 0;
      for await (const _e of session.sendToolResultStream('c1', 'tool-out')) {
        seen++;
        break;
      }
      expect(seen).toBe(1);
      expect(session.turns).toBe(0);
      expect(session.hasImages).toBe(false);
      // The tool-call flag survives the rollback — the bug is that the
      // delta path would now hard-error against the wiped cache.
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // The natural retry must self-heal: route through the cold start
      // path (chatSessionStart), NOT a second tool delta, and resolve
      // with the replayed reply.
      const retry = await session.sendToolResult('c1', 'tool-out');
      expect(retry.text).toBe('recovered-reply');
      // The tool delta was attempted exactly once (the interrupted
      // replay); the retry never re-dispatched it.
      expect(chatStreamSessionContinueTool).toHaveBeenCalledTimes(1);
      // The retry resolved through the start path.
      expect(chatSessionStart).toHaveBeenCalledTimes(1);
      expect(session.turns).toBe(1);

      // The replay history includes the prior media turn, the assistant
      // tool-call turn, and the pending tool message.
      expect(chatSessionStart.mock.calls[0][0]).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'describe-reply', toolCalls: [{ id: 'c1', name: 'tool_fn', arguments: '{}' }] },
        { role: 'tool', content: 'tool-out', toolCallId: 'c1', isError: undefined },
      ]);
    });

    it('sendToolResult(): a media-held replay that THROWS before success leaves a cold session that the next sendToolResult self-heals', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model, { system: 'You are helpful.' });

      // Turn 1: image start whose assistant reply carries one ok tool
      // call — the legal pre-state for a tool-result dispatch.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('first-call', 'c1'));
      await session.send('describe', { images: [imgA] });
      expect(session.turns).toBe(1);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // Tool result: the delta rejects with the media-held prefix, and
      // the replay's chatSessionStart ALSO rejects once — the cold
      // start prefill fails before success, so the runStartPath
      // rollback resets turnCount → 0 (media-change restart) and the
      // first call throws.
      chatSessionContinueTool.mockRejectedValueOnce(mediaHeldError());
      chatSessionStart.mockRejectedValueOnce(new Error('replay-prefill-fail'));
      await expect(session.sendToolResult('c1', 'tool-out')).rejects.toThrow('replay-prefill-fail');
      expect(session.turns).toBe(0);
      // The tool-call flag survives the rollback.
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      // The retry must self-heal through the cold start path now that
      // chatSessionStart succeeds — NOT a second doomed tool delta.
      const retry = await session.sendToolResult('c1', 'tool-out');
      expect(retry.text).toBe('start-reply');
      // The delta was attempted exactly once (the original media-held
      // try); the retry routed straight through the start path.
      expect(chatSessionContinueTool).toHaveBeenCalledTimes(1);
      // chatSessionStart: turn-1 start + the failed replay + the
      // successful retry replay = 3 total.
      expect(chatSessionStart).toHaveBeenCalledTimes(3);
      expect(session.turns).toBe(1);

      // The retry replay re-rendered the full preserved history plus the
      // pending tool message.
      expect(chatSessionStart.mock.calls[2][0]).toEqual([
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'describe', images: [imgA] },
        { role: 'assistant', content: 'first-call', toolCalls: [{ id: 'c1', name: 'tool_fn', arguments: '{}' }] },
        { role: 'tool', content: 'tool-out', toolCallId: 'c1', isError: undefined },
      ]);
    });
  });

  // -------------------------------------------------------------------
  // sendToolResultStream()
  // -------------------------------------------------------------------

  describe('sendToolResultStream()', () => {
    it('routes through chatStreamSessionContinueTool and advances on success', async () => {
      const { model, chatStreamSessionStart, chatStreamSessionContinueTool } = makeMockModel();
      // Seed the opening streamed turn with a single ok tool call so
      // the subsequent sendToolResultStream has a legal
      // outstanding-call state.
      chatStreamSessionStart.mockImplementationOnce(async function* () {
        yield { text: 'opening', done: false };
        yield finalChunkWithSingleToolCall('opening', 'c1');
      });
      const session = new ChatSession(model);

      for await (const _e of session.sendStream('fire')) void _e;
      expect(session.pendingUnresolvedToolCallCount).toBe(1);

      let sawDone = false;
      for await (const e of session.sendToolResultStream('c1', 'tool-out')) {
        if (e.done) sawDone = true;
      }
      expect(sawDone).toBe(true);
      expect(session.turns).toBe(2);
      expect(chatStreamSessionContinueTool).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinueTool.mock.calls[0][0]).toBe('c1');
      expect(chatStreamSessionContinueTool.mock.calls[0][1]).toBe('tool-out');
    });

    it('does NOT advance turnCount on caller break', async () => {
      const { model, chatStreamSessionStart } = makeMockModel();
      chatStreamSessionStart.mockImplementationOnce(async function* () {
        yield { text: 'opening', done: false };
        yield finalChunkWithSingleToolCall('opening', 'c1');
      });
      const session = new ChatSession(model);

      // Establish outstanding single-call state.
      for await (const _e of session.sendStream('fire')) void _e;
      expect(session.turns).toBe(1);

      for await (const e of session.sendToolResultStream('c1', 'out')) {
        expect(e.done).toBe(false);
        break;
      }
      // The abandoned tool-result stream must not advance turnCount
      // past the opening turn.
      expect(session.turns).toBe(1);
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

    // ---------------------------------------------------------------
    // Unresolved-tool-call guard hydration on cold replay
    // ---------------------------------------------------------------

    it('leaves pendingUnresolvedToolCallCount null for histories with no trailing fan-out', () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);
      expect(session.pendingUnresolvedToolCallCount).toBeNull();

      // A trailing assistant turn whose single tool call was already
      // resolved by a sibling `tool:` message should leave the guard
      // null — the session is ready for a plain user follow-up.
      session.primeHistory([
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'ok', toolCalls: [{ id: 'call-a', name: 'fa', arguments: '{}' }] },
        { role: 'tool', content: 'result-a', toolCallId: 'call-a' },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBeNull();
    });

    it('hydrates pendingUnresolvedToolCallCount from an unresolved single-call assistant turn', () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'user', content: 'fetch weather' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [{ id: 'call-a', name: 'fa', arguments: '{}' }],
        },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);
    });

    it('hydrates pendingUnresolvedToolCallCount from a trailing multi-call assistant turn', () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'user', content: 'fan out' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [
            { id: 'call-a', name: 'fa', arguments: '{}' },
            { id: 'call-b', name: 'fb', arguments: '{}' },
          ],
        },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBe(2);
    });

    it('counts only unresolved siblings when a fan-out was partially resolved', () => {
      // Primed chain saw one of two tool results before being
      // preempted — only the still-unresolved call should remain in
      // the guard count so the server can pick a correct recovery
      // path (1 → sendToolResult for the remaining id).
      const { model } = makeMockModel();
      const session = new ChatSession(model);
      session.primeHistory([
        { role: 'user', content: 'fan out' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [
            { id: 'call-a', name: 'fa', arguments: '{}' },
            { id: 'call-b', name: 'fb', arguments: '{}' },
          ],
        },
        { role: 'tool', content: 'result-a', toolCallId: 'call-a' },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBe(1);
    });

    it('re-priming recomputes pendingUnresolvedToolCallCount from the new trailing assistant turn', () => {
      const { model } = makeMockModel();
      const session = new ChatSession(model);

      session.primeHistory([
        { role: 'user', content: 'fan out' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [
            { id: 'call-a', name: 'fa', arguments: '{}' },
            { id: 'call-b', name: 'fb', arguments: '{}' },
          ],
        },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBe(2);

      session.primeHistory([
        { role: 'user', content: 'plain' },
        { role: 'assistant', content: 'plain-reply' },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBeNull();
    });

    it('rejects sendToolResult after a primed multi-call assistant turn', async () => {
      const { model, chatSessionContinueTool } = makeMockModel();
      const session = new ChatSession(model);

      // primeHistory leaves turnCount=0, so we must run startFromHistory
      // before sendToolResult can dispatch. Mock chatSessionStart to
      // emit a multi-call final result so the post-commit
      // recordToolCallFanout still carries a fan-out flag afterward.
      const { chatSessionStart } = model as unknown as {
        chatSessionStart: ReturnType<typeof vi.fn>;
      };
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult('replayed'),
        toolCalls: [
          { id: 'call-x', name: 'fx', arguments: {}, status: 'ok' },
          { id: 'call-y', name: 'fy', arguments: {}, status: 'ok' },
        ],
      } as unknown as ChatResult);

      session.primeHistory([
        { role: 'user', content: 'fan out' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [
            { id: 'call-a', name: 'fa', arguments: '{}' },
            { id: 'call-b', name: 'fb', arguments: '{}' },
          ],
        },
      ]);
      expect(session.pendingUnresolvedToolCallCount).toBe(2);
      await session.startFromHistory();
      expect(session.pendingUnresolvedToolCallCount).toBe(2);
      await expect(session.sendToolResult('call-x', 'result-x')).rejects.toThrow(
        /previous assistant turn emitted 2 ok tool calls/,
      );
      expect(chatSessionContinueTool).not.toHaveBeenCalled();
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

  // -------------------------------------------------------------------
  // Assistant toolCalls preservation on history commit
  // -------------------------------------------------------------------
  //
  // Regression coverage for the cold-replay structural bug: every
  // assistant history commit path must carry the turn's `toolCalls`
  // through to `ChatMessage.toolCalls` so subsequent `tool` messages
  // remain paired with a declaring assistant turn. Dropping them leads
  // to `chatSessionStart(history)` replays where a `tool_response`
  // delta renders against a preceding assistant turn that never
  // emitted a `tool_call`, silently corrupting the conversation and
  // changing model behavior after image-change restarts and
  // cache-miss cold starts. The assertions check the exact shape the
  // native tokenizer expects (`{id, name, arguments: string}`) and
  // that an ok-only filter applies (non-`ok` entries cannot be
  // rendered by the template).

  describe('assistant toolCalls preservation', () => {
    const makeToolCallResult = (id: string, name: string, args: Record<string, unknown> | string, status = 'ok') =>
      ({
        id,
        name,
        arguments: args,
        status,
        rawContent: '',
      }) as unknown as import('@mlx-node/core').ToolCallResult;

    it('preserves toolCalls on the non-streaming send() delta commit', async () => {
      const { model, chatSessionContinue, chatSessionStart } = makeMockModel();
      // turn 0 starts clean; the delta path on turn 1 carries the tool calls.
      chatSessionContinue.mockResolvedValueOnce({
        ...makeChatResult(''),
        toolCalls: [makeToolCallResult('call_1', 'foo', { a: 1 })],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('turn-0');
      await session.send('turn-1-with-tool-call');
      // Resolve the outstanding call so the next plain send() is legal.
      await session.sendToolResult('call_1', 'tool-out');

      // Force a reshuffle through chatSessionStart by changing the
      // image set so we can observe the replayed history entries.
      await session.send('turn-2-img', { images: [new Uint8Array([9, 9, 9])] });

      // chatSessionStart.mock.calls[1] — the second call is the
      // image-change restart with the full replayed history.
      const restartMessages = chatSessionStart.mock.calls[1][0];
      // Find the assistant entry for the tool-call turn.
      const toolCallAssistant = restartMessages.find((m) => m.role === 'assistant' && m.toolCalls !== undefined) as
        | ChatMessage
        | undefined;
      expect(toolCallAssistant).toBeDefined();
      expect(toolCallAssistant?.toolCalls).toEqual([{ id: 'call_1', name: 'foo', arguments: '{"a":1}' }]);
    });

    it('preserves toolCalls on the streaming sendStream() delta commit', async () => {
      const { model, chatStreamSessionContinue, chatSessionStart } = makeMockModel();
      chatStreamSessionContinue.mockImplementationOnce(async function* () {
        yield { text: '', done: false } as ChatStreamEvent;
        yield {
          ...finalChunk(''),
          toolCalls: [makeToolCallResult('call_stream', 'bar', { q: 'x' })],
        } as unknown as ChatStreamFinal;
      });

      const session = new ChatSession(model);
      await session.send('turn-0');
      for await (const _e of session.sendStream('turn-1-stream-tool-call')) void _e;
      // Resolve the outstanding call so the next plain send() is legal.
      await session.sendToolResult('call_stream', 'tool-out');

      // Image-change restart to observe the replayed history.
      await session.send('turn-2-img', { images: [new Uint8Array([7, 7, 7])] });

      const restartMessages = chatSessionStart.mock.calls[1][0];
      const toolCallAssistant = restartMessages.find((m) => m.role === 'assistant' && m.toolCalls !== undefined) as
        | ChatMessage
        | undefined;
      expect(toolCallAssistant).toBeDefined();
      expect(toolCallAssistant?.toolCalls).toEqual([{ id: 'call_stream', name: 'bar', arguments: '{"q":"x"}' }]);
    });

    it('preserves toolCalls on the turn-0 start path commit', async () => {
      const { model, chatSessionStart } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult(''),
        toolCalls: [makeToolCallResult('call_start', 'baz', { k: true })],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('turn-0-with-tool-call');
      // Resolve the outstanding call so the next plain send() is legal.
      await session.sendToolResult('call_start', 'tool-out');
      // Force a restart via image change so we can inspect the
      // replayed history passed to chatSessionStart.
      await session.send('turn-1-img', { images: [new Uint8Array([3, 3, 3])] });

      const restartMessages = chatSessionStart.mock.calls[1][0];
      const toolCallAssistant = restartMessages.find((m) => m.role === 'assistant' && m.toolCalls !== undefined) as
        | ChatMessage
        | undefined;
      expect(toolCallAssistant).toBeDefined();
      expect(toolCallAssistant?.toolCalls).toEqual([{ id: 'call_start', name: 'baz', arguments: '{"k":true}' }]);
    });

    it('preserves toolCalls on sendToolResult() commit', async () => {
      const { model, chatSessionStart, chatSessionContinueTool } = makeMockModel();
      // Turn 0 emits an outstanding ok tool call so sendToolResult is
      // legal. The tool-result turn then emits ANOTHER tool call —
      // the model chaining tool use — and the preceding assistant
      // entry for that reply must retain its toolCalls on history.
      chatSessionStart.mockResolvedValueOnce(makeChatResultWithSingleToolCall('start-reply', 'call-A'));
      chatSessionContinueTool.mockResolvedValueOnce({
        ...makeChatResult('mid-reply'),
        toolCalls: [makeToolCallResult('call-B', 'second_tool', { n: 2 })],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('turn-0');
      await session.sendToolResult('call-A', 'result-A');
      // Resolve the chained call so we can make a plain send() next.
      await session.sendToolResult('call-B', 'result-B');

      // Force a restart via image change so we can inspect the
      // replayed history passed to chatSessionStart.
      await session.send('turn-img', { images: [new Uint8Array([5, 5, 5])] });

      const restartMessages = chatSessionStart.mock.calls[1][0];
      const toolCallAssistants = restartMessages.filter((m) => m.role === 'assistant' && m.toolCalls !== undefined);
      // Two assistant turns carried tool calls: the initial start
      // (call-A) and the post-result chained reply (call-B).
      expect(toolCallAssistants).toHaveLength(2);
      expect(toolCallAssistants[0].toolCalls).toEqual([{ id: 'call-A', name: 'tool_fn', arguments: '{}' }]);
      expect(toolCallAssistants[1].toolCalls).toEqual([{ id: 'call-B', name: 'second_tool', arguments: '{"n":2}' }]);
    });

    it('preserves toolCalls on startFromHistory() commit', async () => {
      const { model, chatSessionStart } = makeMockModel();
      chatSessionStart.mockResolvedValueOnce({
        ...makeChatResult(''),
        toolCalls: [makeToolCallResult('call_cold', 'cold_tool', { p: 7 })],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      session.primeHistory([{ role: 'user', content: 'cold-prime' }]);
      await session.startFromHistory();
      // Resolve the outstanding call so the next plain send() is legal.
      await session.sendToolResult('call_cold', 'tool-out');

      // Force a restart via image change so we can inspect the
      // replayed history passed to chatSessionStart.
      await session.send('img-turn', { images: [new Uint8Array([2, 2, 2])] });

      const restartMessages = chatSessionStart.mock.calls[1][0];
      const toolCallAssistant = restartMessages.find((m) => m.role === 'assistant' && m.toolCalls !== undefined) as
        | ChatMessage
        | undefined;
      expect(toolCallAssistant).toBeDefined();
      expect(toolCallAssistant?.toolCalls).toEqual([{ id: 'call_cold', name: 'cold_tool', arguments: '{"p":7}' }]);
    });

    it('filters non-ok tool call results out of the history entry', async () => {
      const { model, chatSessionContinue, chatSessionStart } = makeMockModel();
      // A mix: one ok call plus one parse_error entry. The parse_error
      // cannot be re-rendered by the jinja template (no well-formed
      // name/arguments pair) and must be dropped from the history
      // entry — only the ok call should survive.
      chatSessionContinue.mockResolvedValueOnce({
        ...makeChatResult(''),
        toolCalls: [
          makeToolCallResult('call_ok', 'ok_tool', { x: 1 }),
          makeToolCallResult('call_bad', 'bad_tool', 'unparsed raw', 'parse_error'),
        ],
      } as unknown as ChatResult);

      const session = new ChatSession(model);
      await session.send('turn-0');
      // Turn 1 produces the mixed-status tool calls. The session's
      // fan-out invariant rejects a plain send while those calls are
      // outstanding, so drive through sendToolResult against the one
      // ok call to clear the obligation before forcing a restart.
      await session.send('turn-1-mixed');
      await session.sendToolResult('call_ok', 'ok-result');
      await session.send('turn-img', { images: [new Uint8Array([6, 6, 6])] });

      const restartMessages = chatSessionStart.mock.calls[1][0];
      const toolCallAssistant = restartMessages.find((m) => m.role === 'assistant' && m.toolCalls !== undefined) as
        | ChatMessage
        | undefined;
      expect(toolCallAssistant).toBeDefined();
      // Only the ok entry survives; the parse_error one is filtered.
      expect(toolCallAssistant?.toolCalls).toEqual([{ id: 'call_ok', name: 'ok_tool', arguments: '{"x":1}' }]);
    });

    it('omits toolCalls field when the turn produced none', async () => {
      const { model, chatSessionStart } = makeMockModel();
      const session = new ChatSession(model);
      await session.send('turn-0');
      await session.send('turn-1');
      // Force a restart to inspect the history shape.
      await session.send('turn-img', { images: [new Uint8Array([8, 8, 8])] });

      const restartMessages = chatSessionStart.mock.calls[1][0];
      const assistantEntries = restartMessages.filter((m) => m.role === 'assistant');
      expect(assistantEntries.length).toBeGreaterThan(0);
      for (const entry of assistantEntries) {
        // The key is either absent or undefined — never an empty array.
        expect(entry.toolCalls).toBeUndefined();
      }
    });
  });
});
