/**
 * Unit tests for the `Qwen35Session` wrapper.
 *
 * These tests exercise the JS-side orchestration only — the underlying
 * `Qwen35Model` is stubbed out with `vi.fn()` mocks so the suite runs
 * fast in CI without booting MLX. The gated Rust integration test at
 * `crates/mlx-core/tests/qwen3_5_delta_chat.rs` covers the actual
 * native session path end-to-end.
 */
import type { ChatConfig, ChatMessage, ChatResult, ChatStreamChunk } from '@mlx-node/core';
import { Qwen35Session } from '@mlx-node/lm';
import type { ChatStreamEvent, Qwen35Model } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

/** Build a minimal `ChatResult` stub sufficient for the session class. */
function makeChatResult(rawText: string): ChatResult {
  return {
    text: rawText,
    rawText,
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
 * Construct a mocked `Qwen35Model` with spyable `chatSessionStart`,
 * `chatSessionContinue`, and `resetCaches` entry points. Returns the
 * typed mock for use in assertions.
 */
function makeMockModel() {
  const chatSessionStart = vi.fn(
    async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> =>
      makeChatResult('turn-1-reply'),
  );
  const chatSessionContinue = vi.fn(
    async (_userMessage: string, _images?: Uint8Array[] | null, _config?: ChatConfig | null): Promise<ChatResult> =>
      makeChatResult('turn-n-reply'),
  );
  const resetCaches = vi.fn(() => undefined);

  const model = {
    chatSessionStart,
    chatSessionContinue,
    resetCaches,
  } as unknown as Qwen35Model;

  return { model, chatSessionStart, chatSessionContinue, resetCaches };
}

describe('Qwen35Session', () => {
  it('routes the first send through chatSessionStart with the full messages array', async () => {
    const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
    const session = new Qwen35Session(model, { system: 'Be concise.' });

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
    // reuseCache is forced on by the session class.
    expect(config?.reuseCache).toBe(true);
  });

  it('omits the system prompt from turn 1 when none was configured', async () => {
    const { model, chatSessionStart } = makeMockModel();
    const session = new Qwen35Session(model);

    await session.send('Hello');

    const [messages] = chatSessionStart.mock.calls[0];
    expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
  });

  it('routes subsequent sends through chatSessionContinue with just the user string', async () => {
    const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
    const session = new Qwen35Session(model);

    await session.send('First message');
    await session.send('Second message');
    await session.send('Third message');

    expect(chatSessionStart).toHaveBeenCalledTimes(1);
    expect(chatSessionContinue).toHaveBeenCalledTimes(2);
    expect(session.turns).toBe(3);

    expect(chatSessionContinue.mock.calls[0][0]).toBe('Second message');
    expect(chatSessionContinue.mock.calls[1][0]).toBe('Third message');

    // reuseCache forced on across continue calls as well.
    for (const call of chatSessionContinue.mock.calls) {
      expect(call[2]?.reuseCache).toBe(true);
    }
  });

  it('forces reuseCache=true even when the caller explicitly passes false', async () => {
    const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
    const session = new Qwen35Session(model);

    await session.send('First', { reuseCache: false });
    await session.send('Second', { reuseCache: false });

    expect(chatSessionStart.mock.calls[0][1]?.reuseCache).toBe(true);
    expect(chatSessionContinue.mock.calls[0][2]?.reuseCache).toBe(true);
  });

  it('merges defaultConfig and per-call config (with per-call taking precedence)', async () => {
    const { model, chatSessionStart } = makeMockModel();
    const session = new Qwen35Session(model, {
      defaultConfig: { maxNewTokens: 32, temperature: 0.2 },
    });

    await session.send('Hello', { temperature: 0.8 });

    const [, config] = chatSessionStart.mock.calls[0];
    expect(config?.maxNewTokens).toBe(32);
    expect(config?.temperature).toBe(0.8);
    expect(config?.reuseCache).toBe(true);
  });

  it('rejects concurrent send() calls with a clear error', async () => {
    // Resolve the first call manually so we can race a second call while
    // the first is still in flight.
    let resolveFirst: (r: ChatResult) => void = () => {
      /* overwritten below */
    };
    const first = new Promise<ChatResult>((r) => {
      resolveFirst = r;
    });
    const chatSessionStart = vi.fn(async () => first);
    const chatSessionContinue = vi.fn(async () => makeChatResult('continue'));
    const resetCaches = vi.fn();
    const model = {
      chatSessionStart,
      chatSessionContinue,
      resetCaches,
    } as unknown as Qwen35Model;
    const session = new Qwen35Session(model);

    const firstPromise = session.send('First');
    // The first call is awaiting — try a concurrent send().
    await expect(session.send('Second')).rejects.toThrow(/concurrent send\(\) not allowed/);

    // Unblock the first call and verify it resolves cleanly.
    resolveFirst(makeChatResult('first'));
    await firstPromise;
    expect(session.turns).toBe(1);

    // After the first call completes, a new send() should succeed.
    await session.send('Third');
    expect(session.turns).toBe(2);
    expect(chatSessionContinue).toHaveBeenCalledTimes(1);
  });

  it('reset() clears the turn counter and calls resetCaches on the model', async () => {
    const { model, chatSessionStart, resetCaches } = makeMockModel();
    const session = new Qwen35Session(model);

    await session.send('First');
    await session.send('Second');
    expect(session.turns).toBe(2);

    await session.reset();

    expect(resetCaches).toHaveBeenCalledTimes(1);
    expect(session.turns).toBe(0);

    // After reset, the next send should go through chatSessionStart again.
    await session.send('Fresh start');
    expect(chatSessionStart).toHaveBeenCalledTimes(2);
  });

  it('releases the inFlight flag even when the underlying call rejects', async () => {
    const { model, chatSessionStart } = makeMockModel();
    chatSessionStart.mockRejectedValueOnce(new Error('boom'));
    const session = new Qwen35Session(model);

    await expect(session.send('Hello')).rejects.toThrow('boom');

    // turnCount should NOT have advanced on failure.
    expect(session.turns).toBe(0);

    // The next send() should not be blocked by a stale inFlight flag.
    await session.send('Retry');
    expect(session.turns).toBe(1);
    expect(chatSessionStart).toHaveBeenCalledTimes(2);
  });

  // ---------------------------------------------------------------------
  // Streaming (sendStream) tests — cover the same turn-routing / inFlight
  // / reuseCache / turnCount semantics as `send()` but over the
  // AsyncGenerator surface backed by `chatStreamSessionStart` /
  // `chatStreamSessionContinue`.
  // ---------------------------------------------------------------------

  /** Fake final chunk shape used by the streaming generator bridge. */
  function finalChunk(text: string): ChatStreamEvent {
    return {
      text,
      done: true,
      finishReason: 'stop',
      toolCalls: [],
      thinking: null,
      numTokens: 2,
      promptTokens: 1,
      reasoningTokens: 0,
      rawText: text,
    } satisfies ChatStreamEvent;
  }

  /** Build a mocked `Qwen35Model` with streaming entry points spied. */
  function makeMockStreamModel() {
    const chatStreamSessionStart = vi.fn(async function* streamStart(
      _messages: ChatMessage[],
      _config?: ChatConfig | null,
    ): AsyncGenerator<ChatStreamEvent> {
      yield { text: 'Hello', done: false } satisfies ChatStreamEvent;
      yield { text: ' world', done: false } satisfies ChatStreamEvent;
      yield finalChunk('Hello world');
    });
    const chatStreamSessionContinue = vi.fn(async function* streamContinue(
      _userMessage: string,
      _config?: ChatConfig | null,
    ): AsyncGenerator<ChatStreamEvent> {
      yield { text: 'Again', done: false } satisfies ChatStreamEvent;
      yield finalChunk('Again!');
    });
    const chatSessionStart = vi.fn(
      async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> =>
        makeChatResult('non-stream-turn-1'),
    );
    const chatSessionContinue = vi.fn(
      async (_userMessage: string, _images?: Uint8Array[] | null, _config?: ChatConfig | null): Promise<ChatResult> =>
        makeChatResult('non-stream-turn-n'),
    );
    const resetCaches = vi.fn(() => undefined);

    const model = {
      chatStreamSessionStart,
      chatStreamSessionContinue,
      chatSessionStart,
      chatSessionContinue,
      resetCaches,
    } as unknown as Qwen35Model;

    return {
      model,
      chatStreamSessionStart,
      chatStreamSessionContinue,
      chatSessionStart,
      chatSessionContinue,
      resetCaches,
    };
  }

  describe('sendStream', () => {
    it('routes the first sendStream through chatStreamSessionStart with the full messages array', async () => {
      const { model, chatStreamSessionStart, chatStreamSessionContinue } = makeMockStreamModel();
      const session = new Qwen35Session(model, { system: 'Be concise.' });

      expect(session.turns).toBe(0);
      const events: ChatStreamEvent[] = [];
      for await (const e of session.sendStream('Hi there!')) {
        events.push(e);
      }

      expect(chatStreamSessionStart).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();
      expect(session.turns).toBe(1);
      expect(events.length).toBe(3);
      expect(events[events.length - 1].done).toBe(true);

      const [messages, config] = chatStreamSessionStart.mock.calls[0];
      expect(messages).toEqual([
        { role: 'system', content: 'Be concise.' },
        { role: 'user', content: 'Hi there!' },
      ]);
      expect(config?.reuseCache).toBe(true);
    });

    it('routes subsequent sendStream calls through chatStreamSessionContinue with just the user string', async () => {
      const { model, chatStreamSessionStart, chatStreamSessionContinue } = makeMockStreamModel();
      const session = new Qwen35Session(model);

      for await (const _e of session.sendStream('First')) {
        void _e;
      }
      for await (const _e of session.sendStream('Second')) {
        void _e;
      }
      for await (const _e of session.sendStream('Third')) {
        void _e;
      }

      expect(chatStreamSessionStart).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinue).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(3);
      expect(chatStreamSessionContinue.mock.calls[0][0]).toBe('Second');
      expect(chatStreamSessionContinue.mock.calls[1][0]).toBe('Third');
      for (const call of chatStreamSessionContinue.mock.calls) {
        // With the T2 signature fix, position 1 is the `images` guard
        // parameter and `config` now lives at position 2.
        expect((call as unknown as [string, unknown, ChatConfig | null | undefined])[2]?.reuseCache).toBe(true);
      }
    });

    it('forces reuseCache=true even when the caller explicitly passes false', async () => {
      const { model, chatStreamSessionStart, chatStreamSessionContinue } = makeMockStreamModel();
      const session = new Qwen35Session(model);

      for await (const _e of session.sendStream('First', { reuseCache: false })) {
        void _e;
      }
      for await (const _e of session.sendStream('Second', { reuseCache: false })) {
        void _e;
      }

      expect(chatStreamSessionStart.mock.calls[0][1]?.reuseCache).toBe(true);
      // T2 signature fix: config is now at position 2 (images guard at 1).
      expect(
        (chatStreamSessionContinue.mock.calls[0] as unknown as [string, unknown, ChatConfig | null | undefined])[2]
          ?.reuseCache,
      ).toBe(true);
    });

    it('rejects concurrent sendStream calls while another send is in flight', async () => {
      // Freeze the first stream until we manually pull from it.
      let releaseFirst: () => void = () => {
        /* overwritten */
      };
      const firstUnblock = new Promise<void>((r) => {
        releaseFirst = r;
      });
      const chatStreamSessionStart = vi.fn(async function* streamStart(
        _messages: ChatMessage[],
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        await firstUnblock;
        yield finalChunk('done');
      });
      const chatStreamSessionContinue = vi.fn(async function* streamContinue(
        _userMessage: string,
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        yield finalChunk('done');
      });
      const resetCaches = vi.fn();
      const model = {
        chatStreamSessionStart,
        chatStreamSessionContinue,
        resetCaches,
      } as unknown as Qwen35Model;
      const session = new Qwen35Session(model);

      // Kick off the first stream (don't iterate yet — generator is lazy,
      // but the `sendStream` wrapper should set `inFlight = true` as soon
      // as iteration begins).
      const firstIter = session.sendStream('First');
      const firstNextPromise = firstIter.next();

      // Second call should be rejected while the first is still open.
      await expect(async () => {
        for await (const _e of session.sendStream('Second')) {
          void _e;
        }
      }).rejects.toThrow(/concurrent send/i);

      // Unblock and drain the first stream so state cleans up.
      releaseFirst();
      // Drain remaining events from the first iterator.
      await firstNextPromise;
      for (;;) {
        const { done } = await firstIter.next();
        if (done) break;
      }

      expect(session.turns).toBe(1);

      // After the first stream terminates, a new stream should succeed.
      for await (const _e of session.sendStream('Third')) {
        void _e;
      }
      expect(session.turns).toBe(2);
      expect(chatStreamSessionContinue).toHaveBeenCalledTimes(1);
    });

    it('increments turnCount only when the final done chunk is yielded', async () => {
      const { model } = makeMockStreamModel();
      const session = new Qwen35Session(model);

      // Fully drain the stream — must see the `done: true` chunk.
      let sawDone = false;
      for await (const e of session.sendStream('Hello')) {
        if (e.done) sawDone = true;
      }
      expect(sawDone).toBe(true);
      expect(session.turns).toBe(1);
    });

    it('does NOT increment turnCount when the caller breaks mid-stream', async () => {
      const { model, chatStreamSessionStart } = makeMockStreamModel();
      const session = new Qwen35Session(model);

      for await (const e of session.sendStream('Hello')) {
        // Break on the first delta — never see the final chunk.
        expect(e.done).toBe(false);
        break;
      }

      // Cancellation path: turn counter stays at 0 so the next stream
      // routes through chatStreamSessionStart again (semantically: the
      // caller abandoned the turn).
      expect(session.turns).toBe(0);

      for await (const _e of session.sendStream('Retry')) {
        void _e;
      }
      expect(chatStreamSessionStart).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(1);
    });

    it('clears inFlight on exception from the underlying stream', async () => {
      // First call throws after a partial delta; subsequent calls
      // complete normally. Use a shared counter so the second `sendStream`
      // retry routes through the same `chatStreamSessionStart` mock but
      // with the non-throwing branch.
      let callCount = 0;
      const chatStreamSessionStart = vi.fn(async function* streamStart(
        _messages: ChatMessage[],
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        callCount++;
        yield { text: 'partial', done: false } satisfies ChatStreamEvent;
        if (callCount === 1) {
          throw new Error('boom');
        }
        yield finalChunk('recovered');
      });
      const chatStreamSessionContinue = vi.fn(async function* streamContinue(
        _userMessage: string,
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        yield finalChunk('ok');
      });
      const resetCaches = vi.fn();
      const model = {
        chatStreamSessionStart,
        chatStreamSessionContinue,
        resetCaches,
      } as unknown as Qwen35Model;
      const session = new Qwen35Session(model);

      await expect(async () => {
        for await (const _e of session.sendStream('Hello')) {
          void _e;
        }
      }).rejects.toThrow('boom');
      expect(session.turns).toBe(0);

      // After a failure, a follow-up stream should not be blocked by a
      // stale `inFlight` flag — and since turnCount is still 0 it should
      // re-route through chatStreamSessionStart.
      for await (const _e of session.sendStream('Retry')) {
        void _e;
      }
      expect(chatStreamSessionStart).toHaveBeenCalledTimes(2);
      expect(session.turns).toBe(1);
    });

    it('clears inFlight on caller break (normal termination path)', async () => {
      const { model } = makeMockStreamModel();
      const session = new Qwen35Session(model);

      for await (const _e of session.sendStream('Hello')) {
        void _e;
        break;
      }
      // Must be able to start a new stream immediately — no stale
      // inFlight flag should block it.
      for await (const _e of session.sendStream('Retry')) {
        void _e;
      }
      // No assertion on turn count here — that's covered by the
      // dedicated "does NOT increment" test.
    });

    it('does NOT increment turnCount when the stream throws a guard-violation error', async () => {
      // Simulates the Rust side's `send_stream_error` helper: a
      // guard-violation (text-only, missing im_end_id, reuse_cache
      // disabled, etc.) arrives as a thrown exception from the async
      // generator rather than a fake `done: true` chunk. The session
      // must catch this in its `finally` block, clear `inFlight`, and
      // leave `turnCount` untouched so the NEXT `sendStream()` routes
      // back through `chatStreamSessionStart` — otherwise the session
      // would be bricked on its first failure.
      // eslint-disable-next-line require-yield
      const failingStart = vi.fn(async function* streamStart(
        _messages: ChatMessage[],
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        // Throw BEFORE any yield to simulate a pre-decode guard
        // violation coming through as an `Err` on the Rust mpsc.
        await Promise.resolve();
        throw new Error('tokenizer: im_end_id not found');
      });
      const chatStreamSessionContinue = vi.fn(async function* streamContinue(
        _userMessage: string,
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        yield finalChunk('should not be called');
      });
      const resetCaches = vi.fn();
      const model = {
        chatStreamSessionStart: failingStart,
        chatStreamSessionContinue,
        resetCaches,
      } as unknown as Qwen35Model;
      const session = new Qwen35Session(model);

      await expect(async () => {
        for await (const _e of session.sendStream('Hi')) {
          void _e;
        }
      }).rejects.toThrow('tokenizer: im_end_id not found');

      // Turn counter must NOT have advanced.
      expect(session.turns).toBe(0);
      // Continue path must NOT have been called.
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();

      // Swap in a succeeding start mock. The next sendStream() must
      // still route through the turn-0 start path, proving the
      // session wasn't bricked by the earlier failure.
      const succeedingStart = vi.fn(async function* streamStart(
        _messages: ChatMessage[],
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        yield { text: 'ok', done: false } satisfies ChatStreamEvent;
        yield finalChunk('ok');
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (model as any).chatStreamSessionStart = succeedingStart;

      for await (const _e of session.sendStream('Retry')) {
        void _e;
      }
      expect(succeedingStart).toHaveBeenCalledTimes(1);
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();
      expect(session.turns).toBe(1);
    });

    it('does NOT increment turnCount when the final chunk has finishReason="error"', async () => {
      // Belt-and-suspenders test: even if a hypothetical future code
      // path were to yield a `done: true` chunk with
      // `finishReason: 'error'` as a regular stream event (instead of
      // throwing), `Qwen35Session.sendStream` must NOT treat it as a
      // successful completion. Refusing to advance the counter keeps
      // the session routing on the turn-0 `chatStreamSessionStart`
      // path so the next send re-initializes rather than piling onto
      // a non-existent session.
      const errorFinalChunk: ChatStreamEvent = {
        text: 'fake error chunk',
        done: true,
        finishReason: 'error',
        toolCalls: [],
        thinking: null,
        numTokens: 0,
        promptTokens: 0,
        reasoningTokens: 0,
        rawText: 'fake error chunk',
      };
      const chatStreamSessionStart = vi.fn(async function* streamStart(
        _messages: ChatMessage[],
        _config?: ChatConfig | null,
      ): AsyncGenerator<ChatStreamEvent> {
        yield errorFinalChunk;
      });
      const chatStreamSessionContinue = vi.fn();
      const resetCaches = vi.fn();
      const model = {
        chatStreamSessionStart,
        chatStreamSessionContinue,
        resetCaches,
      } as unknown as Qwen35Model;
      const session = new Qwen35Session(model);

      // Iteration completes normally — no exception — but we should
      // see the error chunk yielded and turnCount stays at 0.
      const events: ChatStreamEvent[] = [];
      for await (const e of session.sendStream('Hi')) {
        events.push(e);
      }
      expect(events).toEqual([errorFinalChunk]);
      expect(session.turns).toBe(0);
      expect(chatStreamSessionContinue).not.toHaveBeenCalled();
    });
  });
});

// Keep this reference in module scope to prevent oxlint's
// no-unused-vars diagnostic on the import used only for `satisfies`
// annotations above when the test file is consumed standalone.
void ({} as ChatStreamChunk);
