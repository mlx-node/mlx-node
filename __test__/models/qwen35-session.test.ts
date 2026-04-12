/**
 * Unit tests for the `Qwen35Session` wrapper.
 *
 * These tests exercise the JS-side orchestration only — the underlying
 * `Qwen35Model` is stubbed out with `vi.fn()` mocks so the suite runs
 * fast in CI without booting MLX. The gated Rust integration test at
 * `crates/mlx-core/tests/qwen3_5_delta_chat.rs` covers the actual
 * native session path end-to-end.
 */
import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';
import { Qwen35Session } from '@mlx-node/lm';
import type { Qwen35Model } from '@mlx-node/lm';
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
    async (_userMessage: string, _config?: ChatConfig | null): Promise<ChatResult> => makeChatResult('turn-n-reply'),
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
      expect(call[1]?.reuseCache).toBe(true);
    }
  });

  it('forces reuseCache=true even when the caller explicitly passes false', async () => {
    const { model, chatSessionStart, chatSessionContinue } = makeMockModel();
    const session = new Qwen35Session(model);

    await session.send('First', { reuseCache: false });
    await session.send('Second', { reuseCache: false });

    expect(chatSessionStart.mock.calls[0][1]?.reuseCache).toBe(true);
    expect(chatSessionContinue.mock.calls[0][1]?.reuseCache).toBe(true);
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
});
