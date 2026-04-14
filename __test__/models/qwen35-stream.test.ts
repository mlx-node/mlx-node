import type { ChatMessage, ChatStreamChunk, ChatStreamHandle } from '@mlx-node/core';
import type { ChatStreamEvent } from '@mlx-node/lm';
import { _runChatStream } from '@mlx-node/lm';
import { describe, it, expect } from 'vite-plus/test';

/**
 * Creates a fake native chatStream callback method that emits `numTokens`
 * delta chunks followed by a final done chunk, matching the Rust signature.
 */
function fakeNativeMethod(numTokens: number) {
  return (
    _messages: unknown[],
    _config: unknown,
    callback: (err: Error | null, chunk: ChatStreamChunk) => void,
  ): Promise<ChatStreamHandle> => {
    let cancelledFlag = false;
    const handle = {
      cancel: () => {
        cancelledFlag = true;
      },
    } as ChatStreamHandle;

    setTimeout(() => {
      for (let i = 0; i < numTokens; i++) {
        if (cancelledFlag) break;
        callback(null, { text: `token${i}`, done: false } as ChatStreamChunk);
      }
      if (!cancelledFlag) {
        callback(null, {
          text: 'final text',
          done: true,
          finishReason: 'eos',
          toolCalls: [],
          thinking: 'some thinking',
          numTokens,
          rawText: 'raw final text',
        } as ChatStreamChunk);
      }
    }, 0);

    return Promise.resolve(handle);
  };
}

describe.sequential('_runChatStream bridge', () => {
  it('should yield delta chunks followed by a final chunk', async () => {
    const events: ChatStreamEvent[] = [];
    const messages: ChatMessage[] = [{ role: 'user', content: 'Hi' }];
    const native = fakeNativeMethod(3);
    const gen = _runChatStream((callback) => native(messages, null, callback));

    for await (const event of gen) {
      events.push(event);
    }

    expect(events).toHaveLength(4);
    expect(events[0]).toEqual({ text: 'token0', done: false });
    expect(events[1]).toEqual({ text: 'token1', done: false });
    expect(events[2]).toEqual({ text: 'token2', done: false });
    expect(events[3].done).toBe(true);
  });

  it('should populate final chunk fields correctly', async () => {
    let finalEvent: ChatStreamEvent | null = null;
    const messages: ChatMessage[] = [{ role: 'user', content: 'Hi' }];
    const native = fakeNativeMethod(2);
    const gen = _runChatStream((callback) => native(messages, null, callback));

    for await (const event of gen) {
      if (event.done) finalEvent = event;
    }

    expect(finalEvent).toEqual({
      text: 'final text',
      done: true,
      finishReason: 'eos',
      toolCalls: [],
      thinking: 'some thinking',
      numTokens: 2,
      promptTokens: 0,
      reasoningTokens: 0,
      rawText: 'raw final text',
      performance: undefined,
    });
  });

  it('should cancel generation on break via finally block', async () => {
    let cancelCalled = false;
    const native = (
      _messages: unknown[],
      _config: unknown,
      callback: (err: Error | null, chunk: ChatStreamChunk) => void,
    ): Promise<ChatStreamHandle> => {
      const handle = {
        cancel: () => {
          cancelCalled = true;
        },
      } as ChatStreamHandle;

      setTimeout(() => {
        for (let i = 0; i < 100; i++) {
          callback(null, { text: `t${i}`, done: false } as ChatStreamChunk);
        }
        callback(null, { text: '', done: true, finishReason: 'length', numTokens: 100 } as ChatStreamChunk);
      }, 0);

      return Promise.resolve(handle);
    };

    const events: string[] = [];
    const messages: ChatMessage[] = [{ role: 'user', content: 'Hi' }];
    for await (const event of _runChatStream((callback) => native(messages, null, callback))) {
      if (!event.done) {
        events.push(event.text);
        if (events.length >= 3) break;
      }
    }

    expect(events).toHaveLength(3);
    expect(cancelCalled).toBe(true);
  });

  it('should propagate errors from callback', async () => {
    const native = (
      _messages: unknown[],
      _config: unknown,
      callback: (err: Error | null, chunk: ChatStreamChunk) => void,
    ): Promise<ChatStreamHandle> => {
      setTimeout(() => {
        callback(new Error('generation failed'), null as unknown as ChatStreamChunk);
      }, 0);
      return Promise.resolve({ cancel: () => {} } as ChatStreamHandle);
    };

    const messages: ChatMessage[] = [{ role: 'user', content: 'Hi' }];
    await expect(async () => {
      for await (const _event of _runChatStream((callback) => native(messages, null, callback))) {
        // Should throw
      }
    }).rejects.toThrow('generation failed');
  });
});
