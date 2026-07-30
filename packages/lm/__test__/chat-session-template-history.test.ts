import type { ChatConfig, ChatMessage, ChatResult, ToolCallResult, ToolDefinition } from '@mlx-node/core';
import { describe, expect, it } from 'vitest';

import { ChatSession, type SessionCapableModel } from '../src/chat-session.js';
import type { ChatStreamEvent } from '../src/stream.js';

function chatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'assistant reply',
    toolCalls: [],
    thinking: 'private reasoning',
    thinkingEnabled: true,
    numTokens: 4,
    promptTokens: 10,
    reasoningTokens: 2,
    finishReason: 'stop',
    rawText: 'assistant reply',
    cachedTokens: 0,
    ...overrides,
  };
}

class RecordingModel implements SessionCapableModel {
  readonly startCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly continueCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly continueToolCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly startStreamCalls: Array<{
    messages: ChatMessage[];
    config: ChatConfig | null | undefined;
  }> = [];
  readonly results: ChatResult[] = [];
  readonly startStreamRuns: Array<ChatStreamEvent[] | Error> = [];
  readonly promptTokenCounts: number[] = [];
  readonly templateTools: Array<ToolDefinition[] | null | undefined> = [];

  applyChatTemplate(
    _messages: ChatMessage[],
    _addGenerationPrompt?: boolean | null,
    tools?: ToolDefinition[] | null,
  ): Uint32Array {
    this.templateTools.push(tools);
    return new Uint32Array(this.promptTokenCounts.shift() ?? 1);
  }

  contextLimits() {
    return {
      trainedWindowTokens: 4096,
      effectiveWindowTokens: 4096,
      pagedBlockCapacity: 256,
      pagedBlockSize: 16,
    };
  }

  async chatSessionStart(messages: ChatMessage[], config?: ChatConfig | null): Promise<ChatResult> {
    this.startCalls.push({ messages, config });
    return this.nextResult();
  }

  async chatSessionContinue(messages: ChatMessage[], config?: ChatConfig | null): Promise<ChatResult> {
    this.continueCalls.push({ messages, config });
    return this.nextResult();
  }

  async chatSessionContinueTool(messages: ChatMessage[], config?: ChatConfig | null): Promise<ChatResult> {
    this.continueToolCalls.push({ messages, config });
    return this.nextResult();
  }

  chatStreamSessionStart(messages: ChatMessage[], config?: ChatConfig | null): AsyncGenerator<ChatStreamEvent> {
    this.startStreamCalls.push({ messages, config });
    return this.nextStartStream();
  }

  chatStreamSessionContinue(): AsyncGenerator<ChatStreamEvent> {
    return this.emptyStream();
  }

  chatStreamSessionContinueTool(): AsyncGenerator<ChatStreamEvent> {
    return this.emptyStream();
  }

  resetCaches(): void {}

  private nextResult(): ChatResult {
    const result = this.results.shift();
    if (result === undefined) throw new Error('test model has no queued result');
    return result;
  }

  private async *emptyStream(): AsyncGenerator<ChatStreamEvent> {
    for (const event of [] as ChatStreamEvent[]) yield event;
  }

  private async *nextStartStream(): AsyncGenerator<ChatStreamEvent> {
    const run = this.startStreamRuns.shift() ?? [];
    if (run instanceof Error) throw run;
    for (const event of run) yield event;
  }
}

const tools: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'lookup',
      description: 'Look up a value',
      parameters: {
        type: 'object',
        properties: '{"query":{"type":"string"}}',
        required: ['query'],
      },
    },
  },
];

describe('ChatSession template-rendered continuation history', () => {
  it('passes the complete replayable transcript and preserves thinking provenance', async () => {
    const model = new RecordingModel();
    model.results.push(
      chatResult({
        text: 'first',
        thinking: 'reason one',
        thinkingEnabled: true,
      }),
      chatResult({
        text: 'second',
        thinking: undefined,
        thinkingEnabled: false,
      }),
    );
    const session = new ChatSession(model);

    await session.send('one', { config: { tools } });
    await session.send('two');

    expect(model.continueCalls).toHaveLength(1);
    expect(model.continueCalls[0]?.messages).toEqual([
      { role: 'user', content: 'one' },
      {
        role: 'assistant',
        content: 'first',
        reasoningContent: 'reason one',
        thinkingEnabled: true,
      },
      { role: 'user', content: 'two' },
    ]);
    expect(model.continueCalls[0]?.config?.tools).toEqual(tools);
  });

  it('passes the declaring assistant call and structured tool result to the template', async () => {
    const model = new RecordingModel();
    const toolCall: ToolCallResult = {
      id: 'call_1',
      name: 'lookup',
      arguments: { query: 'mlx' },
      status: 'ok',
      rawContent: '',
    };
    model.results.push(
      chatResult({
        text: '',
        toolCalls: [toolCall],
        thinking: 'need a lookup',
        thinkingEnabled: true,
      }),
      chatResult({ text: 'done', thinking: undefined, thinkingEnabled: false }),
    );
    const session = new ChatSession(model);

    await session.send('look this up', { config: { tools } });
    await session.sendToolResult('call_1', 'lookup failed', { isError: true });

    expect(model.continueToolCalls).toHaveLength(1);
    expect(model.continueToolCalls[0]?.messages).toEqual([
      { role: 'user', content: 'look this up' },
      {
        role: 'assistant',
        content: '',
        toolCalls: [{ id: 'call_1', name: 'lookup', arguments: '{"query":"mlx"}' }],
        reasoningContent: 'need a lookup',
        thinkingEnabled: true,
      },
      {
        role: 'tool',
        content: 'lookup failed',
        toolCallId: 'call_1',
        isError: true,
      },
    ]);
    expect(model.continueToolCalls[0]?.config?.tools).toEqual(tools);
  });
});

describe('ChatSession active tool transactionality', () => {
  it.each([
    {
      name: 'complete-history',
      run: (session: ChatSession<RecordingModel>) =>
        session.preflightContextCapacity([{ role: 'user', content: 'preflight only' }], { tools }),
    },
    {
      name: 'pending-message',
      run: (session: ChatSession<RecordingModel>) =>
        session.preflightPendingContextCapacity({ role: 'user', content: 'preflight only' }, { tools }),
    },
  ])('does not persist tools from $name preflight', async ({ run }) => {
    const model = new RecordingModel();
    const session = new ChatSession(model);

    await run(session);
    model.results.push(chatResult());
    await session.send('committed turn');

    expect(model.templateTools).toEqual([tools, null]);
    expect(model.startCalls[0]?.config?.tools).toBeUndefined();
  });

  it('does not persist tools when context-capacity validation rejects a turn', async () => {
    const model = new RecordingModel();
    const session = new ChatSession(model);
    model.promptTokenCounts.push(4097);

    await expect(session.send('oversized', { config: { tools } })).rejects.toThrow('context_length_exceeded');

    model.results.push(chatResult());
    await session.send('retry');

    expect(model.startCalls).toHaveLength(1);
    expect(model.startCalls[0]?.config?.tools).toBeUndefined();
  });

  it('does not persist tools when native inference rejects a turn', async () => {
    const model = new RecordingModel();
    const session = new ChatSession(model);

    await expect(session.send('failed', { config: { tools } })).rejects.toThrow('test model has no queued result');

    model.results.push(chatResult());
    await session.send('retry');

    expect(model.startCalls).toHaveLength(2);
    expect(model.startCalls[0]?.config?.tools).toEqual(tools);
    expect(model.startCalls[1]?.config?.tools).toBeUndefined();
  });

  it('does not persist tools when a stream throws before committing', async () => {
    const model = new RecordingModel();
    const session = new ChatSession(model);
    model.startStreamRuns.push(new Error('stream failed'));

    await expect(
      (async () => {
        for await (const _event of session.sendStream('failed', {
          config: { tools },
        })) {
          // Consume the stream so the queued failure is observed.
        }
      })(),
    ).rejects.toThrow('stream failed');

    model.results.push(chatResult());
    await session.send('retry');

    expect(model.startStreamCalls[0]?.config?.tools).toEqual(tools);
    expect(model.startCalls[0]?.config?.tools).toBeUndefined();
  });

  it('does not persist tools when the caller abandons a stream', async () => {
    const model = new RecordingModel();
    const session = new ChatSession(model);
    model.startStreamRuns.push([{ text: 'partial', done: false }]);

    for await (const _event of session.sendStream('abandoned', {
      config: { tools },
    })) {
      break;
    }

    model.results.push(chatResult());
    await session.send('retry');

    expect(model.startStreamCalls[0]?.config?.tools).toEqual(tools);
    expect(model.startCalls[0]?.config?.tools).toBeUndefined();
  });
});
