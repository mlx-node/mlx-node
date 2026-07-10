import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  Model,
} from '@earendil-works/pi-ai';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import type { ChatStreamFinal, ToolCallResult } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { TurnEmitter } from '../src/provider/events.js';

const MODEL: Model<Api> = {
  id: 'qwen3.5-test',
  name: 'Qwen3.5 Test',
  api: 'openai-completions',
  provider: 'mlx-node',
  baseUrl: 'http://localhost',
  reasoning: true,
  input: ['text'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 32768,
  maxTokens: 8192,
};

function makeEmitter(): { emitter: TurnEmitter; stream: AssistantMessageEventStream } {
  const stream = createAssistantMessageEventStream();
  return { emitter: new TurnEmitter(stream, MODEL), stream };
}

function delta(text: string, isReasoning?: boolean): { text: string; done: false; isReasoning?: boolean } {
  return isReasoning === undefined ? { text, done: false } : { text, done: false, isReasoning };
}

function makeFinal(overrides: Partial<ChatStreamFinal> = {}): ChatStreamFinal {
  return {
    text: '',
    done: true,
    finishReason: 'stop',
    toolCalls: [],
    thinking: null,
    numTokens: 7,
    promptTokens: 100,
    reasoningTokens: 0,
    rawText: '',
    cachedTokens: 60,
    ...overrides,
  };
}

function okCall(id: string, name: string, args: Record<string, unknown>): ToolCallResult {
  return { id, name, arguments: args, status: 'ok', rawContent: JSON.stringify({ name, arguments: args }) };
}

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function types(events: AssistantMessageEvent[]): string[] {
  return events.map((e) => e.type);
}

function emittedText(events: AssistantMessageEvent[]): string {
  return events
    .filter((e): e is Extract<AssistantMessageEvent, { type: 'text_delta' }> => e.type === 'text_delta')
    .map((e) => e.delta)
    .join('');
}

function finalMessage(events: AssistantMessageEvent[]): AssistantMessage {
  const last = events[events.length - 1]!;
  if (last.type === 'done') return last.message;
  if (last.type === 'error') return last.error;
  throw new Error(`stream did not terminate: last event ${last.type}`);
}

describe('TurnEmitter', () => {
  it('emits a plain-text turn with usage and stopReason on the final message', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('Hello'));
    emitter.onDelta(delta(' world'));
    emitter.onFinal(makeFinal({ text: 'Hello world' }));

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_delta', 'text_end', 'done']);
    expect(emittedText(events)).toBe('Hello world');

    const message = finalMessage(events);
    expect(message.role).toBe('assistant');
    expect(message.model).toBe('qwen3.5-test');
    expect(message.api).toBe('openai-completions');
    expect(message.provider).toBe('mlx-node');
    expect(message.content).toEqual([{ type: 'text', text: 'Hello world' }]);
    expect(message.stopReason).toBe('stop');
    expect(message.usage.input).toBe(40); // promptTokens 100 - cachedTokens 60
    expect(message.usage.output).toBe(7);
    expect(message.usage.cacheRead).toBe(60);
    expect(message.usage.cacheWrite).toBe(0);
    expect(message.usage.totalTokens).toBe(107); // promptTokens + numTokens
    expect(message.usage.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 });

    // pi's agent loop consumes result(): it must be the same message as the terminal event's.
    expect(await stream.result()).toBe(message);
    // Every partial-carrying event mutates the same message object.
    const started = events[0]! as Extract<AssistantMessageEvent, { type: 'start' }>;
    expect(started.partial).toBe(message);
  });

  it('treats a missing cachedTokens as 0 cacheRead with full prompt input', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(makeFinal({ cachedTokens: undefined, promptTokens: 50, numTokens: 3 }));
    const message = finalMessage(await collect(stream));
    expect(message.usage.input).toBe(50);
    expect(message.usage.cacheRead).toBe(0);
    expect(message.usage.totalTokens).toBe(53);
  });

  it('never lets a stale cache count drive input negative', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(makeFinal({ promptTokens: 10, cachedTokens: 25, numTokens: 1 }));
    const message = finalMessage(await collect(stream));
    expect(message.usage.input).toBe(0);
  });

  it('routes reasoning deltas to a thinking block and closes it when text starts', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('Let me ', true));
    emitter.onDelta(delta('think.', true));
    emitter.onDelta(delta('The answer is 4.'));
    emitter.onFinal(makeFinal({ text: 'The answer is 4.', reasoningTokens: 2 }));

    const events = await collect(stream);
    expect(types(events)).toEqual([
      'start',
      'thinking_start',
      'thinking_delta',
      'thinking_delta',
      'thinking_end',
      'text_start',
      'text_delta',
      'text_end',
      'done',
    ]);
    const thinkingEnd = events.find((e) => e.type === 'thinking_end')!;
    expect(thinkingEnd.type === 'thinking_end' && thinkingEnd.content).toBe('Let me think.');
    expect(finalMessage(events).content).toEqual([
      { type: 'thinking', thinking: 'Let me think.' },
      { type: 'text', text: 'The answer is 4.' },
    ]);
  });

  it('suppresses <tool_call> markup split across text deltas', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('Checking now.'));
    emitter.onDelta(delta('<tool'));
    emitter.onDelta(delta('_call>{"name":"get_weather"}'));
    emitter.onFinal(
      makeFinal({
        text: 'Checking now.',
        finishReason: 'tool_calls',
        toolCalls: [okCall('call_1', 'get_weather', { location: 'Tokyo' })],
      }),
    );

    const events = await collect(stream);
    expect(types(events)).toEqual([
      'start',
      'text_start',
      'text_delta',
      'text_end',
      'toolcall_start',
      'toolcall_delta',
      'toolcall_end',
      'done',
    ]);
    expect(emittedText(events)).toBe('Checking now.');
    expect(emittedText(events)).not.toContain('<tool');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('toolUse');
    expect(message.content).toEqual([
      { type: 'text', text: 'Checking now.' },
      { type: 'toolCall', id: 'call_1', name: 'get_weather', arguments: { location: 'Tokyo' } },
    ]);
  });

  it('never ratifies a whitespace-only text block ahead of a tool call', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('\n\n'));
    emitter.onDelta(delta('<tool_call>{"name":"ls"}</tool_call>'));
    emitter.onFinal(makeFinal({ finishReason: 'tool_calls', toolCalls: [okCall('call_1', 'ls', {})] }));

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done']);
    expect(finalMessage(events).content).toEqual([{ type: 'toolCall', id: 'call_1', name: 'ls', arguments: {} }]);
  });

  it('releases a held non-tag suffix at final (buffer flush)', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('result: <tool'));
    emitter.onFinal(makeFinal({ text: 'result: <tool' }));

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_delta', 'text_end', 'done']);
    expect(emittedText(events)).toBe('result: <tool');
    expect(finalMessage(events).content).toEqual([{ type: 'text', text: 'result: <tool' }]);
  });

  it('emits one toolcall trio per parsed call and stops with toolUse', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(
      makeFinal({
        finishReason: 'tool_calls',
        toolCalls: [okCall('call_1', 'get_weather', { location: 'Tokyo' }), okCall('call_2', 'ls', { path: '/tmp' })],
      }),
    );

    const events = await collect(stream);
    expect(types(events)).toEqual([
      'start',
      'toolcall_start',
      'toolcall_delta',
      'toolcall_end',
      'toolcall_start',
      'toolcall_delta',
      'toolcall_end',
      'done',
    ]);
    const trioDeltas = events.filter(
      (e): e is Extract<AssistantMessageEvent, { type: 'toolcall_delta' }> => e.type === 'toolcall_delta',
    );
    expect(trioDeltas.map((e) => e.delta)).toEqual(['{"location":"Tokyo"}', '{"path":"/tmp"}']);
    const ends = events.filter(
      (e): e is Extract<AssistantMessageEvent, { type: 'toolcall_end' }> => e.type === 'toolcall_end',
    );
    expect(ends.map((e) => e.contentIndex)).toEqual([0, 1]);
    expect(ends.map((e) => e.toolCall.id)).toEqual(['call_1', 'call_2']);

    const message = finalMessage(events);
    expect(message.stopReason).toBe('toolUse');
    expect(message.content).toEqual([
      { type: 'toolCall', id: 'call_1', name: 'get_weather', arguments: { location: 'Tokyo' } },
      { type: 'toolCall', id: 'call_2', name: 'ls', arguments: { path: '/tmp' } },
    ]);
  });

  it('synthesizes a malformed pi ToolCall for non-ok results so pi validation fails it', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(
      makeFinal({
        finishReason: 'tool_calls',
        toolCalls: [
          {
            id: 'call_9',
            name: '',
            arguments: {},
            status: 'invalid_json',
            error: 'invalid JSON in tool_call',
            rawContent: '{"name": broken',
          },
        ],
      }),
    );

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done']);
    const message = finalMessage(events);
    expect(message.content).toEqual([
      {
        type: 'toolCall',
        id: 'call_9',
        name: 'malformed_tool_call',
        arguments: { raw: '{"name": broken', error: 'invalid JSON in tool_call' },
      },
    ]);
    // No ok call was emitted, so the malformed-only turn does not claim toolUse.
    expect(message.stopReason).toBe('stop');
  });

  it('keeps the parsed name on a non-ok call that has one', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(
      makeFinal({
        toolCalls: [
          {
            id: 'call_3',
            name: 'get_weather',
            arguments: 'not-json',
            status: 'parse_error',
            error: 'arguments not parseable',
            rawContent: '{"name":"get_weather","arguments":"not-json"}',
          },
        ],
      }),
    );
    const message = finalMessage(await collect(stream));
    expect(message.content).toEqual([
      {
        type: 'toolCall',
        id: 'call_3',
        name: 'get_weather',
        arguments: { raw: '{"name":"get_weather","arguments":"not-json"}', error: 'arguments not parseable' },
      },
    ]);
  });

  it('maps finishReason length to stopReason length', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('truncated outpu'));
    emitter.onFinal(makeFinal({ text: 'truncated outpu', finishReason: 'length' }));

    const events = await collect(stream);
    const last = events[events.length - 1]!;
    expect(last.type).toBe('done');
    expect(last.type === 'done' && last.reason).toBe('length');
    expect(finalMessage(events).stopReason).toBe('length');
  });

  it('maps other native finish reasons (repetition) to stop', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(makeFinal({ finishReason: 'repetition' }));
    expect(finalMessage(await collect(stream)).stopReason).toBe('stop');
  });

  it('routes a finishReason=error final to error semantics, closing the open block first', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('so far'));
    emitter.onFinal(makeFinal({ finishReason: 'error' }));

    const events = await collect(stream);
    // The open text block must be balanced (text_end) BEFORE the terminal.
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_end', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('error');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toMatch(/finishReason=error/);
    expect(message.content).toEqual([{ type: 'text', text: 'so far' }]);
  });

  it('recovers held-back buffer text on a finishReason=error final', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('result: <tool'));
    emitter.onFinal(makeFinal({ finishReason: 'error' }));

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_delta', 'text_end', 'error']);
    expect(emittedText(events)).toBe('result: <tool');
    expect(finalMessage(events).content).toEqual([{ type: 'text', text: 'result: <tool' }]);
  });

  it('synthesizes an aborted final with every open block closed before the terminal', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('planning', true));
    emitter.onDelta(delta('partial answ'));
    emitter.onAborted();

    const events = await collect(stream);
    // The open text block must be balanced (text_end) BEFORE the terminal.
    expect(types(events)).toEqual([
      'start',
      'thinking_start',
      'thinking_delta',
      'thinking_end',
      'text_start',
      'text_delta',
      'text_end',
      'error',
    ]);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('aborted');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('aborted');
    expect(message.errorMessage).toBeTruthy();
    expect(message.content).toEqual([
      { type: 'thinking', thinking: 'planning' },
      { type: 'text', text: 'partial answ' },
    ]);
    expect(await stream.result()).toBe(message);
  });

  it('closes an open thinking block before the abort terminal', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('mid-thought', true));
    emitter.onAborted();

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'thinking_start', 'thinking_delta', 'thinking_end', 'error']);
    expect(finalMessage(events).content).toEqual([{ type: 'thinking', thinking: 'mid-thought' }]);
  });

  it('recovers held-back buffer text and closes the block before the abort terminal', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('partial <tool'));
    emitter.onAborted();

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_delta', 'text_end', 'error']);
    expect(emittedText(events)).toBe('partial <tool');
    expect(finalMessage(events).content).toEqual([{ type: 'text', text: 'partial <tool' }]);
  });

  it('emits an error event with stopReason error and errorMessage on onError, block closed first', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onDelta(delta('hi'));
    emitter.onError(new Error('boom'));

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_end', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('error');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toBe('boom');
    expect(message.content).toEqual([{ type: 'text', text: 'hi' }]);
  });

  it('stringifies non-Error throwables into errorMessage', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onError('string failure');
    const message = finalMessage(await collect(stream));
    expect(message.errorMessage).toBe('string failure');
  });

  it('contains a revoked Proxy error value: no throw, one error terminal, result() settles', async () => {
    const { emitter, stream } = makeEmitter();
    // On a revoked Proxy even `err instanceof Error` throws (the prototype
    // walk hits the revoked trap), so onError must guard every read.
    const { proxy, revoke } = Proxy.revocable({}, {});
    revoke();

    expect(() => emitter.onError(proxy)).not.toThrow();

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('error');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toBe('unserializable error');
    expect(await stream.result()).toBe(message);
  });

  it('ignores events after the terminal event', async () => {
    const { emitter, stream } = makeEmitter();
    emitter.onFinal(makeFinal({ text: 'done' }));
    emitter.onDelta(delta('late'));
    emitter.onFinal(makeFinal({ text: 'again' }));
    emitter.onAborted();
    emitter.onError(new Error('late error'));

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'done']);
    expect(finalMessage(events).stopReason).toBe('stop');
  });
});
