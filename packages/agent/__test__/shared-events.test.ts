import {
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessageEvent,
  type Model,
} from '@earendil-works/pi-ai';
import type { ChatStreamFinal } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { TurnEmitter } from '../src/provider/events.js';
import { encodeSharedEvent, SharedEventDecoder } from '../src/provider/shared-events.js';

const model = { id: 'test', api: 'mlx', provider: 'mlx' } as Model<Api>;
const final: ChatStreamFinal = {
  text: 'Hello world',
  done: true,
  finishReason: 'stop',
  toolCalls: [],
  thinking: null,
  thinkingEnabled: true,
  numTokens: 2,
  promptTokens: 10,
  reasoningTokens: 0,
  rawText: 'Hello world',
};

describe('compact shared inference events', () => {
  it('round-trips text, reasoning and tool calls even when queued partials already contain future output', async () => {
    const stream = createAssistantMessageEventStream();
    const emitter = new TurnEmitter(stream, model);
    emitter.onDelta({ text: 'Think', isReasoning: true, done: false });
    emitter.onDelta({ text: 'Hello', done: false });
    emitter.onDelta({ text: ' world', done: false });
    emitter.onFinal({
      ...final,
      toolCalls: [{ id: 'call-1', name: 'bash', arguments: { command: 'test' }, status: 'ok', rawContent: '' }],
    });
    const decoder = new SharedEventDecoder();
    const snapshots: AssistantMessageEvent[] = [];
    let last: AssistantMessageEvent | undefined;
    for await (const event of stream) {
      last = decoder.decode(JSON.parse(JSON.stringify(encodeSharedEvent(event))));
      snapshots.push(structuredClone(last));
    }
    expect(snapshots.filter((event) => event.type === 'text_delta').map((event) => event.partial.content[1])).toEqual([
      { type: 'text', text: 'Hello' },
      { type: 'text', text: 'Hello world' },
    ]);
    expect(last?.type).toBe('done');
    if (last?.type !== 'done') throw new Error('Expected final message');
    expect(decoder.partial!.content).toEqual(last.message.content);
    expect(last.message.content).toContainEqual({
      type: 'toolCall',
      id: 'call-1',
      name: 'bash',
      arguments: { command: 'test' },
    });
  });

  it('keeps wire traffic linear as generated output grows', async () => {
    const wireSize = async (tokens: number) => {
      const stream = createAssistantMessageEventStream();
      const emitter = new TurnEmitter(stream, model);
      for (let i = 0; i < tokens; i++) emitter.onDelta({ text: 'word ', done: false });
      emitter.onFinal({ ...final, text: 'word '.repeat(tokens) });
      let bytes = 0;
      for await (const event of stream) bytes += Buffer.byteLength(JSON.stringify(encodeSharedEvent(event)));
      return bytes;
    };
    const small = await wireSize(1000);
    const large = await wireSize(2000);
    expect(large).toBeLessThan(small * 2.1);
    expect(large).toBeLessThan(250_000);
  });

  it('balances a truncated text block before the client reports cancellation', () => {
    const decoder = new SharedEventDecoder();
    decoder.decode({ type: 'start', partial: { role: 'assistant', content: [] } as never });
    decoder.decode({ type: 'text_start', contentIndex: 0 });
    decoder.decode({ type: 'text_delta', contentIndex: 0, delta: 'partial' });
    expect([...decoder.finishOpenBlocks()]).toMatchObject([{ type: 'text_end', contentIndex: 0, content: 'partial' }]);
    expect([...decoder.finishOpenBlocks()]).toEqual([]);
  });
});
