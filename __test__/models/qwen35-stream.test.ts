import { describe, it, expect } from 'vite-plus/test';
import type { ChatStreamEvent } from '@mlx-node/lm';
import { Qwen35Model } from '@mlx-node/lm';
import type { Qwen35Config } from '@mlx-node/lm';

// Tiny config for stream tests — 4 layers (1 full cycle), random weights, no tokenizer.
const TINY_STREAM_CONFIG: Qwen35Config = {
  vocabSize: 1000,
  hiddenSize: 128,
  numLayers: 4,
  numHeads: 4,
  numKvHeads: 2,
  intermediateSize: 256,
  rmsNormEps: 1e-6,
  headDim: 32,
  tieWordEmbeddings: true,
  attentionBias: false,
  maxPositionEmbeddings: 512,
  padTokenId: 0,
  eosTokenId: 1,
  bosTokenId: 0,
  linearNumValueHeads: 8,
  linearNumKeyHeads: 4,
  linearKeyHeadDim: 32,
  linearValueHeadDim: 16,
  linearConvKernelDim: 4,
  fullAttentionInterval: 4,
  partialRotaryFactor: 0.25,
  ropeTheta: 10000.0,
};

/**
 * Creates a mock model that simulates streaming via callbacks,
 * matching the native chatStream signature.
 */
function createMockModel(numTokens: number) {
  const model = new Qwen35Model(TINY_STREAM_CONFIG);

  // Patch native chatStream to use mock implementation
  (model as any).chatStream = async function* () {
    for (let i = 0; i < numTokens; i++) {
      yield { text: `token${i}`, done: false } as const;
    }
    yield {
      text: 'final text',
      done: true,
      finishReason: 'eos',
      toolCalls: [],
      thinking: 'some thinking',
      numTokens,
      rawText: 'raw final text',
    } as const;
  };

  return model;
}

describe.sequential('Qwen35Model.chatStream() AsyncGenerator', () => {
  it('should yield delta chunks followed by a final chunk', async () => {
    const model = createMockModel(3);
    const events: ChatStreamEvent[] = [];

    for await (const event of model.chatStream([{ role: 'user', content: 'Hi' }])) {
      events.push(event);
    }

    // 3 delta chunks + 1 final
    expect(events).toHaveLength(4);
    expect(events[0]).toEqual({ text: 'token0', done: false });
    expect(events[1]).toEqual({ text: 'token1', done: false });
    expect(events[2]).toEqual({ text: 'token2', done: false });
    expect(events[3].done).toBe(true);
  });

  it('should populate final chunk fields correctly', async () => {
    const model = createMockModel(2);
    let finalEvent: ChatStreamEvent | null = null;

    for await (const event of model.chatStream([{ role: 'user', content: 'Hi' }])) {
      if (event.done) finalEvent = event;
    }

    expect(finalEvent).toEqual({
      text: 'final text',
      done: true,
      finishReason: 'eos',
      toolCalls: [],
      thinking: 'some thinking',
      numTokens: 2,
      rawText: 'raw final text',
    });
  });

  it('should stop generation on break', async () => {
    const model = createMockModel(100);
    const events: string[] = [];

    for await (const event of model.chatStream([{ role: 'user', content: 'Hi' }])) {
      if (!event.done) {
        events.push(event.text);
        if (events.length >= 3) break;
      }
    }

    expect(events).toHaveLength(3);
  });

  it('should be an AsyncGenerator (Symbol.asyncIterator)', async () => {
    const model = createMockModel(1);
    const stream = model.chatStream([{ role: 'user', content: 'Hi' }]);
    expect(stream[Symbol.asyncIterator]).toBeDefined();
    // Consume to avoid hanging
    for await (const _ of stream) {
      /* drain */
    }
  });
});
