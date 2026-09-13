import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { Qwen35MoeModel, type ChatConfig, type ChatMessage, type ChatStreamChunk } from '@mlx-node/core';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

// Opt in with QWEN35_MOE_VISION_MODEL_PATH pointing to an MLX MoE checkpoint
// with vision weights. An explicitly configured but invalid path must fail.
const modelPath = process.env.QWEN35_MOE_VISION_MODEL_PATH;

describe.skipIf(!modelPath)('Qwen3.5 MoE vision reasoning budget', () => {
  let model: Qwen35MoeModel;
  let messages: ChatMessage[];

  beforeAll(async () => {
    model = await Qwen35MoeModel.load(modelPath!);
    expect(model.supportsImages()).toBe(true);
    expect(model.hasBlockPagedCache()).toBe(true);
    messages = [
      {
        role: 'user',
        content: 'Read the text in this image. Think through the layout and each line before answering.',
        images: [readFileSync(resolve(import.meta.dirname, '../../examples/ocr.png'))],
      },
    ];
  }, 300_000);

  afterAll(async () => {
    await model?.resetCaches();
  });

  it.each([0, 1, 2])('caps synchronous image reasoning at %i tokens', async (budget) => {
    await model.resetCaches();
    const result = await model.chatSessionStart(messages, config(budget));
    expect(result.thinkingEnabled).toBe(true);
    expect(result.reasoningTokens).toBeLessThanOrEqual(budget);
    expect(result.numTokens).toBeGreaterThan(0);
    if (budget === 0) expect(result.rawText).toBe('</think>');
  });

  it.each([0, 1, 2])('caps streaming image reasoning at %i tokens before emitting deltas', async (budget) => {
    await model.resetCaches();
    const deltas: ChatStreamChunk[] = [];
    const result = await new Promise<ChatStreamChunk>((resolve, reject) => {
      model
        .chatStreamSessionStart(messages, config(budget), (error, chunk) => {
          if (error) reject(error);
          else if (chunk.done) resolve(chunk);
          else deltas.push(chunk);
        })
        .catch(reject);
    });
    expect(result.thinkingEnabled).toBe(true);
    expect(result.reasoningTokens).toBeLessThanOrEqual(budget);
    expect(result.numTokens).toBeGreaterThan(0);
    if (budget === 0) {
      expect(result.rawText).toBe('</think>');
      // The closing marker itself belongs to the reasoning channel, but no
      // reasoning content may precede it in the emitted stream.
      expect(deltas.map((delta) => delta.text).join('')).toBe('</think>');
    }
  });
});

function config(budget: number): ChatConfig {
  return {
    temperature: 0,
    maxNewTokens: budget + 1,
    reasoningEffort: 'high',
    thinkingTokenBudget: budget,
    includeReasoning: true,
    reuseCache: true,
    enableMtp: false,
  };
}
