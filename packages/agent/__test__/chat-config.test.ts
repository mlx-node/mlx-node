import type { ThinkingLevel } from '@earendil-works/pi-ai';
import { createToolDefinition } from '@mlx-node/lm';
import { LAUNCH_PRESETS } from '@mlx-node/server';
import { describe, expect, it } from 'vite-plus/test';

import { buildChatConfig } from '../src/provider/chat-config.js';

describe('buildChatConfig', () => {
  it('maps every pi thinking level (and undefined) to the native reasoningEffort', () => {
    const table: Array<[ThinkingLevel | undefined, string]> = [
      [undefined, 'none'],
      ['minimal', 'low'],
      ['low', 'low'],
      ['medium', 'medium'],
      ['high', 'high'],
      ['xhigh', 'high'],
      ['max', 'high'],
    ];
    for (const [reasoning, expected] of table) {
      const config = buildChatConfig('qwen3_5', reasoning === undefined ? undefined : { reasoning }, undefined);
      expect(config.reasoningEffort, `reasoning=${String(reasoning)}`).toBe(expected);
    }
  });

  it('uses the LAUNCH_PRESETS sampling base and maxOutputTokens for the model type', () => {
    const qwen = buildChatConfig('qwen3_5', undefined, undefined);
    expect(qwen.temperature).toBe(0.6);
    expect(qwen.topP).toBe(0.95);
    expect(qwen.topK).toBe(20);
    expect(qwen.minP).toBe(0.0);
    expect(qwen.maxNewTokens).toBe(LAUNCH_PRESETS['qwen3_5']!.maxOutputTokens);

    const gemma = buildChatConfig('gemma4', undefined, undefined);
    expect(gemma.temperature).toBe(0.7);
    expect(gemma.topK).toBe(64);
    expect(gemma.maxNewTokens).toBe(LAUNCH_PRESETS['gemma4']!.maxOutputTokens);
  });

  it('lets per-call options override the preset base', () => {
    const config = buildChatConfig('qwen3_5', { maxTokens: 512, temperature: 0.15 }, undefined);
    expect(config.maxNewTokens).toBe(512);
    expect(config.temperature).toBe(0.15);
    // Non-overridden preset fields stay intact.
    expect(config.topK).toBe(20);
    expect(config.topP).toBe(0.95);
  });

  it('does not override when the option is absent', () => {
    const config = buildChatConfig('qwen3_5', {}, undefined);
    expect(config.maxNewTokens).toBe(LAUNCH_PRESETS['qwen3_5']!.maxOutputTokens);
    expect(config.temperature).toBe(0.6);
  });

  it('attaches tools only when non-empty', () => {
    const tool = createToolDefinition('get_weather', 'weather', { location: { type: 'string' } }, ['location']);
    expect(buildChatConfig('qwen3_5', undefined, [tool]).tools).toEqual([tool]);
    expect(buildChatConfig('qwen3_5', undefined, []).tools).toBeUndefined();
    expect(buildChatConfig('qwen3_5', undefined, undefined).tools).toBeUndefined();
  });

  it('never sets reuseCache (ChatSession forces it)', () => {
    const config = buildChatConfig('qwen3_5', { reasoning: 'high', maxTokens: 64 }, undefined);
    expect('reuseCache' in config).toBe(false);
  });

  it('gives lfm2_moe the first-class MoE sampler (LFM2.5-8B-A1B card), NOT the dense lfm2 preset', () => {
    const config = buildChatConfig('lfm2_moe', undefined, undefined);
    // Concrete card values — deliberately not compared against
    // LAUNCH_PRESETS['lfm2'] (the dense 1.2B guidance: temp 0.05, topK 50).
    expect(config.temperature).toBe(0.2);
    expect(config.topK).toBe(80);
    expect(config.repetitionPenalty).toBe(1.05);
    expect(config.maxNewTokens).toBe(8192);
    // Guard against silently re-aliasing onto the dense preset.
    expect(config.temperature).not.toBe(LAUNCH_PRESETS['lfm2']!.sampling.temperature);
    expect(config.topK).not.toBe(LAUNCH_PRESETS['lfm2']!.sampling.topK);
  });

  it('throws a clear error for a model type with no launch preset', () => {
    expect(() => buildChatConfig('harrier', undefined, undefined)).toThrow(/no launch preset .*harrier/i);
  });

  it('does not mutate the shared LAUNCH_PRESETS sampling object', () => {
    const before = { ...LAUNCH_PRESETS['qwen3_5']!.sampling };
    const config = buildChatConfig('qwen3_5', { temperature: 0.99 }, undefined);
    expect(config.temperature).toBe(0.99);
    expect(LAUNCH_PRESETS['qwen3_5']!.sampling).toEqual(before);
  });
});
