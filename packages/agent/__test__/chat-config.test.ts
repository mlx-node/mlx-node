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

  it('resolves lfm2_moe through the lfm2 preset alias (same LFM2.5 family)', () => {
    const config = buildChatConfig('lfm2_moe', undefined, undefined);
    expect(config.maxNewTokens).toBe(LAUNCH_PRESETS['lfm2']!.maxOutputTokens);
    expect(config.temperature).toBe(LAUNCH_PRESETS['lfm2']!.sampling.temperature);
    expect(config.repetitionPenalty).toBe(LAUNCH_PRESETS['lfm2']!.sampling.repetitionPenalty);
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
