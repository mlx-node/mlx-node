import { describe, it, expect } from 'vitest';
import { NativeRewardRegistry } from '@mlx-node/core';
import type { BuiltinRewardType } from '@mlx-node/core';
import { DEFAULT_GRPO_CONFIG, createRewardRegistry } from '@mlx-node/trl';

describe('NativeRewardRegistry NAPI Bindings', () => {
  it('should create registry and register rewards', () => {
    const registry = new NativeRewardRegistry();
    expect(registry.isEmpty).toBe(true);

    registry.register({
      rewardType: 'Length' as BuiltinRewardType,
      minLength: 10,
      maxLength: 100,
      useChars: true,
      weight: 1.0,
    });

    expect(registry.isEmpty).toBe(false);
    expect(registry.names).toContain('length');
  });

  it('should score single completion', () => {
    const registry = new NativeRewardRegistry();
    registry.register({
      rewardType: 'XmlFormat' as BuiltinRewardType,
      requiredTags: ['thinking', 'answer'],
      weight: 1.0,
    });

    const completion = '<thinking>Let me think</thinking><answer>42</answer>';
    const score = registry.score('prompt', completion);
    expect(score).toBe(1.0);
  });

  it('should score batch of completions', () => {
    const registry = new NativeRewardRegistry();
    registry.register({
      rewardType: 'Length' as BuiltinRewardType,
      minLength: 5,
      maxLength: 50,
      useChars: true,
      weight: 1.0,
    });

    const prompts = ['p1', 'p2', 'p3'];
    const completions = ['good length', 'ab', 'also good completion'];
    const scores = registry.scoreBatch(prompts, completions);

    expect(scores).toHaveLength(3);
    expect(scores[0]).toBe(1.0);
    expect(scores[1]).toBeLessThan(1.0);
    expect(scores[2]).toBe(1.0);
  });

  it('should combine multiple rewards with weights', () => {
    const registry = new NativeRewardRegistry();

    registry.register({
      rewardType: 'ToolUse' as BuiltinRewardType,
      allowedTools: ['search'],
      required: true,
      weight: 0.6,
    });

    registry.register({
      rewardType: 'XmlFormat' as BuiltinRewardType,
      requiredTags: ['result'],
      weight: 0.4,
    });

    const completion = '<tool_call><name>search</name><arguments>{}</arguments></tool_call><result>Found it!</result>';
    const score = registry.score('prompt', completion);
    expect(score).toBeGreaterThan(0.8);
  });
});

describe('createRewardRegistry helper', () => {
  it('should create a standalone registry', () => {
    const registry = createRewardRegistry();
    expect(registry).toBeInstanceOf(NativeRewardRegistry);
    expect(registry.isEmpty).toBe(true);
  });
});

describe('DEFAULT_GRPO_CONFIG', () => {
  it('should have reasonable defaults', () => {
    expect(DEFAULT_GRPO_CONFIG.learningRate).toBe(1e-6);
    expect(DEFAULT_GRPO_CONFIG.groupSize).toBe(4);
    expect(DEFAULT_GRPO_CONFIG.clipEpsilon).toBe(0.2);
    expect(DEFAULT_GRPO_CONFIG.temperature).toBe(0.8);
    expect(DEFAULT_GRPO_CONFIG.gradientClipNorm).toBe(1.0);
  });
});
