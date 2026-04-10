import { QWEN35_CONFIGS, getQwen35Config } from '@mlx-node/lm';
import { describe, it, expect } from 'vite-plus/test';

describe('Qwen3.5 Config', () => {
  it('should have valid config presets', () => {
    expect(Object.keys(QWEN35_CONFIGS).length).toBeGreaterThan(0);
    for (const [_name, config] of Object.entries(QWEN35_CONFIGS)) {
      expect(config.vocabSize).toBe(151936);
      expect(config.fullAttentionInterval).toBeGreaterThan(0);
      expect(config.partialRotaryFactor).toBeGreaterThan(0);
      expect(config.partialRotaryFactor).toBeLessThanOrEqual(1.0);
    }
  });

  it('should retrieve config by name', () => {
    const config = getQwen35Config('qwen3.5-0.6b');
    expect(config.hiddenSize).toBe(1024);
    expect(config.numLayers).toBe(28);
    expect(config.numHeads).toBe(16);
    expect(config.numKvHeads).toBe(8);
    expect(config.linearNumValueHeads).toBe(64);
    expect(config.linearNumKeyHeads).toBe(16);
  });

  it('should throw for unknown config name', () => {
    expect(() => getQwen35Config('nonexistent')).toThrow('Unknown');
  });
});
