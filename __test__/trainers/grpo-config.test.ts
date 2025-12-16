import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { applyOverrides, ConfigError, getDefaultConfig, loadTomlConfig } from '@mlx-node/trl';

describe('MLX GRPO configuration', () => {
  it('returns a mutable copy of the default configuration', () => {
    const defaultsA = getDefaultConfig();
    const defaultsB = getDefaultConfig();

    expect(defaultsA).not.toBe(defaultsB);
    expect(defaultsA.learning_rate).toBeCloseTo(1e-6);
    expect(defaultsA.num_generations).toBe(64);
  });

  it('loads TOML configuration and merges with defaults', () => {
    const tomlPath = resolve(process.cwd(), '__test__/trainers/smoke_test.toml');
    const config = loadTomlConfig(tomlPath);

    expect(config.run_name).toBe('smoke-test-run');
    expect(config.max_train_samples).toBe(50);
    expect(config.num_generations).toBe(2);
    expect(config.max_new_tokens).toBe(32);
    expect(config.use_compile).toBe(false);
    expect(config.max_prompt_length).toBe(512);
  });

  it('applies CLI-style overrides without mutating the base config', () => {
    const base = getDefaultConfig();
    const overrides = ['learning_rate=2e-6', 'run_name=test-run', 'use_compile=false', 'seed=123'];
    const updated = applyOverrides(base, overrides);

    expect(updated.learning_rate).toBeCloseTo(2e-6);
    expect(updated.run_name).toBe('test-run');
    expect(updated.use_compile).toBe(false);
    expect(updated.seed).toBe(123);

    // Ensure original remains unchanged
    expect(base.run_name).toBe('Qwen-1.5B-MLX-GRPO-gsm8k');
    expect(base.use_compile).toBe(true);
  });

  it('throws when an override uses an unknown key', () => {
    const base = getDefaultConfig();
    expect(() => applyOverrides(base, ['unknown_key=1'])).toThrow(ConfigError);
  });

  it('throws when an override value is invalid', () => {
    const base = getDefaultConfig();
    expect(() => applyOverrides(base, ['batch_size=not-a-number'])).toThrow(ConfigError);
  });

  it('throws when the TOML file cannot be read', () => {
    const invalidPath = resolve(process.cwd(), 'non-existent-config.toml');
    expect(() => loadTomlConfig(invalidPath)).toThrow(ConfigError);
  });
});
