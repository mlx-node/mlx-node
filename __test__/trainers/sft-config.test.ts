import { describe, expect, it } from 'vitest';
import { getDefaultSFTConfig, mergeSFTConfig, applySFTOverrides, SFTConfigError } from '@mlx-node/trl';

describe('SFT configuration', () => {
  it('returns a mutable copy of the default configuration', () => {
    const defaultsA = getDefaultSFTConfig();
    const defaultsB = getDefaultSFTConfig();

    expect(defaultsA).not.toBe(defaultsB);
    expect(defaultsA.learning_rate).toBeCloseTo(2e-5);
    expect(defaultsA.batch_size).toBe(4);
    expect(defaultsA.num_epochs).toBe(3);
    expect(defaultsA.max_seq_length).toBe(2048);
    expect(defaultsA.completion_only).toBe(false); // default is false for TRL parity
    expect(defaultsA.label_smoothing).toBe(0.0);
  });

  it('merges partial config with defaults', () => {
    const defaults = getDefaultSFTConfig();
    const partial = {
      learning_rate: 1e-4,
      batch_size: 8,
      num_epochs: 5,
    };
    const merged = mergeSFTConfig(defaults, partial);

    expect(merged.learning_rate).toBeCloseTo(1e-4);
    expect(merged.batch_size).toBe(8);
    expect(merged.num_epochs).toBe(5);
    // Unchanged defaults
    expect(merged.max_seq_length).toBe(2048);
    expect(merged.completion_only).toBe(false); // default is false for TRL parity
  });

  it('applies CLI-style overrides without mutating the base config', () => {
    const base = getDefaultSFTConfig();
    const overrides = ['learning_rate=1e-4', 'run_name=my-sft-run', 'completion_only=true', 'seed=42'];
    const updated = applySFTOverrides(base, overrides);

    expect(updated.learning_rate).toBeCloseTo(1e-4);
    expect(updated.run_name).toBe('my-sft-run');
    expect(updated.completion_only).toBe(true);
    expect(updated.seed).toBe(42);

    // Ensure original remains unchanged
    expect(base.run_name).toBe('sft-run');
    expect(base.completion_only).toBe(false); // default is false for TRL parity
  });

  it('throws when an override uses an unknown key', () => {
    const base = getDefaultSFTConfig();
    expect(() => applySFTOverrides(base, ['unknown_key=1'])).toThrow(SFTConfigError);
  });

  it('throws when an override value is invalid', () => {
    const base = getDefaultSFTConfig();
    expect(() => applySFTOverrides(base, ['batch_size=not-a-number'])).toThrow(SFTConfigError);
  });

  it('throws when batch_size is not an integer', () => {
    const base = getDefaultSFTConfig();
    expect(() => applySFTOverrides(base, ['batch_size=2.5'])).toThrow(SFTConfigError);
  });
});
