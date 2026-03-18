import { describe, expect, it } from 'vite-plus/test';
import { getDefaultSFTConfig, mergeSFTConfig, applySFTOverrides, SFTConfigError } from '@mlx-node/trl';

describe('SFT configuration', () => {
  it('returns a mutable copy of the default configuration', () => {
    const defaultsA = getDefaultSFTConfig();
    const defaultsB = getDefaultSFTConfig();

    expect(defaultsA).not.toBe(defaultsB);
    expect(defaultsA.learningRate).toBeCloseTo(2e-5);
    expect(defaultsA.batchSize).toBe(4);
    expect(defaultsA.numEpochs).toBe(3);
    expect(defaultsA.maxSeqLength).toBe(2048);
    expect(defaultsA.completionOnly).toBe(false); // default is false for TRL parity
    expect(defaultsA.labelSmoothing).toBe(0.0);
  });

  it('merges partial config with defaults', () => {
    const defaults = getDefaultSFTConfig();
    const partial = {
      learningRate: 1e-4,
      batchSize: 8,
      numEpochs: 5,
    };
    const merged = mergeSFTConfig(defaults, partial);

    expect(merged.learningRate).toBeCloseTo(1e-4);
    expect(merged.batchSize).toBe(8);
    expect(merged.numEpochs).toBe(5);
    // Unchanged defaults
    expect(merged.maxSeqLength).toBe(2048);
    expect(merged.completionOnly).toBe(false); // default is false for TRL parity
  });

  it('applies CLI-style overrides without mutating the base config', () => {
    const base = getDefaultSFTConfig();
    const overrides = ['learningRate=1e-4', 'runName=my-sft-run', 'completionOnly=true', 'seed=42'];
    const updated = applySFTOverrides(base, overrides);

    expect(updated.learningRate).toBeCloseTo(1e-4);
    expect(updated.runName).toBe('my-sft-run');
    expect(updated.completionOnly).toBe(true);
    expect(updated.seed).toBe(42);

    // Ensure original remains unchanged
    expect(base.runName).toBe('sft-run');
    expect(base.completionOnly).toBe(false); // default is false for TRL parity
  });

  it('throws when an override uses an unknown key', () => {
    const base = getDefaultSFTConfig();
    expect(() => applySFTOverrides(base, ['unknown_key=1'])).toThrow(SFTConfigError);
  });

  it('throws when an override value is invalid', () => {
    const base = getDefaultSFTConfig();
    expect(() => applySFTOverrides(base, ['batchSize=not-a-number'])).toThrow(SFTConfigError);
  });

  it('throws when batchSize is not an integer', () => {
    const base = getDefaultSFTConfig();
    expect(() => applySFTOverrides(base, ['batchSize=2.5'])).toThrow(SFTConfigError);
  });
});
