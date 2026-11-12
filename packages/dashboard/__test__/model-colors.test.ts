/**
 * @vitest-environment happy-dom
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  ensureModelColors,
  MODEL_COLORS_STORAGE_KEY,
  modelColor,
  modelKey,
  modelSeries,
  resetModelColors,
} from '../ui/src/lib/model-colors.js';

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

beforeEach(() => {
  resetModelColors();
  localStorage.clear();
});

afterEach(() => {
  resetModelColors();
  localStorage.clear();
});

describe('model colors', () => {
  it('uses the grouping key of the API', () => {
    expect(modelKey('qwen3')).toBe('qwen3');
    expect(modelKey('')).toBe('unknown');
  });

  it('gives the first 8 models a solid slot, the next 8 the same hues as alt, and folds the rest to other', () => {
    const names = Array.from({ length: 18 }, (_, i) => `m${String(i).padStart(2, '0')}`);
    ensureModelColors(names);
    expect(modelColor('m00')).toEqual({ ordinal: 0, color: 'var(--model-1)', pattern: 'solid' });
    expect(modelColor('m07')).toEqual({ ordinal: 7, color: 'var(--model-8)', pattern: 'solid' });
    expect(modelColor('m08')).toEqual({ ordinal: 8, color: 'var(--model-1)', pattern: 'alt' });
    expect(modelColor('m15')).toEqual({ ordinal: 15, color: 'var(--model-8)', pattern: 'alt' });
    expect(modelColor('m16')).toEqual({ ordinal: 16, color: 'var(--model-other)', pattern: 'other' });
  });

  it('never gives unknown a slot', () => {
    expect(modelColor('unknown').pattern).toBe('other');
    expect(modelColor('').color).toBe('var(--model-other)');
    ensureModelColors(['a']);
    expect(modelColor('a').ordinal).toBe(0);
  });

  it('is append-only: a model keeps its color whatever is registered or asked for later', () => {
    ensureModelColors(['b', 'a']);
    ensureModelColors(['c', 'a', 'b']);
    expect(modelColor('b').ordinal).toBe(0);
    expect(modelColor('a').ordinal).toBe(1);
    expect(modelColor('c').ordinal).toBe(2);
    expect(modelSeries(['c', 'b']).map((series) => series.color)).toEqual(['var(--model-3)', 'var(--model-1)']);
  });

  it('registers unseen series names sorted by code unit, keeps the caller order and labels', () => {
    ensureModelColors(['z']);
    const series = modelSeries(['qwen', 'Gemma', 'z', 'alpha'], new Map([['qwen', 'Qwen 3']]));
    expect(series.map((s) => s.key)).toEqual(['qwen', 'Gemma', 'z', 'alpha']);
    expect(series.map((s) => s.label)).toEqual(['Qwen 3', 'Gemma', 'z', 'alpha']);
    expect(modelColor('Gemma').ordinal).toBe(1);
    expect(modelColor('alpha').ordinal).toBe(2);
    expect(modelColor('qwen').ordinal).toBe(3);
    expect(series.every((s) => s.pattern === 'solid')).toBe(true);
  });

  it('marks models 9-16 as alt series', () => {
    ensureModelColors(Array.from({ length: 9 }, (_, i) => `n${i}`));
    expect(modelSeries(['n8'])[0]).toEqual({ key: 'n8', label: 'n8', color: 'var(--model-1)', pattern: 'alt' });
  });

  it('persists the registry and reads it back on load', async () => {
    ensureModelColors(['first', 'second']);
    await flushMicrotasks();
    expect(JSON.parse(localStorage.getItem(MODEL_COLORS_STORAGE_KEY) ?? 'null')).toEqual(['first', 'second']);

    localStorage.setItem(MODEL_COLORS_STORAGE_KEY, JSON.stringify(['kept', 7, '', 'unknown', 'kept', 'next']));
    vi.resetModules();
    const fresh = await import('../ui/src/lib/model-colors.js');
    expect(fresh.modelColor('kept').ordinal).toBe(0);
    expect(fresh.modelColor('next').ordinal).toBe(1);
    expect(fresh.modelColor('new').ordinal).toBe(2);
    fresh.resetModelColors();
  });

  it('survives unreadable storage', async () => {
    localStorage.setItem(MODEL_COLORS_STORAGE_KEY, '{not json');
    vi.resetModules();
    const fresh = await import('../ui/src/lib/model-colors.js');
    expect(fresh.modelColor('x').ordinal).toBe(0);
    fresh.resetModelColors();
  });

  it('stops registering at 256 names', () => {
    ensureModelColors(Array.from({ length: 300 }, (_, i) => `cap${i}`));
    expect(modelColor('cap255').ordinal).toBe(255);
    expect(modelColor('cap256')).toEqual({ ordinal: -1, color: 'var(--model-other)', pattern: 'other' });
  });

  it('forgets everything on reset', async () => {
    ensureModelColors(['one']);
    await flushMicrotasks();
    resetModelColors();
    expect(localStorage.getItem(MODEL_COLORS_STORAGE_KEY)).toBeNull();
    ensureModelColors(['two']);
    expect(modelColor('two').ordinal).toBe(0);
  });
});
