/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { describe, expect, it } from 'vite-plus/test';

import {
  alignedRange,
  ENGINE_RANGE_OPTIONS,
  engineRange,
  isEngineRangeKey,
  isRangeKey,
  RANGE_OPTIONS,
  RANGE_OPTIONS_SHORT,
  useRange,
  type RangeKey,
} from '../ui/src/lib/range.js';

const NOW = new Date(2026, 10, 11, 14, 37, 21, 500).getTime();

describe('alignedRange', () => {
  it('1h: twelve 5-minute buckets ending at the end of the current one, epoch aligned', () => {
    const range = alignedRange('1h', NOW);
    expect(range.bucket).toBe('5m');
    expect(range.to % 300_000).toBe(0);
    expect(range.to).toBeGreaterThan(NOW);
    expect(range.to - NOW).toBeLessThanOrEqual(300_000);
    expect(range.from).toBe(range.to - 12 * 300_000);
    expect(range.starts).toHaveLength(12);
    expect(range.starts[0]).toBe(range.from);
    expect(range.starts[11]).toBe(range.to - 300_000);
  });

  it('24h: 24 hour buckets ending at the start of the next local hour', () => {
    const range = alignedRange('24h', NOW);
    expect(range.bucket).toBe('hour');
    expect(range.to).toBe(new Date(2026, 10, 11, 15, 0, 0, 0).getTime());
    expect(range.from).toBe(range.to - 24 * 3_600_000);
    expect(range.starts).toHaveLength(24);
    expect(range.starts[1] - range.starts[0]).toBe(3_600_000);
  });

  it('7d and 30d: local midnights, ending at the next one', () => {
    for (const [key, days] of [
      ['7d', 7],
      ['30d', 30],
    ] as const) {
      const range = alignedRange(key, NOW);
      expect(range.bucket).toBe('day');
      expect(range.to).toBe(new Date(2026, 10, 12).getTime());
      expect(range.starts).toHaveLength(days);
      expect(range.from).toBe(range.starts[0]);
      expect(range.starts[days - 1]).toBe(new Date(2026, 10, 11).getTime());
      for (const start of range.starts) {
        const at = new Date(start);
        expect(at.getHours() * 60 + at.getMinutes()).toBe(0);
      }
    }
  });

  it('is equal inside one bucket, so the request path stays stable', () => {
    const keys: RangeKey[] = ['1h', '24h', '7d', '30d'];
    for (const key of keys) {
      const first = alignedRange(key, new Date(2026, 10, 11, 14, 35, 0, 1).getTime());
      const later = alignedRange(key, new Date(2026, 10, 11, 14, 39, 59, 999).getTime());
      expect(later, key).toEqual(first);
    }
    expect(alignedRange('1h', new Date(2026, 10, 11, 14, 40, 0, 0).getTime()).to).not.toBe(alignedRange('1h', NOW).to);
  });
});

describe('engineRange', () => {
  it('uses fixed, epoch-aligned buckets with a set number of points', () => {
    const expected = { '1h': [60_000, 60], '6h': [300_000, 72], '24h': [900_000, 96], '7d': [3_600_000, 168] } as const;
    for (const [key, [bucketMs, points]] of Object.entries(expected)) {
      if (!isEngineRangeKey(key)) throw new Error(key);
      const range = engineRange(key, NOW);
      expect(range.bucketMs).toBe(bucketMs);
      expect(range.starts).toHaveLength(points);
      expect(range.to % bucketMs).toBe(0);
      expect(range.to).toBeGreaterThan(NOW);
      expect(range.from).toBe(range.to - points * bucketMs);
      expect(range.starts[0]).toBe(range.from);
    }
  });
});

describe('range keys', () => {
  it('narrows by exact compare only', () => {
    expect(['1h', '24h', '7d', '30d'].every(isRangeKey)).toBe(true);
    expect(['', '90d', '1H', ' 1h', '6h'].some(isRangeKey)).toBe(false);
    expect(['1h', '6h', '24h', '7d'].every(isEngineRangeKey)).toBe(true);
    expect(['30d', '1d', ''].some(isEngineRangeKey)).toBe(false);
  });

  it('offers exactly the keys as options', () => {
    expect(RANGE_OPTIONS.map((option) => option.value)).toEqual(['1h', '24h', '7d', '30d']);
    expect(RANGE_OPTIONS.map((option) => option.label)).toEqual(['1 hour', '24 hours', '7 days', '30 days']);
    expect(RANGE_OPTIONS_SHORT.map((option) => option.label)).toEqual(['1h', '24h', '7d', '30d']);
    expect(ENGINE_RANGE_OPTIONS.map((option) => option.value)).toEqual(['1h', '6h', '24h', '7d']);
  });
});

describe('useRange', () => {
  function mount(initial: string): { text: () => string; set: (key: RangeKey) => void; unmount: () => void } {
    let setter: ((key: RangeKey) => void) | null = null;
    function Probe(): ReactElement {
      const [range, setRange] = useRange('24h');
      const location = useLocation();
      setter = setRange;
      return createElement('span', null, `${range} ${location.search}`);
    }
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const container = document.createElement('div');
    const root = createRoot(container);
    act(() => root.render(createElement(MemoryRouter, { initialEntries: [initial] }, createElement(Probe))));
    return {
      text: () => container.textContent ?? '',
      set: (key) => act(() => setter?.(key)),
      unmount: () => act(() => root.unmount()),
    };
  }

  it('reads ?range= and falls back on an unknown value', () => {
    const known = mount('/metrics?range=7d');
    expect(known.text()).toBe('7d ?range=7d');
    known.unmount();
    const unknown = mount('/metrics?range=90d');
    expect(unknown.text().startsWith('24h ')).toBe(true);
    unknown.unmount();
  });

  it('writes ?range= and keeps the other params', () => {
    const view = mount('/metrics?tab=speed');
    view.set('30d');
    expect(view.text()).toBe('30d ?tab=speed&range=30d');
    view.unmount();
  });
});
