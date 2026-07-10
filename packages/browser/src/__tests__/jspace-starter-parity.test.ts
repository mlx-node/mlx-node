import { describe, expect, it } from 'vite-plus/test';

import { JACOBIAN_PRESETS } from '../../demo/jlens-core/jacobian-presets';

// The /jspace model-free STARTER grid (demo/jspace/starters/<slug>.json) and the
// lesson bake (demo/learn/widgets/jlens/baked/<slug>.json) are written from ONE
// serialized envelope per bake run and must stay byte-identical. A redirected bake
// (JLENS_OUT_DIR set) or any manual edit that desyncs the two copies fails here.
//
// NOTE: this is a Vitest *browser* suite (no node:fs), so we read the raw file
// bytes via Vite `?raw` eager globs rather than readFileSync — same byte-parity
// guarantee, browser-native. `?raw` returns the exact on-disk content (including
// the trailing newline the bake emits), so `toBe` is a true byte comparison, not a
// parse→re-serialize round-trip.
const STARTER_RAW = import.meta.glob('../../demo/jspace/starters/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;
const BAKED_RAW = import.meta.glob('../../demo/learn/widgets/jlens/baked/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const rawFor = (map: Record<string, string>, slug: string): string | undefined => {
  const key = Object.keys(map).find((k) => k.endsWith(`/${slug}.json`));
  return key === undefined ? undefined : map[key];
};

describe('starter ↔ lesson bake parity', () => {
  for (const preset of JACOBIAN_PRESETS) {
    it(`${preset.slug}.json is byte-identical in both dirs`, () => {
      const starter = rawFor(STARTER_RAW, preset.slug);
      const baked = rawFor(BAKED_RAW, preset.slug);
      expect(typeof starter, `starter ${preset.slug}.json is present`).toBe('string');
      expect(typeof baked, `baked ${preset.slug}.json is present`).toBe('string');
      expect(starter).toBe(baked);
    });
  }
});
