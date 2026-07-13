import { describe, expect, it } from 'vitest';

import { JACOBIAN_LAYERS } from '../../demo/jlens-core/jacobian-presets';
import { STARTERS } from '../../demo/jspace/starters';
import { GALLERY, STARTER_SLUGS } from '../../demo/jspace/starters/gallery';

describe('GALLERY', () => {
  it('ships exactly 8 tiles, french-season headline first', () => {
    expect(GALLERY).toHaveLength(8);
    expect(GALLERY[0]!.slug).toBe('french-season');
  });
  it('has unique slugs and STARTER_SLUGS mirrors them in order', () => {
    const slugs = GALLERY.map((g) => g.slug);
    expect(new Set(slugs).size).toBe(8);
    expect(STARTER_SLUGS).toEqual(slugs);
  });
  it('grades 5 strong + 3 weak', () => {
    expect(GALLERY.filter((g) => g.grade === 'strong')).toHaveLength(5);
    expect(GALLERY.filter((g) => g.grade === 'weak')).toHaveLength(3);
  });
  it('every band has onset <= peak and peak is a displayed layer', () => {
    for (const g of GALLERY) {
      expect(g.band.onset).toBeLessThanOrEqual(g.band.peak);
      expect(JACOBIAN_LAYERS).toContain(g.band.peak);
    }
  });
  it('every prompt is non-empty and every entry pins at least one concept', () => {
    for (const g of GALLERY) {
      expect(g.prompt.length).toBeGreaterThan(0);
      expect(g.concepts.length).toBeGreaterThan(0);
    }
  });
  it('STARTERS registry keys equal STARTER_SLUGS in order', () => {
    expect(Object.keys(STARTERS)).toEqual(STARTER_SLUGS);
  });
  it('every baked starter frame is structurally sound (both runs, aligned shapes)', () => {
    for (const g of GALLERY) {
      const frame = STARTERS[g.slug]!;
      expect(frame, `frame for ${g.slug}`).toBeDefined();
      expect(frame.slug).toBe(g.slug);
      expect(frame.prompt).toBe(g.prompt);
      expect(frame.concepts).toEqual(g.concepts);
      expect(frame.layers).toEqual(JACOBIAN_LAYERS);
      // partialFlags + pinned tracks are index-aligned with concepts.
      expect(frame.partialFlags).toHaveLength(g.concepts.length);
      for (const run of [frame.logit, frame.jacobian]) {
        expect(run.topK).toBe(10);
        expect(run.layers).toEqual(JACOBIAN_LAYERS);
        expect(run.pinned).toHaveLength(g.concepts.length);
        // one cell per (layer, position); each pin's rank track is the same length.
        expect(run.cells).toHaveLength(JACOBIAN_LAYERS.length * run.promptLen);
        for (const p of run.pinned) expect(p.ranks).toHaveLength(run.cells.length);
      }
      // Only the jacobian run applied a fitted Jacobian; the logit run did not.
      expect(frame.jacobian.jacobianApplied).toBe(true);
      expect(frame.logit.jacobianApplied).toBe(false);
    }
  });
});
