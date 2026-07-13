import { describe, expect, it } from 'vitest';

import { GALLERY, STARTER_SLUGS } from '../../demo/jspace/starters/gallery';
import { JACOBIAN_LAYERS } from '../../demo/jlens-core/jacobian-presets';

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
  it('grades 6 strong + 2 weak', () => {
    expect(GALLERY.filter((g) => g.grade === 'strong')).toHaveLength(6);
    expect(GALLERY.filter((g) => g.grade === 'weak')).toHaveLength(2);
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
});
