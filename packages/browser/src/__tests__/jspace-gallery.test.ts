import { describe, expect, it } from 'vitest';

import { JACOBIAN_LAYERS } from '../../demo/jlens-core/jacobian-presets';
import { STARTERS } from '../../demo/jspace/starters';
import { GALLERY, STARTER_SLUGS } from '../../demo/jspace/starters/gallery';

describe('GALLERY', () => {
  it('ships exactly 7 tiles, french-season headline first', () => {
    expect(GALLERY).toHaveLength(7);
    expect(GALLERY[0]!.slug).toBe('french-season');
  });
  it('has unique slugs and STARTER_SLUGS mirrors them in order', () => {
    const slugs = GALLERY.map((g) => g.slug);
    expect(new Set(slugs).size).toBe(7);
    expect(STARTER_SLUGS).toEqual(slugs);
  });
  it('grades 5 strong + 2 weak', () => {
    expect(GALLERY.filter((g) => g.grade === 'strong')).toHaveLength(5);
    expect(GALLERY.filter((g) => g.grade === 'weak')).toHaveLength(2);
  });
  it('never pins the model output — the unspoken-word invariant (codex #1)', () => {
    // eiffel-capital was dropped because `Paris` IS the ℓ24 argmax (a SPOKEN answer).
    // Guard the whole gallery in committed data so a bad re-bake can't reintroduce one.
    for (const g of GALLERY) {
      const jac = STARTERS[g.slug]!.jacobian;
      const outputArgmax = jac.cells[(JACOBIAN_LAYERS.length - 1) * jac.promptLen + (jac.promptLen - 1)]!.argmaxId;
      for (const pin of jac.pinned) {
        expect(pin.tokenId, `${g.slug} pins its ℓ24 greedy output ${outputArgmax}`).not.toBe(outputArgmax);
      }
    }
  });
  it("band.peak sits on the primary concept's min-rank layer (codex #6)", () => {
    for (const g of GALLERY) {
      const jac = STARTERS[g.slug]!.jacobian;
      const pos = jac.promptLen - 1;
      const ranksByLayer = JACOBIAN_LAYERS.map((_, li) => jac.pinned[0]!.ranks[li * jac.promptLen + pos]!);
      const best = Math.min(...ranksByLayer);
      const peakIdx = JACOBIAN_LAYERS.indexOf(g.band.peak);
      expect(ranksByLayer[peakIdx], `${g.slug} band.peak ℓ${g.band.peak} is not a min-rank layer`).toBe(best);
    }
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
