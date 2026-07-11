import { describe, expect, it } from 'vite-plus/test';

import type { BakedFile } from '../../demo/jlens-core/revive';
import { reviveRun } from '../../demo/jlens-core/revive';
import { compareToBakedFrame } from '../../demo/jlens-core/self-test';
import bakedJson from '../../demo/learn/widgets/jlens/baked/french-season.json';

const baked = bakedJson as unknown as BakedFile;
/** Fresh objects each call — reviveRun copies into new typed arrays. */
const frame = () => reviveRun(baked.jacobian);

/** Index of a pin's BEST (minimum) rank — the value the comparator reads. */
const argMinRank = (ranks: Int32Array) =>
  ranks.reduce((best, r, i) => (r < ranks[best]! ? i : best), 0);

describe('self-test oracle', () => {
  it('passes when live output IS the baked frame', () => {
    const v = compareToBakedFrame(frame(), frame());
    expect(v.ok).toBe(true);
    expect(v.topOneAgreement).toBe(1);
    expect(v.worstPinDelta).toBe(0);
  });

  it('tolerates benign GPU noise (best rank off by 1, one top-2 tie swap)', () => {
    const live = frame();
    // Perturb the MINIMUM rank — that is what worstPinDelta compares. Bumping any
    // other index would leave the metric untouched and the test would prove nothing.
    // The bump must land on a pin whose minimum is UNIQUE, or another tied cell
    // still holds the min and the delta stays 0. Data-verified against the
    // committed french-season bake: pins 0-2 (' season'/' summer'/' autumn') all
    // have min rank 1 with 11/5/5 tied cells; only pin[3] (' autom', min 556,
    // single occurrence) has a unique minimum. (The brief's Step-4 draft targeted
    // pin[2], which is tied 5× in this bake — see task-10 report.)
    const pin = live.pinned[3]!;
    pin.ranks[argMinRank(pin.ranks)] += 1;
    // REASSIGN a fresh top-2-swapped array (do not swap in place). reviveRun
    // copies the typed arrays (topKLogits/topKProbs/ranks) but passes the plain
    // `topKIds`/`topKTexts` through BY REFERENCE (revive.ts:84,87), so both
    // frame() calls alias the singleton imported JSON's cell arrays. An in-place
    // `[a[0],a[1]]=[a[1],a[0]]` would mutate the array the baked side also reads,
    // so the two would still agree. A fresh array (as test 3 does via .map)
    // perturbs only `live`. (The brief's Step-4 draft swapped in place.)
    const c = live.cells[0]!;
    c.topKIds = [c.topKIds[1]!, c.topKIds[0]!, ...c.topKIds.slice(2)];

    const v = compareToBakedFrame(live, frame());
    expect(v.ok).toBe(true);
    expect(v.worstPinDelta).toBe(1); // the perturbation was seen…
    expect(v.topOneAgreement).toBeLessThan(1); // …and so was the swap
  });

  // The direction that actually matters. A test that can only pass proves nothing.
  it('FAILS on a corrupted pack (the f16 storage class)', () => {
    const live = frame();
    for (const c of live.cells) c.topKIds = c.topKIds.map((id) => (id + 31337) % 248320);
    const v = compareToBakedFrame(live, frame());
    expect(v.ok).toBe(false);
    expect(v.topOneAgreement).toBeLessThan(0.9);
  });

  it('FAILS when a pinned concept moves by orders of magnitude', () => {
    const live = frame();
    live.pinned[0]!.ranks = live.pinned[0]!.ranks.map(() => 900) as never;
    expect(compareToBakedFrame(live, frame()).ok).toBe(false);
  });
});

// A trust gate must reject a MIS-SHAPED envelope outright — the fuzzy metrics
// assume index alignment, so a truncated frame still scores ~0.99 and swapped
// pins can slip past min-over-ranks. The structural gate closes that.
describe('self-test structural gate', () => {
  it('FAILS a frame truncated by one cell (would otherwise score ~0.99)', () => {
    const live = frame();
    live.cells = live.cells.slice(0, live.cells.length - 1);
    const v = compareToBakedFrame(live, frame());
    expect(v.ok).toBe(false);
    expect(v.reason).toContain('structural mismatch');
  });

  it('FAILS when pin identities/order differ (swapped tracks)', () => {
    const live = frame();
    [live.pinned[0], live.pinned[1]] = [live.pinned[1]!, live.pinned[0]!];
    const v = compareToBakedFrame(live, frame());
    expect(v.ok).toBe(false);
    expect(v.reason).toContain('structural mismatch');
  });

  it('FAILS when promptLen or topK disagree', () => {
    const a = frame();
    a.promptLen -= 1;
    expect(compareToBakedFrame(a, frame()).ok).toBe(false);
    const b = frame();
    b.topK += 1;
    expect(compareToBakedFrame(b, frame()).ok).toBe(false);
  });

  it('FAILS when the pin count differs', () => {
    const live = frame();
    live.pinned = live.pinned.slice(0, live.pinned.length - 1);
    expect(compareToBakedFrame(live, frame()).ok).toBe(false);
  });

  // A within-layer cell permutation keeps counts, pins, layers, promptLen and
  // topK aligned, so the fuzzy top-1 metric still scores ~0.98 — only the per-cell
  // (layer, position) identity check catches it. reviveRun makes fresh cell
  // objects, so swapping live array elements does not alias the baked side (F1b).
  it('FAILS when two cells are spatially swapped (would otherwise score ~0.98)', () => {
    const live = frame();
    [live.cells[0], live.cells[1]] = [live.cells[1]!, live.cells[0]!];
    const v = compareToBakedFrame(live, frame());
    expect(v.ok).toBe(false);
    expect(v.reason).toContain('structural mismatch');
  });
});
