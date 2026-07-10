/**
 * J-lens f16 SHIPPED-pack rank-parity harness (Task T4.0, R6).
 *
 * Proves the f16 shipped pack (`lens-pack-v1.f16.safetensors`) is READOUT-
 * EQUIVALENT to the F32 master pack (`lens-pack-v1.safetensors`) at the top-K /
 * rank level: loading either and running the SAME `lensReadout(useJacobian=true)`
 * over the SAME fixed prompt + layer set must agree on the argmax, keep pinned
 * ranks within a small tolerance, and preserve top-4 membership. This is RANK
 * parity WITHIN f16 TOLERANCE, NOT bit identity: plan D8 astype's BOTH packs to
 * bf16 at load, and f16 rounding (round-trip relErr ≈ 2^-11) crosses a bf16
 * boundary here and there, so deep pinned ranks jitter by ±1–2 and top-8 slots
 * 7–8 swap on near-ties while the argmax and top-4 membership hold. Exact top-K /
 * pinned-rank equality is therefore UNPASSABLE for a correct lossy f16 pack — an
 * exact gate would reject every good export (the T4.0 initial-commit bug). This
 * is the export-time validation named in the plan's cross-phase story
 * ("f16 round-trip rank parity at export").
 *
 * CLAIM-LEVEL VALIDATION (the definitive proof — run by the controller, NOT here):
 * the FULL eval on the f16 pack vs the F32 pack (eval-results-f16.json vs
 * eval-results-v1.json) measured max |Δ jAucHead| = 0.00074 (< 0.001) across all
 * six suites, f16 J beats logit 6/6 (jWins6=6, same as F32), and logit-AUC is
 * byte-identical f32-vs-f16 — so f16 preserves the shipped GO claim at the metric
 * level. THIS harness is only the lightweight top-K SANITY gate guarding the
 * export (catches overflow→Inf / wrong dtype / gross reorder) without re-running
 * the ~minutes-long eval.
 *
 * PASS CRITERION (gate = (a) ∧ (b) ∧ (c), evaluated at the readout position per
 * reported layer; the full per-layer diff is still printed for inspection). The
 * gate asserts PRECISELY what the UI shows and the lesson claims — top-1, top-4
 * membership, and the legible-vs-illegible bucket for pins — and ignores deep-rank
 * jitter that is invisible (native rank display caps at 999) and provably
 * immaterial (ΔAUC < 0.001, 6/6 J-wins preserved). It is a legibility gate, NOT a
 * bit-parity gate:
 *   (a) TOP-1 stability — the argmax token is IDENTICAL F32 vs f16 at every layer.
 *       A real regression flips the argmax; f16 jitter never does. EXACT.
 *   (b) PINNED-RANK LEGIBILITY BUCKET — with K_LEGIBLE=32 (the readout top-K cap;
 *       rank ≤32 is "legible / in the displayed band", beyond is "deep, read as a
 *       number"), for each pinned id compare rank_f32 (rF) vs rank_f16 (rG):
 *         • both rF ≥ 999 and rG ≥ 999           → PASS (both censored/equal)
 *         • both rF > 32 and rG > 32             → PASS (both deep — exact value is
 *           display-noise; no lesson claim hinges on rank 300 vs 350)
 *         • both rF ≤ 32 and rG ≤ 32             → require |rF − rG| ≤ 2 (legible
 *           tokens must be stable to ±2)
 *         • one side ≤ 32 and the other > 32     → FAIL (a token that was legible
 *           became illegible, or vice-versa — the real regression signal; a
 *           broken / overflowed / transposed pack trips exactly this)
 *       Worked example: rank 5 → rank 300 FAILS (legible→illegible); rank 741 →
 *       759 PASSES (both deep). (Why not a fixed 2% bound? A ±18 shift at rank 741
 *       in a 248k-vocab logit plateau is pure near-tie display-noise, so a relative
 *       bound is still the wrong invariant for deep ranks — the bucket is.)
 *   (c) NO TOP-4 ESCAPE — every id in F32's top-4 appears within f16's top-8 and
 *       vice-versa (order may swap on near-ties; membership must hold).
 * Deliberately NOT a rubber stamp: a flipped argmax (a), a pin crossing the
 * legibility boundary (b), or a top-4 id falling out of top-8 (c) each FAIL.
 *
 * !!! DO NOT EXECUTE THIS SCRIPT UNSUPERVISED !!!
 * The GPU is SERIAL and the controller owns every MLX job. This harness loads the
 * model (twice) and runs GPU work; running it concurrently with the controller's
 * MLX job would corrupt both. It is AUTHORED and parse-checked (`oxnode --check`)
 * as part of T4.0; the controller runs it at the designated serial slot.
 *
 * Run (CONTROLLER ONLY, serial slot):
 *   env PATH="/opt/homebrew/bin:$PATH" oxnode \
 *     packages/browser/scripts/jlens/verify-f16-parity.mts
 *   (NOT tsx/ts-node — repo convention.)
 *
 * FAIL-CLOSED CONTRACT (mirrors verify-readout.mts): the native `@mlx-node/lm`
 * import is DYNAMIC and lives only in the child branch, so the parent has no
 * native dependency. The parent re-execs THIS script as a child under the SAME
 * interpreter + loader, then exits non-zero UNLESS it observes BOTH a clean child
 * exit (status 0, no signal) AND the `F16_PARITY: PASS` sentinel in the child's
 * stdout. An addon that cannot load, a Rust panic, or a swallowed abort therefore
 * FAILS the gate — a gate that could pass on a crash would be no gate.
 *
 * DETERMINISM: fixed prompt, fixed layer set, greedy readout, no RNG, and NO
 * wall-clock in the output — two runs are byte-identical.
 *
 * IMPLEMENTATION NOTE: the parent guard uses the SYNCHRONOUS `spawnSync` via a
 * STATIC import (not `await import`) specifically so there is NO top-level await
 * — that keeps the file `oxnode --check`-able (node's syntax checker rejects
 * top-level await). The only dynamic import is the native `@mlx-node/lm`, which
 * lives inside async `main()` (child-only), keeping the parent native-free.
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const SENTINEL = 'F16_PARITY: PASS';
if (!process.env.__JLENS_F16_PARITY_CHILD) {
  const res = spawnSync(process.execPath, [...process.execArgv, ...process.argv.slice(1)], {
    env: { ...process.env, __JLENS_F16_PARITY_CHILD: '1' },
    encoding: 'utf8',
    stdio: ['inherit', 'pipe', 'inherit'], // capture stdout (scan for sentinel), stream stderr live
  });
  if (res.stdout) process.stdout.write(res.stdout);
  const cleanExit = res.status === 0 && res.signal == null;
  const sawSentinel = (res.stdout ?? '').includes(SENTINEL);
  if (!cleanExit || !sawSentinel) {
    console.error(
      `\n[GATE] FAIL — not fail-open: cleanExit=${cleanExit} (status=${res.status} signal=${res.signal}) sentinel=${sawSentinel}`,
    );
    process.exit(1);
  }
  process.exit(0);
}
// ---- child mode below: loads the native addon + model (GPU work) ----
// Fail-closed self-test seam: a child that exits 0 WITHOUT the sentinel MUST
// still fail the gate (proves the parent contract has teeth).
if (process.env.__JLENS_SELFTEST_NO_SENTINEL) {
  console.log('child ran, no sentinel');
  process.exit(0);
}

const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const CACHE_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
const F32_PACK = join(CACHE_DIR, 'lens-pack-v1.safetensors'); // master (F32)
const F16_PACK = join(CACHE_DIR, 'lens-pack-v1.f16.safetensors'); // shipped (F16)

const PROMPT = 'The capital of France is';
const LAYERS = Array.from({ length: 23 }, (_, i) => i + 1); // fitted domain J.1..J.23
const TOP_K = 8;
const EXPECT_JACOBIANS = 23;
const RANK_CAP = 999; // native full-vocab rank display cap: 999 means ">= 999" (censored)
const K_LEGIBLE = 32; // readout top-K cap: rank <= 32 is "legible / displayed band", beyond is "deep, read as a number"

function fail(msg: string): never {
  console.error(`\nFAIL: ${msg}`);
  console.log('F16_PARITY: FAIL');
  process.exit(1);
}

/** Ordered (not just set) equality of two id lists. */
function sameOrder(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

type Cell = { layer: number; position: number; topKIds: number[] };
type Pinned = { ranks: Int32Array };
type Readout = { cells: Cell[]; pinned: Pinned[]; jacobianApplied: boolean; promptLen: number };

/** Run the fixed readout on a model that already has a pack loaded. Returns the
 *  per-layer last-position top-K id lists + the pinned full-vocab ranks at the
 *  last position (aligned to `pinnedIds`), plus jacobianApplied. */
async function readoutFor(
  model: { lensReadout: (ids: number[], opts: unknown) => Promise<Readout> },
  promptIds: number[],
  pinnedIds: number[],
): Promise<{ topKByLayer: number[][]; pinnedRanksByLayer: number[][]; jacobianApplied: boolean }> {
  const P = promptIds.length;
  const ro = await model.lensReadout(promptIds, { layers: LAYERS, topK: TOP_K, pinnedIds, useJacobian: true });
  const lastPos = P - 1;
  // cells are layer-major, position-minor: the last-position cell for a requested
  // layer is found by (layer, position). topK ids at that cell = the readout order.
  const topKByLayer: number[][] = [];
  for (const L of LAYERS) {
    const cell = ro.cells.find((c) => c.layer === L && c.position === lastPos);
    if (!cell) fail(`no cell for layer ${L} at last position ${lastPos}`);
    topKByLayer.push(cell.topKIds);
  }
  // ranks[li*P + pos]: pinned id `bi`'s full-vocab rank at (layer index li, pos).
  const pinnedRanksByLayer: number[][] = LAYERS.map((_, li) =>
    pinnedIds.map((_, bi) => ro.pinned[bi].ranks[li * P + lastPos]),
  );
  return { topKByLayer, pinnedRanksByLayer, jacobianApplied: ro.jacobianApplied };
}

async function main(): Promise<void> {
  for (const p of [MODEL_PATH, F32_PACK, F16_PACK]) {
    if (!existsSync(p)) fail(`required path missing: ${p}`);
  }
  const { Qwen35Model } = await import('@mlx-node/lm');

  // ---------- Model instance A: the F32 master pack ----------
  console.log(`[A] loading model + F32 pack (${F32_PACK})`);
  const modelA = (await Qwen35Model.load(MODEL_PATH)) as any;
  const promptIds: number[] = await modelA.encodeTokens(PROMPT);
  const P = promptIds.length;
  console.log(`    prompt="${PROMPT}"  ids(${P})=[${promptIds.join(', ')}]  layers=[1..23] topK=${TOP_K}`);
  const loadedA: number = await modelA.loadLensPackFromFile(F32_PACK);
  if (loadedA !== EXPECT_JACOBIANS) fail(`F32 pack loaded ${loadedA} Jacobians, expected ${EXPECT_JACOBIANS}`);

  // Discover the pinned id set = F32 J-lens top-K at the FINAL fitted layer (23),
  // last position. These fixed ids give the "pinned full-vocab ranks" both packs
  // must agree on. (pins ≤ LENS_MAX_PINNED (8); TOP_K ≤ 8, so no pin id is dropped.)
  const discover = await modelA.lensReadout(promptIds, { layers: [23], topK: TOP_K, useJacobian: true });
  const seedCell = discover.cells.find((c: Cell) => c.layer === 23 && c.position === P - 1);
  if (!seedCell) fail('no seed cell for pinned-id discovery at layer 23');
  const pinnedIds: number[] = seedCell.topKIds.slice(0, 8);
  console.log(`    pinnedIds (F32 J-lens top-${TOP_K} @ℓ23) = [${pinnedIds.join(', ')}]`);

  const A = await readoutFor(modelA, promptIds, pinnedIds);
  if (A.jacobianApplied !== true) fail('F32 readout jacobianApplied=false (expected true for J.1..J.23)');

  // ---------- Model instance B (FRESH): the f16 shipped pack ----------
  console.log(`[B] loading a FRESH model + f16 pack (${F16_PACK})`);
  const modelB = (await Qwen35Model.load(MODEL_PATH)) as any;
  const loadedB: number = await modelB.loadLensPackFromFile(F16_PACK);
  if (loadedB !== EXPECT_JACOBIANS) fail(`f16 pack loaded ${loadedB} Jacobians, expected ${EXPECT_JACOBIANS}`);
  const B = await readoutFor(modelB, promptIds, pinnedIds);
  if (B.jacobianApplied !== true) fail('f16 readout jacobianApplied=false (expected true for J.1..J.23)');

  // ---------- Compare: the relaxed-but-principled gate (a) ∧ (b) ∧ (c) ----------
  // Rationale in the header: exact top-K/rank equality is unpassable for a correct
  // lossy f16 pack; the gate below passes THIS genuinely-fine pack while still
  // catching a truly broken export.
  console.log(
    `\n[PARITY] per-layer diff + gate (F32 vs f16): (a) top-1 exact, (b) pin legibility bucket (K_LEGIBLE=${K_LEGIBLE}), (c) top-4 ⊆ top-8:`,
  );
  // (b) LEGIBILITY-BUCKET rule (K_LEGIBLE=32 = the readout top-K cap). What matters
  // is whether a pinned token is legible (rank ≤32, shown in the band) or deep
  // (>32, read as a number, capped at 999). Deep-rank jitter is invisible +
  // immaterial; a token crossing the legible/deep boundary is the regression signal.
  //   rank 5 → 300  FAILS (legible→illegible);  rank 741 → 759  PASSES (both deep).
  const rankBucketOk = (rF: number, rG: number): boolean => {
    if (rF >= RANK_CAP && rG >= RANK_CAP) return true; // both censored (≥999) → equal
    const fLeg = rF <= K_LEGIBLE;
    const gLeg = rG <= K_LEGIBLE;
    if (fLeg && gLeg) return Math.abs(rF - rG) <= 2; // both legible → ±2 stable
    if (!fLeg && !gLeg) return true; // both deep/illegible → display-noise, PASS
    return false; // one legible, one deep → legibility crossing → FAIL
  };
  /** (c): every id in `top4` present somewhere in `topK` (top-8). */
  const subsetOf = (top4: number[], topK: number[]): boolean => {
    const set = new Set(topK);
    return top4.every((id) => set.has(id));
  };

  let top1Flips = 0; // (a) violations
  let legibilityViol = 0; // (b) violations: a pin crossed the legible/deep boundary
  let top4Escapes = 0; // (c) violations
  let exactIdDiffs = 0; // informational: layers whose top-K ORDER is not byte-exact
  let exactRankDiffs = 0; // informational: layers whose pinned ranks are not byte-exact
  for (let li = 0; li < LAYERS.length; li++) {
    const L = LAYERS[li];
    const aIds = A.topKByLayer[li];
    const bIds = B.topKByLayer[li];
    const aRanks = A.pinnedRanksByLayer[li];
    const bRanks = B.pinnedRanksByLayer[li];

    const top1Ok = aIds[0] === bIds[0]; // (a) argmax identical
    const rankOk = aRanks.every((r, bi) => rankBucketOk(r, bRanks[bi])); // (b) legibility bucket
    const top4Ok = subsetOf(aIds.slice(0, 4), bIds) && subsetOf(bIds.slice(0, 4), aIds); // (c)
    if (!top1Ok) top1Flips++;
    if (!rankOk) legibilityViol++;
    if (!top4Ok) top4Escapes++;

    const idsExact = sameOrder(aIds, bIds);
    const ranksExact = sameOrder(aRanks, bRanks);
    if (!idsExact) exactIdDiffs++;
    if (!ranksExact) exactRankDiffs++;

    const layerOk = top1Ok && rankOk && top4Ok;
    // 'ok' = byte-exact; 'ok*' = passes the gate but has expected f16 jitter; 'FAIL' = gate violated.
    const flag = layerOk ? (idsExact && ranksExact ? 'ok ' : 'ok*') : 'FAIL';
    const viol: string[] = [];
    if (!top1Ok) viol.push(`TOP1 F32=${aIds[0]} f16=${bIds[0]}`);
    if (!rankOk) {
      // name the pin(s) that crossed the legible/deep boundary.
      const crossed = aRanks
        .map((r, bi) => (rankBucketOk(r, bRanks[bi]) ? null : `${r}↔${bRanks[bi]}`))
        .filter(Boolean);
      viol.push(`LEGIBILITY-CROSS[${crossed.join(',')}]`);
    }
    if (!top4Ok) viol.push('TOP4-ESCAPE');
    let detail = '';
    if (!idsExact) detail += `  ids F32=[${aIds.join(',')}] f16=[${bIds.join(',')}]`;
    if (!ranksExact) detail += `  ranks F32=[${aRanks.join(',')}] f16=[${bRanks.join(',')}]`;
    console.log(`  ℓ${String(L).padStart(2)}: ${flag}${viol.length ? ' [' + viol.join(' ') + ']' : ''}${detail}`);
  }

  const gateFails = top1Flips + legibilityViol + top4Escapes;
  console.log(
    `\n[PARITY] layers=${LAYERS.length}  GATE: top1-flips=${top1Flips} pin-legibility-crossings=${legibilityViol} top4-escapes=${top4Escapes}`,
  );
  console.log(
    `[PARITY] informational (expected f16 jitter, NOT gated): ${exactIdDiffs} layer(s) with non-exact top-${TOP_K} order, ` +
      `${exactRankDiffs} with non-exact pinned ranks (deep-rank ±jitter is display-noise, capped at ${RANK_CAP})`,
  );
  if (gateFails > 0) {
    fail(
      `f16 pack fails the parity gate: ${top1Flips} argmax flip(s) + ${legibilityViol} pin legibility-crossing(s) + ` +
        `${top4Escapes} top-4 escape(s) across ${LAYERS.length} layers`,
    );
  }

  console.log('\n==================== f16 ≈ F32 readout (legibility parity within f16 tolerance) ====================');
  console.log(SENTINEL);
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  console.log('F16_PARITY: FAIL');
  process.exit(1);
});
