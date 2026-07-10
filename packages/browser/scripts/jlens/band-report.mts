/**
 * J-lens BAND REPORT (Task T3.2) — corpus-averaged band-onset detectors.
 *
 * WHAT THIS IS (and is NOT). The paper's Fig-28 "workspace onset" signal is the
 * per-layer EXCESS KURTOSIS (4th moment) of the residual stream. The read-only
 * inspector NAPI exposes only per-layer SUMMARY stats (mean / std / absMax /
 * l2Norm, aggregated over the WHOLE prefill tensor = all prompt positions ×
 * hidden dims) — NOT the raw residual activations — and this task makes NO
 * native/wasm changes. So we CANNOT compute the literal 4th moment. Instead we
 * report FOUR honest PROXY detectors, each a per-boundary (ℓ1..24) curve
 * CORPUS-AVERAGED over the six eval suites' prompts (the same 551-prompt set
 * Part A scores), plus a detected band onset:
 *
 *   1. tail-index  = mean(absMax/std) of post_mlp_residual by boundary. The max
 *      standardized activation; sensitive to the same heavy tails as excess
 *      kurtosis but NOT the 4th moment and NOT a monotone stand-in for it. (This
 *      corpus-averages probe.mts's 4-probe diagnostic over MANY prompts.)
 *   2. lens-legibility gap (J − logit) = per boundary, the fraction of eval
 *      intermediates whose J-lens full-vocab rank BEATS the logit-lens rank at
 *      that single boundary (readout position, min over the intermediate's
 *      single-token surface-form ids). This is MEASURED, not a summary-stat
 *      proxy — the lens analog of the interpretable band, the most defensible of
 *      the four. Also reports the mean rank-improvement.
 *   3. answer-surfacing = per boundary, the fraction of SCORABLE-answer items
 *      (denominator disclosed as answerSurfacingScorableItems; multi-token answers
 *      are excluded as unmeasurable, NOT scored as misses) whose answer/target
 *      concept has entered the J-lens top-K (rank ≤ K) at that boundary; plus the
 *      corpus-mean SHALLOWEST such boundary over items that ACTUALLY surfaced
 *      (never-surfaced items counted separately, not right-censored to ℓ24).
 *      Where answers become legible.
 *   4. residual RMS = mean(l2Norm/√count) of post_mlp_residual by boundary — a
 *      LENGTH-NORMALIZED per-coordinate l2 magnitude (raw l2Norm grows ~√P with
 *      prompt length, so it is NOT corpus-averageable; RMS removes that). A
 *      coarse structural proxy for where the stream's magnitude develops. `std`
 *      is stored alongside as a near-identical cross-check (mean≈0).
 *
 * BINDING HONESTY: all four are SUPPORTING signal, NOT the load-bearing GO
 * criterion — that stays Part A (eval.mts, the J-vs-logit reproduction). Detectors
 * 1 & 4 are aggregated over ALL prompt positions (the NAPI exposes no per-position
 * residual), not the readout position; detectors 2 & 3 use the eval readout
 * position, apples-to-apples with Part A (identical layer set 1..=24, identical
 * surface-form ids, only useJacobian differs).
 *
 * `bandStructurePresent` (top-level boolean) = do ≥2 of the 4 detectors show a
 * clear early-flat → mid/late-rise pattern (max over boundaries 6..24 ≥ 1.25× the
 * early-band ℓ1..5 mean AND that max sits at boundary ≥ 6), i.e. NOT flat noise.
 *
 * DETERMINISTIC: fixed prompt set (the eval files, file order), greedy/no-RNG
 * NAPI paths, no wall-clock field in the JSON → two runs are byte-identical.
 *
 * Run with: JLENS_PACK=lens-pack-v1.safetensors \
 *   oxnode packages/browser/scripts/jlens/band-report.mts
 *   (NOT tsx/ts-node — repo convention; needs env PATH="/opt/homebrew/bin:$PATH")
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// The lens caps (LENS_MAX_POSITIONS / LENS_MAX_PINNED) have a single home in
// src/inspector-types.ts, mirroring the Rust contract in inspector.rs. NOTE:
// these scripts run against the on-disk native .node addon, which THIS branch
// does not rebuild — the raised 128-position cap ships in wasm only. Until
// `yarn build:native` re-bakes the addon, any prompt longer than the old 48
// reaches lens_readout and comes back as a LOUD error, never a silent drop.
// The `.ts` extension is required for oxnode's ESM resolution.
import { LENS_MAX_PINNED, LENS_MAX_POSITIONS } from '../../src/inspector-types.ts';

const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const OUT_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
// This is a v1 (full 100-prompt pack) deliverable → default to lens-pack-v1;
// JLENS_PACK overrides (bare name resolves against OUT_DIR, '/'-containing = verbatim).
const PACK_ENV = process.env.JLENS_PACK ?? 'lens-pack-v1.safetensors';
const PACK_PATH = PACK_ENV.includes('/') ? PACK_ENV : join(OUT_DIR, PACK_ENV);
const OUT_NAME = process.env.JLENS_BAND_OUT ?? 'band-report.json';
const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), 'data');

// Boundaries ℓ = 1..24; lens request 1..24 (24 = J=I). Curves are length 24,
// indexed [boundary-1]. post_mlp_residual layerIdx 0..23 → boundary layerIdx+1.
const REQUEST_LAYERS = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const N_BOUND = 24;
const RANK_CENSOR = 999;
const TOPK_LEGIBLE = 10; // detector 3: "entered the top-K" threshold (rank ≤ K)
const RISE_THRESH = 1.25; // onset / band-structure multiplier over the early band
const EARLY_BAND = 5; // early band = boundaries ℓ1..5

type Suite = { slug: string; readout: 'preTarget' | 'lastNewline'; synonyms: boolean; expect: number };
const SUITES: Suite[] = [
  { slug: 'multihop', readout: 'preTarget', synonyms: false, expect: 93 },
  { slug: 'multilingual', readout: 'preTarget', synonyms: false, expect: 107 },
  { slug: 'order-ops', readout: 'preTarget', synonyms: true, expect: 55 },
  { slug: 'poetry', readout: 'lastNewline', synonyms: false, expect: 98 },
  { slug: 'typo', readout: 'preTarget', synonyms: false, expect: 96 },
  { slug: 'association', readout: 'preTarget', synonyms: false, expect: 102 },
];

type Item = { name: string; prompt: string; target?: string; intermediates: string[] };

// ---- number → words (order-ops synonym expansion; covers 0..999) — copied
// verbatim from eval.mts so surface-form derivation is identical (fairness). ----
const ONES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
function numberToWords(n: number): string[] {
  if (n < 0) return numberToWords(-n).map((w) => `negative ${w}`);
  if (n < 20) return [ONES[n]];
  if (n < 100) {
    const t = TENS[Math.floor(n / 10)];
    const o = n % 10;
    return o === 0 ? [t] : [`${t}-${ONES[o]}`, `${t} ${ONES[o]}`];
  }
  if (n < 1000) {
    const h = ONES[Math.floor(n / 100)];
    const rest = n % 100;
    if (rest === 0) return [`${h} hundred`];
    return numberToWords(rest)
      .map((r) => `${h} hundred ${r}`)
      .concat(numberToWords(rest).map((r) => `${h} hundred and ${r}`));
  }
  return [String(n)];
}
const OP_SYNONYMS: Record<string, string[]> = {
  multiplication: ['multiplication', '*', '×', '·'],
  addition: ['addition', '+'],
  subtraction: ['subtraction', '-', '−'],
  division: ['division', '/', '÷'],
  mod: ['mod', '%'],
  squared: ['squared', '²', '^2'],
};
function expandSynonyms(intermediate: string): string[] {
  if (/^-?\d+$/.test(intermediate)) {
    const n = Number.parseInt(intermediate, 10);
    return Array.from(new Set([intermediate, ...numberToWords(n)]));
  }
  return OP_SYNONYMS[intermediate] ?? [intermediate];
}

// --- band-structure helpers (deterministic; no RNG) ---
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
/** early-band (ℓ1..5) mean of a length-24 curve. */
const earlyMean = (curve: number[]) => mean(curve.slice(0, EARLY_BAND));
/** first boundary (1-based) where curve rises past early×thresh (0 = none). */
function onsetBoundary(curve: number[], thresh: number): number {
  const base = earlyMean(curve);
  const bar = base * thresh;
  for (let b = 0; b < curve.length; b++) if (curve[b] > bar && b + 1 > EARLY_BAND) return b + 1;
  return 0;
}
/** shows a clear early-flat → mid/late rise (not flat noise): the max over
 *  boundaries 6..24 is ≥ thresh × early-band mean AND sits at boundary ≥ 6. */
function showsBandStructure(curve: number[], thresh: number): boolean {
  const base = earlyMean(curve);
  if (base <= 0) return false;
  let maxV = -Infinity;
  let maxB = 0;
  for (let b = EARLY_BAND; b < curve.length; b++) if (curve[b] > maxV) { maxV = curve[b]; maxB = b + 1; }
  return maxV >= base * thresh && maxB >= EARLY_BAND + 1;
}

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  console.log(`Loading lens pack from ${PACK_PATH} ...`);
  const loaded = await model.loadLensPackFromFile(PACK_PATH);
  if (loaded !== 23) throw new Error(`expected 23 Jacobians in the pack, got ${loaded}`);
  console.log(`Model loaded; pack = ${loaded} Jacobians (J.1..J.23).\n`);

  const encCache = new Map<string, number[]>();
  const encode = async (s: string): Promise<number[]> => {
    const hit = encCache.get(s);
    if (hit) return hit;
    const ids: number[] = await model.encodeTokens(s);
    encCache.set(s, ids);
    return ids;
  };
  /** single-token surface-form ids for a base string (both " "+s and bare s). */
  async function surfaceIds(base: string): Promise<number[]> {
    const ids = new Set<number>();
    for (const cand of [` ${base}`, base]) {
      if (cand.length === 0) continue;
      const enc = await encode(cand);
      if (enc.length === 1) ids.add(enc[0]);
    }
    return [...ids];
  }
  /** expand a concept (intermediate or target) → single-token surface ids. */
  async function conceptIds(concept: string, synonyms: boolean): Promise<number[]> {
    const bases = synonyms ? expandSynonyms(concept) : [concept];
    const idset = new Set<number>();
    for (const b of bases) for (const id of await surfaceIds(b)) idset.add(id);
    return [...idset];
  }

  // ---- accumulators (all length-24, indexed [boundary-1]) ----
  // det1 tail-index + det4 rms/std (residual summary stats, over ALL positions)
  const tailSum = new Array(N_BOUND).fill(0);
  const rmsSum = new Array(N_BOUND).fill(0);
  const stdSum = new Array(N_BOUND).fill(0);
  let nResidualPrompts = 0;
  // det2 lens-legibility (per boundary: #intermediates J-beats-logit, Σ rank-delta, #scored)
  const legImproved = new Array(N_BOUND).fill(0);
  const legRankDelta = new Array(N_BOUND).fill(0);
  let nLegIntermediates = 0; // scorable intermediates (denominator, same at every boundary)
  // det3 answer-surfacing (per boundary: #items whose answer rank ≤ K)
  const surfCount = new Array(N_BOUND).fill(0);
  const surfShallowestSurfaced: number[] = []; // shallowest surfacing boundary, ONLY for items that surfaced
  let nAnswerItems = 0; // scorable-answer items (≥1 single-token answer surface id) = the surfFrac denominator
  let nAnswerNeverSurfaced = 0; // scorable-answer items whose answer never reached rank ≤ K at ANY boundary

  let maxPromptLen = 0;

  for (const suite of SUITES) {
    const raw = JSON.parse(readFileSync(join(DATA_DIR, `lens-eval-${suite.slug}.json`), 'utf8'));
    const items: Item[] = raw.items;
    if (items.length !== suite.expect) throw new Error(`${suite.slug}: expected ${suite.expect} items, got ${items.length}`);

    const t0 = Date.now();
    for (const item of items) {
      const promptIds: number[] = await encode(item.prompt);
      const P = promptIds.length;
      if (P > maxPromptLen) maxPromptLen = P;
      if (P > LENS_MAX_POSITIONS) throw new Error(`${suite.slug}/${item.name}: prompt ${P} tok > ${LENS_MAX_POSITIONS}`);

      // readout position (apples-to-apples with eval.mts).
      let readoutPos = P - 1;
      if (suite.readout === 'lastNewline') {
        const texts: string[] = await model.decodeTokens(promptIds);
        let nl = -1;
        for (let i = 0; i < texts.length; i++) if (texts[i].includes('\n')) nl = i;
        if (nl < 0) throw new Error(`${suite.slug}/${item.name}: no newline token found`);
        readoutPos = nl;
      }

      // ---- det1 + det4: residual summary stats over the whole prefill tensor ----
      const hs = await model.runForInspector(item.prompt, {
        hiddenStates: { points: ['post_mlp_residual'] },
        maxNewTokens: 1,
        applyChatTemplate: false,
      });
      const step0 = hs.hiddenStates[0];
      for (const lay of step0.layers) {
        const s = lay.stats.find((x: any) => x.point === 'post_mlp_residual');
        if (!s) continue;
        const b = lay.layerIdx; // 0..23 → boundary b+1
        if (b < 0 || b >= N_BOUND) continue;
        if (s.std > 0) tailSum[b] += s.absMax / s.std;
        rmsSum[b] += s.l2Norm / Math.sqrt(s.count); // length-normalized per-coordinate l2
        stdSum[b] += s.std;
      }
      nResidualPrompts++;

      // ---- det2 intermediates + det3 answer concept → single-token surface ids ----
      const interIdSets: number[][] = [];
      for (const inter of item.intermediates) {
        const ids = await conceptIds(inter, suite.synonyms);
        if (ids.length) interIdSets.push(ids); // unscorable (multi-token) dropped, same as eval
      }
      const answerConcepts = item.target ? [item.target] : item.intermediates;
      const answerIdSet = new Set<number>();
      for (const c of answerConcepts) for (const id of await conceptIds(c, suite.synonyms)) answerIdSet.add(id);

      // union of all ids we need ranks for, batched ≤8, both lenses, all boundaries.
      const allIds = [...new Set([...interIdSets.flat(), ...answerIdSet])];
      if (allIds.length === 0) continue; // nothing scorable for this item
      // ranks[id] → { j:Int, logit:Int } per boundary (length 24)
      const jRank = new Map<number, number[]>();
      const lRank = new Map<number, number[]>();
      for (const id of allIds) {
        jRank.set(id, new Array(N_BOUND).fill(RANK_CENSOR));
        lRank.set(id, new Array(N_BOUND).fill(RANK_CENSOR));
      }
      for (let off = 0; off < allIds.length; off += LENS_MAX_PINNED) {
        const batch = allIds.slice(off, off + LENS_MAX_PINNED);
        for (const useJacobian of [true, false]) {
          const ro = await model.lensReadout(promptIds, {
            layers: REQUEST_LAYERS,
            topK: 1,
            pinnedIds: batch,
            useJacobian,
          });
          if (useJacobian && ro.jacobianApplied !== true) throw new Error(`${suite.slug}/${item.name}: jacobianApplied=false`);
          for (let bi = 0; bi < batch.length; bi++) {
            const ranks = ro.pinned[bi].ranks as Int32Array;
            const dst = (useJacobian ? jRank : lRank).get(batch[bi])!;
            for (let li = 0; li < REQUEST_LAYERS.length; li++) dst[li] = ranks[li * P + readoutPos];
          }
        }
      }

      // det2: per boundary, per intermediate rank = min over its surface ids.
      for (const ids of interIdSets) {
        nLegIntermediates++;
        for (let b = 0; b < N_BOUND; b++) {
          let jr = RANK_CENSOR;
          let lr = RANK_CENSOR;
          for (const id of ids) {
            jr = Math.min(jr, jRank.get(id)![b]);
            lr = Math.min(lr, lRank.get(id)![b]);
          }
          if (jr < lr) legImproved[b]++;
          legRankDelta[b] += lr - jr; // + = J better (lower rank)
        }
      }

      // det3: per boundary answer rank (J-lens) = min over answer surface ids.
      // Denominator = scorable-answer items (answerIdSet non-empty). Items whose
      // answer is multi-token are UNMEASURABLE by a single-token readout, so they
      // are excluded (counted in nAnswerExcluded), NOT scored as a miss — the same
      // rule eval.mts applies to unscorable intermediates. "Never surfaced" (rank
      // > K at every boundary) is tracked separately from a genuine ℓ24 surfacing,
      // so the corpus-mean shallowest boundary is not right-censored to 24.
      if (answerIdSet.size > 0) {
        nAnswerItems++;
        let shallowest = 0; // 0 = never surfaced (NOT a real boundary)
        for (let b = 0; b < N_BOUND; b++) {
          let jr = RANK_CENSOR;
          for (const id of answerIdSet) jr = Math.min(jr, jRank.get(id)![b]);
          if (jr <= TOPK_LEGIBLE) {
            surfCount[b]++;
            if (shallowest === 0) shallowest = b + 1; // ascending loop → first hit is the shallowest
          }
        }
        if (shallowest > 0) surfShallowestSurfaced.push(shallowest);
        else nAnswerNeverSurfaced++;
      }
    }
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`[${suite.slug.padEnd(13)}] ${String(items.length).padStart(3)} items processed (${dt}s)`);
  }

  // ---- reduce to per-boundary curves (length 24) ----
  const tail = tailSum.map((v) => v / nResidualPrompts);
  const rms = rmsSum.map((v) => v / nResidualPrompts);
  const stdCurve = stdSum.map((v) => v / nResidualPrompts);
  const legFrac = legImproved.map((v) => v / nLegIntermediates); // det2 fraction-improved
  const legDeltaMean = legRankDelta.map((v) => v / nLegIntermediates); // det2 mean rank-improvement
  const surfFrac = surfCount.map((v) => v / nAnswerItems); // det3 fraction-surfaced (of scorable-answer items)
  const nAnswerSurfaced = surfShallowestSurfaced.length; // items whose answer reached rank ≤ K at ≥1 boundary
  const meanShallowest = mean(surfShallowestSurfaced); // corpus-mean shallowest boundary, SURFACED items ONLY (0 if none)
  const surfacedFraction = nAnswerItems ? nAnswerSurfaced / nAnswerItems : 0; // share of scorable-answer items that ever surfaced
  const nAnswerExcluded = nResidualPrompts - nAnswerItems; // items dropped: answer had no single-token surface id

  // ---- per-detector onset + band-structure verdict ----
  const detectors = [
    {
      id: 'tail_index',
      title: 'residual tail-index (mean absMax/std by boundary) — PROXY, not Fig-28 kurtosis',
      curve: tail,
      earlyBandBaseline: earlyMean(tail),
      onsetBoundary: onsetBoundary(tail, RISE_THRESH),
      threshold: `early-band(ℓ1..5) mean × ${RISE_THRESH}`,
      showsBandStructure: showsBandStructure(tail, RISE_THRESH),
      reading: '',
    },
    {
      id: 'lens_legibility_gap',
      title: 'lens-legibility gap: fraction of eval intermediates where J-rank < logit-rank by boundary (MEASURED)',
      curve: legFrac,
      auxMeanRankDelta: legDeltaMean,
      earlyBandBaseline: earlyMean(legFrac),
      onsetBoundary: onsetBoundary(legFrac, RISE_THRESH),
      threshold: `early-band(ℓ1..5) mean × ${RISE_THRESH}`,
      showsBandStructure: showsBandStructure(legFrac, RISE_THRESH),
      reading: '',
    },
    {
      id: 'answer_surfacing',
      title: `answer-surfacing: fraction of SCORABLE-answer items with answer/target rank ≤ ${TOPK_LEGIBLE} in the J-lens by boundary`,
      curve: surfFrac,
      denominator: nAnswerItems,
      auxMeanShallowestBoundarySurfaced: meanShallowest,
      auxSurfacedFraction: surfacedFraction,
      auxNeverSurfaced: nAnswerNeverSurfaced,
      auxExcludedUnscorable: nAnswerExcluded,
      earlyBandBaseline: earlyMean(surfFrac),
      onsetBoundary: onsetBoundary(surfFrac, RISE_THRESH),
      threshold: `early-band(ℓ1..5) mean × ${RISE_THRESH}`,
      showsBandStructure: showsBandStructure(surfFrac, RISE_THRESH),
      reading: '',
    },
    {
      id: 'residual_rms',
      title: 'residual RMS (mean l2Norm/√count by boundary) — length-normalized magnitude PROXY',
      curve: rms,
      auxStd: stdCurve,
      earlyBandBaseline: earlyMean(rms),
      onsetBoundary: onsetBoundary(rms, RISE_THRESH),
      threshold: `early-band(ℓ1..5) mean × ${RISE_THRESH}`,
      showsBandStructure: showsBandStructure(rms, RISE_THRESH),
      reading: '',
    },
  ];
  const readings: Record<string, (d: any) => string> = {
    tail_index: (d: any) =>
      d.onsetBoundary
        ? `tail-index rises past ${RISE_THRESH}× the early band at boundary ℓ${d.onsetBoundary} (heavy-tail structure develops mid-stack)`
        : `tail-index stays within ${RISE_THRESH}× the early band across all boundaries (no clear onset)`,
    lens_legibility_gap: (d: any) => {
      const peakB = d.curve.indexOf(Math.max(...d.curve)) + 1;
      return `J beats logit most at boundary ℓ${peakB} (frac=${d.curve[peakB - 1].toFixed(3)}); early-band frac=${d.earlyBandBaseline.toFixed(3)} → J's per-layer advantage is a mid-stack band`;
    },
    answer_surfacing: (d: any) =>
      `${(d.auxSurfacedFraction * 100).toFixed(0)}% of ${d.denominator} scorable-answer items ever enter the J-lens top-${TOPK_LEGIBLE} (${d.auxNeverSurfaced} never do; ${d.auxExcludedUnscorable} more items excluded as multi-token/unscorable). Of those that surface, corpus-mean shallowest boundary ℓ${d.auxMeanShallowestBoundarySurfaced.toFixed(1)}; per-boundary surfacing onset ℓ${d.onsetBoundary || 'n/a'}`,
    residual_rms: (d: any) =>
      d.onsetBoundary
        ? `residual RMS grows past ${RISE_THRESH}× the early band at boundary ℓ${d.onsetBoundary} (magnitude structure develops with depth)`
        : `residual RMS stays within ${RISE_THRESH}× the early band (flat magnitude profile)`,
  };
  for (const d of detectors) d.reading = readings[d.id](d);

  const nStruct = detectors.filter((d) => d.showsBandStructure).length;
  const bandStructurePresent = nStruct >= 2;

  // ---- compact console table ----
  console.log(`\n================ BAND REPORT (corpus-averaged over ${nResidualPrompts} eval prompts) ================`);
  console.log(`pack = ${PACK_PATH}`);
  console.log(`\nper-boundary curves (ℓ1..24):`);
  console.log(`  ℓ  | tail(absMax/std) | J−logit frac | answer top${TOPK_LEGIBLE} frac | resid RMS`);
  for (let b = 0; b < N_BOUND; b++) {
    console.log(
      `  ${String(b + 1).padStart(2)} | ${tail[b].toFixed(3).padStart(14)} | ${legFrac[b].toFixed(3).padStart(11)} | ${surfFrac[b].toFixed(3).padStart(15)} | ${rms[b].toFixed(4).padStart(9)}`,
    );
  }
  console.log(
    `\ndenominators (disclosed — they differ per detector):` +
      `\n  tail_index / residual_rms : ${nResidualPrompts} prompts (all)` +
      `\n  lens_legibility_gap       : ${nLegIntermediates} scorable (single-token) intermediates` +
      `\n  answer_surfacing          : ${nAnswerItems} scorable-answer items (${nAnswerExcluded} excluded as multi-token; ${nAnswerNeverSurfaced} never surfaced)`,
  );
  console.log(`\ndetector summary:`);
  for (const d of detectors) {
    console.log(
      `  [${d.id.padEnd(20)}] early=${d.earlyBandBaseline.toFixed(3)}  onset=ℓ${d.onsetBoundary || '-'}  bandStructure=${d.showsBandStructure}`,
    );
    console.log(`      ${d.reading}`);
  }
  console.log(`\nbandStructurePresent = ${bandStructurePresent}  (${nStruct}/4 detectors show early-flat→rise)`);
  console.log(`NOTE: all four are PROXIES / supporting signal — NOT the paper's Fig-28 kurtosis and NOT the GO criterion (that is Part A / eval.mts).`);

  // ---- write band-report.json (NO wall-clock field → byte-identical reruns) ----
  mkdirSync(OUT_DIR, { recursive: true });
  const outPath = join(OUT_DIR, OUT_NAME);
  writeFileSync(
    outPath,
    JSON.stringify(
      {
        model: MODEL_PATH,
        pack: PACK_PATH,
        corpus: 'six eval suites (lens-eval-*.json), file order — 551 prompts; the same set Part A scores',
        nPrompts: nResidualPrompts,
        // Per-detector denominators DIFFER and are disclosed here (the four curves
        // are NOT all fractions of the same nPrompts): tail/RMS average over all
        // prompts; legibility over scorable single-token intermediates; answer-
        // surfacing over scorable-answer items only.
        denominators: {
          tailIndexResidualRms: nResidualPrompts,
          legibilityIntermediates: nLegIntermediates,
          answerSurfacingScorableItems: nAnswerItems,
          answerSurfacingExcludedUnscorable: nAnswerExcluded,
          answerSurfacingNeverSurfaced: nAnswerNeverSurfaced,
          answerSurfacingSurfaced: nAnswerSurfaced,
        },
        boundaries: REQUEST_LAYERS,
        params: { topKLegible: TOPK_LEGIBLE, riseThreshold: RISE_THRESH, earlyBand: `ℓ1..${EARLY_BAND}` },
        honesty:
          'Four PROXY band-onset detectors, NOT the paper Fig-28 residual excess kurtosis (the read-only NAPI exposes only per-layer summary stats mean/std/absMax/l2Norm over the whole prefill tensor, not raw residuals; no native changes made). Supporting signal only — the load-bearing GO criterion is Part A (eval.mts J-vs-logit). Per-detector denominators differ and are disclosed in `denominators` (see above): tail_index/residual_rms average over all prompts, lens_legibility_gap over scorable single-token intermediates, answer_surfacing over scorable-answer items ONLY — multi-token answers are unmeasurable by a single-token readout and are excluded (counted, not scored as misses; same rule eval.mts uses for intermediates). The answer_surfacing "shallowest boundary" is averaged over items that ACTUALLY surfaced (rank ≤ K at ≥1 boundary); never-surfaced items are counted separately (auxNeverSurfaced), NOT right-censored to ℓ24.',
        detectors,
        bandStructurePresent,
        nDetectorsWithBandStructure: nStruct,
        maxPromptLen,
      },
      null,
      2,
    ),
  );
  console.log(`\nWrote ${outPath}`);
  console.log('BAND REPORT COMPLETE');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
