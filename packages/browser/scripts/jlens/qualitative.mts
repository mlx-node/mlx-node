/**
 * J-lens QUALITATIVE PASS (Task T3.3) — criterion-(3) evidence.
 *
 * The Phase-3 GO/NO-GO has four criteria; three are decided by the aggregate eval
 * (eval.mts) + band report (band-report.mts). This script decides the fourth,
 * QUALITATIVE criterion: are there ≥2 slices where the J-lens surfaces an
 * INTERPRETABLE intermediate concept in its top-K (rank ≤ K) that the plain logit
 * lens does NOT surface at that same layer? That is the paper's core claim in
 * miniature — the fitted Jacobian makes a mid-stack concept legible that the raw
 * unembedding does not.
 *
 * WHAT IT DOES. Over a small CURATED slice of the six eval suites (the paper's
 * story archetypes — multi-hop facts, arithmetic, associations, translation,
 * poetry line-break), for each scorable intermediate concept it computes, per
 * boundary ℓ, the J-lens rank and the logit-lens rank (min over the concept's
 * single-token surface ids — identical derivation to eval.mts/band-report.mts, so
 * the two lenses are apples-to-apples; only useJacobian differs). It then emits
 * every "J-only-legible" slice: jRank ≤ TOPK_LEGIBLE < lRank. For the shallowest
 * such slice of each item it also dumps the actual J-lens vs logit-lens top-K
 * decoded tokens at that boundary, so the evidence is human-auditable.
 *
 * HONESTY. This is a small, curated, deterministic slice for QUALITATIVE evidence
 * and preset selection — NOT a second aggregate metric (that is eval.mts, the
 * load-bearing GO criterion). Concepts with no single-token surface form are
 * UNSCORABLE and skipped (same rule as eval.mts), never scored as a miss.
 *
 * DETERMINISTIC: fixed curated set (first `take` items of each suite, file order),
 * greedy/no-RNG readout, no wall-clock field → two runs are byte-identical.
 *
 * Run with: JLENS_PACK=lens-pack-v1.safetensors \
 *   oxnode packages/browser/scripts/jlens/qualitative.mts
 *   (NOT tsx/ts-node — repo convention; needs env PATH="/opt/homebrew/bin:$PATH")
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const OUT_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
const PACK_ENV = process.env.JLENS_PACK ?? 'lens-pack-v1.safetensors';
const PACK_PATH = PACK_ENV.includes('/') ? PACK_ENV : join(OUT_DIR, PACK_ENV);
const OUT_NAME = process.env.JLENS_QUAL_OUT ?? 'qualitative-results.json';
const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), 'data');

const REQUEST_LAYERS = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24 (24 = J=I)
const RANK_CENSOR = 999;
const LENS_MAX_PINNED = 8;
const LENS_MAX_POSITIONS = 48;
const TOPK_LEGIBLE = 10; // "in the top-K" threshold (rank ≤ K)
const TOPK_DUMP = 10; // # decoded tokens dumped per lens at an evidence boundary
const HEADLINE_MAX_BOUND = 23; // fitted-J domain is boundaries 1..23 (24 = J=I, excluded from evidence)

type Suite = { slug: string; readout: 'preTarget' | 'lastNewline'; synonyms: boolean; take: number };
// Curated archetypes (paper's story prompts), first `take` items of each suite (deterministic, file order).
const SUITES: Suite[] = [
  { slug: 'multihop', readout: 'preTarget', synonyms: false, take: 4 }, // multi-hop facts (sport/where-born)
  { slug: 'order-ops', readout: 'preTarget', synonyms: true, take: 3 }, // arithmetic (calc)
  { slug: 'association', readout: 'preTarget', synonyms: false, take: 3 }, // associations (list-of-animals)
  { slug: 'multilingual', readout: 'preTarget', synonyms: false, take: 2 }, // translation
  { slug: 'poetry', readout: 'lastNewline', synonyms: false, take: 2 }, // poetry line-break
];

type Item = { name: string; prompt: string; target?: string; intermediates: string[] };

// ---- number → words (order-ops synonym expansion) — copied verbatim from
// eval.mts / band-report.mts so surface-form derivation is identical (fairness). ----
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

type Slice = {
  suite: string;
  item: string;
  prompt: string;
  concept: string;
  boundary: number; // ℓ (1..23)
  jRank: number;
  logitRank: number;
};

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
  async function surfaceIds(base: string): Promise<number[]> {
    const ids = new Set<number>();
    for (const cand of [` ${base}`, base]) {
      if (cand.length === 0) continue;
      const enc = await encode(cand);
      if (enc.length === 1) ids.add(enc[0]);
    }
    return [...ids];
  }
  async function conceptIds(concept: string, synonyms: boolean): Promise<number[]> {
    const bases = synonyms ? expandSynonyms(concept) : [concept];
    const idset = new Set<number>();
    for (const b of bases) for (const id of await surfaceIds(b)) idset.add(id);
    return [...idset];
  }

  const slices: Slice[] = [];
  // Per-item evidence: shallowest J-only-legible slice + top-K token dumps at that boundary.
  type Evidence = Slice & { jTopK: string[]; logitTopK: string[] };
  const evidence: Evidence[] = [];
  let nConceptsScored = 0;
  let nConceptsUnscorable = 0;
  let nItems = 0;

  for (const suite of SUITES) {
    const raw = JSON.parse(readFileSync(join(DATA_DIR, `lens-eval-${suite.slug}.json`), 'utf8'));
    const items: Item[] = (raw.items as Item[]).slice(0, suite.take); // deterministic first `take`, file order
    for (const item of items) {
      nItems++;
      const promptIds: number[] = await encode(item.prompt);
      const P = promptIds.length;
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

      // scorable intermediate concepts → single-token surface ids.
      const concepts: { name: string; ids: number[] }[] = [];
      for (const inter of item.intermediates) {
        const ids = await conceptIds(inter, suite.synonyms);
        if (ids.length === 0) { nConceptsUnscorable++; continue; } // unscorable — skip, never a miss
        concepts.push({ name: inter, ids });
        nConceptsScored++;
      }
      if (concepts.length === 0) continue;

      const allIds = [...new Set(concepts.flatMap((c) => c.ids))];
      // per-boundary ranks for every id, both lenses.
      const jRank = new Map<number, number[]>();
      const lRank = new Map<number, number[]>();
      for (const id of allIds) {
        jRank.set(id, new Array(24).fill(RANK_CENSOR));
        lRank.set(id, new Array(24).fill(RANK_CENSOR));
      }
      // top-K decoded tokens at the readout position, per boundary, per lens (cells are
      // independent of pinnedIds, so we capture them from the first batch only).
      const jTopKByBound: string[][] = Array.from({ length: 24 }, () => []);
      const lTopKByBound: string[][] = Array.from({ length: 24 }, () => []);
      let capturedCells = false;

      for (let off = 0; off < allIds.length; off += LENS_MAX_PINNED) {
        const batch = allIds.slice(off, off + LENS_MAX_PINNED);
        for (const useJacobian of [true, false]) {
          const ro = await model.lensReadout(promptIds, {
            layers: REQUEST_LAYERS,
            topK: TOPK_DUMP,
            pinnedIds: batch,
            useJacobian,
          });
          if (useJacobian && ro.jacobianApplied !== true) throw new Error(`${suite.slug}/${item.name}: jacobianApplied=false`);
          for (let bi = 0; bi < batch.length; bi++) {
            const ranks = ro.pinned[bi].ranks as Int32Array;
            const dst = (useJacobian ? jRank : lRank).get(batch[bi])!;
            for (let li = 0; li < REQUEST_LAYERS.length; li++) dst[li] = ranks[li * P + readoutPos];
          }
          if (!capturedCells) {
            for (let li = 0; li < REQUEST_LAYERS.length; li++) {
              const cell = ro.cells[li * P + readoutPos];
              (useJacobian ? jTopKByBound : lTopKByBound)[li] = (cell?.topKTexts ?? []).slice(0, TOPK_DUMP);
            }
          }
        }
        capturedCells = true; // both lenses of the first batch captured
      }

      // per concept, per boundary (fitted-J domain 1..23): J-only-legible slice.
      let shallowest: Slice | null = null;
      for (const c of concepts) {
        for (let b = 0; b < HEADLINE_MAX_BOUND; b++) {
          let jr = RANK_CENSOR;
          let lr = RANK_CENSOR;
          for (const id of c.ids) {
            jr = Math.min(jr, jRank.get(id)![b]);
            lr = Math.min(lr, lRank.get(id)![b]);
          }
          if (jr <= TOPK_LEGIBLE && lr > TOPK_LEGIBLE) {
            const s: Slice = {
              suite: suite.slug, item: item.name, prompt: item.prompt,
              concept: c.name, boundary: b + 1, jRank: jr, logitRank: lr,
            };
            slices.push(s);
            if (!shallowest || b + 1 < shallowest.boundary) shallowest = s;
          }
        }
      }
      if (shallowest) {
        const li = shallowest.boundary - 1;
        evidence.push({ ...shallowest, jTopK: jTopKByBound[li], logitTopK: lTopKByBound[li] });
      }
    }
  }

  // criterion (3): ≥2 J-only-legible slices across ≥2 distinct prompts.
  const distinctItems = new Set(slices.map((s) => `${s.suite}/${s.item}`));
  const distinctConcepts = new Set(slices.map((s) => `${s.suite}/${s.item}/${s.concept}`));
  const criterion3Pass = slices.length >= 2 && distinctItems.size >= 2;

  // ---- console ----
  console.log(`==================== QUALITATIVE PASS (criterion 3) ====================`);
  console.log(`pack = ${PACK_PATH}`);
  console.log(`curated set: ${nItems} items across ${SUITES.length} suites; ${nConceptsScored} scorable concepts (${nConceptsUnscorable} unscorable skipped).`);
  console.log(`\nJ-only-legible slices (J-rank ≤ ${TOPK_LEGIBLE} < logit-rank, fitted-J boundaries ℓ1..${HEADLINE_MAX_BOUND}):`);
  console.log(`  total slices = ${slices.length}  across ${distinctItems.size} distinct prompts, ${distinctConcepts.size} distinct concepts.`);
  console.log(`\nper-item evidence (shallowest J-only-legible slice + top-${TOPK_DUMP} decoded tokens at that boundary):`);
  for (const e of evidence) {
    console.log(`\n  [${e.suite}/${e.item}] concept "${e.concept}" @ boundary ℓ${e.boundary}: J-rank=${e.jRank}  logit-rank=${e.logitRank > RANK_CENSOR - 1 ? '≥999' : e.logitRank}`);
    console.log(`    prompt : ${JSON.stringify(e.prompt.length > 90 ? e.prompt.slice(0, 90) + '…' : e.prompt)}`);
    console.log(`    J  top-${TOPK_DUMP} : ${e.jTopK.map((t) => JSON.stringify(t)).join(' ')}`);
    console.log(`    logit    : ${e.logitTopK.map((t) => JSON.stringify(t)).join(' ')}`);
  }
  console.log(`\ncriterion (3) [≥2 J-only-legible slices across ≥2 prompts] = ${criterion3Pass ? 'PASS' : 'FAIL'}  (${slices.length} slices / ${distinctItems.size} prompts)`);
  console.log(`NOTE: curated qualitative evidence + preset source — NOT a second aggregate metric (that is eval.mts, the load-bearing GO criterion).`);

  // ---- write (no wall-clock field → byte-identical reruns) ----
  mkdirSync(OUT_DIR, { recursive: true });
  const outPath = join(OUT_DIR, OUT_NAME);
  writeFileSync(
    outPath,
    JSON.stringify(
      {
        model: MODEL_PATH,
        pack: PACK_PATH,
        curatedSuites: SUITES.map((s) => ({ slug: s.slug, take: s.take })),
        nItems,
        nConceptsScored,
        nConceptsUnscorable,
        topKLegible: TOPK_LEGIBLE,
        fittedJBoundaries: `ℓ1..${HEADLINE_MAX_BOUND}`,
        honesty:
          'Curated, deterministic qualitative evidence for GO/NO-GO criterion (3) + preset selection — NOT a second aggregate metric (eval.mts is the load-bearing criterion). A slice = a concept with J-lens rank ≤ topKLegible AND logit-lens rank > topKLegible at the same boundary (min over single-token surface ids; unscorable multi-token concepts skipped, never scored as misses; only useJacobian differs between lenses).',
        criterion3Pass,
        nSlices: slices.length,
        nDistinctPrompts: distinctItems.size,
        nDistinctConcepts: distinctConcepts.size,
        slices,
        evidence,
      },
      null,
      2,
    ),
  );
  console.log(`\nWrote ${outPath}`);
  console.log('QUALITATIVE PASS COMPLETE');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
