/**
 * J-lens EVAL harness (Task T1.4) — normalized log-k pass@k AUC, J-lens vs
 * logit-lens, over the paper's six §methods-comparison prompt suites.
 *
 * WHAT THIS MEASURES (paper Fig-52 / vendored data/README.md metric — the
 * AUDIT-CORRECTED protocol, NOT a band-restricted variant):
 *   - For each item, each `intermediate` gets a RANK = the MINIMUM, over the
 *     full fitted source-layer set (boundaries 1..=23) AND over the
 *     intermediate's single-token SURFACE-FORM set, of the intermediate's
 *     1-based full-vocab rank at that item's single readout position. "Recovered
 *     at any layer" — this is the paper/README metric; there is NO mid-band
 *     restriction.
 *   - pass@k(item) = fraction of the item's intermediates whose rank ≤ k.
 *   - suite pass@k = mean over items.
 *   - Reported number per suite = NORMALIZED pass@k AUC: sweep k over a LOG
 *     grid, take the area under pass@k vs ln(k), normalized by ln(k_max) so a
 *     lens that always ranks the intermediate #1 scores exactly 1.0.
 *   - Computed for BOTH the J-lens (useJacobian=true) and the logit-lens
 *     baseline (useJacobian=false) on the SAME layer set, apples-to-apples.
 *
 * HONESTY: the tuned lens is NOT fitted here (the repo ships none) — we compare
 * ONLY vs the logit lens. All tuned-lens / causal / swap results belong to
 * Anthropic's models, not this run. A <6/6 win is a "partial reproduction on a
 * 0.8B", never "reproduced the paper".
 *
 * NAPI CONTRACT honored (crates/mlx-core/src/inspector.rs):
 *   - headline layer set = [1..=23] (the fitted-J domain); [1..=24] (24 = J=I)
 *     is a supplementary column only. Boundary 0 is unfitted and ERRORS.
 *   - pinnedIds capped at LENS_MAX_PINNED=8 (silently truncated past that) — we
 *     BATCH each item's surface-form ids into groups of ≤8 and merge ranks.
 *   - ranks are display-capped at 999 (right-censored: 999 == "≥999", a miss).
 *     The log-k grid max is 377 < 999, so a censored rank is a miss at every k.
 *   - topK clamped to 32; we request topK=1 (irrelevant to the pinned metric).
 *   - LENS_MAX_POSITIONS=48; every eval prompt ≤ 41 tok (T1.3), asserted here.
 *
 * Run with: oxnode packages/browser/scripts/jlens/eval.mts
 *   (NOT tsx/ts-node — repo convention; needs env PATH="/opt/homebrew/bin:$PATH")
 *
 * NOTE (deliberately NOT the verify-*.mts fail-closed re-exec wrapper): this is
 * a ~3-minute analysis run whose value is the LIVE per-suite progress, so it
 * runs in-process. A native load failure still routes through main().catch ->
 * exit 1, and a clean run prints the `EVAL COMPLETE` sentinel + writes results.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const PACK_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-pilot.safetensors';
const OUT_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), 'data');

// Fitted-J domain (boundaries 1..=23) is the HEADLINE metric; 24 (J=I) is a
// supplementary column only. We request [1..=24] once per lens and derive both
// slices from the same call.
const HEADLINE_LAYERS = Array.from({ length: 23 }, (_, i) => i + 1); // 1..23
const REQUEST_LAYERS = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const N_HEADLINE = HEADLINE_LAYERS.length; // 23

// Log-spaced pass@k grid — every point < 999 (the rank censor), so a censored
// rank misses at every k. ln(k_max) normalizes a perfect lens to 1.0.
const K_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377];
const LOGK = K_GRID.map((k) => Math.log(k));
const LOGK_SPAN = LOGK[LOGK.length - 1] - LOGK[0]; // ln(377)

const LENS_MAX_PINNED = 8;
const LENS_MAX_POSITIONS = 48;
const RANK_CENSOR = 999;

type Suite = { slug: string; readout: 'preTarget' | 'lastNewline'; synonyms: boolean; expect: number };
// readout: preTarget = final prompt token (== token immediately preceding
// `target`, since these prompts end exactly where target begins); lastNewline =
// last "\n" token (poetry, end of line 1). typo/association also use the final
// prompt token, i.e. preTarget with no `target` field present.
const SUITES: Suite[] = [
  { slug: 'multihop', readout: 'preTarget', synonyms: false, expect: 93 },
  { slug: 'multilingual', readout: 'preTarget', synonyms: false, expect: 107 },
  { slug: 'order-ops', readout: 'preTarget', synonyms: true, expect: 55 },
  { slug: 'poetry', readout: 'lastNewline', synonyms: false, expect: 98 },
  { slug: 'typo', readout: 'preTarget', synonyms: false, expect: 96 },
  { slug: 'association', readout: 'preTarget', synonyms: false, expect: 102 },
];

type Item = { name: string; prompt: string; target?: string; intermediates: string[] };

// ---- number → words (order-ops synonym expansion; covers 0..999) ----
const ONES = [
  'zero',
  'one',
  'two',
  'three',
  'four',
  'five',
  'six',
  'seven',
  'eight',
  'nine',
  'ten',
  'eleven',
  'twelve',
  'thirteen',
  'fourteen',
  'fifteen',
  'sixteen',
  'seventeen',
  'eighteen',
  'nineteen',
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

// operation key → {word form + symbol forms}. README: "operations → symbol and
// word forms". The word form is the key itself; symbols are the math glyphs.
const OP_SYNONYMS: Record<string, string[]> = {
  multiplication: ['multiplication', '*', '×', '·'],
  addition: ['addition', '+'],
  subtraction: ['subtraction', '-', '−'],
  division: ['division', '/', '÷'],
  mod: ['mod', '%'],
  squared: ['squared', '²', '^2'],
};

/** Expand an order-ops intermediate KEY into its synonym base strings. */
function expandSynonyms(intermediate: string): string[] {
  if (/^-?\d+$/.test(intermediate)) {
    const n = Number.parseInt(intermediate, 10);
    return Array.from(new Set([intermediate, ...numberToWords(n)]));
  }
  return OP_SYNONYMS[intermediate] ?? [intermediate];
}

function normLogKAuc(passAtK: number[]): number {
  // trapezoid of pass vs ln(k), normalized by the ln-span (perfect => 1.0).
  let area = 0;
  for (let i = 0; i + 1 < passAtK.length; i++) {
    area += 0.5 * (passAtK[i] + passAtK[i + 1]) * (LOGK[i + 1] - LOGK[i]);
  }
  return area / LOGK_SPAN;
}

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  const loaded = await model.loadLensPackFromFile(PACK_PATH);
  if (loaded !== 23) throw new Error(`expected 23 Jacobians in the pilot pack, got ${loaded}`);
  console.log(`Model loaded; pilot pack = ${loaded} Jacobians (J.1..J.23).\n`);

  // memoized tokenizer encode (surface-form candidates repeat heavily).
  const encCache = new Map<string, number[]>();
  const encode = async (s: string): Promise<number[]> => {
    const hit = encCache.get(s);
    if (hit) return hit;
    const ids: number[] = await model.encodeTokens(s);
    encCache.set(s, ids);
    return ids;
  };

  /** Single-token surface-form ids for one base string: both `" "+s` and bare
   *  `s`; keep only forms that encode to EXACTLY ONE token. */
  async function surfaceIds(base: string): Promise<number[]> {
    const ids = new Set<number>();
    for (const cand of [` ${base}`, base]) {
      if (cand.length === 0) continue;
      const enc = await encode(cand);
      if (enc.length === 1) ids.add(enc[0]);
    }
    return [...ids];
  }

  // per-suite fallback bookkeeping (intermediates with ZERO single-token forms).
  const deviations: { suite: string; item: string; intermediate: string; firstTokenOf: string }[] = [];

  type SuiteResult = {
    slug: string;
    nItems: number;
    jAucHead: number;
    logitAucHead: number;
    jAucSupp: number;
    logitAucSupp: number;
    jPassHead: number[];
    logitPassHead: number[]; // pass@k curves (headline)
  };
  const results: SuiteResult[] = [];

  let maxPromptLen = 0;

  for (const suite of SUITES) {
    const raw = JSON.parse(readFileSync(join(DATA_DIR, `lens-eval-${suite.slug}.json`), 'utf8'));
    const items: Item[] = raw.items;
    if (items.length !== suite.expect) {
      throw new Error(`${suite.slug}: expected ${suite.expect} items, got ${items.length}`);
    }

    // accumulate pass@k over items, separately for J/logit and headline/supp.
    const acc = {
      jHead: new Array(K_GRID.length).fill(0),
      logitHead: new Array(K_GRID.length).fill(0),
      jSupp: new Array(K_GRID.length).fill(0),
      logitSupp: new Array(K_GRID.length).fill(0),
    };

    const t0 = Date.now();
    for (const item of items) {
      const promptIds: number[] = await encode(item.prompt);
      const P = promptIds.length;
      if (P > maxPromptLen) maxPromptLen = P;
      if (P > LENS_MAX_POSITIONS) {
        throw new Error(`${suite.slug}/${item.name}: prompt ${P} tok > LENS_MAX_POSITIONS=${LENS_MAX_POSITIONS}`);
      }

      // readout position.
      let readoutPos = P - 1;
      if (suite.readout === 'lastNewline') {
        const texts: string[] = await model.decodeTokens(promptIds);
        let nl = -1;
        for (let i = 0; i < texts.length; i++) if (texts[i].includes('\n')) nl = i;
        if (nl < 0) throw new Error(`${suite.slug}/${item.name}: no newline token found`);
        readoutPos = nl;
      }

      // per-intermediate single-token surface-form id sets.
      const interIds: number[][] = [];
      for (const inter of item.intermediates) {
        const bases = suite.synonyms ? expandSynonyms(inter) : [inter];
        const idset = new Set<number>();
        for (const b of bases) for (const id of await surfaceIds(b)) idset.add(id);
        if (idset.size === 0) {
          // fallback: FIRST token of the leading-space form of the ORIGINAL key.
          const enc = await encode(` ${inter}`);
          idset.add(enc[0]);
          deviations.push({ suite: suite.slug, item: item.name, intermediate: inter, firstTokenOf: ` ${inter}` });
        }
        interIds.push([...idset]);
      }

      // union of ids for this item, batched ≤8; rank tracks per id.
      const allIds = [...new Set(interIds.flat())];
      type Rank = { head: number; supp: number };
      const idRank = new Map<number, { j: Rank; logit: Rank }>();
      for (const id of allIds)
        idRank.set(id, {
          j: { head: RANK_CENSOR, supp: RANK_CENSOR },
          logit: { head: RANK_CENSOR, supp: RANK_CENSOR },
        });

      for (let off = 0; off < allIds.length; off += LENS_MAX_PINNED) {
        const batch = allIds.slice(off, off + LENS_MAX_PINNED);
        if (batch.length > LENS_MAX_PINNED) throw new Error('pinned batch exceeds cap'); // invariant
        for (const useJacobian of [true, false]) {
          const ro = await model.lensReadout(promptIds, {
            layers: REQUEST_LAYERS,
            topK: 1,
            pinnedIds: batch,
            useJacobian,
          });
          if (useJacobian && ro.jacobianApplied !== true)
            throw new Error(`${suite.slug}/${item.name}: jacobianApplied=false`);
          for (let bi = 0; bi < batch.length; bi++) {
            const ranks = ro.pinned[bi].ranks as Int32Array;
            let head = RANK_CENSOR;
            let supp = RANK_CENSOR;
            for (let li = 0; li < REQUEST_LAYERS.length; li++) {
              const r = ranks[li * P + readoutPos];
              if (li < N_HEADLINE && r < head) head = r; // layers 1..23
              if (r < supp) supp = r; // layers 1..24
            }
            const rec = idRank.get(batch[bi])!;
            if (useJacobian) rec.j = { head, supp };
            else rec.logit = { head, supp };
          }
        }
      }

      // per-intermediate rank = min over its surface-form ids; then pass@k.
      const jHead: number[] = [];
      const jSupp: number[] = [];
      const logitHead: number[] = [];
      const logitSupp: number[] = [];
      for (const ids of interIds) {
        let jh = RANK_CENSOR,
          js = RANK_CENSOR,
          lh = RANK_CENSOR,
          ls = RANK_CENSOR;
        for (const id of ids) {
          const rec = idRank.get(id)!;
          jh = Math.min(jh, rec.j.head);
          js = Math.min(js, rec.j.supp);
          lh = Math.min(lh, rec.logit.head);
          ls = Math.min(ls, rec.logit.supp);
        }
        jHead.push(jh);
        jSupp.push(js);
        logitHead.push(lh);
        logitSupp.push(ls);
      }
      const M = item.intermediates.length;
      for (let ki = 0; ki < K_GRID.length; ki++) {
        const k = K_GRID[ki];
        acc.jHead[ki] += jHead.filter((r) => r <= k).length / M;
        acc.jSupp[ki] += jSupp.filter((r) => r <= k).length / M;
        acc.logitHead[ki] += logitHead.filter((r) => r <= k).length / M;
        acc.logitSupp[ki] += logitSupp.filter((r) => r <= k).length / M;
      }
    }

    const n = items.length;
    const jPassHead = acc.jHead.map((s) => s / n);
    const logitPassHead = acc.logitHead.map((s) => s / n);
    const jPassSupp = acc.jSupp.map((s) => s / n);
    const logitPassSupp = acc.logitSupp.map((s) => s / n);
    const r: SuiteResult = {
      slug: suite.slug,
      nItems: n,
      jAucHead: normLogKAuc(jPassHead),
      logitAucHead: normLogKAuc(logitPassHead),
      jAucSupp: normLogKAuc(jPassSupp),
      logitAucSupp: normLogKAuc(logitPassSupp),
      jPassHead,
      logitPassHead,
    };
    results.push(r);
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    const win = r.jAucHead > r.logitAucHead ? 'J' : r.jAucHead < r.logitAucHead ? 'logit' : 'tie';
    console.log(
      `[${suite.slug.padEnd(13)}] n=${String(n).padStart(3)} (${dt}s)  ` +
        `J-AUC=${r.jAucHead.toFixed(4)}  logit-AUC=${r.logitAucHead.toFixed(4)}  ` +
        `Δ=${(r.jAucHead - r.logitAucHead >= 0 ? '+' : '') + (r.jAucHead - r.logitAucHead).toFixed(4)}  win=${win}`,
    );
  }

  // ---- headline table + X/6 ----
  const jWins = results.filter((r) => r.jAucHead > r.logitAucHead).length;
  console.log(`\n================ HEADLINE (min-over-layers 1..=23, normalized log-k pass@k AUC) ================`);
  console.log(`suite          nItems   J-AUC   logit-AUC     Δ(J−logit)   winner`);
  for (const r of results) {
    const d = r.jAucHead - r.logitAucHead;
    const win = d > 0 ? 'J' : d < 0 ? 'logit' : 'tie';
    console.log(
      `${r.slug.padEnd(13)} ${String(r.nItems).padStart(5)}   ${r.jAucHead.toFixed(4)}   ${r.logitAucHead.toFixed(4)}     ` +
        `${(d >= 0 ? '+' : '') + d.toFixed(4)}      ${win}`,
    );
  }
  console.log(`\nJ beats logit on ${jWins}/6 suites (headline, layers 1..=23).`);
  console.log(`Supplementary [1..=24, 24=J=I] AUCs (NOT the pass/fail number):`);
  for (const r of results) {
    console.log(`  ${r.slug.padEnd(13)} J=${r.jAucSupp.toFixed(4)}  logit=${r.logitAucSupp.toFixed(4)}`);
  }
  if (deviations.length) {
    console.log(
      `\nSurface-form fallbacks (zero single-token forms → first token of leading-space form): ${deviations.length}`,
    );
    for (const d of deviations.slice(0, 20))
      console.log(`  ${d.suite}/${d.item}: "${d.intermediate}" -> first token of "${d.firstTokenOf}"`);
  } else {
    console.log(`\nSurface-form fallbacks: 0 (every scored intermediate had ≥1 single-token surface form).`);
  }
  console.log(`\nmax eval prompt length = ${maxPromptLen} tok (LENS_MAX_POSITIONS=${LENS_MAX_POSITIONS}).`);

  mkdirSync(OUT_DIR, { recursive: true });
  const outPath = join(OUT_DIR, 'eval-results.json');
  writeFileSync(
    outPath,
    JSON.stringify(
      {
        model: MODEL_PATH,
        pack: PACK_PATH,
        generatedAt: new Date().toISOString(),
        metric: 'normalized log-k pass@k AUC; rank = min over surface-form ids AND layers 1..=23; perfect=1.0',
        kGrid: K_GRID,
        headlineLayers: HEADLINE_LAYERS,
        jWins6: jWins,
        results,
        deviations,
        maxPromptLen,
        note: 'Tuned-lens/causal/swap results belong to Anthropic models, not this run. <6/6 = partial reproduction on a 0.8B.',
      },
      null,
      2,
    ),
  );
  console.log(`\nWrote ${outPath}`);
  console.log('EVAL COMPLETE');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
