/**
 * Headless re-bake of the /jspace GALLERY frames + a `vetting.json` honesty
 * artifact. Native addon only — NO rebuild (uses the committed
 * packages/core/mlx-core.darwin-arm64.node).
 *
 * WHY a SEPARATE script (not scripts/jlens/bake.mts): the lesson bake iterates
 * JACOBIAN_PRESETS (3 curated lesson prompts) and its output must stay pinned to
 * the lesson widget. The /jspace gallery is a DIFFERENT, larger vetted set
 * (GALLERY, 8 tiles) that lives ONLY under demo/jspace/starters/. Keeping the two
 * bakes apart means the lesson's "core instrument" is untouched (Task-2 decision).
 * The serialize/atomic-write helpers below are copied VERBATIM from bake.mts so
 * every frame is byte-for-byte structurally identical to the existing envelope
 * (same keys, same plain-number[] typed arrays, same byte-escaped topKTexts).
 *
 * PACK CHOICE — the FULL-PRECISION pack (`lens-pack-v1.safetensors`), the one the
 * gallery vetting (scripts/jlens/vet-candidates.mts) and the gallery `band`/`grade`
 * calibration were run at. The f16 pack degrades the two WEAK multi-hop/error tiles
 * below the top-K gate (e.g. eiffel `Paris` rank ~5 at full precision vs ~13 at f16),
 * so it would spuriously fail the non-drift assertion. The one tile the browser
 * self-test actually re-checks is `french-season` (JSpaceApp.runSelfTest hardcodes
 * it); it is a STRONG tile and the pack precision does not move its pins: an
 * empirical compareToBakedFrame of this full-precision french-season bake against
 * the committed f16 lesson frame (the browser's self-test reference) returns
 * ok=true, topOneAgreement 0.99, worstPinDelta 0 — comfortably inside self-test.ts's
 * >=0.9 / <=3 thresholds. So the full-precision pack keeps the weak tiles honest AND
 * the in-browser self-test green.
 *
 * Run (CONTROLLER ONLY — single serial Metal GPU):
 *   env PATH="/opt/homebrew/bin:$PATH" oxnode packages/browser/scripts/jlens/bake-gallery.mts
 */
import { existsSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { derivePins, type Encode } from '../../demo/jlens-core/derive-pins.ts';
import { JACOBIAN_LAYERS } from '../../demo/jlens-core/jacobian-presets.ts';
import { GALLERY } from '../../demo/jspace/starters/gallery.ts';
import { LENS_MAX_PINNED } from '../../src/inspector-types.ts';
import { buildIdToBytes, escapeByteFragments } from '../../src/lens-byte-escape.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '../../demo/jspace/starters');

const MODEL = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
// Full-precision pack (see PACK CHOICE header) — the vetting/calibration precision.
const PACK = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-v1.safetensors';
const META = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-v1.meta.json';

const TOP_K = 10; // match the existing frames (french-season cells carry 10 topKIds).
const EXPECTED_JACOBIANS = 23; // J.1..J.23 in the v1 pack (bake.mts:103).
const BAKED_DATE = '2026-07-13';

// ---- serialized (JSON-safe) mirror of LensReadoutRun — copied from bake.mts ----
type SerializedCell = {
  layer: number;
  position: number;
  argmaxId: number;
  topKIds: number[];
  topKLogits: number[]; // Float32Array -> number[]
  topKProbs: number[]; // Float32Array -> number[]
  topKTexts: string[];
};
type SerializedPinned = { tokenId: number; tokenText: string; ranks: number[] }; // Int32Array -> number[]
type SerializedRun = {
  promptLen: number;
  topK: number;
  useJacobian: boolean;
  jacobianApplied: boolean;
  layers: number[];
  tokens: { id: number; text: string }[];
  cells: SerializedCell[];
  pinned: SerializedPinned[];
};

/** Convert a native LensReadoutRun into a JSON-safe SerializedRun: every typed
 *  array (topKLogits/topKProbs Float32Array, ranks Int32Array) becomes number[].
 *  Byte-level-BPE fragment tokens whose standalone HF decode collapsed to `�` are
 *  re-rendered as faithful hex escapes (`‹E5›`) via `escapeByteFragments`, keeping
 *  the bake in lockstep with the committed data + the live worker fix.
 *  COPIED VERBATIM from scripts/jlens/bake.mts:133-161 so the two bakes emit
 *  byte-identical frames. */
function serializeRun(ro: any, idToBytes: (id: number) => Uint8Array | null): SerializedRun {
  return {
    promptLen: ro.promptLen,
    topK: ro.topK,
    useJacobian: ro.useJacobian,
    jacobianApplied: ro.jacobianApplied,
    layers: Array.from(ro.layers as ArrayLike<number>),
    tokens: (ro.tokens as { id: number; text: string }[]).map((t) => ({
      id: t.id,
      text: escapeByteFragments(t.text, t.id, idToBytes),
    })),
    cells: (ro.cells as any[]).map((c) => ({
      layer: c.layer,
      position: c.position,
      argmaxId: c.argmaxId,
      topKIds: Array.from(c.topKIds as ArrayLike<number>),
      topKLogits: Array.from(c.topKLogits as ArrayLike<number>),
      topKProbs: Array.from(c.topKProbs as ArrayLike<number>),
      topKTexts: (c.topKTexts as string[]).map((text, i) =>
        escapeByteFragments(text, (c.topKIds as ArrayLike<number>)[i]!, idToBytes),
      ),
    })),
    pinned: (ro.pinned as any[]).map((p) => ({
      tokenId: p.tokenId,
      tokenText: escapeByteFragments(p.tokenText, p.tokenId, idToBytes),
      ranks: Array.from(p.ranks as ArrayLike<number>),
    })),
  };
}

/** Atomic write: `.tmp.<pid>` -> rename. COPIED from bake.mts:164-168. */
function writeAtomic(path: string, data: string): void {
  const tmp = `${path}.tmp.${process.pid}`;
  writeFileSync(tmp, data);
  renameSync(tmp, path);
}

/** ROBUST PIN RESOLUTION (Task-2 fix #1). `derivePins` pins the FIRST token of
 *  encoding ` ${concept}`, which for some concepts is a sub-word FRAGMENT that
 *  ranks far below the STANDALONE vocab token the model actually surfaces (giza:
 *  the ' Africa' fragment sits at rank 6 while the whole-word 'Africa' is rank 2 at
 *  ℓ17/ℓ18). Once we have a readout we KNOW the per-cell top-K, so we can re-pin to
 *  the exact surfacing token: across every displayed-layer cell at the last
 *  position, find any top-K token whose decoded text — a single leading space
 *  trimmed, case-folded — EQUALS the concept word, and pin the one with the BEST
 *  (lowest) rank. Fall back to derivePins' id ONLY when no top-K exact-text match
 *  exists. This changes a pin ONLY when the surfacing token differs from the guess
 *  — and when it differs, the surfacing token is the correct one. */
function resolvePinIds(
  concepts: string[],
  jacReadout: any,
  lastPos: number,
  fallbackIds: number[],
): { ids: number[]; changed: boolean } {
  const cellsAtLast = (jacReadout.cells as any[]).filter((c) => c.position === lastPos);
  let changed = false;
  const ids = concepts.map((concept, ci) => {
    const target = concept.trim().toLowerCase();
    let bestId: number | null = null;
    let bestRank = Infinity;
    for (const cell of cellsAtLast) {
      const texts = cell.topKTexts as ArrayLike<string>;
      const kids = cell.topKIds as ArrayLike<number>;
      for (let k = 0; k < texts.length; k++) {
        const decoded = String(texts[k]).replace(/^ /, '').toLowerCase();
        if (decoded === target) {
          const rank = k + 1; // top-K arrays are sorted descending by logit.
          if (rank < bestRank) {
            bestRank = rank;
            bestId = kids[k]!;
          }
        }
      }
    }
    if (bestId != null && bestId !== fallbackIds[ci]) changed = true;
    return bestId != null ? bestId : fallbackIds[ci]!;
  });
  return { ids, changed };
}

/** Dropped candidates from scripts/jlens/vet-candidates.mts's roster (the three
 *  phenomena whose gate the 0.8B base model fails). Verbatim from the Task-2
 *  brief; the raw per-layer evidence is in
 *  /Users/brooklyn/workspace/github/mlx-node/.cache/jlens/vet-candidates-results.json. */
const DROPPED_ROWS = [
  {
    slug: 'prompt-injection',
    grade: 'FAIL',
    shipped: false,
    reason:
      "Total floor; only prompt-echo pins ('ignore','translate') hit — disqualified. 0.8B base model has no injection-awareness.",
  },
  {
    slug: 'ascii-emoji-face',
    grade: 'FAIL',
    shipped: false,
    reason:
      'Zero emotion-word hits across all emoticons on both lenses. The emoticon→emotion mapping does not form on Qwen3.5-0.8B.',
  },
  {
    slug: 'count-and-introspect',
    grade: 'FAIL',
    shipped: false,
    reason:
      "Introspection floor (consciousness/self/aware never surface); counting fails Gate 3 (next number IS the model's actual output).",
  },
];

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL} ...`);
  const model = (await Qwen35Model.load(MODEL)) as any;
  // Byte-level vocab resolver so fragment tokens (`�`) bake as faithful hex
  // escapes instead of the lossy replacement char (same util as the live worker).
  const idToBytes = buildIdToBytes(readFileSync(join(MODEL, 'tokenizer.json'), 'utf8'));
  console.log(`Loading lens pack from ${PACK} ...`);
  const loaded = await model.loadLensPackFromFile(PACK);
  if (loaded !== EXPECTED_JACOBIANS) {
    throw new Error(`expected ${EXPECTED_JACOBIANS} Jacobians in the pack, got ${loaded}`);
  }
  console.log(`Model loaded; pack = ${loaded} Jacobians.\n`);

  // nPrompts from the meta sidecar keeps meta's key shape identical to the
  // existing frames (BakedMeta requires it) and honest to the fit corpus.
  const meta = JSON.parse(readFileSync(META, 'utf8')) as { n_prompts: number; fit_date: string };
  const bakeMeta = {
    model: 'qwen3.5-0.8b',
    lensVersion: 'v1',
    nPrompts: meta.n_prompts,
    bakedDate: BAKED_DATE,
    method: 'gallery bake',
  } as const;

  // derivePins needs {id,text}[]; the native model returns ids and texts from two
  // calls. Adapting here keeps ONE pin predicate across bake and the browsers.
  const encodeWithText: Encode = async (text) => {
    const ids: number[] = await model.encodeTokens(text);
    const texts: string[] = await model.decodeTokens(ids);
    return ids.map((id, i) => ({ id, text: texts[i] ?? '' }));
  };

  // "±2 displayed-layers" is a distance along the DISPLAYED JACOBIAN_LAYERS axis,
  // NOT raw layer numbers. band.peak is guaranteed to be a displayed layer
  // (jspace-gallery.test.ts asserts JACOBIAN_LAYERS.contains(band.peak)).
  const layerIndex = (L: number): number => JACOBIAN_LAYERS.indexOf(L);

  const vetting = {
    model: 'Qwen3.5-0.8B',
    layers: JACOBIAN_LAYERS.length,
    topK: TOP_K,
    vettedAt: BAKED_DATE,
    candidates: [] as any[],
  };

  for (const entry of GALLERY) {
    // 1. tokenize the RAW prompt (no chat template — D9).
    const promptIds: number[] = await model.encodeTokens(entry.prompt);
    if (promptIds.length === 0) throw new Error(`${entry.slug}: prompt tokenized to zero tokens`);
    if (promptIds.length > 48) throw new Error(`${entry.slug}: ${promptIds.length} > 48 tokens (addon cap)`);

    // next-token prediction reads the LAST position; needed for pin resolution too.
    const P = promptIds.length;
    const lastPos = P - 1;

    // 2. derive pins exactly as the widget does (` ${concept}` first token, with
    //    the whitespace-fallback for digits so a pin never tracks bare space).
    const { pinnedIds: derivedIds, partialFlags } = await derivePins(entry.concepts, encodeWithText);
    if (derivedIds.length !== entry.concepts.length) {
      throw new Error(
        `${entry.slug}: ${entry.concepts.length - derivedIds.length} concept(s) tokenized to zero tokens — concept↔pin alignment broken`,
      );
    }
    if (derivedIds.length > LENS_MAX_PINNED) {
      throw new Error(`${entry.slug}: ${derivedIds.length} pins > LENS_MAX_PINNED=${LENS_MAX_PINNED}`);
    }

    // 3. TWO readouts over the same layer set (logit + jacobian). The per-cell
    //    top-K is INDEPENDENT of the pins, so this first pass gives us the top-K we
    //    need to resolve robust pins (step 3b) — only the pinned rank tracks depend
    //    on the pin ids, so we re-run only if resolution moved a pin.
    const readouts = async (ids: number[]) => {
      const lg = await model.lensReadout(promptIds, { layers: JACOBIAN_LAYERS, topK: TOP_K, pinnedIds: ids, useJacobian: false });
      const jc = await model.lensReadout(promptIds, { layers: JACOBIAN_LAYERS, topK: TOP_K, pinnedIds: ids, useJacobian: true });
      if (jc.jacobianApplied !== true) {
        throw new Error(`${entry.slug}: jacobianApplied=false (a true request that silently downgraded is a bug)`);
      }
      return { lg, jc };
    };
    let { lg: logit, jc: jac } = await readouts(derivedIds);

    // 3b. Re-pin each concept to the exact top-K token that actually surfaces it
    //     (Task-2 fix #1); re-run the readouts so the baked pinned rank tracks
    //     reflect the resolved ids. `changed` stays false when derivePins already
    //     guessed the surfacing token (french-season / arithmetic tiles).
    const { ids: pinnedIds, changed } = resolvePinIds(entry.concepts, jac, lastPos, derivedIds);
    if (changed) {
      ({ lg: logit, jc: jac } = await readouts(pinnedIds));
    }

    // A pin the reader sees as a concept track must be a real token, not a space.
    for (const pin of jac.pinned as { tokenId: number; tokenText: string }[]) {
      if (pin.tokenText.trim() === '') {
        throw new Error(
          `${entry.slug}: pinned token ${pin.tokenId} is whitespace-only (${JSON.stringify(pin.tokenText)})`,
        );
      }
    }

    // 4. NON-DRIFT ASSERTION (the correctness gate). At the LAST token position
    //    (next-token prediction), find the PRIMARY concept's (pinned[0] ===
    //    concepts[0]) best rank across the displayed layers. It must reach
    //    top-K AND peak within ±2 displayed-layers of the declared band.peak. A
    //    failure means the concept string does not tokenize to the effective
    //    vetted id (a dark tile) — fix the concept in gallery.ts, never weaken this.
    const perLayer = JACOBIAN_LAYERS.map((L, li) => ({
      L,
      r: (jac.pinned[0]?.ranks?.[li * P + lastPos] ?? 999) as number,
    }));
    const inTopK = perLayer.filter((x) => x.r <= TOP_K);
    if (inTopK.length === 0) {
      throw new Error(
        `${entry.slug}: primary concept '${entry.concepts[0]}' never in top-${TOP_K} at the last position — the concept string does not tokenize to the effective vetted id`,
      );
    }
    const peak = inTopK.reduce((a, b) => (b.r < a.r ? b : a));

    // 4b. GRADE-CONSISTENCY ASSERTION (Task-2 fix #3). The displayed grade must
    //     match the measured legibility: ANY tile's primary concept must reach
    //     top-K, and a STRONG tile's must reach top-3 across the displayed layers.
    //     FAIL LOUD (slug + measured rank) rather than ship a mislabeled tile — do
    //     NOT weaken this to pass; regrade the tile in gallery.ts or fix the pin.
    if (peak.r > TOP_K) {
      throw new Error(
        `${entry.slug}: primary concept '${entry.concepts[0]}' best rank ${peak.r} > TOP_K=${TOP_K} across displayed layers`,
      );
    }
    if (entry.grade === 'strong' && peak.r > 3) {
      throw new Error(
        `${entry.slug}: grade='strong' but primary concept '${entry.concepts[0]}' best rank ${peak.r} > 3 across displayed layers — regrade WEAK or fix the pin`,
      );
    }

    const bandIdx = layerIndex(entry.band.peak);
    if (bandIdx < 0) throw new Error(`${entry.slug}: band.peak ℓ${entry.band.peak} is not a displayed layer`);
    const drift = Math.abs(layerIndex(peak.L) - bandIdx);
    if (drift > 2) {
      throw new Error(
        `${entry.slug}: measured peak ℓ${peak.L} (rank ${peak.r}) is ${drift} displayed-layers from declared band.peak ℓ${entry.band.peak} (>2) — concept/band drift`,
      );
    }

    // 5. serialize (typed arrays -> number[], byte-escaped texts) + atomic write.
    const frame = {
      slug: entry.slug,
      prompt: entry.prompt,
      concepts: entry.concepts.slice(),
      partialFlags,
      layers: Array.from(JACOBIAN_LAYERS),
      meta: bakeMeta,
      logit: serializeRun(logit, idToBytes),
      jacobian: serializeRun(jac, idToBytes),
    };
    writeAtomic(join(OUT, `${entry.slug}.json`), JSON.stringify(frame, null, 2) + '\n');

    vetting.candidates.push({
      slug: entry.slug,
      prompt: entry.prompt,
      targetConcept: entry.concepts[0],
      targetIds: pinnedIds,
      jacobian: { pass: true, peakLayer: peak.L, peakRank: peak.r, bandLayers: inTopK.map((x) => x.L) },
      grade: entry.grade.toUpperCase(),
      shipped: true,
    });
    console.log(`baked ${entry.slug}: peak ℓ${peak.L} rank ${peak.r}  (band.peak ℓ${entry.band.peak}, drift ${drift})`);
  }

  // Dropped rows (from vet-candidates.mts roster): injection / emoji-face / count-and-introspect.
  for (const d of DROPPED_ROWS) vetting.candidates.push(d);
  writeAtomic(join(OUT, 'vetting.json'), JSON.stringify(vetting, null, 2) + '\n');

  // Remove superseded legacy frames.
  for (const gone of ['spanish-opposite', 'arithmetic-parens']) {
    const p = join(OUT, `${gone}.json`);
    if (existsSync(p)) {
      unlinkSync(p);
      console.log(`removed legacy ${gone}.json`);
    }
  }

  const shipped = vetting.candidates.filter((c) => c.shipped).length;
  const dropped = vetting.candidates.length - shipped;
  console.log(`\nvetting.json: ${shipped} shipped + ${dropped} dropped`);
  console.log('GALLERY BAKE: OK');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
