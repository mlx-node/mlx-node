/**
 * J-lens BAKE (Task T4.3) — precompute curated J-lens frames into committed,
 * import-bundled per-slug JSON so the live lesson (T4.4) can render the
 * Jacobian-vs-logit contrast INSTANTLY, with no in-browser GPU readout.
 *
 * WHY import-bundled (NOT demo/public): vite.config.ts:59 sets
 *   publicDir: command === 'build' ? false : 'public'
 * so on a production build the entire demo/public/ tree is DROPPED — only the
 * browserDeployAssets() allow-list ships. A baked JSON under demo/public/jlens/
 * would work under `yarn dev` but SILENTLY not ship to mlx.void.app. So the baked
 * JSON lives IN the source tree at demo/learn/widgets/jlens/baked/<slug>.json and
 * is imported (Vite bundles it; the SSG prerender gets it synchronously). This
 * honors the plan's INTENT (small per-slug JSON feeding the widget) via a
 * bundled-import location. (Documented deviation, T4.3.)
 *
 * ON-DISK SCHEMA — one file per preset, demo/learn/widgets/jlens/baked/<slug>.json:
 *   {
 *     slug:         string,        // JacobianPreset.slug
 *     prompt:       string,        // raw prompt (NO chat template)
 *     concepts:     string[],      // authored concept list; index-aligned with pins
 *     partialFlags: boolean[],     // per-pin: the concept tokenized to >1 token
 *     layers:       number[],      // JACOBIAN_LAYERS actually read (== each run.layers)
 *     meta: {
 *       model:      'qwen3.5-0.8b',
 *       lensVersion:'v1',
 *       nPrompts:   number,        // meta sidecar n_prompts
 *       bakedDate:  string,        // JLENS_BAKE_DATE ?? meta.fit_date (deterministic)
 *       method:     'precomputed offline'
 *     },
 *     logit:    SerializedRun,     // useJacobian:false
 *     jacobian: SerializedRun      // useJacobian:true
 *   }
 *
 * SerializedRun is a `LensReadoutRun` (packages/browser/src/inspector-types.ts)
 * with EVERY Float32Array/Int32Array converted to a plain number[] via Array.from:
 *   - cells[].topKLogits : Float32Array -> number[]
 *   - cells[].topKProbs  : Float32Array -> number[]
 *   - pinned[].ranks     : Int32Array   -> number[]
 * because JSON.stringify turns a typed array into {"0":..,"1":..} NOT an array.
 * T4.4 adds a revive() that wraps these back into typed arrays before feeding the
 * SAME buildLensSlice(run) path used by the live RPC (design decision D11).
 *
 * SCOPE: exactly ONE LensReadoutRun per (preset, mode) = top-10 per cell + the
 * <=8 concept-pin rank tracks. The plan's "top-10-union ∪ curated pins" (100%
 * clickable) recipe needs >8 pins (batched) AND a data-model beyond
 * LensReadoutRun/buildLensSlice — that is the OPTIONAL Phase-5 slice-stack, not
 * this task.
 *
 * Run (CONTROLLER ONLY — loads the model on the single, serial Metal GPU):
 *   env PATH="/opt/homebrew/bin:$PATH" oxnode packages/browser/scripts/jlens/bake.mts
 * Overridable env (all optional):
 *   JLENS_MODEL      model dir            (default: the local qwen3.5-0.8b bf16 checkpoint)
 *   JLENS_PACK       lens pack path       (default: the SHIPPED f16 pack — browser parity)
 *                    a bare filename resolves against the pack's cache dir; a value
 *                    containing '/' is used verbatim.
 *   JLENS_META       meta sidecar path    (default: lens-pack-v1.meta.json)
 *   JLENS_BAKE_DATE  override bakedDate    (default: meta.fit_date)
 *   JLENS_OUT_DIR    output dir           (default: demo/learn/widgets/jlens/baked)
 * DETERMINISM: this script NEVER calls Date.now()/new Date(); bakedDate comes from
 * JLENS_BAKE_DATE or the meta sidecar's fit_date, so a re-run is byte-identical.
 */
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { JACOBIAN_LAYERS, JACOBIAN_PRESETS } from '../../demo/jlens-core/jacobian-presets.ts';

const HERE = dirname(fileURLToPath(import.meta.url));

// Defaults mirror scripts/jlens/eval.mts (:54, :178-179) but pin the SHIPPED f16
// pack so baked frames match what browsers load (T4.0 F16_PARITY PASS).
const DEFAULT_MODEL = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const DEFAULT_PACK = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-v1.f16.safetensors';
const DEFAULT_META = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-v1.meta.json';
const JLENS_CACHE_DIR = dirname(DEFAULT_PACK);

const MODEL_PATH = process.env.JLENS_MODEL ?? DEFAULT_MODEL;
// bare filename -> resolve against the jlens cache dir; '/'-bearing -> verbatim.
const PACK_ENV = process.env.JLENS_PACK;
const PACK_PATH = PACK_ENV ? (PACK_ENV.includes('/') ? PACK_ENV : join(JLENS_CACHE_DIR, PACK_ENV)) : DEFAULT_PACK;
const META_PATH = process.env.JLENS_META ?? DEFAULT_META;
const OUT_DIR = process.env.JLENS_OUT_DIR ?? join(HERE, '../../demo/learn/widgets/jlens/baked');

const TOP_K = 10;
const LENS_MAX_PINNED = 8; // crates/mlx-core NAPI contract cap.
const EXPECTED_JACOBIANS = 23; // J.1..J.23 in the v1 pack (eval.mts:179).

// ---- serialized (JSON-safe) mirror of LensReadoutRun -----------------------
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
 *  array (topKLogits/topKProbs Float32Array, ranks Int32Array) becomes number[]. */
function serializeRun(ro: any): SerializedRun {
  return {
    promptLen: ro.promptLen,
    topK: ro.topK,
    useJacobian: ro.useJacobian,
    jacobianApplied: ro.jacobianApplied,
    layers: Array.from(ro.layers as ArrayLike<number>),
    tokens: (ro.tokens as { id: number; text: string }[]).map((t) => ({ id: t.id, text: t.text })),
    cells: (ro.cells as any[]).map((c) => ({
      layer: c.layer,
      position: c.position,
      argmaxId: c.argmaxId,
      topKIds: Array.from(c.topKIds as ArrayLike<number>),
      topKLogits: Array.from(c.topKLogits as ArrayLike<number>),
      topKProbs: Array.from(c.topKProbs as ArrayLike<number>),
      topKTexts: (c.topKTexts as string[]).slice(),
    })),
    pinned: (ro.pinned as any[]).map((p) => ({
      tokenId: p.tokenId,
      tokenText: p.tokenText,
      ranks: Array.from(p.ranks as ArrayLike<number>),
    })),
  };
}

/** Atomic write: `.tmp.<pid>` -> rename (matches export-shipped-pack.mts:316-321). */
function writeAtomic(path: string, data: string): void {
  const tmp = `${path}.tmp.${process.pid}`;
  writeFileSync(tmp, data);
  renameSync(tmp, path);
}

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  console.log(`Loading lens pack from ${PACK_PATH} ...`);
  const loaded = await model.loadLensPackFromFile(PACK_PATH);
  if (loaded !== EXPECTED_JACOBIANS) {
    throw new Error(`expected ${EXPECTED_JACOBIANS} Jacobians in the pack, got ${loaded}`);
  }
  console.log(`Model loaded; pack = ${loaded} Jacobians (J.1..J.${EXPECTED_JACOBIANS}).`);

  // Deterministic bakedDate from the meta sidecar (NO wall clock).
  const meta = JSON.parse(readFileSync(META_PATH, 'utf8')) as { n_prompts: number; fit_date: string };
  const bakedDate = process.env.JLENS_BAKE_DATE ?? meta.fit_date;
  const bakeMeta = {
    model: 'qwen3.5-0.8b',
    lensVersion: 'v1',
    nPrompts: meta.n_prompts,
    bakedDate,
    method: 'precomputed offline',
  } as const;
  console.log(`Baking ${JACOBIAN_PRESETS.length} preset(s) over layers [${JACOBIAN_LAYERS.join(', ')}], topK=${TOP_K}.`);
  console.log(`bakedDate=${bakedDate} (source: ${process.env.JLENS_BAKE_DATE ? 'JLENS_BAKE_DATE' : 'meta.fit_date'}); out=${OUT_DIR}\n`);

  mkdirSync(OUT_DIR, { recursive: true });

  // Qwen emits a standalone space token before a digit, so ` 7` tokenizes to
  // [' ', '7']. Pinning the first token would then track bare whitespace. Learn
  // the standalone-space id once so the pin loop can detect that case.
  const spaceEnc: number[] = await model.encodeTokens(' ');
  const standaloneSpaceId: number | null = spaceEnc.length === 1 ? spaceEnc[0]! : null;

  for (const preset of JACOBIAN_PRESETS) {
    // 1. tokenize the RAW prompt (no chat template — D9).
    const promptIds: number[] = await model.encodeTokens(preset.prompt);
    if (promptIds.length === 0) throw new Error(`${preset.slug}: prompt tokenized to zero tokens`);

    // 2. derive pins exactly as the widget does: tokenize ` ${concept}`, pin the
    //    FIRST token id, flag multi-token concepts. When that first token is the
    //    standalone space (` 7`), fall back to the space-less form — a whitespace
    //    pin's rank track says nothing about the concept it claims to follow.
    const pinnedIds: number[] = [];
    const partialFlags: boolean[] = [];
    for (const concept of preset.concepts) {
      let enc: number[] = await model.encodeTokens(` ${concept}`);
      if (enc.length > 0 && standaloneSpaceId !== null && enc[0] === standaloneSpaceId) {
        enc = await model.encodeTokens(concept);
      }
      if (enc.length === 0) continue; // mirror the widget's skip-empty behavior
      pinnedIds.push(enc[0]);
      partialFlags.push(enc.length > 1);
    }
    // Keep concepts index-aligned with pins so T4.4 can zip them 1:1. None of the
    // curated concepts tokenize to zero tokens; fail loud if that ever changes.
    if (pinnedIds.length !== preset.concepts.length) {
      throw new Error(
        `${preset.slug}: ${preset.concepts.length - pinnedIds.length} concept(s) tokenized to zero tokens — ` +
          `concept↔pin alignment broken`,
      );
    }
    if (pinnedIds.length > LENS_MAX_PINNED) {
      throw new Error(`${preset.slug}: ${pinnedIds.length} pins > LENS_MAX_PINNED=${LENS_MAX_PINNED}`);
    }

    // 3. TWO readouts over the same layer set so the widget can flip the toggle
    //    and show the contrast instantly.
    const logit = await model.lensReadout(promptIds, {
      layers: JACOBIAN_LAYERS,
      topK: TOP_K,
      pinnedIds,
      useJacobian: false,
    });
    const jacobian = await model.lensReadout(promptIds, {
      layers: JACOBIAN_LAYERS,
      topK: TOP_K,
      pinnedIds,
      useJacobian: true,
    });
    // Keep the eval.mts guard: a `true` request that silently downgraded is a bug.
    if (jacobian.jacobianApplied !== true) {
      throw new Error(`${preset.slug}: jacobianApplied=false (expected a real Jacobian on the non-final boundaries)`);
    }
    // A pin the reader sees as a concept track must be a real token, not a space.
    for (const pin of jacobian.pinned) {
      if (pin.tokenText.trim() === '') {
        throw new Error(
          `${preset.slug}: pinned token ${pin.tokenId} is whitespace-only (${JSON.stringify(pin.tokenText)}) — ` +
            `its rank track would say nothing about the concept it is labeled with`,
        );
      }
    }

    // 4. serialize (typed arrays -> number[]) and write atomically.
    const envelope = {
      slug: preset.slug,
      prompt: preset.prompt,
      concepts: preset.concepts.slice(),
      partialFlags,
      layers: Array.from(JACOBIAN_LAYERS),
      meta: bakeMeta,
      logit: serializeRun(logit),
      jacobian: serializeRun(jacobian),
    };
    const outPath = join(OUT_DIR, `${preset.slug}.json`);
    writeAtomic(outPath, JSON.stringify(envelope, null, 2) + '\n');
    console.log(
      `[${preset.slug.padEnd(18)}] P=${String(promptIds.length).padStart(2)} pins=${pinnedIds.length} ` +
        `cells=${logit.cells.length} → ${outPath}`,
    );
  }

  console.log('\nBAKE: OK');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
