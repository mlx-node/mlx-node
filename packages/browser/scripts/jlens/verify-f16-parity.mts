/**
 * J-lens f16 SHIPPED-pack rank-parity harness (Task T4.0, R6).
 *
 * Proves the f16 shipped pack (`lens-pack-v1.f16.safetensors`) is READOUT-
 * EQUIVALENT to the F32 master pack (`lens-pack-v1.safetensors`): loading either
 * and running the SAME `lensReadout(useJacobian=true)` over the SAME fixed prompt
 * and layer set yields IDENTICAL top-K id orderings AND identical pinned
 * full-vocab ranks. This is RANK parity, NOT bit identity — plan D8 astype's the
 * pack to bf16 at load, so F32 and f16 collapse to bf16 operands that differ (if
 * at all) only in the last bf16 bit; the readout ranking must be unchanged. This
 * is the export-time validation named in the plan's cross-phase validation story
 * ("f16 round-trip rank parity at export").
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
  // must agree on. (LENS_MAX_PINNED=8; TOP_K<=8, so no truncation.)
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

  // ---------- Compare: top-K id ordering + pinned ranks, per layer ----------
  console.log(`\n[PARITY] per-layer top-${TOP_K} id ordering + pinned rank parity (F32 vs f16):`);
  let idMismatches = 0;
  let rankMismatches = 0;
  for (let li = 0; li < LAYERS.length; li++) {
    const L = LAYERS[li];
    const idsOk = sameOrder(A.topKByLayer[li], B.topKByLayer[li]);
    const ranksOk = sameOrder(A.pinnedRanksByLayer[li], B.pinnedRanksByLayer[li]);
    if (!idsOk) idMismatches++;
    if (!ranksOk) rankMismatches++;
    const flag = idsOk && ranksOk ? 'ok' : 'DIFF';
    let detail = '';
    if (!idsOk) detail += `  ids F32=[${A.topKByLayer[li].join(',')}] f16=[${B.topKByLayer[li].join(',')}]`;
    if (!ranksOk)
      detail += `  ranks F32=[${A.pinnedRanksByLayer[li].join(',')}] f16=[${B.pinnedRanksByLayer[li].join(',')}]`;
    console.log(`  ℓ${String(L).padStart(2)}: ${flag}${detail}`);
  }

  console.log(
    `\n[PARITY] layers=${LAYERS.length}  id-order mismatches=${idMismatches}  pinned-rank mismatches=${rankMismatches}`,
  );
  if (idMismatches > 0 || rankMismatches > 0) {
    fail(
      `f16 pack diverges from F32: ${idMismatches} id-order + ${rankMismatches} rank mismatch(es) across ${LAYERS.length} layers`,
    );
  }

  console.log('\n==================== f16 == F32 readout (rank parity) ====================');
  console.log(SENTINEL);
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  console.log('F16_PARITY: FAIL');
  process.exit(1);
});
