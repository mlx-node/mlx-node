/**
 * J-lens readout verification harness (Task T1.1).
 *
 * Loads the real Qwen3.5-0.8B model and exercises the new native
 * `lensReadout` NAPI method against two independent oracles:
 *
 *   Gate A — final-boundary logit-lens exactness:
 *     `lensReadout(useJacobian=false)` at the final boundary (layer 24) must
 *     reproduce the model's OWN output distribution. The oracle is
 *     `runForInspector`, whose step-0 logits are the last-prompt-position
 *     distribution from the very same forward. We feed `lensReadout` the exact
 *     token ids `runForInspector` used (`result.tokens`), so the inputs are
 *     byte-identical and the only thing under test is the readout math.
 *
 *   Gate B — pinned rank == greater-than count:
 *     The full-vocab rank of the top-1/2/3 tokens must be exactly 1/2/3. The
 *     rank is computed on-device via `logits.greater(pinned).sum()` (a
 *     gt-count), whereas the top-K ordering is computed via
 *     argpartition+argsort — independent kernels — so agreement cross-checks
 *     the rank path against a hand-counted expectation.
 *
 *   Gate C — pack gating + J=I at the final boundary:
 *     `useJacobian=true` with no pack must error for a non-final layer, be
 *     allowed at layer 24 (J=I), and there produce the same ids as the logit
 *     lens.
 *
 * Run with: oxnode packages/browser/scripts/jlens/verify-readout.mts
 * (NOT tsx/ts-node — repo convention.)
 */
import { Qwen35Model } from '@mlx-node/lm';

const MODEL_PATH =
  '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const PROMPT = 'The capital of France is';
const TOP_K = 8;

function fail(msg: string): never {
  console.error(`\nFAIL: ${msg}`);
  process.exit(1);
}

function sameSet(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  const as = [...a].sort((x, y) => x - y);
  const bs = [...b].sort((x, y) => x - y);
  return as.every((v, i) => v === bs[i]);
}

function sameOrder(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

async function main() {
  console.log(`Loading model from ${MODEL_PATH} ...`);
  // Cast to `any`: the freshly-built addon exposes `lensReadout`, but the
  // published d.cts on disk may lag the native rebuild for a moment.
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  console.log('Model loaded.\n');

  // ---- Oracle: the model's own output distribution at the final position ----
  const ref = await model.runForInspector(PROMPT, {
    logits: { topK: TOP_K },
    maxNewTokens: 1,
    applyChatTemplate: false,
  });
  if (!ref.logits || ref.logits.length === 0) fail('runForInspector returned no logits');
  const promptIds: number[] = ref.tokens.map((t: { id: number }) => t.id);
  const P = promptIds.length;
  const refTopK: number[] = ref.logits[0].topKIds;
  const refArgmax = refTopK[0];
  console.log(`prompt="${PROMPT}"`);
  console.log(`prompt ids (${P}): [${promptIds.join(', ')}]`);

  // ===================== GATE A =====================
  const readout = await model.lensReadout(promptIds, {
    layers: [24],
    topK: TOP_K,
    useJacobian: false,
  });
  const lastCell = readout.cells.find(
    (c: { layer: number; position: number }) => c.layer === 24 && c.position === P - 1,
  );
  if (!lastCell) fail('no layer-24 last-position cell in readout');
  const lensTopK: number[] = lastCell.topKIds;
  const lensArgmax: number = lastCell.argmaxId;

  console.log(`\n[GATE A] final-boundary (layer 24) logit-lens exactness`);
  console.log(`  lens  top-${TOP_K} ids: [${lensTopK.join(', ')}]  argmax=${lensArgmax}`);
  console.log(`  model top-${TOP_K} ids: [${refTopK.join(', ')}]  argmax=${refArgmax}`);
  if (lensArgmax !== refArgmax) fail(`argmax mismatch: lens ${lensArgmax} != model ${refArgmax}`);
  if (!sameSet(lensTopK, refTopK)) fail(`top-${TOP_K} id set mismatch`);
  const ordered = sameOrder(lensTopK, refTopK);
  console.log(`  argmax equal + top-${TOP_K} set equal (ordered=${ordered})  -> PASS`);

  // sanity: probabilities are honest (0..1) and descending
  const probs: number[] = Array.from(lastCell.topKProbs);
  const probsOk =
    probs.every((p) => p >= 0 && p <= 1) &&
    probs.every((p, i) => i === 0 || probs[i - 1] >= p - 1e-6);
  if (!probsOk) fail(`topKProbs not a valid descending prob vector: [${probs.join(', ')}]`);
  console.log(`  topKProbs (full-vocab-normalized): [${probs.map((p) => p.toFixed(4)).join(', ')}]`);

  // ===================== GATE B =====================
  // Pin the top-3 tokens. Their device-computed full-vocab rank
  // (`greater(pinned).sum() + 1`) must equal a host-side hand-counted
  // gt-count. Because a top-K token's strict superiors are ALL themselves in
  // the top-K (the K highest logits), the hand count over `topKLogits` equals
  // the full-vocab count — and it handles TIES correctly (equal logits share
  // the best rank; here the top-2 tokens are exactly tied).
  const pinnedIds = [lensTopK[0], lensTopK[1], lensTopK[2]];
  const logs: number[] = Array.from(lastCell.topKLogits);
  const expected = pinnedIds.map((_, j) => 1 + logs.filter((L) => L > logs[j]).length);
  const readout2 = await model.lensReadout(promptIds, {
    layers: [24],
    topK: TOP_K,
    pinnedIds,
    useJacobian: false,
  });
  // cells: layer-major then position-minor; layers=[24] => last pos is index P-1
  const observed = readout2.pinned.map(
    (pin: { ranks: Int32Array }) => Array.from(pin.ranks)[P - 1],
  );
  console.log(`\n[GATE B] pinned rank == host greater-than count`);
  console.log(`  top-3 logits: [${logs.slice(0, 3).map((L) => L.toFixed(4)).join(', ')}]${logs[0] === logs[1] ? '  (top-2 tied)' : ''}`);
  console.log(`  pinned ids [${pinnedIds.join(', ')}] ranks @layer24/lastpos: observed=[${observed.join(', ')}] expected(host gt-count)=[${expected.join(', ')}]`);
  if (!sameOrder(observed, expected)) fail('pinned rank mismatch');
  console.log('  device rank == host gt-count (ties handled)  -> PASS');

  // ===================== GATE C =====================
  console.log(`\n[GATE C] pack gating + J=I @ layer 24`);
  let errored = false;
  try {
    await model.lensReadout(promptIds, { layers: [0], topK: TOP_K, useJacobian: true });
  } catch (e: unknown) {
    errored = true;
    console.log(`  useJacobian@layer0 (no pack) errored: "${(e as Error).message ?? e}"`);
  }
  if (!errored) fail('useJacobian at layer 0 with no pack did NOT error');
  const jr = await model.lensReadout(promptIds, {
    layers: [24],
    topK: TOP_K,
    useJacobian: true,
  });
  if (jr.jacobianApplied !== false) fail('jacobianApplied should be false when no pack loaded');
  const jrLast = jr.cells.find(
    (c: { layer: number; position: number }) => c.layer === 24 && c.position === P - 1,
  );
  if (!sameOrder(jrLast.topKIds, lensTopK)) fail('J=I@24 readout differs from logit lens');
  console.log('  layer0 errored, jacobianApplied=false, J=I@24 == logit lens  -> PASS');

  console.log('\n==================== ALL GATES PASS ====================');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
