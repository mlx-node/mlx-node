/**
 * J-lens readout verification harness (Task T1.1).
 *
 * Loads the real Qwen3.5-0.8B model and exercises the native `lensReadout`
 * NAPI method against independent oracles. Every gate below is a HARD
 * assertion: a mismatch calls `fail()` (non-zero exit). The harness is THE
 * gate for the native readout core.
 *
 *   Gate A — final-boundary logit-lens exactness:
 *     `lensReadout(useJacobian=false)` at the final boundary (layer 24) must
 *     reproduce the model's OWN output distribution. The oracle is
 *     `runForInspector`, whose step-0 logits are the last-prompt-position
 *     distribution from the very same forward. We feed `lensReadout` the exact
 *     token ids `runForInspector` used (`result.tokens`), so the inputs are
 *     byte-identical and the only thing under test is the readout math. We
 *     assert (a) the ORDERED top-K id list is equal (not just the set), and
 *     (b) the top-K logit VALUES agree within a tight tolerance — they come
 *     from the identical bf16 graph (RMSNorm(h_24) · W_U), so they should be
 *     bit-identical; TOL_A is a small slack for any reduction-order drift.
 *
 *   Negative control (proves Gate A has teeth):
 *     The SAME comparison pointed at a WRONG boundary (layer 0) must diverge —
 *     different top-K ids and large logit-value differences. If Gate A had no
 *     teeth this would silently pass; asserting divergence proves it does not.
 *
 *   Gate B — pinned rank == greater-than count:
 *     The full-vocab rank of each pinned id, computed on-device via
 *     `logits.greater(pinned).sum()+1` (a gt-count), must EXACTLY equal a
 *     host-side hand-counted gt-count over that cell's own top-K logits — a
 *     different kernel path (argpartition+argsort) than the rank path. This
 *     asserts exact rank==gt-count for every pinned id, which also covers the
 *     tie case wherever equal logits occur (equal logits share the best rank).
 *
 *   Gate C — pack gating + honest jacobianApplied + J=I at the final boundary:
 *     `useJacobian=true` with no pack must ERROR for a non-final layer (naming
 *     the missing layer), must be ALLOWED at layer 24 (J=I) with
 *     `jacobianApplied=false`, and there must produce the same ids as the
 *     logit lens.
 *
 * Run with: oxnode packages/browser/scripts/jlens/verify-readout.mts
 * (NOT tsx/ts-node — repo convention.)
 *
 * FAIL-CLOSED CONTRACT (gate integrity):
 *   The native `@mlx-node/lm` import is DYNAMIC and lives only in the child
 *   branch (inside `main()`), so the parent process has NO native dependency.
 *   On invocation the parent re-execs THIS script under the SAME interpreter +
 *   loader (see below) as a child, then exits non-zero UNLESS it observes BOTH
 *   (a) a clean child exit (status 0, no terminating signal) AND (b) the
 *   `ALL GATES PASS` sentinel in the child's stdout. An addon that cannot load
 *   — a Rust panic during NAPI registration, an ABI/arch mismatch, or a hard
 *   `abort()` — happens in the CHILD, before the sentinel is ever printed, so
 *   it FAILS the gate. A native abort that some runtimes swallow to exit 0 is
 *   still caught by the missing-sentinel check. A gate that could pass on a
 *   crash would be no gate; this design cannot.
 *
 *   Re-exec detail: under `oxnode` the oxc TypeScript loader is injected via
 *   `process.execArgv` (`--import .../register.mjs`), NOT via `process.argv0`
 *   (which is plain `node`). We therefore re-exec with
 *   `[...process.execArgv, ...process.argv.slice(1)]` so the child runs under
 *   the identical interpreter + loader and loads the addon exactly as we do.
 */
const SENTINEL = 'ALL GATES PASS';
if (!process.env.__JLENS_GATE_CHILD) {
  const { spawnSync } = await import('node:child_process');
  const res = spawnSync(process.execPath, [...process.execArgv, ...process.argv.slice(1)], {
    env: { ...process.env, __JLENS_GATE_CHILD: '1' },
    encoding: 'utf8',
    stdio: ['inherit', 'pipe', 'inherit'], // capture stdout (scan for sentinel), stream stderr live
  });
  if (res.stdout) process.stdout.write(res.stdout); // still show the child's report to the user
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
// ---- child mode below: run the actual gates (native addon loaded here only) ----
// Fail-closed self-test seam: a child that exits 0 WITHOUT ever printing the
// sentinel MUST still fail the gate. Proves the contract above is enforced.
if (process.env.__JLENS_SELFTEST_NO_SENTINEL) { console.log('child ran, no sentinel'); process.exit(0); }

const MODEL_PATH =
  '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const PROMPT = 'The capital of France is';
const TOP_K = 8;

// Gate A logit-value tolerance. The lens computes `RMSNorm(h_24) · W_U` and so
// does the model's own head (tied embeddings), over the identical bf16
// operands — the values should be bit-identical. TOL_A allows only a sliver of
// slack for possible reduction-order drift; it is far below one bf16 ULP at a
// logit magnitude of ~14 (2^(3-7) ≈ 0.0625), so it cannot mask a wrong-boundary
// readout (see the negative control, which diverges by many logits).
const TOL_A = 1e-2;

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

// Max |a[i] - b[i]| for two equal-length aligned logit vectors.
function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main() {
  // Dynamic import (child branch only): keeps the parent guard native-free so a
  // failed addon registration is caught by the parent as a non-clean/no-sentinel
  // child, and a *catchable* import error routes through `main().catch` -> exit 1.
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  // Cast to `any`: the freshly-built addon exposes `lensReadout`; the published
  // d.cts on disk may lag a native rebuild for a moment.
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
  const refLogits: number[] = Array.from(ref.logits[0].topKLogits as Float32Array);
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
  const lensLogits: number[] = Array.from(lastCell.topKLogits as Float32Array);
  const lensArgmax: number = lastCell.argmaxId;

  console.log(`\n[GATE A] final-boundary (layer 24) logit-lens exactness`);
  console.log(`  lens  top-${TOP_K} ids: [${lensTopK.join(', ')}]  argmax=${lensArgmax}`);
  console.log(`  model top-${TOP_K} ids: [${refTopK.join(', ')}]  argmax=${refArgmax}`);
  // (a) ordered id equality — HARD assert (not just the set, and not just logged).
  if (lensArgmax !== refArgmax) fail(`argmax mismatch: lens ${lensArgmax} != model ${refArgmax}`);
  if (!sameSet(lensTopK, refTopK)) fail(`top-${TOP_K} id set mismatch`);
  if (!sameOrder(lensTopK, refTopK)) fail(`top-${TOP_K} id ORDER mismatch (set equal, order differs)`);
  // (b) logit-value equality within tolerance — ids are order-aligned, so
  // element i of each logit vector is the same token id.
  const diffA = maxAbsDiff(lensLogits, refLogits);
  console.log(`  lens  top-${TOP_K} logits: [${lensLogits.map((L) => L.toFixed(4)).join(', ')}]`);
  console.log(`  model top-${TOP_K} logits: [${refLogits.map((L) => L.toFixed(4)).join(', ')}]`);
  console.log(`  ordered ids equal; max|Δlogit| = ${diffA.toExponential(3)} (tol ${TOL_A})`);
  if (diffA > TOL_A) fail(`top-${TOP_K} logit-value mismatch: max|Δ|=${diffA} > tol ${TOL_A}`);
  console.log(`  argmax + ordered top-${TOP_K} ids + logit values all agree  -> PASS`);

  // sanity: probabilities are honest (0..1) and descending
  const probs: number[] = Array.from(lastCell.topKProbs);
  const probsOk =
    probs.every((p) => p >= 0 && p <= 1) &&
    probs.every((p, i) => i === 0 || probs[i - 1] >= p - 1e-6);
  if (!probsOk) fail(`topKProbs not a valid descending prob vector: [${probs.join(', ')}]`);
  console.log(`  topKProbs (full-vocab-normalized): [${probs.map((p) => p.toFixed(4)).join(', ')}]`);

  // ============ NEGATIVE CONTROL (Gate A has teeth) ============
  // Point the SAME comparison at the wrong boundary (layer 0 = embedding output
  // through the unembedding). It must diverge from the model's own output —
  // different top-K ids and logit values far past TOL_A. This proves the Gate A
  // assertions above would FAIL if fed mismatched data.
  const ro0 = await model.lensReadout(promptIds, { layers: [0], topK: TOP_K, useJacobian: false });
  const cell0 = ro0.cells.find(
    (c: { layer: number; position: number }) => c.layer === 0 && c.position === P - 1,
  );
  if (!cell0) fail('no layer-0 last-position cell for negative control');
  const l0TopK: number[] = cell0.topKIds;
  const l0Logits: number[] = Array.from(cell0.topKLogits as Float32Array);
  const orderedEqual0 = sameOrder(l0TopK, refTopK);
  // Compare logit magnitudes at matched positions only where ids coincide;
  // otherwise the raw magnitude gap alone already blows past TOL_A.
  const l0Diff = sameOrder(l0TopK, refTopK) ? maxAbsDiff(l0Logits, refLogits) : Infinity;
  console.log(`\n[NEG CTRL] wrong-boundary (layer 0) must diverge from the model output`);
  console.log(`  layer-0 top-${TOP_K} ids: [${l0TopK.join(', ')}]  argmax=${cell0.argmaxId}`);
  console.log(`  ordered-id-equal-to-model=${orderedEqual0}  logit max|Δ|=${l0Diff === Infinity ? 'n/a (id set differs)' : l0Diff.toFixed(4)}`);
  if (orderedEqual0 && l0Diff <= TOL_A) {
    fail('negative control did NOT diverge — layer 0 matched the model output within tol; Gate A would be toothless');
  }
  console.log(`  layer 0 diverges from the model output  -> Gate A assertions have teeth`);

  // ===================== GATE B =====================
  // Pin the top-3 tokens at layer24/lastpos. Their device-computed full-vocab
  // rank (`greater(pinned).sum()+1`) must EXACTLY equal a host hand-counted
  // gt-count over that cell's own topKLogits. A top-K token's strict superiors
  // are ALL themselves in the top-K, so the host count over `topKLogits` equals
  // the full-vocab count — and it handles TIES correctly (equal logits share
  // the best rank). We assert exact equality for every pinned id.
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
  // Report whether a tie is actually present in THIS data (honest wording — we
  // only claim a tie is exercised when equal logits truly occur).
  const tieObserved = expected.some((r, j) => expected.some((r2, k) => k !== j && r2 === r));
  console.log(`\n[GATE B] pinned rank == host greater-than count (exact, all pinned ids)`);
  console.log(`  top-3 logits: [${logs.slice(0, 3).map((L) => L.toFixed(4)).join(', ')}]`);
  console.log(`  pinned ids [${pinnedIds.join(', ')}] ranks @layer24/lastpos: observed=[${observed.join(', ')}] expected(host gt-count)=[${expected.join(', ')}]`);
  if (!sameOrder(observed, expected)) fail('pinned rank != host gt-count');
  console.log(
    `  device rank == host gt-count for every pinned id  -> PASS` +
      (tieObserved
        ? ` (a logit TIE is present in this data — shared best rank exercised)`
        : ` (no tie present in this data; exact rank==gt-count still asserted)`),
  );

  // ===================== GATE C =====================
  console.log(`\n[GATE C] pack gating + honest jacobianApplied + J=I @ layer 24`);
  let errMsg = '';
  try {
    await model.lensReadout(promptIds, { layers: [0], topK: TOP_K, useJacobian: true });
  } catch (e: unknown) {
    errMsg = (e as Error).message ?? String(e);
    console.log(`  useJacobian@layer0 (no pack) errored: "${errMsg}"`);
  }
  if (!errMsg) fail('useJacobian at layer 0 with no pack did NOT error');
  // The error must name the missing non-final layer (honesty contract).
  if (!errMsg.includes('[0]')) fail(`error message does not name the missing layer 0: "${errMsg}"`);
  const jr = await model.lensReadout(promptIds, {
    layers: [24],
    topK: TOP_K,
    useJacobian: true,
  });
  // Final-boundary-only useJacobian is allowed (J=I) and must NOT claim applied.
  if (jr.jacobianApplied !== false) fail('jacobianApplied should be false for a J=I@24-only request');
  const jrLast = jr.cells.find(
    (c: { layer: number; position: number }) => c.layer === 24 && c.position === P - 1,
  );
  if (!sameOrder(jrLast.topKIds, lensTopK)) fail('J=I@24 readout differs from logit lens');
  console.log('  layer0 errored (names layer 0), jacobianApplied=false, J=I@24 == logit lens  -> PASS');

  console.log('\n==================== ALL GATES PASS ====================');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
