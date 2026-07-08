/**
 * J-lens FIT + finite-difference GDN gate (Task T1.2).
 *
 * Loads the real Qwen3.5-0.8B model and exercises the native offline-fit
 * surface (`verifyJacobianFiniteDiff`, `fitJacobianLens`) against the T1.1
 * readout. Every gate is a HARD assertion (`fail()` -> non-zero exit).
 *
 *   GATE 1 — THE finite-difference gate (pilot GO/NO-GO):
 *     `verifyJacobianFiniteDiff` compares, in f32, the autograd directional
 *     derivative `⟨∂L/∂z_l, δ⟩` against the numerical central difference
 *     `(L(+εδ)−L(−εδ))/2ε` along a random unit direction δ, for a GatedDeltaNet
 *     boundary (ℓ=5) AND a full-attention boundary (ℓ=7). BOTH must agree to a
 *     small relative error — the proof MLX autograd flows through the 18 GDN
 *     recurrent layers and the 6 full-attention layers. A broken VJP would blow
 *     the relative error up (or NaN).
 *
 *   GATE 2 — pipeline smoke (fit -> pack -> readout):
 *     `fitJacobianLens` fits 1 prompt end-to-end, exports a `J.{layer}` pack,
 *     which is loaded by T1.1's `loadLensPackFromFile` and read out via
 *     `lensReadout(useJacobian=true)` at a mid layer. We assert (a) no NaN,
 *     (b) `jacobianApplied === true`, and (c) the Jacobian readout DIFFERS from
 *     the logit lens at the same layer (the J is doing something). Prints a few
 *     top tokens and the measured secPerPrompt (prices the T1.3 corpus run).
 *
 * Run with: oxnode packages/browser/scripts/jlens/verify-fit.mts
 * (NOT tsx/ts-node — repo convention.)
 *
 * FAIL-CLOSED CONTRACT (identical to verify-readout.mts): the native
 * `@mlx-node/lm` import is DYNAMIC and lives only in the child branch; the
 * parent re-execs THIS script under the same interpreter + loader and exits
 * non-zero UNLESS it observes BOTH a clean child exit AND the `ALL GATES PASS`
 * sentinel. An addon that cannot load fails in the child, before the sentinel.
 */
const SENTINEL = 'ALL GATES PASS';
if (!process.env.__JLENS_GATE_CHILD) {
  const { spawnSync } = await import('node:child_process');
  const res = spawnSync(process.execPath, [...process.execArgv, ...process.argv.slice(1)], {
    env: { ...process.env, __JLENS_GATE_CHILD: '1' },
    encoding: 'utf8',
    stdio: ['inherit', 'pipe', 'inherit'],
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
// ---- child mode below: run the actual gates (native addon loaded here only) ----
if (process.env.__JLENS_SELFTEST_NO_SENTINEL) { console.log('child ran, no sentinel'); process.exit(0); }

import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

/** Parse the tensor-name keys out of a safetensors file header (first 8 bytes =
 *  u64 LE header length, then a JSON object keyed by tensor name + `__metadata__`). */
function safetensorsKeys(path: string): string[] {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8'));
  return Object.keys(header).filter((k) => k !== '__metadata__');
}

const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
// A ~50-token WikiText-like prompt: long enough for the fit's skip-first-16 to
// leave valid positions, short enough to keep the GDN recurrence fast.
const PROMPT =
  'The Rhine is one of the major European rivers, flowing from the Swiss Alps ' +
  'northward through Germany and the Netherlands before emptying into the North ' +
  'Sea near Rotterdam, a route it has followed for thousands of years.';

// Finite-diff gate config.
const GDN_BOUNDARY = 5; // block 5 is linear/GDN (is_linear_layer(5) = true)
const ATTN_BOUNDARY = 7; // block 7 is full-attention (is_linear_layer(7) = false)
// f32 central-difference step, in h-space norm units. The gate takes the
// directional derivative along g_l/‖g_l‖ (max signal), so ε in the 3e-2..1e-1
// sweet spot balances truncation (O(ε²)) against f32 roundoff.
const EPSILON = 5e-2;
// A correct VJP matches to well under this in f32 along the gradient direction;
// a broken one is O(1)+/NaN, so the threshold is decisive with generous slack.
const FD_TOL = 5e-2;

// Smoke-fit config: dimBatch=32 => ceil(1024/32)=32 backward passes (fast).
const FIT_DIM_BATCH = 32;
const FIT_SKIP_FIRST = 16;
const READOUT_LAYER = 5; // a fitted (non-final) boundary
const TOP_K = 8;

function fail(msg: string): never {
  console.error(`\nFAIL: ${msg}`);
  process.exit(1);
}
function sameOrder(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  console.log('Model loaded.\n');

  // Prompt ids (byte-identical to what the model's own forward tokenizes).
  const ref = await model.runForInspector(PROMPT, {
    logits: { topK: 1 },
    maxNewTokens: 1,
    applyChatTemplate: false,
  });
  const promptIds: number[] = ref.tokens.map((t: { id: number }) => t.id);
  const P = promptIds.length;
  console.log(`prompt ("${PROMPT.slice(0, 40)}...")  seq_len=${P}`);
  if (P < FIT_SKIP_FIRST + 2) fail(`prompt too short for skip_first=${FIT_SKIP_FIRST}`);

  // ===================== GATE 1 — finite-difference (GO/NO-GO) =====================
  console.log(`\n[GATE 1] finite-difference through BOTH mixer types (f32, ε=${EPSILON})`);
  const fd = await model.verifyJacobianFiniteDiff(promptIds, [GDN_BOUNDARY, ATTN_BOUNDARY], EPSILON);
  for (let i = 0; i < fd.boundaries.length; i++) {
    const l = fd.boundaries[i];
    const kind = fd.isGdn[i] ? 'GDN     ' : 'ATTN    ';
    console.log(
      `  ℓ=${l} (${kind}) predicted=${fd.predicted[i].toExponential(6)} ` +
        `numerical=${fd.numerical[i].toExponential(6)} relErr=${fd.relError[i].toExponential(3)}`,
    );
  }
  // Assert BOTH boundaries: rel error finite and under tolerance.
  const gdnIdx = fd.boundaries.indexOf(GDN_BOUNDARY);
  const attnIdx = fd.boundaries.indexOf(ATTN_BOUNDARY);
  if (gdnIdx < 0 || attnIdx < 0) fail('finite-diff did not return both requested boundaries');
  if (fd.isGdn[gdnIdx] !== true) fail(`boundary ${GDN_BOUNDARY} was not classified GDN`);
  if (fd.isGdn[attnIdx] !== false) fail(`boundary ${ATTN_BOUNDARY} was not classified full-attention`);
  for (const [name, idx] of [['GDN', gdnIdx], ['ATTN', attnIdx]] as const) {
    const re = fd.relError[idx];
    if (!Number.isFinite(re)) fail(`${name} boundary rel error is not finite (${re}) — autograd does NOT flow`);
    if (re > FD_TOL) fail(`${name} boundary rel error ${re} > tol ${FD_TOL} — autograd does NOT flow correctly`);
  }
  console.log(`  BOTH mixer types pass (relErr < ${FD_TOL}) -> autograd flows through GDN AND attention  -> PASS`);

  // Causality sub-check: a probe injected ONLY at position p must not change
  // h_final BEFORE p (the Σ_{t'≥t} assumption the autograd-vs-finite-diff gate
  // is blind to), and MUST change it at/after p (so the leak check has teeth).
  console.log(`  causality (single-position perturbation at p=${fd.causalitySplitPos}):`);
  for (let i = 0; i < fd.boundaries.length; i++) {
    const kind = fd.isGdn[i] ? 'GDN ' : 'ATTN';
    console.log(
      `    ℓ=${fd.boundaries[i]} (${kind}) ||Δh_final[<p]||=${fd.causalityLeakBefore[i].toExponential(3)} ` +
        `||Δh_final[≥p]||=${fd.causalitySignalAfter[i].toExponential(3)}`,
    );
    if (fd.causalityLeakBefore[i] > 1e-3) {
      fail(`boundary ${fd.boundaries[i]}: causality LEAK — positions before p changed (${fd.causalityLeakBefore[i]}); non-causal`);
    }
    if (!(fd.causalitySignalAfter[i] > 0)) {
      fail(`boundary ${fd.boundaries[i]}: single-position perturbation had NO effect at/after p — leak check is toothless`);
    }
  }
  console.log(`  causal (Δ before p ≈ 0, Δ at/after p > 0) for both boundaries  -> PASS`);

  // Basis-batch consistency: one J_l row computed solo (bv=1) must equal the
  // same row inside a bv=8 batch (a broadcast/shared probe would sum the basis
  // rows and corrupt every row of J_l).
  console.log(
    `  basis-batch consistency @ℓ${fd.batchConsistencyBoundary} row ${fd.batchConsistencyOutDim}: ` +
      `maxAbsDiff=${fd.batchConsistencyMaxAbsDiff.toExponential(3)} rel=${fd.batchConsistencyRel.toExponential(3)}`,
  );
  if (fd.batchConsistencyMaxAbsDiff > 1e-3 || fd.batchConsistencyRel > 1e-3) {
    fail(`bv=1 and bv=8 J rows disagree (maxAbs=${fd.batchConsistencyMaxAbsDiff}, rel=${fd.batchConsistencyRel}) — value_and_grad summed the basis rows`);
  }
  console.log(`  bv=1 row == bv=8 row -> probes are per-basis-row independent  -> PASS`);

  // ===================== GATE 2 — pipeline smoke (fit -> pack -> readout) =====================
  console.log(`\n[GATE 2] pipeline smoke: fit 1 prompt -> export pack -> lensReadout(useJacobian=true)`);
  const workDir = mkdtempSync(join(tmpdir(), 'jlens-fit-'));
  const checkpointPath = join(workDir, 'jlens-master.safetensors');
  const packPath = join(workDir, 'jlens-pack.safetensors');
  try {
    const fit = await model.fitJacobianLens({
      promptIds: [promptIds],
      dimBatch: FIT_DIM_BATCH,
      skipFirst: FIT_SKIP_FIRST,
      checkpointPath,
      packPath,
      resume: false,
    });
    console.log(
      `  fit: nDone=${fit.nDone} promptsThisRun=${fit.promptsThisRun} skipped=${fit.promptsSkipped} ` +
        `nValidLast=${fit.nValidLast} secPerPrompt=${fit.secPerPrompt.toFixed(2)} ` +
        `maxJnorm/sqrt(d)=${fit.maxJNormOverSqrtD.toFixed(4)} peakMemMb=${fit.peakMemoryMb.toFixed(0)}`,
    );
    if (fit.promptsThisRun !== 1) fail(`expected 1 prompt fit, got ${fit.promptsThisRun}`);
    if (!(fit.maxJNormOverSqrtD > 0) || !Number.isFinite(fit.maxJNormOverSqrtD)) {
      fail(`fitted Jacobian norm is degenerate (${fit.maxJNormOverSqrtD})`);
    }

    // KEY-ALIGNMENT assertion (catches a boundary off-by-one before eval): the
    // emitted safetensors keys must be EXACTLY J.1..J.23 — no J.0 (embedding
    // output, unfitted), no J.24 (final = identity).
    const keys = safetensorsKeys(packPath).sort((a, b) => Number(a.slice(2)) - Number(b.slice(2)));
    const expectedKeys = Array.from({ length: 23 }, (_, i) => `J.${i + 1}`);
    console.log(`  pack keys (${keys.length}): ${keys.join(', ')}`);
    if (keys.join(',') !== expectedKeys.join(',')) {
      fail(`pack keys != exactly J.1..J.23 (boundary off-by-one?): got [${keys.join(', ')}]`);
    }
    console.log(`  pack keys are EXACTLY J.1..J.23 (no J.0, no J.24)  -> boundary alignment OK`);

    // Load the exported pack via T1.1's loader.
    const loaded = await model.loadLensPackFromFile(packPath);
    console.log(`  loadLensPackFromFile: ${loaded} Jacobians loaded`);
    if (loaded !== 23) fail(`expected 23 Jacobians in the pack, got ${loaded}`);

    // Jacobian readout vs logit lens at the same mid layer.
    const jro = await model.lensReadout(promptIds, { layers: [READOUT_LAYER], topK: TOP_K, useJacobian: true });
    const lro = await model.lensReadout(promptIds, { layers: [READOUT_LAYER], topK: TOP_K, useJacobian: false });
    if (jro.jacobianApplied !== true) fail('jacobianApplied should be true after loading a pack');
    const jCell = jro.cells.find((c: { position: number }) => c.position === P - 1);
    const lCell = lro.cells.find((c: { position: number }) => c.position === P - 1);
    if (!jCell || !lCell) fail(`no layer-${READOUT_LAYER} last-position cell`);
    const jIds: number[] = jCell.topKIds;
    const lIds: number[] = lCell.topKIds;
    const jTexts: string[] = jCell.topKTexts;
    const lTexts: string[] = lCell.topKTexts;
    // (a) no NaN in the Jacobian readout logits.
    const jLogits = Array.from(jCell.topKLogits as Float32Array);
    if (jLogits.some((x) => !Number.isFinite(x))) fail(`Jacobian readout produced non-finite logits: [${jLogits}]`);
    console.log(`  J-lens  @ℓ${READOUT_LAYER} top-${TOP_K}: [${jIds.join(', ')}]  ("${jTexts.slice(0, 4).join('", "')}"...)`);
    console.log(`  logit   @ℓ${READOUT_LAYER} top-${TOP_K}: [${lIds.join(', ')}]  ("${lTexts.slice(0, 4).join('", "')}"...)`);
    // (c) the two readouts must DIFFER — the Jacobian is doing something.
    if (sameOrder(jIds, lIds)) {
      fail(`Jacobian readout == logit lens at ℓ${READOUT_LAYER}; J is not doing anything (J≈I?)`);
    }
    console.log(`  jacobianApplied=true, no NaN, J-lens ≠ logit lens at ℓ${READOUT_LAYER}  -> PASS`);
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }

  console.log('\n==================== ALL GATES PASS ====================');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
