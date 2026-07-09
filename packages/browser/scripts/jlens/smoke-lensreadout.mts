/**
 * NATIVE correctness + latency smoke for the `lensReadout` NAPI method that the
 * worker RPC (T2.1) wraps. Proves the call SHAPE the worker handler forwards is
 * right and gives an in-process native latency FLOOR (the in-browser Chrome
 * benchmark is run separately by the controller — this script never touches the
 * worker or a browser).
 *
 * Logit-lens only (useJacobian:false), so NO lens pack is loaded — a plain
 * logit lens needs none. Model load mirrors eval.mts.
 *
 * Run with: env PATH="/opt/homebrew/bin:$PATH" oxnode packages/browser/scripts/jlens/smoke-lensreadout.mts
 *   (NOT tsx/ts-node — repo convention.)
 */
const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';

// ~14–16 layers × ~32 positions. Non-final boundaries are fine with the logit
// lens (useJacobian:false needs no fitted Jacobian).
const LAYERS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 23, 24];
const TOP_K = 5;

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  console.log('Model loaded (no lens pack — logit lens).');

  // Tokenize a ~32-token prompt, then clamp to <= 48 (LENS_MAX_POSITIONS).
  const prompt =
    'The history of computing spans many centuries, from the earliest mechanical ' +
    'calculators through the vacuum tube era to the modern silicon microprocessors ' +
    'that power todays laptops, phones, and vast cloud data centers around the world.';
  const encoded: number[] = await model.encodeTokens(prompt);
  const promptIds = encoded.slice(0, 32);
  console.log(`Prompt tokenized: ${encoded.length} tokens; using first ${promptIds.length}.`);

  const opts = { layers: LAYERS, topK: TOP_K, pinnedIds: [] as number[], useJacobian: false };

  // ---- correctness (single run) ----
  const result = (await model.lensReadout(promptIds, opts)) as any;
  const P = result.promptLen;
  const L = result.layers.length;

  const checks: [string, boolean][] = [];
  checks.push([`cells.length (${result.cells.length}) === layers.length*promptLen (${L}*${P}=${L * P})`, result.cells.length === L * P]);
  checks.push([`promptLen (${P}) === promptIds.length (${promptIds.length})`, P === promptIds.length]);
  checks.push([`layers.length (${L}) === requested (${LAYERS.length})`, L === LAYERS.length]);
  checks.push([`result.topK (${result.topK}) === TOP_K (${TOP_K})`, result.topK === TOP_K]);

  let allLogitsOk = true;
  let allProbsOk = true;
  for (const cell of result.cells) {
    if (cell.topKLogits.length !== result.topK) allLogitsOk = false;
    if (cell.topKProbs.length !== result.topK) allProbsOk = false;
  }
  checks.push([`every cell topKLogits.length === topK (${result.topK})`, allLogitsOk]);
  checks.push([`every cell topKProbs.length === topK (${result.topK})`, allProbsOk]);
  checks.push([`jacobianApplied === false (logit lens)`, result.jacobianApplied === false]);
  checks.push([`useJacobian === false`, result.useJacobian === false]);

  let failed = 0;
  for (const [label, ok] of checks) {
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}`);
    if (!ok) failed++;
  }
  if (failed > 0) throw new Error(`${failed} assertion(s) failed`);

  // ---- latency (>=4 runs, discard run 1 as warm-up, report median) ----
  const RUNS = 5;
  const timings: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    await model.lensReadout(promptIds, opts);
    const dt = performance.now() - t0;
    if (i > 0) timings.push(dt); // discard run 0 (warm-up)
    console.log(`run ${i}: ${dt.toFixed(1)} ms${i === 0 ? ' (warm-up, discarded)' : ''}`);
  }
  const med = median(timings);
  console.log(`\nMEDIAN native lensReadout latency over ${timings.length} timed runs: ${med.toFixed(1)} ms`);
  console.log('SMOKE OK');
}

main().catch((e) => {
  console.error('SMOKE FAILED:', e);
  process.exit(1);
});
