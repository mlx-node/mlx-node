#!/usr/bin/env node
/**
 * MTP speculative-decode smoke benchmark.
 *
 * Runs the same prompt twice on a Qwen3.5 / Qwen3.5-MoE checkpoint that
 * carries MTP heads — once with the speculative loop ON, once OFF — at
 * T=0. Speculative decoding is distributionally lossless, NOT bitwise
 * identical to a separate single-token AR run: the batched verify
 * forward and sequential AR decode reduce matmuls in a different order
 * (~1e-2 per-logit difference), which flips rare argmax near-ties. vLLM,
 * MTPLX and dflash-mlx all document this. A near-tie can flip at ANY
 * offset — character position cannot tell a benign near-tie from a real
 * bug — so text divergence is reported as info only; the blocking
 * correctness gate is MTP acceptance health.
 * Reports per-run decode tok/s and the MTP speedup ratio.
 *
 * Plan W6 perf target: >= 1.6x at depth=3 on M3 Max bf16.
 *
 *   oxnode examples/qwen35-mtp-smoke.ts [model-name] [--depth N] [--max-tokens N] [--prompt "..."]
 *   oxnode examples/qwen35-mtp-smoke.ts [model-name] --adaptive [--max-tokens N]
 *
 * `--adaptive` opts INTO the W6.8 adaptive-depth policy: it omits
 * `mtpDepth` from the config so the native side runs
 * `AdaptiveDepthPolicy` (per-depth EMA + DFlash 3-state machine).
 * With `--adaptive` unset (default), `mtpDepth` is pinned to `--depth`
 * (default 3) — matches pre-W6.8 behaviour for parity testing.
 *
 * Defaults: qwen3.5-4b, depth=3, max-tokens=200.
 */

import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import type { ChatConfig, ChatResult, SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    depth: { type: 'string' },
    'max-tokens': { type: 'string' },
    prompt: { type: 'string' },
    'no-warmup': { type: 'boolean' },
    adaptive: { type: 'boolean' },
  },
  allowPositionals: true,
});

const modelName = positionals[0] || 'qwen3.5-4b';
const depth = Number(values.depth ?? '3');
const maxTokens = Number(values['max-tokens'] ?? '200');
const prompt =
  values.prompt ??
  'Write a concise three-paragraph essay on why deterministic sampling at temperature 0 is useful for testing speculative decoding implementations.';
const skipWarmup = values['no-warmup'] === true;
const adaptive = values.adaptive === true;

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

console.log(`Loading model from: ${MODEL_PATH}`);
const loaded = await loadModel(MODEL_PATH);
if (loaded instanceof HarrierModel) {
  console.error('Embedding model is not session-capable.');
  process.exit(1);
}
const model = loaded as unknown as SessionCapableModel;

const hasMtp = typeof model.hasMtpWeights === 'function' && model.hasMtpWeights();
if (!hasMtp) {
  console.error(`Model ${modelName} does not carry MTP heads. Smoke aborted.`);
  process.exit(2);
}
console.log(
  `MTP heads detected. Running ${
    adaptive ? 'adaptive depth (W6.8 policy)' : `depth=${depth} pinned`
  }, max_new_tokens=${maxTokens}`,
);

// W6.8: when `--adaptive`, omit `mtpDepth` so `extract_chat_params`
// defaults `mtp_adaptive_depth = true`. Otherwise pin `mtpDepth = depth`
// which implicitly opts OUT of adaptive (matches pre-W6.8 behaviour).
const baseConfig: ChatConfig = {
  temperature: 0,
  topK: 1,
  topP: 1,
  maxNewTokens: maxTokens,
  reasoningEffort: 'none',
  includeReasoning: false,
  ...(adaptive ? {} : { mtpDepth: depth }),
  reportPerformance: true,
};

async function runOnce(label: string, enableMtp: boolean): Promise<ChatResult> {
  const session = new ChatSession(model, {
    system: 'You are a precise assistant. Be concise.',
    defaultConfig: { ...baseConfig, enableMtp },
  });
  const result = await session.send(prompt);
  const perf = result.performance;
  const tps = perf ? perf.decodeTokensPerSecond.toFixed(2) : 'n/a';
  const ttft = perf ? perf.ttftMs.toFixed(0) : 'n/a';
  console.log(
    `${label} | enableMtp=${enableMtp} | tokens=${result.numTokens} | TTFT=${ttft}ms | decode=${tps} tok/s | stop=${result.finishReason}`,
  );
  return result;
}

// COLD acceptance capture — MUST be the FIRST send on the freshly loaded
// model so the cache is guaranteed cold (paged_adapter empty, nothing
// published to the content-addressed prefix cache yet). This is the
// honest MTP acceptance source. On the paged path
// (`MLX_QWEN35_PAGED_OVERRIDE=1`) a *warm* MTP turn sees
// `cached_prefix_len > 0`, which flips `want_prompt_hidden` false and
// skips the prompt-prefill committed-history seed — so any MTP turn that
// follows an AR/warmup turn on the SAME prompt under-reports acceptance
// (~1.25 vs the true cold ~1.50). `model.resetCaches()` does NOT fix this
// (the published prefix blocks survive a reset), and the shared system
// prompt makes "use a different prompt" fragile — only a never-seen cold
// cache is reliable. The dense path re-seeds via its zero-delta reset, so
// it is unaffected, but this cold-first run keeps BOTH paths honest.
console.log('\n--- Cold acceptance (first send, guaranteed cold cache) ---');
const coldMtp = await runOnce('cold MTP', true);

// Warmup — pays compile + cache costs so the measured tok/s below is
// steady-state. Same prompt both modes.
if (!skipWarmup) {
  console.log('\n--- Warmup (each mode) ---');
  await runOnce('warmup AR', false);
  await runOnce('warmup MTP', true);
}

// Measured runs are WARM/steady-state for a fair, thermally-comparable
// tok/s ratio and the parity check. Their MTP acceptance is intentionally
// NOT used for the gate (it is warm-confounded on the paged path) — the
// gate reads `coldMtp` above.
console.log('\n--- Measured (warm; tok/s ratio + parity) ---');
const ar = await runOnce('measured AR', false);
const mtp = await runOnce('measured MTP', true);

const arText = ar.text;
const mtpText = mtp.text;
const parity = arText === mtpText;

console.log('\n--- Parity (T=0) ---');
// Speculative decoding is distributionally lossless but NOT bitwise
// identical to a separate sequential AR run (batched verify matmul and
// per-row GEMV reduce in a different order). A flipped argmax near-tie
// decorrelates everything downstream, so character offset cannot tell a
// benign near-tie from a real bug — an early divergence is just as
// consistent with a near-tie as a late one (confirmed: the recurring
// offset-16 flip is AR/verify ranking the same top-2 within one bf16
// ulp). Text divergence is therefore INFO ONLY; the blocking correctness
// gate is acceptance health below — a real verify-path bug collapses
// acceptance far below the native-head floor.
if (parity) {
  console.log(`OK: AR and MTP produced identical output (${arText.length} chars).`);
} else {
  let i = 0;
  while (i < Math.min(arText.length, mtpText.length) && arText[i] === mtpText[i]) i++;
  const minLen = Math.min(arText.length, mtpText.length);
  const wStart = Math.max(0, i - 80);
  const wEnd = i + 80;
  console.log(`Outputs diverge at character offset ${i}/${minLen} (info only — see acceptance gate).`);
  console.log(`AR  window [${wStart}..${wEnd}]: ${JSON.stringify(arText.slice(wStart, wEnd))}`);
  console.log(`MTP window [${wStart}..${wEnd}]: ${JSON.stringify(mtpText.slice(wStart, wEnd))}`);
}

console.log('\n--- Speedup ---');
const arTps = ar.performance?.decodeTokensPerSecond ?? 0;
const mtpTps = mtp.performance?.decodeTokensPerSecond ?? 0;
if (arTps > 0 && mtpTps > 0) {
  const ratio = mtpTps / arTps;
  console.log(`Decode tok/s: AR=${arTps.toFixed(2)} MTP=${mtpTps.toFixed(2)} ratio=${ratio.toFixed(2)}x`);
  console.log(`Plan target: >= 1.6x at depth=3. ${ratio >= 1.6 ? 'PASS' : 'BELOW TARGET'}`);
} else {
  console.log('Could not compute speedup (missing performance metrics).');
}

console.log('\n--- MTP acceptance (from the cold run) ---');
let acceptanceOk = true;
const mtpPerf = coldMtp.performance;
if (mtpPerf?.mtpCycles != null) {
  const perPos = mtpPerf.mtpAcceptanceByPosition ?? [];
  const perPosStr = perPos.map((p) => p.toFixed(3)).join(', ');
  const meanAcc = mtpPerf.mtpMeanAcceptedTokens ?? 0;
  // Headline: mlx-vlm-comparable mean accepted tokens/cycle, INCLUDING the
  // always-verified token each cycle commits — matches mlx-vlm's
  // `mean_accepted_tokens = (accepted_drafts + rounds)/rounds`. The
  // drafts-only `mean_accepted` is kept alongside as the historical value.
  const meanAccTotal = mtpPerf.mtpMeanAcceptedTokensTotal ?? meanAcc + 1;
  console.log(
    `cycles=${mtpPerf.mtpCycles} ` +
      `mean accepted tokens/cycle (incl. verified, mlx-vlm-comparable)=${meanAccTotal.toFixed(2)} ` +
      `mean_accepted(drafts-only)=${meanAcc.toFixed(2)}/cycle ` +
      `per_position=[${perPosStr}]`,
  );
  console.log(
    'Reference (MTPLX, stock Qwen3.6-27B native MTP heads, T=0, depth=3): ' +
      'per_position≈[0.73, 0.43, 0.17] cycle-history / [0.90, 0.78, 0.62] committed-history.',
  );
  // BLOCKING correctness gate. A real verify-path bug (wrong logits,
  // cache, or rollback state) corrupts the target argmax and collapses
  // acceptance far below the native-head floor. Healthy native heads
  // clear these bounds with wide margin; a broken verifier lands near
  // zero. This is what the relaxed text-parity check delegates to.
  const pos0 = perPos[0] ?? 0;
  if (meanAcc < 0.5 || pos0 < 0.4) {
    acceptanceOk = false;
    console.log(
      `FAIL: acceptance (mean=${meanAcc.toFixed(2)}, pos0=${pos0.toFixed(3)}) is below the ` +
        `native-head floor — indicates a broken verify/draft path.`,
    );
  }
} else {
  acceptanceOk = false;
  console.log('FAIL: no MTP acceptance recorded — mtpCycles missing; the MTP run executed no speculative cycle.');
}

if (!acceptanceOk) {
  process.exit(3);
}
