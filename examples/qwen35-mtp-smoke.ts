#!/usr/bin/env node
/**
 * MTP speculative-decode smoke benchmark.
 *
 * Runs the same prompt twice on a Qwen3.5 / Qwen3.5-MoE checkpoint that
 * carries MTP heads — once with the speculative loop ON, once OFF —
 * at T=0 so the outputs must match token-for-token (parity gate).
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

// Warmup — first run pays compile + cache costs. Same prompt both modes.
if (!skipWarmup) {
  console.log('\n--- Warmup (each mode) ---');
  await runOnce('warmup AR', false);
  await runOnce('warmup MTP', true);
}

console.log('\n--- Measured ---');
const ar = await runOnce('measured AR', false);
const mtp = await runOnce('measured MTP', true);

const arText = ar.text;
const mtpText = mtp.text;
const parity = arText === mtpText;

console.log('\n--- Parity (T=0) ---');
if (parity) {
  console.log(`OK: AR and MTP produced identical output (${arText.length} chars).`);
} else {
  console.log(`MISMATCH: AR and MTP outputs differ.`);
  console.log(`AR  first 200 chars: ${JSON.stringify(arText.slice(0, 200))}`);
  console.log(`MTP first 200 chars: ${JSON.stringify(mtpText.slice(0, 200))}`);

  // Show first divergence offset
  let i = 0;
  while (i < Math.min(arText.length, mtpText.length) && arText[i] === mtpText[i]) i++;
  console.log(`Diverged at character offset: ${i}`);
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

console.log('\n--- MTP acceptance ---');
const mtpPerf = mtp.performance;
if (mtpPerf?.mtpCycles != null) {
  const perPos = mtpPerf.mtpAcceptanceByPosition ?? [];
  const perPosStr = perPos.map((p) => p.toFixed(3)).join(', ');
  console.log(
    `cycles=${mtpPerf.mtpCycles} ` +
      `mean_accepted=${(mtpPerf.mtpMeanAcceptedTokens ?? 0).toFixed(2)}/cycle ` +
      `per_position=[${perPosStr}]`,
  );
  console.log('Reference (MTPLX, stock Qwen3.6-27B native MTP heads, depth=3): per_position≈[1.00, 0.98, 0.94].');
} else {
  console.log(
    'No MTP acceptance recorded — mtpCycles is missing. The MTP run may not have executed any speculative cycle.',
  );
}

if (!parity) {
  process.exit(3);
}
