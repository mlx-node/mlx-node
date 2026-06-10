#!/usr/bin/env node
/**
 * GDN in_proj_qkvz int8 W8A8 prefill — single-arm COHERENCE probe.
 *
 * Generates ~120 greedy (T=0) tokens from a long (>=256-token) prompt so the
 * int8 qkvz prefill branch is eligible (M = prompt_tokens >= MLX_INT8_PREFILL_MIN_M
 * default 256). The arm (int8-qkvz vs bf16 baseline) is selected by the caller
 * via the `MLX_INT8_PREFILL_QKVZ` env var, read at LOAD TIME on the Rust side
 * (`GatedDeltaNet::finalize_in_proj` quantizes the qkvz weight to int8 ONLY when
 * the flag is truthy at load). A single loaded model therefore CANNOT serve both
 * arms — each arm must be a fresh process with the env set/unset.
 *
 * Set `MLX_INT8_PREFILL_QKVZ_DEBUG=1` to confirm the int8 qkvz branch fired on
 * prefill (it prints `[int8-prefill-qkvz] fired: ...` to stderr) and did NOT
 * fire on decode (M=1 < min_m).
 *
 * Emits exactly one `RESULT_JSON:` line with the generated token ids + text so a
 * bash orchestrator can diff the two arms for token-agreement / coherence.
 *
 * Usage:
 *   MLX_INT8_PREFILL_QKVZ=1 MLX_INT8_PREFILL_QKVZ_DEBUG=1 PATH=/usr/bin:$PATH \
 *     oxnode examples/qwen35-qkvz-int8-coherence.ts \
 *     --model /Volumes/P4510/.cache/models/qwen3.5-4b --max-new 120
 */

import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const DEFAULT_MODEL = '/Volumes/P4510/.cache/models/qwen3.5-4b';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: DEFAULT_MODEL },
    'max-new': { type: 'string', default: '120' },
  },
});

const modelPath = values.model!;
const maxNew = Number.parseInt(values['max-new']!, 10);

// A long, on-topic prompt (>=256 tokens) so prefill M crosses the int8 min_m
// threshold. The continuation must stay coherent if qkvz int8 is correct.
const SENT =
  'The history of computing stretches back through centuries of mechanical invention and mathematical insight. ';
const prompt =
  'Read the following passage carefully and then write a clear, coherent multi-paragraph continuation that stays strictly on topic.\n' +
  SENT.repeat(48) +
  '\nNow continue the passage in your own words, keeping it focused and grammatical.';

const qkvzOn = (() => {
  const v = (process.env.MLX_INT8_PREFILL_QKVZ ?? '').trim();
  return v !== '' && v !== '0' && v.toLowerCase() !== 'false';
})();

const loaded = await loadModel(modelPath);

const session = new ChatSession(loaded as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant.',
});

const res = await session.send(prompt, {
  config: {
    maxNewTokens: maxNew,
    temperature: 0,
    reportPerformance: true,
    reuseCache: false,
    // Surface token ids if the API exposes them; text is the load-bearing signal.
  },
});

// qwen3.5-4b is a reasoning model — most of the greedy output lands in
// `rawText` (full decoded stream incl. any <think>…</think>) and `thinking`;
// `text` is the post-reasoning answer. `rawText` is the load-bearing
// token-by-token signal for an A/B coherence diff.
const out = {
  arm: qkvzOn ? 'int8-qkvz' : 'bf16-baseline',
  model: modelPath,
  maxNew,
  promptTokens: res.promptTokens ?? null,
  numTokens: res.numTokens ?? null,
  reasoningTokens: res.reasoningTokens ?? null,
  finishReason: res.finishReason ?? null,
  ttftMs: res.performance?.ttftMs ?? null,
  prefillTps: res.performance?.prefillTokensPerSecond ?? null,
  rawText: res.rawText ?? '',
  text: res.text ?? '',
  thinking: res.thinking ?? '',
};

console.log(`RESULT_JSON:${JSON.stringify(out)}`);
