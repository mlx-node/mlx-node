#!/usr/bin/env node
/**
 * Cold-cache MTP acceptance probe.
 *
 * The standard smoke (qwen35-mtp-smoke.ts) runs `measured AR` BEFORE
 * `measured MTP` on the SAME prompt with a SHARED model, so by the MTP
 * turn the paged KV cache is warm (`cached_prefix_len > 0`). On the paged
 * path that makes `want_prompt_hidden == false`, so `prefill_mtp_commit`
 * never fires and the committed-history MTP cache starts EMPTY — which
 * structurally depresses paged acceptance below dense.
 *
 * This probe isolates the variable: it runs MTP as the FIRST turn the
 * model ever sees (truly cold cache, cached_prefix_len == 0), so on the
 * paged path the prompt-prefill seed DOES run. If the cold-cache paged
 * acceptance jumps toward the committed-history profile [0.854, 0.715,
 * 0.585] / ~2.15-mean while the warm smoke sits at ~1.25, the prefill
 * boost is real and the smoke comparison was confounded.
 *
 *   MLX_QWEN35_PAGED_OVERRIDE=1 oxnode examples/qwen35-mtp-cold-cache-probe.ts <model> [depth] [maxTokens]
 *   oxnode examples/qwen35-mtp-cold-cache-probe.ts <model> [depth] [maxTokens]   # dense reference
 */
import { resolve } from 'node:path';

import type { ChatConfig, SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

const modelName = process.argv[2] || 'qwen3.6-27b-nvfp4-mtp';
const depth = Number(process.argv[3] ?? '3');
const maxTokens = Number(process.argv[4] ?? '120');
const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

const paged = process.env.MLX_QWEN35_PAGED_OVERRIDE === '1';
console.log(`Loading ${modelName} (paged=${paged}, depth=${depth}, maxTokens=${maxTokens})`);
const loaded = await loadModel(MODEL_PATH);
if (loaded instanceof HarrierModel) {
  console.error('Embedding model is not session-capable.');
  process.exit(1);
}
const model = loaded as unknown as SessionCapableModel;
if (!(typeof model.hasMtpWeights === 'function' && model.hasMtpWeights())) {
  console.error(`Model ${modelName} does not carry MTP heads.`);
  process.exit(2);
}

// argv[5] optionally overrides the prompt. The default is a novel prose
// essay (hard / prompt-dependent acceptance). Pass a predictable prompt
// (e.g. "Count from 1 to 100...") to probe the depth-3 acceptance CEILING:
// correctly-loaded native MTP heads hit ~2.9/cycle on counting regardless
// of checkpoint, so a collapsed ceiling on a predictable prompt is a
// loading/dequant defect, not prompt-dependence.
const prompt =
  process.argv[5] ||
  'Write a concise three-paragraph essay on why deterministic sampling at temperature 0 is useful for testing speculative decoding implementations.';
const baseConfig: ChatConfig = {
  temperature: 0,
  topK: 1,
  topP: 1,
  maxNewTokens: maxTokens,
  reasoningEffort: 'none',
  includeReasoning: false,
  mtpDepth: depth,
  reportPerformance: true,
};

function report(label: string, perf: NonNullable<Awaited<ReturnType<ChatSession['send']>>['performance']>): void {
  if (perf?.mtpCycles == null) {
    console.log(`${label}: NO MTP CYCLES (mtpCycles missing)`);
    return;
  }
  const perPos = (perf.mtpAcceptanceByPosition ?? []).map((p) => p.toFixed(3)).join(', ');
  const mean = (perf.mtpMeanAcceptedTokens ?? 0).toFixed(2);
  const tps = perf.decodeTokensPerSecond.toFixed(2);
  console.log(
    `${label}: cycles=${perf.mtpCycles} mean_accepted=${mean}/cycle per_position=[${perPos}] decode=${tps} tok/s`,
  );
}

// COLD CACHE: MTP is the very first turn this model instance sees.
const session = new ChatSession(model, {
  system: 'You are a precise assistant. Be concise.',
  defaultConfig: { ...baseConfig, enableMtp: true },
});
const mtp = await session.send(prompt);
report(`COLD MTP (paged=${paged})`, mtp.performance!);
