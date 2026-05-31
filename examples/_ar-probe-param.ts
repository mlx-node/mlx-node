/**
 * Parameterized AR (autoregressive, MTP-OFF) decode probe.
 *
 * Sibling of examples/_ar-probe.ts, but accepts a maxTokens and prompt
 * argument so the cross-engine A/B harness can drive AR on the SAME
 * prompt/maxTokens it uses for the MTP probe (qwen35-mtp-cold-cache-probe.ts)
 * and for MTPLX --no-mtp. Greedy/deterministic (T=0, topK=1, topP=1).
 *
 *   oxnode examples/_ar-probe-param.ts <model> [maxTokens] [prompt]
 *
 * Prints a single parseable line:
 *   AR <model>: decode=X tok/s | tokens=N
 */
import { resolve } from 'node:path';

import { ChatSession, loadModel } from '@mlx-node/lm';

const name = process.argv[2] ?? 'mtplx-optimized-speed-mtp';
const maxTokens = Number(process.argv[3] ?? '120');
const prompt =
  process.argv[4] ||
  'Write a concise three-paragraph essay on why deterministic sampling at temperature 0 is useful for testing speculative decoding implementations.';
const P = resolve(process.cwd(), '.cache', 'models', name);

const m = (await loadModel(P)) as unknown as ConstructorParameters<typeof ChatSession>[0];
const s = new ChatSession(m, {
  system: 'You are a precise assistant. Be concise.',
  defaultConfig: {
    temperature: 0,
    topK: 1,
    topP: 1,
    maxNewTokens: maxTokens,
    reasoningEffort: 'none',
    includeReasoning: false,
    enableMtp: false,
    reportPerformance: true,
  },
});
const r = await s.send(prompt);
const tps = r.performance?.decodeTokensPerSecond?.toFixed(2) ?? 'NA';
console.log(`AR ${name}: decode=${tps} tok/s | tokens=${r.numTokens}`);
