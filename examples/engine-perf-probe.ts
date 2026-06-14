#!/usr/bin/env node
/**
 * engine-perf-probe — family-agnostic single-arm perf primitive for the
 * cross-binary (branch vs origin/main artifact) perf-parity oracle.
 *
 * One invocation = one model load + warmup + N measured turns under WHATEVER
 * native trio currently sits in packages/core/. The ARM (branch vs baseline)
 * is selected by the orchestrator swapping the .node + metallibs BEFORE launch
 * — there is NO in-binary toggle (this branch carries no legacy /
 * MLX_ENGINE_LEGACY path; the baseline is the real origin/main artifact).
 *
 * Metrics come from the native reportPerformance path (measured AFTER load, so
 * cold-mmap load variance does not pollute decode/ttft).
 *
 * Usage:
 *   oxnode examples/engine-perf-probe.ts \
 *     --model-path /Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
 *     --mode decode|ttft --prompt-tokens 512 --max-new 256 --reps 4 --warmup 1 \
 *     [--emit-text] [--label branch|baseline]
 *
 * Output: exactly one line `RESULT_JSON:{...}`.
 */
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    'model-path': { type: 'string' },
    mode: { type: 'string', default: 'decode' }, // 'ttft' | 'decode'
    'prompt-tokens': { type: 'string', default: '512' },
    'max-new': { type: 'string', default: '256' },
    reps: { type: 'string', default: '4' },
    warmup: { type: 'string', default: '1' },
    'emit-text': { type: 'boolean', default: false },
    label: { type: 'string', default: '' },
  },
});

const modelPath = values['model-path'];
if (!modelPath) {
  console.error('engine-perf-probe: --model-path is required');
  process.exit(2);
}
const mode = values.mode!;
const promptTokens = Number.parseInt(values['prompt-tokens']!, 10);
const maxNew = Number.parseInt(values['max-new']!, 10);
const reps = Number.parseInt(values.reps!, 10);
const warmup = Number.parseInt(values.warmup!, 10);
const emitText = values['emit-text']!;

const SENT =
  'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
function buildPrompt(nonce: string): string {
  const copies = Math.max(1, Math.ceil(promptTokens / 16));
  return `${nonce}Read the following text and then answer in detail.\n${SENT.repeat(copies)}\nNow write a long continuation.`;
}

function median(xs: number[]): number {
  const f = xs.filter((x) => Number.isFinite(x));
  if (f.length === 0) return Number.NaN;
  const s = [...f].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const loaded = await loadModel(modelPath);

async function oneTurn(nonce: string): Promise<{
  ttftMs: number;
  prefillTps: number;
  decodeTps: number;
  text: string;
  thinking: string;
  rawText: string;
}> {
  // Fresh session per turn → turn-1 cold prefill (no warm-continue confound).
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant.',
  });
  const res = await session.send(buildPrompt(nonce), {
    config: { maxNewTokens: maxNew, temperature: 0, reportPerformance: true },
  });
  const p = res.performance;
  const r = res as unknown as { thinking?: string | null; rawText?: string };
  return {
    ttftMs: p?.ttftMs ?? Number.NaN,
    prefillTps: p?.prefillTokensPerSecond ?? Number.NaN,
    decodeTps: p?.decodeTokensPerSecond ?? Number.NaN,
    text: res.text ?? '',
    thinking: r.thinking ?? '',
    // rawText = full undecorated generation (text + thinking + tool spans) →
    // the robust byte-identity field for cross-arm parity checks.
    rawText: r.rawText ?? res.text ?? '',
  };
}

for (let i = 0; i < warmup; i++) await oneTurn(`warmup-${i} `);

const ttftMs: number[] = [];
const prefillTps: number[] = [];
const decodeTps: number[] = [];
let firstText = '';
let firstThinking = '';
const hasher = createHash('sha256');

for (let r = 0; r < reps; r++) {
  // ttft: unique nonce per rep → cold prefill (miss the content-addressed
  // prefix cache) so we measure real prefill cost. decode: keep prompt FIXED
  // so --emit-text byte-checks are deterministic across arms.
  const nonce = mode === 'ttft' ? `rep-${r} ` : '';
  const t = await oneTurn(nonce);
  ttftMs.push(t.ttftMs);
  prefillTps.push(t.prefillTps);
  decodeTps.push(t.decodeTps);
  if (r === 0) {
    firstText = t.text;
    firstThinking = t.thinking;
  }
  hasher.update(t.rawText);
}

const out = {
  modelPath,
  label: values.label,
  mode,
  promptTokens,
  maxNew,
  reps,
  warmup,
  ttftMs,
  prefillTps,
  decodeTps,
  medTtftMs: median(ttftMs),
  medPrefillTps: median(prefillTps),
  medDecodeTps: median(decodeTps),
  ...(emitText
    ? {
        rawTextHash: hasher.digest('hex'),
        firstText: firstText.slice(0, 300),
        firstThinking: firstThinking.slice(0, 300),
      }
    : {}),
};

console.log(`RESULT_JSON:${JSON.stringify(out)}`);
