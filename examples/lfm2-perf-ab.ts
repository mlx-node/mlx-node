#!/usr/bin/env node
/**
 * lfm2 perf A/B harness — single-arm measurement primitive.
 *
 * One invocation = one model load + warmup + N measured reps in ONE
 * thermal/process arm. The ARM (baseline vs optimized) is selected by the
 * caller via a `MLX_LFM2_DISABLE_<OPT>` env var, read once at process
 * start on the Rust side (the idiomatic toggle pattern in this repo).
 *
 * The thermally-fair A/B is done by the orchestrator
 * (`examples/lfm2-perf-pair.py`), which launches this script alternately
 * with/without the toggle env, pairs adjacent runs, and takes the median
 * of per-pair ratios (drift-canceling) plus a control band.
 *
 * Metrics come from the native `reportPerformance` path (measured AFTER
 * model load, so load variance does not pollute them).
 *
 * Usage:
 *   [MLX_LFM2_DISABLE_X=1] oxnode examples/lfm2-perf-ab.ts \
 *     --model lfm2.5-1.2b-thinking-mlx --mode ttft|decode \
 *     --prompt-tokens 1500 --max-new 4 --reps 4 --warmup 1 [--emit-text]
 *
 * Output: exactly one line beginning `RESULT_JSON:` followed by JSON.
 */

import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: 'lfm2.5-1.2b-thinking-mlx' },
    mode: { type: 'string', default: 'decode' }, // 'ttft' | 'decode'
    'prompt-tokens': { type: 'string', default: '64' },
    'max-new': { type: 'string', default: '256' },
    reps: { type: 'string', default: '4' },
    warmup: { type: 'string', default: '1' },
    'emit-text': { type: 'boolean', default: false },
  },
});

const modelName = values.model!;
const mode = values.mode!;
const promptTokens = Number.parseInt(values['prompt-tokens']!, 10);
const maxNew = Number.parseInt(values['max-new']!, 10);
const reps = Number.parseInt(values.reps!, 10);
const warmup = Number.parseInt(values.warmup!, 10);
const emitText = values['emit-text']!;

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

const SENT = 'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
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

const relevantToggles: Record<string, string> = {};
for (const [k, v] of Object.entries(process.env)) {
  if (k.startsWith('MLX_LFM2_') || k === 'MLX_NO_COMPILE' || k === 'MLX_DISABLE_COMPILE') {
    relevantToggles[k] = v ?? '';
  }
}

const loaded = await loadModel(MODEL_PATH);

async function oneTurn(
  nonce: string,
): Promise<{ ttftMs: number; prefillTps: number; decodeTps: number; text: string }> {
  // Fresh session per turn → turn-1 cold prefill (no warm-continue confound).
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant.',
  });
  const res = await session.send(buildPrompt(nonce), {
    config: { maxNewTokens: maxNew, temperature: 0, reportPerformance: true },
  });
  const p = res.performance;
  return {
    ttftMs: p?.ttftMs ?? Number.NaN,
    prefillTps: p?.prefillTokensPerSecond ?? Number.NaN,
    decodeTps: p?.decodeTokensPerSecond ?? Number.NaN,
    text: res.text ?? '',
  };
}

for (let i = 0; i < warmup; i++) await oneTurn(`warmup-${i} `);

const ttftMs: number[] = [];
const prefillTps: number[] = [];
const decodeTps: number[] = [];
let firstText = '';
const hasher = createHash('sha256');

for (let r = 0; r < reps; r++) {
  // ttft: unique nonce per rep → cold prefill (miss the content-addressed
  // prefix cache) so we measure real prefill cost. decode: decodeTps is
  // cache-independent; keep the prompt FIXED so --emit-text is deterministic
  // across arms for byte-identical checks.
  const nonce = mode === 'ttft' ? `rep-${r} ` : '';
  const t = await oneTurn(nonce);
  ttftMs.push(t.ttftMs);
  prefillTps.push(t.prefillTps);
  decodeTps.push(t.decodeTps);
  if (r === 0) firstText = t.text;
  hasher.update(t.text);
}

const out = {
  model: modelName,
  mode,
  promptTokens,
  maxNew,
  reps,
  warmup,
  toggles: relevantToggles,
  ttftMs,
  prefillTps,
  decodeTps,
  medTtftMs: median(ttftMs),
  medPrefillTps: median(prefillTps),
  medDecodeTps: median(decodeTps),
  ...(emitText ? { textHash: hasher.digest('hex'), firstText: firstText.slice(0, 400) } : {}),
};

console.log(`RESULT_JSON:${JSON.stringify(out)}`);
