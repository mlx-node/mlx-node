#!/usr/bin/env node
/**
 * Qwen3.5 int8 W8A8 prefill A/B harness — single-arm measurement primitive.
 *
 * Stage 4 of the int8 W8A8 prefill GEMM integration: measures the REALIZED
 * prefill TTFT win from the MLP-only int8 path (Stage 3, landed) on the real
 * bf16 Qwen3.5-4B.
 *
 * One invocation = one model load + warmup + N measured reps in ONE
 * thermal/process arm. The ARM (treatment vs baseline) is selected by the
 * caller via the `MLX_INT8_PREFILL` env var, read at LOAD TIME on the Rust
 * side (`MLP::finalize_gate_up` quantizes the gate/up/down weights to int8
 * ONLY when the flag is truthy at load). A single loaded model therefore
 * CANNOT be shared across arms — each arm must be a fresh process with the
 * env set/unset accordingly. The paired orchestrator
 * (`examples/qwen35-int8-prefill-pair.py`) does exactly that.
 *
 * Toggle polarity (note: OPPOSITE of the lfm2 DISABLE-style toggles):
 *   treatment = MLX_INT8_PREFILL=1     (int8 MLP path ON)
 *   baseline  = MLX_INT8_PREFILL unset (bf16 fused path, unchanged default)
 *
 * Metrics come from the native `reportPerformance` path (measured AFTER
 * model load, so load variance does not pollute them).
 *
 * Usage:
 *   [MLX_INT8_PREFILL=1] PATH=/usr/bin:$PATH oxnode \
 *     examples/qwen35-int8-prefill-ab.ts \
 *     --model /Volumes/P4510/.cache/models/qwen3.5-4b \
 *     --mode ttft --prompt-tokens 1024 --max-new 4 --reps 4 --warmup 1
 *
 * Output: exactly one line beginning `RESULT_JSON:` followed by JSON.
 */

import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const DEFAULT_MODEL = '/Volumes/P4510/.cache/models/qwen3.5-4b';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: DEFAULT_MODEL },
    mode: { type: 'string', default: 'ttft' }, // 'ttft' | 'decode'
    'prompt-tokens': { type: 'string', default: '1024' },
    'max-new': { type: 'string', default: '4' },
    reps: { type: 'string', default: '4' },
    warmup: { type: 'string', default: '1' },
    'emit-text': { type: 'boolean', default: false },
  },
});

const modelPath = values.model!;
const mode = values.mode!;
const promptTokens = Number.parseInt(values['prompt-tokens']!, 10);
const maxNew = Number.parseInt(values['max-new']!, 10);
const reps = Number.parseInt(values.reps!, 10);
const warmup = Number.parseInt(values.warmup!, 10);
const emitText = values['emit-text']!;

// Neutral prose, ~16 tokens/sentence, so `copies = ceil(promptTokens/16)`
// builds a prompt of roughly `promptTokens` tokens.
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

// Snapshot the relevant int8 env so the orchestrator's RESULT_JSON records
// exactly which arm produced these numbers.
const relevantToggles: Record<string, string> = {};
for (const [k, v] of Object.entries(process.env)) {
  if (k.startsWith('MLX_INT8_') || k === 'MLX_NO_COMPILE' || k === 'MLX_DISABLE_COMPILE') {
    relevantToggles[k] = v ?? '';
  }
}

const loaded = await loadModel(modelPath);

async function oneTurn(
  nonce: string,
): Promise<{ ttftMs: number; prefillTps: number; decodeTps: number; text: string; promptTok: number }> {
  // Fresh session per turn → turn-1 cold prefill (no warm-continue confound).
  // reuseCache:false ensures no cross-turn cache reuse leaks into prefill.
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant.',
  });
  const res = await session.send(buildPrompt(nonce), {
    config: { maxNewTokens: maxNew, temperature: 0, reportPerformance: true, reuseCache: false },
  });
  const p = res.performance;
  return {
    ttftMs: p?.ttftMs ?? Number.NaN,
    prefillTps: p?.prefillTokensPerSecond ?? Number.NaN,
    decodeTps: p?.decodeTokensPerSecond ?? Number.NaN,
    text: res.text ?? '',
    promptTok: res.promptTokens ?? Number.NaN,
  };
}

for (let i = 0; i < warmup; i++) await oneTurn(`warmup-${i} `);

const ttftMs: number[] = [];
const prefillTps: number[] = [];
const decodeTps: number[] = [];
let firstText = '';
let promptTokActual = Number.NaN;
const hasher = createHash('sha256');

for (let r = 0; r < reps; r++) {
  // ttft: unique nonce per rep → cold prefill (miss any content-addressed
  // prefix cache, including cross-process) so we measure real prefill cost.
  // decode: decodeTps is cache-independent; keep prompt FIXED for determinism.
  const nonce = mode === 'ttft' ? `rep-${r} session-${process.pid} ` : '';
  const t = await oneTurn(nonce);
  ttftMs.push(t.ttftMs);
  prefillTps.push(t.prefillTps);
  decodeTps.push(t.decodeTps);
  if (r === 0) {
    firstText = t.text;
    promptTokActual = t.promptTok;
  }
  hasher.update(t.text);
}

const out = {
  model: modelPath,
  mode,
  promptTokens,
  promptTokensActual: promptTokActual,
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
