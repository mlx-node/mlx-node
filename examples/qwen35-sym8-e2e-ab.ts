#!/usr/bin/env node
/**
 * Qwen3.5 sym8 E2E perf harness — single-arm measurement primitive (Phase 4).
 *
 * Unlike the old env-toggle probes, the sym8 A/B compares two DIFFERENT
 * checkpoint directories with the SAME binary:
 *   sym8 arm : a per-output-channel symmetric int8 checkpoint
 *              (config quantization mode == "sym8"; eager Rust forward,
 *              int8 W8A8 GEMM at prefill M>=3, fused int8 qmv at decode M<=2)
 *   base arm : the 8-bit affine baseline checkpoint
 *              (stock quantized_matmul; compiled C++ decode path)
 *
 * One invocation = one model load + warmup + N measured reps in ONE
 * thermal/process arm. Metrics come from the native `reportPerformance` path
 * (measured AFTER model load, so storage/load variance does not pollute them).
 *
 * If MLX_SYM8_DEBUG=1 is set (NOT recommended for timed runs — it eprintlns on
 * every sym8 forward), the harness counts `[sym8] gemm` / `[sym8] qmv` stderr
 * lines and reports sym8GemmCount / sym8QmvCount as dispatch proof.
 *
 * Usage:
 *   PATH=/usr/bin:$PATH oxnode examples/qwen35-sym8-e2e-ab.ts \
 *     --model /tmp/qwen35-4b-sym8-mlx \
 *     --mode ttft --prompt-tokens 1024 --max-new 4 --reps 4 --warmup 1
 *
 * Output: exactly one line beginning `RESULT_JSON:` followed by JSON.
 */

import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const DEFAULT_MODEL = '/tmp/qwen35-4b-sym8-mlx';

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

// Count native sym8 dispatch lines (`[sym8] gemm|qmv ...` on stderr, emitted
// only under MLX_SYM8_DEBUG=1) without swallowing the stream.
let sym8GemmCount = 0;
let sym8QmvCount = 0;
const GEMM_RE = /\[sym8\] gemm /g;
const QMV_RE = /\[sym8\] qmv /g;
const origStderrWrite = process.stderr.write.bind(process.stderr);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(process.stderr as any).write = (chunk: any, ...rest: any[]): boolean => {
  try {
    const s = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8');
    const g = s.match(GEMM_RE);
    if (g) sym8GemmCount += g.length;
    const q = s.match(QMV_RE);
    if (q) sym8QmvCount += q.length;
  } catch {
    /* counting is best-effort; never break the write */
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (origStderrWrite as any)(chunk, ...rest);
};

// Neutral prose, ~16 tokens/sentence, so `copies = ceil(promptTokens/16)`
// builds a prompt of roughly `promptTokens` tokens.
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
  if (k.startsWith('MLX_SYM8_') || k === 'MLX_NO_COMPILE' || k === 'MLX_DISABLE_COMPILE') {
    relevantToggles[k] = v ?? '';
  }
}

const loaded = await loadModel(modelPath);

async function oneTurn(
  nonce: string,
): Promise<{ ttftMs: number; prefillTps: number; decodeTps: number; text: string; promptTok: number }> {
  // Fresh session per turn → turn-1 cold prefill (no warm-continue confound).
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
sym8GemmCount = 0;
sym8QmvCount = 0;

const ttftMs: number[] = [];
const prefillTps: number[] = [];
const decodeTps: number[] = [];
let firstText = '';
let promptTokActual = Number.NaN;
const hasher = createHash('sha256');

for (let r = 0; r < reps; r++) {
  // ttft: unique nonce per rep → cold prefill (miss any content-addressed
  // prefix cache) so we measure real prefill cost.
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
  sym8GemmCount,
  sym8QmvCount,
  ttftMs,
  prefillTps,
  decodeTps,
  medTtftMs: median(ttftMs),
  medPrefillTps: median(prefillTps),
  medDecodeTps: median(decodeTps),
  ...(emitText ? { textHash: hasher.digest('hex'), firstText: firstText.slice(0, 400) } : {}),
};

console.log(`RESULT_JSON:${JSON.stringify(out)}`);
