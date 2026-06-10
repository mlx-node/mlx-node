#!/usr/bin/env node
/**
 * Qwen3.5 int8 W8A8 quant-prefill A/B harness — single-arm measurement primitive.
 *
 * PROBE (Option C): measures the prefill TTFT delta of routing prefill matmuls
 * of an ALREADY-QUANTIZED 8-bit-affine model through the symmetric-int8 W8A8
 * kernel instead of the affine `quantized_matmul`. Same model, same weights —
 * the ONLY difference is the prefill GEMM. This is the go/no-go probe before any
 * new-kernel work; it reuses the existing W8A8 kernel and adds a one-time
 * affine->symmetric int8 requant at LOAD.
 *
 * One invocation = one model load + warmup + N measured reps in ONE
 * thermal/process arm. The ARM (treatment vs baseline) is selected by the
 * caller via the `MLX_INT8_QUANT_PREFILL` env var, which is read at LOAD TIME on
 * the Rust side (`QuantizedLinear::new` -> `build_int8_w8a8_opt` builds the
 * symmetric-int8 weight ONLY when the flag is truthy AND the layer is 8-bit
 * affine). A single loaded model therefore CANNOT be shared across arms — each
 * arm must be a fresh process with the env set/unset accordingly. The paired
 * orchestrator (`examples/qwen35-int8-quant-prefill-pair.py`) does exactly that.
 *
 * Toggle polarity (note: this is a LOAD-TIME OPT-IN, default OFF):
 *   treatment = MLX_INT8_QUANT_PREFILL=1     (symmetric int8 W8A8 prefill ON)
 *   baseline  = MLX_INT8_QUANT_PREFILL unset (affine W8A16 quantized_matmul, default)
 * IMPORTANT: the model MUST be an 8-bit affine quantized checkpoint (e.g.
 * `*-UD-Q8_K_XL-mlx`). On a 4-bit / mxfp* / nvfp4 / bf16 model the treatment
 * arm is a NO-OP (the int8 view is never built) and the A/B reduces to noise.
 * Set `MLX_INT8_QUANT_PREFILL_DEBUG=1` (the orchestrator does) so the native
 * side prints `[int8-quant-prefill] fired: M=.. K=.. N=..` to stderr when the
 * W8A8 path actually executes — this is the only firing proof. The harness
 * tee's that stderr and reports `int8FiredCount` in RESULT_JSON.
 *
 * Metrics come from the native `reportPerformance` path (measured AFTER model
 * load, so load variance does not pollute them).
 *
 * Usage:
 *   [MLX_INT8_QUANT_PREFILL=1 MLX_INT8_QUANT_PREFILL_DEBUG=1] PATH=/usr/bin:$PATH oxnode \
 *     examples/qwen35-int8-quant-prefill-ab.ts \
 *     --model /Volumes/P4510/models/Qwen3.5-0.8B-UD-Q8_K_XL-mlx \
 *     --mode ttft --prompt-tokens 1024 --max-new 4 --reps 4 --warmup 1
 *
 * Output: exactly one line beginning `RESULT_JSON:` followed by JSON.
 */

import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';

import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const DEFAULT_MODEL = '/Volumes/P4510/models/Qwen3.5-0.8B-UD-Q8_K_XL-mlx';

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

// Count native int8-prefill firings (`[int8-quant-prefill] fired:` on stderr)
// without swallowing the stream — we still pass everything through to the real
// stderr so the orchestrator and a human can see it. process.stderr.write is
// patched (not piped) so ordering/flush behaviour is unchanged.
let int8FiredCount = 0;
const FIRED_RE = /\[int8-quant-prefill\] fired:/g;
const origStderrWrite = process.stderr.write.bind(process.stderr);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(process.stderr as any).write = (chunk: any, ...rest: any[]): boolean => {
  try {
    const s = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8');
    const m = s.match(FIRED_RE);
    if (m) int8FiredCount += m.length;
  } catch {
    /* counting is best-effort; never break the write */
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (origStderrWrite as any)(chunk, ...rest);
};

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
// Only count firings during the MEASURED reps (warmup firings are uninteresting
// and would otherwise inflate the per-rep firing-density sanity number).
int8FiredCount = 0;

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
  // >0 only when MLX_INT8_QUANT_PREFILL=1 AND MLX_INT8_QUANT_PREFILL_DEBUG=1 AND
  // the model is 8-bit affine AND M cleared the threshold. The orchestrator uses
  // this to assert the treatment arm actually exercised the W8A8 path.
  int8FiredCount,
  ttftMs,
  prefillTps,
  decodeTps,
  medTtftMs: median(ttftMs),
  medPrefillTps: median(prefillTps),
  medDecodeTps: median(decodeTps),
  ...(emitText ? { textHash: hasher.digest('hex'), firstText: firstText.slice(0, 400) } : {}),
};

console.log(`RESULT_JSON:${JSON.stringify(out)}`);
