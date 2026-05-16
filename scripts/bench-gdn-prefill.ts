// Qwen3.5-4B-mlx prefill latency benchmark.
//
// Used by autoresearch.sh as the experiment runner. Loads the model once,
// warms up, then runs N timed prefill measurements and emits:
//   METRIC prefill_ms_median=...
//   METRIC prefill_ms_min=...
//   METRIC max_abs_diff=...     (1.0 if first-N decoded tokens differ from
//                                 baseline, else 0.0)
//   METRIC peak_mem_gb=...
//
// Timing strategy:
//   - We request maxNewTokens=1 with temperature=0 and time the entire
//     sendStream call. Prefill dominates this (decode of 1 token is ~10 ms).
//   - For correctness, the experiment is also run once with maxNewTokens=8
//     and the concatenated raw text is compared to baseline.
//
// Usage:
//   oxnode scripts/bench-gdn-prefill.ts \
//     --model .cache/models/Qwen3.5-4B-mlx \
//     --prompt-tokens 1024 \
//     --runs 5 \
//     --warmup 2 \
//     --baseline .cache/autoresearch-baseline.json \
//     --out /tmp/bench-result.json

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

interface BenchResult {
  prefillMsMedian: number;
  prefillMsMin: number;
  prefillMsMax: number;
  prefillMsAll: number[];
  fingerprintText: string;
  promptTokenCount: number;
  peakMemGb: number;
}

interface BaselineSnapshot {
  fingerprintText: string;
  promptTokenCount: number;
}

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    'prompt-tokens': { type: 'string', default: '1024' },
    runs: { type: 'string', default: '5' },
    warmup: { type: 'string', default: '2' },
    'fingerprint-tokens': { type: 'string', default: '8' },
    baseline: { type: 'string' },
    'write-baseline': { type: 'boolean', default: false },
    out: { type: 'string' },
  },
});

const modelPath = values.model;
if (!modelPath) {
  console.error('ERROR: --model is required');
  process.exit(2);
}
const promptTokens = parseInt(values['prompt-tokens']!, 10);
const runs = parseInt(values.runs!, 10);
const warmup = parseInt(values.warmup!, 10);
const fingerprintTokens = parseInt(values['fingerprint-tokens']!, 10);

// Deterministic prompt: repeat a fixed phrase. Greedy sampling on the same
// prompt with the same weights produces identical output every run.
const PHRASE =
  'The quick brown fox jumps over the lazy dog while the curious cat watches from the windowsill. ';
const approxChars = promptTokens * 4;
const repeats = Math.ceil(approxChars / PHRASE.length);
const PROMPT = PHRASE.repeat(repeats);

async function timeStream(
  session: ChatSession,
  prompt: string,
  maxNewTokens: number,
): Promise<{ ms: number; text: string; promptTokenCount: number }> {
  const start = process.hrtime.bigint();
  const stream = session.sendStream(prompt, {
    config: { maxNewTokens, temperature: 0 },
  });

  let firstDeltaNs: bigint | null = null;
  let text = '';
  let promptTokenCount = -1;

  for await (const event of stream) {
    if (firstDeltaNs === null) {
      firstDeltaNs = process.hrtime.bigint();
    }
    if (event.done === true) {
      promptTokenCount = event.promptTokens;
      // Prefer rawText if present (matches what we sent to the model)
      text = event.rawText ?? text;
      break;
    } else {
      text += event.text ?? '';
    }
  }

  if (firstDeltaNs === null) throw new Error('stream produced no events');
  return {
    ms: Number(firstDeltaNs - start) / 1_000_000,
    text,
    promptTokenCount,
  };
}

function median(xs: number[]): number {
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

async function freshSession(model: SessionCapableModel): Promise<ChatSession> {
  return new ChatSession(model, {
    system: 'You are a helpful assistant. Be concise.',
  });
}

async function main() {
  console.error(`[bench] loading model: ${modelPath}`);
  const model = (await loadModel(resolve(modelPath))) as unknown as SessionCapableModel;

  console.error('[bench] kernel warmup...');
  for (let i = 0; i < warmup; i++) {
    const s = await freshSession(model);
    await timeStream(s, PROMPT, 1);
  }

  console.error(`[bench] timing ${runs} prefill runs (prompt ~${promptTokens} tokens)`);
  const prefills: number[] = [];
  let observedPromptTokens = 0;

  for (let i = 0; i < runs; i++) {
    const session = await freshSession(model);
    const { ms, promptTokenCount } = await timeStream(session, PROMPT, 1);
    prefills.push(ms);
    if (i === 0) observedPromptTokens = promptTokenCount;
    console.error(`[bench]   run ${i + 1}/${runs}: ${ms.toFixed(2)} ms`);
  }

  // Correctness fingerprint: separate run, generate N tokens
  console.error(`[bench] capturing fingerprint (${fingerprintTokens} tokens)`);
  const fpSession = await freshSession(model);
  const { text: fingerprintText } = await timeStream(fpSession, PROMPT, fingerprintTokens);
  console.error(`[bench]   fingerprint: ${JSON.stringify(fingerprintText.slice(0, 80))}`);

  const result: BenchResult = {
    prefillMsMedian: median(prefills),
    prefillMsMin: Math.min(...prefills),
    prefillMsMax: Math.max(...prefills),
    prefillMsAll: prefills,
    fingerprintText,
    promptTokenCount: observedPromptTokens,
    peakMemGb: process.memoryUsage.rss() / 1024 ** 3,
  };

  if (values['write-baseline'] && values.baseline) {
    const snap: BaselineSnapshot = {
      fingerprintText,
      promptTokenCount: observedPromptTokens,
    };
    mkdirSync(dirname(values.baseline), { recursive: true });
    writeFileSync(values.baseline, JSON.stringify(snap, null, 2));
    console.error(`[bench] wrote baseline → ${values.baseline}`);
  }

  let maxAbsDiff = 0;
  let mismatch = false;
  if (values.baseline && !values['write-baseline'] && existsSync(values.baseline)) {
    const baseline: BaselineSnapshot = JSON.parse(readFileSync(values.baseline, 'utf8'));
    if (baseline.fingerprintText !== fingerprintText) {
      mismatch = true;
      maxAbsDiff = 1;
      console.error(
        `[bench] MISMATCH:\n  baseline: ${JSON.stringify(baseline.fingerprintText)}\n  current:  ${JSON.stringify(fingerprintText)}`,
      );
    }
  }

  if (values.out) {
    mkdirSync(dirname(values.out), { recursive: true });
    writeFileSync(values.out, JSON.stringify(result, null, 2));
  }

  console.log(`METRIC prefill_ms_median=${result.prefillMsMedian.toFixed(4)}`);
  console.log(`METRIC prefill_ms_min=${result.prefillMsMin.toFixed(4)}`);
  console.log(`METRIC prefill_ms_max=${result.prefillMsMax.toFixed(4)}`);
  console.log(`METRIC max_abs_diff=${maxAbsDiff}`);
  console.log(`METRIC peak_mem_gb=${result.peakMemGb.toFixed(3)}`);
  console.log(`METRIC prompt_token_count=${result.promptTokenCount}`);

  if (mismatch) {
    console.error('[bench] FAIL: correctness gate failed (fingerprint differs from baseline)');
    process.exit(3);
  }
}

main().catch((e) => {
  console.error('[bench] error:', e);
  process.exit(1);
});
