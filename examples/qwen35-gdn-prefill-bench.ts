#!/usr/bin/env node
/**
 * Qwen3.5 GDN prefill A/B — per-step recurrence vs chunked-ops (CUDA prefill lever).
 *
 * GDN (gated-delta linear attention) prefill on the non-Metal backend runs an O(T)
 * per-step recurrence. `gated_delta_chunked_ops` collapses it to O(T/64) chunk-serial
 * dense batched matmuls (cuBLAS / tensor cores). This harness measures the prefill TTFT
 * win and gates output parity between the two kernels.
 *
 *   oxnode examples/qwen35-gdn-prefill-bench.ts <model-path> \
 *     --lengths 212,512,1024,2048,4096,10293 --repeats 3 --max-tokens 24 \
 *     --json /tmp/gdn-prefill.json
 *
 * The `MLX_GDN_KERNEL` override is read fresh per call in Rust, so each cell forks one
 * isolated worker with the env set (`perstep` | `chunked_ops`). Runs are interleaved
 * (perstep/chunked back-to-back per repeat) so cross-invocation thermal drift cancels.
 *
 * IMPORTANT: chunked_ops only fires on the non-Metal (`!use_kernel`) branch. On macOS /
 * Metal both kernels resolve to the same Metal kernel, so the A/B delta is ~0 there — this
 * harness produces a real speedup curve only on the CUDA (DGX) build. On Mac it still runs
 * end-to-end (useful as a wiring smoke test; parity is then trivially identical).
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { hostname } from 'node:os';
import { parseArgs } from 'node:util';

import type { ChatConfig, ChatResult } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

type Kernel = 'perstep' | 'chunked_ops';
const KERNELS: Kernel[] = ['perstep', 'chunked_ops'];

type WorkerOut = {
  kernel: Kernel;
  targetLen: number;
  promptTokens: number;
  numTokens: number;
  finishReason: string;
  ttftMs: number | null;
  prefillTps: number | null;
  decodeTps: number | null;
  textSha: string;
  text: string;
};

const sha256 = (s: string) => createHash('sha256').update(s).digest('hex').slice(0, 16);

/**
 * Build a deterministic prompt of roughly `targetTokens` tokens. English averages
 * ~1.3 tokens/word, so word count ≈ targetTokens * 0.75. The exact token count is
 * reported back from the model; the sweep points are approximate by design.
 */
function buildPrompt(targetTokens: number): string {
  const lexicon =
    'the quick brown fox jumps over a lazy dog while gentle rain falls across the quiet valley ' +
    'and distant mountains hold the morning light as rivers carry stories toward the patient sea';
  const words = lexicon.split(' ');
  const want = Math.max(8, Math.round(targetTokens * 0.75));
  const out: string[] = [];
  for (let i = 0; i < want; i++) out.push(words[i % words.length]);
  return 'Summarize the following passage in one short sentence.\n\n' + out.join(' ') + '\n\nSummary:';
}

function greedyConfig(maxTokens: number): ChatConfig {
  return {
    maxNewTokens: maxTokens,
    temperature: 0,
    topK: 1,
    reasoningEffort: 'none',
    includeReasoning: false,
    reportPerformance: true,
  };
}

// ----------------------------------------------------------------------------- worker
async function worker(): Promise<void> {
  const modelPath = process.env.GDN_BENCH_MODEL!;
  const kernel = process.env.MLX_GDN_KERNEL as Kernel;
  const targetLen = Number(process.env.GDN_BENCH_LEN);
  const maxTokens = Number(process.env.GDN_BENCH_MAXTOK ?? '24');

  const loaded = await loadModel(modelPath);
  if (loaded instanceof HarrierModel) throw new Error('Embedding model is not session-capable.');
  const model = loaded as unknown as ConstructorParameters<typeof ChatSession>[0];

  const session = new ChatSession(model, {
    system: 'You are a precise assistant. Be concise.',
    defaultConfig: greedyConfig(maxTokens),
  });
  const result: ChatResult = await session.send(buildPrompt(targetLen));

  const out: WorkerOut = {
    kernel,
    targetLen,
    promptTokens: result.promptTokens,
    numTokens: result.numTokens,
    finishReason: result.finishReason,
    ttftMs: result.performance?.ttftMs ?? null,
    prefillTps: result.performance?.prefillTokensPerSecond ?? null,
    decodeTps: result.performance?.decodeTokensPerSecond ?? null,
    textSha: sha256(result.text),
    text: result.text.slice(0, 240),
  };
  process.stdout.write('\n@@GDN@@' + JSON.stringify(out) + '@@GDN@@\n');
}

// ----------------------------------------------------------------------------- master
function runCell(modelPath: string, kernel: Kernel, targetLen: number, maxTokens: number): WorkerOut | null {
  const child = spawnSync(process.execPath, [process.argv[1], '--worker'], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
    maxBuffer: 64 * 1024 * 1024,
    env: {
      ...process.env,
      MLX_GDN_KERNEL: kernel,
      GDN_BENCH_MODEL: modelPath,
      GDN_BENCH_LEN: String(targetLen),
      GDN_BENCH_MAXTOK: String(maxTokens),
    },
  });
  if (child.status !== 0) {
    console.error(`  [cell failed] kernel=${kernel} len=${targetLen} status=${child.status}`);
    return null;
  }
  const m = child.stdout.match(/@@GDN@@(.*)@@GDN@@/s);
  if (!m) {
    console.error(`  [no result marker] kernel=${kernel} len=${targetLen}`);
    return null;
  }
  return JSON.parse(m[1]) as WorkerOut;
}

const median = (xs: number[]): number | null => {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length === 0) return null;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
};

// Parity verdict between greedy per-step and chunked outputs.
// chunked-ops is a different algorithm (not bit-exact: ~1e-2 fp32 vs per-step), so a late
// token flip is fp drift (acceptable). An EARLY flip means a real chunk-math bug.
function parity(ps: string, ch: string): { verdict: 'identical' | 'late-drift' | 'EARLY-DIVERGE'; firstDiff: number } {
  let i = 0;
  while (i < ps.length && i < ch.length && ps[i] === ch[i]) i++;
  if (i === ps.length && i === ch.length) return { verdict: 'identical', firstDiff: -1 };
  return { verdict: i < 24 ? 'EARLY-DIVERGE' : 'late-drift', firstDiff: i };
}

async function master(): Promise<void> {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    options: {
      lengths: { type: 'string', default: '212,512,1024,2048,4096,10293' },
      repeats: { type: 'string', default: '3' },
      'max-tokens': { type: 'string', default: '24' },
      json: { type: 'string' },
    },
  });
  const modelPath = positionals[0];
  if (!modelPath) {
    console.error('usage: qwen35-gdn-prefill-bench.ts <model-path> [--lengths a,b,c] [--repeats N] [--max-tokens N] [--json path]');
    process.exit(2);
  }
  const lengths = values.lengths!.split(',').map((s) => Number(s.trim())).filter(Number.isFinite);
  const repeats = Number(values.repeats);
  const maxTokens = Number(values['max-tokens']);

  console.error(`host=${hostname()} model=${modelPath}`);
  console.error(`lengths=${lengths.join(',')} repeats=${repeats} maxTokens=${maxTokens}\n`);

  type Row = {
    targetLen: number;
    promptTokens: number;
    ttft: Record<Kernel, number | null>;
    prefillTps: Record<Kernel, number | null>;
    speedup: number | null;
    parity: ReturnType<typeof parity>;
    samples: WorkerOut[];
  };
  const rows: Row[] = [];

  for (const targetLen of lengths) {
    const byKernel: Record<Kernel, WorkerOut[]> = { perstep: [], chunked_ops: [] };
    // Interleave perstep/chunked per repeat so thermal drift cancels in the paired ratio.
    for (let r = 0; r < repeats; r++) {
      for (const kernel of KERNELS) {
        const o = runCell(modelPath, kernel, targetLen, maxTokens);
        if (o) byKernel[kernel].push(o);
        const tag = o ? `ttft=${o.ttftMs?.toFixed(1)}ms prefill=${o.prefillTps?.toFixed(0)}tok/s ptok=${o.promptTokens}` : 'FAILED';
        console.error(`  len~${targetLen} rep${r} ${kernel.padEnd(11)} ${tag}`);
      }
    }
    const psTtft = median(byKernel.perstep.map((o) => o.ttftMs ?? NaN));
    const chTtft = median(byKernel.chunked_ops.map((o) => o.ttftMs ?? NaN));
    const lastPs = byKernel.perstep.at(-1);
    const lastCh = byKernel.chunked_ops.at(-1);
    rows.push({
      targetLen,
      promptTokens: lastPs?.promptTokens ?? lastCh?.promptTokens ?? 0,
      ttft: { perstep: psTtft, chunked_ops: chTtft },
      prefillTps: {
        perstep: median(byKernel.perstep.map((o) => o.prefillTps ?? NaN)),
        chunked_ops: median(byKernel.chunked_ops.map((o) => o.prefillTps ?? NaN)),
      },
      speedup: psTtft != null && chTtft != null && chTtft > 0 ? psTtft / chTtft : null,
      parity: lastPs && lastCh ? parity(lastPs.text, lastCh.text) : { verdict: 'EARLY-DIVERGE', firstDiff: 0 },
      samples: [...byKernel.perstep, ...byKernel.chunked_ops],
    });
    console.error('');
  }

  // ---- report ----
  console.log('\nGDN prefill A/B — per-step vs chunked_ops  (TTFT median, lower is better)\n');
  console.log('ptok    perstep-ttft  chunked-ttft  speedup   ps-prefill   ch-prefill   parity');
  console.log('------  ------------  ------------  --------  -----------  -----------  ----------------');
  let anyEarly = false;
  for (const r of rows) {
    if (r.parity.verdict === 'EARLY-DIVERGE') anyEarly = true;
    const f = (x: number | null, u = '') => (x == null ? '   n/a   ' : x.toFixed(u === 'ms' ? 1 : 0) + u);
    console.log(
      [
        String(r.promptTokens).padStart(6),
        (f(r.ttft.perstep, 'ms')).padStart(12),
        (f(r.ttft.chunked_ops, 'ms')).padStart(12),
        (r.speedup == null ? '  n/a ' : r.speedup.toFixed(2) + 'x').padStart(8),
        (f(r.prefillTps.perstep)).padStart(11),
        (f(r.prefillTps.chunked_ops)).padStart(11),
        `${r.parity.verdict}${r.parity.firstDiff >= 0 ? '@' + r.parity.firstDiff : ''}`,
      ].join('  '),
    );
  }

  if (values.json) {
    await writeFile(values.json, JSON.stringify({ host: hostname(), modelPath, rows }, null, 2));
    console.error(`\nwrote ${values.json}`);
  }

  if (anyEarly) {
    console.error('\nPARITY GATE FAILED: chunked_ops diverges early from per-step (likely a chunk-math bug).');
    process.exit(1);
  }
  console.error('\nPARITY GATE PASSED (chunked_ops matches per-step or differs only by late fp drift).');
}

if (process.argv.includes('--worker')) {
  worker().catch((e) => {
    console.error(e);
    process.exit(1);
  });
} else {
  master().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
