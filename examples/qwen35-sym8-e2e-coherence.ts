#!/usr/bin/env node
/**
 * Qwen3.5 sym8 E2E COHERENCE gate (Phase 4) — REAL-checkpoint 3-model agreement.
 *
 * Unlike examples/qwen35-fakequant-agreement.ts (which compared bf16 ROUND-TRIP
 * simulations of each quant), this harness loads three REAL checkpoints:
 *   - bf16 : unquantized reference            (real bf16 GEMMs)
 *   - sym8 : per-output-channel symmetric int8 (NEW int8 W8A8 kernels: qmv at
 *            M<=2 decode, GEMM at M>=3 prefill — the Phase-2/3 kernel stack)
 *   - aff8 : 8-bit affine baseline             (stock quantized_matmul path)
 *
 * Greedy T=0, reuseCache=false, ~120 new tokens per prompt. Each model loads in
 * its OWN process (compiled-path weight globals are process-wide). PASS = sym8
 * coherent + its token agreement vs bf16 lands in the same band as aff8's.
 *
 * Usage:
 *   PATH=/usr/bin:$PATH oxnode examples/qwen35-sym8-e2e-coherence.ts --compare \
 *     --bf16 /Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
 *     --sym8 /tmp/qwen35-0.8b-sym8-mlx \
 *     --aff8 /Volumes/P4510/models/Qwen3.5-0.8B-UD-Q8_K_XL-mlx --max-new 120
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

import { MxArray, Qwen3Tokenizer, type ChatMessage } from '@mlx-node/core';
import { loadModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    // worker arm (single model):
    model: { type: 'string' },
    arm: { type: 'string', default: 'unknown' },
    out: { type: 'string' },
    'max-new': { type: 'string', default: '120' },
    // orchestrator:
    compare: { type: 'boolean', default: false },
    bf16: { type: 'string', default: '/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16' },
    sym8: { type: 'string', default: '/tmp/qwen35-0.8b-sym8-mlx' },
    aff8: { type: 'string', default: '/Volumes/P4510/models/Qwen3.5-0.8B-UD-Q8_K_XL-mlx' },
  },
});

const maxNew = Number.parseInt(values['max-new']!, 10);
const SYSTEM = 'You are a helpful assistant.';
const TEXT_HEAD_CHARS = 400;

// ── Six diverse prompts: prose, code, counting, list, factual QA, reasoning QA.
const PROMPTS: { id: string; text: string }[] = [
  {
    id: 'prose',
    text: 'Write a short vivid paragraph describing a thunderstorm rolling over a quiet coastal village at dusk.',
  },
  {
    id: 'code',
    text: 'Write a Python function `is_prime(n)` that returns True if n is prime and False otherwise. Include a short docstring.',
  },
  {
    id: 'counting',
    text: 'Count from 1 to 20, putting each number on its own line.',
  },
  {
    id: 'list',
    text: 'List five practical tips for staying focused while working from home. Use a numbered list.',
  },
  {
    id: 'qa-factual',
    text: 'Explain in two or three sentences why the sky appears blue during the day.',
  },
  {
    id: 'qa-reasoning',
    text: 'A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain your reasoning briefly.',
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// WORKER: load one model, greedily decode every prompt, persist token ids.
// ─────────────────────────────────────────────────────────────────────────────
async function runWorker(): Promise<void> {
  const modelPath = values.model!;
  const tokenizer = await Qwen3Tokenizer.fromPretrained(join(modelPath, 'tokenizer.json'));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const model = (await loadModel(modelPath)) as any;

  const results: {
    id: string;
    nPrompt: number;
    tokenIds: number[];
    numTokens: number;
    finishReason: string;
    idsHash: string;
    textHead: string;
  }[] = [];

  for (const p of PROMPTS) {
    const messages: ChatMessage[] = [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: p.text },
    ];
    const promptIds = await tokenizer.applyChatTemplate(messages, true);
    const nPrompt = promptIds.length;
    const inputArray = MxArray.fromUint32(new Uint32Array(promptIds), BigInt64Array.from([1n, BigInt(nPrompt)]));
    // Greedy (T=0), fresh prompt (no chat-prefix reuse) => deterministic argmax.
    const res = await model.generate(inputArray, {
      maxNewTokens: maxNew,
      temperature: 0,
      reuseCache: false,
    });
    const tokenIds: number[] = Array.from(res.tokens ?? []);
    const text: string = String(res.text ?? '');
    results.push({
      id: p.id,
      nPrompt,
      tokenIds,
      numTokens: res.numTokens ?? tokenIds.length,
      finishReason: String(res.finishReason ?? ''),
      idsHash: createHash('sha256').update(tokenIds.join(',')).digest('hex'),
      textHead: text.slice(0, TEXT_HEAD_CHARS),
    });
  }

  const out = { arm: values.arm, model: modelPath, pid: process.pid, maxNew, results };
  if (values.out) writeFileSync(values.out, JSON.stringify(out));
  console.log(`RESULT_JSON:${JSON.stringify(out)}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// ORCHESTRATOR
// ─────────────────────────────────────────────────────────────────────────────
interface PromptResult {
  id: string;
  nPrompt: number;
  tokenIds: number[];
  numTokens: number;
  finishReason: string;
  idsHash: string;
  textHead: string;
}
interface ArmResult {
  arm: string;
  model: string;
  pid: number;
  maxNew: number;
  results: PromptResult[];
}

function spawnArm(label: string, modelPath: string, outPath: string): Promise<ArmResult> {
  return new Promise((resolve, reject) => {
    const args = [process.argv[1], '--model', modelPath, '--arm', label, '--out', outPath, '--max-new', String(maxNew)];
    const child = spawn(process.execPath, args, {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => {
      stdout += String(d);
    });
    child.stderr.on('data', (d) => {
      stderr += String(d);
    });
    child.on('error', reject);
    child.on('close', (code) => {
      const m = stdout.match(/RESULT_JSON:(\{.*\})\s*$/m);
      let result: ArmResult | null = null;
      if (m) {
        result = JSON.parse(m[1]) as ArmResult;
      } else {
        try {
          result = JSON.parse(readFileSync(outPath, 'utf-8')) as ArmResult;
        } catch {
          result = null;
        }
      }
      if (result == null) {
        reject(
          new Error(
            `[${label}] arm produced no RESULT_JSON (exit=${code}). stderr tail:\n` +
              stderr.split('\n').slice(-25).join('\n'),
          ),
        );
        return;
      }
      resolve(result);
    });
  });
}

function divergenceIndex(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n; // -1 == identical over full common length
}
function agreementRatio(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return Number.NaN;
  let same = 0;
  for (let i = 0; i < n; i++) if (a[i] === b[i]) same++;
  return same / n;
}
function mean(xs: number[]): number {
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : Number.NaN;
}

function compareArm(
  ref: ArmResult,
  arm: ArmResult,
): {
  perPrompt: {
    id: string;
    agreement: number;
    divergenceIndex: number;
    firstTokenSame: boolean;
    refLen: number;
    armLen: number;
    refHead: string;
    armHead: string;
  }[];
  meanAgreement: number;
  tokenWeightedAgreement: number;
  meanFirstDivergence: number;
  identicalPrompts: number;
} {
  const byId = new Map(ref.results.map((r) => [r.id, r]));
  const perPrompt = [];
  let totMatch = 0;
  let totCompared = 0;
  const divs: number[] = [];
  let identical = 0;
  for (const r of arm.results) {
    const refR = byId.get(r.id);
    if (!refR) continue;
    const agree = agreementRatio(refR.tokenIds, r.tokenIds);
    const div = divergenceIndex(refR.tokenIds, r.tokenIds);
    const n = Math.min(refR.tokenIds.length, r.tokenIds.length);
    totCompared += n;
    totMatch += Math.round(agree * n);
    divs.push(div === -1 ? n : div);
    if (div === -1 && refR.tokenIds.length === r.tokenIds.length) identical++;
    perPrompt.push({
      id: r.id,
      agreement: agree,
      divergenceIndex: div,
      firstTokenSame: refR.tokenIds.length > 0 && r.tokenIds.length > 0 && refR.tokenIds[0] === r.tokenIds[0],
      refLen: refR.tokenIds.length,
      armLen: r.tokenIds.length,
      refHead: refR.textHead,
      armHead: r.textHead,
    });
  }
  return {
    perPrompt,
    meanAgreement: mean(perPrompt.map((p) => p.agreement)),
    tokenWeightedAgreement: totCompared ? totMatch / totCompared : Number.NaN,
    meanFirstDivergence: mean(divs),
    identicalPrompts: identical,
  };
}

async function runCompare(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'sym8-e2e-coherence-'));
  // SEPARATE PROCESSES, run SEQUENTIALLY — one GPU inference process at a time.
  const bf16 = await spawnArm('bf16', values.bf16!, join(dir, 'bf16.json'));
  const sym8 = await spawnArm('sym8', values.sym8!, join(dir, 'sym8.json'));
  const aff8 = await spawnArm('aff8', values.aff8!, join(dir, 'aff8.json'));

  const symVsBf16 = compareArm(bf16, sym8);
  const affVsBf16 = compareArm(bf16, aff8);

  const cmp = {
    maxNew,
    models: { bf16: values.bf16, sym8: values.sym8, aff8: values.aff8 },
    nPrompts: bf16.results.length,
    sym8_vs_bf16: symVsBf16,
    aff8_vs_bf16: affVsBf16,
    verdict: {
      sym8_mean_agreement: symVsBf16.meanAgreement,
      aff8_mean_agreement: affVsBf16.meanAgreement,
      sym8_token_weighted_agreement: symVsBf16.tokenWeightedAgreement,
      aff8_token_weighted_agreement: affVsBf16.tokenWeightedAgreement,
      sym8_mean_first_divergence: symVsBf16.meanFirstDivergence,
      aff8_mean_first_divergence: affVsBf16.meanFirstDivergence,
      agreement_gap_aff_minus_sym: affVsBf16.meanAgreement - symVsBf16.meanAgreement,
    },
  };
  console.log(`CMP_JSON:${JSON.stringify(cmp)}`);
}

// ─────────────────────────────────────────────────────────────────────────────
if (values.compare) {
  await runCompare();
} else {
  await runWorker();
}
