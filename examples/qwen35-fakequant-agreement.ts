#!/usr/bin/env node
/**
 * Qwen3.5 FAKE-QUANT end-to-end agreement harness (Option-B sym8 vs current aff8).
 *
 * Greedy-decodes (T=0, reuseCache=false) a SET of diverse prompts for N tokens
 * from THREE bf16 models produced by examples/qwen35-fakequant-writer.py:
 *   - bf16 : the unmodified source (reference)
 *   - sym8 : per-OUTPUT-CHANNEL symmetric int8 round-trip -> bf16  (Option B)
 *   - aff8 : MLX affine per-GROUP(64) int8 round-trip -> bf16      (current Q8)
 *
 * Because each model is plain bf16, normal bf16 inference simulates that quant's
 * WEIGHT quality with NO new kernels. We then ask: does sym8's greedy output
 * track bf16 about as well as aff8's does? That isolates the only new accuracy
 * variable in Option B — the weight quant — end to end.
 *
 * Each model loads in its OWN process (the compiled Qwen3.5 forward path uses
 * process-wide weight globals; separate processes guarantee no cross-model
 * weight contamination). Worker emits one RESULT_JSON line; orchestrator
 * (--compare) spawns all three, then prints one CMP_JSON line.
 *
 * Usage:
 *   PATH=/usr/bin:$PATH oxnode examples/qwen35-fakequant-agreement.ts --compare \
 *     --bf16 /Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
 *     --sym8 /tmp/qwen35-0.8b-sym8 --aff8 /tmp/qwen35-0.8b-aff8 --max-new 64
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
    'max-new': { type: 'string', default: '64' },
    // orchestrator:
    compare: { type: 'boolean', default: false },
    bf16: { type: 'string', default: '/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16' },
    sym8: { type: 'string', default: '/tmp/qwen35-0.8b-sym8' },
    aff8: { type: 'string', default: '/tmp/qwen35-0.8b-aff8' },
  },
});

const maxNew = Number.parseInt(values['max-new']!, 10);
const SYSTEM = 'You are a helpful assistant.';

// ── Six diverse prompts: prose, code, lists/counting, Q&A. Deterministic,
//    self-contained (NOT shell history). Greedy T=0 makes them reproducible.
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
  {
    id: 'prose2',
    text: 'Describe a bustling morning market in a Mediterranean town in one detailed paragraph.',
  },
  {
    id: 'code2',
    text: 'Write a JavaScript function that reverses a string in place and returns it. Add a one-line comment.',
  },
  {
    id: 'code3',
    text: 'Write a SQL query that selects the names and salaries of the top 5 highest-paid employees from a table called employees.',
  },
  {
    id: 'counting2',
    text: 'List the first ten even numbers, separated by commas.',
  },
  {
    id: 'qa-factual2',
    text: 'In two sentences, explain what photosynthesis is and why it matters for life on Earth.',
  },
  {
    id: 'qa-howto',
    text: 'Give step-by-step instructions for making a simple cup of tea. Use a numbered list.',
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
    const inputArray = MxArray.fromUint32(
      new Uint32Array(promptIds),
      BigInt64Array.from([1n, BigInt(nPrompt)]),
    );
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
      textHead: text.slice(0, 160),
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
    const args = [
      process.argv[1],
      '--model',
      modelPath,
      '--arm',
      label,
      '--out',
      outPath,
      '--max-new',
      String(maxNew),
    ];
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

function compareArm(ref: ArmResult, arm: ArmResult): {
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
  // token-weighted agreement: total matched / total compared (big outputs dominate)
  tokenWeightedAgreement: number;
  meanFirstDivergence: number; // identical prompts excluded (clamped to maxNew)
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
    // first-divergence: identical-over-common => count as full length (no divergence)
    divs.push(div === -1 ? n : div);
    if (div === -1 && refR.tokenIds.length === r.tokenIds.length) identical++;
    perPrompt.push({
      id: r.id,
      agreement: agree,
      divergenceIndex: div,
      firstTokenSame:
        refR.tokenIds.length > 0 && r.tokenIds.length > 0 && refR.tokenIds[0] === r.tokenIds[0],
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
  const dir = mkdtempSync(join(tmpdir(), 'fakequant-agree-'));
  // SEPARATE PROCESSES — load-time weight globals must not be shared.
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
      // does sym8 track bf16 about as well as aff8?
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
