#!/usr/bin/env node
/**
 * Qwen3.5 int8 W8A8 PREFILL accuracy harness (Option-C symmetric-int8 probe).
 *
 * GOAL — quantify how much the symmetric-int8 W8A8 prefill path degrades an
 * ALREADY-8bit-affine-quantized checkpoint vs. its OWN affine W8A16 baseline.
 * Both arms load the SAME quantized model. The only delta is the load-time
 * `MLX_INT8_QUANT_PREFILL` flag, which (in `quantized_linear.rs`) builds a
 * symmetric per-channel int8 view of each 8-bit-affine weight at load and
 * routes prefill matmuls (M = batch*seq >= MLX_INT8_QUANT_PREFILL_MIN_M,
 * default 256) through the W8A8 kernel instead of affine `quantized_matmul`.
 *
 *   baseline  (W8A16): MLX_INT8_QUANT_PREFILL unset → affine quantized_matmul
 *   treatment (W8A8):  MLX_INT8_QUANT_PREFILL=1 MLX_INT8_QUANT_PREFILL_DEBUG=1
 *
 * The flag is read at LOAD, so the two arms MUST be SEPARATE PROCESSES. This
 * one file is both the per-arm worker and the orchestrator:
 *
 *   --compare  (orchestrator): spawns the two worker processes itself, captures
 *              each worker's stdout RESULT_JSON + stderr, then prints a single
 *              CMP_JSON line with the comparison verdict.
 *   (default)  (worker / single arm): loads the model, greedily decodes (T=0)
 *              a >=300-token prompt, and writes RESULT_JSON to --out (and stdout).
 *
 * ── What is and isn't measurable through the public API ──────────────────────
 * The Qwen3.5 NAPI surface exposes NO final-position logit/logprob accessor
 * (generate() returns only {tokens,text,numTokens,finishReason}; ChatResult has
 * no logits). So a literal cosine-similarity of two raw logit VECTORS at the
 * last prompt position is NOT computable today without a native change (a
 * one-line "return prefill logits" accessor on Qwen35Model). What IS observable
 * — and is the strongest available proxy for "did the W8A8 prefill perturb the
 * final-position distribution" — is the GREEDY (T=0, argmax) continuation:
 *   • the FIRST greedy token id == argmax of the final prompt-position logits,
 *   • the divergence index of the two token-id sequences == the first step where
 *     the two final-position argmaxes differ,
 *   • a per-position agreement ratio over the common length grades the drift.
 * The comparator reports all three plus byte-identical text, and confirms the
 * debug "[int8-quant-prefill] fired" lines appeared in the TREATMENT arm only
 * (path engaged). See `logitCosineNote` in the CMP_JSON for the exact caveat.
 *
 * Usage:
 *   # orchestrated A/B (recommended) — spawns both arms for you:
 *   PATH=/usr/bin:$PATH oxnode examples/qwen35-int8-quant-accuracy.ts --compare \
 *     --model /path/to/qwen3.5-Xb-8bit --prompt-tokens 384 --max-new 64
 *
 *   # single arm (worker) — env selects the arm; write result to --out:
 *   MLX_INT8_QUANT_PREFILL=1 MLX_INT8_QUANT_PREFILL_DEBUG=1 \
 *     PATH=/usr/bin:$PATH oxnode examples/qwen35-int8-quant-accuracy.ts \
 *     --model /path/to/qwen3.5-Xb-8bit --out /tmp/treat.json --arm treatment
 *
 * Worker prints exactly one `RESULT_JSON:{...}` line on stdout.
 * Orchestrator prints exactly one `CMP_JSON:{...}` line on stdout.
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
    model: { type: 'string', default: '/Volumes/P4510/.cache/models/qwen3.5-4b-8bit' },
    'prompt-tokens': { type: 'string', default: '384' },
    'max-new': { type: 'string', default: '64' },
    system: { type: 'string', default: 'You are a helpful assistant.' },
    // worker-only:
    out: { type: 'string' }, // where the worker writes its RESULT_JSON
    arm: { type: 'string', default: 'unknown' }, // label only ('baseline' | 'treatment')
    // orchestrator:
    compare: { type: 'boolean', default: false },
  },
});

const modelPath = values.model!;
const promptTokens = Number.parseInt(values['prompt-tokens']!, 10);
const maxNew = Number.parseInt(values['max-new']!, 10);
const systemPrompt = values.system!;

// ── Deterministic, self-contained prompt (NOT shell history). ~16 tok/sentence.
// `copies = ceil(promptTokens/16)` pushes prefill M well over the 256 int8
// threshold so the W8A8 path actually fires in the treatment arm.
const SENT = 'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
function buildUserPrompt(): string {
  const copies = Math.max(1, Math.ceil(promptTokens / 16));
  return (
    'Read the following text, then write a clear factual paragraph explaining what a ' +
    'transformer neural network is and why attention matters.\n' +
    `${SENT.repeat(copies)}\n` +
    'Now write the explanation.'
  );
}

// Snapshot the int8 env so each arm's JSON self-documents which path it took.
function snapshotToggles(): Record<string, string> {
  const t: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (k.startsWith('MLX_INT8_') || k === 'MLX_NO_COMPILE' || k === 'MLX_DISABLE_COMPILE') {
      t[k] = v ?? '';
    }
  }
  return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKER: load the (already-quantized) model, greedily decode T=0, persist.
// ─────────────────────────────────────────────────────────────────────────────
async function runWorker(): Promise<void> {
  const toggles = snapshotToggles();

  const tokenizer = await Qwen3Tokenizer.fromPretrained(join(modelPath, 'tokenizer.json'));
  const messages: ChatMessage[] = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: buildUserPrompt() },
  ];
  // add_generation_prompt=true so the prompt ends on the assistant turn; the
  // model's first emitted token is the argmax of the FINAL prompt-position logits.
  const promptIds = await tokenizer.applyChatTemplate(messages, true);
  const nPrompt = promptIds.length;
  const inputArray = MxArray.fromUint32(new Uint32Array(promptIds), BigInt64Array.from([1n, BigInt(nPrompt)]));

  // Native generate(): greedy (T=0) gives the deterministic argmax continuation
  // — exact token ids (sharper than text alone). reuseCache is implicitly false
  // here (one fresh prompt, no chat-session prefix reuse).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const model = (await loadModel(modelPath)) as any;
  const res = await model.generate(inputArray, {
    maxNewTokens: maxNew,
    temperature: 0,
  });

  const tokenIds: number[] = Array.from(res.tokens ?? []);
  const text: string = String(res.text ?? '');
  const textHash = createHash('sha256').update(text).digest('hex');
  // Hash the token-id sequence as canonical text so the digest is stable across
  // platforms/endianness and trivially accepts the number[] continuation.
  const idsHash = createHash('sha256').update(tokenIds.join(',')).digest('hex');

  const out = {
    arm: values.arm,
    model: modelPath,
    pid: process.pid,
    nPromptTokens: nPrompt,
    promptTokensRequested: promptTokens,
    maxNew,
    toggles,
    int8QuantPrefillEnabled:
      (process.env.MLX_INT8_QUANT_PREFILL ?? '') !== '' &&
      process.env.MLX_INT8_QUANT_PREFILL !== '0' &&
      process.env.MLX_INT8_QUANT_PREFILL !== 'false',
    numTokens: res.numTokens ?? tokenIds.length,
    finishReason: res.finishReason ?? '',
    tokenIds,
    idsHash,
    textHash,
    text: text.slice(0, 1200),
  };

  if (values.out) writeFileSync(values.out, JSON.stringify(out));
  console.log(`RESULT_JSON:${JSON.stringify(out)}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// ORCHESTRATOR: spawn the two arms (separate processes, flag set per-arm),
// capture each worker's RESULT_JSON + stderr, then compute the comparison.
// ─────────────────────────────────────────────────────────────────────────────
const DEBUG_FIRED_RE = /\[int8-quant-prefill\] fired:/;

interface ArmResult {
  arm: string;
  tokenIds: number[];
  idsHash: string;
  textHash: string;
  text: string;
  finishReason: string;
  nPromptTokens: number;
  int8QuantPrefillEnabled: boolean;
  toggles: Record<string, string>;
}

interface SpawnedArm {
  result: ArmResult;
  stderr: string;
  firedCount: number;
  exitCode: number | null;
}

function spawnArm(label: string, extraEnv: Record<string, string>, outPath: string): Promise<SpawnedArm> {
  return new Promise((resolve, reject) => {
    const args = [
      process.argv[1], // this script
      '--model',
      modelPath,
      '--prompt-tokens',
      String(promptTokens),
      '--max-new',
      String(maxNew),
      '--system',
      systemPrompt,
      '--out',
      outPath,
      '--arm',
      label,
    ];
    // Inherit the env, then strip ALL MLX_INT8_QUANT_PREFILL* keys so the
    // baseline arm is genuinely unset even if the orchestrator was invoked
    // with them set; then apply this arm's overrides.
    const env: NodeJS.ProcessEnv = { ...process.env };
    for (const k of Object.keys(env)) {
      if (k.startsWith('MLX_INT8_QUANT_PREFILL')) delete env[k];
    }
    Object.assign(env, extraEnv);

    const child = spawn(process.execPath, args, {
      env,
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
        // Fall back to the persisted file if stdout was noisy.
        try {
          result = JSON.parse(readFileSync(outPath, 'utf-8')) as ArmResult;
        } catch {
          result = null;
        }
      }
      if (result == null) {
        // Worker crashed before producing a result (bad model path, OOM, ...).
        // Surface its stderr tail so the failure is legible, not an ENOENT.
        reject(
          new Error(
            `[${label}] arm produced no RESULT_JSON (exit=${code}). stderr tail:\n` +
              stderr.split('\n').slice(-20).join('\n'),
          ),
        );
        return;
      }
      const firedCount = (stderr.match(new RegExp(DEBUG_FIRED_RE, 'g')) ?? []).length;
      resolve({ result, stderr, firedCount, exitCode: code });
    });
  });
}

function divergenceIndex(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n; // -1 == identical; else first length-mismatch index
}

function agreementRatio(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return Number.NaN;
  let same = 0;
  for (let i = 0; i < n; i++) if (a[i] === b[i]) same++;
  return same / n;
}

async function runCompare(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'int8-quant-acc-'));
  const baseOut = join(dir, 'baseline.json');
  const treatOut = join(dir, 'treatment.json');

  // SEPARATE PROCESSES — the load-time flag cannot be toggled in-process.
  const baseline = await spawnArm('baseline', {}, baseOut);
  const treatment = await spawnArm(
    'treatment',
    { MLX_INT8_QUANT_PREFILL: '1', MLX_INT8_QUANT_PREFILL_DEBUG: '1' },
    treatOut,
  );

  const b = baseline.result;
  const t = treatment.result;

  const idsIdentical = b.idsHash === t.idsHash;
  const textIdentical = b.textHash === t.textHash;
  const divIdx = divergenceIndex(b.tokenIds, t.tokenIds);
  const firstTokenSame = b.tokenIds.length > 0 && t.tokenIds.length > 0 && b.tokenIds[0] === t.tokenIds[0];
  const agree = agreementRatio(b.tokenIds, t.tokenIds);

  // (c) path-engaged confirmation: treatment must fire, baseline must NOT.
  const pathEngagedTreatment = treatment.firedCount > 0;
  const pathQuietBaseline = baseline.firedCount === 0;

  const cmp = {
    model: modelPath,
    promptTokens,
    maxNew,
    arms: {
      baseline: {
        pid: undefined as number | undefined,
        nPromptTokens: b.nPromptTokens,
        int8QuantPrefillEnabled: b.int8QuantPrefillEnabled,
        finishReason: b.finishReason,
        firedCount: baseline.firedCount,
        exitCode: baseline.exitCode,
        textHead: b.text.slice(0, 200),
      },
      treatment: {
        nPromptTokens: t.nPromptTokens,
        int8QuantPrefillEnabled: t.int8QuantPrefillEnabled,
        finishReason: t.finishReason,
        firedCount: treatment.firedCount,
        exitCode: treatment.exitCode,
        textHead: t.text.slice(0, 200),
      },
    },
    // (a) final-position argmax proxy for logit cosine — see logitCosineNote.
    firstGreedyTokenSame: firstTokenSame,
    // (b) byte-identical greedy continuation?
    greedyTokenIdsIdentical: idsIdentical,
    greedyTextIdentical: textIdentical,
    // graded drift:
    divergenceIndex: divIdx, // -1 == identical, else first differing token step
    tokenAgreementRatio: agree, // fraction of matched-length positions that agree
    // (c) path-engaged confirmation:
    pathEngagedTreatment,
    pathQuietBaseline,
    logitCosineNote:
      'TRUE final-position logit cosine similarity is NOT exposed by the current ' +
      'Qwen3.5 NAPI surface (generate() returns no logits). The greedy T=0 ' +
      'continuation is the available proxy: firstGreedyTokenSame == final-position ' +
      'argmax agreement; divergenceIndex == first step the two argmaxes differ. ' +
      'For a literal cosine-of-logit-vectors metric, add a native prefill-logits ' +
      'accessor to Qwen35Model and feed both arms the same prompt ids.',
    verdict:
      pathEngagedTreatment && pathQuietBaseline
        ? idsIdentical
          ? 'IDENTICAL (W8A8 prefill caused no greedy drift)'
          : `DRIFT@${divIdx} (W8A8 prefill diverged from affine baseline)`
        : 'PATH-NOT-ENGAGED (treatment did not fire / baseline fired — check flags & M>=256)',
  };

  console.log(`CMP_JSON:${JSON.stringify(cmp)}`);
}

// ─────────────────────────────────────────────────────────────────────────────
if (values.compare) {
  await runCompare();
} else {
  await runWorker();
}
