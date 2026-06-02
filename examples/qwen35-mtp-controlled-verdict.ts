#!/usr/bin/env node
/**
 * Controlled MTP A/B verdict harness — trustworthy chained-ON vs chained-OFF.
 *
 * WHY THIS EXISTS: a naive same-binary A/B (run all chained=0, then all
 * chained=1) is thermal-confounded. On M-series the GPU throttles over a
 * sustained run, so the config measured later looks slower regardless of
 * code. A prior run produced 2.6x swings on an IDENTICAL cell and a ~2x
 * swing in the AR baseline (which is flag-INVARIANT by construction), making
 * every absolute tok/s and ratio untrustworthy.
 *
 * METHODOLOGY (what makes this trustworthy):
 *   1. The MTP/AR RATIO is measured WITHIN one process, AR and MTP back-to-back
 *      at the same thermal moment, so the ratio is self-normalized against the
 *      AR baseline — robust to absolute throttling.
 *   2. Pairs are INTERLEAVED with ALTERNATING order (AR-first / MTP-first on
 *      odd/even repeats) to cancel the intra-pair drift (MTP always slightly
 *      later than its AR).
 *   3. COOLDOWN between pairs lets transient thermal spikes settle.
 *   4. WARMUP runs pay the compile/cache cost before any timed pair.
 *   5. We report the MEDIAN ratio over many repeats plus the spread (min/max),
 *      not a single noisy sample.
 *   6. AR-STABILITY GATE: AR is flag-invariant, so its coefficient of variation
 *      across ALL pairs is our thermal-trust gauge. CV > 10% => the run was
 *      thermally unstable; absolute numbers are untrustworthy (ratios still are,
 *      because they are self-normalized) and the verdict is flagged.
 *
 * The flag MLX_MTP_CHAINED_CYCLES is read ONCE per process (OnceLock in Rust),
 * so each config needs its own process: the parent spawns this same file as a
 * child per config with the env set. Child prints progress to stderr and one
 * `__BENCH_RESULT__ {json}` line to stdout.
 *
 *   oxnode examples/qwen35-mtp-controlled-verdict.ts [model] \
 *     --depths 1,3 --repeats 6 --max-tokens 160 --cooldown 1500 --warmup 2
 */

import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import type { ChatConfig, ChatResult, SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    depths: { type: 'string' },
    repeats: { type: 'string' },
    'max-tokens': { type: 'string' },
    cooldown: { type: 'string' },
    warmup: { type: 'string' },
    prompt: { type: 'string' },
  },
  allowPositionals: true,
});

// Prompt presets — MTP acceptance is PROMPT-GATED, so the bench prompt picks the
// regime: `essay` is prose (hard, low acceptance, the conservative worst case);
// `counting`/`code` are predictable spans (high acceptance, MTP's best case).
// `--prompt` also accepts a raw string. The chosen prompt propagates to children
// automatically (the parent forwards `...process.argv.slice(2)` in `spawnChild`).
const PROMPT_PRESETS: Record<string, string> = {
  essay:
    'Write a concise three-paragraph essay on why deterministic sampling at temperature 0 is useful for testing speculative decoding implementations.',
  counting: 'Count from 1 to 120, one number per line, with no other text.',
  code: 'Write a Python function that implements binary search over a sorted list, then merge sort, then quicksort. Include docstrings and a complexity table.',
};

const MODEL_NAME = positionals[0] || 'qwen3.6-27b-nvfp4-mtp';
const DEPTHS = (values.depths ?? '1,3')
  .split(',')
  .map((d) => Number(d.trim()))
  .filter((d) => Number.isFinite(d));
const REPEATS = Number(values.repeats ?? '6');
const MAX_TOKENS = Number(values['max-tokens'] ?? '160');
const COOLDOWN_MS = Number(values.cooldown ?? '1500');
const WARMUP = Number(values.warmup ?? '2');
const PROMPT_ARG = values.prompt ?? 'essay';
const PROMPT = PROMPT_PRESETS[PROMPT_ARG] ?? PROMPT_ARG;
const PROMPT_LABEL = PROMPT_ARG in PROMPT_PRESETS ? PROMPT_ARG : 'custom';
const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', MODEL_NAME);
const IS_CHILD = process.env.__MTP_BENCH_CHILD === '1';
const RESULT_SENTINEL = '__BENCH_RESULT__';

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

interface Sample {
  depth: number;
  repeat: number;
  arTps: number;
  mtpTps: number;
  ratio: number;
  // drafts-only mean accepted per cycle (historical; excludes the
  // always-verified token).
  meanAccepted: number;
  // mlx-vlm-comparable: mean accepted tokens/cycle INCLUDING the
  // always-verified token = drafts-only + 1.0 = mlx-vlm's
  // `(accepted_drafts + rounds)/rounds`.
  meanAcceptedTotal: number;
  accByPosition: number[];
  cycles: number;
}
interface ChildResult {
  chained: number;
  model: string;
  error?: string;
  samples: Sample[];
}

// ----------------------------------------------------------------------------
// CHILD: load once, warmup, measure interleaved alternating-order pairs.
// ----------------------------------------------------------------------------
async function runChild(): Promise<void> {
  const chained = Number(process.env.MLX_MTP_CHAINED_CYCLES ?? '0');
  const log = (m: string) => console.error(`[child chained=${chained}] ${m}`);
  log(`loading ${MODEL_PATH}`);
  const loaded = await loadModel(MODEL_PATH);
  if (loaded instanceof HarrierModel) {
    process.stdout.write(
      `${RESULT_SENTINEL} ${JSON.stringify({ chained, model: MODEL_NAME, error: 'not_session_capable', samples: [] })}\n`,
    );
    return;
  }
  const model = loaded as unknown as SessionCapableModel;
  if (!(typeof model.hasMtpWeights === 'function' && model.hasMtpWeights())) {
    process.stdout.write(
      `${RESULT_SENTINEL} ${JSON.stringify({ chained, model: MODEL_NAME, error: 'no_mtp_heads', samples: [] })}\n`,
    );
    return;
  }

  const prompt = PROMPT;

  async function gen(depth: number, enableMtp: boolean): Promise<ChatResult> {
    const cfg: ChatConfig = {
      temperature: 0,
      topK: 1,
      topP: 1,
      maxNewTokens: MAX_TOKENS,
      reasoningEffort: 'none',
      includeReasoning: false,
      mtpDepth: depth,
      reportPerformance: true,
      enableMtp,
    };
    const session = new ChatSession(model, { system: 'You are a precise assistant. Be concise.', defaultConfig: cfg });
    return session.send(prompt);
  }

  // Warmup at the first depth — pays compile + cache for both modes.
  const d0 = DEPTHS[0];
  for (let w = 0; w < WARMUP; w++) {
    log(`warmup ${w + 1}/${WARMUP}`);
    await gen(d0, false);
    await gen(d0, true);
    await sleep(COOLDOWN_MS);
  }

  const samples: Sample[] = [];
  for (const depth of DEPTHS) {
    for (let r = 0; r < REPEATS; r++) {
      // Alternate intra-pair order to cancel "MTP always measured later" drift.
      const arFirst = r % 2 === 0;
      let ar: ChatResult, mtp: ChatResult;
      if (arFirst) {
        ar = await gen(depth, false);
        mtp = await gen(depth, true);
      } else {
        mtp = await gen(depth, true);
        ar = await gen(depth, false);
      }
      const arTps = ar.performance?.decodeTokensPerSecond ?? 0;
      const mtpTps = mtp.performance?.decodeTokensPerSecond ?? 0;
      const p = mtp.performance;
      const sample: Sample = {
        depth,
        repeat: r,
        arTps,
        mtpTps,
        ratio: arTps > 0 ? mtpTps / arTps : 0,
        meanAccepted: p?.mtpMeanAcceptedTokens ?? 0,
        meanAcceptedTotal: p?.mtpMeanAcceptedTokensTotal ?? (p?.mtpMeanAcceptedTokens ?? 0) + 1,
        accByPosition: p?.mtpAcceptanceByPosition ?? [],
        cycles: p?.mtpCycles ?? 0,
      };
      samples.push(sample);
      log(
        `depth=${depth} r=${r + 1}/${REPEATS} (${arFirst ? 'AR,MTP' : 'MTP,AR'}) AR=${arTps.toFixed(2)} MTP=${mtpTps.toFixed(2)} ratio=${sample.ratio.toFixed(3)} acc/cyc(incl.verified)=${sample.meanAcceptedTotal.toFixed(2)} drafts-only=${sample.meanAccepted.toFixed(2)}`,
      );
      await sleep(COOLDOWN_MS);
    }
  }
  const result: ChildResult = { chained, model: MODEL_NAME, samples };
  process.stdout.write(`${RESULT_SENTINEL} ${JSON.stringify(result)}\n`);
}

// ----------------------------------------------------------------------------
// PARENT: spawn one child per config, aggregate, gate, print verdict.
// ----------------------------------------------------------------------------
function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
function cv(xs: number[]): number {
  if (xs.length < 2) return 0;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  if (mean === 0) return 0;
  const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
  return Math.sqrt(variance) / mean;
}

function spawnChild(chained: number): ChildResult {
  console.error(`\n=== spawning child: MLX_MTP_CHAINED_CYCLES=${chained} ===`);
  const child = spawnSync('oxnode', [process.argv[1], ...process.argv.slice(2)], {
    encoding: 'utf8',
    timeout: 45 * 60 * 1000,
    maxBuffer: 64 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'inherit'],
    env: { ...process.env, __MTP_BENCH_CHILD: '1', MLX_MTP_CHAINED_CYCLES: String(chained) },
  });
  if (child.status !== 0 || !child.stdout) {
    throw new Error(`child chained=${chained} failed (status=${child.status}, signal=${child.signal})`);
  }
  const line = child.stdout.split('\n').find((l) => l.startsWith(RESULT_SENTINEL));
  if (!line) throw new Error(`child chained=${chained} produced no ${RESULT_SENTINEL} line`);
  return JSON.parse(line.slice(RESULT_SENTINEL.length).trim()) as ChildResult;
}

async function runParent(): Promise<void> {
  console.error(
    `Controlled MTP verdict: model=${MODEL_NAME} prompt=${PROMPT_LABEL} depths=[${DEPTHS}] repeats=${REPEATS} maxTokens=${MAX_TOKENS} cooldown=${COOLDOWN_MS}ms warmup=${WARMUP}`,
  );
  // chained=0 first then chained=1; ratio is self-normalized so run order does
  // not bias the verdict, but we still report cross-child AR drift.
  const off = spawnChild(0);
  const on = spawnChild(1);
  for (const r of [off, on]) {
    if (r.error) {
      console.log(`\nABORTED: child chained=${r.chained} returned error '${r.error}'. Need an MTP-capable checkpoint.`);
      process.exit(2);
    }
  }

  const allAr = [...off.samples, ...on.samples].map((s) => s.arTps);
  const arCv = cv(allAr);
  const arOffMed = median(off.samples.map((s) => s.arTps));
  const arOnMed = median(on.samples.map((s) => s.arTps));
  const arDrift = arOffMed > 0 ? Math.abs(arOnMed - arOffMed) / arOffMed : 0;
  const trustworthy = arCv <= 0.1 && arDrift <= 0.1;

  console.log('\n================ CONTROLLED MTP VERDICT ================');
  console.log(`model=${MODEL_NAME}  prompt=${PROMPT_LABEL}  repeats=${REPEATS}/cell  maxTokens=${MAX_TOKENS}`);
  console.log(
    `AR-stability gate: CV(all AR)=${(arCv * 100).toFixed(1)}%  cross-child AR drift=${(arDrift * 100).toFixed(1)}%  ` +
      `(AR off-med=${arOffMed.toFixed(2)} on-med=${arOnMed.toFixed(2)})  => ${trustworthy ? 'TRUSTWORTHY' : 'THERMALLY UNSTABLE (ratios still self-normalized; absolute tok/s untrustworthy)'}`,
  );
  console.log('');
  // `acc/cyc(incl.verified)` is the mlx-vlm-comparable headline
  // (mlx-vlm's `(accepted_drafts + rounds)/rounds`); `drafts` is the
  // historical drafts-only value kept alongside.
  console.log(
    'depth | cfg      | median ratio | ratio[min..max] | median MTP tok/s | acc/cyc(incl.verified) | drafts | per-position acceptance',
  );
  console.log(
    '------|----------|--------------|-----------------|------------------|------------------------|--------|------------------------',
  );

  interface DepthVerdict {
    depth: number;
    medOff: number;
    medOn: number;
    delta: number;
    verdict: string;
  }
  const verdicts: DepthVerdict[] = [];
  for (const depth of DEPTHS) {
    const offD = off.samples.filter((s) => s.depth === depth);
    const onD = on.samples.filter((s) => s.depth === depth);
    const rOff = offD.map((s) => s.ratio);
    const rOn = onD.map((s) => s.ratio);
    const medOff = median(rOff);
    const medOn = median(rOn);
    const accOff = offD[offD.length - 1];
    const accOn = onD[onD.length - 1];
    for (const [cfg, d, med, rs, acc] of [
      ['OFF', offD, medOff, rOff, accOff] as const,
      ['ON ', onD, medOn, rOn, accOn] as const,
    ]) {
      console.log(
        `  ${depth}   | chained ${cfg} | ${med.toFixed(3)}        | ` +
          `${Math.min(...rs).toFixed(3)}..${Math.max(...rs).toFixed(3)}   | ` +
          `${median(d.map((s) => s.mtpTps)).toFixed(2)}            | ` +
          `${acc.meanAcceptedTotal.toFixed(2)}                   | ` +
          `${acc.meanAccepted.toFixed(2)}   | [${acc.accByPosition.map((p) => p.toFixed(3)).join(', ')}]`,
      );
    }
    // Verdict: ON wins only if its median ratio clears OFF's median by more than
    // the larger half-spread of either distribution (i.e. beyond ratio noise).
    const delta = medOn - medOff;
    const halfSpread = Math.max((Math.max(...rOn) - Math.min(...rOn)) / 2, (Math.max(...rOff) - Math.min(...rOff)) / 2);
    let verdict: string;
    if (delta > halfSpread)
      verdict = `chained-ON WINS (+${(delta * 100).toFixed(1)}pp ratio, beyond noise ±${(halfSpread * 100).toFixed(1)}pp)`;
    else if (delta < -halfSpread) verdict = `chained-ON REGRESSES (${(delta * 100).toFixed(1)}pp ratio)`;
    else verdict = `INCONCLUSIVE (Δ=${(delta * 100).toFixed(1)}pp within noise ±${(halfSpread * 100).toFixed(1)}pp)`;
    verdicts.push({ depth, medOff, medOn, delta, verdict });
  }

  console.log('\n---- per-depth verdict ----');
  for (const v of verdicts) console.log(`depth ${v.depth}: ${v.verdict}`);
  console.log('\nNOTE: acceptance is deterministic and is the reliable signal; chained-ON that lowers');
  console.log('acceptance is a true regression regardless of tok/s. Ratios are self-normalized; trust');
  console.log(`them over absolute tok/s when the AR-stability gate flags thermal instability.`);
  console.log('\n' + RESULT_SENTINEL + '_PARENT ' + JSON.stringify({ trustworthy, arCv, arDrift, verdicts }));
}

if (IS_CHILD) {
  await runChild();
} else {
  await runParent();
}
