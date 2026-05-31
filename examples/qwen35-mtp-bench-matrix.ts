#!/usr/bin/env node
/**
 * Qwen3.5 MTP benchmark matrix.
 *
 * MTP env flags are cached in Rust once per process, so the master process
 * forks one worker per matrix cell. Progress goes to stderr; the final report
 * is JSON on stdout and can also be written with `--json`.
 *
 *   oxnode examples/qwen35-mtp-bench-matrix.ts qwen3.6-27b-nvfp4-mtp-oproj8 \
 *     --depths default,1,2,3,adaptive,adaptive-ev --env-variants perf-flat,paged-on,sparse-off \
 *     --max-tokens 128 --profile --json /tmp/qwen-mtp-matrix.json
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { hostname } from 'node:os';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import {
  getProfilingData,
  resetProfilingData,
  setProfilingEnabled,
  type GenerationProfile,
  type PhaseProfile,
  type ProfilingSummary,
} from '@mlx-node/core';
import type { ChatConfig, ChatResult, PerformanceMetrics, SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

type AdaptiveDepthKind = 'throughput' | 'expected-value';
type DepthSpec =
  | { mode: 'default' }
  | { mode: 'pinned'; mtpDepth: 1 | 2 | 3 }
  | { mode: 'adaptive'; adaptiveKind: AdaptiveDepthKind; mtpDepth?: 1 | 2 | 3 };

type EnvVariant = {
  name: string;
  env: Record<string, string>;
  category: 'flat' | 'paged' | 'ablation';
  experimental?: boolean;
};

type CellSpec = {
  cellId: string;
  depth: DepthSpec;
  envVariant: EnvVariant;
};

type BenchRun = {
  numTokens: number;
  promptTokens: number;
  cachedTokens: number;
  finishReason: string;
  text: string;
  textSha256: string;
  textLength: number;
  performance?: Pick<
    PerformanceMetrics,
    | 'ttftMs'
    | 'prefillTokensPerSecond'
    | 'decodeTokensPerSecond'
    | 'mtpMeanAcceptedTokens'
    | 'mtpAcceptanceByPosition'
    | 'mtpCycles'
    | 'mtpMeanDepth'
  > & {
    phaseMsByName?: Record<string, number>;
  };
};

type MtpAcceptanceTrace = Record<string, unknown>;

type SampleResult = {
  repeat: number;
  status: 'ok' | 'failed' | 'skipped';
  model?: {
    hasMtpWeights: boolean;
    hasBlockPagedCache: boolean;
  };
  skipReason?: string;
  error?: string;
  durationMs: number;
  ar?: BenchRun;
  mtp?: BenchRun;
  comparison?: {
    decodeSpeedup: number | null;
    textEqual: boolean;
    firstDiffChar: number | null;
    acceptanceOk: boolean;
    planPos0GateOk: boolean;
  };
  profiling?: {
    summary: ProfilingSummary;
    arGeneration?: GenerationProfile;
    mtpGeneration?: GenerationProfile;
  };
  acceptanceTrace?: MtpAcceptanceTrace[];
};

type CellResult = {
  cellId: string;
  depth: DepthSpec;
  envVariant: EnvVariant;
  samples: SampleResult[];
  aggregate: {
    okSamples: number;
    failedSamples: number;
    skippedSamples: number;
    textEqualRate: number | null;
    firstDiffCharMin: number | null;
    arDecodeTpsMean: number | null;
    arDecodeTpsMin: number | null;
    speedupMean: number | null;
    speedupMin: number | null;
    speedupStddev: number | null;
    mtpDecodeTpsMean: number | null;
    mtpDecodeTpsMin: number | null;
    meanAcceptedMean: number | null;
    meanAcceptedMin: number | null;
    pos0AcceptanceMean: number | null;
    acceptanceByPositionMean: number[];
    acceptanceByPositionMin: number[];
    mtpCyclesMean: number | null;
    mtpMeanDepthMean: number | null;
    phaseMsByNameMean: Record<string, number>;
    phaseMsPerAcceptedTokenMean: Record<string, number>;
  };
};

type GateProfile = 'safety' | 'flat-m1' | 'flat-m2' | 'paged-parity' | 'custom';

type GateFailure = {
  cellId: string;
  metric: string;
  actual: number | null;
  threshold: number;
  comparator: '>=' | '<=';
};

type GateSummary = {
  profile: GateProfile;
  passed: boolean;
  evaluatedCellIds: string[];
  failures: GateFailure[];
  warnings: string[];
};

type GateConfig = {
  profile: GateProfile;
  minSpeedup?: number;
  minMeanAccepted?: number;
  minPosAcceptance?: number[];
  gatePaged: boolean;
  allowTextDivergence: boolean;
};

type MatrixReport = {
  schemaVersion: 2;
  tool: 'qwen35-mtp-matrix';
  createdAt: string;
  host: string;
  model: {
    name: string;
    path: string;
    hasMtpWeights: boolean | null;
    hasBlockPagedCache: boolean | null;
  };
  runConfig: {
    prompt: string;
    maxNewTokens: number;
    temperature: number;
    topK: number;
    topP: number;
    draftTemperature: number | null;
    draftTopK: number | null;
    draftTopP: number | null;
    reasoningEffort: 'none';
    includeReasoning: false;
    repeats: number;
    warmup: boolean;
    profile: boolean;
    acceptanceTrace: boolean;
    depths: DepthSpec[];
    envVariants: EnvVariant[];
  };
  matrix: CellResult[];
  summary: {
    bestBySpeedup: string | null;
    bestExperimentalBySpeedup: string | null;
    failures: Array<{ cellId: string; reason: string }>;
    gate?: GateSummary;
  };
};

const COMPARED_ENV_KEYS = [
  'MLX_QWEN35_PAGED_OVERRIDE',
  'MLX_MTP_VERIFY_PAGED_ATTN',
  'MLX_MTP_SPARSE_ACCEPT',
  'MLX_MTP_VERIFY_ASYNC_EVAL',
  'MLX_MTP_DEFER_VERIFY_HIDDEN',
  'MLX_MTP_TARGET_DISTRIBUTION_FIRST',
  'MLX_MTP_USE_TAPE_REPLAY',
  'MLX_MTP_CHAINED_CYCLES',
  'MLX_MTP_NO_PROMPT_PREFILL',
  'MLX_MTP_SMALL_M_QMV',
  'MLX_MTP_SMALL_M_QMV_BITS',
  'MLX_MTP_SMALL_M_QMV_MODES',
  'MLX_MTP_SMALL_M_QMM',
  'MLX_MTP_BATCH_TARGET_ARRAYS',
  'MLX_MTP_NATIVE_SPARSE_VERIFY',
  'MLX_MTP_GREEDY_ARGMAX_ONLY_VERIFY',
  'MLX_MTP_SAMPLER_PARITY',
  'MLX_MTP_DRAFT_TEMPERATURE',
  'MLX_MTP_DRAFT_TOP_K',
  'MLX_MTP_DRAFT_TOP_P',
  'MLX_MTP_DRAFT_TEMPERATURE_SCALE',
  'MLX_MTP_ADAPTIVE_DEPTH_MODE',
  'MLX_MTP_EV_BASE_DEPTH',
  'MLX_MTP_EV_ACCEPT_PRIORS',
  'MLX_MTP_EV_EWMA_ALPHA',
  'MLX_MTP_EV_DRAFT_COST_S',
  'MLX_MTP_EV_EXTRA_VERIFY_COST_S',
  'MLX_MTP_EV_BASELINE_TOK_S',
  'MLX_MTP_EV_SAFETY_MARGIN',
  'MLX_MTP_EV_CONFIDENCE_WEIGHT',
  'MLX_MTP_EV_MIN_EXTRA_ACCEPT_PROBABILITY',
  'MLX_MTP_EV_ALLOW_DEEPEN',
  'MLX_MTP_TRACE_ACCEPTANCE',
] as const;

const PERF_FLAT_ENV: Record<string, string> = {
  MLX_QWEN35_PAGED_OVERRIDE: '0',
  MLX_MTP_VERIFY_PAGED_ATTN: '0',
  MLX_MTP_SPARSE_ACCEPT: '1',
  MLX_MTP_VERIFY_ASYNC_EVAL: '1',
  MLX_MTP_DEFER_VERIFY_HIDDEN: '1',
  MLX_MTP_USE_TAPE_REPLAY: '1',
  MLX_MTP_CHAINED_CYCLES: '1',
  MLX_MTP_NO_PROMPT_PREFILL: '0',
  MLX_MTP_SMALL_M_QMV: '1',
  MLX_MTP_SMALL_M_QMM: '0',
  MLX_MTP_BATCH_TARGET_ARRAYS: '1',
};

const ENV_VARIANTS: Record<string, EnvVariant> = {
  default: { name: 'default', env: {}, category: 'flat' },
  'perf-flat': {
    name: 'perf-flat',
    env: PERF_FLAT_ENV,
    category: 'flat',
  },
  'perf-flat-mtplx-sampler': {
    name: 'perf-flat-mtplx-sampler',
    env: { ...PERF_FLAT_ENV, MLX_MTP_SAMPLER_PARITY: 'mtplx' },
    category: 'ablation',
  },
  'perf-flat-mtplx-sampler-native-off': {
    name: 'perf-flat-mtplx-sampler-native-off',
    env: {
      ...PERF_FLAT_ENV,
      MLX_MTP_SAMPLER_PARITY: 'mtplx',
      MLX_MTP_NATIVE_SPARSE_VERIFY: '0',
    },
    category: 'ablation',
  },
  'perf-flat-native-sparse': {
    name: 'perf-flat-native-sparse',
    env: {
      ...PERF_FLAT_ENV,
      MLX_MTP_SAMPLER_PARITY: 'mtplx',
      MLX_MTP_NATIVE_SPARSE_VERIFY: '1',
    },
    category: 'ablation',
  },
  'perf-flat-argmax-only': {
    name: 'perf-flat-argmax-only',
    env: {
      ...PERF_FLAT_ENV,
      MLX_MTP_SAMPLER_PARITY: 'mtplx',
      MLX_MTP_NATIVE_SPARSE_VERIFY: '1',
      MLX_MTP_GREEDY_ARGMAX_ONLY_VERIFY: '1',
    },
    category: 'ablation',
  },
  'target-first-on': {
    name: 'target-first-on',
    env: { MLX_MTP_TARGET_DISTRIBUTION_FIRST: '1' },
    category: 'ablation',
  },
  'perf-flat-target-first-on': {
    name: 'perf-flat-target-first-on',
    env: { ...PERF_FLAT_ENV, MLX_MTP_TARGET_DISTRIBUTION_FIRST: '1' },
    category: 'ablation',
  },
  'perf-flat-qmv-off': {
    name: 'perf-flat-qmv-off',
    env: {
      MLX_QWEN35_PAGED_OVERRIDE: '0',
      MLX_MTP_VERIFY_PAGED_ATTN: '0',
      MLX_MTP_SPARSE_ACCEPT: '1',
      MLX_MTP_VERIFY_ASYNC_EVAL: '1',
      MLX_MTP_DEFER_VERIFY_HIDDEN: '1',
      MLX_MTP_USE_TAPE_REPLAY: '1',
      MLX_MTP_CHAINED_CYCLES: '1',
      MLX_MTP_NO_PROMPT_PREFILL: '0',
      MLX_MTP_SMALL_M_QMV: '0',
      MLX_MTP_SMALL_M_QMM: '0',
      MLX_MTP_BATCH_TARGET_ARRAYS: '1',
    },
    category: 'ablation',
  },
  'perf-flat-ev-deepen': {
    name: 'perf-flat-ev-deepen',
    env: {
      MLX_QWEN35_PAGED_OVERRIDE: '0',
      MLX_MTP_VERIFY_PAGED_ATTN: '0',
      MLX_MTP_SPARSE_ACCEPT: '1',
      MLX_MTP_VERIFY_ASYNC_EVAL: '1',
      MLX_MTP_DEFER_VERIFY_HIDDEN: '1',
      MLX_MTP_USE_TAPE_REPLAY: '1',
      MLX_MTP_CHAINED_CYCLES: '1',
      MLX_MTP_NO_PROMPT_PREFILL: '0',
      MLX_MTP_SMALL_M_QMV: '1',
      MLX_MTP_SMALL_M_QMM: '0',
      MLX_MTP_BATCH_TARGET_ARRAYS: '1',
      MLX_MTP_EV_ALLOW_DEEPEN: '1',
    },
    category: 'ablation',
  },
  'perf-flat-qmv-broad': {
    name: 'perf-flat-qmv-broad',
    env: {
      MLX_QWEN35_PAGED_OVERRIDE: '0',
      MLX_MTP_VERIFY_PAGED_ATTN: '0',
      MLX_MTP_SPARSE_ACCEPT: '1',
      MLX_MTP_VERIFY_ASYNC_EVAL: '1',
      MLX_MTP_DEFER_VERIFY_HIDDEN: '1',
      MLX_MTP_USE_TAPE_REPLAY: '1',
      MLX_MTP_CHAINED_CYCLES: '1',
      MLX_MTP_NO_PROMPT_PREFILL: '0',
      MLX_MTP_SMALL_M_QMV: '1',
      MLX_MTP_SMALL_M_QMV_BITS: '4,5,6,8',
      MLX_MTP_SMALL_M_QMV_MODES: 'affine,nvfp4',
      MLX_MTP_SMALL_M_QMM: '0',
      MLX_MTP_BATCH_TARGET_ARRAYS: '1',
    },
    category: 'ablation',
  },
  'paged-off': {
    name: 'paged-off',
    env: { MLX_QWEN35_PAGED_OVERRIDE: '0', MLX_MTP_VERIFY_PAGED_ATTN: '0' },
    category: 'flat',
  },
  'paged-on': {
    name: 'paged-on',
    env: { MLX_QWEN35_PAGED_OVERRIDE: '1', MLX_MTP_VERIFY_PAGED_ATTN: '1' },
    category: 'paged',
    experimental: true,
  },
  'sparse-off': { name: 'sparse-off', env: { MLX_MTP_SPARSE_ACCEPT: '0' }, category: 'ablation' },
  'sparse-on': { name: 'sparse-on', env: { MLX_MTP_SPARSE_ACCEPT: '1' }, category: 'ablation' },
  'async-off': {
    name: 'async-off',
    env: { MLX_MTP_VERIFY_ASYNC_EVAL: '0' },
    category: 'ablation',
  },
  'defer-hidden-off': {
    name: 'defer-hidden-off',
    env: {
      MLX_MTP_DEFER_VERIFY_HIDDEN: '0',
      MLX_QWEN35_PAGED_OVERRIDE: '0',
      MLX_MTP_VERIFY_PAGED_ATTN: '0',
    },
    category: 'ablation',
  },
  'tape-off': { name: 'tape-off', env: { MLX_MTP_USE_TAPE_REPLAY: '0' }, category: 'ablation' },
  'mtplx-qmv': { name: 'mtplx-qmv', env: { MLX_MTP_SMALL_M_QMV: '1' }, category: 'ablation' },
  'small-m-qmm': { name: 'small-m-qmm', env: { MLX_MTP_SMALL_M_QMM: '1' }, category: 'ablation' },
  'batch-target-off': {
    name: 'batch-target-off',
    env: {
      MLX_QWEN35_PAGED_OVERRIDE: '0',
      MLX_MTP_VERIFY_PAGED_ATTN: '0',
      MLX_MTP_SPARSE_ACCEPT: '1',
      MLX_MTP_VERIFY_ASYNC_EVAL: '1',
      MLX_MTP_DEFER_VERIFY_HIDDEN: '1',
      MLX_MTP_USE_TAPE_REPLAY: '1',
      MLX_MTP_CHAINED_CYCLES: '1',
      MLX_MTP_NO_PROMPT_PREFILL: '0',
      MLX_MTP_SMALL_M_QMV: '1',
      MLX_MTP_SMALL_M_QMM: '0',
      MLX_MTP_BATCH_TARGET_ARRAYS: '0',
    },
    category: 'ablation',
  },
  'chained-on': {
    name: 'chained-on',
    env: {
      MLX_MTP_CHAINED_CYCLES: '1',
      MLX_QWEN35_PAGED_OVERRIDE: '0',
      MLX_MTP_VERIFY_PAGED_ATTN: '0',
    },
    category: 'ablation',
  },
  'prompt-prefill-off': {
    name: 'prompt-prefill-off',
    env: { MLX_MTP_NO_PROMPT_PREFILL: '1' },
    category: 'ablation',
  },
};

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    child: { type: 'boolean' },
    cell: { type: 'string' },
    repeat: { type: 'string' },
    depths: { type: 'string' },
    'env-variants': { type: 'string' },
    repeats: { type: 'string' },
    'max-tokens': { type: 'string' },
    temperature: { type: 'string' },
    'top-k': { type: 'string' },
    'top-p': { type: 'string' },
    'draft-temperature': { type: 'string' },
    'draft-top-k': { type: 'string' },
    'draft-top-p': { type: 'string' },
    prompt: { type: 'string' },
    json: { type: 'string' },
    profile: { type: 'boolean' },
    'accept-trace': { type: 'boolean' },
    'no-warmup': { type: 'boolean' },
    gate: { type: 'string' },
    'include-paged-experimental': { type: 'boolean' },
    'min-speedup': { type: 'string' },
    'min-mean-accepted': { type: 'string' },
    'min-pos-acceptance': { type: 'string' },
    'gate-paged': { type: 'boolean' },
    'allow-text-divergence': { type: 'boolean' },
  },
  allowPositionals: true,
});

const modelName = positionals[0] || 'qwen3.5-4b';
const modelPath = resolve(process.cwd(), '.cache', 'models', modelName);
const maxTokens = Number(values['max-tokens'] ?? '128');
const temperature = parseNumber(values.temperature, 'temperature', 0);
const topK = parseNumber(values['top-k'], 'top-k', 1);
const topP = parseNumber(values['top-p'], 'top-p', 1);
const draftTemperature = parseOptionalNumber(values['draft-temperature'], '--draft-temperature');
const draftTopK = parseOptionalNumber(values['draft-top-k'], '--draft-top-k');
const draftTopP = parseOptionalNumber(values['draft-top-p'], '--draft-top-p');
const prompt =
  values.prompt ??
  'Write a concise three-paragraph essay on why deterministic sampling at temperature 0 is useful for testing speculative decoding implementations.';
const profile = values.profile === true;
const captureAcceptanceTrace = values['accept-trace'] === true || envFlagTruthy(process.env.MLX_MTP_TRACE_ACCEPTANCE);
const warmup = values['no-warmup'] !== true;
const includePagedExperimental = values['include-paged-experimental'] === true;
const gateConfig = parseGateConfig();

if (values.child) {
  const cell = parseCell(values.cell);
  const repeat = Number(values.repeat ?? '0');
  const sample = await runWorker(cell, repeat);
  console.log(`MTP_MATRIX_RESULT ${JSON.stringify(sample)}`);
} else {
  const depths = parseDepths(values.depths ?? 'default,1,2,3,adaptive');
  const envVariants = parseEnvVariants(values['env-variants'] ?? defaultEnvVariants(includePagedExperimental));
  const repeats = Number(values.repeats ?? '1');
  const matrix = await runMaster(depths, envVariants, repeats);
  const report = buildReport(depths, envVariants, repeats, matrix);
  if (gateConfig) {
    report.summary.gate = evaluateGate(report, gateConfig);
    if (!report.summary.gate.passed) {
      process.exitCode = 2;
    }
  }
  if (values.json) {
    await writeFile(values.json, JSON.stringify(report, null, 2));
    console.error(`Wrote ${values.json}`);
  }
  console.log(JSON.stringify(report, null, 2));
}

async function runMaster(depths: DepthSpec[], envVariants: EnvVariant[], repeats: number): Promise<CellResult[]> {
  const matrix: CellResult[] = [];
  for (const depth of depths) {
    for (const envVariant of envVariants) {
      const cell: CellSpec = {
        cellId: `${formatDepth(depth)}__${envVariant.name}`,
        depth,
        envVariant,
      };
      const samples: SampleResult[] = [];
      for (let repeat = 0; repeat < repeats; repeat++) {
        console.error(`Running ${cell.cellId} repeat ${repeat + 1}/${repeats}`);
        const child = spawnSync(
          process.argv[0],
          [
            process.argv[1],
            modelName,
            '--child',
            '--cell',
            JSON.stringify(cell),
            '--repeat',
            String(repeat),
            '--max-tokens',
            String(maxTokens),
            '--temperature',
            String(temperature),
            '--top-k',
            String(topK),
            '--top-p',
            String(topP),
            ...(draftTemperature == null ? [] : ['--draft-temperature', String(draftTemperature)]),
            ...(draftTopK == null ? [] : ['--draft-top-k', String(draftTopK)]),
            ...(draftTopP == null ? [] : ['--draft-top-p', String(draftTopP)]),
            '--prompt',
            prompt,
            ...(profile ? ['--profile'] : []),
            ...(captureAcceptanceTrace ? ['--accept-trace'] : []),
            ...(warmup ? [] : ['--no-warmup']),
          ],
          {
            cwd: process.cwd(),
            env: childEnv(envVariant, depth),
            encoding: 'utf8',
            maxBuffer: 1024 * 1024 * 64,
          },
        );
        process.stderr.write(child.stderr);
        const marker = child.stdout.split('\n').find((line) => line.startsWith('MTP_MATRIX_RESULT '));
        if (!marker) {
          const stdoutTail = tail(child.stdout);
          const stderrTail = tail(child.stderr);
          samples.push({
            repeat,
            status: 'failed',
            durationMs: 0,
            error: [
              `worker exited without result; status=${child.status} signal=${child.signal ?? 'none'}`,
              stdoutTail ? `stdout tail:\n${stdoutTail}` : '',
              stderrTail ? `stderr tail:\n${stderrTail}` : '',
            ]
              .filter(Boolean)
              .join('\n\n'),
          });
          continue;
        }
        const sample = JSON.parse(marker.slice('MTP_MATRIX_RESULT '.length)) as SampleResult;
        if (captureAcceptanceTrace) {
          const trace = parseAcceptanceTrace(child.stderr);
          if (trace.length) sample.acceptanceTrace = trace;
        }
        samples.push(sample);
      }
      matrix.push({ ...cell, samples, aggregate: aggregateSamples(samples) });
    }
  }
  return matrix;
}

async function runWorker(cell: CellSpec, repeat: number): Promise<SampleResult> {
  const started = Date.now();
  try {
    console.error(`Loading model from: ${modelPath}`);
    const loaded = await loadModel(modelPath);
    if (loaded instanceof HarrierModel) {
      throw new Error('Embedding model is not session-capable.');
    }
    const model = loaded as unknown as SessionCapableModel;
    const hasMtpWeights = model.hasMtpWeights?.() === true;
    const hasBlockPagedCache = model.hasBlockPagedCache?.() === true;
    if (!hasMtpWeights) {
      return {
        repeat,
        status: 'skipped',
        model: { hasMtpWeights, hasBlockPagedCache },
        skipReason: 'model has no MTP weights',
        durationMs: Date.now() - started,
      };
    }
    if (
      cell.envVariant.env.MLX_MTP_VERIFY_PAGED_ATTN != null &&
      cell.envVariant.env.MLX_QWEN35_PAGED_OVERRIDE !== '0' &&
      !hasBlockPagedCache
    ) {
      console.error('Paged-verify env variant has no effect because hasBlockPagedCache=false');
    }

    if (profile) {
      resetProfilingData();
      setProfilingEnabled(true);
    }
    if (warmup) {
      await runPair(model, cell, 'warmup');
      if (profile) {
        resetProfilingData();
      }
    }

    const { ar, mtp } = await runPair(model, cell, 'measured');
    const profiling = profile ? profilingForPair() : undefined;
    if (profile) {
      setProfilingEnabled(false);
    }

    return {
      repeat,
      status: 'ok',
      model: { hasMtpWeights, hasBlockPagedCache },
      durationMs: Date.now() - started,
      ar,
      mtp,
      comparison: compareRuns(ar, mtp),
      profiling,
    };
  } catch (error) {
    if (profile) {
      setProfilingEnabled(false);
    }
    return {
      repeat,
      status: 'failed',
      durationMs: Date.now() - started,
      error: error instanceof Error ? (error.stack ?? error.message) : String(error),
    };
  }
}

async function runPair(
  model: SessionCapableModel,
  cell: CellSpec,
  label: string,
): Promise<{ ar: BenchRun; mtp: BenchRun }> {
  const baseConfig: ChatConfig = {
    temperature,
    topK,
    topP,
    maxNewTokens: maxTokens,
    reasoningEffort: 'none',
    includeReasoning: false,
    reportPerformance: true,
  };
  const ar = await runOnce(model, { ...baseConfig, enableMtp: false }, `${label} AR`);
  const mtpConfig: ChatConfig = {
    ...baseConfig,
    enableMtp: true,
    ...(cell.depth.mode === 'pinned'
      ? { mtpDepth: cell.depth.mtpDepth, mtpAdaptiveDepth: false }
      : cell.depth.mode === 'adaptive'
        ? { mtpAdaptiveDepth: true, ...(cell.depth.mtpDepth == null ? {} : { mtpDepth: cell.depth.mtpDepth }) }
        : {}),
  };
  const mtp = await runOnce(model, mtpConfig, `${label} MTP`);
  return { ar, mtp };
}

async function runOnce(model: SessionCapableModel, defaultConfig: ChatConfig, label: string): Promise<BenchRun> {
  const session = new ChatSession(model, {
    system: 'You are a precise assistant. Be concise.',
    defaultConfig,
  });
  const result = await session.send(prompt);
  const tps = result.performance?.decodeTokensPerSecond?.toFixed(2) ?? 'n/a';
  const mean = result.performance?.mtpMeanAcceptedTokens?.toFixed(2);
  console.error(`${label}: tokens=${result.numTokens} decode=${tps} tok/s mtpMean=${mean ?? '-'}`);
  return benchRun(result);
}

function benchRun(result: ChatResult): BenchRun {
  return {
    numTokens: result.numTokens,
    promptTokens: result.promptTokens,
    cachedTokens: result.cachedTokens,
    finishReason: result.finishReason,
    text: result.text,
    textSha256: sha256(result.text),
    textLength: result.text.length,
    performance: result.performance
      ? {
          ttftMs: result.performance.ttftMs,
          prefillTokensPerSecond: result.performance.prefillTokensPerSecond,
          decodeTokensPerSecond: result.performance.decodeTokensPerSecond,
          mtpMeanAcceptedTokens: result.performance.mtpMeanAcceptedTokens,
          mtpAcceptanceByPosition: result.performance.mtpAcceptanceByPosition,
          mtpCycles: result.performance.mtpCycles,
          mtpMeanDepth: result.performance.mtpMeanDepth,
          phaseMsByName: phaseMap(result.performance.profilePhases),
        }
      : undefined,
  };
}

function compareRuns(ar: BenchRun, mtp: BenchRun): SampleResult['comparison'] {
  const arTps = ar.performance?.decodeTokensPerSecond ?? 0;
  const mtpTps = mtp.performance?.decodeTokensPerSecond ?? 0;
  const perPos = mtp.performance?.mtpAcceptanceByPosition ?? [];
  const meanAccepted = mtp.performance?.mtpMeanAcceptedTokens ?? 0;
  const pos0 = perPos[0] ?? 0;
  return {
    decodeSpeedup: arTps > 0 && mtpTps > 0 ? mtpTps / arTps : null,
    textEqual: ar.textSha256 === mtp.textSha256 && ar.textLength === mtp.textLength,
    firstDiffChar: firstDiffChar(ar.text, mtp.text),
    acceptanceOk: meanAccepted >= 0.5 && pos0 >= 0.4,
    planPos0GateOk: pos0 >= 0.4,
  };
}

function firstDiffChar(left: string, right: string): number | null {
  const min = Math.min(left.length, right.length);
  for (let i = 0; i < min; i++) {
    if (left[i] !== right[i]) return i;
  }
  return left.length === right.length ? null : min;
}

function profilingForPair(): SampleResult['profiling'] {
  const data = getProfilingData();
  return {
    summary: data.summary,
    arGeneration: data.generations.find((g) => !g.mtpCycles),
    mtpGeneration: [...data.generations].reverse().find((g) => !!g.mtpCycles),
  };
}

function aggregateSamples(samples: SampleResult[]): CellResult['aggregate'] {
  const ok = samples.filter((s) => s.status === 'ok');
  const phaseMaps = ok.map((s) => s.mtp?.performance?.phaseMsByName);
  const phaseMsByNameMean = meanPhaseMaps(phaseMaps);
  return {
    okSamples: ok.length,
    failedSamples: samples.filter((s) => s.status === 'failed').length,
    skippedSamples: samples.filter((s) => s.status === 'skipped').length,
    textEqualRate: mean(ok.map((s) => (s.comparison?.textEqual == null ? undefined : s.comparison.textEqual ? 1 : 0))),
    firstDiffCharMin: minFinite(ok.map((s) => s.comparison?.firstDiffChar)),
    arDecodeTpsMean: mean(ok.map((s) => s.ar?.performance?.decodeTokensPerSecond)),
    arDecodeTpsMin: minFinite(ok.map((s) => s.ar?.performance?.decodeTokensPerSecond)),
    speedupMean: mean(ok.map((s) => s.comparison?.decodeSpeedup)),
    speedupMin: minFinite(ok.map((s) => s.comparison?.decodeSpeedup)),
    speedupStddev: stddev(ok.map((s) => s.comparison?.decodeSpeedup)),
    mtpDecodeTpsMean: mean(ok.map((s) => s.mtp?.performance?.decodeTokensPerSecond)),
    mtpDecodeTpsMin: minFinite(ok.map((s) => s.mtp?.performance?.decodeTokensPerSecond)),
    meanAcceptedMean: mean(ok.map((s) => s.mtp?.performance?.mtpMeanAcceptedTokens)),
    meanAcceptedMin: minFinite(ok.map((s) => s.mtp?.performance?.mtpMeanAcceptedTokens)),
    pos0AcceptanceMean: mean(ok.map((s) => s.mtp?.performance?.mtpAcceptanceByPosition?.[0])),
    acceptanceByPositionMean: vectorAggregate(
      ok.map((s) => s.mtp?.performance?.mtpAcceptanceByPosition),
      mean,
    ),
    acceptanceByPositionMin: vectorAggregate(
      ok.map((s) => s.mtp?.performance?.mtpAcceptanceByPosition),
      minFinite,
    ),
    mtpCyclesMean: mean(ok.map((s) => s.mtp?.performance?.mtpCycles)),
    mtpMeanDepthMean: mean(ok.map((s) => s.mtp?.performance?.mtpMeanDepth)),
    phaseMsByNameMean,
    phaseMsPerAcceptedTokenMean: meanPhaseMaps(
      ok.map((s) => {
        const phases = s.mtp?.performance?.phaseMsByName;
        const cycles = s.mtp?.performance?.mtpCycles ?? 0;
        const meanAccepted = s.mtp?.performance?.mtpMeanAcceptedTokens ?? 0;
        const acceptedTokens = cycles * meanAccepted;
        if (!phases || acceptedTokens <= 0) return undefined;
        return Object.fromEntries(Object.entries(phases).map(([name, ms]) => [name, ms / acceptedTokens]));
      }),
    ),
  };
}

function buildReport(
  depths: DepthSpec[],
  envVariants: EnvVariant[],
  repeats: number,
  matrix: CellResult[],
): MatrixReport {
  const nonExperimental = matrix.filter((cell) => !cell.envVariant.experimental);
  const best = nonExperimental
    .filter((cell) => cell.aggregate.speedupMean != null)
    .sort((a, b) => (b.aggregate.speedupMean ?? 0) - (a.aggregate.speedupMean ?? 0))[0];
  const bestExperimental = matrix
    .filter((cell) => cell.envVariant.experimental && cell.aggregate.speedupMean != null)
    .sort((a, b) => (b.aggregate.speedupMean ?? 0) - (a.aggregate.speedupMean ?? 0))[0];
  const modelSample = matrix.flatMap((cell) => cell.samples).find((sample) => sample.model)?.model;
  return {
    schemaVersion: 2,
    tool: 'qwen35-mtp-matrix',
    createdAt: new Date().toISOString(),
    host: hostname(),
    model: {
      name: modelName,
      path: modelPath,
      hasMtpWeights: modelSample?.hasMtpWeights ?? null,
      hasBlockPagedCache: modelSample?.hasBlockPagedCache ?? null,
    },
    runConfig: {
      prompt,
      maxNewTokens: maxTokens,
      temperature,
      topK,
      topP,
      draftTemperature: draftTemperature ?? null,
      draftTopK: draftTopK ?? null,
      draftTopP: draftTopP ?? null,
      reasoningEffort: 'none',
      includeReasoning: false,
      repeats,
      warmup,
      profile,
      acceptanceTrace: captureAcceptanceTrace,
      depths,
      envVariants,
    },
    matrix,
    summary: {
      bestBySpeedup: best?.cellId ?? null,
      bestExperimentalBySpeedup: bestExperimental?.cellId ?? null,
      failures: matrix.flatMap((cell) =>
        cell.samples
          .filter((sample) => sample.status === 'failed')
          .map((sample) => ({ cellId: cell.cellId, reason: sample.error ?? 'failed' })),
      ),
    },
  };
}

function evaluateGate(report: MatrixReport, config: GateConfig): GateSummary {
  if (config.profile === 'paged-parity' || config.gatePaged) {
    return evaluatePagedParityGate(report);
  }

  const thresholds = gateThresholds(config);
  const cells = report.matrix.filter((cell) => {
    if (cell.envVariant.experimental) return false;
    if (cell.envVariant.category === 'paged') return false;
    return config.profile === 'safety' || cell.envVariant.category === 'flat';
  });
  const failures: GateFailure[] = [];
  const warnings: string[] = [];
  for (const cell of cells) {
    if (cell.aggregate.okSamples === 0) {
      warnings.push(`${cell.cellId}: no successful samples`);
      continue;
    }
    checkMin(failures, cell.cellId, 'speedupMin', cell.aggregate.speedupMin, thresholds.minSpeedup);
    if ((cell.aggregate.textEqualRate ?? 1) < 1) {
      if (config.allowTextDivergence) {
        warnings.push(`${cell.cellId}: AR/MTP text diverged at char ${cell.aggregate.firstDiffCharMin ?? 'unknown'}`);
      } else {
        checkMin(failures, cell.cellId, 'textEqualRate', cell.aggregate.textEqualRate, 1);
      }
    }
    checkMin(failures, cell.cellId, 'meanAcceptedMin', cell.aggregate.meanAcceptedMin, thresholds.minMeanAccepted);
    checkMin(failures, cell.cellId, 'mtpCyclesMean', cell.aggregate.mtpCyclesMean, 1);
    for (let i = 0; i < thresholds.minPosAcceptance.length; i++) {
      checkMin(
        failures,
        cell.cellId,
        `acceptanceByPositionMin[${i}]`,
        cell.aggregate.acceptanceByPositionMin[i] ?? null,
        thresholds.minPosAcceptance[i],
      );
    }
  }
  if (cells.length === 0) {
    warnings.push('no cells matched the requested gate profile');
  }
  return {
    profile: config.profile,
    passed: failures.length === 0 && cells.length > 0,
    evaluatedCellIds: cells.map((cell) => cell.cellId),
    failures,
    warnings,
  };
}

function evaluatePagedParityGate(report: MatrixReport): GateSummary {
  const flat = report.matrix.find((cell) => cell.cellId === 'depth3__paged-off');
  const paged = report.matrix.find((cell) => cell.cellId === 'depth3__paged-on');
  const failures: GateFailure[] = [];
  const warnings: string[] = [];
  if (!flat) warnings.push('depth3__paged-off is required for paged-parity');
  if (!paged) warnings.push('depth3__paged-on is required for paged-parity');
  if (flat && paged) {
    const flatAccepted = flat.aggregate.meanAcceptedMean;
    const pagedAccepted = paged.aggregate.meanAcceptedMean;
    const ratio =
      flatAccepted != null && flatAccepted > 0 && pagedAccepted != null ? pagedAccepted / flatAccepted : null;
    checkMin(failures, paged.cellId, 'meanAcceptedVsFlatRatio', ratio, 0.9);
    checkMin(failures, paged.cellId, 'speedupMin', paged.aggregate.speedupMin, 1.0);
  }
  return {
    profile: 'paged-parity',
    passed: failures.length === 0 && warnings.length === 0,
    evaluatedCellIds: [flat?.cellId, paged?.cellId].filter((v): v is string => !!v),
    failures,
    warnings,
  };
}

function gateThresholds(
  config: GateConfig,
): Required<Pick<GateConfig, 'minSpeedup' | 'minMeanAccepted' | 'minPosAcceptance'>> {
  if (config.profile === 'flat-m2') {
    return {
      minSpeedup: config.minSpeedup ?? 2.0,
      minMeanAccepted: config.minMeanAccepted ?? 2.5,
      minPosAcceptance: config.minPosAcceptance ?? [0.9, 0.8, 0.6],
    };
  }
  if (config.profile === 'flat-m1') {
    return {
      minSpeedup: config.minSpeedup ?? 1.6,
      minMeanAccepted: config.minMeanAccepted ?? 2.1,
      minPosAcceptance: config.minPosAcceptance ?? [0.9, 0.75, 0.45],
    };
  }
  if (config.profile === 'custom') {
    return {
      minSpeedup: config.minSpeedup ?? 1.0,
      minMeanAccepted: config.minMeanAccepted ?? 0.5,
      minPosAcceptance: config.minPosAcceptance ?? [0.4],
    };
  }
  return {
    minSpeedup: config.minSpeedup ?? 0,
    minMeanAccepted: config.minMeanAccepted ?? 0.5,
    minPosAcceptance: config.minPosAcceptance ?? [0.4],
  };
}

function checkMin(
  failures: GateFailure[],
  cellId: string,
  metric: string,
  actual: number | null | undefined,
  threshold: number,
): void {
  const value = typeof actual === 'number' && Number.isFinite(actual) ? actual : null;
  if (value == null || value < threshold) {
    failures.push({ cellId, metric, actual: value, threshold, comparator: '>=' });
  }
}

function parseDepths(raw: string): DepthSpec[] {
  return raw.split(',').map((part) => {
    const value = part.trim();
    if (value === 'default') return { mode: 'default' };
    if (value === 'adaptive') return { mode: 'adaptive', adaptiveKind: 'throughput' };
    if (value === 'adaptive-ev') return { mode: 'adaptive', adaptiveKind: 'expected-value', mtpDepth: 3 };
    const depth = Number(value);
    if (depth === 1 || depth === 2 || depth === 3) return { mode: 'pinned', mtpDepth: depth };
    throw new Error(`Unsupported depth '${value}'. Use default,1,2,3,adaptive,adaptive-ev.`);
  });
}

function parseEnvVariants(raw: string): EnvVariant[] {
  return raw.split(',').map((part) => {
    const name = part.trim();
    const variant = ENV_VARIANTS[name];
    if (!variant) {
      throw new Error(`Unknown env variant '${name}'. Available: ${Object.keys(ENV_VARIANTS).join(', ')}`);
    }
    return variant;
  });
}

function defaultEnvVariants(includeExperimentalPaged: boolean): string {
  return includeExperimentalPaged ? 'perf-flat,paged-off,sparse-off,paged-on' : 'perf-flat,paged-off,sparse-off';
}

function parseCell(raw: string | undefined): CellSpec {
  if (!raw) throw new Error('--cell is required in child mode');
  return JSON.parse(raw) as CellSpec;
}

function parseGateConfig(): GateConfig | null {
  if (!values.gate && !values['gate-paged']) return null;
  const profile = parseGateProfile(values.gate ?? (values['gate-paged'] ? 'paged-parity' : undefined));
  return {
    profile,
    minSpeedup: parseOptionalNumber(values['min-speedup'], '--min-speedup'),
    minMeanAccepted: parseOptionalNumber(values['min-mean-accepted'], '--min-mean-accepted'),
    minPosAcceptance: parseOptionalNumberList(values['min-pos-acceptance'], '--min-pos-acceptance'),
    gatePaged: values['gate-paged'] === true,
    allowTextDivergence: values['allow-text-divergence'] === true,
  };
}

function parseGateProfile(raw: string | undefined): GateProfile {
  const value = raw ?? 'safety';
  if (
    value === 'safety' ||
    value === 'flat-m1' ||
    value === 'flat-m2' ||
    value === 'paged-parity' ||
    value === 'custom'
  ) {
    return value;
  }
  throw new Error(`Unsupported --gate '${value}'. Use safety,flat-m1,flat-m2,paged-parity,custom.`);
}

function parseOptionalNumber(raw: string | undefined, label: string): number | undefined {
  if (raw == null || raw.trim() === '') return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${label} must be a finite number`);
  return value;
}

function parseNumber(raw: string | undefined, label: string, fallback: number): number {
  const value = parseOptionalNumber(raw, `--${label}`);
  return value ?? fallback;
}

function parseOptionalNumberList(raw: string | undefined, label: string): number[] | undefined {
  if (raw == null || raw.trim() === '') return undefined;
  const values = raw.split(',').map((part) => parseOptionalNumber(part, label));
  return values.map((value) => {
    if (value == null) throw new Error(`${label} contains an empty item`);
    return value;
  });
}

function childEnv(envVariant: EnvVariant, depth: DepthSpec): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env };
  for (const key of COMPARED_ENV_KEYS) {
    delete env[key];
  }
  for (const [key, value] of Object.entries(envVariant.env)) {
    env[key] = value;
  }
  if (draftTemperature != null) env.MLX_MTP_DRAFT_TEMPERATURE = String(draftTemperature);
  if (draftTopK != null) env.MLX_MTP_DRAFT_TOP_K = String(Math.trunc(draftTopK));
  if (draftTopP != null) env.MLX_MTP_DRAFT_TOP_P = String(draftTopP);
  if (depth.mode === 'adaptive' && depth.adaptiveKind === 'expected-value') {
    env.MLX_MTP_ADAPTIVE_DEPTH_MODE = 'expected-value';
    env.MLX_MTP_EV_BASE_DEPTH = env.MLX_MTP_EV_BASE_DEPTH ?? '1';
  }
  if (captureAcceptanceTrace) env.MLX_MTP_TRACE_ACCEPTANCE = '1';
  return env;
}

function parseAcceptanceTrace(stderr: string | null | undefined): MtpAcceptanceTrace[] {
  if (!stderr) return [];
  const prefix = 'MTP_TRACE_ACCEPTANCE ';
  const rows: MtpAcceptanceTrace[] = [];
  for (const line of stderr.split('\n')) {
    if (!line.startsWith(prefix)) continue;
    const payload = line.slice(prefix.length).trim();
    if (!payload) continue;
    try {
      rows.push(JSON.parse(payload) as MtpAcceptanceTrace);
    } catch (error) {
      rows.push({
        schema_version: 1,
        parse_error: error instanceof Error ? error.message : String(error),
        raw: payload,
      });
    }
  }
  return rows;
}

function envFlagTruthy(value: string | undefined): boolean {
  if (value == null) return false;
  const normalized = value.trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'on';
}

function formatDepth(depth: DepthSpec): string {
  if (depth.mode === 'default') return 'default';
  if (depth.mode === 'adaptive') {
    return depth.adaptiveKind === 'expected-value' ? 'adaptive-ev' : 'adaptive';
  }
  return `depth${depth.mtpDepth}`;
}

function tail(value: string | null | undefined, maxChars = 4000): string {
  if (!value) return '';
  const trimmed = value.trim();
  return trimmed.length <= maxChars ? trimmed : trimmed.slice(trimmed.length - maxChars);
}

function phaseMap(phases: PhaseProfile[] | undefined): Record<string, number> | undefined {
  if (!phases?.length) return undefined;
  return Object.fromEntries(phases.map((phase) => [phase.name, phase.totalMs]));
}

function sha256(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

function mean(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function minFinite(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (finite.length === 0) return null;
  return Math.min(...finite);
}

function stddev(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  if (finite.length < 2) return null;
  const avg = mean(finite)!;
  const variance = finite.reduce((sum, value) => sum + (value - avg) ** 2, 0) / finite.length;
  return Math.sqrt(variance);
}

function vectorAggregate(
  vectors: Array<number[] | null | undefined>,
  aggregate: (values: Array<number | null | undefined>) => number | null,
): number[] {
  const maxLength = Math.max(0, ...vectors.map((v) => v?.length ?? 0));
  const result: number[] = [];
  for (let i = 0; i < maxLength; i++) {
    const value = aggregate(vectors.map((v) => v?.[i]));
    if (value != null) result.push(value);
  }
  return result;
}

function meanPhaseMaps(maps: Array<Record<string, number> | null | undefined>): Record<string, number> {
  const names = new Set<string>();
  for (const map of maps) {
    for (const name of Object.keys(map ?? {})) names.add(name);
  }
  const result: Record<string, number> = {};
  for (const name of names) {
    const value = mean(maps.map((map) => map?.[name]));
    if (value != null) result[name] = value;
  }
  return result;
}
