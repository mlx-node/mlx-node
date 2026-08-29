import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { captureTeacherLogits, scoreAgainstTeacher, type EvalReport } from '@mlx-node/core';

import { readJsonlTexts } from './calibrate.js';

/** Options for {@link evalCache}. */
export interface EvalCacheOptions {
  /** Reference checkpoint to capture from — bf16 unless you deliberately
   * anchor on a quantized one, which is recorded rather than refused. */
  teacher: string;
  /** JSONL eval dataset; each row is `{"text": "..."}`. */
  dataset: string;
  /** Directory to write the teacher cache into. */
  cache: string;
  /** Number of dataset rows to capture (default 64). */
  rows?: number;
  /** Tokens kept per row (default 512); the first primes the forward, so
   * values below 2 are raised to 2. */
  seq?: number;
  /** Retained support width per position (default 1024); clamped to the
   * teacher's vocabulary. */
  topK?: number;
  /** Positions per head projection (default 64). */
  logitChunk?: number;
}

/** Options for {@link evalScore}. */
export interface EvalScoreOptions {
  /** Candidate checkpoint to score. */
  model: string;
  /** Teacher cache directory written by {@link evalCache}. */
  cache: string;
  /** Positions per head projection (default 64). */
  logitChunk?: number;
}

const DEFAULT_LOGIT_CHUNK = 64;

/**
 * Capture the teacher's next-token distribution over the eval set.
 *
 * Runs ONCE per (teacher, dataset, seq, top-k); every candidate is then scored
 * against what this wrote. Full rows go to native untouched — native tokenizes
 * raw (no chat template) and truncates to `seq` TOKENS rather than characters,
 * so a JS-side char cap would be tokenizer-blind. `seq` below 2 is raised to 2:
 * the first token primes the forward and has no target of its own.
 */
export async function evalCache(opts: EvalCacheOptions): Promise<number> {
  const teacherPath = resolve(opts.teacher);
  const cacheDir = resolve(opts.cache);
  const texts = readJsonlTexts(resolve(opts.dataset), opts.rows ?? 64);
  if (texts.length === 0) {
    throw new Error(`No {"text": ...} rows found in dataset ${opts.dataset}`);
  }
  mkdirSync(cacheDir, { recursive: true });
  return captureTeacherLogits(
    teacherPath,
    texts,
    opts.seq ?? 512,
    opts.topK ?? 1024,
    opts.logitChunk ?? DEFAULT_LOGIT_CHUNK,
    cacheDir,
  );
}

/** Teacher-force a candidate over the cached token ids and report its quality. */
export async function evalScore(opts: EvalScoreOptions): Promise<EvalReport> {
  return scoreAgainstTeacher(resolve(opts.model), resolve(opts.cache), opts.logitChunk ?? DEFAULT_LOGIT_CHUNK);
}

/**
 * Render a report the way an A/B reader wants it: every candidate number next
 * to the teacher's, and the top-K KL next to the mass it left out.
 */
export function formatReport(modelPath: string, report: EvalReport): string {
  const signed = (x: number, digits: number) => `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`;
  const nllDelta = signed(report.meanNll - report.teacherMeanNll, 4);
  const pplDelta = signed((report.perplexity / report.teacherPerplexity - 1) * 100, 2);
  return [
    `model            ${modelPath}`,
    `teacher          ${report.teacherPath}${report.teacherQuantized ? '   [QUANTIZED — numbers are relative to this, not to bf16]' : ''}`,
    `rows/positions   ${report.rows} / ${report.positions}`,
    `nll              ${report.meanNll.toFixed(4)}   (teacher ${report.teacherMeanNll.toFixed(4)}, ${nllDelta})`,
    `perplexity       ${report.perplexity.toFixed(3)}   (teacher ${report.teacherPerplexity.toFixed(3)}, ${pplDelta}%)`,
    `kl_topk          ${report.meanKlTopk.toFixed(5)}  (K=${report.topK}, teacher tail mass ${report.teacherTailMass.toFixed(5)})`,
    `top1_agreement   ${(report.top1Agreement * 100).toFixed(2)}%`,
  ].join('\n');
}

function printHelp() {
  console.log(`
Teacher-forced output quality for converted checkpoints

Usage:
  mlx eval cache --teacher <bf16-model> --dataset <jsonl> --cache <dir> [options]
  mlx eval score --model <checkpoint> --cache <dir> [options]

  cache   Run the bf16 reference ONCE and store its next-token distribution.
  score   Teacher-force a candidate over the cached token ids and report.

Cache Arguments:
  --teacher <path>      Reference checkpoint (bf16; a quantized one is
                        accepted but every number is then relative to it)
  --dataset <jsonl>     Eval dataset ({"text": "..."} rows) — must be HELD OUT
                        from any calibration or imatrix set
  --cache <dir>         Directory to write the teacher cache into
  --rows <int>          Dataset rows to capture (default: 64)
  --seq <int>           Tokens kept per row (default: 512, minimum: 2)
  --top-k <int>         Retained support per position (default: 1024,
                        clamped to the teacher's vocabulary)

Score Arguments:
  --model <path>        Candidate checkpoint to score
  --cache <dir>         Teacher cache directory
  --json                Emit the report as one JSON object

Shared Arguments:
  --logit-chunk <int>   Positions per head projection (default: ${DEFAULT_LOGIT_CHUNK})
  --help, -h            Show this help message

What it measures:
  Both checkpoints are teacher-forced over the SAME token ids (score mode reads
  them from the cache and never re-tokenizes). Reported per scored position:
  full-vocabulary NLL and perplexity, top-1 agreement with the teacher, and KL
  against the teacher over its cached top-K support. teacher_tail_mass is the
  probability mass that support leaves out — a KL is only comparable across
  checkpoints while that number is small.

  qwen3_5 and qwen3_5_moe. Runs the reference AR prefill lane, so it says
  nothing about paged, MTP or speculative decoding.

Example:
  mlx eval cache --teacher .cache/models/qwen3.8-27b --dataset eval.jsonl \\
    --cache /tmp/teacher-27b --rows 64 --seq 512
  mlx eval score --model .cache/models/qwen3.8-27b-unsloth-nvfp4-mlx --cache /tmp/teacher-27b
`);
}

function parsePositiveInt(flag: string, raw?: string): number | undefined {
  if (raw === undefined) return undefined;
  if (!/^[1-9]\d*$/.test(raw)) {
    console.error(`Error: ${flag} requires a positive integer value`);
    process.exit(1);
  }
  return Number(raw);
}

function requireModelDir(flag: string, path: string): void {
  if (!existsSync(join(path, 'config.json'))) {
    console.error(`Error: ${flag} model config not found: ${join(path, 'config.json')}`);
    process.exit(1);
  }
}

export async function run(argv: string[]) {
  const mode = argv[0];
  const { values: args } = parseArgs({
    args: argv.slice(1),
    options: {
      teacher: { type: 'string' },
      model: { type: 'string', short: 'm' },
      dataset: { type: 'string' },
      cache: { type: 'string' },
      rows: { type: 'string' },
      seq: { type: 'string' },
      'top-k': { type: 'string' },
      'logit-chunk': { type: 'string' },
      json: { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help || mode === undefined || mode === '--help' || mode === '-h') {
    printHelp();
    return;
  }
  if (mode !== 'cache' && mode !== 'score') {
    console.error(`Error: unknown subcommand "${mode}" — expected "cache" or "score"\n`);
    console.error('Use --help for usage information');
    process.exit(1);
  }

  const logitChunk = parsePositiveInt('--logit-chunk', args['logit-chunk']) ?? DEFAULT_LOGIT_CHUNK;
  const startTime = Date.now();

  try {
    if (mode === 'cache') {
      if (!args.teacher || !args.dataset || !args.cache) {
        console.error('Error: cache mode requires --teacher, --dataset and --cache\n');
        console.error('Use --help for usage information');
        process.exit(1);
      }
      const teacherPath = resolve(args.teacher);
      const datasetPath = resolve(args.dataset);
      requireModelDir('--teacher', teacherPath);
      if (!existsSync(datasetPath)) {
        console.error(`Error: dataset not found: ${datasetPath}`);
        process.exit(1);
      }
      const rows = parsePositiveInt('--rows', args.rows) ?? 64;
      const seq = parsePositiveInt('--seq', args.seq) ?? 512;
      const topK = parsePositiveInt('--top-k', args['top-k']) ?? 1024;

      console.log(`Teacher:  ${teacherPath}`);
      console.log(`Dataset:  ${datasetPath}`);
      console.log(`Cache:    ${resolve(args.cache)}`);
      console.log(`Capture:  ${rows} rows x ${seq} tokens, top-${topK}`);
      console.log('');
      console.log('Running teacher-forced capture (this may take a while)...');

      const captured = await evalCache({
        teacher: teacherPath,
        dataset: datasetPath,
        cache: args.cache,
        rows,
        seq,
        topK,
        logitChunk,
      });
      console.log(`\n✓ Captured ${captured} rows`);
    } else {
      if (!args.model || !args.cache) {
        console.error('Error: score mode requires --model and --cache\n');
        console.error('Use --help for usage information');
        process.exit(1);
      }
      const modelPath = resolve(args.model);
      requireModelDir('--model', modelPath);
      const report = await evalScore({ model: modelPath, cache: args.cache, logitChunk });
      if (args.json) {
        console.log(JSON.stringify({ model: modelPath, ...report }));
        return;
      }
      console.log(formatReport(modelPath, report));
    }
    console.log(`✓ Duration: ${((Date.now() - startTime) / 1000).toFixed(2)}s`);
  } catch (error: any) {
    console.error(`\nEval ${mode} failed:`, error.message);
    process.exit(1);
  }
}
