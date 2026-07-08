import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { calibrateActivationAmaxRaw } from '@mlx-node/core';

/** Options for {@link calibrate}. */
export interface CalibrateOptions {
  /** Model directory (an `--q-recipe nvidia` checkpoint) to calibrate in place. */
  input: string;
  /** JSONL calibration dataset; each row is `{"text": "..."}`. */
  dataset: string;
  /** Number of dataset rows to run (default 1024, matching modelopt hf_ptq). */
  calibSize?: number;
  /** Approximate prefill length per row in tokens (default 512). */
  calibSeq?: number;
  /** Progress callback invoked after each row (1-based done, total). */
  onProgress?: (done: number, total: number) => void;
}

/** Result of a calibration run. */
export interface CalibrateResult {
  /** Number of projections that gained an `input_amax` (collected amax count). */
  projectionsCalibrated: number;
  /** Absolute path of the `config.json` that was edited in place. */
  configPath: string;
}

/**
 * Read the first `count` `{"text": ...}` rows from a JSONL calibration file.
 * Blank lines and rows without a string `text` field are skipped.
 */
function readCalibTexts(datasetPath: string, count: number): string[] {
  const raw = readFileSync(datasetPath, 'utf-8');
  const texts: string[] = [];
  for (const line of raw.split('\n')) {
    if (texts.length >= count) break;
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    let row: unknown;
    try {
      row = JSON.parse(trimmed);
    } catch {
      continue;
    }
    const text = (row as { text?: unknown }).text;
    if (typeof text === 'string' && text.length > 0) {
      texts.push(text);
    }
  }
  return texts;
}

/**
 * Drive NVIDIA modelopt-style static FP8 activation-amax calibration over a
 * model, using RAW-text PREFILL (no chat template, no generated token).
 *
 * Thin wrapper over the native {@link calibrateActivationAmaxRaw}: read the
 * first `calibSize` `{"text": ...}` rows, then hand them to native code, which
 * loads the model + tokenizer, arms the process-global activation collector,
 * and for each row tokenizes the RAW text (no `<|im_start|>`/`<|im_end|>`
 * control tokens), truncates to `calibSeq` tokens, and runs a PREFILL-ONLY
 * forward so every mxfp8 attention/GDN projection's tap fires once over
 * realistic raw-text activations (modelopt `MaxCalibrator` is defined over
 * raw-text prefill, NOT chat-templated prompts + a decode step). Caches are
 * reset between rows.
 *
 * On success it ATOMICALLY writes `input_amax` into `<input>/config.json`
 * (temp file + rename) under BOTH the `quantization` and `quantization_config`
 * aliases. On ANY error before the final write, `config.json` is left
 * UNTOUCHED — a failed calibration never mutates the live model in place.
 */
export async function calibrate(opts: CalibrateOptions): Promise<CalibrateResult> {
  const modelPath = resolve(opts.input);
  const calibSize = opts.calibSize ?? 1024;
  const calibSeq = opts.calibSeq ?? 512;

  const rows = readCalibTexts(resolve(opts.dataset), calibSize);
  if (rows.length === 0) {
    throw new Error(`No {"text": ...} rows found in dataset ${opts.dataset}`);
  }
  // Hand the FULL row text to native untouched. `calibrateActivationAmaxRaw` is
  // the SOLE, tokenizer-correct calibration boundary: it tokenizes each row
  // (add_special_tokens=false) and truncates to EXACTLY `calibSeq` tokens. Any
  // JS-side char cap is tokenizer-blind — space runs and byte-level merges pack
  // far more than a fixed chars/token estimate, so a fixed cap could slice a row
  // BELOW `calibSeq` tokens and bias `input_amax` low — so we do NOT pre-truncate.
  //
  // One native call: load + arm + raw-prefill-only over every row + drain +
  // atomic config write, returning the number of projections calibrated.
  const projectionsCalibrated = await calibrateActivationAmaxRaw(modelPath, rows, calibSeq);
  // Native runs as a single blocking pass (no per-row callback), so report
  // completion once.
  opts.onProgress?.(rows.length, rows.length);
  return { projectionsCalibrated, configPath: join(modelPath, 'config.json') };
}

function printHelp() {
  console.log(`
Calibrate FP8 Activation amax (NVIDIA modelopt parity)

Usage:
  mlx calibrate --input <model> --dataset <jsonl> [options]

Required Arguments:
  --input, -i <path>    Model directory to calibrate (an --q-recipe nvidia model)
  --dataset <jsonl>     Calibration dataset JSONL ({"text": "..."} rows)
                        Default mix: ~/.cache/nvidia-calib/cnn_nemotron_v2_calib.jsonl

Optional Arguments:
  --calib-size <int>    Number of dataset rows to run (default: 1024)
  --calib-seq <int>     Approx prefill length per row in tokens (default: 512)
  --help, -h            Show this help message

What it does:
  Runs the model over the NVIDIA calibration mix as RAW-text PREFILL (no chat
  template, no generated token) with the activation collector armed, recording
  each attention/GDN mxfp8 projection's per-tensor max|activation| (modelopt
  MaxCalibrator semantics). The collected input_amax is written ATOMICALLY into
  the model's config.json IN PLACE (both the "quantization" and
  "quantization_config" blocks) so a later inference run fake-quantizes those
  activations to E4M3 for W8A8 numeric parity with NVIDIA modelopt. Only the
  mxfp8 attn/GDN sites are calibrated; the mxfp4 FFN keeps bf16 activations.
  A failed run leaves config.json untouched.

Example:
  mlx calibrate -i ./qwen3.6-27b-nvidia-mxfp4-mlx \\
    --dataset ~/.cache/nvidia-calib/cnn_nemotron_v2_calib.jsonl --calib-size 1024
`);
}

export async function run(argv: string[]) {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      input: { type: 'string', short: 'i' },
      dataset: { type: 'string' },
      'calib-size': { type: 'string' },
      'calib-seq': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  if (!args.input || !args.dataset) {
    console.error('Error: Both --input and --dataset are required\n');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  const parsePositiveInt = (flag: string, raw?: string): number | undefined => {
    if (raw === undefined) return undefined;
    if (!/^[1-9]\d*$/.test(raw)) {
      console.error(`Error: ${flag} requires a positive integer value`);
      process.exit(1);
    }
    return Number(raw);
  };

  const inputPath = resolve(args.input);
  const datasetPath = resolve(args.dataset);
  const calibSize = parsePositiveInt('--calib-size', args['calib-size']) ?? 1024;
  const calibSeq = parsePositiveInt('--calib-seq', args['calib-seq']) ?? 512;

  if (!existsSync(join(inputPath, 'config.json'))) {
    console.error(`Error: model config not found: ${join(inputPath, 'config.json')}`);
    process.exit(1);
  }
  if (!existsSync(datasetPath)) {
    console.error(`Error: dataset not found: ${datasetPath}`);
    process.exit(1);
  }

  console.log(`Input:      ${inputPath}`);
  console.log(`Dataset:    ${datasetPath}`);
  console.log(`Calib size: ${calibSize} rows`);
  console.log(`Calib seq:  ~${calibSeq} tokens/row`);
  console.log('');
  console.log('Running raw-text prefill calibration (this may take a while)...');

  const startTime = Date.now();
  let lastLogged = 0;
  try {
    const { projectionsCalibrated, configPath } = await calibrate({
      input: inputPath,
      dataset: datasetPath,
      calibSize,
      calibSeq,
      onProgress: (done, total) => {
        // Log roughly every 5% (and always on the last row).
        const step = Math.max(1, Math.floor(total / 20));
        if (done === total || done - lastLogged >= step) {
          lastLogged = done;
          console.log(`  calibrated ${done}/${total} rows`);
        }
      },
    });

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    if (projectionsCalibrated === 0) {
      console.warn(
        '\nWarning: 0 projections calibrated — the model exercised no activation-fp8 (mxfp8 attn/GDN) sites.',
      );
      console.warn('         Is this an --q-recipe nvidia checkpoint? config.json was left unchanged.');
    } else {
      console.log(`\n✓ Calibrated ${projectionsCalibrated} projections`);
      console.log(`✓ Wrote input_amax into: ${configPath}`);
    }
    console.log(`✓ Duration: ${duration}s`);
  } catch (error: any) {
    console.error('\nCalibration failed:', error.message);
    process.exit(1);
  }
}
