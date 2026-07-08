import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { startActivationCalibration, finishActivationCalibration } from '@mlx-node/core';
import { loadSession } from '@mlx-node/lm';

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

/** Rough tokens→chars factor for bounding per-row prefill length. */
const CHARS_PER_TOKEN = 4;

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
 * model.
 *
 * Arms the process-global activation collector, prefills the model over each
 * calibration prompt through a normal chat session (one token is enough — the
 * prefill trips every mxfp8 attention/GDN projection's tap once), then drains
 * the per-tensor `max|activation|` into the model `config.json`. The session is
 * `reset()` between rows so each row is a fresh turn-0 prefill (the native KV
 * cache is wiped, avoiding unbounded context growth across rows).
 *
 * Edits `<input>/config.json` in place, writing `input_amax` onto the attn/GDN
 * quantization entries under BOTH the `quantization` and `quantization_config`
 * aliases.
 */
export async function calibrate(opts: CalibrateOptions): Promise<CalibrateResult> {
  const modelPath = resolve(opts.input);
  const calibSize = opts.calibSize ?? 1024;
  const calibSeq = opts.calibSeq ?? 512;
  const maxChars = calibSeq * CHARS_PER_TOKEN;

  const texts = readCalibTexts(resolve(opts.dataset), calibSize);
  if (texts.length === 0) {
    throw new Error(`No {"text": ...} rows found in dataset ${opts.dataset}`);
  }

  const session = await loadSession(modelPath);

  startActivationCalibration();
  try {
    for (let i = 0; i < texts.length; i++) {
      // Bound the prefill length: truncate the raw text to ~calibSeq tokens via
      // a chars-per-token proxy. Exact token truncation is unnecessary — amax is
      // a running max over activations and is robust to the precise length.
      const prompt = texts[i].slice(0, maxChars);
      // A single-token prefill trips every projection's tap during the prompt
      // forward; we don't need the generated token, just the forward pass.
      for await (const event of session.sendStream(prompt, {
        config: { maxNewTokens: 1, temperature: 0 },
      })) {
        if (event.done) break;
      }
      // Wipe the native KV cache + turn counter so the next row is a fresh
      // turn-0 prefill (otherwise the reused cache would accumulate context
      // across all rows and blow past the model's max length).
      await session.reset();
      opts.onProgress?.(i + 1, texts.length);
    }
  } catch (err) {
    // Best-effort drain so a failed calibration doesn't leave the process-global
    // collector armed. This also flushes whatever partial amax was collected;
    // the caller treats the throw as a hard failure (CLI exits nonzero), so the
    // partially-written config is surfaced as an error, not silently trusted.
    try {
      finishActivationCalibration(modelPath);
    } catch {
      // ignore — the original error is the one that matters
    }
    throw err;
  }

  const projectionsCalibrated = finishActivationCalibration(modelPath);
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
  Runs the model over the NVIDIA calibration mix with the activation collector
  armed, recording each attention/GDN mxfp8 projection's per-tensor
  max|activation| (modelopt MaxCalibrator semantics). The collected input_amax
  is written into the model's config.json IN PLACE (both the "quantization" and
  "quantization_config" blocks) so a later inference run fake-quantizes those
  activations to E4M3 for W8A8 numeric parity with NVIDIA modelopt. Only the
  mxfp8 attn/GDN sites are calibrated; the mxfp4 FFN keeps bf16 activations.

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
