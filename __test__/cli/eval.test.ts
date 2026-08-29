import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { convertModel } from '@mlx-node/core';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { evalCache, evalScore } from '../../packages/cli/src/commands/eval.js';

/**
 * End-to-end for `mlx eval`: capture a bf16 teacher, then score checkpoints
 * against it.
 *
 * Two properties, and the second is what makes the first mean anything:
 *
 *   1. IDENTITY — scoring the teacher against its OWN capture must report zero
 *      divergence and perfect agreement. If that fails the metric is wrong and
 *      every later number it produces is worthless.
 *   2. DISCRIMINATION — a 4-bit quantization of the same checkpoint, on the same
 *      token ids, must come out measurably worse. A metric that cannot separate
 *      those two is not a metric.
 *
 * Presence-gated on the 0.8B so CI without weights auto-skips.
 *
 * To run locally:
 *   vp test __test__/cli/eval.test.ts
 *   (or set QWEN35_08B_MODEL_PATH to override the model path).
 */

function findBaseModel(): string | null {
  const env = process.env.QWEN35_08B_MODEL_PATH;
  const candidates = [
    env,
    join(homedir(), '.mlx-node', 'models', 'qwen3.5-0.8b'),
    resolve(process.cwd(), '.cache/models/qwen3.5-0.8b'),
  ].filter(Boolean) as string[];
  for (const dir of candidates) {
    if (existsSync(join(dir, 'config.json'))) return dir;
  }
  return null;
}

const baseModel = findBaseModel();

/**
 * Held-out eval text, written inline so the run is hermetic and can never be
 * pointed at a calibration or imatrix set by accident — scoring a calibrated
 * checkpoint on its own calibration data is train-on-test.
 */
const EVAL_ROWS = [
  'The harbour master kept a ledger of every vessel that entered the bay, noting the tide, the wind, and the temper of the crew. In winter the entries grew shorter, because the ink froze in the well and the lamp burned through its oil before the last ship was counted.',
  'A compiler that optimises for the common case must still be correct in the rare one. The rare case is where the specification lives, and a program that is fast and wrong is slower than one that is slow and right, because the wrong answer has to be computed twice.',
  'She learned the river the way other people learn a language: first the nouns, the shoals and the bends and the drowned oak at the third crossing, and only much later the grammar, which was the way the water answered a week of rain.',
  'The argument for keeping the old bridge was never about the bridge. It was about the road on either side of it, and the fields the road divided, and the fact that no one had ever agreed where the parish ended and the moor began.',
];

describe.skipIf(baseModel === null)('mlx eval (0.8B teacher-forced quality)', () => {
  let scratch: string;
  let cacheDir: string;
  let datasetPath: string;
  let quantModel: string;

  beforeAll(async () => {
    scratch = mkdtempSync(join(tmpdir(), 'mlx-eval-'));
    cacheDir = join(scratch, 'teacher-cache');
    datasetPath = join(scratch, 'eval.jsonl');
    writeFileSync(datasetPath, EVAL_ROWS.map((text) => JSON.stringify({ text })).join('\n'));

    const rows = await evalCache({
      teacher: baseModel!,
      dataset: datasetPath,
      cache: cacheDir,
      rows: EVAL_ROWS.length,
      seq: 128,
      topK: 512,
      logitChunk: 64,
    });
    expect(rows).toBe(EVAL_ROWS.length);

    quantModel = join(scratch, 'qwen35-08b-q4');
    await convertModel({
      inputDir: baseModel!,
      outputDir: quantModel,
      modelType: 'qwen3_5',
      quantize: true,
      quantBits: 4,
      quantGroupSize: 64,
      dtype: 'bfloat16',
    });
  }, 1_800_000);

  afterAll(() => {
    if (scratch) rmSync(scratch, { recursive: true, force: true });
  });

  it('reports zero divergence when a checkpoint is scored against itself', async () => {
    const report = await evalScore({ model: baseModel!, cache: cacheDir, logitChunk: 64 });

    expect(report.rows).toBe(EVAL_ROWS.length);
    expect(report.positions).toBeGreaterThan(0);
    expect(report.teacherPath).toBe(resolve(baseModel!));
    // A tolerance, not `=== 0`, only because this is a SECOND forward of the
    // same bf16 weights. The exact-zero assertion lives in the pure-numeric
    // unit test, where there is no model to be nondeterministic.
    expect(report.meanKlTopk).toBeLessThan(1e-6);
    expect(report.top1Agreement).toBe(1);
    expect(Math.abs(report.meanNll - report.teacherMeanNll)).toBeLessThan(1e-5);
    // The cached support must carry nearly all the mass, otherwise the KL above
    // is measuring almost nothing.
    expect(report.teacherTailMass).toBeLessThan(0.1);
    // Self-identity alone cannot catch a forward that is self-consistently
    // WRONG — an off-by-one between inputs and targets, or a double-applied
    // final_norm, still reports zero divergence against its own capture. Both
    // land near ln(vocab) ~ 12.4 on this vocabulary; real next-token prediction
    // on English prose does not.
    expect(report.meanNll).toBeGreaterThan(0.5);
    expect(report.meanNll).toBeLessThan(6);
  }, 900_000);

  it('scores a 4-bit quantization of the same checkpoint as worse', async () => {
    const teacher = await evalScore({ model: baseModel!, cache: cacheDir, logitChunk: 64 });
    const quantized = await evalScore({ model: quantModel, cache: cacheDir, logitChunk: 64 });

    expect(quantized.positions).toBe(teacher.positions);
    expect(quantized.meanKlTopk).toBeGreaterThan(teacher.meanKlTopk);
    expect(quantized.meanNll).toBeGreaterThan(teacher.meanNll);
    expect(quantized.perplexity).toBeGreaterThan(teacher.perplexity);
    expect(quantized.top1Agreement).toBeLessThan(1);
    // The teacher's own NLL is read from the cache, so it must be identical no
    // matter which candidate is in front of it.
    expect(quantized.teacherMeanNll).toBe(teacher.teacherMeanNll);
  }, 900_000);

  it('refuses a cache captured on a different vocabulary', async () => {
    // The one staleness the cached token ids cannot catch: a candidate whose
    // head is a different size answers a different question entirely.
    const forged = join(scratch, 'forged-cache');
    cpSync(cacheDir, forged, { recursive: true });
    const meta = JSON.parse(readFileSync(join(forged, 'meta.json'), 'utf-8'));
    meta.vocab_size += 1;
    writeFileSync(join(forged, 'meta.json'), JSON.stringify(meta));

    await expect(evalScore({ model: baseModel!, cache: forged, logitChunk: 64 })).rejects.toThrow(/vocabulary/);
  }, 900_000);
});
