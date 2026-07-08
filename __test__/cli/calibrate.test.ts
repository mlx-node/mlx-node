import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { convertModel } from '@mlx-node/core';
import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import { calibrate } from '../../packages/cli/src/commands/calibrate.js';

/**
 * End-to-end smoke for `mlx calibrate`: drive the 0.8B nvidia-recipe checkpoint
 * over a small subset of the NVIDIA calibration mix and assert the collected
 * per-tensor FP8 activation `input_amax` lands in the model `config.json`.
 *
 * The calibration tap fires on the mxfp8 attention/GDN projections only, so the
 * post-calibration config must gain `input_amax` on:
 *   - every `*.self_attn.{q,k,v,o}_proj`
 *   - BOTH split GDN input projections `*.linear_attn.in_proj_qkv` /
 *     `*.linear_attn.in_proj_z` (the merged `in_proj_qkvz` amax fans out to both)
 *   - `*.linear_attn.out_proj`
 * and must NOT gain it on the mxfp4 FFN (`*.mlp.gate_proj`). The write mirrors
 * into BOTH the `quantization` and `quantization_config` aliases.
 *
 * Presence-gated + heavy: runs only when the base 0.8B checkpoint and the
 * calibration JSONL are on disk, so CI without the weights auto-skips. It
 * converts a FRESH nvidia model into a temp dir each run so the assertion proves
 * THIS calibration wrote the amax (not a leftover from a prior run).
 *
 * To run locally:
 *   vp test __test__/cli/calibrate.test.ts
 *   (or set QWEN35_08B_MODEL_PATH / NVIDIA_CALIB_JSONL to override the paths).
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

function findDataset(): string | null {
  const env = process.env.NVIDIA_CALIB_JSONL;
  const candidates = [env, join(homedir(), '.cache', 'nvidia-calib', 'cnn_nemotron_v2_calib.jsonl')].filter(
    Boolean,
  ) as string[];
  for (const f of candidates) {
    if (existsSync(f)) return f;
  }
  return null;
}

const baseModel = findBaseModel();
const dataset = findDataset();
const canRun = baseModel !== null && dataset !== null;

/** Collect all quantization-block keys (per-layer overrides + top-level scalars). */
function keysEndingWith(block: Record<string, unknown>, suffix: string): string[] {
  return Object.keys(block).filter((k) => k.endsWith(suffix));
}

function hasInputAmax(block: Record<string, unknown>, key: string): boolean {
  const entry = block[key] as Record<string, unknown> | undefined;
  return entry !== undefined && typeof entry === 'object' && 'input_amax' in entry;
}

describe.skipIf(!canRun)('mlx calibrate (0.8B nvidia)', () => {
  let scratch: string;
  let configPath: string;

  beforeAll(async () => {
    scratch = mkdtempSync(join(tmpdir(), 'mlx-calibrate-'));
    const modelDir = join(scratch, 'nvidia-08b-cal');
    await convertModel({
      inputDir: baseModel!,
      outputDir: modelDir,
      modelType: 'qwen3_5',
      quantize: true,
      quantRecipe: 'nvidia',
      dtype: 'bfloat16',
    });
    configPath = join(modelDir, 'config.json');

    // Sanity: a freshly-converted nvidia model must NOT yet carry input_amax.
    const before = JSON.parse(readFileSync(configPath, 'utf-8'));
    const qBefore = before.quantization as Record<string, unknown>;
    for (const k of keysEndingWith(qBefore, '.self_attn.q_proj')) {
      expect(hasInputAmax(qBefore, k)).toBe(false);
    }

    const { projectionsCalibrated } = await calibrate({
      input: modelDir,
      dataset: dataset!,
      calibSize: 16,
      calibSeq: 128,
    });
    // Every attn/GDN projection across all layers gets one entry.
    expect(projectionsCalibrated).toBeGreaterThan(0);
  }, 600_000);

  afterAll(() => {
    if (scratch) rmSync(scratch, { recursive: true, force: true });
  });

  it('writes input_amax onto attn/GDN keys in BOTH aliases', () => {
    const config = JSON.parse(readFileSync(configPath, 'utf-8'));
    for (const alias of ['quantization', 'quantization_config']) {
      const block = config[alias] as Record<string, unknown>;
      expect(block, `${alias} block must exist`).toBeTruthy();

      // self_attn.q_proj is an activation-fp8 site → must gain input_amax.
      const qKeys = keysEndingWith(block, '.self_attn.q_proj');
      expect(qKeys.length, `${alias}: at least one q_proj entry`).toBeGreaterThan(0);
      for (const k of qKeys) {
        expect(hasInputAmax(block, k), `${alias}: ${k} must have input_amax`).toBe(true);
        const amax = (block[k] as Record<string, number>).input_amax;
        expect(amax, `${alias}: ${k} input_amax finite positive`).toBeGreaterThan(0);
      }
    }
  });

  it('fans the merged GDN input-projection amax out to BOTH split entries', () => {
    const config = JSON.parse(readFileSync(configPath, 'utf-8'));
    const block = config.quantization as Record<string, unknown>;

    const qkvKeys = keysEndingWith(block, '.linear_attn.in_proj_qkv');
    const zKeys = keysEndingWith(block, '.linear_attn.in_proj_z');
    expect(qkvKeys.length, 'in_proj_qkv entries present').toBeGreaterThan(0);
    expect(zKeys.length, 'in_proj_z entries present').toBeGreaterThan(0);
    for (const k of qkvKeys) {
      expect(hasInputAmax(block, k), `${k} must have input_amax`).toBe(true);
    }
    for (const k of zKeys) {
      expect(hasInputAmax(block, k), `${k} must have input_amax`).toBe(true);
    }
  });

  it('does NOT write input_amax onto the mxfp4 FFN', () => {
    const config = JSON.parse(readFileSync(configPath, 'utf-8'));
    const block = config.quantization as Record<string, unknown>;
    const gateKeys = keysEndingWith(block, '.mlp.gate_proj');
    expect(gateKeys.length, 'gate_proj entries present').toBeGreaterThan(0);
    for (const k of gateKeys) {
      expect(hasInputAmax(block, k), `${k} must NOT have input_amax`).toBe(false);
    }
  });
});
