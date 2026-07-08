import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { convertModel } from '@mlx-node/core';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { calibrate } from '../../packages/cli/src/commands/calibrate.js';

/**
 * Whole-branch codex [medium] regression: the CLI must NOT pre-truncate a row
 * with ANY fixed UTF-16-char cap before handing it to the native
 * `calibrateActivationAmaxRaw`. Every char cap is tokenizer-blind — e.g.
 * `' '.repeat(70000)` encodes to ~548 tokens under the local Qwen tokenizer
 * (runs of spaces merge to ~128 chars/token), so even a generous 16×
 * chars/token guard would slice it to a SHORTER prefix than modelopt's token
 * window and bias `input_amax` low while still reporting success. The native
 * side is the SOLE authoritative boundary — it tokenizes each row
 * (add_special_tokens=false) and truncates to EXACTLY `calibSeq` TOKENS.
 *
 * We `vi.mock('@mlx-node/core')` and delegate EVERY export to the real binding
 * via `vi.importActual` (so the presence-gated e2e block below keeps using the
 * real `convertModel` + real native calibrate), overriding ONLY
 * `calibrateActivationAmaxRaw` — and only when a test installs `nativeAmaxSpy`.
 */
let nativeAmaxSpy: ((modelPath: string, texts: string[], calibSeq: number) => Promise<number>) | null = null;

vi.mock('@mlx-node/core', async () => {
  const actual = (await vi.importActual<Record<string, unknown>>('@mlx-node/core')) as Record<string, unknown>;
  return new Proxy(actual, {
    get(target, prop, receiver): unknown {
      if (prop === 'calibrateActivationAmaxRaw' && nativeAmaxSpy) {
        return nativeAmaxSpy;
      }
      return Reflect.get(target, prop, receiver);
    },
  });
});

/**
 * End-to-end smoke for `mlx calibrate`: drive the 0.8B nvidia-recipe checkpoint
 * over a small subset of the NVIDIA calibration mix and assert the collected
 * per-tensor FP8 activation `input_amax` lands in the model `config.json`.
 *
 * Exercises the native RAW-text PREFILL path (`calibrateActivationAmaxRaw` via
 * `calibrate()`): no chat template, no generated token — each row is tokenized
 * raw and prefilled once with the collector armed, then the amax is written
 * atomically into config.json.
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

/**
 * Weightless unit coverage for the char-slice fix (codex [medium]): drives the
 * CLI `calibrate()` with a spied native `calibrateActivationAmaxRaw` and asserts
 * the CLI hands native the FULL row text UNSLICED regardless of length — NO
 * fixed UTF-16-char cap of any kind. Native token-truncation is the sole
 * boundary. Runs without any model weights.
 */
describe('mlx calibrate — native token-truncation is the only boundary (codex [medium])', () => {
  let scratch: string;

  beforeEach(() => {
    scratch = mkdtempSync(join(tmpdir(), 'mlx-calib-charslice-'));
  });

  afterEach(() => {
    nativeAmaxSpy = null;
    if (scratch) rmSync(scratch, { recursive: true, force: true });
  });

  it('hands the FULL row to native unsliced regardless of length (no fixed char cap)', async () => {
    const calibSeq = 128;
    // Concrete tokenizer-blind failure: `' '.repeat(70000)` encodes to ~548
    // tokens under the local Qwen tokenizer (runs of spaces merge to ~128
    // chars/token), so it carries WELL over `calibSeq` tokens and native must
    // see all 70000 chars to truncate at the right token boundary. The removed
    // f217371f "16× memory guard" (calibSeq*4*16 = 8192) would slice this to
    // 8192 chars = only ~256 tokens, recreating the original under-feed bug.
    const OLD_16X_GUARD = calibSeq * 4 * 16; // 8192 — removed pre-fix slice length
    const rowLen = 70000;
    expect(rowLen, 'row must exceed the removed 16× guard').toBeGreaterThan(OLD_16X_GUARD);
    const rowText = ' '.repeat(rowLen);

    const datasetPath = join(scratch, 'calib.jsonl');
    writeFileSync(datasetPath, `${JSON.stringify({ text: rowText })}\n`, 'utf-8');

    // Capture the exact `texts` the CLI hands native. The input dir is never
    // read by `calibrate()` (native is mocked), so any path string works.
    let capturedTexts: string[] | null = null;
    nativeAmaxSpy = async (_modelPath, texts, _seq) => {
      capturedTexts = texts;
      return 1;
    };

    // Fresh module graph so the freshly-imported CLI resolves
    // `calibrateActivationAmaxRaw` through the spy-aware mock proxy.
    vi.resetModules();
    const { calibrate: calibrateFresh } = await import('../../packages/cli/src/commands/calibrate.js');
    await calibrateFresh({
      input: join(scratch, 'model'),
      dataset: datasetPath,
      calibSize: 1,
      calibSeq,
    });

    expect(capturedTexts, 'native was invoked').not.toBeNull();
    const texts = capturedTexts as unknown as string[];
    expect(texts.length, 'exactly one row').toBe(1);
    // The crux: native received the FULL row, not any fixed-char slice. Against
    // pre-fix code (`t.slice(0, calibSeq * 4 * 16)`) this is 8192 vs 70000 →
    // RED. Native (not the CLI) is the sole token-truncation boundary.
    expect(texts[0].length, 'full row length reaches native').toBe(rowLen);
    expect(texts[0], 'row content is untruncated').toBe(rowText);
  });
});
