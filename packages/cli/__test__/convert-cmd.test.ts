/**
 * Validation tests for `mlx convert` argument checks on GGUF input.
 *
 * The regression this guards against: `--q-mode sym8` with a .gguf input used
 * to pass CLI validation and only fail later inside the native GGUF backend
 * ("Invalid quant_mode 'sym8': must be 'affine', 'mxfp8', 'mxfp4', or
 * 'nvfp4'"). The CLI must reject sym8 for GGUF upfront, before any tensor
 * loading.
 *
 * `@mlx-node/core` is mocked so the native addon is never loaded and no
 * conversion runs; `process.exit` is mocked to throw so the test proves
 * validation halts before reaching the (mocked) native call.
 */
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, it, expect, vi, beforeEach, afterEach } from 'vite-plus/test';

vi.mock('@mlx-node/core', () => ({
  convertModel: vi.fn(async () => ({ numTensors: 0, numParameters: 0, outputPath: '', tensorNames: [] })),
  convertForeignWeights: vi.fn(() => ({})),
  convertGgufToSafetensors: vi.fn(async () => ({})),
}));

import { convertModel, convertGgufToSafetensors } from '@mlx-node/core';

import { run as runConvert } from '../src/commands/convert.js';

let tmp: string;
let ggufPath: string;

beforeEach(() => {
  vi.clearAllMocks();
  tmp = mkdtempSync(join(tmpdir(), 'mlx-convert-cmd-'));
  ggufPath = join(tmp, 'model.gguf');
  writeFileSync(ggufPath, '');
});

afterEach(() => {
  vi.restoreAllMocks();
  rmSync(tmp, { recursive: true, force: true });
});

describe('mlx convert GGUF validation', () => {
  it('rejects --q-mode sym8 for .gguf input upfront instead of failing in the native backend', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`process.exit(${code})`);
    }) as never);

    await expect(
      runConvert(['--input', ggufPath, '--output', join(tmp, 'out'), '--quantize', '--q-mode', 'sym8']),
    ).rejects.toThrow('process.exit(1)');

    const errors = errSpy.mock.calls.map((c) => String(c[0])).join('\n');
    expect(errors).toContain('--q-mode sym8 is not supported for GGUF input');
    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(convertGgufToSafetensors).not.toHaveBeenCalled();
  });
});

describe('mlx convert model-type auto-detection', () => {
  // Drive run() against a synthetic config.json (no --model-type) and read back
  // the modelType handed to the mocked native convertModel. This guards the
  // gemma4_unified pass-through: collapsing it to 'gemma4' would dead-code the
  // native recipe_for("gemma4_unified") arm and misroute gemma-QAT unified
  // checkpoints into the E2B-only prequantized importer.
  const detectModelTypeFromConfig = async (config: Record<string, unknown>): Promise<unknown> => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`process.exit(${code})`);
    }) as never);

    const inputDir = mkdtempSync(join(tmpdir(), 'mlx-convert-detect-in-'));
    writeFileSync(join(inputDir, 'config.json'), JSON.stringify(config));
    const outputDir = join(tmp, 'out');

    await runConvert(['--input', inputDir, '--output', outputDir]);

    const mock = vi.mocked(convertModel);
    expect(mock).toHaveBeenCalledTimes(1);
    const opts = mock.mock.calls[0]![0] as { modelType?: unknown };
    rmSync(inputDir, { recursive: true, force: true });
    return opts.modelType;
  };

  const detectModelType = async (configModelType: string): Promise<unknown> =>
    detectModelTypeFromConfig({ model_type: configModelType });

  it("passes 'gemma4_unified' through unchanged (does NOT collapse to 'gemma4')", async () => {
    expect(await detectModelType('gemma4_unified')).toBe('gemma4_unified');
  });

  it("collapses 'gemma4' to 'gemma4'", async () => {
    expect(await detectModelType('gemma4')).toBe('gemma4');
  });

  it("collapses 'gemma4_text' to 'gemma4'", async () => {
    expect(await detectModelType('gemma4_text')).toBe('gemma4');
  });

  it("detects an architecture-only unified config (no model_type) as 'gemma4_unified'", async () => {
    // Mirrors the runtime loader: a config with no `model_type` but with
    // `architectures: ['Gemma4UnifiedForConditionalGeneration']` must resolve
    // to 'gemma4_unified' so Gemma4Recipe::sanitize runs. Without the
    // converter arm, modelType would stay undefined and the output would be
    // unloadable.
    expect(await detectModelTypeFromConfig({ architectures: ['Gemma4UnifiedForConditionalGeneration'] })).toBe(
      'gemma4_unified',
    );
  });
});
