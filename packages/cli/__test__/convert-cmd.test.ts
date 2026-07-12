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
  convertGgufToSafetensors: vi.fn(async () => ({
    numTensors: 0,
    numParameters: 0,
    sourceFormat: 'gguf',
    outputPath: '',
    tensorNames: [],
  })),
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

describe('mlx convert Unsloth MXFP messaging', () => {
  it('documents the official fixed map without the stale mechanical-upgrade recipe', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    await runConvert(['--help']);

    const help = logSpy.mock.calls.map((call) => String(call[0])).join('\n');
    expect(help).toContain('Qwen3.5 MXFP map: early FFNs=mxfp4');
    expect(help).toContain('Use --q-mode nvfp4 for the official DGX map');
    expect(help).toContain('Plain affine keeps legacy Dynamic 2.0');
    expect(help).not.toContain('Recommended combo: --q-recipe unsloth --q-bits 4 --q-mxfp');
  });

  it('reports the requested DGX map and forwards it without replacing its NVFP4 FFNs', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const imatrixPath = join(tmp, 'imatrix.gguf_file');
    writeFileSync(imatrixPath, '');
    writeFileSync(join(tmp, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }));

    await runConvert([
      '--input',
      tmp,
      '--output',
      join(tmp, 'out'),
      '--model-type',
      'qwen3_5',
      '--quantize',
      '--q-mode',
      'nvfp4',
      '--q-recipe',
      'unsloth',
      '--imatrix-path',
      imatrixPath,
    ]);

    const logs = logSpy.mock.calls.map((call) => String(call[0])).join('\n');
    expect(logs).toContain('requested official Unsloth DGX/NVFP4 map');
    expect(logs).toContain('backend verifies Qwen family/shape');
    expect(logs).toContain('early FFN=nvfp4');
    expect(logs).toContain('final 8 FFN + attention/GDN/head=mxfp8');
    expect(logs).not.toContain('early FFN=mxfp4');
    expect(logs).not.toContain('Quantize:   official Unsloth DGX/NVFP4 map');
    expect(vi.mocked(convertModel)).toHaveBeenCalledWith(
      expect.objectContaining({
        quantBits: 4,
        quantMode: 'nvfp4',
        quantMxfp: false,
        quantRecipe: 'unsloth',
        imatrixPath,
      }),
    );
  });

  it('reports the requested MXFP map and forwards the selector to SafeTensors conversion', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const imatrixPath = join(tmp, 'imatrix.gguf_file');
    writeFileSync(imatrixPath, '');
    writeFileSync(join(tmp, 'config.json'), JSON.stringify({ model_type: 'qwen3_5_moe' }));

    await runConvert([
      '--input',
      tmp,
      '--output',
      join(tmp, 'out'),
      '--model-type',
      'qwen3_5_moe',
      '--quantize',
      '--q-recipe',
      'unsloth',
      '--q-mxfp',
      '--imatrix-path',
      imatrixPath,
    ]);

    const logs = logSpy.mock.calls.map((call) => String(call[0])).join('\n');
    expect(logs).toContain('requested official Unsloth MXFP map');
    expect(logs).toContain('backend verifies Qwen family/shape');
    expect(logs).toContain('early FFN=mxfp4');
    expect(logs).toContain('final 8 FFN + attention/GDN/head=mxfp8');
    expect(logs).not.toContain('unsloth recipe defaults to 3-bit base');
    expect(logs).not.toContain('eligible 8b->mxfp8/4b->mxfp4');
    expect(logs).not.toContain('Quantize:   official Unsloth MXFP map');

    expect(vi.mocked(convertModel)).toHaveBeenCalledWith(
      expect.objectContaining({
        quantize: true,
        quantRecipe: 'unsloth',
        quantMxfp: true,
        imatrixPath,
      }),
    );
  });

  it('preserves the generic --q-mxfp upgrade messaging for other recipes', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});

    await runConvert([
      '--input',
      join(tmp, 'model'),
      '--output',
      join(tmp, 'out'),
      '--model-type',
      'qwen3_5',
      '--quantize',
      '--q-recipe',
      'qwen3_5',
      '--q-mxfp',
    ]);

    const logs = logSpy.mock.calls.map((call) => String(call[0])).join('\n');
    expect(logs).toContain('eligible 8b->mxfp8/4b->mxfp4');
    expect(logs).not.toContain('official Unsloth MXFP map');
    expect(vi.mocked(convertModel)).toHaveBeenCalledWith(
      expect.objectContaining({ quantRecipe: 'qwen3_5', quantMxfp: true }),
    );
  });

  it.each([
    ['a non-Qwen config', { input: 'tmp', configModelType: 'lfm2', modelArgs: [] }],
    [
      'a non-Qwen config with a mismatched explicit Qwen override',
      { input: 'tmp', configModelType: 'lfm2', modelArgs: ['--model-type', 'qwen3_5'] },
    ],
    ['an ambiguous input without a detected family', { input: 'missing', configModelType: undefined, modelArgs: [] }],
    [
      'an unverified explicit Qwen override',
      { input: 'missing', configModelType: undefined, modelArgs: ['--model-type', 'qwen3_5'] },
    ],
  ])('does not claim the official map was applied for %s', async (_label, scenario) => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const imatrixPath = join(tmp, 'imatrix.gguf');
    writeFileSync(imatrixPath, '');
    if (scenario.configModelType) {
      writeFileSync(join(tmp, 'config.json'), JSON.stringify({ model_type: scenario.configModelType }));
    }

    await runConvert([
      '--input',
      scenario.input === 'tmp' ? tmp : join(tmp, 'model'),
      '--output',
      join(tmp, 'out'),
      ...scenario.modelArgs,
      '--quantize',
      '--q-recipe',
      'unsloth',
      '--q-mxfp',
      '--imatrix-path',
      imatrixPath,
    ]);

    const logs = logSpy.mock.calls.map((call) => String(call[0])).join('\n');
    expect(logs).toContain('requested official Unsloth MXFP map');
    expect(logs).toContain('backend verifies Qwen family/shape');
    expect(logs).not.toContain('Quantize:   official Unsloth MXFP map');
  });

  it('marks a GGUF official-map selection as requested until the backend verifies its shape', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const imatrixPath = join(tmp, 'imatrix.gguf');
    writeFileSync(imatrixPath, '');

    await runConvert([
      '--input',
      ggufPath,
      '--output',
      join(tmp, 'out'),
      '--quantize',
      '--q-recipe',
      'unsloth',
      '--q-mxfp',
      '--imatrix-path',
      imatrixPath,
    ]);

    const logs = logSpy.mock.calls.map((call) => String(call[0])).join('\n');
    expect(logs).toContain('requested official Unsloth MXFP map');
    expect(logs).toContain('backend verifies Qwen family/shape');
    expect(logs).not.toContain('Quantize:   official Unsloth MXFP map');
    expect(vi.mocked(convertGgufToSafetensors)).toHaveBeenCalledWith(
      expect.objectContaining({ quantRecipe: 'unsloth', quantMxfp: true, imatrixPath }),
    );
  });
});
