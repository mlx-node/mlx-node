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
  convertModel: vi.fn(async () => ({})),
  convertForeignWeights: vi.fn(() => ({})),
  convertGgufToSafetensors: vi.fn(async () => ({})),
}));

import { convertGgufToSafetensors } from '@mlx-node/core';

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
