import { join } from 'node:path';

import { describe, expect, it, vi } from 'vite-plus/test';

import { runFirstRunWizard } from '../src/commands/agent/wizard.js';

const GIB = 2 ** 30;
const MODELS_DIR = '/tmp/test-models';

describe('runFirstRunWizard', () => {
  it('downloads both the chosen Inkling-2-Mini build and its autoInstall DFlash2 companion', async () => {
    const downloads: string[][] = [];
    const logs: string[] = [];
    const select = vi
      .fn<
        ({ message, choices }: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>
      >()
      .mockResolvedValueOnce('Inkling-2-Mini')
      .mockResolvedValueOnce('UD-Q4_K_M');

    const slug = await runFirstRunWizard({
      io: { select, isTTY: true, log: (line) => logs.push(line) },
      download: async (argv) => {
        downloads.push(argv);
      },
      modelsDir: MODELS_DIR,
      totalMemBytes: 64 * GIB,
    });

    expect(slug).toBe('inkling-2-mini-gguf-ud-q4_k_m');
    expect(downloads).toEqual([
      [
        '-m',
        'unsloth/Inkling-2-Mini-GGUF',
        '-g',
        '*UD-Q4_K_M.gguf',
        '-g',
        'mmproj-BF16.gguf',
        '-g',
        'dspark-Inkling-2-Mini-Q8_0.gguf',
        '--assets-repo',
        'thinkingmachines/Inkling-2-Mini',
        '--complete',
        '-o',
        join(MODELS_DIR, 'inkling-2-mini-gguf-ud-q4_k_m'),
      ],
      ['-m', 'incoai/Inkling-2-Mini-DFlash2', '--complete', '-o', join(MODELS_DIR, 'inkling-2-mini-dflash2')],
    ]);
    expect(logs.some((line) => line.includes('Inkling-2-Mini DFlash2'))).toBe(true);
  });

  it('keeps Qwen3.8-27B DFlash2 optional and downloads only the chosen build', async () => {
    const downloads: string[][] = [];
    const select = vi
      .fn<
        ({ message, choices }: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>
      >()
      .mockResolvedValueOnce('Qwen3.8-27B')
      .mockResolvedValueOnce('UD-Q4_K_XL');

    const slug = await runFirstRunWizard({
      io: { select, isTTY: true, log: () => {} },
      download: async (argv) => {
        downloads.push(argv);
      },
      modelsDir: MODELS_DIR,
      totalMemBytes: 64 * GIB,
    });

    expect(slug).toBe('qwen3.8-27b-gguf');
    expect(downloads).toHaveLength(1);
    expect(downloads[0]).toContain('unsloth/Qwen3.8-27B-GGUF');
  });

  it('continues with Inkling-2-Mini and logs a retry hint if the companion download fails', async () => {
    const logs: string[] = [];
    const select = vi
      .fn<
        ({ message, choices }: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>
      >()
      .mockResolvedValueOnce('Inkling-2-Mini')
      .mockResolvedValueOnce('Original');
    const download = vi.fn().mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error('network reset'));

    const slug = await runFirstRunWizard({
      io: { select, isTTY: true, log: (line) => logs.push(line) },
      download,
      modelsDir: MODELS_DIR,
      totalMemBytes: 64 * GIB,
    });

    expect(slug).toBe('inkling-2-mini');
    expect(download).toHaveBeenCalledTimes(2);
    expect(logs.some((line) => line.includes('Warning:') && line.includes('incoai/Inkling-2-Mini-DFlash2'))).toBe(true);
  });

  it('chains the autoInstall companion in the non-TTY hint for Inkling-2-Mini only', async () => {
    const err = await runFirstRunWizard({
      io: { select: vi.fn(), isTTY: false, log: () => {} },
      download: vi.fn(),
      modelsDir: MODELS_DIR,
      totalMemBytes: 64 * GIB,
    }).catch((e: unknown) => e as Error);

    expect(err).toBeInstanceOf(Error);
    if (!(err instanceof Error)) return;
    expect(err.message).toContain(
      `&& mlx download model -m incoai/Inkling-2-Mini-DFlash2 --complete -o ${join(MODELS_DIR, 'inkling-2-mini-dflash2')}`,
    );
    expect(err.message).not.toContain('z-lab/Qwen3.8-27B-DFlash2');
  });
});
