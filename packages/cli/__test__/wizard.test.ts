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

  it('downloads both the chosen Qwen3.8-27B build and its autoInstall DFlash2 companion', async () => {
    const downloads: string[][] = [];
    const logs: string[] = [];
    const select = vi
      .fn<
        ({ message, choices }: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>
      >()
      .mockResolvedValueOnce('Qwen3.8-27B')
      .mockResolvedValueOnce('UD-Q4_K_XL');

    const slug = await runFirstRunWizard({
      io: { select, isTTY: true, log: (line) => logs.push(line) },
      download: async (argv) => {
        downloads.push(argv);
      },
      modelsDir: MODELS_DIR,
      totalMemBytes: 64 * GIB,
    });

    expect(slug).toBe('qwen3.8-27b-gguf');
    expect(downloads).toEqual([
      [
        '-m',
        'unsloth/Qwen3.8-27B-GGUF',
        '-g',
        '*UD-Q4_K_XL*',
        '-g',
        'config.json',
        '-g',
        'mmproj-BF16.gguf',
        '--assets-repo',
        'Qwen/Qwen3.8-27B',
        '--complete',
        '-o',
        join(MODELS_DIR, 'qwen3.8-27b-gguf'),
      ],
      ['-m', 'z-lab/Qwen3.8-27B-DFlash2', '--complete', '-o', join(MODELS_DIR, 'qwen3.8-27b-dflash2')],
    ]);
    expect(logs.some((line) => line.includes('DFlash2'))).toBe(true);
  });

  it('downloads only the chosen build for models without a companion draft', async () => {
    const downloads: string[][] = [];
    const select = vi
      .fn<
        ({ message, choices }: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>
      >()
      .mockResolvedValueOnce('Hy4-flash');

    const slug = await runFirstRunWizard({
      io: { select, isTTY: true, log: () => {} },
      download: async (argv) => {
        downloads.push(argv);
      },
      modelsDir: MODELS_DIR,
      totalMemBytes: 128 * GIB,
      platform: 'darwin',
    });

    expect(slug).toBe('hy4-flash-gguf');
    expect(downloads).toHaveLength(1);
    expect(downloads[0]).toContain('AngelSlim/Hy4-flash-GGUF');
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

  it('chains the autoInstall companion in the non-TTY hint for entries with autoInstall drafts only', async () => {
    const err = await runFirstRunWizard({
      io: { select: vi.fn(), isTTY: false, log: () => {} },
      download: vi.fn(),
      modelsDir: MODELS_DIR,
      totalMemBytes: 128 * GIB,
      platform: 'darwin',
    }).catch((e: unknown) => e as Error);

    expect(err).toBeInstanceOf(Error);
    if (!(err instanceof Error)) return;
    expect(err.message).toContain(
      `&& mlx download model -m z-lab/Qwen3.8-27B-DFlash2 --complete -o ${join(MODELS_DIR, 'qwen3.8-27b-dflash2')}`,
    );
    expect(err.message).toContain(
      `&& mlx download model -m incoai/Inkling-2-Mini-DFlash2 --complete -o ${join(MODELS_DIR, 'inkling-2-mini-dflash2')}`,
    );
    const lines = err.message.split('\n');
    const flashNextLine = lines.find((line) => line.includes('unsloth/Qwen3.8-Flash-Next-GGUF'));
    const hy4Line = lines.find((line) => line.includes('AngelSlim/Hy4-flash-GGUF'));
    expect(flashNextLine).toBeDefined();
    expect(flashNextLine).not.toContain(' && ');
    expect(hy4Line).toBeDefined();
    expect(hy4Line).not.toContain(' && ');
  });

  it('omits darwin-only Hy4-flash from interactive choices and non-TTY hints on win32 and linux', async () => {
    for (const platform of ['win32', 'linux'] as const) {
      const seenChoices: string[] = [];
      const select = vi
        .fn<
          ({
            message,
            choices,
          }: {
            message: string;
            choices: Array<{ name: string; value: string }>;
          }) => Promise<string>
        >()
        .mockImplementationOnce(async ({ choices }) => {
          seenChoices.push(...choices.map((c) => c.value));
          return 'Qwen3.8-Flash-Next';
        });

      const slug = await runFirstRunWizard({
        io: { select, isTTY: true, log: () => {} },
        download: async () => {},
        modelsDir: MODELS_DIR,
        totalMemBytes: 128 * GIB,
        platform,
      });
      expect(slug).toBe('qwen3.8-flash-next-gguf');
      expect(seenChoices, platform).toEqual(['Qwen3.8-27B', 'Inkling-2-Mini', 'Qwen3.8-Flash-Next']);

      const err = await runFirstRunWizard({
        io: { select: vi.fn(), isTTY: false, log: () => {} },
        download: vi.fn(),
        modelsDir: MODELS_DIR,
        totalMemBytes: 128 * GIB,
        platform,
      }).catch((e: unknown) => e as Error);
      expect(err).toBeInstanceOf(Error);
      if (!(err instanceof Error)) return;
      expect(err.message).not.toContain('AngelSlim/Hy4-flash-GGUF');
      expect(err.message).toContain('unsloth/Qwen3.8-Flash-Next-GGUF');
    }
  });
});
