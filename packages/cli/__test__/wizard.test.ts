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

  it('offers Hy4-flash SPARK-4Bit builds (and omits darwin Q4_K_M) on win32 and linux Spark hosts', async () => {
    const cases: Array<readonly [NodeJS.Platform, { backend: string; totalMemoryBytes: number }]> = [
      ['win32', { backend: 'cuda', totalMemoryBytes: 102 * GIB }],
      ['linux', { backend: 'cuda', totalMemoryBytes: 128 * GIB }],
    ];
    for (const [platform, gpu] of cases) {
      const seenChoices: string[] = [];
      const seenBuilds: string[] = [];
      const downloads: string[][] = [];
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
          return 'Hy4-flash';
        })
        .mockImplementationOnce(async ({ choices }) => {
          seenBuilds.push(...choices.map((c) => c.value));
          return 'SPARK-4Bit';
        });

      const slug = await runFirstRunWizard({
        io: { select, isTTY: true, log: () => {} },
        download: async (argv) => {
          downloads.push(argv);
        },
        modelsDir: MODELS_DIR,
        totalMemBytes: 128 * GIB,
        platform,
        gpu,
      });
      expect(slug, platform).toBe('hy4-flash-gguf-spark-4bit');
      expect(seenChoices, platform).toEqual(['Qwen3.8-27B', 'Inkling-2-Mini', 'Qwen3.8-Flash-Next', 'Hy4-flash']);
      expect(seenBuilds, platform).toEqual(['SPARK-4Bit', 'SPARK-4Bit-Fast']);
      expect(downloads, platform).toEqual([
        [
          '-m',
          'AngelSlim/Hy4-flash-GGUF',
          '-g',
          'Hy4-flash-SPARK-4Bit.gguf',
          '--assets-repo',
          'tencent/Hy4-flash',
          '--complete',
          '-o',
          join(MODELS_DIR, 'hy4-flash-gguf-spark-4bit'),
        ],
      ]);

      const err = await runFirstRunWizard({
        io: { select: vi.fn(), isTTY: false, log: () => {} },
        download: vi.fn(),
        modelsDir: MODELS_DIR,
        totalMemBytes: 128 * GIB,
        platform,
        gpu,
      }).catch((e: unknown) => e as Error);
      expect(err).toBeInstanceOf(Error);
      if (!(err instanceof Error)) return;
      expect(err.message).toContain('Hy4-flash-SPARK-4Bit.gguf');
      expect(err.message).not.toContain('*Q4_K_M*');
      expect(err.message).toContain('unsloth/Qwen3.8-Flash-Next-GGUF');
    }
  });

  it('omits Hy4-flash on win32 and linux when GPU is a 24 GiB discrete CUDA card, CPU-only, or unknown', async () => {
    const unsupportedGpus = [
      ['24 GiB discrete CUDA', { backend: 'cuda', totalMemoryBytes: 24 * GIB }],
      ['CPU-only 128 GiB', { backend: 'cpu', totalMemoryBytes: 128 * GIB }],
      ['missing GPU info', undefined],
    ] as const;
    for (const platform of ['win32', 'linux'] as const) {
      for (const [label, gpu] of unsupportedGpus) {
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

        await runFirstRunWizard({
          io: { select, isTTY: true, log: () => {} },
          download: async () => {},
          modelsDir: MODELS_DIR,
          totalMemBytes: 128 * GIB,
          platform,
          gpu,
        });
        expect(seenChoices, `${platform} ${label}`).toEqual([
          'Qwen3.8-27B',
          'Inkling-2-Mini',
          'Qwen3.8-Flash-Next',
        ]);

        const err = await runFirstRunWizard({
          io: { select: vi.fn(), isTTY: false, log: () => {} },
          download: vi.fn(),
          modelsDir: MODELS_DIR,
          totalMemBytes: 128 * GIB,
          platform,
          gpu,
        }).catch((e: unknown) => e as Error);
        expect(err).toBeInstanceOf(Error);
        if (!(err instanceof Error)) return;
        expect(err.message, `${platform} ${label}`).not.toContain('AngelSlim/Hy4-flash-GGUF');
        expect(err.message, `${platform} ${label}`).toContain('unsloth/Qwen3.8-Flash-Next-GGUF');
      }
    }
  });
});
