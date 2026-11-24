import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

import {
  type CatalogEntry,
  catalogBuildRepo,
  catalogBuildSlug,
  catalogDraftSlug,
  MODEL_CATALOG,
  visibleCatalog,
} from '@mlx-node/agent';
import { describe, expect, it } from 'vite-plus/test';

import { runFirstRunWizard, type WizardIO } from '../../packages/cli/src/commands/agent/wizard.js';

/**
 * Parse a rendered command's argument string with a REAL POSIX shell and
 * return the words it would hand the program (NUL-delimited so embedded
 * newlines can't confuse the split). This is the ground truth for "does
 * the copy-pasteable hint reconstruct the intended argv".
 */
function shellSplit(renderedArgs: string): string[] {
  const out = execFileSync('/bin/sh', ['-c', `printf '%s\\0' ${renderedArgs}`]);
  const words = out.toString('utf8').split('\0');
  words.pop(); // printf leaves a trailing NUL → one empty tail element
  return words;
}

/** The models dir every wizard run below is given. */
const MODELS_DIR = '/custom/models';

const GIB = 2 ** 30;
/** RAM at which all four chat models are offered (Flash Next and Hy4-flash need 96 GiB). */
const LARGE_MAC = 128 * GIB;
/** RAM at which only Qwen3.8-27B and Inkling-2-Mini are offered. */
const SMALL_MAC = 64 * GIB;

/** The platform-resolved repo of the first build of `entry`, the build the non-TTY hint names. */
function repoOf(entry: CatalogEntry): string {
  return catalogBuildRepo(entry.builds[0]);
}

/** The argv of the first build of `entry`, built from the catalog fields, for the non-TTY hint round trips. */
function expectedArgv(entry: CatalogEntry, modelsDir: string = MODELS_DIR): string[] {
  const build = entry.builds[0];
  const argv = ['-m', repoOf(entry)];
  for (const glob of build.globs ?? []) argv.push('-g', glob);
  if (build.assetsRepo !== undefined) argv.push('--assets-repo', build.assetsRepo);
  argv.push('--complete');
  argv.push('-o', join(modelsDir, catalogBuildSlug(build)));
  return argv;
}

/** The chained argv list rendered on one non-TTY hint line (model, plus auto-installed draft when configured). */
function expectedHintArgvs(entry: CatalogEntry, modelsDir: string = MODELS_DIR): string[][] {
  const argvs = [expectedArgv(entry, modelsDir)];
  if (entry.draft?.autoInstall === true) {
    argvs.push(['-m', entry.draft.hfRepo, '--complete', '-o', join(modelsDir, catalogDraftSlug(entry.draft))]);
  }
  return argvs;
}

interface SelectCall {
  message: string;
  choices: Array<{ name: string; value: string }>;
}

/**
 * A scripted terminal: each question takes the next of `answers`; a question past the end takes the first choice,
 * which is what Enter picks in `@inquirer/prompts` `select`.
 */
function makeIO(overrides: Partial<WizardIO> & { answers?: string[] } = {}): {
  io: WizardIO;
  selectCalls: SelectCall[];
  logs: string[];
} {
  const selectCalls: SelectCall[] = [];
  const logs: string[] = [];
  const answers = [...(overrides.answers ?? [])];
  const io: WizardIO = {
    isTTY: overrides.isTTY ?? true,
    log: (line) => logs.push(line),
    select:
      overrides.select ??
      (async (opts) => {
        selectCalls.push(opts);
        return answers.shift() ?? opts.choices[0]!.value;
      }),
  };
  return { io, selectCalls, logs };
}

function makeDownload(): { download: (argv: string[]) => Promise<unknown>; calls: string[][] } {
  const calls: string[][] = [];
  return {
    download: async (argv) => {
      calls.push(argv);
      return undefined;
    },
    calls,
  };
}

describe('runFirstRunWizard', () => {
  it('offers the visible chat models with the default first and the first build size on each line', async () => {
    const { io, selectCalls } = makeIO();
    const { download } = makeDownload();

    await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });

    const question = selectCalls[0]!;
    expect(question.message).toBe('Model to download');
    expect(question.choices.map((c) => c.value)).toEqual([
      'Qwen3.8-27B',
      'Inkling-2-Mini',
      'Qwen3.8-Flash-Next',
      'Hy4-flash',
    ]);
    expect(question.choices.map((c) => c.name)).toEqual([
      'Qwen3.8-27B (~17.7 GB) — Best tool use — recommended default',
      'Inkling-2-Mini (~18.7 GB) — Looped MoE with vision and DFlash2 draft, 1M context',
      'Qwen3.8-Flash-Next (~111.3 GB) — 125B MoE, 512 experts streamed from SSD',
      'Hy4-flash (~168.3 GB) — 512-expert MoE streamed from SSD, 256K context',
    ]);

    // Hidden entries are never offered (the catalog must actually carry one
    // for this assertion to mean anything).
    const hidden = MODEL_CATALOG.filter((e) => e.hidden);
    expect(hidden.length).toBeGreaterThan(0);
    for (const entry of hidden) {
      expect(question.choices.map((c) => c.value)).not.toContain(entry.label);
    }
  });

  it('offers only the models that fit the memory it is given', async () => {
    const { io, selectCalls } = makeIO();
    const { download } = makeDownload();

    await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: SMALL_MAC });

    expect(selectCalls[0]!.choices.map((c) => c.value)).toEqual(['Qwen3.8-27B', 'Inkling-2-Mini']);
  });

  it('Enter, Enter installs the first build of the default model to its folder', async () => {
    const { io, selectCalls } = makeIO();
    const { download, calls } = makeDownload();

    const folder = await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });

    expect(selectCalls.map((call) => call.message)).toEqual(['Model to download', 'Build of Qwen3.8-27B']);
    expect(selectCalls[1]!.choices).toEqual([
      { name: 'UD-Q4_K_XL (~17.7 GB) — Unsloth GGUF', value: 'UD-Q4_K_XL' },
      { name: 'UD-Q4_K_M (~16.6 GB) — Unsloth GGUF · 1.1 GB smaller', value: 'UD-Q4_K_M' },
    ]);
    expect(folder).toBe('qwen3.8-27b-gguf');
    expect(calls).toEqual([
      [
        '-m',
        'unsloth/Qwen3.8-27B-GGUF',
        '-g',
        '*UD-Q4_K_XL*',
        '-g',
        'config.json',
        '--assets-repo',
        'Qwen/Qwen3.8-27B',
        '--complete',
        '-o',
        '/custom/models/qwen3.8-27b-gguf',
      ],
    ]);
  });

  it('installs Qwen3.8-27B UD-Q4_K_M to its own folder with its own glob', async () => {
    const { io } = makeIO({ answers: ['Qwen3.8-27B', 'UD-Q4_K_M'] });
    const { download, calls } = makeDownload();

    const folder = await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });

    expect(folder).toBe('qwen3.8-27b-gguf-ud-q4_k_m');
    expect(calls).toEqual([
      [
        '-m',
        'unsloth/Qwen3.8-27B-GGUF',
        '-g',
        '*UD-Q4_K_M*',
        '-g',
        'config.json',
        '--assets-repo',
        'Qwen/Qwen3.8-27B',
        '--complete',
        '-o',
        '/custom/models/qwen3.8-27b-gguf-ud-q4_k_m',
      ],
    ]);
  });

  it('asks Inkling-2-Mini for one of its three builds, first build first', async () => {
    const { io, selectCalls } = makeIO({ answers: ['Inkling-2-Mini'] });
    const { download, calls } = makeDownload();

    const folder = await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });

    expect(selectCalls.map((call) => call.message)).toEqual(['Model to download', 'Build of Inkling-2-Mini']);
    expect(selectCalls[1]!.choices).toEqual([
      { name: 'UD-Q4_K_XL (~18.7 GB) — Unsloth GGUF', value: 'UD-Q4_K_XL' },
      { name: 'UD-Q4_K_M (~17.7 GB) — Unsloth GGUF · 1 GB smaller', value: 'UD-Q4_K_M' },
      { name: 'Original (~20.2 GB) — Thinking Machines · NVFP4 + MXFP8', value: 'Original' },
    ]);
    expect(folder).toBe('inkling-2-mini-gguf');
    expect(calls).toEqual([
      [
        '-m',
        'unsloth/Inkling-2-Mini-GGUF',
        '-g',
        '*UD-Q4_K_XL.gguf',
        '-g',
        'mmproj-BF16.gguf',
        '-g',
        'dspark-Inkling-2-Mini-Q8_0.gguf',
        '--assets-repo',
        'thinkingmachines/Inkling-2-Mini',
        '--complete',
        '-o',
        '/custom/models/inkling-2-mini-gguf',
      ],
      ['-m', 'incoai/Inkling-2-Mini-DFlash2', '--complete', '-o', '/custom/models/inkling-2-mini-dflash2'],
    ]);
  });

  it('installs Inkling-2-Mini UD-Q4_K_M with a weight glob that ends in .gguf', async () => {
    const { io } = makeIO({ answers: ['Inkling-2-Mini', 'UD-Q4_K_M'] });
    const { download, calls } = makeDownload();

    const folder = await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });

    expect(folder).toBe('inkling-2-mini-gguf-ud-q4_k_m');
    expect(calls).toEqual([
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
        '/custom/models/inkling-2-mini-gguf-ud-q4_k_m',
      ],
      ['-m', 'incoai/Inkling-2-Mini-DFlash2', '--complete', '-o', '/custom/models/inkling-2-mini-dflash2'],
    ]);
  });

  it('installs Inkling-2-Mini Original from its own repository with no globs and no assets repo', async () => {
    const { io } = makeIO({ answers: ['Inkling-2-Mini', 'Original'] });
    const { download, calls } = makeDownload();

    const folder = await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });

    expect(folder).toBe('inkling-2-mini');
    expect(calls).toEqual([
      ['-m', 'thinkingmachines/Inkling-2-Mini', '--complete', '-o', '/custom/models/inkling-2-mini'],
      ['-m', 'incoai/Inkling-2-Mini-DFlash2', '--complete', '-o', '/custom/models/inkling-2-mini-dflash2'],
    ]);
  });

  it('asks a one-build model one question', async () => {
    const { io, selectCalls } = makeIO({ answers: ['Hy4-flash'] });
    const { download, calls } = makeDownload();

    const folder = await runFirstRunWizard({ io, download, modelsDir: '/other/models', totalMemBytes: LARGE_MAC });

    expect(selectCalls.map((call) => call.message)).toEqual(['Model to download']);
    expect(folder).toBe('hy4-flash-gguf');
    expect(calls).toEqual([
      [
        '-m',
        'AngelSlim/Hy4-flash-GGUF',
        '-g',
        '*Q4_K_M*',
        '--assets-repo',
        'tencent/Hy4-flash',
        '--complete',
        '-o',
        '/other/models/hy4-flash-gguf',
      ],
    ]);
  });

  it('refuses a build answer that is not a build of the model and downloads nothing', async () => {
    const { io } = makeIO({ answers: ['Qwen3.8-27B', 'Q8_0'] });
    const { download, calls } = makeDownload();

    await expect(runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC })).rejects.toThrow(
      'Wizard choice "Q8_0" is not a build of Qwen3.8-27B',
    );
    expect(calls).toHaveLength(0);
  });

  it('throws without prompting or downloading when not a TTY', async () => {
    const { io, selectCalls } = makeIO({ isTTY: false });
    const { download, calls } = makeDownload();

    await expect(runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC })).rejects.toThrow(
      /mlx download model/,
    );

    // The hint must list every offered repo so headless users can copy a command.
    try {
      await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      for (const entry of visibleCatalog(LARGE_MAC)) {
        expect(message).toContain(repoOf(entry));
      }
    }

    expect(selectCalls).toHaveLength(0);
    expect(calls).toHaveLength(0);
  });

  it('pins the non-TTY hint commands to modelsDir exactly like the interactive download', async () => {
    const { io } = makeIO({ isTTY: false });
    const { download, calls } = makeDownload();

    try {
      await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      for (const entry of visibleCatalog(LARGE_MAC)) {
        const line = message.split('\n').find((l) => l.includes(repoOf(entry)))!;
        expect(line).toBeDefined();
        const commands = line.trim().split(' && ');
        // The hint must reconstruct the interactive argv(s) exactly — including
        // globs, assets repo, and any auto-installed companion draft command.
        expect(
          commands.map((cmd) => {
            expect(cmd.startsWith('mlx download model ')).toBe(true);
            return shellSplit(cmd.slice('mlx download model '.length));
          }),
        ).toEqual(expectedHintArgvs(entry));
      }
    }
    expect(calls).toHaveLength(0);
  });

  it('shell-quotes hint elements so a models dir with spaces and metacharacters survives copy-paste', async () => {
    // Space, single quote, `$`, `;` and `&` in one path. Deliberately
    // harmless commands after the metacharacters: if quoting regresses,
    // the real /bin/sh below would EXECUTE them.
    const modelsDir = `/tmp/My Models/it's;echo x&$HOME`;
    const { io } = makeIO({ isTTY: false });
    const { download, calls } = makeDownload();

    let message = '';
    try {
      await runFirstRunWizard({ io, download, modelsDir, totalMemBytes: LARGE_MAC });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      message = (error as Error).message;
    }

    for (const entry of visibleCatalog(LARGE_MAC)) {
      const slug = catalogBuildSlug(entry.builds[0]);
      const line = message.split('\n').find((l) => l.includes(repoOf(entry)));
      expect(line).toBeDefined();
      const commands = line!.trim().split(' && ');
      // Round-trip: a real POSIX shell parses each displayed command back
      // into EXACTLY the argv the interactive path would pass — globs as
      // single words, one `-o` word, `$HOME` unexpanded, `;`/`&` inert.
      expect(
        commands.map((cmd) => {
          expect(cmd.startsWith('mlx download model ')).toBe(true);
          return shellSplit(cmd.slice('mlx download model '.length));
        }),
      ).toEqual(expectedHintArgvs(entry, modelsDir));
      // Known-correct quoted form: safe words bare, unsafe -o value
      // single-quoted with the embedded quote escaped as '\''.
      expect(line).toContain(`-o '/tmp/My Models/it'\\''s;echo x&$HOME/${slug}'`);
      if (entry.draft?.autoInstall === true) {
        expect(line).toContain(`-o '/tmp/My Models/it'\\''s;echo x&$HOME/${catalogDraftSlug(entry.draft)}'`);
      }
    }
    expect(calls).toHaveLength(0);
  });

  it('renders plain safe paths unquoted (round-trips too)', async () => {
    const { io } = makeIO({ isTTY: false });
    const { download } = makeDownload();

    try {
      await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR, totalMemBytes: LARGE_MAC });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      const entry = visibleCatalog(LARGE_MAC)[0]!;
      const slug = catalogBuildSlug(entry.builds[0]);
      // Safe paths stay readable: the -o value renders unquoted...
      expect(message).toContain(`-o /custom/models/${slug}`);
      // ...and the whole line still parses back to the intended argv.
      const line = message.split('\n').find((l) => l.includes(repoOf(entry)))!;
      const renderedArgs = line.trim().replace(/^mlx download model /, '');
      expect(shellSplit(renderedArgs)).toEqual(expectedArgv(entry));
    }
  });
});
