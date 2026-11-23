import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

import { type CatalogEntry, catalogBuildRepo, catalogBuildSlug, MODEL_CATALOG, visibleCatalog } from '@mlx-node/agent';
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

/** The platform-resolved repo of the first build of `entry`, the build the wizard installs. */
function repoOf(entry: CatalogEntry): string {
  return catalogBuildRepo(entry.builds[0]);
}

/**
 * The argv contract the wizard must hand `mlx download model` for one entry:
 * the platform-resolved repo of its first build, that build's globs, the
 * sidecar assets repo, and the output under `<modelsDir>/<build folder>`.
 * Written out literally — a change to the wizard's argv shape must fail here.
 */
function expectedArgv(entry: CatalogEntry, modelsDir: string = MODELS_DIR): string[] {
  const build = entry.builds[0];
  const argv = ['-m', repoOf(entry)];
  for (const glob of build.globs ?? []) argv.push('-g', glob);
  if (build.assetsRepo !== undefined) argv.push('--assets-repo', build.assetsRepo);
  // The prescribed selection is the complete model, never a partial glob run.
  argv.push('--complete');
  argv.push('-o', join(modelsDir, catalogBuildSlug(build)));
  return argv;
}

interface SelectCall {
  message: string;
  choices: Array<{ name: string; value: string }>;
}

function makeIO(overrides: Partial<WizardIO> & { chosen?: string } = {}): {
  io: WizardIO;
  selectCalls: SelectCall[];
  logs: string[];
} {
  const selectCalls: SelectCall[] = [];
  const logs: string[] = [];
  const io: WizardIO = {
    isTTY: overrides.isTTY ?? true,
    log: (line) => logs.push(line),
    select:
      overrides.select ??
      (async (opts) => {
        selectCalls.push(opts);
        return overrides.chosen ?? opts.choices[0]!.value;
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
  it('offers exactly the visible catalog with the default entry first', async () => {
    const { io, selectCalls } = makeIO();
    const { download } = makeDownload();

    await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR });

    expect(selectCalls).toHaveLength(1);
    const choices = selectCalls[0]!.choices;

    const visible = visibleCatalog();
    expect(choices.map((c) => c.value).sort()).toEqual(visible.map((e) => e.label).sort());

    // Default entry is preselected by coming first.
    const defaultEntry = visible.find((e) => e.isDefault)!;
    expect(defaultEntry).toBeDefined();
    expect(choices[0]!.value).toBe(defaultEntry.label);

    // Hidden entries are never offered (the catalog must actually carry one
    // for this assertion to mean anything).
    const hidden = MODEL_CATALOG.filter((e) => e.hidden);
    expect(hidden.length).toBeGreaterThan(0);
    for (const entry of hidden) {
      expect(choices.map((c) => c.value)).not.toContain(entry.label);
    }

    // Labels carry the first build's size and the description.
    for (const entry of visible) {
      const choice = choices.find((c) => c.value === entry.label)!;
      expect(choice.name).toContain(entry.label);
      expect(choice.name).toContain(`(~${entry.builds[0].sizeGb} GB)`);
      expect(choice.name).toContain(entry.description);
    }
  });

  it('downloads the first build of the chosen model and returns its folder', async () => {
    const visible = visibleCatalog();
    const entry = visible[visible.length - 1]!;
    const { io } = makeIO({ chosen: entry.label });
    const { download, calls } = makeDownload();

    const result = await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR });

    expect(result).toBe(catalogBuildSlug(entry.builds[0]));
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual(expectedArgv(entry));
  });

  it('carries the build globs and assets repo into the download argv', async () => {
    // The catalog must actually carry a multi-variant GGUF entry for this to
    // mean anything: the wizard exists to install one weight variant plus
    // the base-model tokenizer sidecars, never every quant in the repo.
    const entry = visibleCatalog().find((e) => e.builds[0].globs !== undefined && e.builds[0].assetsRepo !== undefined);
    expect(entry).toBeDefined();
    expect(entry!.builds[0].globs!.length).toBeGreaterThan(0);
    const { io } = makeIO({ chosen: entry!.label });
    const { download, calls } = makeDownload();

    await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR });

    expect(calls[0]).toEqual(expectedArgv(entry!));
  });

  it('pins the download output to the build folder under modelsDir', async () => {
    const entry = visibleCatalog()[0]!;
    const { io } = makeIO({ chosen: entry.label });
    const { download, calls } = makeDownload();

    await runFirstRunWizard({ io, download, modelsDir: '/other/models' });

    expect(calls[0]).toEqual(expectedArgv(entry, '/other/models'));
    expect(calls[0]!.slice(-2)).toEqual(['-o', '/other/models/qwen3.8-27b-gguf']);
  });

  it('throws without prompting or downloading when not a TTY', async () => {
    const { io, selectCalls } = makeIO({ isTTY: false });
    const { download, calls } = makeDownload();

    await expect(runFirstRunWizard({ io, download, modelsDir: MODELS_DIR })).rejects.toThrow(/mlx download model/);

    // The hint must list every offered repo so headless users can copy a command.
    try {
      await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      for (const entry of visibleCatalog()) {
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
      await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      for (const entry of visibleCatalog()) {
        const line = message.split('\n').find((l) => l.includes(repoOf(entry)))!;
        expect(line).toBeDefined();
        const renderedArgs = line.trim().replace(/^mlx download model /, '');
        // The hint must reconstruct the interactive argv exactly — including
        // the globs and assets repo, so a headless user gets the same file set.
        expect(shellSplit(renderedArgs)).toEqual(expectedArgv(entry));
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
      await runFirstRunWizard({ io, download, modelsDir });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      message = (error as Error).message;
    }

    for (const entry of visibleCatalog()) {
      const slug = catalogBuildSlug(entry.builds[0]);
      const line = message.split('\n').find((l) => l.includes(repoOf(entry)));
      expect(line).toBeDefined();
      const renderedArgs = line!.trim().replace(/^mlx download model /, '');
      // Round-trip: a real POSIX shell parses the displayed command back
      // into EXACTLY the argv the interactive path would pass — globs as
      // single words, one `-o` word, `$HOME` unexpanded, `;`/`&` inert.
      expect(shellSplit(renderedArgs)).toEqual(expectedArgv(entry, modelsDir));
      // Known-correct quoted form: safe words bare, unsafe -o value
      // single-quoted with the embedded quote escaped as '\''.
      expect(line).toContain(`-o '/tmp/My Models/it'\\''s;echo x&$HOME/${slug}'`);
    }
    expect(calls).toHaveLength(0);
  });

  it('renders plain safe paths unquoted (round-trips too)', async () => {
    const { io } = makeIO({ isTTY: false });
    const { download } = makeDownload();

    try {
      await runFirstRunWizard({ io, download, modelsDir: MODELS_DIR });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      const entry = visibleCatalog()[0]!;
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
