import { join } from 'node:path';

import { MODEL_CATALOG, visibleCatalog } from '@mlx-node/agent';
import { describe, expect, it } from 'vite-plus/test';

import { runFirstRunWizard, type WizardIO } from '../../packages/cli/src/commands/agent/wizard.js';

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

    await runFirstRunWizard({ io, download });

    expect(selectCalls).toHaveLength(1);
    const choices = selectCalls[0]!.choices;

    const visible = visibleCatalog();
    expect(choices.map((c) => c.value).sort()).toEqual(visible.map((e) => e.hfRepo).sort());

    // Default entry is preselected by coming first.
    const defaultEntry = visible.find((e) => e.isDefault)!;
    expect(defaultEntry).toBeDefined();
    expect(choices[0]!.value).toBe(defaultEntry.hfRepo);

    // Hidden entries are never offered (the catalog must actually carry one
    // for this assertion to mean anything).
    const hidden = MODEL_CATALOG.filter((e) => e.hidden);
    expect(hidden.length).toBeGreaterThan(0);
    for (const entry of hidden) {
      expect(choices.map((c) => c.value)).not.toContain(entry.hfRepo);
    }

    // Labels carry size and description.
    for (const entry of visible) {
      const choice = choices.find((c) => c.value === entry.hfRepo)!;
      expect(choice.name).toContain(entry.label);
      expect(choice.name).toContain(`(~${entry.sizeGb} GB)`);
      expect(choice.name).toContain(entry.description);
    }
  });

  it('passes the chosen repo to download and returns it', async () => {
    const visible = visibleCatalog();
    const chosen = visible[visible.length - 1]!.hfRepo;
    const { io } = makeIO({ chosen });
    const { download, calls } = makeDownload();

    const result = await runFirstRunWizard({ io, download });

    expect(result).toBe(chosen);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual(['-m', chosen]);
  });

  it('pins the download output under modelsDir when provided', async () => {
    const chosen = visibleCatalog()[0]!.hfRepo;
    const { io } = makeIO({ chosen });
    const { download, calls } = makeDownload();

    await runFirstRunWizard({ io, download, modelsDir: '/custom/models' });

    const slug = chosen.split('/').pop()!.toLowerCase();
    expect(calls[0]).toEqual(['-m', chosen, '-o', join('/custom/models', slug)]);
  });

  it('throws without prompting or downloading when not a TTY', async () => {
    const { io, selectCalls } = makeIO({ isTTY: false });
    const { download, calls } = makeDownload();

    await expect(runFirstRunWizard({ io, download })).rejects.toThrow(/mlx download model/);

    // The hint must list every offered repo so headless users can copy a command.
    try {
      await runFirstRunWizard({ io, download });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      for (const entry of visibleCatalog()) {
        expect(message).toContain(entry.hfRepo);
      }
    }

    expect(selectCalls).toHaveLength(0);
    expect(calls).toHaveLength(0);
  });

  it('pins the non-TTY hint commands to modelsDir exactly like the interactive download', async () => {
    const { io } = makeIO({ isTTY: false });
    const { download, calls } = makeDownload();

    try {
      await runFirstRunWizard({ io, download, modelsDir: '/custom/models' });
      expect.unreachable('wizard must throw without a TTY');
    } catch (error) {
      const message = (error as Error).message;
      for (const entry of visibleCatalog()) {
        const slug = entry.hfRepo.split('/').pop()!.toLowerCase();
        expect(message).toContain(`mlx download model -m ${entry.hfRepo} -o ${join('/custom/models', slug)}`);
      }
    }
    expect(calls).toHaveLength(0);
  });
});
