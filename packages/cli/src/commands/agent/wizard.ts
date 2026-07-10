/**
 * First-run wizard for `mlx agent`: when no local model is discovered,
 * offer the curated catalog (`visibleCatalog`) and download the pick via
 * the injected `download` (production: `run` from `../download-model.js`).
 *
 * IO is injectable (same pattern as `TokenPromptIO` in `hf-token.ts`) so
 * tests can drive the wizard without an interactive prompt; the real
 * `@inquirer/prompts` `select` is wired in `./index.ts`, never here.
 */

import { join } from 'node:path';

import { visibleCatalog } from '@mlx-node/agent';

export interface WizardIO {
  select: (opts: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>;
  isTTY: boolean;
  log: (line: string) => void;
}

export interface WizardDeps {
  io: WizardIO;
  /** Runs `mlx download model` with the given argv. */
  download: (argv: string[]) => Promise<unknown>;
  /**
   * Resolved models dir. When set, the download is pinned to
   * `<modelsDir>/<slug>` (the same layout `mlx download model` defaults
   * to) so a custom `--models-dir` receives the model it will re-scan.
   */
  modelsDir?: string;
}

/** Repo slug → local directory name, mirroring `mlx download model`. */
function repoSlug(hfRepo: string): string {
  return hfRepo.split('/').pop()!.toLowerCase();
}

/**
 * Offer the catalog (default entry first), download the chosen repo, and
 * return its HF slug. Non-TTY sessions cannot prompt: throws with the
 * manual `mlx download model` commands instead — the caller must not
 * touch stdin on that path (pi print mode owns it).
 */
export async function runFirstRunWizard(deps: WizardDeps): Promise<string> {
  const catalog = visibleCatalog();

  if (!deps.io.isTTY) {
    const commands = catalog.map((entry) => `  mlx download model -m ${entry.hfRepo}`).join('\n');
    throw new Error(
      `No local models found. Run in a terminal for the setup wizard, or download one directly:\n${commands}`,
    );
  }

  const ordered = [...catalog.filter((entry) => entry.isDefault), ...catalog.filter((entry) => !entry.isDefault)];
  deps.io.log('No local models found — first-run setup: pick a model to download.');
  const chosen = await deps.io.select({
    message: 'Model to download',
    choices: ordered.map((entry) => ({
      name: `${entry.label} (~${entry.sizeGb} GB) — ${entry.description}`,
      value: entry.hfRepo,
    })),
  });

  const argv = ['-m', chosen];
  if (deps.modelsDir) {
    argv.push('-o', join(deps.modelsDir, repoSlug(chosen)));
  }
  await deps.download(argv);
  return chosen;
}
