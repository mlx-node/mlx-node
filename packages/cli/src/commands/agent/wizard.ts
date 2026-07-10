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
 * argv for `mlx download model`, shared by the interactive download AND
 * the non-TTY hint so both target the same destination: with a models
 * dir in play the output is pinned to `<modelsDir>/<slug>` — otherwise a
 * copy-pasted hint would download to the DEFAULT dir and a re-run under
 * a custom `--models-dir` would still find nothing.
 */
function downloadModelArgv(hfRepo: string, modelsDir: string | undefined): string[] {
  const argv = ['-m', hfRepo];
  if (modelsDir) {
    argv.push('-o', join(modelsDir, repoSlug(hfRepo)));
  }
  return argv;
}

/**
 * Quote ONE argv element for copy-paste into a POSIX shell. Elements of
 * only safe characters pass through untouched (keeps the common hint
 * readable); anything else is single-quoted, with embedded single
 * quotes escaped as `'\''` (close quote, literal `'`, reopen). Only the
 * DISPLAYED hint needs this — the interactive path hands `download`
 * real argv elements, no shell in between.
 */
function shellQuote(arg: string): string {
  if (/^[A-Za-z0-9@%+=:,./_-]+$/.test(arg)) {
    return arg;
  }
  return `'${arg.replaceAll("'", String.raw`'\''`)}'`;
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
    const commands = catalog
      .map(
        (entry) => `  mlx download model ${downloadModelArgv(entry.hfRepo, deps.modelsDir).map(shellQuote).join(' ')}`,
      )
      .join('\n');
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

  await deps.download(downloadModelArgv(chosen, deps.modelsDir));
  return chosen;
}
