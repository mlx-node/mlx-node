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

import {
  type CatalogBuild,
  type CatalogEntry,
  catalogBuildRepo,
  catalogBuildSlug,
  catalogDraftSlug,
  visibleCatalog,
} from '@mlx-node/agent';

export interface WizardIO {
  select: (opts: { message: string; choices: Array<{ name: string; value: string }> }) => Promise<string>;
  isTTY: boolean;
  log: (line: string) => void;
}

export interface WizardDeps {
  io: WizardIO;
  /** Runs `mlx download model` with the given argv. */
  download: (argv: string[]) => Promise<unknown>;
  /** Resolved models dir: the download lands in `<modelsDir>/<build folder>`, the dir `mlx agent` re-scans. */
  modelsDir: string;
  /** Physical RAM in bytes: the models offered are `visibleCatalog(totalMemBytes)`. */
  totalMemBytes: number;
}

/**
 * argv for `mlx download model`, shared by the interactive download AND
 * the non-TTY hint so both target the same destination: the output is
 * always pinned to the build's folder under the models dir, the folder the
 * dashboard installs the same build to (a later build of a repository has
 * its own folder, so the CLI's default `<repo name>` would be wrong for it).
 *
 * The GGUF selection fields ride along so the wizard installs exactly the
 * same file set the dashboard does: one weight variant out of a
 * multi-quant GGUF repo, plus the base-model tokenizer sidecars that GGUF
 * repos lack (without them the runtime falls back to its embedded tokenizer
 * extraction and tool calling silently breaks). Both fields describe the
 * GGUF build only — a platform override (the pre-converted CUDA repo)
 * installs as a plain safetensors directory.
 */
function downloadModelArgv(build: CatalogBuild, modelsDir: string): string[] {
  const hfRepo = catalogBuildRepo(build);
  const own = hfRepo === build.hfRepo;
  const argv = ['-m', hfRepo];
  for (const glob of own ? (build.globs ?? []) : []) {
    argv.push('-g', glob);
  }
  if (own && build.assetsRepo !== undefined) {
    argv.push('--assets-repo', build.assetsRepo);
  }
  // The catalog's prescribed selection is the COMPLETE model, so the marker
  // must not read as a partial (arbitrary-subset) glob run: the dashboard
  // only treats full markers as installed and only offers updates for them.
  argv.push('--complete');
  argv.push('-o', join(modelsDir, catalogBuildSlug(build)));
  return argv;
}

/** argv for `mlx download model` of a companion draft, pinned to the folder the dashboard installs it to. */
function downloadDraftArgv(draft: NonNullable<CatalogEntry['draft']>, modelsDir: string): string[] {
  return ['-m', draft.hfRepo, '--complete', '-o', join(modelsDir, catalogDraftSlug(draft))];
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

/** One line of the build question: `UD-Q4_K_M (~16.6 GB) — Unsloth GGUF · 1.1 GB smaller`. */
function buildChoiceName(build: CatalogBuild): string {
  const facts = build.note === undefined ? build.source : `${build.source} · ${build.note}`;
  return `${build.id} (~${build.sizeGb} GB) — ${facts}`;
}

/**
 * The build of `entry` to install: the only build of a one-build model with no question, else the answer to
 * "Build of {model}", whose choices are in catalog order so the first build is the one Enter takes.
 */
async function chooseBuild(io: WizardIO, entry: CatalogEntry): Promise<CatalogBuild> {
  if (entry.builds.length === 1) {
    return entry.builds[0];
  }
  const chosen = await io.select({
    message: `Build of ${entry.label}`,
    choices: entry.builds.map((build) => ({ name: buildChoiceName(build), value: build.id })),
  });
  const build = entry.builds.find((candidate) => candidate.id === chosen);
  if (build === undefined) {
    throw new Error(`Wizard choice "${chosen}" is not a build of ${entry.label}`);
  }
  return build;
}

/**
 * Offer the catalog (default entry first), then the builds of the chosen model when it has more than one, download
 * the chosen build and return the folder it installs to. Enter on both questions installs the first build of the
 * default model. Non-TTY sessions cannot prompt: throws with the manual `mlx download model` commands for the first
 * build of each model instead — the caller must not touch stdin on that path (the binary's print mode owns it).
 */
export async function runFirstRunWizard(deps: WizardDeps): Promise<string> {
  const catalog = visibleCatalog(deps.totalMemBytes);

  if (!deps.io.isTTY) {
    const commands = catalog
      .map((entry) => {
        const modelCmd = `mlx download model ${downloadModelArgv(entry.builds[0], deps.modelsDir).map(shellQuote).join(' ')}`;
        if (entry.draft?.autoInstall !== true) return `  ${modelCmd}`;
        const draftCmd = `mlx download model ${downloadDraftArgv(entry.draft, deps.modelsDir).map(shellQuote).join(' ')}`;
        return `  ${modelCmd} && ${draftCmd}`;
      })
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
      name: `${entry.label} (~${entry.builds[0].sizeGb} GB) — ${entry.description}`,
      value: entry.label,
    })),
  });

  const chosenEntry = ordered.find((entry) => entry.label === chosen);
  if (chosenEntry === undefined) {
    throw new Error(`Wizard choice "${chosen}" is not in the visible catalog`);
  }
  const build = await chooseBuild(deps.io, chosenEntry);
  await deps.download(downloadModelArgv(build, deps.modelsDir));
  if (chosenEntry.draft?.autoInstall === true) {
    const draftArgv = downloadDraftArgv(chosenEntry.draft, deps.modelsDir);
    deps.io.log(`Downloading companion ${chosenEntry.draft.label} (~${chosenEntry.draft.sizeGb} GB)...`);
    try {
      await deps.download(draftArgv);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      deps.io.log(
        `Warning: failed to download companion ${chosenEntry.draft.label} (${detail}). Continuing with ${chosenEntry.label}; retry later with:\n` +
          `  mlx download model ${draftArgv.map(shellQuote).join(' ')}`,
      );
    }
  }
  return catalogBuildSlug(build);
}
