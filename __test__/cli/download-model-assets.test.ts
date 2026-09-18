import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

/**
 * The `--assets-repo` top-up must survive the completion short-circuits: a
 * directory whose weight files already look complete (a retry after a sidecar
 * fetch failed, or an install from before the flag existed) returns before
 * the download loop, so the sidecar pass has to run before that return — the
 * regression the review flagged. Everything network-shaped is mocked; the
 * filesystem is real (a temp output dir).
 */
const GGUF = 'Tiny-UD-Q4_K_XL.gguf';
const PRIMARY = 'unsloth/Tiny-GGUF';
const ASSETS = 'base/Tiny';

interface ManifestEntry {
  type: 'file';
  path: string;
  size: number;
}

const hub = vi.hoisted(() => ({
  manifests: {} as Record<string, ManifestEntry[]>,
  /** Resolvable revisions per repo; absent means "unresolved" (the legacy path). */
  shas: {} as Record<string, string>,
  listedRepos: [] as string[],
  downloaded: [] as string[],
  snapshotDir: '',
}));

vi.mock('@huggingface/hub', () => ({
  // No resolvable upstream revision → the CLI takes its legacy completeness
  // path, which is exactly where the repair must happen.
  // The CLI calls it as `modelInfo({ name, additionalFields, accessToken })`.
  modelInfo: async (params: { name: string }) => ({ sha: hub.shas[params.name] }),
  listFiles: async function* (params: { repo: { name: string } }) {
    hub.listedRepos.push(params.repo.name);
    for (const entry of hub.manifests[params.repo.name] ?? []) yield entry;
  },
  downloadFileToCacheDir: async (params: { path: string }) => {
    hub.downloaded.push(params.path);
    const snapshot = join(hub.snapshotDir, params.path.replaceAll('/', '_'));
    // Serve the manifest's byte count: a mismatched size makes the downloader's
    // post-copy verification retry (the mock IS the upstream here).
    const entry = Object.values(hub.manifests)
      .flat()
      .find((file) => file.path === params.path);
    writeFileSync(snapshot, 'x'.repeat(Math.max(1, entry?.size ?? 1)));
    return snapshot;
  },
}));

import { run } from '../../packages/cli/src/commands/download-model.js';

describe('download model --assets-repo', () => {
  let outputDir: string;
  let cacheDir: string;

  beforeEach(() => {
    outputDir = mkdtempSync(join(tmpdir(), 'mlx-assets-out-'));
    cacheDir = mkdtempSync(join(tmpdir(), 'mlx-assets-cache-'));
    hub.snapshotDir = mkdtempSync(join(tmpdir(), 'mlx-assets-snap-'));
    hub.listedRepos = [];
    hub.downloaded = [];
    hub.shas = {};
    hub.manifests = {
      [PRIMARY]: [{ type: 'file', path: GGUF, size: 300 }],
      [ASSETS]: [
        { type: 'file', path: 'config.json', size: 12 },
        { type: 'file', path: 'tokenizer.json', size: 20 },
      ],
    };
  });

  afterEach(() => {
    for (const dir of [outputDir, cacheDir, hub.snapshotDir]) rmSync(dir, { recursive: true, force: true });
  });

  it('tops up the sidecars even when the glob-matched weights already look complete', async () => {
    // The weights (and nothing else) are already on disk: the run's remaining
    // work is the tokenizer pair, and the glob completeness check would
    // otherwise return before any fetch. The assets revision must be
    // resolvable — an unpinned repair is skipped rather than fetched from
    // mutable `main` (see the skip test below).
    hub.shas[ASSETS] = 'b'.repeat(40);
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));

    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--assets-repo', ASSETS, '--cache-dir', cacheDir]);

    expect(hub.listedRepos).toContain(ASSETS);
    expect([...hub.downloaded].sort()).toEqual(['config.json', 'tokenizer.json']);
    expect(readdirSync(outputDir).sort()).toEqual(['config.json', GGUF, 'tokenizer.json'].sort());
  });

  it('records full scope and sidecar provenance with --complete', async () => {
    // The wizard's selection IS the prescribed complete model: a partial
    // marker never reads as installed downstream, so its update affordance
    // (and the repair job behind it) would be unreachable. The weight is NOT
    // pre-created here: the marker is written on a run that actually
    // downloads (a complete-on-disk dir short-circuits instead) and only when
    // the primary revision resolved.
    hub.shas[PRIMARY] = 'a'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    await run([
      '-m',
      PRIMARY,
      '-o',
      outputDir,
      '-g',
      '*UD-Q4_K_XL*',
      '--assets-repo',
      ASSETS,
      '--complete',
      '--cache-dir',
      cacheDir,
    ]);

    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      scope?: string;
      assetsRepo?: string;
      assetsRevision?: string;
      files: string[];
    };
    expect(marker.scope).toBe('full');
    // The sidecars are part of the install: deleting one later must invalidate
    // the marker, which is only possible when the marker LISTS it.
    expect(marker.files).toEqual(expect.arrayContaining([GGUF, 'config.json', 'tokenizer.json']));
    // Both revisions were resolvable: the sidecar source is pinned so update
    // discovery can compare it later.
    expect(marker.assetsRepo).toBe(ASSETS);
    expect(marker.assetsRevision).toBe('b'.repeat(40));
  });

  it('records partial scope without --complete', async () => {
    hub.shas[PRIMARY] = 'a'.repeat(40);
    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--cache-dir', cacheDir]);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      scope?: string;
    };
    expect(marker.scope).toBe('partial');
  });

  it('does not certify a selection whose only GGUF is a companion', async () => {
    // The Gemma entry globs mmproj by name. A partial upstream upload can leave
    // only the projector (plus config.json): the files land, but no completion
    // marker may be written — a directory with no model weights is not an
    // install, and the dashboard would otherwise present it as one.
    hub.manifests[PRIMARY] = [
      { type: 'file', path: 'mmproj-BF16.gguf', size: 44 },
      { type: 'file', path: 'config.json', size: 12 },
    ];
    // The revision IS resolvable, so a marker WOULD be written if this
    // selection were (wrongly) certified — otherwise the assertion below
    // passes for the unrelated legacy reason.
    hub.shas[PRIMARY] = 'a'.repeat(40);
    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*.gguf', '-g', 'config.json', '--cache-dir', cacheDir]);

    expect(existsSync(join(outputDir, 'mmproj-BF16.gguf'))).toBe(true);
    expect(existsSync(join(outputDir, 'config.json'))).toBe(true);
    expect(existsSync(join(outputDir, '.mlx-download-complete.json'))).toBe(false);
  });

  it('keeps recorded sidecars through a no-glob prune (and still prunes the foreign stale one)', async () => {
    // A no-glob run prunes marker entries absent from the PRIMARY repo's tree.
    // Sidecars were never in that tree, so pruning them here would delete the
    // tool-calling files this very run just verified — while a genuine stale
    // primary artifact must still go.
    hub.shas[PRIMARY] = 'a'.repeat(40);
    const marker = {
      repo: PRIMARY,
      revision: 'a'.repeat(40),
      files: [GGUF, 'config.json', 'tokenizer.json', 'stale-old-quant.gguf'],
      scope: 'full',
      assetsRepo: ASSETS,
      assetsRevision: 'b'.repeat(40),
      completedAt: new Date().toISOString(),
    };
    writeFileSync(join(outputDir, '.mlx-download-complete.json'), JSON.stringify(marker));
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));
    writeFileSync(join(outputDir, 'stale-old-quant.gguf'), 'x'.repeat(7));

    await run(['-m', PRIMARY, '-o', outputDir, '--cache-dir', cacheDir]);

    expect(existsSync(join(outputDir, 'tokenizer.json'))).toBe(true);
    const written = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      files: string[];
      assetsRepo?: string;
      assetsRevision?: string;
    };
    expect(written.files).toContain('tokenizer.json');
    // Provenance survives a run that passed no --assets-repo: it touched
    // nothing there, and erasing the pair would disable update discovery.
    expect(written.assetsRepo).toBe(ASSETS);
    expect(written.assetsRevision).toBe('b'.repeat(40));
    // The stale primary artifact is still pruned.
    expect(existsSync(join(outputDir, 'stale-old-quant.gguf'))).toBe(false);
    expect(written.files).not.toContain('stale-old-quant.gguf');
  });

  it('claims the new revision for a --complete update, unlike a plain glob run', async () => {
    // A glob run deliberately under-claims the revision (its marker union can
    // carry files it did not verify). A --complete run's selection IS the
    // prescribed model, so under-claiming leaves the dashboard offering an
    // update the CLI already applied — forever.
    const seed = (revision: string) => {
      writeFileSync(
        join(outputDir, '.mlx-download-complete.json'),
        JSON.stringify({
          repo: PRIMARY,
          revision,
          files: [GGUF, 'config.json', 'tokenizer.json'],
          scope: 'full',
          assetsRepo: ASSETS,
          assetsRevision: 'b'.repeat(40),
          completedAt: new Date().toISOString(),
        }),
      );
      writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
      writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
      writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));
    };
    hub.shas[PRIMARY] = 'c'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    seed('a'.repeat(40));

    await run([
      '-m',
      PRIMARY,
      '-o',
      outputDir,
      '-g',
      '*UD-Q4_K_XL*',
      '--assets-repo',
      ASSETS,
      '--complete',
      '--cache-dir',
      cacheDir,
    ]);
    const claimed = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      revision: string;
    };
    expect(claimed.revision).toBe('c'.repeat(40));

    // Control: WITHOUT --complete the same run keeps the conservative
    // under-claim, so the wizard's flag is what changes this, not the globs.
    seed('a'.repeat(40));
    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--assets-repo', ASSETS, '--cache-dir', cacheDir]);
    const conservative = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      revision: string;
    };
    expect(conservative.revision).toBe('a'.repeat(40));
  });

  it('prunes a sidecar the assets manifest no longer supplies', async () => {
    // The blanket exemption must apply only when the assets manifest was NOT
    // consulted: with it consulted, a candidate the listing dropped has to go —
    // otherwise the marker advances past a file that is still on disk and the
    // runtime keeps consuming a sidecar upstream removed.
    hub.shas[PRIMARY] = 'a'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    hub.manifests[ASSETS] = [{ type: 'file', path: 'config.json', size: 12 }]; // tokenizer.json dropped upstream
    writeFileSync(
      join(outputDir, '.mlx-download-complete.json'),
      JSON.stringify({
        repo: PRIMARY,
        revision: 'a'.repeat(40),
        files: [GGUF, 'config.json', 'tokenizer.json'],
        scope: 'full',
        assetsRepo: ASSETS,
        assetsRevision: 'b'.repeat(40),
        completedAt: new Date().toISOString(),
      }),
    );
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));

    await run([
      '-m',
      PRIMARY,
      '-o',
      outputDir,
      '-g',
      '*UD-Q4_K_XL*',
      '--assets-repo',
      ASSETS,
      '--complete',
      '--cache-dir',
      cacheDir,
    ]);

    expect(existsSync(join(outputDir, 'tokenizer.json'))).toBe(false);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      files: string[];
    };
    expect(marker.files).not.toContain('tokenizer.json');
    expect(marker.files).toContain('config.json');
  });

  it('persists sidecar provenance on a repair-then-return path', async () => {
    // The repair paths fetch/verify sidecars and then RETURN without the
    // download loop. Without re-finalizing, the files just verified stay
    // unlisted (a deletion could not invalidate the marker) and an advanced
    // assets revision stays unpinned — the badge that can never clear. The
    // legacy no-SHA path is the reachable one here: the marker-current
    // short-circuit additionally demands a converted (safetensors) shape,
    // which a GGUF install never satisfies.
    hub.shas[ASSETS] = 'b'.repeat(40); // PRIMARY stays unresolved -> legacy path
    writeFileSync(
      join(outputDir, '.mlx-download-complete.json'),
      JSON.stringify({
        repo: PRIMARY,
        revision: 'a'.repeat(40),
        files: ['model.safetensors', 'config.json'],
        scope: 'full',
        completedAt: new Date().toISOString(),
      }),
    );
    writeFileSync(join(outputDir, 'model.safetensors'), 'x'.repeat(64));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));

    await run(['-m', PRIMARY, '-o', outputDir, '--assets-repo', ASSETS, '--cache-dir', cacheDir]);

    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      files: string[];
      assetsRepo?: string;
      assetsRevision?: string;
    };
    // The tokenizer the repair verified is now LISTED, and the source pinned —
    // including the case where --assets-repo was added to an older install.
    expect(marker.files).toEqual(expect.arrayContaining(['model.safetensors', 'config.json', 'tokenizer.json']));
    expect(marker.assetsRepo).toBe(ASSETS);
    expect(marker.assetsRevision).toBe('b'.repeat(40));
  });

  it('refuses to prune away the last config on the sync path', async () => {
    // A weight-only GGUF repo relies on the assets repo for config.json. When
    // upstream drops it and this sync would prune the installed copy, the run
    // must fail with everything unchanged — the alternative is certifying a
    // directory nothing can load.
    hub.shas[PRIMARY] = 'c'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    hub.manifests[ASSETS] = [{ type: 'file', path: 'tokenizer.json', size: 20 }]; // config.json gone upstream
    const seeded = {
      repo: PRIMARY,
      revision: 'a'.repeat(40),
      files: [GGUF, 'config.json', 'tokenizer.json'],
      scope: 'full',
      assetsRepo: ASSETS,
      assetsRevision: 'b'.repeat(40),
      completedAt: new Date().toISOString(),
    };
    writeFileSync(join(outputDir, '.mlx-download-complete.json'), JSON.stringify(seeded));
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));

    await expect(
      run([
        '-m',
        PRIMARY,
        '-o',
        outputDir,
        '-g',
        '*UD-Q4_K_XL*',
        '--assets-repo',
        ASSETS,
        '--complete',
        '--cache-dir',
        cacheDir,
      ]),
    ).rejects.toThrow(/no loadable checkpoint/);

    // Nothing was mutated: the installation still loads, and the marker still
    // lists the config it has.
    expect(existsSync(join(outputDir, 'config.json'))).toBe(true);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      files: string[];
    };
    expect(marker.files).toContain('config.json');
    // The run downgrades the marker to `partial` BEFORE downloading, so the
    // refusal must restore what it found: a `partial` scope would flip
    // isModelInstalled false and strand the install with no update path.
    expect(marker).toEqual(seeded);
  });

  it('prunes an unselected variant during a complete sync (prescription, not the whole tree)', async () => {
    // Full-run semantics alone compare old entries against the WHOLE remote
    // tree, so another quant still published upstream would be retained and
    // carried into the marker while the new revision is claimed — a file this
    // run neither selected nor verified, which discovery can expose as a
    // loadable model.
    hub.shas[PRIMARY] = 'c'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    hub.manifests[PRIMARY] = [
      { type: 'file', path: GGUF, size: 300 },
      { type: 'file', path: 'Tiny-UD-Q8_K_XL.gguf', size: 900 }, // published, NOT in the glob prescription
      { type: 'file', path: 'config.json', size: 12 },
    ];
    hub.manifests[ASSETS] = [{ type: 'file', path: 'tokenizer.json', size: 20 }];
    writeFileSync(
      join(outputDir, '.mlx-download-complete.json'),
      JSON.stringify({
        repo: PRIMARY,
        revision: 'a'.repeat(40),
        files: [GGUF, 'Tiny-UD-Q8_K_XL.gguf', 'config.json', 'tokenizer.json'],
        scope: 'full',
        assetsRepo: ASSETS,
        assetsRevision: 'b'.repeat(40),
        completedAt: new Date().toISOString(),
      }),
    );
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'Tiny-UD-Q8_K_XL.gguf'), 'x'.repeat(900));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));

    await run([
      '-m',
      PRIMARY,
      '-o',
      outputDir,
      '-g',
      '*UD-Q4_K_XL*',
      '--assets-repo',
      ASSETS,
      '--complete',
      '--cache-dir',
      cacheDir,
    ]);

    expect(existsSync(join(outputDir, 'Tiny-UD-Q8_K_XL.gguf'))).toBe(false);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      files: string[];
      revision: string;
    };
    expect(marker.files).not.toContain('Tiny-UD-Q8_K_XL.gguf');
    expect(marker.revision).toBe('c'.repeat(40));
  });

  it('prunes a removed candidate on the repair-return path', async () => {
    // Reachable with an unresolvable primary revision and a complete-looking
    // glob set: the repair runs, and preserving every old marker entry would
    // keep a removed tokenizer on disk and listed while the new assets
    // revision is recorded — reported current forever.
    hub.shas[ASSETS] = 'b'.repeat(40); // PRIMARY unresolved -> legacy repair path
    hub.manifests[ASSETS] = [{ type: 'file', path: 'config.json', size: 12 }]; // tokenizer.json gone
    writeFileSync(
      join(outputDir, '.mlx-download-complete.json'),
      JSON.stringify({
        repo: PRIMARY,
        revision: 'a'.repeat(40),
        files: [GGUF, 'config.json', 'tokenizer.json'],
        scope: 'full',
        assetsRepo: ASSETS,
        assetsRevision: 'a'.repeat(40),
        completedAt: new Date().toISOString(),
      }),
    );
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));

    await run([
      '-m',
      PRIMARY,
      '-o',
      outputDir,
      '-g',
      '*UD-Q4_K_XL*',
      '--assets-repo',
      ASSETS,
      '--complete',
      '--cache-dir',
      cacheDir,
    ]);

    expect(existsSync(join(outputDir, 'tokenizer.json'))).toBe(false);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      files: string[];
      assetsRevision?: string;
    };
    expect(marker.files).not.toContain('tokenizer.json');
    expect(marker.assetsRevision).toBe('b'.repeat(40));
  });

  it('does not certify a selection whose only GGUF is an MTP sidecar', async () => {
    // MTP weights ship beside a target and nothing pairs a standalone GGUF MTP
    // file — the same rule as a projector, so the same outcome: files land, no
    // completion marker.
    hub.manifests[PRIMARY] = [{ type: 'file', path: 'mtp-Qwen3.8-27B-Q4_0.gguf', size: 44 }];
    hub.shas[PRIMARY] = 'a'.repeat(40);
    await run(['-m', PRIMARY, '-o', outputDir, '-g', 'mtp-*.gguf', '--cache-dir', cacheDir]);

    expect(existsSync(join(outputDir, 'mtp-Qwen3.8-27B-Q4_0.gguf'))).toBe(true);
    expect(existsSync(join(outputDir, '.mlx-download-complete.json'))).toBe(false);
  });

  it('refuses to certify a fresh install whose selection has no config.json', async () => {
    // Weight-only primary repo + an assets manifest that omits config.json:
    // there is no previous marker and nothing to prune, so the only thing that
    // can catch this is the fresh-install gate. Without it the wizard reports
    // success for a directory nothing can load.
    hub.shas[PRIMARY] = 'a'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    hub.manifests[ASSETS] = [{ type: 'file', path: 'tokenizer.json', size: 20 }]; // no config.json anywhere

    await expect(
      run([
        '-m',
        PRIMARY,
        '-o',
        outputDir,
        '-g',
        '*UD-Q4_K_XL*',
        '--assets-repo',
        ASSETS,
        '--complete',
        '--cache-dir',
        cacheDir,
      ]),
    ).rejects.toThrow(/no config[.]json/);

    // The refusal cannot leave its downloads behind: a markerless directory of
    // loadable weights renders as a PRESENT but foreign install — a disabled
    // card the dashboard cannot repair. Nothing predated this run, so the
    // honest "nothing was published" removes the directory entirely.
    expect(existsSync(outputDir)).toBe(false);
  });

  it('refuses to install sidecars when their revision cannot be pinned', async () => {
    // Installing against mutable `main` publishes sidecars with no provenance:
    // installed, and never updatable. A retryable failure is the honest answer.
    hub.shas[PRIMARY] = 'a'.repeat(40); // ASSETS deliberately unresolved

    await expect(
      run([
        '-m',
        PRIMARY,
        '-o',
        outputDir,
        '-g',
        '*UD-Q4_K_XL*',
        '--assets-repo',
        ASSETS,
        '--complete',
        '--cache-dir',
        cacheDir,
      ]),
    ).rejects.toThrow(/Could not resolve the latest revision/);

    expect(existsSync(join(outputDir, '.mlx-download-complete.json'))).toBe(false);
  });

  it('does not touch the assets repo when the flag is absent', async () => {
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));

    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--cache-dir', cacheDir]);

    expect(hub.listedRepos).not.toContain(ASSETS);
    expect(hub.downloaded).toEqual([]);
    expect(readdirSync(outputDir)).toEqual([GGUF]);
  });

  it('refuses an update BEFORE any bytes land when the prune would break the install', async () => {
    // Regression: the prune guard ran AFTER the in-place download loop, so its
    // "The installed directory is unchanged" refusal was a lie — the new
    // weight had already been copied in next to the old one, and the restored
    // marker then labeled a mixed-revision directory with the old snapshot's
    // name. Every guard input is manifest-derived, so the refusal must fire
    // before the first write.
    const NEW = 'Tiny-UD-Q5_K_XL.gguf';
    hub.shas[PRIMARY] = 'c'.repeat(40);
    hub.shas[ASSETS] = 'b'.repeat(40);
    hub.manifests[PRIMARY] = [{ type: 'file', path: NEW, size: 900 }]; // renamed weight, old one gone upstream
    hub.manifests[ASSETS] = [{ type: 'file', path: 'tokenizer.json', size: 20 }]; // config.json gone upstream
    const seeded = {
      repo: PRIMARY,
      revision: 'a'.repeat(40),
      files: [GGUF, 'config.json', 'tokenizer.json'],
      scope: 'full',
      assetsRepo: ASSETS,
      assetsRevision: 'b'.repeat(40),
      completedAt: new Date().toISOString(),
    };
    writeFileSync(join(outputDir, '.mlx-download-complete.json'), JSON.stringify(seeded));
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'tokenizer.json'), 'x'.repeat(20));

    await expect(
      run(['-m', PRIMARY, '-o', outputDir, '--assets-repo', ASSETS, '--cache-dir', cacheDir]),
    ).rejects.toThrow(/no loadable checkpoint/);

    // "Unchanged" is now literal: the new weight never landed, nothing was
    // fetched at all, the old files keep their bytes, and the marker the run
    // found is the marker it left.
    expect(existsSync(join(outputDir, NEW))).toBe(false);
    expect(hub.downloaded).toEqual([]);
    expect(readFileSync(join(outputDir, GGUF), 'utf8')).toBe('x'.repeat(300));
    expect(existsSync(join(outputDir, 'config.json'))).toBe(true);
    expect(JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8'))).toEqual(seeded);
  });

  it('refuses a repair-path sidecar removal BEFORE fetching anything', async () => {
    // Same ordering bug on the early-return path: the repair fetched sidecars
    // into the install before checking whether the removals it was about to
    // apply would leave the directory unloadable. The plan is computable
    // without a single write, so the refusal must come first.
    hub.shas[ASSETS] = 'b'.repeat(40); // PRIMARY unresolved -> legacy repair path
    // config.json and chat_template.jinja dropped upstream; only the tokenizer remains.
    hub.manifests[ASSETS] = [{ type: 'file', path: 'tokenizer.json', size: 20 }];
    const seeded = {
      repo: PRIMARY,
      revision: 'a'.repeat(40),
      files: [GGUF, 'config.json', 'chat_template.jinja'],
      scope: 'full',
      assetsRepo: ASSETS,
      assetsRevision: 'a'.repeat(40),
      completedAt: new Date().toISOString(),
    };
    writeFileSync(join(outputDir, '.mlx-download-complete.json'), JSON.stringify(seeded));
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    writeFileSync(join(outputDir, 'chat_template.jinja'), 'x'.repeat(9));

    await expect(
      run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--assets-repo', ASSETS, '--cache-dir', cacheDir]),
    ).rejects.toThrow(/no loadable checkpoint/);

    // The refusal precedes the fetch: tokenizer.json was planned but never
    // downloaded or written, and the stale candidates it would have replaced
    // are still on disk.
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(outputDir, 'tokenizer.json'))).toBe(false);
    expect(existsSync(join(outputDir, 'config.json'))).toBe(true);
    expect(existsSync(join(outputDir, 'chat_template.jinja'))).toBe(true);
    expect(JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8'))).toEqual(seeded);
  });

  it('skips the repair entirely when the assets revision cannot be pinned', async () => {
    // `planAssetSidecars` runs with `requireRevision: false` on this path, so
    // a transient resolution failure used to fall through to a fetch from
    // MUTABLE `main` — overwriting live install files — while the marker
    // write below kept the OLD pinned `assetsRevision`: provenance that lies
    // about where the bytes came from (and a branch move mid-run mixes
    // revisions). The repair must skip rather than mutate the install
    // unpinned; the seeded marker records an older sidecar revision and has
    // to stay byte-identical.
    //
    // Both revisions stay unresolved — `hub.shas` has no entry, `modelInfo`
    // returns no sha — so the run takes the legacy "already downloaded"
    // early-return path, the one the repair runs in front of.
    const seeded = {
      repo: PRIMARY,
      revision: 'a'.repeat(40),
      files: ['model.safetensors', 'config.json', 'tokenizer.json'],
      scope: 'full',
      assetsRepo: ASSETS,
      assetsRevision: 'b'.repeat(40),
      completedAt: new Date().toISOString(),
    };
    writeFileSync(join(outputDir, '.mlx-download-complete.json'), JSON.stringify(seeded));
    writeFileSync(join(outputDir, 'model.safetensors'), 'x'.repeat(64));
    writeFileSync(join(outputDir, 'config.json'), 'x'.repeat(12));
    // tokenizer.json is in the marker AND still offered by the assets
    // manifest, but missing on disk — exactly what the repair would fetch.

    await run(['-m', PRIMARY, '-o', outputDir, '--assets-repo', ASSETS, '--cache-dir', cacheDir]);

    // The plan was still consulted (the skip happens after planning, not
    // instead of it), but nothing was fetched and nothing was written: the
    // missing sidecar stays missing and the marker is byte-identical.
    expect(hub.listedRepos).toContain(ASSETS);
    expect(hub.downloaded).toEqual([]);
    expect(existsSync(join(outputDir, 'tokenizer.json'))).toBe(false);
    expect(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')).toBe(JSON.stringify(seeded));
  });
});
