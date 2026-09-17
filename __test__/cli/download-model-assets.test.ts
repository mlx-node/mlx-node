import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
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
    // otherwise return before any fetch.
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));

    await run([
      '-m',
      PRIMARY,
      '-o',
      outputDir,
      '-g',
      '*UD-Q4_K_XL*',
      '--assets-repo',
      ASSETS,
      '--cache-dir',
      cacheDir,
    ]);

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
    };
    expect(marker.scope).toBe('full');
    // Both revisions were resolvable: the sidecar source is pinned so update
    // discovery can compare it later.
    expect(marker.assetsRepo).toBe(ASSETS);
    expect(marker.assetsRevision).toBe('b'.repeat(40));
  });

  it('records no assets provenance when the sidecar revision is unresolved', async () => {
    // Unknown provenance must stay unknown: recording an empty revision would
    // compare unequal to every upstream sha and raise a permanent update badge.
    hub.shas[PRIMARY] = 'a'.repeat(40); // primary resolved, assets NOT
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
      assetsRepo?: string;
      assetsRevision?: string;
    };
    expect(marker.assetsRepo).toBeUndefined();
    expect(marker.assetsRevision).toBeUndefined();
  });

  it('records partial scope without --complete', async () => {
    hub.shas[PRIMARY] = 'a'.repeat(40);
    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--cache-dir', cacheDir]);
    const marker = JSON.parse(readFileSync(join(outputDir, '.mlx-download-complete.json'), 'utf-8')) as {
      scope?: string;
    };
    expect(marker.scope).toBe('partial');
  });

  it('does not touch the assets repo when the flag is absent', async () => {
    writeFileSync(join(outputDir, GGUF), 'x'.repeat(300));

    await run(['-m', PRIMARY, '-o', outputDir, '-g', '*UD-Q4_K_XL*', '--cache-dir', cacheDir]);

    expect(hub.listedRepos).not.toContain(ASSETS);
    expect(hub.downloaded).toEqual([]);
    expect(readdirSync(outputDir)).toEqual([GGUF]);
  });
});
