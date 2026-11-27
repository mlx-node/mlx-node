import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vite-plus/test';

import {
  DOWNLOAD_COMPLETE_MARKER,
  markCompletionPartial,
  readCompletion,
  writeCompletion,
  type DownloadCompletion,
} from '../../packages/cli/src/commands/download-marker.js';
import { run } from '../../packages/cli/src/commands/download-model.js';
import {
  DOWNLOAD_COMPLETE_MARKER as DASHBOARD_MARKER,
  readCompletion as dashboardReadCompletion,
} from '../../packages/dashboard/src/models.js';

const hub = vi.hoisted(() => ({
  manifest: [] as Array<{ type: 'file'; path: string; size: number }>,
  sha: '',
  snapshotDir: '',
}));

// Only safetensors files appear in this suite; guard against unexpected GGUF addon hits.
vi.mock('../../packages/cli/src/commands/download-gguf-check.js', () => ({
  checkGgufTargets: async ({ paths }: { paths: readonly string[] }) => {
    if (paths.some((path) => path.toLowerCase().endsWith('.gguf'))) {
      throw new Error(`unexpected GGUF target in download-marker test: ${paths.join(', ')}`);
    }
    return { targets: [], headerBytes: 0 };
  },
}));

// Only `run` below reaches the hub: the upstream is `hub.manifest` at `hub.sha`.
vi.mock('@huggingface/hub', () => ({
  downloadFile: async () => new Blob(['{}']),
  modelInfo: async () => ({ sha: hub.sha }),
  listFiles: async function* () {
    for (const entry of hub.manifest) yield entry;
  },
  downloadFileToCacheDir: async (params: { path: string }) => {
    const size = hub.manifest.find((file) => file.path === params.path)?.size ?? 1;
    const snapshot = join(hub.snapshotDir, params.path);
    writeFileSync(snapshot, params.path === 'config.json' ? '{}'.padEnd(size) : 'x'.repeat(size));
    return snapshot;
  },
}));

describe('download marker', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-marker-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  const completion: DownloadCompletion = {
    repo: 'google/gemma-4-26B-A4B-it',
    revision: 'a'.repeat(40),
    files: ['config.json', 'model.safetensors'],
    completedAt: '2026-07-29T00:00:00.000Z',
  };

  it('readCompletion returns null when the marker file is absent', async () => {
    expect(await readCompletion(dir)).toBeNull();
  });

  it('readCompletion returns null on malformed JSON (mutation: a throw here would crash the gate)', async () => {
    writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), '{not json');
    expect(await readCompletion(dir)).toBeNull();
  });

  it('readCompletion returns null when a field has the wrong type', async () => {
    writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), JSON.stringify({ ...completion, files: ['config.json', 42] }));
    expect(await readCompletion(dir)).toBeNull();
  });

  it('round-trips through writeCompletion/readCompletion and leaves no temp file', async () => {
    await writeCompletion(dir, completion);
    expect(await readCompletion(dir)).toEqual(completion);
    // temp+rename must not leave `.tmp` droppings (mutation: writing the
    // final path directly could tear the marker on a crash)
    expect(readdirSync(dir)).toEqual([DOWNLOAD_COMPLETE_MARKER]);
    expect(existsSync(join(dir, `${DOWNLOAD_COMPLETE_MARKER}.tmp`))).toBe(false);
  });

  it('CONTRACT: filename is identical to the dashboard marker constant', () => {
    expect(DOWNLOAD_COMPLETE_MARKER).toBe(DASHBOARD_MARKER);
  });

  it('CONTRACT: a CLI-written marker is accepted by the dashboard readCompletion', async () => {
    await writeCompletion(dir, completion);
    expect(dashboardReadCompletion(dir)).toEqual(completion);
  });

  it('an adopted record survives the read, the partial mark and the write', async () => {
    const adopted = { 'config.json': { size: 12, mtimeNs: '1763700000123456789' } };
    writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), JSON.stringify({ ...completion, adopted }));
    const read = await readCompletion(dir);
    expect(read).toEqual({ ...completion, adopted });
    await writeCompletion(dir, markCompletionPartial(read!));
    expect(dashboardReadCompletion(dir)).toEqual({ ...completion, scope: 'partial', adopted });
    await writeCompletion(dir, (await readCompletion(dir))!);
    expect(dashboardReadCompletion(dir)?.adopted).toEqual(adopted);
  });

  it('a malformed adopted record is carried as it is', async () => {
    writeFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), JSON.stringify({ ...completion, adopted: 'x' }));
    await writeCompletion(dir, (await readCompletion(dir))!);
    expect(
      (JSON.parse(readFileSync(join(dir, DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as { adopted: unknown }).adopted,
    ).toBe('x');
  });

  it('a sync to a new revision keeps the adopted record in the marker it writes', async () => {
    const repo = 'org/Tiny';
    hub.manifest = [
      { type: 'file', path: 'config.json', size: 12 },
      { type: 'file', path: 'model.safetensors', size: 30 },
    ];
    hub.sha = 'a'.repeat(40);
    hub.snapshotDir = mkdtempSync(join(tmpdir(), 'mlx-marker-snap-'));
    const cacheDir = mkdtempSync(join(tmpdir(), 'mlx-marker-cache-'));
    onTestFinished(() => {
      for (const path of [hub.snapshotDir, cacheDir]) rmSync(path, { recursive: true, force: true });
    });
    writeFileSync(join(dir, 'config.json'), '{}'.padEnd(12));
    writeFileSync(join(dir, 'model.safetensors'), 'x'.repeat(30));
    writeFileSync(join(dir, 'README.md'), '# my notes');
    const adopted = { 'config.json': { size: 12, mtimeNs: '1' }, 'model.safetensors': { size: 30, mtimeNs: '2' } };
    writeFileSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      JSON.stringify({
        repo,
        revision: 'c'.repeat(40),
        files: ['config.json', 'model.safetensors'],
        scope: 'full',
        completedAt: 'x',
        adopted,
      }),
    );
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    onTestFinished(() => log.mockRestore());

    await run(['-m', repo, '-o', dir, '--cache-dir', cacheDir]);

    const written = dashboardReadCompletion(dir);
    expect(written?.revision).toBe('a'.repeat(40));
    expect(written?.adopted).toEqual(adopted);
  });

  it('marks an existing completion partial without losing its ownership identity', async () => {
    const partial = markCompletionPartial(completion);
    await writeCompletion(dir, partial);
    expect(await readCompletion(dir)).toEqual({ ...completion, scope: 'partial' });
    expect(dashboardReadCompletion(dir)).toEqual({ ...completion, scope: 'partial' });
  });
});
