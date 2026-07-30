import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  DOWNLOAD_COMPLETE_MARKER,
  markCompletionPartial,
  readCompletion,
  writeCompletion,
  type DownloadCompletion,
} from '../../packages/cli/src/commands/download-marker.js';
import {
  DOWNLOAD_COMPLETE_MARKER as DASHBOARD_MARKER,
  readCompletion as dashboardReadCompletion,
} from '../../packages/dashboard/src/models.js';

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

  it('marks an existing completion partial without losing its ownership identity', async () => {
    const partial = markCompletionPartial(completion);
    await writeCompletion(dir, partial);
    expect(await readCompletion(dir)).toEqual({ ...completion, scope: 'partial' });
    expect(dashboardReadCompletion(dir)).toEqual({ ...completion, scope: 'partial' });
  });
});
