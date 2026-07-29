import { mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { DownloadCompletion } from '../../packages/cli/src/commands/download-marker.js';
import {
  buildMarkerFiles,
  computePruneList,
  fileUpToDate,
  isCompletionCurrent,
} from '../../packages/cli/src/commands/download-sync.js';

const SHA = 'b'.repeat(40);

function entry(overrides: Partial<ListFileEntry> & { path: string; size: number }): ListFileEntry {
  return { type: 'file', ...overrides };
}

describe('fileUpToDate', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-sync-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('false when the local file is missing', async () => {
    expect(await fileUpToDate(join(dir, 'model.safetensors'), entry({ path: 'model.safetensors', size: 4 }))).toBe(
      false,
    );
  });

  it('false when sizes differ', async () => {
    const p = join(dir, 'model.safetensors');
    writeFileSync(p, 'abcd');
    expect(await fileUpToDate(p, entry({ path: 'model.safetensors', size: 999 }))).toBe(false);
  });

  it('false for SAME-SIZE different content vs lfs.oid — the mutation a size-only check misses', async () => {
    const p = join(dir, 'model.safetensors');
    writeFileSync(p, 'xxxx'); // same size as 'abcd', different bytes
    const size = statSync(p).size;
    // sha256("abcd")
    const oidOfAbcd = '88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589';
    expect(
      await fileUpToDate(
        p,
        entry({ path: 'model.safetensors', size, lfs: { oid: oidOfAbcd, size, pointerSize: 130 } }),
      ),
    ).toBe(false);
  });

  it('true when content matches lfs.oid (sha256)', async () => {
    const p = join(dir, 'model.safetensors');
    writeFileSync(p, 'abcd');
    const oidOfAbcd = '88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589';
    expect(
      await fileUpToDate(
        p,
        entry({ path: 'model.safetensors', size: 4, lfs: { oid: oidOfAbcd, size: 4, pointerSize: 130 } }),
      ),
    ).toBe(true);
  });

  it('verifies non-LFS files by git blob sha1', async () => {
    const p = join(dir, 'config.json');
    writeFileSync(p, 'abcd');
    // git hash-object of "abcd" = sha1("blob 4\0abcd"), verified via `printf 'abcd' | git hash-object --stdin`
    const gitOid = '85df50785d62d3b05ab03d9cbf7e4a0b49449730';
    expect(await fileUpToDate(p, entry({ path: 'config.json', size: 4, oid: gitOid }))).toBe(true);
    expect(await fileUpToDate(p, entry({ path: 'config.json', size: 4, oid: 'f'.repeat(40) }))).toBe(false);
  });

  it('falls back to size-only when the entry has no usable hash', async () => {
    const p = join(dir, 'weights.bin');
    writeFileSync(p, 'abcd');
    expect(await fileUpToDate(p, entry({ path: 'weights.bin', size: 4 }))).toBe(true);
  });
});

describe('isCompletionCurrent', () => {
  const completion: DownloadCompletion = {
    repo: 'org/model',
    revision: SHA,
    files: ['config.json'],
    completedAt: '2026-07-29T00:00:00.000Z',
  };

  it('true only when repo AND revision match', () => {
    expect(isCompletionCurrent(completion, 'org/model', SHA)).toBe(true);
    expect(isCompletionCurrent(completion, 'org/other', SHA)).toBe(false);
    expect(isCompletionCurrent(completion, 'org/model', 'c'.repeat(40))).toBe(false);
    expect(isCompletionCurrent(null, 'org/model', SHA)).toBe(false);
  });
});

describe('computePruneList', () => {
  it('returns only old-marker files gone from the REMOTE (not merely unselected)', () => {
    const previous = ['config.json', 'old-shard.safetensors', 'kept.safetensors'];
    const remote = ['config.json', 'kept.safetensors', 'new-shard.safetensors'];
    expect(computePruneList(previous, remote, '/out')).toEqual(['old-shard.safetensors']);
  });

  it('never lists files absent from the old marker (mutation: pruning by disk scan would delete user files)', () => {
    expect(computePruneList([], ['config.json'], '/out')).toEqual([]);
  });

  it('drops traversal and absolute entries instead of deleting outside outputDir', () => {
    const previous = ['../escape.txt', '/etc/passwd', 'sub/../../escape2.txt', ''];
    expect(computePruneList(previous, [], '/out')).toEqual([]);
  });

  it('keeps a legitimate nested path', () => {
    expect(computePruneList(['sub/gone.gguf'], [], '/out')).toEqual(['sub/gone.gguf']);
  });
});

describe('buildMarkerFiles', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-marker-files-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('unions the selection with still-remote, still-on-disk previous files, sorted', () => {
    writeFileSync(join(dir, 'model-Q8.gguf'), 'x'); // previous file, still on disk
    const previous: DownloadCompletion = {
      repo: 'org/model',
      revision: SHA,
      files: ['model-Q8.gguf', 'gone-upstream.gguf', 'deleted-locally.gguf'],
      completedAt: '2026-07-29T00:00:00.000Z',
    };
    const remote = ['model-Q8.gguf', 'model-Q4.gguf', 'config.json', 'deleted-locally.gguf'];
    const selected = ['model-Q4.gguf', 'config.json'];
    expect(buildMarkerFiles(previous, remote, selected, dir)).toEqual([
      'config.json',
      'model-Q4.gguf',
      'model-Q8.gguf',
    ]);
  });

  it('with no previous marker returns just the sorted selection', () => {
    expect(buildMarkerFiles(null, ['b.json', 'a.json'], ['b.json', 'a.json'], dir)).toEqual(['a.json', 'b.json']);
  });
});
