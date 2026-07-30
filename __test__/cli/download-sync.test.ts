import { mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { DownloadCompletion } from '../../packages/cli/src/commands/download-marker.js';
import {
  assertCompletionRepoCompatible,
  buildMarkerFiles,
  canShortCircuitFullRun,
  computeLegacyWeightPruneList,
  computePruneList,
  fileUpToDate,
  isCompletionCurrent,
  markerRevisionToClaim,
  sameRepoCompletion,
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

  it('rejects a partial/in-progress marker even when repo and revision match', () => {
    expect(isCompletionCurrent({ ...completion, scope: 'partial' }, 'org/model', SHA)).toBe(false);
  });
});

describe('canShortCircuitFullRun', () => {
  let dir: string;
  const completion: DownloadCompletion = {
    repo: 'org/model',
    revision: SHA,
    files: ['config.json'],
    completedAt: '2026-07-29T00:00:00.000Z',
  };

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-shortcircuit-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('denies when the local shape is incomplete even though every marker file exists (mutation: a glob-selection marker must not satisfy a full run)', () => {
    // A `--glob "*.json"` run wrote a jsons-only marker; every marker file is
    // on disk but the model has no weights. Dropping the shape gate makes a
    // later full run print "already up to date" and never download weights.
    writeFileSync(join(dir, 'config.json'), '{}');
    expect(canShortCircuitFullRun(completion, dir, false)).toBe(false);
  });

  it('denies when a marker file is missing even though the shape is complete', () => {
    expect(canShortCircuitFullRun(completion, dir, true)).toBe(false);
  });

  it('allows only when the shape is complete AND every marker file exists', () => {
    writeFileSync(join(dir, 'config.json'), '{}');
    expect(canShortCircuitFullRun(completion, dir, true)).toBe(true);
  });

  it('denies a partial glob marker even when its files happen to look loadable', () => {
    writeFileSync(join(dir, 'config.json'), '{}');
    expect(canShortCircuitFullRun({ ...completion, scope: 'partial' }, dir, true)).toBe(false);
  });
});

describe('sameRepoCompletion', () => {
  const completion: DownloadCompletion = {
    repo: 'unsloth/model',
    revision: SHA,
    files: ['model.safetensors'],
    completedAt: '2026-07-29T00:00:00.000Z',
  };

  it('nulls a FOREIGN-repo marker (mutation: slug collision lets repo B prune repo A files)', () => {
    // `unsloth/model` and `bartowski/model` share the default output dir slug;
    // repo B's sync must treat repo A's marker as no marker at all.
    expect(sameRepoCompletion(completion, 'bartowski/model')).toBeNull();
  });

  it('passes a same-repo marker through unchanged, and null through as null', () => {
    expect(sameRepoCompletion(completion, 'unsloth/model')).toBe(completion);
    expect(sameRepoCompletion(null, 'unsloth/model')).toBeNull();
  });
});

describe('assertCompletionRepoCompatible', () => {
  const completion: DownloadCompletion = {
    repo: 'unsloth/model',
    revision: SHA,
    files: ['model.safetensors'],
    completedAt: '2026-07-29T00:00:00.000Z',
  };

  it('accepts a marker for the requested repo and a marker-less directory', () => {
    expect(() => assertCompletionRepoCompatible(completion, 'unsloth/model', '/models/model')).not.toThrow();
    expect(() => assertCompletionRepoCompatible(null, 'unsloth/model', '/models/model')).not.toThrow();
  });

  it('refuses a same-basename directory owned by another repo', () => {
    expect(() => assertCompletionRepoCompatible(completion, 'bartowski/model', '/models/model')).toThrow(
      /owned by "unsloth\/model".*--output/,
    );
  });
});

describe('markerRevisionToClaim', () => {
  const OLD = 'a'.repeat(40);
  const NEW = 'b'.repeat(40);
  const previous: DownloadCompletion = {
    repo: 'org/model',
    revision: OLD,
    files: ['model.safetensors'],
    completedAt: '2026-07-29T00:00:00.000Z',
  };

  it('glob sync at a CHANGED revision keeps the OLD revision (mutation: stamping the new sha launders unverified files into a current marker)', () => {
    // The glob run verified only its selection; the union still carries
    // rev-OLD files. Claiming NEW would let the next full run short-circuit
    // forever with stale weights; under-claiming keeps it syncing.
    expect(markerRevisionToClaim(previous, NEW, true)).toBe(OLD);
  });

  it('no-glob sync claims the remote revision — it verified everything it records', () => {
    expect(markerRevisionToClaim(previous, NEW, false)).toBe(NEW);
  });

  it('glob run with NO previous marker claims the remote revision (selection-only marker)', () => {
    expect(markerRevisionToClaim(null, NEW, true)).toBe(NEW);
  });

  it('glob run at an UNCHANGED revision claims it', () => {
    expect(markerRevisionToClaim(previous, OLD, true)).toBe(OLD);
  });
});

describe('computePruneList', () => {
  it('returns only old-marker files gone from the REMOTE (not merely unselected)', () => {
    const previous = ['config.json', 'old-shard.safetensors', 'kept.safetensors'];
    const remote = ['config.json', 'kept.safetensors', 'new-shard.safetensors'];
    expect(computePruneList(previous, remote, '/out', false)).toEqual(['old-shard.safetensors']);
  });

  it('never lists files absent from the old marker (mutation: pruning by disk scan would delete user files)', () => {
    expect(computePruneList([], ['config.json'], '/out', false)).toEqual([]);
  });

  it('drops traversal and absolute entries instead of deleting outside outputDir', () => {
    const previous = ['../escape.txt', '/etc/passwd', 'sub/../../escape2.txt', ''];
    expect(computePruneList(previous, [], '/out', false)).toEqual([]);
  });

  it('prunes nested marker entries proven absent by the recursive remote listing', () => {
    expect(computePruneList(['mtp/weights.safetensors'], [], '/out', false)).toEqual(['mtp/weights.safetensors']);
  });

  it('never prunes during a narrow glob sync, even when a previous weight disappeared upstream', () => {
    const previous = ['config.json', 'model-old.safetensors'];
    const remote = ['config.json', 'model-new.safetensors', 'tokenizer.json'];
    expect(computePruneList(previous, remote, '/out', true)).toEqual([]);
  });
});

describe('computeLegacyWeightPruneList', () => {
  it('removes a superseded single-file weight before certifying a new sharded layout', () => {
    const local = ['config.json', 'model.safetensors', 'adapter.safetensors'];
    const remote = [
      'config.json',
      'model.safetensors.index.json',
      'model-00001-of-00002.safetensors',
      'model-00002-of-00002.safetensors',
    ];
    expect(computeLegacyWeightPruneList(local, remote, '/out', false)).toEqual(['model.safetensors']);
  });

  it('removes superseded shards and their index before certifying a single-file layout', () => {
    const local = [
      'model.safetensors.index.json',
      'model-00001-of-00002.safetensors',
      'model-00002-of-00002.safetensors',
    ];
    expect(computeLegacyWeightPruneList(local, ['config.json', 'model.safetensors'], '/out', false)).toEqual(local);
  });

  it('does not delete adapters, user files, glob-unselected files, or files for non-safetensors repos', () => {
    const local = ['adapter.safetensors', 'README.md', 'model.safetensors'];
    expect(computeLegacyWeightPruneList(local, ['config.json', 'model.gguf'], '/out', false)).toEqual([]);
    expect(
      computeLegacyWeightPruneList(local, ['config.json', 'model-00001-of-00001.safetensors'], '/out', true),
    ).toEqual([]);
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
    expect(buildMarkerFiles(previous, remote, selected, dir, false)).toEqual([
      'config.json',
      'model-Q4.gguf',
      'model-Q8.gguf',
    ]);
  });

  it('with no previous marker returns just the sorted selection', () => {
    expect(buildMarkerFiles(null, ['b.json', 'a.json'], ['b.json', 'a.json'], dir, false)).toEqual([
      'a.json',
      'b.json',
    ]);
  });

  it('keeps a recursively listed nested file after verifying it at the new revision', () => {
    mkdirSync(join(dir, 'sub'));
    writeFileSync(join(dir, 'sub', 'nested.json'), 'x');
    const previous: DownloadCompletion = {
      repo: 'org/model',
      revision: SHA,
      files: ['sub/nested.json'],
      completedAt: '2026-07-29T00:00:00.000Z',
    };
    expect(buildMarkerFiles(previous, ['config.json', 'sub/nested.json'], ['config.json'], dir, false)).toEqual([
      'config.json',
      'sub/nested.json',
    ]);
  });

  it('drops a nested previous file that is absent upstream even if stale bytes remain on disk', () => {
    mkdirSync(join(dir, 'sub'));
    writeFileSync(join(dir, 'sub', 'nested.json'), 'stale');
    const previous: DownloadCompletion = {
      repo: 'org/model',
      revision: SHA,
      files: ['sub/nested.json'],
      completedAt: '2026-07-29T00:00:00.000Z',
    };
    expect(buildMarkerFiles(previous, ['config.json'], ['config.json'], dir, false)).toEqual(['config.json']);
  });

  it('keeps disappeared previous weights in a glob marker so the next full sync can prune them safely', () => {
    writeFileSync(join(dir, 'model-old.safetensors'), 'x');
    const previous: DownloadCompletion = {
      repo: 'org/model',
      revision: SHA,
      files: ['model-old.safetensors'],
      completedAt: '2026-07-29T00:00:00.000Z',
    };
    expect(buildMarkerFiles(previous, ['config.json', 'model-new.safetensors'], ['config.json'], dir, true)).toEqual([
      'config.json',
      'model-old.safetensors',
    ]);
  });
});
