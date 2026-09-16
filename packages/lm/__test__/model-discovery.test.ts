import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, describe, expect, it } from 'vite-plus/test';

import { discoverLocalChatModels } from '../src/model-discovery.js';

const tmp = mkdtempSync(join(tmpdir(), 'mlx-discovery-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

// chmod 000 yields EACCES only on POSIX for a non-root user.
const canChmod = process.platform !== 'win32' && process.getuid?.() !== 0;

describe('discoverLocalChatModels', () => {
  it('returns [] for a dir that does not exist — nothing is installed', async () => {
    await expect(discoverLocalChatModels(join(tmp, 'absent'))).resolves.toEqual([]);
  });

  it('throws when the dir exists but cannot be scanned — "scan failed" is not "empty"', async () => {
    // ENOTDIR from a regular file stands in for EACCES/EIO: any non-ENOENT
    // readdir failure must propagate. The desktop supervisor treats a
    // confirmed-empty library as permanent (no retries), so misreporting an
    // I/O error as empty would suppress recovery after the error clears.
    const file = join(tmp, 'not-a-dir');
    writeFileSync(file, 'x');
    await expect(discoverLocalChatModels(file)).rejects.toThrow();
  });

  it('treats a non-model directory as a clean skip, not a scan failure', async () => {
    // No config.json → definitively not a model: the entry is skipped and the
    // empty result means "nothing installed", so onEntryFailure stays quiet.
    const dir = join(tmp, 'only-junk');
    mkdirSync(join(dir, 'not-a-model'), { recursive: true });
    const failures: string[] = [];
    await expect(discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).resolves.toEqual(
      [],
    );
    expect(failures).toEqual([]);
  });

  it.skipIf(!canChmod)('reports an entry it could not evaluate via onEntryFailure', async () => {
    // config.json exists but cannot be opened (EACCES): the entry may be a
    // model missing from the result, so the scan is incomplete — exactly the
    // signal that keeps an empty result from being read as permanent.
    const dir = join(tmp, 'unreadable-entry');
    mkdirSync(join(dir, 'model-a'), { recursive: true });
    const config = join(dir, 'model-a', 'config.json');
    writeFileSync(config, '{}');
    chmodSync(config, 0o000);
    const failures: string[] = [];
    await expect(discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).resolves.toEqual(
      [],
    );
    expect(failures).toEqual([join(dir, 'model-a')]);
    chmodSync(config, 0o600);
  });
});
