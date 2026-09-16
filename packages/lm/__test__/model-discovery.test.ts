import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, describe, expect, it } from 'vite-plus/test';

import { discoverLocalChatModels } from '../src/model-discovery.js';

const tmp = mkdtempSync(join(tmpdir(), 'mlx-discovery-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

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
});
