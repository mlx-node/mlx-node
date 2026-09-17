import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, describe, expect, it } from 'vite-plus/test';

import { discoverLocalChatModels } from '../src/model-discovery.js';

const tmp = mkdtempSync(join(tmpdir(), 'mlx-discovery-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

// chmod 000 yields EACCES only on POSIX for a non-root user.
const canChmod = process.platform !== 'win32' && process.getuid?.() !== 0;

function minimalGguf(architecture: string): Buffer {
  const header = Buffer.alloc(24);
  header.write('GGUF');
  header.writeUInt32LE(3, 4);
  header.writeBigUInt64LE(1n, 16);
  const string = (value: string): Buffer => {
    const bytes = Buffer.from(value);
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(bytes.length));
    return Buffer.concat([length, bytes]);
  };
  const stringType = Buffer.alloc(4);
  stringType.writeUInt32LE(8);
  return Buffer.concat([header, string('general.architecture'), stringType, string(architecture)]);
}

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

  it('ignores generated GGUF asset directories while discovering the source checkpoint', async () => {
    const dir = join(tmp, 'gguf-assets');
    mkdirSync(dir);
    const gguf = join(dir, 'Qwen3.8-Flash-Next.gguf');
    writeFileSync(gguf, minimalGguf('qwen4exp'));
    for (const name of ['.mlx-qwen4-assets-v2-0123456789abcdef', '.mlx-qwen4-assets-tmp-publishing']) {
      const assets = join(dir, name);
      mkdirSync(assets);
      writeFileSync(join(assets, 'config.json'), JSON.stringify({ model_type: 'qwen4_exp' }));
      writeFileSync(join(assets, 'tokenizer.json'), '{}');
      if (!name.includes('-tmp-')) writeFileSync(join(assets, 'complete'), '1');
    }
    const failures: string[] = [];
    expect(await discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) })).toEqual([
      expect.objectContaining({ path: gguf, modelType: 'qwen4_exp' }),
    ]);
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

  it.skipIf(!canChmod).each(['direct', 'nested'] as const)(
    'reports an unreadable %s Qwen4 GGUF and discovers it after access is restored',
    async (layout) => {
      const dir = join(tmp, `unreadable-gguf-${layout}`);
      const repository = layout === 'direct' ? dir : join(dir, 'qwen4');
      mkdirSync(repository, { recursive: true });
      const gguf = join(repository, 'Qwen3.8-Flash-Next-00001-of-00002.GGUF');
      writeFileSync(gguf, minimalGguf('qwen4exp'));
      chmodSync(gguf, 0o000);
      const failures: string[] = [];
      try {
        await expect(
          discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) }),
        ).resolves.toEqual([]);
        expect(failures).toEqual([gguf]);
      } finally {
        chmodSync(gguf, 0o600);
      }
      expect(await discoverLocalChatModels(dir)).toEqual([
        expect.objectContaining({ path: gguf, modelType: 'qwen4_exp' }),
      ]);
    },
  );

  it.skipIf(!canChmod)('reports unreadable inventories in config-less Qwen4 directories', async () => {
    const dir = join(tmp, 'unreadable-inventory');
    const repository = join(dir, 'qwen4');
    mkdirSync(repository, { recursive: true });
    writeFileSync(join(repository, 'model.gguf'), minimalGguf('qwen4exp'));
    // Search permission allows a config lookup to return ENOENT, but listing
    // the GGUF-only directory still fails and must mark the scan incomplete.
    chmodSync(repository, 0o111);
    const failures: string[] = [];
    try {
      await expect(
        discoverLocalChatModels(dir, { onEntryFailure: (_e, path) => failures.push(path) }),
      ).resolves.toEqual([]);
      expect(failures).toEqual([repository]);
    } finally {
      chmodSync(repository, 0o700);
    }
    expect(await discoverLocalChatModels(dir)).toHaveLength(1);
  });
});
