import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ggufArchitecture } from '@mlx-node/core';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { readGgufArchitecture } from '../src/gguf-metadata.js';
import { detectModelType } from '../src/model-detection.js';

function u32(value: number): Buffer {
  const result = Buffer.alloc(4);
  result.writeUInt32LE(value);
  return result;
}
function u64(value: number | bigint): Buffer {
  const result = Buffer.alloc(8);
  result.writeBigUInt64LE(BigInt(value));
  return result;
}
function str(value: string): Buffer {
  const text = Buffer.from(value);
  return Buffer.concat([u64(text.length), text]);
}
function entry(key: string, type: number, value: Buffer): Buffer {
  return Buffer.concat([str(key), u32(type), value]);
}
function header(entries: Buffer[]): Buffer {
  return Buffer.concat([Buffer.from('GGUF'), u32(3), u64(0), u64(entries.length), ...entries]);
}
const architecture = (value = 'qwen35') => entry('general.architecture', 8, str(value));
let root: string;
let path: string;
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'gguf-metadata-'));
  path = join(root, 'model.gguf');
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe('native-free GGUF metadata', () => {
  it.each([
    [0, 1],
    [1, 1],
    [2, 2],
    [3, 2],
    [4, 4],
    [5, 4],
    [6, 4],
    [7, 1],
    [10, 8],
    [11, 8],
    [12, 8],
  ])('matches native detection past scalar and array type %i', (type, size) => {
    writeFileSync(
      path,
      header([
        entry('scalar', type, Buffer.alloc(size)),
        entry('array', 9, Buffer.concat([u32(type), u64(3), Buffer.alloc(size * 3)])),
        architecture(),
      ]),
    );
    expect(readGgufArchitecture(path)).toBe('qwen35');
    expect(readGgufArchitecture(path)).toBe(ggufArchitecture(path));
  });

  it('skips tokenizer strings across buffer boundaries and honors the last architecture', () => {
    writeFileSync(
      path,
      header([
        architecture('gemma4'),
        entry(
          'tokenizer.ggml.tokens',
          9,
          Buffer.concat([u32(8), u64(3), str('x'.repeat(130_000)), str('中文'), str('')]),
        ),
        architecture('muse-glimmer'),
        entry('trailing-metadata', 8, str('more')),
      ]),
    );
    expect(readGgufArchitecture(path)).toBe('muse-glimmer');
    expect(readGgufArchitecture(path)).toBe(ggufArchitecture(path));
  });

  it.each([
    ['bad magic', Buffer.from('oops')],
    ['old version', Buffer.concat([Buffer.from('GGUF'), u32(2), u64(0), u64(0)])],
    ['missing architecture', header([])],
    ['empty architecture', header([architecture('')])],
    ['wrong architecture type', header([entry('general.architecture', 4, u32(3))])],
    ['truncated header', header([architecture()]).subarray(0, 18)],
    ['truncated string', header([architecture()]).subarray(0, -1)],
    ['truncated scalar array', header([entry('array', 9, Buffer.concat([u32(4), u64(3), u32(1)]))])],
    ['truncated string array', header([entry('array', 9, Buffer.concat([u32(8), u64(2), str('only-one')]))])],
    ['invalid metadata type', header([entry('bad', 99, Buffer.alloc(0))])],
    ['invalid array type', header([entry('bad', 9, Buffer.concat([u32(99), u64(0)]))])],
    ['oversized string', header([entry('bad', 8, u64(2n ** 63n))])],
    ['oversized array', header([entry('bad', 9, Buffer.concat([u32(4), u64(2n ** 63n)]))])],
  ])('rejects %s without allocation or unbounded reads', (_name, data) => {
    writeFileSync(path, data);
    expect(() => readGgufArchitecture(path)).toThrow();
  });

  it.skipIf(process.platform === 'win32')('refuses FIFO metadata files without blocking discovery', async () => {
    execFileSync('mkfifo', [path]);
    expect(() => readGgufArchitecture(path)).toThrow('regular file');
    execFileSync('mkfifo', [join(root, 'config.json')]);
    await expect(detectModelType(root)).rejects.toThrow('Cannot detect model type');
  });
});
