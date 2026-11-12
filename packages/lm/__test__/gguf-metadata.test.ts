import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ggufArchitecture } from '@mlx-node/core';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { readGgufArchitecture, readGgufHeader } from '../src/gguf-metadata.js';
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
  ])('matches native detection past scalar and array type %i', async (type, size) => {
    writeFileSync(
      path,
      header([
        entry('scalar', type, Buffer.alloc(size)),
        entry('array', 9, Buffer.concat([u32(type), u64(3), Buffer.alloc(size * 3)])),
        architecture(),
      ]),
    );
    await expect(readGgufArchitecture(path)).resolves.toBe('qwen35');
    await expect(readGgufArchitecture(path)).resolves.toBe(ggufArchitecture(path));
  });

  it('skips tokenizer strings across buffer boundaries', async () => {
    writeFileSync(
      path,
      header([
        entry(
          'tokenizer.ggml.tokens',
          9,
          Buffer.concat([u32(8), u64(3), str('x'.repeat(130_000)), str('中文'), str('')]),
        ),
        architecture('muse-glimmer'),
        entry('trailing-metadata', 8, str('more')),
      ]),
    );
    await expect(readGgufArchitecture(path)).resolves.toBe('muse-glimmer');
    await expect(readGgufArchitecture(path)).resolves.toBe(ggufArchitecture(path));
  });

  // Splash `GgufFile::parse`: a key holds one value.
  it('refuses a repeated metadata key like the native parser', async () => {
    writeFileSync(path, header([architecture('gemma4'), architecture('muse-glimmer')]));
    await expect(readGgufArchitecture(path)).rejects.toThrow('duplicate GGUF metadata key: general.architecture');
    expect(() => ggufArchitecture(path)).toThrow('duplicate GGUF metadata key: general.architecture');
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
  ])('rejects %s without allocation or unbounded reads', async (_name, data) => {
    writeFileSync(path, data);
    await expect(readGgufArchitecture(path)).rejects.toThrow();
  });

  it.skipIf(process.platform === 'win32')('refuses FIFO metadata files without blocking discovery', async () => {
    execFileSync('mkfifo', [path]);
    await expect(readGgufArchitecture(path)).rejects.toThrow('regular file');
    execFileSync('mkfifo', [join(root, 'config.json')]);
    await expect(detectModelType(root)).rejects.toThrow('Cannot detect model type');
  });
});

/** A GGUF of `entries` and F32 `tensors`, and the length of its header. */
function tinyGguf(entries: Buffer[], tensors: [string, number[]][]): { bytes: Buffer; headerLength: number } {
  let offset = 0;
  const table = tensors.map(([name, dims]) => {
    const descriptor = Buffer.concat([str(name), u32(dims.length), ...dims.map((d) => u64(d)), u32(0), u64(offset)]);
    offset += Math.ceil((dims.reduce((a, b) => a * b, 1) * 4) / 32) * 32;
    return descriptor;
  });
  const head = Buffer.concat([
    Buffer.from('GGUF'),
    u32(3),
    u64(tensors.length),
    u64(entries.length),
    ...entries,
    ...table,
  ]);
  const padded = Math.ceil(head.length / 32) * 32;
  return { bytes: Buffer.concat([head, Buffer.alloc(padded - head.length + offset)]), headerLength: head.length };
}

describe('readGgufHeader', () => {
  const source =
    (bytes: Uint8Array, reads: [number, number][] = []) =>
    async (position: number, length: number) => {
      reads.push([position, length]);
      return bytes.slice(position, position + length);
    };

  it('returns the header through the tensor table, across read edges', async () => {
    const tokens = Array.from({ length: 248_000 }, (_, i) => `t${i}`);
    const file = tinyGguf(
      [
        entry('general.architecture', 8, str('qwen35')),
        entry('tokenizer.ggml.tokens', 9, Buffer.concat([u32(8), u64(tokens.length), ...tokens.map(str)])),
        entry('qwen35.block_count', 4, u32(2)),
      ],
      [
        ['a.weight', [8]],
        ['b.weight', [4, 4]],
      ],
    );
    const reads: [number, number][] = [];
    const header = await readGgufHeader(source(file.bytes, reads), file.bytes.length, 1000);
    expect(header.length).toBe(file.headerLength);
    expect(Buffer.from(header).equals(Buffer.from(file.bytes.subarray(0, file.headerLength)))).toBe(true);
    expect(reads.length).toBeGreaterThan(2);
    expect(reads.every(([position], i) => i === 0 || position === reads[i - 1][0] + reads[i - 1][1])).toBe(true);
  });

  it('refuses more than 16384 tensors', async () => {
    const bytes = tinyGguf([entry('general.architecture', 8, str('llama'))], [['a.weight', [8]]]).bytes;
    bytes.writeBigUInt64LE(16385n, 8);
    await expect(readGgufHeader(source(bytes), bytes.length)).rejects.toThrow(/implausible GGUF header counts/);
  });
});
