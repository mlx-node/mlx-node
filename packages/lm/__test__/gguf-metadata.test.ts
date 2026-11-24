import { execFileSync } from 'node:child_process';
import { closeSync, ftruncateSync, mkdtempSync, openSync, rmSync, writeFileSync, writeSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ggufArchitecture } from '@mlx-node/core';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  readGgufArchitecture,
  readGgufArchitectureSync,
  readGgufHeader,
  readGgufSourceArchitecture,
} from '../src/gguf-metadata.js';
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
    expect(readGgufArchitectureSync(path)).toBe('qwen35');
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
    expect(readGgufArchitectureSync(path)).toBe('muse-glimmer');
  });

  // Splash `GgufFile::parse`: a key holds one value.
  it('refuses a repeated metadata key like the native parser', async () => {
    writeFileSync(path, header([architecture('gemma4'), architecture('muse-glimmer')]));
    await expect(readGgufArchitecture(path)).rejects.toThrow('duplicate GGUF metadata key: general.architecture');
    expect(() => ggufArchitecture(path)).toThrow('duplicate GGUF metadata key: general.architecture');
    expect(() => readGgufArchitectureSync(path)).toThrow('duplicate GGUF metadata key: general.architecture');
  });

  const bytes = (data: Buffer) => (file: string) => writeFileSync(file, data);
  const manyKeys = Array.from({ length: 16384 }, (_, i) => entry(`key.${i}`, 0, Buffer.alloc(1)));
  const manyTensors = header([architecture()]);
  manyTensors.writeBigUInt64LE(16385n, 8);
  /** A sparse file whose one key is 200 MiB long: under the string limit, over the header limit. */
  const longKey = (file: string): void => {
    const fd = openSync(file, 'w');
    try {
      writeSync(fd, Buffer.concat([Buffer.from('GGUF'), u32(3), u64(0), u64(1), u64(200 * 1024 * 1024)]));
      ftruncateSync(fd, 210 * 1024 * 1024);
    } finally {
      closeSync(fd);
    }
  };
  it.each<[string, (file: string) => void, string]>([
    ['bad magic', bytes(Buffer.from('oops')), 'Not a GGUF file'],
    ['old version', bytes(Buffer.concat([Buffer.from('GGUF'), u32(2), u64(0), u64(0)])), 'Unsupported GGUF version'],
    ['missing architecture', bytes(header([])), 'non-empty general.architecture'],
    ['empty architecture', bytes(header([architecture('')])), 'non-empty general.architecture'],
    [
      'wrong architecture type',
      bytes(header([entry('general.architecture', 4, u32(3))])),
      'non-empty general.architecture',
    ],
    ['truncated header', bytes(header([architecture()]).subarray(0, 18)), 'Truncated or oversized GGUF metadata'],
    ['truncated string', bytes(header([architecture()]).subarray(0, -1)), 'Truncated or oversized GGUF metadata'],
    [
      'truncated scalar array',
      bytes(header([entry('array', 9, Buffer.concat([u32(4), u64(3), u32(1)]))])),
      'Truncated or oversized GGUF metadata',
    ],
    [
      'truncated string array',
      bytes(header([entry('array', 9, Buffer.concat([u32(8), u64(2), str('only-one')]))])),
      'Truncated or oversized GGUF metadata',
    ],
    ['invalid metadata type', bytes(header([entry('bad', 99, Buffer.alloc(0))])), 'Unsupported GGUF metadata type'],
    [
      'invalid array type',
      bytes(header([entry('bad', 9, Buffer.concat([u32(99), u64(0)]))])),
      'Unsupported GGUF array element type',
    ],
    ['oversized string', bytes(header([entry('bad', 8, u64(2n ** 63n))])), 'Oversized GGUF metadata length'],
    [
      'oversized array',
      bytes(header([entry('bad', 9, Buffer.concat([u32(4), u64(2n ** 63n)]))])),
      'Oversized GGUF metadata length',
    ],
    // The header limits of `readGgufHeader` and the native parser: 16384 keys, 16384 tensors, 128 MiB.
    [
      'more keys than the limit',
      bytes(header([...manyKeys, architecture()])),
      'implausible GGUF header counts: 0 tensors, 16385 keys',
    ],
    ['more tensors than the limit', bytes(manyTensors), 'implausible GGUF header counts: 16385 tensors, 1 keys'],
    ['a key past the header byte limit', longKey, `GGUF header exceeds ${128 * 1024 * 1024} bytes`],
  ])('rejects %s without allocation or unbounded reads', async (_name, write, expected) => {
    write(path);
    await expect(readGgufArchitecture(path)).rejects.toThrow(expected);
    expect(() => readGgufArchitectureSync(path)).toThrow(expected);
  });

  it('reads a skipped array in discovery without holding it', async () => {
    const ignored = 32 * 1024 * 1024;
    const bytes = header([
      entry('ignored', 9, Buffer.concat([u32(0), u64(ignored), Buffer.alloc(ignored)])),
      architecture(),
      entry('trailing-metadata', 8, str('more')),
    ]);
    let requested = 0;
    const read = async (position: number, length: number) => {
      requested += length;
      return bytes.subarray(position, position + length);
    };
    await expect(readGgufSourceArchitecture(read, bytes.length)).resolves.toBe('qwen35');
    expect(requested).toBeLessThan(1024 * 1024);
  });

  it.skipIf(process.platform === 'win32')('refuses FIFO metadata files without blocking discovery', async () => {
    execFileSync('mkfifo', [path]);
    await expect(readGgufArchitecture(path)).rejects.toThrow('regular file');
    expect(() => readGgufArchitectureSync(path)).toThrow('regular file');
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
