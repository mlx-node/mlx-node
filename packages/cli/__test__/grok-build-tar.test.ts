import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';

import { describe, expect, it } from 'vite-plus/test';

import { unpackAgentTarGz } from '../src/commands/agent/grok-build.js';

const BLOCK = 512;

function tarHeader(opts: {
  name: string;
  size: number;
  typeflag: string;
  base256?: boolean;
  corruptChecksum?: boolean;
}): Buffer {
  const header = Buffer.alloc(BLOCK);
  header.write(opts.name, 0, Math.min(opts.name.length, 100), 'utf8');
  header.write('0000755\0', 100, 'ascii');
  header.write('0000000\0', 108, 'ascii');
  header.write('0000000\0', 116, 'ascii');
  if (opts.base256) {
    header[124] = 0x80;
    let remaining = BigInt(opts.size);
    for (let i = 135; i >= 125; i--) {
      header[i] = Number(remaining & 0xffn);
      remaining >>= 8n;
    }
  } else {
    header.write(`${opts.size.toString(8).padStart(11, '0')}\0`, 124, 'ascii');
  }
  header.write('00000000000\0', 136, 'ascii');
  header.fill(0x20, 148, 156);
  header[156] = opts.typeflag === '' ? 0 : opts.typeflag.charCodeAt(0);
  header.write('ustar\0', 257, 'ascii');
  header.write('00', 263, 'ascii');

  let sum = 0;
  for (let i = 0; i < BLOCK; i++) sum += header[i]!;
  if (opts.corruptChecksum) sum += 1;
  header.write(`${sum.toString(8).padStart(6, '0')}\0 `, 148, 'ascii');
  return header;
}

function padBlock(data: Buffer): Buffer {
  const rem = data.length % BLOCK;
  return rem === 0 ? data : Buffer.concat([data, Buffer.alloc(BLOCK - rem)]);
}

function paxRecord(key: string, value: string): Buffer {
  const body = ` ${key}=${value}\n`;
  let len = Buffer.byteLength(body) + 1;
  while (true) {
    const line = `${len}${body}`;
    const byteLen = Buffer.byteLength(line);
    if (byteLen === len) return Buffer.from(line, 'utf8');
    len = byteLen;
  }
}

function tarEntry(opts: {
  name: string;
  data: Buffer;
  typeflag?: string;
  headerSize?: number;
  base256?: boolean;
  corruptChecksum?: boolean;
}): Buffer {
  const header = tarHeader({
    name: opts.name,
    size: opts.headerSize ?? opts.data.length,
    typeflag: opts.typeflag ?? '0',
    base256: opts.base256,
    corruptChecksum: opts.corruptChecksum,
  });
  return Buffer.concat([header, padBlock(opts.data)]);
}

describe('unpackAgentTarGz', () => {
  it('unpacks a standard ustar single-file archive with octal size', () => {
    const payload = Buffer.from('ELF-or-MachO-or-PE-payload\x00\xff'.repeat(64));
    const tar = Buffer.concat([tarEntry({ name: 'mlx-agent', data: payload }), Buffer.alloc(BLOCK * 2)]);
    expect(unpackAgentTarGz(gzipSync(tar))).toEqual(payload);
  });

  it('handles GNU base-256 sizes, directory prefix entries, and GNU long-name L blocks', () => {
    const payload = Buffer.from('binary-contents-1234567890');
    const dir = tarEntry({ name: './', data: Buffer.alloc(0), typeflag: '5' });
    const longName = tarEntry({
      name: '././@LongLink',
      data: Buffer.from(`${'a'.repeat(140)}/mlx-agent\0`),
      typeflag: 'L',
    });
    const file = tarEntry({ name: 'truncated', data: payload, base256: true });
    const tar = Buffer.concat([dir, longName, file, Buffer.alloc(BLOCK * 2)]);
    expect(unpackAgentTarGz(gzipSync(tar))).toEqual(payload);
  });

  it('skips PAX x/g extended headers and applies PAX size when present', () => {
    const payload = Buffer.from('pax-sized-payload-bytes!'.repeat(30));
    const globalPax = tarEntry({ name: 'pax_global_header', data: paxRecord('comment', 'release'), typeflag: 'g' });
    const nextPax = tarEntry({
      name: 'PaxHeaders/mlx-agent',
      data: Buffer.concat([paxRecord('path', 'bin/mlx-agent'), paxRecord('size', String(payload.length))]),
      typeflag: 'x',
    });
    // Header size field is 0; PAX `size` must override it.
    const file = tarEntry({ name: 'mlx-agent', data: payload, headerSize: 0 });
    const tar = Buffer.concat([globalPax, nextPax, file, Buffer.alloc(BLOCK * 2)]);
    expect(unpackAgentTarGz(gzipSync(tar))).toEqual(payload);
  });

  it('unpacks a real tar -czf archive', () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-tar-real-'));
    try {
      const payload = Buffer.from('real-tar-binary-\x00\x01\x02'.repeat(100));
      writeFileSync(join(dir, 'mlx-agent'), payload);
      const gz = execFileSync('tar', ['-czf', '-', '-C', dir, 'mlx-agent'], {
        env: { ...process.env, COPYFILE_DISABLE: '1' },
      });
      expect(unpackAgentTarGz(gz)).toEqual(payload);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects corrupt checksums, truncated headers/payloads, empty archives, and non-gzip bytes', () => {
    const payload = Buffer.from('hello');
    const badChecksum = Buffer.concat([
      tarEntry({ name: 'mlx-agent', data: payload, corruptChecksum: true }),
      Buffer.alloc(BLOCK * 2),
    ]);
    expect(() => unpackAgentTarGz(gzipSync(badChecksum))).toThrow('checksum');

    // Empty gzip payload and zero-only tar archive.
    expect(() => unpackAgentTarGz(gzipSync(Buffer.alloc(0)))).toThrow('empty');
    expect(() => unpackAgentTarGz(gzipSync(Buffer.alloc(BLOCK * 2)))).toThrow('no file payload');

    // Zero-byte regular file.
    const emptyFile = Buffer.concat([tarEntry({ name: 'mlx-agent', data: Buffer.alloc(0) }), Buffer.alloc(BLOCK * 2)]);
    expect(() => unpackAgentTarGz(gzipSync(emptyFile))).toThrow('no file payload');

    // Truncated header (< 512 bytes).
    expect(() => unpackAgentTarGz(gzipSync(Buffer.alloc(100)))).toThrow('truncated');

    // Truncated entry payload (header claims 1024 bytes, only 512 present).
    const truncPayload = Buffer.concat([
      tarHeader({ name: 'mlx-agent', size: 1024, typeflag: '0' }),
      Buffer.alloc(BLOCK),
    ]);
    expect(() => unpackAgentTarGz(gzipSync(truncPayload))).toThrow('truncated');

    // Not gzip at all.
    expect(() => unpackAgentTarGz(Buffer.from('not a gzip stream'))).toThrow();
  });
});
