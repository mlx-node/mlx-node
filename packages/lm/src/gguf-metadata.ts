/** Read GGUF metadata without linking the native addon or loading tensor weights. */
import { closeSync, constants, fstatSync, openSync, readSync } from 'node:fs';

const MAX_GGUF_LENGTH = 256 * 1024 * 1024;
const SCALAR_BYTES = new Map([
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
]);

/** Buffered header cursor; skipping an array never allocates its payload. */
class HeaderReader {
  private readonly buffer = Buffer.alloc(64 * 1024);
  private start = 0;
  private end = 0;
  private position = 0;

  constructor(
    private readonly fd: number,
    private readonly size: number,
  ) {}

  skip(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.size - this.position) {
      throw new Error('Truncated or oversized GGUF metadata');
    }
    this.position += length;
  }

  bytes(length: number): Buffer {
    const position = this.position;
    this.skip(length);
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF string exceeds maximum length');
    const result = Buffer.alloc(length);
    let offset = 0;
    while (offset < length) {
      const current = position + offset;
      if (current < this.start || current >= this.end) {
        this.start = current;
        this.end = current + readSync(this.fd, this.buffer, 0, this.buffer.length, current);
        if (this.end === current) throw new Error('Truncated GGUF metadata');
      }
      const count = Math.min(length - offset, this.end - current);
      this.buffer.copy(result, offset, current - this.start, current - this.start + count);
      offset += count;
    }
    return result;
  }

  u32(): number {
    return this.bytes(4).readUInt32LE();
  }
  u64(): number {
    const value = this.bytes(8).readBigUInt64LE();
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Oversized GGUF metadata length');
    return Number(value);
  }
  length(): number {
    const length = this.u64();
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF metadata exceeds maximum length');
    return length;
  }
  string(): string {
    return new TextDecoder('utf-8', { fatal: true }).decode(this.bytes(this.length()));
  }

  skipValue(type: number, depth = 0): void {
    const size = SCALAR_BYTES.get(type);
    if (size !== undefined) return this.skip(size);
    if (type === 8) return this.skip(this.length());
    if (type !== 9 || depth > 16) throw new Error('Unsupported GGUF metadata type');
    const element = this.u32();
    const count = this.length();
    const elementSize = SCALAR_BYTES.get(element);
    if (elementSize !== undefined) return this.skip(count * elementSize);
    if (element !== 8 && element !== 9) throw new Error('Unsupported GGUF array element type');
    for (let index = 0; index < count; index++) this.skipValue(element, depth + 1);
  }
}

export function readGgufArchitecture(path: string): string {
  // O_NONBLOCK plus the descriptor type check prevents a renamed FIFO from
  // hanging model discovery in the control-panel worker.
  const fd = openSync(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const info = fstatSync(fd);
    if (!info.isFile()) throw new Error('GGUF path must be a regular file');
    const reader = new HeaderReader(fd, info.size);
    if (reader.bytes(4).toString() !== 'GGUF') throw new Error('Not a GGUF file');
    if (reader.u32() < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
    reader.u64(); // Tensor count: discovery only needs the metadata section.
    const count = reader.u64();
    let architecture: string | undefined;
    for (let index = 0; index < count; index++) {
      const key = reader.string();
      const type = reader.u32();
      if (key === 'general.architecture') {
        architecture = type === 8 ? reader.string() : undefined;
        if (type !== 8) reader.skipValue(type);
      } else {
        reader.skipValue(type);
      }
    }
    if (!architecture) throw new Error('GGUF does not declare a non-empty general.architecture');
    return architecture;
  } finally {
    closeSync(fd);
  }
}
