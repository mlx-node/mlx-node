/** Read GGUF metadata without linking the native addon or loading tensor weights. */
import { constants } from 'node:fs';
import { open, type FileHandle } from 'node:fs/promises';

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
    private readonly file: FileHandle,
    private readonly size: number,
  ) {}

  skip(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.size - this.position) {
      throw new Error('Truncated or oversized GGUF metadata');
    }
    this.position += length;
  }

  async bytes(length: number): Promise<Buffer> {
    const position = this.position;
    this.skip(length);
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF string exceeds maximum length');
    const result = Buffer.alloc(length);
    let offset = 0;
    while (offset < length) {
      const current = position + offset;
      if (current < this.start || current >= this.end) {
        this.start = current;
        const { bytesRead } = await this.file.read(this.buffer, 0, this.buffer.length, current);
        this.end = current + bytesRead;
        if (this.end === current) throw new Error('Truncated GGUF metadata');
      }
      const count = Math.min(length - offset, this.end - current);
      this.buffer.copy(result, offset, current - this.start, current - this.start + count);
      offset += count;
    }
    return result;
  }

  async u32(): Promise<number> {
    return (await this.bytes(4)).readUInt32LE();
  }
  async u64(): Promise<number> {
    const value = (await this.bytes(8)).readBigUInt64LE();
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Oversized GGUF metadata length');
    return Number(value);
  }
  async length(): Promise<number> {
    const length = await this.u64();
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF metadata exceeds maximum length');
    return length;
  }
  async string(): Promise<string> {
    return new TextDecoder('utf-8', { fatal: true }).decode(await this.bytes(await this.length()));
  }

  async skipValue(type: number, depth = 0): Promise<void> {
    const size = SCALAR_BYTES.get(type);
    if (size !== undefined) return this.skip(size);
    if (type === 8) return this.skip(await this.length());
    if (type !== 9 || depth > 16) throw new Error('Unsupported GGUF metadata type');
    const element = await this.u32();
    const count = await this.length();
    const elementSize = SCALAR_BYTES.get(element);
    if (elementSize !== undefined) return this.skip(count * elementSize);
    if (element !== 8 && element !== 9) throw new Error('Unsupported GGUF array element type');
    for (let index = 0; index < count; index++) await this.skipValue(element, depth + 1);
  }
}

export async function readGgufArchitecture(path: string): Promise<string> {
  // O_NONBLOCK plus the descriptor type check prevents a renamed FIFO from
  // hanging model discovery in the control-panel worker.
  const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const info = await file.stat();
    if (!info.isFile()) throw new Error('GGUF path must be a regular file');
    const reader = new HeaderReader(file, info.size);
    if ((await reader.bytes(4)).toString() !== 'GGUF') throw new Error('Not a GGUF file');
    if ((await reader.u32()) < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
    await reader.u64(); // Tensor count: discovery only needs the metadata section.
    const count = await reader.u64();
    let architecture: string | undefined;
    for (let index = 0; index < count; index++) {
      const key = await reader.string();
      const type = await reader.u32();
      if (key === 'general.architecture') {
        architecture = type === 8 ? await reader.string() : undefined;
        if (type !== 8) await reader.skipValue(type);
      } else {
        await reader.skipValue(type);
      }
    }
    if (!architecture) throw new Error('GGUF does not declare a non-empty general.architecture');
    return architecture;
  } finally {
    await file.close();
  }
}
