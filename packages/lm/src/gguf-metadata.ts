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

/** Splash `install/gguf.py` `MAX_BYTES`: a GGUF header is refused past this. */
const MAX_HEADER_BYTES = 128 * 1024 * 1024;
/** Splash `install/gguf.py` `MAX_FIELDS`: metadata keys and tensors of one GGUF. */
const MAX_HEADER_COUNT = 16384;
/** Splash `install/hub.py` `HubRepo.open` block size: a header costs a few range reads. */
const HUB_READ_BYTES = 8 * 1024 * 1024;

/** `length` bytes of the source from `position` (fewer only at its end). */
export type GgufByteSource = (position: number, length: number) => Promise<Uint8Array>;

/**
 * Sequential header cursor over a byte source. Every byte from 0 to the
 * furthest one read is kept, so the header can be returned as read; skipping
 * an array advances the cursor without reading it until a later read needs
 * the bytes after it.
 */
class HeaderReader {
  private readonly chunks: Uint8Array[] = [];
  private fetched = 0;
  position = 0;

  constructor(
    private readonly read: GgufByteSource,
    private readonly size: number,
    private readonly chunk: number,
    private readonly limit = Number.POSITIVE_INFINITY,
  ) {}

  skip(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.size - this.position) {
      throw new Error('Truncated or oversized GGUF metadata');
    }
    this.position += length;
    if (this.position > this.limit) throw new Error(`GGUF header exceeds ${this.limit} bytes`);
  }

  /** Reads until every byte before `end` is held. */
  async fill(end: number): Promise<void> {
    while (this.fetched < end) {
      const length = Math.min(Math.max(this.chunk, end - this.fetched), this.size - this.fetched);
      const bytes = await this.read(this.fetched, length);
      if (bytes.length === 0) throw new Error('Truncated GGUF metadata');
      this.chunks.push(bytes);
      this.fetched += bytes.length;
    }
  }

  /** Bytes `[start, end)` of what was read. */
  slice(start: number, end: number): Uint8Array {
    const result = new Uint8Array(end - start);
    let at = 0;
    for (const bytes of this.chunks) {
      const from = Math.max(start - at, 0);
      const to = Math.min(end - at, bytes.length);
      if (from < to) result.set(bytes.subarray(from, to), at + from - start);
      at += bytes.length;
      if (at >= end) break;
    }
    return result;
  }

  async bytes(length: number): Promise<Buffer> {
    const position = this.position;
    this.skip(length);
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF string exceeds maximum length');
    await this.fill(position + length);
    return Buffer.from(this.slice(position, position + length));
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

function fileSource(file: FileHandle): GgufByteSource {
  return async (position, length) => {
    const buffer = Buffer.alloc(length);
    const { bytesRead } = await file.read(buffer, 0, length, position);
    return buffer.subarray(0, bytesRead);
  };
}

/**
 * The raw header of a GGUF, bytes `[0, end of the tensor table)`, as Splash's
 * installer keeps it (`install/gguf.py` `Metadata(.., tensors=True).raw`):
 * every metadata value and every tensor descriptor, no tensor data. At most
 * 16384 keys and 16384 tensors, 128 MiB in all; reads of `chunk` bytes.
 */
export async function readGgufHeader(read: GgufByteSource, size: number, chunk = HUB_READ_BYTES): Promise<Uint8Array> {
  const reader = new HeaderReader(read, size, chunk, MAX_HEADER_BYTES);
  if ((await reader.bytes(4)).toString() !== 'GGUF') throw new Error('Not a GGUF file');
  if ((await reader.u32()) < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
  const tensors = await reader.u64();
  const keys = await reader.u64();
  if (tensors > MAX_HEADER_COUNT || keys > MAX_HEADER_COUNT) {
    throw new Error(`implausible GGUF header counts: ${tensors} tensors, ${keys} keys`);
  }
  for (let index = 0; index < keys; index++) {
    reader.skip(await reader.length());
    await reader.skipValue(await reader.u32());
  }
  for (let index = 0; index < tensors; index++) {
    reader.skip(await reader.length());
    const rank = await reader.u32();
    if (rank === 0 || rank > 4) throw new Error('invalid GGUF tensor rank');
    reader.skip(rank * 8 + 4 + 8);
  }
  await reader.fill(reader.position);
  return reader.slice(0, reader.position);
}

export async function readGgufArchitecture(path: string): Promise<string> {
  // O_NONBLOCK plus the descriptor type check prevents a renamed FIFO from
  // hanging model discovery in the control-panel worker.
  const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const info = await file.stat();
    if (!info.isFile()) throw new Error('GGUF path must be a regular file');
    const reader = new HeaderReader(fileSource(file), info.size, 64 * 1024);
    if ((await reader.bytes(4)).toString() !== 'GGUF') throw new Error('Not a GGUF file');
    if ((await reader.u32()) < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
    await reader.u64(); // Tensor count: discovery only needs the metadata section.
    const count = await reader.u64();
    let architecture: string | undefined;
    const keys = new Set<string>();
    for (let index = 0; index < count; index++) {
      const key = await reader.string();
      if (keys.has(key)) throw new Error(`duplicate GGUF metadata key: ${key}`);
      keys.add(key);
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
