/** Read GGUF metadata without linking the native addon or loading tensor weights. */
import { closeSync, constants, fstatSync, openSync, readSync } from 'node:fs';
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

/** Refuses a header that counts more keys or tensors than {@link MAX_HEADER_COUNT}, as the native parser does. */
function checkHeaderCounts(tensors: number, keys: number): void {
  if (tensors > MAX_HEADER_COUNT || keys > MAX_HEADER_COUNT) {
    throw new Error(`implausible GGUF header counts: ${tensors} tensors, ${keys} keys`);
  }
}

/** Refuses a header read past {@link MAX_HEADER_BYTES}. */
function checkHeaderEnd(position: number): void {
  if (position > MAX_HEADER_BYTES) throw new Error(`GGUF header exceeds ${MAX_HEADER_BYTES} bytes`);
}

/** `length` bytes of the source from `position` (fewer only at its end). */
export type GgufByteSource = (position: number, length: number) => Promise<Uint8Array>;

/** Sequential header cursor over a byte source; `load` decides what is kept. */
abstract class HeaderCursor {
  position = 0;

  constructor(
    protected readonly read: GgufByteSource,
    protected readonly size: number,
    protected readonly chunk: number,
  ) {}

  /** Bytes `[position, position + length)`, already bounds-checked by `skip`. */
  protected abstract load(position: number, length: number): Promise<Uint8Array>;

  skip(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.size - this.position) {
      throw new Error('Truncated or oversized GGUF metadata');
    }
    this.position += length;
    checkHeaderEnd(this.position);
  }

  async bytes(length: number): Promise<Buffer> {
    const position = this.position;
    this.skip(length);
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF string exceeds maximum length');
    return Buffer.from(await this.load(position, length));
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

  async skipArrayBody(element: number, count: number, depth = 0): Promise<void> {
    if (depth > 16) throw new Error('Unsupported GGUF metadata type');
    const elementSize = SCALAR_BYTES.get(element);
    if (elementSize !== undefined) return this.skip(count * elementSize);
    if (element !== 8 && element !== 9) throw new Error('Unsupported GGUF array element type');
    for (let index = 0; index < count; index++) await this.skipValue(element, depth + 1);
  }

  async skipValue(type: number, depth = 0): Promise<void> {
    const size = SCALAR_BYTES.get(type);
    if (size !== undefined) return this.skip(size);
    if (type === 8) return this.skip(await this.length());
    if (type !== 9 || depth > 16) throw new Error('Unsupported GGUF metadata type');
    const element = await this.u32();
    const count = await this.length();
    await this.skipArrayBody(element, count, depth);
  }
}

/** Holds one window of `chunk` bytes: a skipped array is never read or kept. */
class WindowReader extends HeaderCursor {
  private window: Uint8Array = new Uint8Array(0);
  private start = 0;

  protected async load(position: number, length: number): Promise<Uint8Array> {
    const end = position + length;
    if (position < this.start || end > this.start + this.window.length) {
      const stop = Math.min(position + Math.max(this.chunk, length), this.size);
      const parts: Uint8Array[] = [];
      let at = position;
      while (at < stop) {
        const bytes = await this.read(at, stop - at);
        if (bytes.length === 0) break;
        parts.push(bytes);
        at += bytes.length;
      }
      if (at < end) throw new Error('Truncated GGUF metadata');
      this.window = parts.length === 1 ? parts[0] : Buffer.concat(parts);
      this.start = position;
    }
    return this.window.subarray(position - this.start, end - this.start);
  }
}

/** Keeps every byte from 0 to the furthest one read, so the header can be returned as read. */
class HeaderReader extends HeaderCursor {
  private readonly chunks: Uint8Array[] = [];
  private fetched = 0;

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

  protected async load(position: number, length: number): Promise<Uint8Array> {
    await this.fill(position + length);
    return this.slice(position, position + length);
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
  const reader = new HeaderReader(read, size, chunk);
  if ((await reader.bytes(4)).toString() !== 'GGUF') throw new Error('Not a GGUF file');
  if ((await reader.u32()) < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
  const tensors = await reader.u64();
  const keys = await reader.u64();
  checkHeaderCounts(tensors, keys);
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

/** Maximum length of `tokenizer.chat_template` decoded during lightweight discovery. */
const MAX_CHAT_TEMPLATE_BYTES = 1024 * 1024;
/** Maximum length of `tokenizer.ggml.model` decoded during lightweight discovery (`"gpt2"` is 4 bytes). */
const MAX_TOKENIZER_MODEL_BYTES = 64;
/** Maximum length of `clip.vision.projector_type` decoded during lightweight discovery (`"inkling2v"` is 9 bytes). */
const MAX_PROJECTOR_TYPE_BYTES = 64;
const GGUF_TYPE_INT32 = 5;
const GGUF_TYPE_STRING = 8;
const GGUF_TYPE_ARRAY = 9;

export interface GgufDiscoveryMetadata {
  architecture: string;
  chatTemplate?: string;
  /**
   * Whether the GGUF carries the embedded GPT-2 BPE metadata that native
   * `write_embedded_gpt2_tokenizer` (`crates/mlx-core/src/utils/gguf.rs`)
   * reconstructs when no sidecar `tokenizer.json` is present.
   */
  hasEmbeddedGpt2Tokenizer?: boolean;
  /** `clip.vision.projector_type` of a vision projector GGUF. */
  projectorType?: string;
}

export async function readGgufDiscoveryMetadata(path: string): Promise<GgufDiscoveryMetadata> {
  // O_NONBLOCK plus the descriptor type check prevents a renamed FIFO from
  // hanging model discovery in the control-panel worker.
  const file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const info = await file.stat();
    if (!info.isFile()) throw new Error('GGUF path must be a regular file');
    return await readGgufSourceDiscoveryMetadata(fileSource(file), info.size);
  } finally {
    await file.close();
  }
}

export async function readGgufArchitecture(path: string): Promise<string> {
  return (await readGgufDiscoveryMetadata(path)).architecture;
}

/** `general.architecture` (plus lightweight tokenizer/template metadata) of a GGUF source, read in one pass. */
export async function readGgufSourceDiscoveryMetadata(
  read: GgufByteSource,
  size: number,
  chunk = 64 * 1024,
): Promise<GgufDiscoveryMetadata> {
  const reader = new WindowReader(read, size, chunk);
  if ((await reader.bytes(4)).toString() !== 'GGUF') throw new Error('Not a GGUF file');
  if ((await reader.u32()) < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
  const tensors = await reader.u64();
  const count = await reader.u64();
  checkHeaderCounts(tensors, count);
  let architecture: string | undefined;
  let chatTemplate: string | undefined;
  let tokenizerModel: string | undefined;
  let projectorType: string | undefined;
  let tokensCount: number | undefined;
  let tokenTypeCount: number | undefined;
  let hasMerges = false;
  const keys = new Set<string>();
  for (let index = 0; index < count; index++) {
    const key = await reader.string();
    if (keys.has(key)) throw new Error(`duplicate GGUF metadata key: ${key}`);
    keys.add(key);
    const type = await reader.u32();
    if (key === 'general.architecture') {
      architecture = type === GGUF_TYPE_STRING ? await reader.string() : undefined;
      if (type !== GGUF_TYPE_STRING) await reader.skipValue(type);
    } else if (key === 'tokenizer.chat_template' && type === GGUF_TYPE_STRING) {
      const length = await reader.length();
      if (length <= MAX_CHAT_TEMPLATE_BYTES) {
        chatTemplate = new TextDecoder('utf-8').decode(await reader.bytes(length));
      } else {
        reader.skip(length);
      }
    } else if (key === 'tokenizer.ggml.model' && type === GGUF_TYPE_STRING) {
      const length = await reader.length();
      if (length <= MAX_TOKENIZER_MODEL_BYTES) {
        tokenizerModel = new TextDecoder('utf-8').decode(await reader.bytes(length));
      } else {
        reader.skip(length);
      }
    } else if (key === 'clip.vision.projector_type' && type === GGUF_TYPE_STRING) {
      const length = await reader.length();
      if (length <= MAX_PROJECTOR_TYPE_BYTES) {
        projectorType = new TextDecoder('utf-8').decode(await reader.bytes(length));
      } else {
        reader.skip(length);
      }
    } else if (key === 'tokenizer.ggml.tokens' && type === GGUF_TYPE_ARRAY) {
      const element = await reader.u32();
      const length = await reader.length();
      if (element === GGUF_TYPE_STRING) tokensCount = length;
      await reader.skipArrayBody(element, length);
    } else if (key === 'tokenizer.ggml.token_type' && type === GGUF_TYPE_ARRAY) {
      const element = await reader.u32();
      const length = await reader.length();
      if (element === GGUF_TYPE_INT32) tokenTypeCount = length;
      await reader.skipArrayBody(element, length);
    } else if (key === 'tokenizer.ggml.merges' && type === GGUF_TYPE_ARRAY) {
      const element = await reader.u32();
      const length = await reader.length();
      if (element === GGUF_TYPE_STRING) hasMerges = true;
      await reader.skipArrayBody(element, length);
    } else {
      await reader.skipValue(type);
    }
  }
  if (!architecture) throw new Error('GGUF does not declare a non-empty general.architecture');
  const hasEmbeddedGpt2Tokenizer =
    tokenizerModel === 'gpt2' &&
    tokensCount !== undefined &&
    tokenTypeCount !== undefined &&
    tokensCount === tokenTypeCount &&
    hasMerges;
  return {
    architecture,
    ...(chatTemplate !== undefined ? { chatTemplate } : {}),
    ...(hasEmbeddedGpt2Tokenizer ? { hasEmbeddedGpt2Tokenizer: true } : {}),
    ...(projectorType !== undefined ? { projectorType } : {}),
  };
}

/** `general.architecture` of a GGUF source, read through one `chunk`-byte window. */
export async function readGgufSourceArchitecture(
  read: GgufByteSource,
  size: number,
  chunk = 64 * 1024,
): Promise<string> {
  return (await readGgufSourceDiscoveryMetadata(read, size, chunk)).architecture;
}

/**
 * {@link HeaderCursor} for a caller that cannot wait: the same checks, over a descriptor read with `readSync` through
 * one `chunk`-byte window.
 */
class SyncWindowReader {
  position = 0;
  private window: Buffer = Buffer.alloc(0);
  private start = 0;

  constructor(
    private readonly fd: number,
    private readonly size: number,
    private readonly chunk: number,
  ) {}

  skip(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.size - this.position) {
      throw new Error('Truncated or oversized GGUF metadata');
    }
    this.position += length;
    checkHeaderEnd(this.position);
  }

  bytes(length: number): Buffer {
    const position = this.position;
    this.skip(length);
    if (length > MAX_GGUF_LENGTH) throw new Error('GGUF string exceeds maximum length');
    const end = position + length;
    if (position < this.start || end > this.start + this.window.length) {
      const window = Buffer.alloc(Math.min(Math.max(this.chunk, length), this.size - position));
      let at = 0;
      while (at < window.length) {
        const read = readSync(this.fd, window, at, window.length - at, position + at);
        if (read === 0) break;
        at += read;
      }
      if (position + at < end) throw new Error('Truncated GGUF metadata');
      this.window = window.subarray(0, at);
      this.start = position;
    }
    return this.window.subarray(position - this.start, end - this.start);
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

  skipArrayBody(element: number, count: number, depth = 0): void {
    if (depth > 16) throw new Error('Unsupported GGUF metadata type');
    const elementSize = SCALAR_BYTES.get(element);
    if (elementSize !== undefined) return this.skip(count * elementSize);
    if (element !== 8 && element !== 9) throw new Error('Unsupported GGUF array element type');
    for (let index = 0; index < count; index++) this.skipValue(element, depth + 1);
  }

  skipValue(type: number, depth = 0): void {
    const size = SCALAR_BYTES.get(type);
    if (size !== undefined) return this.skip(size);
    if (type === 8) return this.skip(this.length());
    if (type !== 9 || depth > 16) throw new Error('Unsupported GGUF metadata type');
    const element = this.u32();
    const count = this.length();
    this.skipArrayBody(element, count, depth);
  }
}

/** {@link readGgufDiscoveryMetadata} without waiting, for a synchronous caller. */
export function readGgufDiscoveryMetadataSync(path: string): GgufDiscoveryMetadata {
  const fd = openSync(path, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const info = fstatSync(fd);
    if (!info.isFile()) throw new Error('GGUF path must be a regular file');
    const reader = new SyncWindowReader(fd, info.size, 64 * 1024);
    if (reader.bytes(4).toString() !== 'GGUF') throw new Error('Not a GGUF file');
    if (reader.u32() < 3) throw new Error('Unsupported GGUF version (only v3+ supported)');
    const tensors = reader.u64();
    const count = reader.u64();
    checkHeaderCounts(tensors, count);
    let architecture: string | undefined;
    let chatTemplate: string | undefined;
    let tokenizerModel: string | undefined;
    let tokensCount: number | undefined;
    let tokenTypeCount: number | undefined;
    let hasMerges = false;
    const keys = new Set<string>();
    for (let index = 0; index < count; index++) {
      const key = reader.string();
      if (keys.has(key)) throw new Error(`duplicate GGUF metadata key: ${key}`);
      keys.add(key);
      const type = reader.u32();
      if (key === 'general.architecture') {
        architecture = type === GGUF_TYPE_STRING ? reader.string() : undefined;
        if (type !== GGUF_TYPE_STRING) reader.skipValue(type);
      } else if (key === 'tokenizer.chat_template' && type === GGUF_TYPE_STRING) {
        const length = reader.length();
        if (length <= MAX_CHAT_TEMPLATE_BYTES) {
          chatTemplate = new TextDecoder('utf-8').decode(reader.bytes(length));
        } else {
          reader.skip(length);
        }
      } else if (key === 'tokenizer.ggml.model' && type === GGUF_TYPE_STRING) {
        const length = reader.length();
        if (length <= MAX_TOKENIZER_MODEL_BYTES) {
          tokenizerModel = new TextDecoder('utf-8').decode(reader.bytes(length));
        } else {
          reader.skip(length);
        }
      } else if (key === 'tokenizer.ggml.tokens' && type === GGUF_TYPE_ARRAY) {
        const element = reader.u32();
        const length = reader.length();
        if (element === GGUF_TYPE_STRING) tokensCount = length;
        reader.skipArrayBody(element, length);
      } else if (key === 'tokenizer.ggml.token_type' && type === GGUF_TYPE_ARRAY) {
        const element = reader.u32();
        const length = reader.length();
        if (element === GGUF_TYPE_INT32) tokenTypeCount = length;
        reader.skipArrayBody(element, length);
      } else if (key === 'tokenizer.ggml.merges' && type === GGUF_TYPE_ARRAY) {
        const element = reader.u32();
        const length = reader.length();
        if (element === GGUF_TYPE_STRING) hasMerges = true;
        reader.skipArrayBody(element, length);
      } else {
        reader.skipValue(type);
      }
    }
    if (!architecture) throw new Error('GGUF does not declare a non-empty general.architecture');
    const hasEmbeddedGpt2Tokenizer =
      tokenizerModel === 'gpt2' &&
      tokensCount !== undefined &&
      tokenTypeCount !== undefined &&
      tokensCount === tokenTypeCount &&
      hasMerges;
    return {
      architecture,
      ...(chatTemplate !== undefined ? { chatTemplate } : {}),
      ...(hasEmbeddedGpt2Tokenizer ? { hasEmbeddedGpt2Tokenizer: true } : {}),
    };
  } finally {
    closeSync(fd);
  }
}

/** {@link readGgufArchitecture} without waiting, for a synchronous caller: the same answer and the same refusals. */
export function readGgufArchitectureSync(path: string): string {
  return readGgufDiscoveryMetadataSync(path).architecture;
}
