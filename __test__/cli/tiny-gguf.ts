/** A small valid GGUF v3: string / u32 / u16 metadata and zero-filled F32 tensors. */
export interface TinyGguf {
  arch: string;
  metadata?: [string, string | { u32: number } | { u16: number } | { strings: string[] }][];
  tensors?: [string, number[]][];
}

export function tinyGgufBytes(opts: TinyGguf): Uint8Array<ArrayBuffer> {
  return tinyGguf(opts).bytes;
}

/** The file and the length of its header, bytes `[0, end of the tensor table)`. */
export function tinyGguf(opts: TinyGguf): { bytes: Uint8Array<ArrayBuffer>; headerLength: number } {
  const chunks: Buffer[] = [];
  const u32 = (v: number) => {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(v);
    chunks.push(b);
  };
  const u64 = (v: number) => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(v));
    chunks.push(b);
  };
  const str = (v: string) => {
    const bytes = Buffer.from(v);
    u64(bytes.length);
    chunks.push(bytes);
  };
  const metadata = [['general.architecture', opts.arch] as const, ...(opts.metadata ?? [])];
  const tensors = opts.tensors ?? [['a.weight', [8]]];
  chunks.push(Buffer.from('GGUF'));
  u32(3);
  u64(tensors.length);
  u64(metadata.length);
  for (const [key, value] of metadata) {
    str(key);
    if (typeof value === 'string') {
      u32(8);
      str(value);
    } else if ('u32' in value) {
      u32(4);
      u32(value.u32);
    } else if ('u16' in value) {
      u32(2);
      const b = Buffer.alloc(2);
      b.writeUInt16LE(value.u16);
      chunks.push(b);
    } else {
      u32(9);
      u32(8);
      u64(value.strings.length);
      for (const s of value.strings) str(s);
    }
  }
  let offset = 0;
  for (const [name, dims] of tensors) {
    str(name);
    u32(dims.length);
    for (const d of dims) u64(d);
    u32(0);
    u64(offset);
    const size = dims.reduce((a, b) => a * b, 1) * 4;
    offset += Math.ceil(size / 32) * 32;
  }
  const header = Buffer.concat(chunks);
  const padded = Math.ceil(header.length / 32) * 32;
  return {
    bytes: new Uint8Array(Buffer.concat([header, Buffer.alloc(padded - header.length + offset)])),
    headerLength: header.length,
  };
}
