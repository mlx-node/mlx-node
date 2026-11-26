/**
 * Strict base64 decoding (lazily backed by the native SIMD decoder in
 * `@mlx-node/core`, with a zero-regex RFC 4648 §3.5 fallback when the native
 * addon is unavailable or predates `decodeBase64Strict`) and image media-type
 * allowlist.
 *
 * Leaf module with zero static addon imports so request mappers and offline
 * replay scripts can import it without eagerly loading `@mlx-node/core`.
 */

import { createRequire } from 'node:module';

/** Maximum number of images accepted across a single request (Splash parity). */
export const MAX_REQUEST_IMAGES = 64;

/** Image MIME types supported by the native `image` crate (`gif`, `jpeg`, `png`, `webp`). */
export const ALLOWED_IMAGE_MEDIA_TYPES: ReadonlySet<string> = new Set([
  'image/jpeg',
  'image/png',
  'image/gif',
  'image/webp',
]);

/** Whether `mediaType` is one of the four supported raster image MIME types. */
export function isAllowedImageMediaType(mediaType: string): boolean {
  return ALLOWED_IMAGE_MEDIA_TYPES.has(mediaType.trim().toLowerCase());
}

type NativeDecodeBase64Strict = (encoded: string, errorMessage?: string | null) => Uint8Array;

let cachedNativeDecode: NativeDecodeBase64Strict | null | undefined;
let standardDecodeTable: Int8Array | null = null;
let urlSafeDecodeTable: Int8Array | null = null;

function resolveNativeDecode(): NativeDecodeBase64Strict | null {
  if (cachedNativeDecode !== undefined) {
    return cachedNativeDecode;
  }
  try {
    // Named `require` (not an alias) so `scripts/test-lanes.ts` sees the lazy
    // `@mlx-node/core` edge and assigns tests calling `decodeBase64Strict` to
    // the `addon` lane, while keeping module import itself addon-free.
    const require = createRequire(import.meta.url);
    const core = require('@mlx-node/core') as { decodeBase64Strict?: NativeDecodeBase64Strict };
    cachedNativeDecode = typeof core.decodeBase64Strict === 'function' ? core.decodeBase64Strict : null;
  } catch {
    cachedNativeDecode = null;
  }
  return cachedNativeDecode;
}

/**
 * Test hook: override (`fn` or `null`) or reset (`undefined`) the lazily
 * resolved native SIMD base64 decoder. Intentionally NOT re-exported from
 * `@mlx-node/server`.
 */
export function __setNativeBase64DecoderForTest(fn: NativeDecodeBase64Strict | null | undefined): void {
  cachedNativeDecode = fn;
}

/**
 * Test hook: return the resolved native SIMD base64 decoder (`null` when
 * unavailable or overridden to `null`). Intentionally NOT re-exported from
 * `@mlx-node/server`.
 */
export function __getResolvedNativeBase64DecoderForTest(): NativeDecodeBase64Strict | null {
  return resolveNativeDecode();
}

function buildDecodeTable(char62: number, char63: number): Int8Array {
  const table = new Int8Array(128);
  table.fill(-1);
  for (let i = 0; i < 26; i++) {
    table[0x41 + i] = i; // A-Z
    table[0x61 + i] = 26 + i; // a-z
  }
  for (let i = 0; i < 10; i++) {
    table[0x30 + i] = 52 + i; // 0-9
  }
  table[char62] = 62;
  table[char63] = 63;
  return table;
}

function getDecodeTable(urlSafe: boolean): Int8Array {
  if (urlSafe) {
    urlSafeDecodeTable ??= buildDecodeTable(0x2d, 0x5f);
    return urlSafeDecodeTable;
  }
  standardDecodeTable ??= buildDecodeTable(0x2b, 0x2f);
  return standardDecodeTable;
}

/**
 * Fallback decoder matching `crates/mlx-core/src/base64.rs` semantics when the
 * native addon is unavailable or predates `decodeBase64Strict` (zero regexes,
 * single-pass typed-array decode, strict RFC 4648 §3.5 canonical zero trailing
 * bits, plain tightly sized `Uint8Array`).
 */
function decodeBase64Fallback(encoded: string, errorMessage: string): Uint8Array {
  // If standard (`+`/`/`) and URL-safe (`-`/`_`) alphabets are mixed, selecting
  // the URL-safe table makes `+` and `/` fail lookup cleanly.
  const table = getDecodeTable(encoded.includes('-') || encoded.includes('_'));
  const out = new Uint8Array(Math.floor((encoded.length * 3) / 4));
  let outIdx = 0;
  let chunk = 0;
  let chunkLen = 0;
  let padCount = 0;

  for (let i = 0; i < encoded.length; i++) {
    const c = encoded.charCodeAt(i);
    if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) {
      continue;
    }
    if (c === 0x3d) {
      padCount++;
      if (padCount > 2) {
        throw new Error(errorMessage);
      }
      continue;
    }
    if (padCount > 0 || c >= 128) {
      throw new Error(errorMessage);
    }
    const val = table[c]!;
    if (val < 0) {
      throw new Error(errorMessage);
    }
    chunk = (chunk << 6) | val;
    chunkLen++;
    if (chunkLen === 4) {
      out[outIdx++] = (chunk >> 16) & 0xff;
      out[outIdx++] = (chunk >> 8) & 0xff;
      out[outIdx++] = chunk & 0xff;
      chunk = 0;
      chunkLen = 0;
    }
  }

  if (padCount > 0) {
    if (chunkLen === 2 && padCount === 2) {
      if ((chunk & 0x0f) !== 0) {
        throw new Error(errorMessage);
      }
      out[outIdx++] = (chunk >> 4) & 0xff;
    } else if (chunkLen === 3 && padCount === 1) {
      if ((chunk & 0x03) !== 0) {
        throw new Error(errorMessage);
      }
      out[outIdx++] = (chunk >> 10) & 0xff;
      out[outIdx++] = (chunk >> 2) & 0xff;
    } else {
      throw new Error(errorMessage);
    }
  } else if (chunkLen === 1) {
    throw new Error(errorMessage);
  } else if (chunkLen === 2) {
    if ((chunk & 0x0f) !== 0) {
      throw new Error(errorMessage);
    }
    out[outIdx++] = (chunk >> 4) & 0xff;
  } else if (chunkLen === 3) {
    if ((chunk & 0x03) !== 0) {
      throw new Error(errorMessage);
    }
    out[outIdx++] = (chunk >> 10) & 0xff;
    out[outIdx++] = (chunk >> 2) & 0xff;
  }

  if (outIdx === 0) {
    throw new Error(errorMessage);
  }
  return outIdx === out.byteLength ? out : out.slice(0, outIdx);
}

/**
 * Strictly decode base64 via the lazily loaded native SIMD implementation in
 * `@mlx-node/core` (or the equivalent RFC 4648 §3.5 fallback when the native
 * addon is unavailable or predates `decodeBase64Strict`). Accepts ASCII
 * whitespace `' '`/`'\t'`/`'\n'`/`'\r'`, unpadded or properly padded standard
 * base64, and URL-safe base64; enforces canonical zero trailing bits per RFC
 * 4648 §3.5; and returns a plain `Uint8Array` (avoiding
 * `Buffer.prototype.toJSON` so JSON replacers observe `Uint8Array` cleanly).
 * Throws `errorMessage` on empty, non-string, mixed-alphabet, non-canonical
 * trailing-bit, or malformed base64.
 */
export function decodeBase64Strict(encoded: string, errorMessage = 'Could not process image'): Uint8Array {
  if (typeof encoded !== 'string') {
    throw new Error(errorMessage);
  }
  const nativeDecode = resolveNativeDecode();
  if (nativeDecode !== null) {
    return nativeDecode(encoded, errorMessage);
  }
  return decodeBase64Fallback(encoded, errorMessage);
}
