/**
 * Lenient-yet-strict base64 normalization and image media-type allowlist.
 *
 * Leaf module with zero addon imports so it can be shared freely across server
 * request mappers without pulling in `@mlx-node/core` or `@mlx-node/lm`.
 */

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

/**
 * Normalize base64 input: strip ASCII whitespace (`' '`, `'\t'`, `'\n'`, `'\r'`),
 * map URL-safe `'-'`/`'_'` to `'+'`/`'/'`, validate the alphabet and padding
 * (accepting unpadded or properly padded strings, rejecting empty strings,
 * `len % 4 === 1`, interior `'='`, excess `'='`, and invalid characters), and
 * return canonical padded base64 — or `null` if invalid.
 */
export function normalizeBase64(encoded: string): string | null {
  const stripped = encoded
    .replace(/[\t\n\r ]+/g, '')
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  if (stripped.length === 0) return null;

  let padCount = 0;
  while (padCount < stripped.length && stripped.charCodeAt(stripped.length - 1 - padCount) === 0x3d) {
    padCount++;
  }
  if (padCount > 2) return null;

  const dataLen = stripped.length - padCount;
  if (dataLen === 0) return null;

  if (padCount > 0) {
    if (stripped.length % 4 !== 0) return null;
  } else if (dataLen % 4 === 1) {
    return null;
  }

  for (let i = 0; i < dataLen; i++) {
    const c = stripped.charCodeAt(i);
    const valid =
      (c >= 0x41 && c <= 0x5a) || // A-Z
      (c >= 0x61 && c <= 0x7a) || // a-z
      (c >= 0x30 && c <= 0x39) || // 0-9
      c === 0x2b || // +
      c === 0x2f; // /
    if (!valid) return null;
  }

  if (padCount === 0) {
    const rem = dataLen % 4;
    if (rem === 2) return `${stripped}==`;
    if (rem === 3) return `${stripped}=`;
  }
  return stripped;
}

/**
 * Strictly decode base64 (accepting whitespace-wrapped, unpadded, and URL-safe
 * variants) into a plain `Uint8Array` (avoiding `Buffer.prototype.toJSON` so
 * JSON replacers observe `Uint8Array` cleanly). Throws `errorMessage` on empty
 * or malformed base64.
 */
export function decodeBase64Strict(encoded: string, errorMessage = 'Could not process image'): Uint8Array {
  const normalized = normalizeBase64(encoded);
  if (normalized === null) {
    throw new Error(errorMessage);
  }
  const buf = Buffer.from(normalized, 'base64');
  if (buf.byteLength === 0) {
    throw new Error(errorMessage);
  }
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}
