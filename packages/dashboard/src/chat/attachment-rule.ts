/**
 * Limits and pure helpers for message attachments, shared by the core route and the page. No Node imports: the page
 * imports this file as it is.
 */

export const ATTACH_MAX = 4;
/** First look at a file that is not a picture: a NUL or bad UTF-8 in these bytes means "not a text file". */
export const TEXT_SNIFF_BYTES = 8_192;
/** The agent's 5,000-token file cap (attach_file.rs MAX_FILE_TOKENS) as bytes of the numbered text it renders. */
export const FILE_NUMBERED_MAX_BYTES = 20_000;
/** Message text plus numbered file text; the agent cuts a whole prompt over 100,000 bytes to an excerpt. */
export const MESSAGE_FILES_MAX_BYTES = 90_000;
export const IMAGE_MAX_BYTES = 1_000_000;
export const IMAGE_MAX_BASE64_CHARS = 1_333_336;
/**
 * Stored base64 of one chat's pictures past which a message with pictures is refused: the reader stops at a 64 MiB
 * session file, and this leaves 24 MiB of it for text and tool output.
 */
export const CHAT_PICTURES_MAX_BASE64 = 40 * 1024 * 1024;
export const IMAGE_SOURCE_MAX_BYTES = 20 * 1024 * 1024;
export const IMAGE_MAX_SIDE = 2_000;
export const IMAGE_MAX_PIXELS = 2_400_000;
export const IMAGE_MIN_SIDE = 8;
export const IMAGE_MIN_PIXELS = 512;
export const NAME_IN_MAX_CHARS = 255;
export const NAME_MAX_CHARS = 120;
/** Upper estimate of one picture's prompt tokens (Qwen's largest is about 2,756). */
export const IMAGE_TOKENS = 2_800;
/** Share of the context window one message may fill; the rest is the agent's system prompt, tools and the answer. */
export const WINDOW_SHARE = 0.8;

export type ImageMediaType = 'image/png' | 'image/jpeg';

export type AttachmentInput =
  | { kind: 'file'; name: string; text: string }
  /** `data` is base64 with no `data:` prefix. */
  | { kind: 'image'; mediaType: ImageMediaType; data: string };

export type UserAttachment =
  | { kind: 'file'; name: string; bytes: number }
  | { kind: 'image'; mediaType: string; bytes: number };

const URI_PREFIX = 'file:///attached/';
/** UTF-8 length of '→', the agent's separator between a line number and its line. */
const ARROW_BYTES = 3;

/** No lone surrogate. `String.prototype.isWellFormed` is ES2024, and the page project compiles this file as ES2023. */
export function wellFormed(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const next = i + 1 < text.length ? text.charCodeAt(i + 1) : 0;
      if (next < 0xdc00 || next > 0xdfff) return false;
      i += 1;
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      return false;
    }
  }
  return true;
}

/** UTF-8 length; a lone surrogate counts as the 3 bytes of U+FFFD, as TextEncoder writes it. */
export function utf8Bytes(text: string): number {
  let bytes = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code < 0x80) bytes += 1;
    else if (code < 0x800) bytes += 2;
    else if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const low = text.charCodeAt(i + 1);
      if (low >= 0xdc00 && low <= 0xdfff) {
        bytes += 4;
        i += 1;
      } else bytes += 3;
    } else bytes += 3;
  }
  return bytes;
}

function digitCount(n: number): number {
  let digits = 1;
  for (let rest = n; rest >= 10; rest = Math.floor(rest / 10)) digits += 1;
  return digits;
}

/**
 * Bytes of the text as the agent renders it: Rust `str::lines()` numbered from 1 as `N→line`, joined by `\n`. A line
 * ends at `\n` and loses one `\r` before it; the text after the last `\n` is a line only when it is not empty.
 */
export function numberedBytes(text: string): number {
  const parts = text.split('\n');
  const last = parts.length - 1;
  let lines = 0;
  let bytes = 0;
  for (let i = 0; i <= last; i += 1) {
    let line = parts[i]!;
    if (i === last) {
      if (line === '') break;
    } else if (line.endsWith('\r')) line = line.slice(0, -1);
    lines += 1;
    bytes += utf8Bytes(line) + digitCount(lines) + ARROW_BYTES;
  }
  return lines === 0 ? 0 : bytes + lines - 1;
}

function isUnsafeNameChar(code: number): boolean {
  return (
    code < 0x20 ||
    (code >= 0x7f && code <= 0x9f) ||
    code === 0x23 || // #
    code === 0x22 || // "
    code === 0x3c || // <
    code === 0x3e || // >
    code === 0x2f || // /
    code === 0x5c // \
  );
}

/**
 * The name stored and shown for a file: its base name, controls and `# " < > / \` as `_`, at most
 * {@link NAME_MAX_CHARS} UTF-16 units without splitting a surrogate pair; `file` when nothing is left.
 */
export function safeName(name: string): string {
  let start = 0;
  for (let i = 0; i < name.length; i += 1) {
    const code = name.charCodeAt(i);
    if (code === 0x2f || code === 0x5c) start = i + 1;
  }
  let out = '';
  for (let i = start; i < name.length; i += 1) {
    const code = name.charCodeAt(i);
    out += isUnsafeNameChar(code) ? '_' : name[i];
  }
  if (out.length > NAME_MAX_CHARS) {
    let end = NAME_MAX_CHARS;
    const before = out.charCodeAt(end - 1);
    if (before >= 0xd800 && before <= 0xdbff) end -= 1;
    out = out.slice(0, end);
  }
  return out === '' || out === '.' || out === '..' ? 'file' : out;
}

/** The resource uri of an attached file: a label, never a real path. Holds no `#`, where the agent's FILE_REGEX stops. */
export function attachmentUri(name: string): string {
  return URI_PREFIX + safeName(name);
}

export function nameOfUri(uri: string): string {
  const name = uri.slice(uri.lastIndexOf('/') + 1);
  return name === '' ? 'file' : name;
}

function isBase64Char(code: number): boolean {
  return (
    (code >= 0x41 && code <= 0x5a) || // A-Z
    (code >= 0x61 && code <= 0x7a) || // a-z
    (code >= 0x30 && code <= 0x39) || // 0-9
    code === 0x2b || // +
    code === 0x2f // /
  );
}

function paddingOf(data: string): number {
  if (data.endsWith('==')) return 2;
  return data.endsWith('=') ? 1 : 0;
}

/** Standard padded base64: length a multiple of 4, at most two `=` at the end and none elsewhere. */
export function isBase64(data: string): boolean {
  if (data.length === 0 || data.length % 4 !== 0) return false;
  const end = data.length - paddingOf(data);
  for (let i = 0; i < end; i += 1) if (!isBase64Char(data.charCodeAt(i))) return false;
  return true;
}

/** Decoded length of base64 text, padded or not. */
export function base64Bytes(data: string): number {
  return Math.floor((data.length * 3) / 4) - paddingOf(data);
}

const MAGIC: Readonly<Record<ImageMediaType, readonly number[]>> = {
  'image/png': [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
  'image/jpeg': [0xff, 0xd8, 0xff],
};

/** The decoded first bytes are the signature of `mediaType`, which must be PNG or JPEG. */
export function magicMatches(mediaType: string, data: string): boolean {
  if (mediaType !== 'image/png' && mediaType !== 'image/jpeg') return false;
  const magic = MAGIC[mediaType];
  let head: string;
  try {
    head = atob(data.slice(0, 16));
  } catch {
    return false;
  }
  if (head.length < magic.length) return false;
  for (let i = 0; i < magic.length; i += 1) if (head.charCodeAt(i) !== magic[i]) return false;
  return true;
}

export function attachmentView(input: AttachmentInput): UserAttachment {
  return input.kind === 'file'
    ? { kind: 'file', name: safeName(input.name), bytes: utf8Bytes(input.text) }
    : { kind: 'image', mediaType: input.mediaType, bytes: base64Bytes(input.data) };
}

/** The base64 characters the agent stores for the pictures among `attachments`. */
export function storedPictureChars(attachments: Iterable<UserAttachment>): number {
  let chars = 0;
  for (const item of attachments) if (item.kind === 'image') chars += Math.ceil(item.bytes / 3) * 4;
  return chars;
}

/** Prompt tokens of a message and its attachments, estimated: three bytes a token, {@link IMAGE_TOKENS} a picture. */
export function estimateTokens(text: string, attachments: readonly AttachmentInput[]): number {
  let bytes = utf8Bytes(text);
  let images = 0;
  for (const item of attachments) {
    if (item.kind === 'file') bytes += numberedBytes(item.text);
    else images += 1;
  }
  return Math.ceil(bytes / 3) + images * IMAGE_TOKENS;
}
