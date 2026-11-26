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

/**
 * A PDF page image: the size the agent's `read_file` passes through untouched (`read_file/image.rs`), so one draw
 * serves the message and a reader alike.
 */
export const PAGE_LONG_SIDE = 1_280;
export const PAGE_MAX_PIXELS = 1_048_576;
export const PAGE_MAX_BYTES = 589_824;
export const PAGE_MAX_BASE64_CHARS = 786_432;
/** JPEG qualities tried in order, on white, until the page fits {@link PAGE_MAX_BYTES}. */
export const PAGE_JPEG_QUALITIES = [0.85, 0.7, 0.5] as const;
/** The PDF file picked: the agent's own PDF limit. */
export const PDF_SOURCE_MAX_BYTES = 50 * 1024 * 1024;
/** Pages one message carries: the agent's own count for one PDF read. */
export const PDF_MESSAGE_PAGES = 20;
/** Pages the PDF tool reads. */
export const PDF_TOOL_MAX_PAGES = 320;
/**
 * Base64 of the pictures and pages one message carries. The agent trims old pictures from a 13 MiB request down to
 * 8 MiB; this leaves 2 MiB of that floor for text, so pages are never dropped in the turn that sends them.
 */
export const MESSAGE_IMAGES_MAX_BASE64 = 6 * 1024 * 1024;
/** Images one conversation holds: the server refuses a request with more. */
export const CHAT_IMAGES_MAX = 64;
/** Flash Next takes this many images per conversation. */
export const FLASH_NEXT_FAMILY = 'qwen4_exp';
export const FLASH_NEXT_IMAGES_MAX = 4;
/** The first line of the note a PDF carries in the prompt; a text file that starts with it is refused. */
export const PDF_NOTE_MARKER = 'Portato PDF note 1';

export type ImageMediaType = 'image/png' | 'image/jpeg';

export type AttachmentInput =
  | { kind: 'file'; name: string; text: string }
  /** `data` is base64 with no `data:` prefix. */
  | { kind: 'image'; mediaType: ImageMediaType; data: string }
  /** The pages of a PDF, in order: base64 JPEG, no `data:` prefix. */
  | { kind: 'pdf'; name: string; images: string[] };

export type PdfWay = 'message' | 'tool';

export type UserAttachment =
  | { kind: 'file'; name: string; bytes: number }
  | { kind: 'image'; mediaType: string; bytes: number }
  /** `bytes`, `chars`: decoded size and stored base64 of the pages the message carries (none by the tool). */
  | { kind: 'pdf'; name: string; pages: number; way: PdfWay; bytes: number; chars: number; id?: string };

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

function isAsciiWhitespace(code: number): boolean {
  return code === 0x20 || code === 0x09 || code === 0x0a || code === 0x0d;
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

function normalizeBase64Prefix(prefix: string): string | null {
  if (prefix.length === 0) return null;
  let padCount = 0;
  while (padCount < prefix.length && prefix.charCodeAt(prefix.length - 1 - padCount) === 0x3d) {
    padCount += 1;
  }
  if (padCount > 2) return null;
  const dataLen = prefix.length - padCount;
  if (dataLen === 0) return null;
  if (padCount > 0) {
    if (prefix.length % 4 !== 0) return null;
  } else if (dataLen % 4 === 1) {
    return null;
  }
  for (let i = 0; i < dataLen; i += 1) {
    if (!isBase64Char(prefix.charCodeAt(i))) return null;
  }
  if (padCount === 0) {
    const rem = dataLen % 4;
    if (rem === 2) return `${prefix}==`;
    if (rem === 3) return `${prefix}=`;
  }
  return prefix;
}

/** Standard or URL-safe base64, padded or unpadded, ignoring ASCII whitespace. */
export function isBase64(data: string): boolean {
  let dataLen = 0;
  let padCount = 0;
  for (let i = 0; i < data.length; i += 1) {
    const code = data.charCodeAt(i);
    if (isAsciiWhitespace(code)) continue;
    if (code === 0x3d) {
      padCount += 1;
      if (padCount > 2) return false;
    } else {
      if (padCount > 0) return false;
      if (!isBase64Char(code) && code !== 0x2d && code !== 0x5f) return false;
      dataLen += 1;
    }
  }
  if (dataLen === 0) return false;
  if (padCount > 0) return (dataLen + padCount) % 4 === 0;
  return dataLen % 4 !== 1;
}

/** Decoded length of base64 text, ignoring ASCII whitespace and padding. */
export function base64Bytes(data: string): number {
  let unpaddedLen = 0;
  for (let i = 0; i < data.length; i += 1) {
    const code = data.charCodeAt(i);
    if (isAsciiWhitespace(code) || code === 0x3d) continue;
    unpaddedLen += 1;
  }
  return Math.floor((unpaddedLen * 3) / 4);
}

const MAGIC: Readonly<Record<ImageMediaType, readonly number[]>> = {
  'image/png': [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
  'image/jpeg': [0xff, 0xd8, 0xff],
};

/** The decoded first bytes are the signature of `mediaType`, which must be PNG or JPEG. */
export function magicMatches(mediaType: string, data: string): boolean {
  if (mediaType !== 'image/png' && mediaType !== 'image/jpeg') return false;
  const magic = MAGIC[mediaType];
  // Normalize whitespace and URL-safe chars BEFORE slicing the first 16 chars
  // (12 decoded bytes, enough for PNG's 8-byte header).
  let prefix = '';
  for (let i = 0; i < data.length && prefix.length < 16; i += 1) {
    const code = data.charCodeAt(i);
    if (isAsciiWhitespace(code)) continue;
    if (code === 0x2d) prefix += '+';
    else if (code === 0x5f) prefix += '/';
    else prefix += data[i];
  }
  const normalized = normalizeBase64Prefix(prefix);
  if (normalized === null) return false;
  let head: string;
  try {
    head = atob(normalized);
  } catch {
    return false;
  }
  if (head.length < magic.length) return false;
  for (let i = 0; i < magic.length; i += 1) if (head.charCodeAt(i) !== magic[i]) return false;
  return true;
}

/** Base64 characters of `bytes` decoded bytes, padded, as the agent stores a picture. */
function storedChars(bytes: number): number {
  return Math.ceil(bytes / 3) * 4;
}

export function attachmentView(input: AttachmentInput): UserAttachment {
  if (input.kind === 'file') return { kind: 'file', name: safeName(input.name), bytes: utf8Bytes(input.text) };
  if (input.kind === 'image') return { kind: 'image', mediaType: input.mediaType, bytes: base64Bytes(input.data) };
  let bytes = 0;
  let chars = 0;
  for (const page of input.images) {
    const size = base64Bytes(page);
    bytes += size;
    chars += storedChars(size);
  }
  return { kind: 'pdf', name: safeName(input.name), pages: input.images.length, way: 'message', bytes, chars };
}

/** The base64 characters the agent stores for the pictures and PDF pages among `attachments`. */
export function storedPictureChars(attachments: Iterable<UserAttachment>): number {
  let chars = 0;
  for (const item of attachments) {
    if (item.kind === 'image') chars += storedChars(item.bytes);
    else if (item.kind === 'pdf') chars += item.chars;
  }
  return chars;
}

/** The image blocks `attachments` put in the conversation: one per picture, one per page a message carries. */
export function imageCount(attachments: Iterable<UserAttachment>): number {
  let images = 0;
  for (const item of attachments) {
    if (item.kind === 'image') images += 1;
    else if (item.kind === 'pdf' && item.way === 'message') images += item.pages;
  }
  return images;
}

/** What {@link heldImages} reads of a transcript entry. */
export interface HeldEntry {
  kind: string;
  attachments?: readonly UserAttachment[];
  pictures?: readonly unknown[];
}

/**
 * The images a conversation holds: the pictures and pages of its messages and the pictures its tool calls read. An
 * upper count: a compaction may have dropped some.
 */
export function heldImages(entries: Iterable<HeldEntry>): number {
  let images = 0;
  for (const entry of entries) {
    if (entry.kind === 'user' && entry.attachments !== undefined) images += imageCount(entry.attachments);
    else if (entry.kind === 'tool' && entry.pictures !== undefined) images += entry.pictures.length;
  }
  return images;
}

/** A page as drawn, in pixels. */
export interface PageSize {
  width: number;
  height: number;
}

/**
 * The drawn size of a page `wPt` x `hPt` points: the long side at most {@link PAGE_LONG_SIDE}, the area at most
 * {@link PAGE_MAX_PIXELS}, up or down, each side floored.
 */
export function pdfPageSize(wPt: number, hPt: number): PageSize {
  const scale = Math.min(PAGE_LONG_SIDE / Math.max(wPt, hPt), Math.sqrt(PAGE_MAX_PIXELS / (wPt * hPt)));
  return { width: Math.max(1, Math.floor(wPt * scale)), height: Math.max(1, Math.floor(hPt * scale)) };
}

/** The engine's `smart_resize` (`vision/processing.rs`): sides rounded to `factor`, area kept in the bounds. */
function smartResize(height: number, width: number, factor: number, minPixels: number, maxPixels: number): PageSize {
  let h = height;
  let w = width;
  if (h < factor) {
    w = Math.floor((w * factor) / h);
    h = factor;
  }
  if (w < factor) {
    h = Math.floor((h * factor) / w);
    w = factor;
  }
  let hBar = Math.round(h / factor) * factor;
  let wBar = Math.round(w / factor) * factor;
  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((h * w) / maxPixels);
    hBar = Math.floor(h / beta / factor) * factor;
    wBar = Math.floor(w / beta / factor) * factor;
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (h * w));
    hBar = Math.ceil((h * beta) / factor) * factor;
    wBar = Math.ceil((w * beta) / factor) * factor;
  }
  return { width: wBar, height: hBar };
}

function qwenTokens(width: number, height: number, minPixels: number, maxPixels: number): number {
  const size = smartResize(height, width, 32, minPixels, maxPixels);
  return Math.floor(((size.height / 16) * (size.width / 16)) / 4);
}

/** Gemma4's `compute_target_size` (`gemma4/image_processor.rs`): a 645,120 px budget, sides floored to 48. */
function gemmaTokens(width: number, height: number): number {
  const side = 48;
  const factor = Math.sqrt(645_120 / (height * width));
  let h = Math.floor((factor * height) / side) * side;
  let w = Math.floor((factor * width) / side) * side;
  if (h === 0 && w === 0) {
    h = side;
    w = side;
  } else if (h === 0) {
    h = side;
    w = Math.max(side, Math.min(Math.floor(width / height) * side, 280 * side));
  } else if (w === 0) {
    w = side;
    h = Math.max(side, Math.min(Math.floor(height / width) * side, 280 * side));
  }
  return Math.floor((Math.floor(h / 16) * Math.floor(w / 16)) / 9);
}

/**
 * Prompt tokens of one page image `width` x `height` on a model of this family, by the engine's own formula; an
 * unknown family counts {@link IMAGE_TOKENS}. The chat template's wrapper tokens are not counted.
 */
export function pageTokens(family: string | null, width: number, height: number): number {
  switch (family) {
    case 'qwen3_5':
    case 'qwen3_5_moe':
      return qwenTokens(width, height, 147_384, 2_822_400);
    case FLASH_NEXT_FAMILY:
      return qwenTokens(width, height, 65_536, 262_144);
    case 'gemma4':
      return gemmaTokens(width, height);
    case 'inkling2':
      return Math.ceil(height / 40) * Math.ceil(width / 40);
    default:
      return IMAGE_TOKENS;
  }
}

function pagesTokens(family: string | null, sizes: readonly PageSize[]): number {
  let tokens = 0;
  for (const size of sizes) tokens += pageTokens(family, size.width, size.height);
  return tokens;
}

/**
 * What the note of a PDF says: `message`, its pages are the message's images `from` (from 1, among its image blocks)
 * on; `tool`, they are files the PDF tool reads by `id`.
 */
export type PdfNote =
  | { name: string; pages: number; way: 'message'; from: number }
  | { name: string; pages: number; way: 'tool'; id: string };

/** The note a PDF carries in the prompt: `key: value` lines the model reads, then one sentence. */
export function pdfNote(note: PdfNote): string {
  const head = [PDF_NOTE_MARKER, `name: ${safeName(note.name)}`, `pages: ${note.pages}`, `way: ${note.way}`];
  if (note.way === 'tool') {
    const args = `{"pdf": "${note.id}", "question": "<what to find out>"}`;
    return [
      ...head,
      `id: ${note.id}`,
      `The pages are image files on disk, not in this message. To read them, call the workflow tool with name read-pdf and args ${args}. Add "pages": "40-60" to read only some pages.`,
    ].join('\n');
  }
  const to = note.from + note.pages - 1;
  const said =
    note.pages === 1
      ? `Image ${note.from} of this message is page 1 of this PDF.`
      : `Images ${note.from}-${to} of this message are pages 1-${note.pages} of this PDF, in order.`;
  return [...head, `images: ${note.from}-${to}`, said].join('\n');
}

/** Line `index` of `text`, without the `\r` before its `\n`; null past the last line. */
function lineAt(lines: readonly string[], index: number): string | null {
  const line = lines[index];
  if (line === undefined) return null;
  return line.endsWith('\r') ? line.slice(0, -1) : line;
}

/** Whether the first line of `text` is the note's first line. */
export function isPdfNoteText(text: string): boolean {
  const end = text.indexOf('\n');
  return lineAt([end === -1 ? text : text.slice(0, end)], 0) === PDF_NOTE_MARKER;
}

/** The value of a `key: value` line whose key is `key`; the key ends at the first `: `. */
function fieldOf(line: string | null, key: string): string | null {
  if (line === null) return null;
  const at = line.indexOf(': ');
  if (at === -1 || line.slice(0, at) !== key) return null;
  return line.slice(at + 2);
}

/** A whole number written as decimal digits with no leading zero, at least 1; null for anything else. */
function countOf(text: string | null): number | null {
  if (text === null || text === '' || text.length > 6 || text.charCodeAt(0) === 0x30) return null;
  let n = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code < 0x30 || code > 0x39) return null;
    n = n * 10 + (code - 0x30);
  }
  return n;
}

/** A tool id: 8 lowercase letters or digits, `-`, the page count. */
function isPdfId(id: string, pages: number): boolean {
  if (id.length < 10 || id.charCodeAt(8) !== 0x2d) return false;
  for (let i = 0; i < 8; i += 1) {
    const code = id.charCodeAt(i);
    if (!((code >= 0x61 && code <= 0x7a) || (code >= 0x30 && code <= 0x39))) return false;
  }
  return countOf(id.slice(9)) === pages;
}

/** The note `text` is, read line by line; null when it is not one the app writes. */
export function readPdfNote(text: string): PdfNote | null {
  const lines = text.split('\n');
  if (lineAt(lines, 0) !== PDF_NOTE_MARKER) return null;
  const name = fieldOf(lineAt(lines, 1), 'name');
  const pages = countOf(fieldOf(lineAt(lines, 2), 'pages'));
  const way = fieldOf(lineAt(lines, 3), 'way');
  if (name === null || name === '' || safeName(name) !== name || pages === null) return null;
  if (way === 'tool') {
    const id = fieldOf(lineAt(lines, 4), 'id');
    if (id === null || pages > PDF_TOOL_MAX_PAGES || !isPdfId(id, pages)) return null;
    return { name, pages, way, id };
  }
  if (way !== 'message' || pages > PDF_MESSAGE_PAGES) return null;
  const range = fieldOf(lineAt(lines, 4), 'images');
  const dash = range === null ? -1 : range.indexOf('-');
  if (range === null || dash === -1) return null;
  const from = countOf(range.slice(0, dash));
  const to = countOf(range.slice(dash + 1));
  if (from === null || to === null || to - from + 1 !== pages) return null;
  return { name, pages, way, from };
}

/** The note of each PDF among `attachments`, in order: the pictures come first among the image blocks, then the pages. */
export function pdfNotes(attachments: readonly AttachmentInput[]): string[] {
  let from = 1;
  for (const item of attachments) if (item.kind === 'image') from += 1;
  const notes: string[] = [];
  for (const item of attachments) {
    if (item.kind !== 'pdf') continue;
    notes.push(pdfNote({ name: item.name, pages: item.images.length, way: 'message', from }));
    from += item.images.length;
  }
  return notes;
}

/**
 * An attachment as the token estimate reads it: a PDF may add each page's drawn size. Its pages go in the message,
 * and a page with no size counts {@link IMAGE_TOKENS}.
 */
export type TokenItem =
  | AttachmentInput
  | { kind: 'pdf'; name: string; images: readonly string[]; sizes: readonly PageSize[] };

function tokensOf(bytes: number, pictures: number, pages: number): number {
  return Math.ceil(bytes / 3) + pictures * IMAGE_TOKENS + pages;
}

/**
 * Prompt tokens of a message and its attachments, estimated: three bytes a token, {@link IMAGE_TOKENS} a picture, a
 * PDF its note and its pages by {@link pageTokens} for `family`.
 */
export function estimateTokens(text: string, attachments: readonly TokenItem[], family: string | null = null): number {
  let bytes = utf8Bytes(text);
  let pictures = 0;
  for (const item of attachments) {
    if (item.kind === 'file') bytes += numberedBytes(item.text);
    else if (item.kind === 'image') pictures += 1;
  }
  let from = pictures + 1;
  let pages = 0;
  for (const item of attachments) {
    if (item.kind !== 'pdf') continue;
    const count = item.images.length;
    bytes += numberedBytes(pdfNote({ name: item.name, pages: count, way: 'message', from }));
    from += count;
    const sizes = 'sizes' in item ? item.sizes : [];
    for (let i = 0; i < count; i += 1) {
      const size = sizes[i];
      pages += size === undefined ? IMAGE_TOKENS : pageTokens(family, size.width, size.height);
    }
  }
  return tokensOf(bytes, pictures, pages);
}

/** A PDF of the draft as the plan reads it: each page's drawn size, in order, and the base64 of its pages. */
export interface PdfDraft {
  name: string;
  sizes: readonly PageSize[];
  chars: number;
}

export interface PdfPlanInput {
  text: string;
  /** The message's text files and pictures. */
  attachments: readonly AttachmentInput[];
  /** Its PDFs, in the order added. */
  pdfs: readonly PdfDraft[];
  /** The model row's family; null when the endpoint names none. */
  family: string | null;
  /** The model's context window in tokens; null when no agent listed it. */
  window: number | null;
  contextUsed: number | null;
  /** {@link heldImages} of the conversation. */
  held: number;
  /** {@link storedPictureChars} of the conversation. */
  stored: number;
  /** The PDF tool is there in this mode. */
  tool: boolean;
}

export interface PdfPlan {
  ways: PdfWay[];
  /** `one`: two PDFs need the tool. `room`: one does not fit the message and this mode has no tool. */
  problem: 'one' | 'room' | null;
  /** The message as it goes: its text, files, pictures, each note and the pages it carries. */
  tokens: number;
  /** The page tokens the tool's readers read. */
  readTokens: number;
}

/**
 * Which way each PDF goes, in the order added: into the message while its pages fit every limit of the message and
 * the conversation, else to the PDF tool. Worked out again on every change, so a model change moves a PDF.
 */
export function planPdfs(input: PdfPlanInput): PdfPlan {
  let bytes = utf8Bytes(input.text);
  let pictures = 0;
  for (const item of input.attachments) {
    if (item.kind === 'file') bytes += numberedBytes(item.text);
    else if (item.kind === 'image') pictures += 1;
  }
  let chars = storedPictureChars(input.attachments.map(attachmentView));
  let stored = input.stored + chars;
  let images = pictures;
  let pages = 0;
  let pageTok = 0;
  let readTokens = 0;
  let tools = 0;
  let misses = 0;
  const ways: PdfWay[] = [];
  const flashNext = input.family === FLASH_NEXT_FAMILY;
  for (const pdf of input.pdfs) {
    const count = pdf.sizes.length;
    const tokens = pagesTokens(input.family, pdf.sizes);
    const noteBytes = numberedBytes(pdfNote({ name: pdf.name, pages: count, way: 'message', from: images + 1 }));
    const window = input.window;
    const fits =
      pages + count <= PDF_MESSAGE_PAGES &&
      chars + pdf.chars <= MESSAGE_IMAGES_MAX_BASE64 &&
      input.held + images + count <= CHAT_IMAGES_MAX &&
      stored + pdf.chars <= CHAT_PICTURES_MAX_BASE64 &&
      (!flashNext || input.held + images + count <= FLASH_NEXT_IMAGES_MAX) &&
      (window === null ||
        (input.contextUsed ?? 0) + tokensOf(bytes + noteBytes, pictures, pageTok + tokens) <= window * WINDOW_SHARE);
    if (fits || !input.tool) {
      if (!fits) misses += 1;
      ways.push('message');
      bytes += noteBytes;
      pages += count;
      pageTok += tokens;
      chars += pdf.chars;
      stored += pdf.chars;
      images += count;
    } else {
      ways.push('tool');
      tools += 1;
      readTokens += tokens;
      bytes += numberedBytes(pdfNote({ name: pdf.name, pages: count, way: 'tool', id: `xxxxxxxx-${count}` }));
    }
  }
  const problem = misses > 0 ? 'room' : tools > 1 ? 'one' : null;
  return { ways, problem, tokens: tokensOf(bytes, pictures, pageTok), readTokens };
}
