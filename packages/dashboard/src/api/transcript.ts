/**
 * Pure transcript projection: pi session entries → the rows the SPA renders.
 *
 * Everything here is data → data with no I/O, no database and no transport, so
 * it is unit-testable directly and reusable by any caller that already holds the
 * parsed entries. Moved verbatim out of the former `api.ts`.
 */

import type { SessionEntry } from '@earendil-works/pi-coding-agent';

/** One transcript message projected from a pi session entry. */
export interface TranscriptEntry {
  role: string;
  text: string;
  /** `summary` is a one-line digest of the call's arguments (path / command / …). */
  toolCalls: Array<{ id: string; name: string; arguments: unknown; summary: string }>;
  ts: number;
  /** Present for `toolResult` messages. */
  toolName?: string;
  isError?: boolean;
  /** The model that produced an `assistant` message (verbatim id), for its logo/name. */
  model?: string;
  /**
   * One-line digest of the ORIGINATING tool call's arguments (its `path` /
   * `command` / …), joined by `toolCallId`. Lets a collapsed result row show what
   * it acted on without expanding. Present only for `toolResult` messages.
   */
  title?: string;
  /** Decoded image blocks (base64), rendered inline as thumbnails by the UI. */
  images?: Array<{ mimeType: string; data: string }>;
  /**
   * Short labels (e.g. `HEIC · 32 KB`) for binary blobs a tool returned verbatim
   * as text, or images too large to inline. The UI shows a chip instead of the
   * raw bytes.
   */
  binaryNotes?: string[];
  /**
   * Model-facing image-read notes (`[Image: original …, displayed at …. Multiply
   * coordinates by … …]`) split out of {@link text}. Pure coordinate-mapping
   * plumbing next to a rendered thumbnail, so the UI hides these by default.
   */
  imageNotes?: string[];
}

/** A base64 image bigger than this (~1.5 MB decoded) is chipped, not inlined. */
const MAX_INLINE_IMAGE_B64 = 2_000_000;
/** Cap inlined images per message so one turn can't bloat the transcript payload. */
const MAX_INLINE_IMAGES = 8;

/**
 * Whether a string is raw binary read verbatim as "text" — a `read` of a `.heic`,
 * PDF, etc. returns the file bytes in a `text` block, and control bytes (NUL and
 * friends, excluding tab/newline/CR) never occur in real transcript prose. Such a
 * block must not be dumped into the rendered text; it becomes a chip instead.
 */
export function isBinaryText(text: string): boolean {
  // Scan a bounded prefix (enough to classify; cheap for a multi-MB blob) for any
  // C0 control byte other than tab (9), newline (10), or carriage return (13).
  const limit = Math.min(text.length, 4096);
  for (let i = 0; i < limit; i++) {
    const code = text.charCodeAt(i);
    if (code <= 8 || code === 11 || code === 12 || (code >= 14 && code <= 31)) return true;
  }
  return false;
}

/** Human byte size for a chip label, from a raw byte count. */
function sizeLabel(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
}

/** A short `HEIC · 32 KB`-style label for a binary blob, sniffing common magics. */
export function describeBinary(text: string): string {
  let format = 'binary';
  if (text.startsWith('%PDF')) format = 'PDF';
  else if (/^.{0,16}ftyp(heic|heix|hevc|mif1|msf1)/.test(text)) format = 'HEIC';
  else if (text.startsWith('\u0089PNG')) format = 'PNG';
  else if (text.startsWith('\u00ff\u00d8\u00ff')) format = 'JPEG';
  else if (text.startsWith('GIF8')) format = 'GIF';
  else if (text.startsWith('PK')) format = 'ZIP';
  return `${format} · ${sizeLabel(text.length)}`;
}

/**
 * Rendered prose for a message. Skips two non-prose payloads a tool can put in a
 * `text` block: raw binary read verbatim (see {@link isBinaryText}), which would
 * otherwise dump kilobytes of garbage. Those are surfaced via
 * {@link extractBinaryNotes} as chips instead.
 */
export function extractText(content: unknown): string {
  if (typeof content === 'string') return isBinaryText(content) ? '' : content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'text') {
      const text = (block as { text?: unknown }).text;
      if (typeof text === 'string' && !isBinaryText(text)) parts.push(text);
    }
  }
  return parts.join('\n');
}

/**
 * A model-facing image-read note: a whole line reading
 * `[Image: original …, displayed at …. Multiply coordinates by … to map …]`.
 * Anchored to the bracket delimiters and one of the tool's fixed clauses so it
 * only ever matches the generated note, never human prose that mentions images.
 */
const IMAGE_DETAIL_RE = /^\s*\[Image:[^\]]*(?:displayed at|Multiply coordinates by)[^\]]*\]\s*$/;

/**
 * Split the coordinate-mapping image note(s) out of a message's prose. The note
 * is plumbing for the model sitting right next to a rendered thumbnail, so the
 * UI keeps it collapsed; the remaining prose is returned as {@link visible}.
 */
export function partitionImageDetails(text: string): { visible: string; notes: string[] } {
  if (text === '' || !text.includes('[Image:')) return { visible: text, notes: [] };
  const notes: string[] = [];
  const kept: string[] = [];
  for (const line of text.split('\n')) {
    if (IMAGE_DETAIL_RE.test(line)) notes.push(line.trim());
    else kept.push(line);
  }
  if (notes.length === 0) return { visible: text, notes };
  return {
    visible: kept
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim(),
    notes,
  };
}

/** Inline-able image blocks (`{type:'image', data, mimeType}`), capped. */
export function extractImages(content: unknown): Array<{ mimeType: string; data: string }> {
  if (!Array.isArray(content)) return [];
  const images: Array<{ mimeType: string; data: string }> = [];
  for (const block of content) {
    if (images.length >= MAX_INLINE_IMAGES) break;
    if (!block || typeof block !== 'object' || (block as { type?: unknown }).type !== 'image') continue;
    const { data, mimeType } = block as { data?: unknown; mimeType?: unknown };
    if (typeof data === 'string' && data.length <= MAX_INLINE_IMAGE_B64) {
      images.push({ mimeType: typeof mimeType === 'string' ? mimeType : 'image/png', data });
    }
  }
  return images;
}

/**
 * Chip labels for binary payloads NOT rendered as prose or thumbnails: raw binary
 * read verbatim into a `text` block, and images too large to inline.
 */
export function extractBinaryNotes(content: unknown): string[] {
  const notes: string[] = [];
  const consider = (block: unknown): void => {
    if (typeof block === 'string') {
      if (isBinaryText(block)) notes.push(describeBinary(block));
      return;
    }
    if (!block || typeof block !== 'object') return;
    const b = block as { type?: unknown; text?: unknown; data?: unknown };
    if (b.type === 'text' && typeof b.text === 'string' && isBinaryText(b.text)) {
      notes.push(describeBinary(b.text));
    } else if (b.type === 'image' && typeof b.data === 'string' && b.data.length > MAX_INLINE_IMAGE_B64) {
      // base64 decodes to ~3/4 its length.
      notes.push(`image · ${sizeLabel(Math.floor((b.data.length * 3) / 4))} (too large to preview)`);
    }
  };
  if (typeof content === 'string') consider(content);
  else if (Array.isArray(content)) for (const block of content) consider(block);
  return notes;
}

/**
 * A one-line digest of a tool call's arguments — the `path` it read/edited, the
 * `command` bash ran, the `agent` a subagent spawned — so a collapsed row can show
 * what it did. Whitespace is flattened and the result capped to bound the payload;
 * the UI truncates the rest to the row width.
 */
export function summarizeToolArgs(args: unknown): string {
  const pick = (): string => {
    if (typeof args === 'string') return args;
    if (args === null || typeof args !== 'object') return '';
    const o = args as Record<string, unknown>;
    // Salient keys first (path covers read/ls/edit/write; command bash; agent
    // subagent), then any first string field as a last resort.
    const keys = [
      'command',
      'path',
      'file_path',
      'filePath',
      'file',
      'pattern',
      'glob',
      'query',
      'url',
      'agent',
      'task',
      'description',
      'name',
    ];
    for (const k of keys) {
      const v = o[k];
      if (typeof v === 'string' && v.trim() !== '') return v;
    }
    for (const v of Object.values(o)) {
      if (typeof v === 'string' && v.trim() !== '') return v;
    }
    return '';
  };
  return pick().replace(/\s+/g, ' ').trim().slice(0, 300);
}

export function extractToolCalls(content: unknown): TranscriptEntry['toolCalls'] {
  if (!Array.isArray(content)) return [];
  const calls: TranscriptEntry['toolCalls'] = [];
  for (const block of content) {
    if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'toolCall') {
      const call = block as { id?: unknown; name?: unknown; arguments?: unknown };
      calls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.name === 'string' ? call.name : '',
        arguments: call.arguments ?? null,
        summary: summarizeToolArgs(call.arguments),
      });
    }
  }
  return calls;
}

/** Map every tool call's `id` → its raw arguments, for joining results back. */
export function collectCallArgs(entries: SessionEntry[]): Map<string, unknown> {
  const map = new Map<string, unknown>();
  for (const entry of entries) {
    if (entry.type !== 'message') continue;
    const content = (entry.message as { content?: unknown }).content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (block && typeof block === 'object' && (block as { type?: unknown }).type === 'toolCall') {
        const call = block as { id?: unknown; arguments?: unknown };
        if (typeof call.id === 'string') map.set(call.id, call.arguments);
      }
    }
  }
  return map;
}

export function mapTranscriptEntry(entry: SessionEntry, callArgs: Map<string, unknown>): TranscriptEntry | null {
  if (entry.type !== 'message') return null;
  const msg = entry.message as {
    role?: unknown;
    content?: unknown;
    timestamp?: unknown;
    toolName?: unknown;
    isError?: unknown;
    toolCallId?: unknown;
    model?: unknown;
  };
  const role = typeof msg.role === 'string' ? msg.role : 'unknown';
  const ts =
    typeof entry.timestamp === 'string' && !Number.isNaN(Date.parse(entry.timestamp))
      ? Date.parse(entry.timestamp)
      : typeof msg.timestamp === 'number'
        ? msg.timestamp
        : 0;
  const { visible, notes: imageNotes } = partitionImageDetails(extractText(msg.content));
  const mapped: TranscriptEntry = {
    role,
    text: visible,
    toolCalls: extractToolCalls(msg.content),
    ts,
  };
  if (role === 'toolResult') {
    if (typeof msg.toolName === 'string') mapped.toolName = msg.toolName;
    if (typeof msg.isError === 'boolean') mapped.isError = msg.isError;
    // Recover what the call acted on (path/command) by joining `toolCallId` back
    // to its originating tool call's arguments.
    if (typeof msg.toolCallId === 'string' && callArgs.has(msg.toolCallId)) {
      const title = summarizeToolArgs(callArgs.get(msg.toolCallId));
      if (title !== '') mapped.title = title;
    }
  }
  if (role === 'assistant' && typeof msg.model === 'string' && msg.model !== '') {
    mapped.model = msg.model;
  }
  const images = extractImages(msg.content);
  if (images.length > 0) mapped.images = images;
  const binaryNotes = extractBinaryNotes(msg.content);
  if (binaryNotes.length > 0) mapped.binaryNotes = binaryNotes;
  if (imageNotes.length > 0) mapped.imageNotes = imageNotes;
  return mapped;
}
