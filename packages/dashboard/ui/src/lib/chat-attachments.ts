import {
  ATTACH_MAX,
  type AttachmentInput,
  FILE_NUMBERED_MAX_BYTES,
  IMAGE_MAX_BYTES,
  IMAGE_MAX_PIXELS,
  IMAGE_MAX_SIDE,
  IMAGE_MIN_PIXELS,
  IMAGE_MIN_SIDE,
  IMAGE_SOURCE_MAX_BYTES,
  type ImageMediaType,
  magicMatches,
  numberedBytes,
  safeName,
  TEXT_SNIFF_BYTES,
  utf8Bytes,
} from '../../../src/chat/attachment-rule.js';
import { CHAT, fill } from './chat-strings';
import type { ChatMode } from './chat-types';
import { normalizeMime, readFileAsDataUrl } from './file-read';
import { nextDraftId } from './system-one';

/** A file read into the composer, ready to send. */
export type DraftAttachment =
  | { id: string; kind: 'file'; name: string; bytes: number; numbered: number; text: string }
  | { id: string; kind: 'image'; name: string; mediaType: ImageMediaType; bytes: number; dataUrl: string };

/** A picture decoded upright (the photo's orientation applied). */
export interface DecodedPicture {
  width: number;
  height: number;
  /** The picture drawn again at this size and encoded; JPEG on a white background. */
  encode(width: number, height: number, type: ImageMediaType, quality?: number): Promise<Blob>;
  close(): void;
}

export interface ImageCodec {
  /** Throws when the bytes are not a picture the page can decode. */
  decode(file: Blob): Promise<DecodedPicture>;
}

export interface IngestOptions {
  mode: ChatMode;
  /** Chat and the model row has images = true. */
  seesImages: boolean;
}

export interface IngestResult {
  added: DraftAttachment[];
  /** One sentence per refused file, in order; files past the count cap share one sentence. */
  refused: string[];
}

/** Names in a refusal are cut in the middle to this many characters. */
const NOTE_NAME_MAX = 32;
const NOTE_NAME_TAIL = 8;
const JPEG_QUALITIES = [0.9, 0.8, 0.7, 0.5] as const;
const PICTURE_TYPES = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp']);

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

/** `name` cut in the middle to {@link NOTE_NAME_MAX} characters, with no surrogate pair split. */
export function noteName(name: string): string {
  if (name.length <= NOTE_NAME_MAX) return name;
  let head = NOTE_NAME_MAX - NOTE_NAME_TAIL - 1;
  if (isHighSurrogate(name.charCodeAt(head - 1))) head -= 1;
  let tail = name.length - NOTE_NAME_TAIL;
  if (isLowSurrogate(name.charCodeAt(tail))) tail += 1;
  return `${name.slice(0, head)}…${name.slice(tail)}`;
}

const browserPicture = (bitmap: ImageBitmap): DecodedPicture => ({
  width: bitmap.width,
  height: bitmap.height,
  encode: (width, height, type, quality) =>
    new Promise<Blob>((resolve, reject) => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx === null) {
        reject(new Error('No 2D canvas'));
        return;
      }
      if (type === 'image/jpeg') {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);
      }
      ctx.drawImage(bitmap, 0, 0, width, height);
      canvas.toBlob((blob) => (blob === null ? reject(new Error('Could not encode')) : resolve(blob)), type, quality);
    }),
  close: () => bitmap.close(),
});

export const browserCodec: ImageCodec = {
  decode: async (file) => browserPicture(await createImageBitmap(file, { imageOrientation: 'from-image' })),
};

function hasNul(bytes: Uint8Array): boolean {
  for (const byte of bytes) if (byte === 0) return true;
  return false;
}

type Read<T> = { ok: true; value: T } | { ok: false; sentence: string };

const refuse = <T>(template: string, name: string): Read<T> => ({
  ok: false,
  sentence: fill(template, { name: noteName(name) }),
});

/** Content decides: a NUL or bad UTF-8 in the first look is not text, whatever the type or the extension says. */
async function readText(file: File, name: string, mode: ChatMode): Promise<Read<DraftAttachment>> {
  const e = CHAT.attach.errors;
  const tooLong = mode === 'work' ? e.textTooLongWork : e.textTooLong;
  const head = new Uint8Array(await file.slice(0, TEXT_SNIFF_BYTES).arrayBuffer());
  if (hasNul(head)) return refuse(e.notText, name);
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(head, { stream: true });
  } catch {
    return refuse(e.notText, name);
  }
  if (file.size > FILE_NUMBERED_MAX_BYTES) return refuse(tooLong, name);
  let text: string;
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer());
  } catch {
    return refuse(e.notText, name);
  }
  if (text.includes('\0')) return refuse(e.notText, name);
  if (text.trim() === '') return refuse(e.empty, name);
  const numbered = numberedBytes(text);
  if (numbered > FILE_NUMBERED_MAX_BYTES) return refuse(tooLong, name);
  return { ok: true, value: { id: nextDraftId('file'), kind: 'file', name, bytes: utf8Bytes(text), numbered, text } };
}

/** The encoded bytes as a data URL of `type`, or null when they are not that type's bytes. */
async function encoded(picture: DecodedPicture, width: number, height: number, type: ImageMediaType, quality?: number) {
  const blob = await picture.encode(width, height, type, quality);
  const url = await readFileAsDataUrl(blob);
  const data = url.slice(url.indexOf(',') + 1);
  return magicMatches(type, data) ? { bytes: blob.size, dataUrl: `data:${type};base64,${data}` } : null;
}

/**
 * Every picture is decoded and drawn again: upright, no metadata, inside the side and pixel caps. PNG and WebP leave
 * as PNG; a JPEG, or a PNG over the byte cap, leaves as JPEG at falling quality. The file's own bytes are never sent.
 */
async function readPicture(
  file: File,
  name: string,
  source: string,
  codec: ImageCodec,
): Promise<Read<DraftAttachment>> {
  const e = CHAT.attach.errors;
  if (file.size > IMAGE_SOURCE_MAX_BYTES) return refuse(e.imageTooLarge, name);
  let picture: DecodedPicture;
  try {
    picture = await codec.decode(file);
  } catch {
    return refuse(e.unreadable, name);
  }
  try {
    const { width, height } = picture;
    const pixels = width * height;
    const scale = Math.min(1, IMAGE_MAX_SIDE / Math.max(width, height), Math.sqrt(IMAGE_MAX_PIXELS / pixels));
    const w = Math.max(1, Math.floor(width * scale));
    const h = Math.max(1, Math.floor(height * scale));
    // The agent's floor runs on the drawn size: a 4000 x 15 strip draws as 2000 x 7.
    if (w < IMAGE_MIN_SIDE || h < IMAGE_MIN_SIDE || w * h < IMAGE_MIN_PIXELS) return refuse(e.imageTooSmall, name);
    const made = (out: { bytes: number; dataUrl: string }, mediaType: ImageMediaType): Read<DraftAttachment> => ({
      ok: true,
      value: { id: nextDraftId('image'), kind: 'image', name, mediaType, ...out },
    });
    if (source !== 'image/jpeg' && source !== 'image/jpg') {
      const png = await encoded(picture, w, h, 'image/png');
      if (png === null) return refuse(e.unreadable, name);
      if (png.bytes <= IMAGE_MAX_BYTES) return made(png, 'image/png');
    }
    for (const quality of JPEG_QUALITIES) {
      const jpeg = await encoded(picture, w, h, 'image/jpeg', quality);
      if (jpeg === null) return refuse(e.unreadable, name);
      if (jpeg.bytes <= IMAGE_MAX_BYTES) return made(jpeg, 'image/jpeg');
    }
    return refuse(e.imageNoFit, name);
  } catch {
    return refuse(e.unreadable, name);
  } finally {
    picture.close();
  }
}

/**
 * Reads files into drafts, in order, until `held` plus the added ones reach {@link ATTACH_MAX}; the rest get one
 * sentence with their count. PNG, JPEG and WebP take the picture path; a PDF is refused; anything else is judged as
 * text by its bytes.
 */
export async function ingestFiles(
  files: readonly File[],
  options: IngestOptions & { held: number },
  codec: ImageCodec = browserCodec,
): Promise<IngestResult> {
  const e = CHAT.attach.errors;
  const added: DraftAttachment[] = [];
  const refused: string[] = [];
  for (let i = 0; i < files.length; i += 1) {
    if (options.held + added.length >= ATTACH_MAX) {
      refused.push(fill(e.tooMany, { n: files.length - i }));
      break;
    }
    const file = files[i]!;
    const name = safeName(file.name);
    const type = normalizeMime(file);
    let read: Read<DraftAttachment>;
    if (PICTURE_TYPES.has(type)) {
      if (!options.seesImages) read = refuse(options.mode === 'work' ? e.imageWork : e.imageModel, name);
      else read = await readPicture(file, name, type, codec);
    } else if (type === 'application/pdf') read = refuse(e.pdf, name);
    else {
      try {
        read = await readText(file, name, options.mode);
      } catch {
        read = refuse(e.unreadable, name);
      }
    }
    if (read.ok) added.push(read.value);
    else refused.push(read.sentence);
  }
  return { added, refused };
}

/** The place's attachments, read and written by one queued ingest. */
export interface AttachTarget {
  read(): readonly DraftAttachment[];
  write(items: DraftAttachment[]): void;
}

export type AttachQueue = (
  files: readonly File[],
  options: IngestOptions,
  target: AttachTarget,
) => Promise<IngestResult>;

/**
 * One queue for every pick, paste and drop into a store: each ingest reads the held count when its turn starts and
 * writes before the next one starts. A write made beside the queue while a read runs (a failed send put back) keeps its
 * place: the commit takes only what still fits under {@link ATTACH_MAX} and refuses the rest.
 */
export function attachQueue(codec: ImageCodec = browserCodec): AttachQueue {
  let tail: Promise<unknown> = Promise.resolve();
  return (files, options, target) => {
    const run = async (): Promise<IngestResult> => {
      const result = await ingestFiles(files, { ...options, held: target.read().length }, codec);
      const room = Math.max(0, ATTACH_MAX - target.read().length);
      const added = result.added.slice(0, room);
      if (added.length > 0) target.write([...target.read(), ...added]);
      const cut = result.added.length - added.length;
      if (cut === 0) return result;
      return { added, refused: [...result.refused, fill(CHAT.attach.errors.tooMany, { n: cut })] };
    };
    const next = tail.then(run);
    tail = next.catch(() => undefined);
    return next;
  };
}

/** The wire form of a draft: a picture's `data` is its data URL after the first comma. */
export function draftInput(draft: DraftAttachment): AttachmentInput {
  return draft.kind === 'file'
    ? { kind: 'file', name: draft.name, text: draft.text }
    : { kind: 'image', mediaType: draft.mediaType, data: draft.dataUrl.slice(draft.dataUrl.indexOf(',') + 1) };
}
