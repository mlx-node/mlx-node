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
  PAGE_JPEG_QUALITIES,
  PAGE_MAX_BYTES,
  type PageSize,
  PDF_MESSAGE_PAGES,
  PDF_SOURCE_MAX_BYTES,
  pdfPageSize,
  safeName,
  TEXT_SNIFF_BYTES,
  type TokenItem,
  utf8Bytes,
} from '../../../src/chat/attachment-rule.js';
import { CHAT, fill } from './chat-strings';
import type { ChatMode } from './chat-types';
import { normalizeMime, readFileAsDataUrl } from './file-read';
import { browserPdfCodec, type PdfCodec, type PdfDocument, PdfOpenError } from './pdf-pages';
import { nextDraftId } from './system-one';

/** A file read into the composer, ready to send. */
export type DraftAttachment =
  | { id: string; kind: 'file'; name: string; bytes: number; numbered: number; text: string }
  | { id: string; kind: 'image'; name: string; mediaType: ImageMediaType; bytes: number; dataUrl: string }
  /**
   * A PDF as its page images: base64 JPEG per page with no `data:` prefix, and each page's drawn size. `bytes` is the
   * JPEG bytes of every page, `chars` their base64.
   */
  | {
      id: string;
      kind: 'pdf';
      name: string;
      bytes: number;
      chars: number;
      images: readonly string[];
      sizes: readonly PageSize[];
    };

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

/**
 * How far a PDF's read is: pages drawn of its pages, the JPEG bytes of those pages, and the data URL of page 1 once it
 * is drawn.
 */
export interface PdfProgress {
  done: number;
  total: number;
  bytes: number;
  thumbnail?: string;
}

/** The composer's side of a PDF's read, which shows its chip from the moment the read starts. */
export interface PdfWatch {
  /** The read of the PDF that becomes draft `id` starts. The signal aborts when the person removes its chip. */
  start(id: string, name: string): AbortSignal;
  progress(id: string, progress: PdfProgress): void;
  /** The read ended with nothing added: refused or cancelled. */
  end(id: string): void;
}

export interface IngestOptions {
  mode: ChatMode;
  /** Chat and the model row has images = true. */
  seesImages: boolean;
  /** The model row has images = true, in either mode: PDF pages go to it as images. */
  seesPages: boolean;
  watch?: PdfWatch;
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

/** `sentence` null: the person cancelled the read, so nothing is said. */
type Read<T> = { ok: true; value: T } | { ok: false; sentence: string | null };

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

const CANCELLED: Read<never> = { ok: false, sentence: null };

/** A page as it is sent: base64 JPEG, its JPEG bytes and its drawn size; null when it does not draw or fit. */
async function readPage(doc: PdfDocument, n: number): Promise<{ data: string; bytes: number; size: PageSize } | null> {
  let page: Awaited<ReturnType<PdfDocument['page']>>;
  try {
    page = await doc.page(n);
  } catch {
    return null;
  }
  try {
    const size = pdfPageSize(page.width, page.height);
    const { width, height } = size;
    if (!(width >= IMAGE_MIN_SIDE && height >= IMAGE_MIN_SIDE && width * height >= IMAGE_MIN_PIXELS)) return null;
    const drawn = await page.draw(width, height);
    try {
      for (const quality of PAGE_JPEG_QUALITIES) {
        const blob = await drawn.encode(quality);
        if (blob.size > PAGE_MAX_BYTES) continue;
        const url = await readFileAsDataUrl(blob);
        const data = url.slice(url.indexOf(',') + 1);
        return magicMatches('image/jpeg', data) ? { data, bytes: blob.size, size } : null;
      }
      return null;
    } finally {
      drawn.close();
    }
  } catch {
    return null;
  } finally {
    page.close();
  }
}

/** One page per task: the composer keeps drawing its chip and its count between pages. */
function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Every page is drawn again, one after another, at the size the agent's `read_file` passes through untouched, as
 * JPEG at falling quality until it fits. Any page that does not draw or fit refuses the whole PDF. The file's own
 * bytes are never sent.
 */
async function readPdf(
  file: File,
  name: string,
  codec: PdfCodec,
  watch: PdfWatch | undefined,
): Promise<Read<DraftAttachment>> {
  const e = CHAT.attach.errors;
  if (file.size > PDF_SOURCE_MAX_BYTES) return refuse(e.pdfTooLarge, name);
  const id = nextDraftId('pdf');
  const signal = watch?.start(id, name) ?? new AbortController().signal;
  const failed = (read: Read<DraftAttachment>): Read<DraftAttachment> => {
    watch?.end(id);
    return read;
  };
  let doc: PdfDocument;
  try {
    doc = await codec.open(await file.arrayBuffer());
  } catch (err) {
    if (signal.aborted) return failed(CANCELLED);
    const locked = err instanceof PdfOpenError && err.reason === 'locked';
    return failed(refuse(locked ? e.pdfLocked : e.pdfDamaged, name));
  }
  try {
    const total = doc.pages;
    if (signal.aborted) return failed(CANCELLED);
    if (total === 0) return failed(refuse(e.pdfEmpty, name));
    if (total > PDF_MESSAGE_PAGES) {
      return failed({ ok: false, sentence: fill(e.pdfTooLong, { name: noteName(name), n: total }) });
    }
    watch?.progress(id, { done: 0, total, bytes: 0 });
    const images: string[] = [];
    const sizes: PageSize[] = [];
    let bytes = 0;
    let chars = 0;
    for (let n = 1; n <= total; n += 1) {
      const page = await readPage(doc, n);
      if (signal.aborted) return failed(CANCELLED);
      if (page === null) return failed({ ok: false, sentence: fill(e.pdfPage, { name: noteName(name), n }) });
      images.push(page.data);
      sizes.push(page.size);
      bytes += page.bytes;
      chars += page.data.length;
      watch?.progress(
        id,
        n === 1
          ? { done: n, total, bytes, thumbnail: `data:image/jpeg;base64,${page.data}` }
          : { done: n, total, bytes },
      );
      await nextTask();
      if (signal.aborted) return failed(CANCELLED);
    }
    return { ok: true, value: { id, kind: 'pdf', name, bytes, chars, images, sizes } };
  } finally {
    await doc.close().catch(() => undefined);
  }
}

/** A `.pdf` name takes the PDF path whatever type the browser gives, so it can never pass as a text file. */
function isPdf(file: File, type: string): boolean {
  return type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

/**
 * Reads files into drafts, in order, until `held` plus the added ones reach {@link ATTACH_MAX}; the rest get one
 * sentence with their count. A PDF counts as one file. PNG, JPEG and WebP take the picture path; a PDF is drawn page
 * by page for a model that sees images, in either mode; anything else is judged as text by its bytes. A PDF whose
 * chip is removed after its pages are drawn, while later files of the same pick are still read, is not added.
 */
export async function ingestFiles(
  files: readonly File[],
  options: IngestOptions & { held: number },
  codec: ImageCodec = browserCodec,
  pdfCodec: PdfCodec = browserPdfCodec,
): Promise<IngestResult> {
  const e = CHAT.attach.errors;
  const added: DraftAttachment[] = [];
  const refused: string[] = [];
  const stops = new Map<string, AbortSignal>();
  const outer = options.watch;
  const watch: PdfWatch | undefined =
    outer === undefined
      ? undefined
      : {
          start: (id, name) => {
            const signal = outer.start(id, name);
            stops.set(id, signal);
            return signal;
          },
          progress: (id, progress) => outer.progress(id, progress),
          end: (id) => outer.end(id),
        };
  const kept = (): DraftAttachment[] => added.filter((item) => stops.get(item.id)?.aborted !== true);
  for (let i = 0; i < files.length; i += 1) {
    if (options.held + kept().length >= ATTACH_MAX) {
      refused.push(fill(e.tooMany, { n: files.length - i }));
      break;
    }
    const file = files[i]!;
    const name = safeName(file.name);
    const type = normalizeMime(file);
    let read: Read<DraftAttachment>;
    if (isPdf(file, type)) {
      if (!options.seesPages) read = refuse(e.pdfModel, name);
      else {
        try {
          read = await readPdf(file, name, pdfCodec, watch);
        } catch {
          read = refuse(e.pdfDamaged, name);
        }
      }
    } else if (PICTURE_TYPES.has(type)) {
      if (!options.seesImages) read = refuse(options.mode === 'work' ? e.imageWork : e.imageModel, name);
      else read = await readPicture(file, name, type, codec);
    } else {
      try {
        read = await readText(file, name, options.mode);
      } catch {
        read = refuse(e.unreadable, name);
      }
    }
    if (read.ok) added.push(read.value);
    else if (read.sentence !== null) refused.push(read.sentence);
  }
  return { added: kept(), refused };
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
export function attachQueue(codec: ImageCodec = browserCodec, pdfCodec: PdfCodec = browserPdfCodec): AttachQueue {
  let tail: Promise<unknown> = Promise.resolve();
  return (files, options, target) => {
    const run = async (): Promise<IngestResult> => {
      const result = await ingestFiles(files, { ...options, held: target.read().length }, codec, pdfCodec);
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

/** A PDF being drawn, as its chip and the composer's caption read it. */
export interface PdfDrawing {
  done: number;
  total: number;
  /** When its read started and when it last drew a page (epoch ms). */
  startedAt: number;
  at: number;
}

function oneDecimal(n: number): string {
  return n >= 10 ? String(Math.round(n)) : n.toFixed(1);
}

/** Pages per second as words: per second from 1 a second, else per minute. */
export function pageRate(perSecond: number): string {
  return perSecond >= 1
    ? fill(CHAT.pdf.rateSecond, { n: oneDecimal(perSecond) })
    : fill(CHAT.pdf.rateMinute, { n: oneDecimal(perSecond * 60) });
}

/** Elapsed time shows once it passes 1 s, and the rate once 8 pages and 1 s have passed. */
const DRAW_CLOCK_MS = 1_000;
const DRAW_RATE_PAGES = 8;

/** Pages drawn of a PDF's pages, "1 page" for a one-page PDF. */
export function drawCount(done: number, total: number): string {
  return total === 1 ? fill(CHAT.attach.pdfDrawCountOne, { done }) : fill(CHAT.attach.pdfDrawCount, { done, total });
}

/**
 * Of the PDFs whose reads run, the one being drawn now: open, with pages left. A PDF drawn in full stays among them
 * until the rest of its pick is read, and is not the one drawing.
 */
export function drawingNow<T extends { done: number; total: number | null }>(
  pending: readonly T[],
): (T & { total: number }) | undefined {
  return pending.find((entry): entry is T & { total: number } => entry.total !== null && entry.done < entry.total);
}

/**
 * The caption while a PDF draws: pages done of its pages, then the time and the rate when they mean something. The
 * time runs on `now` between pages; the rate is the pages drawn over the time they took.
 */
export function drawCaption(drawing: PdfDrawing, now: number): string {
  const parts = [drawCount(drawing.done, drawing.total)];
  const elapsed = Math.max(now, drawing.at) - drawing.startedAt;
  if (elapsed >= DRAW_CLOCK_MS) {
    parts.push(fill(CHAT.attach.pdfDrawTime, { n: (elapsed / 1_000).toFixed(1) }));
    const drawn = drawing.at - drawing.startedAt;
    if (drawing.done >= DRAW_RATE_PAGES && drawn >= DRAW_CLOCK_MS) parts.push(pageRate(drawing.done / (drawn / 1_000)));
  }
  return parts.join(' · ');
}

/** The wire form of a draft: a picture's `data` is its data URL after the first comma; a PDF sends its pages. */
export function draftInput(draft: DraftAttachment): AttachmentInput {
  if (draft.kind === 'file') return { kind: 'file', name: draft.name, text: draft.text };
  if (draft.kind === 'pdf') return { kind: 'pdf', name: draft.name, images: [...draft.images] };
  return { kind: 'image', mediaType: draft.mediaType, data: draft.dataUrl.slice(draft.dataUrl.indexOf(',') + 1) };
}

/** A draft as the token estimate reads it: a PDF with each page's drawn size. */
export function draftTokens(draft: DraftAttachment): TokenItem {
  return draft.kind === 'pdf'
    ? { kind: 'pdf', name: draft.name, images: draft.images, sizes: draft.sizes }
    : draftInput(draft);
}
