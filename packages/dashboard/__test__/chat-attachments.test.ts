/**
 * @vitest-environment happy-dom
 */

import { describe, expect, it, vi } from 'vite-plus/test';

import {
  ATTACH_MAX,
  CHAT_PICTURES_MAX_BASE64,
  FILE_NUMBERED_MAX_BYTES,
  IMAGE_MAX_BYTES,
  IMAGE_SOURCE_MAX_BYTES,
  type ImageMediaType,
  magicMatches,
  numberedBytes,
  PAGE_MAX_BYTES,
  type PageSize,
  PDF_SOURCE_MAX_BYTES,
} from '../src/chat/attachment-rule.js';
import {
  attachQueue,
  type DecodedPicture,
  type DraftAttachment,
  drawCaption,
  drawCount,
  draftInput,
  drawingNow,
  type ImageCodec,
  ingestFiles,
  noteName,
  pageRate,
  type PdfProgress,
  type PdfWatch,
} from '../ui/src/lib/chat-attachments.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { normalizeMime } from '../ui/src/lib/file-read.js';
import { type PdfCodec, type PdfOpenFailure, PdfOpenError } from '../ui/src/lib/pdf-pages.js';
import { type AttachFacts, attachProblem } from '../ui/src/pages/chat/blockers.js';
import {
  forgetPlace,
  placeEpoch,
  type PendingPdf,
  readPending,
  stopPending,
  writePending,
} from '../ui/src/pages/chat/drafts.js';

const e = CHAT.attach.errors;
const CHAT_IMAGES = { mode: 'chat', seesImages: true, seesPages: true, held: 0 } as const;
const CHAT_TEXT = { mode: 'chat', seesImages: false, seesPages: false, held: 0 } as const;
const WORK = { mode: 'work', seesImages: false, seesPages: false, held: 0 } as const;
const WORK_PAGES = { mode: 'work', seesImages: false, seesPages: true, held: 0 } as const;

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const JPEG_MAGIC = [0xff, 0xd8, 0xff];

function bytesOf(head: readonly number[], marker: string, size = head.length + marker.length): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(size);
  out.set(head, 0);
  for (let i = 0; i < marker.length && head.length + i < size; i += 1) out[head.length + i] = marker.charCodeAt(i);
  return out;
}

function file(parts: BlobPart[], name: string, type = ''): File {
  return new File(parts, name, { type });
}

/** A file whose `size` reads as `size` while its bytes stay small. */
function sized(source: File, size: number): File {
  Object.defineProperty(source, 'size', { value: size });
  return source;
}

interface EncodeCall {
  width: number;
  height: number;
  type: ImageMediaType;
  quality: number | undefined;
}

/** A codec that decodes every file to `width` x `height` and encodes to `sizeOf(call)` bytes marked DRAWN. */
function fakeCodec(
  width: number,
  height: number,
  sizeOf: (call: EncodeCall) => number = () => 1_000,
): { codec: ImageCodec; calls: EncodeCall[]; decoded: Blob[]; closed: () => number } {
  const calls: EncodeCall[] = [];
  const decoded: Blob[] = [];
  let closed = 0;
  const codec: ImageCodec = {
    decode: async (source) => {
      decoded.push(source);
      const picture: DecodedPicture = {
        width,
        height,
        encode: async (w, h, type, quality) => {
          const call = { width: w, height: h, type, quality };
          calls.push(call);
          return new Blob([bytesOf(type === 'image/png' ? PNG_MAGIC : JPEG_MAGIC, 'DRAWN', sizeOf(call))], { type });
        },
        close: () => {
          closed += 1;
        },
      };
      return picture;
    },
  };
  return { codec, calls, decoded, closed: () => closed };
}

/** A page of a fake PDF: its size in points, and the JPEG bytes each quality gives (1,000 by default). */
interface FakePage {
  width: number;
  height: number;
  bytesAt?: (quality: number) => number;
  /** The draw throws. */
  broken?: boolean;
}

const A4: FakePage = { width: 595.28, height: 841.89 };

/** A codec whose every file opens as `pages`, or fails to open with `fails`. */
function fakePdf(
  pages: readonly FakePage[],
  options: { fails?: PdfOpenFailure } = {},
): {
  codec: PdfCodec;
  opened: () => number;
  draws: Array<{ n: number; width: number; height: number }>;
  encodes: Array<{ n: number; quality: number }>;
  closed: () => number;
} {
  let opened = 0;
  let closed = 0;
  const draws: Array<{ n: number; width: number; height: number }> = [];
  const encodes: Array<{ n: number; quality: number }> = [];
  const codec: PdfCodec = {
    open: async () => {
      opened += 1;
      if (options.fails !== undefined) throw new PdfOpenError(options.fails);
      return {
        pages: pages.length,
        page: async (n) => {
          const page = pages[n - 1]!;
          return {
            width: page.width,
            height: page.height,
            draw: async (width, height) => {
              if (page.broken === true) throw new Error('bad operator');
              draws.push({ n, width, height });
              return {
                encode: async (quality) => {
                  encodes.push({ n, quality });
                  const size = page.bytesAt?.(quality) ?? 1_000;
                  return new Blob([bytesOf(JPEG_MAGIC, `PAGE${n}`, size)], { type: 'image/jpeg' });
                },
                close: () => undefined,
              };
            },
            close: () => undefined,
          };
        },
        close: async () => {
          closed += 1;
        },
      };
    },
  };
  return { codec, opened: () => opened, draws, encodes, closed: () => closed };
}

/** A watch that records what the composer is told. */
function recordWatch(onProgress?: (progress: PdfProgress, stop: AbortController) => void): {
  watch: PdfWatch;
  started: string[];
  progress: PdfProgress[];
  ended: string[];
} {
  const started: string[] = [];
  const progress: PdfProgress[] = [];
  const ended: string[] = [];
  const stops = new Map<string, AbortController>();
  const watch: PdfWatch = {
    start: (id) => {
      started.push(id);
      const stop = new AbortController();
      stops.set(id, stop);
      return stop.signal;
    },
    progress: (id, value) => {
      progress.push(value);
      onProgress?.(value, stops.get(id)!);
    },
    end: (id) => ended.push(id),
  };
  return { watch, started, progress, ended };
}

const pdfFile = (name = 'paper.pdf', type = 'application/pdf'): File => file(['%PDF-1.7 body'], name, type);

function decodedData(draft: DraftAttachment): string {
  if (draft.kind !== 'image') throw new Error('not a picture');
  return atob(draft.dataUrl.slice(draft.dataUrl.indexOf(',') + 1));
}

describe('normalizeMime', () => {
  it('the type when the browser gives one, else the extension', () => {
    expect(normalizeMime(file([], 'a.PNG', 'IMAGE/PNG'))).toBe('image/png');
    expect(normalizeMime(file([], 'photo.JPG'))).toBe('image/jpeg');
    expect(normalizeMime(file([], 'x.webp'))).toBe('image/webp');
    expect(normalizeMime(file([], 'doc.pdf'))).toBe('application/pdf');
    expect(normalizeMime(file([], 'notes.txt'))).toBe('');
  });
});

describe('ingestFiles: content decides what is text', () => {
  it('a .ts typed video/mp2t and an .svg typed image/svg+xml are text files', async () => {
    const ts = file(['export const a = 1;\n'], 'a.ts', 'video/mp2t');
    const svg = file(['<svg xmlns="http://www.w3.org/2000/svg"/>'], 'a.svg', 'image/svg+xml');
    const result = await ingestFiles([ts, svg], CHAT_TEXT);
    expect(result.refused).toEqual([]);
    expect(result.added.map((item) => [item.kind, item.name])).toEqual([
      ['file', 'a.ts'],
      ['file', 'a.svg'],
    ]);
    const first = result.added[0]!;
    expect(first).toMatchObject({ kind: 'file', text: 'export const a = 1;\n', bytes: 20 });
    expect(first.kind === 'file' ? first.numbered : -1).toBe(numberedBytes('export const a = 1;\n'));
  });

  it('a zip named .txt and typed text/plain is not a text file', async () => {
    const zip = file([bytesOf([0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00], 'stored')], 'notes.txt', 'text/plain');
    const result = await ingestFiles([zip], CHAT_TEXT);
    expect(result.added).toEqual([]);
    expect(result.refused).toEqual([fill(e.notText, { name: 'notes.txt' })]);
  });

  it('a 40 MB file with a NUL in its first 8,192 bytes: not a text file, and nothing past the first slice is read', async () => {
    const head = new Uint8Array(8_192).fill(0x61);
    head[100] = 0;
    const big = sized(file([head], 'archive.docx', 'application/vnd.openxmlformats'), 40_000_000);
    const whole = vi.fn(() => Promise.reject(new Error('read past the first slice')));
    Object.defineProperty(big, 'arrayBuffer', { value: whole });
    const slice = vi.spyOn(big, 'slice');
    const result = await ingestFiles([big], CHAT_TEXT);
    expect(result.refused).toEqual([fill(e.notText, { name: 'archive.docx' })]);
    expect(whole).not.toHaveBeenCalled();
    expect(slice.mock.calls).toEqual([[0, 8_192]]);
  });

  it('a long text file: over 20,000 bytes, or over 20,000 numbered bytes; Work names its own sentence', async () => {
    const long = file(['a'.repeat(FILE_NUMBERED_MAX_BYTES + 1)], 'long.md');
    expect((await ingestFiles([long], CHAT_TEXT)).refused).toEqual([fill(e.textTooLong, { name: 'long.md' })]);
    expect((await ingestFiles([long], WORK)).refused).toEqual([fill(e.textTooLongWork, { name: 'long.md' })]);
    // 10,000 bytes on disk, but the line numbers take it past the cap.
    const lines = 'a\n'.repeat(5_000);
    expect(lines.length).toBeLessThanOrEqual(FILE_NUMBERED_MAX_BYTES);
    expect(numberedBytes(lines)).toBeGreaterThan(FILE_NUMBERED_MAX_BYTES);
    expect((await ingestFiles([file([lines], 'lines.txt')], CHAT_TEXT)).refused).toEqual([
      fill(e.textTooLong, { name: 'lines.txt' }),
    ]);
    const fits = file(['a'.repeat(FILE_NUMBERED_MAX_BYTES - 4)], 'fits.txt');
    expect((await ingestFiles([fits], CHAT_TEXT)).added).toHaveLength(1);
  });

  it('bad UTF-8, a blank file, a damaged PDF and HEIC bytes each get their sentence', async () => {
    const heic = bytesOf([0x00, 0x00, 0x00, 0x18], 'ftypheic');
    const result = await ingestFiles(
      [
        file([new Uint8Array([0x61, 0xc3, 0x28])], 'bad.txt'),
        file([' \n\t '], 'blank.txt'),
        file(['%PDF-1.7'], 'paper.pdf', 'application/pdf'),
        file([heic], 'photo.heic', 'image/heic'),
      ],
      CHAT_IMAGES,
      undefined,
      fakePdf([], { fails: 'damaged' }).codec,
    );
    expect(result.added).toEqual([]);
    expect(result.refused).toEqual([
      fill(e.notText, { name: 'bad.txt' }),
      fill(e.empty, { name: 'blank.txt' }),
      fill(e.pdfDamaged, { name: 'paper.pdf' }),
      fill(e.notText, { name: 'photo.heic' }),
    ]);
  });

  it('a character cut by the first look at byte 8,192 does not fail it', async () => {
    const text = `${'a'.repeat(8_191)}é and more`;
    const result = await ingestFiles([file([text], 'cut.txt')], CHAT_TEXT);
    expect(result.refused).toEqual([]);
    expect(result.added[0]).toMatchObject({ kind: 'file', text });
  });

  it('a NUL past the first look still refuses the file', async () => {
    const result = await ingestFiles([file([`${'a'.repeat(9_000)}\0`], 'late.txt')], CHAT_TEXT);
    expect(result.refused).toEqual([fill(e.notText, { name: 'late.txt' })]);
  });

  it('names are cleaned and cut in the middle to 32 characters in a sentence', async () => {
    const name = `${'n'.repeat(40)}.bin`;
    const result = await ingestFiles([file([new Uint8Array([0, 1, 2])], `dir/${name}`)], CHAT_TEXT);
    expect(result.refused).toEqual([fill(e.notText, { name: noteName(name) })]);
    expect(noteName(name)).toHaveLength(32);
    expect(noteName(name).endsWith('nnnn.bin')).toBe(true);
    expect(noteName('short.txt')).toBe('short.txt');
    const pair = `${'x'.repeat(22)}😀${'y'.repeat(20)}`;
    expect(noteName(pair).isWellFormed()).toBe(true);
  });
});

describe('ingestFiles: count and image gate', () => {
  it('stops at 4 with held files counted; the rest get one sentence with their count', async () => {
    const texts = (n: number): File[] => Array.from({ length: n }, (_, i) => file([`text ${String(i)}`], `${i}.txt`));
    const five = await ingestFiles(texts(5), CHAT_TEXT);
    expect(five.added).toHaveLength(ATTACH_MAX);
    expect(five.refused).toEqual([fill(e.tooMany, { n: 1 })]);
    const held = await ingestFiles(texts(4), { ...CHAT_TEXT, held: 3 });
    expect(held.added).toHaveLength(1);
    expect(held.refused).toEqual([fill(e.tooMany, { n: 3 })]);
  });

  it('a refused file does not take a place', async () => {
    const result = await ingestFiles(
      [file(['%PDF'], 'a.pdf', 'application/pdf'), ...['1', '2', '3', '4'].map((n) => file([n], `${n}.txt`))],
      CHAT_TEXT,
    );
    expect(result.added).toHaveLength(4);
    expect(result.refused).toEqual([fill(e.pdfModel, { name: 'a.pdf' })]);
  });

  it('a picture with images off: the model sentence in Chat, the Work sentence in Work, never decoded', async () => {
    const { codec, decoded } = fakeCodec(100, 100);
    const png = file([bytesOf(PNG_MAGIC, 'x')], 'shot.png', 'image/png');
    expect((await ingestFiles([png], CHAT_TEXT, codec)).refused).toEqual([fill(e.imageModel, { name: 'shot.png' })]);
    expect((await ingestFiles([png], WORK, codec)).refused).toEqual([fill(e.imageWork, { name: 'shot.png' })]);
    expect(decoded).toEqual([]);
  });
});

describe('ingestFiles: every picture is drawn again', () => {
  it("a small JPEG is still redrawn: the result holds the drawn bytes, never the file's own", async () => {
    const { codec, calls, closed } = fakeCodec(640, 480);
    const own = file([bytesOf(JPEG_MAGIC, 'SOURCE-WITH-EXIF')], 'photo.jpg', 'image/jpeg');
    const result = await ingestFiles([own], CHAT_IMAGES, codec);
    expect(result.refused).toEqual([]);
    const draft = result.added[0]!;
    expect(draft).toMatchObject({ kind: 'image', name: 'photo.jpg', mediaType: 'image/jpeg', bytes: 1_000 });
    expect(calls).toEqual([{ width: 640, height: 480, type: 'image/jpeg', quality: 0.9 }]);
    expect(decodedData(draft)).toContain('DRAWN');
    expect(decodedData(draft)).not.toContain('SOURCE');
    expect(draft.kind === 'image' && draft.dataUrl.startsWith('data:image/jpeg;base64,')).toBe(true);
    expect(closed()).toBe(1);
  });

  it('a file typed image/jpg takes the picture path and leaves as image/jpeg', async () => {
    const { codec, calls } = fakeCodec(64, 64);
    const result = await ingestFiles([file([bytesOf(JPEG_MAGIC, 'j')], 'old.jpg', 'image/jpg')], CHAT_IMAGES, codec);
    expect(result.added[0]).toMatchObject({ kind: 'image', mediaType: 'image/jpeg' });
    expect(calls.map((call) => call.type)).toEqual(['image/jpeg']);
  });

  it('PNG and WebP leave as PNG; the wire data starts with the PNG signature', async () => {
    const { codec, calls } = fakeCodec(300, 200);
    const result = await ingestFiles(
      [file([bytesOf(PNG_MAGIC, 'p')], 'a.png', 'image/png'), file(['RIFFxxxxWEBP'], 'b.webp', 'image/webp')],
      CHAT_IMAGES,
      codec,
    );
    expect(result.added.map((item) => (item.kind === 'image' ? item.mediaType : item.kind))).toEqual([
      'image/png',
      'image/png',
    ]);
    expect(calls.map((call) => call.type)).toEqual(['image/png', 'image/png']);
    const input = draftInput(result.added[0]!);
    expect(input.kind === 'image' && magicMatches('image/png', input.data)).toBe(true);
  });

  it('scale = min(1, 2000 / long side, sqrt(2,400,000 / pixels)), floored', async () => {
    const size = async (width: number, height: number): Promise<[number, number]> => {
      const { codec, calls } = fakeCodec(width, height);
      await ingestFiles([file([bytesOf(JPEG_MAGIC, 'j')], 'p.jpg', 'image/jpeg')], CHAT_IMAGES, codec);
      return [calls[0]!.width, calls[0]!.height];
    };
    expect(await size(1_000, 800)).toEqual([1_000, 800]);
    // Side cap: 4000 x 1000 -> 2000 x 500 (1 MP, under the pixel cap).
    expect(await size(4_000, 1_000)).toEqual([2_000, 500]);
    // Pixel cap: 1900 x 1900 = 3.61 MP -> scale sqrt(2.4 / 3.61).
    const scale = Math.sqrt(2_400_000 / (1_900 * 1_900));
    expect(await size(1_900, 1_900)).toEqual([Math.floor(1_900 * scale), Math.floor(1_900 * scale)]);
    // A 48 MP photo: the pixel cap wins over the side cap.
    const [w, h] = await size(8_000, 6_000);
    expect(w * h).toBeLessThanOrEqual(2_400_000);
    expect(Math.max(w, h)).toBeLessThanOrEqual(2_000);
  });

  it('a PNG over 1,000,000 bytes falls to JPEG at falling quality; over the cap at every quality: imageNoFit', async () => {
    const fallback = fakeCodec(1_500, 1_500, (call) =>
      call.type === 'image/png' ? IMAGE_MAX_BYTES + 1 : call.quality === 0.7 ? IMAGE_MAX_BYTES : IMAGE_MAX_BYTES + 1,
    );
    const png = file([bytesOf(PNG_MAGIC, 'p')], 'big.png', 'image/png');
    const made = await ingestFiles([png], CHAT_IMAGES, fallback.codec);
    expect(made.added[0]).toMatchObject({ mediaType: 'image/jpeg', bytes: IMAGE_MAX_BYTES });
    expect(fallback.calls.map((call) => [call.type, call.quality])).toEqual([
      ['image/png', undefined],
      ['image/jpeg', 0.9],
      ['image/jpeg', 0.8],
      ['image/jpeg', 0.7],
    ]);
    const never = fakeCodec(1_500, 1_500, () => IMAGE_MAX_BYTES + 1);
    const noFit = await ingestFiles([png], CHAT_IMAGES, never.codec);
    expect(noFit.added).toEqual([]);
    expect(noFit.refused).toEqual([fill(e.imageNoFit, { name: 'big.png' })]);
    expect(never.calls.map((call) => call.quality)).toEqual([undefined, 0.9, 0.8, 0.7, 0.5]);
  });

  it('too small, too large before decode, and a decode error', async () => {
    const tiny = fakeCodec(4, 400);
    const jpeg = file([bytesOf(JPEG_MAGIC, 'j')], 'p.jpg', 'image/jpeg');
    expect((await ingestFiles([jpeg], CHAT_IMAGES, tiny.codec)).refused).toEqual([
      fill(e.imageTooSmall, { name: 'p.jpg' }),
    ]);
    expect(tiny.closed()).toBe(1);
    const few = fakeCodec(20, 20);
    expect((await ingestFiles([jpeg], CHAT_IMAGES, few.codec)).refused).toEqual([
      fill(e.imageTooSmall, { name: 'p.jpg' }),
    ]);
    // 4000 x 15 passes at its own size but redraws to 2000 x 7, under the agent's 8 px floor.
    const strip = fakeCodec(4_000, 15);
    const png = file([bytesOf(PNG_MAGIC, 'p')], 'strip.png', 'image/png');
    const thin = await ingestFiles([png], CHAT_IMAGES, strip.codec);
    expect(thin.added).toEqual([]);
    expect(thin.refused).toEqual([fill(e.imageTooSmall, { name: 'strip.png' })]);
    expect(strip.calls).toEqual([]);
    const huge = fakeCodec(100, 100);
    const over = sized(file([bytesOf(JPEG_MAGIC, 'j')], 'raw.jpg', 'image/jpeg'), IMAGE_SOURCE_MAX_BYTES + 1);
    expect((await ingestFiles([over], CHAT_IMAGES, huge.codec)).refused).toEqual([
      fill(e.imageTooLarge, { name: 'raw.jpg' }),
    ]);
    expect(huge.decoded).toEqual([]);
    const broken: ImageCodec = { decode: () => Promise.reject(new Error('bad')) };
    expect((await ingestFiles([jpeg], CHAT_IMAGES, broken)).refused).toEqual([fill(e.unreadable, { name: 'p.jpg' })]);
  });
});

describe('ingestFiles: a PDF is drawn page by page', () => {
  it('a .pdf name typed text/plain takes the PDF path, never the text path', async () => {
    const pdf = fakePdf([A4, A4]);
    const result = await ingestFiles([pdfFile('notes.pdf', 'text/plain')], CHAT_IMAGES, undefined, pdf.codec);
    expect(result.refused).toEqual([]);
    expect(result.added.map((item) => item.kind)).toEqual(['pdf']);
    expect(pdf.opened()).toBe(1);
  });

  it('a model without images refuses the PDF in Chat and in Work, unopened; an image model takes it in Work', async () => {
    const pdf = fakePdf([A4]);
    expect((await ingestFiles([pdfFile()], CHAT_TEXT, undefined, pdf.codec)).refused).toEqual([
      fill(e.pdfModel, { name: 'paper.pdf' }),
    ]);
    expect((await ingestFiles([pdfFile()], WORK, undefined, pdf.codec)).refused).toEqual([
      fill(e.pdfModel, { name: 'paper.pdf' }),
    ]);
    expect(pdf.opened()).toBe(0);
    const work = await ingestFiles([pdfFile()], WORK_PAGES, undefined, pdf.codec);
    expect(work.refused).toEqual([]);
    expect(work.added.map((item) => item.kind)).toEqual(['pdf']);
  });

  it('each page drawn at the page rule and sent as JPEG; the draft holds the pages, their sizes and bytes', async () => {
    const pdf = fakePdf([A4, { width: 612, height: 792 }, { width: 960, height: 540 }]);
    const result = await ingestFiles([pdfFile()], CHAT_IMAGES, undefined, pdf.codec);
    expect(pdf.draws).toEqual([
      { n: 1, width: 861, height: 1_217 },
      { n: 2, width: 900, height: 1_164 },
      { n: 3, width: 1_280, height: 720 },
    ]);
    expect(pdf.encodes.map((call) => call.quality)).toEqual([0.85, 0.85, 0.85]);
    const draft = result.added[0]!;
    if (draft.kind !== 'pdf') throw new Error('not a PDF');
    expect(draft.sizes).toEqual(pdf.draws.map(({ width, height }) => ({ width, height })));
    expect(draft.images).toHaveLength(3);
    expect(draft.images.every((page) => magicMatches('image/jpeg', page))).toBe(true);
    expect(atob(draft.images[1]!)).toContain('PAGE2');
    expect([draft.bytes, draft.chars]).toEqual([3_000, draft.images.join('').length]);
    expect(draftInput(draft)).toEqual({ kind: 'pdf', name: 'paper.pdf', images: draft.images });
    expect(pdf.closed()).toBe(1);
  });

  it('a page over the byte cap at 0.85 falls to 0.7; a page over it at every quality refuses the PDF by page', async () => {
    const fitsAt = (fit: number) => (quality: number) => (quality <= fit ? PAGE_MAX_BYTES : PAGE_MAX_BYTES + 1);
    const ladder = fakePdf([A4, { ...A4, bytesAt: fitsAt(0.7) }]);
    const made = await ingestFiles([pdfFile()], CHAT_IMAGES, undefined, ladder.codec);
    expect(made.refused).toEqual([]);
    expect(ladder.encodes).toEqual([
      { n: 1, quality: 0.85 },
      { n: 2, quality: 0.85 },
      { n: 2, quality: 0.7 },
    ]);
    const never = fakePdf([A4, A4, { ...A4, bytesAt: () => PAGE_MAX_BYTES + 1 }]);
    const refused = await ingestFiles([pdfFile()], CHAT_IMAGES, undefined, never.codec);
    expect(refused.added).toEqual([]);
    expect(refused.refused).toEqual([fill(e.pdfPage, { name: 'paper.pdf', n: 3 })]);
    expect(never.encodes.filter((call) => call.n === 3).map((call) => call.quality)).toEqual([0.85, 0.7, 0.5]);
    expect(never.closed()).toBe(1);
  });

  it('a page that does not draw, or draws under 8 px a side, refuses the PDF with its number', async () => {
    const broken = await ingestFiles([pdfFile()], CHAT_IMAGES, undefined, fakePdf([A4, { ...A4, broken: true }]).codec);
    expect(broken.refused).toEqual([fill(e.pdfPage, { name: 'paper.pdf', n: 2 })]);
    const sliver = await ingestFiles([pdfFile()], CHAT_IMAGES, undefined, fakePdf([{ width: 0.5, height: 300 }]).codec);
    expect(sliver.refused).toEqual([fill(e.pdfPage, { name: 'paper.pdf', n: 1 })]);
  });

  it('locked, damaged, empty, over 50 MB and over 20 pages: one sentence each, nothing drawn', async () => {
    const at = async (codec: PdfCodec, source: File = pdfFile()) =>
      (await ingestFiles([source], CHAT_IMAGES, undefined, codec)).refused;
    expect(await at(fakePdf([], { fails: 'locked' }).codec)).toEqual([fill(e.pdfLocked, { name: 'paper.pdf' })]);
    expect(await at(fakePdf([], { fails: 'damaged' }).codec)).toEqual([fill(e.pdfDamaged, { name: 'paper.pdf' })]);
    expect(await at(fakePdf([]).codec)).toEqual([fill(e.pdfEmpty, { name: 'paper.pdf' })]);
    const big = fakePdf([A4]);
    expect(await at(big.codec, sized(pdfFile(), PDF_SOURCE_MAX_BYTES + 1))).toEqual([
      fill(e.pdfTooLarge, { name: 'paper.pdf' }),
    ]);
    expect(big.opened()).toBe(0);
    for (const count of [21, 321]) {
      const long = fakePdf(Array.from({ length: count }, () => A4));
      expect(await at(long.codec)).toEqual([fill(e.pdfTooLong, { name: 'paper.pdf', n: count })]);
      expect(long.draws).toEqual([]);
    }
    const twenty = await ingestFiles([pdfFile()], CHAT_IMAGES, undefined, fakePdf(Array(20).fill(A4)).codec);
    expect(twenty.added).toHaveLength(1);
  });

  it('the watch: the chip starts before the open, counts every page, gets page 1, and ends only on a refusal', async () => {
    const seen = recordWatch();
    const result = await ingestFiles(
      [pdfFile()],
      { ...CHAT_IMAGES, watch: seen.watch },
      undefined,
      fakePdf([A4, A4]).codec,
    );
    expect(seen.started).toEqual([result.added[0]!.id]);
    expect(seen.progress.map(({ done, total, bytes }) => [done, total, bytes])).toEqual([
      [0, 2, 0],
      [1, 2, 1_000],
      [2, 2, 2_000],
    ]);
    expect(seen.progress[1]!.thumbnail?.startsWith('data:image/jpeg;base64,/9j/')).toBe(true);
    expect(seen.progress[2]!.thumbnail).toBeUndefined();
    expect(seen.ended).toEqual([]);
    const locked = recordWatch();
    await ingestFiles(
      [pdfFile()],
      { ...CHAT_IMAGES, watch: locked.watch },
      undefined,
      fakePdf([], { fails: 'locked' }).codec,
    );
    expect(locked.ended).toEqual(locked.started);
  });

  it('removing the chip after page 3 stops the read: nothing added, nothing said, no page 4', async () => {
    const pdf = fakePdf(Array(6).fill(A4));
    const seen = recordWatch((progress, stop) => {
      if (progress.done === 3) stop.abort();
    });
    const result = await ingestFiles([pdfFile()], { ...CHAT_IMAGES, watch: seen.watch }, undefined, pdf.codec);
    expect(result).toEqual({ added: [], refused: [] });
    expect(pdf.draws.map((call) => call.n)).toEqual([1, 2, 3]);
    expect(seen.ended).toEqual(seen.started);
    expect(pdf.closed()).toBe(1);
  });

  it('a PDF removed once drawn, while the rest of its pick is read, is not added and frees its place', async () => {
    const stops: AbortController[] = [];
    const watch: PdfWatch = {
      start: () => {
        const stop = new AbortController();
        stops.push(stop);
        return stop.signal;
      },
      progress: (_id, progress) => {
        if (stops.length === 2 && progress.done === 1) stops[0]!.abort();
      },
      end: () => undefined,
    };
    const result = await ingestFiles(
      [pdfFile('a.pdf'), pdfFile('b.pdf'), file(['x'], 'x.txt')],
      { ...CHAT_IMAGES, held: 2, watch },
      undefined,
      fakePdf([A4, A4]).codec,
    );
    expect(result.added.map((item) => item.name)).toEqual(['b.pdf', 'x.txt']);
    expect(result.refused).toEqual([]);
  });

  it('a PDF is one of the 4 files, whatever its pages', async () => {
    const texts = ['1', '2', '3'].map((n) => file([n], `${n}.txt`));
    const result = await ingestFiles([pdfFile(), ...texts], CHAT_IMAGES, undefined, fakePdf(Array(12).fill(A4)).codec);
    expect(result.refused).toEqual([]);
    expect(result.added.map((item) => item.kind)).toEqual(['pdf', 'file', 'file', 'file']);
  });
});

describe('pending PDFs', () => {
  const entry = (id: string): PendingPdf => ({
    id,
    name: `${id}.pdf`,
    done: 0,
    total: null,
    bytes: 0,
    thumbnail: null,
    startedAt: 1,
    at: 1,
    stop: new AbortController(),
  });

  it('removing a chip stops its read and drops it; an id with no read is false', () => {
    const a = entry('a');
    const b = entry('b');
    writePending('pending-1', 'a', a);
    writePending('pending-1', 'b', b);
    writePending('pending-1', 'a', { ...a, done: 2, total: 5 });
    expect(readPending('pending-1').map((item) => [item.id, item.done])).toEqual([
      ['a', 2],
      ['b', 0],
    ]);
    expect(stopPending('pending-1', 'a')).toBe(true);
    expect(a.stop.signal.aborted).toBe(true);
    expect(b.stop.signal.aborted).toBe(false);
    expect(readPending('pending-1').map((item) => item.id)).toEqual(['b']);
    expect(stopPending('pending-1', 'a')).toBe(false);
    forgetPlace('pending-1');
  });

  it('a forgotten place stops every read in it, and a read that started before keeps no chip there', () => {
    const a = entry('a');
    writePending('pending-2', 'a', a);
    const epoch = placeEpoch('pending-2');
    forgetPlace('pending-2');
    expect(a.stop.signal.aborted).toBe(true);
    expect(readPending('pending-2')).toEqual([]);
    writePending('pending-2', 'a', { ...a, done: 1, total: 3 }, epoch);
    expect(readPending('pending-2')).toEqual([]);
  });
});

describe('drawCaption, drawingNow and pageRate', () => {
  it('pages drawn of all; the time once past 1 s; the rate once 8 pages and 1 s have passed', () => {
    const at = (done: number, ms: number) =>
      drawCaption({ done, total: 20, startedAt: 5_000, at: 5_000 + ms }, 5_000 + ms);
    expect(at(5, 900)).toBe(fill(CHAT.attach.pdfDrawCount, { done: 5, total: 20 }));
    expect(at(5, 1_100)).toBe(`${fill(CHAT.attach.pdfDrawCount, { done: 5, total: 20 })} · 1.1 s`);
    expect(at(8, 2_000)).toBe(`${fill(CHAT.attach.pdfDrawCount, { done: 8, total: 20 })} · 2.0 s · 4.0 pages/s`);
    expect(at(12, 800)).toBe(fill(CHAT.attach.pdfDrawCount, { done: 12, total: 20 }));
  });

  it('the time runs between pages; the rate stays the pages drawn over their time', () => {
    const drawing = { done: 8, total: 20, startedAt: 5_000, at: 7_000 };
    const count = fill(CHAT.attach.pdfDrawCount, { done: 8, total: 20 });
    expect(drawCaption(drawing, 9_500)).toBe(`${count} · 4.5 s · 4.0 pages/s`);
    expect(drawCaption({ done: 0, total: 20, startedAt: 5_000, at: 5_000 }, 7_300)).toBe(
      `${fill(CHAT.attach.pdfDrawCount, { done: 0, total: 20 })} · 2.3 s`,
    );
    expect(drawCaption({ done: 9, total: 20, startedAt: 5_000, at: 5_600 }, 6_200)).toBe(
      `${fill(CHAT.attach.pdfDrawCount, { done: 9, total: 20 })} · 1.2 s`,
    );
  });

  it('a one-page PDF counts "of 1 page"', () => {
    expect(drawCount(0, 1)).toBe('0 of 1 page');
    expect(drawCaption({ done: 1, total: 1, startedAt: 5_000, at: 5_100 }, 5_100)).toBe(
      fill(CHAT.attach.pdfDrawCountOne, { done: 1 }),
    );
    expect(drawCount(1, 2)).toBe(fill(CHAT.attach.pdfDrawCount, { done: 1, total: 2 }));
  });

  it('the PDF drawing now is the open one with pages left, not one drawn in full or one not open yet', () => {
    const at = (done: number, total: number | null) => ({ done, total });
    expect(drawingNow([at(3, 3), at(1, 5)])).toEqual(at(1, 5));
    expect(drawingNow([at(0, null), at(3, 3)])).toBeUndefined();
    expect(drawingNow([at(0, 4)])).toEqual(at(0, 4));
  });

  it('per second from 1 a second, else per minute; whole numbers from 10', () => {
    expect(pageRate(34.2)).toBe('34 pages/s');
    expect(pageRate(1)).toBe('1.0 pages/s');
    expect(pageRate(9.4 / 60)).toBe('9.4 pages/min');
    expect(pageRate(0.5)).toBe('30 pages/min');
  });
});

/** A PDF draft of `pages` A4 pages, each `chars` of base64. */
function pdfDraft(name: string, pages: number, chars = 200_000, size: PageSize = { width: 861, height: 1_217 }) {
  return {
    id: `pdf-${name}`,
    kind: 'pdf' as const,
    name,
    bytes: (pages * chars * 3) / 4,
    chars: pages * chars,
    images: Array<string>(pages).fill('/9j/AAAA'),
    sizes: Array<PageSize>(pages).fill(size),
  };
}

const PICTURE_DRAFT: DraftAttachment = {
  id: 'pic',
  kind: 'image',
  name: 'p.png',
  mediaType: 'image/png',
  bytes: 3_000_000,
  dataUrl: `data:image/png;base64,${'A'.repeat(4_000_000)}`,
};

const FACTS: AttachFacts = {
  mode: 'chat',
  seesImages: true,
  seesPages: true,
  text: 'What is on these pages?',
  files: [],
  windowTokens: null,
  family: 'qwen3_5',
  contextUsed: null,
  reading: 0,
  heldPictures: 0,
  heldImages: 0,
  pdfTool: false,
};

describe('attachProblem with PDFs', () => {
  it('first match wins: reading, then the model, then two PDFs for the tool, then the picture room', () => {
    const one = pdfDraft('a.pdf', 5);
    expect(attachProblem({ ...FACTS, files: [one] })).toBeNull();
    expect(attachProblem({ ...FACTS, files: [one], seesPages: false, reading: 1 })).toBe(CHAT.attach.reading);
    expect(attachProblem({ ...FACTS, files: [one], seesPages: false, heldImages: 60 })).toBe(
      CHAT.attach.blockedPdfModel,
    );
    const tool = { ...FACTS, pdfTool: true, heldImages: 60 };
    expect(attachProblem({ ...tool, files: [one, pdfDraft('b.pdf', 5)] })).toBe(CHAT.attach.blockedPdfOne);
    expect(attachProblem({ ...tool, files: [one] })).toBeNull();
    const full = { ...FACTS, heldPictures: CHAT_PICTURES_MAX_BASE64 - 1_000_000 };
    expect(attachProblem({ ...full, files: [PICTURE_DRAFT] })).toBe(CHAT.sendErrors['pictures-full'][0]);
  });

  it('a Work PDF on an image model passes while a Work picture is still refused', () => {
    const work = { ...FACTS, mode: 'work' as const, seesImages: false };
    expect(attachProblem({ ...work, files: [pdfDraft('a.pdf', 3)] })).toBeNull();
    expect(attachProblem({ ...work, files: [pdfDraft('a.pdf', 3), PICTURE_DRAFT] })).toBe(CHAT.attach.blockedWork);
  });

  it('with no tool in this mode, pages that do not fit the message keep Send off', () => {
    expect(attachProblem({ ...FACTS, files: [pdfDraft('a.pdf', 12), pdfDraft('b.pdf', 9)] })).toBe(
      CHAT.attach.blockedPdfRoom,
    );
    expect(attachProblem({ ...FACTS, files: [pdfDraft('a.pdf', 12), pdfDraft('b.pdf', 8)] })).toBeNull();
    expect(attachProblem({ ...FACTS, files: [pdfDraft('a.pdf', 5)], heldImages: 60 })).toBe(CHAT.attach.blockedPdfRoom);
    expect(attachProblem({ ...FACTS, files: [pdfDraft('a.pdf', 5)], windowTokens: 6_000, contextUsed: 1_000 })).toBe(
      CHAT.attach.blockedPdfRoom,
    );
  });
});

describe('attachQueue', () => {
  it('two ingests started together add at most 4 in total', async () => {
    const queue = attachQueue();
    let store: readonly DraftAttachment[] = [];
    const target = { read: () => store, write: (items: DraftAttachment[]) => (store = items) };
    const batch = (tag: string): File[] => ['1', '2', '3'].map((n) => file([`${tag} ${n}`], `${tag}${n}.txt`));
    const [first, second] = await Promise.all([
      queue(batch('a'), { mode: 'chat', seesImages: false, seesPages: false }, target),
      queue(batch('b'), { mode: 'chat', seesImages: false, seesPages: false }, target),
    ]);
    expect(store.map((item) => item.name)).toEqual(['a1.txt', 'a2.txt', 'a3.txt', 'b1.txt']);
    expect(first.refused).toEqual([]);
    expect(second.added.map((item) => item.name)).toEqual(['b1.txt']);
    expect(second.refused).toEqual([fill(e.tooMany, { n: 2 })]);
  });

  /** A codec whose decodes wait for `release`. */
  function gatedCodec(): { codec: ImageCodec; release: () => void } {
    const inner = fakeCodec(800, 600).codec;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    return { codec: { decode: async (source) => gate.then(() => inner.decode(source)) }, release };
  }
  const photo = (name: string): File => file([new Uint8Array(PNG_MAGIC)], name, 'image/png');
  const texts = (n: number): File[] => Array.from({ length: n }, (_, i) => file([`text ${i}`], `t${i}.txt`));

  // Mutation: drop the cap at commit -> the store holds 5.
  it('a read that ends after another queue filled the store is cut at the commit', async () => {
    const gated = gatedCodec();
    let store: readonly DraftAttachment[] = [];
    const target = { read: () => store, write: (items: DraftAttachment[]) => (store = items) };
    const late = attachQueue(gated.codec)([photo('late.png')], CHAT_IMAGES, target);
    await attachQueue()(texts(4), CHAT_TEXT, target);
    expect(store).toHaveLength(ATTACH_MAX);
    gated.release();
    const result = await late;
    expect(store.map((item) => item.name)).toEqual(['t0.txt', 't1.txt', 't2.txt', 't3.txt']);
    expect(result.added).toEqual([]);
    expect(result.refused).toEqual([fill(e.tooMany, { n: 1 })]);
  });

  it('files written beside the queue while a read runs keep their place; the read is cut', async () => {
    const gated = gatedCodec();
    let store: readonly DraftAttachment[] = [];
    const target = { read: () => store, write: (items: DraftAttachment[]) => (store = items) };
    const reading = attachQueue(gated.codec)([photo('a.png'), photo('b.png'), photo('c.png')], CHAT_IMAGES, target);
    await new Promise((resolve) => setTimeout(resolve, 0));
    store = (await ingestFiles(texts(2), CHAT_TEXT)).added;
    gated.release();
    const result = await reading;
    expect(store.map((item) => item.name)).toEqual(['t0.txt', 't1.txt', 'a.png', 'b.png']);
    expect(result.added.map((item) => item.name)).toEqual(['a.png', 'b.png']);
    expect(result.refused).toEqual([fill(e.tooMany, { n: 1 })]);
  });

  it('a failed ingest does not stop the next one', async () => {
    const queue = attachQueue();
    let store: readonly DraftAttachment[] = [];
    let fail = true;
    const target = {
      read: () => {
        if (fail) {
          fail = false;
          throw new Error('store gone');
        }
        return store;
      },
      write: (items: DraftAttachment[]) => (store = items),
    };
    const options = { mode: 'chat', seesImages: false, seesPages: false } as const;
    const [failed, next] = await Promise.allSettled([
      queue([file(['x'], 'x.txt')], options, target),
      queue([file(['y'], 'y.txt')], options, target),
    ]);
    expect(failed.status).toBe('rejected');
    expect(next.status).toBe('fulfilled');
    expect(store.map((item) => item.name)).toEqual(['y.txt']);
  });
});

describe('draftInput', () => {
  it('a file sends its name and text; a picture its data after the first comma', () => {
    expect(draftInput({ id: 'f', kind: 'file', name: 'a.ts', bytes: 1, numbered: 5, text: 'a' })).toEqual({
      kind: 'file',
      name: 'a.ts',
      text: 'a',
    });
    expect(
      draftInput({
        id: 'i',
        kind: 'image',
        name: 'p.png',
        mediaType: 'image/png',
        bytes: 3,
        dataUrl: 'data:image/png;base64,iVBORw0KGgo=',
      }),
    ).toEqual({ kind: 'image', mediaType: 'image/png', data: 'iVBORw0KGgo=' });
  });

  it('a PDF sends its name and its pages only: the route takes exact keys, and the sizes stay in the page', () => {
    expect(draftInput(pdfDraft('r.pdf', 2))).toEqual({ kind: 'pdf', name: 'r.pdf', images: ['/9j/AAAA', '/9j/AAAA'] });
  });
});
