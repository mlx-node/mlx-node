/**
 * @vitest-environment happy-dom
 */

import { describe, expect, it, vi } from 'vite-plus/test';

import {
  ATTACH_MAX,
  FILE_NUMBERED_MAX_BYTES,
  IMAGE_MAX_BYTES,
  IMAGE_SOURCE_MAX_BYTES,
  type ImageMediaType,
  magicMatches,
  numberedBytes,
} from '../src/chat/attachment-rule.js';
import {
  attachQueue,
  type DecodedPicture,
  type DraftAttachment,
  draftInput,
  type ImageCodec,
  ingestFiles,
  noteName,
} from '../ui/src/lib/chat-attachments.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { normalizeMime } from '../ui/src/lib/file-read.js';

const e = CHAT.attach.errors;
const CHAT_IMAGES = { mode: 'chat', seesImages: true, held: 0 } as const;
const CHAT_TEXT = { mode: 'chat', seesImages: false, held: 0 } as const;
const WORK = { mode: 'work', seesImages: false, held: 0 } as const;

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

  it('bad UTF-8, a blank file, a PDF and HEIC bytes each get their sentence', async () => {
    const heic = bytesOf([0x00, 0x00, 0x00, 0x18], 'ftypheic');
    const result = await ingestFiles(
      [
        file([new Uint8Array([0x61, 0xc3, 0x28])], 'bad.txt'),
        file([' \n\t '], 'blank.txt'),
        file(['%PDF-1.7'], 'paper.pdf', 'application/pdf'),
        file([heic], 'photo.heic', 'image/heic'),
      ],
      CHAT_IMAGES,
    );
    expect(result.added).toEqual([]);
    expect(result.refused).toEqual([
      fill(e.notText, { name: 'bad.txt' }),
      fill(e.empty, { name: 'blank.txt' }),
      fill(e.pdf, { name: 'paper.pdf' }),
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
    expect(result.refused).toEqual([fill(e.pdf, { name: 'a.pdf' })]);
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

describe('attachQueue', () => {
  it('two ingests started together add at most 4 in total', async () => {
    const queue = attachQueue();
    let store: readonly DraftAttachment[] = [];
    const target = { read: () => store, write: (items: DraftAttachment[]) => (store = items) };
    const batch = (tag: string): File[] => ['1', '2', '3'].map((n) => file([`${tag} ${n}`], `${tag}${n}.txt`));
    const [first, second] = await Promise.all([
      queue(batch('a'), { mode: 'chat', seesImages: false }, target),
      queue(batch('b'), { mode: 'chat', seesImages: false }, target),
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
    const options = { mode: 'chat', seesImages: false } as const;
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
});
