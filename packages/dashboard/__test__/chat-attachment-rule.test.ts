import { describe, expect, it } from 'vite-plus/test';

import {
  attachmentUri,
  attachmentView,
  base64Bytes,
  CHAT_PICTURES_MAX_BASE64,
  estimateTokens,
  FILE_NUMBERED_MAX_BYTES,
  heldImages,
  IMAGE_MAX_BASE64_CHARS,
  IMAGE_MAX_BYTES,
  IMAGE_TOKENS,
  isBase64,
  isPdfNoteText,
  magicMatches,
  MESSAGE_IMAGES_MAX_BASE64,
  NAME_MAX_CHARS,
  nameOfUri,
  numberedBytes,
  PAGE_MAX_BASE64_CHARS,
  PAGE_MAX_BYTES,
  pageTokens,
  type PageSize,
  pdfNote,
  type PdfDraft,
  type PdfPlanInput,
  pdfPageSize,
  planPdfs,
  readPdfNote,
  safeName,
  storedPictureChars,
  type UserAttachment,
  utf8Bytes,
  wellFormed,
} from '../src/chat/attachment-rule.js';

/** Byte length of the agent's rendering, given the lines Rust `str::lines()` yields. */
function rendered(lines: string[]): number {
  return Buffer.byteLength(lines.map((line, i) => `${i + 1}→${line}`).join('\n'), 'utf8');
}

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48]).toString('base64');
const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46, 0, 1]).toString('base64');
const WEBP = Buffer.from('RIFF\u0000\u0000\u0000\u0000WEBPVP8 ', 'latin1').toString('base64');

describe('utf8Bytes', () => {
  it('counts as the UTF-8 encoder does', () => {
    for (const text of ['', 'a', 'é', '中文', '😀', 'a\u0000b', '\ud800', 'x\udc00y']) {
      expect(utf8Bytes(text)).toBe(new TextEncoder().encode(text).length);
    }
  });
});

describe('numberedBytes', () => {
  it('matches the agent rendering line by line', () => {
    const cases: Array<[string, string[]]> = [
      ['', []],
      ['a', ['a']],
      ['a\n', ['a']],
      ['a\r\nb', ['a', 'b']],
      ['a\n\nb\n', ['a', '', 'b']],
      ['\n', ['']],
      ['a\rb', ['a\rb']],
      ['a\r', ['a\r']],
      ['第一行\n第二行\r\n第三行', ['第一行', '第二行', '第三行']],
    ];
    for (const [text, lines] of cases) expect(numberedBytes(text), JSON.stringify(text)).toBe(rendered(lines));
  });

  it('counts two-digit line numbers', () => {
    const lines = Array.from({ length: 10 }, (_, i) => `line ${i}`);
    expect(numberedBytes(lines.join('\n'))).toBe(rendered(lines));
    expect(numberedBytes(lines.join('\n'))).toBe(rendered(lines.slice(0, 9)) + 1 + Buffer.byteLength('10→line 9'));
  });

  it('a text of exactly the cap passes and one byte more does not', () => {
    // One line: '1→' is 4 bytes, the rest is the line.
    const exact = 'x'.repeat(FILE_NUMBERED_MAX_BYTES - 4);
    expect(numberedBytes(exact)).toBe(FILE_NUMBERED_MAX_BYTES);
    expect(numberedBytes(exact) <= FILE_NUMBERED_MAX_BYTES).toBe(true);
    expect(numberedBytes(`${exact}x`) <= FILE_NUMBERED_MAX_BYTES).toBe(false);
  });
});

describe('safeName', () => {
  it('keeps the base name only', () => {
    expect(safeName('/Users/me/project/notes.md')).toBe('notes.md');
    expect(safeName('C:\\Users\\me\\a.txt')).toBe('a.txt');
    expect(safeName('dir/')).toBe('file');
  });

  it('replaces #, quotes, angle brackets and controls', () => {
    expect(safeName('a#L1-L2.ts')).toBe('a_L1-L2.ts');
    expect(safeName('say "hi" <b>.txt')).toBe('say _hi_ _b_.txt');
    expect(safeName('a\u0000b\nc\td\u007fe\u0085f.txt')).toBe('a_b_c_d_e_f.txt');
  });

  it('cuts a long name to the limit without splitting a pair', () => {
    expect(safeName('n'.repeat(300))).toBe('n'.repeat(NAME_MAX_CHARS));
    const atCut = `${'n'.repeat(NAME_MAX_CHARS - 1)}😀tail`;
    expect(safeName(atCut)).toBe('n'.repeat(NAME_MAX_CHARS - 1));
    const beforeCut = `${'n'.repeat(NAME_MAX_CHARS - 2)}😀tail`;
    expect(safeName(beforeCut)).toBe(`${'n'.repeat(NAME_MAX_CHARS - 2)}😀`);
  });

  it('names an empty or dot-only base name "file"', () => {
    expect(safeName('')).toBe('file');
    expect(safeName('.')).toBe('file');
    expect(safeName('a/..')).toBe('file');
  });

  it('is stable when applied twice', () => {
    for (const name of ['a/b#c', `${'n'.repeat(130)}`, '', 'x"y'])
      expect(safeName(safeName(name))).toBe(safeName(name));
  });
});

describe('attachment uri', () => {
  it('never holds # and gives back the stored name', () => {
    for (const name of ['a#b.ts', 'dir/x#L1-2.md', 'plain.txt', '', 'n'.repeat(300), 'é 中.txt']) {
      const uri = attachmentUri(name);
      expect(uri.startsWith('file:///attached/')).toBe(true);
      expect(uri.includes('#')).toBe(false);
      expect(nameOfUri(uri)).toBe(safeName(name));
    }
  });

  it('reads "file" from a uri that ends in a slash', () => {
    expect(nameOfUri('file:///attached/')).toBe('file');
  });
});

describe('base64', () => {
  it('takes 0, 1 or 2 padding characters, unpadded, whitespace-wrapped, and URL-safe base64', () => {
    for (const length of [3, 4, 5, 6]) {
      const data = Buffer.alloc(length, 7).toString('base64');
      expect(isBase64(data)).toBe(true);
      expect(base64Bytes(data)).toBe(length);
      const unpadded = data.replaceAll('=', '');
      expect(isBase64(unpadded)).toBe(true);
      expect(base64Bytes(unpadded)).toBe(length);
      const wrapped = ` ${unpadded.slice(0, 2)}\r\n\t${unpadded.slice(2)} `;
      expect(isBase64(wrapped)).toBe(true);
      expect(base64Bytes(wrapped)).toBe(length);
    }
    expect(isBase64('AQI-')).toBe(true);
    expect(isBase64('AQI_')).toBe(true);
    expect(isBase64('AQ I')).toBe(true);
    expect(isBase64('AQI')).toBe(true);
    expect(base64Bytes('AQI')).toBe(2);
    expect(base64Bytes('AQ I\r\nD')).toBe(3);
    expect(Buffer.alloc(3).toString('base64').endsWith('=')).toBe(false);
    expect(Buffer.alloc(5).toString('base64').endsWith('==')).toBe(false);
    expect(Buffer.alloc(4).toString('base64').endsWith('==')).toBe(true);
  });

  it('refuses a bad character, len % 4 === 1, bad padded length, inner padding and empty text', () => {
    expect(isBase64('iVBORw0KGgo!AAAA')).toBe(false);
    expect(isBase64('A')).toBe(false);
    expect(isBase64('ABCDE')).toBe(false);
    expect(isBase64('AQ=')).toBe(false);
    expect(isBase64('A=BC')).toBe(false);
    expect(isBase64('A===')).toBe(false);
    expect(isBase64('')).toBe(false);
    expect(isBase64(' \r\n\t ')).toBe(false);
  });

  it('sizes the base64 cap from the byte cap', () => {
    expect(IMAGE_MAX_BASE64_CHARS).toBe(4 * Math.ceil(IMAGE_MAX_BYTES / 3));
  });
});

describe('magicMatches', () => {
  it('takes PNG and JPEG bytes under their own type, including whitespace-wrapped, unpadded, and URL-safe', () => {
    expect(magicMatches('image/png', PNG)).toBe(true);
    expect(magicMatches('image/jpeg', JPEG)).toBe(true);
    expect(magicMatches('image/png', `  \r\n${PNG.slice(0, 6)}\n${PNG.slice(6)}`)).toBe(true);
    expect(magicMatches('image/png', 'iVBORw0KGgo')).toBe(true);
    const urlSafeJpeg = JPEG.replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
    expect(magicMatches('image/jpeg', urlSafeJpeg)).toBe(true);
    expect(magicMatches('image/jpeg', `\r\n _9j_4AAQ`)).toBe(true);
  });

  it('refuses a wrong pair, WebP bytes and text that is not base64', () => {
    expect(magicMatches('image/jpeg', PNG)).toBe(false);
    expect(magicMatches('image/png', JPEG)).toBe(false);
    expect(magicMatches('image/png', WEBP)).toBe(false);
    expect(magicMatches('image/webp', WEBP)).toBe(false);
    expect(magicMatches('image/png', '!!!!')).toBe(false);
    expect(magicMatches('image/png', '')).toBe(false);
  });
});

describe('attachmentView and estimateTokens', () => {
  it('views a file by stored name and text bytes, a picture by type and decoded bytes', () => {
    expect(attachmentView({ kind: 'file', name: 'src/a#1.ts', text: 'é\n' })).toEqual({
      kind: 'file',
      name: 'a_1.ts',
      bytes: 3,
    });
    expect(attachmentView({ kind: 'image', mediaType: 'image/png', data: PNG })).toEqual({
      kind: 'image',
      mediaType: 'image/png',
      bytes: 14,
    });
  });

  it('adds numbered file bytes to the text and a fixed count per picture', () => {
    expect(estimateTokens('abcd', [])).toBe(2);
    expect(
      estimateTokens('ab', [
        { kind: 'file', name: 'a.txt', text: 'x\ny' },
        { kind: 'image', mediaType: 'image/jpeg', data: JPEG },
      ]),
    ).toBe(Math.ceil((2 + rendered(['x', 'y'])) / 3) + IMAGE_TOKENS);
  });
});

describe('wellFormed', () => {
  it('refuses a lone surrogate and takes pairs, with the same answer as String.isWellFormed', () => {
    const cases = [
      '',
      'plain',
      'a\u{1F600}b',
      '\u{1F600}',
      '\ud83d',
      '\ude00',
      'a\ud83d',
      '\ude00a',
      '\ud83d\ud83d\ude00',
      '\ud83d\ude00\ude00',
      'x\udc00\ud800y',
    ];
    for (const text of cases) expect([text, wellFormed(text)]).toEqual([text, text.isWellFormed()]);
    expect(wellFormed('a\ud83d')).toBe(false);
    expect(wellFormed('a\u{1F600}b')).toBe(true);
  });
});

describe('pdfPageSize', () => {
  // Mutations: the area term dropped (A4 905x1280); the side term dropped (slide 1365x768); ceil for floor → red.
  it('fits the long side to 1,280 px and the area to 1,048,576 px, up or down, each side floored', () => {
    expect(pdfPageSize(595.276, 841.89)).toEqual({ width: 861, height: 1217 });
    expect(pdfPageSize(612, 792)).toEqual({ width: 900, height: 1164 });
    expect(pdfPageSize(960, 540)).toEqual({ width: 1280, height: 720 });
    expect(pdfPageSize(720, 540)).toEqual({ width: 1182, height: 886 });
    expect(pdfPageSize(100, 50)).toEqual({ width: 1280, height: 640 });
  });

  it('keeps every drawn page inside what the agent passes through untouched', () => {
    expect(PAGE_MAX_BASE64_CHARS).toBe(4 * Math.ceil(PAGE_MAX_BYTES / 3));
    for (const [w, h] of [
      [595.276, 841.89],
      [612, 792],
      [960, 540],
      [3, 2000],
      [14_400, 14_400],
    ] as const) {
      const size = pdfPageSize(w, h);
      expect(Math.max(size.width, size.height)).toBeLessThanOrEqual(1_280);
      expect(size.width * size.height).toBeLessThanOrEqual(1_048_576);
    }
  });
});

describe('pageTokens', () => {
  const A4 = { width: 861, height: 1217 };
  const LETTER = { width: 900, height: 1164 };
  const SLIDE = { width: 1280, height: 720 };
  const SLIDE_43 = { width: 1182, height: 886 };
  const tokens = (family: string | null) =>
    [A4, LETTER, SLIDE, SLIDE_43].map((page) => pageTokens(family, page.width, page.height));

  // Mutations: Qwen's /4 dropped; Flash Next given Qwen's max; Gemma's pool 3 as 2; Inkling's ceil as floor; the
  // engine's round half away from zero as half down (the slide's 22.5) → red.
  it("gives each family's engine count for the common page sizes", () => {
    expect(tokens('qwen3_5')).toEqual([1_026, 1_008, 920, 1_036]);
    expect(tokens('qwen3_5_moe')).toEqual([1_026, 1_008, 920, 1_036]);
    expect(tokens('inkling2')).toEqual([682, 690, 576, 690]);
    expect(tokens('gemma4')).toEqual([266, 266, 264, 266]);
    expect(tokens('qwen4_exp')).toEqual([247, 252, 252, 234]);
  });

  it('keeps the flat picture count for a family it does not know', () => {
    expect(tokens('lfm2')).toEqual(Array(4).fill(IMAGE_TOKENS));
    expect(tokens(null)).toEqual(Array(4).fill(IMAGE_TOKENS));
  });
});

describe('pdfNote and readPdfNote', () => {
  const MESSAGE = { name: 'bitter_lesson.pdf', pages: 2, way: 'message', from: 1 } as const;
  const TOOL = { name: 'report-2026.pdf', pages: 212, way: 'tool', id: 'k3f9a2c1-212' } as const;

  it('writes key lines the model reads and one sentence', () => {
    expect(pdfNote(MESSAGE)).toBe(
      [
        'Portato PDF note 1',
        'name: bitter_lesson.pdf',
        'pages: 2',
        'way: message',
        'images: 1-2',
        'Images 1-2 of this message are pages 1-2 of this PDF, in order.',
      ].join('\n'),
    );
    expect(pdfNote({ name: 'one.pdf', pages: 1, way: 'message', from: 3 }).split('\n').slice(4)).toEqual([
      'images: 3-3',
      'Image 3 of this message is page 1 of this PDF.',
    ]);
    expect(pdfNote(TOOL).split('\n')).toEqual([
      'Portato PDF note 1',
      'name: report-2026.pdf',
      'pages: 212',
      'way: tool',
      'id: k3f9a2c1-212',
      'The pages are image files on disk, not in this message. To read them, call the workflow tool with name read-pdf and args {"pdf": "k3f9a2c1-212", "question": "<what to find out>"}. Add "pages": "40-60" to read only some pages.',
    ]);
  });

  // Mutation: the key split at the last ": " → red.
  it('reads back both ways, a name with ": " and a note with CRLF line ends', () => {
    expect(readPdfNote(pdfNote(MESSAGE))).toEqual(MESSAGE);
    expect(readPdfNote(pdfNote(TOOL))).toEqual(TOOL);
    const colon = { ...MESSAGE, name: 'notes: draft 2.pdf', from: 4 };
    expect(readPdfNote(pdfNote(colon))).toEqual(colon);
    expect(readPdfNote(pdfNote(MESSAGE).replaceAll('\n', '\r\n'))).toEqual(MESSAGE);
    expect(pdfNote({ ...MESSAGE, name: 'a\nb#c.pdf' }).split('\n')[1]).toBe('name: a_b_c.pdf');
  });

  // Mutation: any text that names a PDF read as a note → red.
  it('takes a note only when its first line is the marker and every key line is one the app writes', () => {
    const note = pdfNote(MESSAGE);
    const lines = note.split('\n');
    const swap = (index: number, line: string) => lines.map((l, i) => (i === index ? line : l)).join('\n');
    expect(isPdfNoteText(note)).toBe(true);
    expect(isPdfNoteText('Portato PDF note 1\r\nname: a.pdf')).toBe(true);
    expect(isPdfNoteText('Portato PDF note 1')).toBe(true);
    expect(isPdfNoteText(`hello\n${note}`)).toBe(false);
    expect(isPdfNoteText(' Portato PDF note 1\n')).toBe(false);
    const refused = [
      'bitter_lesson.pdf',
      `hello\n${note}`,
      `﻿${note}`,
      swap(0, 'Portato PDF note 2'),
      swap(1, 'name: '),
      swap(1, 'title: bitter_lesson.pdf'),
      swap(2, 'pages: 0'),
      swap(2, 'pages: 02'),
      swap(2, 'pages: two'),
      swap(3, 'way: mail'),
      swap(4, 'images: 1-3'),
      swap(4, 'images: 1'),
      swap(4, 'images: 0-1'),
      lines.slice(0, 4).join('\n'),
      pdfNote({ ...MESSAGE, pages: 21 }),
      pdfNote({ ...TOOL, id: 'k3f9a2c1-211' }),
      pdfNote({ ...TOOL, id: 'K3F9A2C1-212' }),
      pdfNote({ ...TOOL, id: 'k3f9a2c-212' }),
      pdfNote({ ...TOOL, pages: 321, id: 'k3f9a2c1-321' }),
    ];
    for (const text of refused) expect([text, readPdfNote(text)]).toEqual([text, null]);
  });
});

describe('PDF pages as pictures: attachmentView, storedPictureChars, heldImages', () => {
  const jpeg = (bytes: number) => {
    const data = new Uint8Array(bytes);
    data.set([0xff, 0xd8, 0xff, 0xe0]);
    return Buffer.from(data).toString('base64');
  };

  it('views a PDF by name, page count, decoded bytes and the base64 the agent stores per page', () => {
    expect(attachmentView({ kind: 'pdf', name: 'docs/a#1.pdf', images: [jpeg(300), jpeg(301)] })).toEqual({
      kind: 'pdf',
      name: 'a_1.pdf',
      pages: 2,
      way: 'message',
      bytes: 601,
      chars: 400 + 404,
    });
  });

  // Mutation: PDF pages left out of the stored base64 or the image count (the picture room and the 64 pass wrongly) → red.
  it('counts the pages a message carries as pictures; a tool PDF holds none', () => {
    const picture: UserAttachment = { kind: 'image', mediaType: 'image/png', bytes: 300 };
    const pages: UserAttachment = { kind: 'pdf', name: 'a.pdf', pages: 3, way: 'message', bytes: 900, chars: 1_200 };
    const tool: UserAttachment = { kind: 'pdf', name: 'b.pdf', pages: 212, way: 'tool', bytes: 0, chars: 0, id: 'x' };
    const file: UserAttachment = { kind: 'file', name: 'c.md', bytes: 9 };
    expect(storedPictureChars([picture, pages, tool, file])).toBe(400 + 1_200);
    const entries = [
      { kind: 'user', attachments: [picture, pages, file] },
      { kind: 'assistant' },
      { kind: 'user', attachments: [tool] },
      { kind: 'tool', pictures: [{}, {}, {}] },
      { kind: 'tool' },
      { kind: 'wake', attachments: [picture] },
    ];
    expect(heldImages(entries)).toBe(1 + 3 + 3);
  });
});

describe('estimateTokens with PDFs', () => {
  const A4: PageSize = { width: 861, height: 1217 };
  const PAGE = '/9j/4AAQ';

  it("adds each PDF's note and its pages by the family's count; a page with no size counts a picture", () => {
    const note = (from: number, pages: number) =>
      numberedBytes(pdfNote({ name: 'a.pdf', pages, way: 'message', from }));
    const picture = { kind: 'image', mediaType: 'image/jpeg', data: PAGE } as const;
    const sized = { kind: 'pdf', name: 'a.pdf', images: [PAGE, PAGE], sizes: [A4, A4] } as const;
    expect(estimateTokens('ab', [picture, sized], 'inkling2')).toBe(
      Math.ceil((2 + note(2, 2)) / 3) + IMAGE_TOKENS + 2 * 682,
    );
    expect(estimateTokens('ab', [{ kind: 'pdf', name: 'a.pdf', images: [PAGE] }], 'inkling2')).toBe(
      Math.ceil((2 + note(1, 1)) / 3) + IMAGE_TOKENS,
    );
  });
});

describe('planPdfs', () => {
  const A4: PageSize = { width: 861, height: 1217 };
  const pdf = (pages: number, chars = pages * 1_000, name = 'a.pdf'): PdfDraft => ({
    name,
    sizes: Array(pages).fill(A4),
    chars,
  });
  const base: PdfPlanInput = {
    text: 'q',
    attachments: [],
    pdfs: [],
    family: 'qwen3_5',
    window: null,
    contextUsed: null,
    held: 0,
    stored: 0,
    tool: true,
  };
  const plan = (over: Partial<PdfPlanInput>) => planPdfs({ ...base, ...over });
  const ways = (over: Partial<PdfPlanInput>) => plan(over).ways;
  /** A picture whose stored base64 is `chars` long. */
  const picture = (chars: number) => ({ kind: 'image', mediaType: 'image/png', data: 'A'.repeat(chars) }) as const;

  it('puts up to 20 pages of a message in it, in the order added; more go to the tool', () => {
    expect(ways({ pdfs: [pdf(20)] })).toEqual(['message']);
    expect(ways({ pdfs: [pdf(21)] })).toEqual(['tool']);
    expect(ways({ pdfs: [pdf(15), pdf(10), pdf(5)] })).toEqual(['message', 'tool', 'message']);
  });

  it('keeps the pictures and pages of a message to 6 MiB of base64', () => {
    const pictures = [picture(1_000_000)];
    const edge = MESSAGE_IMAGES_MAX_BASE64 - 1_000_000;
    expect(ways({ attachments: pictures, pdfs: [pdf(2, edge)] })).toEqual(['message']);
    expect(ways({ attachments: pictures, pdfs: [pdf(2, edge + 4)] })).toEqual(['tool']);
  });

  // Mutation: the message's pictures left out of the 64 → red.
  it('keeps the images the conversation holds and this message brings to 64', () => {
    expect(ways({ held: 60, pdfs: [pdf(4)] })).toEqual(['message']);
    expect(ways({ held: 60, pdfs: [pdf(5)] })).toEqual(['tool']);
    expect(ways({ held: 60, attachments: [picture(8)], pdfs: [pdf(4)] })).toEqual(['tool']);
  });

  it("keeps the chat's stored pictures and pages to 40 MiB", () => {
    const stored = CHAT_PICTURES_MAX_BASE64 - 4_000;
    expect(ways({ stored, pdfs: [pdf(2, 4_000)] })).toEqual(['message']);
    expect(ways({ stored, pdfs: [pdf(2, 4_004)] })).toEqual(['tool']);
    expect(ways({ stored, attachments: [picture(8)], pdfs: [pdf(2, 4_000)] })).toEqual(['tool']);
  });

  it('gives Flash Next 4 images in all', () => {
    expect(ways({ family: 'qwen4_exp', pdfs: [pdf(4)] })).toEqual(['message']);
    expect(ways({ family: 'qwen4_exp', pdfs: [pdf(5)] })).toEqual(['tool']);
    expect(ways({ family: 'qwen4_exp', held: 1, pdfs: [pdf(4)] })).toEqual(['tool']);
    expect(ways({ family: 'qwen3_5', held: 1, pdfs: [pdf(5)] })).toEqual(['message']);
  });

  // Mutation: the context used left out → red.
  it("keeps the context used and the message to 80% of a known window, pages by the family's count", () => {
    const sent = plan({ window: 10_000, pdfs: [pdf(2)] });
    expect(sent.ways).toEqual(['message']);
    expect(sent.tokens).toBe(
      estimateTokens('q', [{ kind: 'pdf', name: 'a.pdf', images: ['x', 'x'], sizes: [A4, A4] }], 'qwen3_5'),
    );
    expect(sent.tokens).toBeGreaterThan(2 * 1_026);
    expect(ways({ window: 10_000, contextUsed: 8_000 - sent.tokens, pdfs: [pdf(2)] })).toEqual(['message']);
    expect(ways({ window: 10_000, contextUsed: 8_001 - sent.tokens, pdfs: [pdf(2)] })).toEqual(['tool']);
    expect(ways({ window: null, contextUsed: 1_000_000, pdfs: [pdf(2)] })).toEqual(['message']);
  });

  // Mutation: a second tool PDF allowed → red.
  it('lets one PDF of a message go to the tool; a second keeps the message from going', () => {
    const one = plan({ pdfs: [pdf(21), pdf(3)] });
    expect(one).toMatchObject({ ways: ['tool', 'message'], problem: null, readTokens: 21 * 1_026 });
    expect(one.tokens).toBeLessThan(3 * 1_026 + 200);
    expect(plan({ pdfs: [pdf(21), pdf(30, 30_000, 'b.pdf')] })).toMatchObject({
      ways: ['tool', 'tool'],
      problem: 'one',
      readTokens: 51 * 1_026,
    });
  });

  // Mutation: with no tool in the mode, a PDF that does not fit answered as the tool's → red.
  it('without the tool in this mode, a PDF that does not fit stays in the message and the message cannot go', () => {
    expect(plan({ tool: false, pdfs: [pdf(21)] })).toMatchObject({ ways: ['message'], problem: 'room', readTokens: 0 });
    expect(plan({ tool: false, held: 63, pdfs: [pdf(1)] })).toMatchObject({ ways: ['message'], problem: null });
    expect(plan({ tool: false, held: 63, pdfs: [pdf(2)] })).toMatchObject({ problem: 'room' });
  });
});
