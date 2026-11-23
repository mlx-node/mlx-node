import { describe, expect, it } from 'vite-plus/test';

import {
  attachmentUri,
  attachmentView,
  base64Bytes,
  estimateTokens,
  FILE_NUMBERED_MAX_BYTES,
  IMAGE_MAX_BASE64_CHARS,
  IMAGE_MAX_BYTES,
  IMAGE_TOKENS,
  isBase64,
  magicMatches,
  NAME_MAX_CHARS,
  nameOfUri,
  numberedBytes,
  safeName,
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
  it('takes 0, 1 or 2 padding characters and counts the bytes', () => {
    for (const length of [3, 4, 5, 6]) {
      const data = Buffer.alloc(length, 7).toString('base64');
      expect(isBase64(data)).toBe(true);
      expect(base64Bytes(data)).toBe(length);
    }
    expect(Buffer.alloc(3).toString('base64').endsWith('=')).toBe(false);
    expect(Buffer.alloc(5).toString('base64').endsWith('==')).toBe(false);
    expect(Buffer.alloc(4).toString('base64').endsWith('==')).toBe(true);
  });

  it('refuses a bad character, a bad length, inner padding and empty text', () => {
    expect(isBase64('AQI-')).toBe(false);
    expect(isBase64('AQ I')).toBe(false);
    expect(isBase64('AQI')).toBe(false);
    expect(isBase64('A=BC')).toBe(false);
    expect(isBase64('A===')).toBe(false);
    expect(isBase64('')).toBe(false);
  });

  it('sizes the base64 cap from the byte cap', () => {
    expect(IMAGE_MAX_BASE64_CHARS).toBe(4 * Math.ceil(IMAGE_MAX_BYTES / 3));
  });
});

describe('magicMatches', () => {
  it('takes PNG and JPEG bytes under their own type', () => {
    expect(magicMatches('image/png', PNG)).toBe(true);
    expect(magicMatches('image/jpeg', JPEG)).toBe(true);
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
