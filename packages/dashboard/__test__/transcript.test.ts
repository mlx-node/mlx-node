/**
 * Direct unit coverage for the pure transcript projection. These functions used
 * to be reachable only through the HTTP session-detail route, so a regression in
 * (say) the binary sniffing showed up only as a confusing end-to-end failure.
 */

import type { SessionEntry } from '@earendil-works/pi-coding-agent';
import { describe, expect, it } from 'vite-plus/test';

import {
  collectCallArgs,
  describeBinary,
  extractBinaryNotes,
  extractImages,
  extractText,
  extractToolCalls,
  isBinaryText,
  mapTranscriptEntry,
  partitionImageDetails,
  summarizeToolArgs,
} from '../src/api/transcript.js';

const NUL = String.fromCharCode(0);
/** A 1x1 PNG, base64 — small enough to inline. */
const PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

/** Build a `message` session entry with the shape the projection reads. */
function messageEntry(message: Record<string, unknown>, timestamp?: string): SessionEntry {
  return { type: 'message', id: 'x', parentId: null, timestamp, message } as unknown as SessionEntry;
}

describe('isBinaryText', () => {
  it('accepts prose containing tab, newline and carriage return', () => {
    expect(isBinaryText('hello\tworld\r\nsecond line')).toBe(false);
    expect(isBinaryText('')).toBe(false);
    expect(isBinaryText('émoji 🎉 and ünicode stay text')).toBe(false);
  });

  it('rejects any other C0 control byte', () => {
    expect(isBinaryText(`${NUL}binary`)).toBe(true);
    expect(isBinaryText(`ok${String.fromCharCode(11)}`)).toBe(true); // vertical tab
    expect(isBinaryText(`ok${String.fromCharCode(12)}`)).toBe(true); // form feed
    expect(isBinaryText(`ok${String.fromCharCode(31)}`)).toBe(true); // unit separator
    expect(isBinaryText(`ok${String.fromCharCode(8)}`)).toBe(true); // backspace
  });

  it('only scans a bounded prefix, so a control byte past 4096 chars is not seen', () => {
    expect(isBinaryText('a'.repeat(4096) + NUL)).toBe(false);
    expect(isBinaryText('a'.repeat(4095) + NUL)).toBe(true);
  });
});

describe('describeBinary', () => {
  it('sniffs the common magics', () => {
    expect(describeBinary('%PDF-1.7 ...')).toMatch(/^PDF · /);
    expect(describeBinary(`${NUL.repeat(4)}ftypheic more`)).toMatch(/^HEIC · /);
    expect(describeBinary(`${NUL.repeat(4)}ftypmif1 more`)).toMatch(/^HEIC · /);
    expect(describeBinary('\u0089PNG\r\n')).toMatch(/^PNG · /);
    expect(describeBinary('\u00ff\u00d8\u00ff\u00e0')).toMatch(/^JPEG · /);
    expect(describeBinary('GIF89a')).toMatch(/^GIF · /);
    expect(describeBinary('PK\u0003\u0004')).toMatch(/^ZIP · /);
    expect(describeBinary('no magic here')).toMatch(/^binary · /);
  });

  it('labels the size in B / KB / MB', () => {
    expect(describeBinary('x'.repeat(512))).toBe('binary · 512 B');
    expect(describeBinary('x'.repeat(32_768))).toBe('binary · 32 KB');
    expect(describeBinary('x'.repeat(2_097_152))).toBe('binary · 2.0 MB');
  });
});

describe('extractText', () => {
  it('returns a plain string as-is and joins text blocks with a newline', () => {
    expect(extractText('hello')).toBe('hello');
    expect(
      extractText([
        { type: 'text', text: 'a' },
        { type: 'text', text: 'b' },
      ]),
    ).toBe('a\nb');
  });

  it('drops binary payloads instead of dumping them into prose', () => {
    expect(extractText(`${NUL}raw bytes`)).toBe('');
    expect(
      extractText([
        { type: 'text', text: 'keep' },
        { type: 'text', text: `${NUL}drop` },
      ]),
    ).toBe('keep');
  });

  it('ignores non-text blocks and non-array, non-string content', () => {
    expect(extractText([{ type: 'image', data: PNG_B64 }])).toBe('');
    expect(extractText(null)).toBe('');
    expect(extractText(42)).toBe('');
  });
});

describe('partitionImageDetails', () => {
  const note =
    '[Image: original 6720x4480, displayed at 2000x1333. Multiply coordinates by 3.36 to map to original image.]';

  it('splits the generated coordinate note out of the prose', () => {
    const { visible, notes } = partitionImageDetails(`Read image file [image/png]\n${note}`);
    expect(visible).toBe('Read image file [image/png]');
    expect(notes).toEqual([note]);
  });

  it('leaves human prose that merely mentions an image alone', () => {
    const prose = 'Please look at [Image: the one I sent] and tell me what you think';
    expect(partitionImageDetails(prose)).toEqual({ visible: prose, notes: [] });
  });

  it('is a no-op when there is no note marker at all', () => {
    expect(partitionImageDetails('just prose')).toEqual({ visible: 'just prose', notes: [] });
    expect(partitionImageDetails('')).toEqual({ visible: '', notes: [] });
  });

  it('collapses the blank runs the removed lines leave behind', () => {
    const { visible } = partitionImageDetails(`before\n\n${note}\n\n\nafter`);
    expect(visible).toBe('before\n\nafter');
  });
});

describe('extractImages', () => {
  it('inlines image blocks and defaults a missing mimeType', () => {
    expect(extractImages([{ type: 'image', data: PNG_B64, mimeType: 'image/png' }])).toEqual([
      { mimeType: 'image/png', data: PNG_B64 },
    ]);
    expect(extractImages([{ type: 'image', data: 'abc' }])).toEqual([{ mimeType: 'image/png', data: 'abc' }]);
  });

  it('caps at 8 inlined images per message', () => {
    const many = Array.from({ length: 20 }, () => ({ type: 'image', data: 'a', mimeType: 'image/png' }));
    expect(extractImages(many)).toHaveLength(8);
  });

  it('skips an image too large to inline', () => {
    expect(extractImages([{ type: 'image', data: 'a'.repeat(2_000_001) }])).toEqual([]);
    expect(extractImages([{ type: 'image', data: 'a'.repeat(2_000_000) }])).toHaveLength(1);
  });
});

describe('extractBinaryNotes', () => {
  it('chips a raw-binary text block instead of rendering it', () => {
    expect(extractBinaryNotes([{ type: 'text', text: `${NUL.repeat(4)}ftypheic rest` }])[0]).toMatch(/^HEIC · /);
    expect(extractBinaryNotes(`${NUL}raw`)[0]).toMatch(/^binary · /);
  });

  it('chips an oversize image with its decoded size', () => {
    const notes = extractBinaryNotes([{ type: 'image', data: 'a'.repeat(4_000_000) }]);
    expect(notes[0]).toMatch(/^image · 2\.9 MB \(too large to preview\)$/);
  });

  it('says nothing about ordinary prose or an inlineable image', () => {
    expect(extractBinaryNotes([{ type: 'text', text: 'hello' }])).toEqual([]);
    expect(extractBinaryNotes([{ type: 'image', data: PNG_B64 }])).toEqual([]);
    expect(extractBinaryNotes(null)).toEqual([]);
  });
});

describe('summarizeToolArgs', () => {
  it('prefers the salient keys in priority order', () => {
    expect(summarizeToolArgs({ command: 'ls -la', path: '/tmp' })).toBe('ls -la');
    expect(summarizeToolArgs({ path: '/tmp/a.ts', description: 'x' })).toBe('/tmp/a.ts');
    expect(summarizeToolArgs({ file_path: '/tmp/b.ts' })).toBe('/tmp/b.ts');
    expect(summarizeToolArgs({ agent: 'explore' })).toBe('explore');
  });

  it('falls back to the first non-empty string field, then to empty', () => {
    expect(summarizeToolArgs({ unknownKey: 'fallback' })).toBe('fallback');
    expect(summarizeToolArgs({ blank: '   ', later: 'used' })).toBe('used');
    expect(summarizeToolArgs({ n: 1 })).toBe('');
    expect(summarizeToolArgs(null)).toBe('');
    expect(summarizeToolArgs('a raw string')).toBe('a raw string');
  });

  it('flattens whitespace and caps the digest at 300 chars', () => {
    expect(summarizeToolArgs({ command: '  git   status\n--short  ' })).toBe('git status --short');
    expect(summarizeToolArgs({ command: 'x'.repeat(500) })).toHaveLength(300);
  });
});

describe('extractToolCalls / collectCallArgs', () => {
  const call = { type: 'toolCall', id: 'call_1', name: 'read', arguments: { path: '/src/lib.rs' } };

  it('projects a tool call with its arg digest', () => {
    expect(extractToolCalls([call])).toEqual([
      { id: 'call_1', name: 'read', arguments: { path: '/src/lib.rs' }, summary: '/src/lib.rs' },
    ]);
  });

  it('defaults a missing id/name and a missing arguments object', () => {
    expect(extractToolCalls([{ type: 'toolCall' }])).toEqual([{ id: '', name: '', arguments: null, summary: '' }]);
    expect(extractToolCalls('not an array')).toEqual([]);
  });

  it('maps every call id to its raw arguments across entries', () => {
    const map = collectCallArgs([
      messageEntry({ role: 'assistant', content: [call] }),
      messageEntry({ role: 'user', content: 'no calls here' }),
      { type: 'session_info', name: 'x' } as unknown as SessionEntry,
    ]);
    expect(map.get('call_1')).toEqual({ path: '/src/lib.rs' });
    expect(map.size).toBe(1);
  });
});

describe('mapTranscriptEntry', () => {
  const noArgs = new Map<string, unknown>();

  it('ignores a non-message entry', () => {
    expect(mapTranscriptEntry({ type: 'session_info' } as unknown as SessionEntry, noArgs)).toBeNull();
  });

  it('prefers the entry timestamp, then the message timestamp, then 0', () => {
    const iso = '2026-07-20T10:00:02.000Z';
    expect(mapTranscriptEntry(messageEntry({ role: 'user', content: 'a' }, iso), noArgs)?.ts).toBe(Date.parse(iso));
    expect(mapTranscriptEntry(messageEntry({ role: 'user', content: 'a', timestamp: 1234 }), noArgs)?.ts).toBe(1234);
    expect(mapTranscriptEntry(messageEntry({ role: 'user', content: 'a' }, 'not-a-date'), noArgs)?.ts).toBe(0);
    expect(mapTranscriptEntry(messageEntry({ content: 'a' }), noArgs)?.role).toBe('unknown');
  });

  it('carries the model only for an assistant message', () => {
    expect(mapTranscriptEntry(messageEntry({ role: 'assistant', content: 'a', model: 'qwen3_5' }), noArgs)?.model).toBe(
      'qwen3_5',
    );
    expect(
      mapTranscriptEntry(messageEntry({ role: 'user', content: 'a', model: 'qwen3_5' }), noArgs)?.model,
    ).toBeUndefined();
    expect(
      mapTranscriptEntry(messageEntry({ role: 'assistant', content: 'a', model: '' }), noArgs)?.model,
    ).toBeUndefined();
  });

  it('joins a toolResult back to its originating call arguments as `title`', () => {
    const args = new Map<string, unknown>([['call_1', { path: '/src/lib.rs' }]]);
    const mapped = mapTranscriptEntry(
      messageEntry({ role: 'toolResult', toolCallId: 'call_1', toolName: 'read', isError: false, content: 'fn main' }),
      args,
    );
    expect(mapped).toMatchObject({ role: 'toolResult', toolName: 'read', isError: false, title: '/src/lib.rs' });
  });

  it('omits `title` when the call id is unknown, and never sets it on a non-result', () => {
    expect(
      mapTranscriptEntry(messageEntry({ role: 'toolResult', toolCallId: 'ghost', content: 'x' }), noArgs)?.title,
    ).toBeUndefined();
    expect(
      mapTranscriptEntry(
        messageEntry({ role: 'assistant', toolCallId: 'call_1', content: 'x' }),
        new Map([['call_1', { path: '/p' }]]),
      )?.title,
    ).toBeUndefined();
  });

  it('omits the optional arrays when empty, and fills them when present', () => {
    const plain = mapTranscriptEntry(messageEntry({ role: 'user', content: 'hi' }), noArgs);
    expect(plain?.images).toBeUndefined();
    expect(plain?.binaryNotes).toBeUndefined();
    expect(plain?.imageNotes).toBeUndefined();

    const rich = mapTranscriptEntry(
      messageEntry({
        role: 'toolResult',
        content: [
          { type: 'text', text: 'Read image file [image/png]' },
          { type: 'text', text: '[Image: original 10x10, displayed at 5x5. Multiply coordinates by 2 to map.]' },
          { type: 'image', data: PNG_B64, mimeType: 'image/png' },
          { type: 'text', text: `${NUL.repeat(4)}ftypheic blob` },
        ],
      }),
      noArgs,
    );
    expect(rich?.text).toBe('Read image file [image/png]');
    expect(rich?.images).toEqual([{ mimeType: 'image/png', data: PNG_B64 }]);
    expect(rich?.imageNotes?.[0]).toContain('Multiply coordinates by 2');
    expect(rich?.binaryNotes?.[0]).toMatch(/^HEIC · /);
    // The raw bytes never reach the rendered text.
    expect(rich?.text).not.toContain('blob');
  });
});
