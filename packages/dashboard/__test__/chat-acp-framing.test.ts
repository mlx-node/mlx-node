import { describe, expect, it } from 'vite-plus/test';

import { encodeFrame, FramingError, LineSplitter } from '../src/chat/acp/framing.js';

function collect(maxBytes?: number): { lines: string[]; splitter: LineSplitter } {
  const lines: string[] = [];
  return { lines, splitter: new LineSplitter((line) => lines.push(line), maxBytes) };
}

describe('LineSplitter', () => {
  it('splits on newlines across chunk borders', () => {
    const { lines, splitter } = collect();
    splitter.push(Buffer.from('{"a":1}\n{"b":'));
    expect(lines).toEqual(['{"a":1}']);
    splitter.push(Buffer.from('2}\n\n{"c":3}\n'));
    expect(lines).toEqual(['{"a":1}', '{"b":2}', '', '{"c":3}']);
  });

  it('decodes a multi-byte character split between two chunks', () => {
    const { lines, splitter } = collect();
    const bytes = Buffer.from('{"t":"日本"}\n');
    splitter.push(bytes.subarray(0, 8));
    splitter.push(bytes.subarray(8));
    expect(lines).toEqual(['{"t":"日本"}']);
  });

  it('drops a torn last line at end', () => {
    const { lines, splitter } = collect();
    splitter.push(Buffer.from('{"a":1}\n{"torn"'));
    splitter.end();
    splitter.push(Buffer.from('{"b":2}\n'));
    expect(lines).toEqual(['{"a":1}', '{"b":2}']);
  });

  it('a line over the limit is a framing error, also when it arrives in pieces, and the splitter stays dead', () => {
    const { lines, splitter } = collect(16);
    splitter.push(Buffer.from('0123456789'));
    expect(() => splitter.push(Buffer.from('0123456789'))).toThrow(FramingError);
    expect(() => splitter.push(Buffer.from('\n'))).toThrow(FramingError);
    expect(lines).toEqual([]);
  });

  it('a line of exactly the limit passes', () => {
    const { lines, splitter } = collect(4);
    splitter.push(Buffer.from('abcd\n'));
    expect(lines).toEqual(['abcd']);
  });
});

describe('encodeFrame', () => {
  it('writes one JSON object and one newline; a newline inside a string stays escaped', () => {
    const frame = encodeFrame({ text: 'a\nb' });
    expect(frame.endsWith('\n')).toBe(true);
    expect(frame.indexOf('\n')).toBe(frame.length - 1);
    expect(JSON.parse(frame)).toEqual({ text: 'a\nb' });
  });
});
