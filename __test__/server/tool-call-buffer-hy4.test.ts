import { describe, expect, it } from 'vite-plus/test';

import { ToolCallTagBuffer } from '../../packages/lm/src/tool-call-buffer.js';

function hy4Block(x: string): string {
  return (
    `<tool_calls:${x}><tool_call:${x}>get_weather` +
    `<arg_key:${x}>city</arg_key:${x}><arg_value:${x}>Paris</arg_value:${x}>` +
    `<arg_key:${x}>days</arg_key:${x}><arg_value:${x}>3</arg_value:${x}>` +
    `</tool_call:${x}></tool_calls:${x}>`
  );
}

/** Text that reaches the wire: released deltas, the pre-tag prefix, and the flush. */
function streamInChunks(text: string, size: number): { wire: string; suppressed: boolean } {
  const buffer = new ToolCallTagBuffer();
  let wire = '';
  for (let at = 0; at < text.length; at += size) {
    const pushed = buffer.push(text.slice(at, at + size));
    wire += pushed.safeText + pushed.cleanPrefix;
  }
  return { wire: wire + buffer.flush(), suppressed: buffer.suppressed };
}

describe('ToolCallTagBuffer: Hy4-flash', () => {
  for (const x of ['6124c78e', 'opensource']) {
    it(`suppresses a :${x} tool-call block split anywhere and keeps the prose before it`, () => {
      const prose = 'Let me check the weather.\n';
      const content = prose + hy4Block(x) + '\ntrailing';
      for (let cut = 0; cut <= content.length; cut++) {
        const buffer = new ToolCallTagBuffer();
        const first = buffer.push(content.slice(0, cut));
        const second = buffer.push(content.slice(cut));
        const wire = first.safeText + first.cleanPrefix + second.safeText + second.cleanPrefix + buffer.flush();
        expect(wire).toBe(prose);
        expect(first.tagFound || second.tagFound).toBe(true);
        expect(buffer.suppressed).toBe(true);
      }
      for (const size of [1, 2, 3, 5, 7]) {
        expect(streamInChunks(content, size)).toEqual({ wire: prose, suppressed: true });
      }
    });
  }

  it('releases a Hy4 text turn with lookalike but incomplete tags', () => {
    const texts = [
      'The <tool_calls> wrapper has no suffix, <tool_callsX> no colon.',
      'An inner <tool_call:opensource>name</tool_call:opensource> with no outer block is text.',
      'Args look like <arg_key:6124c78e>k</arg_key:6124c78e><arg_value:6124c78e>v</arg_value:6124c78e>.',
      'It ends on an unfinished <tool_calls',
      'It ends on an unfinished <tool_call',
    ];
    for (const text of texts) {
      for (const size of [1, 2, 3, text.length]) {
        expect(streamInChunks(text, size)).toEqual({ wire: text, suppressed: false });
      }
    }
  });

  it('suppresses a Qwen3 <tool_call> block exactly as before', () => {
    const content =
      'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\nafter';
    for (let cut = 0; cut <= content.length; cut++) {
      const buffer = new ToolCallTagBuffer();
      const first = buffer.push(content.slice(0, cut));
      const second = buffer.push(content.slice(cut));
      const wire = first.safeText + first.cleanPrefix + second.safeText + second.cleanPrefix + buffer.flush();
      expect(wire).toBe('Checking.\n');
      expect(buffer.suppressed).toBe(true);
    }
    const plain = 'A literal <tool_calls are not tags> stays text.';
    expect(streamInChunks(plain, 1)).toEqual({ wire: plain, suppressed: false });
  });
});
