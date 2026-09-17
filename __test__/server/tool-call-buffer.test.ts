import { ToolCallTagBuffer } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { recoverSuppressedToolCallText } from '../../packages/server/src/mappers/anthropic-response.js';

describe('ToolCallTagBuffer', () => {
  it('suppresses Gemma4 structural tool-call tags split across chunks', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('title <|too')).toEqual({
      safeText: 'title ',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.push('l_call>call:list_files{path:<|"|>.<|"|>}')).toEqual({
      safeText: '',
      tagFound: true,
      cleanPrefix: '',
    });
    expect(buffer.push('<|tool_response>')).toEqual({
      safeText: '',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.flush()).toBe('');
  });

  it('strips parsed tool-call and tool-response blocks when tool use is disallowed', () => {
    const raw =
      '<|channel>thought\nneed a file list<channel|><|tool_call>call:list_files{path:<|"|>.<|"|>}<tool_call|><|tool_response>';

    expect(recoverSuppressedToolCallText(raw)).toBe('');
    expect(recoverSuppressedToolCallText(`visible ${raw}`)).toBe('visible ');
  });

  it('suppresses LFM2 sentinel blocks and everything after them', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('The weather in Paris is ')).toEqual({
      safeText: 'The weather in Paris is ',
      tagFound: false,
      cleanPrefix: '',
    });
    // Start sentinel split across chunks holds the prefix boundary.
    expect(buffer.push('<|tool_call_')).toEqual({
      safeText: '',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.push('start|>[get_weather(location=')).toEqual({
      safeText: '',
      tagFound: true,
      cleanPrefix: '',
    });
    // Interior, the end sentinel, AND post-call prose are all suppressed:
    // the streamed text stays a verbatim prefix of the raw output so the
    // done-path recovery emits the right tail whether the call parses
    // (cleaned text) or not (verbatim raw block).
    expect(buffer.push('"Paris")]<|tool_call_end|>sunny today.')).toEqual({
      safeText: '',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.flush()).toBe('');
  });

  it('suppresses the LFM2 post-call echo and prose to stream end', () => {
    const buffer = new ToolCallTagBuffer();

    buffer.push('<|tool_call_start|>[get_weather(location="Paris")]');
    expect(buffer.push('<|tool_call_end|>[get_weather(location="Paris")]').safeText).toBe('');
    // Echo bytes and trailing prose are held alike — nothing leaks.
    expect(buffer.push('<|tool_call_end|>done.')).toEqual({
      safeText: '',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.flush()).toBe('');
  });

  it('holds a whole LFM2 block landing in one delta to the end', () => {
    const buffer = new ToolCallTagBuffer();

    const pushed = buffer.push('intro <|tool_call_start|>[f()]<|tool_call_end|>outro');
    expect(pushed.tagFound).toBe(true);
    expect(pushed.cleanPrefix).toBe('intro ');
    // The trailing prose is recovered from `finalText` by the done-path,
    // not released by the buffer itself.
    expect(buffer.flush()).toBe('');
  });

  it('keeps later LFM2 sentinel blocks suppressed too', () => {
    const buffer = new ToolCallTagBuffer();

    buffer.push('<|tool_call_start|>[f()]<|tool_call_end|>');
    // A second block's prose and interior stay suppressed — a malformed
    // later block must not leave released text ahead of the raw tail.
    const second = buffer.push('between <|tool_call_start|>[g()');
    expect(second).toEqual({ safeText: '', tagFound: false, cleanPrefix: '' });
    expect(buffer.flush()).toBe('');
  });

  it('recovers text around an LFM2 sentinel block when tool use is disallowed', () => {
    const raw = 'before <|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|> after';

    expect(recoverSuppressedToolCallText(raw)).toBe('before  after');
    expect(recoverSuppressedToolCallText('<|tool_call_start|>[f()]')).toBe('');
  });

  it('drops the LFM2 echoed call body during tools-disallowed recovery', () => {
    // The echo is capped by a second end sentinel: recovery must drop
    // through the LAST end, not leak the `[f()]` echo as visible text.
    const raw = 'A<|tool_call_start|>[f()]<|tool_call_end|>[f()]<|tool_call_end|>tail';

    expect(recoverSuppressedToolCallText(raw)).toBe('Atail');
  });
});
