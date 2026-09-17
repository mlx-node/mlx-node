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

  it('suppresses LFM2 sentinel-wrapped calls and releases post-call prose', () => {
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
    // Interior text is suppressed; the end sentinel can arrive with prose.
    expect(buffer.push('"Paris")]<|tool_call_end|>sunny today.')).toEqual({
      safeText: 'sunny today.',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.flush()).toBe('');
  });

  it('suppresses the LFM2 post-call echo through the second end sentinel', () => {
    const buffer = new ToolCallTagBuffer();

    buffer.push('<|tool_call_start|>[get_weather(location="Paris")]');
    const mid = buffer.push('<|tool_call_end|>[get_weather(location="Paris")]');
    expect(mid.safeText).toBe('');
    // The echo resolves only once its own end sentinel lands.
    expect(buffer.push('<|tool_call_end|>done.')).toEqual({
      safeText: 'done.',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.flush()).toBe('');
  });

  it('drops held LFM2 echo text at flush when no resolving end arrives', () => {
    const buffer = new ToolCallTagBuffer();

    buffer.push('<|tool_call_start|>[f(x=1)]<|tool_call_end|>');
    expect(buffer.push('[f(x=1)]').safeText).toBe('');
    // Held suspect text is markup, never released.
    expect(buffer.flush()).toBe('');
  });

  it('releases post-call prose when the whole LFM2 block lands in one final delta', () => {
    const buffer = new ToolCallTagBuffer();

    const pushed = buffer.push('intro <|tool_call_start|>[f()]<|tool_call_end|>outro');
    expect(pushed.tagFound).toBe(true);
    expect(pushed.cleanPrefix).toBe('intro ');
    // The end sentinel and prose were still buffered behind the tagFound
    // early-return — flush() must resolve them, not drop the prose with
    // the suppressed call body.
    expect(buffer.flush()).toBe('outro');
  });

  it('re-enters suppression on a second LFM2 start sentinel after prose', () => {
    const buffer = new ToolCallTagBuffer();

    buffer.push('<|tool_call_start|>[f()]<|tool_call_end|>');
    // A second call block re-enters interior mode; the genuine prose
    // before it is released through cleanPrefix.
    const second = buffer.push('between <|tool_call_start|>[g()');
    expect(second).toEqual({ safeText: '', tagFound: true, cleanPrefix: 'between ' });
    // The unclosed second body is held markup — dropped at flush.
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
