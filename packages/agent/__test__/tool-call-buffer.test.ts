import { describe, expect, it } from 'vite-plus/test';

import { ToolCallTagBuffer } from '../src/provider/tool-call-buffer.js';

describe('ToolCallTagBuffer', () => {
  it('passes plain text through unchanged', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('hello world')).toEqual({
      safeText: 'hello world',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.suppressed).toBe(false);
    expect(buffer.flush()).toBe('');
  });

  it('suppresses a <tool_call> tag split across deltas', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('<tool')).toEqual({ safeText: '', tagFound: false, cleanPrefix: '' });
    expect(buffer.push('_call>')).toEqual({ safeText: '', tagFound: true, cleanPrefix: '' });
    expect(buffer.suppressed).toBe(true);
    expect(buffer.flush()).toBe('');
  });

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

  it('recovers a false-alarm tag prefix once later text disambiguates it', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('<tool')).toEqual({ safeText: '', tagFound: false, cleanPrefix: '' });
    // "<tool shed" cannot extend into any structural tag, so the held
    // prefix is released together with the new text.
    expect(buffer.push(' shed is red')).toEqual({
      safeText: '<tool shed is red',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.suppressed).toBe(false);
    expect(buffer.flush()).toBe('');
  });

  it('recovers a still-ambiguous held prefix at flush', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('trailing <tool_cal')).toEqual({
      safeText: 'trailing ',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.flush()).toBe('<tool_cal');
    // flush drains the pending buffer — a second flush yields nothing.
    expect(buffer.flush()).toBe('');
  });

  it('returns text before the tag as cleanPrefix and suppresses the rest', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('Answer: <tool_call>{"name":"ls"}')).toEqual({
      safeText: '',
      tagFound: true,
      cleanPrefix: 'Answer: ',
    });
    expect(buffer.suppressed).toBe(true);
  });

  it('suppresses all text after a completed tool-call block until stream end', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('<tool_call>{"name":"ls"}').tagFound).toBe(true);
    expect(buffer.push('</tool_call>')).toEqual({ safeText: '', tagFound: false, cleanPrefix: '' });
    expect(buffer.push(' trailing prose')).toEqual({
      safeText: '',
      tagFound: false,
      cleanPrefix: '',
    });
    expect(buffer.suppressed).toBe(true);
    expect(buffer.flush()).toBe('');
  });

  it('picks the earliest tag when multiple tags appear in one chunk', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('a<|channel>x<tool_call>y')).toEqual({
      safeText: '',
      tagFound: true,
      cleanPrefix: 'a',
    });
  });

  it('holds back only the ambiguous suffix and emits the safe part immediately', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('Hello <')).toEqual({ safeText: 'Hello ', tagFound: false, cleanPrefix: '' });
    expect(buffer.push('world')).toEqual({ safeText: '<world', tagFound: false, cleanPrefix: '' });
    expect(buffer.flush()).toBe('');
  });
});
