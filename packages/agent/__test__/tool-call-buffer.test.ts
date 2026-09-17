import { ToolCallTagBuffer } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

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

  // LFM2 cases mirror __test__/server/tool-call-buffer.test.ts — both
  // exercise the shared ToolCallTagBuffer (now in @mlx-node/lm).
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
});
