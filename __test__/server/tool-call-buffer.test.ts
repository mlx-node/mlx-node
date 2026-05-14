import { ToolCallTagBuffer } from '@mlx-node/lm/tools';
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

  it('resets suppression state for a new stream', () => {
    const buffer = new ToolCallTagBuffer();

    expect(buffer.push('<tool_call>{"name":"x"}</tool_call>')).toEqual({
      safeText: '',
      tagFound: true,
      cleanPrefix: '',
    });
    expect(buffer.suppressed).toBe(true);

    buffer.reset();

    expect(buffer.suppressed).toBe(false);
    expect(buffer.push('visible')).toEqual({
      safeText: 'visible',
      tagFound: false,
      cleanPrefix: '',
    });
  });

  it('strips parsed tool-call and tool-response blocks when tool use is disallowed', () => {
    const raw =
      '<|channel>thought\nneed a file list<channel|><|tool_call>call:list_files{path:<|"|>.<|"|>}<tool_call|><|tool_response>';

    expect(recoverSuppressedToolCallText(raw)).toBe('');
    expect(recoverSuppressedToolCallText(`visible ${raw}`)).toBe('visible ');
  });
});
