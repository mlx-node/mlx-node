import { describe, expect, it } from 'vite-plus/test';

import { recoverSuppressedToolCallText } from '../../packages/server/src/mappers/anthropic-response.js';
import { ToolCallTagBuffer } from '../../packages/server/src/tool-call-buffer.js';

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
});
