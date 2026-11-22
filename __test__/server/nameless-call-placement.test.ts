import { describe, expect, it } from 'vite-plus/test';

import { visibleAnswerText } from '../../packages/server/src/mappers/anthropic-response.js';
import { StopSequenceBuffer } from '../../packages/server/src/stop-sequence-buffer.js';

// The native shapes of a turn with a nameless call, one per family parser, read
// as the non-streaming Messages handler reads them: the visible answer, then the
// stop scan over it (`messages.ts` `handleNonStreaming`). Mutation: place
// in `rawText` → red (the reasoning's copy of the block is found first and the
// block goes after the text, past the stop).

interface Call {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: string;
  rawContent: string;
}

interface Turn {
  text: string;
  rawText: string;
  publicRawText?: string;
  toolCalls: Call[];
}

function answer(turn: Turn, stops: string[] = []): string {
  const visible = visibleAnswerText(turn, true);
  const buffer = new StopSequenceBuffer(stops);
  const pushed = buffer.push(visible);
  const flushed = buffer.flush();
  return (pushed.matched ?? flushed.matched) === null ? visible : pushed.safeText + flushed.safeText;
}

function nameless(rawContent: string): Call {
  return { id: 'call_n', name: '', arguments: {}, status: 'missing_name', rawContent };
}

// `tools/chatml.rs` parses the content after the first `</think>`;
// `engine/finalize.rs` `raw_text_with_reasoning_suppressed` cuts `publicRawText`
// at that close, and `rawText` keeps the reasoning when it is requested.
describe('ChatML families (Qwen3, Qwen3.5, Nemotron-H, LFM2)', () => {
  const block = '<tool_call>\n<function=>\n<parameter=q>\nxSTOPy\n</parameter>\n</function>\n</tool_call>';
  const okBlock = '<tool_call>\n<function=search>\n<parameter=q>\nz\n</parameter>\n</function>\n</tool_call>';
  const okCall: Call = { id: 'call_ok', name: 'search', arguments: { q: 'z' }, status: 'ok', rawContent: okBlock };

  function turn(content: string, text: string, toolCalls: Call[], quoted: string): Turn {
    const publicRawText = `\n\n${content}`;
    return { text, rawText: `see ${quoted}\n</think>${publicRawText}`, publicRawText, toolCalls };
  }

  it('the block sits between the prose', () => {
    expect(answer(turn(`BEFORE ${block} AFTER`, 'BEFORE  AFTER', [nameless(block)], block))).toBe(
      `BEFORE ${block} AFTER`,
    );
  });

  it('a stop inside the block cuts the text after it', () => {
    expect(answer(turn(`BEFORE ${block} AFTER`, 'BEFORE  AFTER', [nameless(block)], block), ['STOP'])).toBe(
      'BEFORE <tool_call>\n<function=>\n<parameter=q>\nx',
    );
  });

  it('a stop in the prose after the block keeps the block', () => {
    const plain = block.replace('xSTOPy', 'xy');
    expect(answer(turn(`BEFORE ${plain} AF HALT TER`, 'BEFORE  AF HALT TER', [nameless(plain)], plain), ['HALT'])).toBe(
      `BEFORE ${plain} AF `,
    );
  });

  it('a named call leaves the text and the block keeps its place', () => {
    const content = `BEFORE ${okBlock} MID ${block} AFTER`;
    expect(answer(turn(content, 'BEFORE  MID  AFTER', [okCall, nameless(block)], `${okBlock}${block}`))).toBe(
      `BEFORE  MID ${block} AFTER`,
    );
  });

  it('with thinking off the raw texts are the same', () => {
    const raw = `BEFORE ${block} AFTER`;
    const result = { text: 'BEFORE  AFTER', rawText: raw, publicRawText: raw, toolCalls: [nameless(block)] };
    expect(answer(result, ['STOP'])).toBe('BEFORE <tool_call>\n<function=>\n<parameter=q>\nx');
  });

  it('a block only inside the reasoning is no call and no text', () => {
    const result = turn('ANSWER', 'ANSWER', [], block);
    expect(answer(result, ['STOP'])).toBe('ANSWER');
  });
});

// `hy_v4_flash/finalize.rs` `parse_content`: the text is the content before
// `<tool_calls:X>` and text after the block is dropped (vLLM
// `hy_v4_tool_parser.py`), so after the text is the block's place.
describe('Hy4-flash', () => {
  const args = '<arg_key:opensource>q</arg_key:opensource><arg_value:opensource>xSTOPy</arg_value:opensource>';
  const call = `<tool_call:opensource>${args}</tool_call:opensource>`;
  const content = `PRE <tool_calls:opensource>${call}</tool_calls:opensource>`;
  const result: Turn = {
    text: 'PRE',
    rawText: `<think:opensource>see ${call}</think:opensource>${content}`,
    publicRawText: content,
    toolCalls: [nameless(call)],
  };

  it('the block follows the text', () => {
    expect(answer(result)).toBe(`PRE${call}`);
  });

  it('a stop inside the block cuts the text after it', () => {
    expect(answer(result, ['STOP'])).toBe(
      'PRE<tool_call:opensource><arg_key:opensource>q</arg_key:opensource><arg_value:opensource>x',
    );
  });
});

// `k2_horizon/finalize.rs` `classify_k2_tool_call`: a first line that is a tag
// is a nameless call; `rawContent` is the whole `<ifm|tool_call>` block.
describe('K2-Horizon', () => {
  const call = '<ifm|tool_call>\n<ifm|arg_key>q</ifm|arg_key>\n<ifm|arg_value>xSTOPy</ifm|arg_value>\n</ifm|tool_call>';
  const content = `PRE\n<ifm|tool_calls>\n${call}\n</ifm|tool_calls>`;
  const result: Turn = {
    text: 'PRE',
    rawText: `see ${call}</ifm|think>${content}`,
    publicRawText: content,
    toolCalls: [nameless(call)],
  };

  it('a call after the text follows it', () => {
    expect(answer(result)).toBe(`PRE${call}`);
  });

  it('a stop inside the block cuts the text after it', () => {
    expect(answer(result, ['STOP'])).toBe('PRE<ifm|tool_call>\n<ifm|arg_key>q</ifm|arg_key>\n<ifm|arg_value>x');
  });
});

// `gemma4/output_parser.rs` `drain_tool_call`: `rawContent` is the body between
// `<|tool_call>` and `<tool_call|>`, and the result has no `publicRawText`.
describe('Gemma4', () => {
  const body = 'call:{q:xSTOPy}';
  const result: Turn = {
    text: 'PRE',
    rawText: `<|channel>thought\nsee ${body}<channel|>PRE<|tool_call>${body}<tool_call|>`,
    toolCalls: [nameless(body)],
  };

  it('a call after the text follows it', () => {
    expect(answer(result)).toBe(`PRE${body}`);
  });

  it('a stop inside the block cuts the text after it', () => {
    expect(answer(result, ['STOP'])).toBe('PREcall:{q:x');
  });
});
