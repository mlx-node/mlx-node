import { describe, expect, it } from 'vite-plus/test';

import {
  buildAnthropicContent,
  containsToolCallMarkup,
  recoverSuppressedToolCallText,
  visibleAnswerText,
} from '../../packages/server/src/mappers/anthropic-response.js';

function hy4Call(x: string, value = 'Paris'): string {
  return (
    `<tool_call:${x}>get_weather` +
    `<arg_key:${x}>city</arg_key:${x}><arg_value:${x}>${value}</arg_value:${x}>` +
    `<arg_key:${x}>days</arg_key:${x}><arg_value:${x}>3</arg_value:${x}>` +
    `</tool_call:${x}>`
  );
}

function hy4Block(x: string, value?: string): string {
  return `<tool_calls:${x}>${hy4Call(x, value)}</tool_calls:${x}>`;
}

function dsmlBlock(value = 'value'): string {
  return (
    '<｜DSML｜ calls>\n' +
    '<｜DSML｜ invoke name="lookup">\n' +
    `<｜DSML｜ parameter name="query" string="true">${value}</｜DSML｜ parameter>\n` +
    '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n' +
    '</｜DSML｜ invoke>\n' +
    '</｜DSML｜ calls>'
  );
}

describe('recoverSuppressedToolCallText: Hy4-flash', () => {
  for (const x of ['6124c78e', 'opensource']) {
    describe(`:${x}`, () => {
      it('removes a complete block and keeps the prose around it', () => {
        expect(recoverSuppressedToolCallText(`Let me check.\n${hy4Block(x)}\nDone.`)).toBe('Let me check.\n\nDone.');
        expect(recoverSuppressedToolCallText(hy4Block(x))).toBe('');
      });

      it('removes every block', () => {
        expect(recoverSuppressedToolCallText(`a ${hy4Block(x)} b ${hy4Block(x)} c`)).toBe('a  b  c');
      });

      it('removes an unclosed block to the end', () => {
        const raw = `pre <tool_calls:${x}><tool_call:${x}>get_weather<arg_key:${x}>city</arg_key:${x}><arg_value:${x}>Par`;
        expect(recoverSuppressedToolCallText(raw)).toBe('pre ');
        expect(recoverSuppressedToolCallText(`pre <tool_calls:${x}>`)).toBe('pre ');
      });

      it('removes stray inner blocks and close tags', () => {
        expect(recoverSuppressedToolCallText(`x ${hy4Call(x)} y`)).toBe('x  y');
        expect(
          recoverSuppressedToolCallText(`x <arg_key:${x}>k</arg_key:${x}><arg_value:${x}>v</arg_value:${x}> y`),
        ).toBe('x  y');
        expect(
          recoverSuppressedToolCallText(
            `a </tool_calls:${x}> b </tool_call:${x}> c </arg_key:${x}> d </arg_value:${x}> e`,
          ),
        ).toBe('a  b  c  d  e');
      });

      it('a value holding the block close does not end the block', () => {
        const raw = `${hy4Block(x, `see </tool_calls:${x}> here`)}after`;
        expect(recoverSuppressedToolCallText(raw)).toBe('after');
      });
    });
  }

  it('keeps text that is not a Hy4 tag', () => {
    for (const raw of ['use <tool_calls: x> here', '<tool_calls:a b>', '<tool_calls:>', 'a <tool_calls b']) {
      expect(recoverSuppressedToolCallText(raw)).toBe(raw);
    }
  });
});

describe('recoverSuppressedToolCallText: DeepSeek V4.1 DSML', () => {
  it('removes a complete block and keeps the prose around it', () => {
    expect(recoverSuppressedToolCallText(`summary\n\n${dsmlBlock()}\nafter`)).toBe('summary\n\n\nafter');
    expect(recoverSuppressedToolCallText(dsmlBlock())).toBe('');
  });

  it('removes every block', () => {
    expect(recoverSuppressedToolCallText(`a ${dsmlBlock()} b ${dsmlBlock()} c`)).toBe('a  b  c');
  });

  it('removes an unclosed block to the end', () => {
    const raw =
      'pre <｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n<｜DSML｜ parameter name="query" string="true">val';
    expect(recoverSuppressedToolCallText(raw)).toBe('pre ');
  });

  it('removes stray inner blocks and close tags', () => {
    expect(recoverSuppressedToolCallText('x <｜DSML｜ invoke name="f">\n</｜DSML｜ invoke> y')).toBe('x  y');
    expect(recoverSuppressedToolCallText('x <｜DSML｜ parameter name="q" string="true">v</｜DSML｜ parameter> y')).toBe(
      'x  y',
    );
    expect(recoverSuppressedToolCallText('a </｜DSML｜ calls> b </｜DSML｜ invoke> c </｜DSML｜ parameter> d')).toBe(
      'a  b  c  d',
    );
  });

  it('a parameter value holding the block close does not end the block', () => {
    expect(recoverSuppressedToolCallText(`${dsmlBlock('see </｜DSML｜ calls> here')}after`)).toBe('after');
  });
});

// Outputs pinned from the implementation before the Hy4 and DSML cases.
describe('recoverSuppressedToolCallText: other families unchanged', () => {
  const pinned: [string, string][] = [
    [
      '<think>\nneed weather\n</think>\n\nLet me look.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\nDone.',
      '<think>\nneed weather\n</think>\n\nLet me look.\n\nDone.',
    ],
    ['pre <tool_call>{"name":"a"}</tool_call> mid <tool_call>{"name":"b","arguments":{"q":"x"}', 'pre  mid '],
    ['a </tool_call> b', 'a </tool_call> b'],
    [
      '<|channel>thought\nplan<channel|>Sure. <|tool_call>call:list_files{path:<|"|>.<|"|>}<tool_call|><|tool_response>ok<tool_response|> end<turn|>',
      'Sure.  end',
    ],
    ['<|turn>model\nhello <|tool>decl<tool|>world', 'hello world'],
    [
      'The weather <|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>[get_weather(city="Paris")]<|tool_call_end|> is sunny <|tool_call_start|>[f(',
      'The weather  is sunny ',
    ],
  ];
  for (const [raw, out] of pinned) {
    it(JSON.stringify(raw.slice(0, 40)), () => {
      expect(recoverSuppressedToolCallText(raw)).toBe(out);
    });
  }
});

describe('visibleAnswerText with tool use not allowed', () => {
  const ok = {
    id: 'call_1',
    name: 'get_weather',
    arguments: { city: 'Paris', days: '3' },
    status: 'ok',
    rawContent: '',
  };

  for (const x of ['6124c78e', 'opensource']) {
    it(`a Hy4 :${x} turn shows no markup`, () => {
      const call = { ...ok, rawContent: hy4Call(x) };
      expect(visibleAnswerText({ text: '', rawText: hy4Block(x), toolCalls: [call] }, false)).toBe('');
      expect(
        visibleAnswerText({ text: '', rawText: `${hy4Block(x)}\nThe forecast follows.`, toolCalls: [call] }, false),
      ).toBe('\nThe forecast follows.');
      const result = {
        text: '',
        rawText: hy4Block(x),
        toolCalls: [call],
        thinking: null,
        finishReason: 'tool_calls',
      } as unknown as Parameters<typeof buildAnthropicContent>[0];
      expect(buildAnthropicContent(result, false)).toEqual([{ type: 'text', text: '' }]);
    });
  }

  it('a DSML turn reaches the recovery and shows no markup', () => {
    expect(containsToolCallMarkup(dsmlBlock())).toBe(true);
    const call = { ...ok, name: 'lookup', rawContent: '' };
    expect(visibleAnswerText({ text: '', rawText: `${dsmlBlock()}\ntrailing`, toolCalls: [call] }, false)).toBe(
      '\ntrailing',
    );
  });
});
