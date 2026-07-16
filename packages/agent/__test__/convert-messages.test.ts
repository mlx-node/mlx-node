import type {
  AssistantMessage,
  Context,
  StopReason,
  Tool,
  ToolResultMessage,
  Usage,
  UserMessage,
} from '@earendil-works/pi-ai';
import { Type } from '@earendil-works/pi-ai';
import type { TSchema } from '@earendil-works/pi-ai';
import { describe, expect, it } from 'vite-plus/test';

import { contextToChatMessages, toolsToDefinitions } from '../src/provider/convert-messages.js';

function zeroUsage(): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function userMsg(content: UserMessage['content']): UserMessage {
  return { role: 'user', content, timestamp: 1 };
}

function assistantMsg(content: AssistantMessage['content'], stopReason: StopReason = 'stop'): AssistantMessage {
  return {
    role: 'assistant',
    content,
    api: 'openai-completions',
    provider: 'mlx-node',
    model: 'test-model',
    usage: zeroUsage(),
    stopReason,
    timestamp: 2,
  };
}

function toolResultMsg(toolCallId: string, content: ToolResultMessage['content'], isError = false): ToolResultMessage {
  return { role: 'toolResult', toolCallId, toolName: 'get_weather', content, isError, timestamp: 3 };
}

describe('contextToChatMessages', () => {
  it('converts system + user + assistant-with-2-toolCalls + 2 toolResults preserving order and ids', () => {
    const context: Context = {
      systemPrompt: 'You are a helpful agent.',
      messages: [
        userMsg('Weather in Tokyo and Osaka?'),
        assistantMsg(
          [
            { type: 'text', text: 'Checking both cities.' },
            { type: 'toolCall', id: 'call_1', name: 'get_weather', arguments: { location: 'Tokyo' } },
            { type: 'toolCall', id: 'call_2', name: 'get_weather', arguments: { location: 'Osaka' } },
          ],
          'toolUse',
        ),
        toolResultMsg('call_1', [{ type: 'text', text: 'Tokyo: sunny' }]),
        toolResultMsg('call_2', [{ type: 'text', text: 'Osaka: rain' }], true),
        userMsg('Thanks!'),
      ],
    };

    expect(contextToChatMessages(context)).toEqual([
      { role: 'system', content: 'You are a helpful agent.' },
      { role: 'user', content: 'Weather in Tokyo and Osaka?' },
      {
        role: 'assistant',
        content: 'Checking both cities.',
        toolCalls: [
          { id: 'call_1', name: 'get_weather', arguments: '{"location":"Tokyo"}' },
          { id: 'call_2', name: 'get_weather', arguments: '{"location":"Osaka"}' },
        ],
      },
      { role: 'tool', content: 'Tokyo: sunny', toolCallId: 'call_1', isError: false },
      { role: 'tool', content: 'Osaka: rain', toolCallId: 'call_2', isError: true },
      { role: 'user', content: 'Thanks!' },
    ]);
  });

  it('omits the system message when the context has no systemPrompt', () => {
    const converted = contextToChatMessages({ messages: [userMsg('hi')] });
    expect(converted).toEqual([{ role: 'user', content: 'hi' }]);
  });

  it('replaces image parts with [image omitted] placeholder lines for text-only models', () => {
    const context: Context = {
      messages: [
        userMsg([
          { type: 'text', text: 'What is in this picture?' },
          { type: 'image', data: 'aGk=', mimeType: 'image/png' },
        ]),
        assistantMsg([{ type: 'toolCall', id: 'call_1', name: 'screenshot', arguments: {} }], 'toolUse'),
        toolResultMsg('call_1', [
          { type: 'image', data: 'aGk=', mimeType: 'image/png' },
          { type: 'text', text: 'captured' },
        ]),
      ],
    };

    const [user, , tool] = contextToChatMessages(context);
    expect(user!.content).toBe('What is in this picture?\n[image omitted]');
    expect(tool!.content).toBe('[image omitted]\ncaptured');
    expect(user!.images).toBeUndefined();
    expect(tool!.images).toBeUndefined();
  });

  it('keeps user images direct and moves tool-result images to a synthetic user message', () => {
    const context: Context = {
      messages: [
        userMsg([
          { type: 'text', text: 'What is in this picture?' },
          { type: 'image', data: 'aGk=', mimeType: 'image/png' },
          { type: 'text', text: 'Be concise.' },
        ]),
        assistantMsg([{ type: 'toolCall', id: 'call_1', name: 'screenshot', arguments: {} }], 'toolUse'),
        toolResultMsg('call_1', [
          { type: 'image', data: 'AQID', mimeType: 'image/png' },
          { type: 'text', text: 'captured' },
          { type: 'image', data: 'BAU=', mimeType: 'image/jpeg' },
        ]),
      ],
    };

    const [user, , tool, toolImages] = contextToChatMessages(context, true);
    expect(user).toMatchObject({ role: 'user', content: 'What is in this picture?\nBe concise.' });
    expect(user!.images?.map((image) => [...image])).toEqual([[104, 105]]);
    expect(tool).toMatchObject({ role: 'tool', content: 'captured', toolCallId: 'call_1', isError: false });
    expect(tool!.images).toBeUndefined();
    expect(toolImages).toMatchObject({ role: 'user', content: 'Attached image(s) from tool result:' });
    expect(toolImages!.images?.map((image) => [...image])).toEqual([
      [1, 2, 3],
      [4, 5],
    ]);
  });

  it('groups consecutive tool-result images after all parallel textual results', () => {
    const converted = contextToChatMessages(
      {
        messages: [
          assistantMsg(
            [
              { type: 'toolCall', id: 'call_1', name: 'read_first', arguments: {} },
              { type: 'toolCall', id: 'call_2', name: 'read_second', arguments: {} },
            ],
            'toolUse',
          ),
          toolResultMsg('call_1', [
            { type: 'image', data: 'AQ==', mimeType: 'image/png' },
            { type: 'image', data: 'Ag==', mimeType: 'image/png' },
          ]),
          toolResultMsg(
            'call_2',
            [
              { type: 'text', text: 'second result' },
              { type: 'image', data: 'Aw==', mimeType: 'image/jpeg' },
            ],
            true,
          ),
        ],
      },
      true,
    );

    expect(converted).toHaveLength(4);
    expect(converted[0]).toMatchObject({ role: 'assistant' });
    expect(converted[1]).toEqual({
      role: 'tool',
      content: '(see attached image)',
      toolCallId: 'call_1',
      isError: false,
    });
    expect(converted[2]).toEqual({
      role: 'tool',
      content: 'second result',
      toolCallId: 'call_2',
      isError: true,
    });
    expect(converted[3]).toMatchObject({ role: 'user', content: 'Attached image(s) from tool result:' });
    expect(converted[3]!.images?.map((image) => [...image])).toEqual([[1], [2], [3]]);
  });

  it('repairs a missing sibling tool result before the grouped image user turn', () => {
    const converted = contextToChatMessages(
      {
        messages: [
          assistantMsg(
            [
              { type: 'toolCall', id: 'call_1', name: 'read_first', arguments: {} },
              { type: 'toolCall', id: 'call_2', name: 'read_second', arguments: {} },
            ],
            'toolUse',
          ),
          toolResultMsg('call_1', [{ type: 'image', data: 'AQ==', mimeType: 'image/png' }]),
          userMsg('continue'),
        ],
      },
      true,
    );

    expect(converted.slice(1)).toEqual([
      { role: 'tool', content: '(see attached image)', toolCallId: 'call_1', isError: false },
      { role: 'tool', content: 'No result provided', toolCallId: 'call_2', isError: true },
      {
        role: 'user',
        content: 'Attached image(s) from tool result:',
        images: [Buffer.from([1])],
      },
      { role: 'user', content: 'continue' },
    ]);
  });

  it("strips Pi's stale non-vision note from an image-bearing tool result", () => {
    const note = '[Current model does not support images. The image will be omitted from this request.]';
    const converted = contextToChatMessages(
      {
        messages: [
          toolResultMsg('call_image', [
            { type: 'text', text: `Read image file [image/png]\n${note}` },
            { type: 'image', data: 'AQID', mimeType: 'image/png' },
          ]),
          userMsg([{ type: 'text', text: `Literal text without an image:\n${note}` }]),
        ],
      },
      true,
    );

    expect(converted[0]).toMatchObject({ role: 'tool', content: 'Read image file [image/png]' });
    expect(converted[0]!.images).toBeUndefined();
    expect(converted[1]).toMatchObject({ role: 'user', content: 'Attached image(s) from tool result:' });
    expect(converted[1]!.images?.map((image) => [...image])).toEqual([[1, 2, 3]]);
    expect(converted[2]?.content).toBe(`Literal text without an image:\n${note}`);
  });

  it("strips Pi's stale non-vision note from a failed text-only image tool result", () => {
    const note = '[Current model does not support images. The image will be omitted from this request.]';
    const processingFailure = '[Image omitted: Failed to process image bytes]';
    const converted = contextToChatMessages(
      {
        messages: [
          toolResultMsg('call_failed_image', [
            { type: 'text', text: `Read image file [image/png]\n${processingFailure}\n${note}` },
          ]),
        ],
      },
      true,
    );

    expect(converted).toEqual([
      {
        role: 'tool',
        content: `Read image file [image/png]\n${processingFailure}`,
        toolCallId: 'call_failed_image',
        isError: false,
      },
    ]);
  });

  it("preserves Pi's non-vision note when it is literal direct-user text beside an image", () => {
    const note = '[Current model does not support images. The image will be omitted from this request.]';
    const converted = contextToChatMessages(
      {
        messages: [
          userMsg([
            { type: 'text', text: `Do not rewrite this literal line:\n${note}` },
            { type: 'image', data: 'AQID', mimeType: 'image/png' },
          ]),
        ],
      },
      true,
    );

    expect(converted[0]?.content).toBe(`Do not rewrite this literal line:\n${note}`);
    expect(converted[0]!.images?.map((image) => [...image])).toEqual([[1, 2, 3]]);
  });

  it('joins multiple user text parts with newlines', () => {
    const converted = contextToChatMessages({
      messages: [
        userMsg([
          { type: 'text', text: 'line one' },
          { type: 'text', text: 'line two' },
        ]),
      ],
    });
    expect(converted[0]!.content).toBe('line one\nline two');
  });

  it('preserves assistant reasoning bodies for byte-stable native template replay', () => {
    const converted = contextToChatMessages({
      messages: [
        assistantMsg([
          { type: 'thinking', thinking: 'pondering...' },
          { type: 'thinking', thinking: 'still pondering...' },
          { type: 'text', text: 'The answer is 4.' },
        ]),
      ],
    });
    expect(converted).toEqual([
      {
        role: 'assistant',
        content: 'The answer is 4.',
        reasoningContent: 'pondering...still pondering...',
      },
    ]);
  });

  it('preserves the per-turn MLX thinking mode used to render historical assistant bytes', () => {
    const thinkingOff = assistantMsg([{ type: 'text', text: 'Direct answer.' }]) as AssistantMessage & {
      mlxThinkingEnabled?: boolean;
    };
    thinkingOff.mlxThinkingEnabled = false;
    const thinkingOn = assistantMsg([{ type: 'text', text: 'No visible reasoning this time.' }]) as AssistantMessage & {
      mlxThinkingEnabled?: boolean;
    };
    thinkingOn.mlxThinkingEnabled = true;

    expect(contextToChatMessages({ messages: [thinkingOff, thinkingOn] })).toEqual([
      { role: 'assistant', content: 'Direct answer.', thinkingEnabled: false },
      { role: 'assistant', content: 'No visible reasoning this time.', thinkingEnabled: true },
    ]);
  });

  it('skips husk assistant messages (aborted/error with no text and no toolCalls)', () => {
    const converted = contextToChatMessages({
      messages: [
        userMsg('q1'),
        assistantMsg([], 'aborted'),
        assistantMsg([{ type: 'thinking', thinking: 'only thoughts' }], 'error'),
        userMsg('q2'),
      ],
    });
    expect(converted).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'user', content: 'q2' },
    ]);
  });

  it('DROPS error/aborted assistant messages even when they carry partial text (mirrors pi transformMessages)', () => {
    // The core WB-4 regression: an aborted/errored turn with PARTIAL content is
    // an incomplete turn pi never replays. Priming it into the reset session
    // garbles the continuation. Mutation guard: reverting to the empty-husk-only
    // check keeps these partial turns and this test fails.
    const converted = contextToChatMessages({
      messages: [
        userMsg('q1'),
        assistantMsg([{ type: 'text', text: 'partial ans' }], 'aborted'),
        assistantMsg([{ type: 'text', text: 'half an error' }], 'error'),
        userMsg('q2'),
      ],
    });
    expect(converted).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'user', content: 'q2' },
    ]);
  });

  it('DROPS an error/aborted assistant that carries a tool call, leaving no dangling toolCalls', () => {
    const converted = contextToChatMessages({
      messages: [
        userMsg('q1'),
        assistantMsg([{ type: 'toolCall', id: 'call_7', name: 'ls', arguments: {} }], 'error'),
        userMsg('q2'),
      ],
    });
    // The dropped turn's tool call is NOT tracked, so no synthetic tool result
    // is emitted for it either — it vanishes entirely.
    expect(converted).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'user', content: 'q2' },
    ]);
    expect(converted.some((m) => m.role === 'tool')).toBe(false);
  });

  it('synthesizes a No-result tool message for a RETAINED assistant tool call with no following result', () => {
    // pi's insertSyntheticToolResults: a completed turn whose tool call is never
    // answered gets a synthetic error result before the next user/assistant, so
    // no tool call is left unresolved in the primed history. Mutation guard:
    // dropping the orphan-repair pass omits the synthetic tool message → fails.
    const converted = contextToChatMessages({
      messages: [
        userMsg('do it'),
        assistantMsg([{ type: 'toolCall', id: 'call_9', name: 'ls', arguments: {} }], 'toolUse'),
        userMsg('never mind'),
      ],
    });
    expect(converted).toEqual([
      { role: 'user', content: 'do it' },
      { role: 'assistant', content: '', toolCalls: [{ id: 'call_9', name: 'ls', arguments: '{}' }] },
      { role: 'tool', content: 'No result provided', toolCallId: 'call_9', isError: true },
      { role: 'user', content: 'never mind' },
    ]);
  });

  it('synthesizes a No-result tool message for a trailing unresolved tool call at end of history', () => {
    const converted = contextToChatMessages({
      messages: [assistantMsg([{ type: 'toolCall', id: 'call_end', name: 'ls', arguments: {} }], 'toolUse')],
    });
    expect(converted).toEqual([
      { role: 'assistant', content: '', toolCalls: [{ id: 'call_end', name: 'ls', arguments: '{}' }] },
      { role: 'tool', content: 'No result provided', toolCallId: 'call_end', isError: true },
    ]);
  });

  it('keeps completed assistant messages with empty content (stop is not a husk)', () => {
    const converted = contextToChatMessages({ messages: [assistantMsg([], 'stop')] });
    expect(converted).toEqual([{ role: 'assistant', content: '' }]);
  });
});

describe('toolsToDefinitions', () => {
  it('returns undefined for undefined and empty tool lists', () => {
    expect(toolsToDefinitions(undefined)).toBeUndefined();
    expect(toolsToDefinitions([])).toBeUndefined();
  });

  it("converts pi's real bash tool schema with a parseable properties JSON round-trip", () => {
    // Exact schema construction from pi-coding-agent dist/core/tools/bash.js.
    const bashSchema = Type.Object({
      command: Type.String({ description: 'Bash command to execute' }),
      timeout: Type.Optional(Type.Number({ description: 'Timeout in seconds (optional, no default timeout)' })),
    });
    const bashTool: Tool = {
      name: 'bash',
      description: 'Executes bash commands',
      parameters: bashSchema,
    };

    const defs = toolsToDefinitions([bashTool]);
    expect(defs).toHaveLength(1);
    const def = defs![0]!;
    expect(def.type).toBe('function');
    expect(def.function.name).toBe('bash');
    expect(def.function.description).toBe('Executes bash commands');
    expect(def.function.parameters?.type).toBe('object');
    expect(def.function.parameters?.required).toEqual(['command']);
    expect(JSON.parse(def.function.parameters!.properties!)).toEqual({
      command: { description: 'Bash command to execute', type: 'string' },
      timeout: { description: 'Timeout in seconds (optional, no default timeout)', type: 'number' },
    });
  });

  it('preserves key order as given and is byte-stable across calls', () => {
    const schema = {
      type: 'object',
      properties: {
        zeta: { type: 'string' },
        alpha: { type: 'number' },
      },
      required: ['zeta'],
    } as unknown as TSchema;
    const tool: Tool = { name: 't', description: 'd', parameters: schema };

    const first = toolsToDefinitions([tool])![0]!.function.parameters!.properties!;
    const second = toolsToDefinitions([tool])![0]!.function.parameters!.properties!;
    expect(second).toBe(first); // byte-stable across calls
    expect(first).toBe('{"zeta":{"type":"string"},"alpha":{"type":"number"}}');
  });

  it('serializes a schema without properties/required as empty properties', () => {
    const tool: Tool = {
      name: 'noargs',
      description: 'takes nothing',
      parameters: { type: 'object' } as unknown as TSchema,
    };
    const def = toolsToDefinitions([tool])![0]!;
    expect(def.function.parameters?.properties).toBe('{}');
    expect(def.function.parameters?.required).toBeUndefined();
  });
});
