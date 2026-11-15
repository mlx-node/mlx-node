import type { ServerResponse } from 'node:http';
import { Writable } from 'node:stream';

import type { ToolCallResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { handleCreateMessage } from '../../packages/server/src/endpoints/messages.js';
import { nestStrayArguments, withNestedArguments } from '../../packages/server/src/mappers/tool-arguments.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import type { AnthropicToolDefinition } from '../../packages/server/src/types-anthropic.js';

/** The `use_tool` definition the agent sends on a Messages backend (recorded from mlx-agent v0.3.3). */
const USE_TOOL: AnthropicToolDefinition = {
  name: 'use_tool',
  description: 'Call a discovered MCP integration tool with `tool_name` and `tool_input`.',
  input_schema: {
    type: 'object',
    properties: {
      tool_name: { description: 'Discovered MCP target name', type: 'string' },
      tool_input: {
        description: 'Inline remote arguments; use the discovered input schema',
        type: 'object',
        additionalProperties: true,
      },
    },
    required: ['tool_name', 'tool_input'],
  },
};
const SEARCH_TOOL: AnthropicToolDefinition = {
  name: 'search_tool',
  input_schema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
};
const PAGE = 'https://www.example.com/dy/article/PROBE0001.html';
const FLAT = { tool_name: 'mlx_web__web_fetch', url: PAGE };
const NESTED = { tool_name: 'mlx_web__web_fetch', tool_input: { url: PAGE } };

function call(name: string, args: Record<string, unknown> | string, status = 'ok'): ToolCallResult {
  return { id: 'call_0001', name, arguments: args, status, rawContent: '' };
}

describe('nestStrayArguments', () => {
  it('moves the flat keys of a use_tool call into tool_input', () => {
    expect(nestStrayArguments(FLAT, USE_TOOL.input_schema)).toEqual(NESTED);
    expect(
      JSON.stringify(
        nestStrayArguments({ tool_name: 'mlx_web__web_search', query: 'q', count: 3 }, USE_TOOL.input_schema),
      ),
    ).toBe('{"tool_name":"mlx_web__web_search","tool_input":{"query":"q","count":3}}');
  });

  it('keeps a call that already has every required property', () => {
    expect(nestStrayArguments(NESTED, USE_TOOL.input_schema)).toBeNull();
    expect(nestStrayArguments({ ...NESTED, extra: 'y' }, USE_TOOL.input_schema)).toBeNull();
  });

  it('keeps a call with no stray key to move', () => {
    expect(nestStrayArguments({ tool_name: 'mlx_web__web_fetch' }, USE_TOOL.input_schema)).toBeNull();
  });

  it('keeps a call that misses two required properties', () => {
    const objectFirst = { ...USE_TOOL.input_schema, required: ['tool_input', 'tool_name'] };
    expect(nestStrayArguments({ url: PAGE }, objectFirst)).toBeNull();
  });

  it('keeps a call whose missing property is not an object', () => {
    const schema = {
      ...USE_TOOL.input_schema,
      properties: { tool_name: { type: 'string' }, tool_input: { type: 'string' } },
    };
    expect(nestStrayArguments(FLAT, schema)).toBeNull();
  });

  it('keeps a call against a schema with a root union (the file forms)', () => {
    const schema = {
      ...USE_TOOL.input_schema,
      oneOf: [{ required: ['tool_name', 'tool_input'] }, { required: ['file'] }],
    };
    expect(nestStrayArguments(FLAT, schema)).toBeNull();
  });

  it.each(['anyOf', 'allOf', 'not', '$ref', 'patternProperties'])(
    'keeps a call against a schema with a root %s',
    (key) => {
      expect(nestStrayArguments(FLAT, { ...USE_TOOL.input_schema, [key]: {} })).toBeNull();
    },
  );

  it('keeps a call whose root admits extra keys', () => {
    expect(nestStrayArguments(FLAT, { ...USE_TOOL.input_schema, additionalProperties: true })).toBeNull();
    expect(nestStrayArguments(FLAT, { ...USE_TOOL.input_schema, additionalProperties: { type: 'string' } })).toBeNull();
  });

  it('keeps the arguments-file form the agent accepts without declaring it', () => {
    const fileForm = { tool_name: 'mlx_web__web_fetch', tool_input_file: '/tmp/args.json' };
    expect(nestStrayArguments(fileForm, USE_TOOL.input_schema)).toBeNull();
    expect(nestStrayArguments({ ...fileForm, url: PAGE }, USE_TOOL.input_schema)).toBeNull();
  });

  it('keeps a call whose stray keys the missing object does not admit', () => {
    const schema = {
      ...USE_TOOL.input_schema,
      properties: {
        tool_name: { type: 'string' },
        tool_input: { type: 'object', properties: { query: { type: 'string' } }, additionalProperties: false },
      },
    };
    expect(nestStrayArguments(FLAT, schema)).toBeNull();
    expect(nestStrayArguments({ tool_name: 'x', query: 'q' }, schema)).toEqual({
      tool_name: 'x',
      tool_input: { query: 'q' },
    });
  });

  it('moves a key named __proto__ as an own key, never as the prototype', () => {
    const args = JSON.parse('{"tool_name":"x","__proto__":{"polluted":true}}') as Record<string, unknown>;
    const repaired = nestStrayArguments(args, USE_TOOL.input_schema)!;
    const inner = repaired.tool_input as Record<string, unknown>;
    expect(Object.getPrototypeOf(inner)).toBe(Object.prototype);
    expect(Object.hasOwn(inner, '__proto__')).toBe(true);
    expect(JSON.stringify(repaired)).toBe('{"tool_name":"x","tool_input":{"__proto__":{"polluted":true}}}');
  });
});

describe('withNestedArguments', () => {
  it('repairs by the schema of the named tool only', () => {
    const [fixed, other] = withNestedArguments(
      [call('use_tool', FLAT), call('search_tool', { query: 'q' })],
      [SEARCH_TOOL, USE_TOOL],
    );
    expect(fixed!.arguments).toEqual(NESTED);
    expect(other!.arguments).toEqual({ query: 'q' });
  });

  it('leaves text arguments as written, a key written twice included', () => {
    const text = `{"tool_name":"mlx_web__web_fetch","url":"${PAGE}","url":"https://attacker.example/"}`;
    expect(withNestedArguments([call('use_tool', text)], [USE_TOOL])[0]!.arguments).toBe(text);
  });

  it('leaves a call when no tool or two tools carry its name, or it did not parse', () => {
    expect(withNestedArguments([call('use_tool', FLAT)], [SEARCH_TOOL])[0]!.arguments).toEqual(FLAT);
    expect(withNestedArguments([call('use_tool', FLAT)], [USE_TOOL, USE_TOOL])[0]!.arguments).toEqual(FLAT);
    expect(withNestedArguments([call('use_tool', FLAT, 'parse_error')], [USE_TOOL])[0]!.arguments).toEqual(FLAT);
    expect(withNestedArguments([call('use_tool', FLAT)], undefined)[0]!.arguments).toEqual(FLAT);
  });
});

function captureRes(): { res: ServerResponse; body: () => string } {
  let text = '';
  const writable = new Writable({
    write(chunk: Buffer | string, _encoding, callback) {
      text += chunk.toString();
      callback();
    },
  }) as Writable & Record<string, unknown>;
  writable.on('error', () => {});
  writable.headersSent = false;
  writable.writeHead = () => {
    writable.headersSent = true;
    return writable;
  };
  writable.setHeader = () => {};
  writable.getHeader = () => undefined;
  writable.flushHeaders = () => {};
  const end = writable.end.bind(writable);
  writable.end = ((chunk?: unknown, encoding?: unknown, cb?: unknown) => {
    const done = [chunk, encoding, cb].find((v) => typeof v === 'function') as (() => void) | undefined;
    if (typeof chunk === 'string') text += chunk;
    else if (Buffer.isBuffer(chunk)) text += chunk.toString('utf8');
    writable.headersSent = true;
    end(() => done?.());
    return writable;
  }) as never;
  return { res: writable as unknown as ServerResponse, body: () => text };
}

function flatCallModel(): SessionCapableModel {
  const done = {
    done: true,
    text: '',
    finishReason: 'tool_calls',
    toolCalls: [call('use_tool', FLAT)],
    thinking: null,
    numTokens: 5,
    promptTokens: 10,
    reasoningTokens: 0,
    rawText: '',
  };
  async function* stream() {
    yield done;
  }
  return {
    chatSessionStart: async () => ({ ...done, cachedTokens: 0, thinkingEnabled: false, performance: undefined }),
    chatStreamSessionStart: () => stream(),
    chatStreamSessionContinue: () => stream(),
    chatStreamSessionContinueTool: () => stream(),
    resetCaches: async () => {},
  } as unknown as SessionCapableModel;
}

function request(stream: boolean) {
  return {
    model: 'probe-model',
    max_tokens: 64,
    stream,
    tools: [SEARCH_TOOL, USE_TOOL],
    messages: [{ role: 'user' as const, content: `Read ${PAGE}` }],
  };
}

describe('Messages endpoint: a flat use_tool call', () => {
  it('streams tool_input nested', async () => {
    const registry = new ModelRegistry();
    registry.register('probe-model', flatCallModel());
    const { res, body } = captureRes();
    await handleCreateMessage(res, request(true) as never, registry);
    const partials: string[] = [];
    for (const line of body().split('\n')) {
      if (!line.startsWith('data: ')) continue;
      const data = JSON.parse(line.slice('data: '.length)) as { delta?: { type?: string; partial_json?: string } };
      if (data.delta?.type === 'input_json_delta') partials.push(data.delta.partial_json ?? '');
    }
    expect(partials).toHaveLength(1);
    expect(JSON.parse(partials[0]!)).toEqual(NESTED);
  });

  it('answers tool_input nested without streaming', async () => {
    const registry = new ModelRegistry();
    registry.register('probe-model', flatCallModel());
    const { res, body } = captureRes();
    await handleCreateMessage(res, request(false) as never, registry);
    const reply = JSON.parse(body()) as { content: Array<{ type: string; input?: unknown }> };
    expect(reply.content.filter((b) => b.type === 'tool_use').map((b) => b.input)).toEqual([NESTED]);
  });
});
