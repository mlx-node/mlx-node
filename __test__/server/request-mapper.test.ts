import { describe, expect, it } from 'vite-plus/test';

import { mapRequest, reconstructMessagesFromChain } from '../../packages/server/src/mappers/request.js';

describe('mapRequest', () => {
  it('maps a string input to a single user message', () => {
    const { messages, config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
    });

    expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
    expect(config.reportPerformance).toBe(true);
  });

  it('maps message array with developer role to system', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: [{ type: 'message', role: 'developer', content: 'You are helpful.' }],
    });

    expect(messages).toEqual([{ role: 'system', content: 'You are helpful.' }]);
  });

  it('maps message array with user role unchanged', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: [{ type: 'message', role: 'user', content: 'Hi there' }],
    });

    expect(messages).toEqual([{ role: 'user', content: 'Hi there' }]);
  });

  it('maps typed content parts (input_text) to plain string', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: [
        {
          type: 'message',
          role: 'user',
          content: [
            { type: 'input_text', text: 'Part 1' },
            { type: 'input_text', text: ' Part 2' },
          ],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'user', content: 'Part 1 Part 2' }]);
  });

  it('maps function_call input to assistant message with toolCalls', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: [
        {
          type: 'function_call',
          id: 'fc-1',
          call_id: 'call_abc',
          name: 'get_weather',
          arguments: '{"city":"SF"}',
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'assistant',
        content: '',
        toolCalls: [{ name: 'get_weather', arguments: '{"city":"SF"}', id: 'call_abc' }],
      },
    ]);
  });

  it('maps function_call_output input to tool message with toolCallId', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: [
        {
          type: 'function_call_output',
          call_id: 'call_abc',
          output: '72F and sunny',
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'tool',
        content: '72F and sunny',
        toolCallId: 'call_abc',
      },
    ]);
  });

  it('prepends instructions as system message', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      instructions: 'Be concise.',
    });

    expect(messages).toEqual([
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'Hello' },
    ]);
  });

  it('maps max_output_tokens to maxNewTokens in config', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      max_output_tokens: 256,
    });

    expect(config.maxNewTokens).toBe(256);
  });

  it('maps temperature to config', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      temperature: 0.7,
    });

    expect(config.temperature).toBe(0.7);
  });

  it('maps top_p to topP in config', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      top_p: 0.9,
    });

    expect(config.topP).toBe(0.9);
  });

  it('maps reasoning.effort to reasoningEffort in config', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      reasoning: { effort: 'high' },
    });

    expect(config.reasoningEffort).toBe('high');
  });

  it('maps tools from flat format to nested format with stringified properties', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tools: [
        {
          type: 'function',
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city'],
          },
        },
      ],
    });

    expect(config.tools).toHaveLength(1);
    const tool = config.tools![0];
    expect(tool.type).toBe('function');
    expect(tool.function.name).toBe('get_weather');
    expect(tool.function.description).toBe('Get weather for a city');
    expect(tool.function.parameters).toEqual({
      type: 'object',
      properties: JSON.stringify({ city: { type: 'string' } }),
      required: ['city'],
    });
  });

  it('maps tool without parameters', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tools: [
        {
          type: 'function',
          name: 'no_params',
        },
      ],
    });

    expect(config.tools).toHaveLength(1);
    expect(config.tools![0].function.parameters).toBeUndefined();
  });

  it('does not pass tools when tool_choice is none', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tool_choice: 'none',
      tools: [
        {
          type: 'function',
          name: 'get_weather',
          description: 'Get weather for a city',
          parameters: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city'],
          },
        },
      ],
    });

    expect(config.tools).toBeUndefined();
  });

  it('passes only the named tool when tool_choice specifies a function', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tool_choice: { type: 'function', name: 'search' },
      tools: [
        {
          type: 'function',
          name: 'get_weather',
          description: 'Get weather',
        },
        {
          type: 'function',
          name: 'search',
          description: 'Search the web',
        },
        {
          type: 'function',
          name: 'calculator',
          description: 'Do math',
        },
      ],
    });

    expect(config.tools).toHaveLength(1);
    expect(config.tools![0].function.name).toBe('search');
  });

  it('passes all tools when tool_choice is auto', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tool_choice: 'auto',
      tools: [
        {
          type: 'function',
          name: 'get_weather',
        },
        {
          type: 'function',
          name: 'search',
        },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('passes all tools when tool_choice is required', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tool_choice: 'required',
      tools: [
        {
          type: 'function',
          name: 'get_weather',
        },
        {
          type: 'function',
          name: 'search',
        },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('passes all tools when tool_choice is unspecified', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tools: [
        {
          type: 'function',
          name: 'get_weather',
        },
        {
          type: 'function',
          name: 'search',
        },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('does not set tools when tool_choice function name does not match any tool', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
      tool_choice: { type: 'function', name: 'nonexistent' },
      tools: [
        {
          type: 'function',
          name: 'get_weather',
        },
      ],
    });

    expect(config.tools).toBeUndefined();
  });

  it('prepends prior messages and sets reuseCache', () => {
    const priorMessages = [
      { role: 'user' as const, content: 'First message' },
      { role: 'assistant' as const, content: 'First response' },
    ];

    const { messages, config } = mapRequest(
      {
        model: 'test-model',
        input: 'Second message',
      },
      priorMessages,
    );

    expect(messages).toEqual([
      { role: 'user', content: 'First message' },
      { role: 'assistant', content: 'First response' },
      { role: 'user', content: 'Second message' },
    ]);
    expect(config.reuseCache).toBe(true);
  });

  it('places instructions before prior messages when both are provided', () => {
    const priorMessages = [
      { role: 'user' as const, content: 'First message' },
      { role: 'assistant' as const, content: 'First response' },
    ];

    const { messages } = mapRequest(
      {
        model: 'test-model',
        input: 'Second message',
        instructions: 'Be concise.',
      },
      priorMessages,
    );

    expect(messages).toEqual([
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: 'First message' },
      { role: 'assistant', content: 'First response' },
      { role: 'user', content: 'Second message' },
    ]);
  });

  it('does not set reuseCache when no prior messages', () => {
    const { config } = mapRequest({
      model: 'test-model',
      input: 'Hello',
    });

    expect(config.reuseCache).toBeUndefined();
  });

  it('defaults item type to message when type is omitted', () => {
    const { messages } = mapRequest({
      model: 'test-model',
      input: [{ role: 'user', content: 'No explicit type' } as any],
    });

    expect(messages).toEqual([{ role: 'user', content: 'No explicit type' }]);
  });
});

describe('reconstructMessagesFromChain', () => {
  it('reconstructs messages from a single response record', () => {
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Hello' }]),
        outputJson: JSON.stringify([
          {
            type: 'message',
            content: [{ text: 'Hi there!' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ]);
  });

  it('reconstructs messages from a multi-response chain', () => {
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'First question' }]),
        outputJson: JSON.stringify([
          {
            type: 'message',
            content: [{ text: 'First answer' }],
          },
        ]),
      },
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Follow-up' }]),
        outputJson: JSON.stringify([
          {
            type: 'message',
            content: [{ text: 'Follow-up answer' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'First question' },
      { role: 'assistant', content: 'First answer' },
      { role: 'user', content: 'Follow-up' },
      { role: 'assistant', content: 'Follow-up answer' },
    ]);
  });

  it('reconstructs reasoning content from chain', () => {
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Think hard' }]),
        outputJson: JSON.stringify([
          {
            type: 'reasoning',
            summary: [{ text: 'Let me think...' }],
          },
          {
            type: 'message',
            content: [{ text: 'The answer is 42' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'Think hard' },
      { role: 'assistant', content: 'The answer is 42', reasoningContent: 'Let me think...' },
    ]);
  });

  it('reconstructs tool calls from chain', () => {
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Get weather' }]),
        outputJson: JSON.stringify([
          {
            type: 'message',
            content: [{ text: '' }],
          },
          {
            type: 'function_call',
            name: 'get_weather',
            arguments: '{"city":"SF"}',
            call_id: 'call_123',
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toHaveLength(2);
    expect(messages[1]).toEqual({
      role: 'assistant',
      content: '',
      toolCalls: [{ name: 'get_weather', arguments: '{"city":"SF"}', id: 'call_123' }],
    });
  });

  it('returns empty array for empty chain', () => {
    const messages = reconstructMessagesFromChain([]);
    expect(messages).toEqual([]);
  });

  it('skips assistant message when output has no text and no tool calls', () => {
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Hello' }]),
        outputJson: JSON.stringify([
          {
            type: 'reasoning',
            summary: [{ text: 'thinking...' }],
          },
          // No message item, no function_call items
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    // Only user message, no assistant since content is empty and no tool calls
    expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
  });
});
