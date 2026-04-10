import { describe, expect, it } from 'vite-plus/test';

import { mapAnthropicRequest } from '../../packages/server/src/mappers/anthropic-request.js';

describe('mapAnthropicRequest', () => {
  it('maps a simple string user message to a single user ChatMessage', () => {
    const { messages, config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
    expect(config.reportPerformance).toBe(true);
  });

  it('prepends system prompt string as first message', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: 'You are a helpful assistant.',
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages).toEqual([
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hi' },
    ]);
  });

  it('prepends system prompt array of blocks as concatenated system message', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: [
        { type: 'text', text: 'You are helpful.' },
        { type: 'text', text: ' Be concise.' },
      ],
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages).toEqual([
      { role: 'system', content: 'You are helpful. Be concise.' },
      { role: 'user', content: 'Hi' },
    ]);
  });

  it('maps multi-turn conversation (user → assistant → user)', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: '4' },
        { role: 'user', content: 'Are you sure?' },
      ],
    });

    expect(messages).toEqual([
      { role: 'user', content: 'What is 2+2?' },
      { role: 'assistant', content: '4' },
      { role: 'user', content: 'Are you sure?' },
    ]);
  });

  it('maps user message with tool_result content blocks to tool role messages', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'call_abc', content: '72F and sunny' }],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'tool', content: '72F and sunny', toolCallId: 'call_abc' }]);
  });

  it('maps mixed user message (text + tool_result) → user message then tool messages', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Here is the weather result:' },
            { type: 'tool_result', tool_use_id: 'call_123', content: 'Rainy' },
            { type: 'tool_result', tool_use_id: 'call_456', content: 'Sunny' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'user', content: 'Here is the weather result:' },
      { role: 'tool', content: 'Rainy', toolCallId: 'call_123' },
      { role: 'tool', content: 'Sunny', toolCallId: 'call_456' },
    ]);
  });

  it('maps assistant message with tool_use to assistant with toolCalls', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'call_xyz',
              name: 'get_weather',
              input: { city: 'San Francisco' },
            },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'assistant',
        content: '',
        toolCalls: [{ id: 'call_xyz', name: 'get_weather', arguments: '{"city":"San Francisco"}' }],
      },
    ]);
  });

  it('maps assistant message with thinking to assistant with reasoningContent', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'Let me reason through this...' },
            { type: 'text', text: 'The answer is 42.' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'assistant',
        content: 'The answer is 42.',
        reasoningContent: 'Let me reason through this...',
      },
    ]);
  });

  it('maps mixed assistant message (thinking + text + tool_use) into a single message', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'I should call the weather tool.' },
            { type: 'text', text: 'Let me check the weather.' },
            {
              type: 'tool_use',
              id: 'call_abc',
              name: 'get_weather',
              input: { city: 'NYC' },
            },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('assistant');
    expect(msg.content).toBe('Let me check the weather.');
    expect(msg.reasoningContent).toBe('I should call the weather tool.');
    expect(msg.toolCalls).toEqual([{ id: 'call_abc', name: 'get_weather', arguments: '{"city":"NYC"}' }]);
  });

  it('maps tool definition from Anthropic input_schema to internal format with JSON.stringify', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tools: [
        {
          name: 'get_weather',
          description: 'Get the weather for a city',
          input_schema: {
            type: 'object',
            properties: { city: { type: 'string', description: 'City name' } },
            required: ['city'],
          },
        },
      ],
    });

    expect(config.tools).toHaveLength(1);
    const tool = config.tools![0];
    expect(tool.type).toBe('function');
    expect(tool.function.name).toBe('get_weather');
    expect(tool.function.description).toBe('Get the weather for a city');
    expect(tool.function.parameters).toEqual({
      type: 'object',
      properties: JSON.stringify({ city: { type: 'string', description: 'City name' } }),
      required: ['city'],
    });
  });

  it('maps tool choice auto → passes all tools', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'auto' },
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('maps tool choice any → passes all tools', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'any' },
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('maps tool choice tool with name → filters to only the named tool', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'tool', name: 'tool_b' },
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
        { name: 'tool_c', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(1);
    expect(config.tools![0].function.name).toBe('tool_b');
  });

  it('passes all tools when tool_choice is absent', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('maps max_tokens to maxNewTokens in config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 512,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.maxNewTokens).toBe(512);
  });

  it('maps temperature to config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      temperature: 0.7,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.temperature).toBe(0.7);
  });

  it('maps top_p to topP in config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      top_p: 0.9,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.topP).toBe(0.9);
  });

  it('maps top_k to topK in config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      top_k: 50,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.topK).toBe(50);
  });

  it('always sets reportPerformance to true', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.reportPerformance).toBe(true);
  });

  it('maps content array with only text blocks to concatenated text', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Part one. ' },
            { type: 'text', text: 'Part two.' },
          ],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'user', content: 'Part one. Part two.' }]);
  });

  it('maps user message with a single image block to images array with decoded Uint8Array', () => {
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } }],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('');
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toEqual(Buffer.from(imageData, 'base64'));
  });

  it('maps user message with text + image to content and images populated', () => {
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is in this image?' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('What is in this image?');
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toEqual(Buffer.from(imageData, 'base64'));
  });

  it('maps user message with only image (no text) to empty content with images', () => {
    const imageData = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageData } }],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('');
    expect(msg.images).toBeDefined();
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toBeInstanceOf(Uint8Array);
    expect(msg.images![0]).toEqual(Buffer.from(imageData, 'base64'));
  });

  it('maps tool_result with text block array content', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_789',
              content: [
                { type: 'text', text: 'Result: ' },
                { type: 'text', text: 'success' },
              ],
            },
          ],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'tool', content: 'Result: success', toolCallId: 'call_789' }]);
  });
});
