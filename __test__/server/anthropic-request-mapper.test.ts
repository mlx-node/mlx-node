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

  it('rejects text/image blocks that precede a tool_result in the same user turn', () => {
    // A tool_result must appear as a CONTIGUOUS PREFIX of the user
    // turn. Text/image blocks that appear BEFORE any tool_result
    // are still rejected — emitting them in-place would put a
    // user `content` message between the preceding assistant
    // fan-out and the tool_result it is answering, orphaning the
    // fan-out. The mapper also cannot reorder blocks to satisfy
    // the prefix invariant without silently changing authorship.
    expect(() =>
      mapAnthropicRequest({
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
      }),
    ).toThrow(/tool_result blocks must appear as a contiguous prefix/i);
  });

  it('rejects an image block that precedes a tool_result in the same user turn', () => {
    // Same prefix invariant as the text-before-tool_result case
    // above — an image block before a tool_result is still
    // rejected because it would orphan the fan-out.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
              { type: 'tool_result', tool_use_id: 'call_abc', content: 'ok' },
            ],
          },
        ],
      }),
    ).toThrow(/tool_result blocks must appear as a contiguous prefix/i);
  });

  it('accepts tool_result followed by trailing text and emits a tool block + user turn', () => {
    // Accept the common Anthropic shape `[tool_result, text("now do X")]`: emit the
    // tool message(s) first, then a trailing user message carrying text/image content.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_123', content: 'Sunny, 72F' },
            { type: 'text', text: 'Now tell me a joke about the weather.' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'tool', content: 'Sunny, 72F', toolCallId: 'call_123' },
      { role: 'user', content: 'Now tell me a joke about the weather.' },
    ]);
  });

  it('accepts multiple tool_result blocks followed by a trailing image', () => {
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'one' },
            { type: 'tool_result', tool_use_id: 'call_b', content: 'two' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(3);
    expect(messages[0]).toEqual({ role: 'tool', content: 'one', toolCallId: 'call_a' });
    expect(messages[1]).toEqual({ role: 'tool', content: 'two', toolCallId: 'call_b' });
    expect(messages[2].role).toBe('user');
    expect(messages[2].content).toBe('');
    expect(messages[2].images).toHaveLength(1);
  });

  it('rejects a tool_result whose content array carries a nested image block', () => {
    // Anthropic allows `tool_result.content` to mix text and image blocks, but the
    // internal `ChatMessage` shape is a NAPI-generated Rust struct with no `images`
    // field on `role: 'tool'`. Refuse this shape outright — hoisting the image onto
    // a trailing user turn would reorder images relative to the caller's declaration
    // and lose the per-tool association after canonicalization reorders tool rows.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'call_screenshot',
                content: [
                  { type: 'text', text: 'Here is the screenshot you asked for:' },
                  { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
                ],
              },
            ],
          },
        ],
      }),
    ).toThrow(/nested image content in tool_result blocks is not representable/i);
  });

  it('rejects a tool_result nested image even when another tool_result in the same turn is plain text', () => {
    // When the caller mixes a plain-text tool_result with one that carries a nested
    // image, reject the whole turn rather than silently hoisting the image onto a
    // trailing user message that loses its owner after canonicalization.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_ok', content: 'ok' },
              {
                type: 'tool_result',
                tool_use_id: 'call_shot',
                content: [
                  { type: 'text', text: 'screenshot:' },
                  { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
                ],
              },
            ],
          },
        ],
      }),
    ).toThrow(/nested image content in tool_result blocks is not representable/i);
  });

  // The flat NAPI `ChatMessage` shape cannot represent interleaved text/image blocks
  // after a tool_result prefix, so the mapper rejects ambiguous shapes. These tests
  // pin the narrow accepted set (pure trailing text, single trailing image) and pin
  // rejection for every interleaved / multi-image shape.
  const iter28Png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

  it('rejects trailing text followed by an image after a tool_result prefix', () => {
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
              { type: 'text', text: 'see the attached screenshot' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
            ],
          },
        ],
      }),
    ).toThrow(/mixing trailing text and image blocks after a tool_result prefix/i);
  });

  it('rejects trailing image followed by text after a tool_result prefix', () => {
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
              { type: 'text', text: 'that was the screenshot' },
            ],
          },
        ],
      }),
    ).toThrow(/mixing trailing text and image blocks after a tool_result prefix/i);
  });

  it('rejects multiple trailing image blocks after a tool_result prefix', () => {
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
            ],
          },
        ],
      }),
    ).toThrow(/multiple trailing image blocks after a tool_result prefix/i);
  });

  it('accepts a single trailing text block after a tool_result prefix', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
            { type: 'text', text: 'now summarise the above' },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({ role: 'tool', content: 'alpha', toolCallId: 'call_a' });
    expect(messages[1]).toEqual({ role: 'user', content: 'now summarise the above' });
  });

  it('accepts a single trailing image block after a tool_result prefix', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({ role: 'tool', content: 'alpha', toolCallId: 'call_a' });
    expect(messages[1].role).toBe('user');
    expect(messages[1].content).toBe('');
    expect(messages[1].images).toHaveLength(1);
    expect(messages[1].images![0]).toEqual(Buffer.from(iter28Png, 'base64'));
  });

  it('accepts multiple trailing text blocks after a tool_result prefix (concatenated)', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
            { type: 'text', text: 'part one ' },
            { type: 'text', text: 'part two' },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({ role: 'tool', content: 'alpha', toolCallId: 'call_a' });
    expect(messages[1]).toEqual({ role: 'user', content: 'part one part two' });
  });

  it('maps a pure-tool_result user turn to a contiguous tool block (counter-test)', () => {
    // A user turn whose blocks are ALL tool_result maps cleanly to contiguous `tool`
    // ChatMessage blocks in caller-supplied order (downstream canonicalization then
    // reorders against the assistant's declared sibling order).
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_123', content: 'Rainy' },
            { type: 'tool_result', tool_use_id: 'call_456', content: 'Sunny' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'tool', content: 'Rainy', toolCallId: 'call_123' },
      { role: 'tool', content: 'Sunny', toolCallId: 'call_456' },
    ]);
  });

  it('maps a pure text/image user turn to a single user message (counter-test)', () => {
    // A user turn containing only text and image blocks maps to a single `user`
    // ChatMessage with both fields populated.
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
    expect(messages[0].role).toBe('user');
    expect(messages[0].content).toBe('What is in this image?');
    expect(messages[0].images).toHaveLength(1);
  });

  it('wraps tool_result.is_error=true content in a JSON envelope', () => {
    // `ChatMessage.content` is a string with no `isError` field, so errored
    // tool_result content is wrapped as `{"is_error":true,"content":<original>}` —
    // an unambiguous, lossless, non-colliding encoding that preserves the raw
    // payload verbatim. Successful tool_result is passed through untouched.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_fail',
              content: 'boom: connection refused',
              is_error: true,
            },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'tool',
        content: JSON.stringify({ is_error: true, content: 'boom: connection refused' }),
        toolCallId: 'call_fail',
      },
    ]);
  });

  it('preserves a JSON tool_result payload losslessly inside the is_error envelope', () => {
    // The envelope preserves the raw payload verbatim as the `content` field, so a
    // downstream reader can either pass the whole envelope through or `JSON.parse`
    // it and recover both the flag and the original JSON text.
    const jsonPayload = '{"error_code":500,"message":"upstream unavailable"}';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_json_fail',
              content: jsonPayload,
              is_error: true,
            },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const toolMsg = messages[0]!;
    expect(toolMsg.role).toBe('tool');
    expect(toolMsg.toolCallId).toBe('call_json_fail');
    // Encoded content is itself valid JSON and `content` inside equals the original.
    const parsed = JSON.parse(toolMsg.content) as { is_error: boolean; content: string };
    expect(parsed).toEqual({ is_error: true, content: jsonPayload });
  });

  it('does not wrap successful tool_result content that looks like a JSON envelope', () => {
    // A successful tool_result whose content happens to be a JSON object — even one
    // structurally resembling `{"is_error":true,"content":...}` — MUST NOT be
    // rewrapped. The envelope shape is reserved exclusively for `is_error === true`.
    const suspicious = '{"is_error":true,"content":"this was supplied by a tool that returned success"}';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'call_ok', content: suspicious, is_error: false }],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'tool', content: suspicious, toolCallId: 'call_ok' }]);
  });

  it('leaves tool_result content untouched when is_error is absent or false', () => {
    // The envelope MUST NOT leak into successful tool_result content — including
    // content that literally starts with `[tool error] `.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_ok_1', content: '72F and sunny', is_error: false },
            { type: 'tool_result', tool_use_id: 'call_ok_2', content: '68F and cloudy' },
            { type: 'tool_result', tool_use_id: 'call_ok_3', content: '[tool error] this is a legitimate payload' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'tool', content: '72F and sunny', toolCallId: 'call_ok_1' },
      { role: 'tool', content: '68F and cloudy', toolCallId: 'call_ok_2' },
      { role: 'tool', content: '[tool error] this is a legitimate payload', toolCallId: 'call_ok_3' },
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
