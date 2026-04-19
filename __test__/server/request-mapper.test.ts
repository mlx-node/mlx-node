import { describe, expect, it } from 'vite-plus/test';

import {
  mapRequest,
  reconstructMessagesFromChain,
  stringifyStoredInputMessages,
} from '../../packages/server/src/mappers/request.js';

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

  describe('assistant message + function_call coalescing (Finding 3)', () => {
    it('coalesces assistant message followed by function_call into a single assistant turn', () => {
      // A single assistant turn that produced both text and tool calls is serialised by
      // the OpenAI Responses API as `[message(assistant, text), function_call, ...]`.
      // The mapper must coalesce that run into ONE `assistant` ChatMessage carrying
      // both text and `toolCalls`, matching the hot-path `ChatSession` shape exactly.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'user', content: 'What is the weather?' },
          { type: 'message', role: 'assistant', content: 'Let me check.' },
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
        { role: 'user', content: 'What is the weather?' },
        {
          role: 'assistant',
          content: 'Let me check.',
          toolCalls: [{ name: 'get_weather', arguments: '{"city":"SF"}', id: 'call_abc' }],
        },
      ]);
    });

    it('coalesces assistant message followed by multiple function_calls into one turn', () => {
      // A fan-out shape: one assistant message then several
      // parallel function_calls. All siblings end up on the SAME
      // assistant turn.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'assistant', content: 'Fetching two cities.' },
          {
            type: 'function_call',
            id: 'fc-1',
            call_id: 'call_a',
            name: 'get_weather',
            arguments: '{"city":"SF"}',
          },
          {
            type: 'function_call',
            id: 'fc-2',
            call_id: 'call_b',
            name: 'get_weather',
            arguments: '{"city":"NYC"}',
          },
        ],
      });

      expect(messages).toEqual([
        {
          role: 'assistant',
          content: 'Fetching two cities.',
          toolCalls: [
            { name: 'get_weather', arguments: '{"city":"SF"}', id: 'call_a' },
            { name: 'get_weather', arguments: '{"city":"NYC"}', id: 'call_b' },
          ],
        },
      ]);
    });

    it('does not coalesce a function_call onto a preceding user message', () => {
      // A function_call after a user message is not a valid
      // fan-out head — the coalesce predicate is gated on the
      // previous message being an assistant. The mapper opens a
      // fresh assistant turn instead.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'user', content: 'Hello' },
          {
            type: 'function_call',
            id: 'fc-1',
            call_id: 'call_x',
            name: 'do_thing',
            arguments: '{}',
          },
        ],
      });

      expect(messages).toEqual([
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [{ name: 'do_thing', arguments: '{}', id: 'call_x' }],
        },
      ]);
    });

    it('does not absorb an assistant message that follows function_call into the same turn', () => {
      // Only a message BEFORE a function_call run is coalesced.
      // A message AFTER a function_call starts a fresh turn —
      // that shape is a subsequent user/system/assistant turn,
      // not a continuation of the fan-out.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          {
            type: 'function_call',
            id: 'fc-1',
            call_id: 'call_a',
            name: 'lookup',
            arguments: '{}',
          },
          {
            type: 'function_call_output',
            call_id: 'call_a',
            output: 'done',
          },
          { type: 'message', role: 'assistant', content: 'All done.' },
        ],
      });

      expect(messages).toEqual([
        {
          role: 'assistant',
          content: '',
          toolCalls: [{ name: 'lookup', arguments: '{}', id: 'call_a' }],
        },
        { role: 'tool', content: 'done', toolCallId: 'call_a' },
        { role: 'assistant', content: 'All done.' },
      ]);
    });
  });

  describe('assistant history replay (client-echoed input[])', () => {
    it('accepts output_text content parts when a client replays an assistant message', () => {
      // Clients that do not use `previous_response_id` (pi-ai, Codex) replay
      // the prior assistant turn as `{type:"message",role:"assistant",content:[{type:"output_text",...}]}`.
      // Rejecting output_text breaks cold-start multi-turn outright.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Remember 42.' }] },
          {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'output_text', text: "Got it. I'll remember 42.", annotations: [] }],
          } as any,
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'What number?' }] },
        ],
      });

      expect(messages).toEqual([
        { role: 'user', content: 'Remember 42.' },
        { role: 'assistant', content: "Got it. I'll remember 42." },
        { role: 'user', content: 'What number?' },
      ]);
    });

    it('coalesces a replayed reasoning item onto the trailing assistant message', () => {
      // Real pi-ai payload shape: user → reasoning → message(assistant) → user.
      // The reasoning summary must land as `reasoningContent` on the assistant turn.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Hi' }] },
          {
            type: 'reasoning',
            id: 'rs_1',
            summary: [{ type: 'summary_text', text: 'Let me think briefly.' }],
          } as any,
          {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'output_text', text: 'Hello!', annotations: [] }],
          } as any,
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Bye' }] },
        ],
      });

      expect(messages).toEqual([
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello!', reasoningContent: 'Let me think briefly.' },
        { role: 'user', content: 'Bye' },
      ]);
    });

    it('coalesces reasoning + function_call into one assistant turn', () => {
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Weather in SF?' }] },
          {
            type: 'reasoning',
            summary: [{ type: 'summary_text', text: 'Tool required.' }],
          } as any,
          {
            type: 'function_call',
            id: 'fc-1',
            call_id: 'call_a',
            name: 'get_weather',
            arguments: '{"city":"SF"}',
          },
        ],
      });

      expect(messages).toEqual([
        { role: 'user', content: 'Weather in SF?' },
        {
          role: 'assistant',
          content: '',
          reasoningContent: 'Tool required.',
          toolCalls: [{ name: 'get_weather', arguments: '{"city":"SF"}', id: 'call_a' }],
        },
      ]);
    });

    it('emits reasoning-only turn as a standalone assistant message when no trailing content', () => {
      // Turn 1 ran out of budget inside thinking → client replays a lone reasoning item.
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Think briefly.' }] },
          {
            type: 'reasoning',
            summary: [{ type: 'summary_text', text: 'thinking only.' }],
          } as any,
          { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Continue.' }] },
        ],
      });

      expect(messages).toEqual([
        { role: 'user', content: 'Think briefly.' },
        { role: 'assistant', content: '', reasoningContent: 'thinking only.' },
        { role: 'user', content: 'Continue.' },
      ]);
    });

    it('treats a replayed refusal part as text for model re-ingestion', () => {
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          {
            type: 'message',
            role: 'assistant',
            content: [{ type: 'refusal', refusal: 'I cannot help with that.' }],
          } as any,
        ],
      });

      expect(messages).toEqual([{ role: 'assistant', content: 'I cannot help with that.' }]);
    });
  });

  describe('input_image content parts', () => {
    it('decodes a base64 data URL into raw image bytes attached to the user turn', () => {
      // 2x2 red PNG.
      const b64 =
        'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8DwnwEDMGEKDQYxAEiRAP9t2B9IAAAAAElFTkSuQmCC';
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              { type: 'input_text', text: 'What is in this image?' },
              { type: 'input_image', image_url: `data:image/png;base64,${b64}` },
            ],
          },
        ],
      });

      expect(messages).toHaveLength(1);
      const u = messages[0];
      expect(u.role).toBe('user');
      expect(u.content).toBe('What is in this image?');
      expect(u.images).toBeDefined();
      expect(u.images).toHaveLength(1);
      expect(Buffer.from(u.images![0]).subarray(0, 4)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47]));
    });

    it('rejects input_image with a remote http(s) URL', () => {
      expect(() =>
        mapRequest({
          model: 'test-model',
          input: [
            {
              type: 'message',
              role: 'user',
              content: [{ type: 'input_image', image_url: 'https://example.com/cat.png' }],
            },
          ],
        }),
      ).toThrow(/base64 data URL/);
    });

    it('rejects input_image attached to a non-user message', () => {
      expect(() =>
        mapRequest({
          model: 'test-model',
          input: [
            {
              type: 'message',
              role: 'assistant',
              content: [{ type: 'input_image', image_url: 'data:image/png;base64,AA==' }],
            } as any,
          ],
        }),
      ).toThrow(/only allowed on user messages/);
    });
  });

  describe('text/image part ordering', () => {
    // The flat internal `ChatMessage` shape cannot express a text part
    // that appears AFTER an image part in the same message: the
    // downstream Jinja serializer always renders text first, then all
    // images. Reject ambiguous orderings rather than silently reorder
    // them and change the caller's intent. Mirrors the rejection in
    // `anthropic-request.ts` for its own user-turn shape.
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

    it('rejects [input_text, input_image, input_text] (text after image)', () => {
      expect(() =>
        mapRequest({
          model: 'test-model',
          input: [
            {
              type: 'message',
              role: 'user',
              content: [
                { type: 'input_text', text: 'before' },
                { type: 'input_image', image_url: `data:image/png;base64,${png}` },
                { type: 'input_text', text: 'after' },
              ],
            },
          ],
        }),
      ).toThrow(/text content part after an image part in the same message/i);
    });

    it('rejects [input_image, input_text] (image first, text after)', () => {
      // Silently reordering `[image, text]` to `[text, image]` would flip
      // caption-vs-question framing for VLM prompts.
      expect(() =>
        mapRequest({
          model: 'test-model',
          input: [
            {
              type: 'message',
              role: 'user',
              content: [
                { type: 'input_image', image_url: `data:image/png;base64,${png}` },
                { type: 'input_text', text: 'what is this?' },
              ],
            },
          ],
        }),
      ).toThrow(/text content part after an image part/i);
    });

    it('rejects [output_text, input_image, output_text] on replayed assistant messages', () => {
      // `output_text` is a text-like part too. Reject text-after-image
      // uniformly regardless of the text variant.
      expect(() =>
        mapRequest({
          model: 'test-model',
          input: [
            {
              type: 'message',
              role: 'user',
              content: [
                { type: 'output_text', text: 'before', annotations: [] },
                { type: 'input_image', image_url: `data:image/png;base64,${png}` },
                { type: 'output_text', text: 'after', annotations: [] },
              ],
            } as any,
          ],
        }),
      ).toThrow(/text content part after an image part/i);
    });

    it('accepts [input_text, input_image] (text before image — representable)', () => {
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              { type: 'input_text', text: 'what colour?' },
              { type: 'input_image', image_url: `data:image/png;base64,${png}` },
            ],
          },
        ],
      });
      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('what colour?');
      expect(messages[0].images).toHaveLength(1);
    });

    it('accepts [input_text, input_text, input_image, input_image] (all text before all images)', () => {
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              { type: 'input_text', text: 'compare ' },
              { type: 'input_text', text: 'these:' },
              { type: 'input_image', image_url: `data:image/png;base64,${png}` },
              { type: 'input_image', image_url: `data:image/png;base64,${png}` },
            ],
          },
        ],
      });
      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('compare these:');
      expect(messages[0].images).toHaveLength(2);
    });

    it('accepts images-only content (no ordering ambiguity)', () => {
      const { messages } = mapRequest({
        model: 'test-model',
        input: [
          {
            type: 'message',
            role: 'user',
            content: [
              { type: 'input_image', image_url: `data:image/png;base64,${png}` },
              { type: 'input_image', image_url: `data:image/png;base64,${png}` },
            ],
          },
        ],
      });
      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('');
      expect(messages[0].images).toHaveLength(2);
    });
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

  it('preserves assistant turn when empty text accompanies reasoning', () => {
    // An empty assistant `message` item alongside a non-empty `reasoning` item must
    // still reconstruct the assistant turn, otherwise cold replay after TTL expiry
    // silently rebuilds a different conversation than the live session saw.
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Hello' }]),
        outputJson: JSON.stringify([
          {
            type: 'reasoning',
            summary: [{ text: 'let me think about this...' }],
          },
          {
            type: 'message',
            content: [{ text: '' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: '', reasoningContent: 'let me think about this...' },
    ]);
  });

  it('preserves assistant turn with reasoning even when no message item is present', () => {
    // Some stored records carry ONLY a `reasoning` item (no `message`); reconstruction
    // must still re-emit the assistant turn, carrying the reasoning summary through.
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Tell me a secret' }]),
        outputJson: JSON.stringify([
          {
            type: 'reasoning',
            summary: [{ text: 'I reasoned silently and produced nothing.' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'Tell me a secret' },
      {
        role: 'assistant',
        content: '',
        reasoningContent: 'I reasoned silently and produced nothing.',
      },
    ]);
  });

  it('skips assistant message when output has no text, no reasoning, and no tool calls', () => {
    // A stored record with NO assistant-facing items at all (no message, no reasoning,
    // no function_call) must produce no assistant turn on reconstruction — otherwise
    // legitimate no-op turns clutter the replayed history with an empty assistant.
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Hello' }]),
        outputJson: JSON.stringify([]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    // Only user message, no assistant since ALL output items were absent.
    expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
  });

  it('preserves assistant turn driven purely by tool calls', () => {
    // Tool-call-only assistant turns must remain reconstructible after the predicate
    // was widened to also accept reasoning-only / empty-text turns.
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'Get weather' }]),
        outputJson: JSON.stringify([
          {
            type: 'function_call',
            name: 'get_weather',
            arguments: '{"city":"NYC"}',
            call_id: 'call_nyc',
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toHaveLength(2);
    expect(messages[1]).toEqual({
      role: 'assistant',
      content: '',
      toolCalls: [{ name: 'get_weather', arguments: '{"city":"NYC"}', id: 'call_nyc' }],
    });
  });

  it('preserves assistant turn when only an empty-text message item is present', () => {
    // The server deliberately emits a `message` item with empty text when a turn
    // completes with no tool calls and no output (e.g. a tool-result continuation
    // where the model acknowledged and produced nothing). `ChatSession` hot-path
    // history always appends an assistant message for every completed turn — the
    // reconstruction predicate keys on item PRESENCE, not accumulated content.
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'thanks' }]),
        outputJson: JSON.stringify([
          {
            type: 'message',
            content: [{ text: '' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'thanks' },
      { role: 'assistant', content: '' },
    ]);
  });

  it('preserves assistant turn when an empty message item accompanies an empty reasoning item', () => {
    // A stored record carrying BOTH a `message` item with empty text and a `reasoning`
    // item with empty summary must reconstruct the assistant turn with empty content
    // and NO reasoningContent field — we omit empty reasoning so the reconstructed
    // shape matches a plain blank successful turn byte-for-byte.
    const chain = [
      {
        inputJson: JSON.stringify([{ role: 'user', content: 'thanks' }]),
        outputJson: JSON.stringify([
          {
            type: 'reasoning',
            summary: [{ text: '' }],
          },
          {
            type: 'message',
            content: [{ text: '' }],
          },
        ]),
      },
    ];

    const messages = reconstructMessagesFromChain(chain);
    expect(messages).toEqual([
      { role: 'user', content: 'thanks' },
      { role: 'assistant', content: '' },
    ]);
    // Pin that `reasoningContent` is absent, not present-but-empty — some downstream
    // paths distinguish `undefined` from `''` when deciding whether to emit <think>.
    expect(messages[1]).not.toHaveProperty('reasoningContent');
  });
});

describe('stored-input codec (images round-trip)', () => {
  // `StoredResponseRecord.inputJson` is serialised with
  // `stringifyStoredInputMessages` and re-parsed by
  // `reconstructMessagesFromChain`. Plain `JSON.stringify` would turn
  // `Uint8Array([1,2,3])` into `{"0":1,"1":2,"2":3}`, which fails the
  // NAPI `Uint8Array` coercion on cold replay through
  // `previous_response_id` chains that carry images. Guard that
  // round-trip here.
  it('encodes Uint8Array images as base64 in the stored JSON', () => {
    const msgs = [
      {
        role: 'user' as const,
        content: 'look',
        images: [new Uint8Array([0xff, 0x00, 0x01, 0x02])],
      },
    ];
    const json = stringifyStoredInputMessages(msgs);
    const parsedRaw = JSON.parse(json);
    expect(parsedRaw[0].images).toHaveLength(1);
    expect(parsedRaw[0].images[0]).toEqual({ __u8__: Buffer.from([0xff, 0x00, 0x01, 0x02]).toString('base64') });
    // Regression guard: the legacy numeric-keyed shape MUST NOT appear.
    expect(parsedRaw[0].images[0]).not.toHaveProperty('0');
  });

  it('reconstructs images back to Uint8Array instances with identical bytes', () => {
    const bytes = new Uint8Array([10, 20, 30, 40, 50]);
    const inputJson = stringifyStoredInputMessages([
      {
        role: 'user',
        content: 'round-trip me',
        images: [bytes],
      },
    ]);
    const [msg] = reconstructMessagesFromChain([{ inputJson, outputJson: '[]' }]);
    expect(msg.role).toBe('user');
    expect(msg.images).toBeDefined();
    expect(msg.images).toHaveLength(1);
    // `Buffer.from(_, 'base64')` returns a Uint8Array subclass — satisfies
    // the NAPI `Array<Uint8Array>` type check downstream.
    expect(msg.images![0]).toBeInstanceOf(Uint8Array);
    expect(Array.from(msg.images![0])).toEqual(Array.from(bytes));
  });

  it('preserves multiple images across round-trip in order', () => {
    const inputJson = stringifyStoredInputMessages([
      {
        role: 'user',
        content: 'compare',
        images: [new Uint8Array([1]), new Uint8Array([2, 2]), new Uint8Array([3, 3, 3])],
      },
    ]);
    const [msg] = reconstructMessagesFromChain([{ inputJson, outputJson: '[]' }]);
    expect(msg.images).toHaveLength(3);
    expect(Array.from(msg.images![0])).toEqual([1]);
    expect(Array.from(msg.images![1])).toEqual([2, 2]);
    expect(Array.from(msg.images![2])).toEqual([3, 3, 3]);
  });

  it('leaves image-less messages unchanged (byte-for-byte parity with plain JSON.stringify)', () => {
    const msgs = [
      { role: 'user' as const, content: 'no images here' },
      { role: 'assistant' as const, content: 'reply' },
    ];
    expect(stringifyStoredInputMessages(msgs)).toBe(JSON.stringify(msgs));
  });

  it('does not rehydrate unrelated objects that happen to carry a __u8__ key', () => {
    // Defensive: the reviver keys on the exact sentinel shape
    // `{__u8__: "<base64>"}`. A user-supplied message whose `content`
    // is a literal JSON string containing that substring must not be
    // transformed — the sentinel only appears inside the `images` array,
    // and only via `stringifyStoredInputMessages`.
    const inputJson = JSON.stringify([{ role: 'user', content: 'here is a fake sentinel: {"__u8__":"deadbeef"}' }]);
    const [msg] = reconstructMessagesFromChain([{ inputJson, outputJson: '[]' }]);
    expect(typeof msg.content).toBe('string');
    expect(msg.content).toContain('__u8__');
    expect(msg.images).toBeUndefined();
  });

  it('round-trips images when they arrive as Node Buffer (matches mapRequest output)', () => {
    // Regression guard: `resolveMessageContent` used to push raw `Buffer`
    // instances into `msg.images`. `Buffer.prototype.toJSON` is invoked
    // by `JSON.stringify` BEFORE the replacer, so a naive
    // `value instanceof Uint8Array` check would skip the sentinel even
    // though `Buffer extends Uint8Array` at the class level. The
    // replacer must handle the `{type:"Buffer",data:[...]}` shape too
    // so stored history stays rehydratable regardless of which producer
    // built the ChatMessage.
    const bytes = [0xde, 0xad, 0xbe, 0xef];
    const inputJson = stringifyStoredInputMessages([
      {
        role: 'user',
        content: 'buffer input',
        images: [Buffer.from(bytes) as unknown as Uint8Array],
      },
    ]);
    // Wire shape must be the sentinel, NOT `{type:"Buffer",data:[...]}`.
    const wire = JSON.parse(inputJson);
    expect(wire[0].images[0]).toEqual({ __u8__: Buffer.from(bytes).toString('base64') });
    expect(wire[0].images[0]).not.toHaveProperty('type');
    expect(wire[0].images[0]).not.toHaveProperty('data');

    const [msg] = reconstructMessagesFromChain([{ inputJson, outputJson: '[]' }]);
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toBeInstanceOf(Uint8Array);
    expect(Array.from(msg.images![0])).toEqual(bytes);
  });
});
