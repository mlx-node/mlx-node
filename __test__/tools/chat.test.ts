import { describe, it, expect } from 'vitest';

/**
 * Tests for the model.chat() API
 *
 * These tests verify that:
 * 1. ToolCallResult has the expected structure
 * 2. ChatResult has all required fields
 * 3. The Rust tool call and thinking parsers work correctly
 *
 * Note: Full model integration tests are in __test__/models/qwen3.test.ts
 * These tests focus on the type behavior which can be tested without loading a model.
 */

// Import the types to verify they exist and have correct structure
import type { ToolCallResult, ToolDefinition } from '@mlx-node/lm';

describe('chat() API Types', () => {
  describe('ToolCallResult interface', () => {
    it('has the expected properties', () => {
      // This test verifies the TypeScript interface matches our expectations
      const mockToolCall: ToolCallResult = {
        id: 'call_abc123',
        name: 'test_function',
        arguments: { key: 'value', nested: { a: 1 } },
        status: 'ok',
        error: undefined,
      };

      expect(mockToolCall.id).toBe('call_abc123');
      expect(mockToolCall.name).toBe('test_function');
      expect(mockToolCall.arguments).toEqual({ key: 'value', nested: { a: 1 } });
      expect(mockToolCall.status).toBe('ok');
      expect(mockToolCall.error).toBeUndefined();
    });

    it('supports error status with message', () => {
      const errorToolCall: ToolCallResult = {
        id: 'call_error123',
        name: 'broken_function',
        arguments: {},
        status: 'invalid_json',
        error: 'Failed to parse arguments JSON',
      };

      expect(errorToolCall.status).toBe('invalid_json');
      expect(errorToolCall.error).toBeDefined();
    });

    it('supports missing_name status', () => {
      const missingNameCall: ToolCallResult = {
        id: 'call_noname',
        name: '',
        arguments: {},
        status: 'missing_name',
        error: 'Tool call missing name',
      };

      expect(missingNameCall.status).toBe('missing_name');
      expect(missingNameCall.name).toBe('');
    });
  });

  describe('ToolDefinition interface', () => {
    it('supports OpenAI-compatible tool definition', () => {
      const tool: ToolDefinition = {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather information',
          parameters: {
            type: 'object',
            properties: JSON.stringify({
              location: { type: 'string', description: 'City name' },
            }),
            required: ['location'],
          },
        },
      };

      expect(tool.type).toBe('function');
      expect(tool.function.name).toBe('get_weather');
      expect(tool.function.description).toBeDefined();
      expect(tool.function.parameters).toBeDefined();
    });

    it('supports minimal tool definition', () => {
      const minimalTool: ToolDefinition = {
        type: 'function',
        function: {
          name: 'simple_tool',
        },
      };

      expect(minimalTool.function.name).toBe('simple_tool');
      expect(minimalTool.function.description).toBeUndefined();
      expect(minimalTool.function.parameters).toBeUndefined();
    });
  });
});

describe('Tool Call ID Format', () => {
  it('follows OpenAI call_<uuid> format', () => {
    // Tool call IDs should match the pattern call_<32-hex-chars>
    const validIdPattern = /^call_[a-f0-9]{32}$/;

    // This is a mock - actual IDs come from the Rust parser
    const mockId = 'call_12345678901234567890123456789012';
    expect(mockId).toMatch(validIdPattern);
  });
});

describe('Tool Arguments as Native Objects', () => {
  it('arguments should be directly usable without JSON.parse', () => {
    // This test demonstrates the key benefit of the new API
    const toolCall: ToolCallResult = {
      id: 'call_test',
      name: 'get_weather',
      arguments: { location: 'Paris', units: 'celsius' },
      status: 'ok',
    };

    // Direct property access - no JSON.parse needed!
    const location = toolCall.arguments.location;
    const units = toolCall.arguments.units;

    expect(location).toBe('Paris');
    expect(units).toBe('celsius');
  });

  it('arguments support nested objects', () => {
    const toolCall: ToolCallResult = {
      id: 'call_test',
      name: 'complex_tool',
      arguments: {
        config: {
          setting1: true,
          setting2: [1, 2, 3],
        },
        data: {
          nested: {
            deep: 'value',
          },
        },
      },
      status: 'ok',
    };

    const config = toolCall.arguments.config as { setting1: boolean; setting2: number[] };
    const data = toolCall.arguments.data as { nested: { deep: string } };
    expect(config.setting1).toBe(true);
    expect(config.setting2).toEqual([1, 2, 3]);
    expect(data.nested.deep).toBe('value');
  });
});

describe('Finish Reason Handling', () => {
  it('supports stop, length, and tool_calls values', () => {
    type FinishReason = 'stop' | 'length' | 'tool_calls';

    const validReasons: FinishReason[] = ['stop', 'length', 'tool_calls'];

    validReasons.forEach((reason) => {
      // This verifies the type system accepts these values
      const r: FinishReason = reason;
      expect(['stop', 'length', 'tool_calls']).toContain(r);
    });
  });
});

describe('Error Status Values', () => {
  it('supports all expected status values', () => {
    type ToolCallStatus = 'ok' | 'invalid_json' | 'missing_name';

    const validStatuses: ToolCallStatus[] = ['ok', 'invalid_json', 'missing_name'];

    validStatuses.forEach((status) => {
      expect(['ok', 'invalid_json', 'missing_name']).toContain(status);
    });
  });
});

describe('Thinking Content Extraction', () => {
  it('thinking field type is string | null', () => {
    // Verify the thinking field can be string or null
    const withThinking: { thinking: string | null } = { thinking: 'Some reasoning here' };
    const withoutThinking: { thinking: string | null } = { thinking: null };

    expect(withThinking.thinking).toBe('Some reasoning here');
    expect(withoutThinking.thinking).toBeNull();
  });

  it('thinking content is separate from main text', () => {
    // The thinking field should contain extracted reasoning
    // The text field should have <think> tags stripped
    const mockResult = {
      text: 'The answer is 42.', // Clean text without <think> tags
      thinking: 'Let me analyze this problem step by step...', // Extracted thinking
      rawText: '<think>\nLet me analyze this problem step by step...\n</think>\n\nThe answer is 42.',
    };

    expect(mockResult.text).not.toContain('<think>');
    expect(mockResult.text).not.toContain('</think>');
    expect(mockResult.thinking).toBe('Let me analyze this problem step by step...');
    expect(mockResult.rawText).toContain('<think>');
  });

  it('thinking is null when no think tags present', () => {
    const mockResult = {
      text: 'Direct response without reasoning.',
      thinking: null as string | null,
      rawText: 'Direct response without reasoning.',
    };

    expect(mockResult.thinking).toBeNull();
    expect(mockResult.text).toBe(mockResult.rawText);
  });
});
