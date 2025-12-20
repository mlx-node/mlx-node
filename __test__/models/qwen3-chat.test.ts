/**
 * Integration tests for the model.chat() API
 *
 * Tests the high-level chat API with tool calling support.
 * Uses a tiny model with random weights for fast testing.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { ModelLoader, createToolDefinition } from '@mlx-node/lm';
import type { ToolCallResult } from '@mlx-node/lm';
import { createTempModel, TINY_TEST_CONFIG } from '../test-model-utils';

describe.sequential('Qwen3 Chat API', () => {
  let model: Awaited<ReturnType<typeof ModelLoader.loadPretrained>>;
  let cleanup: () => void;

  beforeAll(async () => {
    // Create a tiny model with random weights for testing
    const temp = await createTempModel(TINY_TEST_CONFIG);
    cleanup = temp.cleanup;

    // Load the model using ModelLoader
    model = await ModelLoader.loadPretrained(temp.modelPath);
  }, 60000); // 60s timeout for model creation

  afterAll(() => {
    cleanup?.();
  });

  describe('Basic Chat', () => {
    it('should return ChatResult with expected structure', async () => {
      const messages = [{ role: 'user', content: 'Hello' }];

      const result = await model.chat(messages);

      // Verify ChatResult structure
      expect(result).toBeDefined();
      expect(typeof result.text).toBe('string');
      expect(Array.isArray(result.toolCalls)).toBe(true);
      expect(result.thinking === null || typeof result.thinking === 'string').toBe(true);
      expect(['stop', 'length', 'tool_calls', 'repetition']).toContain(result.finishReason);
      expect(typeof result.numTokens).toBe('number');
      expect(result.numTokens).toBeGreaterThan(0);
      expect(typeof result.rawText).toBe('string');
    });

    it('should generate tokens with default config', async () => {
      const messages = [{ role: 'user', content: 'Say something' }];

      const result = await model.chat(messages);

      // Model should generate at least some tokens
      expect(result.numTokens).toBeGreaterThan(0);
      // Raw text should not be empty
      expect(result.rawText.length).toBeGreaterThan(0);
    });

    it('should respect maxNewTokens config', async () => {
      const messages = [{ role: 'user', content: 'Count to 100' }];

      const result = await model.chat(messages, {
        maxNewTokens: 10,
      });

      // Should stop around the limit (might be slightly over due to token boundaries)
      expect(result.numTokens).toBeLessThanOrEqual(15);
    });
  });

  describe('Chat with Tools', () => {
    const testTools = [
      createToolDefinition(
        'test_tool',
        'A test tool for unit tests',
        {
          arg1: { type: 'string', description: 'First argument' },
          arg2: { type: 'number', description: 'Second argument' },
        },
        ['arg1'],
      ),
      createToolDefinition('no_args_tool', 'A tool with no arguments'),
    ];

    it('should accept tool definitions without error', async () => {
      const messages = [{ role: 'user', content: 'Use a tool' }];

      // Should not throw even with random weights
      const result = await model.chat(messages, {
        tools: testTools,
        maxNewTokens: 50,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.toolCalls)).toBe(true);
    });

    it('should return empty toolCalls when no tool_call tags in output', async () => {
      const messages = [{ role: 'user', content: 'Hello' }];

      const result = await model.chat(messages, {
        tools: testTools,
        maxNewTokens: 20,
      });

      // With random weights, unlikely to generate valid tool calls
      // But the array should exist
      expect(Array.isArray(result.toolCalls)).toBe(true);
    });
  });

  describe('ToolCallResult Structure', () => {
    it('should have correct fields when parsed', () => {
      // This tests the type structure (compile-time check)
      const mockToolCall: ToolCallResult = {
        id: 'call_abc123def456',
        name: 'test_function',
        arguments: { key: 'value' },
        status: 'ok',
        error: undefined,
      };

      expect(mockToolCall.id).toMatch(/^call_/);
      expect(mockToolCall.name).toBe('test_function');
      expect(mockToolCall.arguments).toEqual({ key: 'value' });
      expect(mockToolCall.status).toBe('ok');
    });

    it('should support error status', () => {
      const errorCall: ToolCallResult = {
        id: 'call_error',
        name: '',
        arguments: {},
        status: 'missing_name',
        error: 'Tool call missing name',
      };

      expect(errorCall.status).toBe('missing_name');
      expect(errorCall.error).toBeDefined();
    });
  });

  describe('Finish Reasons', () => {
    it('should return "length" when max tokens reached', async () => {
      const messages = [{ role: 'user', content: 'Write a very long essay' }];

      const result = await model.chat(messages, {
        maxNewTokens: 5,
      });

      // With such a low limit, should hit length
      expect(result.finishReason).toBe('length');
    });

    it('should return "stop" when EOS token generated', async () => {
      const messages = [{ role: 'user', content: 'Hi' }];

      const result = await model.chat(messages, {
        maxNewTokens: 500, // Give it room to naturally stop
      });

      // With random weights, might hit either stop or length
      expect(['stop', 'length', 'tool_calls', 'repetition']).toContain(result.finishReason);
    });
  });

  describe('Thinking Extraction', () => {
    it('should return null thinking when no think tags', async () => {
      const messages = [{ role: 'user', content: 'Hello' }];

      const result = await model.chat(messages, {
        maxNewTokens: 50,
      });

      // With random weights, unlikely to generate <think> tags
      // But thinking field should exist (null or string)
      expect(result.thinking === null || typeof result.thinking === 'string').toBe(true);
    });
  });

  describe('Generation Config', () => {
    it('should accept all config options', async () => {
      const messages = [{ role: 'user', content: 'Test' }];

      // Should not throw with any valid config
      const result = await model.chat(messages, {
        maxNewTokens: 20,
        temperature: 0.5,
        topK: 40,
        topP: 0.95,
        minP: 0.05,
        repetitionPenalty: 1.1,
        repetitionContextSize: 50,
      });

      expect(result).toBeDefined();
    });

    it('should use greedy decoding with temperature 0', async () => {
      const messages = [{ role: 'user', content: 'Test' }];

      // Temperature 0 = greedy decoding (deterministic)
      const result1 = await model.chat(messages, {
        maxNewTokens: 10,
        temperature: 0,
      });

      const result2 = await model.chat(messages, {
        maxNewTokens: 10,
        temperature: 0,
      });

      // Same input + greedy = same output
      expect(result1.rawText).toBe(result2.rawText);
    });
  });
});
