/**
 * Integration tests for the Qwen3 chat session API.
 *
 * Tests the session-based high-level chat API with tool calling support.
 * Uses a tiny model with random weights for fast testing.
 *
 * The legacy `model.chat()` NAPI entry point has been replaced with
 * session methods: `chatSessionStart`, `chatSessionContinue`, and
 * `chatSessionContinueTool`. These tests cover the structural contract
 * rather than correctness (since the weights are random).
 */

import { loadModel, Qwen3Model, createToolDefinition } from '@mlx-node/lm';
import type { ToolCallResult } from '@mlx-node/lm';
import { describe, it, expect, beforeAll, afterAll } from 'vite-plus/test';

import { createTempModel, TINY_TEST_CONFIG } from '../test-model-utils';

describe.sequential('Qwen3 Chat Session API', () => {
  let model: Qwen3Model;
  let cleanup: () => void;

  beforeAll(async () => {
    // Create a tiny model with random weights for testing
    const temp = await createTempModel(TINY_TEST_CONFIG);
    cleanup = temp.cleanup;

    // Load the model using loadModel
    model = (await loadModel(temp.modelPath)) as unknown as Qwen3Model;
  }, 60000); // 60s timeout for model creation

  afterAll(() => {
    cleanup?.();
  });

  describe('Basic chatSessionStart', () => {
    it('should return ChatResult with expected structure', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Hello' }];

      const result = await model.chatSessionStart(messages);

      // Verify ChatResult structure
      expect(result).toBeDefined();
      expect(typeof result.text).toBe('string');
      expect(Array.isArray(result.toolCalls)).toBe(true);
      expect(result.thinking == null || typeof result.thinking === 'string').toBe(true);
      expect(['stop', 'length', 'tool_calls', 'repetition']).toContain(result.finishReason);
      expect(typeof result.numTokens).toBe('number');
      expect(result.numTokens).toBeGreaterThan(0);
      expect(typeof result.rawText).toBe('string');
    });

    it('should generate tokens with default config', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Say something' }];

      const result = await model.chatSessionStart(messages);

      // Model should generate at least some tokens
      expect(result.numTokens).toBeGreaterThan(0);
      // Raw text should not be empty
      expect(result.rawText.length).toBeGreaterThan(0);
    });

    it('should respect maxNewTokens config', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Count to 100' }];

      const result = await model.chatSessionStart(messages, {
        maxNewTokens: 10,
      });

      // Should stop around the limit (might be slightly over due to token boundaries)
      expect(result.numTokens).toBeLessThanOrEqual(15);
    });
  });

  describe('Multi-turn session', () => {
    it('should allow continuing an existing session via chatSessionContinue', async () => {
      model.resetCaches();

      // Turn 1: start
      const first = await model.chatSessionStart([{ role: 'user', content: 'Hi' }], { maxNewTokens: 10 });
      expect(first).toBeDefined();
      expect(first.numTokens).toBeGreaterThan(0);

      // Turn 2: continue with the same session
      const second = await model.chatSessionContinue('Follow-up question', null, {
        maxNewTokens: 10,
      });
      expect(second).toBeDefined();
      expect(second.numTokens).toBeGreaterThan(0);
      // Prompt tokens for the continue must be at least as large as the
      // initial prompt tokens — the cached history carries forward.
      expect(second.promptTokens).toBeGreaterThanOrEqual(first.promptTokens);
    });

    it('should reject image input via chatSessionContinue (text-only backend)', async () => {
      model.resetCaches();
      await model.chatSessionStart([{ role: 'user', content: 'Hi' }], { maxNewTokens: 5 });

      const fakeImage = new Uint8Array([0, 1, 2, 3]);
      await expect(model.chatSessionContinue('Caption this', [fakeImage], null)).rejects.toThrow(
        /IMAGE_CHANGE_REQUIRES_SESSION_RESTART/,
      );
    });

    it('should error when chatSessionContinue is called without an active session', async () => {
      model.resetCaches();
      await expect(model.chatSessionContinue('No session first', null, null)).rejects.toThrow();
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
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Use a tool' }];

      // Should not throw even with random weights
      const result = await model.chatSessionStart(messages, {
        tools: testTools,
        maxNewTokens: 50,
      });

      expect(result).toBeDefined();
      expect(Array.isArray(result.toolCalls)).toBe(true);
    });

    it('should return empty toolCalls when no tool_call tags in output', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Hello' }];

      const result = await model.chatSessionStart(messages, {
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
        rawContent: '<tool_call>{"name": "test_function", "arguments": {"key": "value"}}</tool_call>',
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
        rawContent: '<tool_call>{"arguments": {}}</tool_call>',
      };

      expect(errorCall.status).toBe('missing_name');
      expect(errorCall.error).toBeDefined();
    });
  });

  describe('Finish Reasons', () => {
    it('should return "length" when max tokens reached', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Write a very long essay' }];

      const result = await model.chatSessionStart(messages, {
        maxNewTokens: 5,
        ngramSize: 0, // Disable repetition detection so length limit fires first
      });

      // With such a low limit, should hit length
      expect(result.finishReason).toBe('length');
    });

    it('should accept a generous token budget', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Hi' }];

      const result = await model.chatSessionStart(messages, {
        maxNewTokens: 500, // Give it room to naturally stop
      });

      // With random weights, might hit either stop or length
      expect(['stop', 'length', 'tool_calls', 'repetition']).toContain(result.finishReason);
    });
  });

  describe('Thinking Extraction', () => {
    it('should return null thinking when no think tags', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Hello' }];

      const result = await model.chatSessionStart(messages, {
        maxNewTokens: 50,
      });

      // With random weights, unlikely to generate <think> tags
      // But thinking field should exist (null or string)
      expect(result.thinking == null || typeof result.thinking === 'string').toBe(true);
    });
  });

  describe('Generation Config', () => {
    it('should accept all config options', async () => {
      model.resetCaches();
      const messages = [{ role: 'user', content: 'Test' }];

      // Should not throw with any valid config
      const result = await model.chatSessionStart(messages, {
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
      model.resetCaches();
      const result1 = await model.chatSessionStart(messages, {
        maxNewTokens: 10,
        temperature: 0,
      });

      model.resetCaches();
      const result2 = await model.chatSessionStart(messages, {
        maxNewTokens: 10,
        temperature: 0,
      });

      // Same input + greedy = same output
      expect(result1.rawText).toBe(result2.rawText);
    });
  });

  describe('Session delta paths', () => {
    it('chatSessionContinueTool extends the session with a tool-result message', async () => {
      model.resetCaches();

      // Prime the session with an initial turn.
      const r1 = await model.chatSessionStart([{ role: 'user', content: 'call a tool' }], {
        maxNewTokens: 16,
        temperature: 0,
      });
      expect(r1).toBeDefined();
      expect(r1.numTokens).toBeGreaterThan(0);

      // Round-trip a tool-result delta. The content does not need to match
      // a real tool schema — we only care that the cache hoist-and-save-back
      // path completes cleanly.
      const r2 = await model.chatSessionContinueTool('call_test_123', '{"result": 42}', {
        maxNewTokens: 16,
        temperature: 0,
      });
      expect(r2).toBeDefined();
      expect(r2.numTokens).toBeGreaterThan(0);
      expect(typeof r2.rawText).toBe('string');
      expect(typeof r2.finishReason).toBe('string');

      // After the tool result, the next user continue should still work —
      // proves the cache is consistent across delta turns.
      const r3 = await model.chatSessionContinue('ok, thanks', null, {
        maxNewTokens: 16,
        temperature: 0,
      });
      expect(r3).toBeDefined();
      expect(r3.numTokens).toBeGreaterThan(0);
    });

    it('resetCaches restores a clean slate (determinism canary)', async () => {
      const msgs = [{ role: 'user', content: 'say hi in one word' }];

      model.resetCaches();
      const r1 = await model.chatSessionStart(msgs, { maxNewTokens: 8, temperature: 0 });

      model.resetCaches();
      const r2 = await model.chatSessionStart(msgs, { maxNewTokens: 8, temperature: 0 });

      expect(r2.rawText).toBe(r1.rawText);
      expect(r2.numTokens).toBe(r1.numTokens);
    });
  });
});
