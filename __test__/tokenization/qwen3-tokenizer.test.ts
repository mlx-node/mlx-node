/**
 * Qwen3 Tokenizer Tests
 *
 * Comprehensive tests for the Rust-based Qwen3 tokenizer implementation
 */

import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, it, expect, beforeAll } from 'vitest';
import { Qwen3Tokenizer, type ChatMessage } from '@mlx-node/core';

describe('Qwen3Tokenizer', () => {
  let tokenizer: Qwen3Tokenizer;

  const TOKENIZER_PATH = join(
    fileURLToPath(import.meta.url),
    '..',
    '..',
    '..',
    '.cache/models/qwen3-0.6b-mlx-bf16/tokenizer.json',
  );

  beforeAll(async () => {
    tokenizer = await Qwen3Tokenizer.fromPretrained(TOKENIZER_PATH);
  });

  describe('Initialization', () => {
    it('should load tokenizer from default path', () => {
      expect(tokenizer).toBeDefined();
      expect(tokenizer.vocabSize()).toBeGreaterThan(0);
    });

    it('should have correct vocabulary size', () => {
      const vocabSize = tokenizer.vocabSize();
      // Actual vocab size from tokenizer (base vocab + special tokens)
      expect(vocabSize).toBe(151669);
    });

    it('should have correct special token IDs', () => {
      expect(tokenizer.getPadTokenId()).toBe(151643); // <|endoftext|>
      expect(tokenizer.getEosTokenId()).toBe(151645); // <|im_end|>
      expect(tokenizer.getBosTokenId()).toBeNull(); // Qwen3 doesn't use BOS (NAPI returns null for None)
    });
  });

  describe('Basic Encoding', () => {
    it('should encode simple text', async () => {
      const text = 'Hello, world!';
      const tokens = await tokenizer.encode(text, false);

      expect(tokens).toBeInstanceOf(Uint32Array);
      expect(tokens.length).toBeGreaterThan(0);
    });

    it('should encode empty string', async () => {
      const tokens = await tokenizer.encode('', false);
      expect(tokens).toBeInstanceOf(Uint32Array);
      expect(tokens.length).toBe(0);
    });

    it('should encode text with special characters', async () => {
      const text = 'Hello 世界! 🌍';
      const tokens = await tokenizer.encode(text, false);

      expect(tokens.length).toBeGreaterThan(0);
    });

    it('should encode with and without special tokens', async () => {
      const text = 'Hello, world!';
      const withSpecial = await tokenizer.encode(text, true);
      const withoutSpecial = await tokenizer.encode(text, false);

      // Should differ if special tokens are added
      expect(withSpecial).toBeInstanceOf(Uint32Array);
      expect(withoutSpecial).toBeInstanceOf(Uint32Array);
    });
  });

  describe('Basic Decoding', () => {
    it('should decode simple tokens', async () => {
      const text = 'Hello, world!';
      const tokens = await tokenizer.encode(text, false);
      const decoded = await tokenizer.decode(tokens, false);

      expect(decoded).toBe(text);
    });

    it('should decode empty token array', async () => {
      const tokens = new Uint32Array([]);
      const decoded = await tokenizer.decode(tokens, false);

      expect(decoded).toBe('');
    });

    it('should handle special token skipping', async () => {
      const text = 'Hello';
      const tokens = await tokenizer.encode(text, false);

      const withSpecial = await tokenizer.decode(tokens, false);
      const withoutSpecial = await tokenizer.decode(tokens, true);

      expect(withSpecial).toBeTruthy();
      expect(withoutSpecial).toBeTruthy();
    });
  });

  describe('Roundtrip Encode/Decode', () => {
    const testCases = [
      'Hello, world!',
      'The quick brown fox jumps over the lazy dog.',
      'Numbers: 123456789',
      'Special chars: !@#$%^&*()',
      'Unicode: 你好世界 🌍',
      'Mixed: Hello 世界 123!',
      '',
    ];

    testCases.forEach((text) => {
      it(`should roundtrip: "${text.substring(0, 30)}..."`, async () => {
        const tokens = await tokenizer.encode(text, false);
        const decoded = await tokenizer.decode(tokens, false);

        expect(decoded).toBe(text);
      });
    });
  });

  describe('Batch Encoding', () => {
    it('should encode multiple texts in batch', async () => {
      const texts = ['Hello', 'World', 'Test'];
      const batchTokens = await tokenizer.encodeBatch(texts, false);

      expect(batchTokens).toHaveLength(3);
      batchTokens.forEach((tokens) => {
        expect(tokens).toBeInstanceOf(Uint32Array);
        expect(tokens.length).toBeGreaterThan(0);
      });
    });

    it('should encode empty batch', async () => {
      const batchTokens = await tokenizer.encodeBatch([], false);
      expect(batchTokens).toHaveLength(0);
    });

    it('should handle batch with empty strings', async () => {
      const texts = ['Hello', '', 'World'];
      const batchTokens = await tokenizer.encodeBatch(texts, false);

      expect(batchTokens).toHaveLength(3);
      expect(batchTokens[0].length).toBeGreaterThan(0);
      expect(batchTokens[1].length).toBe(0); // Empty string
      expect(batchTokens[2].length).toBeGreaterThan(0);
    });
  });

  describe('Batch Decoding', () => {
    it('should decode multiple token arrays in batch', async () => {
      const texts = ['Hello', 'World', 'Test'];
      const batchTokens = await tokenizer.encodeBatch(texts, false);
      const decodedTexts = await tokenizer.decodeBatch(batchTokens, false);

      expect(decodedTexts).toHaveLength(3);
      expect(decodedTexts[0]).toBe('Hello');
      expect(decodedTexts[1]).toBe('World');
      expect(decodedTexts[2]).toBe('Test');
    });

    it('should decode empty batch', async () => {
      const decodedTexts = await tokenizer.decodeBatch([], false);
      expect(decodedTexts).toHaveLength(0);
    });
  });

  describe('Chat Template', () => {
    it('should apply chat template to single message', async () => {
      const messages: ChatMessage[] = [{ role: 'user', content: 'Hello!' }];

      const tokens = await tokenizer.applyChatTemplate(messages, true);

      expect(tokens).toBeInstanceOf(Uint32Array);
      expect(tokens.length).toBeGreaterThan(0);

      // Decode and verify format
      const formatted = await tokenizer.decode(tokens, false);
      expect(formatted).toContain('<|im_start|>');
      expect(formatted).toContain('<|im_end|>');
      expect(formatted).toContain('user');
      expect(formatted).toContain('Hello!');
    });

    it('should apply chat template with system message', async () => {
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is 2+2?' },
      ];

      const tokens = await tokenizer.applyChatTemplate(messages, true);
      const formatted = await tokenizer.decode(tokens, false);

      expect(formatted).toContain('system');
      expect(formatted).toContain('helpful assistant');
      expect(formatted).toContain('user');
      expect(formatted).toContain('2+2');
    });

    it('should apply chat template with full conversation', async () => {
      const messages: ChatMessage[] = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'Hello!' },
        { role: 'assistant', content: 'Hi! How can I help you?' },
        { role: 'user', content: 'What is the weather?' },
      ];

      const tokens = await tokenizer.applyChatTemplate(messages, true);
      const formatted = await tokenizer.decode(tokens, false);

      // Should have all roles
      const systemCount = (formatted.match(/<\|im_start\|>system/g) || []).length;
      const userCount = (formatted.match(/<\|im_start\|>user/g) || []).length;
      const assistantCount = (formatted.match(/<\|im_start\|>assistant/g) || []).length;

      expect(systemCount).toBe(1);
      expect(userCount).toBe(2);
      expect(assistantCount).toBeGreaterThanOrEqual(1); // At least 1 (conversation + generation prompt)
    });

    it('should add generation prompt when requested', async () => {
      const messages: ChatMessage[] = [{ role: 'user', content: 'Hello!' }];

      const withPrompt = await tokenizer.applyChatTemplate(messages, true);
      const withoutPrompt = await tokenizer.applyChatTemplate(messages, false);

      // With prompt should be longer (includes <|im_start|>assistant\n)
      expect(withPrompt.length).toBeGreaterThan(withoutPrompt.length);

      const formattedWith = await tokenizer.decode(withPrompt, false);
      const formattedWithout = await tokenizer.decode(withoutPrompt, false);

      // With prompt should end with assistant start tag
      expect(formattedWith).toMatch(/<\|im_start\|>assistant\s*$/);

      // Without prompt should not
      expect(formattedWithout).not.toMatch(/<\|im_start\|>assistant\s*$/);
    });
  });

  describe('Token/ID Conversion', () => {
    it('should convert token to ID', () => {
      const id = tokenizer.tokenToId('<|endoftext|>');
      expect(id).toBe(151643);
    });

    it('should convert ID to token for special tokens', () => {
      const token = tokenizer.idToToken(151643);
      expect(token).toBe('<|endoftext|>');

      const imStart = tokenizer.idToToken(151644);
      expect(imStart).toBe('<|im_start|>');

      const imEnd = tokenizer.idToToken(151645);
      expect(imEnd).toBe('<|im_end|>');
    });

    it('should return null for invalid token', () => {
      const id = tokenizer.tokenToId('NONEXISTENT_TOKEN_XYZ123');
      expect(id).toBeNull(); // NAPI returns null for Option::None
    });

    it('should return null for invalid ID', () => {
      const token = tokenizer.idToToken(999999999);
      expect(token).toBeNull(); // NAPI returns null for Option::None
    });
  });

  describe('Special Tokens', () => {
    it('should get special token strings', () => {
      expect(tokenizer.getImStartToken()).toBe('<|im_start|>');
      expect(tokenizer.getImEndToken()).toBe('<|im_end|>');
      expect(tokenizer.getEndoftextToken()).toBe('<|endoftext|>');
    });

    it('should encode and decode special tokens', async () => {
      const text = '<|im_start|>system\nYou are helpful<|im_end|>';
      const tokens = await tokenizer.encode(text, false);
      const decoded = await tokenizer.decode(tokens, false);

      expect(decoded).toBe(text);
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long text', async () => {
      const longText = 'Hello '.repeat(1000);
      const tokens = await tokenizer.encode(longText, false);

      expect(tokens.length).toBeGreaterThan(1000);

      const decoded = await tokenizer.decode(tokens, false);
      expect(decoded).toBe(longText);
    });

    it('should handle text with only whitespace', async () => {
      const text = '   \n\t  ';
      const tokens = await tokenizer.encode(text, false);
      const decoded = await tokenizer.decode(tokens, false);

      expect(decoded).toBe(text);
    });

    it('should handle newlines and tabs', async () => {
      const text = 'Line 1\nLine 2\tTab';
      const tokens = await tokenizer.encode(text, false);
      const decoded = await tokenizer.decode(tokens, false);

      expect(decoded).toBe(text);
    });
  });

  describe('Performance', () => {
    it('should encode and decode efficiently', async () => {
      const start = Date.now();

      for (let i = 0; i < 100; i++) {
        const text = `Test message ${i}`;
        const tokens = await tokenizer.encode(text, false);
        await tokenizer.decode(tokens, false);
      }

      const elapsed = Date.now() - start;

      // Should complete 100 roundtrips in under 1 second
      expect(elapsed).toBeLessThan(1000);
    });

    it('should handle batch encoding efficiently', async () => {
      const texts = Array.from({ length: 100 }, (_, i) => `Test message ${i}`);

      const start = Date.now();
      const batchTokens = await tokenizer.encodeBatch(texts, false);
      const elapsed = Date.now() - start;

      expect(batchTokens).toHaveLength(100);

      // Batch encoding should be fast
      expect(elapsed).toBeLessThan(500);
    });
  });
});
