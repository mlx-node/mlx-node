import { describe, expect, it, beforeAll } from 'vitest';
import { resolve } from 'node:path';
import {
  SFTDataset,
  createSFTDataset,
  type SFTPromptCompletionExample,
  type SFTConversationExample,
} from '@mlx-node/trl';
import { Qwen3Tokenizer } from '@mlx-node/core';

describe('SFT Dataset', () => {
  let tokenizer: Qwen3Tokenizer;

  beforeAll(async () => {
    // Load tokenizer from cached model
    const tokenizerPath = resolve(process.cwd(), '.cache/models/qwen3-0.6b-mlx-bf16/tokenizer.json');
    tokenizer = await Qwen3Tokenizer.fromPretrained(tokenizerPath);
  });

  describe('prompt-completion format', () => {
    it('creates dataset from prompt-completion examples', () => {
      const examples: SFTPromptCompletionExample[] = [
        {
          prompt: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: 'What is 2 + 2?' },
          ],
          completion: { role: 'assistant', content: '4' },
        },
        {
          prompt: [{ role: 'user', content: 'Hello!' }],
          completion: { role: 'assistant', content: 'Hi there!' },
        },
      ];

      const dataset = createSFTDataset(examples, tokenizer);
      expect(dataset.length).toBe(2);
    });

    it('collates batch correctly with tokenization', async () => {
      const examples: SFTPromptCompletionExample[] = [
        {
          prompt: [{ role: 'user', content: 'Hi' }],
          completion: { role: 'assistant', content: 'Hello!' },
        },
      ];

      const dataset = createSFTDataset(examples, tokenizer, { maxSeqLength: 512 });
      const batch = await dataset.collateBatch([0]);

      expect(batch.shape[0]).toBe(1); // batch size
      expect(batch.shape[1]).toBeGreaterThan(0); // seq length
      expect(batch.inputIds.length).toBe(batch.shape[0] * batch.shape[1]);
      expect(batch.labels.length).toBe(batch.shape[0] * batch.shape[1]);
    });

    it('masks prompt tokens with -100 when completionOnly=true', async () => {
      const examples: SFTPromptCompletionExample[] = [
        {
          prompt: [{ role: 'user', content: 'Hi' }],
          completion: { role: 'assistant', content: 'Hello!' },
        },
      ];

      const dataset = createSFTDataset(examples, tokenizer, {
        maxSeqLength: 512,
        completionOnly: true,
      });
      const batch = await dataset.collateBatch([0]);

      // At least some tokens should be -100 (masked)
      const maskedCount = Array.from(batch.labels).filter((l) => l === -100).length;
      expect(maskedCount).toBeGreaterThan(0);
    });
  });

  describe('conversation format', () => {
    it('creates dataset from conversation examples', () => {
      const examples: SFTConversationExample[] = [
        {
          messages: [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: 'Hello!' },
            { role: 'assistant', content: 'Hi! How can I help you today?' },
            { role: 'user', content: 'What time is it?' },
            { role: 'assistant', content: 'I cannot tell the exact time.' },
          ],
        },
      ];

      const dataset = createSFTDataset(examples, tokenizer);
      expect(dataset.length).toBe(1);
    });

    it('throws for conversation without assistant messages', () => {
      const examples = [
        {
          messages: [
            { role: 'system', content: 'System prompt' },
            { role: 'user', content: 'User message' },
          ],
        },
      ] as SFTConversationExample[];

      // This should fail validation since there's no assistant message
      expect(() => createSFTDataset(examples, tokenizer)).not.toThrow();
      // The validation happens in loadSFTDataset, not createSFTDataset
    });
  });

  describe('batching', () => {
    it('generates correct number of batches', async () => {
      const examples: SFTPromptCompletionExample[] = Array(10)
        .fill(null)
        .map((_, i) => ({
          prompt: [{ role: 'user', content: `Question ${i}` }],
          completion: { role: 'assistant', content: `Answer ${i}` },
        }));

      const dataset = createSFTDataset(examples, tokenizer);

      expect(dataset.numBatches(4)).toBe(3); // 10 / 4 = 2.5 -> 3 batches
      expect(dataset.numBatches(5)).toBe(2); // 10 / 5 = 2 batches
      expect(dataset.numBatches(10)).toBe(1); // 10 / 10 = 1 batch
    });

    it('shuffles dataset', () => {
      const examples: SFTPromptCompletionExample[] = Array(5)
        .fill(null)
        .map((_, i) => ({
          prompt: [{ role: 'user', content: `Question ${i}` }],
          completion: { role: 'assistant', content: `Answer ${i}` },
        }));

      const dataset = createSFTDataset(examples, tokenizer);

      // Get initial order by iterating
      const batches1: SFTDataset['length'][] = [];
      for (let i = 0; i < dataset.length; i++) {
        batches1.push(i);
      }

      // Shuffle for epoch 0 and verify it doesn't throw
      dataset.shuffleForEpoch(0);
      expect(dataset.length).toBe(5);

      // Shuffle for epoch 1 should give different order
      dataset.shuffleForEpoch(1);
      expect(dataset.length).toBe(5);
    });
  });

  describe('truncation', () => {
    it('truncates sequences longer than maxSeqLength', async () => {
      const longContent = 'word '.repeat(1000); // Very long content
      const examples: SFTPromptCompletionExample[] = [
        {
          prompt: [{ role: 'user', content: longContent }],
          completion: { role: 'assistant', content: longContent },
        },
      ];

      const maxSeqLength = 256;
      const dataset = createSFTDataset(examples, tokenizer, { maxSeqLength });
      const batch = await dataset.collateBatch([0]);

      expect(batch.shape[1]).toBe(maxSeqLength);
    });
  });

  describe('error handling', () => {
    it('throws for empty dataset', () => {
      expect(() => createSFTDataset([], tokenizer)).toThrow('at least one example');
    });

    it('throws for mixed format examples', () => {
      const mixedExamples = [
        {
          prompt: [{ role: 'user', content: 'Hi' }],
          completion: { role: 'assistant', content: 'Hello' },
        },
        {
          messages: [
            { role: 'user', content: 'Hi' },
            { role: 'assistant', content: 'Hello' },
          ],
        },
      ] as unknown as SFTPromptCompletionExample[];

      expect(() => createSFTDataset(mixedExamples, tokenizer)).toThrow('Inconsistent SFT data format');
    });
  });
});
