import { describe, expect, it } from 'vitest';
import { resolve } from 'node:path';
import { SYSTEM_PROMPT, loadLocalGsm8kDataset, LocalGsm8kDatasetLoader, type ChatMessage } from '@mlx-node/trl';

describe('Local GSM8K dataset loader', () => {
  it('loads a limited number of training examples and extracts answers', async () => {
    const examples = await loadLocalGsm8kDataset('train', { limit: 2 });
    expect(examples).toHaveLength(2);

    const [first] = examples;
    expect(first.answer).toBe('72');
    expect(first.prompt[0]).toEqual({ role: 'system', content: SYSTEM_PROMPT });
    expect(first.prompt[1].role).toBe('user');
    expect(first.prompt[1].content).toContain('Natalia sold clips');
  });

  it('loads evaluation split and respects limit option', async () => {
    const examples = await loadLocalGsm8kDataset('test', { limit: 1 });
    expect(examples).toHaveLength(1);
  });

  it('supports includeOneShot option when formatting prompts', async () => {
    const examples = await loadLocalGsm8kDataset('train', {
      limit: 1,
      includeOneShot: true,
      oneShotExample: {
        question: 'Example Q',
        reasoning: 'Example reasoning',
        answer: 'Example answer',
      },
    });

    const [example] = examples;
    const assistantMessages = example.prompt.filter((message) => message.role === 'assistant');
    expect(assistantMessages).toHaveLength(1);
    expect(assistantMessages[0].content).toContain('Example answer');
  });

  it('allows custom prompt templates', async () => {
    const template = (question: string): ChatMessage[] => [{ role: 'user', content: `Q: ${question}` }];
    const examples = await loadLocalGsm8kDataset('train', { limit: 1, promptTemplate: template });
    expect(examples[0].prompt).toEqual([{ role: 'user', content: expect.stringContaining('Q:') }]);
  });

  it('throws for unknown splits', async () => {
    await expect(loadLocalGsm8kDataset('dev')).rejects.toThrow(/Unsupported GSM8K split/);
  });

  it('throws when JSONL file is missing', async () => {
    await expect(loadLocalGsm8kDataset('train', { basePath: 'non-existent-path' })).rejects.toThrow();
  });

  it('supports dataset loader class wrapper', async () => {
    const loader = new LocalGsm8kDatasetLoader();
    const examples = await loader.load('train', 1);
    expect(examples).toHaveLength(1);
  });

  it('can load from explicit base path', async () => {
    const basePath = resolve(process.cwd(), 'data/gsm8k');
    const examples = await loadLocalGsm8kDataset('train', { limit: 1, basePath });
    expect(examples).toHaveLength(1);
  });
});
