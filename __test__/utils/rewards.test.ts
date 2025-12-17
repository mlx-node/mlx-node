import { describe, expect, it } from 'vitest';
import {
  ALL_REWARD_FUNCTIONS,
  correctnessReward,
  integerReward,
  softFormatReward,
  strictFormatReward,
  xmlCountReward,
  type RewardOutput,
} from '@mlx-node/trl';

// Helper to create RewardOutput array from test inputs
function createRewardOutputs(completionContent: string, answer: string | null): RewardOutput[] {
  const prompt = '<|im_start|>system\nRespond in XML format.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n';
  return [
    {
      prompt,
      completion: {
        text: completionContent,
        rawText: completionContent,
        toolCalls: [],
        thinking: undefined,
        numTokens: completionContent.length,
        finishReason: 'stop',
      },
      expectedAnswer: answer ?? undefined,
    },
  ];
}

// Helper for multiple outputs
function createMultipleRewardOutputs(completions: string[], answers: (string | null)[]): RewardOutput[] {
  return completions.map((completion, i) => ({
    prompt: `<|im_start|>system\nFormat as XML.<|im_end|>\n<|im_start|>user\nQuestion ${i}?<|im_end|>\n`,
    completion: {
      text: completion,
      rawText: completion,
      toolCalls: [],
      thinking: undefined,
      numTokens: completion.length,
      finishReason: 'stop' as const,
    },
    expectedAnswer: (answers.length === 1 ? answers[0] : answers[i]) ?? undefined,
  }));
}

describe('GRPO reward functions', () => {
  it('awards correctness reward for matching numeric answers despite formatting', () => {
    const outputs = createRewardOutputs(`<reasoning>Compute carefully.</reasoning><answer>1,000.0</answer>`, '1000');
    expect(correctnessReward(outputs)).toEqual([2.0]);
  });

  it('returns zero correctness reward when answer is missing', () => {
    const outputs = createRewardOutputs(`<reasoning>...</reasoning><answer>12</answer>`, null);
    expect(correctnessReward(outputs)).toEqual([0.0]);
  });

  it('awards integer reward when completion answer is an integer', () => {
    const outputs = createRewardOutputs(`<reasoning>Think.</reasoning><answer>42</answer>`, 'irrelevant');
    expect(integerReward(outputs)).toEqual([0.5]);
  });

  it('returns zero integer reward for non-integer answers', () => {
    const outputs = createRewardOutputs(`<reasoning>Think.</reasoning><answer>3.14</answer>`, 'irrelevant');
    expect(integerReward(outputs)).toEqual([0.0]);
  });

  it('detects strict XML format when no extra characters are present', () => {
    const outputs = createRewardOutputs(`<reasoning>\nSteps\n</reasoning>\n<answer>\n24\n</answer>\n`, '24');
    expect(strictFormatReward(outputs)).toEqual([0.5]);
    expect(softFormatReward(outputs)).toEqual([0.5]);
  });

  it('downgrades to soft match when extra characters surround the XML', () => {
    const outputs = createRewardOutputs(`Intro<reasoning>Plan</reasoning><answer>Done</answer>Outro`, 'Done');
    expect(strictFormatReward(outputs)).toEqual([0.0]);
    expect(softFormatReward(outputs)).toEqual([0.5]);
  });

  it('computes xml count reward with penalties for trailing text', async () => {
    const outputs = createRewardOutputs(`<reasoning>R</reasoning><answer>A</answer>junk`, 'A');
    const result = await xmlCountReward(outputs);
    const scores = result instanceof Float32Array ? Array.from(result) : result;
    expect(scores[0]).toBeCloseTo(0.25 + 0.25 - 4 * 0.001);
  });

  it('returns zero xml count reward when tags are missing', () => {
    const outputs = createRewardOutputs(`No tags here`, null);
    expect(xmlCountReward(outputs)).toEqual([0.0]);
  });

  it('supports broadcasting a single answer across multiple completions', () => {
    const outputs = createMultipleRewardOutputs(
      [`<reasoning>r</reasoning><answer>10</answer>`, `<reasoning>r</reasoning><answer>11</answer>`],
      ['10'],
    );

    expect(() => correctnessReward(outputs)).not.toThrow();
    expect(correctnessReward(outputs)).toEqual([2.0, 0.0]);
  });

  it('handles empty outputs array', () => {
    const outputs: RewardOutput[] = [];
    expect(correctnessReward(outputs)).toEqual([]);
  });

  it('exposes all reward functions in canonical order', () => {
    expect(ALL_REWARD_FUNCTIONS).toHaveLength(5);
    expect(ALL_REWARD_FUNCTIONS[0]).toBe(correctnessReward);
    expect(ALL_REWARD_FUNCTIONS[4]).toBe(xmlCountReward);
  });
});
