import { describe, expect, it } from 'vitest';
import {
  ALL_REWARD_FUNCTIONS,
  correctnessReward,
  integerReward,
  softFormatReward,
  strictFormatReward,
  xmlCountReward,
} from '../../src/index';

// Helper to create test inputs with the new simpler signature
function createInput(
  completionContent: string,
  answer: string | null,
): { prompts: string[]; completions: string[]; answers: (string | null)[] } {
  const prompt = '<|im_start|>system\nRespond in XML format.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n';
  return {
    prompts: [prompt],
    completions: [completionContent],
    answers: [answer],
  };
}

describe('GRPO reward functions', () => {
  it('awards correctness reward for matching numeric answers despite formatting', () => {
    const { prompts, completions, answers } = createInput(
      `<reasoning>Compute carefully.</reasoning><answer>1,000.0</answer>`,
      '1000',
    );
    expect(correctnessReward(prompts, completions, answers)).toEqual([2.0]);
  });

  it('returns zero correctness reward when answer is missing', () => {
    const { prompts, completions, answers } = createInput(`<reasoning>...</reasoning><answer>12</answer>`, null);
    expect(correctnessReward(prompts, completions, answers)).toEqual([0.0]);
  });

  it('awards integer reward when completion answer is an integer', () => {
    const { prompts, completions, answers } = createInput(
      `<reasoning>Think.</reasoning><answer>42</answer>`,
      'irrelevant',
    );
    expect(integerReward(prompts, completions, answers)).toEqual([0.5]);
  });

  it('returns zero integer reward for non-integer answers', () => {
    const { prompts, completions, answers } = createInput(
      `<reasoning>Think.</reasoning><answer>3.14</answer>`,
      'irrelevant',
    );
    expect(integerReward(prompts, completions, answers)).toEqual([0.0]);
  });

  it('detects strict XML format when no extra characters are present', () => {
    const { prompts, completions, answers } = createInput(
      `<reasoning>\nSteps\n</reasoning>\n<answer>\n24\n</answer>\n`,
      '24',
    );
    expect(strictFormatReward(prompts, completions, answers)).toEqual([0.5]);
    expect(softFormatReward(prompts, completions, answers)).toEqual([0.5]);
  });

  it('downgrades to soft match when extra characters surround the XML', () => {
    const { prompts, completions, answers } = createInput(
      `Intro<reasoning>Plan</reasoning><answer>Done</answer>Outro`,
      'Done',
    );
    expect(strictFormatReward(prompts, completions, answers)).toEqual([0.0]);
    expect(softFormatReward(prompts, completions, answers)).toEqual([0.5]);
  });

  it('computes xml count reward with penalties for trailing text', async () => {
    const { prompts, completions, answers } = createInput(`<reasoning>R</reasoning><answer>A</answer>junk`, 'A');
    const result = await xmlCountReward(prompts, completions, answers);
    const scores = result instanceof Float32Array ? Array.from(result) : result;
    expect(scores[0]).toBeCloseTo(0.25 + 0.25 - 4 * 0.001);
  });

  it('returns zero xml count reward when tags are missing', () => {
    const { prompts, completions, answers } = createInput(`No tags here`, null);
    expect(xmlCountReward(prompts, completions, answers)).toEqual([0.0]);
  });

  it('supports broadcasting a single answer across multiple completions', () => {
    const prompts = [
      '<|im_start|>system\nFormat as XML.<|im_end|>\n<|im_start|>user\nQuestion?<|im_end|>\n',
      '<|im_start|>system\nFormat as XML.<|im_end|>\n<|im_start|>user\nAnother question?<|im_end|>\n',
    ];
    const completions = [`<reasoning>r</reasoning><answer>10</answer>`, `<reasoning>r</reasoning><answer>11</answer>`];
    const answers = ['10'];

    expect(() => correctnessReward(prompts, completions, answers)).not.toThrow();
    expect(correctnessReward(prompts, completions, answers)).toEqual([2.0, 0.0]);
  });

  it('throws when completions and prompts length mismatch', () => {
    const prompts = ['<|im_start|>system\nhi<|im_end|>\n'];
    const completions: string[] = [];
    const answers: (string | null)[] = [];

    expect(() => correctnessReward(prompts, completions, answers)).toThrow(/Expected prompts and completions/);
  });

  it('throws when answer array has incompatible length', () => {
    const prompts = ['<|im_start|>system\nhi<|im_end|>\n'];
    const completions = ['test'];
    const answers: (string | null)[] = [];

    expect(() => correctnessReward(prompts, completions, answers)).toThrow(
      /Answers must contain either one shared value/,
    );
  });

  it('exposes all reward functions in canonical order', () => {
    expect(ALL_REWARD_FUNCTIONS).toHaveLength(5);
    expect(ALL_REWARD_FUNCTIONS[0]).toBe(correctnessReward);
    expect(ALL_REWARD_FUNCTIONS[4]).toBe(xmlCountReward);
  });
});
