import { parseXmlCot, extractXmlAnswer } from './utils/xml-parser';
import type { RewardFunction } from './types';
import type { RewardOutput } from '@mlx-node/core';

// Re-export the types for convenience
export type { RewardFunction } from './types';
export type { RewardOutput } from '@mlx-node/core';

function normalizeAnswerValue(value: string | null | undefined): string | null {
  if (value == null) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const withoutCommas = trimmed.replace(/,/g, '');
  const collapsedWhitespace = withoutCommas.replace(/\s+/g, '');
  if (/^[-+]?\d+(\.0+)?$/.test(collapsedWhitespace)) {
    const parsed = Number.parseFloat(collapsedWhitespace);
    if (Number.isFinite(parsed)) {
      return Math.trunc(parsed).toString();
    }
  }
  return trimmed;
}

/**
 * Reward for correct answers.
 * Extracts answer from XML tags and compares with expected answer.
 * Returns 2.0 for correct, 0.0 for incorrect.
 */
export const correctnessReward: RewardFunction = (outputs) => {
  return outputs.map((output) => {
    const expectedAnswer = output.expectedAnswer;
    if (!expectedAnswer) {
      return 0;
    }

    // Use rawText to preserve original XML structure for parsing
    const extractedAnswer = extractXmlAnswer(output.completion.rawText);

    const normalizedGold = normalizeAnswerValue(expectedAnswer);
    const normalizedPrediction = normalizeAnswerValue(extractedAnswer);

    if (normalizedGold != null && normalizedPrediction === normalizedGold) {
      return 2.0;
    }
    return 0.0;
  });
};

function isIntegerString(value: string | null): boolean {
  if (value == null) return false;
  const trimmed = value.trim();
  if (!trimmed) return false;
  const normalized = trimmed.replace(/,/g, '').replace(/\s+/g, '');
  if (!/^[-+]?\d+$/.test(normalized)) return false;
  const parsed = Number.parseInt(normalized, 10);
  return Number.isFinite(parsed);
}

/**
 * Reward for integer-formatted answers.
 * Returns 0.5 if the extracted answer is a valid integer, 0.0 otherwise.
 */
export const integerReward: RewardFunction = (outputs) => {
  return outputs.map((output) => {
    const parsed = extractXmlAnswer(output.completion.rawText);
    return isIntegerString(parsed) ? 0.5 : 0.0;
  });
};

/**
 * Reward for strict XML format adherence.
 * Returns 0.5 if format matches strictly, 0.0 otherwise.
 */
export const strictFormatReward: RewardFunction = (outputs) => {
  return outputs.map((output) => {
    const result = parseXmlCot(output.completion.rawText);
    return result.isStrictMatch ? 0.5 : 0.0;
  });
};

/**
 * Reward for soft XML format adherence.
 * Returns 0.5 if format matches loosely, 0.0 otherwise.
 */
export const softFormatReward: RewardFunction = (outputs) => {
  return outputs.map((output) => {
    const result = parseXmlCot(output.completion.rawText);
    return result.isSoftMatch ? 0.5 : 0.0;
  });
};

function hasEnclosedTag(content: string, openTag: string, closeTag: string): boolean {
  const lower = content.toLowerCase();
  const openIndex = lower.indexOf(openTag);
  if (openIndex === -1) return false;
  const closeIndex = lower.indexOf(closeTag, openIndex + openTag.length);
  return closeIndex !== -1;
}

function trailingCharactersAfterAnswer(content: string): number {
  const lower = content.toLowerCase();
  const closeIndex = lower.indexOf('</answer>');
  if (closeIndex === -1) return 0;
  const trailing = content.slice(closeIndex + '</answer>'.length).trim();
  return trailing.length;
}

/**
 * Reward for XML tag presence and penalize trailing content.
 * +0.25 for <reasoning> tags, +0.25 for <answer> tags.
 * -0.001 per character after </answer>.
 */
export const xmlCountReward: RewardFunction = (outputs) => {
  return outputs.map((output) => {
    const completion = output.completion.rawText;
    let score = 0.0;
    if (hasEnclosedTag(completion, '<reasoning>', '</reasoning>')) {
      score += 0.25;
    }
    if (hasEnclosedTag(completion, '<answer>', '</answer>')) {
      score += 0.25;
    }
    const trailing = trailingCharactersAfterAnswer(completion);
    if (trailing > 0) {
      score -= trailing * 0.001;
    }
    return score;
  });
};

export const ALL_REWARD_FUNCTIONS: RewardFunction[] = [
  correctnessReward,
  integerReward,
  strictFormatReward,
  softFormatReward,
  xmlCountReward,
];
