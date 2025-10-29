import { parseXmlCot, extractXmlAnswer } from './utils/xml-parser';
import type { RewardFunction } from './types';

// Re-export the type for convenience
export type { RewardFunction } from './types';

function assertConsistentLengths(prompts: string[], completions: string[], answers: (string | null)[]): void {
  if (prompts.length !== completions.length) {
    throw new Error(
      `Expected prompts and completions to have equal length, got ${prompts.length} vs ${completions.length}.`,
    );
  }
  if (!(answers.length === 1 || answers.length === completions.length)) {
    throw new Error(
      `Answers must contain either one shared value or align with completions. Received ${answers.length} entries for ${completions.length} completions.`,
    );
  }
}

function getAnswerForIndex(answers: (string | null)[], index: number): string | null {
  if (answers.length === 0) return null;
  if (answers.length === 1) return answers[0] ?? null;
  return answers[index] ?? null;
}

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

export const correctnessReward: RewardFunction = (prompts, completions, answers) => {
  assertConsistentLengths(prompts, completions, answers);
  const scores: number[] = [];

  completions.forEach((completion, index) => {
    const answer = getAnswerForIndex(answers, index);
    if (!answer) {
      scores.push(0);
      return;
    }

    const extractedAnswer = extractXmlAnswer(completion);

    const normalizedGold = normalizeAnswerValue(answer);
    const normalizedPrediction = normalizeAnswerValue(extractedAnswer);

    if (normalizedGold != null && normalizedPrediction === normalizedGold) {
      scores.push(2.0);
    } else {
      scores.push(0.0);
    }
  });

  return scores;
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

export const integerReward: RewardFunction = (prompts, completions, answers) => {
  assertConsistentLengths(prompts, completions, answers);
  return completions.map((completion) => {
    const parsed = extractXmlAnswer(completion);
    return isIntegerString(parsed) ? 0.5 : 0.0;
  });
};

export const strictFormatReward: RewardFunction = (prompts, completions, answers) => {
  assertConsistentLengths(prompts, completions, answers);
  return completions.map((completion) => {
    const result = parseXmlCot(completion);
    return result.isStrictMatch ? 0.5 : 0.0;
  });
};

export const softFormatReward: RewardFunction = (prompts, completions, answers) => {
  assertConsistentLengths(prompts, completions, answers);
  return completions.map((completion) => {
    const result = parseXmlCot(completion);
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

export const xmlCountReward: RewardFunction = (prompts, completions, answers) => {
  assertConsistentLengths(prompts, completions, answers);
  return completions.map((completion) => {
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
