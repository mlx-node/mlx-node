/** Pure helpers of thinking levels, shared by the core and the page. No Node imports: the page imports this file. */

import { THINKING_LEVELS, type ModelThinking, type ThinkingLevel } from './types.js';

export function isThinkingLevel(value: unknown): value is ThinkingLevel {
  return typeof value === 'string' && (THINKING_LEVELS as readonly string[]).includes(value);
}

/**
 * The level a conversation runs at on a model: the picked one when the model has it, else the nearest level below it
 * that the model has, else the model's default. Absent: the default.
 */
export function stepLevel(level: ThinkingLevel | undefined, thinking: ModelThinking): ThinkingLevel {
  if (level === undefined || thinking.levels.includes(level)) return level ?? thinking.default;
  for (let i = THINKING_LEVELS.indexOf(level) - 1; i >= 0; i--) {
    const below = THINKING_LEVELS[i]!;
    if (thinking.levels.includes(below)) return below;
  }
  return thinking.default;
}
