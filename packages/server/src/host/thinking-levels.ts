/**
 * The thinking levels of a server with `thinkingLevels` (the desktop engine and the server `mlx agent` starts): each
 * chat family's row in `@mlx-node/lm` family-data. `/v1/models` lists them as `reasoningEfforts`, and a Messages
 * request's `extra_body.reasoning_effort` maps through them. The default level sets nothing, so a request that names
 * it runs as one that names none.
 */

import type { ChatConfig } from '@mlx-node/core';
import { familyTraitsFor, THINKING_LEVELS, type ThinkingLevel } from '@mlx-node/lm/family-data';

/** One entry of a `/v1/models` row's `reasoningEfforts`, in the shape the agent reads (`ReasoningEffortOption`). */
export interface ReasoningEffortRow {
  value: ThinkingLevel;
  label: string;
  default?: true;
}

export interface FamilyLevels {
  /** Least thinking first. */
  levels: ThinkingLevel[];
  default: ThinkingLevel;
}

/** What a level sets on a Messages request; empty for the default level. */
export type LevelSetting = Pick<ChatConfig, 'reasoningEffort' | 'thinkingTokenBudget'>;

const LABELS: Record<ThinkingLevel, string> = { none: 'Off', low: 'Low', medium: 'Medium', high: 'High', max: 'Max' };

function isLevel(value: string): value is ThinkingLevel {
  return (THINKING_LEVELS as readonly string[]).includes(value);
}

/** The family's levels; `null` when it offers none or is not a chat family. */
export function levelsFor(family: string): FamilyLevels | null {
  const row = familyTraitsFor(family)?.thinkingLevels;
  if (row === undefined) return null;
  const levels = THINKING_LEVELS.filter((level) => row[level] !== undefined);
  if (levels.length === 0) return null;
  const defaults = levels.filter((level) => row[level] === 'default');
  if (defaults.length !== 1) throw new Error(`Thinking levels of "${family}" need exactly one default level.`);
  return { levels, default: defaults[0]! };
}

/** The `reasoningEfforts` of a `/v1/models` row of the family; `undefined` when it offers no levels. */
export function reasoningEffortsFor(family: string): ReasoningEffortRow[] | undefined {
  const levels = levelsFor(family);
  if (levels === null) return undefined;
  return levels.levels.map((value) => ({
    value,
    label: LABELS[value],
    ...(value === levels.default ? { default: true as const } : {}),
  }));
}

/** What `effort` sets on a Messages request of the family; `null` when the family does not list it. */
export function thinkingFor(family: string, effort: string): LevelSetting | null {
  const row = familyTraitsFor(family)?.thinkingLevels;
  const level = row !== undefined && isLevel(effort) ? row[effort] : undefined;
  if (level === undefined) return null;
  if (level === 'default') return {};
  return {
    ...(level.reasoningEffort !== undefined ? { reasoningEffort: level.reasoningEffort } : {}),
    ...(level.thinkingTokenBudget !== undefined ? { thinkingTokenBudget: level.thinkingTokenBudget } : {}),
  };
}
