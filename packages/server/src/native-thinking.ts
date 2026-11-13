/**
 * The native thinking guard: the two config inputs the server sets and the four result fields it reads, typed here so
 * an addon that predates them still works. Such an addon ignores the inputs (the budget alone still bounds a thought)
 * and sends no outputs, which read as "not known".
 */

import type { ChatConfig } from '@mlx-node/core';

import { TURN_REASONING_CLOSES, type TurnReasoningClose } from './metrics-trace.js';

/** `ChatConfig` with the two guard inputs. */
export type ThinkingChatConfig = ChatConfig & {
  /** An exact repeat inside a thought closes it. Native default: off. */
  thinkingCycleGuard?: boolean;
  /** Plain text written before the template's close when a cut ends thinking. Native default: the close alone. */
  thinkingEndText?: string;
};

/** The native outcome of a thought, every field validated; a field the addon did not send is absent. */
export interface ReasoningOutcome {
  /** The model closed it, the budget or the cycle check cut it, or the turn ended inside it. */
  reasoningClose?: TurnReasoningClose;
  /** Thinking tokens when a cut began. */
  reasoningCutAt?: number;
  /** Period in tokens of the cycle the check saw. */
  reasoningCyclePeriod?: number;
  /** Close tokens after the thought ended. */
  reasoningStrayCloses?: number;
}

function count(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined;
}

/** The validated outcome fields of a `ChatResult` or a terminal stream event. */
export function reasoningOutcomeOf(value: unknown): ReasoningOutcome {
  if (typeof value !== 'object' || value === null) return {};
  const v = value as Record<string, unknown>;
  const out: ReasoningOutcome = {};
  for (const close of TURN_REASONING_CLOSES) if (v.reasoningClose === close) out.reasoningClose = close;
  const cutAt = count(v.reasoningCutAt);
  if (cutAt !== undefined) out.reasoningCutAt = cutAt;
  const period = count(v.reasoningCyclePeriod);
  if (period !== undefined && period > 0) out.reasoningCyclePeriod = period;
  const stray = count(v.reasoningStrayCloses);
  if (stray !== undefined) out.reasoningStrayCloses = stray;
  return out;
}
