/**
 * The native thinking guard's four result fields, as the turn trace records them: a field the turn did not send is
 * absent.
 */

import type { ChatResult } from '@mlx-node/core';

import { TURN_REASONING_CLOSES, type TurnReasoningClose } from './metrics-trace.js';

/** The four fields as a `ChatResult` or a terminal stream chunk carries them. */
export type NativeReasoningOutcome = Pick<
  ChatResult,
  'reasoningClose' | 'reasoningCutAt' | 'reasoningCyclePeriod' | 'reasoningStrayCloses'
>;

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
export function reasoningOutcomeOf(v: NativeReasoningOutcome): ReasoningOutcome {
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
