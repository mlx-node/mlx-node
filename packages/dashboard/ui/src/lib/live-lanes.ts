/**
 * The rows of the live turn lists: one per client session, not one per model call. An agent makes one engine call per
 * step and runs its tool in between, so a session's row stays through that gap as Idle, and the session's next call
 * takes the same row in the same place. A request that names no session is a row of its own.
 */

import { useState } from 'react';

import { LANE_HOLD_SECONDS } from './metric-strings';
import type { RuntimeProcessLive } from './types';

export type LiveTurn = RuntimeProcessLive['turns'][number];

/** An idle session row leaves after this long without a new call. */
export const LANE_HOLD_MS = LANE_HOLD_SECONDS * 1000;

export interface LiveLane {
  /** The trace id of the lane's first call: the row's key for as long as the lane lives. */
  key: string;
  session: string | null;
  /** The running call, or the last one while the lane is idle. */
  turn: LiveTurn;
  /** Snapshot time at which the last call left the list; null while a call runs. */
  endedAt: number | null;
}

/** The lanes after one snapshot: a call keeps its lane, a session's next call takes its idle lane, a new one appends. */
export function nextLanes(prev: readonly LiveLane[], turns: readonly LiveTurn[], now: number): LiveLane[] {
  const byTrace = new Map(turns.map((turn) => [turn.traceId, turn]));
  const kept = new Set<string>();
  const lanes = prev.map((lane): LiveLane => {
    const turn = byTrace.get(lane.turn.traceId);
    if (turn === undefined) return { ...lane, endedAt: lane.endedAt ?? now };
    kept.add(turn.traceId);
    return { ...lane, turn, endedAt: null };
  });
  const fresh = turns.filter((turn) => !kept.has(turn.traceId)).sort((a, b) => a.startedAt - b.startedAt);
  for (const turn of fresh) {
    const idle =
      turn.clientSession === null
        ? -1
        : lanes.findIndex((lane) => lane.endedAt !== null && lane.session === turn.clientSession);
    if (idle >= 0) lanes[idle] = { ...lanes[idle]!, turn, endedAt: null };
    else lanes.push({ key: turn.traceId, session: turn.clientSession, turn, endedAt: null });
  }
  return lanes.filter((lane) => lane.endedAt === null || (lane.session !== null && now - lane.endedAt < LANE_HOLD_MS));
}

/** At most `cap` lanes, in their order; idle lanes give up their place first, the longest idle first. */
export function visibleLanes(lanes: readonly LiveLane[], cap: number): LiveLane[] {
  if (lanes.length <= cap) return [...lanes];
  const idle = lanes.filter((lane) => lane.endedAt !== null).sort((a, b) => a.endedAt! - b.endedAt!);
  const dropped = new Set(idle.slice(0, lanes.length - cap).map((lane) => lane.key));
  return lanes.filter((lane) => !dropped.has(lane.key)).slice(0, cap);
}

/** The lanes of one process across its snapshots; a new pid starts over. */
export function useLiveLanes(turns: readonly LiveTurn[], now: number, pid: number): LiveLane[] {
  const [state, setState] = useState(() => ({ turns, now, pid, lanes: nextLanes([], turns, now) }));
  if (state.turns === turns && state.now === now && state.pid === pid) return state.lanes;
  const lanes = nextLanes(state.pid === pid ? state.lanes : [], turns, now);
  setState({ turns, now, pid, lanes });
  return lanes;
}
