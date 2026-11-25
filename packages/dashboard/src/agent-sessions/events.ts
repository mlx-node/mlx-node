/**
 * Compact timing digest of grok-build (`mlx agent`) `events.jsonl` logs. Pure data → data.
 *
 * `updates.jsonl` coalesces streamed reasoning and text chunks into a single entry stamped with
 * the final chunk's time. Sibling `events.jsonl` records millisecond-level `waiting_for_model`,
 * `first_token`, per-chunk `streaming_reasoning` / `streaming_text`, and `tool_execution`
 * transitions so the timeline can place waits, generation phases, and tool-call argument decode
 * accurately.
 */

import { splitCompleteLines } from './transcript.js';

const MAX_DATE_MS = 8_640_000_000_000_000;

export type EventSegmentKind = 'wait-model' | 'reasoning' | 'answer' | 'tool-call';

export interface EventSegment {
  kind: EventSegmentKind;
  start: number;
  /** `null` when the segment is still open at EOF (in-progress or crashed turn). */
  end: number | null;
  /** Last recorded timestamp belonging to this segment (`=== end` when closed). */
  lastEnd: number;
  /** Only for `wait-model`: what ended the wait. */
  until?: 'text' | 'tool' | 'open';
}

export interface ParsedEvents {
  segments: EventSegment[];
  /** Largest valid timestamp seen in the file; 0 when none. */
  lastTs: number;
}

type ActiveState =
  | { state: 'idle' }
  | { state: 'waiting'; start: number }
  | { state: 'first_token'; at: number }
  | { state: 'reasoning'; start: number; last: number }
  | { state: 'answer'; start: number; last: number }
  | { state: 'tool' };

function parseIsoMs(value: unknown): number {
  if (typeof value !== 'string') return 0;
  const ms = Date.parse(value);
  return Number.isFinite(ms) && ms > 0 && ms <= MAX_DATE_MS ? ms : 0;
}

/**
 * Parse a whole `events.jsonl` file into chronological wait/generation segments.
 * Invalid lines, torn trailing lines, and non-timing events are silently skipped.
 */
export function parseEvents(text: string): ParsedEvents {
  const segments: EventSegment[] = [];
  let lastTs = 0;
  let active: ActiveState = { state: 'idle' };

  const pushClosed = (kind: EventSegmentKind, start: number, end: number, until?: 'text' | 'tool' | 'open'): void => {
    const clampedEnd = Math.max(start, end);
    segments.push(
      until !== undefined
        ? { kind, start, end: clampedEnd, lastEnd: clampedEnd, until }
        : { kind, start, end: clampedEnd, lastEnd: clampedEnd },
    );
  };

  const closeStream = (): void => {
    if (active.state === 'reasoning') pushClosed('reasoning', active.start, active.last);
    else if (active.state === 'answer') pushClosed('answer', active.start, active.last);
  };

  for (const line of splitCompleteLines(text)) {
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch {
      continue;
    }
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) continue;
    const obj = raw as Record<string, unknown>;
    const ts = parseIsoMs(obj.ts);
    if (ts === 0) continue;
    if (ts > lastTs) lastTs = ts;

    const type = obj.type;
    if (type === 'turn_started') {
      closeStream();
      active = { state: 'idle' };
      continue;
    }
    if (type === 'loop_started') {
      if (active.state === 'waiting') pushClosed('wait-model', active.start, ts, 'open');
      else closeStream();
      active = { state: 'idle' };
      continue;
    }
    if (type === 'turn_ended') {
      if (active.state === 'waiting') pushClosed('wait-model', active.start, ts, 'open');
      else closeStream();
      active = { state: 'idle' };
      continue;
    }
    if (type === 'first_token') {
      if (active.state === 'waiting') pushClosed('wait-model', active.start, ts, 'text');
      else closeStream();
      active = { state: 'first_token', at: ts };
      continue;
    }
    if (type === 'phase_changed') {
      const phase = obj.phase;
      if (phase === 'waiting_for_model') {
        if (active.state === 'waiting') continue;
        closeStream();
        active = { state: 'waiting', start: ts };
      } else if (phase === 'streaming_reasoning') {
        if (active.state === 'reasoning') {
          if (ts > active.last) active.last = ts;
        } else {
          if (active.state === 'waiting') pushClosed('wait-model', active.start, ts, 'text');
          else if (active.state === 'answer') pushClosed('answer', active.start, active.last);
          const start: number = active.state === 'first_token' ? Math.min(active.at, ts) : ts;
          active = { state: 'reasoning', start, last: Math.max(start, ts) };
        }
      } else if (phase === 'streaming_text') {
        if (active.state === 'answer') {
          if (ts > active.last) active.last = ts;
        } else {
          if (active.state === 'waiting') pushClosed('wait-model', active.start, ts, 'text');
          else if (active.state === 'reasoning') pushClosed('reasoning', active.start, active.last);
          const start: number = active.state === 'first_token' ? Math.min(active.at, ts) : ts;
          active = { state: 'answer', start, last: Math.max(start, ts) };
        }
      } else if (phase === 'tool_execution') {
        if (active.state === 'tool') continue;
        if (active.state === 'waiting') {
          pushClosed('wait-model', active.start, ts, 'tool');
        } else if (active.state === 'first_token') {
          if (ts > active.at) pushClosed('tool-call', active.at, ts);
        } else if (active.state === 'reasoning') {
          pushClosed('reasoning', active.start, active.last);
          if (ts > active.last) pushClosed('tool-call', active.last, ts);
        } else if (active.state === 'answer') {
          pushClosed('answer', active.start, active.last);
          if (ts > active.last) pushClosed('tool-call', active.last, ts);
        }
        active = { state: 'tool' };
      } else if (phase === 'permission_prompt') {
        if (active.state !== 'tool') {
          closeStream();
          active = { state: 'tool' };
        }
      }
      continue;
    }
    if (type === 'tool_started') {
      if (active.state === 'tool') continue;
      if (active.state === 'waiting') {
        pushClosed('wait-model', active.start, ts, 'tool');
      } else if (active.state === 'first_token') {
        if (ts > active.at) pushClosed('tool-call', active.at, ts);
      } else if (active.state === 'reasoning') {
        pushClosed('reasoning', active.start, active.last);
        if (ts > active.last) pushClosed('tool-call', active.last, ts);
      } else if (active.state === 'answer') {
        pushClosed('answer', active.start, active.last);
        if (ts > active.last) pushClosed('tool-call', active.last, ts);
      }
      active = { state: 'tool' };
      continue;
    }
    if (type === 'permission_requested' || type === 'permission_resolved' || type === 'tool_completed') {
      if (active.state !== 'tool') {
        closeStream();
        active = { state: 'tool' };
      }
    }
  }

  // Open segment at EOF (in-progress or crashed turn without turn_ended).
  if (active.state === 'waiting') {
    segments.push({ kind: 'wait-model', start: active.start, end: null, lastEnd: active.start, until: 'open' });
  } else if (active.state === 'reasoning') {
    segments.push({ kind: 'reasoning', start: active.start, end: null, lastEnd: active.last });
  } else if (active.state === 'answer') {
    segments.push({ kind: 'answer', start: active.start, end: null, lastEnd: active.last });
  }

  return { segments, lastTs };
}
