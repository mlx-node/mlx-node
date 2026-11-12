import type {
  AgentHandoff,
  AgentLane,
  AgentOutcome,
  AgentSpan,
  AgentTimelineMode,
  AgentTokenReport,
  AxisUnit,
} from '../components/arc-x/agent-timeline/agent-timeline-lib';
import { replayRate } from '../components/arc-x/agent-timeline/agent-timeline-lib';
import { sessionUrl } from '../components/session-dialogs';
import { formatClock, formatCountNoun, formatDuration } from './format';
import { axisNumber, timeAxis } from './metrics-view';
import type { SessionTimelineResponse, TimelineLane, TimelineNoSteps, TimelineSpan, TranscriptEntry } from './types';

/** One unit for the whole axis: ms under 2 s, s under 120 s, else min. */
export function timelineAxis(spanMs: number): AxisUnit {
  const axis = timeAxis(spanMs);
  return { unitMs: axis.div, format: (ms: number) => `${axisNumber(ms / axis.div)} ${axis.unit}` };
}

export interface TimelineModel {
  lanes: AgentLane[];
  spans: AgentSpan[];
  handoffs: AgentHandoff[];
  reports: AgentTokenReport[];
  durationMs: number;
  /** Where the turn starts on the axis; > 0 only for a woken turn drawn with the background work before it. */
  turnStartMs: number;
  mode: AgentTimelineMode;
  outcome: AgentOutcome;
  /** Sentences under the timeline. */
  footnote: string;
}

const NO_STEPS_NOTE: Record<Exclude<TimelineNoSteps, 'too-large'>, string> = {
  missing: "No step record: this subagent's session folder is gone.",
  empty: 'No step record: the subagent wrote no events.',
  unreadable: "No step record: the subagent's events could not be read.",
  'out-of-range': "No step record: the subagent's recorded events are outside this handoff's times.",
  'no-spawn': 'No step record: the spawn of this subagent was not recorded.',
  folded: 'Not drawn in detail: only the first 8 subagents get a lane.',
};
const START_NOTE = 'Start not recorded. Drawn from the nearest recorded event.';
const END_NOTE = 'End not recorded. Drawn to the next recorded event.';
const NO_TOOL_NOTE = 'Read from the order of events; the spawn call itself was not recorded.';
const NO_RETURN_NOTE = 'No return was recorded for this subagent.';
const CUT_START_NOTE = 'Began before this turn; drawn from the start of the axis.';
const IDLE_NOTE =
  'No turn was open. Handoffs here are background work, not steps of this agent. The turn starts where this line ends.';
const CUT_END_NOTE = 'Went on after this turn; drawn to the turn end.';
const MB = 1024 * 1024;

function noStepsNote(noSteps: TimelineNoSteps, lane: TimelineLane | undefined): string {
  if (noSteps !== 'too-large') return NO_STEPS_NOTE[noSteps];
  const size =
    lane?.fileBytes === null || lane?.fileBytes === undefined ? '' : ` (${String(Math.ceil(lane.fileBytes / MB))} MB)`;
  return `No step record: the subagent's event file is too large to read here${size}. Open the subagent to see it.`;
}

function toolWords(tools: readonly string[], calls: number, noun: string): string {
  const shown = tools.slice(0, 2).join(', ');
  const more = tools.length > 2 ? ` +${String(tools.length - 2)}` : '';
  return calls <= 1 ? `${noun} · ${tools[0] ?? ''}` : `${String(calls)} ${noun.toLowerCase()}s · ${shown}${more}`;
}

function spanWords(
  span: TimelineSpan,
  lane: TimelineLane | undefined,
): { label: string; facts: string[]; notes: string[] } {
  switch (span.kind) {
    case 'generate':
      return { label: span.phase === 'reasoning' ? 'Reasoning' : 'Writing the answer', facts: [], notes: [] };
    case 'tool':
      return {
        label: toolWords(span.tools, span.calls, 'Tool call'),
        facts: span.failedCalls > 0 && span.calls > 1 ? [`${String(span.failedCalls)} failed`] : [],
        notes: [],
      };
    case 'wait':
      if (span.reason === 'model') {
        return {
          label: span.until === 'text' ? 'Waiting for the first token' : 'Waiting for the model',
          facts: [],
          notes: [],
        };
      }
      return {
        label: 'Waiting for subagents',
        facts: [
          ...(span.calls > 0 ? [formatCountNoun(span.calls, 'spawn call', 'spawn calls')] : []),
          ...(span.failedCalls > 0 ? [`${String(span.failedCalls)} failed`] : []),
        ],
        notes: span.calls === 0 ? [NO_TOOL_NOTE] : [],
      };
    case 'run':
      return { label: 'Ran, no step record', facts: [], notes: [noStepsNote(span.noSteps, lane)] };
    case 'idle':
      return { label: 'Idle between turns', facts: [], notes: [IDLE_NOTE] };
  }
}

function openEdge(span: TimelineSpan): AgentSpan['open'] {
  if (span.inferredStart && span.inferredEnd) return 'both';
  if (span.inferredStart) return 'start';
  if (span.inferredEnd) return 'end';
  return undefined;
}

function cutEdge(span: TimelineSpan): AgentSpan['cut'] {
  if (span.cutStart && span.cutEnd) return 'both';
  if (span.cutStart) return 'start';
  if (span.cutEnd) return 'end';
  return undefined;
}

function outcomeOf(response: SessionTimelineResponse): AgentOutcome {
  const lastEvent = response.range?.lastEventMs ?? null;
  if (response.open && !response.live) {
    return { word: 'Not finished', tone: 'neutral', detail: `last event ${formatClock(lastEvent, true)}` };
  }
  if (response.open) return { word: 'In progress', tone: 'neutral' };
  switch (response.stopReason) {
    case null:
    case 'end_turn':
      return { word: 'Finished', tone: 'success' };
    case 'error':
      return { word: 'Failed', tone: 'danger' };
    case 'cancelled':
      return { word: 'Cancelled', tone: 'warning' };
    default:
      return { word: `Ended · ${response.stopReason}`, tone: 'neutral' };
  }
}

function plural(n: number, one: string, other: string): string {
  return n === 1 ? one : other;
}

/** The API body of one turn → AgentTimeline props and the footnote. `range` must not be null. */
export function timelineModel(
  response: SessionTimelineResponse,
  options: { fresh: boolean; formatDuration?: (ms: number) => string },
): TimelineModel {
  const format = options.formatDuration ?? ((ms: number) => formatDuration(ms));
  const range = response.range ?? {
    startMs: 0,
    turnStartMs: 0,
    endMs: 0,
    lastEventMs: 0,
    startExact: true,
    leadMs: 0,
  };
  const durationMs = Math.max(1, (range.endMs ?? range.lastEventMs) - range.startMs);
  const turnStartMs = Math.max(0, range.turnStartMs - range.startMs);
  const live = response.open && response.live;
  const mode: AgentTimelineMode =
    live && options.fresh
      ? { kind: 'live', originMs: range.startMs }
      : live
        ? { kind: 'stale', atMs: response.asOfMs - range.startMs, asOfText: formatClock(response.asOfMs, true) }
        : { kind: 'recorded' };

  const laneById = new Map(response.lanes.map((lane) => [lane.id, lane]));
  const lanes: AgentLane[] = response.lanes.map((lane) => {
    const extra: string[] = [];
    if (lane.role === 'subagent' && lane.subagents > 0) {
      extra.push(`+${formatCountNoun(lane.subagents, 'subagent', 'subagents')}`);
    }
    if (lane.carriedIn) extra.push('from an earlier turn');
    if (lane.carriedOut) extra.push('still running at the end');
    const outcome: AgentLane['outcome'] =
      lane.status === 'failed'
        ? { text: 'Failed', tone: 'danger' }
        : lane.status === 'cancelled'
          ? { text: 'Cancelled', tone: 'warning' }
          : undefined;
    return {
      id: lane.id,
      name: lane.name,
      ...(lane.model !== null ? { model: lane.model } : {}),
      ...(extra.length > 0 ? { extra: extra.join(' · ') } : {}),
      color: lane.slot === null ? 'var(--lane-main)' : `var(--lane-${String(lane.slot + 1)})`,
      ...(lane.role === 'subagent' && lane.sessionId !== null ? { to: sessionUrl(lane.sessionId) } : {}),
      ...(outcome !== undefined ? { outcome } : {}),
    };
  });

  const lastSpanOf = new Map<string, string>();
  for (const span of response.spans) lastSpanOf.set(span.lane, span.id);
  const spans: AgentSpan[] = response.spans.map((span) => {
    const lane = laneById.get(span.lane);
    const words = spanWords(span, lane);
    const open = openEdge(span);
    const cut = cutEdge(span);
    const notes = [...words.notes];
    if (span.kind !== 'run') {
      if (open === 'start' || open === 'both') notes.push(START_NOTE);
      if (open === 'end' || open === 'both') notes.push(END_NOTE);
    }
    if (cut === 'start' || cut === 'both') notes.push(CUT_START_NOTE);
    if (cut === 'end' || cut === 'both') notes.push(CUT_END_NOTE);
    if (
      lane !== undefined &&
      lane.role === 'subagent' &&
      !response.open &&
      lane.status === 'unknown' &&
      !lane.carriedOut &&
      lastSpanOf.get(lane.id) === span.id
    ) {
      notes.push(NO_RETURN_NOTE);
    }
    return {
      id: span.id,
      laneId: span.lane,
      kind: span.kind,
      startMs: span.startMs,
      endMs: span.endMs,
      status: span.status,
      label: words.label,
      ...(open !== undefined ? { open } : {}),
      ...(cut !== undefined ? { cut } : {}),
      ...(span.retry !== null
        ? {
            retry:
              span.retry.max > 0
                ? `Retry ${String(span.retry.attempt)} of ${String(span.retry.max)}`
                : `Retry ${String(span.retry.attempt)}`,
          }
        : {}),
      ...(words.facts.length > 0 ? { facts: words.facts } : {}),
      ...(notes.length > 0 ? { note: notes.join(' ') } : {}),
      ...((span.kind === 'wait' && span.reason === 'subagent') || span.kind === 'idle' ? { ownsHandoffs: true } : {}),
    };
  });

  const handoffs: AgentHandoff[] = response.handoffs.map((handoff) => ({
    id: handoff.id,
    laneId: handoff.lane,
    kind: handoff.kind,
    atMs: handoff.atMs,
    ...(handoff.failed ? { failed: true } : {}),
  }));
  const reports: AgentTokenReport[] = response.lanes.flatMap((lane) =>
    lane.tokenReports.map((report) => ({ laneId: lane.id, atMs: report.atMs, tokens: report.tokens })),
  );

  const sentences = ['Positions come from recorded event times. Empty stretches have no recorded event.'];
  if (spans.some((span) => span.open !== undefined)) {
    sentences.push('Faded ends were not recorded and are drawn to the nearest recorded event.');
  }
  if (turnStartMs > 0) {
    sentences.push(
      `A background event started this turn; no prompt was typed. The axis begins ${format(turnStartMs)} before the turn, where background work ran while the main agent was idle.`,
    );
  } else if (range.leadMs > 0) {
    sentences.push(
      `The axis starts at the prompt, ${format(range.leadMs)} before the start the agent measured for this turn.`,
    );
  }
  if (response.lanes.some((lane) => lane.clamped)) {
    sentences.push('Some subagent steps crossed their handoff times and are cut there.');
  }
  const notes = response.notes;
  if (notes.unplacedRetries > 0) {
    sentences.push(
      `${formatCountNoun(notes.unplacedRetries, 'model request retry is', 'model request retries are')} not shown on a span.`,
    );
  }
  if (notes.untimedEvents > 0) {
    sentences.push(
      `${formatCountNoun(notes.untimedEvents, 'event', 'events')} had no time and ${plural(notes.untimedEvents, 'is', 'are')} not drawn.`,
    );
  }
  if (notes.foldedSubagents > 0) sentences.push('Only the first 8 subagents have a lane.');
  if (notes.droppedSpans > 0) {
    sentences.push(`${formatCountNoun(notes.droppedSpans, 'very short span is', 'very short spans are')} left out.`);
  }
  const rate = replayRate(durationMs);
  if (mode.kind === 'recorded' && rate > 1) {
    sentences.push(`Replay runs ${String(rate)}× faster than ${turnStartMs > 0 ? 'real time' : 'the turn did'}.`);
  }

  return {
    lanes,
    spans,
    handoffs,
    reports,
    durationMs,
    turnStartMs,
    mode,
    outcome: outcomeOf(response),
    footnote: sentences.join(' '),
  };
}

/** Lanes the loading frame draws: the main agent, one per subagent the group spawned (at most 8), and the folded lane. */
export function expectedLanes(entries: readonly TranscriptEntry[]): number {
  const children = new Set<string>();
  for (const entry of entries) if (entry.kind === 'subagent') children.add(entry.childId);
  return 1 + Math.min(8, children.size) + (children.size > 8 ? 1 : 0);
}
