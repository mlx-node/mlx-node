/**
 * A turn woken by a background workflow, line for line in the order a real session file holds it (made-up text, measured times): turn 1
 * starts a background workflow; its 8 children run on after turn 1 ends; the workflow stops and the agent wakes itself
 * with a hidden prompt for turn 2.
 */

import {
  buildTimeline,
  type ChildEvents,
  digestUpdates,
  type TimelineChildInfo,
  timelineChildReads,
} from '../../src/agent-sessions/timeline.js';
import { parseUpdates } from '../../src/agent-sessions/transcript.js';
import type { SessionTimelineResponse } from '../../src/agent-sessions/types.js';
import { eventsText, type EventLine } from './agent-session-files.js';

export const WOKEN_T = 1_800_000_000_000;
export const WOKEN_PARENT = '01d20000-0000-7000-8000-00000000000a';
export const WOKEN_RUN = 'wf_1';
export const WOKEN_PROMPT_ID = `workflow-completed-${WOKEN_RUN}-30`;
export const HIDDEN_TEXT = 'A background workflow stopped. Look at its result and go on with the task.';
const TYPED_TEXT = 'Research the topic in depth';
const TOOL_INPUT = 'notes/research-report.md';
const TOOL_OUTPUT = 'The report says the plan holds.';
/** Every prose string the parent file writes: prompts, thoughts, answers, tool titles, tool input and output. */
export const WOKEN_PROSE: readonly string[] = [
  HIDDEN_TEXT,
  TYPED_TEXT,
  TOOL_INPUT,
  TOOL_OUTPUT,
  'Planning the work',
  'Reading the workflow result',
  'and the reports',
  'The research runs in the background.',
  'Here is the summary.',
  'workflow title',
  'read_file title',
];

const childId = (n: number): string => `01d20000-0000-7000-8000-0000000000${n.toString(16).padStart(2, '0')}`;

export interface WokenChild {
  id: string;
  subagentId: string;
  description: string;
  spawn: number;
  finish: number;
  /** `turn_completed` usage (input + output) of the child's own file. */
  tokens: number;
}

/** Offsets from WOKEN_T. */
export const WOKEN_CHILDREN: readonly WokenChild[] = [
  { id: childId(0x10), subagentId: 'sa-plan', description: 'Plan', spawn: 60_750, finish: 227_364, tokens: 40_000 },
  ...[
    [870_242, 122_388],
    [1_108_457, 362_504],
    [1_219_540, 334_434],
    [1_282_001, 257_210],
  ].map(([finish, tokens], i) => ({
    id: childId(0x11 + i),
    subagentId: `sa-research-${String(i + 1)}`,
    description: `Research ${String(i + 1)}`,
    spawn: 227_385,
    finish: finish!,
    tokens: tokens!,
  })),
  {
    id: childId(0x15),
    subagentId: 'sa-verify-a',
    description: 'Verify A',
    spawn: 1_282_037,
    finish: 2_079_280,
    tokens: 345_687,
  },
  {
    id: childId(0x16),
    subagentId: 'sa-verify-b',
    description: 'Verify B',
    spawn: 1_282_037,
    finish: 2_177_808,
    tokens: 236_534,
  },
  {
    id: childId(0x17),
    subagentId: 'sa-report',
    description: 'Report',
    spawn: 2_177_848,
    finish: 2_680_118,
    tokens: 39_230,
  },
];

const at = (offset: number): number => WOKEN_T + offset;
const xai = (offset: number, update: Record<string, unknown>): EventLine => ({ at: at(offset), xai: true, update });
const acp = (offset: number, update: Record<string, unknown>): EventLine => ({ at: at(offset), update });

const spawnLine = (child: WokenChild): EventLine =>
  xai(child.spawn, {
    sessionUpdate: 'subagent_spawned',
    subagent_id: child.subagentId,
    child_session_id: child.id,
    subagent_type: 'general',
    description: child.description,
    model: 'child-model',
    workflow_run_id: WOKEN_RUN,
  });
const finishLine = (child: WokenChild): EventLine =>
  xai(child.finish, {
    sessionUpdate: 'subagent_finished',
    subagent_id: child.subagentId,
    child_session_id: child.id,
    status: 'completed',
    duration_ms: child.finish - child.spawn,
    tokens_used: 9_000,
    will_wake: false,
  });
const workflowLine = (offset: number, status: string, revision: number): EventLine =>
  xai(offset, {
    sessionUpdate: 'workflow_updated',
    run_id: WOKEN_RUN,
    name: 'deep-research',
    status,
    revision,
    foreground: false,
    elapsed_ms: offset - 60_702,
  });
const thought = (offset: number, text: string): EventLine =>
  acp(offset, { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } });
const answer = (offset: number, text: string): EventLine =>
  acp(offset, { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } });
const tool = (
  offset: number,
  id: string,
  name: string,
  status: string,
  fields: Record<string, unknown> = {},
): EventLine =>
  acp(offset, {
    sessionUpdate: status === 'pending' ? 'tool_call' : 'tool_call_update',
    toolCallId: id,
    title: `${name} title`,
    status,
    _meta: { 'x.ai/tool': { name } },
    ...fields,
  });
const turnCompleted = (offset: number, elapsed: number, promptId: string, input: number, output: number): EventLine =>
  xai(offset, {
    sessionUpdate: 'turn_completed',
    prompt_id: promptId,
    stop_reason: 'end_turn',
    elapsed_ms: elapsed,
    usage: { inputTokens: input, outputTokens: output },
  });

const [plan, r1, r2, r3, r4, va, vb, report] = WOKEN_CHILDREN as [
  WokenChild,
  WokenChild,
  WokenChild,
  WokenChild,
  WokenChild,
  WokenChild,
  WokenChild,
  WokenChild,
];

export interface WokenOptions {
  /** A typed prompt in place of the hidden one. */
  typed?: boolean;
  /** Leave out the `user_prompt_submit` hook line. */
  noHook?: boolean;
  /** Stop after the hidden prompt and its first thought: turn 2 is open. */
  open?: boolean;
}

/** The parent's lines. */
export function wokenParentLines(options: WokenOptions = {}): EventLine[] {
  const lines: EventLine[] = [
    xai(-2, { sessionUpdate: 'hook_execution', event_name: 'user_prompt_submit', prompt_id: 'typed-1', runs: [] }),
    acp(0, {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: TYPED_TEXT },
      _meta: { modelId: 'main-model', promptIndex: 0 },
    }),
    thought(20_000, 'Planning the work'),
    tool(60_642, 'call-wf', 'workflow', 'pending'),
    workflowLine(60_702, 'active', 1),
    tool(60_705, 'call-wf', 'workflow', 'completed'),
    spawnLine(plan),
    finishLine(plan),
    spawnLine(r1),
    spawnLine(r2),
    spawnLine(r3),
    spawnLine(r4),
    answer(300_000, 'The research runs in the background.'),
    turnCompleted(649_557, 649_621, 'typed-1', 900_000, 4_000),
    finishLine(r1),
    finishLine(r2),
    finishLine(r3),
    finishLine(r4),
    workflowLine(1_282_025, 'active', 7),
    spawnLine(va),
    spawnLine(vb),
    finishLine(va),
    finishLine(vb),
    workflowLine(2_177_840, 'active', 25),
    spawnLine(report),
    finishLine(report),
    workflowLine(2_680_142, 'complete', 30),
  ];
  if (options.noHook !== true) {
    lines.push(
      xai(2_680_208, {
        sessionUpdate: 'hook_execution',
        event_name: 'user_prompt_submit',
        prompt_id: WOKEN_PROMPT_ID,
        runs: [],
      }),
    );
  }
  lines.push(
    acp(2_680_208, {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: options.typed === true ? 'Go on with the result' : HIDDEN_TEXT },
      _meta: { modelId: 'main-model', promptIndex: 1, ...(options.typed === true ? {} : { hideFromScrollback: true }) },
    }),
    thought(2_699_582, 'Reading the workflow result'),
    thought(2_710_000, ' and the reports'),
  );
  if (options.open === true) return lines;
  lines.push(
    tool(2_711_000, 'call-read', 'read_file', 'pending', { rawInput: { path: TOOL_INPUT } }),
    tool(2_712_000, 'call-read', 'read_file', 'completed', { rawOutput: TOOL_OUTPUT }),
    answer(2_720_000, 'Here is the summary.'),
    answer(2_930_000, ' Done.'),
    turnCompleted(2_932_482, 252_337, WOKEN_PROMPT_ID, 1_496_173, 5_968),
  );
  return lines;
}

/** A child's own lines: its prompt, one thought and its `turn_completed` with the measured usage. */
export function wokenChildLines(child: WokenChild): EventLine[] {
  return [
    acp(child.spawn + 10, {
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: `Do ${child.description}` },
      _meta: { modelId: 'child-model', promptIndex: 0 },
    }),
    thought(child.spawn + 20, 'Working'),
    answer(child.finish - 20, 'Finished'),
    turnCompleted(
      child.finish - 5,
      child.finish - 5 - child.spawn,
      `child-${child.subagentId}`,
      child.tokens - 100,
      100,
    ),
  ];
}

export function wokenParentText(options: WokenOptions = {}): string {
  return eventsText(WOKEN_PARENT, wokenParentLines(options));
}

export function wokenChildText(child: WokenChild): string {
  return eventsText(child.id, wokenChildLines(child));
}

/** The timeline route body of the woken session, built as the reader builds it. */
export function wokenTimeline(turn: number, options: WokenOptions = {}): SessionTimelineResponse {
  const parent = digestUpdates(parseUpdates(wokenParentText(options)));
  const children = new Map<string, TimelineChildInfo>();
  for (const child of WOKEN_CHILDREN) {
    children.set(child.id, {
      status: 'completed',
      type: 'general',
      description: child.description,
      model: 'child-model',
      exists: true,
      subagents: 0,
    });
  }
  const input = {
    turn,
    parent,
    live: false,
    asOfMs: WOKEN_T + 3_000_000,
    main: { name: 'Main agent', type: null, model: 'main-model', sessionId: WOKEN_PARENT },
    children,
  };
  const reads = timelineChildReads(input);
  if (reads === 'no-turn') throw new Error('no-turn');
  const events = new Map<string, ChildEvents>();
  for (const id of reads) {
    const child = WOKEN_CHILDREN.find((c) => c.id === id)!;
    events.set(id, { ok: true, source: digestUpdates(parseUpdates(wokenChildText(child))) });
  }
  const body = buildTimeline(input, events);
  if (body === 'no-turn') throw new Error('no-turn');
  return body;
}
