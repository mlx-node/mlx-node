/**
 * The live work tracker: what one conversation's agent process runs in the background, read from the agent's lines
 * in the shapes v0.3.4 sends and stores (helpers, commands and monitors, schedules), the wake an end line announces,
 * the `_x.ai/subagent/list_running` reply the idle guard reads, the helpers' live progress ticks, and each monitor's
 * events with the wake reports they make.
 */

import { describe, expect, it } from 'vite-plus/test';

import { readRunningSubagents, readSubagentProgress } from '../src/chat/acp/messages.js';
import {
  ActivityTracker,
  DOING_TITLE_CHARS,
  TASK_EVENT_CHARS,
  TASK_EVENTS_KEPT,
  WAKE_REPORTS_KEPT,
  wakeDueOf,
  type TrackedLine,
} from '../src/chat/agent/activity-tracker.js';

const xai = (update: Record<string, unknown>): TrackedLine => ({ xai: true, update });
const acp = (update: Record<string, unknown>): TrackedLine => ({ xai: false, update });

const TASK = '01a20df5-8562-75b2-aaf5-d3c1e2ec5273';
const CHILD = '01a1264a-4c3b-7cd3-833e-26c6f665d1db';

const backgrounded = (id: string, monitor?: string): TrackedLine =>
  xai({
    sessionUpdate: 'task_backgrounded',
    tool_call_id: `call-${id}`,
    task_id: id,
    command: 'npm run dev',
    cwd: '/code/app',
    output_file: `/tmp/${id}.log`,
    ...(monitor === undefined ? { description: 'Start the dev server' } : { monitor_description: monitor }),
  });

const completed = (id: string, wake = false): TrackedLine =>
  xai({
    sessionUpdate: 'task_completed',
    task_snapshot: { task_id: id, command: 'npm run dev', exit_code: 0, completed: true, kind: 'bash' },
    will_wake: wake,
  });

const list = (rows: Array<[string, string, string?]>, truncated = false): TrackedLine =>
  xai({
    sessionUpdate: 'background_tasks',
    tasks: rows.map(([id, status, kind]) => ({ task_id: id, command: 'x', cwd: '/', kind: kind ?? 'bash', status })),
    ...(truncated ? { truncated: true } : {}),
  });

const spawned = (child: string, run?: string): TrackedLine =>
  xai({
    sessionUpdate: 'subagent_spawned',
    subagent_id: child,
    parent_session_id: 'parent',
    child_session_id: child,
    subagent_type: 'general-purpose',
    description: 'Find the flaky test',
    ...(run === undefined ? {} : { workflow_run_id: run }),
  });

const finished = (child: string, wake = false): TrackedLine =>
  xai({
    sessionUpdate: 'subagent_finished',
    subagent_id: child,
    child_session_id: child,
    status: 'completed',
    tool_calls: 3,
    turns: 2,
    duration_ms: 900,
    tokens_used: 1200,
    will_wake: wake,
  });

const schedule = (kind: string, id: string, reason?: string): TrackedLine =>
  xai({
    sessionUpdate: kind,
    task_id: id,
    prompt: 'Check the deploy',
    human_schedule: 'every 5 minutes',
    next_fire_at: null,
    ...(reason === undefined ? {} : { reason }),
  });

function counts(t: ActivityTracker): [number, number, number] {
  return [t.helperCount(), t.taskCount(), t.scheduleCount()];
}

describe('commands and monitors', () => {
  it('a task is live from its start line to its end line; a start read after the end does not bring it back', () => {
    const t = new ActivityTracker();
    expect(t.apply(backgrounded(TASK))).toBe(true);
    expect(t.apply(backgrounded(TASK))).toBe(false);
    expect(t.taskCount()).toBe(1);
    expect(t.apply(completed(TASK))).toBe(true);
    expect(t.taskCount()).toBe(0);
    expect(t.apply(backgrounded(TASK))).toBe(false);
    expect(t.apply(completed(TASK))).toBe(false);
    expect(t.taskCount()).toBe(0);
  });

  it('the tool result starts a task too: BackgroundTaskStarted is a command, Monitor a monitor', () => {
    const t = new ActivityTracker();
    const result = (rawOutput: Record<string, unknown>): TrackedLine =>
      acp({ sessionUpdate: 'tool_call_update', toolCallId: 'c1', status: 'completed', rawOutput });
    expect(t.apply(result({ type: 'BackgroundTaskStarted', task_id: 'a', task_type: 'bash', status: 'running' }))).toBe(
      true,
    );
    expect(t.apply(result({ type: 'Monitor', taskId: 'b', timeoutMs: 36_000_000, persistent: false }))).toBe(true);
    expect(t.apply(result({ type: 'Bash', exit_code: 0 }))).toBe(false);
    expect(t.apply(result({ type: 'Monitor', taskId: '' }))).toBe(false);
    expect(t.taskCount()).toBe(2);
    // The same kinds on a stored xAI line or a bare ACP line of another kind change nothing.
    expect(t.apply(acp({ sessionUpdate: 'task_backgrounded', task_id: 'c' }))).toBe(false);
    expect(t.taskCount()).toBe(2);
  });

  it('a monitor read as a command first stays one task when its tool result says monitor', () => {
    const t = new ActivityTracker();
    expect(t.apply(backgrounded('m'))).toBe(true);
    expect(
      t.apply(acp({ sessionUpdate: 'tool_call_update', toolCallId: 'c', rawOutput: { type: 'Monitor', taskId: 'm' } })),
    ).toBe(false);
    expect(t.apply(backgrounded('n', 'Watch the build log'))).toBe(true);
    expect(t.taskCount()).toBe(2);
  });

  it('background_tasks: a running row is live, an ended row ends it, an uncut list without a task ends it', () => {
    const t = new ActivityTracker();
    expect(t.apply(list([['a', 'running']]))).toBe(true);
    expect(t.apply(backgrounded('b'))).toBe(true);
    expect(
      t.apply(
        list([
          ['a', 'running'],
          ['b', 'running', 'monitor'],
        ]),
      ),
    ).toBe(false);
    expect(t.apply(list([['a', 'failed']], true))).toBe(true);
    expect(t.taskCount()).toBe(1);
    // A cut list proves nothing about the tasks it leaves out.
    expect(t.apply(list([], true))).toBe(false);
    expect(t.taskCount()).toBe(1);
    expect(t.apply(list([]))).toBe(true);
    expect(t.taskCount()).toBe(0);
  });

  it('a list read before a start can end it; the next list that says running puts it back', () => {
    const t = new ActivityTracker();
    expect(t.apply(backgrounded(TASK))).toBe(true);
    expect(t.apply(list([]))).toBe(true);
    expect(t.taskCount()).toBe(0);
    // Its own start lines stay ended; only the agent's list brings it back.
    expect(t.apply(backgrounded(TASK))).toBe(false);
    expect(t.apply(list([[TASK, 'running']]))).toBe(true);
    expect(t.taskCount()).toBe(1);
    expect(t.apply(list([[TASK, 'completed']]))).toBe(true);
    expect(t.taskCount()).toBe(0);
  });
});

describe('a Stop of a command or monitor (T1)', () => {
  // Mutation: leave `stopping` out of `live()` → the row never reads Stopping….
  it('the live facts list every task in start order; Stop marks it stopping until its end line', () => {
    const t = new ActivityTracker();
    t.apply(backgrounded('a'));
    t.apply(backgrounded('b', 'Watch the build log'));
    expect(t.live().tasks).toEqual([
      { id: 'a', kind: 'command', stopping: false, events: 0, recent: [] },
      { id: 'b', kind: 'monitor', stopping: false, events: 0, recent: [] },
    ]);
    const before = t.tasksVersion();
    expect(t.setStopping('b', true)).toBe(true);
    expect(t.setStopping('b', true)).toBe(false);
    expect(t.tasksVersion()).toBe(before + 1);
    expect(t.live().tasks.map((task) => [task.id, task.stopping])).toEqual([
      ['a', false],
      ['b', true],
    ]);
    // A Stop that failed gives the task back.
    expect(t.setStopping('b', false)).toBe(true);
    expect(t.setStopping('b', true)).toBe(true);
    expect(t.apply(completed('b'))).toBe(true);
    expect(t.live().tasks).toEqual([{ id: 'a', kind: 'command', stopping: false, events: 0, recent: [] }]);
    expect(t.hasTask('b')).toBe(false);
    // A task the tracker does not hold is never marked.
    expect(t.setStopping('b', true)).toBe(false);
    expect(t.setStopping('nope', true)).toBe(false);
  });

  it('a task the agent no longer knows is dropped and stays ended; the version moves on every task change', () => {
    const t = new ActivityTracker();
    const v0 = t.tasksVersion();
    t.apply(backgrounded('a'));
    const v1 = t.tasksVersion();
    expect(v1).toBeGreaterThan(v0);
    expect(t.dropTask('a')).toBe(true);
    expect(t.dropTask('a')).toBe(false);
    expect(t.tasksVersion()).toBeGreaterThan(v1);
    expect([t.taskCount(), t.live().tasks]).toEqual([0, []]);
    expect(t.apply(backgrounded('a'))).toBe(false);
    // A helper or a schedule moves no task version.
    const v2 = t.tasksVersion();
    t.apply(spawned(CHILD));
    t.apply(schedule('scheduled_task_created', 's1'));
    expect(t.tasksVersion()).toBe(v2);
  });
});

describe('helpers', () => {
  it('a helper is live from subagent_spawned to subagent_finished; a workflow helper is engine work, not the conversation’s own', () => {
    const t = new ActivityTracker();
    expect(t.apply(spawned(CHILD))).toBe(true);
    expect(t.apply(spawned(CHILD))).toBe(false);
    expect(t.apply(spawned('wf-child', 'wf_1'))).toBe(true);
    expect([t.helperCount(), t.ownHelperCount()]).toEqual([2, 1]);
    expect(t.apply(finished(CHILD))).toBe(true);
    expect(t.apply(finished(CHILD))).toBe(false);
    expect([t.helperCount(), t.ownHelperCount()]).toEqual([1, 0]);
  });

  it('the guard keeps a helper list_running names by child session or subagent id and drops the rest', () => {
    const t = new ActivityTracker();
    t.apply(spawned('a'));
    t.apply(xai({ sessionUpdate: 'subagent_spawned', subagent_id: 'sub-b', child_session_id: 'b' }));
    t.apply(spawned('c'));
    expect(t.keepListed(new Set(['a', 'sub-b']), t.helperMark())).toBe(true);
    expect(t.helperCount()).toBe(2);
    expect(t.keepListed(new Set(['a', 'sub-b']), t.helperMark())).toBe(false);
    expect(t.keepListed(new Set(), t.helperMark())).toBe(true);
    expect(t.helperCount()).toBe(0);
  });

  // Mutation: drop a run's agents as any other (list_running never names them) → a live run's agents lose their Stop.
  it("the guard keeps a run's agents while the host holds the run (live, or draining a stop); other unlisted ones go", () => {
    const t = new ActivityTracker();
    t.apply(spawned('w1', 'wf_1'));
    t.apply(spawned('w2', 'wf_2'));
    t.apply(spawned('own'));
    expect(t.keepListed(new Set(), t.helperMark(), new Set(['wf_1']))).toBe(true);
    expect(t.live().agents.map((a) => a.childId)).toEqual(['w1']);
  });

  it('the guard never drops a helper that started after its request went: the agent read its list before it', () => {
    const t = new ActivityTracker();
    t.apply(spawned('a'));
    const asked = t.helperMark();
    t.apply(spawned('b'));
    expect(t.keepListed(new Set(), asked)).toBe(true);
    expect(t.helperCount()).toBe(1);
    expect(t.keepListed(new Set(), asked)).toBe(false);
    expect(t.keepListed(new Set(), t.helperMark())).toBe(true);
    expect(t.helperCount()).toBe(0);
    t.apply(spawned('a'));
    expect(t.keepListed(new Set(), asked)).toBe(false);
    expect(t.helperCount()).toBe(1);
  });
});

describe('agent controls: Stop and Message', () => {
  const ADDRESS = 'agent-address-opaque-1';
  const live = (child: string, address?: string, run?: string): TrackedLine =>
    xai({
      ...spawned(child, run).update,
      ...(address === undefined ? {} : { agentAddress: address }),
    });

  // Mutation: list only the conversation's own helpers → a run's agent gets no row, so no Stop.
  // Mutation: copy the address into `live` → it reaches the page.
  // Mutation: read `agent_address` (the stored alias) → every live row reads no-address.
  it('every agent is listed from start to end, ticked or not, in start order, with its run and what Message can do; never the address', () => {
    const t = new ActivityTracker();
    t.apply(live('a', ADDRESS));
    t.apply(live('wf', undefined, 'wf_1'));
    t.apply(live('b'));
    expect(t.live().agents).toEqual([
      { childId: 'a', run: null, attempts: ['a'], stopping: false, messages: 'on' },
      { childId: 'wf', run: 'wf_1', attempts: ['wf'], stopping: false, messages: 'no-address' },
      { childId: 'b', run: null, attempts: ['b'], stopping: false, messages: 'no-address' },
    ]);
    expect(JSON.stringify(t.live())).not.toContain('agent-address');
    expect(t.agent('a')).toEqual({ childId: 'a', subagentId: 'a', run: null, address: ADDRESS, stopping: false });
    expect(t.agent('stranger')).toBeNull();
    // The process has no message method: every row says so.
    expect(t.live(new Map(), true).agents.map((s) => s.messages)).toEqual(['off', 'off', 'off']);
    t.apply(finished('a'));
    expect(t.live().agents.map((s) => s.childId)).toEqual(['wf', 'b']);
    expect(t.agent('a')).toBeNull();
  });

  // Mutation: `agent` refuses a run's agent again → its Stop is not-live, nothing is sent.
  // Mutation: `setHelperStopping` refuses a run's agent again → its row never reads Stopping….
  it("a run's agent takes Stop: it is found, reads stopping, and a stop the agent took is the host's (T30)", () => {
    const t = new ActivityTracker();
    t.apply(live('wf', undefined, 'wf_1'));
    expect(t.agent('wf')).toEqual({ childId: 'wf', subagentId: 'wf', run: 'wf_1', address: null, stopping: false });
    const v0 = t.agentsVersion();
    expect(t.setHelperStopping('wf', true)).toBe(true);
    expect(t.agentsVersion()).not.toBe(v0);
    expect(t.live().agents).toEqual([
      { childId: 'wf', run: 'wf_1', attempts: ['wf'], stopping: true, messages: 'no-address' },
    ]);
    expect(t.agent('wf')?.stopping).toBe(true);
    expect(t.stopTaken('wf')).toBe(true);
    expect(t.stopTaken('wf')).toBe(false);
    t.apply(finished('wf'));
    expect(t.live().agents).toEqual([]);
    expect(t.live().stoppedByYou).toEqual(['wf']);
    // The process exiting ends its agents, not what this host did.
    t.clear();
    expect(t.live().stoppedByYou).toEqual(['wf']);
  });

  // Mutation: no version move on a start → trackWork sends the page nothing until the first tick (no Stop meanwhile).
  it('the version moves on every change of the list; dropHelper takes one out', () => {
    const t = new ActivityTracker();
    const v0 = t.agentsVersion();
    t.apply(live('a', ADDRESS));
    const v1 = t.agentsVersion();
    expect(v1).not.toBe(v0);
    t.apply(live('wf', undefined, 'wf_1'));
    const v2 = t.agentsVersion();
    expect(v2).not.toBe(v1);
    expect(t.setHelperStopping('a', true)).toBe(true);
    expect(t.setHelperStopping('a', true)).toBe(false);
    expect(t.setHelperStopping('stranger', true)).toBe(false);
    expect(t.agentsVersion()).not.toBe(v2);
    expect(t.setHelperStopping('a', false)).toBe(true);
    expect(t.dropHelper('a', 'a')).toBe(true);
    expect(t.dropHelper('a', 'a')).toBe(false);
    expect([t.live().agents.map((a) => a.childId), t.ownHelperCount(), t.helperCount()]).toEqual([['wf'], 0, 1]);
    const v3 = t.agentsVersion();
    t.apply(finished('wf'));
    expect(t.agentsVersion()).not.toBe(v3);
    t.apply(live('c', ADDRESS));
    const v4 = t.agentsVersion();
    t.clear();
    expect(t.agentsVersion()).not.toBe(v4);
    expect(t.live().agents).toEqual([]);
  });

  // Mutation: keep the helper on a drop → it holds the agent process until the idle guard asks.
  it('a dropped helper ends as an end line ends it: a schedule whose fire started it no longer fires', () => {
    const t = new ActivityTracker();
    t.apply(xai({ sessionUpdate: 'scheduled_task_fired', task_id: 's1', subagent_id: 'run-1', next_fire_at: null }));
    t.apply(live('run-1', ADDRESS));
    expect(t.live().schedules[0]?.firing).toBe('run-1');
    expect(t.dropHelper('run-1', 'run-1')).toBe(true);
    expect([t.helperCount(), t.live().schedules[0]?.firing]).toEqual([0, null]);
  });
});

describe("a workflow agent's contract retry", () => {
  const RUN = 'wf_01a2185639d376b2a1c4a4f482ca7fe1';
  const spawn = (child: string, extra: Record<string, unknown> = {}): TrackedLine =>
    xai({ ...spawned(child, RUN).update, description: 'track:process', ...extra });

  const retry = (child: string, of: string): TrackedLine => spawn(child, { resumed_from: of, retry_of: of });

  // Mutation: no alias → a Stop sent with the first attempt's id finds no agent (not-live), and the retry runs on.
  it("takes the row: both attempts are listed, the first attempt's id reaches the retry, until it ends", () => {
    const t = new ActivityTracker();
    t.apply(spawn('a1'));
    t.apply(finished('a1'));
    expect(t.agent('a1')).toBeNull();
    t.apply(retry('a2', 'a1'));
    expect(t.live().agents).toEqual([
      { childId: 'a2', run: RUN, attempts: ['a1', 'a2'], stopping: false, messages: 'no-address' },
    ]);
    expect(t.agent('a1')?.childId).toBe('a2');
    expect(t.setHelperStopping('a1', true)).toBe(true);
    expect(t.agent('a2')?.stopping).toBe(true);
    expect(t.stopTaken('a1')).toBe(true);
    expect(t.live().stoppedByYou).toEqual(['a2']);
    t.apply(finished('a2'));
    expect([t.agent('a1'), t.agent('a2')]).toEqual([null, null]);
  });

  // Mutation: a drop keeps no ended agent → the retry is a row of its own, and the first attempt's id finds nothing.
  it('a Stop answered as already finished before the finish line still lets the retry take the row', () => {
    const t = new ActivityTracker();
    t.apply(spawn('a1'));
    expect(t.dropHelper('a1', 'a1')).toBe(true);
    expect(t.apply(finished('a1'))).toBe(false);
    t.apply(retry('a2', 'a1'));
    expect(t.agent('a1')?.childId).toBe('a2');
    expect([t.attempts('a1'), t.attempts('a2')]).toEqual([
      ['a1', 'a2'],
      ['a1', 'a2'],
    ]);
  });

  // Mutation: fold on `resumed_from` → the script's call with the same label takes the row of the agent it resumed.
  it("a script's resume of an ended agent, also under its own label, is its own row; the retry after it still folds", () => {
    const t = new ActivityTracker();
    t.apply(spawn('a1'));
    t.apply(finished('a1'));
    t.apply(spawn('c', { resumed_from: 'a1' }));
    t.apply(spawn('e', { resumed_from: 'a1', description: 'track:process-follow-up' }));
    t.apply(retry('a2', 'a1'));
    expect(t.live().agents.map((a) => a.attempts)).toEqual([['c'], ['e'], ['a1', 'a2']]);
  });

  // Mutation: keep the ended agent once a retry took it → a second `retry_of` of the first attempt joins the row.
  it("a `retry_of` of an attempt that is no longer its row's newest is its own row", () => {
    const t = new ActivityTracker();
    t.apply(spawn('a1'));
    t.apply(finished('a1'));
    t.apply(retry('a2', 'a1'));
    t.apply(finished('a2'));
    t.apply(retry('d', 'a1'));
    expect(t.live().agents.map((a) => a.attempts)).toEqual([['d']]);
    expect(t.attempts('a1')).toEqual(['a1', 'a2']);
  });

  // Mutation: keep an ended agent's newest attempt alone → attempt 1's id gives attempt 1 alone once both ended.
  it("names a row's attempts by any attempt's id, while it runs and after it ended", () => {
    const t = new ActivityTracker();
    t.apply(spawn('a1'));
    expect(t.attempts('a1')).toEqual(['a1']);
    t.apply(finished('a1'));
    t.apply(retry('a2', 'a1'));
    expect([t.attempts('a1'), t.attempts('a2')]).toEqual([
      ['a1', 'a2'],
      ['a1', 'a2'],
    ]);
    t.apply(finished('a2'));
    expect([t.attempts('a1'), t.attempts('a2'), t.attempts('other')]).toEqual([['a1', 'a2'], ['a1', 'a2'], ['other']]);
  });

  it('a resumed Task subagent and a spawn from an agent that still runs are agents of their own', () => {
    const t = new ActivityTracker();
    t.apply(xai({ ...spawned('t1').update }));
    t.apply(finished('t1'));
    t.apply(xai({ ...spawned('t2').update, resumed_from: 't1', retry_of: 't1' }));
    t.apply(spawn('w1'));
    t.apply(retry('w2', 'w1'));
    expect(t.live().agents.map((a) => a.attempts)).toEqual([['t2'], ['w1'], ['w2']]);
  });
});

describe('what an agent does now (T31)', () => {
  const tickOf = (child: string) =>
    readSubagentProgress({
      sessionUpdate: 'subagent_progress',
      subagent_id: child,
      child_session_id: child,
      duration_ms: 2_133,
      turn_count: 1,
      tool_call_count: 0,
      tokens_used: 1_105,
      context_window_tokens: 131_072,
      context_usage_pct: 0,
      error_count: 0,
    })!;
  const line = (sessionUpdate: string, fields: Record<string, unknown> = {}) => ({ sessionUpdate, ...fields });
  const nowOf = (t: ActivityTracker) => t.live().helpers[0]?.now;

  // Mutation: read `now` from the ticks only → it stays null: every running row reads Starting.
  it('follows its own thought, message, tool and compaction lines; the newest open call names it', () => {
    const t = new ActivityTracker();
    t.apply(spawned(CHILD, 'wf_1'));
    // Before its first tick the page shows nothing of it, so nothing is sent.
    expect(t.childLine(CHILD, line('agent_thought_chunk', { content: { type: 'text', text: 'Hmm' } }))).toBe(false);
    t.progress(tickOf(CHILD), 1);
    expect(nowOf(t)).toEqual({ kind: 'thinking' });
    expect(t.live().helpers[0]?.contextWindow).toBe(131_072);
    expect(t.childLine(CHILD, line('agent_message_chunk', { content: { type: 'text', text: 'alpha' } }))).toBe(true);
    expect(nowOf(t)).toEqual({ kind: 'writing' });
    expect(t.childLine(CHILD, line('agent_message_chunk', { content: { type: 'text', text: ' piece' } }))).toBe(false);
    const grep = 'grep -rn "External" crates/';
    expect(
      t.childLine(CHILD, line('tool_call', { toolCallId: 't1', title: `${grep}\n| head`, status: 'pending' })),
    ).toBe(true);
    expect(nowOf(t)).toEqual({ kind: 'tool', title: grep });
    t.childLine(CHILD, line('tool_call', { toolCallId: 't2', title: 'read_file', status: 'pending' }));
    t.childLine(CHILD, line('tool_call_update', { toolCallId: 't2', title: 'Read crates/napi/src/lib.rs' }));
    expect(nowOf(t)).toEqual({ kind: 'tool', title: 'Read crates/napi/src/lib.rs' });
    t.childLine(CHILD, line('tool_call_update', { toolCallId: 't2', status: 'completed' }));
    expect(nowOf(t)).toEqual({ kind: 'tool', title: grep });
    t.childLine(CHILD, line('tool_call_update', { toolCallId: 't1', status: 'failed' }));
    // The model takes the results.
    expect(nowOf(t)).toEqual({ kind: 'thinking' });
    expect(t.childLine(CHILD, line('auto_compact_started', { tokens_used: 1, context_window: 2 }))).toBe(true);
    expect(nowOf(t)).toEqual({ kind: 'compacting' });
    t.childLine(CHILD, line('auto_compact_completed'));
    expect(nowOf(t)).toEqual({ kind: 'thinking' });
    const long = 'x'.repeat(DOING_TITLE_CHARS + 30);
    t.childLine(CHILD, line('tool_call', { toolCallId: 't3', title: long }));
    expect(nowOf(t)).toEqual({ kind: 'tool', title: `${'x'.repeat(DOING_TITLE_CHARS)}…` });
    t.childLine(CHILD, line('turn_completed', { prompt_id: 'p1' }));
    expect(nowOf(t)).toEqual({ kind: 'thinking' });
    // Another session's lines (a nested agent's, or one this process does not run) change nothing.
    expect(t.childLine('stranger', line('agent_message_chunk'))).toBe(false);
    expect(t.childLine(CHILD, line('user_message_chunk'))).toBe(false);
  });
});

describe('schedules', () => {
  it('created or fired makes one live; a delete ends it, except the process stopping (`shutdown`)', () => {
    const t = new ActivityTracker();
    expect(t.apply(schedule('scheduled_task_created', 's1'))).toBe(true);
    expect(t.apply(schedule('scheduled_task_fired', 's1'))).toBe(false);
    expect(t.apply(schedule('scheduled_task_fired', 's2'))).toBe(true);
    expect(t.apply(schedule('scheduled_task_deleted', 's1', 'shutdown'))).toBe(false);
    expect(t.scheduleCount()).toBe(2);
    expect(t.apply(schedule('scheduled_task_deleted', 's1', 'expired'))).toBe(true);
    expect(t.apply(schedule('scheduled_task_deleted', 's2', 'deleted'))).toBe(true);
    expect(t.scheduleCount()).toBe(0);
  });

  /** `scheduled_task_fired` as the agent sends it live (`notification_bridge.rs`): the run is its own child session. */
  const fired = (id: string, run: string): TrackedLine =>
    xai({
      sessionUpdate: 'scheduled_task_fired',
      task_id: id,
      prompt: 'Check the deploy',
      human_schedule: 'every 5 minutes',
      next_fire_at: '2026-11-25T14:10:00Z',
      subagent_id: run,
    });
  const schedulesOf = (t: ActivityTracker) => t.live().schedules;
  const NEXT_TS = Date.parse('2026-11-25T14:10:00Z');

  // Mutation: keep `nextTs` off the live schedule → after a fire the rail shows the time the schedule was made with.
  // Mutation: keep a fire's `nextTs` past a created line → after an update the rail shows the fire's old next time.
  it('a fire names the next time; a later fire moves it; a created line (an update) hands it back to the entry', () => {
    const t = new ActivityTracker();
    t.apply(schedule('scheduled_task_created', 's1'));
    expect(schedulesOf(t)).toEqual([{ id: 's1', deleting: null, firing: null }]);
    const version = t.schedulesVersion();
    t.apply(fired('s1', 'run-1'));
    expect(schedulesOf(t)[0]!.nextTs).toBe(NEXT_TS);
    t.apply(
      xai({
        sessionUpdate: 'scheduled_task_fired',
        task_id: 's1',
        prompt: 'Check the deploy',
        human_schedule: 'every 5 minutes',
        next_fire_at: '2026-11-25T14:15:00Z',
        subagent_id: 'run-1',
      }),
    );
    expect(schedulesOf(t)[0]!.nextTs).toBe(Date.parse('2026-11-25T14:15:00Z'));
    expect(t.schedulesVersion()).toBe(version + 2);
    t.apply(schedule('scheduled_task_created', 's1'));
    expect(schedulesOf(t)).toEqual([{ id: 's1', deleting: null, firing: 'run-1' }]);
    expect(t.schedulesVersion()).toBe(version + 3);
    t.apply(schedule('scheduled_task_created', 's1'));
    expect(t.schedulesVersion()).toBe(version + 3);
  });

  // Mutation: clear `firing` only by child session id → a run the idle guard dropped keeps "Running now" for ever.
  it('a fire sets firing to its run until that helper ends, by its end line or the idle guard', () => {
    const t = new ActivityTracker();
    t.apply(schedule('scheduled_task_created', 's1'));
    const version = t.schedulesVersion();
    t.apply(spawned('run-1'));
    expect(t.apply(fired('s1', 'run-1'))).toBe(false);
    expect(t.schedulesVersion()).toBe(version + 1);
    expect(schedulesOf(t)).toEqual([{ id: 's1', deleting: null, firing: 'run-1', nextTs: NEXT_TS }]);
    // The agent re-announces the schedule with its next time: the run still runs.
    t.apply(schedule('scheduled_task_created', 's1'));
    expect(schedulesOf(t)[0]!.firing).toBe('run-1');
    expect(t.apply(finished('run-1'))).toBe(true);
    expect(schedulesOf(t)).toEqual([{ id: 's1', deleting: null, firing: null }]);
    expect(t.schedulesVersion()).toBe(version + 3);

    t.apply(xai({ sessionUpdate: 'subagent_spawned', subagent_id: 'run-2', child_session_id: 'run-2' }));
    t.apply(fired('s1', 'run-2'));
    expect(schedulesOf(t)[0]!.firing).toBe('run-2');
    expect(t.keepListed(new Set(), t.helperMark())).toBe(true);
    expect(schedulesOf(t)[0]!.firing).toBeNull();
  });

  // Mutation: set `firing` with no look at the helpers that ended → a fire read after its run ended reads "Running now".
  it('a fire read after its run ended does not fire; a fire first makes the schedule live', () => {
    const t = new ActivityTracker();
    t.apply(spawned('run-1'));
    t.apply(finished('run-1'));
    expect(t.apply(fired('s1', 'run-1'))).toBe(true);
    expect(schedulesOf(t)).toEqual([{ id: 's1', deleting: null, firing: null, nextTs: NEXT_TS }]);
    expect(t.apply(fired('s2', 'run-2'))).toBe(true);
    expect(schedulesOf(t).map((s) => s.firing)).toEqual([null, 'run-2']);
  });

  // Mutation: leave a Delete's phase out of `live()` for a schedule the process does not hold → a paused schedule being
  // loaded reads "Paused", not "Opening…".
  it("live names each Delete's phase, also for a schedule the process does not hold yet; dropSchedule takes one out", () => {
    const t = new ActivityTracker();
    t.apply(schedule('scheduled_task_created', 's1'));
    t.apply(schedule('scheduled_task_created', 's2'));
    expect(
      t.live(
        new Map([
          ['s2', 'removing'],
          ['cold', 'opening'],
        ]),
      ).schedules,
    ).toEqual([
      { id: 's1', deleting: null, firing: null },
      { id: 's2', deleting: 'removing', firing: null },
      { id: 'cold', deleting: 'opening', firing: null },
    ]);
    const version = t.schedulesVersion();
    expect(t.hasSchedule('s2')).toBe(true);
    expect(t.dropSchedule('s2')).toBe(true);
    expect(t.dropSchedule('s2')).toBe(false);
    expect(t.hasSchedule('s2')).toBe(false);
    expect(t.scheduleCount()).toBe(1);
    expect(t.schedulesVersion()).toBe(version + 1);
    expect(t.apply(schedule('scheduled_task_deleted', 's1', 'deleted'))).toBe(true);
    expect(t.schedulesVersion()).toBe(version + 2);
  });

  it('clear forgets everything (the process exited)', () => {
    const t = new ActivityTracker();
    t.apply(spawned('a'));
    t.apply(backgrounded('b'));
    t.apply(schedule('scheduled_task_created', 'c'));
    expect(counts(t)).toEqual([1, 1, 1]);
    t.clear();
    expect(counts(t)).toEqual([0, 0, 0]);
    // A task the old process ended may be started again by a new process.
    expect(t.apply(backgrounded('b'))).toBe(true);
  });

  it('lines of other kinds and bad ids change nothing', () => {
    const t = new ActivityTracker();
    for (const line of [
      xai({ sessionUpdate: 'turn_completed', prompt_id: 'p' }),
      xai({ sessionUpdate: 'subagent_spawned', child_session_id: '' }),
      xai({ sessionUpdate: 'task_completed', task_snapshot: null }),
      xai({ sessionUpdate: 'background_tasks', tasks: 'none' }),
      xai({ sessionUpdate: 'scheduled_task_created', task_id: 7 }),
      acp({ sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'hi' } }),
    ]) {
      expect(t.apply(line)).toBe(false);
    }
    expect(counts(t)).toEqual([0, 0, 0]);
  });
});

describe('the wake an end line announces', () => {
  it('task_completed and subagent_finished with will_wake; nothing without it', () => {
    expect(wakeDueOf(completed(TASK, true))).toEqual({ kind: 'task', taskId: TASK });
    expect(wakeDueOf(finished(CHILD, true))).toEqual({ kind: 'subagent', subagentId: CHILD });
    expect(wakeDueOf(completed(TASK))).toBeNull();
    expect(wakeDueOf(finished(CHILD))).toBeNull();
    expect(wakeDueOf({ xai: false, update: completed(TASK, true).update })).toBeNull();
    expect(wakeDueOf(xai({ sessionUpdate: 'workflow_updated', will_wake: true }))).toBeNull();
  });
});

describe('the list_running reply', () => {
  it('reads every helper’s child session and subagent id; any other shape is null', () => {
    const reply = {
      result: {
        subagents: [
          { subagentId: 'sub-a', parentSessionId: 'p', childSessionId: 'a', turnCount: 1 },
          { subagentId: 'b', childSessionId: 'b' },
        ],
      },
    };
    expect(readRunningSubagents(reply)).toEqual(new Set(['a', 'sub-a', 'b']));
    expect(readRunningSubagents({ result: { subagents: [] } })).toEqual(new Set());
    expect(readRunningSubagents({ result: null, error: 'session not found' })).toBeNull();
    expect(readRunningSubagents({ subagents: [] })).toBeNull();
    expect(readRunningSubagents({ result: { subagents: ['a'] } })).toBeNull();
    expect(readRunningSubagents(null)).toBeNull();
  });
});

describe('helper progress ticks', () => {
  /** grok `SessionUpdate::SubagentProgress` (`extensions/notification.rs`), as the publisher sends it. */
  const tick = (child: string, over: Record<string, unknown> = {}): Record<string, unknown> => ({
    sessionUpdate: 'subagent_progress',
    subagent_id: child,
    parent_session_id: 'parent',
    child_session_id: child,
    duration_ms: 4_100,
    turn_count: 2,
    tool_call_count: 12,
    tokens_used: 18_200,
    context_window_tokens: 44_000,
    context_usage_pct: 41,
    tools_used: ['read_file', 'grep'],
    error_count: 0,
    ...over,
  });

  it('reads every count; a percent past 100 is clamped; a bad count or no child is no tick', () => {
    expect(readSubagentProgress(tick(CHILD))).toEqual({
      childId: CHILD,
      turns: 2,
      toolCalls: 12,
      tokens: 18_200,
      contextPct: 41,
      errors: 0,
      durationMs: 4_100,
      contextWindow: 44_000,
    });
    expect(readSubagentProgress(tick(CHILD, { context_usage_pct: 250 }))?.contextPct).toBe(100);
    // The window is the agent's model's: a tick with none still counts.
    expect(readSubagentProgress(tick(CHILD, { context_window_tokens: undefined }))?.contextWindow).toBeNull();
    for (const bad of [
      { tool_call_count: -1 },
      { tokens_used: 1.5 },
      { turn_count: '2' },
      { error_count: null },
      { context_usage_pct: Number.NaN },
      { duration_ms: undefined },
      { child_session_id: '' },
      { child_session_id: 7 },
      { sessionUpdate: 'subagent_finished' },
    ]) {
      expect(readSubagentProgress(tick(CHILD, bad))).toBeNull();
    }
  });

  // Mutation: keep a tick for any child id → the stranger is listed.
  it('a tick of a known helper is kept with the host time; one for an unknown child is dropped', () => {
    const t = new ActivityTracker();
    t.apply(spawned(CHILD));
    t.apply(spawned('quiet'));
    expect(t.live().helpers).toEqual([]);
    expect(t.progress(readSubagentProgress(tick(CHILD))!, 500)).toBe(true);
    expect(t.progress(readSubagentProgress(tick('stranger'))!, 600)).toBe(false);
    expect(t.live().helpers).toEqual([
      {
        childId: CHILD,
        turns: 2,
        toolCalls: 12,
        tokens: 18_200,
        contextPct: 41,
        errors: 0,
        durationMs: 4_100,
        contextWindow: 44_000,
        at: 500,
        now: null,
      },
    ]);
    expect(t.tickedCount()).toBe(1);
    // A later tick replaces the numbers; a heartbeat with the same numbers moves the time.
    expect(t.progress(readSubagentProgress(tick(CHILD, { tool_call_count: 13 }))!, 700)).toBe(true);
    expect(t.progress(readSubagentProgress(tick(CHILD, { tool_call_count: 13 }))!, 900)).toBe(true);
    expect(t.live().helpers.map((h) => [h.toolCalls, h.at])).toEqual([[13, 900]]);
  });

  it('subagent_finished, the guard and clear drop a helper with its tick', () => {
    const t = new ActivityTracker();
    t.apply(spawned('a'));
    t.apply(spawned('b'));
    t.apply(spawned('c'));
    for (const child of ['a', 'b', 'c']) t.progress(readSubagentProgress(tick(child))!, 1);
    expect(t.apply(finished('a'))).toBe(true);
    expect(t.live().helpers.map((h) => h.childId)).toEqual(['b', 'c']);
    expect(t.progress(readSubagentProgress(tick('a'))!, 2)).toBe(false);
    expect(t.keepListed(new Set(['c']), t.helperMark())).toBe(true);
    expect(t.live().helpers.map((h) => h.childId)).toEqual(['c']);
    t.clear();
    expect([t.live(), t.tickedCount()]).toEqual([
      { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
      0,
    ]);
  });
});

describe('monitor events (T2)', () => {
  const MON = '01a20df5-8562-75b2-aaf5-d3c1e2ec9a01';
  const turnEnd = xai({ sessionUpdate: 'turn_completed', prompt_id: 'p1', stop_reason: 'end_turn' });

  // Mutation: keep every event (or the first five) → recent is not the last five.
  it('events count up; recent keeps the last 5, each cut at 500 characters with color codes removed', () => {
    const t = new ActivityTracker();
    t.apply(backgrounded(MON, 'Watch the build log'));
    const version = t.tasksVersion();
    for (let i = 1; i <= 7; i++) expect(t.event(MON, `line ${i}`, 1_000 * i)).toBe(true);
    expect(t.tasksVersion()).toBe(version + 7);
    const [task] = t.live().tasks;
    expect(task).toMatchObject({ id: MON, kind: 'monitor', events: 7 });
    expect(task!.recent).toEqual([3, 4, 5, 6, 7].map((i) => ({ at: 1_000 * i, text: `line ${i}` })));
    expect(TASK_EVENTS_KEPT).toBe(5);

    t.event(MON, `\u001b[31m${'x'.repeat(TASK_EVENT_CHARS + 20)}\u001b[39m`, 9_000);
    t.event(MON, `\u001b[1mok\u001b[22m`, 9_500);
    t.event(MON, 'Compiling…', 9_900);
    const recent = t.live().tasks[0]!.recent;
    expect(recent.slice(-3)).toEqual([
      { at: 9_000, text: `${'x'.repeat(TASK_EVENT_CHARS)}…`, cut: true },
      { at: 9_500, text: 'ok' },
      { at: 9_900, text: 'Compiling…' },
    ]);
  });

  it('an event for a task this process does not run, or one that ended, changes nothing', () => {
    const t = new ActivityTracker();
    expect(t.event(MON, 'early', 1)).toBe(false);
    t.apply(backgrounded(MON, 'Watch the build log'));
    t.apply(completed(MON));
    expect(t.event(MON, 'late', 2)).toBe(false);
    expect(t.live()).toEqual({ helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] });
    expect(t.wakeStarted(3)).toBe(false);
  });

  it('the live facts are copies: changing them changes nothing kept', () => {
    const t = new ActivityTracker();
    t.apply(backgrounded(MON, 'Watch the build log'));
    t.event(MON, 'one', 1);
    t.wakeStarted(5);
    const live = t.live();
    live.tasks[0]!.recent[0]!.text = 'changed';
    live.reports[0]!.monitors.push({ id: 'other', description: '' });
    live.reports[0]!.monitors[0]!.description = 'changed';
    expect(t.live().tasks[0]!.recent[0]!.text).toBe('one');
    expect(t.live().reports).toEqual([{ wakeTs: 5, monitors: [{ id: MON, description: '' }] }]);
  });

  // Mutation: no clear at wakeStarted → the next wake reports the same monitor again.
  it('a monitor event before a wake files one report with that wake’s ts; the next wake starts from none', () => {
    const t = new ActivityTracker();
    t.apply(backgrounded(MON, 'Watch the build log'));
    t.apply(backgrounded(TASK, 'Tail the server log'));
    t.event(TASK, 'GET /', 10, 'Tail the log');
    t.event(MON, 'error', 11, 'Watch the build log');
    t.event(TASK, 'GET /a', 12, 'Tail the server log');
    expect(t.wakeStarted(100)).toBe(true);
    expect(t.live().reports).toEqual([
      {
        wakeTs: 100,
        monitors: [
          { id: TASK, description: 'Tail the server log' },
          { id: MON, description: 'Watch the build log' },
        ],
      },
    ]);
    expect(t.wakeStarted(200)).toBe(false);
    expect(t.live().reports).toHaveLength(1);
  });

  // Mutation: drop the turn_completed case → an event the turn read names the next wake.
  it('a turn end clears what was reported: the model read those events in that turn', () => {
    const t = new ActivityTracker();
    t.apply(backgrounded(MON, 'Watch the build log'));
    t.event(MON, 'mid-turn', 10);
    expect(t.apply(turnEnd)).toBe(false);
    expect(t.wakeStarted(100)).toBe(false);
    t.event(MON, 'after the end', 110);
    expect(t.wakeStarted(200)).toBe(true);
    expect(t.live().reports).toEqual([{ wakeTs: 200, monitors: [{ id: MON, description: '' }] }]);
  });

  // Mutation: read only task_backgrounded → a tool result's task still has ticks that reach the reducer.
  it('an output tick of a call that started a task is known, from either start line, until clear', () => {
    const t = new ActivityTracker();
    const tick = (call: string, status = 'in_progress') => ({
      sessionUpdate: 'tool_call_update',
      toolCallId: call,
      status,
    });
    const isTick = (call: string) => t.isTaskTick(tick(call), undefined);
    expect(isTick(`call-${MON}`)).toBe(false);
    t.apply(backgrounded(MON, 'Watch the build log'));
    t.apply(
      acp({
        sessionUpdate: 'tool_call_update',
        toolCallId: 'call-result',
        status: 'completed',
        rawOutput: { type: 'BackgroundTaskStarted', task_id: TASK },
      }),
    );
    expect([isTick(`call-${MON}`), isTick('call-result')]).toEqual([true, true]);
    expect(isTick('call-other')).toBe(false);
    // The call's stored lines are not ticks: its result, and any line with `_meta`.
    expect(t.isTaskTick(tick(`call-${MON}`, 'completed'), undefined)).toBe(false);
    expect(t.isTaskTick(tick(`call-${MON}`), { eventId: 'e1' })).toBe(false);
    expect(
      t.isTaskTick({ sessionUpdate: 'tool_call', toolCallId: `call-${MON}`, status: 'in_progress' }, undefined),
    ).toBe(false);
    // An ended task's call may still send a last tick.
    t.apply(completed(MON));
    expect(isTick(`call-${MON}`)).toBe(true);
    t.clear();
    expect(isTick(`call-${MON}`)).toBe(false);
  });

  it('the list keeps the last 20 reports; clear forgets them (the process exited)', () => {
    const t = new ActivityTracker();
    t.apply(backgrounded(MON, 'Watch the build log'));
    for (let i = 1; i <= 25; i++) {
      t.event(MON, `e${i}`, i);
      t.wakeStarted(1_000 + i);
    }
    expect(WAKE_REPORTS_KEPT).toBe(20);
    expect(t.live().reports.map((r) => r.wakeTs)).toEqual(Array.from({ length: 20 }, (_, i) => 1_006 + i));
    t.clear();
    expect(t.live()).toEqual({ helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] });
  });
});
