/**
 * The agent rows of a workflow run: the attempt fold, the end state rule, the control target. The lines follow a real
 * Work session's `updates.jsonl` (a `workflow` script with four tracks, one of them retried by its output contract):
 * the run's first revisions carry no roster, the agents' spawn lines name the run, and the roster the agent stores at
 * the next step names each agent's newest attempt.
 */

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import {
  activityOf,
  agentPaletteOf,
  hasActivity,
  RUN_AGENTS_SHOWN,
  runRailDetail,
  type RunRailRow,
  runStoppable,
} from '../ui/src/lib/activity.js';
import {
  agentButtons,
  agentLiveOf,
  agentRowOf,
  agentState,
  agentsNow,
  attemptWords,
  controlTarget,
  failedReason,
  FINISH_WAIT_MS,
  runAgents,
  stoppedByYou,
  subagentRows,
  tokenWords,
} from '../ui/src/lib/agents.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { LiveActivity, PermissionInteraction } from '../ui/src/lib/chat-types.js';
import { stopText } from '../ui/src/lib/run-view.js';
import { runView } from './helpers/run-views.js';

const SID = '01a2167d-b0e0-79b0-b308-cddf84241628';
const RUN = 'wf_01a2168534c27fd1a477edb6fad7b0a4';
const T0 = 1_795_674_158_000;
const TRACKS = ['track:process', 'track:holes', 'track:claims-check', 'track:fix-reality'] as const;
const child = (n: number): string => `01a21685-34f1-7d63-8879-00000000000${n}`;
/** The run's contract retry of `track:process`: a new child, `retry_of` and resumed from the first. */
const RETRY = '01a21688-13f5-7b32-873d-247e85ca5c3f';
/** A script call `agent({resume_from})` that resumes `track:process` on its first attempt. */
const FOLLOW = '01a2168a-0000-7000-8000-00000000f011';
/** That call's own contract retry. */
const FOLLOW_RETRY = '01a2168a-0000-7000-8000-00000000f012';
/** A spawn whose `retry_of` names `track:process`'s first attempt after its retry took it. */
const AGAIN = '01a2168a-0000-7000-8000-00000000f013';
/** A script call `agent({resume_from})` with `track:process`'s label that resumes its retry. */
const RESUME = '01a2168a-0000-7000-8000-00000000f014';
const RUN2 = 'wf_01a2168534c27fd1a477edb6fad7b0b5';
const OTHER = '01a21685-34f1-7d63-8879-0000000000b0';

const STORED = { live: false, now: 0, open: false } as const;
const LIVE = (now: number) => ({ live: true, now });

function xai(ms: number, update: object): string {
  return JSON.stringify({
    timestamp: Math.floor(ms / 1000),
    method: '_x.ai/session/update',
    params: { sessionId: SID, update, _meta: { eventId: `${SID}-${ms}`, agentTimestampMs: ms } },
  });
}

const phases = (recon: string) => [
  { title: 'Recon', state: recon },
  { title: 'Verify', state: recon === 'done' ? 'active' : 'pending' },
  { title: 'Synthesize', state: 'pending' },
];

const revision = (ms: number, n: number, extra: object = {}): string =>
  xai(ms, {
    sessionUpdate: 'workflow_updated',
    run_id: RUN,
    revision: n,
    name: 'research-advisory',
    status: 'active',
    foreground: false,
    phases: phases('active'),
    agent_budget: 128,
    agents_used: 0,
    ...extra,
  });

const spawn = (ms: number, id: string, label: string, extra: object = {}): string =>
  xai(ms, {
    sessionUpdate: 'subagent_spawned',
    subagent_id: id,
    parent_session_id: SID,
    child_session_id: id,
    subagent_type: 'general-purpose',
    description: label,
    workflow_run_id: RUN,
    ...extra,
  });

const finished = (
  ms: number,
  id: string,
  tokens: number,
  durationMs: number,
  status = 'completed',
  extra: object = {},
): string =>
  xai(ms, {
    sessionUpdate: 'subagent_finished',
    subagent_id: id,
    child_session_id: id,
    status,
    tool_calls: 9,
    turns: 1,
    duration_ms: durationMs,
    tokens_used: tokens,
    ...extra,
  });
const billed = (n: number) => ({ total_tokens_used: n });

const roster = (agents: Array<[string, string, string, number, number]>) =>
  agents.map(([agent_id, label, state, tokens_used, duration_ms]) => ({
    agent_id,
    label,
    phase: 'Recon',
    state,
    tokens_used,
    duration_ms,
  }));

/** The run's start: two revisions with no roster, then its four agents. */
const START = [
  revision(T0, 1, { phases: phases('pending') }),
  revision(T0 + 1, 2, { phase_entered: 'Recon' }),
  ...TRACKS.map((label, n) => spawn(T0 + 10 + n, child(n), label)),
];
/** `track:process` answered in prose: its first attempt ends, and 47 ms later the run retries it. */
const RETRIED = [
  ...START,
  finished(T0 + 187_923, child(0), 25_262, 187_923),
  spawn(T0 + 187_970, RETRY, 'track:process', {
    resumed_from: child(0),
    retry_of: child(0),
    effective_context_source: 'resumed',
  }),
];
/** `track:process` ended, then a script call resumed it as `track:process-followup`. */
const FOLLOWED = [
  ...START,
  finished(T0 + 187_923, child(0), 25_262, 187_923),
  spawn(T0 + 188_000, FOLLOW, 'track:process-followup', { resumed_from: child(0) }),
];

function read(lines: string[]): TranscriptEntry[] {
  return parseUpdates(`${lines.join('\n')}\n`).transcript;
}

const indexOfChild = (entries: readonly TranscriptEntry[], id: string): number =>
  entries.findIndex((e) => e.kind === 'subagent' && e.childId === id);

describe('runAgents: the rows of a run', () => {
  // Mutation: list the roster alone → a stored transcript's run lists no agent (its roster is not stored yet).
  it("a run with no stored roster lists its agents' own entries, by spawn, running while the run runs", () => {
    const entries = read(START);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => [row.label, row.childId, row.at])).toEqual(
      TRACKS.map((label, n) => [label, child(n), [indexOfChild(entries, child(n))]]),
    );
    expect(agentsNow(rows, LIVE(T0 + 20)).map((agent) => agent.state)).toEqual(TRACKS.map(() => 'running'));
  });

  // Mutation: ignore `retry_of` → track:process is listed twice.
  it('an agent the run retried is one row in its first place: its newest attempt, with both attempts', () => {
    const entries = read(RETRIED);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => row.label)).toEqual([...TRACKS]);
    expect(rows[0]).toMatchObject({ childId: RETRY, attempts: [child(0), RETRY], continues: null });
    expect(rows[0]!.at).toEqual([indexOfChild(entries, child(0)), indexOfChild(entries, RETRY)]);
    expect(attemptWords(rows[0]!)).toEqual([fill(CHAT.agent.facts.attempt, { n: 2 })]);
  });

  // Mutation: match the roster to the entries by position → the rows get each other's numbers.
  // Mutation: order the rows by the roster → the rows reorder when the step's roster is stored.
  it('the stored roster joins each row by its newest attempt; the rows keep their spawn order', () => {
    const at = T0 + 845_501;
    const entries = read([
      ...RETRIED,
      finished(T0 + 308_450, RETRY, 27_462, 120_527),
      revision(at, 14, {
        current_phase: 'Verify',
        phases: phases('done'),
        agents_used: 4,
        agents: roster([
          [child(3), 'track:fix-reality', 'done', 1_937_982, 789_628],
          [child(2), 'track:claims-check', 'done', 187_449, 362_029],
          [child(1), 'track:holes', 'done', 3_997_846, 845_266],
          [RETRY, 'track:process', 'done', 188_922, 308_450],
        ]),
      }),
    ]);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => [row.label, row.childId, row.entry?.childId, row.tokens, row.durationMs])).toEqual([
      ['track:process', RETRY, RETRY, { n: 188_922, kind: 'used' }, 308_450],
      ['track:holes', child(1), child(1), { n: 3_997_846, kind: 'used' }, 845_266],
      ['track:claims-check', child(2), child(2), { n: 187_449, kind: 'used' }, 362_029],
      ['track:fix-reality', child(3), child(3), { n: 1_937_982, kind: 'used' }, 789_628],
    ]);
    expect(agentsNow(rows, STORED).map((agent) => agent.state)).toEqual(['done', 'done', 'done', 'done']);
    expect(tokenWords(rows[0]!)).toBe(fill(CHAT.agent.facts.tokensUsed, { n: '188.9K' }));
  });

  // Mutation: keep a roster row of an attempt the run retried → track:process is listed twice.
  it('a roster stored before a retry names the first attempt: the row is the newest attempt, and it runs', () => {
    const lines = [
      ...START,
      revision(T0 + 100, 3, {
        agents: roster(
          TRACKS.map((label, n) => [child(n), label, 'running', 0, 0] as [string, string, string, number, number]),
        ),
      }),
      finished(T0 + 187_923, child(0), 25_262, 187_923),
      spawn(T0 + 187_970, RETRY, 'track:process', { resumed_from: child(0), retry_of: child(0) }),
    ];
    const rows = runAgents(read(lines)).get(RUN)!;
    expect(agentsNow(rows, LIVE(T0 + 190_000)).map((a) => [a.row.label, a.row.childId, a.state])).toEqual([
      ['track:process', RETRY, 'running'],
      ['track:holes', child(1), 'running'],
      ['track:claims-check', child(2), 'running'],
      ['track:fix-reality', child(3), 'running'],
    ]);
  });

  // Mutation: walk a resumed spawn onto the row it resumed whatever its label → the follow-up loses its row.
  it('a script call that resumes an agent under a new label is its own row: it continues the first', () => {
    const entries = read([
      ...FOLLOWED,
      revision(T0 + 200_000, 5, {
        agents: roster([
          [child(0), 'track:process', 'done', 188_922, 187_923],
          [FOLLOW, 'track:process-followup', 'running', 0, 0],
        ]),
      }),
    ]);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => [row.childId, row.at, row.continues])).toEqual([
      [child(0), [indexOfChild(entries, child(0))], null],
      [child(1), [indexOfChild(entries, child(1))], null],
      [child(2), [indexOfChild(entries, child(2))], null],
      [child(3), [indexOfChild(entries, child(3))], null],
      [FOLLOW, [indexOfChild(entries, FOLLOW)], { childId: child(0), label: 'track:process' }],
    ]);
    expect(attemptWords(rows[4]!)).toEqual([fill(CHAT.agent.facts.continues, { label: 'track:process' })]);
  });

  it("a follow-up call's own retry folds into the follow-up; the agent it resumed keeps its row", () => {
    const entries = read([
      ...FOLLOWED,
      finished(T0 + 250_000, FOLLOW, 9_000, 62_000),
      spawn(T0 + 250_100, FOLLOW_RETRY, 'track:process-followup', { resumed_from: FOLLOW, retry_of: FOLLOW }),
      revision(T0 + 260_000, 6, {
        agents: roster([
          [child(0), 'track:process', 'done', 188_922, 187_923],
          [FOLLOW_RETRY, 'track:process-followup', 'running', 9_000, 62_000],
        ]),
      }),
    ]);
    const rows = runAgents(entries).get(RUN)!;
    expect([rows[0], rows[4]].map((row) => [row!.childId, row!.attempts, row!.continues?.childId ?? null])).toEqual([
      [child(0), [child(0)], null],
      [FOLLOW_RETRY, [FOLLOW, FOLLOW_RETRY], child(0)],
    ]);
  });

  it('a stale roster row whose retry has no roster row is drawn once, as the retry, with both attempts', () => {
    const entries = read([
      ...START.slice(0, 3),
      revision(T0 + 100, 3, { agents: roster([[child(0), 'track:process', 'running', 0, 0]]) }),
      finished(T0 + 187_923, child(0), 25_262, 187_923),
      spawn(T0 + 187_970, RETRY, 'track:process', { resumed_from: child(0), retry_of: child(0) }),
    ]);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => [row.childId, row.entry?.childId, row.at, row.roster])).toEqual([
      [RETRY, RETRY, [indexOfChild(entries, child(0)), indexOfChild(entries, RETRY)], null],
    ]);
  });

  // Mutation: take the newest attempt's run time alone → 120,527 ms.
  // Mutation: take the newest attempt's billed tokens alone → 13 tokens used.
  // Mutation: read a run agent's context at end when it has no roster → 27.5K context at end.
  it('a retried row built from its entries sums the run time and the billed tokens of its attempts', () => {
    const lines = [
      ...START,
      finished(T0 + 187_923, child(0), 25_262, 187_923, 'completed', billed(43)),
      ...RETRIED.slice(START.length + 1),
      finished(T0 + 308_450, RETRY, 27_462, 120_527, 'completed', billed(13)),
    ];
    const rows = runAgents(read(lines)).get(RUN)!;
    expect(rows[0]).toMatchObject({ childId: RETRY, durationMs: 308_450, tokens: { n: 56, kind: 'used' } });
    expect(tokenWords(rows[0]!)).toBe(fill(CHAT.agent.facts.tokensUsed, { n: '56' }));
  });

  // Mutation: sum the attempts that have a figure → 13 tokens used, though the first attempt's use is unknown.
  it('a row built from its entries has no token figure when an attempt has no billed tokens, never a context', () => {
    const rows = runAgents(
      read([...RETRIED, finished(T0 + 308_450, RETRY, 27_462, 120_527, 'completed', billed(13))]),
    ).get(RUN)!;
    expect(rows[0]!.tokens).toBeNull();
    expect(tokenWords(rows[0]!)).toBeNull();
  });

  // Mutation: read the stored state alone → an agent with no end reads running after its run ended.
  // Mutation: read every state as unfinished once the run is not open → a done agent reads not finished.
  it('an agent with no end reads running only while its run is open, else not finished; an ended one keeps its state', () => {
    const rows = runAgents(read([...START, finished(T0 + 362_029, child(2), 24_454, 362_029)])).get(RUN)!;
    const at = (open: boolean) => agentsNow(rows, { live: false, now: 0, open }).map((agent) => agent.state);
    expect(at(true)).toEqual(['running', 'running', 'done', 'running']);
    expect(at(false)).toEqual(['unfinished', 'unfinished', 'done', 'unfinished']);
  });
});

describe('the attempt fold', () => {
  // Mutation: fold on `resumed_from` → a spawn that resumed an ended agent with its label joins its row.
  it('a spawn with no `retry_of` is its own row, whatever its label, run or start time', () => {
    const early = [
      ...START,
      spawn(T0 + 100_000, FOLLOW, 'track:process', { resumed_from: child(0) }),
      finished(T0 + 187_923, child(0), 25_262, 187_923),
    ];
    const sameLabel = [
      ...START,
      finished(T0 + 187_923, child(0), 25_262, 187_923),
      spawn(T0 + 188_000, FOLLOW, 'track:process', { resumed_from: child(0) }),
    ];
    const otherRun = [
      ...START,
      finished(T0 + 187_923, child(0), 25_262, 187_923),
      spawn(T0 + 188_000, OTHER, 'track:process', { resumed_from: child(0), workflow_run_id: RUN2 }),
    ];
    const task = [
      ...START,
      finished(T0 + 187_923, child(0), 25_262, 187_923),
      xai(T0 + 188_000, {
        sessionUpdate: 'subagent_spawned',
        subagent_id: OTHER,
        child_session_id: OTHER,
        description: 'track:process',
        resumed_from: child(0),
      }),
    ];
    for (const [name, lines, id] of [
      ['new label', FOLLOWED, FOLLOW],
      ['same label', sameLabel, FOLLOW],
      ['before its end', early, FOLLOW],
    ] as const) {
      const rows = runAgents(read(lines)).get(RUN)!;
      expect(
        rows.map((row) => row.childId),
        name,
      ).toEqual([child(0), child(1), child(2), child(3), id]);
      expect(rows[4]!.continues?.childId, name).toBe(child(0));
    }
    const split = runAgents(read(otherRun));
    expect(split.get(RUN)![0]!.attempts).toEqual([child(0)]);
    expect(split.get(RUN2)!.map((row) => [row.childId, row.continues])).toEqual([
      [OTHER, { childId: child(0), label: 'track:process' }],
    ]);
    expect(runAgents(read(task)).get(RUN)![0]!.attempts).toEqual([child(0)]);
  });

  // Mutation: fold on `resumed_from` → the call that resumes the retry with its label is the row's third attempt.
  it("a script call with the agent's label that resumes its retry is its own row, also under a stale roster", () => {
    const entries = read([
      ...START,
      revision(T0 + 100, 3, { agents: roster([[child(0), 'track:process', 'running', 0, 0]]) }),
      finished(T0 + 187_923, child(0), 25_262, 187_923),
      spawn(T0 + 187_970, RETRY, 'track:process', { resumed_from: child(0), retry_of: child(0) }),
      finished(T0 + 308_450, RETRY, 27_462, 120_527),
      spawn(T0 + 400_000, RESUME, 'track:process', { resumed_from: RETRY }),
    ]);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => [row.childId, row.attempts, row.continues])).toEqual([
      [RETRY, [child(0), RETRY], null],
      [child(1), [child(1)], null],
      [child(2), [child(2)], null],
      [child(3), [child(3)], null],
      [RESUME, [RESUME], { childId: RETRY, label: 'track:process' }],
    ]);
    expect(controlTarget(rows, child(0))).toBe(RETRY);
    expect(attemptWords(rows[4]!)).toEqual([fill(CHAT.agent.facts.continues, { label: 'track:process' })]);
  });

  // Mutation: fold onto any row that holds the id → the spawn joins the retried row, and its rows clash.
  it("a `retry_of` that names an attempt that is no longer its row's newest is its own row", () => {
    const entries = read([
      ...RETRIED,
      finished(T0 + 308_450, RETRY, 27_462, 120_527),
      spawn(T0 + 400_000, AGAIN, 'track:process', { resumed_from: child(0), retry_of: child(0) }),
    ]);
    const rows = runAgents(entries).get(RUN)!;
    expect(rows.map((row) => [row.childId, row.attempts, row.continues?.childId ?? null])).toEqual([
      [RETRY, [child(0), RETRY], null],
      [child(1), [child(1)], null],
      [child(2), [child(2)], null],
      [child(3), [child(3)], null],
      [AGAIN, [AGAIN], child(0)],
    ]);
    expect(agentRowOf(rows, child(0))?.childId).toBe(RETRY);
  });
});

describe('the end state rule', () => {
  const ended = (status: string, rosterState: string, rosterAt: number) =>
    read([
      ...START,
      finished(T0 + 187_923, child(0), 25_262, 187_923, status),
      revision(rosterAt, 14, {
        agents: roster([[child(0), 'track:process', rosterState, 188_922, 187_923]]),
      }),
    ]);

  // Mutation: read the roster first → a stopped agent reads failed (the agent writes `failed` for a cancelled child).
  it('a finish line that says cancelled is a stop, whatever the newer roster says', () => {
    const [row] = runAgents(ended('cancelled', 'failed', T0 + 200_000)).get(RUN)!;
    expect(agentState(row!, STORED).state).toBe('stopped');
    expect(agentState(row!, LIVE(T0 + 200_001)).state).toBe('stopped');
  });

  it('a completed line with a roster as new that says failed is failed: its answer missed the format', () => {
    const [row] = runAgents(ended('completed', 'failed', T0 + 187_923)).get(RUN)!;
    expect(agentState(row!, STORED).state).toBe('failed');
    expect(agentState(row!, LIVE(T0 + 187_924)).state).toBe('failed');
    expect(failedReason(row!)).toBe(CHAT.agent.wrongFormat);
  });

  // Mutation: end the row at the finish line → it reads Done for 47 ms, then Running (the retry), then Done.
  it('live, a completed line waits for the roster, a retry or 1 s; a stored session takes the line', () => {
    const end = T0 + 187_923;
    const lines = [...START, finished(end, child(0), 25_262, 187_923)];
    const [row] = runAgents(read(lines)).get(RUN)!;
    expect(agentState(row!, LIVE(end + 47))).toMatchObject({ state: 'running', until: end + FINISH_WAIT_MS });
    expect(agentState(row!, LIVE(end + FINISH_WAIT_MS)).state).toBe('done');
    expect(agentState(row!, STORED).state).toBe('done');
    const [retried] = runAgents(
      read([...lines, spawn(end + 47, RETRY, 'track:process', { resumed_from: child(0), retry_of: child(0) })]),
    ).get(RUN)!;
    expect(agentState(retried!, LIVE(end + 2_000))).toMatchObject({ state: 'running', until: null });
    const moved = read([
      ...lines,
      revision(end + 5, 3, { agents: roster([[child(0), 'track:process', 'done', 9, 9]]) }),
    ]);
    expect(agentState(runAgents(moved).get(RUN)![0]!, LIVE(end + 6))).toMatchObject({ state: 'done', until: null });
  });
});

describe('the control target', () => {
  // Mutation: target the chain root → Stop and Message go to the first attempt, which has ended (404).
  it("a retried row's controls name its newest attempt, and its first attempt's id finds the row", () => {
    const rows = runAgents(read(RETRIED)).get(RUN)!;
    const row = agentRowOf(rows, child(0));
    expect(row?.childId).toBe(RETRY);
    expect(agentRowOf(rows, RETRY)).toBe(row);
    expect(agentRowOf(rows, 'nope')).toBeNull();
    expect([controlTarget(rows, child(0)), controlTarget(rows, RETRY), controlTarget(rows, 'nope')]).toEqual([
      RETRY,
      RETRY,
      null,
    ]);
  });

  // Mutation: match the host's agent by the row's newest id only → a retry the host knows first loses its Stop.
  it("the host's agent is the row's under any attempt; Stopped by you only for an attempt this host stopped", () => {
    const row = agentRowOf(runAgents(read(RETRIED)).get(RUN)!, RETRY)!;
    const first = {
      childId: child(0),
      run: RUN,
      attempts: [child(0)],
      stopping: true,
      messages: 'no-address' as const,
    };
    const live = (agents: LiveActivity['agents'], stopped: string[] = []): LiveActivity => ({
      helpers: [],
      agents,
      tasks: [],
      schedules: [],
      reports: [],
      stoppedByYou: stopped,
    });
    expect(agentLiveOf(live([first]), row)).toBe(first);
    expect(agentLiveOf(live([]), row)).toBeNull();
    expect(agentLiveOf(undefined, row)).toBeNull();
    expect([stoppedByYou(live([], [RETRY]), row), stoppedByYou(live([], [child(1)]), row)]).toEqual([true, false]);
  });
});

describe('runAgents keeps its lists while their rows stay the same', () => {
  // Mutation: build new lists each call → every run card re-renders on each streamed flush.
  // Mutation: keep every list once seen → an agent's end never reaches its card.
  // Mutation: keep the map but not each run's list → the run that did not change gets a new list.
  it('streamed text keeps the map; an agent that ends changes its own run alone', () => {
    const entries = read([
      ...START,
      revision(T0 + 50, 1, { run_id: RUN2 }),
      spawn(T0 + 60, OTHER, 'other', { workflow_run_id: RUN2 }),
    ]);
    const first = runAgents(entries);
    // The store replaces only the entry an op names; every other entry keeps its object.
    const streamed: TranscriptEntry[] = [...entries, { kind: 'assistant', ts: T0 + 70, text: 'Looking' }];
    expect(runAgents(streamed, first)).toBe(first);
    const flushed: TranscriptEntry[] = [...entries, { kind: 'assistant', ts: T0 + 70, text: 'Looking at it' }];
    expect(runAgents(flushed, first)).toBe(first);
    const end = indexOfChild(entries, child(1));
    const ended = flushed.map((entry, i): TranscriptEntry =>
      i === end && entry.kind === 'subagent'
        ? { ...entry, endTs: T0 + 845_266, status: 'completed', durationMs: 845_266 }
        : entry,
    );
    const next = runAgents(ended, first);
    expect(next).not.toBe(first);
    expect(next.get(RUN2)).toBe(first.get(RUN2));
    expect(next.get(RUN)).not.toBe(first.get(RUN));
    const row = next.get(RUN)!.find((r) => r.childId === child(1))!;
    expect([agentState(row, STORED).state, row.durationMs]).toEqual(['done', 845_266]);
  });

  // Mutation: leave the earlier attempts' counts out of the compare → the retried row keeps its old tool calls.
  it("an earlier attempt's counts that change give the retried row a new list", () => {
    const entries = read(RETRIED);
    const first = runAgents(entries);
    const end = indexOfChild(entries, child(0));
    const moved = entries.map((entry, i): TranscriptEntry =>
      i === end && entry.kind === 'subagent' ? { ...entry, toolCalls: 12 } : entry,
    );
    const next = runAgents(moved, first);
    expect(next.get(RUN)).not.toBe(first.get(RUN));
    expect(next.get(RUN)![0]!.earlier.toolCalls).toBe(12);
  });
});

describe("the rail: a run's agents count on its row", () => {
  // Mutation: count the phases in state complete or completed → 0 of 3 (the agent writes done).
  it("a held run's row counts its done phases", () => {
    const entries = read([...RETRIED, revision(T0 + 200_000, 5, { current_phase: 'Verify', phases: phases('done') })]);
    const [row] = activityOf({ entries, runs: [runView(RUN)], turnRunning: false }).running;
    expect(row).toMatchObject({ kind: 'workflow', phase: 'Verify', phasesDone: 1, phasesTotal: 3 });
  });

  // Mutation: list a held run's agents as rows of their own → "Running 5" for one run with four agents.
  it('a held run is one row with its running agents counted; a run the host no longer holds leaves them as rows', () => {
    const entries = read(RETRIED);
    const held = activityOf({ entries, runs: [runView(RUN)], turnRunning: false, now: T0 + 190_000 }).running;
    expect(held.map((row) => [row.kind, row.id])).toEqual([['workflow', RUN]]);
    expect(held[0]).toMatchObject({
      kind: 'workflow',
      agentsRunning: 4,
      anchor: entries.findIndex((e) => e.kind === 'workflow'),
    });
    const gone = activityOf({ entries, runs: [], turnRunning: false }).running;
    expect(gone.map((row) => [row.kind, row.id])).toEqual([
      ['subagent', child(1)],
      ['subagent', child(2)],
      ['subagent', child(3)],
      ['subagent', RETRY],
    ]);
  });
});

describe("the rail: a run's agents under its row, and the runs that wait (T21)", () => {
  const held = (...ids: string[]): LiveActivity => ({
    helpers: [],
    agents: ids.map((id) => ({
      childId: id,
      run: RUN,
      attempts: id === RETRY ? [child(0), RETRY] : [id],
      stopping: false,
      messages: 'no-address',
    })),
    tasks: [],
    schedules: [],
    reports: [],
    stoppedByYou: [],
  });

  // Mutation: look only at live runs → the stopping run leaves the rail and its draining agents turn loose rows.
  it('a run that left active keeps the agents it still drains under its row; none of them is a loose row', () => {
    const entries = read([...RETRIED, revision(T0 + 200_000, 5, { status: 'cancelled' })]);
    const view = runView(RUN, { live: false, status: 'cancelled', stopping: true });
    const activity = activityOf({ entries, runs: [view], live: held(RETRY, child(2)), turnRunning: false });
    expect(activity.running.map((row) => [row.kind, row.id])).toEqual([['workflow', RUN]]);
    const row = activity.running[0] as RunRailRow;
    expect(row).toMatchObject({ live: false, stopping: true, stoppable: false, waiting: false });
    expect(row.agents.map((agent) => [agent.id, agent.attempts, agent.live])).toEqual([
      [RETRY, [child(0), RETRY], true],
      [child(2), [child(2)], true],
    ]);
    // Drained: the stopped run leaves the rail, and its agents with no end line are no rows either.
    const drained = activityOf({ entries, runs: [{ ...view, stopping: false }], live: held(), turnRunning: false });
    expect([drained.running, drained.needsRuns]).toEqual([[], []]);
  });

  // Mutation: list a waiting run under Running → "Needs you" never shows the paused run and its Stop.
  it('a run that waits on the person sits in Needs you with why; Stop only where the agent takes it', () => {
    const message = 'Issue #821 could not be reproduced. Check the repro log before resuming.';
    const entries = read([...START, revision(T0 + 300, 3, { status: 'user_paused', pause_message: message })]);
    const paused = runView(RUN, { live: false, status: 'user_paused', waiting: true });
    const activity = activityOf({ entries, runs: [paused], live: held(), turnRunning: false });
    expect(activity.running).toEqual([]);
    expect(activity.needsRuns.map((row) => [row.id, row.waiting, row.stoppable, row.pauseMessage])).toEqual([
      [RUN, true, true, message],
    ]);
    expect(hasActivity(activity)).toBe(true);
    const out = activityOf({
      entries: read([...START, revision(T0 + 300, 3, { status: 'budget_limited' })]),
      runs: [runView(RUN, { live: false, status: 'budget_limited', waiting: true })],
      turnRunning: false,
    });
    expect(out.needsRuns.map((row) => [row.id, row.stoppable])).toEqual([[RUN, false]]);
  });

  // Mutation: count every row with no end → a paused run's Stop confirm says 4 agents stop that the host no longer holds.
  it('a run the host does not hold live counts the agents it lists: those the host still holds', () => {
    const entries = read([...START, revision(T0 + 300, 3, { status: 'user_paused' })]);
    const paused = runView(RUN, { live: false, status: 'user_paused', waiting: true });
    const [one] = activityOf({ entries, runs: [paused], live: held(child(1)), turnRunning: false }).needsRuns;
    expect([one!.listed.map((agent) => agent.id), one!.agentsRunning]).toEqual([[child(1)], 1]);
    const [none] = activityOf({ entries, runs: [paused], live: held(), turnRunning: false }).needsRuns;
    expect([none!.listed, none!.agentsRunning, stopText(none!.agentsRunning)]).toEqual([[], 0, CHAT.run.stopText.none]);
  });

  // Mutation: no cap → a run of 26 agents lists all 26 rows on the rail.
  // Mutation: keep spawn order alone → the asking agent sits past the cap, out of sight.
  it('at most 4 agents under a run, the asking ones first; the rest is a count; failed and done are counted', () => {
    const many = Array.from({ length: 7 }, (_, n) => `01a21685-34f1-7d63-8879-0000000001${n}0`);
    const entries = read([
      ...START.slice(0, 2),
      revision(T0 + 3, 3, { current_phase: 'Recon' }),
      ...many.map((id, n) => spawn(T0 + 10 + n, id, `agent-${n}`)),
      finished(T0 + 100, many[0]!, 10, 90, 'failed'),
      finished(T0 + 101, many[1]!, 10, 91),
    ]);
    const ask: PermissionInteraction = {
      id: 'a1',
      kind: 'permission',
      toolCallId: 'call-a1',
      openedAt: T0 + 200,
      toolName: 'run_terminal_command',
      access: 'bash',
      title: 'Execute `ls`',
      input: '{}',
      command: 'ls',
      path: null,
      resolvedPath: null,
      options: [],
      flags: [],
      from: { label: 'agent-6', run: 'research-advisory', cwd: null, child: many[6]! },
    };
    const { running } = activityOf({
      entries,
      interactions: [ask],
      runs: [runView(RUN)],
      turnRunning: false,
      now: T0 + 5_000,
    });
    const row = running[0] as RunRailRow;
    expect(row.agents.map((agent) => [agent.id, agent.waits])).toEqual([
      [many[6], true],
      [many[2], false],
      [many[3], false],
      [many[4], false],
    ]);
    expect(row.agents).toHaveLength(RUN_AGENTS_SHOWN);
    expect([row.moreAgents, row.agentsRunning, row.failed, row.done, row.step]).toEqual([1, 5, 1, 1, { i: 1, n: 3 }]);
  });
});

describe("a row's buttons, the same on every surface", () => {
  const held = (over: { stopping?: boolean; messages?: 'on' | 'off' | 'no-address' } = {}) => ({
    stopping: false,
    messages: 'on' as const,
    ...over,
  });

  // Mutation: drop the workflow branch → a workflow agent gets Message before the agent build can take it.
  it('a running Task subagent the host holds has Message and Stop; a workflow agent has Stop alone', () => {
    expect(agentButtons({ workflow: false, state: 'running', held: held(), askId: null })).toEqual({
      message: { off: null },
      review: null,
      stop: true,
    });
    expect(
      agentButtons({ workflow: true, state: 'running', held: held({ messages: 'no-address' }), askId: null }),
    ).toEqual({ message: null, review: null, stop: true });
  });

  // Mutation: drop the ask branch → a row that waits on a request offers Message in place of Review.
  it('a row that waits on a request has Review and Stop, of either kind', () => {
    for (const workflow of [false, true])
      expect(agentButtons({ workflow, state: 'running', held: held(), askId: 'p1' })).toEqual({
        message: null,
        review: 'p1',
        stop: true,
      });
  });

  // Mutation: drop `held.stopping` from the guard → a stopping row offers Stop a second time.
  it('none while its stop is on the way, when the host does not hold it, or once it ended', () => {
    const none = { message: null, review: null, stop: false };
    expect(agentButtons({ workflow: true, state: 'running', held: held({ stopping: true }), askId: null })).toEqual(
      none,
    );
    expect(agentButtons({ workflow: false, state: 'running', held: null, askId: 'p1' })).toEqual(none);
    for (const state of ['done', 'failed', 'stopped', 'unfinished'] as const)
      expect(agentButtons({ workflow: false, state, held: held(), askId: null })).toEqual(none);
  });

  // Mutation: draw a Message that cannot reach the agent as on → a press sends into nothing.
  it('a Message that cannot reach the subagent stays, off, with the reason', () => {
    expect(
      agentButtons({ workflow: false, state: 'running', held: held({ messages: 'off' }), askId: null }).message,
    ).toEqual({
      off: CHAT.activity.messageOff,
    });
    expect(
      agentButtons({ workflow: false, state: 'running', held: held({ messages: 'no-address' }), askId: null }).message,
    ).toEqual({ off: CHAT.activity.messageNoAddress });
  });
});

describe('the rail and palette words of a run', () => {
  const rail = (over: Partial<RunRailRow>): RunRailRow => ({
    kind: 'workflow',
    id: RUN,
    title: 'research-advisory',
    status: 'active',
    phase: 'Verify',
    step: { i: 2, n: 3 },
    phasesDone: 1,
    phasesTotal: 3,
    agentsRunning: 1,
    agents: [],
    listed: [],
    moreAgents: 0,
    failed: 0,
    done: 5,
    live: true,
    stopping: false,
    waiting: false,
    stoppable: true,
    pauseMessage: null,
    startTs: T0,
    anchor: 3,
    ...over,
  });

  // Mutation: number a step the script did not declare → "Step undefined of undefined: Hotfix".
  // Mutation: read the step before the wait → a paused run reads where it stood, not why it waits.
  it('where it is and how many run; a step the script did not declare has no number; a wait says why', () => {
    expect(runRailDetail(rail({}))).toBe(`${fill(CHAT.run.step, { i: 2, n: 3, phase: 'Verify' })} · 1 running`);
    expect(runRailDetail(rail({ phase: 'Hotfix', step: null, agentsRunning: 0 }))).toBe('Hotfix');
    expect(runRailDetail(rail({ phase: null, step: null, agentsRunning: 0 }))).toBe(CHAT.run.starting);
    expect(runRailDetail(rail({ stopping: true }))).toBe(CHAT.run.status.stopping);
    const message = 'Issue #821 could not be reproduced.';
    expect(runRailDetail(rail({ live: false, waiting: true, status: 'user_paused', pauseMessage: message }))).toBe(
      fill(CHAT.run.pausedBecause, { message }),
    );
    expect(runRailDetail(rail({ live: false, waiting: true, status: 'budget_limited' }))).toBe(
      CHAT.run.status.outOfAgents,
    );
  });

  // Mutation: drop the budget_limited exception → Stop on a run the agent refuses to stop.
  it('a Stop reaches a live run and one that waits on the person; not one out of agents, nor a stopping one', () => {
    expect(runStoppable(runView(RUN))).toBe(true);
    expect(runStoppable(runView(RUN, { live: false, waiting: true, status: 'user_paused' }))).toBe(true);
    expect(runStoppable(runView(RUN, { live: false, waiting: true, status: 'budget_limited' }))).toBe(false);
    expect(runStoppable(runView(RUN, { stopping: true }))).toBe(false);
    expect(runStoppable(runView(RUN, { live: false, status: 'cancelled' }))).toBe(false);
  });

  // Mutation: list the agents the rail shows (`agents`, at most 4) → the palette loses the fifth running agent.
  it('the palette lists each run, then every agent of it that asks or runs, then each Task subagent', () => {
    const many = Array.from({ length: 5 }, (_, n) => `01a21685-34f1-7d63-8879-0000000002${n}0`);
    const task = '01a21685-34f1-7d63-8879-0000000003a0';
    const entries = read([
      ...START.slice(0, 2),
      ...many.map((id, n) => spawn(T0 + 10 + n, id, `agent-${n}`)),
      spawn(T0 + 30, task, 'Find the flaky test', { workflow_run_id: undefined }),
    ]);
    const activity = activityOf({ entries, runs: [runView(RUN)], turnRunning: false, now: T0 + 5_000 });
    const run = activity.running.find((row) => row.kind === 'workflow') as RunRailRow;
    expect(run.agents).toHaveLength(RUN_AGENTS_SHOWN);
    const items = agentPaletteOf(activity);
    expect(
      items.map((item) => (item.kind === 'run' ? `run:${item.name}` : `${item.run ?? 'task'}:${item.label}`)),
    ).toEqual([
      'run:research-advisory',
      ...many.map((_, n) => `research-advisory:agent-${n}`),
      'task:Find the flaky test',
    ]);
    expect(items[0]).toMatchObject({ kind: 'run', stoppable: true, running: 5, anchor: run.anchor });
    // A run's agent is drawn in its block: it jumps there.
    expect(items.slice(1, 6).every((item) => item.anchor === run.anchor)).toBe(true);
  });
});

describe('Task subagent rows', () => {
  const task = (ms: number, id: string, label: string, extra: object = {}): string =>
    spawn(ms, id, label, { workflow_run_id: undefined, subagent_type: 'explore', ...extra });

  // Mutation: keep a run's agents → the run's tracks draw a second time as Task rows.
  it("one row per Task subagent, none for a run's agents; a resumed one says what it continues", () => {
    const entries = read([
      ...START,
      task(T0 + 20, OTHER, 'Find the flaky test'),
      finished(T0 + 30, OTHER, 900, 10),
      task(T0 + 40, FOLLOW, 'Fix it', { resumed_from: OTHER }),
    ]);
    const rows = subagentRows(entries);
    expect([...rows.keys()]).toEqual([OTHER, FOLLOW]);
    expect(rows.get(OTHER)!).toMatchObject({ run: null, label: 'Find the flaky test', continues: null });
    expect(rows.get(FOLLOW)!.continues).toEqual({ childId: OTHER, label: 'Find the flaky test' });
    expect(attemptWords(rows.get(FOLLOW)!)).toEqual([
      fill(CHAT.agent.facts.continues, { label: 'Find the flaky test' }),
    ]);
  });

  // Mutation: give a Task subagent its billed tokens → its row reads 3,400 tokens used, not its context at end.
  it('an ended Task subagent reads its context at end, also when its finish line names its billed tokens', () => {
    const entries = read([
      ...START,
      task(T0 + 20, OTHER, 'Find the flaky test'),
      finished(T0 + 30, OTHER, 1_200, 10, 'completed', billed(3_400)),
    ]);
    const row = subagentRows(entries).get(OTHER)!;
    expect(row.tokens).toEqual({ n: 1_200, kind: 'context' });
    expect(tokenWords(row)).toBe(fill(CHAT.agent.facts.contextAtEnd, { n: '1,200' }));
  });

  // Mutation: return a new map every time → every finished turn's rows re-render on each stream event.
  it('an update that changes no row keeps the map; a finish line makes a new row and map', () => {
    const entries = read([...START, task(T0 + 20, OTHER, 'Find the flaky test')]);
    const first = subagentRows(entries);
    const grown: TranscriptEntry[] = [...entries, { kind: 'assistant', ts: T0 + 25, text: 'Waiting.' }];
    expect(subagentRows(grown, first)).toBe(first);
    const ended = entries.map((e) =>
      e.kind === 'subagent' && e.childId === OTHER ? { ...e, endTs: T0 + 30, status: 'completed' } : e,
    );
    const next = subagentRows(ended, first);
    expect(next).not.toBe(first);
    expect(next.get(OTHER)!.entry?.endTs).toBe(T0 + 30);
  });
});
