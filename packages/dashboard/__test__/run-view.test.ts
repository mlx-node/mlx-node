/**
 * The words and rules of a workflow run's block and end row: the step rule and chips, the status words of each end,
 * who stopped it, the out-of-agents words, the new limit, where the report lives and when it reads as cut, and when
 * Resume is offered. The real run behind the fixture is `research-advisory` (3 steps, 7 agents, a 16,387-byte report).
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { mergeRunState, parseRunState } from '../src/agent-sessions/run-state.js';
import { parseUpdates } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { readPdfCommand } from '../src/chat/agent/pdf-tool.js';
import { type AgentEnd, type AgentNow, type AgentRow, agentsNow, runAgents } from '../ui/src/lib/agents.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ConversationEvent, RunBy, RunView } from '../ui/src/lib/chat-types.js';
import { formatClock } from '../ui/src/lib/format.js';
import {
  hostStartWords,
  leftWords,
  notStartedWords,
  pdfLine,
  pdfNames,
  pdfProgress,
  type PdfRun,
  pdfRunName,
  pdfUnread,
  rangeText,
  rateWords,
  readerOf,
  readPdfReportHead,
} from '../ui/src/lib/pdf-run.js';
import {
  agentGroups,
  budgetWords,
  chipAgents,
  countWords,
  endRowOf,
  logCount,
  logSteps,
  newLimit,
  nowLine,
  outOfAgentsWords,
  reportCut,
  reportPlace,
  resumeOffer,
  runAnnouncements,
  runBy,
  runCounts,
  type RunEntry,
  runsAt,
  runSnapshot,
  runState,
  shownRunNames,
  startedPdfs,
  stepChips,
  stepOf,
  stepWords,
  stoppedWords,
  stopText,
  summaryWords,
  wakeAt,
} from '../ui/src/lib/run-view.js';
import { runView } from './helpers/run-views.js';

const RUN = 'wf_01a2168534c27fd1a477edb6fad7b0a4';
const T0 = 1_795_674_158_000;
const S = CHAT.run.status;

const run = (over: Partial<RunEntry> = {}): RunEntry => ({
  kind: 'workflow',
  ts: T0,
  updatedTs: T0 + 1_000,
  runId: RUN,
  revision: 14,
  name: 'research-advisory',
  status: 'active',
  elapsedMs: 0,
  phases: [
    { title: 'Recon', state: 'done' },
    { title: 'Verify', state: 'active' },
    { title: 'Synthesize', state: 'pending' },
  ],
  currentPhase: 'Verify',
  agentBudget: 128,
  agentsUsed: 6,
  agents: [],
  phasesEntered: [
    { title: 'Recon', ts: T0 },
    { title: 'Verify', ts: T0 + 500 },
  ],
  ...over,
});

let ids = 0;
const agent = (phase: string | null, state: AgentEnd, over: Partial<AgentRow> = {}): AgentNow => {
  ids += 1;
  const childId = `child-${ids}`;
  const row: AgentRow = {
    childId,
    label: `agent-${ids}`,
    run: RUN,
    phase,
    attempts: [childId],
    at: [],
    roster: null,
    rosterTs: null,
    continues: null,
    tokens: null,
    durationMs: 0,
    ...over,
  };
  return { row, state, until: null };
};

const ctx = (over: { view?: ReturnType<typeof runView>; by?: RunBy | null; live?: boolean } = {}) => ({
  live: over.live ?? true,
  view: 'view' in over ? over.view : runView(RUN),
  by: over.by ?? null,
});

describe('the step rule', () => {
  // Mutation: `indexOf` + 1 → a title the script did not declare reads `Step 0 of 3: Triage`.
  it('a declared step has its number; an undeclared one has none; no step yet reads Starting; no steps, none', () => {
    expect(stepWords(stepOf(run()))).toBe(fill(CHAT.run.step, { i: 2, n: 3, phase: 'Verify' }));
    expect(stepWords(stepOf(run({ currentPhase: 'Triage' })))).toBe('Triage');
    expect(stepWords(stepOf(run({ currentPhase: null })))).toBe(CHAT.run.starting);
    expect(stepWords(stepOf(run({ currentPhase: null, phases: [] })))).toBeNull();
    expect(stepWords(stepOf(run({ currentPhase: 'Triage', phases: [] })))).toBe('Triage');
  });

  it('an undeclared step gets a chip after the declared ones, in order of first sight; agents with no step go under Other', () => {
    const entry = run({
      currentPhase: 'Triage',
      phases: run().phases.map((p) => ({ ...p, state: p.title === 'Recon' ? 'done' : 'pending' })),
      phasesEntered: [
        { title: 'Recon', ts: T0 },
        { title: 'Scan', ts: T0 + 100 },
        { title: 'Triage', ts: T0 + 200 },
      ],
    });
    const agents = [agent('Recon', 'done'), agent('Recon', 'done'), agent('Triage', 'running'), agent(null, 'done')];
    const chips = stepChips(entry, agents, runState(entry, agents, ctx()));
    expect(chips.map((chip) => [chip.title, chip.mark, chip.done, chip.total])).toEqual([
      ['Recon', 'done', 2, 2],
      ['Verify', 'pending', 0, 0],
      ['Synthesize', 'pending', 0, 0],
      ['Scan', 'done', 0, 0],
      ['Triage', 'active', 0, 1],
      [CHAT.run.otherStep, 'done', 1, 1],
    ]);
    expect(stepChips(run({ phases: [], currentPhase: null }), agents, runState(run(), agents, ctx()))).toEqual([]);
  });

  it('a chip is active while one of its agents runs, even when the wire says done', () => {
    const agents = [agent('Recon', 'running'), agent('Verify', 'done')];
    const chips = stepChips(run(), agents, runState(run(), agents, ctx()));
    expect(chips.map((chip) => [chip.title, chip.mark])).toEqual([
      ['Recon', 'active'],
      ['Verify', 'active'],
      ['Synthesize', 'pending'],
    ]);
  });

  // Mutation: trust the wire's `active` → a failed run's current step still reads current.
  it("an ended run's current step takes the run's end mark: failed, stopped, paused", () => {
    const agents = [agent('Recon', 'done'), agent('Verify', 'failed')];
    for (const [status, mark] of [
      ['failed', 'failed'],
      ['cancelled', 'stopped'],
      ['interrupted', 'stopped'],
      ['back_off_paused', 'paused'],
      ['budget_limited', 'paused'],
    ] as const) {
      const entry = run({ status });
      const state = runState(entry, agents, ctx({ by: 'you' }));
      const verify = stepChips(entry, agents, state).find((chip) => chip.title === 'Verify');
      expect(verify?.mark, status).toBe(mark);
    }
    const early = stepChips(run(), agents, runState(run(), agents, ctx({ view: undefined })));
    expect(early.find((chip) => chip.title === 'Verify')?.mark).toBe('unfinished');
  });
});

describe('who stopped it', () => {
  const stopCall = (id: string, kind: 'stop' | 'pause' | 'resume', status = 'completed'): TranscriptEntry => ({
    kind: 'tool',
    ts: T0,
    id: `call-${kind}-${status}`,
    name: 'workflow',
    title: 'Workflow: stop run',
    status,
    run: { id, kind },
  });
  const typed = (verb: 'pause' | 'resume' | 'stop'): TranscriptEntry => ({
    kind: 'user',
    ts: T0,
    text: `/workflow ${verb} research-advisory`,
    promptIndex: null,
    runControl: { verb, runId: RUN },
  });

  // Mutation: `Stopped by you` for a tool stop → the person reads that they stopped what the main agent stopped.
  it('the stop words name you, the main agent, the engine, or no one', () => {
    const step = stepOf(run());
    expect(stoppedWords('you', step)).toBe(fill(S.stoppedByYouStep, { i: 2, phase: 'Verify' }));
    expect(stoppedWords('main-agent', step)).toBe(fill(S.stoppedByMainAgentStep, { i: 2, phase: 'Verify' }));
    expect(stoppedWords('engine', step)).toBe(
      `${fill(S.stoppedStep, { i: 2, phase: 'Verify' })}. ${CHAT.run.stoppedEngine}`,
    );
    expect(stoppedWords(null, step)).toBe(fill(S.stoppedStep, { i: 2, phase: 'Verify' }));
    expect(stoppedWords(null, stepOf(run({ currentPhase: 'Triage' })))).toBe(fill(S.stoppedPhase, { phase: 'Triage' }));
    expect(stoppedWords('you', stepOf(run({ currentPhase: null })))).toBe(S.stoppedByYou);
  });

  it('a stored transcript names the main agent for its own call, you for a typed command, no one after a resume', () => {
    expect(runBy([stopCall(RUN, 'stop')], RUN)).toBe('main-agent');
    expect(runBy([typed('stop')], RUN)).toBe('you');
    expect(runBy([stopCall(RUN, 'stop'), typed('resume')], RUN)).toBeNull();
    expect(runBy([typed('stop'), stopCall(RUN, 'resume')], RUN)).toBeNull();
    expect(runBy([stopCall(RUN, 'stop', 'failed')], RUN)).toBeNull();
    expect(runBy([stopCall('wf_other', 'stop')], RUN)).toBeNull();
    // The host's own word wins while it holds the run: its Stop is not in the transcript.
    expect(runBy([stopCall(RUN, 'stop')], RUN, runView(RUN, { by: 'you' }))).toBe('you');
    const stopped = runState(run({ status: 'cancelled' }), [], ctx({ by: runBy([stopCall(RUN, 'stop')], RUN) }));
    expect(stopped.word).toBe(fill(S.stoppedByMainAgentStep, { i: 2, phase: 'Verify' }));
    expect(stopped.notes).toEqual([]);
  });
});

describe('the status of each end', () => {
  it('running, asking, retrying, stopping', () => {
    expect(runState(run(), [], ctx())).toMatchObject({ kind: 'running', word: S.running, pulse: true, notes: [] });
    expect(runState(run(), [], { ...ctx(), asking: { label: 'verify:claims', access: 'bash' } })).toMatchObject({
      kind: 'asking',
      word: S.waitingYou,
      notes: [fill(CHAT.run.asks.bash, { label: 'verify:claims' })],
      button: 'review',
    });
    const retry = agent('Recon', 'running', { label: 'track:process', attempts: ['a', 'b'] });
    expect(runState(run(), [retry], ctx()).notes).toEqual([fill(CHAT.run.retrying, { label: 'track:process' })]);
    const stopping = runState(run({ status: 'cancelled' }), [], ctx({ view: runView(RUN, { stopping: true }) }));
    expect(stopping).toMatchObject({ kind: 'stopping', word: S.stopping, notes: [CHAT.run.stoppingNote] });
  });

  it('active on disk: ended early when the host does not hold it; on the Sessions page, not finished or as of', () => {
    expect(runState(run(), [], ctx({ view: undefined }))).toMatchObject({
      kind: 'ended-early',
      word: S.endedEarly,
      line: CHAT.run.endedEarly,
    });
    const time = formatClock(T0 + 1_000);
    expect(runState(run(), [], ctx({ live: false, view: undefined })).word).toBe(fill(S.notFinished, { time }));
    expect(runState(run(), [], { ...ctx({ live: false, view: undefined }), sessionLive: true }).word).toBe(
      fill(S.runningAsOf, { time }),
    );
  });

  it('done, partly failed, failed with its message, out of agents', () => {
    expect(runState(run({ status: 'complete' }), [agent('Recon', 'done')], ctx())).toMatchObject({
      word: S.done,
      tone: 'success',
    });
    const some = [agent('Recon', 'done'), agent('Recon', 'failed'), agent('Verify', 'failed')];
    expect(runState(run({ status: 'complete' }), some, ctx())).toMatchObject({
      word: fill(S.donePartly, { n: 2, m: 3 }),
      tone: 'warning',
    });
    const message = 'Runtime error: Implementation agent failed (line 81, position 30)';
    expect(runState(run({ status: 'failed', pauseMessage: message }), [], ctx())).toMatchObject({
      word: fill(S.failedStep, { i: 2, phase: 'Verify' }),
      tone: 'danger',
      notes: [message],
    });
    expect(runState(run({ status: 'budget_limited', agentsUsed: 128 }), [], ctx())).toMatchObject({
      word: S.outOfAgents,
      line: fill(CHAT.run.outOfAgents.all, { budget: 128 }),
      notes: [CHAT.run.askToResume],
    });
  });

  it('stopped by you says the main agent was not told; interrupted after a stop took too long; else ended early', () => {
    expect(runState(run({ status: 'cancelled' }), [], ctx({ by: 'you' })).notes).toEqual([CHAT.run.notTold]);
    expect(runState(run({ status: 'interrupted' }), [], ctx({ by: 'you' }))).toMatchObject({
      word: S.stopped,
      line: CHAT.run.stoppedSlow,
    });
    const history = (events: string[]) => events.map((event, n) => ({ event, detail: null, ts: T0 + n }));
    const slow = run({
      status: 'interrupted',
      history: history(['phase_entered', 'workflow_cancelled', 'workflow_interrupted']),
    });
    expect(runState(slow, [], ctx({ view: undefined, live: false })).line).toBe(CHAT.run.stoppedSlow);
    const resumed = run({
      status: 'interrupted',
      history: history(['workflow_cancelled', 'workflow_outcome_ignored', 'workflow_resumed', 'workflow_interrupted']),
    });
    expect(runState(resumed, [], ctx({ view: undefined, live: false })).line).toBe(CHAT.run.endedEarly);
  });

  it('paused: by the main agent or you, its running agents were stopped; by its script, its message; and how to go on', () => {
    const paused = run({ status: 'user_paused', pauseMessage: 'Issue #821 could not be reproduced as described.' });
    expect(runState(paused, [], ctx({ by: 'main-agent' }))).toMatchObject({
      word: S.pausedByMainAgent,
      notes: [`${CHAT.run.pausedByMainAgentLine} ${CHAT.run.cancelPaused} ${CHAT.run.askToResume}`],
    });
    expect(runState(paused, [], ctx({ by: 'you' })).word).toBe(S.pausedByYou);
    expect(runState({ ...paused, status: 'blocked' }, [], ctx())).toMatchObject({
      word: S.paused,
      tone: 'warning',
      notes: [`${fill(CHAT.run.pausedBecause, { message: paused.pauseMessage! })} ${CHAT.run.askToResume}`],
    });
  });
});

describe('counts, budget and the glance line', () => {
  it('counts leave zeros out; the Done group holds done and stopped rows; asking and failed rows need attention', () => {
    const asks = agent('Verify', 'running');
    const agents = [
      agent('Recon', 'done'),
      agent('Recon', 'stopped'),
      agent('Recon', 'failed'),
      asks,
      agent('Verify', 'running'),
    ];
    expect(countWords(runCounts(agents))).toEqual([
      fill(CHAT.run.counts.running, { n: 2 }),
      fill(CHAT.run.counts.done, { n: 2 }),
      fill(CHAT.run.counts.failed, { n: 1 }),
    ]);
    const groups = agentGroups(agents, new Set([asks.row.childId]));
    expect([groups.attention.length, groups.running.length, groups.done.length]).toEqual([2, 1, 2]);
  });

  it('the budget is text; the meter shows from 75% used; a budget the agent could not count reads at least', () => {
    expect(budgetWords(run())).toEqual({ text: fill(CHAT.run.agentsUsed, { used: 6, budget: 128 }), meter: false });
    expect(budgetWords(run({ agentsUsed: 96 })).meter).toBe(true);
    expect(budgetWords(run({ agentsUsed: 95 })).meter).toBe(false);
    expect(budgetWords(run({ usageIncomplete: true })).text).toBe(
      fill(CHAT.run.agentsUsedAtLeast, { used: 6, budget: 128 }),
    );
    expect(budgetWords(run({ agentBudget: null })).text).toBe(fill(CHAT.run.agentsUsedNoBudget, { used: 6 }));
  });

  it("an ended run's summary: steps, agents, retries and the agents' total use", () => {
    const agents = [
      agent('Recon', 'done', { attempts: ['a', 'b'], tokens: { n: 188_922, kind: 'used' } }),
      agent('Recon', 'done', { tokens: { n: 14_700_000, kind: 'used' } }),
      agent('Verify', 'done', { tokens: { n: 51_801, kind: 'context' } }),
    ];
    expect(summaryWords(run({ status: 'complete' }), agents).join(' · ')).toBe(
      '3 steps · 3 agents · 1 retry · 14.9M tokens used',
    );
  });

  // Mutation: count a row's tokens only with a roster → a run whose roster was never stored reads no tokens used.
  it("an ended run's summary counts a retried agent with no roster by its attempts' billed tokens", () => {
    const line = (ms: number, update: object) =>
      JSON.stringify({
        timestamp: Math.floor(ms / 1000),
        method: '_x.ai/session/update',
        params: { sessionId: 's', update, _meta: { eventId: `s-${ms}`, agentTimestampMs: ms } },
      });
    const spawn = (ms: number, id: string, extra: object = {}) =>
      line(ms, {
        sessionUpdate: 'subagent_spawned',
        subagent_id: id,
        child_session_id: id,
        description: 'alpha',
        workflow_run_id: RUN,
        ...extra,
      });
    const end = (ms: number, id: string, total: number) =>
      line(ms, {
        sessionUpdate: 'subagent_finished',
        subagent_id: id,
        child_session_id: id,
        status: 'completed',
        tool_calls: 1,
        turns: 1,
        duration_ms: 100,
        tokens_used: 13,
        total_tokens_used: total,
      });
    const entries = parseUpdates(
      `${[
        line(T0, { sessionUpdate: 'workflow_updated', run_id: RUN, revision: 1, name: 'probe', status: 'active' }),
        spawn(T0 + 10, 'a1'),
        end(T0 + 400, 'a1', 43),
        spawn(T0 + 450, 'a2', { resumed_from: 'a1', retry_of: 'a1' }),
        end(T0 + 600, 'a2', 13),
        line(T0 + 700, {
          sessionUpdate: 'workflow_updated',
          run_id: RUN,
          revision: 2,
          name: 'probe',
          status: 'complete',
        }),
      ].join('\n')}\n`,
    ).transcript;
    const entry = entries.find((e): e is RunEntry => e.kind === 'workflow')!;
    const agents = agentsNow(runAgents(entries).get(RUN)!, { live: false, now: 0, open: false });
    expect(agents.map(({ row }) => [row.attempts, row.roster])).toEqual([[['a1', 'a2'], null]]);
    expect(summaryWords(entry, agents)).toEqual([
      CHAT.run.summary.agentsOne,
      CHAT.run.summary.retriesOne,
      fill(CHAT.run.summary.tokens, { tokens: '56' }),
    ]);
  });

  it('the Stop confirm says how many agents stop', () => {
    expect(stopText(0)).toBe(CHAT.run.stopText.none);
    expect(stopText(1)).toBe(CHAT.run.stopText.runningOne);
    expect(stopText(3)).toBe(fill(CHAT.run.stopText.running, { n: 3 }));
  });
});

describe('out of agents and the new limit', () => {
  // Mutation: always `It used all {budget}` → a run stopped with 3 left reads that it used all 128.
  it('names the agents it had left when its next group needed more; at 1,024 it cannot resume', () => {
    expect(outOfAgentsWords(128, 125)).toBe(fill(CHAT.run.outOfAgents.left, { left: 3, budget: 128 }));
    expect(outOfAgentsWords(128, 128)).toBe(fill(CHAT.run.outOfAgents.all, { budget: 128 }));
    expect(outOfAgentsWords(1024, 1000)).toBe(CHAT.run.outOfAgents.atCap);
    expect(runState(run({ status: 'budget_limited', agentBudget: 1024, agentsUsed: 1000 }), [], ctx()).notes).toEqual(
      [],
    );
  });

  // Mutation: min `used + 1` → a run that used 100 of 128 offers 101, which the agent refuses.
  // Mutation: default `2 × budget` past the cap → a budget of 600 offers 1,200.
  it('the field: above the old budget and what it used, starts at twice the budget, capped at 1,024; none at 1,024', () => {
    expect(newLimit(128, 100)).toEqual({ min: 129, initial: 256, max: 1024 });
    expect(newLimit(128, 130)).toEqual({ min: 131, initial: 256, max: 1024 });
    expect(newLimit(600, 600)).toEqual({ min: 601, initial: 1024, max: 1024 });
    expect(newLimit(1023, 1023)).toEqual({ min: 1024, initial: 1024, max: 1024 });
    expect(newLimit(1024, 1024)).toBeNull();
    expect(newLimit(1000, 1024)).toBeNull();
  });
});

describe('the report', () => {
  const lines = readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/parent.jsonl', import.meta.url)), 'utf8');
  const entries = parseUpdates(lines).transcript;
  const wakeIndex = entries.findIndex((entry) => entry.kind === 'wake');
  const before = entries.slice(0, wakeIndex);
  const entry = entries.find((e): e is RunEntry => e.kind === 'workflow')!;

  // Mutation: draw it in both → the end row and the block show one report twice.
  // Mutation: move it when the wake lands → the report jumps from the block to the end row.
  it('one home: the end row while a wake is due or its row is here; else the block; it never moves', () => {
    expect(entry).toMatchObject({ status: 'complete', revision: 26 });
    expect(reportPlace(entry, before, true)).toEqual({ home: 'end-row', writingUp: true });
    expect(reportPlace(entry, entries, false)).toEqual({ home: 'end-row', writingUp: false });
    expect(reportPlace(entry, before, false)).toEqual({ home: 'block', writingUp: false });
    expect(reportPlace({ ...entry, revision: 27 }, entries, false).home).toBe('block');
    expect(reportPlace({ ...entry, report: '' }, entries, true)).toEqual({ home: null, writingUp: false });
    expect(wakeAt(entries, entry)).toBe(wakeIndex);
    expect(wakeAt(before, entry)).toBeNull();
  });

  it('the end row reads the update its wake names: finished, its time, its agents', () => {
    const row = endRowOf(entries[wakeIndex] as Extract<TranscriptEntry, { kind: 'wake' }>)!;
    expect(row).toMatchObject({
      title: fill(CHAT.run.finished.complete, { name: 'research-advisory' }),
      facts: ['38m 44s', fill(CHAT.run.summary.agents, { n: 7 })],
      tone: 'success',
      cut: false,
    });
    expect(row.report).toBe(entry.report);
  });

  // Mutation: compare the length in characters → R's report (16,335 characters) reads as whole.
  it("reads as cut when its UTF-8 length is over 16,380 bytes and it ends in …, as R's report", () => {
    const r = `${'x'.repeat(16_309)}${'→'.repeat(25)}…`;
    expect([r.length, Buffer.byteLength(r, 'utf8')]).toEqual([16_335, 16_387]);
    expect(reportCut(r)).toBe(true);
    expect(reportCut(r.slice(0, -1))).toBe(false);
    expect(reportCut(`${'x'.repeat(16_376)}…`)).toBe(false);
    expect(reportCut(`${'x'.repeat(16_378)}…`)).toBe(true);
    expect(reportCut(`${'😀'.repeat(4_096)}…`)).toBe(true);
  });
});

describe('Resume', () => {
  const offer = (status: string, over: Partial<Parameters<typeof resumeOffer>[0]> = {}) =>
    resumeOffer({
      status,
      agentBudget: 128,
      agentsUsed: 6,
      view: runView(RUN, { status, live: false }),
      agentControl: true,
      ...over,
    });

  // Mutation: offer it for every failed run → a script failure is offered a resume that fails at the same place.
  it('offered for a stop, a script pause and a host failure; not for a script failure, interrupted, or a run not held', () => {
    expect(offer('cancelled')).toEqual({ offered: true, route: 'resume' });
    expect(offer('back_off_paused')).toEqual({ offered: true, route: 'resume' });
    expect(offer('blocked')).toEqual({ offered: true, route: 'resume' });
    expect(offer('failed', { failure: 'host' })).toEqual({ offered: true, route: 'resume' });
    expect(offer('budget_limited')).toEqual({ offered: true, route: 'resume' });
    expect(offer('failed', { failure: 'script' })).toEqual({ offered: false, why: null });
    expect(offer('failed')).toEqual({ offered: false, why: null });
    expect(offer('interrupted')).toEqual({ offered: false, why: null });
    expect(offer('complete')).toEqual({ offered: false, why: null });
    expect(offer('budget_limited', { agentBudget: 1024 })).toEqual({ offered: false, why: null });
    expect(offer('cancelled', { view: undefined })).toEqual({ offered: false, why: CHAT.run.cannotResume });
    expect(offer('cancelled', { view: runView(RUN, { status: 'cancelled', live: false, stopping: true }) })).toEqual({
      offered: false,
      why: null,
    });
  });

  it('a held run is released, not resumed; nothing is offered without the capability', () => {
    expect(offer('active', { hold: 'held', view: runView(RUN) })).toEqual({ offered: true, route: 'release' });
    expect(offer('active', { hold: 'holding', view: runView(RUN) })).toEqual({ offered: false, why: null });
    for (const status of ['cancelled', 'user_paused', 'budget_limited'])
      expect(offer(status, { agentControl: false, view: undefined })).toEqual({ offered: false, why: null });
  });
});

describe('the log and the now line', () => {
  const read = (name: string): string =>
    readFileSync(fileURLToPath(new URL(`./fixtures/workflow-run/${name}`, import.meta.url)), 'utf8');
  const stored = parseUpdates(read('parent.jsonl')).transcript.find((e): e is RunEntry => e.kind === 'workflow')!;
  const merged = mergeRunState(stored, parseRunState(read('state.json'))!, T0) as RunEntry;

  // Mutation: keep only the step's detail when it has log lines → Recon loses "Recon: 4 read-only tracks …".
  it("R's state.json: three steps, each with the script's detail and its one log line", () => {
    const steps = logSteps(merged);
    expect(steps.map((step) => [step.title, step.detail, step.lines.map((line) => line.text)])).toEqual([
      ['Recon', 'four independent read-only tracks', ['Recon: 4 read-only tracks over GHSA-0000-0000-0000']],
      ['Verify', 'adversarial checks of the key claims', ['Verify: adversarial checks of the two key claims']],
      ['Synthesize', 'final maintainer-ready report', ['Synthesize: final maintainer-ready report']],
    ]);
    expect(logCount(steps)).toBe(3);
    expect(logSteps(stored)).toEqual([]);
  });

  // Mutation: read the step's detail before its newest log → the line reads "final maintainer-ready report".
  it("the newest log of the current step with its time; else the step's detail with no time; none before a step", () => {
    const line = nowLine(merged)!;
    expect(line.text).toBe('Synthesize: final maintainer-ready report');
    expect(line.ts).toBe(Date.parse('2026-11-26T06:55:55.702146+00:00'));
    const noLog = { ...merged, history: merged.history!.filter((h) => h.event !== 'log') };
    expect(nowLine(noLog)).toEqual({ text: 'final maintainer-ready report', ts: null });
    expect(nowLine({ ...merged, currentPhase: null })).toBeNull();
    expect(nowLine({ ...stored, phases: stored.phases.map((p) => ({ ...p, detail: '' })) })).toBeNull();
  });
});

describe('the step filter', () => {
  // Mutation: match Other by its title → the agents with no step are in no chip.
  it("a chip names its step's agents; Other names those with no step", () => {
    const recon = agent('Recon', 'done');
    const verify = agent('Verify', 'running');
    const loose = agent(null, 'failed');
    const all = [recon, verify, loose];
    expect(chipAgents(all, { title: 'Verify' })).toEqual([verify]);
    expect(chipAgents(all, { title: 'Other', other: true })).toEqual([loose]);
  });
});

describe('what the polite region says', () => {
  const A = CHAT.run.announce;
  const say = (
    before: { entry: RunEntry; agents: AgentNow[] } | null,
    after: { entry: RunEntry; agents: AgentNow[] },
    held = true,
  ): string[] => {
    const snap = (at: { entry: RunEntry; agents: AgentNow[] }) =>
      runSnapshot(at.entry, at.agents, runState(at.entry, at.agents, ctx(held ? {} : { view: undefined })));
    return runAnnouncements(after.entry, after.agents, before === null ? null : snap(before), snap(after));
  };

  // Mutation: say every failed agent on each update → "Agent agent-… failed." twice.
  it('a new step, a newly failed agent once, the end once; a done agent and a first draw say nothing', () => {
    const one = agent('Recon', 'running');
    const two = agent('Recon', 'running');
    const recon = { entry: run({ currentPhase: 'Recon' }), agents: [one, two] };
    const verify = {
      entry: run(),
      agents: [
        { ...one, state: 'failed' as const },
        { ...two, state: 'done' as const },
      ],
    };
    expect(say(null, recon)).toEqual([]);
    expect(say(recon, verify)).toEqual([
      fill(A.step, { name: 'research-advisory', i: 2, n: 3, phase: 'Verify' }),
      fill(A.agentFailed, { label: one.row.label }),
    ]);
    expect(say(verify, verify)).toEqual([]);
    const done = { entry: run({ status: 'complete' }), agents: verify.agents };
    expect(say(verify, done, false)).toEqual([fill(A.finished, { name: 'research-advisory' })]);
    expect(say(done, done, false)).toEqual([]);
  });
});

describe('where each run is drawn', () => {
  const entries = parseUpdates(
    readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/parent.jsonl', import.meta.url)), 'utf8'),
  ).transcript;
  const launchAt = entries.findIndex((e) => e.kind === 'tool' && e.run?.kind === 'launch');
  const entryAt = entries.findIndex((e) => e.kind === 'workflow');
  const wake = entries.findIndex((e) => e.kind === 'wake');

  // Mutation: leave the workflow entry drawn when its launch holds the block → R draws two blocks.
  it("R: one block at its launch call, the run's entry data only, its wake and result read from the thread", () => {
    const at = runsAt(entries);
    expect([...at.blocks]).toEqual([[launchAt, RUN]]);
    expect([...at.dataOnly]).toEqual([[entryAt, RUN]]);
    const facts = at.runs.get(RUN)!;
    expect(facts).toMatchObject({ place: { at: launchAt, kind: 'launch' }, entryAt, resultAt: wake, by: null });
    expect(facts.launch).toBe(entries[launchAt]);
    expect(facts.wakes).toEqual([entries[wake]]);
    expect(at.names.get(RUN)).toBe('research-advisory');
  });

  // Mutation: match the wake by run id alone → a run stopped after a resume links to its out-of-agents result.
  it("the result link is the newest wake's only while the run still has the status that wake reported", () => {
    const woke = entries[wake] as Extract<TranscriptEntry, { kind: 'wake' }>;
    const shot = { ...woke.workflow!.snapshot!, status: 'budget_limited' };
    const thread = entries.map((e, i) =>
      i === wake ? { ...woke, workflow: { ...woke.workflow!, snapshot: shot } } : e,
    );
    const at = (status: string) => {
      const list = thread.map((e, i) => (i === entryAt ? { ...(e as RunEntry), status } : e));
      return [wakeAt(list, list[entryAt] as RunEntry), runsAt(list).runs.get(RUN)!.resultAt];
    };
    expect(at('cancelled')).toEqual([null, null]);
    expect(at('budget_limited')).toEqual([wake, wake]);
  });

  // Mutation: mark the entry data only for every place → a cut thread with no launch call draws no block.
  it('a thread cut after the launch draws the block at the entry, which is then drawn', () => {
    const cut = entries.slice(entryAt);
    const at = runsAt(cut);
    expect([...at.blocks]).toEqual([[0, RUN]]);
    expect(at.dataOnly.size).toBe(0);
    expect(at.runs.get(RUN)!.launch).toBeNull();
  });

  // Mutation: build a new answer every time → every finished turn re-renders on each stream event.
  it('an entry that does not touch a run keeps the answer; a new revision makes a new one', () => {
    const first = runsAt(entries);
    const grown = [...entries, { kind: 'assistant', ts: T0, text: 'More.' } as TranscriptEntry];
    expect(runsAt(grown, first)).toBe(first);
    const revised = entries.map((e) => (e.kind === 'workflow' ? { ...e, revision: e.revision + 1 } : e));
    const next = runsAt(revised, first);
    expect(next).not.toBe(first);
    expect(next.runs.get(RUN)!.entry).toBe(revised[entryAt]);
  });
});

describe('a launch the host names', () => {
  // R's lines up to its first `workflow_updated`: the launch turn as it stands when the run starts.
  const lines = readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/parent.jsonl', import.meta.url)), 'utf8')
    .split('\n')
    .filter((line) => line !== '');
  const first = lines.findIndex((line) => line.includes('"sessionUpdate":"workflow_updated"'));
  const entries = parseUpdates(`${lines.slice(0, first + 1).join('\n')}\n`).transcript;
  const entryAt = entries.findIndex((e) => e.kind === 'workflow');
  const launched = entries.slice(0, entryAt + 1);
  const turn = { turnId: 't1', phase: 'running', startedAt: T0, wait: null } as const;

  const sync = (shown: TranscriptEntry[], runs: RunView[]): ConversationEvent => ({
    type: 'sync',
    seq: 1,
    conversation: {
      id: 'c1',
      mode: 'work',
      title: 'R',
      cwd: '/Users/dev/napi-rs',
      modelId: 'm',
      createdAt: T0,
      updatedAt: T0,
      hot: true,
      permissionMode: 'ask',
    },
    entries: shown,
    turns: [],
    turn,
    interactions: [],
    notices: [],
    transcriptError: null,
    permissionMarks: [],
    runs,
    live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
  });
  const step = (state: ConversationState, event: ConversationEvent): ConversationState => {
    const next = applyConversationEvent(state, { ...event, seq: state.seq + 1 } as ConversationEvent);
    if (next === 'resync') throw new Error('resync');
    return next;
  };
  /** What the block reads for R in `state`. */
  const read = (state: ConversationState) =>
    runState(runsAt(state.entries).runs.get(RUN)!.entry!, [], {
      live: true,
      view: state.runs.find((view) => view.runId === RUN),
      by: null,
    }).kind;

  // Mutation: drop the ops a runs event carries → the run has no entry when the host's view of it comes.
  it("a runs event brings the entry's ops and its view in one step; a later list without the run ends it early", () => {
    const start = applyConversationEvent(null, sync(entries.slice(0, entryAt), []));
    if (start === 'resync') throw new Error('resync');
    const together = step(start, {
      type: 'runs',
      seq: 0,
      runs: [runView(RUN)],
      ops: [{ op: 'push', entry: entries[entryAt]! }],
    });
    expect(together.entries).toHaveLength(launched.length);
    expect(read(together)).toBe('running');
    const gone = step(together, { type: 'runs', seq: 0, runs: [] });
    expect(read(gone)).toBe('ended-early');
  });
});

describe('a run of the PDF tool', () => {
  const ID = 'abcdefgh-40';
  const P = CHAT.pdf;
  const note: TranscriptEntry = {
    kind: 'user',
    ts: T0 - 5_000,
    text: 'Summarize the findings',
    promptIndex: 0,
    attachments: [{ kind: 'pdf', name: 'report.pdf', pages: 40, way: 'tool', bytes: 0, chars: 0, id: ID }],
  };
  const call = (input: unknown, id = RUN): TranscriptEntry => ({
    kind: 'tool',
    ts: T0 - 1_000,
    id: `call-${id}`,
    name: 'workflow',
    title: 'Workflow: read-pdf',
    status: 'completed',
    input: JSON.stringify(input),
    run: { id, kind: 'launch' },
  });
  const launch = (args: Record<string, unknown>, name = 'read-pdf') =>
    call({ variant: 'Workflow', source: { type: 'name', name }, args });
  const pdfRun = (over: Partial<RunEntry> = {}): RunEntry =>
    run({
      name: 'read-pdf',
      phases: [
        { title: 'Read pages', state: 'active' },
        { title: 'Combine notes', state: 'pending' },
      ],
      currentPhase: 'Read pages',
      phasesEntered: [{ title: 'Read pages', ts: T0 }],
      ...over,
    });
  const pdfOf = (entries: TranscriptEntry[]) => runsAt(entries).runs.get(RUN)?.pdf ?? null;
  const all = (n: number) => Array.from({ length: n }, (_, i) => i + 1);
  const READ: PdfRun = { id: ID, name: 'report.pdf', pages: all(40), by: 'model' };

  /** A reader of the run: its label, its state, and its spawn and finish times (ms after T0). */
  const reader = (label: string, state: AgentEnd, start: number, end?: number): AgentNow => {
    ids += 1;
    const childId = `reader-${ids}`;
    return {
      row: {
        childId,
        label,
        run: RUN,
        phase: 'Read pages',
        entry: {
          kind: 'subagent',
          ts: T0 + start,
          childId,
          type: 'pdf-reader',
          description: label,
          run: RUN,
          ...(end === undefined ? {} : { endTs: T0 + end, durationMs: end - start }),
        },
        attempts: [childId],
        at: [],
        roster: null,
        rosterTs: null,
        continues: null,
        tokens: null,
        durationMs: end === undefined ? 0 : end - start,
      },
      state,
      until: null,
    };
  };
  const progress = (
    agents: AgentNow[],
    over: { entry?: RunEntry; running?: boolean; now?: number; run?: PdfRun } = {},
  ) =>
    pdfProgress({
      run: over.run ?? READ,
      agents,
      entry: over.entry ?? pdfRun(),
      running: over.running ?? true,
      now: over.now ?? T0 + 1_000_000,
    });

  it("is a run of the PDF tool when the model's call names read-pdf and an id; its pages from the call", () => {
    const entries = [note, launch({ pdf: ID, question: 'q', pages: '9-12, 1-4, 3' }), pdfRun()];
    expect(pdfOf(entries)).toEqual({ id: ID, name: 'report.pdf', pages: [1, 2, 3, 4, 9, 10, 11, 12], by: 'model' });
    expect(pdfOf([note, launch({ pdf: ID, question: 'q' }), pdfRun()])?.pages).toEqual(all(40));
    expect(pdfOf([launch({ pdf: ID, question: 'q' }), pdfRun()])?.name).toBeNull();
    expect(pdfOf([note, launch({ pdf: ID }, 'deep-research'), pdfRun()])).toBeNull();
    expect(pdfOf([note, launch({ pdf: '../x-12' }), pdfRun()])).toBeNull();
    expect(pdfOf([note, launch({ pdf: ID }), pdfRun({ name: 'deep-research' })])).toBeNull();
    const script = call({ variant: 'Workflow', source: { type: 'script', script: 'read-pdf' }, args: { pdf: ID } });
    expect(pdfOf([note, script, pdfRun()])).toBeNull();
  });

  // Mutation: the app's own start read as the person's → it shows "You started reading".
  it('a Read it now that opens the turn of the run names its PDF; any other prompt there does not', () => {
    const typed: TranscriptEntry = {
      kind: 'user',
      ts: T0 - 500,
      text: readPdfCommand(ID, 'Summarize the findings', 'report.pdf', 'you'),
      promptIndex: 1,
    };
    expect(pdfOf([note, typed, pdfRun()])).toEqual({ id: ID, name: 'report.pdf', pages: all(40), by: 'you' });
    const own: TranscriptEntry = { ...typed, text: readPdfCommand(ID, 'Summarize the findings', 'report.pdf', 'app') };
    expect(pdfOf([note, own, pdfRun()])).toEqual({ id: ID, name: 'report.pdf', pages: all(40), by: 'app' });
    const end: TranscriptEntry = { kind: 'turn-end', ts: T0 - 200, stopReason: 'end_turn' };
    expect(pdfOf([note, typed, end, pdfRun()])).toBeNull();
    expect(pdfOf([note, pdfRun()])).toBeNull();
    const names = pdfNames([note]);
    expect(hostStartWords(typed.text, names)).toBe(fill(P.hostStart, { name: 'report.pdf' }));
    expect(hostStartWords(typed.text, new Map())).toBe(P.hostStartBare);
    expect(hostStartWords(own.text, names)).toBe('Started reading report.pdf');
    expect(hostStartWords(own.text, new Map())).toBe('Started reading a PDF');
    expect(hostStartWords('/workflow read-pdf {"pdf": "abcdefgh-40", "by": "model"}', names)).toBe(
      'You started reading report.pdf',
    );
    expect(hostStartWords('/workflow read-pdf {"pdf": "nope"}', names)).toBeNull();
    expect(hostStartWords('Summarize the findings', names)).toBeNull();
    expect(pdfRunName({ name: 'report.pdf' })).toBe('Reading report.pdf');
    expect(pdfRunName({ name: null })).toBe(P.runNameBare);
  });

  it('a reader is known by its label; the cuts and the combine are not readers', () => {
    expect(readerOf('pages 33-36', 40)).toEqual({ pages: [33, 34, 35, 36], again: false });
    expect(readerOf('pages 9-12 again', 40)).toEqual({ pages: [9, 10, 11, 12], again: true });
    expect(readerOf('pages 1, 3, 10-11', 40)).toEqual({ pages: [1, 3, 10, 11], again: false });
    expect(readerOf('pages 39-42', 40)).toBeNull();
    expect(readerOf('cut 1', 40)).toBeNull();
    expect(readerOf('combine', 40)).toBeNull();
    expect(rangeText([1, 2, 3, 5, 7, 8])).toBe('1-3, 5, 7-8');
  });

  // Mutation: the rate from one reader → a guess shows after the first reader.
  it('the rate shows once 2 readers ended well, from their pages over their span', () => {
    const one = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'running', 60_000)];
    expect(progress(one).rate).toBeNull();
    const two = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'done', 60_000, 120_000)];
    expect(progress(two).rate).toBeCloseTo(8 / 120);
    expect(rateWords(progress(two).rate!)).toBe('4.0 pages/min');
  });

  // Mutation: the rate over summed reader times → the queue wait between readers is left out.
  it('the rate spans the wait between readers, not the sum of their times', () => {
    const agents = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'done', 180_000, 240_000)];
    expect(progress(agents).rate).toBeCloseTo(8 / 240);
  });

  it('the rate reads the 5 newest readers that ended well', () => {
    const agents = [
      reader('pages 1-4', 'done', 0, 600_000),
      ...[1, 2, 3, 4, 5].map((k) =>
        reader(`pages ${k * 4 + 1}-${k * 4 + 4}`, 'done', 540_000 + k * 60_000, 600_000 + k * 60_000),
      ),
    ];
    expect(progress(agents).rate).toBeCloseTo(20 / 300);
  });

  // Mutation: a retried group counted twice → more pages read than the PDF has.
  it('pages read count a group once, whether it was read at its first try or its second', () => {
    const agents = [
      reader('pages 1-4', 'done', 0, 60_000),
      reader('pages 5-8', 'failed', 60_000, 70_000),
      reader('pages 9-12', 'done', 70_000, 130_000),
      reader('pages 1-4 again', 'done', 130_000, 190_000),
      reader('pages 5-8 again', 'done', 190_000, 250_000),
      reader('cut 1', 'done', 250_000, 260_000),
    ];
    expect(progress(agents).read).toBe(12);
    expect(progress(agents).notStarted).toBe(28);
  });

  // Mutation: a first reader that ended with a second try of its group counted → its pages read and its rate shown.
  it('a group with a second try counts only when that try ended well, and its first reader stays out of the rate', () => {
    const first = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'done', 60_000, 120_000)];
    const trying = progress([...first, reader('pages 1-4 again', 'running', 120_000)]);
    expect(trying).toMatchObject({ read: 4, rate: null });
    const tried = progress([...first, reader('pages 1-4 again', 'done', 120_000, 180_000)]);
    expect(tried.read).toBe(8);
    expect(tried.rate).toBeCloseTo(8 / 120);
  });

  // Mutation: time left in step 2 → the combine step shows a time left for pages already read.
  it('time left: the pages left at the rate, less the time since a reader last ended, only in step 1 while it runs', () => {
    const agents = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'done', 60_000, 120_000)];
    expect(progress(agents, { now: T0 + 120_000 }).leftMs).toBeCloseTo(32 / (8 / 120_000));
    expect(progress(agents, { now: T0 + 180_000 }).leftMs).toBeCloseTo(32 / (8 / 120_000) - 60_000);
    expect(progress(agents, { now: T0 + 100_000_000 }).leftMs).toBe(0);
    const combine = pdfRun({ currentPhase: 'Combine notes' });
    expect(progress(agents, { entry: combine }).leftMs).toBeNull();
    expect(progress(agents, { running: false }).leftMs).toBeNull();
    expect(progress([reader('pages 1-4', 'done', 0, 60_000)]).leftMs).toBeNull();
    const ended = pdfRun({ status: 'complete', currentPhase: 'Combine notes' });
    expect(progress(agents, { entry: ended }).leftMs).toBeNull();
  });

  it('after the end the report says how many pages were read and which were not; the rate is over the read', () => {
    const report = 'PDF read: 36 of 40 pages. Not read: 21-24.\n\nThe answer (p. 3).\n\nNotes by page\n…';
    expect(readPdfReportHead(report)).toEqual({ read: 36, total: 40, unread: [21, 22, 23, 24] });
    expect(readPdfReportHead('PDF read: 5 of 5 pages.\n\nok')).toEqual({ read: 5, total: 5, unread: [] });
    expect(readPdfReportHead('PDF read: 6 of 5 pages.')).toBeNull();
    expect(readPdfReportHead('PDF read: 5 of 5 pages. Not read: soon.')).toBeNull();
    expect(readPdfReportHead('# Findings')).toBeNull();
    const agents = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'done', 60_000, 240_000)];
    const ended = progress(agents, { entry: pdfRun({ status: 'complete', currentPhase: 'Combine notes', report }) });
    expect(ended).toMatchObject({ read: 36, unread: [21, 22, 23, 24], leftMs: null, notStarted: null });
    expect(ended.rate).toBeCloseTo(36 / 240);
    const one = progress([reader('pages 1-4', 'done', 0, 60_000)], {
      entry: pdfRun({ status: 'complete', currentPhase: 'Combine notes', report: 'PDF read: 4 of 4 pages.' }),
      run: { ...READ, pages: [1, 2, 3, 4] },
    });
    expect(one.rate).toBeCloseTo(4 / 60);
    expect(pdfLine(READ, ended)).toEqual({
      text: `${fill(P.line, { range: '1-40', done: 36, total: 40 })} · ${rateWords(36 / 240)}`,
      unread: fill(P.notRead, { list: '21-24' }),
    });
  });

  it('the line: pages, counts, rate and time left when known; one page and one left read as one', () => {
    const agents = [reader('pages 1-4', 'done', 0, 60_000), reader('pages 5-8', 'done', 60_000, 120_000)];
    const live = progress(agents, { now: T0 + 120_000 });
    expect(pdfLine(READ, live)?.text).toBe(
      [fill(P.line, { range: '1-40', done: 8, total: 40 }), '4.0 pages/min', fill(P.left.minutes, { n: 8 })].join(
        ' · ',
      ),
    );
    expect(notStartedWords(live)).toBe(fill(P.notStarted, { n: 32 }));
    const single: PdfRun = { ...READ, pages: [7] };
    expect(pdfLine(single, progress([], { run: single }))?.text).toBe(fill(P.lineOne, { range: '7', done: 0 }));
    expect(notStartedWords(progress([], { run: single }))).toBe(P.notStartedOne);
    expect(leftWords(59_999)).toBe(P.left.minute);
    expect(leftWords(19 * 60_000)).toBe(fill(P.left.minutes, { n: 19 }));
    expect(leftWords(59.6 * 60_000)).toBe(fill(P.left.hours, { n: 1 }));
    expect(leftWords(2 * 3_600_000)).toBe(fill(P.left.hours, { n: 2 }));
  });

  // Mutation: every run named "Reading" → a deep-research run in the run bar reads as a PDF read.
  it('a run of the PDF tool shows "Reading {file}" in place of its workflow name; other runs keep theirs', () => {
    const entries = [note, launch({ pdf: ID, question: 'q' }), pdfRun()];
    expect([...shownRunNames(runsAt(entries))]).toEqual([[RUN, 'Reading report.pdf']]);
    const other = [note, launch({ pdf: ID }, 'deep-research'), pdfRun({ name: 'deep-research' })];
    expect([...shownRunNames(runsAt(other))]).toEqual([]);
  });

  // Mutation: Read it now while the turn runs → a second run starts beside the model's own.
  it('Read it now: a PDF sent to the tool that no run read, once its turn ended, never while an answer runs', () => {
    const entries = [note];
    expect(pdfUnread(entries, 0, false, new Set(), 'conv')).toEqual([{ id: ID, name: 'report.pdf' }]);
    expect(pdfUnread(entries, 0, true, new Set(), 'conv')).toEqual([]);
    const read = [note, launch({ pdf: ID, question: 'q' }), pdfRun()];
    expect(pdfUnread(read, 0, false, startedPdfs(runsAt(read)), 'conv')).toEqual([]);
    const other = [note, launch({ pdf: 'zzzzzzzz-40', question: 'q' }), pdfRun()];
    expect(pdfUnread(other, 0, false, startedPdfs(runsAt(other)), 'conv')).toEqual([{ id: ID, name: 'report.pdf' }]);
    const inMessage: TranscriptEntry = {
      ...note,
      attachments: [{ kind: 'pdf', name: 'short.pdf', pages: 2, way: 'message', bytes: 9, chars: 12 }],
    };
    expect(pdfUnread([inMessage], 0, false, new Set(), 'conv')).toEqual([]);
  });

  // Mutations: the session check dropped → a fork offers a PDF copied from the original, whose page folder is the
  // original's; the check run on live notes too, which name no session → a PDF sent inside the fork loses its line.
  it('Read it now in a fork: none for a PDF copied from the original, as in any conversation for one sent there', () => {
    const stored = (session: string): TranscriptEntry[] => [
      {
        ...note,
        attachments: [{ kind: 'pdf', name: 'report.pdf', pages: 40, way: 'tool', bytes: 0, chars: 0, id: ID, session }],
      },
    ];
    const shown = [{ id: ID, name: 'report.pdf' }];
    expect(pdfUnread(stored('orig'), 0, false, new Set(), 'fork')).toEqual([]);
    expect(pdfUnread(stored('orig'), 0, false, new Set(), 'orig')).toEqual(shown);
    expect(pdfUnread(stored('fork'), 0, false, new Set(), 'fork')).toEqual(shown);
    expect(pdfUnread([note], 0, false, new Set(), 'fork')).toEqual(shown);
    expect(pdfUnread(stored('fork'), 0, true, new Set(), 'fork')).toEqual([]);
  });
});
