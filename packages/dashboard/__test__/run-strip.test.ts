/**
 * The dock strip while the activity rail is a drawer: a line per live run the host holds, up to two, one line for three
 * or four, and the request an agent of one of them waits on.
 */

import { describe, expect, it } from 'vite-plus/test';

import type { AskRow, RunRailRow } from '../ui/src/lib/activity.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { runStripOf, STRIP_RUN_LINES } from '../ui/src/lib/run-strip.js';

const T0 = 1_795_674_158_000;

function run(id: string, over: Partial<RunRailRow> = {}): RunRailRow {
  return {
    kind: 'workflow',
    id,
    title: `research-${id}`,
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
    anchor: 4,
    ...over,
  };
}

function ask(id: string, runTitle: string, label: string): AskRow {
  return {
    id,
    kind: 'permission',
    access: 'bash',
    title: 'ls',
    path: false,
    from: { who: 'workflow', label, run: runTitle, childId: `${id}-child` },
    since: T0,
    front: false,
    autoAsk: false,
  };
}

describe('the dock strip', () => {
  // Mutation: keep every workflow row (drop `row.live`) → a run that waits on the person, or drains a stop, gets a line.
  it('none while the host holds no live run: a waiting run and one that drains a stop have no line', () => {
    expect(runStripOf({ running: [], needs: [] })).toEqual([]);
    const draining = run('a', { live: false, stopping: true, stoppable: false });
    expect(runStripOf({ running: [draining], needs: [] })).toEqual([]);
  });

  // Mutation: no ask line → the strip hides that an agent waits while the rail is closed.
  it('a line per run with its step and counts; an ask of one of its agents follows its line', () => {
    const a = run('a');
    const b = run('b', { phase: 'Hotfix', step: null, agentsRunning: 2, done: 0 });
    const lines = runStripOf({ running: [a, b], needs: [ask('p1', b.title, 'verify:claims')] });
    expect(lines).toEqual([
      {
        kind: 'run',
        id: 'a',
        name: a.title,
        step: fill(CHAT.run.step, { i: 2, n: 3, phase: 'Verify' }),
        counts: ['1 running', '5 done'],
        running: 1,
        startTs: T0,
        stopping: false,
        stoppable: true,
        anchor: 4,
      },
      expect.objectContaining({ kind: 'run', id: 'b', step: 'Hotfix', counts: ['2 running'], running: 2 }),
      { kind: 'ask', id: 'p1', name: b.title, text: fill(CHAT.run.asks.bash, { label: 'verify:claims' }) },
    ]);
  });

  // Mutations: the names left out → a PDF run's line reads "read-pdf"; the ask matched by the shown name → it is lost.
  it('a run shows the name it is given by id; its ask still finds it by its workflow name', () => {
    const pdf = run('p', { title: 'read-pdf' });
    const lines = runStripOf(
      { running: [pdf, run('a')], needs: [ask('p1', 'read-pdf', 'pages 1-4')] },
      new Map([['p', 'Reading report.pdf']]),
    );
    expect(lines.map((line) => [line.kind, 'name' in line ? line.name : null])).toEqual([
      ['run', 'Reading report.pdf'],
      ['ask', 'Reading report.pdf'],
      ['run', 'research-a'],
    ]);
  });

  // Mutation: compare with `>=` → two runs read as one summary line.
  // Mutation: count the agents of the first run only → "3 workflows running · 1 agent running".
  it('three or four runs read as one line with their running agents, then the first ask', () => {
    const runs = [run('a'), run('b', { agentsRunning: 4 }), run('c', { agentsRunning: 2 })];
    expect(STRIP_RUN_LINES).toBe(2);
    expect(runStripOf({ running: runs.slice(0, 2), needs: [] }).map((line) => line.kind)).toEqual(['run', 'run']);
    const lines = runStripOf({ running: runs, needs: [ask('p1', 'research-c', 'x'), ask('p2', 'research-a', 'y')] });
    expect(lines).toEqual([
      {
        kind: 'summary',
        text: fill(CHAT.run.strip.summary, { runs: 3, agents: fill(CHAT.run.strip.agents, { n: 7 }) }),
      },
      expect.objectContaining({ kind: 'ask', id: 'p2', name: 'research-a' }),
    ]);
  });
});
