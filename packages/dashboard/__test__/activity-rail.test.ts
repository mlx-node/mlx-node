/**
 * @vitest-environment happy-dom
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { ActivityRail } from '../ui/src/components/arc-x/activity-rail/activity-rail.js';
import { ChatFrame } from '../ui/src/components/arc-x/chat-frame/chat-frame.js';
import {
  activityOf,
  type AskRow,
  currentPlanItem,
  hasActivity,
  type Activity,
  type ScheduleRow,
  type TaskRow,
} from '../ui/src/lib/activity.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { HelperProgress } from '../ui/src/lib/chat-types.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { ActivityToggle } from '../ui/src/pages/chat/thread-bar.js';
import { flush, mount, unmountAll } from './helpers/mount.js';
import { runView } from './helpers/run-views.js';
import { byRole, observeWidth, waitFor } from './render.js';

let restore: (() => void) | undefined;

afterEach(() => {
  unmountAll();
  restore?.();
  restore = undefined;
});

const item = (content: string, status: string) => ({ content, status, priority: 'medium' });
const plan = (ts: number, ...entries: ReturnType<typeof item>[]): TranscriptEntry => ({ kind: 'plan', ts, entries });
const user = (ts: number): TranscriptEntry => ({ kind: 'user', ts, text: 'go', promptIndex: 0 });
const subagent = (ts: number, childId: string, endTs?: number): TranscriptEntry => ({
  kind: 'subagent',
  ts,
  childId,
  type: 'general-purpose',
  description: `Research ${childId}`,
  ...(endTs !== undefined ? { endTs, status: 'completed' } : {}),
});
const workflow = (ts: number, runId: string): TranscriptEntry => ({
  kind: 'workflow',
  ts,
  updatedTs: ts,
  runId,
  revision: 1,
  name: `review-${runId}`,
  status: 'running',
  elapsedMs: 0,
  phases: [
    { title: 'Find', state: 'done' },
    { title: 'Verify', state: 'active' },
    { title: 'Report', state: 'pending' },
  ],
  currentPhase: 'Verify',
  agentBudget: null,
  agentsUsed: 2,
  agents: [],
  phasesEntered: [],
});

describe('activity model', () => {
  it('the current plan item: in progress first, else the first not done, else the last', () => {
    expect(currentPlanItem([item('a', 'completed'), item('b', 'in_progress'), item('c', 'pending')])?.content).toBe(
      'b',
    );
    expect(currentPlanItem([item('a', 'completed'), item('b', 'pending')])?.content).toBe('b');
    expect(currentPlanItem([item('a', 'completed'), item('b', 'completed')])?.content).toBe('b');
    expect(currentPlanItem([])).toBeNull();
  });

  it('the latest plan shows while an item is open or a turn runs', () => {
    const open = [plan(1, item('old', 'pending')), user(2), plan(3, item('a', 'completed'), item('b', 'pending'))];
    const shown = activityOf({ entries: open, runs: [], turnRunning: false }).plan;
    expect(shown?.anchor).toBe(2);
    expect(shown?.items.map((i) => i.content)).toEqual(['a', 'b']);
    const done = [plan(1, item('old', 'pending')), plan(3, item('a', 'completed'))];
    expect(activityOf({ entries: done, runs: [], turnRunning: true }).plan?.anchor).toBe(1);
    expect(activityOf({ entries: done, runs: [], turnRunning: false }).plan).toBeNull();
  });

  it('running: subagents with no end, workflow runs the host still holds, oldest first', () => {
    const entries = [user(1), workflow(5, 'w1'), subagent(3, 's1'), subagent(4, 's2', 9), workflow(6, 'w2')];
    const activity = activityOf({ entries, runs: [runView('w1')], turnRunning: true });
    expect(activity.running.map((row) => `${row.kind}:${row.id}@${String(row.anchor)}`)).toEqual([
      'subagent:s1@2',
      'workflow:w1@1',
    ]);
    const run = activity.running[1]!;
    expect(run.kind === 'workflow' && [run.phase, run.phasesDone, run.phasesTotal]).toEqual(['Verify', 1, 3]);
    expect(hasActivity(activity)).toBe(true);
    expect(hasActivity(activityOf({ entries: [user(1)], runs: [], turnRunning: false }))).toBe(false);
  });

  it('a conversation not loaded in a running agent lists no stored command or monitor', () => {
    const command: TranscriptEntry = {
      kind: 'tool',
      ts: 2,
      id: 't1',
      name: 'run_terminal_command',
      title: 'cargo test',
      status: 'completed',
      task: { id: 'bg1', kind: 'command', label: 'cargo test' },
    };
    expect(activityOf({ entries: [user(1), command], runs: [], turnRunning: false }).running).toHaveLength(1);
    expect(
      activityOf({ entries: [user(1), command], runs: [], turnRunning: false, agentLive: false }).running,
    ).toHaveLength(0);
  });
});

const SOME: Activity = {
  needs: [],
  needsRuns: [],
  plan: {
    items: [item('Read', 'completed'), item('Reproduce', 'in_progress'), item('Report', 'pending')],
    ts: 1,
    anchor: 4,
  },
  running: [
    {
      kind: 'subagent',
      id: 's1',
      attempts: ['s1'],
      title: 'Research A6IH',
      type: 'general-purpose',
      startTs: Date.now() - 65_000,
      anchor: 7,
      progress: null,
      live: false,
      stopping: false,
      messages: null,
      waits: false,
      opens: false,
    },
    {
      kind: 'workflow',
      id: 'w1',
      title: 'review',
      status: 'active',
      phase: 'Verify',
      step: { i: 2, n: 3 },
      phasesDone: 1,
      phasesTotal: 3,
      agentsRunning: 2,
      agents: [],
      listed: [],
      moreAgents: 0,
      failed: 0,
      done: 0,
      live: true,
      stopping: false,
      waiting: false,
      stoppable: true,
      pauseMessage: null,
      startTs: Date.now() - 5_000,
      anchor: 9,
    },
  ],
  scheduled: [],
};

describe('ActivityRail', () => {
  it('draws the plan, then the running rows with kind, detail and elapsed time; a row jumps to its card', () => {
    const onJump = vi.fn();
    const container = mount(createElement(ActivityRail, { activity: SOME, onJump, onAsk: () => undefined }));
    expect(container.textContent).toContain(fill(CHAT.activity.planCount, { done: 1, total: 3 }));
    expect(Array.from(container.querySelectorAll('li'), (li) => li.textContent)).toEqual(
      expect.arrayContaining(['Read', 'Reproduce', 'Report']),
    );
    const sub = byRole(container, 'button', fill(CHAT.activity.jump, { title: 'Research A6IH' }))!;
    expect(sub.textContent).toContain('Subagent · general-purpose');
    expect(sub.textContent).toContain('1:05');
    const run = byRole(container, 'button', fill(CHAT.activity.jump, { title: 'review' }))!;
    expect(run.textContent).toContain(`${fill(CHAT.run.step, { i: 2, n: 3, phase: 'Verify' })} · 2 running`);
    act(() => sub.click());
    expect(onJump).toHaveBeenCalledWith(7);
    act(() => byRole(container, 'button', fill(CHAT.activity.jump, { title: CHAT.activity.plan }))!.click());
    expect(onJump).toHaveBeenLastCalledWith(4);
    expect(container.textContent).not.toContain(CHAT.activity.scheduled);
  });

  it('a schedule whose stored next time has passed reads Due', () => {
    const container = mount(
      createElement(ActivityRail, {
        activity: {
          needs: [],
          needsRuns: [],
          plan: null,
          running: [],
          scheduled: [
            {
              id: 'a',
              title: 'check PR',
              nextTs: Date.now() - 1_000,
              repeat: 'every 10 minutes',
              anchor: 3,
              state: 'waiting',
            },
            { id: 'b', title: 'nightly', nextTs: Date.now() + 3_600_000, repeat: null, anchor: 5, state: 'waiting' },
          ],
        },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    const due = byRole(container, 'button', fill(CHAT.activity.jump, { title: 'check PR' }))!;
    expect(due.textContent).toContain('Schedule · every 10 minutes');
    expect(due.textContent).toContain(CHAT.activity.due);
    const later = byRole(container, 'button', fill(CHAT.activity.jump, { title: 'nightly' }))!;
    expect(later.textContent).toContain('Next ');
  });

  it('nothing going on: one quiet line', () => {
    const container = mount(
      createElement(ActivityRail, {
        activity: { needs: [], needsRuns: [], plan: null, running: [], scheduled: [] },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    expect(container.textContent).toBe(CHAT.activity.empty);
  });
});

describe('ActivityRail: a helper row ticks (#10)', () => {
  const NOW = 1_794_000_000_000;
  const tick = (over: Partial<HelperProgress> = {}): HelperProgress => ({
    childId: 's1',
    contextWindow: null,
    now: null,
    turns: 2,
    toolCalls: 12,
    tokens: 18_200,
    contextPct: 41,
    errors: 0,
    durationMs: 134_000,
    at: NOW,
    ...over,
  });
  const helperRow = (progress: HelperProgress | null): Activity => ({
    needs: [],
    needsRuns: [],
    plan: null,
    running: [
      {
        kind: 'subagent',
        id: 's1',
        attempts: ['s1'],
        title: 'Find the flaky test',
        type: 'explore',
        startTs: NOW - 134_000,
        anchor: 3,
        progress,
        live: false,
        stopping: false,
        messages: null,
        waits: false,
        opens: false,
      },
    ],
    scheduled: [],
  });
  const draw = (progress: HelperProgress | null) => {
    const container = mount(
      createElement(ActivityRail, { activity: helperRow(progress), onJump: () => undefined, onAsk: () => undefined }),
    );
    const row = byRole(container, 'button', fill(CHAT.activity.jump, { title: 'Find the flaky test' }))!;
    const line = row.querySelector<HTMLElement>('[data-progress]')!;
    return { row, line };
  };
  const a = CHAT.activity;

  afterEach(() => {
    vi.useRealTimers();
  });

  // Mutation: drop the turn from the detail → "turn 2" is missing.
  it('before the first tick it says Starting…; then the turn, the calls, how full and the tokens', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    const before = draw(null);
    expect(before.line.textContent).toBe(a.progressStarting);
    expect(before.row.textContent).toContain('Subagent · explore');
    expect(before.row.textContent).not.toContain(fill(a.progressTurn, { n: 2 }));
    unmountAll();
    const { row, line } = draw(tick());
    expect(row.textContent).toContain(`Subagent · explore · ${fill(a.progressTurn, { n: 2 })}`);
    expect(line.textContent).toBe(
      [
        fill(a.progressCalls, { n: 12 }),
        fill(a.progressContext, { pct: 41 }),
        fill(a.progressTokens, { n: '18.2K' }),
      ].join(' · '),
    );
    expect(line.getAttribute('data-tone')).toBe('normal');
    // One tab stop; the line is its description, read once on focus, and no live region.
    expect(row.getAttribute('aria-describedby')).toBe(line.id);
    expect(row.querySelectorAll('button, [tabindex]')).toHaveLength(0);
    expect(row.querySelector('[aria-live], [role="status"]')).toBeNull();
  });

  it('a full context is in warning ink; errors come first in danger ink, in words', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    const high = draw(tick({ toolCalls: 57, contextPct: 86, tokens: 172_000 })).line;
    expect(high.getAttribute('data-tone')).toBe('warning');
    expect(high.textContent).toBe(
      [
        fill(a.progressCalls, { n: 57 }),
        fill(a.progressContext, { pct: 86 }),
        fill(a.progressTokens, { n: '172K' }),
      ].join(' · '),
    );
    unmountAll();
    const failing = draw(tick({ errors: 2, toolCalls: 1 })).line;
    expect(failing.getAttribute('data-tone')).toBe('danger');
    expect(failing.textContent?.startsWith(`${fill(a.progressErrors, { n: 2 })} · ${a.progressCallsOne} · `)).toBe(
      true,
    );
    unmountAll();
    expect(draw(tick({ errors: 1 })).line.textContent?.startsWith(`${a.progressErrorsOne} · `)).toBe(true);
  });

  // Mutation: read the quiet time from the row's start, not the last tick → "Quiet" shows at once.
  it('no tick for 20 s reads Quiet for m:ss in muted ink, on the rail clock', () => {
    vi.useFakeTimers({ now: NOW, toFake: ['Date', 'setInterval', 'clearInterval'] });
    const { line } = draw(tick({ at: NOW - 15_000 }));
    expect(line.textContent).not.toContain(fill(a.progressQuiet, { time: '' }).trim());
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(line.textContent?.startsWith(fill(a.progressQuiet, { time: '0:20' }))).toBe(true);
    act(() => {
      vi.advanceTimersByTime(15_000);
    });
    expect(line.textContent?.startsWith(`${fill(a.progressQuiet, { time: '0:35' })} · `)).toBe(true);
    expect(line.getAttribute('data-tone')).toBe('muted');
  });
});

const ago = (ms: number): number => Date.now() - ms;
const NEEDS: AskRow[] = [
  {
    id: 'a1',
    kind: 'permission',
    access: 'bash',
    title: 'rm -rf build/',
    path: false,
    from: { who: 'agent' },
    since: ago(42_000),
    front: true,
    autoAsk: false,
  },
  {
    id: 'a2',
    kind: 'permission',
    access: 'edit',
    title: '/Users/dev/proj/src/app.ts',
    path: true,
    from: { who: 'subagent', label: 'fix tests', childId: 'c1' },
    since: ago(10_000),
    front: false,
    autoAsk: false,
  },
  {
    id: 'a3',
    kind: 'permission',
    access: 'bash',
    title: 'cargo publish',
    path: false,
    from: { who: 'workflow', label: 'packer', run: 'Release', childId: 'c2' },
    since: ago(4_000),
    front: false,
    autoAsk: true,
  },
  {
    id: 'a4',
    kind: 'plan',
    access: null,
    title: CHAT.needs.plan,
    path: false,
    from: { who: 'agent' },
    since: ago(26 * 60_000 + 10_000),
    front: false,
    autoAsk: false,
  },
];

const QUESTION: AskRow = {
  id: 'q1',
  kind: 'question',
  access: null,
  title: 'Which database should it use?',
  questions: 2,
  path: false,
  from: { who: 'agent' },
  since: ago(2 * 60_000 + 5_000),
  front: false,
  autoAsk: false,
};

const MCP: AskRow = {
  id: 'm1',
  kind: 'mcp-input',
  access: null,
  title: fill(CHAT.mcp.formTitle, { server: 'github' }),
  path: false,
  from: { who: 'agent' },
  since: ago(42_000),
  front: false,
  autoAsk: false,
};

describe('ActivityRail: Needs you', () => {
  it('comes first: one row per request with its kind and who asks; the front row is marked; a click fronts it', () => {
    const onAsk = vi.fn();
    const container = mount(
      createElement(ActivityRail, { activity: { ...SOME, needs: NEEDS }, onJump: () => undefined, onAsk }),
    );
    const sections = container.querySelectorAll('section');
    expect(sections[0]!.querySelector('h2')?.textContent).toBe(`${CHAT.needs.title}${NEEDS.length}`);
    expect(sections[1]!.textContent).toContain(CHAT.activity.plan);
    const rows = [...sections[0]!.querySelectorAll('button')];
    expect(rows.map((row) => row.querySelector('[class*="detail"]')?.textContent)).toEqual([
      CHAT.needs.kinds.bash,
      `${CHAT.needs.kinds.edit} · ${fill(CHAT.needs.fromSubagent, { label: 'fix tests' })}`,
      `${CHAT.needs.kinds.bash} · ${fill(CHAT.needs.fromWorkflow, { run: 'Release', label: 'packer' })} · ${CHAT.needs.autoAsks}`,
      CHAT.needs.kinds.plan,
    ]);
    expect(rows.map((row) => row.getAttribute('aria-current'))).toEqual(['true', null, null, null]);
    expect(rows.map((row) => row.hasAttribute('data-front'))).toEqual([true, false, false, false]);
    expect(rows[1]!.textContent).toContain('app.ts');
    act(() => rows[2]!.click());
    expect(onAsk).toHaveBeenCalledWith('a3');
  });

  it("a row's name says the wait in words; the clock is hidden and nothing counts down", () => {
    const container = mount(
      createElement(ActivityRail, {
        activity: { needs: NEEDS, needsRuns: [], plan: null, running: [], scheduled: [] },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    const rows = [...container.querySelectorAll('section button')];
    expect(rows[0]!.getAttribute('aria-label')).toBe(
      fill(CHAT.needs.rowName, {
        title: 'rm -rf build/',
        kind: CHAT.needs.kinds.bash.toLowerCase(),
        who: CHAT.needs.fromAgent,
        time: fill(CHAT.needs.seconds, { n: 42 }),
      }),
    );
    expect(rows[3]!.getAttribute('aria-label')).toContain(fill(CHAT.needs.minutes, { n: 26 }));
    expect(rows[3]!.querySelector('[data-wait-clock]')?.getAttribute('aria-hidden')).toBe('true');
    expect(rows[3]!.textContent).toContain('26:10');
    expect(container.textContent).not.toContain('left');
    expect(container.textContent).not.toContain(CHAT.activity.empty);
    const real = Date.now();
    vi.spyOn(Date, 'now').mockReturnValue(real + 60_000);
    act(() => {
      rows[0]!.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    });
    expect(rows[0]!.getAttribute('aria-label')).toContain(CHAT.needs.minute);
    vi.restoreAllMocks();
  });

  it('a question row: its first question, "Question · 2 questions", a wait clock and no time left', () => {
    const container = mount(
      createElement(ActivityRail, {
        activity: { ...SOME, needs: [QUESTION] },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    const row = container.querySelector<HTMLButtonElement>('section button')!;
    expect(row.getAttribute('data-ask')).toBe('question');
    expect(row.querySelector('[class*="title"]')?.textContent).toBe(QUESTION.title);
    expect(row.querySelector('[class*="detail"]')?.textContent).toBe(
      `${CHAT.needs.kinds.question} · ${fill(CHAT.needs.questions, { n: 2 })}`,
    );
    expect(row.querySelector('svg')?.getAttribute('class')).toContain('lucide-circle-question-mark');
    expect(row.textContent).toContain('2:05');
    expect(row.getAttribute('aria-label')).toContain(CHAT.needs.kinds.question.toLowerCase());
    expect(container.textContent).not.toContain('left');
  });

  it('an MCP row: "{server} asks for input", "MCP server", the plug icon; its name keeps the kind, no agent', () => {
    const container = mount(
      createElement(ActivityRail, {
        activity: { ...SOME, needs: [MCP] },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    const row = container.querySelector<HTMLButtonElement>('section button')!;
    expect(row.getAttribute('data-ask')).toBe('mcp-input');
    expect(row.querySelector('[class*="title"]')?.textContent).toBe(MCP.title);
    expect(row.querySelector('[class*="detail"]')?.textContent).toBe(CHAT.needs.kinds.mcpInput);
    expect(row.querySelector('svg')?.getAttribute('class')).toContain('lucide-plug');
    expect(row.getAttribute('aria-label')).toBe(
      fill(CHAT.needs.rowNameServer, {
        title: MCP.title,
        kind: CHAT.needs.kinds.mcpInput,
        time: fill(CHAT.needs.seconds, { n: 42 }),
      }),
    );
    expect(row.getAttribute('aria-label')).not.toContain(CHAT.needs.fromAgent);
  });

  // Mutation: drop the `run` case of `fromWords` → the row reads as the conversation's own agent asking.
  it("a workflow run's MCP request says the run asks, on the row and in its name", () => {
    const container = mount(
      createElement(ActivityRail, {
        activity: { ...SOME, needs: [{ ...MCP, from: { who: 'run', run: 'deep-research' } }] },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    const row = container.querySelector<HTMLButtonElement>('section button')!;
    const who = fill(CHAT.needs.fromRun, { run: 'deep-research' });
    expect(row.querySelector('[class*="detail"]')?.textContent).toBe(`${CHAT.needs.kinds.mcpInput} · ${who}`);
    expect(row.getAttribute('aria-label')).toBe(
      fill(CHAT.needs.rowName, {
        title: MCP.title,
        kind: CHAT.needs.kinds.mcpInput,
        who,
        time: fill(CHAT.needs.seconds, { n: 42 }),
      }),
    );
  });
});

describe('ActivityToggle', () => {
  it('open requests win over running work: a warning dot, their count and the words in its name', () => {
    const container = mount(
      createElement(ActivityToggle, { open: false, running: 4, needs: 2, onToggle: () => undefined }),
    );
    const button = container.querySelector('button')!;
    expect(button.getAttribute('aria-label')).toBe(`${CHAT.activity.show}, ${fill(CHAT.needs.count, { n: 2 })}`);
    expect(button.querySelector('[data-tone]')?.getAttribute('data-tone')).toBe('warning');
    expect(button.textContent).toContain('2');
    const one = mount(createElement(ActivityToggle, { open: true, running: 0, needs: 1, onToggle: () => undefined }));
    expect(one.querySelector('button')!.getAttribute('aria-label')).toBe(
      `${CHAT.activity.hide}, ${CHAT.needs.countOne}`,
    );
    const busy = mount(createElement(ActivityToggle, { open: false, running: 3, needs: 0, onToggle: () => undefined }));
    expect(busy.querySelector('[data-tone]')?.getAttribute('data-tone')).toBe('success');
    expect(busy.querySelector('button')!.getAttribute('aria-label')).toBe(
      `${CHAT.activity.show}, ${fill(CHAT.activity.running, { n: 3 })}`,
    );
  });
});

function Frame({ docked, initial }: { docked: boolean; initial: boolean }): ReactElement {
  const [open, setOpen] = useState(initial);
  return createElement(ChatFrame, {
    barStart: createElement('span', null, 'MODE'),
    barEnd: createElement('button', { type: 'button', onClick: () => setOpen((v) => !v) }, 'Toggle'),
    list: createElement('a', { href: '#c1' }, 'Conversation one'),
    listLabel: 'Chats',
    listOpen: false,
    onListOpenChange: () => undefined,
    dock: createElement('textarea', { 'aria-label': 'Message' }),
    children: createElement('p', null, 'MESSAGE BODY'),
    rail: {
      content: createElement('p', null, 'RAIL BODY'),
      label: CHAT.activity.label,
      docked,
      open,
      onOpenChange: setOpen,
    },
  });
}

describe('ChatFrame rail', () => {
  it('docked: a named column right of the thread, drawn only while open', async () => {
    restore = observeWidth(1400);
    const container = mount(createElement(Frame, { docked: true, initial: true }));
    await flush();
    const grid = container.querySelector('[data-rail]')!;
    expect(grid.getAttribute('data-rail')).toBe('open');
    expect(byRole(container, 'complementary', CHAT.activity.label)?.textContent).toBe('RAIL BODY');
    act(() => byRole(container, 'button', 'Toggle')!.click());
    expect(grid.getAttribute('data-rail')).toBe('closed');
    expect(container.textContent).not.toContain('RAIL BODY');
  });

  it('not docked: the rail is a drawer from the right', async () => {
    restore = observeWidth(1000);
    const container = mount(createElement(Frame, { docked: false, initial: false }));
    await flush();
    expect(container.querySelector('[data-rail]')).toBeNull();
    expect(document.body.textContent).not.toContain('RAIL BODY');
    act(() => byRole(container, 'button', 'Toggle')!.click());
    await waitFor(() => document.body.textContent?.includes('RAIL BODY') === true);
    expect(byRole(document.body, 'dialog', CHAT.activity.label)).not.toBeNull();
  });
});

describe('ActivityRail: Stop a command or monitor (T1)', () => {
  const a = CHAT.activity;
  const taskRow = (id: string, kind: 'command' | 'monitor', title: string, stopping = false, live = true): TaskRow => ({
    kind,
    id,
    title,
    startTs: Date.now() - 64_000,
    anchor: 3,
    stopping,
    live,
    events: 0,
    lastEvent: null,
  });
  const ofRows = (running: TaskRow[]): Activity => ({ needs: [], needsRuns: [], plan: null, running, scheduled: [] });
  const dialog = (): HTMLElement | null => {
    const all = document.querySelectorAll<HTMLElement>('[role="dialog"]');
    return all.length === 0 ? null : all[all.length - 1]!;
  };
  const stopOf = (title: string): HTMLElement | null => byRole(document, 'button', fill(a.stop, { title }));
  const jumpOf = (title: string): HTMLElement | null => byRole(document, 'button', fill(a.jump, { title }));

  /** The rail with rows a test can change, as the page's live facts do. */
  function harness(first: TaskRow[], onStopTask: (row: TaskRow) => void = () => undefined) {
    let set: (rows: TaskRow[]) => void = () => undefined;
    function Host(): ReactElement {
      const [rows, setRows] = useState(first);
      set = setRows;
      return createElement(ActivityRail, {
        activity: ofRows(rows),
        onJump: () => undefined,
        onAsk: () => undefined,
        onStopTask,
      });
    }
    const container = mount(createElement(Host));
    return { container, set: (rows: TaskRow[]) => act(() => set(rows)) };
  }

  // Mutation: put Stop inside the jump button → a button inside a button, one tab stop.
  it('a command row is two buttons, jump then Stop; the confirm names the kind and starts on Keep running', async () => {
    const onStopTask = vi.fn();
    const dev = taskRow('t1', 'command', 'npm run dev');
    const { container } = harness([dev, taskRow('t2', 'monitor', 'Watch the build log')], onStopTask);
    const item = container.querySelector('li')!;
    expect(Array.from(item.querySelectorAll('button'), (b) => b.getAttribute('aria-label'))).toEqual([
      fill(a.jump, { title: 'npm run dev' }),
      fill(a.stop, { title: 'npm run dev' }),
    ]);
    expect(jumpOf('npm run dev')!.querySelector('button')).toBeNull();
    expect(jumpOf('npm run dev')!.textContent).toContain(a.kinds.command);
    expect(jumpOf('npm run dev')!.textContent).toContain('1:04');

    act(() => stopOf('Watch the build log')!.click());
    await waitFor(() => dialog() !== null);
    expect(dialog()!.textContent).toContain(a.stopTitleMonitor);
    act(() => byRole(dialog()!, 'button', a.stopKeep)!.click());
    await waitFor(() => dialog() === null);
    expect(onStopTask).not.toHaveBeenCalled();

    act(() => stopOf('npm run dev')!.click());
    await waitFor(() => dialog() !== null);
    expect(dialog()!.textContent).toContain(a.stopTitleCommand);
    expect(dialog()!.textContent).toContain(a.stopText);
    expect(document.activeElement?.textContent).toBe(a.stopKeep);
    act(() => byRole(dialog()!, 'button', a.stopConfirm)!.click());
    await waitFor(() => onStopTask.mock.calls.length === 1);
    expect(onStopTask).toHaveBeenCalledWith(dev);
  });

  // Mutation: drop the `live` gate → a row the host no longer tracks offers a Stop that can only 404.
  it('a row the host does not track has no Stop; its jump button stays', () => {
    harness([taskRow('t1', 'command', 'npm run dev', false, false), taskRow('t2', 'monitor', 'Watch the build log')]);
    expect(stopOf('npm run dev')).toBeNull();
    expect(jumpOf('npm run dev')).not.toBeNull();
    expect(jumpOf('npm run dev')!.textContent).not.toContain(a.stopping);
    expect(stopOf('Watch the build log')).not.toBeNull();
  });

  // Mutation: keep the Stop on a stopping row → a second kill can be sent while the first is out.
  it('a stopping row reads Stopping… and has no Stop; with no Stop handler no row has one', () => {
    harness([taskRow('t1', 'command', 'npm run dev', true)]);
    expect(jumpOf('npm run dev')!.textContent).toContain(`${a.kinds.command} · ${a.stopping}`);
    expect(stopOf('npm run dev')).toBeNull();
    unmountAll();
    mount(
      createElement(ActivityRail, {
        activity: ofRows([taskRow('t1', 'command', 'npm run dev')]),
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    expect(stopOf('npm run dev')).toBeNull();
  });

  // Mutation: drop the row's leave cleanup → focus falls to the page body when a stopped row leaves.
  it('focus: the Stop going sends it to the jump button; a row leaving sends it to the next row, else the heading', async () => {
    const one = taskRow('t1', 'command', 'npm run dev');
    const two = taskRow('t2', 'monitor', 'Watch the build log');
    const { container, set } = harness([one, two]);
    stopOf('npm run dev')!.focus();
    set([{ ...one, stopping: true }, two]);
    await waitFor(() => document.activeElement === jumpOf('npm run dev'));
    set([two]);
    await waitFor(() => document.activeElement === jumpOf('Watch the build log'));
    set([]);
    // The section goes with its last row: the rail itself keeps focus on the page.
    await waitFor(() => document.activeElement === container.querySelector('[data-activity-rail]'));
  });

  // Mutation: drop the passive focus step → the confirm's button goes with its portal and focus falls to the body.
  it('the confirmed Stop turns the row stopping while its confirm holds focus: the jump button takes it', async () => {
    const one = taskRow('t1', 'command', 'npm run dev');
    let set: (rows: TaskRow[]) => void = () => undefined;
    const harnessed = harness([one], () => set([{ ...one, stopping: true }]));
    set = harnessed.set;
    act(() => stopOf('npm run dev')!.click());
    await waitFor(() => dialog() !== null);
    act(() => byRole(dialog()!, 'button', a.stopConfirm)!.click());
    await waitFor(() => stopOf('npm run dev') === null);
    await waitFor(() => document.activeElement === jumpOf('npm run dev'));
    expect(jumpOf('npm run dev')!.textContent).toContain(a.stopping);
  });

  it('a leaving row with rows above it hands focus to the section heading', async () => {
    const one = taskRow('t1', 'command', 'npm run dev');
    const two = taskRow('t2', 'monitor', 'Watch the build log');
    const { container, set } = harness([one, two]);
    jumpOf('Watch the build log')!.focus();
    set([one]);
    await waitFor(() => document.activeElement === container.querySelector('[data-heading]'));
    expect(document.activeElement?.textContent).toContain(a.nowRunning);
  });

  it('beside the thread the Stop shows on hover and focus; in the drawer it always shows', () => {
    const rows = ofRows([taskRow('t1', 'command', 'npm run dev')]);
    const base = { activity: rows, onJump: () => undefined, onAsk: () => undefined, onStopTask: () => undefined };
    const docked = mount(createElement(ActivityRail, base));
    expect(docked.querySelector('[data-activity-rail]')!.hasAttribute('data-actions-shown')).toBe(false);
    const drawer = mount(createElement(ActivityRail, { ...base, docked: false }));
    expect(drawer.querySelector('[data-activity-rail]')!.hasAttribute('data-actions-shown')).toBe(true);
  });
});

describe('ActivityRail: monitor events (T2)', () => {
  const a = CHAT.activity;
  const monitorRow = (events: number, lastEvent: string | null, stopping = false): TaskRow => ({
    kind: 'monitor',
    id: 'm1',
    title: 'Watch the build log',
    startTs: Date.now() - 250_000,
    anchor: 4,
    stopping,
    live: true,
    events,
    lastEvent,
  });
  const draw = (row: TaskRow) =>
    mount(
      createElement(ActivityRail, {
        activity: { needs: [], needsRuns: [], plan: null, running: [row], scheduled: [] },
        onJump: () => undefined,
        onAsk: () => undefined,
        onStopTask: () => undefined,
      }),
    );
  const jumpOf = (container: HTMLElement): HTMLElement =>
    byRole(container, 'button', fill(a.jump, { title: 'Watch the build log' }))!;
  const described = (button: HTMLElement): string | null => {
    const id = button.getAttribute('aria-describedby');
    return id === null ? null : (document.getElementById(id)?.textContent ?? null);
  };

  // Mutation: describe the button by the last line → the reader hears raw output, not the count.
  it('the detail counts the events and describes the jump button; the newest line shows under it, hidden from readers', () => {
    const container = draw(monitorRow(12, "error TS2345: Argument of type 'string'‮"));
    const jump = jumpOf(container);
    const detail = `${a.kinds.monitor} · ${fill(a.events, { n: 12 })}`;
    expect(jump.textContent).toContain(detail);
    expect(described(jump)).toBe(detail);
    const line = container.querySelector<HTMLElement>('[data-last-event]')!;
    expect(line.getAttribute('aria-hidden')).toBe('true');
    expect(line.textContent).toBe("error TS2345: Argument of type 'string'\\u{202E}");
  });

  it('one event reads "1 event"; while stopping the detail says so and the line stays; with none, no line and no description', () => {
    expect(jumpOf(draw(monitorRow(1, 'done'))).textContent).toContain(`${a.kinds.monitor} · ${a.eventsOne}`);
    unmountAll();
    const stopping = draw(monitorRow(3, 'still here', true));
    expect(jumpOf(stopping).textContent).toContain(`${a.kinds.monitor} · ${a.stopping}`);
    expect(stopping.querySelector('[data-last-event]')!.textContent).toBe('still here');
    unmountAll();
    const quiet = draw(monitorRow(0, null));
    expect(jumpOf(quiet).textContent).toContain(a.kinds.monitor);
    expect(jumpOf(quiet).textContent).not.toContain(fill(a.events, { n: 0 }));
    expect(quiet.querySelector('[data-last-event]')).toBeNull();
    expect(jumpOf(quiet).getAttribute('aria-describedby')).toBeNull();
  });
});

describe('ActivityRail: Delete a schedule (T3)', () => {
  const a = CHAT.activity;
  const scheduleRow = (
    id: string,
    title: string,
    state: ScheduleRow['state'],
    nextTs: number | null = Date.now() + 3_600_000,
  ): ScheduleRow => ({ id, title, nextTs, repeat: 'every 5 minutes', anchor: 5, state });
  const ofRows = (scheduled: ScheduleRow[]): Activity => ({
    needs: [],
    needsRuns: [],
    plan: null,
    running: [],
    scheduled,
  });
  const dialog = (): HTMLElement | null => {
    const all = document.querySelectorAll<HTMLElement>('[role="dialog"]');
    return all.length === 0 ? null : all[all.length - 1]!;
  };
  const deleteOf = (title: string): HTMLElement | null => byRole(document, 'button', fill(a.deleteSchedule, { title }));
  const jumpOf = (title: string): HTMLElement | null => byRole(document, 'button', fill(a.jump, { title }));

  function harness(
    first: ScheduleRow[],
    onDeleteSchedule: ((row: ScheduleRow) => void) | null = () => undefined,
    deleteOff: string | null = null,
  ) {
    let set: (rows: ScheduleRow[]) => void = () => undefined;
    function Host(): ReactElement {
      const [rows, setRows] = useState(first);
      set = setRows;
      return createElement(ActivityRail, {
        activity: ofRows(rows),
        onJump: () => undefined,
        onAsk: () => undefined,
        ...(onDeleteSchedule === null ? {} : { onDeleteSchedule }),
        deleteOff,
      });
    }
    const container = mount(createElement(Host));
    return { container, set: (rows: ScheduleRow[]) => act(() => set(rows)) };
  }

  // Mutation: put Delete inside the jump button → a button inside a button, one tab stop.
  it('a schedule row is two buttons, jump then Delete; the confirm says it will not run again and starts on Keep', async () => {
    const onDelete = vi.fn();
    const row = scheduleRow('s1', 'Check the deploy', 'waiting');
    const { container } = harness([row], onDelete);
    expect(
      Array.from(container.querySelector('li')!.querySelectorAll('button'), (b) => b.getAttribute('aria-label')),
    ).toEqual([fill(a.jump, { title: 'Check the deploy' }), fill(a.deleteSchedule, { title: 'Check the deploy' })]);
    expect(jumpOf('Check the deploy')!.textContent).toContain(`${a.kinds.schedule} · every 5 minutes`);
    act(() => deleteOf('Check the deploy')!.click());
    await waitFor(() => dialog() !== null);
    expect(dialog()!.textContent).toContain(a.deleteTitle);
    expect(dialog()!.textContent).toContain(a.deleteText);
    expect(dialog()!.textContent).not.toContain(a.deleteOverdue);
    expect(document.activeElement?.textContent).toBe(a.deleteKeep);
    act(() => byRole(dialog()!, 'button', a.deleteKeep)!.click());
    await waitFor(() => dialog() === null);
    expect(onDelete).not.toHaveBeenCalled();
    act(() => deleteOf('Check the deploy')!.click());
    await waitFor(() => dialog() !== null);
    act(() => byRole(dialog()!, 'button', a.deleteConfirm)!.click());
    await waitFor(() => onDelete.mock.calls.length === 1);
    expect(onDelete).toHaveBeenCalledWith(row);
  });

  // Mutation: keep the Delete on an opening or removing row → a second delete can be sent while the first is out.
  it('each state has its words at the row end; opening and removing offer no Delete; with no handler none does', () => {
    harness([
      scheduleRow('w', 'waiting one', 'waiting'),
      scheduleRow('f', 'firing one', 'firing'),
      scheduleRow('p', 'paused one', 'paused'),
      scheduleRow('o', 'opening one', 'opening'),
      scheduleRow('r', 'removing one', 'removing'),
    ]);
    expect(jumpOf('waiting one')!.textContent).toContain('Next ');
    expect(jumpOf('firing one')!.textContent).toContain(a.runningNow);
    const paused = jumpOf('paused one')!;
    expect(paused.textContent).toContain(a.paused);
    expect(paused.querySelector(`[title="${a.pausedTip}"]`)?.textContent).toBe(a.paused);
    expect(jumpOf('opening one')!.textContent).toContain(a.opening);
    expect(jumpOf('removing one')!.textContent).toContain(a.removing);
    expect(['waiting one', 'firing one', 'paused one'].map((t) => deleteOf(t) !== null)).toEqual([true, true, true]);
    expect(deleteOf('opening one')).toBeNull();
    expect(deleteOf('removing one')).toBeNull();
    unmountAll();
    harness([scheduleRow('w', 'waiting one', 'waiting')], null);
    expect(deleteOf('waiting one')).toBeNull();
  });

  // Mutation: show the overdue line for any overdue row → a waiting schedule's confirm warns of a load it never does.
  it('the overdue line shows only for a paused schedule whose time passed', async () => {
    const past = Date.now() - 60_000;
    harness([
      scheduleRow('a', 'overdue paused', 'paused', past),
      scheduleRow('b', 'later paused', 'paused'),
      scheduleRow('c', 'overdue waiting', 'waiting', past),
    ]);
    expect(jumpOf('overdue waiting')!.textContent).toContain(a.due);
    const confirmText = async (title: string): Promise<string> => {
      act(() => deleteOf(title)!.click());
      await waitFor(() => dialog() !== null);
      const text = dialog()!.textContent ?? '';
      act(() => byRole(dialog()!, 'button', a.deleteKeep)!.click());
      await waitFor(() => dialog() === null);
      return text;
    };
    expect(await confirmText('overdue paused')).toContain(a.deleteOverdue);
    expect(await confirmText('later paused')).not.toContain(a.deleteOverdue);
    expect(await confirmText('overdue waiting')).not.toContain(a.deleteOverdue);
  });

  // Mutation: render the off Delete as the confirm → a press while the engine restarted loads the agent anyway.
  it('an off Delete stays, carries its reason as tooltip and description, and a press does nothing', async () => {
    const onDelete = vi.fn();
    harness([scheduleRow('s1', 'Check the deploy', 'waiting')], onDelete, CHAT.history.engineLost);
    const button = deleteOf('Check the deploy')!;
    expect(button.getAttribute('aria-disabled')).toBe('true');
    const described = document.getElementById(button.getAttribute('aria-describedby') ?? '');
    expect(described?.textContent).toBe(CHAT.history.engineLost);
    act(() => button.click());
    await flush();
    expect(dialog()).toBeNull();
    expect(onDelete).not.toHaveBeenCalled();
  });

  // Mutation: drop the row's leave cleanup → focus falls to the page body when a deleted row leaves.
  it('a paused row reads Opening… then Removing…; when it leaves, focus goes to the next row, else the heading', async () => {
    const one = scheduleRow('s1', 'Check the deploy', 'paused');
    const two = scheduleRow('s2', 'Nightly report', 'paused');
    const { container, set } = harness([one, two]);
    deleteOf('Check the deploy')!.focus();
    set([{ ...one, state: 'opening' }, two]);
    await waitFor(() => document.activeElement === jumpOf('Check the deploy'));
    expect(jumpOf('Check the deploy')!.textContent).toContain(a.opening);
    set([{ ...one, state: 'removing' }, two]);
    expect(jumpOf('Check the deploy')!.textContent).toContain(a.removing);
    set([two]);
    await waitFor(() => document.activeElement === jumpOf('Nightly report'));
    jumpOf('Nightly report')!.focus();
    set([]);
    await waitFor(() => document.activeElement === container.querySelector('[data-activity-rail]'));
  });
});

describe('ActivityRail: the goal (#17)', () => {
  const goalEntry = (over: Partial<Extract<TranscriptEntry, { kind: 'goal' }>> = {}) =>
    ({
      kind: 'goal',
      ts: 1,
      goalId: 'g1',
      objective: 'Fix the flaky cache tests',
      status: 'active',
      phase: 'executing',
      tokensUsed: 31_000,
      elapsedMs: 724_000,
      asOf: Date.now(),
      ...over,
    }) as const;
  const withGoal = (over: Partial<Extract<TranscriptEntry, { kind: 'goal' }>> = {}): Activity => {
    const goal = activityOf({ entries: [user(0), goalEntry(over)], runs: [], turnRunning: false }).goal!;
    return { ...SOME, needs: NEEDS, goal };
  };

  // Mutation: draw the goal after the plan → the rail's order is Needs you, Plan, Goal.
  it('sits after Needs you and before Plan, docked and in the drawer; its row jumps to where it began', () => {
    for (const docked of [true, false]) {
      const onJump = vi.fn();
      const container = mount(
        createElement(ActivityRail, { activity: withGoal(), onJump, onAsk: () => undefined, docked }),
      );
      const sections = [...container.querySelectorAll('section')];
      expect(sections.map((s) => s.querySelector('h2, button')?.textContent?.slice(0, 4))).toEqual([
        CHAT.needs.title.slice(0, 4),
        CHAT.activity.goal.slice(0, 4),
        CHAT.activity.plan.slice(0, 4),
        CHAT.activity.nowRunning.slice(0, 4),
      ]);
      const row = byRole(container, 'button', CHAT.activity.goalJump)!;
      expect(row.textContent).toContain('Fix the flaky cache tests');
      expect(row.textContent).toContain(`${CHAT.goal.state.working} · ${fill(CHAT.goal.tokens, { n: '31k' })}`);
      act(() => row.click());
      expect(onJump).toHaveBeenCalledWith(1);
      unmountAll();
    }
  });

  it('a paused goal shows with no clock; a cleared one shows none; alone it is not the empty line', () => {
    const paused = mount(
      createElement(ActivityRail, {
        activity: {
          needs: [],
          needsRuns: [],
          plan: null,
          running: [],
          scheduled: [],
          goal: withGoal({ status: 'user_paused' }).goal,
        },
        onJump: () => undefined,
        onAsk: () => undefined,
      }),
    );
    const row = byRole(paused, 'button', CHAT.activity.goalJump)!;
    expect(row.textContent).toBe(`Fix the flaky cache tests${CHAT.goal.state.paused}`);
    expect(paused.textContent).not.toContain(CHAT.activity.empty);
    const none = activityOf({ entries: [user(0), goalEntry({ status: 'cleared' })], runs: [], turnRunning: false });
    expect(none.goal).toBeNull();
  });
});
