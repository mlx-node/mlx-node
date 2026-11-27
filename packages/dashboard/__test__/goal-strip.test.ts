/**
 * @vitest-environment happy-dom
 */

/**
 * The goal strip: what each state says, its buttons, the clock and the tokens while it runs, the
 * narrow drops; the thread row where the goal began; the lines said once at each state change; the `/goal` row's tip.
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CommandMenu } from '../ui/src/components/arc-x/command-menu/command-menu.js';
import { GoalRow, GoalStrip } from '../ui/src/components/arc-x/goal-strip/goal-strip.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import {
  goalMeta,
  goalOf,
  goalRowState,
  goalStateWords,
  goalStatusLine,
  goalTokens,
  openGoalAt,
  type GoalEntry,
  type GoalFacts,
  type GoalView,
} from '../ui/src/lib/goal.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import { rewindGoalOff } from '../ui/src/pages/chat/history-actions.js';
import { slashRows } from '../ui/src/pages/chat/slash.js';
import { cssRules } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { accessibleName, byRole, waitFor } from './render.js';

afterEach(unmountAll);

const OBJECTIVE = 'Fix the flaky cache tests';
const T = 1_791_643_279_000;
const HOT: GoalFacts = { hot: true, waiting: false };

function goal(over: Partial<GoalEntry> = {}): GoalEntry {
  return {
    kind: 'goal',
    ts: T - 60_000,
    goalId: 'g1',
    objective: OBJECTIVE,
    status: 'active',
    phase: 'executing',
    tokensUsed: 31_400,
    elapsedMs: 724_000,
    asOf: T,
    ...over,
  };
}

const typed = (ts: number, text = 'go', extra: Partial<Extract<TranscriptEntry, { kind: 'user' }>> = {}) =>
  ({ kind: 'user', ts, text, promptIndex: 0, ...extra }) as TranscriptEntry;

function viewOf(over: Partial<GoalEntry> = {}, facts: GoalFacts = HOT): GoalView {
  const view = goalOf([typed(T - 10, `/goal ${OBJECTIVE}`), goal(over)], facts);
  if (view === null) throw new Error('no goal');
  return view;
}

/** The state words, clock and tokens the strip shows at `T + 5 s`. */
function said(view: GoalView): { words: string; clock: string | null; tokens: string | null } {
  return { words: goalStateWords(view), ...goalMeta(view, T + 5_000) };
}

describe('the goal as the strip reads it (D1-D11)', () => {
  // Mutation: count the clock from `ts` instead of `asOf` → a goal resumed later shows its old start.
  it('while it runs: the state words, the clock counted on from its last line, the tokens; only Pause', () => {
    const cases: Array<[Partial<GoalEntry>, GoalFacts, ReturnType<typeof said>]> = [
      [{ phase: 'planning', planning: true }, HOT, { words: CHAT.goal.state.planning, clock: '12:09', tokens: null }],
      [{}, HOT, { words: CHAT.goal.state.working, clock: '12:09', tokens: fill(CHAT.goal.tokens, { n: '31k' }) }],
      [
        { verifying: true },
        HOT,
        { words: CHAT.goal.state.checking, clock: '12:09', tokens: fill(CHAT.goal.tokens, { n: '31k' }) },
      ],
      [
        { tokenBudget: 200_000 },
        HOT,
        { words: CHAT.goal.state.working, clock: '12:09', tokens: fill(CHAT.goal.tokensOf, { n: '31k', max: '200k' }) },
      ],
      [{}, { hot: true, waiting: true }, { words: CHAT.goal.state.waiting, clock: '12:09', tokens: null }],
    ];
    for (const [over, facts, expected] of cases) {
      const view = viewOf(over, facts);
      expect([over, said(view)]).toEqual([over, expected]);
      expect(view.running).toBe(true);
      expect(view.actions).toEqual(['pause']);
    }
  });

  it('stopped: the words, no clock, Resume and Clear (Clear alone once out of budget or done)', () => {
    const cases: Array<[Partial<GoalEntry>, GoalFacts, string, readonly string[]]> = [
      [{ status: 'user_paused', phase: 'idle' }, HOT, CHAT.goal.state.paused, ['resume', 'clear']],
      [
        { status: 'no_progress_paused', pauseMessage: 'No progress in 3 rounds.' },
        HOT,
        fill(CHAT.goal.state.pausedWhy, { reason: CHAT.goal.why.no_progress_paused }),
        ['resume', 'clear'],
      ],
      [
        { status: 'back_off_paused' },
        HOT,
        fill(CHAT.goal.state.pausedWhy, { reason: CHAT.goal.why.back_off_paused }),
        ['resume', 'clear'],
      ],
      [
        { status: 'infra_paused' },
        HOT,
        fill(CHAT.goal.state.pausedWhy, { reason: CHAT.goal.why.infra_paused }),
        ['resume', 'clear'],
      ],
      [
        { status: 'blocked', pauseMessage: 'The tests need a GPU.\nAsk the owner.' },
        HOT,
        fill(CHAT.goal.state.blocked, { reason: 'The tests need a GPU.' }),
        ['resume', 'clear'],
      ],
      [{ status: 'blocked' }, HOT, CHAT.goal.state.blockedBare, ['resume', 'clear']],
      [
        { status: 'blocked', pauseMessage: 'needs‮ a GPU' },
        HOT,
        fill(CHAT.goal.state.blocked, { reason: 'needs\\u{202E} a GPU' }),
        ['resume', 'clear'],
      ],
      [
        { status: 'budget_limited', tokenBudget: 200_000, tokensUsed: 201_300 },
        HOT,
        fill(CHAT.goal.state.budget, { tokens: '200k' }),
        ['clear'],
      ],
      [
        { status: 'complete', elapsedMs: 2_470_000, tokensUsed: 96_200 },
        HOT,
        fill(CHAT.goal.state.done, { time: '41:10', tokens: '96k' }),
        ['clear'],
      ],
      [{}, { hot: false, waiting: false }, CHAT.goal.state.cold, ['resume', 'clear']],
    ];
    for (const [over, facts, words, actions] of cases) {
      const view = viewOf(over, facts);
      expect([over, said(view), view.actions, view.running]).toEqual([
        over,
        { words, clock: null, tokens: null },
        actions,
        false,
      ]);
    }
  });

  // Mutation: hide a done goal at any later entry → the strip leaves at the agent's last words, before a message.
  it('done stays until a message is typed after it; a steer does not count; cleared shows none', () => {
    const done = goal({ status: 'complete', asOf: T + 100 });
    const answer: TranscriptEntry = { kind: 'assistant', ts: T + 200, text: 'All green.' };
    expect(goalOf([typed(T, 'x'), done, answer], HOT)?.state).toBe('done');
    expect(goalOf([typed(T, 'x'), done, typed(T + 300, 'thanks', { interjection: true })], HOT)?.state).toBe('done');
    expect(goalOf([typed(T, 'x'), done, typed(T + 300, 'not that', { note: true })], HOT)?.state).toBe('done');
    expect(goalOf([typed(T, 'x'), done, typed(T + 300, 'next')], HOT)).toBeNull();
    expect(goalOf([typed(T, 'x'), goal({ status: 'cleared' })], HOT)).toBeNull();
    // A new goal replaces the old one: only the last row counts.
    const two = [goal({ status: 'user_paused' }), typed(T + 1, 'y'), goal({ goalId: 'g2', objective: 'Ship it' })];
    expect(goalOf(two, HOT)?.entry.goalId).toBe('g2');
    expect(goalOf([], HOT)).toBeNull();
  });

  it('tokens: plain below 1,000, then thousands, then millions', () => {
    expect([950, 1_000, 31_999, 200_000, 1_234_567].map(goalTokens)).toEqual(['950', '1k', '31k', '200k', '1.2M']);
  });

  // Mutation: `rewindGoalOff` always null → the message that set the open goal can be rewound.
  it('rewind: off for a message before the open goal’s row; on after it, or once the goal is done', () => {
    const entries = [typed(T - 10, 'one'), typed(T - 5, `/goal ${OBJECTIVE}`), goal(), typed(T + 10, 'later')];
    const at = openGoalAt(entries);
    expect(at).toBe(2);
    expect([0, 1, 3].map((i) => rewindGoalOff(i, at))).toEqual([CHAT.history.offGoal, CHAT.history.offGoal, null]);
    expect(openGoalAt([typed(T, 'x'), goal({ status: 'complete' })])).toBeNull();
    expect(openGoalAt([typed(T, 'x'), goal({ status: 'cleared' })])).toBeNull();
  });
});

describe('GoalStrip', () => {
  function strip(view: GoalView, onAction = vi.fn(), pending: 'pause' | null = null) {
    return { container: mount(createElement(GoalStrip, { goal: view, pending, onAction })), onAction };
  }

  it('a group named Goal: the objective whole in its title, the state, the clock hidden from reading, Pause', () => {
    const { container, onAction } = strip(viewOf({ tokenBudget: 200_000 }));
    const group = byRole(container, 'group')!;
    expect(accessibleName(group)).toBe(CHAT.goal.label);
    expect(group.querySelector(`[title="${OBJECTIVE}"]`)?.textContent).toBe(OBJECTIVE);
    expect(group.textContent).toContain(CHAT.goal.state.working);
    expect(container.querySelector('[data-goal-clock]')?.getAttribute('aria-hidden')).toBe('true');
    expect(container.querySelector('[data-goal-tokens]')?.textContent).toBe(
      fill(CHAT.goal.tokensOf, { n: '31k', max: '200k' }),
    );
    const buttons = [...group.querySelectorAll('button')].map((b) => b.textContent);
    expect(buttons).toEqual([CHAT.goal.pause]);
    act(() => byRole(container, 'button', CHAT.goal.pause)!.click());
    expect(onAction).toHaveBeenCalledWith('pause');
  });

  it('paused by the agent: its reason under the line; a press that waits turns every button off', () => {
    const reason = 'No progress in 3 rounds: the same test fails.';
    const onAction = vi.fn();
    const { container } = strip(viewOf({ status: 'no_progress_paused', pauseMessage: reason }), onAction, 'pause');
    expect(container.textContent).toContain(reason);
    const buttons = [...container.querySelectorAll('button')].filter((b) => b.textContent !== '');
    expect(buttons.map((b) => b.textContent).slice(0, 2)).toEqual([CHAT.goal.resume, CHAT.goal.clear]);
    for (const button of buttons.slice(0, 2)) {
      expect(button.getAttribute('aria-disabled')).toBe('true');
      act(() => button.click());
    }
    expect(onAction).not.toHaveBeenCalled();
  });

  // Mutation: drop `busy` from the buttons → Resume posts while an answer runs and comes back as a failed toast.
  it('while an answer runs: Resume and Clear are off and say why; Pause (Stop) stays on', () => {
    const onAction = vi.fn();
    const paused = mount(
      createElement(GoalStrip, { goal: viewOf({ status: 'user_paused' }), pending: null, busy: true, onAction }),
    );
    for (const label of [CHAT.goal.resume, CHAT.goal.clear]) {
      const button = byRole(paused, 'button', label)!;
      expect(button.getAttribute('aria-disabled')).toBe('true');
      expect(document.getElementById(button.getAttribute('aria-describedby') ?? '')?.textContent).toBe(CHAT.modelBusy);
      act(() => button.click());
    }
    expect(onAction).not.toHaveBeenCalled();
    rerender(
      createElement(GoalStrip, { goal: viewOf({ status: 'user_paused' }), pending: null, busy: false, onAction }),
    );
    expect(byRole(paused, 'button', CHAT.goal.resume)!.getAttribute('aria-disabled')).toBeNull();
    act(() => byRole(paused, 'button', CHAT.goal.resume)!.click());
    expect(onAction).toHaveBeenCalledWith('resume');
    const running = mount(createElement(GoalStrip, { goal: viewOf(), pending: null, busy: true, onAction }));
    const pause = byRole(running, 'button', CHAT.goal.pause)!;
    expect(pause.getAttribute('aria-disabled')).toBeNull();
    act(() => pause.click());
    expect(onAction).toHaveBeenLastCalledWith('pause');
  });

  // Mutation: key the buttons by action → Pause's node leaves and the focus drops to the page.
  it('after Pause the focus stays on the same button, which now reads Resume', async () => {
    const onAction = vi.fn();
    const container = mount(createElement(GoalStrip, { goal: viewOf(), pending: null, onAction }));
    const pause = byRole(container, 'button', CHAT.goal.pause)!;
    act(() => pause.focus());
    rerender(createElement(GoalStrip, { goal: viewOf({ status: 'user_paused' }), pending: null, onAction }));
    expect(document.activeElement).toBe(pause);
    await waitFor(() => pause.textContent === CHAT.goal.resume);
  });

  it('narrow (D13): the tokens go first, then the word Goal and a running goal’s state words', () => {
    const rules = cssRules('components/arc-x/goal-strip/goal-strip.module.css');
    const hidden = rules
      .filter((rule) => rule.conditional && rule.decls.get('display') === 'none')
      .map((rule) => rule.selectors);
    expect(hidden).toEqual([['.tokens'], ['.label', '.box[data-running] .state']]);
    const { container } = strip(viewOf());
    expect(container.querySelector('[data-running]')).not.toBeNull();
    expect(container.querySelector('.label, [class*="label"]')?.getAttribute('aria-hidden')).toBe('true');
  });
});

describe('the thread row and the status lines', () => {
  it('the row says what the goal did last, by name too', () => {
    const cases: Array<[Partial<GoalEntry>, string]> = [
      [{ status: 'complete', elapsedMs: 2_470_000 }, fill(CHAT.goal.rowDone, { time: '41:10' })],
      [{ status: 'cleared' }, CHAT.goal.rowCleared],
      [{}, CHAT.goal.state.working],
      [{ status: 'no_progress_paused' }, fill(CHAT.goal.state.pausedWhy, { reason: CHAT.goal.why.no_progress_paused })],
    ];
    for (const [over, state] of cases) expect([over, goalRowState(goal(over))]).toEqual([over, state]);
    const container = mount(createElement(GoalRow, { entry: goal({ status: 'complete', elapsedMs: 2_470_000 }) }));
    const row = container.querySelector('article')!;
    expect(accessibleName(row)).toBe(fill(CHAT.goal.row, { objective: OBJECTIVE, state: 'Done in 41:10' }));
    expect(row.textContent).toContain(OBJECTIVE);
  });

  // Mutation: say a line for every goal line → the ticks would be read out.
  it('each change of state is said once; the same state, another goal or no goal before says nothing', () => {
    const before = { goalId: 'g1', status: 'active' };
    const now = (over: Partial<GoalEntry>) => goalStatusLine(before, [typed(T, 'x'), goal(over)]);
    expect(now({ status: 'user_paused' })).toBe(CHAT.goal.status.paused);
    expect(now({ status: 'infra_paused' })).toBe(CHAT.goal.status.paused);
    expect(now({ status: 'complete' })).toBe(CHAT.goal.status.done);
    expect(now({ status: 'cleared' })).toBe(CHAT.goal.status.cleared);
    expect(now({ status: 'blocked', pauseMessage: 'needs a GPU' })).toBe(
      fill(CHAT.goal.status.blocked, { reason: 'needs a GPU' }),
    );
    expect(now({ status: 'blocked' })).toBe(CHAT.goal.status.blockedBare);
    expect(now({ tokensUsed: 90_000 })).toBeNull();
    expect(now({ goalId: 'g2', status: 'user_paused' })).toBeNull();
    expect(goalStatusLine(null, [goal({ status: 'complete' })])).toBeNull();
  });
});

describe('the /goal row of the command list', () => {
  it('carries the tip, read with the row and shown on hover', () => {
    const row = slashRows('work', null).rows.find((r) => r.name === 'goal')!;
    expect(row.tip).toBe(CHAT.goal.commandTip);
    expect(
      slashRows('work', null)
        .rows.filter((r) => r.tip !== undefined)
        .map((r) => r.name),
    ).toEqual(['goal']);
    const container = mount(
      createElement(CommandMenu, {
        id: 'cmds',
        rows: [row],
        active: 0,
        optionId: (i: number) => `cmds-${String(i)}`,
        later: false,
        onPick: () => undefined,
        onHover: () => undefined,
      }),
    );
    const option = container.querySelector('[role="option"]')!;
    expect(option.getAttribute('title')).toBe(CHAT.goal.commandTip);
    expect(option.textContent).toContain(CHAT.goal.commandTip);
  });
});
