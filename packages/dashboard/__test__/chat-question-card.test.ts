/**
 * @vitest-environment happy-dom
 */

/**
 * The question card (`ui/src/pages/chat/work/question.tsx`): one question per step, options, Other, Skip. Nothing is
 * picked on arrival, Send and Enter-to-send count only after the arm delay, and the answer goes as option indexes.
 */

import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { QuestionInteraction } from '../src/chat/types.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { CARD_ARM_MS, type InteractionAnswer } from '../ui/src/pages/chat/work/permission.js';
import {
  answerOf,
  keepQuestionDrafts,
  optionView,
  previewOf,
  QuestionCard,
} from '../ui/src/pages/chat/work/question.js';
import { accessibleName } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;
let offset = 0;

afterEach(() => {
  vi.restoreAllMocks();
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
  document.body.innerHTML = '';
});

function useClock(): void {
  offset = 0;
  const real = performance.now.bind(performance);
  vi.spyOn(performance, 'now').mockImplementation(() => real() + offset);
}

function advance(ms: number): void {
  offset += ms;
}

const DB = {
  text: 'Which database should the cart use?',
  multi: false,
  options: [
    { label: 'Postgres (Recommended)', description: 'Relational. Already used by the orders service.', preview: null },
    { label: 'Redis', description: 'In memory. Fast, but the cart is lost on restart.', preview: null },
  ],
};
const CHECKS = {
  text: 'Which checks should run before a merge?',
  multi: true,
  options: [
    { label: 'Unit tests', description: 'Fast. Under a minute.', preview: null },
    { label: 'Type check', description: 'tsc over all packages.', preview: null },
    { label: 'Lint', description: '', preview: null },
  ],
};
const DEPLOY = {
  text: 'Deploy after merge?',
  multi: false,
  options: [{ label: 'Yes', description: '', preview: null }],
};

const LAYOUT = {
  text: 'Which layout should the shop use?',
  multi: false,
  options: [
    {
      label: 'Sidebar layout (Recommended)',
      description: 'Nav on the left.',
      preview: '┌──────┬─────────┐\n│ nav  │ content │',
    },
    { label: 'Top bar layout', description: 'Nav on top.', preview: '<b>top</b>\n┌───────────────┐' },
    { label: 'No nav', description: 'One page.', preview: null },
  ],
};

let ids = 0;

function card(
  questions: QuestionInteraction['questions'],
  extra: Partial<QuestionInteraction> = {},
): QuestionInteraction {
  ids += 1;
  return {
    id: `ask-${ids}`,
    kind: 'question',
    toolCallId: 'toolu_1',
    openedAt: 0,
    planMode: false,
    questions,
    ...extra,
  };
}

interface Mounted {
  root: HTMLElement;
  sent: InteractionAnswer[];
  status: string[];
  composer: HTMLTextAreaElement;
}

function mount(
  questions: QuestionInteraction['questions'] | QuestionInteraction,
  onAnswer?: (answer: InteractionAnswer) => Promise<void>,
): Mounted {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const composer = document.createElement('textarea');
  document.body.appendChild(composer);
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  const sent: InteractionAnswer[] = [];
  const status: string[] = [];
  act(() =>
    root.render(
      createElement(QuestionCard, {
        interaction: Array.isArray(questions) ? card(questions) : questions,
        folder: '/Users/me/code/shop',
        position: { index: 1, count: 1 },
        composerRef: { current: composer },
        onStatus: (text: string) => status.push(text),
        onAnswer: (answer: InteractionAnswer) => {
          sent.push(answer);
          return onAnswer?.(answer) ?? Promise.resolve();
        },
      }),
    ),
  );
  mounted = { root, container };
  return { root: container, sent, status, composer };
}

function rows(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('[data-row]')];
}

function checked(root: HTMLElement): boolean[] {
  return rows(root).map((row) => row.getAttribute('aria-checked') === 'true');
}

function button(root: HTMLElement, name: string): HTMLButtonElement | undefined {
  return [...root.querySelectorAll<HTMLButtonElement>('button')].find((b) => accessibleName(b) === name);
}

function disabled(element: HTMLButtonElement): boolean {
  return element.disabled || element.getAttribute('aria-disabled') === 'true';
}

function click(element: HTMLElement): void {
  act(() => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  });
}

function key(target: Element, name: string, init: KeyboardEventInit = {}): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init }));
  });
}

function type(area: HTMLTextAreaElement, text: string): void {
  act(() => {
    Reflect.set(HTMLTextAreaElement.prototype, 'value', text, area);
    area.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function unmount(): void {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
  document.body.innerHTML = '';
}

function preview(root: HTMLElement): { caption: string; text: string; label: string | null } | null {
  const box = root.querySelector<HTMLElement>('[data-ask-preview]');
  if (box === null) return null;
  const well = box.querySelector<HTMLElement>('pre')!;
  return {
    caption: box.querySelector('p')!.textContent!,
    text: well.textContent!,
    label: well.getAttribute('aria-label'),
  };
}

async function settle(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

describe('optionView and answerOf', () => {
  it('a "(Recommended)" label shows as a tag, never as a pick', () => {
    expect(optionView(DB.options[0]!)).toEqual({
      label: 'Postgres',
      description: 'Relational. Already used by the orders service.',
      tag: CHAT.ask.recommended,
    });
    expect(optionView({ label: '(recommended)', description: '', preview: null })).toEqual({
      label: '(recommended)',
      description: '',
    });
  });

  it('single: one pick, or the own text alone; multi: picks and the own text; blank text is no answer', () => {
    expect(answerOf({ picks: [1], otherOn: false, otherText: 'x' }, DB)).toEqual({ picks: [1] });
    expect(answerOf({ picks: [], otherOn: true, otherText: ' Dynamo ' }, DB)).toEqual({ picks: [], other: 'Dynamo' });
    expect(answerOf({ picks: [], otherOn: true, otherText: '  ' }, DB)).toBeNull();
    expect(answerOf({ picks: [0, 2], otherOn: true, otherText: 'e2e' }, CHECKS)).toEqual({
      picks: [0, 2],
      other: 'e2e',
    });
    expect(answerOf({ picks: [1], otherOn: false, otherText: 'e2e' }, CHECKS)).toEqual({ picks: [1] });
    expect(answerOf({ picks: [], otherOn: true, otherText: '' }, CHECKS)).toBeNull();
  });
});

describe('QuestionCard', () => {
  it('Q1: one question, nothing picked on arrival (also with a Recommended option), focus on the first option', () => {
    useClock();
    const { root, sent } = mount([DB]);
    const group = root.querySelector('[data-interaction-card]')!;
    expect(group.getAttribute('role')).toBe('group');
    expect(group.textContent).toContain(CHAT.ask.title);
    expect(group.textContent).toContain('code/shop');
    expect(root.querySelector('[role="radiogroup"]')?.getAttribute('aria-label')).toBe(DB.text);
    expect(rows(root).map((row) => row.textContent)).toEqual([
      `1Postgres${CHAT.ask.recommended}Relational. Already used by the orders service.`,
      '2RedisIn memory. Fast, but the cart is lost on restart.',
      `3${CHAT.ask.other}`,
    ]);
    expect(checked(root)).toEqual([false, false, false]);
    expect(document.activeElement).toBe(rows(root)[0]);
    expect(root.querySelector('ol')).toBeNull();
    expect(button(root, CHAT.ask.skip)).toBeDefined();
    expect(button(root, CHAT.ask.next)).toBeUndefined();
    expect(disabled(button(root, CHAT.ask.send)!)).toBe(true);
    expect(group.textContent).toContain(CHAT.ask.skipNote);
    expect(group.textContent).toContain(CHAT.ask.waitNote);
    const read = (group.getAttribute('aria-describedby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent ?? '')
      .join('\n');
    expect(read).toContain(CHAT.ask.skipNote);
    expect(read).toContain(CHAT.ask.waitNote);
    advance(CARD_ARM_MS);
    click(button(root, CHAT.ask.send)!);
    expect(sent).toEqual([]);
  });

  it('arrival leaves focus in the composer while the user types there', () => {
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const composer = document.createElement('textarea');
    document.body.appendChild(composer);
    composer.value = 'half a message';
    composer.focus();
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() =>
      root.render(
        createElement(QuestionCard, {
          interaction: card([DB]),
          folder: null,
          position: { index: 1, count: 1 },
          composerRef: { current: composer },
          onAnswer: () => Promise.resolve(),
        }),
      ),
    );
    mounted = { root, container };
    expect(document.activeElement).toBe(composer);
  });

  it('Enter on an unpicked option only picks; Enter and Send count only after the arm delay; the body is indexes', () => {
    useClock();
    const { root, sent } = mount([DB]);
    // A stray Enter from the card before: no pick, no send.
    key(rows(root)[1]!, 'Enter');
    expect(checked(root)).toEqual([false, false, false]);
    act(() => rows(root)[1]!.click());
    expect(checked(root)).toEqual([false, true, false]);
    key(rows(root)[1]!, 'Enter');
    click(button(root, CHAT.ask.send)!);
    expect(sent).toEqual([]);
    advance(CARD_ARM_MS);
    key(rows(root)[0]!, 'Enter');
    expect(checked(root)).toEqual([true, false, false]);
    expect(sent).toEqual([]);
    key(rows(root)[0]!, 'Enter');
    expect(sent).toEqual([{ answers: [{ picks: [0] }] }]);
  });

  it('keys 1–9 pick while focus is in the card and not in a text field; the Other key moves focus into its field', () => {
    useClock();
    const { root } = mount([DB]);
    const section = root.querySelector('[data-interaction-card]')!;
    key(rows(root)[0]!, '2');
    expect(checked(root)).toEqual([false, true, false]);
    expect(document.activeElement).toBe(rows(root)[1]);
    key(section, '9');
    expect(checked(root)).toEqual([false, true, false]);
    key(rows(root)[1]!, '3');
    expect(checked(root)).toEqual([false, false, true]);
    const field = root.querySelector<HTMLTextAreaElement>('[data-other-field]')!;
    expect(document.activeElement).toBe(field);
    key(field, '1');
    expect(checked(root)).toEqual([false, false, true]);
    key(rows(root)[0]!, '1', { metaKey: true });
    expect(checked(root)).toEqual([false, false, true]);
  });

  it('Other alone is an answer once it has text', () => {
    useClock();
    const { root, sent } = mount([DB]);
    act(() => rows(root)[2]!.click());
    const field = root.querySelector<HTMLTextAreaElement>('[data-other-field]')!;
    expect(document.activeElement).toBe(field);
    expect(disabled(button(root, CHAT.ask.send)!)).toBe(true);
    type(field, '  DynamoDB ');
    expect(disabled(button(root, CHAT.ask.send)!)).toBe(false);
    advance(CARD_ARM_MS);
    key(field, 'Enter');
    expect(sent).toEqual([{ answers: [{ picks: [], other: 'DynamoDB' }] }]);
  });

  it('Q2: steps, multi select with Other, Back keeps picks, the last step sends; an unanswered one goes as null', () => {
    useClock();
    const { root, sent, status } = mount([DB, CHECKS, DEPLOY]);
    const steps = root.querySelector('ol')!;
    expect(steps.getAttribute('aria-label')).toBe(CHAT.ask.stepsLabel);
    expect(root.textContent).toContain(fill(CHAT.ask.titleMany, { n: 3 }));
    expect(steps.querySelector('[aria-current="step"]')?.textContent).toContain('1');
    expect(button(root, CHAT.ask.skipMany)).toBeDefined();
    expect(button(root, CHAT.ask.back)).toBeUndefined();
    act(() => rows(root)[0]!.click());
    click(button(root, CHAT.ask.next)!);
    expect(status).toEqual([fill(CHAT.status.askStep, { i: 2, n: 3 })]);
    expect(root.textContent).toContain(fill(CHAT.ask.step, { i: 2, n: 3 }));
    expect(root.textContent).toContain(CHAT.ask.multiHint);
    expect(root.querySelector('[role="group"][aria-label]')?.getAttribute('aria-label')).toBe(CHECKS.text);
    expect(checked(root)).toEqual([false, false, false, false]);
    expect(document.activeElement).toBe(rows(root)[0]);
    key(rows(root)[0]!, ' ');
    key(rows(root)[0]!, '3');
    expect(checked(root)).toEqual([true, false, true, false]);
    key(rows(root)[0]!, '4');
    const field = root.querySelector<HTMLTextAreaElement>('[data-other-field]')!;
    expect(document.activeElement).toBe(field);
    type(field, 'also run the e2e smoke test');
    expect(checked(root)).toEqual([true, false, true, true]);
    click(button(root, CHAT.ask.back)!);
    expect(checked(root)).toEqual([true, false, false]);
    click(button(root, CHAT.ask.next)!);
    expect(checked(root)).toEqual([true, false, true, true]);
    click(button(root, CHAT.ask.next)!);
    expect(button(root, CHAT.ask.next)).toBeUndefined();
    expect(root.textContent).toContain(fill(CHAT.ask.answeredCount, { done: 2, n: 3 }));
    // Send on the last step is armed like Allow.
    click(button(root, CHAT.ask.sendMany)!);
    expect(sent).toEqual([]);
    advance(CARD_ARM_MS);
    click(button(root, CHAT.ask.sendMany)!);
    expect(sent).toEqual([
      { answers: [{ picks: [0] }, { picks: [0, 2], other: 'also run the e2e smoke test' }, null] },
    ]);
  });

  it('Skip sends "skip"; a failed send turns the buttons on again', async () => {
    useClock();
    let fail = true;
    const { root, sent } = mount([DB], () => (fail ? Promise.reject(new Error('down')) : Promise.resolve()));
    click(button(root, CHAT.ask.skip)!);
    expect(sent).toEqual([{ reply: 'skip' }]);
    await settle();
    expect(disabled(button(root, CHAT.ask.skip)!)).toBe(false);
    fail = false;
    act(() => rows(root)[1]!.click());
    advance(CARD_ARM_MS);
    click(button(root, CHAT.ask.send)!);
    expect(sent.at(-1)).toEqual({ answers: [{ picks: [1] }] });
    expect(root.querySelector('[data-interaction-card]')?.getAttribute('aria-busy')).toBe('true');
  });

  it('Q3: the focused or picked option shows its preview as plain text under the list; single select only', () => {
    useClock();
    const { root, sent } = mount([LAYOUT, { ...CHECKS, options: CHECKS.options.map((o) => ({ ...o, preview: 'p' })) }]);
    // Arrival: focus is on option 1 and nothing is picked; its preview shows.
    expect(checked(root)).toEqual([false, false, false, false]);
    const label = fill(CHAT.ask.preview, { label: 'Sidebar layout' });
    expect(preview(root)).toEqual({ caption: label, text: LAYOUT.options[0]!.preview, label });
    // Arrow keys move and pick: the next option's preview, drawn as text.
    key(rows(root)[0]!, 'ArrowDown');
    expect(checked(root)).toEqual([false, true, false, false]);
    expect(preview(root)?.text).toBe('<b>top</b>\n┌───────────────┐');
    expect(root.querySelector('[data-ask-preview] b')).toBeNull();
    // An option with no preview shows none; focus on Other falls back to the picked option.
    key(rows(root)[1]!, 'ArrowDown');
    expect(preview(root)).toBeNull();
    // A click focuses the row it picks.
    act(() => {
      rows(root)[1]!.focus();
      rows(root)[1]!.click();
    });
    expect(preview(root)?.caption).toBe(fill(CHAT.ask.preview, { label: 'Top bar layout' }));
    act(() => rows(root)[3]!.focus());
    expect(preview(root)?.caption).toBe(fill(CHAT.ask.preview, { label: 'Top bar layout' }));
    // Focus out of the list: the picked option's preview stays.
    act(() => button(root, CHAT.ask.skipMany)!.focus());
    expect(preview(root)?.caption).toBe(fill(CHAT.ask.preview, { label: 'Top bar layout' }));
    // A multi-select question shows no preview.
    click(button(root, CHAT.ask.next)!);
    expect(root.querySelector('[data-ask-preview]')).toBeNull();
    act(() => rows(root)[0]!.click());
    expect(root.querySelector('[data-ask-preview]')).toBeNull();
    advance(CARD_ARM_MS);
    click(button(root, CHAT.ask.sendMany)!);
    expect(sent).toEqual([{ answers: [{ picks: [1] }, { picks: [0] }] }]);
  });

  it('previewOf: the focused option, else the picked one; none for Other, for an option with none, or multi', () => {
    const none = { picks: [], otherOn: false, otherText: '' };
    expect(previewOf(LAYOUT, none, null)).toBeNull();
    expect(previewOf(LAYOUT, none, 1)).toBe(1);
    expect(previewOf(LAYOUT, { ...none, picks: [1] }, null)).toBe(1);
    expect(previewOf(LAYOUT, { ...none, picks: [1] }, 0)).toBe(0);
    expect(previewOf(LAYOUT, { ...none, picks: [1] }, 3)).toBe(1);
    expect(previewOf(LAYOUT, { ...none, picks: [1] }, 2)).toBeNull();
    expect(previewOf(LAYOUT, { picks: [], otherOn: true, otherText: 'x' }, 3)).toBeNull();
    expect(previewOf({ ...LAYOUT, multi: true }, none, 0)).toBeNull();
  });

  it('Q4: plan mode adds Talk it over and Plan now; each sends the answers so far', async () => {
    useClock();
    const plain = mount([DB]);
    expect(button(plain.root, CHAT.ask.discuss)).toBeUndefined();
    expect(button(plain.root, CHAT.ask.planNow)).toBeUndefined();
    expect(plain.root.textContent).not.toContain(CHAT.ask.discussNote);
    unmount();

    let fail = true;
    const { root, sent } = mount(card([DB, CHECKS], { planMode: true }), () =>
      fail ? Promise.reject(new Error('down')) : Promise.resolve(),
    );
    const group = root.querySelector('[data-interaction-card]')!;
    const actions = [...root.querySelectorAll<HTMLButtonElement>('button')].map((b) => accessibleName(b));
    expect(actions.slice(0, 3)).toEqual([CHAT.ask.skipMany, CHAT.ask.discuss, CHAT.ask.planNow]);
    for (const note of [CHAT.ask.skipNote, CHAT.ask.discussNote, CHAT.ask.planNowNote, CHAT.ask.waitNote]) {
      expect(group.textContent).toContain(note);
    }
    const read = (group.getAttribute('aria-describedby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent ?? '')
      .join('\n');
    expect(read).toContain(CHAT.ask.discussNote);
    expect(read).toContain(CHAT.ask.planNowNote);
    // Nothing answered yet: still a plan reply, with every question open.
    click(button(root, CHAT.ask.discuss)!);
    expect(sent).toEqual([{ reply: 'discuss', answers: [null, null] }]);
    expect(disabled(button(root, CHAT.ask.planNow)!)).toBe(true);
    await settle();
    expect(disabled(button(root, CHAT.ask.planNow)!)).toBe(false);
    fail = false;
    act(() => rows(root)[1]!.click());
    click(button(root, CHAT.ask.next)!);
    key(rows(root)[0]!, '1');
    key(rows(root)[0]!, '4');
    type(root.querySelector<HTMLTextAreaElement>('[data-other-field]')!, 'and e2e');
    click(button(root, CHAT.ask.planNow)!);
    expect(sent.at(-1)).toEqual({ reply: 'plan-now', answers: [{ picks: [1] }, { picks: [0], other: 'and e2e' }] });
    expect(root.querySelector('[data-interaction-card]')?.getAttribute('aria-busy')).toBe('true');
  });

  it('draft kept: the step and the picks come back when the card is shown again, until it closes', async () => {
    useClock();
    const interaction = card([DB, CHECKS, DEPLOY]);
    const first = mount(interaction);
    act(() => rows(first.root)[1]!.click());
    click(button(first.root, CHAT.ask.next)!);
    key(rows(first.root)[0]!, '3');
    key(rows(first.root)[0]!, '4');
    type(first.root.querySelector<HTMLTextAreaElement>('[data-other-field]')!, 'smoke');
    // Another conversation opens: the card leaves the page.
    keepQuestionDrafts('c-1', [interaction.id]);
    unmount();
    keepQuestionDrafts('c-2', []);
    keepQuestionDrafts('c-1', [interaction.id]);
    const again = mount(interaction);
    expect(again.root.querySelector('[aria-current="step"]')?.textContent).toContain('2');
    expect(again.root.querySelector('[role="group"][aria-label]')?.getAttribute('aria-label')).toBe(CHECKS.text);
    expect(checked(again.root)).toEqual([false, false, true, true]);
    expect(again.root.querySelector<HTMLTextAreaElement>('[data-other-field]')!.value).toBe('smoke');
    expect(document.activeElement).toBe(rows(again.root)[0]);
    click(button(again.root, CHAT.ask.back)!);
    expect(checked(again.root)).toEqual([false, true, false]);
    // The send counts only after the arm delay of the card shown again.
    click(button(again.root, CHAT.ask.next)!);
    click(button(again.root, CHAT.ask.next)!);
    click(button(again.root, CHAT.ask.sendMany)!);
    expect(again.sent).toEqual([]);
    advance(CARD_ARM_MS);
    click(button(again.root, CHAT.ask.sendMany)!);
    expect(again.sent).toEqual([{ answers: [{ picks: [1] }, { picks: [2], other: 'smoke' }, null] }]);
    await settle();
    unmount();
    // Answered: shown again it starts empty.
    const after = mount(interaction);
    expect(after.root.querySelector('[role="radiogroup"]')?.getAttribute('aria-label')).toBe(DB.text);
    expect(checked(after.root)).toEqual([false, false, false]);
  });

  it("a card that closed with no answer drops its draft; other conversations' cards keep theirs", () => {
    const stopped = card([DB]);
    const elsewhere = card([DB]);
    keepQuestionDrafts('c-1', [stopped.id]);
    keepQuestionDrafts('c-2', [elsewhere.id]);
    const one = mount(stopped);
    act(() => rows(one.root)[0]!.click());
    unmount();
    const two = mount(elsewhere);
    act(() => rows(two.root)[1]!.click());
    unmount();
    keepQuestionDrafts('c-1', [stopped.id]);
    expect(checked(mount(stopped).root)).toEqual([true, false, false]);
    unmount();
    keepQuestionDrafts('c-1', []);
    expect(checked(mount(stopped).root)).toEqual([false, false, false]);
    unmount();
    expect(checked(mount(elsewhere).root)).toEqual([false, true, false]);
  });

  it('the texts are drawn as plain text', () => {
    useClock();
    const { root } = mount([
      { text: '<b>bold?</b>', multi: false, options: [{ label: '<i>x</i>', description: '<u>d</u>', preview: null }] },
    ]);
    expect(root.querySelector('b, i, u')).toBeNull();
    expect(root.textContent).toContain('<b>bold?</b>');
    expect(root.textContent).toContain('<i>x</i>');
  });
});
