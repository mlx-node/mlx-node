/**
 * @vitest-environment happy-dom
 */

/**
 * The MCP input card (`ui/src/pages/chat/work/elicit.tsx`): the server's message and form, Decline and Send. Send
 * and Enter-to-send count only after the arm delay; values go by field (indexes for picks, a date and time with the
 * local offset); a refused Send marks each wrong field, says a summary at once and focuses the first wrong field. A
 * form the app cannot draw offers Decline only. Typed values outlive a remount of the card until it closes. The link
 * card shows the whole address and its site; Open page is a real link that opens and answers only after the arm
 * delay, once.
 */

import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import type { InteractionAnswerResult, McpField, McpInteraction } from '../src/chat/types.js';
import { rfc3339 } from '../ui/src/components/arc-x/schema-field/schema-field.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { McpCard, errorText, keepMcpDrafts, valuesOf } from '../ui/src/pages/chat/work/elicit.js';
import { asksSentence, CARD_ARM_MS, type InteractionAnswer } from '../ui/src/pages/chat/work/permission.js';
import { accessibleName } from './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;
let offset = 0;
let cards = 0;

afterEach(() => {
  vi.restoreAllMocks();
  unmount();
  document.body.innerHTML = '';
});

function unmount(): void {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
}

function useClock(): void {
  offset = 0;
  const real = performance.now.bind(performance);
  vi.spyOn(performance, 'now').mockImplementation(() => real() + offset);
}

function advance(ms: number): void {
  offset += ms;
}

const BASE = { description: null, required: false };
const FIELDS: McpField[] = [
  {
    ...BASE,
    name: 'repo',
    title: 'Repository',
    required: true,
    type: 'choice',
    options: [{ label: 'shop-api' }, { label: 'shop-web' }],
    initial: null,
  },
  {
    ...BASE,
    name: 'title',
    title: 'Title',
    description: 'One line.',
    required: true,
    type: 'text',
    format: null,
    minLength: 3,
    maxLength: null,
    initial: '',
  },
  {
    ...BASE,
    name: 'labels',
    title: 'Labels',
    type: 'choices',
    options: [{ label: 'bug' }, { label: 'feature' }],
    minItems: null,
    maxItems: null,
    initial: [],
  },
  { ...BASE, name: 'priority', title: 'Priority', type: 'integer', min: 1, max: 5, initial: '' },
  { ...BASE, name: 'mine', title: 'Assign to me', type: 'boolean', initial: true },
  {
    ...BASE,
    name: 'due',
    title: 'Due',
    type: 'text',
    format: 'date-time',
    minLength: null,
    maxLength: null,
    initial: '',
  },
];

/** A new id per card, so the page-wide drafts of one case never reach the next. */
function card(over: Partial<McpInteraction> = {}): McpInteraction {
  cards += 1;
  return {
    id: `mcp-${cards}`,
    kind: 'mcp-input',
    toolCallId: `mcp-elicit-${cards}`,
    openedAt: 0,
    mode: 'form',
    server: 'github',
    message: 'Which repository should the issue go to?',
    form: { fields: FIELDS, leftOut: 0 },
    ...over,
  };
}

interface Mounted {
  root: HTMLElement;
  sent: InteractionAnswer[];
  composer: HTMLTextAreaElement;
}

function mount(
  interaction: McpInteraction,
  onAnswer?: (answer: InteractionAnswer) => Promise<InteractionAnswerResult>,
  composer = document.createElement('textarea'),
): Mounted {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  if (!composer.isConnected) document.body.appendChild(composer);
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  const sent: InteractionAnswer[] = [];
  act(() =>
    root.render(
      createElement(McpCard, {
        interaction,
        folder: '/Users/me/code/shop',
        position: { index: 1, count: 1 },
        composerRef: { current: composer },
        onAnswer: (answer: InteractionAnswer) => {
          sent.push(answer);
          return onAnswer?.(answer) ?? Promise.resolve({});
        },
      }),
    ),
  );
  mounted = { root, container };
  return { root: container, sent, composer };
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

function key(target: Element, name: string): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true }));
  });
}

function field(root: HTMLElement, index: number): HTMLElement {
  return root.querySelector<HTMLElement>(`[data-field="${index}"]`)!;
}

function input(root: HTMLElement, index: number): HTMLInputElement {
  return field(root, index).querySelector<HTMLInputElement>('input')!;
}

function type(control: HTMLInputElement, value: string): void {
  act(() => {
    Reflect.set(HTMLInputElement.prototype, 'value', value, control);
    control.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

/** Fills the form: shop-web, "Cart", bug, 3, not me, a due date. */
function fill6(root: HTMLElement): void {
  act(() => field(root, 0).querySelectorAll<HTMLInputElement>('input')[1]!.click());
  type(input(root, 1), 'Cart');
  act(() => field(root, 2).querySelectorAll<HTMLInputElement>('input')[0]!.click());
  type(input(root, 3), '3');
  act(() => field(root, 4).querySelector<HTMLButtonElement>('button[role="switch"]')!.click());
  type(input(root, 5), '2026-08-19T10:00');
}

async function settle(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

describe('errorText and valuesOf', () => {
  it('each refusal has its words; a bound fills them', () => {
    expect(errorText({ code: 'required' })).toBe(CHAT.mcp.errors.required);
    expect(errorText({ code: 'minLength', n: 3 })).toBe(fill(CHAT.mcp.errors.minLength, { n: 3 }));
    expect(errorText({ code: 'maxItems', n: 2 })).toBe(fill(CHAT.mcp.errors.maxItems, { n: 2 }));
  });

  it('one value per field; a date and time gains the local offset; the rest go as typed', () => {
    expect(valuesOf(FIELDS, [1, ' Cart ', [0], '3', false, '2026-08-19T10:00'])).toEqual([
      1,
      ' Cart ',
      [0],
      '3',
      false,
      rfc3339('2026-08-19T10:00'),
    ]);
    expect(valuesOf(FIELDS, [null])).toEqual([null, null, null, null, null, null]);
  });
});

describe('McpCard', () => {
  it('M1: the server, its message and its form; notes; focus on the first control; Send waits for the arm delay', () => {
    useClock();
    const { root, sent } = mount(card());
    const group = root.querySelector('[data-interaction-card]')!;
    expect(group.getAttribute('role')).toBe('group');
    expect(group.textContent).toContain(fill(CHAT.mcp.formTitle, { server: 'github' }));
    expect(group.textContent).toContain(fill(CHAT.mcp.from, { server: 'github' }));
    const well = root.querySelector('[role="region"]')!;
    expect(accessibleName(well)).toBe(CHAT.mcp.message);
    expect(well.textContent).toBe('Which repository should the issue go to?');
    expect(group.textContent).toContain(fill(CHAT.mcp.formNote, { server: 'github' }));
    expect(group.textContent).toContain(CHAT.mcp.secretNote);
    expect(group.textContent).not.toContain(CHAT.mcp.leftOutOne);
    expect(root.querySelectorAll('[data-field]')).toHaveLength(6);
    expect(accessibleName(field(root, 0).querySelector('[role="radiogroup"]')!)).toBe(
      fill(CHAT.mcp.fieldName, { title: 'Repository', needed: CHAT.mcp.needed }),
    );
    expect(document.activeElement).toBe(field(root, 0).querySelector('input'));
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect(button(root, CHAT.mcp.decline)).toBeDefined();
    fill6(root);
    click(button(root, CHAT.mcp.send)!);
    expect(sent).toEqual([]);
    advance(CARD_ARM_MS);
    click(button(root, CHAT.mcp.send)!);
    expect(sent).toEqual([{ mcp: 'accept', values: [1, 'Cart', [0], '3', false, rfc3339('2026-08-19T10:00')] }]);
    expect(group.getAttribute('aria-busy')).toBe('true');
  });

  it('Enter in a one-line field sends once armed', () => {
    useClock();
    const { root, sent } = mount(card());
    type(input(root, 1), 'Cart');
    key(input(root, 1), 'Enter');
    expect(sent).toEqual([]);
    advance(CARD_ARM_MS);
    key(input(root, 1), 'Enter');
    expect(sent).toEqual([{ mcp: 'accept', values: [null, 'Cart', [], '', true, ''] }]);
  });

  // Mutation: drop the focus effect on `refusals` → focus stays on Send.
  it('M2: a refused Send marks each wrong field, says a summary, focuses the first wrong one; typing clears its mark', async () => {
    useClock();
    const { root, sent } = mount(card(), () =>
      Promise.resolve({ fields: [null, { code: 'minLength', n: 3 }, null, { code: 'max', n: 5 }, null, null] }),
    );
    type(input(root, 1), 'ab');
    type(input(root, 3), '9');
    advance(CARD_ARM_MS);
    const send = button(root, CHAT.mcp.send)!;
    act(() => send.focus());
    click(send);
    await settle();
    expect(sent).toHaveLength(1);
    const alert = root.querySelector('[role="alert"]')!;
    expect(alert.textContent).toBe(fill(CHAT.mcp.errorCount, { n: 2 }));
    const group = root.querySelector('[data-interaction-card]')!;
    expect(group.getAttribute('aria-describedby')?.split(' ')).toContain(alert.id);
    expect(input(root, 1).getAttribute('aria-invalid')).toBe('true');
    expect(field(root, 1).textContent).toContain(fill(CHAT.mcp.errors.minLength, { n: 3 }));
    expect(input(root, 3).getAttribute('aria-invalid')).toBe('true');
    expect(field(root, 3).textContent).toContain(fill(CHAT.mcp.errors.max, { n: 5 }));
    expect(input(root, 5).hasAttribute('aria-invalid')).toBe(false);
    expect(document.activeElement).toBe(input(root, 1));
    expect(disabled(button(root, CHAT.mcp.send)!)).toBe(false);
    type(input(root, 1), 'abc');
    expect(input(root, 1).hasAttribute('aria-invalid')).toBe(false);
    expect(root.querySelector('[role="alert"]')?.textContent).toBe(CHAT.mcp.errorCountOne);
  });

  it('M4: a form the app cannot draw offers Decline only, says why, and focuses Decline', () => {
    useClock();
    const { root, sent } = mount(card({ form: undefined, unusable: 'field' }));
    const group = root.querySelector('[data-interaction-card]')!;
    expect(group.textContent).toContain(fill(CHAT.mcp.unusable, { why: CHAT.mcp.why.field }));
    expect(group.textContent).not.toContain(CHAT.mcp.secretNote);
    expect(button(root, CHAT.mcp.send)).toBeUndefined();
    expect(root.querySelector('[data-field]')).toBeNull();
    const decline = button(root, CHAT.mcp.decline)!;
    expect(document.activeElement).toBe(decline);
    click(decline);
    expect(sent).toEqual([{ mcp: 'decline' }]);
  });

  it('an empty form: focus on Decline; Send gives no values', () => {
    useClock();
    const { root, sent } = mount(card({ form: { fields: [], leftOut: 0 } }));
    expect(document.activeElement).toBe(button(root, CHAT.mcp.decline));
    advance(CARD_ARM_MS);
    click(button(root, CHAT.mcp.send)!);
    expect(sent).toEqual([{ mcp: 'accept', values: [] }]);
  });

  it('fields left out are counted under the form', () => {
    const one = mount(card({ form: { fields: FIELDS.slice(0, 1), leftOut: 1 } }));
    expect(one.root.textContent).toContain(CHAT.mcp.leftOutOne);
    unmount();
    const many = mount(card({ form: { fields: FIELDS.slice(0, 1), leftOut: 3 } }));
    expect(many.root.textContent).toContain(fill(CHAT.mcp.leftOut, { n: 3 }));
  });

  it('typed values outlive a remount until the card is answered', async () => {
    useClock();
    const interaction = card();
    const first = mount(interaction);
    type(input(first.root, 1), 'Cart');
    unmount();
    const again = mount(interaction);
    expect(input(again.root, 1).value).toBe('Cart');
    advance(CARD_ARM_MS);
    click(button(again.root, CHAT.mcp.send)!);
    await settle();
    unmount();
    const after = mount(interaction);
    expect(input(after.root, 1).value).toBe('');
  });

  it("a card that closed with no answer drops its typed values; other conversations' cards keep theirs", () => {
    const stopped = card();
    const elsewhere = card();
    keepMcpDrafts('c-1', [stopped.id]);
    keepMcpDrafts('c-2', [elsewhere.id]);
    type(input(mount(stopped).root, 1), 'Cart');
    unmount();
    type(input(mount(elsewhere).root, 1), 'Other');
    unmount();
    keepMcpDrafts('c-1', [stopped.id]);
    expect(input(mount(stopped).root, 1).value).toBe('Cart');
    unmount();
    keepMcpDrafts('c-1', []);
    expect(input(mount(stopped).root, 1).value).toBe('');
    unmount();
    expect(input(mount(elsewhere).root, 1).value).toBe('Other');
  });

  it('a failed send turns the buttons on again', async () => {
    useClock();
    const { root, sent } = mount(card(), () => Promise.reject(new Error('down')));
    click(button(root, CHAT.mcp.decline)!);
    expect(sent).toEqual([{ mcp: 'decline' }]);
    expect(disabled(button(root, CHAT.mcp.send)!)).toBe(true);
    await settle();
    expect(disabled(button(root, CHAT.mcp.send)!)).toBe(false);
    expect(disabled(button(root, CHAT.mcp.decline)!)).toBe(false);
  });

  it('arrival leaves focus in the composer while the user types there', () => {
    const composer = document.createElement('textarea');
    document.body.appendChild(composer);
    composer.value = 'half a message';
    composer.focus();
    mount(card(), undefined, composer);
    expect(document.activeElement).toBe(composer);
  });

  it('the server texts are drawn as plain text', () => {
    const { root } = mount(card({ server: '<i>s</i>', message: '<b>bold?</b>' }));
    expect(root.querySelector('b, i')).toBeNull();
    expect(root.textContent).toContain('<b>bold?</b>');
    expect(root.textContent).toContain(fill(CHAT.mcp.formTitle, { server: '<i>s</i>' }));
  });
});

const LINK = 'https://github.com/login/oauth/authorize?client_id=abc&state=x%20y';

function linkCard(over: Partial<McpInteraction> = {}): McpInteraction {
  return card({
    mode: 'url',
    message: 'Sign in to GitHub to let this server create issues.',
    form: undefined,
    link: { url: LINK, host: 'github.com', secure: true },
    ...over,
  });
}

function openLink(root: HTMLElement): HTMLAnchorElement | undefined {
  return root.querySelector<HTMLAnchorElement>('a[href]') ?? undefined;
}

/** Clicks the link and says whether the browser would follow it; the test DOM never does. */
function follows(link: HTMLAnchorElement): boolean {
  let followed = false;
  const last = (event: Event): void => {
    followed = !event.defaultPrevented;
    event.preventDefault();
  };
  document.addEventListener('click', last);
  try {
    click(link);
  } finally {
    document.removeEventListener('click', last);
  }
  return followed;
}

describe('McpCard, a page to open', () => {
  it('M3: the message, the whole address and its site; the note; focus on Decline; a real link out', () => {
    const { root } = mount(linkCard());
    const group = root.querySelector('[data-interaction-card]')!;
    expect(group.querySelector('h3')?.textContent).toBe(fill(CHAT.mcp.linkTitle, { server: 'github' }));
    expect(group.textContent).toContain(fill(CHAT.mcp.from, { server: 'github' }));
    const wells = [...root.querySelectorAll('[role="region"]')];
    expect(wells.map((w) => [accessibleName(w), w.textContent])).toEqual([
      [CHAT.mcp.message, 'Sign in to GitHub to let this server create issues.'],
      [CHAT.mcp.address, LINK],
    ]);
    expect(group.textContent).toContain(fill(CHAT.mcp.site, { host: 'github.com' }));
    expect(group.textContent).toContain(CHAT.mcp.linkNote);
    expect(group.textContent).not.toContain(CHAT.mcp.notSecure);
    expect(group.textContent).not.toContain(CHAT.mcp.secretNote);
    expect(button(root, CHAT.mcp.send)).toBeUndefined();
    expect(document.activeElement).toBe(button(root, CHAT.mcp.decline));
    const link = openLink(root)!;
    expect(link.getAttribute('href')).toBe(LINK);
    expect(link.getAttribute('target')).toBe('_blank');
    expect(link.getAttribute('rel')).toBe('noreferrer');
    expect(accessibleName(link)).toBe(fill(CHAT.mcp.openLabel, { host: 'github.com' }));
    expect(link.textContent).toBe(CHAT.mcp.open);
    // The card names what it asks about: the message, the address and the site.
    const described = (group.getAttribute('aria-describedby') ?? '').split(' ');
    const texts = described.map((id) => document.getElementById(id)?.textContent);
    expect(texts).toContain(LINK);
    expect(texts).toContain(fill(CHAT.mcp.site, { host: 'github.com' }));
  });

  // Mutation: drop `!armed(event)` from the link's click → the early click opens the page.
  it('Open page opens and answers only after the arm delay, once', () => {
    useClock();
    const { root, sent } = mount(linkCard());
    const link = openLink(root)!;
    expect(follows(link)).toBe(false);
    expect(sent).toEqual([]);
    advance(CARD_ARM_MS);
    expect(follows(link)).toBe(true);
    expect(sent).toEqual([{ mcp: 'accept' }]);
    expect(root.querySelector('[data-interaction-card]')!.getAttribute('aria-busy')).toBe('true');
    expect(disabled(button(root, CHAT.mcp.decline)!)).toBe(true);
    expect(follows(link)).toBe(false);
    expect(sent).toHaveLength(1);
  });

  it('a middle click never follows the link', () => {
    useClock();
    advance(CARD_ARM_MS);
    const { root, sent } = mount(linkCard());
    const event = new MouseEvent('auxclick', { bubbles: true, cancelable: true, button: 1 });
    act(() => {
      openLink(root)!.dispatchEvent(event);
    });
    expect(event.defaultPrevented).toBe(true);
    expect(sent).toEqual([]);
  });

  it('a failed send lets Open page and Decline work again', async () => {
    useClock();
    const { root, sent } = mount(linkCard(), () => Promise.reject(new Error('down')));
    advance(CARD_ARM_MS);
    expect(follows(openLink(root)!)).toBe(true);
    await settle();
    expect(disabled(button(root, CHAT.mcp.decline)!)).toBe(false);
    click(button(root, CHAT.mcp.decline)!);
    expect(sent).toEqual([{ mcp: 'accept' }, { mcp: 'decline' }]);
  });

  it('its arrival is said with the server, as a page to open', () => {
    expect(asksSentence(linkCard())).toBe(fill(CHAT.status.asksMcpLink, { server: 'github' }));
    expect(asksSentence(card())).toBe(fill(CHAT.status.asksMcp, { server: 'github' }));
  });

  it('plain http is flagged', () => {
    const { root } = mount(
      linkCard({ link: { url: 'http://intranet.local/', host: 'intranet.local', secure: false } }),
    );
    expect(root.textContent).toContain(CHAT.mcp.notSecure);
    expect(root.textContent).toContain(fill(CHAT.mcp.site, { host: 'intranet.local' }));
    expect(root.textContent).not.toContain(CHAT.mcp.lookalike);
  });

  // Mutation: drop the `lookalike` line → an `xn--` site reads like any other.
  it('a site name in its xn-- form gets a warning line, read with the card; with plain http both lines show', () => {
    const host = 'xn--gthub-n4a.com';
    const { root } = mount(linkCard({ link: { url: `https://${host}/login`, host, secure: true, lookalike: true } }));
    const group = root.querySelector('[data-interaction-card]')!;
    const warnings = [...group.querySelectorAll('li')].map((li) => li.textContent);
    expect(warnings).toEqual([CHAT.mcp.lookalike]);
    const described = (group.getAttribute('aria-describedby') ?? '').split(' ');
    expect(described.map((id) => document.getElementById(id)?.textContent)).toContain(CHAT.mcp.lookalike);
    expect(root.textContent).toContain(fill(CHAT.mcp.site, { host }));
    unmount();
    const plain = mount(linkCard({ link: { url: `http://${host}/`, host, secure: false, lookalike: true } }));
    const both = [...plain.root.querySelectorAll('[data-interaction-card] li')].map((li) => li.textContent);
    expect(both).toEqual([CHAT.mcp.notSecure, CHAT.mcp.lookalike]);
  });

  // Mutation: leave the notes out of the card's description → "Check the site name first." is never read with it.
  it('its notes are read with the card, as on a form', () => {
    const { root } = mount(linkCard());
    const group = root.querySelector('[data-interaction-card]')!;
    const described = (group.getAttribute('aria-describedby') ?? '').split(' ');
    expect(described.map((id) => document.getElementById(id)?.textContent)).toContain(CHAT.mcp.linkNote);
    unmount();
    const form = mount(card());
    const formGroup = form.root.querySelector('[data-interaction-card]')!;
    const read = (formGroup.getAttribute('aria-describedby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent ?? '')
      .join('\n');
    expect(read).toContain(CHAT.mcp.secretNote);
    expect(read).toContain(fill(CHAT.mcp.formNote, { server: 'github' }));
  });

  it('M4: an address that is not safe offers Decline only and says why', () => {
    const { root, sent } = mount(linkCard({ link: undefined, unusable: 'address' }));
    expect(root.textContent).toContain(fill(CHAT.mcp.unusable, { why: CHAT.mcp.why.address }));
    expect(openLink(root)).toBeUndefined();
    expect(root.textContent).not.toContain(CHAT.mcp.linkNote);
    const decline = button(root, CHAT.mcp.decline)!;
    expect(document.activeElement).toBe(decline);
    click(decline);
    expect(sent).toEqual([{ mcp: 'decline' }]);
  });
});
