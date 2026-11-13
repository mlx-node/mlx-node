/** @vitest-environment happy-dom */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { MAX_FEEDBACK_CHARS } from '../src/chat/route-input.js';
import { CWD_PROBLEM_MESSAGES } from '../src/chat/work-folder.js';
import type { ApiCall } from '../src/runtime.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type {
  ConversationInfo,
  ConversationRow,
  PermissionInteraction,
  PlanInteraction,
} from '../ui/src/lib/chat-types.js';
import { flagLines, folderName } from '../ui/src/lib/permission-text.js';
import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { toast } from '../ui/src/lib/toast.js';
import Chat from '../ui/src/pages/chat.js';
import { chooseFolder, forgetPlace, NEW_PLACE, writeDraft } from '../ui/src/pages/chat/drafts.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import { CARD_ARM_MS, cardButtons, cardNotes, FEEDBACK_MAX_CHARS } from '../ui/src/pages/chat/work/permission.js';
import { APP_ROUTES, NAV_ROUTES } from '../ui/src/shell/routes.js';
import { answer, end, user } from './helpers/chat-thread-fixtures.js';
import {
  accessibleName,
  byRole,
  type ChatSync,
  deferred,
  READY_CHAT_HOST,
  STUB_FAILURE,
  renderPage,
  type RenderedPage,
  stubApi,
  stubError,
  type StubbedApi,
  waitFor,
} from './render.js';

const NOW = Date.now();
const FOLDER = '/tmp/work/proj';

let where = { pathname: '', search: '' };
let page: RenderedPage | undefined;
let api: StubbedApi | undefined;
let calls: ApiCall[] = [];
const used: string[] = [];

function LocationSpy(): null {
  const at = useLocation();
  where = { pathname: at.pathname, search: at.search };
  return null;
}

function app(initial: string): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: [initial] },
    createElement(LocationSpy),
    createElement(Routes, null, createElement(Route, { path: '/chat', element: createElement(Chat) })),
  );
}

function info(id: string, over: Partial<ConversationInfo> = {}): ConversationInfo {
  used.push(id);
  return {
    id,
    mode: 'work',
    title: `Title ${id}`,
    cwd: FOLDER,
    modelId: 'qwen3.5-4b',
    createdAt: NOW - 60_000,
    updatedAt: NOW - 60_000,
    hot: false,
    permissionMode: 'ask',
    ...over,
  };
}

function row(id: string, over: Partial<ConversationInfo> = {}): ConversationRow {
  return { ...info(id, over), turns: 1 };
}

function sync(id: string, over: Partial<ChatSync> = {}, conversation: Partial<ConversationInfo> = {}): ChatSync {
  return {
    type: 'sync',
    seq: 0,
    conversation: info(id, conversation),
    entries: [user('Fix the build'), answer('Hi there'), end()],
    turns: [],
    turn: { turnId: 't1', phase: 'running', startedAt: NOW, wait: null },
    interactions: [],
    notices: [],
    transcriptError: null,
    permissionMarks: [],
    ...over,
  };
}

const ALL_OPTIONS: PermissionInteraction['options'] = [
  { optionId: 'enable-always-approve', kind: 'allow_always', name: 'Approve everything' },
  { optionId: 'always-allow', kind: 'allow_always', name: 'Always allow cargo' },
  { optionId: 'allow-once', kind: 'allow_once', name: 'Yes' },
  { optionId: 'reject-always', kind: 'reject_once', name: 'Never' },
  { optionId: 'reject-once', kind: 'reject_once', name: 'No' },
];

function permission(over: Partial<PermissionInteraction> = {}): PermissionInteraction {
  return {
    id: 'i1',
    kind: 'permission',
    toolCallId: 'c1',
    toolName: 'run_terminal_command',
    access: 'bash',
    title: 'Run the tests',
    input: '{"command":"cargo test"}',
    command: 'cargo test --workspace && rm -rf target/debug/incremental',
    path: null,
    resolvedPath: null,
    options: ALL_OPTIONS,
    flags: [],
    ...over,
  };
}

const PLAN: PlanInteraction = {
  id: 'i-plan',
  kind: 'plan',
  toolCallId: 'c-plan',
  plan: '1. Read the failing test\n2. Patch `src/main.rs`',
};

async function open(url: string, until: (text: string) => boolean, setup: Open = {}): Promise<RenderedPage> {
  calls = [];
  api = stubApi(
    { '/chat/conversations': { conversations: setup.rows ?? [] }, ...setup.routes },
    {
      onCall: (call) => calls.push(call),
      chat: { host: READY_CHAT_HOST, conversations: setup.conversations ?? {} },
    },
  );
  page = await renderPage(app(url), until);
  const want = JSON.stringify(READY_CHAT_HOST);
  await waitFor(() => JSON.stringify(chatHostSnapshot().state) === want, 2_000);
  return page;
}

interface Open {
  routes?: Record<string, unknown>;
  conversations?: Record<string, ChatSync>;
  rows?: ConversationRow[];
}

function pathOf(call: ApiCall): string {
  const bare = call.path.split('?')[0]!;
  return bare.startsWith('/api') ? bare.slice('/api'.length) : bare;
}

function writes(): Array<{ method: string; path: string; body: unknown }> {
  return calls
    .filter((call) => call.method !== 'GET')
    .map((call) => ({ method: call.method, path: pathOf(call), body: call.body }));
}

function composer(): HTMLTextAreaElement {
  const area = byRole(page!.container, 'textbox', CHAT.composerLabel);
  if (area === null) throw new Error('no composer');
  return area as HTMLTextAreaElement;
}

function type(text: string, area: HTMLTextAreaElement = composer()): void {
  act(() => {
    Reflect.set(HTMLTextAreaElement.prototype, 'value', text, area);
    area.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function reachable(scope: ParentNode = document.body): HTMLButtonElement[] {
  return [...scope.querySelectorAll('button')].filter((b) => b.closest('[inert]') === null);
}

function button(label: string, scope: ParentNode = document.body): HTMLButtonElement {
  const found = reachable(scope).find((b) => accessibleName(b) === label);
  if (found === undefined) throw new Error(`no button ${label}`);
  return found;
}

function click(element: HTMLElement): void {
  act(() => element.click());
}

/** A click once the card has been on screen long enough for its allow buttons to count. */
function laterClick(element: HTMLElement): void {
  const real = performance.now.bind(performance);
  const later = vi.spyOn(performance, 'now').mockImplementation(() => real() + CARD_ARM_MS);
  click(element);
  later.mockRestore();
}

function shown(scope: Element = page!.container): string {
  const copy = scope.cloneNode(true) as Element;
  for (const hidden of copy.querySelectorAll('[inert]')) hidden.remove();
  return copy.textContent ?? '';
}

function card(): HTMLElement | null {
  return page!.container.querySelector<HTMLElement>('[data-interaction-card]');
}

/** The answer buttons of the open card, in tab order. */
function cardButtonLabels(): string[] {
  return [...card()!.querySelectorAll('button[data-option]')].map((b) => accessibleName(b));
}

function sendFace(): HTMLButtonElement {
  const form = page!.container.querySelector<HTMLFormElement>(`form[aria-label="${CHAT.composerLabel}"]`)!;
  const found = [...form.querySelectorAll('button')].at(-1);
  if (found === undefined) throw new Error('no send button');
  return found;
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  page?.unmount();
  api?.();
  page = undefined;
  api = undefined;
  writeDraft(NEW_PLACE, '');
  for (const id of used.splice(0)) forgetPlace(id);
  chooseFolder(null);
  vi.restoreAllMocks();
});

describe('Work permission card: what is drawn', () => {
  it('Deny, Allow once, then the "always" option; reject_always and enable-always-approve are never drawn', () => {
    const labels = cardButtons(permission(), 'ask').map((b) => [b.label, b.option.optionId]);
    expect(labels).toEqual([
      [CHAT.deny, 'reject-once'],
      [CHAT.allowOnce, 'allow-once'],
      [CHAT.allowAlwaysCommand, 'always-allow'],
    ]);
  });

  it('the "always" label follows what the agent saves', () => {
    const always = (over: Partial<PermissionInteraction>): string | undefined =>
      cardButtons(permission(over), 'ask').find((b) => b.always)?.label;
    expect(always({ access: 'mcp' })).toBe(CHAT.allowAlwaysTool);
    expect(always({ access: 'other' })).toBe(CHAT.allowAlwaysOther);
    expect(always({ access: 'edit' })).toBeUndefined();
    expect(
      always({
        access: 'edit',
        options: [...ALL_OPTIONS, { optionId: 'allow-edits-session', kind: 'allow_always', name: 'Allow edits' }],
      }),
    ).toBe(CHAT.allowEditsSession);
  });

  it('an option whose id and kind do not match is not drawn', () => {
    const options: PermissionInteraction['options'] = [
      { optionId: 'reject-once', kind: 'allow_once', name: 'Sure' },
      { optionId: 'allow-once', kind: 'allow_always', name: 'Forever' },
    ];
    expect(cardButtons(permission({ options }), 'ask')).toEqual([]);
  });

  it('warning lines come in the fixed order', () => {
    expect(flagLines(['provider-jobs', 'installs', 'elevated', 'prints-secrets', 'deletes'])).toEqual([
      CHAT.permFlags.elevated,
      CHAT.permFlags.deletes,
      CHAT.permFlags.installs,
      CHAT.permFlags['prints-secrets'],
      CHAT.permFlags['provider-jobs'],
    ]);
    expect(flagLines(['unchecked', 'deletes'])).toEqual([CHAT.permFlags.deletes, 'This command is too long to check.']);
    expect(flagLines([])).toEqual([]);
  });

  it("notes: the agent's own wording on one line, what Deny does, and where command grants live", () => {
    const bash = permission({
      options: [{ optionId: 'always-allow', kind: 'allow_always', name: 'Always\nallow cargo' }],
    });
    expect(cardNotes('bash', cardButtons(bash, 'ask'))).toEqual([
      fill(CHAT.agentWording, { name: 'Always allow cargo' }),
      `${CHAT.denyNote} ${CHAT.grantsNote}`,
    ]);
    expect(cardNotes('other', cardButtons(permission({ access: 'other' }), 'ask'))[1]).toBe(CHAT.denyNote);
    expect(cardNotes('bash', [])).toEqual([CHAT.denyNote]);
  });

  it("the page's folder sentences are the server's", () => {
    expect(CHAT.cwdProblems).toEqual(CWD_PROBLEM_MESSAGES);
  });

  it('a folder is named by its last segment', () => {
    expect(folderName('/Users/me/code/app/')).toBe('app');
    expect(folderName('/')).toBe('/');
  });
});

describe('Work permission card: on the page', () => {
  it('shows the whole command and its warnings, focuses Deny, and Deny sends reject-once', async () => {
    const interaction = permission({ flags: ['deletes', 'elevated'] });
    await open('/chat?c=p-deny', (text) => text.includes(CHAT.permRun), {
      routes: { 'POST /chat/conversations/p-deny/interactions/i1': {} },
      conversations: { 'p-deny': sync('p-deny', { interactions: [interaction] }) },
    });
    const exact = card()!.querySelector(`pre[aria-label="${CHAT.permCommand}"]`);
    expect(exact?.textContent).toBe(interaction.command);
    expect(shown(card()!)).toContain(CHAT.permFlags.elevated);
    expect(shown(card()!).indexOf(CHAT.permFlags.elevated)).toBeLessThan(
      shown(card()!).indexOf(CHAT.permFlags.deletes),
    );
    expect(cardButtonLabels()).toEqual([CHAT.deny, CHAT.allowOnce, CHAT.allowAlwaysCommand]);
    await waitFor(() => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.deny);
    click(document.activeElement as HTMLButtonElement);
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      { method: 'POST', path: '/chat/conversations/p-deny/interactions/i1', body: { optionId: 'reject-once' } },
    ]);
    act(() => api!.emitChat('p-deny', { type: 'interaction-closed', seq: 1, id: 'i1', reason: 'answered' }));
    act(() => api!.emitChat('p-deny', { type: 'turn', seq: 2, turn: null, turnId: 't1', outcome: 'denied' }));
    await waitFor(() => shown().includes(CHAT.outcomes.denied));
    await waitFor(() => card() === null);
  });

  it('a command too long to check says so on the card, beside the whole command', async () => {
    const interaction = permission({ flags: ['deletes', 'unchecked'], command: `rm a; ${'sh;'.repeat(2_000)}` });
    await open('/chat?c=p-long', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-long': sync('p-long', { interactions: [interaction] }) },
    });
    const exact = card()!.querySelector(`pre[aria-label="${CHAT.permCommand}"]`);
    expect(exact?.textContent).toBe(interaction.command);
    expect(shown(card()!)).toContain('This command is too long to check.');
  });

  it('a pressed answer waits; the other buttons are off until the request returns', async () => {
    await open('/chat?c=p-once', (text) => text.includes(CHAT.permRun), {
      routes: { 'POST /chat/conversations/p-once/interactions/i1': {} },
      conversations: { 'p-once': sync('p-once', { interactions: [permission()] }) },
    });
    laterClick(button(CHAT.allowOnce, card()!));
    expect(card()!.getAttribute('aria-busy')).toBe('true');
    expect(button(CHAT.deny, card()!).disabled).toBe(true);
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ optionId: 'allow-once' });
  });

  it('an edit request shows the whole path and offers edits for this conversation', async () => {
    const interaction = permission({
      access: 'edit',
      toolName: 'search_replace',
      command: null,
      path: `${FOLDER}/src/main.rs`,
      options: [
        { optionId: 'reject-once', kind: 'reject_once', name: 'No' },
        { optionId: 'allow-once', kind: 'allow_once', name: 'Yes' },
        { optionId: 'allow-edits-session', kind: 'allow_always', name: 'Yes, and allow edits' },
      ],
    });
    await open('/chat?c=p-edit', (text) => text.includes(CHAT.permEdit), {
      conversations: { 'p-edit': sync('p-edit', { interactions: [interaction] }) },
    });
    expect(card()!.querySelector(`pre[aria-label="${CHAT.permFile}"]`)?.textContent).toBe(interaction.path);
    expect(cardButtonLabels()).toEqual([CHAT.deny, CHAT.allowOnce, CHAT.allowEditsSession]);
    expect(shown(card()!)).not.toContain(CHAT.grantsNote);
    expect(shown(card()!)).toContain(CHAT.editsSessionNote);
  });

  it('an edit through a link shows where it writes, and the whole input, as exact wells the card is described by', async () => {
    const input = JSON.stringify({ file_path: 'docs/x.plist', old_string: 'a', new_string: 'b' }, null, 2);
    const interaction = permission({
      access: 'edit',
      toolName: 'search_replace',
      command: null,
      input,
      path: 'docs/x.plist',
      resolvedPath: '/Users/someone/Library/LaunchAgents/x.plist',
      flags: ['outside-folder'],
    });
    await open('/chat?c=p-link', (text) => text.includes(CHAT.permEdit), {
      conversations: { 'p-link': sync('p-link', { interactions: [interaction] }) },
    });
    const lands = card()!.querySelector(`pre[aria-label="${CHAT.permLands}"]`);
    expect(lands?.textContent).toBe(interaction.resolvedPath);
    expect(card()!.querySelector(`pre[aria-label="${CHAT.permInput}"]`)?.textContent).toBe(input);
    expect(card()!.getAttribute('aria-describedby')?.split(' ')).toContain(lands!.id);
    expect(shown(card()!)).toContain(CHAT.permFlags['outside-folder']);
  });

  it('an edit whose file the app cannot tell says so in place of the outside line', async () => {
    const interaction = permission({
      access: 'edit',
      toolName: 'search_replace',
      command: null,
      input: '{}',
      path: 'n b.txt',
      landingUnknown: true,
      flags: ['outside-folder'],
    });
    await open('/chat?c=p-unknown', (text) => text.includes(CHAT.permEdit), {
      conversations: { 'p-unknown': sync('p-unknown', { interactions: [interaction] }) },
    });
    expect(shown(card()!)).toContain(CHAT.permLandingUnknown);
    expect(shown(card()!)).not.toContain(CHAT.permFlags['outside-folder']);
    expect(card()!.querySelector(`pre[aria-label="${CHAT.permLands}"]`)).toBeNull();
  });

  it('a card that arrives while the user types does not take the focus', async () => {
    await open('/chat?c=p-typing', (text) => text.includes('Hi there'), {
      conversations: { 'p-typing': sync('p-typing') },
    });
    act(() => composer().focus());
    type('and also check the docs');
    act(() => api!.emitChat('p-typing', { type: 'interaction', seq: 1, interaction: permission() }));
    await waitFor(() => card() !== null);
    expect(document.activeElement).toBe(composer());
  });

  it('a request that is no longer open closes its card with a toast', async () => {
    const error = vi.spyOn(toast, 'error');
    await open('/chat?c=p-late', (text) => text.includes(CHAT.permRun), {
      routes: { 'POST /chat/conversations/p-late/interactions/i1': stubError('E_NOT_FOUND', 'gone') },
      conversations: { 'p-late': sync('p-late', { interactions: [permission()] }) },
    });
    click(button(CHAT.deny, card()!));
    await waitFor(() => error.mock.calls.length === 1);
    expect(error).toHaveBeenCalledWith(CHAT.permClosed);
    await waitFor(() => card() === null);
  });

  it('an answer the server refuses for its content keeps the card and shows the server sentence', async () => {
    const error = vi.spyOn(toast, 'error');
    await open('/chat?c=p-bad', (text) => text.includes(CHAT.permRun), {
      routes: {
        'POST /chat/conversations/p-bad/interactions/i1': stubError('E_BAD_REQUEST', 'That option was not offered.'),
      },
      conversations: { 'p-bad': sync('p-bad', { interactions: [permission()] }) },
    });
    click(button(CHAT.deny, card()!));
    await waitFor(() => error.mock.calls.length === 1);
    expect(error).toHaveBeenCalledWith('That option was not offered.');
    await waitFor(() => !button(CHAT.deny, card()!).disabled);
    expect(card()).not.toBeNull();
  });

  it('a click on a button that allows, in the moment the card appears, is not an answer', async () => {
    await open('/chat?c=p-quick', (text) => text.includes(CHAT.permRun), {
      routes: { 'POST /chat/conversations/p-quick/interactions/i1': {} },
      conversations: { 'p-quick': sync('p-quick', { interactions: [permission()] }) },
    });
    click(button(CHAT.allowOnce, card()!));
    click(button(CHAT.allowAlwaysCommand, card()!));
    expect(card()!.hasAttribute('aria-busy')).toBe(false);
    expect(writes()).toEqual([]);
    laterClick(button(CHAT.allowOnce, card()!));
    await waitFor(() => writes().length === 1);
  });

  it('a Chat conversation never draws a card', async () => {
    await open('/chat?c=p-chat', (text) => text.includes('Hi there'), {
      conversations: { 'p-chat': sync('p-chat', { interactions: [permission()] }, { mode: 'chat', cwd: null }) },
    });
    expect(card()).toBeNull();
  });
});

describe('Work plan card', () => {
  it('Approve sends approved', async () => {
    await open('/chat?c=p-plan', (text) => text.includes(CHAT.planTitle), {
      routes: { 'POST /chat/conversations/p-plan/interactions/i-plan': {} },
      conversations: { 'p-plan': sync('p-plan', { interactions: [PLAN] }) },
    });
    expect(shown(card()!)).toContain('Read the failing test');
    laterClick(button(CHAT.planApprove, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-plan/interactions/i-plan',
      body: { decision: 'approved' },
    });
  });

  it('feedback is written in the card and sent as a rejection with its text', async () => {
    await open('/chat?c=p-feedback', (text) => text.includes(CHAT.planTitle), {
      routes: { 'POST /chat/conversations/p-feedback/interactions/i-plan': {} },
      conversations: { 'p-feedback': sync('p-feedback', { interactions: [PLAN] }) },
    });
    click(button(CHAT.planFeedback, card()!));
    const field = byRole(card()!, 'textbox', CHAT.planFeedbackLabel) as HTMLTextAreaElement | null;
    expect(field).not.toBeNull();
    expect([FEEDBACK_MAX_CHARS, field!.maxLength]).toEqual([MAX_FEEDBACK_CHARS, MAX_FEEDBACK_CHARS]);
    await waitFor(() => reachable(card()!).some((b) => accessibleName(b) === CHAT.planSend), 2_000);
    expect(button(CHAT.planSend, card()!).disabled).toBe(true);
    type('  Run the linter first  ', field!);
    click(button(CHAT.planSend, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ decision: 'rejected', feedback: 'Run the linter first' });
  });

  it('a plan with no text shows one fixed line, offers no feedback field, and keeps its two answers', async () => {
    for (const [n, plan] of [
      [0, ''],
      [1, ' \n '],
    ] as const) {
      const id = `p-empty-${n}`;
      await open(`/chat?c=${id}`, (text) => text.includes(CHAT.planTitle), {
        routes: { [`POST /chat/conversations/${id}/interactions/i-plan`]: {} },
        conversations: { [id]: sync(id, { interactions: [{ ...PLAN, plan }] }) },
      });
      expect(shown(card()!)).toContain(CHAT.planNoText);
      expect(card()!.textContent).not.toContain(CHAT.planLabel);
      expect(reachable(card()!).some((b) => accessibleName(b) === CHAT.planFeedback)).toBe(false);
      expect(reachable(card()!).some((b) => accessibleName(b) === CHAT.planApprove)).toBe(true);
      click(button(CHAT.planKeepPlanning, card()!));
      await waitFor(() => writes().length === 1);
      expect(byRole(card()!, 'textbox', CHAT.planFeedbackLabel)).toBeNull();
      expect(writes()[0]).toEqual({
        method: 'POST',
        path: `/chat/conversations/${id}/interactions/i-plan`,
        body: { decision: 'rejected' },
      });
    }
  });

  it('Escape in the feedback field goes back to the plan buttons', async () => {
    await open('/chat?c=p-escape', (text) => text.includes(CHAT.planTitle), {
      conversations: { 'p-escape': sync('p-escape', { interactions: [PLAN] }) },
    });
    click(button(CHAT.planFeedback, card()!));
    const field = byRole(card()!, 'textbox', CHAT.planFeedbackLabel) as HTMLTextAreaElement;
    act(() => {
      field.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => byRole(card()!, 'textbox', CHAT.planFeedbackLabel) === null);
    await waitFor(() => reachable(card()!).some((b) => accessibleName(b) === CHAT.planApprove), 2_000);
  });
});

describe('Work: Escape', () => {
  it('Escape stops a Work answer only from the composer', async () => {
    await open('/chat?c=p-work-esc', (text) => text.includes('Hi there'), {
      routes: { '/chat/conversations/p-work-esc/cancel': {} },
      conversations: { 'p-work-esc': sync('p-work-esc') },
    });
    await waitFor(() => accessibleName(sendFace()) === CHAT.stop);
    const press = (target: Element, init: KeyboardEventInit = {}): void =>
      act(() => {
        target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true, ...init }));
      });
    const cancels = (): number => writes().filter((w) => w.path.endsWith('/cancel')).length;
    press(document.body);
    press(page!.container.querySelector('[data-turn]')!);
    press(composer(), { repeat: true });
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
    expect(cancels()).toBe(0);
    press(composer());
    await waitFor(() => cancels() === 1);
  });
});

describe('Work folder', () => {
  it('a folder whose path is too long for the agent keeps Send off and says why', async () => {
    const deep = `/Users/dev/${'a'.repeat(250)}`;
    chooseFolder(deep);
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
    type('Run the tests');
    await waitFor(() => shown().includes(CHAT.folderTooLong));
    expect(sendFace().disabled).toBe(true);
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    expect(writes()).toEqual([]);
  });

  it('with no folder the chip asks for one and nothing can be sent', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
    const chip = reachable().filter((b) => accessibleName(b) === CHAT.folderChoose);
    expect(chip.length).toBeGreaterThan(0);
    for (const text of CHAT.startersWork) expect(button(text).disabled).toBe(true);
    type('Run the tests');
    expect(sendFace().disabled).toBe(true);
  });

  it('the last Work folder is preselected and a new conversation is created in it', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle), {
      rows: [row('p-old', { updatedAt: NOW - 120_000 })],
      routes: { 'POST /chat/conversations': { conversation: info('p-made'), turnId: 't1' } },
    });
    await waitFor(() => reachable().some((b) => accessibleName(b) === fill(CHAT.folderLabel, { folder: FOLDER })));
    expect(composer().placeholder).toBe(fill(CHAT.placeholderWork, { folder: folderName(FOLDER) }));
    for (const text of CHAT.startersWork) expect(button(text).disabled).toBe(false);
    type('Run the tests');
    act(() => sendFace().click());
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toMatchObject({
      method: 'POST',
      path: '/chat/conversations',
      body: { mode: 'work', text: 'Run the tests', cwd: FOLDER },
    });
  });

  it('a Work create that fails late keeps the text typed while it was pending, and Retry sends the failed text', async () => {
    const create = deferred(STUB_FAILURE);
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle), {
      rows: [row('p-old', { updatedAt: NOW - 120_000 })],
      routes: { 'POST /chat/conversations': create.body },
    });
    await waitFor(() => reachable().some((b) => accessibleName(b) === fill(CHAT.folderLabel, { folder: FOLDER })));
    type('Run the tests');
    act(() => sendFace().click());
    await waitFor(() => writes().length === 1);
    type('newer text');
    await act(async () => create.release());
    await waitFor(() => shown().includes(CHAT.sendErrors.other[0]));
    expect(composer().value).toBe('newer text');
    act(() => button(CHAT.sendErrors.other[1]).click());
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toMatchObject({ body: { mode: 'work', text: 'Run the tests', cwd: FOLDER } });
    expect(composer().value).toBe('newer text');
  });

  it('a folder that is gone says so and turns Send off', async () => {
    await open('/chat?c=p-gone', (text) => text.includes('Hi there'), {
      conversations: { 'p-gone': sync('p-gone', { turn: null }, { cwdProblem: 'missing' }) },
    });
    await waitFor(() => shown().includes(CHAT.cwdProblems.missing));
    type('Try again');
    expect(sendFace().disabled).toBe(true);
  });

  it('a refused folder shows the server reason and offers another folder', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle), {
      rows: [row('p-home', { cwd: '/Users/me' })],
      routes: {
        'POST /chat/conversations': stubError('E_BAD_REQUEST', CWD_PROBLEM_MESSAGES.home, 'cwd-invalid'),
      },
    });
    await waitFor(() => composer().placeholder === fill(CHAT.placeholderWork, { folder: 'me' }));
    type('Run the tests');
    act(() => sendFace().click());
    await waitFor(() => shown().includes(CHAT.cwdProblems.home));
    expect(reachable(page!.container.querySelector('main')!).some((b) => accessibleName(b) === CHAT.folderChoose)).toBe(
      true,
    );
    expect(where.search).toBe('?mode=work');
  });
});

describe('Work in the docs', () => {
  /** The text with every run of white space as one space, so a pinned sentence may wrap anywhere. */
  const flat = (path: string): string =>
    splitWhitespace(readFileSync(join(import.meta.dirname, path), 'utf8')).join(' ');
  const doc = flat('../../../docs/dashboard.md');
  const readme = flat('../README.md');

  it('the route count is the route table plus the onboarding route', () => {
    const words = ['Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen'];
    expect(doc).toContain(
      `${words[APP_ROUTES.length + 1 - 9]} routes. ${words[NAV_ROUTES.length - 9]} sit in the sidebar`,
    );
    expect(readme).toContain(`the ${words[APP_ROUTES.length + 1 - 9]!.toLowerCase()} routes`);
  });

  it('says who can read the key in Work, and that the folder dialog is the only main-process surface', () => {
    expect(doc).toContain("the user's own MCP servers, hooks and shell commands can read the provider keys");
    expect(doc).toContain('The folder dialog is the only main-process surface the page adds.');
    expect(doc).not.toContain('stops at the folder step');
  });
});
