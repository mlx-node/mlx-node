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
  ChatHostState,
  ConversationInfo,
  ConversationRow,
  McpInteraction,
  PermissionInteraction,
  PlanInteraction,
  QuestionInteraction,
  QueuedMessage,
  QueueHold,
  QueueState,
} from '../ui/src/lib/chat-types.js';
import { formatClock } from '../ui/src/lib/format.js';
import { flagLines, folderName, NEVER_NOTE_HEAD, NEVER_NOTE_TAIL } from '../ui/src/lib/permission-text.js';
import { FOLDER_PICKED_MESSAGE, PICK_FOLDER_MESSAGE } from '../ui/src/lib/pick-folder.js';
import { splitWhitespace } from '../ui/src/lib/text-scan.js';
import { toast } from '../ui/src/lib/toast.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import Chat from '../ui/src/pages/chat.js';
import {
  chooseFolder,
  forgetPlace,
  NEW_PLACE,
  writeAttachments,
  writeDraft,
  writeFailed,
} from '../ui/src/pages/chat/drafts.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import {
  CARD_ARM_MS,
  cardButtons,
  cardNotes,
  FEEDBACK_MAX_CHARS,
  neverMenuText,
} from '../ui/src/pages/chat/work/permission.js';
import { APP_ROUTES, NAV_ROUTES } from '../ui/src/shell/routes.js';
import { answer, end, RUN_ID, runSample, user, workflowRun } from './helpers/chat-thread-fixtures.js';
import { runView } from './helpers/run-views.js';
import {
  accessibleName,
  allByRole,
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
  observeWidth,
  sequence,
  waitFor,
} from './render.js';

const NOW = Date.now();
const FOLDER = '/tmp/work/proj';

// Happy DOM lays nothing out: with `away` set before a page opens, the thread reads as scrolled away from its end.
const stick = vi.hoisted(() => ({ away: false }));
vi.mock('../ui/src/lib/stick-to-bottom.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../ui/src/lib/stick-to-bottom.js')>();
  return {
    ...real,
    useStickToBottom: (root: HTMLElement | null, key: string) => {
      const live = real.useStickToBottom(root, key);
      return stick.away ? { ...live, pinned: false, showJump: true } : live;
    },
  };
});

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

/** A queue as the host sends it: every row is listed, so its count is its rows. */
function queueView(items: QueuedMessage[], held: QueueHold | null = null): QueueState {
  return { items, held, count: items.length };
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
    runs: [],
    live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
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
    openedAt: NOW,
    ...over,
  };
}

const PLAN: PlanInteraction = {
  id: 'i-plan',
  kind: 'plan',
  toolCallId: 'c-plan',
  openedAt: NOW,
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

/** The page bar: the header that holds the page's h1. */
function bar(): HTMLElement {
  const found = [...page!.container.querySelectorAll<HTMLElement>('header')].find(
    (h) => h.querySelector('h1') !== null,
  );
  if (found === undefined) throw new Error('no bar');
  return found;
}

/** The strip under the composer on New; an empty node while it is not shown. */
function tray(): HTMLElement {
  return (
    page!.container.querySelector<HTMLElement>(`[role="group"][aria-label="${CHAT.tray.label}"]`) ??
    document.createElement('div')
  );
}

function composerForm(): HTMLFormElement {
  const found = page!.container.querySelector<HTMLFormElement>(`form[aria-label="${CHAT.composerLabel}"]`);
  if (found === null) throw new Error('no composer form');
  return found;
}

/** The folder chips in a scope: any button named "Choose folder…" or "Folder: …". */
/** The tray chip's name with no folder: the fixture host's scratch root, the home folder unknown. */
const NONE_LABEL = fill(CHAT.folderLabelNone, { root: READY_CHAT_HOST.scratchRoot });

function folderChips(scope: ParentNode): string[] {
  const prefix = fill(CHAT.folderLabel, { folder: '' });
  return reachable(scope)
    .map((b) => accessibleName(b))
    .filter((name) => name === CHAT.folderChoose || name.startsWith(prefix));
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

/** The card's visible "Asked by" line (its screen-reader copy is a separate node). */
function askedBy(): string | null {
  return card()?.querySelector('[data-asked-by]')?.textContent ?? null;
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

afterEach(async () => {
  page?.unmount();
  // A call still in flight fails when the port closes; its toast must not land in the next test.
  const closed = api?.closed;
  api?.();
  await closed;
  await new Promise((resolve) => setImmediate(resolve));
  page = undefined;
  api = undefined;
  stick.away = false;
  writeDraft(NEW_PLACE, '');
  writeFailed(NEW_PLACE, null);
  for (const id of used.splice(0)) forgetPlace(id);
  chooseFolder(null);
  vi.restoreAllMocks();
  toast.dismiss();
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

describe('Work MCP link card', () => {
  const link: McpInteraction = {
    id: 'i-link',
    kind: 'mcp-input',
    toolCallId: 'mcp-elicit-1',
    openedAt: Date.now(),
    mode: 'url',
    server: 'github',
    message: 'Sign in to GitHub to let this server create issues.',
    link: { url: 'https://github.com/login/oauth/authorize?client_id=abc', host: 'github.com', secure: true },
  };

  it('focuses Decline, and Open page sends a bare accept once armed', async () => {
    await open('/chat?c=p-link', (text) => text.includes(fill(CHAT.mcp.linkTitle, { server: 'github' })), {
      routes: { 'POST /chat/conversations/p-link/interactions/i-link': {} },
      conversations: { 'p-link': sync('p-link', { interactions: [link] }) },
    });
    await waitFor(() => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.mcp.decline);
    const anchor = card()!.querySelector<HTMLAnchorElement>('a[href]')!;
    expect(anchor.getAttribute('href')).toBe(link.link!.url);
    const stay = (event: Event): void => event.preventDefault();
    document.addEventListener('click', stay);
    try {
      click(anchor);
      expect(writes()).toEqual([]);
      laterClick(anchor);
    } finally {
      document.removeEventListener('click', stay);
    }
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      { method: 'POST', path: '/chat/conversations/p-link/interactions/i-link', body: { mcp: 'accept' } },
    ]);
  });
});

describe('Work question card', () => {
  const asked: QuestionInteraction = {
    id: 'i-ask',
    kind: 'question',
    toolCallId: 'c-ask',
    openedAt: Date.now(),
    planMode: true,
    questions: [
      {
        text: 'Which layout?',
        multi: false,
        options: [
          { label: 'Sidebar layout', description: '', preview: 'nav | content' },
          { label: 'Top bar layout', description: '', preview: 'nav\ncontent' },
        ],
      },
      {
        text: 'Which checks?',
        multi: true,
        options: [
          { label: 'Unit tests', description: '', preview: null },
          { label: 'Lint', description: '', preview: null },
        ],
      },
    ],
  };
  const picked = (): boolean[] =>
    [...card()!.querySelectorAll<HTMLElement>('[data-row]')].map((r) => r.getAttribute('aria-checked') === 'true');
  const link = (id: string): HTMLAnchorElement => {
    const nav = page!.container.querySelector(`nav[aria-label="${CHAT.conversations}"]`)!;
    return [...nav.querySelectorAll('a')].find((a) => a.textContent?.includes(`Title ${id}`) === true)!;
  };

  // Mutation: start the card's state empty (no draft) → back on the conversation the step and picks are gone.
  it('keeps its step and picks across a conversation switch; Talk it over sends the answers so far', async () => {
    await open('/chat?c=p-ask', (text) => text.includes(CHAT.ask.titleMany.split('{n}')[0]!), {
      rows: [row('p-ask'), row('p-other-ask')],
      routes: { 'POST /chat/conversations/p-ask/interactions/i-ask': {} },
      conversations: {
        'p-ask': sync('p-ask', { interactions: [asked] }),
        'p-other-ask': sync('p-other-ask', { turn: null }),
      },
    });
    expect(card()!.textContent).toContain(fill(CHAT.ask.preview, { label: 'Sidebar layout' }));
    click(card()!.querySelectorAll<HTMLElement>('[data-row]')[1]!);
    click(button(CHAT.ask.next, card()!));
    click(card()!.querySelectorAll<HTMLElement>('[data-row]')[1]!);
    expect(picked()).toEqual([false, true, false]);

    click(link('p-other-ask'));
    await waitFor(() => card() === null);
    click(link('p-ask'));
    await waitFor(() => card() !== null);
    expect(card()!.querySelector('[role="group"][aria-label]')?.getAttribute('aria-label')).toBe('Which checks?');
    expect(picked()).toEqual([false, true, false]);
    click(button(CHAT.ask.back, card()!));
    expect(picked()).toEqual([false, true, false]);
    expect(card()!.textContent).toContain(fill(CHAT.ask.preview, { label: 'Top bar layout' }));

    click(button(CHAT.ask.discuss, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      {
        method: 'POST',
        path: '/chat/conversations/p-ask/interactions/i-ask',
        body: { reply: 'discuss', answers: [{ picks: [1] }, { picks: [1] }] },
      },
    ]);
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

describe('Work: workflow runs', () => {
  const runEntries = runSample(workflowRun()).entries;
  const RUN_TITLE = `${CHAT.run.namePrefix}deep-research`;
  const runCard = (): HTMLElement | null =>
    page!.container.querySelector<HTMLElement>(`[data-run="${RUN_ID}"]`)?.closest('section') ?? null;
  const runStatus = (): string => runCard()?.querySelector('header [data-tone]')?.textContent ?? '';
  /** A finished step of the running turn: the turn has shown work, so an open request of its own hides the wait line. */
  const listed = (): TranscriptEntry => ({
    kind: 'tool',
    ts: NOW + 10,
    id: 'ls-1',
    name: 'list_dir',
    title: 'List docs',
    status: 'completed',
    endTs: NOW + 20,
  });
  const helperAsk = (over: Partial<PermissionInteraction> = {}): PermissionInteraction =>
    permission({
      id: 'i-helper',
      command: 'curl -s https://example.com',
      from: { label: 'researcher-0', run: 'deep-research', cwd: null, child: 'child-0' },
      ...over,
    });

  it('Stop run asks first, then POSTs the run stop route; the card says Stopping… once the host does', async () => {
    await open('/chat?c=p-run-stop', (text) => text.includes(RUN_TITLE), {
      routes: { [`POST /chat/conversations/p-run-stop/runs/${RUN_ID}/stop`]: {} },
      conversations: {
        'p-run-stop': sync('p-run-stop', {
          entries: runEntries,
          turn: null,
          runs: [runView(RUN_ID)],
        }),
      },
    });
    expect(runStatus()).toBe(CHAT.run.status.running);
    expect(accessibleName(sendFace())).not.toBe(CHAT.stop);
    click(button(CHAT.run.buttons.stop, runCard()!));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    expect(writes()).toEqual([]);
    click(button(CHAT.run.buttons.stopWorkflow, byRole(document.body, 'dialog')!));
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      { method: 'POST', path: `/chat/conversations/p-run-stop/runs/${RUN_ID}/stop`, body: undefined },
    ]);
    act(() => api!.emitChat('p-run-stop', { type: 'runs', seq: 1, runs: [runView(RUN_ID, { stopping: true })] }));
    await waitFor(() => runStatus() === CHAT.run.status.stopping);
    act(() => api!.emitChat('p-run-stop', { type: 'runs', seq: 2, runs: [] }));
    await waitFor(() => runStatus() === CHAT.run.status.endedEarly);
  });

  it('a refused stop shows the server sentence in the confirmation', async () => {
    await open('/chat?c=p-run-gone', (text) => text.includes(RUN_TITLE), {
      routes: {
        [`POST /chat/conversations/p-run-gone/runs/${RUN_ID}/stop`]: stubError('E_NOT_FOUND', 'No such run.'),
      },
      conversations: {
        'p-run-gone': sync('p-run-gone', {
          entries: runEntries,
          turn: null,
          runs: [runView(RUN_ID)],
        }),
      },
    });
    click(button(CHAT.run.buttons.stop, runCard()!));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    click(button(CHAT.run.buttons.stopWorkflow, byRole(document.body, 'dialog')!));
    await waitFor(() => (byRole(document.body, 'dialog')?.textContent ?? '').includes('No such run.'));
    expect(byRole(document.body, 'dialog')!.textContent).toContain(CHAT.run.stopError);
  });

  it('a stop that fails after the card went to Stopping… shows a toast; the card runs again', async () => {
    const error = vi.spyOn(toast, 'error');
    const held = deferred(STUB_FAILURE);
    await open('/chat?c=p-run-fail', (text) => text.includes(RUN_TITLE), {
      routes: { [`POST /chat/conversations/p-run-fail/runs/${RUN_ID}/stop`]: held.body },
      conversations: {
        'p-run-fail': sync('p-run-fail', {
          entries: runEntries,
          turn: null,
          runs: [runView(RUN_ID)],
        }),
      },
    });
    click(button(CHAT.run.buttons.stop, runCard()!));
    await waitFor(() => byRole(document.body, 'dialog') !== null);
    click(button(CHAT.run.buttons.stopWorkflow, byRole(document.body, 'dialog')!));
    await waitFor(() => writes().length === 1);
    act(() => api!.emitChat('p-run-fail', { type: 'runs', seq: 1, runs: [runView(RUN_ID, { stopping: true })] }));
    await waitFor(() => runStatus() === CHAT.run.status.stopping);
    act(() => api!.emitChat('p-run-fail', { type: 'runs', seq: 2, runs: [runView(RUN_ID)] }));
    await waitFor(() => runStatus() === CHAT.run.status.running);
    held.release();
    await waitFor(() => error.mock.calls.length === 1);
    expect(error.mock.calls[0]![0]).toBe(CHAT.run.stopError);
  });

  it('a wake turn shows Stop like any turn, and Stop cancels it', async () => {
    await open('/chat?c=p-wake', (text) => text.includes(RUN_TITLE), {
      routes: { '/chat/conversations/p-wake/cancel': {} },
      conversations: {
        'p-wake': sync('p-wake', {
          entries: runEntries,
          turn: { turnId: 'w1', phase: 'running', startedAt: NOW, wait: null, origin: 'wake' },
          runs: [runView(RUN_ID)],
        }),
      },
    });
    await waitFor(() => accessibleName(sendFace()) === CHAT.stop);
    click(sendFace());
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.path).toBe('/chat/conversations/p-wake/cancel');
  });

  it("a helper's request names who asks and where it works; the run card waits for you", async () => {
    await open('/chat?c=p-run-ask', (text) => text.includes(RUN_TITLE) && text.includes(CHAT.permRun), {
      conversations: {
        'p-run-ask': sync('p-run-ask', {
          entries: runEntries,
          turn: null,
          runs: [runView(RUN_ID)],
          interactions: [
            helperAsk({
              from: { label: 'researcher-0', run: 'deep-research', cwd: '/tmp/work/other', child: 'child-0' },
            }),
          ],
        }),
      },
    });
    const meta = shown(card()!);
    expect(askedBy()).toBe(fill(CHAT.permFrom, { label: 'researcher-0', run: 'deep-research' }));
    expect(card()!.querySelector('[data-works-in]')?.textContent).toBe('Works in /tmp/work/other');
    expect(meta).toContain(CHAT.denyRunNote);
    expect(meta).not.toContain(CHAT.denyNote);
    expect(runStatus()).toBe(CHAT.run.status.waitingYou);
    expect(shown(runCard()!)).toContain(fill(CHAT.run.asks[helperAsk().access], { label: 'researcher-0' }));
    expect(page!.container.querySelector('[data-wait-line]')).toBeNull();
  });

  it('a helper of no run says "Asked by helper"; a card with no helper names nobody', async () => {
    await open('/chat?c=p-plain-helper', (text) => text.includes(CHAT.permRun), {
      conversations: {
        'p-plain-helper': sync('p-plain-helper', {
          interactions: [helperAsk({ from: { label: 'explore-1', run: null, cwd: null, child: 'child-1' } })],
        }),
      },
    });
    expect(askedBy()).toBe(fill(CHAT.permFromNoRun, { label: 'explore-1' }));
    expect(shown(card()!)).toContain(CHAT.denyNote);
    expect(card()!.querySelector('[data-works-in]')).toBeNull();
    page!.unmount();
    api?.();
    await open('/chat?c=p-own', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-own': sync('p-own', { interactions: [permission()] }) },
    });
    expect(card()!.querySelector('[data-asked-by]')).toBeNull();
  });

  it("a run helper's open request does not hide the typed turn's wait line; the turn's own request does", async () => {
    await open('/chat?c=p-both', (text) => text.includes(RUN_TITLE) && text.includes(CHAT.permRun), {
      conversations: {
        'p-both': sync('p-both', {
          entries: [...runEntries, user('And the docs?', NOW), listed()],
          turn: { turnId: 't2', phase: 'running', startedAt: NOW, wait: null },
          runs: [runView(RUN_ID)],
          interactions: [helperAsk()],
        }),
      },
    });
    await waitFor(() => page!.container.querySelector('[data-wait-line]') !== null);
    page!.unmount();
    api?.();
    await open('/chat?c=p-own-ask', (text) => text.includes(CHAT.permRun), {
      conversations: {
        'p-own-ask': sync('p-own-ask', {
          entries: [user('Fix the build', NOW), listed()],
          interactions: [permission()],
        }),
      },
    });
    expect(page!.container.querySelector('[data-wait-line]')).toBeNull();
  });

  it('a live run keeps delete and the model picker off with its reason; rename stays on', async () => {
    await open('/chat?c=p-run-busy', (text) => text.includes(RUN_TITLE), {
      conversations: {
        'p-run-busy': sync('p-run-busy', {
          entries: runEntries,
          turn: null,
          runs: [runView(RUN_ID)],
        }),
      },
    });
    expect(button(CHAT.remove, page!.container).getAttribute('aria-disabled')).toBe('true');
    expect(button(CHAT.rename, page!.container).disabled).toBe(false);
    const picker = button(
      fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates.idle }),
      page!.container,
    );
    expect(picker.disabled).toBe(true);
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
    chooseFolder({ kind: 'folder', path: deep });
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
    type('Run the tests');
    await waitFor(() => shown().includes(CHAT.folderTooLong));
    expect(sendFace().disabled).toBe(true);
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    expect(writes()).toEqual([]);
  });

  it('with no Work conversation the place is No folder, and Send asks for a conversation without a folder', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkNoneTitle), {
      routes: { 'POST /chat/conversations': { conversation: info('p-made'), turnId: 't1' } },
    });
    expect(folderChips(tray())).toEqual([NONE_LABEL]);
    for (const text of CHAT.startersWorkNone) expect(button(text).disabled).toBe(false);
    type('Run the tests');
    act(() => sendFace().click());
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toMatchObject({ method: 'POST', path: '/chat/conversations' });
    const body = writes()[0]!.body as Record<string, unknown>;
    expect([body.noFolder, 'cwd' in body]).toEqual([true, false]);
  });

  it('the last Work folder is preselected and a new conversation is created in it', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle), {
      rows: [row('p-old', { updatedAt: NOW - 120_000 })],
      routes: { 'POST /chat/conversations': { conversation: info('p-made'), turnId: 't1' } },
    });
    await waitFor(() => folderChips(tray()).includes(fill(CHAT.folderLabel, { folder: FOLDER })));
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
    await waitFor(() => folderChips(tray()).includes(fill(CHAT.folderLabel, { folder: FOLDER })));
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

describe('Work folder chip: the tray on New, the bar once open', () => {
  const label = fill(CHAT.folderLabel, { folder: FOLDER });

  it('a new Work conversation with a folder: the folder in the tray, none in the composer or the bar', async () => {
    chooseFolder({ kind: 'folder', path: FOLDER });
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
    await waitFor(() => folderChips(tray()).length > 0);
    expect(folderChips(composerForm())).toEqual([]);
    expect(folderChips(bar())).toEqual([]);
    expect(folderChips(tray())).toEqual([label]);
    expect(composer().placeholder).toBe(fill(CHAT.placeholderWork, { folder: folderName(FOLDER) }));
  });

  it('a new Work conversation with no folder: "No folder" in the tray, nothing in the bar', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkNoneTitle));
    expect(folderChips(tray())).toEqual([NONE_LABEL]);
    expect(folderChips(bar())).toEqual([]);
  });

  it('an open Work conversation: no chip in the composer, the read-only one in the bar', async () => {
    await open('/chat?c=p-open', (text) => text.includes('Hi there'), {
      rows: [row('p-open')],
      conversations: { 'p-open': sync('p-open', { turn: null }) },
    });
    await waitFor(() => folderChips(bar()).length > 0);
    expect(folderChips(composerForm())).toEqual([]);
    expect(folderChips(bar())).toEqual([label]);
    click(button(label, bar()));
    await waitFor(() => document.querySelector(`[aria-label="${CHAT.folderPath}"]`) !== null);
    const popover = document.querySelector<HTMLElement>(`[aria-label="${CHAT.folderPath}"]`)!;
    expect(reachable(popover).some((b) => accessibleName(b) === CHAT.folderCopy)).toBe(true);
    expect(popover.querySelector('[role="radio"]')).toBeNull();
  });

  const OTHER = '/tmp/work/other';
  const otherLabel = fill(CHAT.folderLabel, { folder: OTHER });

  /** The purpose of every folder request the page sent. */
  let pickPurposes: unknown[] = [];

  /** The native dialog: every request is answered with `path`. Returns the undo. */
  function answerPicks(path: string): () => void {
    pickPurposes = [];
    const onMessage = (event: MessageEvent): void => {
      const data = event.data as { type?: unknown; requestId?: unknown; purpose?: unknown };
      if (data?.type !== PICK_FOLDER_MESSAGE) return;
      pickPurposes.push(data.purpose);
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { type: FOLDER_PICKED_MESSAGE, requestId: data.requestId, path },
          source: window,
        }),
      );
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }

  it('a folder chosen from the tray chip names it there and leaves focus in the composer', async () => {
    const undo = answerPicks(FOLDER);
    try {
      await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkNoneTitle));
      const chip = button(NONE_LABEL, tray());
      act(() => chip.focus());
      expect(document.activeElement).toBe(chip);
      click(chip);
      await waitFor(() => document.querySelector(`[role="dialog"][aria-label="${CHAT.folderPath}"]`) !== null);
      click(button(CHAT.folderChoose, document.querySelector<HTMLElement>(`[aria-label="${CHAT.folderPath}"]`)!));
      await waitFor(() => folderChips(tray()).includes(label));
      // Mutation: ask for the Locate dialog → MAIN opens "Locate model files" → red.
      expect(pickPurposes).toEqual(['work']);
      expect(folderChips(composerForm())).toEqual([]);
      await waitFor(() => document.activeElement === composer());
    } finally {
      undo();
    }
  });

  it('the tray chip with an empty Work list offers No folder and "Choose folder…", which shows the folder picked there', async () => {
    chooseFolder({ kind: 'folder', path: FOLDER });
    const undo = answerPicks(OTHER);
    try {
      await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
      await waitFor(() => folderChips(tray()).includes(label));
      click(button(label, tray()));
      const popover = (): HTMLElement | null =>
        document.querySelector<HTMLElement>(`[role="dialog"][aria-label="${CHAT.folderPath}"]`);
      await waitFor(() => popover() !== null);
      expect([...popover()!.querySelectorAll('[role="radio"]')].map((o) => o.textContent)).toEqual([
        `${CHAT.folderNone}${fill(CHAT.folderNoneNote, { root: READY_CHAT_HOST.scratchRoot })}`,
      ]);
      click(button(CHAT.folderChoose, popover()!));
      await waitFor(() => folderChips(tray()).includes(otherLabel));
      expect(popover()).toBeNull();
      expect(composer().placeholder).toBe(fill(CHAT.placeholderWork, { folder: folderName(OTHER) }));
      expect(folderChips(composerForm())).toEqual([]);
      await waitFor(() => document.activeElement === composer());
    } finally {
      undo();
    }
  });

  it('a recent folder picked from the tray chip leaves focus in the composer, even with a frame that lands late', async () => {
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle), {
      rows: [row('p-a', { updatedAt: NOW - 1_000 }), row('p-b', { cwd: OTHER, updatedAt: NOW - 2_000 })],
    });
    await waitFor(() => folderChips(tray()).includes(label));
    const chip = button(label, tray());
    act(() => chip.focus());
    click(chip);
    const option = (): HTMLElement | undefined =>
      [...document.querySelectorAll<HTMLElement>(`[aria-label="${CHAT.folderPath}"] [role="radio"]`)].find((o) =>
        (o.textContent ?? '').includes('other'),
      );
    await waitFor(() => option() !== undefined);
    // The frame lands after the popover unmounts and before Radix's restore-focus timer (a slow frame in a browser).
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
      queueMicrotask(() => callback(performance.now()));
      return 0;
    });
    click(option()!);
    await waitFor(() => folderChips(tray()).includes(otherLabel));
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 50)));
    expect(document.querySelector(`[role="dialog"][aria-label="${CHAT.folderPath}"]`)).toBeNull();
    expect(document.activeElement).toBe(composer());
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

describe('Work: needs you', () => {
  const own = permission({ id: 'i-own', command: 'rm -rf build/', openedAt: NOW - 42_000 });
  const sub = permission({
    id: 'i-sub',
    access: 'edit',
    command: null,
    path: '/tmp/work/proj/src/app.ts',
    openedAt: NOW - 10_000,
    from: { label: 'fix tests', run: null, cwd: null, child: 'child-1' },
  });
  const plan: PlanInteraction = { ...PLAN, openedAt: NOW - 5_000 };
  const runAsk = permission({
    id: 'i-run',
    command: 'cargo publish',
    from: { label: 'packer', run: 'Release', cwd: null, child: 'child-2' },
  });
  const rail = (): HTMLElement | null => page!.container.querySelector<HTMLElement>('section[data-needs]');
  const rows = (): HTMLButtonElement[] => [...(rail()?.querySelectorAll('button') ?? [])];
  const activityButton = (): HTMLButtonElement =>
    [...bar().querySelectorAll('button')].find((b) =>
      [CHAT.activity.show, CHAT.activity.hide].some((name) => accessibleName(b).startsWith(name)),
    )!;
  const said = (): string[] => allByRole(page!.container, 'status').map((node) => node.textContent ?? '');

  it('the rail lists every open request oldest first; a row shows its card and focuses its first button', async () => {
    await open('/chat?c=p-needs', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-needs': sync('p-needs', { interactions: [own, sub, plan] }) },
    });
    await waitFor(() => rows().length === 3);
    expect(rows().map((row) => row.getAttribute('aria-current'))).toEqual(['true', null, null]);
    expect(accessibleName(activityButton())).toBe(`${CHAT.activity.hide}, ${fill(CHAT.needs.count, { n: 3 })}`);
    expect(activityButton().querySelector('[data-tone]')?.getAttribute('data-tone')).toBe('warning');
    expect(shown(card()!)).toContain(fill(CHAT.permCount, { i: 1, n: 3 }));
    click(rows()[2]!);
    await waitFor(() => card()?.getAttribute('data-ask-id') === plan.id);
    await waitFor(
      () => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.planFeedback,
    );
    expect(shown(card()!)).toContain(fill(CHAT.permCount, { i: 3, n: 3 }));
    expect(rows().map((row) => row.getAttribute('aria-current'))).toEqual([null, null, 'true']);
  });

  it("the composer says to answer on the card while its own answer asks; a run helper's ask leaves it alone", async () => {
    await open('/chat?c=p-own-ph', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-own-ph': sync('p-own-ph', { interactions: [own] }) },
    });
    expect(composer().placeholder).toBe(CHAT.needs.composerPlaceholder);
    page!.unmount();
    api?.();
    await open('/chat?c=p-run-ph', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-run-ph': sync('p-run-ph', { turn: null, interactions: [runAsk] }) },
    });
    expect(composer().placeholder).toBe(CHAT.placeholderReply);
    await waitFor(() => rows().length === 1);
  });

  // Mutation: let `ownAsks` keep every MCP card → the box says "Answer on the card above…" with no answer running.
  it("a run's MCP request leaves the composer alone too; the rail says the run asks", async () => {
    const runMcp: McpInteraction = {
      id: 'i-run-mcp',
      kind: 'mcp-input',
      toolCallId: 'mcp-elicit-run',
      openedAt: NOW - 4_000,
      mode: 'form',
      server: 'github',
      message: 'Which repository?',
      form: { fields: [], leftOut: 0 },
      run: 'deep-research',
    };
    await open('/chat?c=p-run-mcp', (text) => text.includes(fill(CHAT.mcp.formTitle, { server: 'github' })), {
      conversations: { 'p-run-mcp': sync('p-run-mcp', { turn: null, interactions: [runMcp] }) },
    });
    expect(composer().placeholder).toBe(CHAT.placeholderReply);
    await waitFor(() => rows().length === 1);
    expect(rows()[0]!.textContent).toContain(fill(CHAT.needs.fromRun, { run: 'deep-research' }));
    page!.unmount();
    api?.();
    const { run: _run, ...own } = runMcp;
    await open('/chat?c=p-own-mcp', (text) => text.includes(fill(CHAT.mcp.formTitle, { server: 'github' })), {
      conversations: { 'p-own-mcp': sync('p-own-mcp', { interactions: [own] }) },
    });
    expect(composer().placeholder).toBe(CHAT.needs.composerPlaceholder);
  });

  it('scrolled away: one short strip line says how many wait, and Go to request brings the card into focus', async () => {
    stick.away = true;
    await open('/chat?c=p-strip', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-strip': sync('p-strip', { interactions: [own, runAsk] }) },
    });
    const strip = composerForm().querySelector<HTMLElement>('[data-tone="neutral"]')!;
    expect(strip.textContent).toContain(CHAT.needs.composerOne);
    expect(CHAT.needs.composerOne.length).toBeLessThan(50);
    expect(fill(CHAT.needs.composerMany, { n: 12 }).length).toBeLessThan(50);
    expect(strip.getAttribute('title')).toBe(fill(CHAT.needs.composerOneTip, { title: 'rm -rf build/' }));
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    click(button(CHAT.needs.composerGo, strip));
    await waitFor(() => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.deny);
    expect(card()!.getAttribute('data-ask-id')).toBe(own.id);
  });

  it('pinned to the end, the card is in view: no strip', async () => {
    await open('/chat?c=p-nostrip', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-nostrip': sync('p-nostrip', { interactions: [own] }) },
    });
    expect(composerForm().textContent).not.toContain(CHAT.needs.composerGo);
  });

  it('a growing count is said once; a request the agent withdraws is said', async () => {
    await open('/chat?c=p-said', (text) => text.includes(CHAT.permRun), {
      conversations: { 'p-said': sync('p-said', { interactions: [own] }) },
    });
    act(() => api!.emitChat('p-said', { type: 'interaction', seq: 1, interaction: sub }));
    await waitFor(() => said().includes(fill(CHAT.status.asksMore, { n: 2 })));
    act(() => api!.emitChat('p-said', { type: 'interaction-closed', seq: 2, id: sub.id, reason: 'expired' }));
    await waitFor(() => said().includes(CHAT.status.askWithdrawn));
    await waitFor(() => rows().length === 1);
    expect(card()!.getAttribute('data-ask-id')).toBe(own.id);
  });

  it('in the drawer a row closes the drawer first, then focus lands on the card', async () => {
    const restore = observeWidth(1000);
    try {
      await open('/chat?c=p-drawer', (text) => text.includes(CHAT.permRun), {
        conversations: { 'p-drawer': sync('p-drawer', { interactions: [own, plan] }) },
      });
      await waitFor(
        () => accessibleName(activityButton()) === `${CHAT.activity.show}, ${fill(CHAT.needs.count, { n: 2 })}`,
      );
      expect(rail()).toBeNull();
      click(activityButton());
      await waitFor(() => byRole(document.body, 'dialog', CHAT.activity.label) !== null);
      const row = [...document.body.querySelectorAll<HTMLButtonElement>('section[data-needs] button')][1]!;
      click(row);
      await waitFor(() => byRole(document.body, 'dialog', CHAT.activity.label) === null);
      await waitFor(
        () => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.planFeedback,
      );
    } finally {
      restore();
    }
  });

  it('the docked rail opens by itself for a request with the "auto" preference, not with "off"', async () => {
    await open('/chat?c=p-auto-rail', (text) => text.includes(CHAT.permRun), {
      conversations: {
        'p-auto-rail': sync('p-auto-rail', { turn: null, entries: [user('Hi', NOW)], interactions: [own] }),
      },
    });
    await waitFor(() => page!.container.querySelector('[data-rail]')?.getAttribute('data-rail') === 'open');
    expect(rows()).toHaveLength(1);
    page!.unmount();
    api?.();
    localStorage.setItem('mlx.chat.activityRail', 'off');
    await open('/chat?c=p-off-rail', (text) => text.includes(CHAT.permRun), {
      conversations: {
        'p-off-rail': sync('p-off-rail', { turn: null, entries: [user('Hi', NOW)], interactions: [own] }),
      },
    });
    expect(page!.container.querySelector('[data-rail]')?.getAttribute('data-rail')).toBe('closed');
    expect(accessibleName(activityButton())).toBe(`${CHAT.activity.show}, ${CHAT.needs.countOne}`);
  });
});

describe('Work: deny with a note', () => {
  const NOTE = 'use pnpm, not npm';
  const said = (): string[] => allByRole(page!.container, 'status').map((node) => node.textContent ?? '');
  const field = (): HTMLTextAreaElement | null =>
    card() === null ? null : (byRole(card()!, 'textbox', CHAT.denyNoteLabel) as HTMLTextAreaElement | null);

  async function writeStep(id: string, interaction: PermissionInteraction = permission()): Promise<void> {
    await open(`/chat?c=${id}`, (text) => text.includes(CHAT.permRun), {
      routes: { [`POST /chat/conversations/${id}/interactions/${interaction.id}`]: {} },
      conversations: { [id]: sync(id, { interactions: [interaction] }) },
    });
    const more = button(CHAT.denyMore, card()!);
    expect(more.getAttribute('aria-haspopup')).toBe('menu');
    click(more);
    await waitFor(() => byRole(document.body, 'menu', CHAT.denyMore) !== null);
    click(byRole(document.body, 'menuitem', CHAT.denyWhy)!);
    await waitFor(() => field() !== null);
  }

  it('Deny stays the first button; the arrow opens "Deny and say why…"; the note step sends reject-once with the note', async () => {
    await writeStep('n-send');
    const area = field()!;
    await waitFor(() => document.activeElement === area);
    expect(shown(card()!)).toContain(CHAT.denyWritingNote);
    expect(shown(card()!)).toContain(fill(CHAT.feedbackCount, { n: 0, max: FEEDBACK_MAX_CHARS }));
    expect(area.maxLength).toBe(MAX_FEEDBACK_CHARS);
    expect(button(CHAT.denySend, card()!).disabled).toBe(true);
    type('   ', area);
    expect(button(CHAT.denySend, card()!).disabled).toBe(true);
    type(`  ${NOTE}  `, area);
    click(button(CHAT.denySend, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      {
        method: 'POST',
        path: '/chat/conversations/n-send/interactions/i1',
        body: { optionId: 'reject-once', note: NOTE },
      },
    ]);
    await waitFor(() => said().includes(CHAT.status.deniedNote));
    await waitFor(() => document.activeElement === composer());
  });

  it('the buttons: Deny first with its arrow beside it, then the allow buttons; Deny alone still sends reject-once', async () => {
    await open('/chat?c=n-plain', (text) => text.includes(CHAT.permRun), {
      routes: { 'POST /chat/conversations/n-plain/interactions/i1': {} },
      conversations: { 'n-plain': sync('n-plain', { interactions: [permission()] }) },
    });
    const order = reachable(card()!).map((b) => accessibleName(b));
    expect(order.slice(0, 4)).toEqual([CHAT.deny, CHAT.denyMore, CHAT.allowOnce, CHAT.allowAlwaysCommand]);
    expect(shown(card()!)).toContain(`${CHAT.denyNote} ${CHAT.grantsNote}`);
    click(button(CHAT.deny, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ optionId: 'reject-once' });
  });

  it('Escape in the note field goes back to the buttons with the focus on Deny; Cmd+Enter sends', async () => {
    await writeStep('n-escape');
    act(() => {
      field()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => field() === null);
    await waitFor(() => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.deny);
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menuitem', CHAT.denyWhy) !== null);
    click(byRole(document.body, 'menuitem', CHAT.denyWhy)!);
    await waitFor(() => field() !== null);
    const area = field()!;
    type(NOTE, area);
    act(() => {
      area.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });
    expect(writes()).toEqual([]);
    act(() => {
      area.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', metaKey: true, bubbles: true }));
    });
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ optionId: 'reject-once', note: NOTE });
  });

  it('a send that fails keeps the field and its text', async () => {
    const error = vi.spyOn(toast, 'error');
    await open('/chat?c=n-fail', (text) => text.includes(CHAT.permRun), {
      routes: { 'POST /chat/conversations/n-fail/interactions/i1': stubError('E_UNAVAILABLE', 'down') },
      conversations: { 'n-fail': sync('n-fail', { interactions: [permission()] }) },
    });
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menuitem', CHAT.denyWhy) !== null);
    click(byRole(document.body, 'menuitem', CHAT.denyWhy)!);
    await waitFor(() => field() !== null);
    const area = field()!;
    type(NOTE, area);
    click(button(CHAT.denySend, card()!));
    await waitFor(() => error.mock.calls.length === 1);
    expect(error).toHaveBeenCalledWith(CHAT.sendErrors.other[0]);
    await waitFor(() => !button(CHAT.denySend, card()!).disabled);
    expect(area.isConnected).toBe(true);
    expect(area.value).toBe(NOTE);
  });

  it("a helper's card names the helper in the menu and says what Deny does to it", async () => {
    const helper = permission({ from: { label: 'tests', run: 'ci-run', cwd: null, child: 'child-1' } });
    await open('/chat?c=n-helper', (text) => text.includes(CHAT.permRun), {
      conversations: { 'n-helper': sync('n-helper', { interactions: [helper] }) },
    });
    expect(shown(card()!)).toContain(CHAT.denyRunNote);
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menu', CHAT.denyMore) !== null);
    expect(byRole(document.body, 'menuitem', fill(CHAT.denyWhyHelper, { label: 'tests' }))).not.toBeNull();
  });

  it('a card that offers no reject-once has no Deny and no arrow', async () => {
    const options: PermissionInteraction['options'] = [{ optionId: 'allow-once', kind: 'allow_once', name: 'Yes' }];
    await open('/chat?c=n-none', (text) => text.includes(CHAT.permRun), {
      conversations: { 'n-none': sync('n-none', { interactions: [permission({ options })] }) },
    });
    const names = reachable(card()!).map((b) => accessibleName(b));
    expect(names).not.toContain(CHAT.deny);
    expect(names).not.toContain(CHAT.denyMore);
  });

  it('the thread after: the row says "Denied · note sent", the note is a row of its own, a helper row counts its denied call', async () => {
    const tool: TranscriptEntry = {
      kind: 'tool',
      ts: NOW + 10,
      id: 'c1',
      name: 'run_terminal_command',
      title: 'npm install left-pad',
      status: 'failed',
      input: '{"command":"npm install left-pad"}',
      output: 'The user elected to avoid running the run_terminal_command tool. The tool was not executed.',
      endTs: NOW + 20,
    };
    const spawn: TranscriptEntry = {
      kind: 'subagent',
      ts: NOW + 30,
      childId: 'child-1',
      type: 'general-purpose',
      description: 'Run the unit tests',
    };
    const note: TranscriptEntry = { kind: 'user', ts: NOW + 25, text: NOTE, promptIndex: null, note: true };
    const marks = [
      { kind: 'denied', at: NOW + 21, toolCallId: 'c1', access: 'bash', by: 'note', card: null },
      { kind: 'denied', at: NOW + 40, toolCallId: 'h1', access: 'bash', by: 'note', card: null, child: 'child-1' },
    ] as const;
    await open('/chat?c=n-thread', (text) => text.includes(CHAT.noteRow), {
      conversations: {
        'n-thread': sync('n-thread', {
          turn: null,
          entries: [user('Add left-pad', NOW), tool, note, spawn, answer('Added with pnpm.', NOW + 50), end()],
          permissionMarks: [...marks],
        }),
      },
    });
    const row = page!.container.querySelector('[data-note-row]');
    expect(row?.textContent).toContain(CHAT.noteRow);
    expect(row?.textContent).toContain(NOTE);
    expect(shown()).toContain(CHAT.permission.deniedNote);
    expect(shown()).toContain(CHAT.permission.subagentDeniedOne);
  });
});

describe('Work: never allow (#14)', () => {
  const RULE = `n_${'a'.repeat(32)}`;
  const OTHER = `n_${'b'.repeat(32)}`;
  const NEVER = { short: 'cargo test', exact: 'cargo test --workspace' };
  const pickStep = (): HTMLElement | null => card()?.querySelector<HTMLElement>('[data-never-pick]') ?? null;
  const said = (): string[] => allByRole(page!.container, 'status').map((node) => node.textContent ?? '');

  async function neverStep(id: string, interaction: PermissionInteraction, result: unknown = {}): Promise<void> {
    await open(`/chat?c=${id}`, (text) => text.includes(CHAT.permRun), {
      routes: {
        [`POST /chat/conversations/${id}/interactions/${interaction.id}`]: result,
        [`DELETE /chat/conversations/${id}/never/${RULE}`]: {},
      },
      conversations: { [id]: sync(id, { interactions: [interaction] }) },
    });
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menu', CHAT.denyMore) !== null);
    click(byRole(document.body, 'menuitem', neverMenuText(interaction.never!))!);
    await waitFor(() => pickStep() !== null);
  }

  it('the arrow menu has "Never allow cargo test…"; the step offers both scopes and sends the one picked; the toast undoes it', async () => {
    const success = vi.spyOn(toast, 'success');
    const update = vi.spyOn(toast, 'update');
    await neverStep('nv-pick', permission({ never: NEVER }), {
      never: { saved: true, ruleId: RULE, words: NEVER.exact },
    });
    expect(neverMenuText(NEVER)).toBe(fill(CHAT.never.menu, { words: 'cargo test' }));
    expect(shown(card()!)).toContain(CHAT.never.note);
    const short = byRole(card()!, 'radio', NEVER.short)!;
    const exact = byRole(card()!, 'radio', NEVER.exact)!;
    await waitFor(() => document.activeElement === short);
    expect(short.getAttribute('aria-checked')).toBe('true');
    expect(byRole(card()!, 'radiogroup', CHAT.never.pick)).not.toBeNull();
    click(exact);
    expect(exact.getAttribute('aria-checked')).toBe('true');
    expect(writes()).toEqual([]);
    click(button(CHAT.never.confirm, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/nv-pick/interactions/i1',
      body: { never: 'exact' },
    });
    await waitFor(() => success.mock.calls.length === 1);
    const [title, options] = success.mock.calls[0]!;
    expect(title).toBe(fill(CHAT.never.toast, { words: NEVER.exact }));
    expect(options?.action?.label).toBe(CHAT.never.undo);
    await waitFor(() => document.activeElement === composer());
    act(() => options!.action!.onClick('t-never'));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({
      method: 'DELETE',
      path: `/chat/conversations/nv-pick/never/${RULE}`,
      body: undefined,
    });
    await waitFor(() => update.mock.calls.some(([id, patch]) => id === 't-never' && patch.title === CHAT.never.undone));
    await waitFor(() => said().includes(CHAT.status.neverRemoved));
  });

  it('one scope shows its words as a line and sends short; Escape goes back to Deny; a rule not saved says so', async () => {
    const error = vi.spyOn(toast, 'error');
    await neverStep('nv-one', permission({ command: 'rm -rf build', never: { short: 'rm', exact: 'rm' } }), {
      never: { saved: false },
    });
    expect(byRole(card()!, 'radiogroup', CHAT.never.pick)).toBeNull();
    expect(card()!.querySelector('[data-never-words]')?.textContent).toBe('rm');
    await waitFor(() => document.activeElement === button(CHAT.never.confirm, card()!));
    act(() => {
      document.activeElement!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });
    await waitFor(() => pickStep() === null);
    await waitFor(() => document.activeElement !== null && accessibleName(document.activeElement) === CHAT.deny);
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menuitem', fill(CHAT.never.menu, { words: 'rm' })) !== null);
    click(byRole(document.body, 'menuitem', fill(CHAT.never.menu, { words: 'rm' }))!);
    await waitFor(() => pickStep() !== null);
    click(button(CHAT.never.confirm, card()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]!.body).toEqual({ never: 'short' });
    await waitFor(() => error.mock.calls.length === 1);
    expect(error).toHaveBeenCalledWith(CHAT.never.saveFailed);
  });

  it('a card with no scopes, or one for a file, has only "Deny and say why…" in the menu', async () => {
    await open('/chat?c=nv-none', (text) => text.includes(CHAT.permRun), {
      conversations: { 'nv-none': sync('nv-none', { interactions: [permission()] }) },
    });
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menu', CHAT.denyMore) !== null);
    const items = allByRole(byRole(document.body, 'menu', CHAT.denyMore)!, 'menuitem').map((n) => accessibleName(n));
    expect(items).toEqual([CHAT.denyWhy]);
    page!.unmount();
    api!();
    const edit = permission({ access: 'edit', command: null, path: 'src/a.ts', resolvedPath: null, never: NEVER });
    await open('/chat?c=nv-edit', (text) => text.includes('Hi there'), {
      conversations: { 'nv-edit': sync('nv-edit', { interactions: [edit] }) },
    });
    await waitFor(() => card() !== null);
    click(button(CHAT.denyMore, card()!));
    await waitFor(() => byRole(document.body, 'menu', CHAT.denyMore) !== null);
    expect(byRole(document.body, 'menuitem', neverMenuText(NEVER))).toBeNull();
  });

  it("the mode chip counts the folder's rules; its view lists them, removes one, and goes back to the row", async () => {
    const rules = [
      { id: RULE, words: 'git push', at: NOW - 1_000 },
      { id: OTHER, words: 'rm -rf', at: NOW - 2_000 },
    ];
    await open('/chat?c=nv-chip', (text) => text.includes('Hi there'), {
      routes: {
        'GET /chat/conversations/nv-chip/never': { folder: FOLDER, rules },
        [`DELETE /chat/conversations/nv-chip/never/${RULE}`]: {},
      },
      conversations: { 'nv-chip': sync('nv-chip', { turn: null }, { neverRules: 2 }) },
    });
    const chip = page!.container.querySelector<HTMLButtonElement>('[data-permission-chip]')!;
    click(chip);
    await waitFor(() => document.querySelector('[data-never-chip-row]') !== null);
    const row = document.querySelector<HTMLButtonElement>('[data-never-chip-row]')!;
    expect(row.textContent).toBe(fill(CHAT.never.chipRow, { n: 2 }));
    click(row);
    await waitFor(() => document.querySelectorAll('[data-never-rule]').length === 2);
    const view = document.querySelector<HTMLElement>('[data-never-rules]')!;
    expect(shown(view)).toContain(fill(CHAT.never.listTitle, { folder: folderName(FOLDER) }));
    expect(shown(view)).toContain(CHAT.never.listNote);
    await waitFor(() => document.activeElement === button(CHAT.never.back, view));
    expect([...view.querySelectorAll('[data-never-rule]')].map((r) => r.getAttribute('data-never-rule'))).toEqual([
      RULE,
      OTHER,
    ]);
    click(button(fill(CHAT.never.removeLabel, { words: 'git push' }), view));
    await waitFor(() => view.querySelectorAll('[data-never-rule]').length === 1);
    expect(writes()).toEqual([
      { method: 'DELETE', path: `/chat/conversations/nv-chip/never/${RULE}`, body: undefined },
    ]);
    await waitFor(() => document.activeElement === button(fill(CHAT.never.removeLabel, { words: 'rm -rf' }), view));
    click(button(CHAT.never.back, view));
    await waitFor(() => document.querySelector('[data-never-rules]') === null);
    await waitFor(() => document.activeElement === document.querySelector('[data-never-chip-row]'));
  });

  it('no rules, no row', async () => {
    await open('/chat?c=nv-zero', (text) => text.includes('Hi there'), {
      conversations: { 'nv-zero': sync('nv-zero', { turn: null }) },
    });
    click(page!.container.querySelector<HTMLButtonElement>('[data-permission-chip]')!);
    await waitFor(() => document.querySelector('[role="radiogroup"]') !== null);
    expect(document.querySelector('[data-never-chip-row]')).toBeNull();
  });

  it('the thread: "Never allowed · git push", the rule note as "Your rule" with Remove rule; a rule removed elsewhere reads "rule removed"', async () => {
    const tool: TranscriptEntry = {
      kind: 'tool',
      ts: NOW + 10,
      id: 'c1',
      name: 'run_terminal_command',
      title: 'git push origin',
      status: 'failed',
      input: '{"command":"git push origin"}',
      output: 'The user elected to avoid running the run_terminal_command tool. The tool was not executed.',
      endTs: NOW + 20,
    };
    const text = `${NEVER_NOTE_HEAD}git push${NEVER_NOTE_TAIL}`;
    const note: TranscriptEntry = { kind: 'user', ts: NOW + 25, text, promptIndex: null, note: true };
    await open('/chat?c=nv-thread', (shownText) => shownText.includes(CHAT.never.ruleLine), {
      routes: { [`DELETE /chat/conversations/nv-thread/never/${RULE}`]: {} },
      conversations: {
        'nv-thread': sync(
          'nv-thread',
          {
            turn: null,
            entries: [user('Ship it', NOW), tool, note, answer('Not pushed.', NOW + 50), end()],
            permissionMarks: [
              {
                kind: 'denied',
                at: NOW + 21,
                toolCallId: 'c1',
                access: 'bash',
                by: 'rule',
                ruleId: RULE,
                words: 'git push',
                card: null,
              },
            ],
          },
          { neverRules: 1 },
        ),
      },
    });
    expect(shown()).toContain(fill(CHAT.never.row, { words: 'git push' }));
    expect(page!.container.querySelector('[data-note-row]')).toBeNull();
    const ruleRow = page!.container.querySelector<HTMLElement>('[data-rule-row]')!;
    expect(shown(ruleRow)).toContain(fill(CHAT.never.ruleText, { words: 'git push' }));
    expect(shown(ruleRow)).not.toContain(NEVER_NOTE_TAIL);
    click(button(fill(CHAT.never.removeLabel, { words: 'git push' }), ruleRow));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'DELETE',
      path: `/chat/conversations/nv-thread/never/${RULE}`,
      body: undefined,
    });
    act(() => api!.emitChat('nv-thread', { type: 'rule-gone', seq: 1, ruleId: RULE }));
    await waitFor(() => shown().includes(CHAT.never.rowGone));
    expect(reachable(ruleRow).map((b) => accessibleName(b))).not.toContain(
      fill(CHAT.never.removeLabel, { words: 'git push' }),
    );
  });

  it('a call a rule denies while the page is open is said in the status line', async () => {
    await open('/chat?c=nv-live', (text) => text.includes('Hi there'), {
      conversations: { 'nv-live': sync('nv-live', {}, { neverRules: 1 }) },
    });
    act(() =>
      api!.emitChat('nv-live', {
        type: 'permission-mark',
        seq: 1,
        mark: {
          kind: 'denied',
          at: NOW + 5,
          toolCallId: 'c9',
          access: 'bash',
          by: 'rule',
          ruleId: RULE,
          words: 'git push',
          card: null,
        },
      }),
    );
    await waitFor(() => said().includes(fill(CHAT.status.neverDenied, { words: 'git push' })));
  });
});

describe('Work: the "Plan first" chip', () => {
  const planChip = (): HTMLButtonElement => button(CHAT.planMode.chip, composerForm());
  const modeLine = (mode: 'plan' | 'default', ts: number): TranscriptEntry => ({ kind: 'plan-mode', ts, mode });
  const noChip = (): boolean => !reachable(composerForm()).some((b) => accessibleName(b) === CHAT.planMode.chip);
  // `/plan` + Enter turns the toggle on and leaves the box empty.
  const armPlan = (): void => {
    type('/plan');
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
  };
  const statusText = (): string =>
    [...page!.container.querySelectorAll('[role="status"]')].map((n) => n.textContent ?? '').join(' ');

  it('a press arms the next message, which carries `plan`; the agent planning turns it to Planning; Chat has none', async () => {
    await open('/chat?c=p-planchip', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-planchip/prompt': {} },
      conversations: { 'p-planchip': sync('p-planchip', { turn: null }) },
    });
    expect(noChip()).toBe(true);
    armPlan();
    expect(composer().value).toBe('');
    expect(planChip().getAttribute('aria-pressed')).toBe('true');
    expect(planChip().textContent).toBe(CHAT.planMode.chip);
    type('Plan the cache');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-planchip/prompt',
      body: { text: 'Plan the cache', plan: true },
    });
    act(() =>
      api!.emitChat('p-planchip', {
        type: 'ops',
        seq: 1,
        ops: [
          { op: 'push', entry: user('Plan the cache', NOW + 1) },
          { op: 'push', entry: modeLine('plan', NOW + 2) },
        ],
      }),
    );
    await waitFor(() => planChip().textContent === CHAT.planMode.chipOn);
    expect(planChip().getAttribute('aria-pressed')).toBe('true');
    expect(shown()).toContain(CHAT.planMode.on);
    expect(statusText()).toContain(CHAT.planMode.status.on);
  });

  it('the chip follows the agent: Planning on open, a press turns it off while an answer runs, and the next message carries `plan: false`', async () => {
    await open('/chat?c=p-planlive', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-planlive/prompt': {} },
      conversations: {
        'p-planlive': sync('p-planlive', {
          entries: [user('Plan it'), modeLine('plan', NOW), answer('Hi there')],
        }),
      },
    });
    expect(planChip().textContent).toBe(CHAT.planMode.chipOn);
    expect(planChip().getAttribute('aria-pressed')).toBe('true');
    expect(planChip().disabled).toBe(false);
    // A mode line the conversation opened with is not news.
    expect(statusText()).not.toContain(CHAT.planMode.status.on);
    click(planChip());
    expect(planChip().getAttribute('aria-pressed')).toBe('false');
    expect(planChip().textContent).toBe(CHAT.planMode.chip);
    act(() => api!.emitChat('p-planlive', { type: 'turn', seq: 1, turn: null, turnId: 't1', outcome: 'completed' }));
    await waitFor(() => accessibleName(sendFace()) !== CHAT.stop);
    expect(planChip().getAttribute('aria-pressed')).toBe('false');
    type('Build it');
    click(sendFace());
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-planlive/prompt',
      body: { text: 'Build it', plan: false },
    });
  });

  it('the "no plan mode" notice turns an armed chip off once; a press after it holds', async () => {
    await open('/chat?c=p-noplan', (text) => text.includes('Hi there'), {
      conversations: { 'p-noplan': sync('p-noplan', { turn: null }) },
    });
    armPlan();
    expect(planChip().getAttribute('aria-pressed')).toBe('true');
    act(() => api!.emitChat('p-noplan', { type: 'notice', seq: 1, notice: { code: 'no-plan-mode', tone: 'warning' } }));
    await waitFor(() => noChip());
    expect(shown()).toContain(CHAT.planMode.notice);
    armPlan();
    expect(planChip().getAttribute('aria-pressed')).toBe('true');
  });

  it('switching to a conversation in plan mode shows Planning and says nothing', async () => {
    await open('/chat?c=p-planfrom', (text) => text.includes('Hi there'), {
      rows: [row('p-planfrom'), row('p-planto')],
      conversations: {
        'p-planfrom': sync('p-planfrom', { turn: null }),
        'p-planto': sync('p-planto', {
          turn: null,
          entries: [user('Plan it'), modeLine('plan', NOW), answer('Planned answer')],
        }),
      },
    });
    expect(noChip()).toBe(true);
    const nav = page!.container.querySelector(`nav[aria-label="${CHAT.conversations}"]`)!;
    const link = [...nav.querySelectorAll('a')].find((a) => a.textContent?.includes('Title p-planto') === true)!;
    click(link);
    await waitFor(() => shown().includes('Planned answer') && planChip().textContent === CHAT.planMode.chipOn);
    expect(statusText()).not.toContain(CHAT.planMode.status.on);
  });

  it('a Chat conversation has no chip', async () => {
    await open('/chat?c=p-planchat', (text) => text.includes('Hi there'), {
      conversations: { 'p-planchat': sync('p-planchat', { turn: null }, { mode: 'chat', cwd: null }) },
    });
    expect(reachable(composerForm()).some((b) => accessibleName(b) === CHAT.planMode.chip)).toBe(false);
  });
});

describe('background work holds the agent process (H-a)', () => {
  const ID = 'p-bg';
  type Active = ChatHostState['active'][number];
  const here = (over: Partial<Active> = {}): Active => ({
    conversationId: ID,
    mode: 'work',
    busy: false,
    waiting: false,
    asks: 0,
    background: false,
    helpers: false,
    tasks: 0,
    op: null,
    ...over,
  });
  const other = (tasks: number): Active => ({ ...here({ tasks }), conversationId: 'p-bg-other' });
  const hostWith = (work: Partial<ChatHostState['modes']['work']>, active: Active[]): ChatHostState => ({
    ...READY_CHAT_HOST,
    modes: { ...READY_CHAT_HOST.modes, work: { ...READY_CHAT_HOST.modes.work, state: 'ready', ...work } },
    active,
  });
  const pushHost = (state: ChatHostState): void => act(() => api!.emitChatHost({ type: 'state', state }));
  const strips = (): HTMLElement[] =>
    [...page!.container.querySelectorAll<HTMLElement>('[role="status"][title]')].filter(
      (n) => n.closest('form') !== null,
    );
  const engineStrip = (): HTMLElement | undefined =>
    strips().find((n) => n.getAttribute('title')?.startsWith(CHAT.engineRestartTip) === true);
  const compactRow = (): HTMLElement | null => byRole(document.body, 'menuitem', CHAT.history.compact);
  const noteOf = (row: HTMLElement): string =>
    document.getElementById(row.getAttribute('aria-describedby') ?? '')?.textContent ?? '';
  const tooltipOf = async (trigger: HTMLElement): Promise<string> => {
    await act(async () => trigger.focus());
    await waitFor(() => byRole(document.body, 'tooltip') !== null);
    // What a screen reader speaks: the tooltip's `aria-hidden` width measure is left out.
    const spoken = (node: Node): string =>
      node instanceof Element && node.getAttribute('aria-hidden') === 'true'
        ? ''
        : node.nodeType === Node.TEXT_NODE
          ? (node.textContent ?? '')
          : [...node.childNodes].map(spoken).join('');
    const text = spoken(byRole(document.body, 'tooltip')!);
    await act(async () => trigger.blur());
    await waitFor(() => byRole(document.body, 'tooltip') === null);
    return text;
  };
  const deleteTrigger = (): HTMLElement => {
    const found = reachable(bar()).find((b) => accessibleName(b) === CHAT.remove);
    if (found === undefined) throw new Error('no delete');
    return found;
  };
  const picker = (): HTMLButtonElement =>
    button(fill(CHAT.pillName, { model: 'qwen3.5-4b MLX', state: CHAT.pillStates.idle }), page!.container);
  const settle = (): Promise<void> => act(() => new Promise<void>((resolve) => setTimeout(resolve, 20)));

  it('the engine restarted under live commands: the strip counts every conversation’s, says why in its title; only then', async () => {
    await open(`/chat?c=${ID}`, (text) => text.includes('Hi there'), {
      rows: [row(ID)],
      conversations: { [ID]: sync(ID, { turn: null }) },
    });
    expect(engineStrip()).toBeUndefined();
    pushHost(hostWith({ stale: true, tasks: 3 }, [here({ tasks: 1 }), other(2)]));
    await settle();
    expect(engineStrip()).toBeUndefined();

    pushHost(hostWith({ stale: true, engineLost: true, tasks: 3 }, [here({ tasks: 1 }), other(2)]));
    await waitFor(() => engineStrip() !== undefined);
    const three = fill(CHAT.engineRestartStops, { n: 3 });
    expect(engineStrip()!.textContent).toBe(three);
    expect(three.length).toBeLessThan(50);
    expect(engineStrip()!.getAttribute('title')).toBe(
      `${CHAT.engineRestartTip} ${fill(CHAT.engineRestartOthers, { n: 2 })}`,
    );
    expect(engineStrip()!.getAttribute('data-tone')).toBe('warning');

    click(button(CHAT.history.threadMenu));
    await waitFor(() => compactRow() !== null);
    expect(compactRow()!.getAttribute('aria-disabled')).toBe('true');
    expect(noteOf(compactRow()!)).toBe(CHAT.history.engineLost);
    act(() => {
      compactRow()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => compactRow() === null);

    pushHost(hostWith({ stale: true, engineLost: true, tasks: 1 }, [here({ tasks: 1 })]));
    await waitFor(() => engineStrip()?.textContent === CHAT.engineRestartStopsOne);
    expect(engineStrip()!.getAttribute('title')).toBe(CHAT.engineRestartTip);

    pushHost(hostWith({ stale: true, engineLost: true, tasks: 0 }, [here()]));
    await waitFor(() => engineStrip() === undefined);
  });

  it('the delete tip names the command or the subagent; a command leaves the model change on, a subagent does not', async () => {
    await open(`/chat?c=${ID}`, (text) => text.includes('Hi there'), {
      rows: [row(ID)],
      conversations: { [ID]: sync(ID, { turn: null }) },
    });
    pushHost(hostWith({ tasks: 1 }, [here({ tasks: 1 })]));
    await waitFor(() => deleteTrigger().getAttribute('aria-disabled') === 'true');
    expect(await tooltipOf(deleteTrigger())).toBe(CHAT.deleteBusyTask);
    expect(picker().disabled).toBe(false);

    pushHost(hostWith({ tasks: 1, background: 1 }, [here({ tasks: 1, background: true, helpers: true })]));
    await waitFor(() => picker().disabled);
    expect(await tooltipOf(deleteTrigger())).toBe(CHAT.deleteBusyHelper);

    pushHost(hostWith({}, [here()]));
    await waitFor(() => deleteTrigger().getAttribute('aria-disabled') === null);
    expect(picker().disabled).toBe(false);
  });
});

describe('Work: the queue', () => {
  const row = (id: string, text: string, plan = false) => ({
    id,
    text,
    files: [],
    plan,
    queuedAt: NOW,
    editing: false,
  });
  const enter = (): void =>
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
  const statusText = (): string =>
    [...page!.container.querySelectorAll('[role="status"]')].map((n) => n.textContent ?? '').join(' ');
  const queueList = (): HTMLElement | null => byRole(page!.container, 'list', CHAT.queue.label);

  it('Enter while an answer runs queues the message with its Plan first value and empties the box', async () => {
    await open('/chat?c=p-queue', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-queue/prompt': { queued: 'q1' } },
      conversations: { 'p-queue': sync('p-queue') },
    });
    expect(accessibleName(sendFace())).toBe(CHAT.stop);
    type('also update the changelog');
    expect(accessibleName(sendFace())).toBe(CHAT.queue.add);
    expect(button(CHAT.stop, composerForm())).not.toBeNull();
    enter();
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-queue/prompt',
      body: { text: 'also update the changelog', plan: false, whenBusy: 'queue' },
    });
    expect(composer().value).toBe('');
    await waitFor(() => statusText().includes(fill(CHAT.queue.status.queued, { n: 1 })));
    expect(writes().some((w) => w.path.endsWith('/cancel'))).toBe(false);
  });

  it('the list from the stream: rows, Remove, a row opened and left, the held header and Send next, Clear', async () => {
    await open('/chat?c=p-qlist', (text) => text.includes('Hi there'), {
      routes: {
        'DELETE /chat/conversations/p-qlist/queue/q1': {},
        'DELETE /chat/conversations/p-qlist/queue': {},
        'PATCH /chat/conversations/p-qlist/queue/q2': {},
        'POST /chat/conversations/p-qlist/queue/release': {},
      },
      conversations: {
        'p-qlist': sync('p-qlist', {
          queue: queueView([row('q1', 'first row'), row('q2', 'second row', true)]),
        }),
      },
    });
    await waitFor(() => queueList() !== null);
    expect(shown(queueList()!)).toContain('first row');
    expect(shown(composerForm())).toContain(fill(CHAT.queue.title, { n: 2 }));
    expect(shown(composerForm())).toContain(CHAT.queue.plan);

    click(button(CHAT.queue.remove, queueList()!));
    await waitFor(() => writes().some((w) => w.method === 'DELETE'));
    expect(writes().at(-1)).toEqual({
      method: 'DELETE',
      path: '/chat/conversations/p-qlist/queue/q1',
      body: undefined,
    });

    click(button(`${CHAT.queue.edit}: second row`, queueList()!));
    await waitFor(() => writes().some((w) => w.method === 'PATCH'));
    expect(writes().at(-1)).toEqual({
      method: 'PATCH',
      path: '/chat/conversations/p-qlist/queue/q2',
      body: { editing: true },
    });
    const edit = byRole(page!.container, 'textbox', CHAT.queue.editLabel)!;
    act(() => {
      edit.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => writes().filter((w) => w.method === 'PATCH').length === 2);
    expect(writes().at(-1)).toMatchObject({ body: { editing: false } });
    // Esc in the row's edit leaves the edit, never the answer.
    expect(writes().some((w) => w.path.endsWith('/cancel'))).toBe(false);
    // With the edit open, Esc in the message box closes that edit first, and does not stop the answer either.
    click(button(`${CHAT.queue.edit}: second row`, queueList()!));
    await waitFor(() => writes().filter((w) => w.method === 'PATCH').length === 3);
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => writes().filter((w) => w.method === 'PATCH').length === 4);
    expect(writes().at(-1)).toMatchObject({ body: { editing: false } });
    expect(byRole(page!.container, 'textbox', CHAT.queue.editLabel)).toBeNull();
    expect(writes().some((w) => w.path.endsWith('/cancel'))).toBe(false);

    act(() =>
      api!.emitChat('p-qlist', {
        type: 'queue',
        seq: 1,
        queue: queueView([row('q2', 'second row', true)], { why: 'stopped' }),
      }),
    );
    await waitFor(() => shown(composerForm()).includes(CHAT.queue.held.stopped));
    expect(statusText()).toContain(CHAT.queue.status.held);
    click(button(CHAT.queue.sendNext, composerForm()));
    await waitFor(() => writes().some((w) => w.path.endsWith('/queue/release')));
    click(button(CHAT.queue.clear, composerForm()));
    await waitFor(() => writes().some((w) => w.method === 'DELETE' && w.path === '/chat/conversations/p-qlist/queue'));

    act(() => api!.emitChat('p-qlist', { type: 'queue', seq: 2, queue: queueView([]) }));
    await waitFor(() => queueList() === null);
  });

  // Mutation: drop the re-hold on a view where the host gave the row back → the list sends the old text mid-edit.
  it('an open edit the host gave back is held again; a save of a row that went says so and keeps the text', async () => {
    const error = vi.spyOn(toast, 'error');
    const warning = vi.spyOn(toast, 'warning');
    const saved = deferred({});
    const gone = stubError('E_NOT_FOUND', 'That queued message is gone.');
    await open('/chat?c=p-qedit', (text) => text.includes('Hi there'), {
      routes: { 'PATCH /chat/conversations/p-qedit/queue/q1': sequence({}, {}, saved.body, {}, gone) },
      conversations: { 'p-qedit': sync('p-qedit', { queue: queueView([row('q1', 'first row')]) }) },
    });
    const view = (seq: number, text: string, editing: boolean): void =>
      act(() =>
        api!.emitChat('p-qedit', {
          type: 'queue',
          seq,
          queue: queueView([{ ...row('q1', text), editing }]),
        }),
      );
    const settle = (): Promise<void> =>
      act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
      });
    const editBox = (): HTMLTextAreaElement | null =>
      byRole(page!.container, 'textbox', CHAT.queue.editLabel) as HTMLTextAreaElement | null;
    await waitFor(() => queueList() !== null);
    click(button(`${CHAT.queue.edit}: first row`, queueList()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toMatchObject({ method: 'PATCH', body: { editing: true } });
    view(1, 'first row', true);
    await settle();
    expect(writes()).toHaveLength(1);

    // No page watched the conversation for a moment: the host ended the edit, and the open edit holds it again.
    view(2, 'first row', false);
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({
      method: 'PATCH',
      path: '/chat/conversations/p-qedit/queue/q1',
      body: { editing: true },
    });

    // A save ends the edit on the host before its reply: that view is no reason to hold the row again.
    type('first row, edited', editBox()!);
    click(button(CHAT.queue.save, queueList()!));
    await waitFor(() => writes().length === 3);
    view(3, 'first row, edited', false);
    await settle();
    saved.release();
    await waitFor(() => editBox() === null);
    await settle();
    expect(writes()).toHaveLength(3);

    // The row went before the save reached the host: the edit is not lost, it goes to the empty message box.
    click(button(`${CHAT.queue.edit}: first row, edited`, queueList()!));
    await waitFor(() => writes().length === 4);
    type('edited again', editBox()!);
    click(button(CHAT.queue.save, queueList()!));
    await waitFor(() => writes().length === 5);
    expect(writes()[4]).toMatchObject({ method: 'PATCH', body: { text: 'edited again' } });
    await waitFor(() => composer().value === 'edited again');
    expect(warning).toHaveBeenCalledWith(CHAT.queue.editGoneBack);
    expect(error).not.toHaveBeenCalled();
    expect(editBox()).toBeNull();
  });

  // Mutation: skip the failed record when the box holds a newer draft → the refused message is gone with no way back.
  it('a queue refused while a newer draft fills the box keeps the message and its file for Try again, which queues it', async () => {
    const refused = deferred(stubError('E_CONFLICT', '5 messages wait. Remove one to queue more.', 'queue-full'));
    const file = { id: 'f1', kind: 'file' as const, name: 'build.log', bytes: 8, numbered: 8, text: 'line one' };
    writeAttachments('p-qlost', [file]);
    await open('/chat?c=p-qlost', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-qlost/prompt': sequence(refused.body, { queued: 'q9' }) },
      conversations: { 'p-qlost': sync('p-qlost') },
    });
    type('first message');
    enter();
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toMatchObject({
      body: { attachments: [{ kind: 'file', name: 'build.log', text: 'line one' }] },
    });
    expect(composer().value).toBe('');
    expect(shown(composerForm())).not.toContain('build.log');
    type('second draft');
    refused.release();
    await waitFor(() => shown(composerForm()).includes(CHAT.queue.full));
    expect(composer().value).toBe('second draft');
    click(button(CHAT.tryAgain, composerForm()));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-qlost/prompt',
      body: {
        text: 'first message',
        attachments: [{ kind: 'file', name: 'build.log', text: 'line one' }],
        plan: false,
        whenBusy: 'queue',
      },
    });
    await waitFor(() => !shown(composerForm()).includes(CHAT.queue.full));
    expect(composer().value).toBe('second draft');
  });

  // Mutation: count the listed rows on the page → four listed offer Queue, and the host refuses it.
  it('four listed and one whose answer starts: the host counts five, so Queue is off', async () => {
    const four = [2, 3, 4, 5].map((n) => row(`q${n}`, `row ${n}`));
    await open('/chat?c=p-qstart', (text) => text.includes('Hi there'), {
      conversations: { 'p-qstart': sync('p-qstart', { queue: { items: four, held: null, count: 5 } }) },
    });
    type('a sixth');
    await waitFor(() => shown(composerForm()).includes(CHAT.queue.full));
    expect(sendFace().disabled).toBe(true);
    enter();
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(writes()).toEqual([]);
    act(() => api!.emitChat('p-qstart', { type: 'queue', seq: 1, queue: queueView(four) }));
    await waitFor(() => !shown(composerForm()).includes(CHAT.queue.full));
    enter();
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toMatchObject({ path: '/chat/conversations/p-qstart/prompt', body: { whenBusy: 'queue' } });
  });

  it('five queued: the reason says so and Queue is off; idle with held rows, Enter sends at once', async () => {
    const five = [1, 2, 3, 4, 5].map((n) => row(`q${n}`, `row ${n}`));
    await open('/chat?c=p-qfull', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-qfull/prompt': { turnId: 't2' } },
      conversations: { 'p-qfull': sync('p-qfull', { queue: queueView(five) }) },
    });
    type('a sixth');
    await waitFor(() => shown(composerForm()).includes(CHAT.queue.full));
    expect(sendFace().disabled).toBe(true);
    enter();
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(writes()).toEqual([]);

    act(() => api!.emitChat('p-qfull', { type: 'turn', seq: 1, turn: null, turnId: 't1', outcome: 'cancelled' }));
    act(() => api!.emitChat('p-qfull', { type: 'queue', seq: 2, queue: queueView(five, { why: 'stopped' }) }));
    await waitFor(() => accessibleName(sendFace()) === CHAT.send);
    expect(shown(composerForm())).not.toContain(CHAT.queue.full);
    enter();
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-qfull/prompt',
      body: { text: 'a sixth', plan: false },
    });
  });
});

describe('Work: Send now (#3)', () => {
  const row = (id: string, text: string) => ({ id, text, files: [], plan: false, queuedAt: NOW, editing: false });
  const cmdEnter = (): void =>
    act(() => {
      composer().dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Enter', metaKey: true, bubbles: true, cancelable: true }),
      );
    });
  const statusText = (): string =>
    [...page!.container.querySelectorAll('[role="status"]')].map((n) => n.textContent ?? '').join(' ');
  const queueList = (): HTMLElement | null => byRole(page!.container, 'list', CHAT.queue.label);
  const STEER_TS = NOW - 30_000;
  const steer: TranscriptEntry = {
    kind: 'user',
    ts: STEER_TS,
    text: 'use pnpm instead',
    promptIndex: null,
    interjection: true,
  };

  it('a row goes now through its route; ⌘Enter sends the typed text now; while starting Send now is off', async () => {
    await open('/chat?c=p-steer', (text) => text.includes('Hi there'), {
      routes: {
        'POST /chat/conversations/p-steer/queue/q1/send': { steered: true },
        'POST /chat/conversations/p-steer/prompt': { steered: true },
      },
      conversations: {
        'p-steer': sync('p-steer', { queue: queueView([row('q1', 'also the changelog')]) }),
      },
    });
    await waitFor(() => queueList() !== null);
    click(button(CHAT.queue.sendNow, queueList()!));
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-steer/queue/q1/send',
      body: { how: 'now' },
    });
    await waitFor(() => statusText().includes(CHAT.queue.status.sent));

    type('use pnpm instead');
    cmdEnter();
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-steer/prompt',
      body: { text: 'use pnpm instead', plan: false, whenBusy: 'now' },
    });
    expect(composer().value).toBe('');

    act(() =>
      api!.emitChat('p-steer', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't2', phase: 'starting', startedAt: NOW, wait: null },
      }),
    );
    const sendNow = (): HTMLElement => button(CHAT.queue.sendNow, queueList()!);
    await waitFor(() => sendNow().getAttribute('aria-disabled') === 'true');
    act(() => sendNow().focus());
    await waitFor(
      () => byRole(document.body, 'tooltip')?.textContent?.includes(CHAT.queue.sendNowOffStarting) === true,
    );
    click(sendNow());
    // ⌘Enter while the answer starts queues, as Enter does.
    type('later on');
    cmdEnter();
    await waitFor(() => writes().length === 3);
    expect(writes()[2]).toMatchObject({ body: { text: 'later on', whenBusy: 'queue' } });
  });

  // Mutation: toast `steer-not-now` → a row pressed as its answer ended warns, although the list sends it by itself.
  it('a Send now refused while the answer runs warns, refused as it ended is quiet; ⌘Enter text queued says why', async () => {
    const warning = vi.spyOn(toast, 'warning');
    const error = vi.spyOn(toast, 'error');
    await open('/chat?c=p-notnow', (text) => text.includes('Hi there'), {
      routes: {
        'POST /chat/conversations/p-notnow/queue/q1/send': stubError(
          'E_CONFLICT',
          'The answer takes no message at this step.',
          'steer-busy',
        ),
        'POST /chat/conversations/p-notnow/queue/q2/send': stubError(
          'E_CONFLICT',
          'No answer takes it now. The message stays queued.',
          'steer-not-now',
        ),
        'POST /chat/conversations/p-notnow/prompt': sequence(
          { queued: 'q3', steer: 'refused' },
          { queued: 'q4', steer: 'late' },
        ),
      },
      conversations: {
        'p-notnow': sync('p-notnow', {
          queue: queueView([row('q1', 'also the changelog'), row('q2', 'and the readme')]),
        }),
      },
    });
    await waitFor(() => queueList() !== null);
    const sendNowOf = (text: string): HTMLElement => {
      const item = allByRole(queueList()!, 'listitem').find((li) => (li.textContent ?? '').includes(text))!;
      return button(CHAT.queue.sendNow, item);
    };
    click(sendNowOf('also the changelog'));
    await waitFor(() => warning.mock.calls.some((c) => c[0] === CHAT.queue.notThisStep));
    expect(shown(queueList()!)).toContain('also the changelog');
    click(sendNowOf('and the readme'));
    await waitFor(() => writes().length === 2);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(warning).toHaveBeenCalledTimes(1);
    expect(error).not.toHaveBeenCalled();
    type('refused one');
    cmdEnter();
    await waitFor(() => warning.mock.calls.length === 2);
    expect(warning.mock.calls[1]![0]).toBe(CHAT.queue.notThisStep);
    type('slow one');
    cmdEnter();
    await waitFor(() => warning.mock.calls.some((c) => c[0] === CHAT.queue.sendNowLate));
  });

  // Mutation: keep the full list as a `blocked` reason → ⌘Enter is off with five rows waiting.
  it('five rows wait: Queue is off, ⌘Enter still sends now; files or a command keep it off', async () => {
    const five = [1, 2, 3, 4, 5].map((n) => row(`q${n}`, `row ${n}`));
    await open('/chat?c=p-steerfull', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-steerfull/prompt': { steered: true } },
      conversations: { 'p-steerfull': sync('p-steerfull', { queue: queueView(five) }) },
    });
    type('use pnpm instead');
    await waitFor(() => shown(composerForm()).includes(CHAT.queue.full));
    expect(sendFace().disabled).toBe(true);
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    type('/goal ship it');
    cmdEnter();
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(writes()).toEqual([]);
    type('use pnpm instead');
    cmdEnter();
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-steerfull/prompt',
      body: { text: 'use pnpm instead', plan: false, whenBusy: 'now' },
    });
    await waitFor(() => statusText().includes(CHAT.queue.status.sent));
  });

  // Mutation: offer Send now with no answer running → a press as the list drains its row is a 409.
  it('no answer runs: the rows have no Send now; they go by themselves', async () => {
    await open('/chat?c=p-steeridle', (text) => text.includes('Hi there'), {
      conversations: {
        'p-steeridle': sync('p-steeridle', { queue: queueView([row('q1', 'also the changelog')]) }),
      },
    });
    await waitFor(() => queueList() !== null);
    expect(byRole(queueList()!, 'button', CHAT.queue.sendNow)).not.toBeNull();
    act(() => api!.emitChat('p-steeridle', { type: 'turn', seq: 1, turn: null, turnId: 't1', outcome: 'completed' }));
    await waitFor(() => byRole(queueList()!, 'button', CHAT.queue.sendNow) === null);
    expect(shown(queueList()!)).toContain('also the changelog');
  });

  it('the steer bubble: its caption is always shown; after a Stop it says it may not have been read', async () => {
    await open('/chat?c=p-bubble', (text) => text.includes('use pnpm instead'), {
      conversations: {
        'p-bubble': sync('p-bubble', { entries: [user('Fix the build'), answer('Hi there'), steer] }),
      },
    });
    const caption = `${CHAT.queue.steered} · ${formatClock(STEER_TS)}`;
    await waitFor(() => page!.text().includes(caption));
    expect(page!.text()).not.toContain(CHAT.queue.steerUnread);
    act(() => api!.emitChat('p-bubble', { type: 'turn', seq: 1, turn: null, turnId: 't1', outcome: 'cancelled' }));
    await waitFor(() => page!.text().includes(CHAT.queue.steerUnread));
    expect(page!.text()).not.toContain(caption);
  });

  it('Chat: the rows have no Send now and ⌘Enter queues', async () => {
    await open('/chat?c=p-chatq', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-chatq/prompt': { queued: 'q2' } },
      conversations: {
        'p-chatq': sync(
          'p-chatq',
          { queue: queueView([row('q1', 'and the train times')]) },
          {
            mode: 'chat',
            cwd: null,
          },
        ),
      },
    });
    await waitFor(() => queueList() !== null);
    expect(byRole(queueList()!, 'button', CHAT.queue.sendNow)).toBeNull();
    type('and the weather');
    cmdEnter();
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toMatchObject({ body: { text: 'and the weather', whenBusy: 'queue' } });
  });
});

describe('Work: Stop a command or monitor (T1)', () => {
  const LABEL = 'Start the dev server';
  const command: TranscriptEntry = {
    kind: 'tool',
    ts: NOW - 60_000,
    id: 'call-1',
    name: 'run_terminal_command',
    title: 'run_terminal_command',
    status: 'completed',
    input: JSON.stringify({ command: 'npm run dev', description: LABEL, is_background: true }),
    endTs: NOW - 59_000,
    task: { id: 'task-1', kind: 'command', label: LABEL },
  };
  const said = (): string[] => allByRole(page!.container, 'status').map((node) => node.textContent ?? '');
  const stopButton = (): HTMLElement | null =>
    byRole(document.body, 'button', fill(CHAT.activity.stop, { title: LABEL }));
  const jump = (): HTMLElement | null => byRole(document.body, 'button', fill(CHAT.activity.jump, { title: LABEL }));
  const dialog = (): HTMLElement | null => byRole(document.body, 'dialog');
  const stopPath = (id: string): string => `/chat/conversations/${id}/tasks/task-1/stop`;

  async function openStop(id: string, reply: unknown): Promise<void> {
    await open(`/chat?c=${id}`, (text) => text.includes(LABEL), {
      routes: { [`POST ${stopPath(id)}`]: reply },
      conversations: {
        [id]: sync(
          id,
          {
            turn: null,
            entries: [user('Start it', NOW - 61_000), command, answer('Started.'), end()],
            live: {
              helpers: [],
              agents: [],
              tasks: [{ id: 'task-1', kind: 'command', stopping: false, events: 0, recent: [] }],
              schedules: [],
              reports: [],
              stoppedByYou: [],
            },
          },
          { hot: true },
        ),
      },
    });
    await waitFor(() => stopButton() !== null);
  }

  async function confirmStop(): Promise<void> {
    click(stopButton()!);
    await waitFor(() => dialog() !== null);
    expect(writes()).toEqual([]);
    click(button(CHAT.activity.stopConfirm, dialog()!));
  }

  // Mutation: send the stop to the run route → the agent kills nothing.
  it('Stop asks first, then POSTs the task stop route and says Stopped.; the row reads Stopping…, then leaves', async () => {
    await openStop('p-task-stop', {});
    expect(page!.text()).toContain(CHAT.task.inBackground);
    await confirmStop();
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([{ method: 'POST', path: stopPath('p-task-stop'), body: undefined }]);
    await waitFor(() => said().includes(CHAT.status.taskStopped));
    act(() =>
      api!.emitChat('p-task-stop', {
        type: 'activity',
        seq: 1,
        live: {
          helpers: [],
          agents: [],
          tasks: [{ id: 'task-1', kind: 'command', stopping: true, events: 0, recent: [] }],
          schedules: [],
          reports: [],
          stoppedByYou: [],
        },
      }),
    );
    await waitFor(() => (jump()?.textContent ?? '').includes(CHAT.activity.stopping));
    expect(stopButton()).toBeNull();
    const ended = { ...command, task: { ...command.task!, endTs: NOW, killed: true as const } };
    act(() => api!.emitChat('p-task-stop', { type: 'ops', seq: 2, ops: [{ op: 'set', index: 1, entry: ended }] }));
    await waitFor(() => jump() === null);
    expect(page!.text()).toContain(CHAT.task.stopped);
  });

  // Mutation: offer Stop on every stored row → a task the host dropped offers a Stop that can only 404.
  it('Stop shows only while the host tracks the task: it goes when the live facts drop it', async () => {
    await openStop('p-task-live', {});
    act(() =>
      api!.emitChat('p-task-live', {
        type: 'activity',
        seq: 1,
        live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
      }),
    );
    await waitFor(() => stopButton() === null);
    expect(jump()).not.toBeNull();
    act(() =>
      api!.emitChat('p-task-live', {
        type: 'activity',
        seq: 2,
        live: {
          helpers: [],
          agents: [],
          tasks: [{ id: 'task-1', kind: 'command', stopping: false, events: 0, recent: [] }],
          schedules: [],
          reports: [],
          stoppedByYou: [],
        },
      }),
    );
    await waitFor(() => stopButton() !== null);
  });

  it('a task that already ended (404) is an info toast, no error', async () => {
    const info = vi.spyOn(toast, 'info');
    const error = vi.spyOn(toast, 'error');
    await openStop('p-task-gone', stubError('E_NOT_FOUND', 'No such task.'));
    await confirmStop();
    await waitFor(() => info.mock.calls.some((c) => c[0] === CHAT.activity.stopGone));
    expect(error).not.toHaveBeenCalled();
    expect(said()).not.toContain(CHAT.status.taskStopped);
  });

  // Mutation: drop the toast's action → a failed Stop has no way to try again.
  it('a failed Stop is an error toast whose Try again sends the Stop once more', async () => {
    const error = vi.spyOn(toast, 'error');
    await openStop('p-task-fail', stubError('E_UNAVAILABLE', 'The agent could not be reached.'));
    await confirmStop();
    await waitFor(() => error.mock.calls.length === 1);
    const [title, input] = error.mock.calls[0]!;
    expect(title).toBe(fill(CHAT.activity.stopError, { title: LABEL }));
    expect(input?.action?.label).toBe(CHAT.activity.tryAgain);
    act(() => input!.action!.onClick('t'));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({ method: 'POST', path: stopPath('p-task-fail'), body: undefined });
  });
});

describe('Work: Delete a schedule (T3)', () => {
  const PROMPT = 'Check the deploy';
  const schedule: TranscriptEntry = {
    kind: 'tool',
    ts: NOW - 60_000,
    id: 'call-s',
    name: 'scheduler_create',
    title: 'scheduler_create',
    status: 'completed',
    input: JSON.stringify({ interval: '5m', prompt: PROMPT }),
    endTs: NOW - 59_000,
    schedule: { id: 's1', prompt: PROMPT, repeat: 'every 5 minutes', nextTs: NOW + 300_000 },
  };
  const said = (): string[] => allByRole(page!.container, 'status').map((node) => node.textContent ?? '');
  const deleteButton = (): HTMLElement | null =>
    byRole(document.body, 'button', fill(CHAT.activity.deleteSchedule, { title: PROMPT }));
  const jump = (): HTMLElement | null => byRole(document.body, 'button', fill(CHAT.activity.jump, { title: PROMPT }));
  const dialog = (): HTMLElement | null => byRole(document.body, 'dialog');
  const deletePath = (id: string): string => `/chat/conversations/${id}/schedules/s1/delete`;
  const live = (deleting: 'opening' | 'removing' | null) => ({
    helpers: [],
    agents: [],
    tasks: [],
    schedules: [{ id: 's1', deleting, firing: null }],
    reports: [],
    stoppedByYou: [],
  });

  async function openDelete(id: string, reply: unknown, hot = true): Promise<void> {
    await open(`/chat?c=${id}`, (text) => text.includes('Scheduled.'), {
      routes: { [`POST ${deletePath(id)}`]: reply },
      conversations: {
        [id]: sync(
          id,
          {
            turn: null,
            entries: [user('Schedule it', NOW - 61_000), schedule, answer('Scheduled.'), end()],
            live: hot
              ? live(null)
              : { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
          },
          { hot },
        ),
      },
    });
    await waitFor(() => deleteButton() !== null);
  }

  async function confirmDelete(): Promise<void> {
    click(deleteButton()!);
    await waitFor(() => dialog() !== null);
    expect(writes()).toEqual([]);
    click(button(CHAT.activity.deleteConfirm, dialog()!));
  }

  // Mutation: send the delete to the task stop route → the agent deletes nothing.
  it('Delete asks first, then POSTs the schedule delete route and says Schedule deleted.; Removing…, then it leaves', async () => {
    await openDelete('p-sched-del', {});
    expect(jump()!.textContent).toContain(fill(CHAT.activity.next, { time: formatClock(NOW + 300_000) }));
    await confirmDelete();
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([{ method: 'POST', path: deletePath('p-sched-del'), body: undefined }]);
    await waitFor(() => said().includes(CHAT.status.scheduleDeleted));
    act(() => api!.emitChat('p-sched-del', { type: 'activity', seq: 1, live: live('removing') }));
    await waitFor(() => (jump()?.textContent ?? '').includes(CHAT.activity.removing));
    expect(deleteButton()).toBeNull();
    const ended = { ...schedule, schedule: { ...schedule.schedule!, endTs: NOW } };
    act(() => api!.emitChat('p-sched-del', { type: 'ops', seq: 2, ops: [{ op: 'set', index: 1, entry: ended }] }));
    await waitFor(() => jump() === null);
  });

  // Mutation: read the row of a cold conversation as waiting → it shows a next time it will not keep.
  it('a conversation no agent holds: the row reads Paused and still offers Delete; the load reads Opening…', async () => {
    await openDelete('p-sched-cold', {}, false);
    expect(jump()!.textContent).toContain(CHAT.activity.paused);
    await confirmDelete();
    await waitFor(() => writes().length === 1);
    act(() => api!.emitChat('p-sched-cold', { type: 'activity', seq: 1, live: live('opening') }));
    await waitFor(() => (jump()?.textContent ?? '').includes(CHAT.activity.opening));
    expect(deleteButton()).toBeNull();
  });

  it('a schedule that is already gone (404) is an info toast, no error', async () => {
    const info = vi.spyOn(toast, 'info');
    const error = vi.spyOn(toast, 'error');
    await openDelete('p-sched-gone', stubError('E_NOT_FOUND', 'No such schedule.'));
    await confirmDelete();
    await waitFor(() => info.mock.calls.some((c) => c[0] === CHAT.activity.deleteGone));
    expect(error).not.toHaveBeenCalled();
    expect(said()).not.toContain(CHAT.status.scheduleDeleted);
  });

  // Mutation: drop the toast's action → a failed Delete has no way to try again.
  it('a failed Delete is an error toast whose Try again sends the Delete once more', async () => {
    const error = vi.spyOn(toast, 'error');
    await openDelete('p-sched-fail', stubError('E_UNAVAILABLE', 'The agent could not be reached.'));
    await confirmDelete();
    await waitFor(() => error.mock.calls.length === 1);
    const [title, input] = error.mock.calls[0]!;
    expect(title).toBe(CHAT.activity.deleteError);
    expect(input?.description).toBeUndefined();
    expect(input?.action?.label).toBe(CHAT.activity.tryAgain);
    act(() => input!.action!.onClick('t'));
    await waitFor(() => writes().length === 2);
    expect(writes()[1]).toEqual({ method: 'POST', path: deletePath('p-sched-fail'), body: undefined });
  });

  // Mutation: read `load-slow` as any failure → the toast does not say the agent was slow to open the conversation.
  it('a Delete the agent could not open the conversation for in time says so, with Try again', async () => {
    const error = vi.spyOn(toast, 'error');
    await openDelete('p-sched-slow', stubError('E_UNAVAILABLE', 'slow', 'load-slow'));
    await confirmDelete();
    await waitFor(() => error.mock.calls.length === 1);
    const [title, input] = error.mock.calls[0]!;
    expect(title).toBe(CHAT.history.loadSlow);
    expect(input?.action?.label).toBe(CHAT.activity.tryAgain);
  });

  // Mutation: leave the engine-lost reason off the rail → a Delete loads the agent and stops the user's commands.
  it('the engine restarted under live commands: Delete is off with the reason, and a press sends nothing', async () => {
    await openDelete('p-sched-lost', {});
    act(() =>
      api!.emitChatHost({
        type: 'state',
        state: {
          ...READY_CHAT_HOST,
          modes: {
            ...READY_CHAT_HOST.modes,
            work: { ...READY_CHAT_HOST.modes.work, state: 'ready', stale: true, engineLost: true, tasks: 1 },
          },
        },
      }),
    );
    await waitFor(() => deleteButton()?.getAttribute('aria-disabled') === 'true');
    const reason = document.getElementById(deleteButton()!.getAttribute('aria-describedby') ?? '');
    expect(reason?.textContent).toBe(CHAT.history.engineLost);
    click(deleteButton()!);
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 20)));
    expect(dialog()).toBeNull();
    expect(writes()).toEqual([]);
  });
});

describe('Work: Rewind to here (#7)', () => {
  const H = CHAT.history;
  const PLAN = {
    token: 'v1:6',
    removes: { messages: 1, steps: 3 },
    background: { commands: 1, schedules: 2, runs: 0 },
    helpers: false,
    attachments: false,
    files: [],
    filesMore: 0,
  };
  const THREAD: TranscriptEntry[] = [
    user('Fix the build'),
    answer('Done'),
    end(),
    user('Run the dev server'),
    answer('Started'),
    end(),
  ];
  const dialog = (): HTMLElement | null => byRole(document.body, 'dialog', H.rewindTitle);

  // Mutation: draw the rule with no `data-rewind-mode` → the line check fails.
  it('a plan with no files: the dialog says the agent changed none and what the rewind ends; the rule after', async () => {
    await open('/chat?c=w-rw', (text) => text.includes('Started'), {
      routes: {
        'POST /chat/conversations/w-rw/rewind/plan': PLAN,
        'POST /chat/conversations/w-rw/rewind': {},
      },
      conversations: { 'w-rw': sync('w-rw', { entries: THREAD, turn: null }) },
    });
    const menus = [...page!.container.querySelectorAll<HTMLElement>('[data-message-menu]')];
    expect(menus).toHaveLength(2);
    click(menus[1]!);
    await waitFor(() => byRole(document.body, 'menuitem', H.rewind) !== null);
    click(byRole(document.body, 'menuitem', H.rewind)!);
    await waitFor(() => dialog()?.textContent?.includes(H.filesNone) === true);
    expect(dialog()!.querySelector('[role="radiogroup"]')).toBeNull();
    const lines = [...dialog()!.querySelectorAll('li')].map((li) => li.textContent);
    expect(lines).toEqual([H.backgroundOne, fill(H.schedules, { n: 2 })]);
    click(button(H.confirm, dialog()!));
    await waitFor(() => writes().filter((w) => w.path === '/chat/conversations/w-rw/rewind').length === 1);
    act(() => api!.emitChat('w-rw', sync('w-rw', { seq: 1, entries: THREAD.slice(0, 3), turn: null })));
    act(() =>
      api!.emitChat('w-rw', {
        type: 'permission-mark',
        seq: 2,
        mark: { kind: 'rewind', at: NOW + 1, mode: 'conversation', files: 0, from: NOW - 1 },
      }),
    );
    act(() =>
      api!.emitChat('w-rw', {
        type: 'rewind',
        seq: 3,
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: 3 },
      }),
    );
    await waitFor(() => dialog() === null);
    await waitFor(() => page!.container.querySelector('[data-thread-mark="rewind"]') !== null);
    const rule = page!.container.querySelector('[data-thread-mark="rewind"]')!;
    expect(rule.getAttribute('data-rewind-mode')).toBe('conversation');
    expect(rule.textContent).toContain(H.rule);
    expect(composer().value).toBe('Run the dev server');
  });

  // Mutation: post `conversation` whatever the choice → the body check fails.
  it('files: the dry run lists them; Files only sends `files`, keeps the thread and the box, and the rule says so', async () => {
    const files = [
      { path: 'src/upload.ts', state: 'clean' },
      { path: 'src/upload.test.ts', state: 'clean' },
    ];
    await open('/chat?c=w-rf', (text) => text.includes('Started'), {
      routes: {
        'POST /chat/conversations/w-rf/rewind/plan': { ...PLAN, files },
        'POST /chat/conversations/w-rf/rewind': {},
      },
      conversations: { 'w-rf': sync('w-rf', { entries: THREAD, turn: null }) },
    });
    const success = vi.spyOn(toast, 'success');
    click([...page!.container.querySelectorAll<HTMLElement>('[data-message-menu]')][1]!);
    await waitFor(() => byRole(document.body, 'menuitem', H.rewind) !== null);
    click(byRole(document.body, 'menuitem', H.rewind)!);
    await waitFor(() => dialog()?.querySelector('[role="radiogroup"]') != null);
    const radios = [...dialog()!.querySelectorAll<HTMLElement>('[role="radio"]')];
    expect(radios.map((r) => r.getAttribute('aria-checked'))).toEqual(['true', 'false', 'false']);
    await waitFor(() => document.activeElement === radios[0]);
    expect(dialog()!.querySelectorAll('li[data-file-state]')).toHaveLength(2);
    click(radios[2]!);
    await waitFor(() => (dialog()?.textContent ?? '').includes(fill(H.filesOnlyLead, { n: 2 })));
    await waitFor(() => {
      try {
        return button(H.confirmFiles, dialog()!) !== null;
      } catch {
        return false;
      }
    });
    click(button(H.confirmFiles, dialog()!));
    await waitFor(() => writes().filter((w) => w.path === '/chat/conversations/w-rf/rewind').length === 1);
    expect(writes().find((w) => w.path === '/chat/conversations/w-rf/rewind')?.body).toEqual({
      token: PLAN.token,
      mode: 'files',
    });
    act(() =>
      api!.emitChat('w-rf', {
        type: 'permission-mark',
        seq: 1,
        mark: { kind: 'rewind', at: NOW + 1, mode: 'files', files: 2 },
      }),
    );
    act(() =>
      api!.emitChat('w-rf', {
        type: 'rewind',
        seq: 2,
        n: 1,
        end: { outcome: 'done', mode: 'files', restored: 2, entry: 3 },
      }),
    );
    await waitFor(() => dialog() === null);
    expect(shown()).toContain('Started');
    expect(composer().value).toBe('');
    expect(success).toHaveBeenLastCalledWith(fill(H.doneFilesOnly, { files: fill(H.nFiles, { n: 2 }) }));
    await waitFor(() => page!.container.querySelector('[data-thread-mark="rewind"]') !== null);
    const rule = page!.container.querySelector('[data-thread-mark="rewind"]')!;
    expect(rule.getAttribute('data-rewind-mode')).toBe('files');
    expect(rule.textContent).toContain(fill(H.ruleFilesOnly, { files: fill(H.nFiles, { n: 2 }) }));
    success.mockRestore();
  });
});

describe('Work: the goal (#17)', () => {
  const OBJECTIVE = 'Fix the flaky cache tests';
  const THREAD: TranscriptEntry[] = [
    user('Fix the build'),
    answer('Done'),
    end(),
    user(`/goal ${OBJECTIVE}`),
    {
      kind: 'goal',
      ts: NOW - 50_000,
      goalId: 'g1',
      objective: OBJECTIVE,
      status: 'user_paused',
      phase: 'idle',
      tokensUsed: 31_400,
      elapsedMs: 724_000,
      asOf: NOW - 40_000,
    },
    answer('Paused at the second test'),
    end('cancelled'),
  ];
  const noteOf = (row: HTMLElement): string =>
    document.getElementById(row.getAttribute('aria-describedby') ?? '')?.textContent ?? '';
  const strip = (): HTMLElement => byRole(page!.container, 'group', CHAT.goal.label)!;
  const goalWrites = (): unknown[] => writes().filter((w) => w.path.endsWith('/goal'));

  // Mutation: leave the goal rule out of `/rewind`'s reason → the slot is empty, Send is on and Enter does nothing.
  it('paused: Rewind before its row is off in the menus and for /rewind; Resume waits for a running answer', async () => {
    await open('/chat?c=w-goal', (text) => text.includes('Paused at the second test'), {
      routes: { 'POST /chat/conversations/w-goal/goal': { turnId: 't2' } },
      conversations: { 'w-goal': sync('w-goal', { entries: THREAD, turn: null }, { hot: true }) },
    });
    expect(strip().textContent).toContain(CHAT.goal.state.paused);
    const menus = [...page!.container.querySelectorAll<HTMLElement>('[data-message-menu]')];
    expect(menus).toHaveLength(2);
    for (const menu of menus) {
      click(menu);
      await waitFor(() => byRole(document.body, 'menuitem', CHAT.history.rewind) !== null);
      const row = byRole(document.body, 'menuitem', CHAT.history.rewind)!;
      expect(row.getAttribute('aria-disabled')).toBe('true');
      expect(noteOf(row)).toBe(CHAT.history.offGoal);
      act(() => {
        row.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      });
      await waitFor(() => byRole(document.body, 'menuitem', CHAT.history.rewind) === null);
    }
    act(() => composer().focus());
    type('/rewind');
    expect(shown()).toContain(CHAT.history.offGoal);
    expect(sendFace().disabled).toBe(true);
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 20)));
    expect(writes().filter((w) => w.path.includes('/rewind') || w.path.endsWith('/prompt'))).toEqual([]);
    type('');

    act(() =>
      api!.emitChat('w-goal', {
        type: 'turn',
        seq: 1,
        turn: { turnId: 't1', phase: 'running', startedAt: NOW, wait: null },
      }),
    );
    await waitFor(() => button(CHAT.goal.resume, strip()).getAttribute('aria-disabled') === 'true');
    expect(noteOf(button(CHAT.goal.resume, strip()))).toBe(CHAT.modelBusy);
    click(button(CHAT.goal.resume, strip()));
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 20)));
    expect(goalWrites()).toEqual([]);
    act(() => api!.emitChat('w-goal', { type: 'turn', seq: 2, turn: null, turnId: 't1', outcome: 'completed' }));
    await waitFor(() => button(CHAT.goal.resume, strip()).getAttribute('aria-disabled') === null);
    click(button(CHAT.goal.resume, strip()));
    await waitFor(() => goalWrites().length === 1);
    expect(goalWrites()).toEqual([
      { method: 'POST', path: '/chat/conversations/w-goal/goal', body: { action: 'resume' } },
    ]);
  });

  // Mutation: drop the goal term of `steerShown` → the row waiting between two rounds has no Send now.
  it('running between two rounds: only Pause; the queue waits for the goal and its row can go now', async () => {
    const entries = THREAD.slice(0, 4).concat(
      { ...THREAD[4]!, status: 'active', phase: 'executing' } as TranscriptEntry,
      answer('Round one'),
      end(),
    );
    const queued = { id: 'q1', text: 'use pnpm', files: [], queuedAt: NOW, editing: false };
    await open('/chat?c=w-goal2', (text) => text.includes('Round one'), {
      routes: { 'POST /chat/conversations/w-goal2/queue/q1/send': { steered: true } },
      conversations: {
        'w-goal2': sync('w-goal2', { entries, turn: null, queue: queueView([queued]) }, { hot: true }),
      },
    });
    expect([...strip().querySelectorAll('button')].map((b) => b.textContent)).toEqual([CHAT.goal.pause]);
    const list = byRole(page!.container, 'list', CHAT.queue.label)!;
    expect(shown(composerForm())).toContain(CHAT.queue.held.goal);
    const sendNow = button(CHAT.queue.sendNow, list);
    expect(sendNow.getAttribute('aria-disabled')).toBeNull();
    click(sendNow);
    await waitFor(() => writes().length === 1);
    expect(writes()).toEqual([
      { method: 'POST', path: '/chat/conversations/w-goal2/queue/q1/send', body: { how: 'now' } },
    ]);
  });
});

describe('Work: the / command list (#6)', () => {
  const listNames = (): string[] =>
    [...document.querySelectorAll('[role="option"]')].map((o) => o.querySelector('span')?.textContent ?? '');
  const keyDown = (init: KeyboardEventInit): void =>
    act(() => {
      composer().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init }));
    });
  const idle = { turn: null, turns: [], entries: [user('Fix the build'), answer('Hi there'), end()] };

  it('before the session loads: the app commands and the four built-ins, and more after the first message', async () => {
    await open('/chat?c=p-cmd', (text) => text.includes('Hi there'), {
      conversations: { 'p-cmd': sync('p-cmd', idle) },
    });
    act(() => composer().focus());
    type('/');
    await waitFor(() => listNames().length > 0);
    expect(listNames()).toEqual([
      '/compact',
      '/context',
      '/fork',
      '/plan',
      '/recap',
      '/rewind',
      '/goal',
      '/loop',
      '/deep-research',
      '/workflow',
    ]);
    expect(shown()).toContain(CHAT.slash.later);
    // A skill typed in full may go: the host checks it once the session is loaded.
    type('/commit fix');
    expect(sendFace().disabled).toBe(false);

    act(() =>
      api!.emitChat('p-cmd', {
        type: 'commands',
        seq: 1,
        commands: [
          {
            name: 'goal',
            kind: 'builtin',
            description: 'Set, manage, or check an autonomous goal',
            hint: '<objective>',
          },
          { name: 'review-pr', kind: 'skill', description: 'Review a pull request', hint: '<number>' },
          { name: 'release', kind: 'workflow', description: 'Workflow: cut a release', hint: null },
        ],
      }),
    );
    type('/');
    await waitFor(() => listNames().includes('/review-pr'));
    expect(listNames()).toEqual([
      '/compact',
      '/context',
      '/fork',
      '/plan',
      '/recap',
      '/rewind',
      '/goal',
      '/review-pr',
      '/release',
    ]);
    expect(shown()).not.toContain(CHAT.slash.later);
    // Built-ins keep our words; skills show the agent's, as plain text.
    expect(shown()).toContain(CHAT.slash.about.goal);
    expect(shown()).toContain('Review a pull request');
    // Not in the list now: unknown.
    type('/commit fix');
    expect(sendFace().disabled).toBe(true);
    expect(shown()).toContain(CHAT.slash.unknown);
    // Mutation: pass every allowed built-in → `/loop` goes although this session's agent does not offer it.
    type('/loop 5m check the build');
    expect(sendFace().disabled).toBe(true);
    expect(shown()).toContain(CHAT.slash.unknown);
    type('/goal ship it');
    expect(sendFace().disabled).toBe(false);
  });

  it('a filled agent command shows its words, then goes as typed through the prompt route', async () => {
    await open('/chat?c=p-goal', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/p-goal/prompt': { turnId: 't2' } },
      conversations: { 'p-goal': sync('p-goal', idle) },
    });
    act(() => composer().focus());
    type('/go');
    await waitFor(() => listNames().length === 1);
    keyDown({ key: 'Enter' });
    expect(composer().value).toBe('/goal ');
    expect(shown()).toContain(CHAT.slash.hint.goal);
    expect(writes()).toEqual([]);
    type('/goal ship the cache fix');
    keyDown({ key: 'Enter' });
    await waitFor(() => writes().length === 1);
    expect(writes()[0]).toEqual({
      method: 'POST',
      path: '/chat/conversations/p-goal/prompt',
      body: { text: '/goal ship the cache fix', plan: false },
    });
  });
});
