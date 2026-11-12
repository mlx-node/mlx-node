/** @vitest-environment happy-dom */

/**
 * The permission mode on the page: the chip and its menu, the two confirms and their arm, stale facts, the disabled
 * reasons, the header chip, the list marks, the palette, the cards by mode and the marks in the thread. The page shows
 * what the stream says; it never moves the mode up without a trusted press made after the arm time.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, type NavigateFunction, Route, Routes, useNavigate } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type {
  ChatHostState,
  ConversationInfo,
  ConversationRow,
  PermissionInteraction,
  PermissionMark,
  PermissionMode,
} from '../ui/src/lib/chat-types.js';
import { allowedSummary, autoAskText, markLine, placeMarks } from '../ui/src/lib/permission-text.js';
import type { ProvidersResponse } from '../ui/src/lib/providers-types.js';
import { toast } from '../ui/src/lib/toast.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';
import Chat from '../ui/src/pages/chat.js';
import { forgetPlace, NEW_PLACE, writeDraft, chooseFolder } from '../ui/src/pages/chat/drafts.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import {
  keySentence,
  permissionModeBlocked,
  permissionModeName,
  permissionModeShown,
  workKeyProviders,
} from '../ui/src/pages/chat/work/permission-mode.js';
import { CARD_ARM_MS, cardButtons, cardReason } from '../ui/src/pages/chat/work/permission.js';
import { type AppCommand, usePageCommands } from '../ui/src/shell/commands.js';
import { answer, end, user } from './helpers/chat-thread-fixtures.js';
import {
  accessibleName,
  byRole,
  type ChatSync,
  deferred,
  READY_CHAT_HOST,
  renderPage,
  type RenderedPage,
  sequence,
  stubApi,
  stubError,
  type StubbedApi,
  waitFor,
} from './render.js';

const NOW = Date.now();
const FOLDER = '/tmp/work/proj';
const T0 = 1_794_411_000_000;
const P = CHAT.permission;
const RANK: Record<PermissionMode, number> = { ask: 0, auto: 1, bypass: 2 };

let page: RenderedPage | undefined;
let api: StubbedApi | undefined;
let calls: ApiCall[] = [];
let commands: readonly AppCommand[] = [];
let navigate: NavigateFunction | undefined;
const used: string[] = [];

function CommandSpy(): null {
  commands = usePageCommands();
  return null;
}

function NavSpy(): null {
  navigate = useNavigate();
  return null;
}

function app(initial: string): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: [initial] },
    createElement(CommandSpy),
    createElement(NavSpy),
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

function sync(id: string, over: Partial<ChatSync> = {}, conversation: Partial<ConversationInfo> = {}): ChatSync {
  return {
    type: 'sync',
    seq: 0,
    conversation: info(id, conversation),
    entries: [user('Fix the build'), answer('Hi there'), end()],
    turns: [],
    turn: null,
    interactions: [],
    notices: [],
    transcriptError: null,
    permissionMarks: [],
    ...over,
  };
}

function permission(over: Partial<PermissionInteraction> = {}): PermissionInteraction {
  return {
    id: 'i1',
    kind: 'permission',
    toolCallId: 'c1',
    toolName: 'run_terminal_command',
    access: 'bash',
    title: 'Run it',
    input: '{"command":"curl https://example.com"}',
    command: 'curl https://example.com',
    path: null,
    resolvedPath: null,
    options: [
      { optionId: 'always-allow', kind: 'allow_always', name: 'Always allow curl' },
      { optionId: 'allow-once', kind: 'allow_once', name: 'Yes' },
      { optionId: 'reject-once', kind: 'reject_once', name: 'No' },
    ],
    flags: [],
    ...over,
  };
}

function runEntry(id: string, command: string, ts: number, output = 'ok'): TranscriptEntry {
  return {
    kind: 'tool',
    ts,
    id,
    name: 'run_terminal_command',
    title: command,
    status: 'completed',
    input: JSON.stringify({ command }),
    output,
    endTs: ts + 50,
  };
}

function raisedHost(id: string, mode: Exclude<PermissionMode, 'ask'>, base: ChatHostState = READY_CHAT_HOST) {
  return { ...base, permissionRaised: [{ id, mode }] };
}

const FAILED_AGENT: ChatHostState = {
  ...READY_CHAT_HOST,
  modes: { ...READY_CHAT_HOST.modes, work: { ...READY_CHAT_HOST.modes.work, state: 'failed' } },
};

interface Open {
  routes?: Record<string, unknown>;
  conversations?: Record<string, ChatSync>;
  rows?: ConversationRow[];
  host?: ChatHostState;
}

async function open(url: string, until: (text: string) => boolean, setup: Open = {}): Promise<RenderedPage> {
  calls = [];
  const host = setup.host ?? READY_CHAT_HOST;
  api = stubApi(
    {
      '/chat/conversations': { conversations: setup.rows ?? [] },
      ...setup.routes,
    },
    { onCall: (call) => calls.push(call), chat: { host, conversations: setup.conversations ?? {} } },
  );
  page = await renderPage(app(url), until);
  const want = JSON.stringify(host);
  await waitFor(() => JSON.stringify(chatHostSnapshot().state) === want, 2_000);
  return page;
}

/** Open a Work conversation in a mode, with the host saying the same. */
async function openIn(id: string, mode: PermissionMode, setup: Open & { sync?: Partial<ChatSync> } = {}) {
  const host = setup.host ?? (mode === 'ask' ? READY_CHAT_HOST : raisedHost(id, mode));
  return open(`/chat?c=${id}`, (text) => text.includes('Hi there') || text.includes(CHAT.permRun), {
    ...setup,
    host,
    routes: { [`POST /chat/conversations/${id}/permission-mode`]: { mode }, ...setup.routes },
    conversations: { [id]: sync(id, setup.sync ?? {}, { permissionMode: mode }), ...setup.conversations },
  });
}

function pathOf(call: ApiCall): string {
  const bare = call.path.split('?')[0]!;
  return bare.startsWith('/api') ? bare.slice('/api'.length) : bare;
}

/** Every mode call the page made. */
function switches(): unknown[] {
  return calls
    .filter((call) => call.method === 'POST' && pathOf(call).endsWith('/permission-mode'))
    .map((call) => call.body);
}

function shown(scope: Element = document.body): string {
  const copy = scope.cloneNode(true) as Element;
  for (const hidden of copy.querySelectorAll('[inert]')) hidden.remove();
  return copy.textContent ?? '';
}

function chip(place: 'composer' | 'header' = 'composer'): HTMLButtonElement {
  const found = page!.container.querySelector<HTMLButtonElement>(`[data-permission-chip="${place}"]`);
  if (found === null) throw new Error(`no ${place} chip`);
  return found;
}

function menu(): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[role="dialog"][aria-label="${P.name}"]`);
}

function confirmView(): HTMLElement | null {
  return document.querySelector<HTMLElement>('[data-permission-confirm]');
}

function radio(mode: PermissionMode): HTMLElement {
  const rows = [...menu()!.querySelectorAll<HTMLElement>('[role="radio"]')];
  return rows[RANK[mode]]!;
}

function buttonIn(scope: ParentNode, label: string): HTMLButtonElement {
  const found = [...scope.querySelectorAll('button')].find((b) => accessibleName(b) === label);
  if (found === undefined) throw new Error(`no button ${label}`);
  return found;
}

function iconOf(element: Element): string {
  const names = (element.querySelector('svg')?.getAttribute('class') ?? '').split(' ');
  return names.find((name) => name.startsWith('lucide-')) ?? '';
}

function click(element: HTMLElement): void {
  act(() => element.click());
}

let offset = 0;
/** Moves the page's clock: the arm compares the event time with the time the confirm text was shown. */
function useClock(): void {
  offset = 0;
  const real = performance.now.bind(performance);
  vi.spyOn(performance, 'now').mockImplementation(() => real() + offset);
}

function advance(ms: number): void {
  offset += ms;
}

/** A click on the action, trusted or not, at the clock's time. */
function press(element: HTMLElement, trusted: boolean): void {
  const event = new MouseEvent('click', { bubbles: true, cancelable: true });
  if (trusted) Object.defineProperty(event, 'isTrusted', { value: true });
  act(() => {
    element.dispatchEvent(event);
  });
}

function key(target: EventTarget, name: string): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true }));
  });
}

async function openMenu(place: 'composer' | 'header' = 'composer'): Promise<HTMLElement> {
  click(chip(place));
  await waitFor(() => menu() !== null);
  return menu()!;
}

async function openConfirm(target: 'auto' | 'bypass'): Promise<HTMLElement> {
  if (menu() === null) await openMenu();
  click(radio(target));
  await waitFor(() => confirmView()?.getAttribute('data-permission-confirm') === target);
  return confirmView()!;
}

function action(target: 'auto' | 'bypass'): HTMLButtonElement {
  return buttonIn(confirmView()!, P[target].confirm);
}

const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 20));

function permissionCommands(): string[] {
  return commands.filter((c) => c.id.startsWith('chat:permission-')).map((c) => c.label);
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  page?.unmount();
  api?.();
  page = undefined;
  api = undefined;
  navigate = undefined;
  writeDraft(NEW_PLACE, '');
  for (const id of used.splice(0)) forgetPlace(id);
  chooseFolder(null);
  vi.restoreAllMocks();
});

describe('rules of the page', () => {
  it('the host list wins over the conversation field; Chat is always Ask', () => {
    expect(permissionModeShown(null, info('a', { permissionMode: 'auto' }))).toBe('auto');
    expect(permissionModeShown([], info('a', { permissionMode: 'bypass' }))).toBe('ask');
    expect(permissionModeShown([{ id: 'a', mode: 'bypass' }], info('a'))).toBe('bypass');
    expect(permissionModeShown([{ id: 'b', mode: 'bypass' }], info('a'))).toBe('ask');
    expect(permissionModeShown([{ id: 'a', mode: 'auto' }], info('a', { mode: 'chat' }))).toBe('ask');
    expect(permissionModeShown([{ id: 'a', mode: 'auto' }], null)).toBe('ask');
  });

  it('the block reasons, first match wins; an open card blocks Auto only', () => {
    const all = { connected: false, isNew: true, cardOpen: true, blocker: 'Agent', cwdProblem: 'missing' } as const;
    const down = CHAT.blockers.reconnecting;
    expect(permissionModeBlocked(all)).toEqual({ chip: down, auto: down, bypass: down });
    const fresh = P.newScreen;
    expect(permissionModeBlocked({ ...all, connected: true })).toEqual({ chip: fresh, auto: fresh, bypass: fresh });
    const up = { ...all, connected: true, isNew: false };
    expect(permissionModeBlocked(up)).toEqual({ chip: null, auto: P.cardOpen, bypass: 'Agent' });
    expect(permissionModeBlocked({ ...up, cardOpen: false, blocker: undefined })).toEqual({
      chip: null,
      auto: CHAT.cwdProblems.missing,
      bypass: CHAT.cwdProblems.missing,
    });
    expect(
      permissionModeBlocked({
        connected: true,
        isNew: false,
        cardOpen: false,
        blocker: undefined,
        cwdProblem: undefined,
      }),
    ).toEqual({ chip: null, auto: null, bypass: null });
  });

  it('names the keys in the Work environment, worded for each confirm', () => {
    const host: ChatHostState['providers'] = { ...READY_CHAT_HOST.providers, search: 'exa' };
    expect(workKeyProviders(host, null)).toEqual(['exa']);
    const list = (ready: boolean): ProvidersResponse =>
      ({
        search: 'brave',
        firecrawl: true,
        fetch: 'firecrawl',
        providers: [{ id: 'firecrawl', work: { ready, missing: ready ? null : 'firecrawl-cli' } }],
      }) as unknown as ProvidersResponse;
    expect(workKeyProviders(host, list(true))).toEqual(['brave', 'firecrawl']);
    expect(workKeyProviders(host, list(false))).toEqual(['brave']);
    expect(keySentence([], 'auto')).toBeNull();
    expect(keySentence([], 'bypass')).toBeNull();
    expect(keySentence(['exa'], 'auto')).toBe(fill(P.auto.key, { provider: 'Exa' }));
    expect(keySentence(['exa'], 'bypass')).toBe(fill(P.bypass.key, { provider: 'Exa' }));
    expect(keySentence(['brave', 'firecrawl'], 'bypass')).toBe(
      fill(P.bypass.keys, { first: 'Brave', second: 'Firecrawl' }),
    );
  });

  it('the accessible name of each mode', () => {
    expect(permissionModeName('ask', FOLDER)).toBe('Permission mode: Ask. Each command and edit asks first.');
    expect(permissionModeName('auto', FOLDER)).toBe(
      'Permission mode: Auto. Plain commands and edits in proj run. The rest asks.',
    );
    expect(permissionModeName('bypass', FOLDER)).toBe(
      'Permission mode: Bypass. Commands, edits and tools run without asking. A plan, or a request that offers Deny but not "Allow once", still waits for you.',
    );
  });

  it('a card shows Deny and Allow once outside Ask, every offered option in Ask, and the reason of the shown mode', () => {
    const card = permission({ autoAsk: 'not-known' });
    expect(cardButtons(card, 'auto').map((b) => b.label)).toEqual([CHAT.deny, CHAT.allowOnce]);
    expect(cardButtons(card, 'bypass').map((b) => b.label)).toEqual([CHAT.deny, CHAT.allowOnce]);
    expect(cardButtons(card, 'ask').map((b) => b.label)).toEqual([CHAT.deny, CHAT.allowOnce, CHAT.allowAlwaysCommand]);
    expect(cardReason(card, 'auto', FOLDER)).toBe(autoAskText('not-known', FOLDER));
    expect(cardReason(card, 'ask', FOLDER)).toBeNull();
    expect(cardReason(card, 'bypass', FOLDER)).toBeNull();
    const noOnce = permission({
      options: [
        { optionId: 'always-allow', kind: 'allow_always', name: 'Always' },
        { optionId: 'reject-once', kind: 'reject_once', name: 'No' },
      ],
    });
    const noOnceLine = (mode: string): string => fill(P.noAllowOnce, { mode });
    expect(cardReason(noOnce, 'bypass', FOLDER)).toBe(`${noOnceLine('Bypass')} ${P.noAllowOnceAsk}`);
    expect(cardReason({ ...noOnce, autoAsk: 'not-covered' }, 'auto', FOLDER)).toBe(
      `${noOnceLine('Auto')} ${P.noAllowOnceAsk}`,
    );
    expect(cardReason({ ...noOnce, autoAsk: undefined }, 'auto', FOLDER)).toBe(
      `${noOnceLine('Auto')} ${P.noAllowOnceAsk}`,
    );
    expect(cardReason(noOnce, 'ask', FOLDER)).toBeNull();
    const denyOnly = permission({ options: [{ optionId: 'reject-once', kind: 'reject_once', name: 'No' }] });
    expect(cardReason(denyOnly, 'bypass', FOLDER)).toBe(noOnceLine('Bypass'));
    expect(cardReason({ ...denyOnly, autoAsk: 'not-covered' }, 'auto', FOLDER)).toBe(noOnceLine('Auto'));
    expect(cardReason({ ...card, autoAsk: undefined }, 'auto', FOLDER)).toBeNull();
    expect(autoAskText('unknown-target', FOLDER)).toBe(
      P.asksStart + fill(P.asks['unknown-target'], { folder: 'proj' }),
    );
    expect(autoAskText('not-covered', null)).toBe(
      'Auto is on. This still asks: Auto covers commands and file edits only.',
    );
  });

  it('the mode lines by from, mode and by', () => {
    const line = (mark: Partial<Extract<PermissionMark, { kind: 'mode' }>>) =>
      markLine({ kind: 'mode', at: 1, mode: 'ask', from: 'ask', by: 'user', ...mark });
    expect(line({ mode: 'auto', from: 'ask' })).toEqual({ label: 'Mode: Auto', text: P.lineAuto, tone: 'neutral' });
    expect(line({ mode: 'auto', from: 'ask', editsUnasked: true, userHooks: true }).text).toBe(
      `${P.lineAuto} ${P.lineEditsUnasked} ${P.lineHooks}`,
    );
    expect(line({ mode: 'auto', from: 'bypass' }).text).toBe(P.lineAutoDown);
    expect(line({ mode: 'bypass', from: 'ask' })).toEqual({
      label: 'Mode: Bypass',
      text: P.bypass.line,
      tone: 'neutral',
    });
    expect(line({ mode: 'bypass', from: 'auto', userHooks: true }).text).toBe(`${P.bypass.line} ${P.bypass.lineHooks}`);
    expect(line({ mode: 'ask', from: 'bypass' })).toEqual({ label: 'Mode: Ask', text: P.askUser, tone: 'neutral' });
    expect(line({ mode: 'ask', from: 'bypass', by: 'restart' }).text).toBe(
      'Bypass ended when the Control Panel restarted. Each command and edit asks again.',
    );
    expect(line({ mode: 'ask', from: 'auto', by: 'restart' }).text).toBe(
      'Auto ended when the Control Panel restarted. Each command and edit asks again.',
    );
    expect(line({ mode: 'ask', from: 'auto', by: 'record-failed' }).text).toBe(
      'The app could not write its record, so Auto ended. Each command and edit asks again.',
    );
    expect(markLine({ kind: 'settings', at: 1 })).toEqual({
      label: P.settingsLabel,
      text: P.settingsText,
      tone: 'warning',
    });
  });

  it('an allowed call in one line: the first line of its command or path, else what kind of call', () => {
    const mark = (access: 'bash' | 'edit' | 'mcp' | 'other', card: PermissionInteraction | null) =>
      allowedSummary({ kind: 'allowed', mode: 'bypass', at: 1, toolCallId: 'x', access, card });
    expect(mark('bash', permission({ command: 'ls\n-la' }))).toBe('ls');
    expect(mark('edit', permission({ access: 'edit', command: null, path: '~/notes/out.txt' }))).toBe(
      '~/notes/out.txt',
    );
    expect(mark('bash', null)).toBe(P.allowedCommand);
    expect(mark('edit', null)).toBe(P.allowedEdit);
    expect(mark('mcp', null)).toBe(P.allowedTool);
    expect(mark('other', permission({ access: 'other', command: null, path: null }))).toBe(P.allowedTool);
  });

  it('places a line before the first later entry, after the last one, and an allowed call on its row', () => {
    const entries = [user('a', 10), runEntry('c1', 'ls', 20), answer('b', 30)];
    const up: PermissionMark = { kind: 'mode', at: 15, mode: 'bypass', from: 'ask', by: 'user' };
    const row: PermissionMark = {
      kind: 'allowed',
      mode: 'bypass',
      at: 20,
      toolCallId: 'c1',
      access: 'bash',
      card: null,
    };
    const helper: PermissionMark = {
      kind: 'allowed',
      mode: 'bypass',
      at: 25,
      toolCallId: 'other',
      access: 'bash',
      card: null,
    };
    const down: PermissionMark = { kind: 'mode', at: 40, mode: 'ask', from: 'bypass', by: 'user' };
    const placed = placeMarks(entries, [up, row, helper, down]);
    expect(placed.before.get(1)).toEqual([up]);
    expect(placed.before.get(2)).toEqual([helper]);
    expect(placed.allowed.get('c1')).toBe(row);
    expect(placed.end).toEqual([down]);
  });
});

describe('the chip', () => {
  it('each mode: word, icon, tone, name; a menu button, not a toggle', async () => {
    const want = {
      ask: { word: 'Ask', icon: 'lucide-shield', tone: null, keep: null },
      auto: { word: 'Auto', icon: 'lucide-shield-check', tone: 'active', keep: null },
      bypass: { word: 'Bypass', icon: 'lucide-shield-off', tone: 'danger', keep: '' },
    } as const;
    for (const mode of ['ask', 'auto', 'bypass'] as const) {
      await openIn(`pm-chip-${mode}`, mode);
      const c = chip();
      expect(c.getAttribute('data-permission-mode')).toBe(mode);
      expect(shown(c)).toBe(want[mode].word);
      expect(iconOf(c)).toBe(want[mode].icon);
      expect(c.getAttribute('data-chip-tone')).toBe(want[mode].tone);
      expect(c.getAttribute('data-keep-label')).toBe(want[mode].keep);
      expect(accessibleName(c)).toBe(permissionModeName(mode, FOLDER));
      expect(c.hasAttribute('aria-pressed')).toBe(false);
      expect(c.getAttribute('aria-haspopup')).toBe('dialog');
      page!.unmount();
      api!();
      page = undefined;
      api = undefined;
    }
  });

  it('the menu: three radios, the checked one focused, the notes with the folder name, the foot line', async () => {
    await openIn('pm-menu', 'auto');
    const box = await openMenu();
    expect(box.querySelector('[role="radiogroup"]')?.getAttribute('aria-label')).toBe(P.name);
    const radios = [...box.querySelectorAll('[role="radio"]')];
    expect(radios.map((r) => r.getAttribute('aria-checked'))).toEqual(['false', 'true', 'false']);
    await waitFor(() => document.activeElement === radio('auto'));
    const text = shown(box);
    expect(text).toContain(P.notes.ask);
    expect(text).toContain('Plain commands and edits in proj run. The rest asks.');
    expect(text).toContain(P.notes.bypass);
    expect(text).toContain(P.menuFoot);
    expect(iconOf(radio('bypass'))).toBe('lucide-shield-off');
    expect(radio('bypass').getAttribute('data-tone')).toBe('danger');
    expect(switches()).toEqual([]);
  });

  it('a lower row posts at once with no confirm; picking the mode that is on does nothing', async () => {
    await openIn('pm-down', 'bypass', {
      routes: { 'POST /chat/conversations/pm-down/permission-mode': { mode: 'auto' } },
    });
    await openMenu();
    click(radio('bypass'));
    await settle();
    expect(switches()).toEqual([]);
    expect(confirmView()).toBeNull();
    click(radio('auto'));
    await waitFor(() => switches().length === 1);
    expect(switches()).toEqual([{ mode: 'auto' }]);
    expect(confirmView()).toBeNull();
    await waitFor(() => menu() === null);
    // The chip follows the stream, never its own click.
    expect(chip().getAttribute('data-permission-mode')).toBe('bypass');
    act(() => api!.emitChatHost({ type: 'state', state: raisedHost('pm-down', 'auto') }));
    await waitFor(() => chip().getAttribute('data-permission-mode') === 'auto');
  });

  it('a lower row still posts while a move up is in flight', async () => {
    useClock();
    const held = deferred({ mode: 'bypass' });
    await openIn('pm-inflight', 'auto', {
      routes: {
        'POST /chat/conversations/pm-inflight/permission-mode': sequence(held.body, { mode: 'auto' }, { mode: 'ask' }),
      },
    });
    await openConfirm('bypass');
    advance(CARD_ARM_MS);
    press(action('bypass'), true);
    await waitFor(() => switches().length === 1);
    // The up waits on the server; Escape on its confirm calls it back, then the user moves down from the menu.
    key(document.activeElement!, 'Escape');
    await waitFor(() => menu() === null && confirmView() === null);
    await waitFor(() => switches().length === 2);
    await openMenu();
    click(radio('ask'));
    await waitFor(() => switches().length === 3);
    expect(switches()).toEqual([{ mode: 'bypass' }, { mode: 'auto' }, { mode: 'ask' }]);
    held.release();
    await settle();
  });

  it('Back on a confirm whose up is in flight calls the up back; Back with nothing in flight sends nothing', async () => {
    useClock();
    const held = deferred({ mode: 'auto' });
    await openIn('pm-back', 'ask', {
      routes: { 'POST /chat/conversations/pm-back/permission-mode': sequence(held.body, { mode: 'ask' }) },
    });
    await openConfirm('auto');
    click(buttonIn(confirmView()!, P.back));
    await waitFor(() => confirmView() === null && menu() !== null);
    expect(switches()).toEqual([]);
    await openConfirm('auto');
    advance(CARD_ARM_MS);
    press(action('auto'), true);
    await waitFor(() => switches().length === 1);
    click(buttonIn(confirmView()!, P.back));
    await waitFor(() => switches().length === 2);
    expect(switches()).toEqual([{ mode: 'auto' }, { mode: 'ask' }]);
    key(document.activeElement!, 'Escape');
    await waitFor(() => menu() === null);
    // The palette dialog closed while the same up is still in flight calls it back too.
    act(() => void commands.find((c) => c.id === 'chat:permission-bypass')!.run());
    await waitFor(() => confirmView()?.getAttribute('data-permission-confirm') === 'bypass');
    click(buttonIn(confirmView()!, P.back));
    await waitFor(() => switches().length === 3);
    expect(switches()[2]).toEqual({ mode: 'ask' });
    held.release();
    await settle();
  });

  it('an up in flight in one conversation does not block or spin the confirm of another', async () => {
    useClock();
    const held = deferred({ mode: 'bypass' });
    const rows: ConversationRow[] = [
      { ...info('pm-f1'), turns: 1 },
      { ...info('pm-f2'), turns: 1 },
    ];
    await open('/chat?c=pm-f1', (text) => text.includes('Hi there'), {
      rows,
      routes: {
        'POST /chat/conversations/pm-f1/permission-mode': held.body,
        'POST /chat/conversations/pm-f2/permission-mode': { mode: 'auto' },
      },
      conversations: { 'pm-f1': sync('pm-f1'), 'pm-f2': sync('pm-f2') },
    });
    await openConfirm('bypass');
    advance(CARD_ARM_MS);
    press(action('bypass'), true);
    await waitFor(() => switches().length === 1);
    act(() => void navigate!('/chat?c=pm-f2'));
    await waitFor(() => page!.container.textContent?.includes('Title pm-f2') === true);
    await waitFor(() => menu() === null && confirmView() === null);
    await waitFor(() => chip().getAttribute('aria-disabled') === null);
    await openConfirm('auto');
    expect(action('auto').getAttribute('aria-busy')).not.toBe('true');
    advance(CARD_ARM_MS);
    press(action('auto'), true);
    await waitFor(() => switches().length === 2);
    expect(calls.filter((c) => c.method === 'POST').map((c) => pathOf(c))).toEqual([
      '/chat/conversations/pm-f1/permission-mode',
      '/chat/conversations/pm-f2/permission-mode',
    ]);
    held.release();
    await settle();
  });

  it('a 409 mode-changed is no toast; any other failure is the toast, and the chip does not move', async () => {
    const error = vi.spyOn(toast, 'error');
    await openIn('pm-409', 'auto', {
      routes: {
        'POST /chat/conversations/pm-409/permission-mode': stubError('E_CONFLICT', 'no', 'mode-changed'),
      },
    });
    await openMenu();
    click(radio('ask'));
    await waitFor(() => switches().length === 1);
    await settle();
    expect(error).not.toHaveBeenCalled();
    page!.unmount();
    api!();
    await openIn('pm-503', 'auto', {
      routes: { 'POST /chat/conversations/pm-503/permission-mode': stubError('E_UNAVAILABLE', 'no') },
    });
    await openMenu();
    click(radio('ask'));
    await waitFor(() => error.mock.calls.length === 1);
    expect(error).toHaveBeenCalledWith(P.failed);
    expect(chip().getAttribute('data-permission-mode')).toBe('auto');
  });

  it('a higher row opens its confirm and posts nothing; focus on the text; Tab order Back, then the action', async () => {
    await openIn('pm-up', 'ask');
    const view = await openConfirm('auto');
    await waitFor(() => document.activeElement?.hasAttribute('data-permission-text') === true);
    const buttons = [...view.querySelectorAll('button')].map((b) => accessibleName(b));
    expect(buttons).toEqual([P.back, P.auto.confirm]);
    const text = view.querySelector('[data-permission-text]')!;
    expect(text.compareDocumentPosition(buttonIn(view, P.back)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(document.querySelector(`[role="dialog"][aria-label="${P.auto.title}"]`)).not.toBeNull();
    expect(switches()).toEqual([]);
  });

  it('the Auto confirm text: with and without a key, with hooks, and the ends line', async () => {
    await openIn('pm-auto-text', 'ask');
    let text = shown(await openConfirm('auto'));
    expect(text).toContain(P.auto.title);
    expect(text).toContain(fill(P.auto.line1, { folder: 'proj' }));
    expect(text).toContain(fill(P.auto.line2, { folder: 'proj' }));
    expect(text).toContain(fill(P.auto.line3, { folder: 'proj' }));
    expect(text).toContain(`${P.auto.line4Start} ${P.auto.line4Code} ${P.auto.line4End}`);
    expect(text).toContain(
      'Auto ends when you switch back, when the Control Panel restarts (quitting the app does that), or when the app cannot write its record. A new conversation starts in Ask.',
    );
    expect(text).not.toContain(P.auto.hooks);
    expect(text).not.toContain('key.');
    page!.unmount();
    api!();
    await openIn('pm-auto-key', 'ask', {
      host: { ...READY_CHAT_HOST, providers: { ...READY_CHAT_HOST.providers, search: 'brave' } },
      sync: {},
      conversations: { 'pm-auto-key': sync('pm-auto-key', {}, { userHooks: true }) },
    });
    text = shown(await openConfirm('auto'));
    expect(text).toContain(`${P.auto.line4End} ${fill(P.auto.key, { provider: 'Brave' })}`);
    expect(text).toContain(P.auto.hooks);
  });

  it('the Bypass confirm text: keys, hooks, the plan sentence, the ends line and the open requests', async () => {
    const twoKeys = {
      search: 'brave',
      firecrawl: true,
      fetch: 'firecrawl',
      providers: [{ id: 'firecrawl', work: { ready: true, missing: null } }],
    };
    await openIn('pm-b-text', 'auto', { routes: { '/providers': twoKeys } });
    let view = await openConfirm('bypass');
    await waitFor(() => shown(view).includes('Firecrawl'));
    let text = shown(view);
    expect(text).toContain(P.bypass.title);
    expect(text).toContain(fill(P.bypass.line1, { folder: 'proj' }));
    expect(text).toContain('A plan, or a request that offers Deny but not "Allow once", still waits for you.');
    expect(text).toContain(P.bypass.line2);
    expect(text).toContain(`${P.bypass.line3} ${fill(P.bypass.keys, { first: 'Brave', second: 'Firecrawl' })}`);
    expect(text).toContain(P.bypass.line4);
    expect(text).toContain(
      'Bypass ends when you switch back, when the Control Panel restarts (quitting the app does that), or when the app cannot write its record. A new conversation starts in Ask.',
    );
    expect(text).not.toContain(P.bypass.hooks);
    expect(text).not.toContain('allowed at once');
    expect(buttonIn(view, P.bypass.confirm).className).toContain('danger');
    page!.unmount();
    api!();

    await openIn('pm-b-one', 'ask', {
      host: { ...READY_CHAT_HOST, providers: { ...READY_CHAT_HOST.providers, search: 'exa' } },
      conversations: {
        'pm-b-one': sync('pm-b-one', { interactions: [permission()] }, { userHooks: true }),
      },
    });
    text = shown(await openConfirm('bypass'));
    expect(text).toContain(`${P.bypass.line3} ${fill(P.bypass.key, { provider: 'Exa' })}`);
    expect(text).toContain(P.bypass.hooks);
    expect(text).toContain(P.bypass.cardOpen);
    page!.unmount();
    api!();

    const plan = { id: 'p1', kind: 'plan', plan: 'Do it' } as unknown as PermissionInteraction;
    const noOnce = permission({
      id: 'i3',
      options: [{ optionId: 'reject-once', kind: 'reject_once', name: 'No' }],
    });
    await openIn('pm-b-two', 'ask', {
      conversations: {
        'pm-b-two': sync('pm-b-two', { interactions: [permission(), permission({ id: 'i2' }), noOnce, plan] }),
      },
    });
    view = await openConfirm('bypass');
    text = shown(view);
    expect(text).toContain(fill(P.bypass.cardsOpen, { n: 2 }));
    expect(text).not.toContain('key, as in');
  });

  it('the action counts only a trusted press made after the arm time, and posts once', async () => {
    useClock();
    await openIn('pm-arm', 'ask', {
      routes: { 'POST /chat/conversations/pm-arm/permission-mode': { mode: 'auto' } },
    });
    await openConfirm('auto');
    press(action('auto'), false);
    press(action('auto'), true);
    advance(CARD_ARM_MS);
    press(action('auto'), false);
    await settle();
    expect(switches()).toEqual([]);
    press(action('auto'), true);
    await waitFor(() => switches().length === 1);
    expect(switches()).toEqual([{ mode: 'auto' }]);
    await waitFor(() => menu() === null && confirmView() === null);
  });

  it('the clock starts at the confirm, not when the menu opened', async () => {
    useClock();
    await openIn('pm-clock', 'ask');
    await openMenu();
    advance(CARD_ARM_MS * 2);
    await openConfirm('bypass');
    press(action('bypass'), true);
    await settle();
    expect(switches()).toEqual([]);
  });
});

describe('stale facts', () => {
  it('Auto confirm armed, Back, then the Bypass row: a press at once posts nothing', async () => {
    useClock();
    await openIn('pm-back', 'ask');
    await openConfirm('auto');
    advance(CARD_ARM_MS);
    click(buttonIn(confirmView()!, P.back));
    await waitFor(() => confirmView() === null && menu() !== null);
    await waitFor(() => document.activeElement === radio('auto'));
    await openConfirm('bypass');
    press(action('bypass'), true);
    await settle();
    expect(switches()).toEqual([]);
  });

  it('a change of the open-request count starts the arm again', async () => {
    useClock();
    const id = 'pm-count';
    await openIn(id, 'ask', {
      routes: { [`POST /chat/conversations/${id}/permission-mode`]: { mode: 'bypass' } },
      conversations: { [id]: sync(id, { interactions: [permission()] }) },
    });
    await openConfirm('bypass');
    expect(shown(confirmView()!)).toContain(P.bypass.cardOpen);
    advance(CARD_ARM_MS);
    act(() =>
      api!.emitChat(id, {
        type: 'interaction',
        seq: 1,
        interaction: permission({ id: 'i2' }),
      }),
    );
    await waitFor(() => shown(confirmView()!).includes(fill(P.bypass.cardsOpen, { n: 2 })));
    press(action('bypass'), true);
    await settle();
    expect(switches()).toEqual([]);
    advance(CARD_ARM_MS);
    press(action('bypass'), true);
    await waitFor(() => switches().length === 1);
  });

  it('a key sentence that arrives late starts the arm again', async () => {
    useClock();
    const id = 'pm-late';
    const late = deferred({
      search: 'brave',
      firecrawl: true,
      fetch: 'firecrawl',
      providers: [{ id: 'firecrawl', work: { ready: true, missing: null } }],
    });
    await openIn(id, 'ask', {
      routes: { '/providers': late.body, [`POST /chat/conversations/${id}/permission-mode`]: { mode: 'bypass' } },
    });
    await openConfirm('bypass');
    advance(CARD_ARM_MS);
    late.release();
    await waitFor(() => shown(confirmView()!).includes('Firecrawl'));
    press(action('bypass'), true);
    await settle();
    expect(switches()).toEqual([]);
    advance(CARD_ARM_MS);
    press(action('bypass'), true);
    await waitFor(() => switches().length === 1);
  });

  it('another conversation: both chips and the palette dialog are closed', async () => {
    const rows: ConversationRow[] = [
      { ...info('pm-c1', { permissionMode: 'auto' }), turns: 1 },
      { ...info('pm-c2', { permissionMode: 'auto' }), turns: 1 },
    ];
    await open('/chat?c=pm-c1', (text) => text.includes('Hi there'), {
      rows,
      host: {
        ...READY_CHAT_HOST,
        permissionRaised: [
          { id: 'pm-c1', mode: 'auto' },
          { id: 'pm-c2', mode: 'auto' },
        ],
      },
      conversations: {
        'pm-c1': sync('pm-c1', {}, { permissionMode: 'auto' }),
        'pm-c2': sync('pm-c2', {}, { permissionMode: 'auto' }),
      },
    });
    await openConfirm('bypass');
    act(() => void navigate!('/chat?c=pm-c2'));
    await waitFor(() => page!.container.querySelector('[data-permission-chip="header"]') !== null);
    await waitFor(() => menu() === null && confirmView() === null);
    await openMenu('header');
    act(() => void navigate!('/chat?c=pm-c1'));
    await waitFor(() => menu() === null);
    expect(chip('header').getAttribute('aria-expanded')).toBe('false');
    await waitFor(() => permissionCommands().includes(`${fill(P.command, { mode: 'Bypass' })}…`));
    act(() => void commands.find((c) => c.id === 'chat:permission-bypass')!.run());
    await waitFor(() => confirmView() !== null);
    act(() => void navigate!('/chat?c=pm-c2'));
    await waitFor(() => confirmView() === null);
    expect(switches()).toEqual([]);
  });

  it('the palette dialog of one conversation is never drawn over another that was loaded before', async () => {
    const rows: ConversationRow[] = [
      { ...info('pm-d1'), turns: 1 },
      { ...info('pm-d2'), turns: 1 },
    ];
    await open('/chat?c=pm-d2', (text) => text.includes('Hi there'), {
      rows,
      conversations: { 'pm-d1': sync('pm-d1'), 'pm-d2': sync('pm-d2') },
    });
    act(() => void navigate!('/chat?c=pm-d1'));
    await waitFor(() => permissionCommands().includes(`${fill(P.command, { mode: 'Bypass' })}…`));
    act(() => void commands.find((c) => c.id === 'chat:permission-bypass')!.run());
    await waitFor(() => confirmView() !== null);
    const drawn: string[] = [];
    const watch = new MutationObserver((records) => {
      for (const record of records)
        for (const node of record.addedNodes)
          if (
            node instanceof Element &&
            (node.matches('[data-permission-confirm]') || node.querySelector('[data-permission-confirm]') !== null)
          )
            drawn.push(node.outerHTML);
    });
    watch.observe(document.body, { childList: true, subtree: true });
    act(() => void navigate!('/chat?c=pm-d2'));
    await waitFor(() => confirmView() === null);
    await settle();
    watch.disconnect();
    expect(drawn).toEqual([]);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(switches()).toEqual([]);
  });

  it('the shown mode changes under an open confirm: back to the menu', async () => {
    const id = 'pm-moved';
    await openIn(id, 'ask');
    await openConfirm('bypass');
    act(() => api!.emitChatHost({ type: 'state', state: raisedHost(id, 'auto') }));
    await waitFor(() => confirmView() === null && menu() !== null);
    expect(radio('auto').getAttribute('aria-checked')).toBe('true');
    expect(switches()).toEqual([]);
  });

  it('the target becomes blocked: back to the menu, the row disabled; the chip blocked: closed for good', async () => {
    const id = 'pm-blocked';
    await openIn(id, 'ask');
    await openConfirm('bypass');
    act(() => api!.emitChatHost({ type: 'state', state: FAILED_AGENT }));
    await waitFor(() => confirmView() === null && menu() !== null);
    expect(radio('bypass').getAttribute('aria-disabled')).toBe('true');
    expect(radio('auto').getAttribute('aria-disabled')).toBe('true');
    click(radio('bypass'));
    await settle();
    expect(confirmView()).toBeNull();
    act(() => api!.emitChatHost({ type: 'state', state: READY_CHAT_HOST }));
    await waitFor(() => radio('auto').getAttribute('aria-disabled') === null);
    await openConfirm('auto');
    api!.dropPort();
    await waitFor(() => menu() === null && confirmView() === null, 3_000);
    await waitFor(() => chip().getAttribute('aria-disabled') === 'true', 3_000);
    expect(switches()).toEqual([]);
  });

  it('Escape closes the menu and the confirm, posts nothing, and returns focus to the chip', async () => {
    await openIn('pm-esc', 'ask');
    await openConfirm('bypass');
    await waitFor(() => document.activeElement?.hasAttribute('data-permission-text') === true);
    key(document.activeElement!, 'Enter');
    key(document.activeElement!, ' ');
    key(document.activeElement!, 'Escape');
    await waitFor(() => menu() === null && confirmView() === null);
    await waitFor(() => document.activeElement === chip());
    expect(switches()).toEqual([]);
  });
});

describe('block reasons', () => {
  async function tooltipOf(element: HTMLElement): Promise<string> {
    act(() => element.focus());
    await waitFor(() => document.querySelector('[role="tooltip"]') !== null);
    return document.querySelector('[role="tooltip"]')!.textContent ?? '';
  }

  it('the New screen: the chip is disabled and says why', async () => {
    chooseFolder(FOLDER);
    await open('/chat?mode=work', (text) => text.includes(CHAT.heroWorkTitle));
    expect(chip().getAttribute('aria-disabled')).toBe('true');
    expect(chip().getAttribute('data-permission-mode')).toBe('ask');
    expect(await tooltipOf(chip())).toContain(P.newScreen);
    click(chip());
    await settle();
    expect(menu()).toBeNull();
  });

  it('the port down: the chip is disabled in every mode', async () => {
    await openIn('pm-port', 'bypass');
    expect(chip().getAttribute('aria-disabled')).toBeNull();
    api!.dropPort();
    await waitFor(() => chip().getAttribute('aria-disabled') === 'true', 3_000);
    expect(await tooltipOf(chip())).toContain(CHAT.blockers.reconnecting);
    click(chip());
    await settle();
    expect(menu()).toBeNull();
    expect(permissionCommands()).toEqual([]);
  });

  it('an open card disables the Auto row only; Bypass stays open to pick', async () => {
    await openIn('pm-card', 'ask', {
      conversations: { 'pm-card': sync('pm-card', { interactions: [permission()] }) },
    });
    await openMenu();
    expect(radio('auto').getAttribute('aria-disabled')).toBe('true');
    expect(radio('bypass').getAttribute('aria-disabled')).toBeNull();
    click(radio('auto'));
    await settle();
    expect(confirmView()).toBeNull();
    await openConfirm('bypass');
  });

  it('the agent blocker and the folder disable the higher rows, never the lower ones', async () => {
    await openIn('pm-agent', 'auto', { host: raisedHost('pm-agent', 'auto', FAILED_AGENT) });
    await openMenu();
    expect(radio('bypass').getAttribute('aria-disabled')).toBe('true');
    expect(radio('ask').getAttribute('aria-disabled')).toBeNull();
    page!.unmount();
    api!();
    await openIn('pm-cwd', 'bypass', {
      conversations: { 'pm-cwd': sync('pm-cwd', {}, { permissionMode: 'bypass', cwdProblem: 'missing' }) },
    });
    await openMenu();
    expect(radio('ask').getAttribute('aria-disabled')).toBeNull();
    expect(radio('auto').getAttribute('aria-disabled')).toBeNull();
  });

  it('a running turn with no open card does not disable anything', async () => {
    await openIn('pm-run', 'ask', {
      conversations: {
        'pm-run': sync('pm-run', { turn: { turnId: 't1', phase: 'running', startedAt: NOW, wait: null } }),
      },
    });
    expect(chip().getAttribute('aria-disabled')).toBeNull();
    await openMenu();
    expect(radio('auto').getAttribute('aria-disabled')).toBeNull();
    expect(radio('bypass').getAttribute('aria-disabled')).toBeNull();
  });
});

describe('header and list', () => {
  it('the header chip shows only in Auto and Bypass; Bypass keeps its word; its menu moves down in one click', async () => {
    await openIn('pm-h-ask', 'ask');
    expect(page!.container.querySelector('[data-permission-chip="header"]')).toBeNull();
    page!.unmount();
    api!();
    await openIn('pm-h-auto', 'auto');
    expect(chip('header').getAttribute('data-keep-label')).toBeNull();
    expect(chip('header').getAttribute('data-chip-tone')).toBe('active');
    page!.unmount();
    api!();
    await openIn('pm-h-b', 'bypass', {
      routes: { 'POST /chat/conversations/pm-h-b/permission-mode': { mode: 'ask' } },
    });
    expect(chip('header').getAttribute('data-keep-label')).toBe('');
    expect(chip('header').getAttribute('data-chip-tone')).toBe('danger');
    expect(shown(chip('header'))).toBe('Bypass');
    await openMenu('header');
    click(radio('ask'));
    await waitFor(() => switches().length === 1);
    expect(switches()).toEqual([{ mode: 'ask' }]);
  });

  it('list marks: Auto icon with hidden text, Bypass icon and word, none in Ask; also in search results', async () => {
    const rows: ConversationRow[] = [
      { ...info('pm-l-auto'), turns: 1 },
      { ...info('pm-l-b'), turns: 1 },
      { ...info('pm-l-ask'), turns: 1 },
    ];
    await open('/chat?c=pm-l-ask', (text) => text.includes('Hi there'), {
      rows,
      host: {
        ...READY_CHAT_HOST,
        permissionRaised: [
          { id: 'pm-l-auto', mode: 'auto' },
          { id: 'pm-l-b', mode: 'bypass' },
        ],
      },
      conversations: { 'pm-l-ask': sync('pm-l-ask') },
    });
    const row = (id: string): HTMLElement | null =>
      page!.container.querySelector<HTMLElement>(`a[data-conversation-row="${id}"]`);
    await waitFor(() => row('pm-l-b') !== null);
    const check = (): void => {
      const auto = row('pm-l-auto')!.querySelector('[data-permission-mode]')!;
      expect(auto.getAttribute('data-permission-mode')).toBe('auto');
      expect(iconOf(auto)).toBe('lucide-shield-check');
      expect(shown(auto)).toBe('Auto');
      expect(row('pm-l-auto')!.textContent).not.toContain('Bypass');
      const loud = row('pm-l-b')!.querySelector('[data-permission-mode]')!;
      expect(iconOf(loud)).toBe('lucide-shield-off');
      expect(row('pm-l-b')!.textContent).toContain('Bypass · ');
      expect(row('pm-l-ask')!.querySelector('[data-permission-mode]')).toBeNull();
    };
    check();
    const search = byRole(document.body, 'searchbox', CHAT.listSearch) as HTMLInputElement;
    act(() => {
      Reflect.set(HTMLInputElement.prototype, 'value', 'Title', search);
      search.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitFor(() => shown(page!.container).includes(fill(CHAT.listCount, { n: 3 })));
    check();
  });
});

describe('palette', () => {
  it('the commands by mode; Ask and a move down post at once', async () => {
    const label = (mode: string): string => fill(P.command, { mode });
    await openIn('pm-pal', 'bypass', {
      routes: { 'POST /chat/conversations/pm-pal/permission-mode': { mode: 'auto' } },
    });
    await waitFor(() => permissionCommands().length > 0);
    expect(permissionCommands()).toEqual([label('Ask'), label('Auto')]);
    act(() => void commands.find((c) => c.id === 'chat:permission-auto')!.run());
    await waitFor(() => switches().length === 1);
    expect(switches()).toEqual([{ mode: 'auto' }]);
    expect(confirmView()).toBeNull();
    act(() => api!.emitChatHost({ type: 'state', state: raisedHost('pm-pal', 'auto') }));
    await waitFor(() => permissionCommands().includes(`${label('Bypass')}…`));
    expect(permissionCommands()).toEqual([label('Ask'), `${label('Bypass')}…`]);
    act(() => void commands.find((c) => c.id === 'chat:permission-ask')!.run());
    await waitFor(() => switches().length === 2);
    expect(switches()[1]).toEqual({ mode: 'ask' });
    act(() => api!.emitChatHost({ type: 'state', state: READY_CHAT_HOST }));
    await waitFor(() => permissionCommands().includes(`${label('Auto')}…`));
    expect(permissionCommands()).toEqual([`${label('Auto')}…`, `${label('Bypass')}…`]);
  });

  it('"Bypass…" and "Auto…" open the dialog and post nothing; its action needs the trusted press after the arm', async () => {
    useClock();
    await openIn('pm-dialog', 'ask', {
      routes: { 'POST /chat/conversations/pm-dialog/permission-mode': { mode: 'bypass' } },
    });
    await waitFor(() => permissionCommands().length === 2);
    act(() => void commands.find((c) => c.id === 'chat:permission-auto')!.run());
    await waitFor(() => confirmView()?.getAttribute('data-permission-confirm') === 'auto');
    await waitFor(() => document.activeElement?.hasAttribute('data-permission-text') === true);
    click(buttonIn(confirmView()!, P.back));
    await waitFor(() => confirmView() === null);
    act(() => void commands.find((c) => c.id === 'chat:permission-bypass')!.run());
    await waitFor(() => confirmView()?.getAttribute('data-permission-confirm') === 'bypass');
    const dialog = confirmView()!.closest<HTMLElement>('[role="dialog"]')!;
    expect(shown(dialog)).toContain(P.bypass.title);
    expect(shown(dialog)).toContain(fill(P.bypass.line1, { folder: 'proj' }));
    press(action('bypass'), true);
    advance(CARD_ARM_MS);
    press(action('bypass'), false);
    await settle();
    expect(switches()).toEqual([]);
    press(action('bypass'), true);
    await waitFor(() => switches().length === 1);
    expect(switches()).toEqual([{ mode: 'bypass' }]);
    await waitFor(() => confirmView() === null);
  });
});

describe('the page never moves the mode up by itself', () => {
  it('(e) an address parameter posts nothing and opens nothing', async () => {
    await open('/chat?c=pm-e&mode=bypass&permissionMode=bypass', (text) => text.includes('Hi there'), {
      routes: { 'POST /chat/conversations/pm-e/permission-mode': { mode: 'bypass' } },
      conversations: { 'pm-e': sync('pm-e') },
    });
    await settle();
    expect(switches()).toEqual([]);
    expect(chip().getAttribute('data-permission-mode')).toBe('ask');
    expect(menu()).toBeNull();
    expect(confirmView()).toBeNull();
  });

  it('only the confirm action sends a move up; the page sends only moves down', () => {
    const root = join(import.meta.dirname, '../ui/src');
    const files: string[] = [];
    const walk = (dir: string): void => {
      for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) walk(path);
        else if (name.endsWith('.ts') || name.endsWith('.tsx')) files.push(path);
      }
    };
    walk(root);
    const count = (text: string, needle: string): number => {
      let n = 0;
      for (let at = text.indexOf(needle); at !== -1; at = text.indexOf(needle, at + 1)) n += 1;
      return n;
    };
    const holders = files.filter((path) => readFileSync(path, 'utf8').includes('/permission-mode`'));
    expect(holders.map((path) => path.slice(root.length + 1))).toEqual(['pages/chat/work/permission-mode.tsx']);
    for (const path of files) {
      const text = readFileSync(path, 'utf8');
      const ups = count(text, '.onSet(shown.target, true)') + count(text, '.onSet(target, true)');
      expect([path, ups]).toEqual([path, path === holders[0] ? 2 : 0]);
      if (path === holders[0]) {
        expect(count(text, 'onConfirm={')).toBe(2);
        expect(count(text, ', true)')).toBe(ups);
      }
      if (path.endsWith('pages/chat.tsx')) {
        expect(count(text, 'setPermissionMode(')).toBe(
          count(text, "setPermissionMode('ask', false)") + count(text, "setPermissionMode('auto', false)"),
        );
      }
    }
  });
});

describe('status, placeholder and cards', () => {
  it('the status region says the new mode; the Bypass placeholder', async () => {
    const id = 'pm-status';
    await openIn(id, 'ask');
    const status = (): string => page!.container.querySelector('div.sr-only[role="status"]')?.textContent ?? '';
    const area = (): HTMLTextAreaElement => page!.container.querySelector('textarea')!;
    expect(area().placeholder).toBe(CHAT.placeholderReply);
    act(() => api!.emitChatHost({ type: 'state', state: raisedHost(id, 'auto') }));
    await waitFor(() => status() === permissionModeName('auto', FOLDER));
    act(() => api!.emitChatHost({ type: 'state', state: raisedHost(id, 'bypass') }));
    await waitFor(() => status() === permissionModeName('bypass', FOLDER));
    expect(area().placeholder).toBe('Reply. Bypass is on.');
    act(() => api!.emitChatHost({ type: 'state', state: READY_CHAT_HOST }));
    await waitFor(() => status() === permissionModeName('ask', FOLDER));
    expect(area().placeholder).toBe(CHAT.placeholderReply);
  });

  it('a card with no "Allow once": Ask shows its options; Bypass Deny and the line; back in Ask both again', async () => {
    const id = 'pm-cards';
    const card = permission({
      options: [
        { optionId: 'always-allow', kind: 'allow_always', name: 'Always allow curl' },
        { optionId: 'reject-once', kind: 'reject_once', name: 'No' },
      ],
    });
    await openIn(id, 'ask', { conversations: { [id]: sync(id, { interactions: [card] }) } });
    const labels = (): string[] =>
      [...page!.container.querySelectorAll('[data-interaction-card] button[data-option]')].map((b) =>
        accessibleName(b),
      );
    const cardText = (): string => shown(page!.container.querySelector('[data-interaction-card]')!);
    expect(labels()).toEqual([CHAT.deny, CHAT.allowAlwaysCommand]);
    const noOnce = `${fill(P.noAllowOnce, { mode: 'Bypass' })} ${P.noAllowOnceAsk}`;
    expect(cardText()).not.toContain(fill(P.noAllowOnce, { mode: 'Bypass' }));
    act(() => api!.emitChatHost({ type: 'state', state: raisedHost(id, 'bypass') }));
    await waitFor(() => labels().length === 1);
    expect(labels()).toEqual([CHAT.deny]);
    expect(cardText()).toContain(noOnce);
    act(() => api!.emitChatHost({ type: 'state', state: READY_CHAT_HOST }));
    await waitFor(() => labels().length === 2);
    expect(labels()).toEqual([CHAT.deny, CHAT.allowAlwaysCommand]);
  });

  it('an Auto card: its reason and two buttons in Auto; no reason and every option in Ask', async () => {
    const id = 'pm-autocard';
    await openIn(id, 'auto', {
      conversations: {
        [id]: sync(id, { interactions: [permission({ autoAsk: 'not-known' })] }, { permissionMode: 'auto' }),
      },
    });
    const card = (): HTMLElement => page!.container.querySelector<HTMLElement>('[data-interaction-card]')!;
    const labels = (): string[] => [...card().querySelectorAll('button[data-option]')].map((b) => accessibleName(b));
    expect(shown(card())).toContain(autoAskText('not-known', FOLDER));
    expect(labels()).toEqual([CHAT.deny, CHAT.allowOnce]);
    expect(shown(card())).not.toContain(CHAT.grantsNote);
    act(() => api!.emitChatHost({ type: 'state', state: READY_CHAT_HOST }));
    await waitFor(() => labels().length === 3);
    expect(labels()).toEqual([CHAT.deny, CHAT.allowOnce, CHAT.allowAlwaysCommand]);
    expect(shown(card())).not.toContain(P.asksStart);
  });
});

describe('the thread', () => {
  it('mode lines in time order, allowed rows for each mode, a call with no row, and the restart and record lines', async () => {
    const command = 'npm test -- --run';
    const card: PermissionInteraction = permission({ command, input: JSON.stringify({ command }), options: [] });
    const mcpInput = '{"title":"Flaky test"}';
    const mcpCard: PermissionInteraction = permission({
      toolName: 'github__create_issue',
      access: 'mcp',
      command: null,
      input: mcpInput,
      options: [],
    });
    const marks: PermissionMark[] = [
      { kind: 'mode', at: T0 + 100, mode: 'auto', from: 'ask', by: 'user', userHooks: true },
      { kind: 'allowed', mode: 'auto', at: T0 + 250, toolCallId: 'c1', access: 'bash', card },
      { kind: 'mode', at: T0 + 300, mode: 'bypass', from: 'auto', by: 'user', userHooks: true },
      { kind: 'allowed', mode: 'bypass', at: T0 + 450, toolCallId: 'c2', access: 'bash', card: null },
      { kind: 'allowed', mode: 'bypass', at: T0 + 550, toolCallId: 'm1', access: 'mcp', card: mcpCard },
      {
        kind: 'allowed',
        mode: 'bypass',
        at: T0 + 560,
        toolCallId: 'helper-1',
        access: 'bash',
        card: { ...card, command: 'touch b\nmore' },
      },
      { kind: 'allowed', mode: 'bypass', at: T0 + 570, toolCallId: 'helper-2', access: 'mcp', card: null },
      { kind: 'mode', at: T0 + 2_000, mode: 'auto', from: 'bypass', by: 'user' },
      { kind: 'mode', at: T0 + 2_100, mode: 'ask', from: 'auto', by: 'user' },
      { kind: 'mode', at: T0 + 3_000, mode: 'ask', from: 'bypass', by: 'restart' },
      { kind: 'mode', at: T0 + 4_000, mode: 'ask', from: 'auto', by: 'record-failed' },
    ];
    const entries: TranscriptEntry[] = [
      user('Fix the build', T0),
      runEntry('c1', command, T0 + 200, 'all passed'),
      runEntry('c2', 'pwd', T0 + 400, '/tmp/work/proj'),
      {
        kind: 'tool',
        ts: T0 + 500,
        id: 'm1',
        name: 'github__create_issue',
        title: 'github__create_issue',
        status: 'completed',
        input: mcpInput,
        output: 'created #12',
        endTs: T0 + 520,
      },
      answer('Done', T0 + 900),
      end('end_turn', T0 + 1_500),
    ];
    await open('/chat?c=pm-thread', (text) => text.includes('Done'), {
      conversations: { 'pm-thread': sync('pm-thread', { entries, permissionMarks: marks }) },
    });
    const text = shown(page!.container);
    const at = (needle: string): number => {
      const index = text.indexOf(needle);
      expect(index, needle).toBeGreaterThan(-1);
      return index;
    };
    const autoLine = `${P.lineAuto} ${P.lineHooks}`;
    const topLine = `${P.bypass.line} ${P.bypass.lineHooks}`;
    expect(at('Fix the build')).toBeLessThan(at(autoLine));
    expect(at(autoLine)).toBeLessThan(at(command));
    expect(at(command)).toBeLessThan(at(topLine));
    expect(at(topLine)).toBeLessThan(at('pwd'));
    expect(at(`${P.allowed.bypass}touch b`)).toBeLessThan(at('Done'));
    expect(at(`${P.allowed.bypass}${P.allowedTool}`)).toBeLessThan(at('Done'));
    expect(at('Done')).toBeLessThan(at(P.lineAutoDown));
    expect(at(P.lineAutoDown)).toBeLessThan(at(P.askUser));
    expect(at(P.askUser)).toBeLessThan(at('Bypass ended when the Control Panel restarted.'));
    expect(at('Bypass ended when the Control Panel restarted.')).toBeLessThan(
      at('The app could not write its record, so Auto ended.'),
    );
    expect(text).toContain('Mode: Bypass');
    const rows = [...page!.container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')];
    const label = (r: HTMLElement): string => r.getAttribute('aria-label') ?? '';
    const autoRow = rows.find((r) => label(r).includes(command))!;
    expect(label(autoRow)).toContain(P.allowed.auto);
    click(autoRow);
    await waitFor(() => page!.container.querySelector(`pre[aria-label="${CHAT.permCommand}"]`) !== null);
    expect(page!.container.querySelector(`pre[aria-label="${CHAT.permCommand}"]`)?.textContent).toBe(command);
    const plain = rows.find((r) => label(r).includes('pwd'))!;
    expect(label(plain)).toContain(P.allowed.bypass);
    const tool = rows.find((r) => label(r).includes('github__create_issue'))!;
    expect(label(tool)).toContain(P.allowed.bypass);
    click(tool);
    await waitFor(() => page!.container.querySelector(`pre[aria-label="${CHAT.permInput}"]`) !== null);
    expect(page!.container.querySelector(`pre[aria-label="${CHAT.permInput}"]`)?.textContent).toBe(mcpInput);
  });

  it('a settings mark that arrives live is also a toast', async () => {
    const warning = vi.spyOn(toast, 'warning');
    await openIn('pm-set', 'bypass');
    act(() => api!.emitChat('pm-set', { type: 'permission-mark', seq: 1, mark: { kind: 'settings', at: Date.now() } }));
    await waitFor(() => warning.mock.calls.length === 1);
    expect(warning).toHaveBeenCalledWith(P.settingsLabel, { description: P.settingsText });
    await waitFor(() => shown(page!.container).includes(P.settingsLabel));
  });
});
