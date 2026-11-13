/**
 * @vitest-environment happy-dom
 */

/**
 * Render guards for the Sessions list and session detail pages: the live guard
 * on rename/delete, the 409 message staying in its dialog, the copied resume
 * command coming from the server, and the parent/subagent links. The pages are
 * mounted against a stub served over a real MessagePort, as in `pages.test.ts`.
 */

import { MessageChannel } from 'node:worker_threads';

import { CHAT } from '@/lib/chat-strings';
import { writeCache } from '@/lib/json-cache';
import { PREFILL_MIN_TOKENS } from '@/lib/prefill';
import { LIVE_FRESH_MS, LIVE_HINT, prefillAverage, turnPrefill, turnTimeParts, turnTokenBars } from '@/lib/sessions';
import { toast } from '@/lib/toast';
import type {
  SessionDeleteResponse,
  SessionDetail as SessionDetailData,
  SessionDetailResponse,
  SessionListResponse,
  SessionRenameResponse,
  SessionRow,
  SessionTimelineResponse,
  TranscriptEntry,
  TurnMetric,
  WakeCause,
} from '@/lib/types';
import SessionDetail from '@/pages/session-detail';
import Sessions from '@/pages/sessions';
import { usePageCommands } from '@/shell/commands';
import { act, createElement } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import type * as Server from '../src/agent-sessions/types.js';
import { type ApiErrorCode, failure } from '../src/api/errors.js';
import type { LiveEvent } from '../src/live.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { HIDDEN_TEXT, wokenParentText, wokenTimeline } from './helpers/woken-turn.js';
import { allByRole, byLabel, byRole, chartTable, renderPage, type RenderedPage, statValue, waitFor } from './render.js';

const PARENT_ID = '019a0000-0000-7000-8000-000000000001';
const LIVE_ID = '019a0000-0000-7000-8000-000000000002';
const CHILD_ID = '019a0000-0000-7000-8000-000000000003';
const GONE_CHILD_ID = '019a0000-0000-7000-8000-000000000004';
const RESUME = `cd '/work/mlx node' && mlx agent --resume ${PARENT_ID}`;

function row(overrides: Partial<SessionRow>): SessionRow {
  return {
    id: PARENT_ID,
    origin: 'agent',
    title: 'Refactor the planner',
    cwd: '/work/mlx-node',
    model: 'qwen3.8-27b',
    branch: 'main',
    createdAt: '2026-10-20T09:30:00.000Z',
    lastActiveAt: '2026-10-20T11:00:00.000Z',
    messageCount: 42,
    live: false,
    kind: null,
    parentId: null,
    subagentCount: 2,
    tokens: { input: 1_000, output: 2_000, cached: 500 },
    ...overrides,
  };
}

function listBody(): SessionListResponse {
  return {
    sessions: [row({}), row({ id: LIVE_ID, title: 'Live one', live: true, subagentCount: 0 })],
    total: 2,
    cwds: ['/work/mlx-node', '/work/other'],
  };
}

function turn(overrides: Partial<TurnMetric>): TurnMetric {
  return {
    promptId: 'p1',
    ts: Date.UTC(2026, 9, 20, 10),
    inputTokens: 100,
    outputTokens: 50,
    cachedTokens: 20,
    reasoningTokens: 0,
    elapsedMs: 4_000,
    ttftMs: 120,
    prefillMs: null,
    prefillTokens: null,
    prefillCalls: null,
    modelCalls: null,
    decodeTps: 60,
    model: 'qwen3.8-27b',
    ...overrides,
  };
}

function detailBody(overrides: Partial<SessionDetailData> = {}, transcript?: TranscriptEntry[]): SessionDetailResponse {
  const ts = Date.UTC(2026, 9, 20, 10);
  return {
    session: {
      ...row({}),
      gitRoot: '/work/mlx-node',
      headCommit: 'abc123',
      remotes: [],
      agentName: null,
      reasoningEffort: null,
      lastTurnSummary: null,
      parent: null,
      subagents: [
        {
          id: CHILD_ID,
          title: 'Explore the cache',
          type: 'explore',
          description: 'Explore the cache',
          status: 'completed',
          model: null,
          exists: true,
        },
        {
          id: GONE_CHILD_ID,
          title: 'Old helper',
          type: 'general',
          description: 'Old helper',
          status: 'failed',
          model: null,
          exists: false,
        },
      ],
      ...overrides,
    },
    transcript: transcript ?? [
      { kind: 'user', ts, text: 'Please **refactor** the planner', promptIndex: 0 },
      { kind: 'thinking', ts, text: 'secret reasoning' },
      {
        kind: 'tool',
        ts,
        id: 't1',
        name: 'read_file',
        title: 'planner.rs',
        status: 'completed',
        input: '{"path":"planner.rs"}',
      },
      { kind: 'plan', ts, entries: [{ content: 'Split the module', status: 'completed', priority: 'high' }] },
      {
        kind: 'subagent',
        ts,
        childId: CHILD_ID,
        type: 'explore',
        description: 'Explore the cache',
        status: 'completed',
        exists: true,
      },
      {
        kind: 'subagent',
        ts,
        childId: GONE_CHILD_ID,
        type: 'general',
        description: 'Old helper',
        status: 'failed',
        exists: false,
      },
      { kind: 'assistant', ts, text: 'Done. See `planner.rs`.' },
      { kind: 'turn-end', ts, stopReason: 'end_turn', elapsedMs: 4_000 },
    ],
    turns: [turn({}), turn({ promptId: 'p2', ttftMs: 80, decodeTps: 40 })],
    resumeCommand: RESUME,
  };
}

const TURN_START = Date.UTC(2026, 9, 20, 10);

function timelineBody(overrides: Partial<SessionTimelineResponse> = {}): SessionTimelineResponse {
  return {
    turn: 1,
    turns: 2,
    open: false,
    live: false,
    stopReason: 'end_turn',
    asOfMs: Date.now(),
    range: {
      startMs: TURN_START,
      turnStartMs: TURN_START,
      endMs: TURN_START + 73_400,
      lastEventMs: TURN_START + 73_400,
      startExact: true,
      leadMs: 0,
    },
    lanes: [
      {
        id: 'main',
        role: 'main',
        name: 'Main agent',
        type: null,
        model: 'qwen3.8-27b',
        slot: null,
        sessionId: PARENT_ID,
        status: 'completed',
        carriedIn: false,
        carriedOut: false,
        noSteps: null,
        fileBytes: null,
        clamped: false,
        subagents: 0,
        tokenReports: [{ atMs: 73_400, tokens: 150 }],
      },
    ],
    spans: [
      {
        id: 'main:0',
        lane: 'main',
        kind: 'wait',
        reason: 'model',
        until: 'text',
        startMs: 0,
        endMs: 13_600,
        status: 'ok',
        inferredStart: false,
        inferredEnd: false,
        cutStart: false,
        cutEnd: false,
        retry: null,
      },
      {
        id: 'main:1',
        lane: 'main',
        kind: 'generate',
        phase: 'answer',
        startMs: 13_600,
        endMs: 73_400,
        status: 'ok',
        inferredStart: false,
        inferredEnd: false,
        cutStart: false,
        cutEnd: false,
        retry: null,
      },
    ],
    handoffs: [],
    notes: { untimedEvents: 0, foldedSubagents: 0, droppedSpans: 0, unplacedRetries: 0 },
    ...overrides,
  };
}

/** The timeline of turn 1 or 2 of `twoTurns()`: 4 s each, the second starting 10 s after the first. */
function twoTurnsTimeline(turn: 1 | 2, tokens: number): SessionTimelineResponse {
  const base = timelineBody();
  const startMs = TURN_START + (turn === 1 ? 0 : 10_000);
  return timelineBody({
    turn,
    range: {
      startMs,
      turnStartMs: startMs,
      endMs: startMs + 4_000,
      lastEventMs: startMs + 4_000,
      startExact: true,
      leadMs: 0,
    },
    lanes: [{ ...base.lanes[0]!, tokenReports: [{ atMs: 4_000, tokens }] }],
    spans: [
      { ...base.spans[0]!, endMs: 1_000 },
      { ...base.spans[1]!, startMs: 1_000, endMs: 4_000 },
    ],
  });
}

/** A live session's open turn: still running when the body was built. */
function openTimeline(asOfMs: number): SessionTimelineResponse {
  return timelineBody({
    turn: 2,
    turns: 1,
    open: true,
    live: true,
    stopReason: null,
    asOfMs,
    range: {
      startMs: asOfMs - 5_000,
      turnStartMs: asOfMs - 5_000,
      endMs: null,
      lastEventMs: asOfMs - 1_000,
      startExact: false,
      leadMs: 0,
    },
    spans: [
      {
        id: 'main:0',
        lane: 'main',
        kind: 'tool',
        tools: ['run_terminal_command'],
        calls: 1,
        failedCalls: 0,
        startMs: 1_000,
        endMs: null,
        status: 'running',
        inferredStart: false,
        inferredEnd: false,
        cutStart: false,
        cutEnd: false,
        retry: null,
      },
    ],
    lanes: [{ ...timelineBody().lanes[0]!, status: 'running', tokenReports: [] }],
  });
}

/** Two finished turns, each with its prompt, an answer and its turn end. */
function twoTurns(): TranscriptEntry[] {
  const ts = TURN_START;
  return [
    { kind: 'user', ts, text: 'Please refactor the planner', promptIndex: 0 },
    { kind: 'assistant', ts: ts + 1_000, text: 'Done.' },
    { kind: 'turn-end', ts: ts + 4_000, stopReason: 'end_turn', elapsedMs: 4_000 },
    { kind: 'user', ts: ts + 10_000, text: 'Now the tests', promptIndex: 1 },
    { kind: 'assistant', ts: ts + 11_000, text: 'Added.' },
    { kind: 'turn-end', ts: ts + 14_000, stopReason: 'end_turn', elapsedMs: 4_000 },
  ];
}

/** A live session: one finished turn and an open one. */
function liveOpen(): SessionDetailResponse {
  const ts = Date.now() - 5_000;
  const transcript: TranscriptEntry[] = [
    { kind: 'user', ts: ts - 60_000, text: 'First', promptIndex: 0 },
    { kind: 'turn-end', ts: ts - 50_000, stopReason: 'end_turn', elapsedMs: 10_000 },
    { kind: 'user', ts, text: 'Run the upload tests', promptIndex: 1 },
    { kind: 'tool', ts: ts + 1_000, id: 'run', name: 'run_terminal_command', title: 'npm test', status: 'in_progress' },
  ];
  return { ...detailBody({ live: true }, transcript), turns: [turn({})] };
}

function timelineCalls(): string[] {
  return calls.filter((call) => call.path.includes('/timeline')).map((call) => call.path);
}

function timelineStatus(label: string): string {
  return byRole(document.body, 'region', label)?.querySelector('header p')?.textContent ?? '';
}

type Reply = { body: unknown; after?: Promise<void> } | { fail: ApiErrorCode; message: string };

const liveWatchers = new Set<(event: LiveEvent) => void>();

/** A "new data was ingested" event, as the runtime sends it. */
async function emitData(): Promise<void> {
  await act(async () => {
    for (const listener of liveWatchers) listener({ type: 'data', version: 1, traces: false, runtime: false });
  });
}

/** The path as the SPA spells it, without the query and the `/api` prefix. */
function routePath(path: string): string {
  const bare = path.split('?')[0];
  return bare.startsWith('/api') ? bare.slice('/api'.length) : bare;
}

/**
 * Serve `routes` keyed by `"METHOD /path?query"` when that key exists, else by
 * `"METHOD /path"`. An array answers the n-th call with its n-th element and
 * repeats the last.
 */
function stubRoutes(routes: Record<string, Reply | Reply[]>, calls: ApiCall[]): () => void {
  const { port1, port2 } = new MessageChannel();
  port1.unref();
  port2.unref();
  const served = new Map<string, number>();
  const dispose = serveRuntimeOverPort(
    {
      call: (call: ApiCall) => {
        calls.push(call);
        const bare = `${call.method} ${routePath(call.path)}`;
        const query = call.path.indexOf('?');
        const exact = query === -1 ? bare : `${bare}${call.path.slice(query)}`;
        const key = routes[exact] !== undefined ? exact : bare;
        const entry = routes[key];
        if (entry === undefined) return Promise.resolve(failure('E_NOT_FOUND', `no stub for ${key}`));
        let reply: Reply;
        if (Array.isArray(entry)) {
          const nth = served.get(key) ?? 0;
          served.set(key, nth + 1);
          reply = entry[Math.min(nth, entry.length - 1)];
        } else {
          reply = entry;
        }
        const result =
          'fail' in reply ? failure(reply.fail, reply.message) : { ok: true as const, status: 200, body: reply.body };
        if (!('fail' in reply) && reply.after !== undefined) return reply.after.then(() => result);
        return Promise.resolve(result);
      },
      subscribe: () => () => {},
      watchLive: (listener: (event: LiveEvent) => void) => {
        liveWatchers.add(listener);
        return () => {
          liveWatchers.delete(listener);
        };
      },
    },
    bindEventTargetPort(port2),
  );
  connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
  return () => {
    disconnectDashboardApi();
    dispose();
    port1.close();
    port2.close();
  };
}

let mounted: RenderedPage | undefined;
let restoreApi: (() => void) | undefined;
let calls: ApiCall[] = [];
let writeText: ReturnType<typeof vi.fn>;
let originalClipboard: PropertyDescriptor | undefined;

beforeEach(() => {
  calls = [];
  writeText = vi.fn(() => Promise.resolve());
  originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restoreApi?.();
  restoreApi = undefined;
  if (originalClipboard !== undefined) Object.defineProperty(navigator, 'clipboard', originalClipboard);
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

async function settle(): Promise<void> {
  for (let i = 0; i < 10; i++) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

async function click(el: Element | null | undefined): Promise<void> {
  expect(el).toBeTruthy();
  await act(async () => {
    (el as HTMLElement).click();
  });
}

async function typeInto(input: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    // The prototype setter, not `input.value =`: React tracks the instance
    // property and would swallow the change event as a no-op.
    Reflect.set(HTMLInputElement.prototype, 'value', value, input);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function rowOf(id: string): HTMLElement {
  const tr = mounted!.container.querySelector<HTMLElement>(`tr[data-session-id="${id}"]`);
  expect(tr).toBeTruthy();
  return tr!;
}

function actionIn(scope: ParentNode, label: string): HTMLButtonElement {
  const button = scope.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);
  expect(button).toBeTruthy();
  return button!;
}

/** Off by the `disabled` attribute (a plain button) or by `aria-disabled` (a confirm trigger, which keeps its tooltip). */
function isOff(button: HTMLButtonElement): boolean {
  return button.disabled || button.getAttribute('aria-disabled') === 'true';
}

/** Text a screen reader speaks: `aria-hidden` subtrees (a tooltip's width measure) are left out. */
function spokenText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? '';
  if (node instanceof Element && node.getAttribute('aria-hidden') === 'true') return '';
  let text = '';
  for (const child of node.childNodes) text += spokenText(child);
  return text;
}

function dialog(): HTMLElement {
  const el = document.querySelector<HTMLElement>('[role="dialog"]');
  expect(el).toBeTruthy();
  return el!;
}

function dialogButton(text: string): HTMLButtonElement {
  const button = [...dialog().querySelectorAll('button')].find((b) => b.textContent?.trim() === text);
  expect(button).toBeTruthy();
  return button!;
}

function LocationProbe() {
  const location = useLocation();
  return createElement('output', { 'data-location': '' }, `${location.pathname}${location.search}`);
}

/** The page's palette commands as plain buttons, so a test can run one without the shell. */
function CommandsProbe() {
  const commands = usePageCommands();
  return createElement(
    'div',
    { 'data-commands': '' },
    commands.map((command) =>
      createElement('button', { key: command.id, type: 'button', onClick: () => void command.run() }, command.label),
    ),
  );
}

function command(label: string): HTMLButtonElement {
  const button = [...mounted!.container.querySelectorAll<HTMLButtonElement>('[data-commands] button')].find(
    (candidate) => candidate.textContent === label,
  );
  expect(button).toBeTruthy();
  return button!;
}

function location(): string {
  return mounted!.container.querySelector('[data-location]')?.textContent ?? '';
}

async function mountList(routes: Record<string, Reply | Reply[]>, url = '/sessions'): Promise<void> {
  restoreApi = stubRoutes(routes, calls);
  mounted = await renderPage(
    createElement(
      MemoryRouter,
      { initialEntries: [url] },
      createElement(Routes, null, createElement(Route, { path: '/sessions', element: createElement(Sessions) })),
      createElement(LocationProbe),
    ),
    (text) => text.includes('Refactor the planner') || text.includes('Live one'),
  );
}

function writes(): Array<{ method: string; path: string; body?: unknown }> {
  return calls
    .filter((c) => c.method !== 'GET')
    .map((c) => ({ method: c.method, path: c.path, ...(c.body === undefined ? {} : { body: c.body }) }));
}

function listGets(): number {
  return calls.filter((c) => c.method === 'GET' && c.path.split('?')[0] === '/api/sessions').length;
}

describe('Sessions page', () => {
  it('an untitled row reads "Session" and the start of its id; the whole id is only its hover text', async () => {
    const id = '01a1cdb0-5c3e-4d6f-9a1b-2c3d4e5f6a7b';
    await mountList({
      'GET /sessions': { body: { ...listBody(), sessions: [row({}), row({ id, title: '' })], total: 2 } },
    });
    const cell = rowOf(id);
    expect(cell.textContent).toContain('Session 01a1cdb0');
    expect(cell.textContent).not.toContain(id);
    expect(cell.querySelector(`[title="${id}"]`)).not.toBeNull();
    const head = [...document.querySelectorAll('th')].find((th) => th.textContent?.includes('Messages'));
    expect(head?.getAttribute('title')).toContain("The agent's own count");
  });

  it('lists sessions with a link to each detail page and the live badge', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } });
    const link = rowOf(PARENT_ID).querySelector('a');
    expect(link?.getAttribute('href')).toBe(`/sessions/${PARENT_ID}`);
    expect(rowOf(LIVE_ID).textContent).toContain('Live');
    expect(rowOf(PARENT_ID).textContent).not.toContain('Live');
    // The list asks for the server's cap rather than its default of 100.
    expect(calls[0].path).toContain('limit=500');
  });

  it('disables rename and delete on a live session', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } });
    const live = rowOf(LIVE_ID);
    expect(isOff(actionIn(live, 'Rename'))).toBe(true);
    expect(isOff(actionIn(live, 'Delete'))).toBe(true);
    // Copying the resume command stays available while live.
    expect(isOff(actionIn(live, 'Copy resume command'))).toBe(false);
    // A disabled confirm trigger opens nothing.
    await click(actionIn(live, 'Delete'));
    await settle();
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(writes()).toEqual([]);

    const idle = rowOf(PARENT_ID);
    expect(isOff(actionIn(idle, 'Rename'))).toBe(false);
    expect(isOff(actionIn(idle, 'Delete'))).toBe(false);
  });

  it('a live session names the reason in the tooltip of its rename and delete actions', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } });
    const live = rowOf(LIVE_ID);

    const remove = actionIn(live, 'Delete');
    await act(async () => remove.focus());
    await waitFor(() => byRole(document.body, 'tooltip') !== null);
    expect(spokenText(byRole(document.body, 'tooltip')!)).toBe(LIVE_HINT);
    await act(async () => remove.blur());
    await waitFor(() => byRole(document.body, 'tooltip') === null);

    const renameWrapper = actionIn(live, 'Rename').parentElement!;
    await act(async () => {
      renameWrapper.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, pointerType: 'mouse' }));
    });
    await waitFor(() => byRole(document.body, 'tooltip') !== null);
    expect(spokenText(byRole(document.body, 'tooltip')!)).toBe(LIVE_HINT);
  });

  it('a cut search placeholder ends in an ellipsis; the actions column fits its three buttons', async () => {
    const rules = cssRules('components/arc-x/compact-field/compact-field.module.css');
    expect(cssValue(rules, '.compact input', 'text-overflow')).toBe('ellipsis');
    await mountList({ 'GET /sessions': { body: listBody() } });
    const cols = [...mounted!.container.querySelectorAll<HTMLTableColElement>('colgroup col')];
    // three 28 px buttons, two 4 px gaps, 12 px cell start padding, 16 px end padding of the last cell
    expect(Number.parseFloat(cols.at(-1)!.style.width)).toBeGreaterThanOrEqual(3 * 28 + 2 * 4 + 12 + 16);
  });

  it('renames through PATCH {title} and refetches the list', async () => {
    const success = vi.spyOn(toast, 'success');
    await mountList({
      'GET /sessions': [{ body: listBody() }, { body: { ...listBody(), sessions: [row({ title: 'New name' })] } }],
      [`PATCH /sessions/${PARENT_ID}`]: { body: { id: PARENT_ID, title: 'New name' } },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Rename'));
    const input = byLabel(dialog(), 'Session title') as HTMLInputElement;
    expect(input.value).toBe('Refactor the planner');
    // The dialog opens on its title field with the old title selected, so typing replaces it and Enter saves.
    await waitFor(() => document.activeElement === input);
    expect([input.selectionStart, input.selectionEnd]).toEqual([0, input.value.length]);
    await typeInto(input, '  New name  ');
    await click(dialogButton('Save'));
    await settle();

    expect(writes()).toEqual([{ method: 'PATCH', path: `/api/sessions/${PARENT_ID}`, body: { title: 'New name' } }]);
    expect(success).toHaveBeenCalledWith('Session renamed', { description: 'New name' });
    await waitFor(() => document.querySelector('[role="dialog"]') === null);
    expect(listGets()).toBe(2);
    expect(mounted!.text()).toContain('New name');
  });

  it('Cancel closes the rename dialog and puts focus back on its Rename button', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } });
    const rename = actionIn(rowOf(PARENT_ID), 'Rename');
    await act(async () => rename.focus());
    await click(rename);
    await click(dialogButton('Cancel'));
    await waitFor(() => document.querySelector('[role="dialog"]') === null);
    await waitFor(() => document.activeElement === actionIn(rowOf(PARENT_ID), 'Rename'));
  });

  it('keeps a 409 inside the rename dialog instead of a toast', async () => {
    const failed = vi.spyOn(toast, 'error');
    const message = `Session ${PARENT_ID} is live (recent write); retry later`;
    await mountList({
      'GET /sessions': { body: listBody() },
      [`PATCH /sessions/${PARENT_ID}`]: { fail: 'E_CONFLICT', message },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Rename'));
    await typeInto(byLabel(dialog(), 'Session title') as HTMLInputElement, 'Other');
    await click(dialogButton('Save'));
    await settle();

    expect(dialog().querySelector('[role="alert"]')?.textContent).toContain(message);
    expect(failed).not.toHaveBeenCalled();
    // The list is refetched so the row's live state catches up.
    expect(listGets()).toBe(2);
  });

  it('deletes through a confirm at the button and names the subagent sessions removed with it', async () => {
    const success = vi.spyOn(toast, 'success');
    await mountList({
      'GET /sessions': [
        { body: listBody() },
        { body: { ...listBody(), sessions: [listBody().sessions[1]], total: 1 } },
      ],
      [`DELETE /sessions/${PARENT_ID}`]: { body: { deleted: [PARENT_ID, CHILD_ID, GONE_CHILD_ID] } },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Delete'));
    expect(writes()).toEqual([]);
    expect(dialog().textContent).toContain('Delete session?');
    expect(dialog().textContent).toContain('and its 2 subagent sessions');
    await click(dialogButton('Delete'));
    await settle();

    expect(writes()).toEqual([{ method: 'DELETE', path: `/api/sessions/${PARENT_ID}` }]);
    expect(success).toHaveBeenCalledWith('Session deleted', {
      description: 'Refactor the planner and 2 subagent sessions',
    });
    await waitFor(() => mounted!.container.querySelector(`tr[data-session-id="${PARENT_ID}"]`) === null);
  });

  it('shows a delete 409 in place and refreshes the list', async () => {
    const message = 'Subagent session is live';
    await mountList({
      'GET /sessions': { body: listBody() },
      [`DELETE /sessions/${PARENT_ID}`]: { fail: 'E_CONFLICT', message },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Delete'));
    await click(dialogButton('Delete'));
    await settle();
    const alert = dialog().querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('Could not delete');
    expect(alert?.textContent).toContain(message);
    expect(listGets()).toBe(2);
  });

  it('copies the resume command the server built, not one made in the page', async () => {
    await mountList({
      'GET /sessions': { body: listBody() },
      [`GET /sessions/${PARENT_ID}`]: { body: detailBody() },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Copy resume command'));
    await settle();
    expect(writeText).toHaveBeenCalledWith(RESUME);
  });

  it('drops a directory filter once the served directories no longer list it', async () => {
    const other = row({ id: LIVE_ID, title: 'Only one here', cwd: '/work/other', subagentCount: 0 });
    await mountList({
      'GET /sessions': [
        { body: listBody() },
        { body: { sessions: [other], total: 1, cwds: ['/work/mlx-node', '/work/other'] } },
        { body: { sessions: [], total: 0, cwds: ['/work/mlx-node'] } },
        { body: { sessions: [row({})], total: 1, cwds: ['/work/mlx-node'] } },
      ],
      [`DELETE /sessions/${LIVE_ID}`]: { body: { deleted: [LIVE_ID] } },
    });
    const trigger = byLabel(mounted!.container, 'Filter by directory')!;
    expect(trigger).toBeTruthy();
    await act(async () => {
      trigger.focus();
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });
    const option = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(
      (o) => o.textContent?.trim() === '/work/other',
    );
    await act(async () => {
      option!.focus();
      option!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });
    await settle();
    expect(calls.at(-1)!.path).toContain('cwd=%2Fwork%2Fother');
    expect(location()).toContain('cwd=%2Fwork%2Fother');

    await click(actionIn(rowOf(LIVE_ID), 'Delete'));
    await click(dialogButton('Delete'));
    await settle();

    await waitFor(() => !calls.at(-1)!.path.includes('cwd='));
    expect(trigger.textContent).toContain('All directories');
    expect(location()).not.toContain('cwd=');
    await waitFor(() => mounted!.container.querySelector(`tr[data-session-id="${PARENT_ID}"]`) !== null);
  });

  it('says how many sessions the cap left out', async () => {
    await mountList({ 'GET /sessions': { body: { ...listBody(), total: 640 } } });
    const table = byRole(mounted!.container, 'region', 'Sessions');
    expect(table?.textContent).toContain('Showing the newest 2 of 640 matching sessions');
    expect(statValue(mounted!.container, 'Sessions')).toBe('640');
  });

  it('restores search and directory from the URL', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } }, '/sessions?q=planner&cwd=%2Fwork%2Fmlx-node');
    expect(calls[0].path).toContain('q=planner');
    expect(calls[0].path).toContain('cwd=%2Fwork%2Fmlx-node');
    expect((byLabel(mounted!.container, 'Search sessions') as HTMLInputElement).value).toBe('planner');
    expect(byLabel(mounted!.container, 'Filter by directory')?.textContent).toContain('/work/mlx-node');
    expect(byRole(mounted!.container, 'group', 'Matching sessions')).toBeTruthy();
  });

  it('the Live filter keeps only live rows', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } }, '/sessions?live=1');
    expect(mounted!.container.querySelector(`tr[data-session-id="${PARENT_ID}"]`)).toBeNull();
    expect(rowOf(LIVE_ID)).toBeTruthy();
    const tile = byRole(byRole(mounted!.container, 'group', 'Live now')!, 'button');
    expect(tile?.getAttribute('aria-pressed')).toBe('true');
    await click(tile);
    await waitFor(() => mounted!.container.querySelector(`tr[data-session-id="${PARENT_ID}"]`) !== null);
    expect(location()).not.toContain('live=');
  });

  it('sums tokens over rows that carry usage and says how many do not', async () => {
    const body = { ...listBody(), sessions: [row({}), row({ id: LIVE_ID, title: 'No usage', tokens: null })] };
    await mountList({ 'GET /sessions': { body } });
    expect(statValue(mounted!.container, 'Tokens')).toBe('3,000');
    const tokens = byRole(mounted!.container, 'group', 'Tokens');
    expect(tokens?.textContent).toContain('1,000 in · 2,000 out');
    expect(tokens?.textContent).toContain('1 session without usage');
    expect(statValue(mounted!.container, 'Prompt from cache')).toBe('50%');
  });
});

function detailPage(id: string, search = '') {
  return createElement(
    MemoryRouter,
    { initialEntries: [`/sessions/${id}${search}`] },
    createElement(
      Routes,
      null,
      createElement(Route, { path: '/sessions/:id', element: createElement(SessionDetail) }),
      createElement(Route, { path: '/sessions', element: createElement('p', null, 'The session list') }),
    ),
    createElement(LocationProbe),
    createElement(CommandsProbe),
  );
}

async function mountDetail(
  id: string,
  routes: Record<string, Reply | Reply[]>,
  until: string,
  search = '',
): Promise<string> {
  restoreApi = stubRoutes(routes, calls);
  mounted = await renderPage(detailPage(id, search), (text) => text.includes(until));
  return mounted.text();
}

async function showTab(name: string): Promise<void> {
  const tab = allByRole(mounted!.container, 'tab').find((candidate) => candidate.textContent?.startsWith(name));
  expect(tab).toBeTruthy();
  // Radix tabs activate on mouse-down, not click.
  await act(async () => {
    tab!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0 }));
  });
  await settle();
}

describe('Session detail page', () => {
  it('renders the header, the session totals and the resume button', async () => {
    const text = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Per turn');
    expect(mounted!.container.querySelector('h1')?.textContent).toBe('Refactor the planner');
    expect(text).toContain('/work/mlx-node');
    expect(text).toContain('main');
    expect(statValue(mounted!.container, 'Turns')).toBe('2');
    // Means over the two turns: first token (120+80)/2, decode (60+40)/2.
    expect(statValue(mounted!.container, 'First token')).toContain('100 ms');
    expect(statValue(mounted!.container, 'Decode speed')).toContain('50.0');
    // The totals open the activity column: no empty measuring box takes a grid gap above them.
    const totals = byRole(mounted!.container, 'group', 'Session totals');
    expect(totals?.parentElement?.firstElementChild).toBe(totals);

    const copy = [...mounted!.container.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Copy resume command'),
    );
    await click(copy);
    await settle();
    expect(writeText).toHaveBeenCalledWith(RESUME);
  });

  it('rates measured prefills from the floor up, as a token-weighted average with the call count', async () => {
    const turns = [
      // 20,700 tokens in 36.25 s and 4,000 in 2 s: 24,700 / 38.25 = 645.75 tok/s.
      turn({ promptId: 'big', prefillMs: 36_250, prefillTokens: 20_700, prefillCalls: 2, modelCalls: 5 }),
      turn({ promptId: 'mid', prefillMs: 2_000, prefillTokens: 4_000, prefillCalls: 1, modelCalls: 1 }),
      // Measured, but under the floor: no rate.
      turn({ promptId: 'small', prefillMs: 40, prefillTokens: 20, prefillCalls: 1 }),
      // The first-call block of today's agent: not measured.
      turn({ promptId: 'old', modelCalls: 5, ttftMs: 400 }),
    ];
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody(), turns } } },
      'Prefill speed per turn',
    );
    expect(statValue(mounted!.container, 'Prefill speed')).toContain('646');
    const tile = byRole(mounted!.container, 'group', 'Prefill speed');
    expect(tile?.textContent).toContain('rated 2 of 3 measured turns');
    expect(tile?.textContent).not.toContain('mean');
    expect(text).toContain(`Tokens per second · measured prefill of ${PREFILL_MIN_TOKENS}+ tokens`);
    const drawn = chartTable(mounted!.container, 'Prefill speed per turn').filter((cells) =>
      cells.slice(1).some((cell) => cell !== 'no data' && cell !== '—' && cell !== ''),
    );
    expect(drawn).toHaveLength(2);
    expect(drawn.some((cells) => cells.join(' ').includes('2 of 5 model calls'))).toBe(true);
  });

  it('says prefill was not measured for a session whose file holds first-call blocks only', async () => {
    const turns = [turn({ modelCalls: 5, ttftMs: 400 }), turn({ modelCalls: 5, ttftMs: 300 })];
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody(), turns } } },
      'Decode speed per turn',
    );
    expect(statValue(mounted!.container, 'Prefill speed')).toBe('—');
    expect(byRole(mounted!.container, 'group', 'Prefill speed')?.textContent).toContain('not measured in this session');
    expect(text).toContain('Prefill was not measured in this session');
    expect(text).toContain('To first token');
    expect(text).not.toContain('new tokens');
    // The inspector of the selected turn names the call its first token came from.
    expect(text).toContain('first of 5 model calls');
  });

  it('says no prefill reached the floor when every measured prefill is small', async () => {
    const turns = [
      turn({ prefillMs: 40, prefillTokens: 20 }),
      turn({ prefillMs: 90, prefillTokens: PREFILL_MIN_TOKENS - 1 }),
    ];
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody(), turns } } },
      'Decode speed per turn',
    );
    expect(statValue(mounted!.container, 'Prefill speed')).toBe('—');
    expect(text).toContain(`No prefill of ${PREFILL_MIN_TOKENS}+ tokens`);
  });

  it('judges the prefill card by the turns in its window, not by the whole session', async () => {
    // 130 turns; only the first 10 carry a measured, rated prefill. The default window shows turns 71-130.
    const turns = Array.from({ length: 130 }, (_, index) =>
      index < 10 ? turn({ prefillMs: 1_000, prefillTokens: 4_000 }) : turn({ modelCalls: 3, ttftMs: 300 }),
    );
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody(), turns } } },
      'Decode speed per turn',
    );
    expect(text).toContain('Turns 71–130 of 130');
    expect(text).toContain('Prefill was not measured for these turns');
    expect(text).not.toContain(`No prefill of ${PREFILL_MIN_TOKENS}+ tokens`);
    expect(text).not.toContain('Prefill was not measured in this session');
  });

  it('links each existing subagent to its child session and leaves a missing one unlinked', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Subagents');
    // The Subagents table on Activity.
    expect(mounted!.container.querySelectorAll(`a[href="/sessions/${CHILD_ID}"]`)).toHaveLength(1);
    expect(mounted!.container.querySelector(`a[href="/sessions/${GONE_CHILD_ID}"]`)).toBeNull();
    expect(mounted!.text()).toContain('folder missing');
    // The transcript row.
    await showTab('Transcript');
    expect(mounted!.container.querySelectorAll(`a[href="/sessions/${CHILD_ID}"]`)).toHaveLength(1);
    expect(mounted!.container.querySelector(`a[href="/sessions/${GONE_CHILD_ID}"]`)).toBeNull();
  });

  it('does not link a transcript subagent row whose child the server says is gone', async () => {
    // A fork copies the source's spawn lines; those children are not in its `subagents`.
    const ts = Date.UTC(2026, 9, 20, 10);
    const fork = detailBody({ kind: 'fork', subagents: [], subagentCount: 0 }, [
      { kind: 'subagent', ts, childId: CHILD_ID, type: 'explore', description: 'Copied spawn', exists: false },
      { kind: 'subagent', ts, childId: GONE_CHILD_ID, type: 'explore', description: 'Live spawn', exists: true },
    ]);
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: fork } }, 'Copied spawn', '?tab=transcript');
    expect(mounted!.container.querySelector(`a[href="/sessions/${CHILD_ID}"]`)).toBeNull();
    expect(mounted!.container.querySelector(`a[href="/sessions/${GONE_CHILD_ID}"]`)).toBeTruthy();
  });

  it('links a subagent child back to its parent', async () => {
    const child = detailBody(
      {
        id: CHILD_ID,
        title: 'Explore the cache',
        kind: 'subagent',
        parentId: PARENT_ID,
        parent: { id: PARENT_ID, title: 'Refactor the planner' },
        subagents: [],
        subagentCount: 0,
      },
      [],
    );
    const text = await mountDetail(
      CHILD_ID,
      { [`GET /sessions/${CHILD_ID}`]: { body: child } },
      'Subagent of',
      '?tab=transcript',
    );
    const back = mounted!.container.querySelector(`a[href="/sessions/${PARENT_ID}"]`);
    expect(back?.textContent).toBe('Refactor the planner');
    expect(text).toContain('No messages in this session.');
  });

  it('says so when a fork parent no longer exists', async () => {
    const fork = detailBody(
      { kind: 'fork', parentId: CHILD_ID, parent: { id: CHILD_ID, title: null }, subagents: [], subagentCount: 0 },
      [],
    );
    const text = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: fork } }, 'Forked from');
    expect(text).toContain('Forked from a session that no longer exists');
    expect(mounted!.container.querySelector(`a[href="/sessions/${CHILD_ID}"]`)).toBeNull();
  });

  it('renders the transcript: markdown prose, collapsed thinking and tools, plan, turn marker', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Per turn');
    await showTab('Transcript');
    const text = mounted!.text();
    // Markdown, not literal asterisks.
    const strong = [...mounted!.container.querySelectorAll('strong')].find((el) => el.textContent === 'refactor');
    expect(strong?.parentElement?.textContent).toBe('Please refactor the planner');
    expect([...mounted!.container.querySelectorAll('code')].some((el) => el.textContent === 'planner.rs')).toBe(true);
    // Collapsed by default.
    expect(text).toContain('Thinking');
    expect(text).not.toContain('secret reasoning');
    expect(text).not.toContain('"path":"planner.rs"');
    expect(text).toContain('Split the module');
    expect(text).toContain('Turn ended · end_turn · 4.00 s');

    const tool = [...mounted!.container.querySelectorAll('button')].find((b) =>
      (b.getAttribute('aria-label') ?? '').includes('read_file'),
    );
    await click(tool);
    expect(mounted!.text()).toContain('"path":"planner.rs"');
  });

  it('shows why the transcript could not be read', async () => {
    const body = { ...detailBody({}, []), transcriptError: 'EACCES: permission denied' };
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body } },
      'Transcript could not be read',
      '?tab=transcript',
    );
    expect(text).toContain('Transcript could not be read: EACCES: permission denied');
  });

  it('renders the 404 message for an unknown id', async () => {
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { fail: 'E_NOT_FOUND', message: `Session ${PARENT_ID} not found` } },
      'not found',
    );
    expect(text).toContain(`Session ${PARENT_ID} not found`);
    expect(text).toContain('Could not open this session');
  });

  it('Rename and Delete are off on a live session', async () => {
    const body = detailBody({ live: true });
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body } }, 'Per turn');
    expect(isOff(actionIn(mounted!.container, 'Rename'))).toBe(true);
    expect(isOff(actionIn(mounted!.container, 'Delete'))).toBe(true);
  });

  it('refetches on window focus, so a session resumed in a terminal turns live', async () => {
    await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: [{ body: detailBody() }, { body: detailBody({ live: true }) }] },
      'Per turn',
    );
    expect(isOff(actionIn(mounted!.container, 'Rename'))).toBe(false);
    await act(async () => {
      window.dispatchEvent(new Event('focus'));
    });
    await waitFor(() => calls.filter((call) => call.path.startsWith(`/api/sessions/${PARENT_ID}`)).length === 2);
    await settle();
    expect(isOff(actionIn(mounted!.container, 'Rename'))).toBe(true);
    expect(isOff(actionIn(mounted!.container, 'Delete'))).toBe(true);
  });

  it('deleting from the detail page returns to the list', async () => {
    const success = vi.spyOn(toast, 'success');
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: detailBody() },
        [`DELETE /sessions/${PARENT_ID}`]: { body: { deleted: [PARENT_ID] } },
      },
      'Per turn',
    );
    await click(actionIn(mounted!.container, 'Delete'));
    await click(dialogButton('Delete'));
    await waitFor(() => location() === '/sessions');
    expect(mounted!.text()).toContain('The session list');
    expect(success).toHaveBeenCalledWith('Session deleted', { description: 'Refactor the planner' });
  });

  it('a session with no title reads "Session 01a1cdb0", never its whole id; the id stays in the facts', async () => {
    const id = '01a1cdb0-5c3e-4d6f-9a1b-2c3d4e5f6a7b';
    await mountDetail(id, { [`GET /sessions/${id}`]: { body: detailBody({ id, title: '' }) } }, 'Per turn');
    expect(mounted!.container.querySelector('h1')?.textContent).toBe('Session 01a1cdb0');
    expect(document.title).not.toContain(id);
    const facts = byRole(mounted!.container, 'region', 'Session details') ?? mounted!.container;
    expect(facts.textContent).toContain(id);
    expect(mounted!.container.querySelector('h1')?.textContent).not.toContain(id);
  });

  it('an app session names its Mode, not the agent profile; both counts say what they count', async () => {
    const body = detailBody({ origin: 'work', agentName: 'grok-build-plan' });
    const text = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body } }, 'Per turn');
    expect(text).toContain(CHAT.mode);
    expect(text).toContain(CHAT.modeWork);
    expect(text).not.toContain('grok-build-plan');
    const label = [...mounted!.container.querySelectorAll('dt')].find((dt) => dt.textContent === 'Agent messages');
    expect(label?.getAttribute('title')).toContain("The agent's own count");
    const tab = allByRole(mounted!.container, 'tab').find((candidate) =>
      candidate.textContent?.startsWith('Transcript'),
    );
    expect(tab?.getAttribute('title')).toBe('Rows in the transcript');
    const terminal = detailBody({ agentName: 'grok-build-plan' });
    mounted!.unmount();
    restoreApi?.();
    const again = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: terminal } }, 'Per turn');
    expect(again).toContain('grok-build-plan');
  });

  it('a turn stopped before its first token adds no usage: with no turn that has usage, the token tiles print —', async () => {
    const stopped = { ...detailBody(), turns: [turn({ inputTokens: 0, outputTokens: 0, cachedTokens: 0 })] };
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: stopped } }, 'Per turn');
    expect(statValue(mounted!.container, 'Turns')).toBe('1');
    expect(statValue(mounted!.container, 'Prompt')).toBe('—');
    expect(statValue(mounted!.container, 'Output')).toBe('—');
  });

  it('the palette Delete session… opens the centered confirm with the same texts', async () => {
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: detailBody() },
        [`DELETE /sessions/${PARENT_ID}`]: { fail: 'E_CONFLICT', message: 'Subagent session is live' },
      },
      'Per turn',
    );
    await click(command('Delete session…'));
    await waitFor(() => document.querySelector('[role="dialog"]') !== null);
    expect(dialog().textContent).toContain('Delete session?');
    expect(dialog().textContent).toContain('and its 2 subagent sessions');
    await click(dialogButton('Delete'));
    await settle();
    expect(writes()).toEqual([{ method: 'DELETE', path: `/api/sessions/${PARENT_ID}` }]);
    const alert = dialog().querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('Could not delete');
    expect(alert?.textContent).toContain('Subagent session is live');
  });

  it('offers no rename or delete command while the session is live', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody({ live: true }) } }, 'Per turn');
    const labels = [...mounted!.container.querySelectorAll('[data-commands] button')].map(
      (button) => button.textContent,
    );
    expect(labels).toContain('Copy resume command');
    expect(labels).not.toContain('Rename session…');
    expect(labels).not.toContain('Delete session…');
  });

  it('selecting a turn updates the inspector and the URL', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Per turn');
    expect(byRole(mounted!.container, 'region', 'Turn 2')).toBeTruthy();
    await click(actionIn(mounted!.container, 'Previous turn'));
    expect(byRole(mounted!.container, 'region', 'Turn 1')).toBeTruthy();
    expect(location()).toContain('turn=1');
    expect(isOff(actionIn(mounted!.container, 'Previous turn'))).toBe(true);
  });

  it('keeps the tab it opened on when a live refetch brings the first finished turn', async () => {
    const ts = Date.now() - 1_000;
    const open: TranscriptEntry[] = [{ kind: 'user', ts, text: 'Keep going', promptIndex: 0 }];
    const first = { ...detailBody({ live: true }, open), turns: [] };
    const later = { ...detailBody({ live: true }, open), turns: [turn({})] };
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: [{ body: first }, { body: later }] }, 'Keep going');
    const selectedTab = (): string | undefined =>
      allByRole(mounted!.container, 'tab').find((tab) => tab.getAttribute('aria-selected') === 'true')?.textContent ??
      undefined;
    expect(selectedTab()).toContain('Transcript');
    await click(actionIn(mounted!.container, 'Refresh'));
    await waitFor(() => calls.filter((call) => call.path.startsWith(`/api/sessions/${PARENT_ID}`)).length === 2);
    await settle();
    expect(selectedTab()).toContain('Transcript');
    expect(mounted!.text()).toContain('Keep going');
    expect(mounted!.text()).not.toContain('Per turn');
    expect(location()).not.toContain('tab=');
  });

  it('shows no Git root when it names the directory with a trailing slash', async () => {
    const atRoot = detailBody({ cwd: '/a/b', gitRoot: '/a/b/' });
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: atRoot } }, 'Per turn');
    expect(mounted!.text()).not.toContain('Git root');
    mounted!.unmount();
    restoreApi?.();
    const below = detailBody({ cwd: '/a/b/c', gitRoot: '/a/b/' });
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: below } }, 'Per turn');
    expect(mounted!.text()).toContain('Git root');
  });

  it('cuts a long subagent title with an ellipsis: its cell text is a block, not an inline run', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Subagents');
    const table = byRole(mounted!.container, 'table', 'Subagents');
    for (const title of ['Explore the cache', 'Old helper (folder missing)']) {
      const span = table?.querySelector<HTMLElement>(`span[title="${title}"]`);
      expect(span, title).toBeTruthy();
      const classes = (span!.getAttribute('class') ?? '').split(' ');
      expect(classes).toContain('truncate');
      expect(classes).toContain('block');
    }
  });

  it('draws the Per turn title in the foreground color inside the tab panel', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Per turn');
    const heading = byRole(mounted!.container, 'heading', 'Per turn');
    expect(heading?.closest('[role="tabpanel"]')).toBeTruthy();
    expect(cssValue(cssRules('shell/page.module.css'), '.sectionTitle', 'color')).toBe('var(--foreground)');
  });

  it('prints each time axis in one format, its unit once', async () => {
    const turns = [
      turn({ elapsedMs: 73_400, ttftMs: 13_600, decodeTps: 60, outputTokens: 3_564 }),
      turn({ promptId: 'p2', elapsedMs: 12_000, ttftMs: 1_500 }),
    ];
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody(), turns } } }, 'Per turn');
    const time = byRole(mounted!.container, 'figure', 'Time per turn');
    const ticks = [...(time?.querySelectorAll('[class*="gutter"] > span') ?? [])].map((node) => node.textContent ?? '');
    expect(ticks.length).toBeGreaterThan(2);
    expect(ticks[0]).toBe('0');
    for (const tick of ticks) expect(Number.isFinite(Number(tick.replaceAll(',', ''))), tick).toBe(true);
    const rows = chartTable(mounted!.container, 'Time per turn');
    for (const cell of rows.flatMap((cells) => cells.slice(1))) {
      expect(Number.isFinite(Number(cell.replaceAll(',', ''))), cell).toBe(true);
    }
    expect(rows[0]!.slice(1)).toEqual(['13.6', '59.4', '0.4', '73.4']);
  });

  it('prints the agent timeline axis in one unit', async () => {
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody({}, twoTurns()) } },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: timelineBody() },
      },
      'Positions come from recorded event times',
      '?turn=1',
    );
    const timeline = byRole(mounted!.container, 'region', 'Turn 1 timeline');
    const axis = [...(timeline?.querySelectorAll<HTMLElement>('[style*="--t"]') ?? [])]
      .map((node) => node.textContent ?? '')
      .filter((text) => text !== '');
    expect(axis).toEqual(['0', '20 s', '40 s', '60 s']);
  });

  it('requests the timeline of the selected turn, and again when the turn changes', async () => {
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: detailBody({}, twoTurns()) },
        [`GET /sessions/${PARENT_ID}/timeline?turn=1`]: { body: twoTurnsTimeline(1, 150) },
        [`GET /sessions/${PARENT_ID}/timeline?turn=2`]: { body: twoTurnsTimeline(2, 230) },
      },
      'Positions come from recorded event times',
      '?turn=1',
    );
    expect(timelineCalls()).toEqual([`/api/sessions/${PARENT_ID}/timeline?turn=1`]);
    expect(byRole(mounted!.container, 'region', 'Turn 1 timeline')?.querySelector('h3')?.textContent).toBe(
      'Please refactor the planner',
    );
    expect(timelineStatus('Turn 1 timeline')).toBe('Finished · 4.00 s · 150 tokens');
    await click(actionIn(mounted!.container, 'Next turn'));
    await waitFor(() => timelineCalls().length === 2);
    expect(timelineCalls()[1]).toBe(`/api/sessions/${PARENT_ID}/timeline?turn=2`);
    await waitFor(() => timelineStatus('Turn 2 timeline') !== '');
    expect(timelineStatus('Turn 2 timeline')).toBe('Finished · 4.00 s · 230 tokens');
    const text = mounted!.text();
    expect(text).not.toContain('Positions come from transcript timestamps');
    expect(text).not.toContain("Lengths come from the turn's totals");
  });

  it('selects the open turn of a live session by default; previous and next reach it', async () => {
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: liveOpen() },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: openTimeline(Date.now()) },
      },
      'Run the upload tests',
      '?tab=activity',
    );
    await waitFor(() => byRole(mounted!.container, 'region', 'Turn 2 · in progress') !== null);
    expect(isOff(actionIn(mounted!.container, 'Next turn'))).toBe(true);
    await click(actionIn(mounted!.container, 'Previous turn'));
    expect(byRole(mounted!.container, 'region', 'Turn 1')).toBeTruthy();
    await click(actionIn(mounted!.container, 'Next turn'));
    expect(byRole(mounted!.container, 'region', 'Turn 2 · in progress')).toBeTruthy();
    expect(timelineStatus('Turn 2 timeline').startsWith('Live')).toBe(true);
  });

  it('with no finished turn shows the empty state and the open turn', async () => {
    const body = { ...liveOpen(), turns: [] };
    body.transcript = body.transcript.slice(2);
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: { ...openTimeline(Date.now()), turn: 1, turns: 0 } },
      },
      'No completed turns yet',
      '?tab=activity',
    );
    await waitFor(() => byRole(mounted!.container, 'region', 'Turn 1 · in progress') !== null);
    expect(mounted!.text()).toContain('No completed turns yet');
  });

  it('refetches the timeline on a data event exactly when the detail query is live', async () => {
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: liveOpen() },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: openTimeline(Date.now()) },
      },
      'Run the upload tests',
      '?tab=activity',
    );
    await waitFor(() => timelineStatus('Turn 2 timeline') !== '');
    await settle();
    const realNow = Date.now.bind(Date);
    vi.spyOn(Date, 'now').mockImplementation(() => realNow() + 60_000);
    await emitData();
    await waitFor(() => timelineCalls().length === 2);
    mounted!.unmount();
    restoreApi?.();
    calls = [];
    vi.restoreAllMocks();

    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: detailBody({}, twoTurns()) },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: timelineBody() },
      },
      'Positions come from recorded event times',
      '?turn=1',
    );
    await settle();
    vi.spyOn(Date, 'now').mockImplementation(() => realNow() + 60_000);
    await emitData();
    await settle();
    expect(timelineCalls()).toHaveLength(1);
  });

  it('a body that turns 10 s old switches the live status to In progress · as of', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'], shouldAdvanceTime: true });
    try {
      await mountDetail(
        PARENT_ID,
        {
          [`GET /sessions/${PARENT_ID}`]: { body: liveOpen() },
          [`GET /sessions/${PARENT_ID}/timeline`]: { body: openTimeline(Date.now()) },
        },
        'Run the upload tests',
        '?tab=activity',
      );
      await waitFor(() => timelineStatus('Turn 2 timeline') !== '');
      const region = (): Element | null => byRole(mounted!.container, 'region', 'Turn 2 timeline');
      expect(timelineStatus('Turn 2 timeline').startsWith('Live')).toBe(true);
      expect(region()?.getAttribute('data-mode')).toBe('live');
      await act(async () => {
        vi.advanceTimersByTime(LIVE_FRESH_MS);
      });
      expect(timelineStatus('Turn 2 timeline').startsWith('In progress · as of')).toBe(true);
      expect(region()?.getAttribute('data-mode')).toBe('stale');
    } finally {
      vi.useRealTimers();
    }
  });

  it('opens a cached old body as stale, never live, and turns live when the refetch lands', async () => {
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    restoreApi = stubRoutes(
      {
        [`GET /sessions/${PARENT_ID}`]: { body: liveOpen() },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: openTimeline(Date.now()), after: held },
      },
      calls,
    );
    writeCache(`/sessions/${PARENT_ID}/timeline?turn=2`, openTimeline(Date.now() - 120_000));
    const seen: string[] = [];
    mounted = await renderPage(detailPage(PARENT_ID, '?tab=activity'), () => {
      const status = timelineStatus('Turn 2 timeline');
      if (status !== '') seen.push(status);
      return status.startsWith('In progress · as of');
    });
    expect(seen.some((status) => status.startsWith('Live'))).toBe(false);
    await act(async () => {
      release();
      await held;
    });
    await waitFor(() => timelineStatus('Turn 2 timeline').startsWith('Live'));
  });

  it('adds no turn for a trailing group that only switches the model', async () => {
    const ts = TURN_START;
    const transcript: TranscriptEntry[] = [
      { kind: 'user', ts, text: 'Only turn', promptIndex: 0 },
      { kind: 'turn-end', ts: ts + 4_000, stopReason: 'end_turn', elapsedMs: 4_000 },
      { kind: 'model', ts: ts + 9_000, modelId: 'gemma-4-e4b' },
    ];
    const text = await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody({}, transcript), turns: [turn({})] } },
        [`GET /sessions/${PARENT_ID}/timeline`]: { body: timelineBody() },
      },
      'Per turn',
      '?tab=activity',
    );
    expect(byRole(mounted!.container, 'region', 'Turn 1')).toBeTruthy();
    expect(isOff(actionIn(mounted!.container, 'Next turn'))).toBe(true);
    expect(text).not.toContain('not finished');
  });

  it('makes no timeline request for a transcript that could not be read, and says why', async () => {
    const body = { ...detailBody({}, []), transcriptError: 'EACCES: permission denied' };
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body } },
      'there is no timeline for this turn',
      '?tab=activity',
    );
    expect(text).toContain('The transcript could not be read, so there is no timeline for this turn.');
    expect(timelineCalls()).toEqual([]);
  });

  it('marks the open thinking of a live session as live; a finished one reads its last turn as cut off', async () => {
    const ts = Date.now() - 1_000;
    const open: TranscriptEntry[] = [
      { kind: 'user', ts, text: 'Keep going', promptIndex: 0 },
      { kind: 'thinking', ts, text: 'still thinking' },
      { kind: 'tool', ts, id: 'cut', name: 'run_terminal_command', title: 'cargo test', status: 'in_progress' },
    ];
    await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: detailBody({ live: true }, open) } },
      'in progress',
      '?tab=transcript',
    );
    expect(mounted!.container.querySelector('[data-live]')).toBeTruthy();
    mounted!.unmount();
    restoreApi?.();

    await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: detailBody({}, open) } },
      'not finished',
      '?tab=transcript',
    );
    expect(mounted!.container.querySelector('[data-live]')).toBeNull();
    expect(mounted!.container.textContent).not.toContain('in progress');
  });
});

describe('Session detail page: a woken turn', () => {
  const warnings: string[] = [];
  const onWarning = (warning: Error): void => {
    warnings.push(`${warning.name}: ${warning.message}`);
  };
  beforeEach(() => {
    warnings.length = 0;
    process.on('warning', onWarning);
  });
  afterEach(() => {
    process.off('warning', onWarning);
    expect(warnings).toEqual([]);
  });
  function wokenDetail(): SessionDetailResponse {
    const parsed = parseUpdates(wokenParentText());
    return { ...detailBody({}, parsed.transcript), turns: parsed.turns };
  }
  const userMessages = (): number => mounted!.container.querySelectorAll('[data-role="user"]').length;
  const hiddenToggle = (): HTMLButtonElement | undefined =>
    [...mounted!.container.querySelectorAll<HTMLButtonElement>('button[aria-expanded]')].find((button) =>
      (button.textContent ?? '').includes('Hidden prompt'),
    );

  it('titles the turn by what woke the agent in the inspector and the transcript', async () => {
    await mountDetail(
      PARENT_ID,
      {
        [`GET /sessions/${PARENT_ID}`]: { body: wokenDetail() },
        [`GET /sessions/${PARENT_ID}/timeline?turn=2`]: { body: { ...wokenTimeline(2), asOfMs: Date.now() } },
      },
      'Positions come from recorded event times',
      '?tab=activity&turn=2',
    );
    expect(byRole(mounted!.container, 'region', 'Turn 2 timeline')?.querySelector('h3')?.textContent).toBe(
      'Woken by workflow deep-research',
    );
    expect(timelineStatus('Turn 2 timeline')).toContain('Finished · 4m 12s · after 33m 51s idle');
    expect(mounted!.text()).not.toContain('(no prompt)');
    await showTab('Transcript');
    await waitFor(() => mounted!.text().includes('No prompt typed'));
    const text = mounted!.text();
    expect(text).toContain('Woken by workflow deep-research');
    expect(text).not.toContain('(no prompt)');
    expect(userMessages()).toBe(1);
    expect(text).not.toContain(HIDDEN_TEXT);
    await click(hiddenToggle());
    expect(mounted!.text()).toContain(HIDDEN_TEXT);
  });

  it('keeps the notice but not the hidden prompt in the Messages filter', async () => {
    await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: wokenDetail() } },
      'No prompt typed',
      '?tab=transcript',
    );
    expect(hiddenToggle()).toBeTruthy();
    const messages = [...mounted!.container.querySelectorAll('button')].find((b) => b.textContent === 'Messages');
    await click(messages);
    await settle();
    expect(mounted!.text()).toContain('No prompt typed');
    expect(hiddenToggle()).toBeUndefined();
  });

  it('a live trailing group with only spawn lines is no turn; with a wake it is the open turn', async () => {
    const ts = Date.now() - 5_000;
    const base: TranscriptEntry[] = [
      { kind: 'user', ts: ts - 60_000, text: 'First', promptIndex: 0 },
      { kind: 'turn-end', ts: ts - 50_000, stopReason: 'end_turn', elapsedMs: 10_000 },
      { kind: 'subagent', ts: ts - 40_000, childId: CHILD_ID, type: 'general', description: 'Background work' },
    ];
    const routes = (transcript: TranscriptEntry[]) => ({
      [`GET /sessions/${PARENT_ID}`]: { body: { ...detailBody({ live: true }, transcript), turns: [turn({})] } },
      [`GET /sessions/${PARENT_ID}/timeline`]: { body: openTimeline(Date.now()) },
    });
    await mountDetail(PARENT_ID, routes(base), 'Per turn', '?tab=activity');
    expect(byRole(mounted!.container, 'region', 'Turn 2 · in progress')).toBeNull();
    expect(isOff(actionIn(mounted!.container, 'Next turn'))).toBe(true);
    mounted!.unmount();
    restoreApi?.();

    const woken: TranscriptEntry[] = [...base, { kind: 'wake', ts, cause: 'workflow', text: 'Go on' }];
    await mountDetail(PARENT_ID, routes(woken), 'Per turn', '?tab=activity');
    await waitFor(() => byRole(mounted!.container, 'region', 'Turn 2 · in progress') !== null);
  });
});

describe('turnTokenBars', () => {
  it('splits the prompt into uncached and cached so the stack sums to input + output', () => {
    // A real turn: grok's inputTokens already includes the cache reads.
    const bars = turnTokenBars({ inputTokens: 332_055, cachedTokens: 288_974, outputTokens: 5_661 });
    expect(bars).toEqual({ uncached: 43_081, cached: 288_974, output: 5_661 });
    expect(bars.uncached + bars.cached + bars.output).toBe(332_055 + 5_661);
    expect(turnTokenBars({ inputTokens: 10, cachedTokens: 25, outputTokens: 1 })).toEqual({
      uncached: 0,
      cached: 10,
      output: 1,
    });
  });
});

describe('turnPrefill / prefillAverage / turnTimeParts', () => {
  it('reads a measured pair, rates it from the floor up, and applies the floor per call', () => {
    expect(turnPrefill({ prefillMs: null, prefillTokens: null, prefillCalls: null })).toBeNull();
    expect(turnPrefill({ prefillMs: 100, prefillTokens: null, prefillCalls: null })).toBeNull();
    expect(turnPrefill({ prefillMs: 100, prefillTokens: PREFILL_MIN_TOKENS - 1, prefillCalls: null })).toEqual({
      tokens: PREFILL_MIN_TOKENS - 1,
      ms: 100,
      calls: 1,
      tps: null,
    });
    expect(turnPrefill({ prefillMs: 100, prefillTokens: PREFILL_MIN_TOKENS, prefillCalls: null })?.tps).toBe(
      PREFILL_MIN_TOKENS * 10,
    );
    // 600 tokens over 3 calls is 200 per call: under the floor.
    expect(turnPrefill({ prefillMs: 300, prefillTokens: 600, prefillCalls: 3 })?.tps).toBeNull();
    expect(turnPrefill({ prefillMs: 300, prefillTokens: 600, prefillCalls: 2 })?.tps).toBe(2_000);
  });

  it('weights the average by tokens over rated turns only', () => {
    const big = turnPrefill({ prefillMs: 30_000, prefillTokens: 30_000, prefillCalls: 1 });
    const mid = turnPrefill({ prefillMs: 10_000, prefillTokens: 1_000, prefillCalls: 1 });
    const small = turnPrefill({ prefillMs: 1_000, prefillTokens: 20, prefillCalls: 1 });
    // 31,000 tokens / 40 s, not the mean of the two rates (550); the small one is left out.
    expect(prefillAverage([big, mid, small, null])).toBe(775);
    expect(prefillAverage([small])).toBeNull();
    expect(prefillAverage([])).toBeNull();
  });

  it('splits a turn into time to first token, decode and the rest', () => {
    const parts = turnTimeParts({ elapsedMs: 10_000, ttftMs: 400, decodeTps: 50, outputTokens: 100 });
    expect(parts).toEqual({ firstTokenMs: 400, decodeMs: 2_000, otherMs: 7_600, totalMs: 10_000 });
    const clamped = turnTimeParts({ elapsedMs: 1_000, ttftMs: 800, decodeTps: 10, outputTokens: 100 });
    expect((clamped.firstTokenMs ?? 0) + (clamped.decodeMs ?? 0) + (clamped.otherMs ?? 0)).toBe(1_000);
  });
});

describe('App conversations on the Sessions pages', () => {
  const CHAT_ID = 'c-0000000000000001';
  const WORK_ID = 'w-0000000000000002';
  const appList = (): SessionListResponse => ({
    sessions: [
      row({}),
      row({ id: CHAT_ID, origin: 'chat', title: 'Chat about Rust', cwd: '', subagentCount: 0, live: true }),
      row({ id: WORK_ID, origin: 'work', title: 'Fix the build', cwd: '/work/app', subagentCount: 0 }),
    ],
    total: 3,
    cwds: ['/work/mlx-node', '/work/app'],
  });

  it('tags each app conversation with its workspace; only a terminal session has no tag', async () => {
    await mountList({ 'GET /sessions': { body: appList() } });
    expect(rowOf(CHAT_ID).textContent).toContain(CHAT.modeChat);
    expect(rowOf(WORK_ID).textContent).toContain(CHAT.modeWork);
    expect(rowOf(PARENT_ID).textContent).not.toContain(CHAT.modeWork);
    expect(rowOf(PARENT_ID).textContent).not.toContain(CHAT.modeChat);
  });

  it('a Chat conversation has no resume command; a live one can still be renamed and deleted', async () => {
    await mountList({ 'GET /sessions': { body: appList() } });
    const chat = rowOf(CHAT_ID);
    expect(chat.querySelector('button[aria-label="Copy resume command"]')).toBeNull();
    expect(isOff(actionIn(chat, 'Rename'))).toBe(false);
    expect(isOff(actionIn(chat, 'Delete'))).toBe(false);
    expect(isOff(actionIn(rowOf(WORK_ID), 'Copy resume command'))).toBe(false);
  });

  it('renames a Work conversation through the chat route', async () => {
    await mountList({
      'GET /sessions': { body: appList() },
      [`PATCH /chat/conversations/${WORK_ID}`]: { body: { id: WORK_ID, title: 'Build fixed' } },
    });
    await click(actionIn(rowOf(WORK_ID), 'Rename'));
    await typeInto(byLabel(dialog(), 'Session title') as HTMLInputElement, 'Build fixed');
    await click(dialogButton('Save'));
    await settle();
    expect(writes()).toEqual([
      { method: 'PATCH', path: `/api/chat/conversations/${WORK_ID}`, body: { title: 'Build fixed' } },
    ]);
  });

  it('deletes a Work conversation through the chat route and says its files stay', async () => {
    await mountList({
      'GET /sessions': [{ body: appList() }, { body: { ...appList(), sessions: appList().sessions.slice(0, 2) } }],
      [`DELETE /chat/conversations/${WORK_ID}`]: { body: {} },
    });
    await click(actionIn(rowOf(WORK_ID), 'Delete'));
    expect(dialog().textContent).toContain(CHAT.deleteTitle);
    expect(dialog().textContent).toContain('/work/app');
    expect(dialog().textContent).toContain('are not touched');
    await click(dialogButton(CHAT.deleteConfirm));
    await settle();
    expect(writes()).toEqual([{ method: 'DELETE', path: `/api/chat/conversations/${WORK_ID}` }]);
    await waitFor(() => mounted!.container.querySelector(`tr[data-session-id="${WORK_ID}"]`) === null);
  });

  it('the detail page of a Chat conversation opens it in Chat and offers no resume command', async () => {
    const body: SessionDetailResponse = {
      ...detailBody({
        id: CHAT_ID,
        origin: 'chat',
        title: 'Chat about Rust',
        cwd: '',
        subagents: [],
        subagentCount: 0,
      }),
      resumeCommand: null,
    };
    await mountDetail(CHAT_ID, { [`GET /sessions/${CHAT_ID}`]: { body } }, 'Chat about Rust');
    const open = [...mounted!.container.querySelectorAll('a')].find((a) => a.textContent === CHAT.commands.openInChat);
    expect(open?.getAttribute('href')).toBe(`/chat?c=${CHAT_ID}`);
    expect(mounted!.container.querySelector('button[aria-label="Copy resume command"]')).toBeNull();
    const labels = [...mounted!.container.querySelectorAll('[data-commands] button')].map((b) => b.textContent);
    expect(labels).toContain(CHAT.commands.openInChat);
    expect(labels).not.toContain('Copy resume command');
    await click(command(CHAT.commands.openInChat));
    await waitFor(() => location() === `/chat?c=${CHAT_ID}`);
  });
});

describe('UI session types', () => {
  it('mirror the server contract exactly', () => {
    // Checked by `tsc`: a field added to or changed on either side fails the typecheck.
    expectTypeOf<SessionRow>().toEqualTypeOf<Server.SessionRow>();
    expectTypeOf<SessionDetailData>().toEqualTypeOf<Server.SessionDetail>();
    expectTypeOf<TranscriptEntry>().toEqualTypeOf<Server.TranscriptEntry>();
    expectTypeOf<WakeCause>().toEqualTypeOf<Server.WakeCause>();
    expectTypeOf<TurnMetric>().toEqualTypeOf<Server.TurnMetric>();
    expectTypeOf<SessionListResponse>().toEqualTypeOf<Server.SessionListResponse>();
    expectTypeOf<SessionDetailResponse>().toEqualTypeOf<Server.SessionDetailResponse>();
    expectTypeOf<SessionRenameResponse>().toEqualTypeOf<Server.SessionRenameResponse>();
    expectTypeOf<SessionDeleteResponse>().toEqualTypeOf<Server.SessionDeleteResponse>();
    expectTypeOf<SessionTimelineResponse>().toEqualTypeOf<Server.SessionTimelineResponse>();
  });
});

describe('home folder as ~', () => {
  const HEALTH = {
    body: { status: 'ok', modelsDir: '', tracesDir: '', runtimeDir: '', homeDir: '/work' },
  };

  it('shows each row directory and filter option under the home folder as ~/…, the full path on hover', async () => {
    await mountList({ 'GET /sessions': { body: listBody() }, 'GET /health': HEALTH });
    await waitFor(() => (rowOf(PARENT_ID).textContent ?? '').includes('~/mlx-node'));
    const cell = rowOf(PARENT_ID).querySelector('[title="/work/mlx-node"]');
    expect(cell?.textContent).toBe('~/mlx-node');
    expect(rowOf(PARENT_ID).textContent).not.toContain('/work/mlx-node');
    const trigger = byLabel(mounted!.container, 'Filter by directory')!;
    await act(async () => {
      trigger.focus();
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });
    const labels = [...document.querySelectorAll<HTMLElement>('[role="option"]')].map((o) => o.textContent?.trim());
    expect(labels).toContain('~/other');
    expect(labels).not.toContain('/work/other');
  });

  it('shows the Directory fact as ~/… and still copies the full path', async () => {
    await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() }, 'GET /health': HEALTH },
      'Per turn',
    );
    await waitFor(() => mounted!.container.querySelector('[title="/work/mlx-node"]')?.textContent === '~/mlx-node');
    const copy = allByRole(mounted!.container, 'button').find((b) => b.getAttribute('aria-label') === 'Copy Directory');
    expect(copy).toBeTruthy();
    await click(copy);
    expect(writeText).toHaveBeenCalledWith('/work/mlx-node');
  });
});
