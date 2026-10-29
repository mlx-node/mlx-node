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

import type {
  SessionDeleteResponse,
  SessionDetail as SessionDetailData,
  SessionDetailResponse,
  SessionListResponse,
  SessionRenameResponse,
  SessionRow,
  TranscriptEntry,
  TurnMetric,
} from '@/lib/types';
import SessionDetail from '@/pages/session-detail';
import Sessions from '@/pages/sessions';
import { act, createElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { toast } from 'sonner';
import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from 'vite-plus/test';

import type * as Server from '../src/agent-sessions/types.js';
import { type ApiErrorCode, failure } from '../src/api/errors.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi } from '../ui/src/lib/api.js';
import { renderPage, type RenderedPage, stubChartMetrics } from './render.js';

const PARENT_ID = '019a0000-0000-7000-8000-000000000001';
const LIVE_ID = '019a0000-0000-7000-8000-000000000002';
const CHILD_ID = '019a0000-0000-7000-8000-000000000003';
const GONE_CHILD_ID = '019a0000-0000-7000-8000-000000000004';
const RESUME = `cd '/work/mlx node' && mlx agent --resume ${PARENT_ID}`;

function row(overrides: Partial<SessionRow>): SessionRow {
  return {
    id: PARENT_ID,
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
    prefillTps: 900,
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
      },
      { kind: 'subagent', ts, childId: GONE_CHILD_ID, type: 'general', description: 'Old helper', status: 'failed' },
      { kind: 'assistant', ts, text: 'Done. See `planner.rs`.' },
      { kind: 'turn-end', ts, stopReason: 'end_turn', elapsedMs: 4_000 },
    ],
    turns: [turn({}), turn({ promptId: 'p2', ttftMs: 80, decodeTps: 40 })],
    resumeCommand: RESUME,
  };
}

type Reply = { body: unknown } | { fail: ApiErrorCode; message: string };

/**
 * Serve `routes` keyed by `"METHOD /path"` (query string ignored). An array
 * answers the n-th call with its n-th element and repeats the last.
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
        const key = `${call.method} ${call.path.split('?')[0].replace(/^\/api/u, '')}`;
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
        return Promise.resolve(
          'fail' in reply ? failure(reply.fail, reply.message) : { ok: true as const, status: 200, body: reply.body },
        );
      },
      subscribe: () => () => {},
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

function actionIn(tr: HTMLElement, label: string): HTMLButtonElement {
  const button = tr.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);
  expect(button).toBeTruthy();
  return button!;
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

async function mountList(routes: Record<string, Reply | Reply[]>): Promise<void> {
  restoreApi = stubRoutes(routes, calls);
  mounted = await renderPage(createElement(MemoryRouter, null, createElement(Sessions)), (text) =>
    text.includes('Refactor the planner'),
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
  it('lists sessions with a link to each detail page and the live badge', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } });
    const link = rowOf(PARENT_ID).querySelector('a');
    expect(link?.getAttribute('href')).toBe(`/sessions/${PARENT_ID}`);
    expect(rowOf(LIVE_ID).textContent).toContain('Live');
    expect(rowOf(PARENT_ID).textContent).not.toContain('Live');
    // The list asks for the server's cap rather than its default of 100.
    expect(calls[0].path).toContain('limit=500');
  });

  it('disables rename and delete on a live session, with the reason as the hint', async () => {
    await mountList({ 'GET /sessions': { body: listBody() } });
    const live = rowOf(LIVE_ID);
    expect(actionIn(live, 'Rename').disabled).toBe(true);
    expect(actionIn(live, 'Delete').disabled).toBe(true);
    expect(actionIn(live, 'Delete').parentElement?.getAttribute('title')).toContain('running agent');
    // Copying the resume command stays available while live.
    expect(actionIn(live, 'Copy resume command').disabled).toBe(false);

    const idle = rowOf(PARENT_ID);
    expect(actionIn(idle, 'Rename').disabled).toBe(false);
    expect(actionIn(idle, 'Delete').disabled).toBe(false);
  });

  it('renames through PATCH {title} and refetches the list', async () => {
    const success = vi.spyOn(toast, 'success');
    await mountList({
      'GET /sessions': [{ body: listBody() }, { body: { ...listBody(), sessions: [row({ title: 'New name' })] } }],
      [`PATCH /sessions/${PARENT_ID}`]: { body: { id: PARENT_ID, title: 'New name' } },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Rename'));
    const input = dialog().querySelector<HTMLInputElement>('input[aria-label="Session title"]')!;
    expect(input.value).toBe('Refactor the planner');
    await typeInto(input, '  New name  ');
    await click(dialogButton('Save'));
    await settle();

    expect(writes()).toEqual([{ method: 'PATCH', path: `/api/sessions/${PARENT_ID}`, body: { title: 'New name' } }]);
    expect(success).toHaveBeenCalledWith('Session renamed', { description: 'New name' });
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(listGets()).toBe(2);
    expect(mounted!.text()).toContain('New name');
  });

  it('keeps a 409 inside the rename dialog instead of a toast', async () => {
    const failed = vi.spyOn(toast, 'error');
    const message = `Session ${PARENT_ID} is live (recent write); retry later`;
    await mountList({
      'GET /sessions': { body: listBody() },
      [`PATCH /sessions/${PARENT_ID}`]: { fail: 'E_CONFLICT', message },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Rename'));
    await typeInto(dialog().querySelector<HTMLInputElement>('input')!, 'Other');
    await click(dialogButton('Save'));
    await settle();

    expect(dialog().querySelector('[role="alert"]')?.textContent).toContain(message);
    expect(failed).not.toHaveBeenCalled();
    // The list is refetched so the row's live state catches up.
    expect(listGets()).toBe(2);
  });

  it('deletes through DELETE and names the subagent sessions removed with it', async () => {
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
    expect(dialog().textContent).toContain('and its 2 subagent sessions');
    await click(dialogButton('Delete'));
    await settle();

    expect(writes()).toEqual([{ method: 'DELETE', path: `/api/sessions/${PARENT_ID}` }]);
    expect(success).toHaveBeenCalledWith('Session deleted', {
      description: 'Refactor the planner and 2 subagent sessions',
    });
    expect(mounted!.container.querySelector(`tr[data-session-id="${PARENT_ID}"]`)).toBeNull();
  });

  it('shows a delete 409 in the dialog', async () => {
    const message = 'Subagent session is live';
    await mountList({
      'GET /sessions': { body: listBody() },
      [`DELETE /sessions/${PARENT_ID}`]: { fail: 'E_CONFLICT', message },
    });
    await click(actionIn(rowOf(PARENT_ID), 'Delete'));
    await click(dialogButton('Delete'));
    await settle();
    expect(dialog().querySelector('[role="alert"]')?.textContent).toContain(message);
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

  it('says how many sessions the cap left out', async () => {
    await mountList({ 'GET /sessions': { body: { ...listBody(), total: 640 } } });
    expect(mounted!.text()).toContain('Showing the newest 2 of 640 matching sessions');
  });
});

function detailPage(id: string) {
  return createElement(
    MemoryRouter,
    { initialEntries: [`/sessions/${id}`] },
    createElement(Routes, null, createElement(Route, { path: '/sessions/:id', element: createElement(SessionDetail) })),
  );
}

async function mountDetail(id: string, routes: Record<string, Reply | Reply[]>, until: string): Promise<string> {
  restoreApi = stubRoutes(routes, calls);
  mounted = await renderPage(detailPage(id), (text) => text.includes(until));
  return mounted.text();
}

describe('Session detail page', () => {
  let restoreCharts: (() => void) | undefined;
  beforeEach(() => {
    restoreCharts = stubChartMetrics();
  });
  afterEach(() => {
    restoreCharts?.();
  });

  it('renders the header, the turn chips and the resume button', async () => {
    const text = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'TTFT');
    expect(mounted!.container.querySelector('h1')?.textContent).toBe('Refactor the planner');
    expect(text).toContain('/work/mlx-node');
    expect(text).toContain('main');
    expect(text).toContain('Turns2');
    // Means over the two turns: TTFT (120+80)/2, decode (60+40)/2.
    expect(text).toContain('TTFT100ms avg');
    expect(text).toContain('Decode50.0tok/s avg');

    const copy = [...mounted!.container.querySelectorAll('button')].find((b) =>
      b.textContent?.includes('Copy resume command'),
    );
    await click(copy);
    await settle();
    expect(writeText).toHaveBeenCalledWith(RESUME);
  });

  it('links each existing subagent to its child session and leaves a missing one unlinked', async () => {
    await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Subagents');
    const childLinks = [...mounted!.container.querySelectorAll(`a[href="/sessions/${CHILD_ID}"]`)];
    // One in the Subagents card, one on the transcript row.
    expect(childLinks).toHaveLength(2);
    expect(mounted!.container.querySelector(`a[href="/sessions/${GONE_CHILD_ID}"]`)).toBeNull();
    expect(mounted!.text()).toContain('folder missing');
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
    const text = await mountDetail(CHILD_ID, { [`GET /sessions/${CHILD_ID}`]: { body: child } }, 'Subagent of');
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
    const text = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body: detailBody() } }, 'Transcript');
    // Markdown, not literal asterisks.
    expect(mounted!.container.querySelector('strong')?.textContent).toBe('refactor');
    expect(text).not.toContain('**refactor**');
    expect(mounted!.container.querySelector('code')?.textContent).toBe('planner.rs');
    // Collapsed by default.
    expect(text).toContain('Thinking');
    expect(text).not.toContain('secret reasoning');
    expect(text).not.toContain('"path":"planner.rs"');
    expect(text).toContain('Split the module');
    expect(text).toContain('Turn ended · end_turn · 4.0 s');

    const tool = [...mounted!.container.querySelectorAll('button')].find((b) => b.textContent?.includes('read_file'));
    await click(tool);
    expect(mounted!.text()).toContain('"path":"planner.rs"');
  });

  it('shows why the transcript could not be read', async () => {
    const body = { ...detailBody({}, []), transcriptError: 'EACCES: permission denied' };
    const text = await mountDetail(PARENT_ID, { [`GET /sessions/${PARENT_ID}`]: { body } }, 'Transcript');
    expect(text).toContain('Transcript could not be read: EACCES: permission denied');
  });

  it('renders the 404 message for an unknown id', async () => {
    const text = await mountDetail(
      PARENT_ID,
      { [`GET /sessions/${PARENT_ID}`]: { fail: 'E_NOT_FOUND', message: `Session ${PARENT_ID} not found` } },
      'not found',
    );
    expect(text).toContain(`Session ${PARENT_ID} not found`);
  });
});

describe('UI session types', () => {
  it('mirror the server contract exactly', () => {
    // Checked by `tsc`: a field added to or changed on either side fails the typecheck.
    expectTypeOf<SessionRow>().toEqualTypeOf<Server.SessionRow>();
    expectTypeOf<SessionDetailData>().toEqualTypeOf<Server.SessionDetail>();
    expectTypeOf<TranscriptEntry>().toEqualTypeOf<Server.TranscriptEntry>();
    expectTypeOf<TurnMetric>().toEqualTypeOf<Server.TurnMetric>();
    expectTypeOf<SessionListResponse>().toEqualTypeOf<Server.SessionListResponse>();
    expectTypeOf<SessionDetailResponse>().toEqualTypeOf<Server.SessionDetailResponse>();
    expectTypeOf<SessionRenameResponse>().toEqualTypeOf<Server.SessionRenameResponse>();
    expectTypeOf<SessionDeleteResponse>().toEqualTypeOf<Server.SessionDeleteResponse>();
  });
});
