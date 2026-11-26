/** @vitest-environment happy-dom */

import { act, createElement, type ReactElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import type { ConversationContext, ConversationInfo } from '../ui/src/lib/chat-types.js';
import Chat from '../ui/src/pages/chat.js';
import { forgetPlace } from '../ui/src/pages/chat/drafts.js';
import { historyActions } from '../ui/src/pages/chat/history-actions.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import { type AppCommand, usePageCommands } from '../ui/src/shell/commands.js';
import { answer, end, metric, user } from './helpers/chat-thread-fixtures.js';
import {
  byRole,
  type ChatSync,
  READY_CHAT_HOST,
  renderPage,
  type RenderedPage,
  sequence,
  stubApi,
  type StubbedApi,
  waitFor,
} from './render.js';

const NOW = Date.now();
const ID = 'p-ctx';
const CONTEXT_PATH = `/chat/conversations/${ID}/context`;
const RING = 'Context: 16K of 32K tokens (50%)';

let page: RenderedPage | undefined;
let api: StubbedApi | undefined;
let calls: ApiCall[] = [];
let commands: readonly AppCommand[] = [];

function CommandSpy(): null {
  commands = usePageCommands();
  return null;
}

function app(initial: string): ReactElement {
  return createElement(
    MemoryRouter,
    { initialEntries: [initial] },
    createElement(CommandSpy),
    createElement(
      Routes,
      null,
      createElement(Route, { path: '/chat', element: createElement(Chat) }),
      createElement(Route, { path: '*', element: createElement('p', null, 'Elsewhere') }),
    ),
  );
}

function info(over: Partial<ConversationInfo> = {}): ConversationInfo {
  return {
    id: ID,
    mode: 'chat',
    title: 'Context probe',
    cwd: null,
    modelId: 'qwen3.5-4b',
    createdAt: NOW - 60_000,
    updatedAt: NOW - 60_000,
    hot: true,
    permissionMode: 'ask',
    ...over,
  };
}

function sync(over: Partial<ChatSync> = {}): ChatSync {
  return {
    type: 'sync',
    seq: 0,
    conversation: info(),
    entries: [user('Hello'), answer('Hi there'), end()],
    turns: [metric({ inputTokens: 1_200, cachedTokens: 800, outputTokens: 512, modelCalls: 2 })],
    turn: null,
    interactions: [],
    notices: [],
    transcriptError: null,
    permissionMarks: [],
    runs: [],
    live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
    contextUsed: 16_384,
    ...over,
  };
}

const HOT: ConversationContext = {
  used: 16_384,
  parts: {
    systemTokens: 432,
    toolCount: 2,
    toolTokens: 261,
    messageTokens: 15_000,
    autoCompactPercent: 85,
    compactions: 0,
  },
};

async function open(routes: Record<string, unknown>): Promise<RenderedPage> {
  calls = [];
  api = stubApi(
    { '/chat/conversations': { conversations: [] }, ...routes },
    {
      onCall: (call) => calls.push(call),
      chat: { host: READY_CHAT_HOST, conversations: { [ID]: sync() } },
    },
  );
  page = await renderPage(app(`/chat?c=${ID}`), (text) => text.includes('Hi there'));
  const want = JSON.stringify(READY_CHAT_HOST);
  await waitFor(() => JSON.stringify(chatHostSnapshot().state) === want, 2_000);
  return page;
}

function pathOf(call: ApiCall): string {
  const bare = call.path.split('?')[0]!;
  return bare.startsWith('/api') ? bare.slice('/api'.length) : bare;
}

const reads = (): number => calls.filter((c) => c.method === 'GET' && pathOf(c) === CONTEXT_PATH).length;
const dialog = (): HTMLElement | null => byRole(document.body, 'dialog', CHAT.history.context);
const ring = (): HTMLElement => byRole(page!.container, 'button', RING)!;

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  page?.unmount();
  api?.();
  page = undefined;
  api = undefined;
  forgetPlace(ID);
  vi.restoreAllMocks();
});

describe('Chat page: the Context panel', () => {
  it('the ring opens it: the parts come from the context route, the totals from the stored turns', async () => {
    await open({ [CONTEXT_PATH]: HOT });
    expect(reads()).toBe(0);
    act(() => ring().click());
    await waitFor(() => dialog() !== null);
    await waitFor(() => (dialog()!.textContent ?? '').includes(CHAT.history.partSystem));
    expect(reads()).toBe(1);
    const text = dialog()!.textContent ?? '';
    expect(text).toContain(fill(CHAT.history.contextHead, { used: '16K', window: '32K', percent: '50%' }));
    expect(text).toContain(fill(CHAT.history.partTools, { n: 2 }));
    expect(text).toContain(fill(CHAT.history.usageReadValue, { n: '1,200', cached: '800' }));
    expect(byRole(dialog()!, 'link', CHAT.inspect)!.getAttribute('href')).toBe(`/sessions/${ID}?tab=activity`);
  });

  it('cold: the line that asks for a message; Compact now closes it and posts compact', async () => {
    await open({ [CONTEXT_PATH]: { used: 16_384, parts: null }, [`POST /chat/conversations/${ID}/compact`]: {} });
    act(() => ring().click());
    await waitFor(() => (dialog()?.textContent ?? '').includes(CHAT.history.contextCold));
    act(() => byRole(dialog()!, 'button', CHAT.history.compact)!.click());
    await waitFor(() => calls.some((c) => c.method === 'POST' && pathOf(c) === `/chat/conversations/${ID}/compact`));
    await waitFor(() => dialog() === null);
  });

  it('the palette and historyActions.showContext open it; an answer’s end while open reads the parts again', async () => {
    await open({ [CONTEXT_PATH]: sequence(HOT, { ...HOT, used: 20_000 }) });
    const command = commands.find((c) => c.id === 'chat:context');
    expect(command?.label).toBe(CHAT.commands.context);
    act(() => void command!.run());
    await waitFor(() => dialog() !== null);
    await waitFor(() => reads() === 1);
    act(() => api!.emitChat(ID, { type: 'ops', seq: 1, ops: [{ op: 'metric', metric: metric({ promptId: 'q2' }) }] }));
    await waitFor(() => reads() === 2);
    act(() => {
      dialog()!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
    });
    await waitFor(() => dialog() === null);
    act(() => historyActions.showContext('another-conversation'));
    expect(dialog()).toBeNull();
    act(() => historyActions.showContext(ID));
    await waitFor(() => dialog() !== null);
  });
});
