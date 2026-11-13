/**
 * @vitest-environment happy-dom
 */

/**
 * The page's Chat streams over a real port: a conversation store opens on its first reader, applies events in `seq`
 * order, re-opens for a fresh `sync` after a gap and after a replacement port, closes while the window is hidden and
 * on `gone`; the host store stays open while hidden and counts list changes. A Chat route failure keeps its reason.
 */

import { afterEach, describe, expect, expectTypeOf, it } from 'vite-plus/test';

import type * as ServerProviders from '../src/chat/providers/types.js';
import type * as Server from '../src/chat/types.js';
import { DashboardApiError, mutate } from '../ui/src/lib/api.js';
import {
  applyConversationEvent,
  createChatHostStore,
  createConversationStore,
  type ConversationState,
} from '../ui/src/lib/chat-store.js';
import type * as Page from '../ui/src/lib/chat-types.js';
import type { ConversationInfo } from '../ui/src/lib/chat-types.js';
import type * as PageProviders from '../ui/src/lib/providers-types.js';
import { READY_CHAT_HOST, stubApi, stubError, type ChatSync, type StubbedApi } from './render.js';

const INFO: ConversationInfo = {
  id: 'c1',
  mode: 'chat',
  title: 'First',
  cwd: null,
  modelId: 'qwen3.5-4b',
  createdAt: 1,
  updatedAt: 1,
  hot: true,
  permissionMode: 'ask',
};

const SYNC: ChatSync = {
  type: 'sync',
  seq: 0,
  conversation: INFO,
  entries: [{ kind: 'user', ts: 1, text: 'hello', promptIndex: 0 }],
  turns: [],
  turn: null,
  interactions: [],
  notices: [],
  transcriptError: null,
  permissionMarks: [],
};

let stops: Array<() => void> = [];

function stub(): StubbedApi {
  const api = stubApi({}, { chat: { conversations: { c1: SYNC } } });
  stops.push(api);
  return api;
}

async function until(check: () => boolean, ms = 2_000): Promise<void> {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error('condition not reached');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

function setVisibility(state: 'visible' | 'hidden'): void {
  Object.defineProperty(document, 'visibilityState', { value: state, configurable: true });
  document.dispatchEvent(new Event('visibilitychange'));
}

afterEach(() => {
  setVisibility('visible');
  for (const stop of stops.reverse()) stop();
  stops = [];
});

describe('conversation store', () => {
  it('opens on the first reader, applies ops in order, and closes with the last reader', async () => {
    const api = stub();
    const store = createConversationStore('c1');
    expect(api.chatWatchers('c1')).toBe(0);
    let renders = 0;
    const off = store.subscribe(() => (renders += 1));
    await until(() => store.getSnapshot().state !== null);
    expect(api.chatWatchers('c1')).toBe(1);
    api.emitChat('c1', {
      type: 'ops',
      seq: 1,
      ops: [{ op: 'push', entry: { kind: 'assistant', ts: 2, text: 'Hi' } }],
    });
    api.emitChat('c1', { type: 'ops', seq: 2, ops: [{ op: 'text', index: 1, append: ' there' }] });
    await until(() => store.getSnapshot().state?.seq === 2);
    expect(store.getSnapshot().state!.entries[1]).toEqual({ kind: 'assistant', ts: 2, text: 'Hi there' });
    expect(renders).toBeGreaterThanOrEqual(2);
    off();
    await until(() => api.chatWatchers('c1') === 0);
  });

  it('asks for a fresh sync after a gap', async () => {
    const api = stub();
    const store = createConversationStore('c1');
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state !== null);
    api.emitChat('c1', { type: 'notice', seq: 1, notice: { code: 'key-printed', provider: 'exa' } });
    await until(() => store.getSnapshot().state?.seq === 1);
    api.emitChat('c1', { type: 'turn', seq: 5, turn: null, outcome: 'completed' });
    await until(() => store.getSnapshot().state?.seq === 0);
    await until(() => api.chatWatchers('c1') === 1);
    off();
  });

  it('re-attaches on a replacement port after the old one dropped', async () => {
    const first = stub();
    const store = createConversationStore('c1');
    const off = store.subscribe(() => {});
    await until(() => first.chatWatchers('c1') === 1);
    first.dropPort();
    first();
    stops = stops.filter((s) => s !== first);
    const second = stub();
    await until(() => second.chatWatchers('c1') === 1);
    expect(store.getSnapshot().state?.conversation.id).toBe('c1');
    off();
  });

  it('closes while the window is hidden and re-opens when it shows', async () => {
    const api = stub();
    const store = createConversationStore('c1');
    const off = store.subscribe(() => {});
    await until(() => api.chatWatchers('c1') === 1);
    setVisibility('hidden');
    await until(() => api.chatWatchers('c1') === 0);
    setVisibility('visible');
    await until(() => api.chatWatchers('c1') === 1);
    off();
  });

  it('reads gone before a sync for an unknown id, and closes on gone', async () => {
    const api = stub();
    const unknown = createConversationStore('missing');
    const offUnknown = unknown.subscribe(() => {});
    await until(() => unknown.getSnapshot().gone === 'unknown');
    expect(unknown.getSnapshot().state).toBeNull();
    await until(() => api.chatWatchers('missing') === 0);
    offUnknown();

    const store = createConversationStore('c1');
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state !== null);
    api.emitChat('c1', { type: 'gone', seq: 1, reason: 'deleted' });
    await until(() => store.getSnapshot().gone === 'deleted');
    await until(() => api.chatWatchers('c1') === 0);
    off();
  });
});

describe('a turn end in the sync', () => {
  it('sets the same state as the live event of that turn, and a repeat keeps it', () => {
    const failure = { code: 'agent-crashed' as const, message: 'The agent stopped.' };
    const started = applyConversationEvent(null, SYNC) as ConversationState;
    const live = applyConversationEvent(started, {
      type: 'turn',
      seq: 1,
      turn: null,
      turnId: 't1',
      outcome: 'tool-refused',
      failure,
    }) as ConversationState;
    const synced = applyConversationEvent(null, {
      ...SYNC,
      seq: 1,
      lastEnd: { turnId: 't1', outcome: 'tool-refused', failure },
    }) as ConversationState;
    expect(synced).toEqual(live);
    expect(synced.lastEnd).toEqual({ turnId: 't1', outcome: 'tool-refused', failure });

    const again = applyConversationEvent(live, {
      ...SYNC,
      seq: 2,
      lastEnd: { turnId: 't1', outcome: 'tool-refused', failure },
    });
    expect((again as ConversationState).lastEnd).toBe(live.lastEnd);
    const next = applyConversationEvent(live, { ...SYNC, seq: 2, lastEnd: { turnId: 't2', outcome: 'completed' } });
    expect((next as ConversationState).lastEnd).toEqual({ turnId: 't2', outcome: 'completed' });
    expect((applyConversationEvent(live, { ...SYNC, seq: 2 }) as ConversationState).lastEnd).toBeNull();
  });
});

describe('host store', () => {
  it('stays open while hidden, counts list changes, and re-attaches on a new port', async () => {
    const first = stub();
    const store = createChatHostStore();
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state !== null);
    expect(store.getSnapshot().state).toEqual(READY_CHAT_HOST);
    setVisibility('hidden');
    first.emitChatHost({ type: 'conversations-changed' });
    await until(() => store.getSnapshot().conversationsVersion === 1);
    expect(first.chatWatchers()).toBe(1);
    first.dropPort();
    first();
    stops = stops.filter((s) => s !== first);
    const second = stub();
    await until(() => second.chatWatchers() === 1);
    off();
    await until(() => second.chatWatchers() === 0);
  });

  it('a state event carries the answers of each resident model to the page', async () => {
    const api = stub();
    const store = createChatHostStore();
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state !== null);
    expect(store.getSnapshot().state?.answers).toEqual([]);
    const answers = [{ modelId: 'qwen3.5-4b', inFlight: 1, maxConcurrent: 1, queued: 1, lastStart: 1_000 }];
    api.emitChatHost({ type: 'state', state: { ...READY_CHAT_HOST, answers } });
    await until(() => store.getSnapshot().state?.answers.length === 1);
    expect(store.getSnapshot().state?.answers).toEqual(answers);
    off();
  });
});

describe('permission modes in the stores', () => {
  const UP: Page.PermissionMark = { kind: 'mode', at: 5, mode: 'bypass', from: 'ask', by: 'user' };
  const ALLOWED: Page.PermissionMark = {
    kind: 'allowed',
    at: 6,
    mode: 'bypass',
    toolCallId: 't1',
    access: 'mcp',
    card: null,
  };
  const RESTART: Page.PermissionMark = { kind: 'mode', at: 9, mode: 'ask', from: 'bypass', by: 'restart' };
  const WORK: ChatSync = { ...SYNC, conversation: { ...INFO, mode: 'work', cwd: '/tmp/w', permissionMode: 'bypass' } };

  it('a sync replaces the marks and a permission-mark event appends one; a gap asks for a fresh sync', () => {
    const first = applyConversationEvent(null, { ...WORK, permissionMarks: [UP] }) as ConversationState;
    expect(first.permissionMarks).toEqual([UP]);
    const next = applyConversationEvent(first, { type: 'permission-mark', seq: 1, mark: ALLOWED }) as ConversationState;
    expect(next.permissionMarks).toEqual([UP, ALLOWED]);
    expect(first.permissionMarks).toEqual([UP]);
    expect(applyConversationEvent(next, { type: 'permission-mark', seq: 3, mark: RESTART })).toBe('resync');
    const again = applyConversationEvent(next, { ...WORK, seq: 4, permissionMarks: [UP] }) as ConversationState;
    expect(again.permissionMarks).toEqual([UP]);
  });

  it('after a reconnect to a restarted app the new sync says Ask, with its restart line', async () => {
    const first = stubApi({}, { chat: { conversations: { c1: { ...WORK, permissionMarks: [UP] } } } });
    stops.push(first);
    const store = createConversationStore('c1');
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state?.conversation.permissionMode === 'bypass');
    first.emitChat('c1', { type: 'permission-mark', seq: 1, mark: ALLOWED });
    await until(() => store.getSnapshot().state?.permissionMarks.length === 2);
    first.dropPort();
    first();
    stops = stops.filter((s) => s !== first);
    const restarted: ChatSync = {
      ...WORK,
      conversation: { ...WORK.conversation, permissionMode: 'ask' },
      permissionMarks: [UP, ALLOWED, RESTART],
    };
    const second = stubApi({}, { chat: { conversations: { c1: restarted } } });
    stops.push(second);
    await until(() => store.getSnapshot().state?.conversation.permissionMode === 'ask');
    expect(store.getSnapshot().state?.permissionMarks).toEqual([UP, ALLOWED, RESTART]);
    off();
  });

  it('the host list of conversations not in Ask follows each state event', async () => {
    const api = stub();
    const store = createChatHostStore();
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state !== null);
    expect(store.getSnapshot().state?.permissionRaised).toEqual([]);
    const raised = [{ id: 'c1', mode: 'bypass' as const }];
    api.emitChatHost({ type: 'state', state: { ...READY_CHAT_HOST, permissionRaised: raised } });
    await until(() => store.getSnapshot().state?.permissionRaised.length === 1);
    expect(store.getSnapshot().state?.permissionRaised).toEqual(raised);
    api.emitChatHost({ type: 'state', state: READY_CHAT_HOST });
    await until(() => store.getSnapshot().state?.permissionRaised.length === 0);
    off();
  });
});

describe('Chat route failures', () => {
  it('keep the code, the fixed message and the reason', async () => {
    const api = stubApi({
      '/chat/conversations': stubError('E_BAD_REQUEST', 'Start the message with a word, not /.', 'slash-text'),
    });
    stops.push(api);
    const error = await mutate('POST', '/chat/conversations', { mode: 'chat', text: '/yolo' }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(DashboardApiError);
    expect(error).toMatchObject({ status: 400, code: 'E_BAD_REQUEST', reason: 'slash-text' });
  });
});

describe('the page copies of the Chat types', () => {
  it('mirror the server contract exactly', () => {
    // Checked by `tsc`: a field added to or changed on either side fails the typecheck.
    expectTypeOf<Page.ChatMode>().toEqualTypeOf<Server.ChatMode>();
    expectTypeOf<Page.ChatFailureCode>().toEqualTypeOf<Server.ChatFailureCode>();
    expectTypeOf<Page.ChatFailure>().toEqualTypeOf<Server.ChatFailure>();
    expectTypeOf<Page.TurnOutcome>().toEqualTypeOf<Server.TurnOutcome>();
    expectTypeOf<Page.ConversationInfo>().toEqualTypeOf<Server.ConversationInfo>();
    expectTypeOf<Page.ConversationRow>().toEqualTypeOf<Server.ConversationRow>();
    expectTypeOf<Page.TurnState>().toEqualTypeOf<Server.TurnState>();
    expectTypeOf<Page.Interaction>().toEqualTypeOf<Server.Interaction>();
    expectTypeOf<Page.ConversationNotice>().toEqualTypeOf<Server.ConversationNotice>();
    expectTypeOf<Page.TranscriptOp>().toEqualTypeOf<Server.TranscriptOp>();
    expectTypeOf<Page.TurnEnd>().toEqualTypeOf<Server.TurnEnd>();
    expectTypeOf<Page.ConversationEvent>().toEqualTypeOf<Server.ConversationEvent>();
    expectTypeOf<Page.AgentModeState>().toEqualTypeOf<Server.AgentModeState>();
    expectTypeOf<Page.ChatHostState>().toEqualTypeOf<Server.ChatHostState>();
    expectTypeOf<Page.ChatHostEvent>().toEqualTypeOf<Server.ChatHostEvent>();
    expectTypeOf<PageProviders.ProviderState>().toEqualTypeOf<ServerProviders.ProviderState>();
    expectTypeOf<PageProviders.ProvidersResponse>().toEqualTypeOf<ServerProviders.ProvidersResponse>();
  });
});
