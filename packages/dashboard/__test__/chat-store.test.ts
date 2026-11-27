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
import { runView } from './helpers/run-views.js';
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
  runs: [],
  live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
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
    api.emitChat('c1', { type: 'notice', seq: 1, notice: { code: 'key-printed', tone: 'danger', provider: 'exa' } });
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

describe('a question card that closes', () => {
  it('counts closes with no answer (Stop, time limit) apart from withdrawn requests; an answer counts nothing', () => {
    const question: Page.QuestionInteraction = {
      id: 'q1',
      kind: 'question',
      toolCallId: 'toolu_1',
      openedAt: 1,
      planMode: false,
      questions: [{ text: 'Which?', multi: false, options: [] }],
    };
    let state = applyConversationEvent(null, SYNC) as ConversationState;
    const step = (event: Server.ConversationEvent): ConversationState =>
      (state = applyConversationEvent(state, event) as ConversationState);
    step({ type: 'interaction', seq: 1, interaction: question });
    step({ type: 'interaction-closed', seq: 2, id: 'q1', reason: 'expired' });
    expect([state.unanswered, state.withdrawn, state.interactions]).toEqual([1, 0, []]);
    step({ type: 'interaction', seq: 3, interaction: { ...question, id: 'q2' } });
    step({ type: 'interaction-closed', seq: 4, id: 'q2', reason: 'cancelled' });
    step({ type: 'interaction', seq: 5, interaction: { ...question, id: 'q3' } });
    step({ type: 'interaction-closed', seq: 6, id: 'q3', reason: 'answered' });
    expect([state.unanswered, state.withdrawn]).toEqual([2, 0]);
  });
});

describe('an MCP card that closes', () => {
  // Mutation: count an expired MCP card as withdrawn → the withdrawn count moves.
  it('counts closes with no answer apart from questions and withdrawn requests; an answer counts nothing', () => {
    const mcp: Page.McpInteraction = {
      id: 'm1',
      kind: 'mcp-input',
      toolCallId: 'mcp-elicit-1',
      openedAt: 1,
      mode: 'form',
      server: 'github',
      message: 'Which repository?',
      form: { fields: [], leftOut: 0 },
    };
    let state = applyConversationEvent(null, SYNC) as ConversationState;
    expect(state.mcpUnanswered).toBe(0);
    const step = (event: Server.ConversationEvent): ConversationState =>
      (state = applyConversationEvent(state, event) as ConversationState);
    step({ type: 'interaction', seq: 1, interaction: mcp });
    step({ type: 'interaction-closed', seq: 2, id: 'm1', reason: 'expired' });
    expect([state.mcpUnanswered, state.unanswered, state.withdrawn, state.interactions]).toEqual([1, 0, 0, []]);
    step({ type: 'interaction', seq: 3, interaction: { ...mcp, id: 'm2' } });
    step({ type: 'interaction-closed', seq: 4, id: 'm2', reason: 'cancelled' });
    step({ type: 'interaction', seq: 5, interaction: { ...mcp, id: 'm3' } });
    step({ type: 'interaction-closed', seq: 6, id: 'm3', reason: 'answered' });
    expect([state.mcpUnanswered, state.unanswered, state.withdrawn]).toEqual([2, 0, 0]);
    expect((applyConversationEvent(state, { ...SYNC, seq: 7 }) as ConversationState).mcpUnanswered).toBe(2);
  });
});

describe('the end of a manual compaction', () => {
  it('is a new object per live end, kept through a state change, never from a sync', () => {
    const started = applyConversationEvent(null, SYNC) as ConversationState;
    expect(started.compactionEnd).toBeNull();
    const running = applyConversationEvent(started, {
      type: 'compaction',
      seq: 1,
      compaction: { phase: 'running', startedAt: 1 },
    }) as ConversationState;
    expect(running.compactionEnd).toBeNull();
    const ended = applyConversationEvent(running, {
      type: 'compaction',
      seq: 2,
      compaction: null,
      end: 'stopped',
    }) as ConversationState;
    expect(ended).toMatchObject({ compaction: null, compactionEnd: { outcome: 'stopped' } });
    const next = applyConversationEvent(ended, {
      type: 'compaction',
      seq: 3,
      compaction: { phase: 'running', startedAt: 2 },
    }) as ConversationState;
    expect(next.compactionEnd).toBe(ended.compactionEnd);
    const again = applyConversationEvent(next, {
      type: 'compaction',
      seq: 4,
      compaction: null,
      end: 'stopped',
    }) as ConversationState;
    expect(again.compactionEnd).toEqual({ outcome: 'stopped' });
    expect(again.compactionEnd).not.toBe(ended.compactionEnd);
    expect((applyConversationEvent(again, { ...SYNC, seq: 5 }) as ConversationState).compactionEnd).toBeNull();
  });
});

describe('a rewind end, the live retry and a blocked prompt in the stores', () => {
  // Mutation: read `rewindEnd` only from the `rewind` event → a page that resynced after the cut never leaves the
  // dialog.
  it('a sync carries the last rewind end; the same number keeps the object, a new one replaces it', () => {
    const end = { outcome: 'done' as const, mode: 'conversation' as const, restored: 0, entry: 1 };
    const synced = applyConversationEvent(null, { ...SYNC, rewindEnd: { n: 1, end } }) as ConversationState;
    expect(synced.rewindEnd).toEqual({ n: 1, end });
    const again = applyConversationEvent(synced, { ...SYNC, seq: 1, rewindEnd: { n: 1, end } }) as ConversationState;
    expect(again.rewindEnd).toBe(synced.rewindEnd);
    const live = applyConversationEvent(again, {
      type: 'rewind',
      seq: 2,
      n: 2,
      end: { outcome: 'stopped' },
    }) as ConversationState;
    expect(live.rewindEnd).toEqual({ n: 2, end: { outcome: 'stopped' } });
    expect((applyConversationEvent(null, SYNC) as ConversationState).rewindEnd).toBeNull();
  });

  it('the running answer’s retry: from the sync, replaced by each retry event, cleared with null', () => {
    const retry = {
      kind: 'notice' as const,
      ts: 5,
      code: 'retry' as const,
      cause: 'stalled' as const,
      attempts: 1,
      max: 3,
    };
    const synced = applyConversationEvent(null, { ...SYNC, retry }) as ConversationState;
    expect(synced.retry).toEqual(retry);
    const closed = { ...retry, endTs: 9, outcome: 'recovered' as const };
    const next = applyConversationEvent(synced, { type: 'retry', seq: 1, retry: closed }) as ConversationState;
    expect(next.retry).toEqual(closed);
    expect(next.entries).toBe(synced.entries);
    expect(
      (applyConversationEvent(next, { type: 'retry', seq: 2, retry: null }) as ConversationState).retry,
    ).toBeNull();
    expect((applyConversationEvent(null, SYNC) as ConversationState).retry).toBeNull();
  });

  it('a turn end with a blocked prompt keeps its text, live and from the sync; commands null clears the list', () => {
    const started = applyConversationEvent(null, { ...SYNC, commands: [] }) as ConversationState;
    const live = applyConversationEvent(started, {
      type: 'turn',
      seq: 1,
      turn: null,
      turnId: 't2',
      outcome: 'hook-stopped',
      blocked: 'deploy to prod',
    }) as ConversationState;
    expect(live.lastEnd).toEqual({ turnId: 't2', outcome: 'hook-stopped', blocked: 'deploy to prod' });
    const synced = applyConversationEvent(null, {
      ...SYNC,
      lastEnd: { turnId: 't2', outcome: 'hook-stopped', blocked: 'deploy to prod' },
    }) as ConversationState;
    expect(synced.lastEnd).toEqual(live.lastEnd);
    const listed = applyConversationEvent(live, {
      type: 'commands',
      seq: 2,
      commands: [{ name: 'commit', kind: 'skill', description: 'Write a commit', hint: null }],
    }) as ConversationState;
    expect(listed.commands).toHaveLength(1);
    expect(
      (applyConversationEvent(listed, { type: 'commands', seq: 3, commands: null }) as ConversationState).commands,
    ).toBeNull();
  });
});

describe('a recap being written', () => {
  it('the wait from a sync or its event; a new end object per live end; a sync keeps the wait and drops the end', () => {
    const started = applyConversationEvent(null, { ...SYNC, recap: { startedAt: 7 } }) as ConversationState;
    expect(started).toMatchObject({ recap: { startedAt: 7 }, recapEnd: null });
    const ended = applyConversationEvent(started, { type: 'recap', seq: 1, recap: null, end: 'none' });
    expect(ended).toMatchObject({ recap: null, recapEnd: { outcome: 'none' } });
    const writing = applyConversationEvent(ended as ConversationState, {
      type: 'recap',
      seq: 2,
      recap: { startedAt: 9 },
    }) as ConversationState;
    expect(writing.recap).toEqual({ startedAt: 9 });
    expect(writing.recapEnd).toBe((ended as ConversationState).recapEnd);
    const shown = applyConversationEvent(writing, { type: 'recap', seq: 3, recap: null, end: 'shown' });
    expect((shown as ConversationState).recapEnd).toEqual({ outcome: 'shown' });
    expect((applyConversationEvent(shown as ConversationState, { ...SYNC, seq: 4 }) as ConversationState).recap).toBe(
      null,
    );
  });
});

describe('a hook batch the agent waits on', () => {
  it('a sync carries it; a hook-run event sets and clears it; a gap asks for a fresh sync', () => {
    const run = { event: 'pre_tool_use', tool: 'run_terminal_command', count: 2, startedAt: 7 };
    const synced = applyConversationEvent(null, { ...SYNC, hookRun: run }) as ConversationState;
    expect(synced.hookRun).toEqual(run);
    expect((applyConversationEvent(null, SYNC) as ConversationState).hookRun).toBeNull();
    const cleared = applyConversationEvent(synced, { type: 'hook-run', seq: 1, hookRun: null }) as ConversationState;
    expect(cleared.hookRun).toBeNull();
    const stop = { event: 'stop', count: 1, startedAt: 9 };
    const set = applyConversationEvent(cleared, { type: 'hook-run', seq: 2, hookRun: stop }) as ConversationState;
    expect(set.hookRun).toEqual(stop);
    expect(set.entries).toBe(cleared.entries);
    expect(applyConversationEvent(set, { type: 'hook-run', seq: 4, hookRun: null })).toBe('resync');
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

  it('a sync and a runs event each replace the live runs; a gap asks for a fresh sync', () => {
    const live = runView('wf_1');
    const first = applyConversationEvent(null, { ...WORK, runs: [live] }) as ConversationState;
    expect(first.runs).toEqual([live]);
    const stopping = applyConversationEvent(first, {
      type: 'runs',
      seq: 1,
      runs: [runView('wf_1', { stopping: true })],
    }) as ConversationState;
    expect(stopping.runs).toEqual([runView('wf_1', { stopping: true })]);
    expect(first.runs).toEqual([live]);
    const gone = applyConversationEvent(stopping, { type: 'runs', seq: 2, runs: [] }) as ConversationState;
    expect(gone.runs).toEqual([]);
    expect(applyConversationEvent(gone, { type: 'runs', seq: 4, runs: [live] })).toBe('resync');
    const again = applyConversationEvent(gone, { ...WORK, seq: 5, runs: [live] }) as ConversationState;
    expect(again.runs).toEqual([live]);
  });

  // Mutation: leave `live` as it was on an `activity` event → the tick does not show.
  it('a sync carries the live facts and an activity event replaces them; a gap asks for a fresh sync', () => {
    const tick = (toolCalls: number): Page.HelperProgress => ({
      childId: 'child-1',
      contextWindow: null,
      now: null,
      turns: 1,
      toolCalls,
      tokens: 9_000,
      contextPct: 27,
      errors: 0,
      durationMs: 2_000,
      at: 5,
    });
    const first = applyConversationEvent(null, {
      ...WORK,
      live: { helpers: [tick(4)], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
    }) as ConversationState;
    expect(first.live).toEqual({
      helpers: [tick(4)],
      agents: [],
      tasks: [],
      schedules: [],
      reports: [],
      stoppedByYou: [],
    });
    const next = applyConversationEvent(first, {
      type: 'activity',
      seq: 1,
      live: { helpers: [tick(7)], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
    }) as ConversationState;
    expect(next.live).toEqual({
      helpers: [tick(7)],
      agents: [],
      tasks: [],
      schedules: [],
      reports: [],
      stoppedByYou: [],
    });
    expect(first.live).toEqual({
      helpers: [tick(4)],
      agents: [],
      tasks: [],
      schedules: [],
      reports: [],
      stoppedByYou: [],
    });
    const gone = applyConversationEvent(next, {
      type: 'activity',
      seq: 2,
      live: { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
    }) as ConversationState;
    expect(gone.live).toEqual({ helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] });
    expect(
      applyConversationEvent(gone, {
        type: 'activity',
        seq: 4,
        live: { helpers: [tick(9)], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] },
      }),
    ).toBe('resync');
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

  it('the mode picked on the New Work screen follows each state event', async () => {
    const api = stub();
    const store = createChatHostStore();
    const off = store.subscribe(() => {});
    await until(() => store.getSnapshot().state !== null);
    expect(store.getSnapshot().state?.permissionDraft).toBeNull();
    const draft = { draft: 'd1', mode: 'bypass' as const };
    api.emitChatHost({ type: 'state', state: { ...READY_CHAT_HOST, permissionDraft: draft } });
    await until(() => store.getSnapshot().state?.permissionDraft !== null);
    expect(store.getSnapshot().state?.permissionDraft).toEqual(draft);
    api.emitChatHost({ type: 'state', state: READY_CHAT_HOST });
    await until(() => store.getSnapshot().state?.permissionDraft === null);
    off();
  });
});

describe('Chat route failures', () => {
  it('keep the code, the fixed message and the reason', async () => {
    const api = stubApi({
      '/chat/conversations': stubError('E_BAD_REQUEST', 'No such command here.', 'slash-text'),
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
    expectTypeOf<Page.ThinkingLevel>().toEqualTypeOf<Server.ThinkingLevel>();
    expectTypeOf<Page.ModelThinking>().toEqualTypeOf<Server.ModelThinking>();
    expectTypeOf<Page.ConversationRow>().toEqualTypeOf<Server.ConversationRow>();
    expectTypeOf<Page.TurnState>().toEqualTypeOf<Server.TurnState>();
    expectTypeOf<Page.Interaction>().toEqualTypeOf<Server.Interaction>();
    expectTypeOf<Page.ConversationNotice>().toEqualTypeOf<Server.ConversationNotice>();
    expectTypeOf<Page.TranscriptOp>().toEqualTypeOf<Server.TranscriptOp>();
    expectTypeOf<Page.TurnEnd>().toEqualTypeOf<Server.TurnEnd>();
    expectTypeOf<Page.RunView>().toEqualTypeOf<Server.RunView>();
    expectTypeOf<Page.HelperProgress>().toEqualTypeOf<Server.HelperProgress>();
    expectTypeOf<Page.LiveActivity>().toEqualTypeOf<Server.LiveActivity>();
    expectTypeOf<Page.CompactionState>().toEqualTypeOf<Server.CompactionState>();
    expectTypeOf<Page.CompactOutcome>().toEqualTypeOf<Server.CompactOutcome>();
    expectTypeOf<Page.RecapState>().toEqualTypeOf<Server.RecapState>();
    expectTypeOf<Page.HookRunState>().toEqualTypeOf<Server.HookRunState>();
    expectTypeOf<Page.RecapEnd>().toEqualTypeOf<Server.RecapEnd>();
    expectTypeOf<Page.ConversationContext>().toEqualTypeOf<Server.ConversationContext>();
    expectTypeOf<Page.ConversationEvent>().toEqualTypeOf<Server.ConversationEvent>();
    expectTypeOf<Page.AgentWatchAnswer>().toEqualTypeOf<Server.AgentWatchAnswer>();
    expectTypeOf<Page.AgentModeState>().toEqualTypeOf<Server.AgentModeState>();
    expectTypeOf<Page.ChatHostState>().toEqualTypeOf<Server.ChatHostState>();
    expectTypeOf<Page.ChatHostEvent>().toEqualTypeOf<Server.ChatHostEvent>();
    expectTypeOf<PageProviders.ProviderState>().toEqualTypeOf<ServerProviders.ProviderState>();
    expectTypeOf<PageProviders.ProvidersResponse>().toEqualTypeOf<ServerProviders.ProvidersResponse>();
    expectTypeOf<PageProviders.ProviderSelectionInput>().toEqualTypeOf<ServerProviders.ProviderSelectionInput>();
  });
});
