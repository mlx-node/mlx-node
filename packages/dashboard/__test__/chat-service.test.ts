/**
 * The Chat service on the scripted fake agent: a conversation is created, streamed, cancelled, survives a crash, is
 * reopened cold after a Control Panel restart, renamed and deleted. Every page-bound event is folded through the
 * page's own reducer.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';

import { describe, expect, it, onTestFinished } from 'vite-plus/test';

import { AgentSessionsReader } from '../src/agent-sessions/reader.js';
import { ApiError } from '../src/api/errors.js';
import { CLAUDE_CODE_OFF_ENV } from '../src/chat/agent/env.js';
import { readPdfCommand, readPdfScript } from '../src/chat/agent/pdf-tool.js';
import { promptParams } from '../src/chat/agent/profiles.js';
import { attachmentView, type AttachmentInput, pdfToolId, readPdfNote } from '../src/chat/attachment-rule.js';
import { readWorkIndexSync } from '../src/chat/index-file.js';
import type { ChatPaths } from '../src/chat/paths.js';
import { MemorySecretStore, type SecretStore } from '../src/chat/providers/secrets.js';
import { ProvidersService, WORK_PATH_ASK_GAP_MS } from '../src/chat/providers/service.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ChatService } from '../src/chat/service.js';
import type {
  ChatHostEvent,
  ChatHostState,
  ConversationEvent,
  CwdProblem,
  Interaction,
  PermissionMode,
} from '../src/chat/types.js';
import { CWD_PROBLEM_MESSAGES } from '../src/chat/work-folder.js';
import { readScratchRecordsSync, stampName } from '../src/chat/work-scratch.js';
import type { RuntimeLive } from '../src/live.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import {
  ageTree,
  endpointModel,
  fakeService,
  INFERENCE_TOKEN,
  memoryProviders,
  NO_NETWORK,
  until,
  type FakeService,
  type Json,
} from './helpers/chat-fake.js';
import { UNLOAD_PATH } from './helpers/fake-engine.js';
import { live, liveModel, liveTurn, process as liveProcess } from './helpers/observability-fixtures.js';
import { runView } from './helpers/run-views.js';

const BASIC = [
  { thought: 'The user greets me.', chunk: 5 },
  { text: 'Hello! How can I help?', chunk: 7 },
  { turnCompleted: { inputTokens: 400, outputTokens: 30 } },
  { end: { stopReason: 'end_turn' } },
];

function scripted(scenario: Json): void {
  scenario.prompts = [
    {
      match: 'slow',
      steps: [{ text: 'Thinking about it', chunk: 3 }, { sleepMs: 5000 }, { end: { stopReason: 'end_turn' } }],
    },
    { match: 'crash', steps: [{ text: 'Partial answer', chunk: 4 }, { sleepMs: 30 }, { exit: 9 }] },
    { match: '', steps: BASIC },
  ];
}

/** A page attached to one conversation: every event, and the state the page's reducer folds them into. */
class Page {
  readonly events: ConversationEvent[] = [];
  state: ConversationState | null = null;
  resyncs = 0;
  readonly detach: () => void;

  constructor(service: ChatService, id: string) {
    this.detach = service.watchConversation(id, (event) => {
      this.events.push(event);
      const next = applyConversationEvent(this.state, event);
      if (next === 'resync') this.resyncs += 1;
      else this.state = next;
    });
  }

  ends(): ConversationEvent[] {
    return this.events.filter((e) => e.type === 'turn' && e.turn === null);
  }

  lastOutcome(): string | undefined {
    const end = this.ends().at(-1);
    return end?.type === 'turn' ? end.outcome : undefined;
  }

  texts(kind: 'user' | 'assistant'): string[] {
    return (this.state?.entries ?? []).flatMap((e) => (e.kind === kind ? [e.text] : []));
  }
}

async function withService(run: (fake: FakeService) => Promise<void>, mutate = scripted): Promise<void> {
  const fake = fakeService('chat-basic', { mutate });
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

/** The failure a route would answer for this call. */
async function apiError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return routeError('chat', error);
  }
  throw new Error('expected a failure');
}

describe('Chat service — a conversation end to end on the fake agent', () => {
  it('creates, streams, cancels, survives a crash, reopens cold, renames and deletes', async () => {
    await withService(async (fake) => {
      const svc = fake.service;
      const created = await svc.create({ mode: 'chat', text: 'hello there' });
      const id = created.conversation.id;
      expect(created.conversation).toMatchObject({ mode: 'chat', cwd: null, modelId: 'qwen3.5-4b', hot: true });
      expect(typeof created.turnId).toBe('string');

      const page = new Page(svc, id);
      await until(() => page.ends().length === 1, 5_000);
      expect(page.resyncs).toBe(0);
      expect(page.events[0]!.type).toBe('sync');
      expect(page.lastOutcome()).toBe('completed');
      expect(page.texts('user')).toEqual(['hello there']);
      expect(page.texts('assistant')).toEqual(['Hello! How can I help?']);
      expect(page.state!.entries.some((e) => e.kind === 'model')).toBe(false);
      expect(page.state!.turns).toHaveLength(1);
      await until(() => page.state!.conversation.title === 'hello there');

      // Cancel: the turn ends as the user's stop.
      await svc.prompt(id, 'slow please');
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      expect(await svc.cancel(id)).toEqual({ cancelled: true });
      await until(() => page.ends().length === 2, 5_000);
      expect(page.lastOutcome()).toBe('cancelled');

      // Crash: the turn ends `agent-crashed`, the page gets a fresh sync from disk, and the next prompt loads it.
      await svc.prompt(id, 'crash now');
      await until(() => page.ends().length === 3, 5_000);
      expect(page.lastOutcome()).toBe('agent-crashed');
      await until(() => page.state!.conversation.hot === false && page.texts('user').includes('crash now'), 5_000);
      await svc.prompt(id, 'hello again');
      await until(() => page.ends().length === 4, 5_000);
      expect(page.lastOutcome()).toBe('completed');
      expect(page.resyncs).toBe(0);
      const load = fake.sent().find((m) => m.method === 'session/load') as Json;
      expect(((load.params as Json)._meta as Json).noReplay).toBe(true);
      for (const m of fake.sent().filter((m) => m.method === 'session/prompt')) {
        for (const block of (m.params as Json).prompt as Json[])
          expect(Object.keys(block).sort()).toEqual(['text', 'type']);
      }

      // Rename while hot goes through the agent.
      const renamed = await svc.patch(id, { title: 'Renamed' });
      expect(renamed.title).toBe('Renamed');
      expect(fake.sent().some((m) => m.method === '_x.ai/session/rename')).toBe(true);
      page.detach();

      // A Control Panel restart: the conversation is cold, read from disk, and continues with a load.
      await svc.close();
      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      expect(cold.state!.conversation).toMatchObject({ id, mode: 'chat', hot: false });
      expect(cold.texts('user')).toEqual(['hello there', 'slow please', 'crash now', 'hello again']);
      expect(cold.state!.entries.some((e) => e.kind === 'model')).toBe(false);
      const loadsBefore = fake.sent().filter((m) => m.method === 'session/load').length;
      await svc2.prompt(id, 'once more');
      await until(() => cold.ends().length === 1, 5_000);
      expect(cold.lastOutcome()).toBe('completed');
      expect(fake.sent().filter((m) => m.method === 'session/load').length).toBe(loadsBefore + 1);
      const firstLoaded = cold.state!.entries.findIndex((e) => e.kind === 'user' && e.text === 'once more');
      expect(cold.state!.entries.slice(0, firstLoaded).some((e) => e.kind === 'model')).toBe(false);

      // Delete while hot goes through the agent; the page is told, the folder and the prompt history are gone.
      await svc2.delete(id);
      await until(() => cold.state!.gone === 'deleted');
      expect(fake.sent().some((m) => m.method === '_x.ai/session/delete')).toBe(true);
      expect(fake.reader.detail(id)).toBeNull();
      const history = readdirSync(fake.paths.chatSessions).filter((d) =>
        existsSync(join(fake.paths.chatSessions, d, 'prompt_history.jsonl')),
      );
      expect(history).toEqual([]);
      expect((await apiError(svc2.prompt(id, 'are you there'))).code).toBe('E_NOT_FOUND');
    });
  }, 30_000);

  it('the context use moves while an answer streams and comes back after a restart', async () => {
    const counted = (scenario: Json): void => {
      scenario.initialize = { models: [{ modelId: 'qwen3.5-4b', totalContextTokens: 262144 }] };
      scenario.prompts = [
        {
          match: '',
          steps: [
            { thought: 'Think.', totalTokens: 535 },
            { text: 'Answer.', totalTokens: 1929 },
            { turnCompleted: { inputTokens: 400, outputTokens: 30 } },
            { end: { stopReason: 'end_turn' } },
          ],
        },
      ];
    };
    const windowOf = async (svc: ChatService): Promise<number | null | undefined> =>
      (await svc.state()).models.find((m) => m.id === 'qwen3.5-4b')?.contextTokens;
    await withService(async (fake) => {
      const svc = fake.service;
      expect(await windowOf(svc)).toBeNull();
      const created = await svc.create({ mode: 'chat', text: 'count me' });
      const id = created.conversation.id;
      const page = new Page(svc, id);
      await until(() => page.ends().length === 1, 5_000);
      expect(page.state!.contextUsed).toBe(1929);
      const sent = page.events.flatMap((e) =>
        e.type === 'ops' ? e.ops.flatMap((op) => (op.op === 'context' ? [op.used] : [])) : [],
      );
      expect(sent.at(-1)).toBe(1929);
      expect(await windowOf(svc)).toBe(262144);
      page.detach();

      // The agent is stopped, as after its idle stop and after a restart: the window stays known.
      await svc.close();
      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      expect(cold.state!.conversation.hot).toBe(false);
      expect(cold.state!.contextUsed).toBe(1929);
      const state = await svc2.state();
      expect(state.modes.chat.state).toBe('stopped');
      expect(state.models.find((m) => m.id === 'qwen3.5-4b')?.contextTokens).toBe(262144);
    }, counted);
  }, 30_000);

  it('a title the agent writes reaches the page as one plain line', async () => {
    await withService(async (fake) => {
      const svc = fake.service;
      const rlo = String.fromCharCode(0x202e);
      const created = await svc.create({ mode: 'chat', text: `Ask ${rlo}about\nthis` });
      const page = new Page(svc, created.conversation.id);
      await until(() => page.state?.conversation.title === 'Ask about this', 5_000);
    });
  });

  it('a prompt sent while the disk is re-read after a crash keeps its place in the view', async () => {
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scripted(scenario);
        (scenario.session as Json).mcpReadyAfterMs = 1_200;
      },
      readSessionDelayMs: 400,
      timings: { mcpReadyMs: 3_000 },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'crash now' });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      expect(page.lastOutcome()).toBe('agent-crashed');
      await svc.prompt(conversation.id, 'hello again');
      await until(() => page.ends().length === 2, 5_000);
      await new Promise((resolve) => setTimeout(resolve, 500));
      expect(page.lastOutcome()).toBe('completed');
      expect(page.texts('user')).toEqual(['crash now', 'hello again']);
      expect(page.texts('assistant')).toEqual(['Partial answer', 'Hello! How can I help?']);
    } finally {
      await fake.close();
    }
  }, 15_000);

  it('a model switch made while the disk is re-read after a crash keeps its place in the view', async () => {
    const fake = fakeService('chat-basic', { mutate: scripted, readSessionDelayMs: 400 });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'crash now' });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      expect(page.lastOutcome()).toBe('agent-crashed');
      await svc.patch(conversation.id, { modelId: 'qwen3.5-9b' });
      await until(() => page.events.filter((e) => e.type === 'sync').length === 2, 5_000);
      const models = page.state!.entries.flatMap((e) => (e.kind === 'model' ? [e.modelId] : []));
      expect(models).toEqual(['qwen3.5-9b']);
      expect(page.texts('user')).toEqual(['crash now']);
    } finally {
      await fake.close();
    }
  }, 15_000);

  it('a compaction is on the stream while it runs, and its marker comes back from disk after a restart', async () => {
    const compacting = (scenario: Json): void => {
      scripted(scenario);
      (scenario.prompts as Json[]).unshift({
        match: 'compact',
        steps: [
          { text: 'Working', chunk: 7 },
          { compact: { ms: 1_000, tokensBefore: 222_822, tokensAfter: 31_204 } },
          { text: ' done.', chunk: 6, totalTokens: 31_204 },
          { turnCompleted: { inputTokens: 400, outputTokens: 30 } },
          { end: { stopReason: 'end_turn' } },
        ],
      });
    };
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'compact please' });
      const id = conversation.id;
      const page = new Page(fake.service, id);
      await until(() => page.state?.compaction != null, 5_000);
      expect(page.state!.compaction).toEqual({
        phase: 'running',
        startedAt: expect.any(Number),
        tokensBefore: 222_822,
      });
      const late = new Page(fake.service, id);
      await until(() => late.state !== null, 5_000);
      expect(late.state!.compaction).toEqual(page.state!.compaction);
      await until(() => page.ends().length === 1, 5_000);
      expect(page.lastOutcome()).toBe('completed');
      expect(page.state!.compaction).toBeNull();
      const marker = {
        kind: 'compaction',
        ts: expect.any(Number),
        outcome: 'done',
        tokensBefore: 222_822,
        tokensAfter: 31_204,
      };
      expect(page.state!.entries.filter((e) => e.kind === 'compaction')).toEqual([marker]);
      expect(page.state!.contextUsed).toBe(31_204);
      page.detach();
      late.detach();
      await fake.service.close();

      const reread = new Page(fake.restart(), id);
      await until(() => reread.state !== null, 5_000);
      expect(reread.state!.compaction).toBeNull();
      expect(reread.state!.entries.map((e) => e.kind)).toEqual([
        'user',
        'assistant',
        'compaction',
        'assistant',
        'turn-end',
      ]);
      expect(reread.state!.entries[2]).toEqual(marker);
      expect(reread.state!.contextUsed).toBe(31_204);
    }, compacting);
  }, 20_000);

  it('renames a cold conversation through the session writer, without starting the agent', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'first' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      await fake.service.close();
      ageTree(fake.paths.chatSessions, Date.now() - 120_000);
      const starts = fake.records().filter((r) => r.kind === 'start').length;

      const svc = fake.restart();
      const renamed = await svc.patch(conversation.id, { title: 'Cold title' });
      expect(renamed).toMatchObject({ title: 'Cold title', hot: false });
      expect(fake.reader.detail(conversation.id)!.session.title).toBe('Cold title');
      expect(fake.records().filter((r) => r.kind === 'start').length).toBe(starts);
    });
  }, 20_000);

  it('deletes a cold conversation through the agent at once, also inside 30 s of its last write', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'first' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      await fake.service.close();

      const svc = fake.restart();
      await svc.delete(conversation.id);
      expect(fake.reader.detail(conversation.id)).toBeNull();
      const del = fake.sent().filter((m) => m.method === '_x.ai/session/delete');
      expect(del).toHaveLength(1);
      expect((del[0]!.params as Json).sessionId).toBe(conversation.id);
      expect(fake.sent().filter((m) => m.method === 'session/load')).toHaveLength(0);
    });
  }, 20_000);

  it('deletes through the session writer when the agent cannot start, with a typed refusal inside 30 s', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'first' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      await fake.service.close();

      const svc = fake.restart();
      fake.endpoint.current = null;
      const busy = await apiError(svc.delete(conversation.id));
      expect([busy.code, busy.reason]).toEqual(['E_CONFLICT', 'conversation-busy']);
      expect(fake.reader.detail(conversation.id)).not.toBeNull();

      ageTree(fake.paths.chatSessions, Date.now() - 120_000);
      await svc.delete(conversation.id);
      expect(fake.reader.detail(conversation.id)).toBeNull();
      expect(fake.sent().filter((m) => m.method === '_x.ai/session/delete')).toHaveLength(0);
    });
  }, 20_000);

  it('a delete through the session writer removes the ledger file of a chat this run loaded', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'crash now' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      expect(readdirSync(fake.paths.webLedger)).toHaveLength(1);

      fake.endpoint.current = null;
      ageTree(fake.paths.chatSessions, Date.now() - 20 * 60_000);
      await fake.service.delete(conversation.id);
      expect(fake.reader.detail(conversation.id)).toBeNull();
      expect(fake.sent().filter((m) => m.method === '_x.ai/session/delete')).toHaveLength(0);
      expect(readdirSync(fake.paths.webLedger)).toEqual([]);
    });
  }, 20_000);
});

function slowDelete(scenario: Json): void {
  scripted(scenario);
  scenario.faults = { ...(scenario.faults as Json | undefined), delayMs: { '_x.ai/session/delete': 250 } };
}

function deleteSent(fake: FakeService): boolean {
  return fake.sent().some((m) => m.method === '_x.ai/session/delete');
}

function prompts(fake: FakeService): number {
  return fake.sent().filter((m) => m.method === 'session/prompt').length;
}

describe('Chat service — refusals and typed failures', () => {
  it('refuses a busy conversation, a model it does not have, and a Work folder the folder rule refuses', async () => {
    await withService(async (fake) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'slow start' });
      const busy = await apiError(svc.prompt(conversation.id, 'again'));
      expect([busy.code, busy.reason]).toEqual(['E_CONFLICT', 'conversation-busy']);
      const busyRename = await apiError(svc.patch(conversation.id, { title: 'x' }));
      expect(busyRename.reason).toBe('conversation-busy');
      await svc.cancel(conversation.id);

      const model = await apiError(svc.create({ mode: 'chat', text: 'hi', modelId: 'not-a-model' }));
      expect([model.code, model.reason]).toEqual(['E_CONFLICT', 'model-unavailable']);
      const work = await apiError(svc.create({ mode: 'work', text: 'hi', cwd: join(fake.root, 'home') }));
      expect([work.code, work.reason]).toEqual(['E_BAD_REQUEST', 'cwd-invalid']);
      expect((await apiError(svc.prompt('no-such-id', 'hi'))).code).toBe('E_NOT_FOUND');
    });
  }, 20_000);

  it('refuses a prompt while a delete of a hot conversation waits for the agent', async () => {
    await withService(async (fake) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hi' });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      const del = svc.delete(conversation.id);
      await until(() => deleteSent(fake));
      const late = await apiError(svc.prompt(conversation.id, 'late prompt'));
      expect([late.code, late.reason]).toEqual(['E_CONFLICT', 'conversation-busy']);
      await del;
      expect(prompts(fake)).toBe(1);
      expect(fake.reader.detail(conversation.id)).toBeNull();
    }, slowDelete);
  }, 20_000);

  it('refuses a prompt and a rename while a delete of a cold conversation waits for the agent', async () => {
    const fake = fakeService('chat-basic', { mutate: slowDelete, readSessionDelayMs: 100 });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hi' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      await fake.service.close();
      const svc = fake.restart();
      const del = svc.delete(conversation.id);
      await until(() => deleteSent(fake), 5_000);
      const late = await apiError(svc.prompt(conversation.id, 'late prompt'));
      expect([late.code, late.reason]).toEqual(['E_CONFLICT', 'conversation-busy']);
      const rename = await apiError(svc.patch(conversation.id, { title: 'late title' }));
      expect([rename.code, rename.reason]).toEqual(['E_CONFLICT', 'conversation-busy']);
      await del;
      expect(prompts(fake)).toBe(1);
      expect(fake.sent().filter((m) => m.method === 'session/load')).toHaveLength(0);
      expect(fake.reader.detail(conversation.id)).toBeNull();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('refuses a prompt whose disk read outlived a delete, and the deleted session stays deleted', async () => {
    let hold: Promise<void> | null = null;
    let reads = 0;
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      readSessionHold: () => {
        reads += 1;
        return hold ?? Promise.resolve();
      },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hi' });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      await svc.agentAction({ action: 'restart', mode: 'chat' });
      let release = (): void => {};
      hold = new Promise<void>((resolve) => {
        release = resolve;
      });
      const before = reads;
      const late = apiError(svc.prompt(conversation.id, 'late prompt'));
      await until(() => reads > before);
      await svc.delete(conversation.id);
      release();
      expect((await late).code).toBe('E_NOT_FOUND');
      expect(prompts(fake)).toBe(1);
      expect(fake.sent().filter((m) => m.method === 'session/load')).toHaveLength(0);
      expect(fake.reader.detail(conversation.id)).toBeNull();
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('answers inference-stopped without an endpoint and agent-slow past the create budget', async () => {
    const fake = fakeService('hang-on-new', { createBudgetMs: 400 });
    try {
      const endpoint = fake.endpoint.current;
      fake.endpoint.current = null;
      const stopped = await apiError(fake.service.create({ mode: 'chat', text: 'hi' }));
      expect([stopped.code, stopped.reason]).toEqual(['E_UNAVAILABLE', 'inference-stopped']);
      fake.endpoint.current = endpoint;
      const started = Date.now();
      const slow = await apiError(fake.service.create({ mode: 'chat', text: 'hi' }));
      expect([slow.code, slow.reason]).toEqual(['E_UNAVAILABLE', 'agent-slow']);
      expect(Date.now() - started).toBeLessThan(3_000);
    } finally {
      await fake.close();
    }
  });

  it('never answers an interaction of a Chat conversation', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hi' });
      const error = await apiError(fake.service.answer(conversation.id, 'i1', { optionId: 'allow-once' }));
      expect(error.code).toBe('E_NOT_FOUND');
    });
  });

  it('gives an unknown conversation one gone event', async () => {
    await withService(async (fake) => {
      const events: ConversationEvent[] = [];
      fake.service.watchConversation('0000-unknown', (e) => events.push(e));
      await until(() => events.length === 1);
      expect(events).toEqual([{ type: 'gone', seq: 1, reason: 'unknown' }]);
    });
  });
});

describe('Chat service — titles', () => {
  it('a new conversation is titled by its prompt at once; the agent title replaces it', async () => {
    await withService(async (fake) => {
      const created = await fake.service.create({ mode: 'chat', text: 'Plan the trip to Kyoto\nwith all the details' });
      expect(created.conversation.title).toBe('Plan the trip to Kyoto');
      const page = new Page(fake.service, created.conversation.id);
      await until(() => page.state?.conversation.title === 'Plan the trip to Kyoto with all the details', 5_000);
      page.detach();
    });
  });

  it('an agent that writes no title leaves the prompt title, in the service, the list and a cold read', async () => {
    await withService(
      async (fake) => {
        const created = await fake.service.create({ mode: 'chat', text: 'Why is the sky blue' });
        const id = created.conversation.id;
        const page = new Page(fake.service, id);
        await until(() => page.ends().length === 1, 5_000);
        expect(page.state?.conversation.title).toBe('Why is the sky blue');
        page.detach();
        expect(fake.reader.list().sessions.find((r) => r.id === id)?.title).toBe('Why is the sky blue');
        await fake.service.close();
        const restarted = fake.restart();
        const cold = new Page(restarted, id);
        await until(() => cold.state !== null);
        expect(cold.state?.conversation.title).toBe('Why is the sky blue');
        cold.detach();
      },
      (scenario) => {
        scripted(scenario);
        scenario.noTitle = true;
      },
    );
  });

  it('a blank agent title does not replace the prompt title', async () => {
    await withService(
      async (fake) => {
        const created = await fake.service.create({ mode: 'chat', text: 'Why is the sea salty' });
        const id = created.conversation.id;
        const page = new Page(fake.service, id);
        await until(() => page.ends().length === 1, 5_000);
        expect(page.state?.conversation.title).toBe('Why is the sea salty');
        expect(fake.reader.list().sessions.find((r) => r.id === id)?.title).toBe('Why is the sea salty');
        page.detach();
      },
      (scenario) => {
        scripted(scenario);
        scenario.blankTitle = true;
      },
    );
  });
});

describe('Chat service — engine facts and unload', () => {
  type Load = RuntimeLive['processes'][number]['loading'][number];
  const load = (model: string, startedAt: number, patch: Partial<Load> = {}): Load => ({
    model,
    startedAt,
    phase: 'weights',
    stepsDone: 0,
    steps: 0,
    ...patch,
  });

  function liveWith(loading: Load[], names: string[], ts = Date.now()): RuntimeLive {
    return live({
      processes: [
        liveProcess({ pid: process.pid, ts, loading, models: names.map((n) => liveModel(n)) }),
        liveProcess({
          pid: process.pid + 1,
          host: 'launch',
          loading: [load('qwen3.5-4b', 99)],
          models: [liveModel('qwen3.5-4b')],
        }),
      ],
    }) as unknown as RuntimeLive;
  }

  it('names the load and the resident chat models of the app engine only, and a new conversation takes the resident model', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-chat-sizes-'));
    mkdirSync(join(dir, 'b'));
    writeFileSync(join(dir, 'b', 'model.safetensors'), 'x'.repeat(200));
    writeFileSync(join(dir, 'b', 'config.json'), 'x'.repeat(100));
    let snapshot = liveWith(
      [load('qwen3.5-9b', 1234, { phase: 'kernels', stepsDone: 3, steps: 50 }), load('gemma-4', 1000)],
      ['qwen3.5-9b', 'mlx-local'],
    );
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      endpointModels: [endpointModel('qwen3.5-4b', join(dir, 'a')), endpointModel('qwen3.5-9b', join(dir, 'b'))],
      live: { subscribe: () => () => {}, readNow: async () => snapshot },
    });
    try {
      let state = await fake.service.state();
      expect(state).toMatchObject({
        inference: 'loading',
        loading: [
          { modelId: 'gemma-4', startedAt: 1000, phase: 'weights', stepsDone: 0, steps: 0 },
          { modelId: 'qwen3.5-9b', startedAt: 1234, phase: 'kernels', stepsDone: 3, steps: 50 },
        ],
        resident: [{ modelId: 'qwen3.5-9b', sizeBytes: 300 }],
        defaultModelId: 'qwen3.5-9b',
      });
      snapshot = liveWith([], []);
      state = await fake.service.state();
      expect(state).toMatchObject({ inference: 'ready', loading: [], resident: [], defaultModelId: 'qwen3.5-4b' });
      snapshot = liveWith([], ['qwen3.5-4b']);
      expect((await fake.service.state()).resident).toEqual([{ modelId: 'qwen3.5-4b', sizeBytes: null }]);
      expect((await fake.service.state()).answers).toEqual([
        { modelId: 'qwen3.5-4b', inFlight: 0, maxConcurrent: 4, queued: 0, lastStart: null },
      ]);
      snapshot = live({
        processes: [
          liveProcess({
            pid: process.pid,
            loading: [],
            models: [liveModel('qwen3.5-4b', { maxConcurrent: 1, inFlight: 1, queued: 1 }), liveModel('mlx-local')],
            turns: [
              liveTurn('t-old', { model: 'qwen3.5-4b', startedAt: 5_000 }),
              liveTurn('t-new', { model: 'qwen3.5-4b', startedAt: 7_000 }),
              liveTurn('t-other', { model: 'mlx-local', startedAt: 9_000 }),
            ],
          }),
        ],
      }) as unknown as RuntimeLive;
      expect((await fake.service.state()).answers).toEqual([
        { modelId: 'qwen3.5-4b', inFlight: 1, maxConcurrent: 1, queued: 1, lastStart: 7_000 },
      ]);
      snapshot = liveWith([], ['qwen3.5-9b']);
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hi' });
      expect(conversation.modelId).toBe('qwen3.5-9b');
    } finally {
      await fake.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("lists the app engine's requests, oldest first, with the session each named, its phase and its prefill counts", async () => {
    let snapshot = live({
      processes: [liveProcess({ pid: process.pid, loading: [], models: [liveModel('qwen3.5-4b')], turns: [] })],
    }) as unknown as RuntimeLive;
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      endpointModels: [endpointModel('qwen3.5-4b', '/nowhere/a'), endpointModel('qwen3.5-9b', '/nowhere/b')],
      live: { subscribe: () => () => {}, readNow: async () => snapshot },
    });
    try {
      expect((await fake.service.state()).requests).toEqual([]);
      const turn = liveTurn('x', { model: 'qwen3.5-4b' });
      snapshot = live({
        processes: [
          liveProcess({
            pid: process.pid,
            loading: [],
            models: [liveModel('qwen3.5-4b', { maxConcurrent: 4, inFlight: 3, queued: 0 }), liveModel('qwen3.5-9b')],
            turns: [
              liveTurn('t-held', {
                model: 'qwen3.5-4b',
                startedAt: 9_000,
                phase: 'queued',
                prefillTotal: null,
                prefillDone: null,
                generatedTokens: null,
                clientSession: 'acp-ours',
              }),
              liveTurn('t-other', {
                model: 'qwen3.5-4b',
                startedAt: 5_000,
                phase: 'prefill',
                prefillTotal: 121_300,
                prefillDone: 69_300,
                clientSession: 'acp-other',
              }),
              liveTurn('t-terminal', { model: 'qwen3.5-9b', startedAt: 7_000, phase: 'decode', clientSession: null }),
            ],
          }),
          liveProcess({
            pid: process.pid + 1,
            host: 'launch',
            loading: [],
            models: [liveModel('qwen3.5-4b')],
            turns: [{ ...turn, startedAt: 1_000, clientSession: 'acp-ours' }],
          }),
        ],
      }) as unknown as RuntimeLive;
      expect((await fake.service.state()).requests).toEqual([
        {
          session: 'acp-other',
          modelId: 'qwen3.5-4b',
          phase: 'prefill',
          startedAt: 5_000,
          prefillDone: 69_300,
          prefillTotal: 121_300,
          generated: 1_204,
        },
        {
          session: null,
          modelId: 'qwen3.5-9b',
          phase: 'decode',
          startedAt: 7_000,
          prefillDone: 1_000,
          prefillTotal: 1_000,
          generated: 1_204,
        },
        {
          session: 'acp-ours',
          modelId: 'qwen3.5-4b',
          phase: 'queued',
          startedAt: 9_000,
          prefillDone: null,
          prefillTotal: null,
          generated: null,
        },
      ]);
    } finally {
      await fake.close();
    }
  });

  it('lists every served model with its base, build, format and size; a split GGUF counts all its parts', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-chat-models-'));
    const repo = join(dir, 'flash-gguf', 'UD-Q4_K_XL');
    mkdirSync(repo, { recursive: true });
    for (const k of [1, 2, 3])
      writeFileSync(join(repo, `Flash-UD-Q4_K_XL-0000${k}-of-00003.gguf`), 'x'.repeat(100 * k));
    writeFileSync(join(repo, 'Flash-UD-Q4_K_XL-00001-of-00002.gguf'), 'x'.repeat(7));
    writeFileSync(join(repo, 'mmproj-F16.gguf'), 'x'.repeat(5));
    mkdirSync(join(dir, 'flash-bf16'));
    writeFileSync(join(dir, 'flash-bf16', 'model.safetensors'), 'x'.repeat(40));
    const split = join(repo, 'Flash-UD-Q4_K_XL-00001-of-00003.gguf');
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      endpointModels: [
        { name: 'Flash-UD-Q4_K_XL', path: split, base: 'Flash', variant: 'UD-Q4_K_XL', format: 'gguf' },
        { name: 'flash-bf16', path: join(dir, 'flash-bf16'), base: 'Flash', variant: 'BF16', format: 'mlx' },
        { name: 'gone', path: join(dir, 'gone'), base: 'gone', variant: 'MLX', format: 'mlx' },
      ],
    });
    try {
      const fields = (state: Awaited<ReturnType<typeof fake.service.state>>) =>
        state.models.map(({ id, base, variant, format, sizeBytes }) => ({ id, base, variant, format, sizeBytes }));
      expect(fields(await fake.service.state())).toEqual([
        { id: 'Flash-UD-Q4_K_XL', base: 'Flash', variant: 'UD-Q4_K_XL', format: 'gguf', sizeBytes: 600 },
        { id: 'flash-bf16', base: 'Flash', variant: 'BF16', format: 'mlx', sizeBytes: 40 },
        { id: 'gone', base: 'gone', variant: 'MLX', format: 'mlx', sizeBytes: null },
      ]);
      writeFileSync(join(dir, 'flash-bf16', 'model.safetensors'), 'x'.repeat(50));
      expect((await fake.service.state()).models[1]!.sizeBytes).toBe(40);
      fake.endpoint.current = { ...fake.endpoint.current!, pid: fake.endpoint.current!.pid + 1 };
      expect((await fake.service.state()).models[1]!.sizeBytes).toBe(50);
    } finally {
      await fake.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('sizes a split GGUF only when the loader can open every part beside the listed path', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-chat-split-'));
    const parts = [1, 2, 3].map((k) => `Flash-UD-Q4_K_XL-0000${k}-of-00003.gguf`);
    const real = join(dir, 'real', 'flash-gguf', 'UD-Q4_K_XL');
    mkdirSync(real, { recursive: true });
    parts.forEach((name, i) => writeFileSync(join(real, name), 'x'.repeat(100 * (i + 1))));
    mkdirSync(join(dir, 'repo-link'));
    symlinkSync(join(dir, 'real', 'flash-gguf'), join(dir, 'repo-link', 'flash-gguf'));
    mkdirSync(join(dir, 'variant-link', 'flash-gguf'), { recursive: true });
    symlinkSync(real, join(dir, 'variant-link', 'flash-gguf', 'UD-Q4_K_XL'));
    mkdirSync(join(dir, 'shard-link'));
    symlinkSync(join(real, parts[0]!), join(dir, 'shard-link', parts[0]!));
    mkdirSync(join(dir, 'partial'));
    for (const name of parts.slice(0, 2)) writeFileSync(join(dir, 'partial', name), 'x'.repeat(10));
    mkdirSync(join(dir, 'folder-part', parts[1]!), { recursive: true });
    for (const name of [parts[0]!, parts[2]!]) writeFileSync(join(dir, 'folder-part', name), 'x'.repeat(10));
    const at = (id: string, path: string) => ({ name: id, path, base: 'Flash', variant: id, format: 'gguf' as const });
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      endpointModels: [
        at('complete', join(real, parts[0]!)),
        at('repo-link', join(dir, 'repo-link', 'flash-gguf', 'UD-Q4_K_XL', parts[0]!)),
        at('variant-link', join(dir, 'variant-link', 'flash-gguf', 'UD-Q4_K_XL', parts[0]!)),
        at('shard-link', join(dir, 'shard-link', parts[0]!)),
        at('partial', join(dir, 'partial', parts[0]!)),
        at('folder-part', join(dir, 'folder-part', parts[0]!)),
      ],
    });
    try {
      expect((await fake.service.state()).models.map(({ id, sizeBytes }) => ({ id, sizeBytes }))).toEqual([
        { id: 'complete', sizeBytes: 600 },
        { id: 'repo-link', sizeBytes: 600 },
        { id: 'variant-link', sizeBytes: 600 },
        { id: 'shard-link', sizeBytes: null },
        { id: 'partial', sizeBytes: null },
        { id: 'folder-part', sizeBytes: null },
      ]);
    } finally {
      await fake.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('gives no size to a split GGUF with two files for one part', async (context) => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-chat-split-case-'));
    writeFileSync(join(dir, 'probe'), '');
    writeFileSync(join(dir, 'PROBE'), '');
    if (readdirSync(dir).length < 2) {
      rmSync(dir, { recursive: true, force: true });
      context.skip();
    }
    const first = join(dir, 'Flash-00001-of-00002.gguf');
    writeFileSync(first, 'x');
    writeFileSync(join(dir, 'Flash-00002-of-00002.gguf'), 'x');
    writeFileSync(join(dir, 'Flash-00002-of-00002.GGUF'), 'x');
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      endpointModels: [{ name: 'Flash', path: first, base: 'Flash', variant: 'GGUF', format: 'gguf' }],
    });
    try {
      expect((await fake.service.state()).models[0]!.sizeBytes).toBeNull();
    } finally {
      await fake.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('unload: refused while a turn runs or the app engine loads, with no engine request; else the engine names the model', async () => {
    let snapshot = liveWith([], ['qwen3.5-4b']);
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      live: { subscribe: () => () => {}, readNow: async () => snapshot },
      engine: (r) => (r.path === UNLOAD_PATH ? { json: { unloaded: ['qwen3.5-4b'] } } : { json: {} }),
    });
    try {
      const engine = await fake.engine();
      const unloads = (): number => engine.requests.filter((r) => r.path === UNLOAD_PATH).length;
      const created = await fake.service.create({ mode: 'chat', text: 'slow one' });
      const page = new Page(fake.service, created.conversation.id);
      expect((await apiError(fake.service.unloadModels())).reason).toBe('engine-busy');
      await fake.service.cancel(created.conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      snapshot = liveWith([load('qwen3.5-9b', 1)], []);
      expect((await apiError(fake.service.unloadModels())).reason).toBe('engine-busy');
      expect(unloads()).toBe(0);
      snapshot = liveWith([], ['qwen3.5-4b']);
      expect(await fake.service.unloadModels()).toEqual({ unloaded: ['qwen3.5-4b'] });
      expect(unloads()).toBe(1);
      engine.script(() => ({ status: 409, json: {} }));
      const busy = await apiError(fake.service.unloadModels());
      expect({ code: busy.code, reason: busy.reason }).toEqual({ code: 'E_CONFLICT', reason: 'engine-busy' });
      engine.script(() => ({ status: 500, json: {} }));
      expect((await apiError(fake.service.unloadModels())).reason).toBe('inference-failed');
    } finally {
      await fake.close();
    }
  });

  it('after an unload, a heartbeat written before it does not list the unloaded model; a later one does', async () => {
    const before = Date.now() - 1_000;
    let snapshot = liveWith([], ['qwen3.5-4b'], before);
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      live: { subscribe: () => () => {}, readNow: async () => snapshot },
      engine: (r) => (r.path === UNLOAD_PATH ? { json: { unloaded: ['qwen3.5-4b'] } } : { json: {} }),
    });
    try {
      expect((await fake.service.state()).resident.map((r) => r.modelId)).toEqual(['qwen3.5-4b']);
      await fake.service.unloadModels();
      let state = await fake.service.state();
      expect(state.resident).toEqual([]);
      expect(state.defaultModelId).toBe('qwen3.5-4b');
      snapshot = liveWith([], ['qwen3.5-4b'], Date.now() + 1_000);
      state = await fake.service.state();
      expect(state.resident.map((r) => r.modelId)).toEqual(['qwen3.5-4b']);
      snapshot = liveWith([], ['qwen3.5-4b'], before);
      expect((await fake.service.state()).resident.map((r) => r.modelId)).toEqual(['qwen3.5-4b']);
    } finally {
      await fake.close();
    }
  });

  it("takes a model's image support from the loaded model's own answer first, then the endpoint file, else none", async () => {
    const seen = (name: string, images: boolean | null) => Object.assign(liveModel(name), { images });
    const snapshot = live({
      processes: [
        liveProcess({
          pid: process.pid,
          loading: [],
          models: [seen('said-no', false), seen('said-yes', true), seen('not-asked', null)],
        }),
      ],
    }) as unknown as RuntimeLive;
    const fake = fakeService('chat-basic', {
      mutate: scripted,
      endpointModels: [
        { ...endpointModel('said-no', '/models/a'), images: true },
        { ...endpointModel('said-yes', '/models/b'), images: false },
        { ...endpointModel('not-asked', '/models/c'), images: true },
        { ...endpointModel('on-disk', '/models/d'), images: true },
        endpointModel('unknown', '/models/e'),
      ],
      live: { subscribe: () => () => {}, readNow: async () => snapshot },
    });
    try {
      expect((await fake.service.state()).models.map((m) => [m.id, m.images])).toEqual([
        ['said-no', false],
        ['said-yes', true],
        ['not-asked', true],
        ['on-disk', true],
        ['unknown', false],
      ]);
    } finally {
      await fake.close();
    }
  });
});

describe('Chat service — attachments', () => {
  const FILE: AttachmentInput = { kind: 'file', name: 'notes.md', text: '# notes\n' };
  const PICTURE: AttachmentInput = { kind: 'image', mediaType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUg==' };
  const SEES = { ...endpointModel('qwen3.5-4b', '/models/qwen3.5-4b'), images: true };
  const BLIND = endpointModel('qwen3.5-9b', '/models/qwen3.5-9b');
  const refusedAs = (error: ApiError) => [error.code, error.reason, error.message];
  const REFUSED = ['E_BAD_REQUEST', 'attachment-refused', 'The attached files were refused.'];

  // Mutation: let a picture through on a model whose row has images false → red.
  it('a picture goes only to a model whose row takes images, refused before any session or prompt', async () => {
    const fake = fakeService('chat-basic', { mutate: scripted, endpointModels: [SEES, BLIND] });
    try {
      const svc = fake.service;
      const prompts = () => fake.sent().filter((m) => m.method === 'session/prompt');
      const blind = await apiError(
        svc.create({ mode: 'chat', text: 'what is this', modelId: BLIND.name, attachments: [PICTURE] }),
      );
      expect(refusedAs(blind)).toEqual(REFUSED);
      expect(fake.sent().filter((m) => m.method === 'session/new')).toEqual([]);

      const attachments = [FILE, PICTURE];
      const { conversation } = await svc.create({ mode: 'chat', text: 'what is this', attachments });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      expect(prompts().map((m) => m.params)).toEqual([promptParams(conversation.id, 'what is this', attachments)]);
      expect(page.state!.entries.filter((e) => e.kind === 'user')).toEqual([
        expect.objectContaining({ text: 'what is this', attachments: attachments.map(attachmentView) }),
      ]);
      page.detach();

      const text = await svc.create({ mode: 'chat', text: 'hello', modelId: BLIND.name });
      const second = new Page(svc, text.conversation.id);
      await until(() => second.ends().length === 1, 5_000);
      expect(refusedAs(await apiError(svc.prompt(text.conversation.id, 'and this?', [PICTURE])))).toEqual(REFUSED);
      expect(prompts()).toHaveLength(2);
      await svc.prompt(text.conversation.id, 'and this file?', [FILE]);
      await until(() => second.ends().length === 2, 5_000);
      expect(prompts()[2]!.params).toEqual(promptParams(text.conversation.id, 'and this file?', [FILE]));
      second.detach();
    } finally {
      await fake.close();
    }
  }, 20_000);

  // Mutation: drop the count of the pictures a chat holds → the last picture goes out.
  it('a picture past the stored pictures a chat can keep is refused; text and files still go', async () => {
    const cap = 1024;
    const fake = fakeService('chat-basic', { mutate: scripted, endpointModels: [SEES], maxPicturesBase64: cap });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hello' });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      const big: AttachmentInput = {
        kind: 'image',
        mediaType: 'image/png',
        data: `iVBO${'A'.repeat(cap - 8)}`,
      };
      await svc.prompt(conversation.id, 'a big one', [big]);
      await until(() => page.ends().length === 2, 10_000);
      const full = await apiError(svc.prompt(conversation.id, 'one more', [PICTURE]));
      expect(refusedAs(full)).toEqual([
        'E_CONFLICT',
        'pictures-full',
        'This chat has too many pictures to keep. Start a new chat to send more.',
      ]);
      await svc.prompt(conversation.id, 'and this file?', [FILE]);
      await until(() => page.ends().length === 3, 5_000);
      const prompts = fake.sent().filter((m) => m.method === 'session/prompt');
      expect(prompts.map((m) => (m.params as { prompt: Array<{ text?: string }> }).prompt[0]!.text)).toEqual([
        'hello',
        'a big one',
        'and this file?',
      ]);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: let a picture through in Work → red.
  it('a picture is refused in Work on any model, before the session; a file is not', async () => {
    await withWork(
      'work-always',
      async (fake, project) => {
        const svc = fake.service;
        const work = await apiError(svc.create({ mode: 'work', text: 'list', cwd: project, attachments: [PICTURE] }));
        expect(refusedAs(work)).toEqual(REFUSED);
        expect(fake.sent().filter((m) => m.method === 'session/new')).toEqual([]);
        const { conversation } = await svc.create({ mode: 'work', text: 'list', cwd: project, attachments: [FILE] });
        await until(() => fake.sent().some((m) => m.method === 'session/prompt'), 5_000);
        const prompt = fake.sent().find((m) => m.method === 'session/prompt')!;
        expect(prompt.params).toEqual(promptParams(conversation.id, 'list', [FILE], { mode: 'agent' }));
        expect(refusedAs(await apiError(svc.prompt(conversation.id, 'again', [PICTURE])))).toEqual(REFUSED);
      },
      { endpointModels: [SEES] },
    );
  }, 20_000);

  const PAGE = '/9j/4AAQ';
  const pdf = (pages: number, name = 'paper.pdf'): AttachmentInput => ({
    kind: 'pdf',
    name,
    images: Array<string>(pages).fill(PAGE),
  });

  // Mutation: the Work picture refusal dropped along with the PDF change → red; a PDF refused in Work → red.
  it('a PDF goes to Work on a model that sees images; a picture in Work is still refused', async () => {
    await withWork(
      'work-always',
      async (fake, project) => {
        const svc = fake.service;
        const { conversation } = await svc.create({ mode: 'work', text: 'read', cwd: project, attachments: [pdf(2)] });
        await until(() => fake.sent().some((m) => m.method === 'session/prompt'), 5_000);
        const prompt = fake.sent().find((m) => m.method === 'session/prompt')!;
        expect(prompt.params).toEqual(promptParams(conversation.id, 'read', [pdf(2)], { mode: 'agent' }));
        expect(refusedAs(await apiError(svc.prompt(conversation.id, 'again', [pdf(1), PICTURE])))).toEqual(REFUSED);
        const picture = await apiError(
          svc.create({ mode: 'work', text: 'look', cwd: project, attachments: [PICTURE] }),
        );
        expect(refusedAs(picture)).toEqual(REFUSED);
        expect(fake.sent().filter((m) => m.method === 'session/new')).toHaveLength(1);
      },
      { endpointModels: [SEES] },
    );
  }, 20_000);

  // Mutation: a PDF let through on a model whose row has images false, in either mode → red.
  it('a PDF on a model that does not see images is refused in Chat and in Work, before any session', async () => {
    const chat = fakeService('chat-basic', { mutate: scripted, endpointModels: [SEES, BLIND] });
    try {
      const svc = chat.service;
      const blind = await apiError(
        svc.create({ mode: 'chat', text: 'read', modelId: BLIND.name, attachments: [pdf(1)] }),
      );
      expect(refusedAs(blind)).toEqual(REFUSED);
      expect(chat.sent().filter((m) => m.method === 'session/new')).toEqual([]);
      const { conversation } = await svc.create({ mode: 'chat', text: 'hello', modelId: BLIND.name });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      expect(refusedAs(await apiError(svc.prompt(conversation.id, 'read', [pdf(1)])))).toEqual(REFUSED);
      page.detach();
    } finally {
      await chat.close();
    }
    await withWork(
      'work-always',
      async (fake, project) => {
        const work = await apiError(
          fake.service.create({ mode: 'work', text: 'read', cwd: project, modelId: BLIND.name, attachments: [pdf(1)] }),
        );
        expect(refusedAs(work)).toEqual(REFUSED);
        expect(fake.sent().filter((m) => m.method === 'session/new')).toEqual([]);
      },
      { endpointModels: [SEES, BLIND] },
    );
  }, 20_000);

  describe('a PDF for the tool', () => {
    const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0]);
    /** Stages `pages` pages through the route's method, as the page sends them. */
    async function staged(svc: ChatService, pages: number): Promise<string> {
      const { stage } = await svc.stagePdf({ stage: null, page: 1, pages, data: JPEG });
      for (let page = 2; page <= pages; page++) await svc.stagePdf({ stage, page, pages, data: JPEG });
      return stage;
    }
    const tool = (stage: string, pages: number): AttachmentInput => ({ kind: 'pdf', name: 'report.pdf', stage, pages });
    const prompts = (fake: FakeService) =>
      fake
        .sent()
        .filter((m) => m.method === 'session/prompt')
        .map((m) => m.params as Json);

    // Mutations: the stage not placed on Send; Read it now not the agent's own command; Read it now with a reader of the
    // Work folder's that takes the tool's name; the page folders kept on delete → red.
    it('Work: Send moves the stage to the page folder of its id, the note names the id; Read it now starts read-pdf', async () => {
      await withWork(
        'work-always',
        async (fake, project) => {
          const svc = fake.service;
          expect((await svc.state()).pdfTool).toEqual({ chat: false, work: true });
          const stage = await staged(svc, 3);
          const item = tool(stage, 3);
          const id = pdfToolId(stage, 3);
          const { conversation } = await svc.create({ mode: 'work', text: 'read', cwd: project, attachments: [item] });
          const page = new Page(svc, conversation.id);
          // The first turn can end before the page is attached: its sync then shows no turn.
          const done = () => {
            const first = page.events[0];
            return page.ends().length + (first?.type === 'sync' && first.turn === null ? 1 : 0);
          };
          await until(() => page.events.length > 0 && done() === 1, 5_000);
          expect(prompts(fake)[0]).toEqual(promptParams(conversation.id, 'read', [item], { mode: 'agent' }));
          const folder = join(fake.paths.workPdf, id);
          expect(readdirSync(folder).sort()).toEqual(['meta.json', 'page-001.jpg', 'page-002.jpg', 'page-003.jpg']);
          expect(JSON.parse(readFileSync(join(folder, 'meta.json'), 'utf8'))).toMatchObject({
            session: conversation.id,
            name: 'report.pdf',
            pages: 3,
          });
          expect(existsSync(join(fake.paths.pdfStage, stage))).toBe(false);
          expect(readFileSync(join(fake.root, 'agent-home', 'workflows', 'read-pdf.rhai'), 'utf8')).toBe(
            readPdfScript(fake.paths.workPdf),
          );
          await svc.prompt(conversation.id, 'and again', [item]);
          await until(() => done() === 2, 5_000);
          expect(prompts(fake)[1]).toEqual(promptParams(conversation.id, 'and again', [item], { mode: 'agent' }));
          const own = join(project, '.mlx-agent', 'agents', 'pdf-reader.md');
          mkdirSync(join(own, '..'), { recursive: true });
          writeFileSync(own, '---\nname: pdf-reader\n---\nmine\n');
          expect(refusedAs(await apiError(svc.readPdf(conversation.id, id)))).toEqual(REFUSED);
          rmSync(join(project, '.mlx-agent'), { recursive: true });
          await svc.readPdf(conversation.id, id);
          await until(() => done() === 3, 5_000);
          const sent = (prompts(fake)[2]!.prompt as Array<{ type: string; text?: string }>)[0]!;
          expect(sent).toEqual({ type: 'text', text: readPdfCommand(id, 'read', 'report.pdf', 'you') });
          const work = new AgentSessionsReader({ agentHome: join(fake.root, 'agent-home') });
          const stored = work.detail(conversation.id)!.transcript;
          const command = stored.findIndex((e) => e.kind === 'user' && e.text.startsWith('/workflow read-pdf '));
          expect(stored.slice(command + 1).find((e) => e.kind === 'user' || e.kind === 'turn-end')?.kind).toBe(
            'turn-end',
          );
          const gone = await apiError(svc.readPdf(conversation.id, pdfToolId('b'.repeat(26), 3)));
          expect([gone.code, gone.message]).toEqual(['E_NOT_FOUND', 'No such PDF.']);
          page.detach();
          await svc.delete(conversation.id);
          expect(existsSync(folder)).toBe(false);
        },
        { endpointModels: [SEES], mutate: scripted },
      );
    }, 30_000);

    // Mutations: a removed row's page folder kept, on Remove or on Clear; a folder removed that a sent message names,
    // or that a row still queued carries → red.
    it('Work: a queued row that is removed or cleared takes its page folder, unless a message or a row names it', async () => {
      let gate = '';
      const hold = (scenario: Json) => {
        gate = join(String(scenario.recordTo), '..', 'gate');
        const end = [{ text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }];
        scenario.prompts = [{ match: 'hold', steps: [{ holdOn: gate }, ...end] }, ...(scenario.prompts as Json[])];
      };
      await withWork(
        'work-always',
        async (fake, project) => {
          const svc = fake.service;
          const folder = (stage: string, pages: number) => join(fake.paths.workPdf, pdfToolId(stage, pages));
          const sent = await staged(svc, 2);
          const created = await svc.create({ mode: 'work', text: 'hold', cwd: project, attachments: [tool(sent, 2)] });
          const id = created.conversation.id;
          const page = new Page(svc, id);
          await until(() => prompts(fake).length === 1, 5_000);
          const queued = async (stage: string, pages: number): Promise<string> => {
            const reply = await svc.prompt(id, 'later', [tool(stage, pages)], undefined, 'queue');
            if (!('queued' in reply)) throw new Error('not queued');
            return reply.queued;
          };
          const [a, b, twice] = [await staged(svc, 3), await staged(svc, 2), await staged(svc, 4)];
          const rowA = await queued(a, 3);
          await queued(b, 2);
          const rowSent = await queued(sent, 2);
          const rowTwice = await queued(twice, 4);
          await queued(twice, 4);
          const on = () =>
            [folder(sent, 2), folder(a, 3), folder(b, 2), folder(twice, 4)].map((path) => existsSync(path));
          expect(on()).toEqual([true, true, true, true]);
          await svc.queueRemove(id, rowA);
          expect(on()).toEqual([true, false, true, true]);
          await svc.queueRemove(id, rowSent);
          expect(on()).toEqual([true, false, true, true]);
          await svc.queueRemove(id, rowTwice);
          expect(on()).toEqual([true, false, true, true]);
          await svc.queueClear(id);
          expect(on()).toEqual([true, false, false, false]);
          writeFileSync(gate, '');
          await until(() => page.ends().length === 1, 5_000);
          page.detach();
          await svc.delete(id);
          expect(on()).toEqual([false, false, false, false]);
        },
        { endpointModels: [SEES], mutate: hold },
      );
    }, 20_000);

    it('is refused in Chat, with a stage that is not complete, and on a model that does not see images', async () => {
      await withWork(
        'work-always',
        async (fake, project) => {
          const svc = fake.service;
          const half = (await svc.stagePdf({ stage: null, page: 1, pages: 3, data: JPEG })).stage;
          const refusedWith = async (input: Parameters<ChatService['create']>[0]) =>
            refusedAs(await apiError(svc.create(input)));
          expect(await refusedWith({ mode: 'work', text: 'read', cwd: project, attachments: [tool(half, 3)] })).toEqual(
            REFUSED,
          );
          const whole = await staged(svc, 2);
          expect(
            await refusedWith({
              mode: 'work',
              text: 'read',
              cwd: project,
              modelId: BLIND.name,
              attachments: [tool(whole, 2)],
            }),
          ).toEqual(REFUSED);
          expect(await refusedWith({ mode: 'chat', text: 'read', attachments: [tool(whole, 2)] })).toEqual(REFUSED);
          expect(fake.sent().filter((m) => m.method === 'session/new')).toEqual([]);
          const late = await apiError(svc.stagePdf({ stage: half, page: 3, pages: 3, data: JPEG }));
          expect(refusedAs(late)).toEqual(REFUSED);
          expect(await svc.dropPdfStage(half)).toEqual({});
          expect(existsSync(join(fake.paths.pdfStage, half))).toBe(false);
        },
        { endpointModels: [SEES, BLIND] },
      );
    }, 20_000);

    // Mutation: the tool reported on while a file of the person's holds its name → red.
    // Mutations: the host start left out; the family check dropped; a start in Ask; a start after a move down to Ask
    // during its checks, or while the host selects the model; that held start left in the thread; no start for a PDF
    // sent inside a fork; Read it now in a fork on a PDF copied from the original (the session check dropped); a tool
    // note that does not name the conversation it is sent in; a start after the model's call of the PDF, after any
    // call of the tool, or after a deny; the start sent as the person's; Read it now while a run of the tool goes on;
    // Read it now during the host's start not answered with its turn → red.
    it('Work Auto and Bypass: on a family that skips the tool, the host starts read-pdf when the turn that sent the PDF ends', async () => {
      const GEMMA = { ...endpointModel('qwen3.5-4b', '/models/qwen3.5-4b'), modelType: 'gemma4', images: true };
      const INKLING = { ...endpointModel('qwen3.5-9b', '/models/qwen3.5-9b'), modelType: 'inkling2', images: true };
      await withWork(
        'chat-harness',
        async (fake, project) => {
          const svc = fake.service;
          const texts = (session: string): string[] =>
            prompts(fake)
              .filter((p) => p.sessionId === session)
              .map((p) => (p.prompt as Array<{ text?: string }>)[0]!.text ?? '');
          const ended = (page: Page) => () => {
            const first = page.events[0];
            return page.ends().length + (first?.type === 'sync' && first.turn === null ? 1 : 0) > 0;
          };
          let drafts = 0;
          /** Sends `text` with a 3-page tool PDF on `model` in `mode`; its page, and its first turn's end as the page saw it. */
          const send = async (text: string, model: string, mode: PermissionMode) => {
            const stage = await staged(svc, 3);
            const draft = `d${++drafts}`;
            if (mode !== 'ask') await svc.setDraftPermissionMode(draft, mode, true, project, null);
            const { conversation, turnId } = await svc.create({
              mode: 'work',
              text,
              cwd: project,
              modelId: model,
              attachments: [tool(stage, 3)],
              draft,
            });
            expect(conversation.permissionMode).toBe(mode);
            const page = new Page(svc, conversation.id);
            return { id: conversation.id, pdf: pdfToolId(stage, 3), turnId, page, ended: ended(page) };
          };
          const settle = () => new Promise((resolve) => setTimeout(resolve, 500));
          const answered = async (p: Awaited<ReturnType<typeof send>>, optionId: string) => {
            await until(() => interactionsOf(p.page).length === 1, 10_000);
            await svc.answer(p.id, interactionsOf(p.page)[0]!.id, { optionId });
            await until(p.ended, 10_000);
            await settle();
          };
          const quiet = async (p: Awaited<ReturnType<typeof send>>) => {
            await until(p.ended, 10_000);
            await settle();
            expect(interactionsOf(p.page)).toEqual([]);
          };

          const skipped = await send('pdfnot', GEMMA.name, 'bypass');
          let pressed: Promise<{ turnId: string }> | null = null;
          let hostTurn: string | null = null;
          const stopWatch = svc.watchConversation(skipped.id, (event) => {
            if (pressed !== null || event.type !== 'turn' || event.turn === null) return;
            if (event.turn.turnId === skipped.turnId) return;
            hostTurn = event.turn.turnId;
            pressed = svc.readPdf(skipped.id, skipped.pdf);
          });
          await until(() => texts(skipped.id).length === 2, 10_000);
          expect(texts(skipped.id)).toEqual(['pdfnot', readPdfCommand(skipped.pdf, 'pdfnot', 'report.pdf', 'app')]);
          await until(() => pressed !== null, 5_000);
          expect(await pressed!).toEqual({ turnId: hostTurn });
          stopWatch();

          const auto = await send('pdfnot', GEMMA.name, 'auto');
          await until(() => texts(auto.id).length === 2, 10_000);
          expect(texts(auto.id)).toEqual(['pdfnot', readPdfCommand(auto.pdf, 'pdfnot', 'report.pdf', 'app')]);

          const facts = svc as unknown as { pdfFacts: (sessionId: string, pdf: string) => Promise<unknown> };
          const realFacts = facts.pdfFacts.bind(svc);
          let watched: { id: string; pdf: string } | null = null;
          let lowered = false;
          facts.pdfFacts = async (sessionId, pdf) => {
            if (!lowered && watched?.id === sessionId && watched.pdf === pdf) {
              lowered = true;
              expect(await svc.setPermissionMode(sessionId, 'ask', false)).toEqual({ mode: 'ask' });
            }
            return realFacts(sessionId, pdf);
          };
          const movedDown = await send('pdfnot', GEMMA.name, 'bypass');
          watched = movedDown;
          await quiet(movedDown);
          facts.pdfFacts = realFacts;
          expect(lowered).toBe(true);
          expect(texts(movedDown.id)).toEqual(['pdfnot']);
          await svc.readPdf(movedDown.id, movedDown.pdf);
          await until(() => texts(movedDown.id).length === 2, 10_000);
          expect(texts(movedDown.id)).toEqual(['pdfnot', readPdfCommand(movedDown.pdf, 'pdfnot', 'report.pdf', 'you')]);

          type Select = (
            running: unknown,
            session: { id: string },
            turn: { turnId: string },
            endpoint: unknown,
          ) => Promise<unknown>;
          const host = (svc as unknown as { hosts: { work: { selectFor: Select } } }).hosts.work;
          const realSelect = host.selectFor.bind(host);
          let selecting: { id: string; turnId: string } | null = null;
          let loweredOnSelect = false;
          host.selectFor = async (running, session, turn, endpoint) => {
            if (!loweredOnSelect && session.id === selecting?.id && turn.turnId !== selecting.turnId) {
              loweredOnSelect = true;
              expect(await svc.setPermissionMode(session.id, 'ask', false)).toEqual({ mode: 'ask' });
            }
            return realSelect(running, session, turn, endpoint);
          };
          const lateMove = await send('pdfnot', GEMMA.name, 'bypass');
          selecting = { id: lateMove.id, turnId: lateMove.turnId };
          await until(() => loweredOnSelect, 10_000);
          await settle();
          host.selectFor = realSelect;
          expect(texts(lateMove.id)).toEqual(['pdfnot']);
          expect(interactionsOf(lateMove.page)).toEqual([]);
          expect(lateMove.page.texts('user')).toEqual(['pdfnot']);
          await svc.readPdf(lateMove.id, lateMove.pdf);
          await until(() => texts(lateMove.id).length === 2, 10_000);
          expect(texts(lateMove.id)).toEqual(['pdfnot', readPdfCommand(lateMove.pdf, 'pdfnot', 'report.pdf', 'you')]);

          const asked = await send('pdfnot', GEMMA.name, 'ask');
          await quiet(asked);
          expect(texts(asked.id)).toEqual(['pdfnot']);

          const forked = (await svc.fork(asked.id)).id;
          expect(await svc.setPermissionMode(forked, 'bypass', true)).toEqual({ mode: 'bypass' });
          const forkPage = new Page(svc, forked);
          const copied = await apiError(svc.readPdf(forked, asked.pdf));
          expect([copied.code, copied.message]).toEqual(['E_NOT_FOUND', 'No such PDF.']);
          const stage = await staged(svc, 3);
          await svc.prompt(forked, 'pdfnot', [tool(stage, 3)]);
          const noteOf = (session: string) =>
            prompts(fake)
              .filter((p) => p.sessionId === session)
              .flatMap((p) => p.prompt as Array<{ type: string; resource?: { text?: string } }>)
              .find((block) => block.type === 'resource')?.resource?.text ?? '';
          await until(() => texts(forked).length === 2, 10_000);
          expect(readPdfNote(noteOf(forked))).toMatchObject({ id: pdfToolId(stage, 3), session: forked });
          expect(readPdfNote(noteOf(asked.id))).toMatchObject({ id: asked.pdf, session: asked.id });
          expect(texts(forked)).toEqual(['pdfnot', readPdfCommand(pdfToolId(stage, 3), 'pdfnot', 'report.pdf', 'app')]);
          forkPage.detach();

          const other = await send('pdfnot', INKLING.name, 'bypass');
          await quiet(other);
          expect(texts(other.id)).toEqual(['pdfnot']);

          const own = await send('pdftool', GEMMA.name, 'bypass');
          await quiet(own);
          expect(texts(own.id)).toEqual(['pdftool']);
          const busy = await apiError(svc.readPdf(own.id, own.pdf));
          expect(busy.code).toBe('E_CONFLICT');
          expect(texts(own.id)).toEqual(['pdftool']);

          const called = await send('pdfcall', GEMMA.name, 'bypass');
          await quiet(called);
          expect(texts(called.id)).toEqual(['pdfcall']);

          const wrong = await send('pdfwrong', GEMMA.name, 'auto');
          await answered(wrong, 'allow-once');
          expect(texts(wrong.id)).toEqual(['pdfwrong', readPdfCommand(wrong.pdf, 'pdfwrong', 'report.pdf', 'app')]);

          const wrongDenied = await send('pdfwrong', GEMMA.name, 'auto');
          await answered(wrongDenied, 'reject-once');
          expect(wrongDenied.page.ends()[0]).toMatchObject({ outcome: 'denied' });
          expect(texts(wrongDenied.id)).toEqual(['pdfwrong']);
          for (const p of [skipped, auto, movedDown, lateMove, asked, other, own, called, wrong, wrongDenied])
            p.page.detach();
        },
        { endpointModels: [GEMMA, INKLING] },
      );
    }, 90_000);

    // Mutation: Read it now with the tool off once the person's file took its name → red.
    it("Work: a file of the person's named like the tool's turns it off, before any agent start", async () => {
      await withWork(
        'work-always',
        async (fake, project) => {
          const first = fake.service;
          const stage = await staged(first, 2);
          const sent = await first.create({ mode: 'work', text: 'read', cwd: project, attachments: [tool(stage, 2)] });
          const page = new Page(first, sent.conversation.id);
          await until(() => {
            const head = page.events[0];
            return page.ends().length > 0 || (head?.type === 'sync' && head.turn === null);
          }, 5_000);
          page.detach();
          await first.close();
          const agents = join(fake.root, 'agent-home', 'agents');
          mkdirSync(agents, { recursive: true });
          writeFileSync(join(agents, 'pdf-reader.md'), '---\nname: pdf-reader\n---\nmine\n');
          const svc = fake.restart();
          expect((await svc.state()).pdfTool).toEqual({ chat: false, work: false });
          const off = await apiError(svc.readPdf(sent.conversation.id, pdfToolId(stage, 2)));
          expect(refusedAs(off)).toEqual(REFUSED);
          const again = await staged(svc, 2);
          const refused = await apiError(
            svc.create({ mode: 'work', text: 'read', cwd: project, attachments: [tool(again, 2)] }),
          );
          expect(refusedAs(refused)).toEqual(REFUSED);
          expect(fake.sent().filter((m) => m.method === 'session/new')).toHaveLength(1);
        },
        { endpointModels: [SEES], mutate: scripted },
      );
    });
  });

  const FULL = [
    'E_CONFLICT',
    'pictures-full',
    'This chat has too many pictures to keep. Start a new chat to send more.',
  ];

  // Mutation: PDF pages left out of the picture room → the PDF goes out.
  it('the pages of a PDF count in the stored pictures a chat can keep', async () => {
    const cap = 1024;
    const fake = fakeService('chat-basic', { mutate: scripted, endpointModels: [SEES], maxPicturesBase64: cap });
    try {
      const svc = fake.service;
      const big: AttachmentInput = {
        kind: 'image',
        mediaType: 'image/png',
        data: `iVBO${'A'.repeat(cap - 8)}`,
      };
      const { conversation } = await svc.create({ mode: 'chat', text: 'a big one', attachments: [big] });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 10_000);
      expect(refusedAs(await apiError(svc.prompt(conversation.id, 'one page more', [pdf(1)])))).toEqual(FULL);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: PDF pages left out of the images a chat holds → the 65th image goes out.
  it('a message that would bring the images a chat holds past 64 is refused; 64 go', async () => {
    const fake = fakeService('chat-basic', { mutate: scripted, endpointModels: [SEES] });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'first', attachments: [pdf(20)] });
      const page = new Page(svc, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      for (const [n, text] of ['second', 'third'].entries()) {
        await svc.prompt(conversation.id, text, [pdf(19, 'b.pdf'), PICTURE]);
        await until(() => page.ends().length === n + 2, 5_000);
      }
      expect(refusedAs(await apiError(svc.prompt(conversation.id, 'five more', [pdf(5)])))).toEqual(FULL);
      await svc.prompt(conversation.id, 'four more', [pdf(4)]);
      await until(() => page.ends().length === 4, 5_000);
      expect(refusedAs(await apiError(svc.prompt(conversation.id, 'one more', [PICTURE])))).toEqual(FULL);
      await svc.prompt(conversation.id, 'text still goes', [FILE]);
      await until(() => page.ends().length === 5, 5_000);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('Chat service — the host stream and redaction', () => {
  it('a prompt title cut inside the inference token shows no part of it', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({
        mode: 'chat',
        text: `see ${'x'.repeat(40)}${INFERENCE_TOKEN} now`,
      });
      expect(conversation.title.endsWith('…')).toBe(true);
      expect(conversation.title).not.toContain(INFERENCE_TOKEN.slice(0, 4));
    });
  });

  it('starts the host stream with a complete state and reports a new conversation', async () => {
    await withService(async (fake) => {
      const events: ChatHostEvent[] = [];
      const off = fake.service.watchHost((e) => events.push(e));
      await until(() => events.length >= 1);
      const first = events[0]!;
      expect(first.type).toBe('state');
      if (first.type !== 'state') return;
      expect(first.state).toMatchObject({ inference: 'ready', defaultModelId: 'qwen3.5-4b' });
      expect(first.state.models.map((m) => m.id)).toEqual(['qwen3.5-4b', 'qwen3.5-9b']);
      expect(first.state.agent).toMatchObject({ binary: 'ok', source: 'test', versionVerified: false });
      await fake.service.create({ mode: 'chat', text: 'hi' });
      await until(() => events.some((e) => e.type === 'conversations-changed'));
      await until(() => events.some((e) => e.type === 'state' && e.state.modes.chat.state === 'ready'));
      expect(JSON.stringify(events)).not.toContain(INFERENCE_TOKEN);
      off();
    });
  });

  it('redacts the inference token from streamed text, also when it is split over chunks', async () => {
    await withService(
      async (fake) => {
        const { conversation } = await fake.service.create({ mode: 'chat', text: 'tell me' });
        const page = new Page(fake.service, conversation.id);
        await until(() => page.ends().length === 1, 5_000);
        const wire = JSON.stringify(page.events);
        expect(wire).not.toContain(INFERENCE_TOKEN);
        expect(wire).not.toContain('INFERENCE-0123');
        expect(page.texts('assistant')).toEqual(['The token is [redacted] ok']);
      },
      (scenario) => {
        const half = Math.floor(INFERENCE_TOKEN.length / 2);
        scenario.prompts = [
          {
            match: '',
            steps: [
              { text: `The token is ${INFERENCE_TOKEN.slice(0, half)}`, chunk: 100 },
              { text: `${INFERENCE_TOKEN.slice(half)} ok`, chunk: 100 },
              { end: { stopReason: 'end_turn' } },
            ],
          },
        ];
      },
    );
  });

  it('redacts the inference token and a provider key from the first sync of a cold stream', async () => {
    const key = 'exa-KEY-0123456789abcdefghij';
    const secrets = new MemorySecretStore();
    await secrets.set('exa', key);
    const fake = fakeService('chat-basic', {
      providers: (paths) => memoryProviders(paths, { secrets }),
      mutate: (scenario) => {
        scenario.prompts = [
          {
            match: '',
            steps: [
              { text: `token ${INFERENCE_TOKEN} key ${key} end`, chunk: 200 },
              { end: { stopReason: 'end_turn' } },
            ],
          },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'tell me' });
      const onDisk = (): string => JSON.stringify(fake.reader.detail(conversation.id)?.transcript ?? []);
      await until(() => onDisk().includes(' end'), 5_000);
      expect(onDisk()).toContain(INFERENCE_TOKEN);
      await fake.service.close();

      const svc2 = fake.restart();
      const events: ConversationEvent[] = [];
      const off = svc2.watchConversation(conversation.id, (event) => events.push(event));
      await until(() => events.length >= 1, 5_000);
      off();
      expect(events[0]!.type).toBe('sync');
      const wire = JSON.stringify(events[0]);
      expect(wire).not.toContain(INFERENCE_TOKEN);
      expect(wire).not.toContain(key);
      expect(wire).toContain('[redacted]');
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('a provider key the Keychain gives only after the first sync is redacted by a fresh sync', async () => {
    const key = 'exa-KEY-late-0123456789abcdef';
    const inner = new MemorySecretStore();
    await inner.set('exa', key);
    let gate: Promise<void> | null = null;
    let open = (): void => {};
    const secrets: SecretStore = {
      get: async (id) => {
        if (id === 'exa' && gate !== null) await gate;
        return inner.get(id);
      },
      set: (id, value) => inner.set(id, value),
      delete: (id) => inner.delete(id),
    };
    const fake = fakeService('chat-basic', {
      providers: (paths) =>
        new ProvidersService({ paths, secrets, fetch: NO_NETWORK, workPath: () => '', keychainCallMs: 20 }),
      mutate: (scenario) => {
        scenario.prompts = [
          { match: '', steps: [{ text: `key ${key} end`, chunk: 200 }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'tell me' });
      const onDisk = (): string => JSON.stringify(fake.reader.detail(conversation.id)?.transcript ?? []);
      await until(() => onDisk().includes(' end'), 5_000);
      await fake.service.close();

      gate = new Promise<void>((resolve) => {
        open = resolve;
      });
      const svc2 = fake.restart();
      const page = new Page(svc2, conversation.id);
      await until(() => page.events.length >= 1, 5_000);
      expect(page.events[0]!.type).toBe('sync');
      // The Keychain had not answered: the first sync could not know the key.
      expect(JSON.stringify(page.events[0])).toContain(key);
      open();
      await until(() => page.events.length >= 2, 5_000);
      const syncs = page.events.filter((event) => event.type === 'sync');
      expect(syncs.length).toBeGreaterThanOrEqual(2);
      expect(JSON.stringify(syncs.at(-1))).not.toContain(key);
      expect(JSON.stringify(page.state)).not.toContain(key);
      expect(page.resyncs).toBe(0);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('the needles of a session text route come with what is known when the Keychain does not answer in time', async () => {
    let open = (): void => {};
    const gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    const inner = new MemorySecretStore();
    await inner.set('exa', 'exa-KEY-slow-0123456789');
    const secrets: SecretStore = {
      get: async (id) => {
        await gate;
        return inner.get(id);
      },
      set: (id, value) => inner.set(id, value),
      delete: (id) => inner.delete(id),
    };
    const fake = fakeService('chat-basic', {
      providers: (paths) =>
        new ProvidersService({ paths, secrets, fetch: NO_NETWORK, workPath: () => '', keychainCallMs: 600_000 }),
      needlesWaitMs: 20,
    });
    try {
      expect(await fake.service.redactionNeedles()).toEqual([INFERENCE_TOKEN]);
      open();
      let needles: string[] = [];
      await until(() => {
        void fake.service.redactionNeedles().then((n) => {
          needles = n;
        });
        return needles.length === 2;
      }, 5_000);
      expect(needles).toEqual([INFERENCE_TOKEN, 'exa-KEY-slow-0123456789']);
    } finally {
      open();
      await fake.close();
    }
  }, 10_000);

  it('blocks Chat when the bundled skill does not match its manifest', async () => {
    const fake = fakeService('chat-basic', { skillsDir: join(import.meta.dirname, 'fixtures', 'no-such-skills') });
    try {
      const error = await apiError(fake.service.create({ mode: 'chat', text: 'hi' }));
      expect([error.code, error.reason]).toEqual(['E_UNAVAILABLE', 'skill-bundle-corrupt']);
      const state = await fake.service.state();
      expect(state.modes.chat).toMatchObject({ state: 'blocked', failure: { code: 'skill-bundle-corrupt' } });
      expect(fake.records()).toEqual([]);
    } finally {
      await fake.close();
    }
  });

  it('installs the Chat skill before the first session and lists it in the session request', async () => {
    await withService(async (fake) => {
      await fake.service.create({ mode: 'chat', text: 'hi' });
      const skill = join(fake.paths.plugins, 'chat-web', 'skills', 'web-tools', 'SKILL.md');
      expect(readFileSync(skill, 'utf8').startsWith('---\nname: web-tools\n')).toBe(true);
      const created = fake.sent().find((m) => m.method === 'session/new') as Json;
      expect(((created.params as Json)._meta as Json).pluginDirs).toEqual([join(fake.paths.plugins, 'chat-web')]);
    });
  });
});

/** A Work service on the fake agent with a project folder inside the throwaway home. */
async function withWork(
  name: string,
  run: (fake: FakeService, project: string) => Promise<void>,
  options: Parameters<typeof fakeService>[1] = {},
): Promise<void> {
  const fake = fakeService(name, options);
  const project = join(fake.root, 'home', 'code', 'app');
  mkdirSync(project, { recursive: true });
  try {
    await run(fake, project);
  } finally {
    await fake.close();
  }
}

/** Every card the page was given, by a sync or as it opened, in order and once each. */
function interactionsOf(page: Page): Interaction[] {
  const seen = new Map<string, Interaction>();
  for (const e of page.events) {
    const cards = e.type === 'interaction' ? [e.interaction] : e.type === 'sync' ? e.interactions : [];
    for (const card of cards) if (!seen.has(card.id)) seen.set(card.id, card);
  }
  return [...seen.values()];
}

function workIndex(fake: FakeService): string[] {
  try {
    const raw = JSON.parse(readFileSync(fake.paths.conversations, 'utf8')) as { work: Array<{ id: string }> };
    return raw.work.map((e) => e.id);
  } catch {
    return [];
  }
}

describe('Chat service — Work', () => {
  it('refuses every folder the rule refuses with its typed reason and never reaches the agent', async () => {
    await withWork('work-permission', async (fake) => {
      const file = join(fake.root, 'home', 'file.txt');
      writeFileSync(file, 'x');
      const locked = join(fake.root, 'home', 'locked');
      mkdirSync(locked);
      chmodSync(locked, 0o000);
      const cases: Array<[string, CwdProblem]> = [
        ['code/app', 'not-absolute'],
        [join(fake.root, 'home', 'gone'), 'missing'],
        [file, 'not-directory'],
        [locked, 'unreadable'],
        ['/', 'root'],
        [join(fake.root, 'home'), 'home'],
        [join(fake.root, 'home', '.'), 'home'],
        [fake.paths.root, 'app-data'],
        [join(fake.root, 'agent-home'), 'app-data'],
        [fake.root, 'home'],
        [dirname(fake.root), 'home'],
      ];
      // macOS `/System/Volumes/Data` firmlink refusal cases live in
      // `chat-work-landing.test.ts` (`darwin-ts` lane) so macOS CI always asserts them.
      mkdirSync(fake.paths.root, { recursive: true });
      mkdirSync(join(fake.root, 'agent-home'), { recursive: true });
      try {
        for (const [cwd, problem] of cases) {
          const error = await apiError(fake.service.create({ mode: 'work', text: 'go', cwd }));
          expect([cwd, error.code, error.reason, error.message]).toEqual([
            cwd,
            'E_BAD_REQUEST',
            'cwd-invalid',
            CWD_PROBLEM_MESSAGES[problem],
          ]);
        }
        const none = await apiError(fake.service.create({ mode: 'work', text: 'go' }));
        expect(none.reason).toBe('cwd-invalid');
      } finally {
        chmodSync(locked, 0o755);
      }
      expect(fake.sent().filter((m) => m.method === 'session/new')).toEqual([]);
    });
  });

  it("a service given its own home takes no folder rule from the user's home", async () => {
    const user = mkdtempSync(join(tmpdir(), 'chat-user-home-'));
    const folder = join(user, 'work');
    mkdirSync(folder);
    symlinkSync(folder, join(user, '.mlx-node'));
    const before = process.env.HOME;
    process.env.HOME = user;
    try {
      await withWork('work-always', async (fake) => {
        process.env.HOME = before;
        const { conversation } = await fake.service.create({ mode: 'work', text: 'list', cwd: folder });
        expect(conversation.cwd).toBe(realpathSync(folder));
      });
    } finally {
      process.env.HOME = before;
      rmSync(user, { recursive: true, force: true });
    }
  });

  it('creates in the resolved folder, writes the index, and answers a card through the service', async () => {
    await withWork('work-always', async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'list', cwd: join(project, '..', 'app') });
      expect(conversation).toMatchObject({ mode: 'work', cwd: realpathSync(project) });
      expect(workIndex(fake)).toEqual([conversation.id]);
      const created = fake.sent().find((m) => m.method === 'session/new') as Json;
      expect((created.params as Json).cwd).toBe(realpathSync(project));
      const page = new Page(svc, conversation.id);
      await until(() => interactionsOf(page).length === 1, 5_000);
      const card = interactionsOf(page)[0]!;
      const notOffered = await apiError(svc.answer(conversation.id, card.id, { optionId: 'enable-always-approve' }));
      expect([notOffered.code, notOffered.message]).toEqual(['E_BAD_REQUEST', 'That option was not offered.']);
      expect((await apiError(svc.answer(conversation.id, card.id, { decision: 'approved' }))).code).toBe('E_NOT_FOUND');
      await svc.answer(conversation.id, card.id, { optionId: 'always-allow' });
      await until(() => page.lastOutcome() !== undefined, 5_000);
      expect(page.lastOutcome()).toBe('completed');
      expect(page.events.some((e) => e.type === 'interaction-closed' && e.reason === 'answered')).toBe(true);
      expect((await apiError(svc.answer(conversation.id, card.id, { optionId: 'allow-once' }))).code).toBe(
        'E_NOT_FOUND',
      );
      page.detach();
    });
  }, 20_000);

  it('a folder that went away gives a typed state and refuses the prompt until it is back', async () => {
    await withWork('work-plan', async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'plan', cwd: project });
      const page = new Page(svc, conversation.id);
      await until(() => interactionsOf(page).length === 1, 5_000);
      await svc.answer(conversation.id, interactionsOf(page)[0]!.id, { decision: 'approved' });
      await until(() => page.ends().length === 1, 5_000);
      rmSync(project, { recursive: true });
      const refused = await apiError(svc.prompt(conversation.id, 'again'));
      expect([refused.code, refused.reason, refused.message]).toEqual([
        'E_BAD_REQUEST',
        'cwd-invalid',
        'That folder does not exist any more.',
      ]);
      const infos = (): Array<string | undefined> =>
        page.events.flatMap((e) => (e.type === 'conversation' ? [e.conversation.cwdProblem] : []));
      await until(() => infos().includes('missing'));
      expect(fake.sent().filter((m) => m.method === 'session/prompt')).toHaveLength(1);
      page.detach();

      const late = new Page(svc, conversation.id);
      await until(() => late.events.length >= 1, 5_000);
      expect(JSON.stringify(late.events[0])).toContain('"cwdProblem":"missing"');
      late.detach();

      mkdirSync(project, { recursive: true });
      const back = new Page(svc, conversation.id);
      await svc.prompt(conversation.id, 'again');
      await until(() => interactionsOf(back).length === 1, 5_000);
      await svc.answer(conversation.id, interactionsOf(back)[0]!.id, { decision: 'approved' });
      await until(() => back.ends().length === 1, 5_000);
      expect(JSON.stringify(back.events[0])).not.toContain('cwdProblem');
      back.detach();
    });
  }, 30_000);

  it('renames, switches model and deletes through the agent; the index entry goes with it', async () => {
    await withWork('work-plan', async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'plan', cwd: project });
      const page = new Page(svc, conversation.id);
      await until(() => interactionsOf(page).length === 1, 5_000);
      await svc.answer(conversation.id, interactionsOf(page)[0]!.id, { decision: 'rejected', feedback: 'No.' });
      await until(() => page.ends().length === 1, 5_000);
      expect(await svc.patch(conversation.id, { title: 'My work' })).toMatchObject({ title: 'My work', mode: 'work' });
      expect(fake.sent().some((m) => m.method === '_x.ai/session/rename')).toBe(true);
      expect(await svc.patch(conversation.id, { modelId: 'qwen3.5-9b' })).toMatchObject({ modelId: 'qwen3.5-9b' });
      expect(workIndex(fake)).toEqual([conversation.id]);
      await svc.delete(conversation.id);
      const del = fake.sent().find((m) => m.method === '_x.ai/session/delete') as Json;
      expect(del.params).toEqual({ sessionId: conversation.id, cwd: realpathSync(project) });
      expect(workIndex(fake)).toEqual([]);
      expect(page.events.at(-1)).toMatchObject({ type: 'gone', reason: 'deleted' });
      expect((await apiError(svc.prompt(conversation.id, 'hi'))).code).toBe('E_NOT_FOUND');
      page.detach();
    });
  }, 20_000);

  it('with no window holding the host stream, an open card is denied once the wait is over', async () => {
    await withWork(
      'work-permission',
      async (fake, project) => {
        const svc = fake.service;
        // The page attaches asynchronously; if the deny fired first it would see only the ended turn in its sync.
        const hostOff = svc.watchHost(() => {});
        const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
        const page = new Page(svc, conversation.id);
        await until(() => interactionsOf(page).length === 1, 5_000);
        await svc.answer(conversation.id, interactionsOf(page)[0]!.id, { optionId: 'allow-once' });
        await until(() => page.lastOutcome() !== undefined, 5_000);
        hostOff();

        await svc.prompt(conversation.id, 'again');
        await until(() => page.ends().length === 2, 5_000);
        expect(page.lastOutcome()).toBe('denied');
        expect(page.events.some((e) => e.type === 'interaction-closed' && e.reason === 'expired')).toBe(true);
        page.detach();
      },
      { pageGoneMs: 30 },
    );
  }, 20_000);

  it('the host state counts the open cards of a conversation; a card the agent gives up on is withdrawn', async () => {
    const gate = join(mkdtempSync(join(tmpdir(), 'mlx-svc-ask-')), 'give-up');
    onTestFinished(() => rmSync(dirname(gate), { recursive: true, force: true }));
    await withWork(
      'work-permission',
      async (fake, project) => {
        const svc = fake.service;
        const states: ChatHostEvent[] = [];
        const hostOff = svc.watchHost((e) => states.push(e));
        const asksOf = (id: string): number | undefined => {
          for (let i = states.length - 1; i >= 0; i--) {
            const e = states[i]!;
            if (e.type === 'state') return e.state.active.find((a) => a.conversationId === id)?.asks;
          }
          return undefined;
        };
        const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
        const page = new Page(svc, conversation.id);
        await until(() => interactionsOf(page).length === 2, 5_000);
        await until(() => asksOf(conversation.id) === 2, 5_000);
        writeFileSync(gate, '');
        await until(() => page.state?.withdrawn === 1, 5_000);
        expect(page.state?.interactions).toHaveLength(1);
        await until(() => asksOf(conversation.id) === 1, 5_000);
        await svc.answer(conversation.id, page.state!.interactions[0]!.id, { optionId: 'allow-once' });
        await until(() => page.lastOutcome() !== undefined, 5_000);
        await until(() => asksOf(conversation.id) === 0, 5_000);
        expect(page.state?.withdrawn).toBe(1);
        hostOff();
        page.detach();
      },
      {
        mutate: (scenario) => {
          const prompt = (scenario.prompts as Json[])[0]!;
          prompt.steps = [{ giveUp: { background: true, on: gate, id: 'toolu_gone' } }, ...(prompt.steps as Json[])];
        },
      },
    );
  }, 20_000);

  it('while a window holds the host stream, the card waits for the user', async () => {
    await withWork(
      'work-permission',
      async (fake, project) => {
        const svc = fake.service;
        const hostOff = svc.watchHost(() => {});
        const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
        const page = new Page(svc, conversation.id);
        await until(() => interactionsOf(page).length === 1, 5_000);
        await svc.answer(conversation.id, interactionsOf(page)[0]!.id, { optionId: 'allow-once' });
        await until(() => page.lastOutcome() !== undefined, 5_000);
        expect(page.lastOutcome()).toBe('completed');

        await svc.prompt(conversation.id, 'again');
        await until(() => interactionsOf(page).length === 2, 5_000);
        hostOff();
        await until(() => page.ends().length === 2, 5_000);
        expect(page.lastOutcome()).toBe('denied');
        page.detach();
      },
      { pageGoneMs: 1 },
    );
  }, 20_000);

  it('a command that prints a provider key: the page gets the notice once and never the key', async () => {
    const key = 'exa-KEY-printed-0123456789';
    const secrets = new MemorySecretStore();
    await secrets.set('exa', key);
    let providers: ProvidersService | null = null;
    await withWork(
      'work-key-printed',
      async (fake, project) => {
        await providers!.select({ search: 'exa' });
        const svc = fake.service;
        const { conversation } = await svc.create({ mode: 'work', text: 'print', cwd: project });
        const page = new Page(svc, conversation.id);
        await until(() => interactionsOf(page).length === 1, 5_000);
        await svc.answer(conversation.id, interactionsOf(page)[0]!.id, { optionId: 'allow-once' });
        await until(() => page.lastOutcome() !== undefined, 5_000);
        const notices = page.events.filter((e) => e.type === 'notice');
        expect(notices.map((e) => e.type === 'notice' && e.notice)).toEqual([
          { code: 'key-printed', tone: 'danger', provider: 'exa' },
        ]);
        expect(JSON.stringify(page.events)).not.toContain(key);
        expect(JSON.stringify(page.events)).toContain('EXA_API_KEY=[redacted]');
        const start = fake.records().find((r) => r.kind === 'start') as Json;
        expect(start.envNames).toContain('EXA_API_KEY');
        expect(JSON.stringify(start)).not.toContain(key);
        page.detach();
      },
      {
        providers: (paths) => {
          providers = memoryProviders(paths, { secrets });
          return providers;
        },
      },
    );
  }, 20_000);

  it('a card title and a plan with the inference token across the cut carry no part of it', async () => {
    const T = INFERENCE_TOKEN;
    const prefix = 'Execute `curl -H "Authorization: Bearer ';
    const command = `curl -H "Authorization: Bearer ${'x'.repeat(200 - prefix.length - (T.length - 1))}${T}" http://127.0.0.1:9/v1/models`;
    const title = `Execute \`${command}\``;
    expect(title.indexOf(T)).toBe(200 - (T.length - 1));
    const plan = `${'p'.repeat(50_000 - (T.length - 1))}${T} end`;
    const cases: Array<[string, (scenario: Json) => void]> = [
      [
        'work-permission',
        (scenario) => {
          const step = ((scenario.prompts as Json[])[0]!.steps as Json[])[0]!.permission as Json;
          step.title = title;
          (step.rawInput as Json).command = command;
        },
      ],
      [
        'work-plan',
        (scenario) => {
          const step = ((scenario.prompts as Json[])[0]!.steps as Json[]).find((s) => s.exitPlan !== undefined)!;
          (step.exitPlan as Json).planContent = plan;
        },
      ],
    ];
    for (const [name, mutate] of cases) {
      await withWork(
        name,
        async (fake, project) => {
          const { conversation } = await fake.service.create({ mode: 'work', text: 'go', cwd: project });
          const page = new Page(fake.service, conversation.id);
          await until(() => interactionsOf(page).length === 1, 5_000);
          const card = interactionsOf(page)[0]!;
          if (card.kind === 'permission') expect(card.title.startsWith(`${prefix}xxx`)).toBe(true);
          else expect(card.kind === 'plan' && card.plan.startsWith('ppp')).toBe(true);
          expect(JSON.stringify(page.events)).not.toContain(T.slice(0, 8));
          await fake.service.cancel(conversation.id);
          await until(() => page.lastOutcome() !== undefined, 5_000);
          page.detach();
        },
        { mutate },
      );
    }
  }, 30_000);

  describe('a folder swapped for the home folder after its check', () => {
    let gate: Promise<void> | null = null;
    let open = (): void => {};
    let reached = false;
    /** The next `providers.settled()` waits for `open()`. */
    const arm = (): void => {
      reached = false;
      gate = new Promise<void>((resolve) => {
        open = resolve;
      });
    };
    const options = {
      providers: (paths: ChatPaths) => {
        const providers = memoryProviders(paths);
        const settled = providers.settled.bind(providers);
        providers.settled = async () => {
          const held = gate;
          gate = null;
          if (held !== null) {
            reached = true;
            await held;
          }
          return settled();
        };
        return providers;
      },
      mutate: (scenario: Json) => {
        scenario.prompts = [
          { match: 'hello', steps: [{ text: 'hi' }, { end: { stopReason: 'end_turn' } }] },
          ...(scenario.prompts as Json[]),
        ];
      },
    };
    const swap = (fake: FakeService, project: string): void => {
      rmSync(project, { recursive: true });
      symlinkSync(join(fake.root, 'home'), project);
    };
    const sentCwds = (fake: FakeService, method: string): unknown[] =>
      fake
        .sent()
        .filter((m) => m.method === method)
        .map((m) => (m.params as Json).cwd);

    it('during create is refused before session/new', async () => {
      await withWork(
        'work-edit',
        async (fake, project) => {
          arm();
          const creating = apiError(fake.service.create({ mode: 'work', text: 'go', cwd: project }));
          await until(() => reached, 5_000);
          swap(fake, project);
          open();
          expect((await creating).reason).toBe('cwd-invalid');
          expect(sentCwds(fake, 'session/new')).toEqual([]);
        },
        options,
      );
    }, 20_000);

    it('before a cold load is refused before session/load', async () => {
      await withWork(
        'work-edit',
        async (fake, project) => {
          const real = realpathSync(project);
          const { conversation } = await fake.service.create({ mode: 'work', text: 'hello', cwd: project });
          const page = new Page(fake.service, conversation.id);
          await until(
            () => page.events.some((e) => (e.type === 'turn' && e.turn === null) || (e.type === 'sync' && e.lastEnd)),
            5_000,
          );
          const before = page.ends().length;
          await fake.service.agentAction({ action: 'restart', mode: 'work' });
          arm();
          const prompting = fake.service.prompt(conversation.id, 'hello');
          await until(() => reached, 5_000);
          swap(fake, project);
          open();
          await prompting;
          await until(() => page.ends().length === before + 1, 10_000);
          expect(page.ends().at(-1)).toMatchObject({ failure: { code: 'cwd-invalid' } });
          expect(sentCwds(fake, 'session/new')).toEqual([real]);
          expect(sentCwds(fake, 'session/load')).toEqual([]);
          page.detach();
        },
        options,
      );
    }, 20_000);
  });

  it('gives Work the Firecrawl folder and key only once the firecrawl CLI is on its PATH', async () => {
    const secrets = new MemorySecretStore();
    await secrets.set('deepseek', 'ds-key-00001');
    await secrets.set('firecrawl', 'fc-key-000001');
    let path = '';
    let clock = 1_000;
    let providers: ProvidersService | null = null;
    await withWork(
      'work-permission',
      async (fake, project) => {
        await providers!.select({ search: 'deepseek', firecrawl: true });
        const svc = fake.service;
        const hostOff = svc.watchHost(() => {});
        const run = async (): Promise<{ dirs: unknown; env: unknown; url: unknown }> => {
          const before = fake.sent().filter((m) => m.method === 'session/new').length;
          const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
          const page = new Page(svc, conversation.id);
          await until(() => interactionsOf(page).length === 1, 5_000);
          await svc.answer(conversation.id, interactionsOf(page)[0]!.id, { optionId: 'reject-once' });
          await until(() => page.lastOutcome() !== undefined, 5_000);
          page.detach();
          const created = fake.sent().filter((m) => m.method === 'session/new')[before] as Json;
          const start = fake
            .records()
            .filter((r) => r.kind === 'start')
            .at(-1) as Json;
          return {
            dirs: ((created.params as Json)._meta as Json).pluginDirs,
            env: start.envNames,
            url: start.firecrawlApiUrl,
          };
        };
        const first = await run();
        expect(first.dirs).toEqual([fake.paths.workGuard, join(fake.paths.plugins, 'deepseek')]);
        expect(first.env).toContain('DEEPSEEK_API_KEY');
        expect(first.env).not.toContain('FIRECRAWL_API_KEY');
        expect(first.url).toBeNull();

        const bin = join(fake.root, 'bin');
        mkdirSync(bin);
        writeFileSync(join(bin, 'firecrawl'), '#!/bin/sh\n');
        chmodSync(join(bin, 'firecrawl'), 0o755);
        path = bin;
        clock += WORK_PATH_ASK_GAP_MS;
        await providers!.list();
        const second = await run();
        expect(second.dirs).toEqual([
          fake.paths.workGuard,
          join(fake.paths.plugins, 'deepseek'),
          join(fake.paths.plugins, 'firecrawl'),
        ]);
        expect(second.env).toEqual(expect.arrayContaining(['DEEPSEEK_API_KEY', 'FIRECRAWL_API_KEY']));
        expect(second.url).toBe('https://api.firecrawl.dev');
        expect(JSON.stringify(fake.records())).not.toContain('fc-key-000001');
        hostOff();
      },
      {
        providers: (paths) => {
          providers = new ProvidersService({
            paths,
            secrets,
            fetch: NO_NETWORK,
            workPath: () => path,
            now: () => clock,
          });
          return providers;
        },
      },
    );
  }, 30_000);
});

describe('Chat service — a Work workflow run', () => {
  /** `work-workflow-run` with the step's options merged in. */
  const workflow =
    (options: Json) =>
    (scenario: Json): void => {
      const steps = (scenario.prompts as Json[])[0]!.steps as Json[];
      steps[1] = { workflow: { ...(steps[1]!.workflow as Json), ...options } };
    };

  it('the page gets the runs; delete, a model change and unload wait for it; a rename does not; Stop run ends it', async () => {
    const flags = mkdtempSync(join(tmpdir(), 'mlx-wf-'));
    const childOn = join(flags, 'child');
    const snapshot = live({
      processes: [liveProcess({ pid: process.pid, models: [liveModel('qwen3.5-4b')] })],
    }) as unknown as RuntimeLive;
    const busy: string[][] = [];
    let service: ChatService | null = null;
    try {
      await withWork(
        'work-workflow-run',
        async (fake, project) => {
          service = fake.service;
          const svc = fake.service;
          const { conversation } = await svc.create({ mode: 'work', text: 'research', cwd: project });
          const id = conversation.id;
          const page = new Page(svc, id);
          await until(() => interactionsOf(page).length === 1, 5_000);
          await svc.answer(id, interactionsOf(page)[0]!.id, { optionId: 'allow-once' });
          await until(() => page.lastOutcome() !== undefined, 5_000);
          expect(page.lastOutcome()).toBe('completed');
          const runs = (): unknown[] => page.events.flatMap((e) => (e.type === 'runs' ? [e.runs] : []));
          expect(runs()).toEqual([[runView('wf_1')]]);
          expect(svc.busyConversations()).toEqual([id]);
          expect(svc.closingConversations()).toEqual([id]);
          const state = await svc.state();
          expect(state.active).toEqual([
            {
              conversationId: id,
              mode: 'work',
              busy: false,
              waiting: false,
              asks: 0,
              background: true,
              helpers: false,
              tasks: 0,
              op: null,
            },
          ]);
          expect(state.modes.work.background).toBe(1);
          const late = new Page(svc, id);
          await until(() => late.events.length > 0, 5_000);
          const sync = late.events[0]!;
          expect(sync.type === 'sync' ? sync.runs : null).toEqual([runView('wf_1')]);
          late.detach();
          expect((await apiError(svc.delete(id))).reason).toBe('conversation-busy');
          expect((await apiError(svc.patch(id, { modelId: 'qwen3.5-9b' }))).reason).toBe('conversation-busy');
          expect((await apiError(svc.unloadModels())).reason).toBe('engine-busy');
          expect((await svc.patch(id, { title: 'Research' })).title).toBe('Research');
          const missing = await apiError(svc.stopRun(id, 'wf_none'));
          expect([missing.code, missing.message]).toEqual(['E_NOT_FOUND', 'No such run.']);
          expect(await svc.stopRun(id, 'wf_1')).toEqual({});
          await until(() => runs().length === 4, 5_000);
          const stopped = { live: false, status: 'cancelled', by: 'you' } as const;
          expect(runs()).toEqual([
            [runView('wf_1')],
            [runView('wf_1', { stopping: true })],
            [runView('wf_1', { ...stopped, stopping: true })],
            [runView('wf_1', stopped)],
          ]);
          await until(() => svc.busyConversations().length === 0, 5_000);
          expect(busy.at(-1)).toEqual([]);
          expect(page.state?.entries.filter((e) => e.kind === 'workflow').map((e) => e.status)).toEqual(['cancelled']);
          page.detach();
          await svc.delete(id);
        },
        {
          mutate: workflow({ childOn }),
          live: { subscribe: () => () => {}, readNow: async () => snapshot },
          onBusyChange: () => busy.push(service?.busyConversations() ?? []),
        },
      );
    } finally {
      rmSync(flags, { recursive: true, force: true });
    }
  }, 20_000);

  it('a stop the agent rejects fails the call and gives the run back', async () => {
    const flags = mkdtempSync(join(tmpdir(), 'mlx-wf-'));
    try {
      await withWork(
        'work-workflow-run',
        async (fake, project) => {
          const svc = fake.service;
          const { conversation } = await svc.create({ mode: 'work', text: 'research', cwd: project });
          const id = conversation.id;
          const page = new Page(svc, id);
          await until(() => interactionsOf(page).length === 1, 5_000);
          await svc.answer(id, interactionsOf(page)[0]!.id, { optionId: 'allow-once' });
          await until(() => page.lastOutcome() !== undefined, 5_000);
          const runs = (): unknown[] => page.events.flatMap((e) => (e.type === 'runs' ? [e.runs] : []));
          expect(runs()).toEqual([[runView('wf_1')]]);
          const error = await apiError(svc.stopRun(id, 'wf_1'));
          expect([error.code, error.message]).toEqual(['E_CONFLICT', 'The run could not be stopped.']);
          expect(runs()).toEqual([[runView('wf_1')], [runView('wf_1', { stopping: true })], [runView('wf_1')]]);
          page.detach();
        },
        { mutate: workflow({ childOn: join(flags, 'child'), stopRejects: true }) },
      );
    } finally {
      rmSync(flags, { recursive: true, force: true });
    }
  }, 20_000);

  it('Stop run of a Chat conversation is 404', async () => {
    await withService(async (fake) => {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello' });
      const error = await apiError(fake.service.stopRun(conversation.id, 'wf_1'));
      expect([error.code, error.message]).toEqual(['E_NOT_FOUND', 'No such run.']);
    });
  }, 20_000);
});

describe('Chat service — background work (H-a)', () => {
  /** One prompt per word: its background step, then a short answer. */
  const prompts =
    (words: Record<string, Json>) =>
    (scenario: Json): void => {
      scenario.prompts = Object.entries(words).map(([match, step]) => ({
        match,
        steps: [step, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }],
      }));
    };

  /** The host state once `check` holds, polled every 10 ms for at most 5 s. */
  async function stateWhen(svc: ChatService, check: (state: ChatHostState) => boolean): Promise<ChatHostState> {
    const end = Date.now() + 5_000;
    for (;;) {
      const state = await svc.state();
      if (check(state)) return state;
      if (Date.now() > end) throw new Error('condition not reached');
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }

  it('a command holds delete, the quit guard and a settings change, not unload or a model change; a helper holds unload', async () => {
    const flags = mkdtempSync(join(tmpdir(), 'mlx-bg-'));
    const helperEnd = join(flags, 'helper-end');
    const snapshot = live({
      processes: [liveProcess({ pid: process.pid, models: [liveModel('qwen3.5-4b')] })],
    }) as unknown as RuntimeLive;
    let providers: ProvidersService | null = null;
    try {
      await withWork(
        'work-workflow-run',
        async (fake, project) => {
          const svc = fake.service;
          const { conversation } = await svc.create({ mode: 'work', text: 'cmd', cwd: project });
          const id = conversation.id;
          const page = new Page(svc, id);
          // The first answer may end before the page attaches: its sync carries the end.
          await until(() => page.state?.lastEnd?.outcome !== undefined, 5_000);
          expect(page.state?.lastEnd?.outcome).toBe('completed');
          await until(() => svc.busyConversations().includes(id), 5_000);
          const state = await svc.state();
          expect(state.active).toEqual([
            {
              conversationId: id,
              mode: 'work',
              busy: false,
              waiting: false,
              asks: 0,
              background: false,
              helpers: false,
              tasks: 1,
              op: null,
            },
          ]);
          expect(state.modes.work).toMatchObject({ background: 0, tasks: 1, engineLost: false });
          expect(state.providers.applyPending).toBe(false);
          expect((await apiError(svc.delete(id))).reason).toBe('conversation-busy');
          expect((await svc.patch(id, { modelId: 'qwen3.5-9b' })).modelId).toBe('qwen3.5-9b');
          expect(await svc.unloadModels()).toEqual({ unloaded: ['qwen3.5-4b'] });
          await providers!.putKey('exa', 'exa-KEY-0123456789abcdef');
          const stale = await stateWhen(svc, (s) => s.modes.work.stale);
          expect(stale.modes.work).toMatchObject({ state: 'ready', tasks: 1 });
          expect(stale.providers.applyPending).toBe(true);
          // The first answer's end is in the sync or is the first event: count from here.
          const endsBefore = page.ends().length;
          await svc.prompt(id, 'helper');
          await until(() => page.ends().length === endsBefore + 1, 5_000);
          expect(page.lastOutcome()).toBe('completed');
          await until(() => svc.busyConversations().includes(id), 5_000);
          const helper = (await svc.state()).active.find((a) => a.conversationId === id);
          expect(helper).toMatchObject({ background: true, helpers: true, tasks: 1 });
          expect((await apiError(svc.unloadModels())).reason).toBe('engine-busy');
          expect((await apiError(svc.patch(id, { modelId: 'qwen3.5-4b' }))).reason).toBe('conversation-busy');
          writeFileSync(helperEnd, '');
          await stateWhen(svc, (s) => s.active.some((a) => a.conversationId === id && !a.helpers));
          expect(await svc.unloadModels()).toEqual({ unloaded: ['qwen3.5-4b'] });
          expect(svc.busyConversations()).toEqual([id]);
          page.detach();
        },
        {
          mutate: prompts({
            cmd: { background: { wake: false } },
            helper: { helper: { endOn: helperEnd, wake: false } },
          }),
          live: { subscribe: () => () => {}, readNow: async () => snapshot },
          providers: (paths) => (providers = memoryProviders(paths)),
          engine: (r) => (r.path === UNLOAD_PATH ? { json: { unloaded: ['qwen3.5-4b'] } } : { json: {} }),
        },
      );
    } finally {
      rmSync(flags, { recursive: true, force: true });
    }
  }, 20_000);
});

describe('Chat service — what MAIN learns of turns and of earlier panel lives', () => {
  it('carries the earlier ends in the host state, and tells of each turn start and end at once', async () => {
    const appEnd = {
      trigger: 'no-ping' as const,
      decidedAt: 1_000,
      exitAt: 3_000,
      silentMs: 120_000,
      code: 0,
      convs: ['01a1d519-7b3c-7d4e-8f50-61728394a5b6'],
      cancelled: null,
    };
    const busy: string[][] = [];
    let service: ChatService | null = null;
    const fake = fakeService('chat-basic', {
      appEnds: () => Promise.resolve([appEnd]),
      onBusyChange: () => busy.push(service?.busyConversations() ?? []),
    });
    service = fake.service;
    try {
      expect((await fake.service.state()).appEnds).toEqual([appEnd]);
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      expect(busy).toEqual([[conversation.id], []]);
      expect(fake.service.busyConversations()).toEqual([]);
    } finally {
      await fake.close();
    }
  }, 20_000);

  it("the host's wake turn counts as running until its end line, and the page sees it as a turn with origin wake", async () => {
    const flags = mkdtempSync(join(tmpdir(), 'mlx-wake-'));
    const startOn = join(flags, 'start');
    const endOn = join(flags, 'end');
    const busy: string[][] = [];
    let service: ChatService | null = null;
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        const steps = (scenario.prompts as Array<{ steps: Array<Record<string, unknown>> }>)[0]!.steps;
        steps[steps.length - 1] = { end: { stopReason: 'end_turn', wake: { startOn, endOn } } };
      },
      onBusyChange: () => busy.push(service?.busyConversations() ?? []),
    });
    service = fake.service;
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello' });
      const page = new Page(fake.service, conversation.id);
      await until(() => busy.length === 2, 5_000);
      expect(busy).toEqual([[conversation.id], []]);
      writeFileSync(startOn, '');
      await until(() => busy.length === 3, 5_000);
      expect(busy[2]).toEqual([conversation.id]);
      expect(fake.service.busyConversations()).toEqual([conversation.id]);
      const wake = page.events.find((e) => e.type === 'turn' && e.turn?.origin === 'wake');
      expect(wake?.type === 'turn' ? wake.turn : null).toMatchObject({ phase: 'running', origin: 'wake' });
      expect((await apiError(fake.service.prompt(conversation.id, 'again'))).reason).toBe('conversation-busy');
      writeFileSync(endOn, '');
      await until(() => busy.length === 4, 5_000);
      expect(busy[3]).toEqual([]);
      expect(page.lastOutcome()).toBe('completed');
      page.detach();
    } finally {
      await fake.close();
      rmSync(flags, { recursive: true, force: true });
    }
  }, 20_000);

  it("a host wake whose lines share one stdout chunk with the turn reply shows after the typed turn's end", async () => {
    const flags = mkdtempSync(join(tmpdir(), 'mlx-wake-'));
    const endOn = join(flags, 'end');
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        const steps = (scenario.prompts as Array<{ steps: Array<Record<string, unknown>> }>)[0]!.steps;
        steps[steps.length - 1] = { end: { stopReason: 'end_turn', wake: { together: true, endOn } } };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      await until(() => page.state?.turn?.origin === 'wake', 5_000);
      expect(fake.service.busyConversations()).toEqual([conversation.id]);
      writeFileSync(endOn, '');
      await until(() => fake.service.busyConversations().length === 0, 5_000);
      await until(() => page.ends().length === 2, 5_000);
      expect(page.state?.turn).toBeNull();
      page.detach();
    } finally {
      await fake.close();
      rmSync(flags, { recursive: true, force: true });
    }
  }, 20_000);

  it('a failed ledger read leaves the host state with no ends', async () => {
    const fake = fakeService('chat-basic', { appEnds: () => Promise.reject(new Error('EACCES')) });
    try {
      expect((await fake.service.state()).appEnds).toEqual([]);
    } finally {
      await fake.close();
    }
  }, 20_000);
});

describe('Chat service — thinking level', () => {
  const FOUR = 'qwen3.5-4b';
  const NINE = 'qwen3.5-9b';
  /** The fake agent lists Off · Low · Medium · High for one model and Low · High for the other (an LFM2 row). */
  const levelled = (scenario: Json): void => {
    scenario.initialize = {
      models: [
        {
          modelId: FOUR,
          totalContextTokens: 32768,
          reasoningEfforts: [
            { value: 'none', label: 'Off' },
            { value: 'low', label: 'Low' },
            { value: 'medium', label: 'Medium' },
            { value: 'high', label: 'High', default: true },
          ],
        },
        {
          modelId: NINE,
          totalContextTokens: 32768,
          reasoningEfforts: [
            { value: 'low', label: 'Low' },
            { value: 'high', label: 'High', default: true },
          ],
        },
      ],
    };
    scenario.prompts = [
      {
        match: 'slow',
        steps: [{ text: 'Thinking', chunk: 3 }, { sleepMs: 1500 }, { end: { stopReason: 'end_turn' } }],
      },
      { match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] },
    ];
  };
  const levelsOf = async (svc: ChatService, id: string) => (await svc.state()).models.find((m) => m.id === id);
  const stored = (fake: FakeService, id: string): Json | null => {
    try {
      return JSON.parse(readFileSync(join(fake.paths.composer, `${id}.json`), 'utf8')) as Json;
    } catch {
      return null;
    }
  };
  const setValues = (fake: FakeService): unknown[] =>
    fake
      .sent()
      .filter((m) => m.method === 'session/set_config_option')
      .map((m) => (m.params as Json).value);

  it('the host state lists levels once an agent listed the model', async () => {
    const fake = fakeService('chat-basic', { mutate: levelled });
    try {
      expect(await levelsOf(fake.service, FOUR)).not.toHaveProperty('thinking');
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hi' });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      expect((await levelsOf(fake.service, FOUR))?.thinking).toEqual({
        levels: ['none', 'low', 'medium', 'high'],
        default: 'high',
      });
      expect((await levelsOf(fake.service, NINE))?.thinking).toEqual({ levels: ['low', 'high'], default: 'high' });
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('before any agent a served model has its family’s levels; an agent’s list replaces them', async () => {
    const INKLING = 'Inkling-2-Mini';
    const GEMMA = 'gemma-4-e4b';
    const NO_FAMILY = 'no-family';
    const CLAIRIC = 'clairic-1';
    const listing = (scenario: Json): void => {
      levelled(scenario);
      scenario.initialize = {
        models: [
          {
            modelId: FOUR,
            totalContextTokens: 32768,
            reasoningEfforts: [
              { value: 'low', label: 'Low' },
              { value: 'high', label: 'High', default: true },
            ],
          },
          // An agent build that lists no efforts: its word wins over the family's.
          { modelId: INKLING, totalContextTokens: 32768 },
        ],
      };
    };
    const fake = fakeService('chat-basic', {
      mutate: listing,
      endpointModels: [
        { ...endpointModel(FOUR, '/models/qwen3.5-4b'), modelType: 'qwen3_5' },
        { ...endpointModel(INKLING, '/models/inkling'), modelType: 'inkling2' },
        { ...endpointModel(GEMMA, '/models/gemma'), modelType: 'gemma4' },
        endpointModel(NO_FAMILY, '/models/plain'),
        { ...endpointModel(CLAIRIC, '/models/clairic'), modelType: 'clairic', kind: 'decision' },
      ],
    });
    try {
      const family = { levels: ['none', 'low', 'medium', 'high'], default: 'high' };
      expect((await levelsOf(fake.service, INKLING))?.thinking).toEqual(family);
      expect((await levelsOf(fake.service, FOUR))?.thinking).toEqual(family);
      expect((await levelsOf(fake.service, GEMMA))?.thinking).toBeNull();
      expect(await levelsOf(fake.service, NO_FAMILY)).not.toHaveProperty('thinking');
      expect((await fake.service.state()).models.map((m) => m.id)).not.toContain(CLAIRIC);

      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hi', modelId: FOUR });
      const page = new Page(fake.service, conversation.id);
      await until(() => page.ends().length === 1, 5_000);
      page.detach();
      expect((await levelsOf(fake.service, FOUR))?.thinking).toEqual({ levels: ['low', 'high'], default: 'high' });
      expect((await levelsOf(fake.service, INKLING))?.thinking).toBeNull();
      expect((await levelsOf(fake.service, GEMMA))?.thinking).toBeNull();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('create and patch store the level, also while an answer runs; the next answer sets it; a restart reads it', async () => {
    const fake = fakeService('chat-basic', { mutate: levelled });
    try {
      const svc = fake.service;
      const created = await svc.create({ mode: 'chat', text: 'hi', thinking: 'low' });
      const id = created.conversation.id;
      expect(created.conversation.thinking).toBe('low');
      const page = new Page(svc, id);
      await until(() => page.ends().length === 1, 5_000);
      expect(setValues(fake)).toEqual(['low']);
      await until(() => stored(fake, id)?.thinking === 'low');

      await svc.prompt(id, 'slow please');
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const patched = await svc.patch(id, { thinking: 'none' });
      expect(patched.thinking).toBe('none');
      expect(page.state!.conversation.thinking).toBe('none');
      expect(stored(fake, id)).toEqual({ queue: [], thinking: 'none' });
      await until(() => page.ends().length === 2, 5_000);
      expect(setValues(fake)).toEqual(['low']);
      await svc.prompt(id, 'again');
      await until(() => page.ends().length === 3, 5_000);
      expect(setValues(fake)).toEqual(['low', 'none']);
      page.detach();

      await svc.close();
      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      expect(cold.state!.conversation.thinking).toBe('none');
      await svc2.prompt(id, 'once more');
      await until(() => cold.ends().length === 1, 5_000);
      expect(setValues(fake)).toEqual(['low', 'none', 'none']);
      cold.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('a model change steps the level down to one the model has (else its default) and clears the notice', async () => {
    const refusing = (scenario: Json): void => {
      levelled(scenario);
      scenario.session = { ...(scenario.session as Json), refuseEffort: true };
    };
    const fake = fakeService('chat-basic', { mutate: refusing });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hi', thinking: 'medium' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.ends().length === 1, 5_000);
      expect(page.lastOutcome()).toBe('completed');
      await until(() => page.state!.notices.some((n) => n.code === 'thinking-not-set'));
      expect((await svc.patch(id, { modelId: NINE })).thinking).toBe('low');
      expect(stored(fake, id)?.thinking).toBe('low');
      expect(page.state!.notices).toEqual([]);
      expect((await svc.patch(id, { modelId: FOUR })).thinking).toBe('low');
      await svc.patch(id, { thinking: 'none' });
      expect((await svc.patch(id, { modelId: NINE })).thinking).toBe('high');
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('a fork copies the level into the copy’s composer file, with no queue', async () => {
    // The answer runs until Stop: a timed one can end before the row is queued, and the row then runs as a turn.
    const held = (scenario: Json): void => {
      levelled(scenario);
      const never = join(String(scenario.recordTo), '..', 'never');
      (scenario.prompts as Json[]).unshift({
        match: 'hold',
        steps: [{ holdOn: never }, { end: { stopReason: 'end_turn' } }],
      });
    };
    const fake = fakeService('chat-basic', { mutate: held });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hi', thinking: 'none' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.ends().length === 1, 5_000);
      // A row held by Stop stays the original's.
      await svc.prompt(id, 'hold please');
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'queued for the original', [], undefined, 'queue');
      await svc.cancel(id);
      await until(() => page.ends().length === 2, 5_000);
      // The row's file write is not awaited by the route, so it can land after the turn end.
      await until(() => (stored(fake, id)?.queue as unknown[] | undefined)?.length === 1, 5_000);
      expect(stored(fake, id)).toMatchObject({ queue: [{ text: 'queued for the original' }], thinking: 'none' });
      const copy = await svc.fork(id);
      expect(stored(fake, copy.id)).toEqual({ queue: [], thinking: 'none' });
      expect(stored(fake, id)).toMatchObject({ queue: [{ text: 'queued for the original' }], thinking: 'none' });
      const opened = new Page(svc, copy.id);
      await until(() => opened.state !== null, 5_000);
      expect(opened.state!.conversation.thinking).toBe('none');
      opened.detach();
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('Chat service — Claude Code in Work', () => {
  // Mutations: the Work host not marked stale (the idle agent keeps Claude Code until it idles out), both hosts marked
  // stale (the idle Chat agent stops too), or the setting read only at construction (the restart reads the default).
  it('off: the idle Work agent stops at once, the next one starts with the switches, Chat stays; a restart keeps it', async () => {
    await withWork(
      'chat-basic',
      async (fake, project) => {
        const svc = fake.service;
        const starts = (): Json[] => fake.records().filter((r) => r.kind === 'start');
        const idle = (id: string): Promise<void> => until(() => !svc.busyConversations().includes(id), 5_000);
        const chat = (await svc.create({ mode: 'chat', text: 'hi' })).conversation;
        await idle(chat.id);
        const work = (await svc.create({ mode: 'work', text: 'hi', cwd: project })).conversation;
        await idle(work.id);
        expect(starts()).toHaveLength(2);
        expect(starts()[1]!.claudeSwitches).toEqual({});

        expect(await svc.setWorkSettings({ claudeCode: false })).toEqual({ claudeCode: false });
        for (const end = Date.now() + 5_000; (await svc.state()).modes.work.state !== 'stopped';) {
          if (Date.now() > end) throw new Error('the Work agent did not stop');
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
        expect((await svc.state()).modes.chat.state).toBe('ready');
        await svc.prompt(work.id, 'again');
        await idle(work.id);
        expect(starts()).toHaveLength(3);
        expect(starts()[2]!.claudeSwitches).toEqual(CLAUDE_CODE_OFF_ENV);

        await fake.service.close();
        expect(fake.restart().getWorkSettings()).toEqual({ claudeCode: false });
      },
      {
        mutate: (scenario) => {
          scripted(scenario);
          scenario.work = { session: { agentName: 'grok-build-plan' } };
        },
      },
    );
  }, 30_000);
});

describe('Chat service — Work without a folder', () => {
  const quiet = (scenario: Json): void => {
    scenario.prompts = [];
  };

  async function withNone(
    run: (fake: FakeService, scratch: string) => Promise<void>,
    options: Parameters<typeof fakeService>[1] = {},
  ): Promise<void> {
    const fake = fakeService('work-workflow-run', { mutate: quiet, ...options });
    try {
      await run(fake, join(fake.root, 'home', 'Portato'));
    } finally {
      await fake.close();
    }
  }

  /** Waits until the conversation's first answer ended, however soon after the create it did. */
  async function answered(fake: FakeService, id: string): Promise<void> {
    const page = new Page(fake.service, id);
    await until(() => page.state !== null && page.state.turn === null && page.texts('user').length === 1, 8_000);
    page.detach();
  }

  /** A No folder conversation whose first answer ended. */
  async function made(fake: FakeService): Promise<{ id: string; cwd: string }> {
    const { conversation } = await fake.service.create({ mode: 'work', text: 'hi', noFolder: true });
    await answered(fake, conversation.id);
    return { id: conversation.id, cwd: conversation.cwd! };
  }

  const sessionNews = (fake: FakeService): Json[] => fake.sent().filter((m) => m.method === 'session/new');

  // Mutations: no folder made (the agent gets no cwd); the folder made under the app data; the mark left out.
  it('makes a new 0700 folder in ~/Portato, starts the agent there and marks the conversation', async () => {
    await withNone(async (fake, scratch) => {
      const { id, cwd } = await made(fake);
      expect(dirname(cwd)).toBe(realpathSync(scratch));
      expect(statSync(cwd).isDirectory()).toBe(true);
      expect(statSync(cwd).mode & 0o777).toBe(0o700);
      expect((sessionNews(fake)[0]!.params as Json).cwd).toBe(cwd);
      const page = new Page(fake.service, id);
      await until(() => page.state !== null, 5_000);
      expect(page.state!.conversation).toMatchObject({ mode: 'work', cwd, noFolder: true });
      page.detach();
      // ~/Portato picked by hand is a folder of its own: its parent is home.
      const picked = await fake.service.create({ mode: 'work', text: 'hi', cwd: scratch });
      expect([picked.conversation.cwd, picked.conversation.noFolder]).toEqual([realpathSync(scratch), undefined]);
    });
  }, 30_000);

  // Mutation: the record written to the file only (the service reads it once at start) → the first No folder
  // conversation is not marked.
  it('a conversation read before ~/Portato exists leaves the first No folder conversation marked', async () => {
    await withNone(async (fake, scratch) => {
      const project = join(fake.root, 'home', 'code');
      mkdirSync(project, { recursive: true });
      const before = await fake.service.create({ mode: 'work', text: 'hi', cwd: project });
      expect(before.conversation.noFolder).toBeUndefined();
      expect(existsSync(scratch)).toBe(false);
      const { conversation } = await fake.service.create({ mode: 'work', text: 'hi', noFolder: true });
      expect(dirname(conversation.cwd!)).toBe(realpathSync(scratch));
      expect(conversation.noFolder).toBe(true);
    });
  }, 30_000);

  // Mutation: the new folder skips `checkWorkFolder` → the agent starts in the agent home.
  it('the folder rule runs on the new folder: a ~/Portato inside the agent home fails and leaves nothing', async () => {
    await withNone(async (fake, scratch) => {
      const inside = join(fake.root, 'agent-home', 'Portato');
      mkdirSync(inside, { recursive: true });
      symlinkSync(inside, scratch);
      const error = await apiError(fake.service.create({ mode: 'work', text: 'hi', noFolder: true }));
      expect([error.reason, error.message]).toEqual(['cwd-invalid', CWD_PROBLEM_MESSAGES['app-data']]);
      expect(readdirSync(inside)).toEqual([]);
      expect(sessionNews(fake)).toEqual([]);
    });
  });

  // Mutation: a raw fs error passed on → "The agent could not be reached." in place of the typed failure.
  it('a file named Portato: the create answers folder-not-made and never reaches the agent', async () => {
    await withNone(async (fake, scratch) => {
      writeFileSync(scratch, 'x');
      const error = await apiError(fake.service.create({ mode: 'work', text: 'hi', noFolder: true }));
      expect([error.code, error.reason, error.message]).toEqual([
        'E_CONFLICT',
        'folder-not-made',
        'The app could not make a folder for this conversation.',
      ]);
      expect(sessionNews(fake)).toEqual([]);
    });
  });

  // Mutation: the record written once session/new answered → the agent runs in a folder the app has no record of.
  it('the new folder is recorded (0600) before session/new, and stays while session/new hangs', async () => {
    await withNone(
      async (fake, scratch) => {
        const create = apiError(fake.service.create({ mode: 'work', text: 'hi', noFolder: true }));
        await until(() => sessionNews(fake).length === 1, 8_000);
        const cwd = (sessionNews(fake)[0]!.params as Json).cwd as string;
        const id = statSync(cwd, { bigint: true });
        expect(readScratchRecordsSync(fake.paths.scratchFolders).get(cwd)).toEqual({
          path: cwd,
          dev: String(id.dev),
          ino: String(id.ino),
        });
        expect(statSync(fake.paths.scratchFolders).mode & 0o777).toBe(0o600);
        expect((await create).reason).toBe('agent-slow');
        expect(readdirSync(scratch)).toEqual([basename(cwd)]);
      },
      {
        mutate: (scenario) => void Object.assign(scenario, { prompts: [], faults: { hangOn: 'session/new' } }),
        createBudgetMs: 400,
      },
    );
  }, 30_000);

  // Mutation: the folder removed when the create gives up → the late session starts in a removed folder, and a retry
  // in the same minute gets that folder again.
  it('a create past its budget keeps its folder until the late session is deleted; a retry gets -2', async () => {
    const start = Date.now();
    const now = (): number => Date.now() - (start % 60_000);
    const stamp = stampName(now());
    await withNone(
      async (fake, scratch) => {
        for (let n = 0; n < 2; n += 1) {
          const error = await apiError(fake.service.create({ mode: 'work', text: 'hi', noFolder: true }));
          expect(error.reason).toBe('agent-slow');
        }
        expect(readdirSync(scratch).sort()).toEqual([stamp, `${stamp}-2`]);
        await until(() => readdirSync(scratch).length === 0, 10_000);
        expect(fake.sent().filter((m) => m.method === '_x.ai/session/delete')).toHaveLength(2);
      },
      {
        mutate: (scenario) =>
          void Object.assign(scenario, { prompts: [], faults: { delayMs: { 'session/new': 1_500 } } }),
        createBudgetMs: 400,
        now,
      },
    );
  }, 30_000);

  // Mutation: no cleanup on a failed create → an empty dated folder per failure.
  it('an index write that fails: the new folder is gone', async () => {
    await withNone(async (fake, scratch) => {
      // A folder where the index file goes: its write cannot land.
      mkdirSync(join(fake.paths.conversations, 'held'), { recursive: true });
      await apiError(fake.service.create({ mode: 'work', text: 'hi', noFolder: true }));
      expect(sessionNews(fake)).toHaveLength(1);
      expect(readdirSync(scratch)).toEqual([]);
    });
  }, 30_000);

  // Mutation: the cleanup also after the conversation is stored → a live conversation loses its folder.
  it('a first turn that cannot start after the conversation is stored keeps the folder', async () => {
    await withNone(async (fake, scratch) => {
      const target = fake.service as unknown as { startTurn: () => never };
      const original = target.startTurn;
      target.startTurn = () => {
        throw routeError('chat', new Error('no turn'));
      };
      try {
        await apiError(fake.service.create({ mode: 'work', text: 'hi', noFolder: true }));
      } finally {
        target.startTurn = original;
      }
      expect(readdirSync(scratch)).toHaveLength(1);
    });
  });

  // Mutations: a recursive remove (the file goes); no remove at all (the empty folder stays).
  it('delete removes the empty folder and keeps one that holds a file', async () => {
    await withNone(async (fake, scratch) => {
      const empty = await made(fake);
      const full = await made(fake);
      writeFileSync(join(full.cwd, 'notes.md'), 'mine');
      await fake.service.delete(empty.id);
      await fake.service.delete(full.id);
      expect(existsSync(empty.cwd)).toBe(false);
      expect(readFileSync(join(full.cwd, 'notes.md'), 'utf8')).toBe('mine');
      expect(existsSync(scratch)).toBe(true);
    });
  }, 30_000);

  // Mutation: the remove does not look for another conversation in the folder → the fork loses its folder.
  it('a fork keeps the folder until it is deleted too', async () => {
    await withNone(async (fake) => {
      const origin = await made(fake);
      const copy = await fake.service.fork(origin.id);
      await fake.service.delete(origin.id);
      expect(existsSync(origin.cwd)).toBe(true);
      await fake.service.delete(copy.id);
      expect(existsSync(origin.cwd)).toBe(false);
    });
  }, 30_000);

  // Mutation: the remove for every Work folder → a user's own empty folder goes.
  it('delete of a conversation in a picked folder never removes it, empty or not', async () => {
    await withNone(async (fake) => {
      const picked = join(fake.root, 'home', 'code', 'empty');
      mkdirSync(picked, { recursive: true });
      const { conversation } = await fake.service.create({ mode: 'work', text: 'hi', cwd: picked });
      await answered(fake, conversation.id);
      await fake.service.delete(conversation.id);
      expect(readdirSync(picked)).toEqual([]);
    });
  }, 30_000);

  // Mutation: the parent folder read in place of the record → a folder picked in ~/Portato is marked and removed.
  it('a folder picked by hand in ~/Portato is never marked and never removed', async () => {
    await withNone(async (fake, scratch) => {
      const picked = join(scratch, 'my-project');
      mkdirSync(picked, { recursive: true });
      const { conversation } = await fake.service.create({ mode: 'work', text: 'hi', cwd: picked });
      expect([conversation.cwd, conversation.noFolder]).toEqual([realpathSync(picked), undefined]);
      await answered(fake, conversation.id);
      await fake.service.delete(conversation.id);
      expect(existsSync(picked)).toBe(true);
    });
  }, 30_000);

  // Mutation: the record matched by path alone → a new folder at the path the app made is removed.
  it('a new folder in place of the one the app made is not its own: delete keeps it', async () => {
    await withNone(async (fake) => {
      const { id, cwd } = await made(fake);
      renameSync(cwd, `${cwd}-moved`);
      mkdirSync(cwd);
      await fake.service.delete(id);
      expect(existsSync(cwd)).toBe(true);
    });
  }, 30_000);

  // Mutation: the delete's in-use check blind to the creates in flight → the new conversation loses its folder.
  it('a delete keeps the folder while another conversation is being made in it', async () => {
    await withNone(
      async (fake) => {
        const first = await made(fake);
        const opening = fake.service.create({ mode: 'work', text: 'hi', cwd: first.cwd });
        await until(() => sessionNews(fake).length === 2, 8_000);
        await fake.service.delete(first.id);
        expect(existsSync(first.cwd)).toBe(true);
        const { conversation } = await opening;
        expect(conversation.cwd).toBe(first.cwd);
        await answered(fake, conversation.id);
      },
      {
        mutate: (scenario) =>
          void Object.assign(scenario, { prompts: [], faults: { delayMs: { 'session/new': 1_500 } } }),
        createBudgetMs: 10_000,
      },
    );
  }, 30_000);

  describe('a fork past its soft deadline', () => {
    function gated(run: (fake: FakeService, go: string) => Promise<void>): Promise<void> {
      const gateDir = mkdtempSync(join(tmpdir(), 'mlx-fork-gate-'));
      onTestFinished(() => rmSync(gateDir, { recursive: true, force: true }));
      const go = join(gateDir, 'go');
      return withNone((fake) => run(fake, go), {
        timings: { forkMs: 300 },
        mutate: (scenario) => void Object.assign(scenario, { prompts: [], fork: { holdOn: go } }),
      });
    }

    // Mutation: the fork's hold let go at its soft deadline → a delete of the origin removes the copy's folder.
    it('keeps the folder through a delete of the origin', async () => {
      await gated(async (fake) => {
        const origin = await made(fake);
        expect((await apiError(fake.service.fork(origin.id))).reason).toBe('agent-slow');
        await fake.service.delete(origin.id);
        expect(existsSync(origin.cwd)).toBe(true);
      });
    }, 30_000);

    // Mutation: a late copy that landed never lets the hold go → the folder stays after its last conversation.
    it('lands its copy in the folder, then lets the folder go', async () => {
      await gated(async (fake, go) => {
        const origin = await made(fake);
        expect((await apiError(fake.service.fork(origin.id))).reason).toBe('agent-slow');
        writeFileSync(go, '');
        const indexed = (): string[] => readWorkIndexSync(fake.paths.conversations).work.map((e) => e.id);
        await until(() => indexed().length === 2, 8_000);
        const copy = indexed().find((id) => id !== origin.id)!;
        await fake.service.delete(origin.id);
        expect(existsSync(origin.cwd)).toBe(true);
        await fake.service.delete(copy);
        expect(existsSync(origin.cwd)).toBe(false);
      });
    }, 30_000);
  });
});
