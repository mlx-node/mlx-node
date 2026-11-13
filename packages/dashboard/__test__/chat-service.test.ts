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
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { ApiError } from '../src/api/errors.js';
import type { ChatPaths } from '../src/chat/paths.js';
import { MemorySecretStore, type SecretStore } from '../src/chat/providers/secrets.js';
import { ProvidersService } from '../src/chat/providers/service.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ChatService } from '../src/chat/service.js';
import type { ChatHostEvent, ConversationEvent, CwdProblem, Interaction } from '../src/chat/types.js';
import { CWD_PROBLEM_MESSAGES } from '../src/chat/work-folder.js';
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
      const firmlinked = `/System/Volumes/Data${realpathSync(join(fake.root, 'home'))}`;
      if (existsSync(firmlinked)) cases.push([firmlinked, 'home'], [`${firmlinked}/../agent-home`, 'app-data']);
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
        expect(notices.map((e) => e.type === 'notice' && e.notice)).toEqual([{ code: 'key-printed', provider: 'exa' }]);
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
          else expect(card.plan.startsWith('ppp')).toBe(true);
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
    let providers: ProvidersService | null = null;
    await withWork(
      'work-permission',
      async (fake, project) => {
        await providers!.select({ search: 'deepseek', firecrawl: true });
        const svc = fake.service;
        const hostOff = svc.watchHost(() => {});
        const run = async (): Promise<{ dirs: unknown; env: unknown }> => {
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
          return { dirs: ((created.params as Json)._meta as Json).pluginDirs, env: start.envNames };
        };
        const first = await run();
        expect(first.dirs).toEqual([fake.paths.workGuard, join(fake.paths.plugins, 'deepseek')]);
        expect(first.env).toContain('DEEPSEEK_API_KEY');
        expect(first.env).not.toContain('FIRECRAWL_API_KEY');

        const bin = join(fake.root, 'bin');
        mkdirSync(bin);
        writeFileSync(join(bin, 'firecrawl'), '#!/bin/sh\n');
        chmodSync(join(bin, 'firecrawl'), 0o755);
        path = bin;
        await providers!.list();
        const second = await run();
        expect(second.dirs).toEqual([
          fake.paths.workGuard,
          join(fake.paths.plugins, 'deepseek'),
          join(fake.paths.plugins, 'firecrawl'),
        ]);
        expect(second.env).toEqual(expect.arrayContaining(['DEEPSEEK_API_KEY', 'FIRECRAWL_API_KEY']));
        expect(JSON.stringify(fake.records())).not.toContain('fc-key-000001');
        hostOff();
      },
      {
        providers: (paths) => {
          providers = new ProvidersService({ paths, secrets, fetch: NO_NETWORK, workPath: () => path });
          return providers;
        },
      },
    );
  }, 30_000);
});
