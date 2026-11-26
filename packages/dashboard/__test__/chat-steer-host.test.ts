/**
 * Send now on the scripted fake agent: a queued row (or ⌘Enter text) goes into the running Work answer through
 * `_x.ai/interject` (soft), only while a prompt is open; the host pushes the steer row itself and the agent stores it
 * when the model reads it, as the same entry. Chat refuses and sends nothing; a starting turn, a held row with no answer
 * and an agent error leave the row queued; a late yes still moves the row; a Stop drops what was not read; a steer that
 * misses the answer runs as its own turn, named `message`. The host itself sends nothing for Chat or in the gap between
 * the two prompts of an automatic continue.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import type { ApiError } from '../src/api/errors.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent, QueueState } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeHost, fakeService, noProviders, until, type FakeService, type Json } from './helpers/chat-fake.js';
import { gate as engineGate } from './helpers/fake-engine.js';

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

class Page {
  readonly events: ConversationEvent[] = [];
  state: ConversationState | null = null;
  readonly detach: () => void;

  constructor(service: ChatService, id: string) {
    this.detach = service.watchConversation(id, (event) => {
      this.events.push(event);
      const next = applyConversationEvent(this.state, event);
      if (next !== 'resync') this.state = next;
    });
  }

  ends(): number {
    return this.events.filter((e) => e.type === 'turn' && e.turn === null).length;
  }

  outcomes(): Array<string | undefined> {
    return this.events.flatMap((e) => (e.type === 'turn' && e.turn === null ? [e.outcome] : []));
  }

  queue(): QueueState {
    return this.state?.queue ?? { items: [], held: null, count: 0 };
  }

  entries(): TranscriptEntry[] {
    return this.state?.entries ?? [];
  }

  steers(): TranscriptEntry[] {
    return this.entries().filter((e) => e.kind === 'user' && e.interjection === true);
  }
}

async function apiError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return routeError('chat', error);
  }
  throw new Error('expected a failure');
}

function prompts(fake: FakeService): string[] {
  return fake
    .sent()
    .filter((m) => m.method === 'session/prompt')
    .map((m) => (((m.params as Json).prompt as Json[])[0]!.text as string) ?? '');
}

function interjects(fake: FakeService): Json[] {
  return fake
    .sent()
    .filter((m) => m.method === '_x.ai/interject')
    .map((m) => m.params as Json);
}

function recorded(fake: FakeService, kind: string): Json[] {
  return fake.records().filter((r) => r.kind === kind);
}

/** An entry without its time: the host stamps the live row, the agent the stored one. */
function untimed(entry: TranscriptEntry): Omit<TranscriptEntry, 'ts'> {
  const { ts: _ts, ...rest } = entry;
  return rest;
}

const END = [
  { text: 'Done.', chunk: 5 },
  { turnCompleted: { inputTokens: 10, outputTokens: 2 } },
  { end: { stopReason: 'end_turn' } },
];

/** `read`: the model reads the steers after the gate; `hold`: the answer ends with them unread. */
function scripts(gate: string): (scenario: Json) => void {
  return (scenario) => {
    scenario.prompts = [
      { match: 'read', steps: [{ holdOn: gate }, { readSteers: true }, ...END] },
      { match: 'hold', steps: [{ holdOn: gate }, ...END] },
      { match: '', steps: END },
    ];
  };
}

async function withWork(
  run: (fake: FakeService, project: string, gate: string) => Promise<void>,
  mutate: (scenario: Json) => void = () => {},
  timings: Parameters<typeof fakeService>[1] = {},
): Promise<void> {
  let gatePath = '';
  const fake = fakeService('work-workflow-run', {
    ...timings,
    mutate: (scenario) => {
      gatePath = join(String(scenario.recordTo), '..', 'gate');
      scripts(gatePath)(scenario);
      mutate(scenario);
    },
  });
  const project = join(fake.root, 'home', 'code', 'app');
  mkdirSync(project, { recursive: true });
  try {
    await run(fake, project, gatePath);
  } finally {
    await fake.close();
  }
}

/** A Work conversation whose first answer runs (its prompt is open) and one queued row. */
async function runningWithRow(
  svc: ChatService,
  project: string,
  text: string,
  row: string,
): Promise<{ id: string; page: Page; rowId: string }> {
  const { conversation } = await svc.create({ mode: 'work', text, cwd: project });
  const id = conversation.id;
  const page = new Page(svc, id);
  await until(() => page.state?.turn?.phase === 'running', 5_000);
  const queued = (await svc.prompt(id, row, [], false, 'queue')) as { queued: string };
  await until(() => page.queue().items.length === 1);
  return { id, page, rowId: queued.queued };
}

describe('Send now — into the running answer', () => {
  it('a row goes through `_x.ai/interject`, leaves the list, shows as a steer, and is stored as the same entry', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { id, page, rowId } = await runningWithRow(svc, project, 'read on', 'also update the changelog');

      expect(await svc.queueSend(id, rowId)).toEqual({ steered: true });
      expect(interjects(fake)).toEqual([
        { sessionId: id, text: 'also update the changelog', interjectionId: expect.any(String) },
      ]);
      await until(() => page.queue().items.length === 0);
      await until(() => page.steers().length === 1);
      const live = page.steers()[0]!;
      expect(live).toMatchObject({
        kind: 'user',
        text: 'also update the changelog',
        interjection: true,
        promptIndex: null,
      });
      // The steer sits inside the running answer: no new turn, no queue row sent as a prompt.
      expect(page.state?.turn?.phase).toBe('running');

      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      expect(page.outcomes()).toEqual(['completed']);
      expect(prompts(fake)).toEqual(['read on']);
      expect(recorded(fake, 'steer-read').map((r) => r.text)).toEqual(['also update the changelog']);
      const liveEntries = page.entries();
      page.detach();

      await svc.close();
      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      expect(cold.steers().map(untimed)).toEqual([untimed(live)]);
      const at = (list: TranscriptEntry[]): number =>
        list.findIndex((e) => e.kind === 'user' && e.interjection === true);
      expect(at(cold.entries())).toBe(at(liveEntries));
      cold.detach();
    });
  }, 30_000);

  it('⌘Enter text (`whenBusy: "now"`) goes with no row; idle it starts an answer', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'read on', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);

      expect(await svc.prompt(id, 'use pnpm instead', [], true, 'now')).toEqual({ steered: true });
      await until(() => page.steers().length === 1);
      expect(page.queue().items).toEqual([]);
      expect(interjects(fake).map((p) => p.text)).toEqual(['use pnpm instead']);

      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      expect(await svc.prompt(id, 'next one', [], false, 'now')).toEqual({ turnId: expect.any(String) });
      await until(() => page.ends() === 2, 5_000);
      expect(prompts(fake)).toEqual(['read on', 'next one']);
      expect(interjects(fake)).toHaveLength(1);
      page.detach();
    });
  }, 30_000);
});

describe('Send now — refused, nothing sent', () => {
  it('Chat refuses with 409 and sends nothing', async () => {
    let gatePath = '';
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        gatePath = join(String(scenario.recordTo), '..', 'gate');
        scripts(gatePath)(scenario);
      },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold on' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const row = (await svc.prompt(id, 'and the train times', [], undefined, 'queue')) as { queued: string };
      expect(await apiError(svc.queueSend(id, row.queued))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'steer-work-only',
      });
      expect(await apiError(svc.prompt(id, 'now please', [], undefined, 'now'))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'steer-work-only',
      });
      expect(page.queue().items.map((item) => item.text)).toEqual(['and the train times']);
      writeFileSync(gatePath, '');
      await until(() => page.ends() === 2, 10_000);
      expect(interjects(fake)).toEqual([]);
      expect(page.steers()).toEqual([]);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: answer `steer-not-now` while the answer still runs → the page says nothing although the row cannot go.
  it('a turn still starting refuses as `steer-busy`, a held row with no answer as `steer-not-now`; the row stays', async () => {
    await withWork(async (fake, project) => {
      const engine = await fake.engine();
      const load = engineGate();
      engine.script((request) =>
        request.method === 'POST' && request.path === '/v1/messages/count_tokens'
          ? { json: { input_tokens: 1 }, until: load.promise }
          : { status: 404, json: {} },
      );
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'hold', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'starting', 5_000);
      const row = (await svc.prompt(id, 'B', [], false, 'queue')) as { queued: string };
      expect(await apiError(svc.queueSend(id, row.queued))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'steer-busy',
      });
      // ⌘Enter while starting: it joins the queue and says why.
      expect(await svc.prompt(id, 'C', [], false, 'now')).toEqual({ queued: expect.any(String), steer: 'refused' });
      expect(interjects(fake)).toEqual([]);

      load.open();
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      expect(await svc.cancel(id)).toEqual({ cancelled: true });
      await until(() => page.ends() === 1, 5_000);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'B' }, { text: 'C' }] });
      expect(await apiError(svc.queueSend(id, row.queued))).toMatchObject({ reason: 'steer-not-now' });
      expect(interjects(fake)).toEqual([]);
      expect(page.queue().items).toHaveLength(2);
      page.detach();
    });
  }, 30_000);

  it('an agent error leaves the row queued, and the list sends it when the answer ends', async () => {
    await withWork(
      async (fake, project, gate) => {
        const svc = fake.service;
        const { id, page, rowId } = await runningWithRow(svc, project, 'hold', 'after the error');
        expect(await apiError(svc.queueSend(id, rowId))).toMatchObject({ code: 'E_CONFLICT', reason: 'steer-busy' });
        expect(interjects(fake)).toHaveLength(1);
        expect(page.queue().items.map((item) => item.text)).toEqual(['after the error']);
        writeFileSync(gate, '');
        await until(() => page.ends() === 2, 10_000);
        expect(prompts(fake)).toEqual(['hold', 'after the error']);
        expect(page.steers()).toEqual([]);
        page.detach();
      },
      (scenario) => {
        scenario.faults = { ...(scenario.faults as Json), interjectError: true };
      },
    );
  }, 30_000);
});

describe('Send now — five rows wait', () => {
  async function fillQueue(svc: ChatService, id: string, page: Page): Promise<void> {
    for (const n of [1, 2, 3, 4, 5]) await svc.prompt(id, `row ${n}`, [], false, 'queue');
    await until(() => page.queue().items.length === 5);
    expect(await apiError(svc.prompt(id, 'a sixth', [], false, 'queue'))).toMatchObject({ reason: 'queue-full' });
  }

  // Mutation: enqueue before the steer when the list is full → 409 `queue-full` and nothing goes.
  it('⌘Enter text goes into the answer: a steer takes no place in the list', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'read on', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await fillQueue(svc, id, page);
      expect(await svc.prompt(id, 'use pnpm instead', [], false, 'now')).toEqual({ steered: true });
      expect(interjects(fake).map((p) => p.text)).toEqual(['use pnpm instead']);
      await until(() => page.steers().length === 1);
      expect(page.queue().items.map((item) => item.text)).toEqual(['row 1', 'row 2', 'row 3', 'row 4', 'row 5']);
      page.detach();
    });
  }, 30_000);

  it('refused while the answer starts: 409 `steer-busy`, nothing queued, nothing sent', async () => {
    await withWork(async (fake, project) => {
      const engine = await fake.engine();
      const load = engineGate();
      engine.script((request) =>
        request.method === 'POST' && request.path === '/v1/messages/count_tokens'
          ? { json: { input_tokens: 1 }, until: load.promise }
          : { status: 404, json: {} },
      );
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'hold', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'starting', 5_000);
      await fillQueue(svc, id, page);
      expect(await apiError(svc.prompt(id, 'now please', [], false, 'now'))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'steer-busy',
      });
      expect(page.queue().items).toHaveLength(5);
      expect(interjects(fake)).toEqual([]);
      load.open();
      page.detach();
    });
  }, 30_000);

  it('a late reply is waited for: the late yes is a steer, a late no is 409 `steer-busy`; no sixth row', async () => {
    await withWork(
      async (fake, project) => {
        const svc = fake.service;
        const { conversation } = await svc.create({ mode: 'work', text: 'read on', cwd: project });
        const id = conversation.id;
        const page = new Page(svc, id);
        await until(() => page.state?.turn?.phase === 'running', 5_000);
        await fillQueue(svc, id, page);
        expect(await svc.prompt(id, 'slow steer', [], false, 'now')).toEqual({ steered: true });
        await until(() => page.steers().length === 1);
        expect(page.steers().map((e) => e.kind === 'user' && e.text)).toEqual(['slow steer']);
        expect(page.queue().items).toHaveLength(5);
        page.detach();
      },
      (scenario) => {
        scenario.faults = { ...(scenario.faults as Json), delayMs: { '_x.ai/interject': 600 } };
      },
      { timings: { steerMs: 100 } },
    );
    await withWork(
      async (fake, project) => {
        const svc = fake.service;
        const { conversation } = await svc.create({ mode: 'work', text: 'read on', cwd: project });
        const id = conversation.id;
        const page = new Page(svc, id);
        await until(() => page.state?.turn?.phase === 'running', 5_000);
        await fillQueue(svc, id, page);
        expect(await apiError(svc.prompt(id, 'slow steer', [], false, 'now'))).toMatchObject({ reason: 'steer-busy' });
        expect(interjects(fake).map((p) => p.text)).toEqual(['slow steer']);
        expect(page.steers()).toEqual([]);
        expect(page.queue().items).toHaveLength(5);
        page.detach();
      },
      (scenario) => {
        scenario.faults = { ...(scenario.faults as Json), delayMs: { '_x.ai/interject': 600 }, interjectError: true };
      },
      { timings: { steerMs: 100 } },
    );
  }, 30_000);
});

describe('Send now — late, stopped, missed', () => {
  it('no reply in time: 503 and the row stays, the list does not send it; the late yes moves it', async () => {
    await withWork(
      async (fake, project, gate) => {
        const svc = fake.service;
        const { id, page, rowId } = await runningWithRow(svc, project, 'read on', 'slow steer');
        expect(await apiError(svc.queueSend(id, rowId))).toMatchObject({ code: 'E_UNAVAILABLE', reason: 'steer-late' });
        expect(page.queue().items.map((item) => item.text)).toEqual(['slow steer']);
        // A second press while the first waits is refused before anything goes out.
        expect(await apiError(svc.queueSend(id, rowId))).toMatchObject({ code: 'E_CONFLICT', reason: 'steer-late' });
        expect(interjects(fake)).toHaveLength(1);
        await until(() => page.steers().length === 1, 5_000);
        await until(() => page.queue().items.length === 0);
        writeFileSync(gate, '');
        await until(() => page.ends() === 1, 5_000);
        await sleep(200);
        expect(prompts(fake)).toEqual(['read on']);
        expect(recorded(fake, 'steer-read').map((r) => r.text)).toEqual(['slow steer']);
        page.detach();
      },
      (scenario) => {
        scenario.faults = { ...(scenario.faults as Json), delayMs: { '_x.ai/interject': 600 } };
      },
      { timings: { steerMs: 100 } },
    );
  }, 30_000);

  it('the answer ends while the reply is late: the list does not send the row too; the late yes moves it', async () => {
    await withWork(
      async (fake, project, gate) => {
        const svc = fake.service;
        const { id, page, rowId } = await runningWithRow(svc, project, 'hold', 'after the end');
        expect(await apiError(svc.queueSend(id, rowId))).toMatchObject({ reason: 'steer-late' });
        writeFileSync(gate, '');
        await until(() => page.ends() === 1, 5_000);
        await until(() => page.steers().length === 1, 5_000);
        await until(() => page.queue().items.length === 0);
        await until(() => page.ends() === 2, 5_000);
        await sleep(200);
        expect(prompts(fake)).toEqual(['hold']);
        expect(interjects(fake)).toHaveLength(1);
        expect(recorded(fake, 'steer-turn').map((r) => r.text)).toEqual(['after the end']);
        page.detach();
      },
      (scenario) => {
        scenario.faults = { ...(scenario.faults as Json), delayMs: { '_x.ai/interject': 600 } };
      },
      { timings: { steerMs: 100 } },
    );
  }, 30_000);

  it('a row edited while the reply is late keeps its new words as a queued message; the late yes shows the old', async () => {
    await withWork(
      async (fake, project, gate) => {
        const svc = fake.service;
        const { id, page, rowId } = await runningWithRow(svc, project, 'read on', 'first words');
        expect(await apiError(svc.queueSend(id, rowId))).toMatchObject({ reason: 'steer-late' });
        await svc.queueEdit(id, rowId, { text: 'second words' });
        await until(() => page.steers().length === 1, 5_000);
        expect(page.steers()[0]).toMatchObject({ text: 'first words' });
        expect(page.queue().items.map((item) => item.text)).toEqual(['second words']);
        writeFileSync(gate, '');
        await until(() => page.ends() === 2, 10_000);
        expect(prompts(fake)).toEqual(['read on', 'second words']);
        expect(recorded(fake, 'steer-read').map((r) => r.text)).toEqual(['first words']);
        expect(page.queue().items).toEqual([]);
        page.detach();
      },
      (scenario) => {
        scenario.faults = { ...(scenario.faults as Json), delayMs: { '_x.ai/interject': 600 } };
      },
      { timings: { steerMs: 100 } },
    );
  }, 30_000);

  it('a Stop drops the steer the model did not read: the live row stays, the disk has none', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page, rowId } = await runningWithRow(svc, project, 'read on', 'stop, use pnpm');
      expect(await svc.queueSend(id, rowId)).toEqual({ steered: true });
      await until(() => page.steers().length === 1);
      expect(await svc.cancel(id)).toEqual({ cancelled: true });
      await until(() => page.ends() === 1, 5_000);
      expect(page.outcomes()).toEqual(['cancelled']);
      expect(page.steers()).toHaveLength(1);
      await sleep(200);
      expect(recorded(fake, 'steer-read')).toEqual([]);
      expect(recorded(fake, 'steer-turn')).toEqual([]);
      page.detach();

      await svc.close();
      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      expect(cold.steers()).toEqual([]);
      expect(cold.entries().some((e) => e.kind === 'user' && e.text === 'stop, use pnpm')).toBe(false);
      cold.detach();
    });
  }, 30_000);

  it('a steer the answer never read runs as its own turn: live a wake named `message`, stored a typed turn', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { id, page, rowId } = await runningWithRow(svc, project, 'hold', 'missed the end');
      expect(await svc.queueSend(id, rowId)).toEqual({ steered: true });
      writeFileSync(gate, '');
      await until(() => page.ends() === 2, 10_000);
      expect(page.outcomes()).toEqual(['completed', 'completed']);
      expect(prompts(fake)).toEqual(['hold']);
      expect(recorded(fake, 'steer-turn').map((r) => r.text)).toEqual(['missed the end']);
      const wake = page.entries().find((e) => e.kind === 'wake');
      expect(wake).toMatchObject({ kind: 'wake', cause: 'message' });
      page.detach();

      await svc.close();
      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      const stored = cold.entries().filter((e) => e.kind === 'user' && e.text === 'missed the end');
      expect(stored).toEqual([expect.objectContaining({ kind: 'user', promptIndex: 1 })]);
      expect(stored[0]).not.toHaveProperty('interjection');
      expect(cold.entries().some((e) => e.kind === 'wake')).toBe(false);
      cold.detach();
    });
  }, 30_000);
});

describe('Send now — the host gate', () => {
  const MODEL = 'qwen3.5-4b';
  const hostInterjects = (sent: Json[]): string[] =>
    sent.filter((m) => m.method === '_x.ai/interject').map((m) => (m.params as Json).text as string);

  it('a Chat host sends nothing, even with a prompt open', async () => {
    let gatePath = '';
    const fake = await fakeHost('chat-basic', {
      mode: 'chat',
      providers: noProviders(),
      mutate: (scenario) => {
        gatePath = join(String(scenario.recordTo), '..', 'gate');
        scripts(gatePath)(scenario);
      },
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'hold on', { modelId: MODEL });
      await until(() => fake.host.isBusy(sessionId) && fake.sent().some((m) => m.method === 'session/prompt'), 5_000);
      expect(await fake.host.steer(sessionId, 'and the train times')).toBe('not-now');
      writeFileSync(gatePath, '');
      expect((await started.done).outcome).toBe('completed');
      expect(hostInterjects(fake.sent())).toEqual([]);
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('the gap between two prompts (a length stop and its continue) refuses; the continue prompt takes it', async () => {
    let gatePath = '';
    const fake = await fakeHost('work-length', {
      mode: 'work',
      providers: noProviders(),
      mutate: (scenario) => {
        gatePath = join(String(scenario.recordTo), '..', 'gate');
        const list = scenario.prompts as Json[];
        list[0]!.steps = [{ holdOn: gatePath }, { readSteers: true }, ...(list[0]!.steps as Json[])];
        (list[1]!.steps as Json[])[1] = { lengthError: { endAfterMs: 1_500 } };
      },
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const started = fake.host.startTurn(sessionId, 'fix it', { modelId: MODEL, cwd: fake.project });
      const sessions = Reflect.get(fake.host, 'sessions') as Map<string, { turn: { between: boolean } | null }>;
      await until(() => sessions.get(sessionId)?.turn?.between === true, 5_000);
      expect(await fake.host.steer(sessionId, 'in the gap')).toBe('not-now');
      await until(() => fake.events.continues.length === 1, 5_000);
      expect(await fake.host.steer(sessionId, 'after the gap')).toBe('sent');
      writeFileSync(gatePath, '');
      expect(await started.done).toMatchObject({ outcome: 'completed', continues: 1 });
      expect(hostInterjects(fake.sent())).toEqual(['after the gap']);
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'steer-read')
          .map((r) => r.text),
      ).toEqual(['after the gap']);
    } finally {
      await fake.close();
    }
  }, 30_000);
});
