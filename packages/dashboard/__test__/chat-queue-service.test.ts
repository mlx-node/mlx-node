/**
 * The queue in the Chat service on the scripted fake agent: a message sent with `whenBusy: 'queue'` while an answer
 * runs waits in the host and goes, through the checks of a typed message, when the answer completed; Stop, a failed
 * answer and an output limit hold the list; a held list never blocks a fresh message; a failed check holds with its
 * sentence; a row open for edit, a due wake and the engine-lost wall keep the next row back; a restart brings the rows
 * back held; delete removes the composer file.
 */

import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { ApiError } from '../src/api/errors.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ChatService } from '../src/chat/service.js';
import type { ChatHostState, ConversationEvent, QueueState } from '../src/chat/types.js';
import { CWD_PROBLEM_MESSAGES } from '../src/chat/work-folder.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeService, memoryProviders, until, type FakeService, type Json } from './helpers/chat-fake.js';

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Polls the host state until `check` holds. */
async function stateUntil(svc: ChatService, check: (state: ChatHostState) => boolean, ms = 5_000): Promise<void> {
  const end = Date.now() + ms;
  while (!check(await svc.state())) {
    if (Date.now() > end) throw new Error('host state not reached');
    await sleep(20);
  }
}

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

  texts(): string[] {
    return this.queue().items.map((item) => item.text);
  }

  users(): string[] {
    return (this.state?.entries ?? []).flatMap((e) => (e.kind === 'user' && e.text !== '' ? [e.text] : []));
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

/** The texts of every `session/prompt` the agent got, in order. */
function prompts(fake: FakeService): string[] {
  return fake
    .sent()
    .filter((m) => m.method === 'session/prompt')
    .map((m) => (((m.params as Json).prompt as Json[])[0]!.text as string) ?? '');
}

function promptModes(fake: FakeService): Array<[string, unknown]> {
  return fake
    .sent()
    .filter((m) => m.method === 'session/prompt')
    .map((m) => {
      const params = m.params as Json;
      return [(params.prompt as Json[])[0]!.text as string, (params._meta as Json | undefined)?.mode];
    });
}

const END = [
  { text: 'Done.', chunk: 5 },
  { turnCompleted: { inputTokens: 10, outputTokens: 2 } },
  { end: { stopReason: 'end_turn' } },
];

type Extra = (gate: string) => Json[];

/** Scripts by word: `hold` waits for the gate file; `failing` and `limited` wait, then fail or hit the length cap. */
function scripts(gate: string, extra: Extra = () => []): (scenario: Json) => void {
  return (scenario) => {
    scenario.prompts = [
      ...extra(gate),
      { match: 'failing', steps: [{ holdOn: gate }, { error: { code: -32603, message: 'Internal error' } }] },
      { match: 'limited', steps: [{ holdOn: gate }, { text: 'Cut', chunk: 3 }, { end: { stopReason: 'max_tokens' } }] },
      { match: 'hold', steps: [{ holdOn: gate }, ...END] },
      { match: '', steps: END },
    ];
  };
}

async function withChat(run: (fake: FakeService, gate: string) => Promise<void>, extra?: Extra): Promise<void> {
  let gatePath = '';
  const fake = fakeService('chat-basic', {
    mutate: (scenario) => {
      gatePath = join(String(scenario.recordTo), '..', 'gate');
      scripts(gatePath, extra)(scenario);
    },
  });
  try {
    await run(fake, gatePath);
  } finally {
    await fake.close();
  }
}

async function withWork(
  run: (fake: FakeService, project: string, gate: string) => Promise<void>,
  extra?: Extra,
): Promise<void> {
  let gatePath = '';
  const fake = fakeService('work-workflow-run', {
    mutate: (scenario) => {
      gatePath = join(String(scenario.recordTo), '..', 'gate');
      scripts(gatePath, extra)(scenario);
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

function composerFile(fake: FakeService, id: string): string {
  return join(fake.paths.composer, `${id}.json`);
}

describe('the queue — sending', () => {
  it('queues while an answer runs, sends each row in order through a new turn, and refuses a busy send without it', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold first' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);

      expect(await svc.prompt(id, 'second', [], undefined, 'queue')).toEqual({ queued: expect.any(String) });
      expect(await svc.prompt(id, 'third', [], undefined, 'queue')).toEqual({ queued: expect.any(String) });
      const busy = await apiError(svc.prompt(id, 'not queued'));
      expect(busy).toMatchObject({ code: 'E_CONFLICT', reason: 'conversation-busy' });
      await until(() => page.texts().length === 2);
      expect(page.queue()).toMatchObject({
        held: null,
        items: [{ text: 'second', editing: false }, { text: 'third' }],
      });
      expect(page.queue().items[0]!.files).toEqual([]);

      // The rows are on disk for a restart, owner-only. Each change is written whole, in order, after the reply.
      const stored = (): unknown[] =>
        existsSync(composerFile(fake, id))
          ? (JSON.parse(readFileSync(composerFile(fake, id), 'utf8')) as { queue: Json[] }).queue.map((row) => row.text)
          : [];
      await until(() => stored().length === 2);
      expect(stored()).toEqual(['second', 'third']);
      expect(statSync(composerFile(fake, id)).mode & 0o777).toBe(0o600);

      writeFileSync(gate, '');
      await until(() => page.ends() === 3, 10_000);
      expect(page.outcomes()).toEqual(['completed', 'completed', 'completed']);
      expect(prompts(fake)).toEqual(['hold first', 'second', 'third']);
      expect(page.users()).toEqual(['hold first', 'second', 'third']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      await until(() => !existsSync(composerFile(fake, id)));
      page.detach();
    });
  }, 30_000);

  it('an idle conversation runs a `whenBusy` message at once; a full queue refuses the sixth', async () => {
    await withChat(async (fake) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hello' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.ends() === 1, 5_000);
      expect(await svc.prompt(id, 'hold now', [], undefined, 'queue')).toEqual({ turnId: expect.any(String) });
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      for (let i = 1; i <= 5; i++) await svc.prompt(id, `row ${i}`, [], undefined, 'queue');
      const full = await apiError(svc.prompt(id, 'row 6', [], undefined, 'queue'));
      expect(full).toMatchObject({ code: 'E_CONFLICT', reason: 'queue-full' });
      await until(() => page.texts().length === 5);
      await svc.cancel(id);
      await until(() => page.ends() === 2, 5_000);
      page.detach();
    });
  }, 30_000);
});

describe('the queue — holds', () => {
  it('Stop holds the list; a fresh message runs at once and the rows stay held; Send next sends them', async () => {
    await withChat(async (fake) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold A' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'B', [], undefined, 'queue');
      expect(await svc.cancel(id)).toEqual({ cancelled: true });
      expect(page.ends()).toBe(0);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' } });
      await until(() => page.ends() === 1, 5_000);
      expect(page.outcomes()).toEqual(['cancelled']);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'B' }] });

      expect(await svc.prompt(id, 'C')).toEqual({ turnId: expect.any(String) });
      await until(() => page.ends() === 2, 5_000);
      await sleep(150);
      expect(prompts(fake)).toEqual(['hold A', 'C']);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'B' }] });

      await svc.queueRelease(id);
      await until(() => page.ends() === 3, 5_000);
      expect(prompts(fake)).toEqual(['hold A', 'C', 'B']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      page.detach();
    });
  }, 30_000);

  it('a failed answer and an output limit hold the list; Clear empties it', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'failing now' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'after the failure', [], undefined, 'queue');
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      expect(page.outcomes()[0]).not.toBe('completed');
      await until(() => page.queue().held !== null);
      expect(page.queue().held).toEqual({ why: 'failed' });
      await svc.queueClear(id);
      await until(() => page.texts().length === 0);
      expect(page.queue().held).toBeNull();

      rmSync(gate);
      await svc.prompt(id, 'limited now');
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'after the limit', [], undefined, 'queue');
      writeFileSync(gate, '');
      await until(() => page.ends() === 2, 5_000);
      expect(page.outcomes()[1]).toBe('output-limit');
      await until(() => page.queue().held !== null);
      expect(page.queue()).toMatchObject({ held: { why: 'failed' }, items: [{ text: 'after the limit' }] });
      expect(prompts(fake)).toEqual(['failing now', 'limited now']);
      page.detach();
    });
  }, 30_000);

  it('a check that fails when the row is due holds the list with its sentence, and the row stays (Work)', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'hold', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'next step', [], false, 'queue');
      rmSync(project, { recursive: true, force: true });
      writeFileSync(gate, '');
      await until(() => page.queue().held !== null, 5_000);
      expect(page.outcomes()).toEqual(['completed']);
      expect(page.queue()).toMatchObject({
        held: { why: 'blocked', reason: CWD_PROBLEM_MESSAGES.missing, code: 'cwd-invalid' },
        items: [{ text: 'next step', plan: false }],
      });
      expect(prompts(fake)).toEqual(['hold']);
      page.detach();
    });
  }, 30_000);

  it('the engine restarted under a live command: the next row waits held; Send next is the yes (Work)', async () => {
    const extra: Extra = (gate) => [
      { match: 'cmd', steps: [{ background: { wake: false } }, { holdOn: gate }, ...END] },
    ];
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'cmd', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await stateUntil(svc, (state) => state.modes.work.tasks === 1);
      await svc.prompt(id, 'later', [], false, 'queue');
      const before = fake.endpoint.current!;
      fake.endpoint.current = { ...before, pid: before.pid + 1 };
      svc.onRuntimeEvent();
      await stateUntil(svc, (state) => state.modes.work.engineLost);
      writeFileSync(gate, '');
      await until(() => page.queue().held !== null, 5_000);
      expect(page.queue().held).toMatchObject({ why: 'blocked', code: 'engine-lost' });
      expect(prompts(fake)).toEqual(['cmd']);

      await svc.queueRelease(id);
      await until(() => page.ends() === 2, 10_000);
      expect(prompts(fake)).toEqual(['cmd', 'later']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      page.detach();
    }, extra);
  }, 30_000);
});

describe('the queue — what keeps a row back', () => {
  it('a row open for edit is not sent; saving its text ends the edit and sends it, then the next', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold one' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const first = (await svc.prompt(id, 'B', [], undefined, 'queue')) as { queued: string };
      await svc.prompt(id, 'C', [], undefined, 'queue');
      await svc.queueEdit(id, first.queued, { editing: true });
      await until(() => page.queue().items[0]?.editing === true);
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      await sleep(300);
      expect(prompts(fake)).toEqual(['hold one']);
      expect(page.queue().held).toBeNull();

      await svc.queueEdit(id, first.queued, { text: 'B edited' });
      await until(() => page.ends() === 3, 10_000);
      expect(prompts(fake)).toEqual(['hold one', 'B edited', 'C']);
      expect((await apiError(svc.queueRemove(id, first.queued))).code).toBe('E_NOT_FOUND');
      page.detach();
    });
  }, 30_000);

  it('a page that leaves with a row open for edit gives the row back', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold one' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const row = (await svc.prompt(id, 'B', [], undefined, 'queue')) as { queued: string };
      await svc.queueEdit(id, row.queued, { editing: true });
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      await sleep(200);
      expect(prompts(fake)).toEqual(['hold one']);
      page.detach();
      await until(() => prompts(fake).length === 2, 5_000);
      expect(prompts(fake)).toEqual(['hold one', 'B']);
    });
  }, 30_000);

  it('a wake that is due keeps the next row until that wake ended (Work)', async () => {
    const extra: Extra = (gate) => [
      { match: 'bg', steps: [{ background: { endOn: `${gate}-bg`, wakeAfterMs: 400 } }, { holdOn: gate }, ...END] },
    ];
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'bg', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'after the wake', [], false, 'queue');
      await stateUntil(svc, (state) => state.modes.work.tasks === 1);
      writeFileSync(`${gate}-bg`, '');
      await stateUntil(svc, (state) => state.modes.work.tasks === 0);
      writeFileSync(gate, '');
      await until(() => page.ends() === 3, 10_000);
      expect(page.outcomes()).toEqual(['completed', 'completed', 'completed']);
      expect(prompts(fake)).toEqual(['bg', 'after the wake']);
      const entries = page.state!.entries;
      const woke = entries.findIndex((e) => e.kind === 'assistant' && e.text === 'The command ended.');
      const queued = entries.findIndex((e) => e.kind === 'user' && e.text === 'after the wake');
      expect(woke).toBeGreaterThan(-1);
      expect(queued).toBeGreaterThan(woke);
      page.detach();
    }, extra);
  }, 30_000);

  // Mutation: drop the service's drain on `onWakeDropped` → the row waits for a wake that never comes.
  it('the agent exits while a wake is due: no turn ends, and the next row still goes (Work)', async () => {
    const extra: Extra = (gate) => [
      { match: 'bg', steps: [{ background: { endOn: `${gate}-bg`, wakeAfterMs: 60_000 } }, { holdOn: gate }, ...END] },
    ];
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'bg', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'after the exit', [], false, 'queue');
      await stateUntil(svc, (state) => state.modes.work.tasks === 1);
      writeFileSync(`${gate}-bg`, '');
      await stateUntil(svc, (state) => state.modes.work.tasks === 0);
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      await sleep(300);
      expect(prompts(fake)).toEqual(['bg']);
      expect(page.texts()).toEqual(['after the exit']);

      const pid = fake
        .records()
        .filter((r) => r.kind === 'start')
        .at(-1)!.pid as number;
      process.kill(pid, 'SIGKILL');
      await until(() => page.ends() === 2, 10_000);
      expect(page.outcomes()).toEqual(['completed', 'completed']);
      expect(prompts(fake)).toEqual(['bg', 'after the exit']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      page.detach();
    }, extra);
  }, 30_000);

  it('a row queued while Compact now runs goes when it ends well; Stop of it holds the list', async () => {
    const fake = fakeService('chat-basic', { mutate: (scenario) => void (scenario.compactNow = { ms: 400 }) });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hello' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.ends() === 1, 5_000);
      await svc.compact(id);
      expect(await svc.prompt(id, 'after compact', [], undefined, 'queue')).toEqual({ queued: expect.any(String) });
      await until(() => page.ends() === 2, 10_000);
      expect(prompts(fake)).toEqual(['hello', 'after compact']);

      await svc.compact(id);
      await svc.prompt(id, 'never by itself', [], undefined, 'queue');
      expect(await svc.cancel(id)).toEqual({ cancelled: true });
      await until(() => page.queue().held !== null, 5_000);
      await until(() => page.state?.compaction === null, 5_000);
      await sleep(200);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'never by itself' }] });
      expect(prompts(fake)).toEqual(['hello', 'after compact']);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('the queue — a message typed during an op', () => {
  // Mutation: queue the row without `unheld` → it waits behind the held rows for Send next.
  it('goes when the op ends though older rows are held; the held rows stay held', async () => {
    let gate = '';
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        gate = join(String(scenario.recordTo), '..', 'gate');
        scripts(gate)(scenario);
        scenario.compactNow = { ms: 400 };
      },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold A' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'B', [], undefined, 'queue');
      await svc.cancel(id);
      await until(() => page.ends() === 1, 5_000);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'B' }] });

      await svc.compact(id);
      expect(await svc.prompt(id, 'typed during compact', [], undefined, 'queue')).toEqual({
        queued: expect.any(String),
      });
      expect(page.texts()).toEqual(['B', 'typed during compact']);
      await until(() => page.ends() === 2, 10_000);
      await sleep(150);
      expect(prompts(fake)).toEqual(['hold A', 'typed during compact']);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'B' }] });

      await svc.queueRelease(id);
      await until(() => page.ends() === 3, 5_000);
      expect(prompts(fake)).toEqual(['hold A', 'typed during compact', 'B']);
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('a Stop of the op holds the row typed during it with the rest', async () => {
    let gate = '';
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        gate = join(String(scenario.recordTo), '..', 'gate');
        scripts(gate)(scenario);
        scenario.compactNow = { ms: 400 };
      },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold A' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'B', [], undefined, 'queue');
      await svc.cancel(id);
      await until(() => page.ends() === 1, 5_000);

      await svc.compact(id);
      await svc.prompt(id, 'typed during compact', [], undefined, 'queue');
      expect(await svc.cancel(id)).toEqual({ cancelled: true });
      await until(() => page.state?.compaction === null, 5_000);
      await sleep(300);
      expect(prompts(fake)).toEqual(['hold A']);
      expect(page.queue()).toMatchObject({
        held: { why: 'stopped' },
        items: [{ text: 'B' }, { text: 'typed during compact' }],
      });
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('a recap is an op too: a row typed while it is written goes when it ends, past the held ones', async () => {
    let gate = '';
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        gate = join(String(scenario.recordTo), '..', 'gate');
        scripts(gate)(scenario);
        scenario.recap = { ms: 400 };
      },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'first' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.ends() === 1, 5_000);
      await svc.prompt(id, 'hold A');
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'B', [], undefined, 'queue');
      await svc.cancel(id);
      await until(() => page.ends() === 2, 5_000);
      await svc.recap(id);
      await until(() => page.state?.recap != null, 5_000);
      expect(await svc.prompt(id, 'typed during recap', [], undefined, 'queue')).toEqual({
        queued: expect.any(String),
      });
      await until(() => page.ends() === 3, 10_000);
      expect(prompts(fake)).toEqual(['first', 'hold A', 'typed during recap']);
      expect(page.queue()).toMatchObject({ held: { why: 'stopped' }, items: [{ text: 'B' }] });
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('the queue — a row the agent never got', () => {
  const FILE = { kind: 'file' as const, name: 'notes.md', text: '# notes\n' };
  const SELECT = '/v1/messages/count_tokens';

  function storedRows(fake: FakeService, id: string): Json[] {
    return existsSync(composerFile(fake, id))
      ? (JSON.parse(readFileSync(composerFile(fake, id), 'utf8')) as { queue: Json[] }).queue
      : [];
  }

  // Mutation: remove the row from the queue as soon as its turn is reserved → a Stop while the model is chosen loses it.
  it('a Stop while the row starts gives it back held, with its file; once the prompt went the row leaves', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const engine = await fake.engine();
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold first' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const queued = (await svc.prompt(id, 'second', [FILE], undefined, 'queue')) as { queued: string };
      await until(() => storedRows(fake, id).length === 1);

      let release = (): void => {};
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      const before = engine.selects().length;
      engine.script((request) =>
        request.path === SELECT ? { json: { input_tokens: 1 }, until: held } : { status: 404, json: {} },
      );
      writeFileSync(gate, '');
      await until(() => engine.selects().length > before, 5_000);
      expect(page.state?.turn?.phase).toBe('starting');
      expect(page.texts()).toEqual([]);
      expect(storedRows(fake, id).map((row) => row.id)).toEqual([queued.queued]);
      expect(page.queue()).toEqual({ items: [], held: null, count: 1 });

      await svc.cancel(id);
      release();
      await until(() => page.ends() === 2, 5_000);
      expect(page.outcomes()).toEqual(['completed', 'cancelled']);
      expect(prompts(fake)).toEqual(['hold first']);
      await until(() => page.texts().length === 1);
      expect(page.queue()).toMatchObject({
        held: { why: 'stopped' },
        items: [{ id: queued.queued, text: 'second', files: [{ kind: 'file', name: 'notes.md' }] }],
      });
      expect(storedRows(fake, id)).toMatchObject([{ id: queued.queued, text: 'second', attachments: [FILE] }]);

      engine.script((request) => (request.path === SELECT ? { json: { input_tokens: 1 } } : { status: 404 }));
      await svc.queueRelease(id);
      await until(() => page.ends() === 3, 5_000);
      expect(page.outcomes()[2]).toBe('completed');
      expect(prompts(fake)).toEqual(['hold first', 'second']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      await until(() => !existsSync(composerFile(fake, id)));
      page.detach();
    });
  }, 30_000);

  // Mutation: count only the listed rows in the view → the page reads 4 of 5 and offers Queue the host refuses.
  it('a row whose answer starts still counts: four listed read as five, and the sixth is refused', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const engine = await fake.engine();
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold first' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      for (let i = 1; i <= 5; i++) await svc.prompt(id, `row ${i}`, [], undefined, 'queue');
      await until(() => page.queue().count === 5);
      let release = (): void => {};
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      const before = engine.selects().length;
      engine.script((request) =>
        request.path === SELECT ? { json: { input_tokens: 1 }, until: held } : { status: 404, json: {} },
      );
      writeFileSync(gate, '');
      await until(() => engine.selects().length > before, 5_000);
      expect(page.state?.turn?.phase).toBe('starting');
      await until(() => page.queue().items.length === 4);
      expect(page.queue().count).toBe(5);
      expect(page.queue().items.map((row) => row.text)).toEqual(['row 2', 'row 3', 'row 4', 'row 5']);
      const full = await apiError(svc.prompt(id, 'row 6', [], undefined, 'queue'));
      expect(full).toMatchObject({ code: 'E_CONFLICT', reason: 'queue-full' });
      await svc.cancel(id);
      release();
      await until(() => page.ends() === 2, 5_000);
      await until(() => page.queue().items.length === 5);
      expect(page.queue().count).toBe(5);
      page.detach();
    });
  }, 30_000);

  it('a start that fails before the prompt went gives the row back held', async () => {
    await withChat(async (fake, gate) => {
      const svc = fake.service;
      const engine = await fake.engine();
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold first' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'second', [], undefined, 'queue');
      engine.script(() => ({ status: 404, json: {} }));
      writeFileSync(gate, '');
      await until(() => page.ends() === 2, 5_000);
      expect(prompts(fake)).toEqual(['hold first']);
      await until(() => page.texts().length === 1);
      expect(page.queue()).toMatchObject({ held: { why: 'failed' }, items: [{ text: 'second' }] });
      expect(storedRows(fake, id).map((row) => row.text)).toEqual(['second']);
      page.detach();
    });
  }, 30_000);

  // Mutation: drop the request a drain gets while another one checks its row → the next row waits for good.
  it('the head removed while the drain checks it: the next row still goes', async () => {
    let hold: Promise<void> | null = null;
    let entered = 0;
    let gatePath = '';
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        gatePath = join(String(scenario.recordTo), '..', 'gate');
        scripts(gatePath)(scenario);
      },
      providers: (paths) => {
        const real = memoryProviders(paths);
        const ready = real.chatSkillReady.bind(real);
        real.chatSkillReady = async () => {
          if (hold !== null) {
            entered += 1;
            await hold;
          }
          return ready();
        };
        return real;
      },
    });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold first' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const a = (await svc.prompt(id, 'A', [], undefined, 'queue')) as { queued: string };
      await svc.prompt(id, 'B', [], undefined, 'queue');
      let release = (): void => {};
      hold = new Promise<void>((resolve) => {
        release = resolve;
      });
      writeFileSync(gatePath, '');
      await until(() => entered === 1, 5_000);
      await svc.queueRemove(id, a.queued);
      hold = null;
      release();
      await until(() => page.ends() === 2, 5_000);
      expect(prompts(fake)).toEqual(['hold first', 'B']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      page.detach();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('the queue — plan, restart, delete', () => {
  it('a queued Work row carries the Plan first value it was queued with', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'hold', cwd: project, plan: false });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'plan this', [], true, 'queue');
      await svc.prompt(id, 'then do it', [], false, 'queue');
      await until(() => page.texts().length === 2);
      expect(page.queue().items.map((item) => item.plan)).toEqual([true, false]);
      writeFileSync(gate, '');
      await until(() => page.ends() === 3, 10_000);
      expect(promptModes(fake)).toEqual([
        ['hold', 'agent'],
        ['plan this', 'plan'],
        ['then do it', 'agent'],
      ]);
      page.detach();
    });
  }, 30_000);

  it('a restart brings the rows back held; Send next sends the first; delete removes the file', async () => {
    await withChat(async (fake) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'chat', text: 'hold A' });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'kept B', [], undefined, 'queue');
      await svc.prompt(id, 'kept C', [], undefined, 'queue');
      page.detach();
      // Close ends only after the rows are written.
      await svc.close();

      const svc2 = fake.restart();
      const cold = new Page(svc2, id);
      await until(() => cold.state !== null, 5_000);
      expect(cold.queue()).toMatchObject({ held: { why: 'restart' }, items: [{ text: 'kept B' }, { text: 'kept C' }] });
      await sleep(200);
      expect(prompts(fake)).toEqual(['hold A']);

      await svc2.queueRelease(id);
      await until(() => cold.ends() === 2, 10_000);
      expect(prompts(fake)).toEqual(['hold A', 'kept B', 'kept C']);

      await svc2.prompt(id, 'hold again');
      await until(() => cold.state?.turn?.phase === 'running', 5_000);
      await svc2.prompt(id, 'left over', [], undefined, 'queue');
      await until(() => existsSync(composerFile(fake, id)));
      await svc2.cancel(id);
      await until(() => cold.ends() === 3, 5_000);
      await svc2.delete(id);
      expect(existsSync(composerFile(fake, id))).toBe(false);
      cold.detach();
    });
  }, 40_000);
});
