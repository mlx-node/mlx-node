/**
 * Rewind through the service on the scripted fake agent: the plan reads the prompt index from disk and holds a token;
 * the commit refuses a stale token, stops the removed part's commands (teardown) and deletes its schedules before the
 * rewind, cuts the thread, rebuilds Chat's address ledger, stores Work's rule, and ends on the stream. Chat sends only
 * `conversation_only` and no dry run. The thread read from disk afterwards equals the live cut. Work's plan asks the
 * agent's dry run (`all`, `force: false`) and lists the files it would put back; each of the three modes reaches the
 * agent and its rule.
 */

import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ConversationEvent, RewindPlan } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeService, until, type FakeService, type Json } from './helpers/chat-fake.js';

interface Watched {
  /** The page's state; read it after {@link idle}. */
  state: () => ConversationState;
  current: () => ConversationState | null;
  events: ConversationEvent[];
  off: () => void;
}

function watch(service: FakeService['service'], id: string): Watched {
  const events: ConversationEvent[] = [];
  let state: ConversationState | null = null;
  const off = service.watchConversation(id, (event) => {
    events.push(event);
    const next = applyConversationEvent(state, event);
    if (next !== 'resync') state = next;
  });
  return { state: () => state!, current: () => state, events, off };
}

/** Waits for the answer to the `sent`-th message and no turn (a command the turn started may run on). */
async function idle(w: Watched, sent: number): Promise<void> {
  await until(() => {
    const state = w.current();
    if (state === null) return false;
    const users = state.entries.filter((e) => e.kind === 'user').length;
    return users === sent && state.entries.at(-1)?.kind !== 'user' && state.turn === null;
  }, 8_000);
}

/** The page's name of the entry at `at`: its index and its time. */
const entryAt = (entries: readonly TranscriptEntry[], at: number): { entry: number; ts: number } => ({
  entry: at,
  ts: entries[at]!.ts,
});

const userAt = (entries: readonly TranscriptEntry[], text: string): number =>
  entries.findIndex((e) => e.kind === 'user' && e.text === text);

/** The time of the page's entry `at`: a plan names the entry by its index and its time. */
const tsAt = (w: Watched, at: number): number => w.state().entries[at]?.ts ?? -1;

const refusal = async (run: Promise<unknown>): Promise<{ code?: string; reason?: string }> =>
  routeError(
    'chat',
    await run.then(
      () => null,
      (error: unknown) => error,
    ),
  ) as unknown as { code: string; reason: string };

/** The parts of an entry the disk and the live view both have (a live thought has times, a stored one none). */
const shape = (entries: readonly TranscriptEntry[]): unknown[] =>
  entries.map((e) => (e.kind === 'user' || e.kind === 'assistant' ? [e.kind, e.text] : [e.kind]));

describe('Rewind through the service: Chat', () => {
  it('plan, stale token, commit: thread cut, end on the stream, only conversation_only, ledger rebuilt, disk = live', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'see https://example.com/page-two');
      await idle(w, 2);
      const at = userAt(w.state().entries, 'see https://example.com/page-two');
      expect(at).toBeGreaterThan(0);
      const ledgerText = (): string =>
        readdirSync(fake.paths.webLedger)
          .map((name) => readFileSync(join(fake.paths.webLedger, name), 'utf8'))
          .join('\n');
      expect(ledgerText()).toContain('example.com/page-two');

      expect(await refusal(fake.service.rewindPlan(id, at + 1, tsAt(w, at + 1)))).toMatchObject({
        reason: 'rewind-not-point',
      });
      const plan: RewindPlan = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(plan).toEqual({
        token: expect.any(String),
        removes: { messages: 1, steps: w.state().entries.length - at - 1 - 1 },
        background: { commands: 0, schedules: 0, runs: 0 },
        helpers: false,
        attachments: false,
        files: [],
        filesMore: 0,
      });
      expect(await refusal(fake.service.rewind(id, { token: 'stale', mode: 'conversation' }))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'rewind-changed',
      });
      expect(await fake.service.rewind(id, { token: plan.token, mode: 'conversation' })).toEqual({});
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: at },
      });
      expect(w.state().entries).toHaveLength(at);
      expect(w.state().turns).toHaveLength(1);
      const order = w.events.map((e) => e.type).filter((t) => t === 'sync' || t === 'rewind');
      expect(order.slice(-2)).toEqual(['sync', 'rewind']);
      expect(w.state().permissionMarks).toEqual([]);
      const rewinds = fake.records().filter((r) => r.kind === 'rewind');
      expect(rewinds).toEqual([
        { kind: 'rewind', target: 1, force: true, mode: 'conversation_only', hasMode: true, hasForce: true },
      ]);
      expect(ledgerText()).not.toContain('example.com/page-two');
      // The token of the plan was used: a second commit is refused.
      expect(await refusal(fake.service.rewind(id, { token: plan.token, mode: 'conversation' }))).toMatchObject({
        reason: 'rewind-changed',
      });

      const disk = fake.reader.detail(id)!;
      expect(shape(disk.transcript)).toEqual(shape(w.state().entries));
      expect(disk.turns).toHaveLength(w.state().turns.length);

      // A later message lands after the cut, and the next rewind finds it by order.
      await fake.service.prompt(id, 'three');
      await idle(w, 2);
      const third = userAt(w.state().entries, 'three');
      expect(third).toBe(at);
      const again = await fake.service.rewindPlan(id, third, tsAt(w, third));
      await fake.service.rewind(id, { token: again.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd?.end.outcome === 'done' && w.state().entries.length === at, 8_000);
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'rewind')
          .map((r) => r.target),
      ).toEqual([1, 1]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('after a Control Panel restart the cold conversation is loaded first, then rewound', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'two');
      await idle(w, 2);
      w.off();
      await fake.service.close();
      const next = fake.restart();
      const after = watch(next, id);
      await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
      const at = userAt(after.state().entries, 'two');
      const plan = await next.rewindPlan(id, at, tsAt(after, at));
      await next.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => after.current()?.rewindEnd !== null, 8_000);
      expect(after.state().rewindEnd).toEqual({
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: at },
      });
      const methods = fake
        .sent()
        .map((m) => m.method)
        .filter((m) => m === 'session/load' || m === '_x.ai/rewind/execute');
      expect(methods.slice(-2)).toEqual(['session/load', '_x.ai/rewind/execute']);
      expect(shape(fake.reader.detail(id)!.transcript)).toEqual(shape(after.state().entries));
      after.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('the agent saying no: no cut, the end says refused; a message in the plan window makes the token stale', async () => {
    const fake = fakeService('chat-basic', { mutate: (scenario) => void (scenario.rewind = { refuse: true }) });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      const before = w.state().entries.length;
      const plan = await fake.service.rewindPlan(id, 0, tsAt(w, 0));
      await fake.service.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'refused' } });
      expect(w.state().entries).toHaveLength(before);
      const next = await fake.service.rewindPlan(id, 0, tsAt(w, 0));
      await fake.service.prompt(id, 'two');
      expect(await refusal(fake.service.rewind(id, { token: next.token, mode: 'conversation' }))).toMatchObject({
        reason: 'rewind-changed',
      });
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: resolve the plan by the index alone → the stale window plans the message that took the place of its own.
  it('a window that still shows a cut message: its plan for that place is refused, not made for the message there now', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'two');
      await idle(w, 2);
      const at = userAt(w.state().entries, 'two');
      const seen = tsAt(w, at);
      const plan = await fake.service.rewindPlan(id, at, seen);
      await fake.service.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      await fake.service.prompt(id, 'three');
      await idle(w, 2);
      expect(userAt(w.state().entries, 'three')).toBe(at);
      expect(await refusal(fake.service.rewindPlan(id, at, seen))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'rewind-changed',
      });
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: do not mark the prompt of a turn that ended unsent → the failed send takes the retry's prompt and the
  // retry is no rewind point.
  it('a send stopped before it reached the agent, then the same text: the retry rewinds, the stopped one is no point', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'again');
      await fake.service.cancel(id);
      await until(() => w.state().turn === null && w.state().lastEnd?.unsent === true, 8_000);
      await fake.service.prompt(id, 'again');
      await idle(w, 3);
      const stopped = userAt(w.state().entries, 'again');
      const retry = w.state().entries.findLastIndex((e) => e.kind === 'user' && e.text === 'again');
      expect(retry).toBeGreaterThan(stopped);
      expect(await refusal(fake.service.rewindPlan(id, stopped, tsAt(w, stopped)))).toMatchObject({
        reason: 'rewind-not-point',
      });
      const plan = await fake.service.rewindPlan(id, retry, tsAt(w, retry));
      await fake.service.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: retry },
      });
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'rewind')
          .map((r) => r.target),
      ).toEqual([1]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: drop the entry from `rewindToken` → the first window's commit runs the second window's plan.
  it('two windows plan two messages: the older plan is refused, the newer one cuts at its own message', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'two');
      await idle(w, 2);
      const second = userAt(w.state().entries, 'two');
      const first = await fake.service.rewindPlan(id, 0, tsAt(w, 0));
      const other = await fake.service.rewindPlan(id, second, tsAt(w, second));
      expect(await refusal(fake.service.rewind(id, { token: first.token, mode: 'conversation' }))).toMatchObject({
        reason: 'rewind-changed',
      });
      expect(fake.records().filter((r) => r.kind === 'rewind')).toEqual([]);
      await fake.service.rewind(id, { token: other.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: second },
      });
      expect(w.state().entries).toHaveLength(second);
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'rewind')
          .map((r) => r.target),
      ).toEqual([1]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('refuses while an answer runs and while the engine is stopped', async () => {
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scenario.prompts = [
          { match: 'slow', steps: [{ sleepMs: 1_500 }, { text: 'late' }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      const plan = await fake.service.rewindPlan(id, 0, tsAt(w, 0));
      await fake.service.prompt(id, 'slow');
      expect(await refusal(fake.service.rewindPlan(id, 0, tsAt(w, 0)))).toMatchObject({ reason: 'conversation-busy' });
      await idle(w, 2);
      const fresh = await fake.service.rewindPlan(id, 0, tsAt(w, 0));
      expect(fresh.token).not.toBe(plan.token);
      const endpoint = fake.endpoint.current;
      fake.endpoint.current = null;
      expect(await refusal(fake.service.rewind(id, { token: fresh.token, mode: 'conversation' }))).toMatchObject({
        reason: 'inference-stopped',
      });
      fake.endpoint.current = endpoint;
      expect(fake.records().filter((r) => r.kind === 'rewind')).toEqual([]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('Rewind through the service: Work', () => {
  function steps(scenario: Json, faults: Json = {}): void {
    scenario.prompts = [
      {
        match: 'cmd',
        steps: [
          { background: { taskId: 't1', wake: false } },
          { text: 'Started.' },
          { end: { stopReason: 'end_turn' } },
        ],
      },
      {
        match: 'sched',
        steps: [{ schedule: { taskId: 's1' } }, { text: 'Made.' }, { end: { stopReason: 'end_turn' } }],
      },
    ];
    scenario.faults = { ...(scenario.faults as Json | undefined), ...faults };
  }

  async function workConversation(fake: FakeService): Promise<{ id: string; w: Watched }> {
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const { conversation } = await fake.service.create({ mode: 'work', text: 'first', cwd: project });
    const id = conversation.id;
    const w = watch(fake.service, id);
    await idle(w, 1);
    for (const [n, text] of ['cmd', 'sched'].entries()) {
      await fake.service.prompt(id, text);
      await idle(w, n + 2);
    }
    return { id, w };
  }

  // Mutation: drop the stops before `execute` → no kill is sent and the command runs on.
  it('the plan names the removed part’s command and schedule; the commit ends them first, cuts, and stores the rule', async () => {
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => steps(scenario) });
    try {
      const { id, w } = await workConversation(fake);
      await until(() => w.state().live.tasks.length === 1, 5_000);
      const at = userAt(w.state().entries, 'cmd');
      const plan = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(plan.background).toEqual({ commands: 1, schedules: 1, runs: 0 });
      await fake.service.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({
        n: 1,
        end: { outcome: 'done', mode: 'conversation', restored: 0, entry: at },
      });
      // The plan's dry run comes first; the commit after the stops.
      const kinds = fake
        .records()
        .filter((r) => r.kind !== 'rewind' || r.force === true)
        .map((r) => r.kind)
        .filter((k) => k === 'kill' || k === 'schedule-delete' || k === 'rewind');
      expect(kinds.at(-1)).toBe('rewind');
      expect(kinds.slice(0, 2).sort()).toEqual(['kill', 'schedule-delete']);
      expect(fake.records().find((r) => r.kind === 'kill')).toMatchObject({ taskId: 't1', source: 'teardown' });
      expect(fake.records().filter((r) => r.kind === 'rewind')).toMatchObject([
        { target: 1, force: false, mode: 'all' },
        { target: 1, force: true, mode: 'conversation_only' },
      ]);
      expect(w.state().entries).toHaveLength(at);
      await until(() => w.state().live.tasks.length === 0, 5_000);
      const marks = w.state().permissionMarks.filter((m) => m.kind === 'rewind');
      expect(marks).toEqual([
        { kind: 'rewind', at: expect.any(Number), mode: 'conversation', files: 0, from: expect.any(Number) },
      ]);
      const log = readFileSync(join(fake.paths.permissionLog, `${id}.jsonl`), 'utf8');
      const line = log
        .split('\n')
        .filter((l) => l !== '')
        .map((l) => JSON.parse(l) as Json)
        .find((l) => l.kind === 'rewind');
      expect(Object.keys(line!).sort()).toEqual(['at', 'files', 'from', 'kind', 'mode']);
      w.off();
    } finally {
      await fake.close();
    }
  }, 40_000);

  // Mutation: send no runs with the commit → the paused run is not stopped, and the cut conversation could resume it.
  it('the plan names the paused run the removed part launched; the commit stops it before the cut (T39)', async () => {
    const paused = { workflow: { runId: 'wf_1', ask: false, completeAfterMs: 0, status: 'user_paused', wake: false } };
    const fake = fakeService('work-workflow-run', {
      mutate: (scenario) => {
        steps(scenario);
        const cmd = (scenario.prompts as Json[])[0]!;
        cmd.steps = [paused, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }];
      },
    });
    try {
      const { id, w } = await workConversation(fake);
      await until(() => w.state().runs[0]?.status === 'user_paused', 5_000);
      const later = userAt(w.state().entries, 'sched');
      const kept = await fake.service.rewindPlan(id, later, tsAt(w, later));
      expect(kept.background).toEqual({ commands: 0, schedules: 1, runs: 0 });
      const at = userAt(w.state().entries, 'cmd');
      const plan = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(plan.background).toEqual({ commands: 0, schedules: 1, runs: 1 });
      await fake.service.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd?.end).toMatchObject({ outcome: 'done' });
      const kinds = fake
        .records()
        .filter((r) => r.kind !== 'rewind' || r.force === true)
        .map((r) => r.kind)
        .filter((k) => k === 'stop' || k === 'rewind');
      expect(kinds).toEqual(['stop', 'rewind']);
      expect(fake.records().find((r) => r.kind === 'stop')).toMatchObject({ runId: 'wf_1' });
      w.off();
    } finally {
      await fake.close();
    }
  }, 40_000);

  it('a stop that fails: nothing is rewound and the end says so', async () => {
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => steps(scenario, { killReply: 'error' }) });
    try {
      const { id, w } = await workConversation(fake);
      await until(() => w.state().live.tasks.length === 1, 5_000);
      const before = w.state().entries.length;
      const plan = await fake.service.rewindPlan(
        id,
        userAt(w.state().entries, 'cmd'),
        tsAt(w, userAt(w.state().entries, 'cmd')),
      );
      await fake.service.rewind(id, { token: plan.token, mode: 'conversation' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'tasks' } });
      expect(fake.records().filter((r) => r.kind === 'rewind' && r.force === true)).toEqual([]);
      expect(w.state().entries).toHaveLength(before);
      expect(w.state().permissionMarks.filter((m) => m.kind === 'rewind')).toEqual([]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 40_000);
});

describe('Rewind files through the service (Work)', () => {
  const EDITS = [
    {
      toolName: 'search_replace',
      file_path: 'src/a.ts',
      before: 'a1\n',
      old_string: 'a1',
      new_string: 'a2',
      ask: false,
    },
    { toolName: 'write', file_path: 'src/b.ts', before: null, content: 'b\n', ask: false },
  ];

  function steps(scenario: Json, rewind: Json = {}): void {
    scenario.prompts = [
      {
        match: 'edit',
        steps: [...EDITS.map((edit) => ({ edit })), { text: 'Edited.' }, { end: { stopReason: 'end_turn' } }],
      },
      {
        match: 'cmd',
        steps: [
          { background: { taskId: 't1', wake: false } },
          { text: 'Started.' },
          { end: { stopReason: 'end_turn' } },
        ],
      },
    ];
    scenario.rewind = rewind;
  }

  async function workConversation(fake: FakeService, words: string[]): Promise<{ id: string; w: Watched }> {
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const { conversation } = await fake.service.create({ mode: 'work', text: 'first', cwd: project });
    const id = conversation.id;
    const w = watch(fake.service, id);
    await idle(w, 1);
    for (const [n, text] of words.entries()) {
      await fake.service.prompt(id, text);
      await idle(w, n + 2);
    }
    return { id, w };
  }

  const rewinds = (fake: FakeService): Json[] => fake.records().filter((r) => r.kind === 'rewind');

  // Mutation: send the dry run without `mode` → the record says it was left out (the agent's default `all`).
  it('the plan asks the dry run and lists the files, conflicts first; Conversation and files puts them back, cuts and stores the rule', async () => {
    const fake = fakeService('work-workflow-run', {
      mutate: (scenario) => steps(scenario, { changed: { 'src/b.ts': 'modified_externally' } }),
    });
    try {
      const { id, w } = await workConversation(fake, ['edit']);
      const at = userAt(w.state().entries, 'edit');
      expect(await refusal(fake.service.rewindPlan(id, at + 1, tsAt(w, at + 1)))).toMatchObject({
        reason: 'rewind-not-point',
      });
      expect(rewinds(fake)).toEqual([]);
      const plan = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(plan.files).toEqual([
        { path: 'src/b.ts', state: 'modified' },
        { path: 'src/a.ts', state: 'clean' },
      ]);
      expect(plan.filesMore).toBe(0);
      expect(rewinds(fake)).toEqual([
        { kind: 'rewind', target: 1, force: false, mode: 'all', hasMode: true, hasForce: true },
      ]);
      const first = await fake.service.rewindPlan(id, 0, tsAt(w, 0));
      expect(first.files).toEqual(plan.files);
      const again = await fake.service.rewindPlan(id, at, tsAt(w, at));
      await fake.service.rewind(id, { token: again.token, mode: 'all' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'done', mode: 'all', restored: 2, entry: at } });
      expect(rewinds(fake).at(-1)).toEqual({
        kind: 'rewind',
        target: 1,
        force: true,
        mode: 'all',
        hasMode: true,
        hasForce: true,
      });
      expect(w.state().entries).toHaveLength(at);
      expect(w.state().permissionMarks.filter((m) => m.kind === 'rewind')).toEqual([
        { kind: 'rewind', at: expect.any(Number), mode: 'all', files: 2, from: expect.any(Number) },
      ]);
      // The restore dropped the agent's points: the files are back, nothing is left to put back.
      expect((await fake.service.rewindPlan(id, 0, tsAt(w, 0))).files).toEqual([]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 40_000);

  // Mutation: commit without the dry run again → the hand edit after the plan is overwritten and the end is `done`.
  it('a file changed after the plan: nothing is rewound, the end says so; a file the agent could not write is named', async () => {
    const hand = join(mkdtempSync(join(tmpdir(), 'mlx-rewind-hand-')), 'changed.json');
    const fake = fakeService('work-workflow-run', {
      mutate: (scenario) => steps(scenario, { changedFile: hand, unwritable: ['src/a.ts'] }),
    });
    try {
      const { id, w } = await workConversation(fake, ['edit']);
      const before = w.state().entries.length;
      const at = userAt(w.state().entries, 'edit');
      const plan = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(plan.files.map((f) => f.state)).toEqual(['clean', 'clean']);
      writeFileSync(hand, JSON.stringify({ 'src/b.ts': 'modified_externally' }));
      await fake.service.rewind(id, { token: plan.token, mode: 'all' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'changed' } });
      expect(rewinds(fake).map((r) => r.force)).toEqual([false, false]);
      expect(w.state().entries).toHaveLength(before);
      expect(w.state().permissionMarks.filter((m) => m.kind === 'rewind')).toEqual([]);

      // The plan again shows the change; the user overwrites it. src/a.ts cannot be written: the agent says success.
      const again = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(again.files).toEqual([
        { path: 'src/b.ts', state: 'modified' },
        { path: 'src/a.ts', state: 'clean' },
      ]);
      await fake.service.rewind(id, { token: again.token, mode: 'all' });
      await until(() => w.state().rewindEnd?.n === 2, 8_000);
      expect(w.state().rewindEnd).toEqual({
        n: 2,
        end: { outcome: 'partial', mode: 'all', restored: 1, entry: at, failed: ['src/a.ts'], failedMore: 0 },
      });
      expect(rewinds(fake).map((r) => r.force)).toEqual([false, false, false, false, true]);
      expect(w.state().entries).toHaveLength(at);
      expect(w.state().permissionMarks.filter((m) => m.kind === 'rewind')).toEqual([
        { kind: 'rewind', at: expect.any(Number), mode: 'all', files: 1, from: expect.any(Number) },
      ]);
      w.off();
    } finally {
      await fake.close();
      rmSync(join(hand, '..'), { recursive: true, force: true });
    }
  }, 40_000);

  // Mutation: stop the removed part's commands for Files only too → a kill is sent.
  it('Files only: the thread stays, no command is stopped, the rule has no span; a second plan finds no files', async () => {
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => steps(scenario) });
    try {
      const { id, w } = await workConversation(fake, ['edit', 'cmd']);
      await until(() => w.state().live.tasks.length === 1, 5_000);
      const before = w.state().entries.length;
      const at = userAt(w.state().entries, 'edit');
      const plan = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(plan.background).toEqual({ commands: 1, schedules: 0, runs: 0 });
      expect(plan.files.map((f) => f.path)).toEqual(['src/a.ts', 'src/b.ts']);
      await fake.service.rewind(id, { token: plan.token, mode: 'files' });
      await until(() => w.state().rewindEnd !== null, 8_000);
      expect(w.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'done', mode: 'files', restored: 2, entry: at } });
      expect(fake.records().filter((r) => r.kind === 'kill')).toEqual([]);
      expect(rewinds(fake).at(-1)).toMatchObject({ force: true, mode: 'files_only', target: 1 });
      expect(w.state().entries).toHaveLength(before);
      expect(w.state().live.tasks).toHaveLength(1);
      const marks = w.state().permissionMarks.filter((m) => m.kind === 'rewind');
      expect(marks).toEqual([{ kind: 'rewind', at: expect.any(Number), mode: 'files', files: 2 }]);
      const log = readFileSync(join(fake.paths.permissionLog, `${id}.jsonl`), 'utf8');
      expect(log).not.toContain('src/');
      const next = await fake.service.rewindPlan(id, at, tsAt(w, at));
      expect(next.files).toEqual([]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 40_000);

  // Mutation: give the dry run a hard deadline → the miss kills the agent (a second start).
  it('a dry run with no answer in time is agent-slow and the agent stays; a cold conversation loads before it', async () => {
    const fake = fakeService('work-workflow-run', {
      mutate: (scenario) => steps(scenario, { previewMs: 1_500 }),
      timings: { rewindPreviewMs: 200 },
    });
    try {
      const { id, w } = await workConversation(fake, ['edit']);
      const at = userAt(w.state().entries, 'edit');
      expect(await refusal(fake.service.rewindPlan(id, at, tsAt(w, at)))).toMatchObject({
        code: 'E_UNAVAILABLE',
        reason: 'agent-slow',
      });
      await new Promise((resolve) => setTimeout(resolve, 1_600));
      expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(1);
      expect((await fake.service.state()).modes.work.state).toBe('ready');
      w.off();
      await fake.service.close();
      const next = fake.restart();
      const after = watch(next, id);
      await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
      const slow = refusal(
        next.rewindPlan(id, userAt(after.state().entries, 'edit'), tsAt(after, userAt(after.state().entries, 'edit'))),
      );
      expect(await slow).toMatchObject({ reason: 'agent-slow' });
      const methods = fake
        .sent()
        .map((m) => m.method)
        .filter((m) => m === 'session/load' || m === '_x.ai/rewind/execute');
      expect(methods.slice(-2)).toEqual(['session/load', '_x.ai/rewind/execute']);
      after.off();
    } finally {
      await fake.close();
    }
  }, 40_000);

  // Mutation: forget the file points when the agent exits → the cold plan lists no files.
  it('after a Control Panel restart the dry run loads the cold conversation, finds the files and the plan rewinds', async () => {
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => steps(scenario) });
    try {
      const { id, w } = await workConversation(fake, ['edit']);
      w.off();
      await fake.service.close();
      const next = fake.restart();
      const after = watch(next, id);
      await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
      expect(after.state().conversation.hot).toBe(false);
      const at = userAt(after.state().entries, 'edit');
      const plan = await next.rewindPlan(id, at, tsAt(after, at));
      expect(plan.files).toEqual([
        { path: 'src/a.ts', state: 'clean' },
        { path: 'src/b.ts', state: 'clean' },
      ]);
      const methods = fake
        .sent()
        .map((m) => m.method)
        .filter((m) => m === 'session/load' || m === '_x.ai/rewind/execute');
      expect(methods.slice(-2)).toEqual(['session/load', '_x.ai/rewind/execute']);
      await until(() => after.state().conversation.hot, 5_000);
      await next.rewind(id, { token: plan.token, mode: 'all' });
      await until(() => after.current()?.rewindEnd !== null, 8_000);
      expect(after.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'done', mode: 'all', restored: 2, entry: at } });
      expect(after.state().entries).toHaveLength(at);
      const spoken = after.state().permissionMarks.filter((m) => m.kind === 'rewind');
      expect(spoken).toEqual([
        { kind: 'rewind', at: expect.any(Number), mode: 'all', files: 2, from: expect.any(Number) },
      ]);
      after.off();
      await next.close();
      const third = fake.restart();
      const reopened = watch(third, id);
      await until(() => reopened.current() !== null && reopened.state().entries.length > 0, 8_000);
      expect(reopened.state().entries).toHaveLength(at);
      expect(reopened.state().permissionMarks.filter((m) => m.kind === 'rewind')).toEqual(spoken);
      reopened.off();
    } finally {
      await fake.close();
    }
  }, 40_000);

  it('Chat refuses a file mode; Work refuses the plan while the engine is stopped', async () => {
    const chat = fakeService('chat-basic');
    try {
      const { conversation } = await chat.service.create({ mode: 'chat', text: 'one' });
      const cw = watch(chat.service, conversation.id);
      await idle(cw, 1);
      const chatPlan = await chat.service.rewindPlan(conversation.id, 0, tsAt(cw, 0));
      expect(chatPlan.files).toEqual([]);
      for (const mode of ['all', 'files'] as const) {
        expect(await refusal(chat.service.rewind(conversation.id, { token: chatPlan.token, mode }))).toMatchObject({
          code: 'E_BAD_REQUEST',
        });
      }
      expect(rewinds(chat)).toEqual([]);
      cw.off();
    } finally {
      await chat.close();
    }
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => steps(scenario) });
    try {
      const { id, w } = await workConversation(fake, ['edit']);
      const endpoint = fake.endpoint.current;
      fake.endpoint.current = null;
      expect(
        await refusal(
          fake.service.rewindPlan(id, userAt(w.state().entries, 'edit'), tsAt(w, userAt(w.state().entries, 'edit'))),
        ),
      ).toMatchObject({
        reason: 'inference-stopped',
      });
      fake.endpoint.current = endpoint;
      expect(rewinds(fake)).toEqual([]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 40_000);
});

describe('A cold conversation the agent opens too slowly', () => {
  // Mutation: send the host's `failed` end as it came → the page reads "could not start" for a slow open.
  it('a rewind ends load-slow on the stream and cuts nothing; a fork of a message that moved is fork-changed', async () => {
    const fake = fakeService('chat-basic', {
      timings: { loadSoftMs: 300 },
      mutate: (scenario) => {
        scenario.faults = { delayMs: { 'session/load': 1_500 } };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'two');
      await idle(w, 2);
      w.off();
      await fake.service.close();
      const next = fake.restart();
      const after = watch(next, id);
      await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
      const entries = after.state().entries;
      const at = userAt(entries, 'two');
      const plan = await next.rewindPlan(id, at, entries[at]!.ts);
      expect(await next.rewind(id, { token: plan.token, mode: 'conversation' })).toEqual({});
      await until(() => after.state().rewindEnd !== null, 8_000);
      expect(after.state().rewindEnd).toEqual({ n: 1, end: { outcome: 'load-slow' } });
      expect(after.state().entries).toEqual(entries);
      expect(fake.records().filter((r) => r.kind === 'rewind')).toEqual([]);

      // Mutation: drop the time check → the fork copies up to the message that is there now.
      expect(await refusal(next.fork(id, { entry: at, ts: entries[at]!.ts + 1 }))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'fork-changed',
      });
      expect(fake.records().filter((r) => r.kind === 'fork')).toEqual([]);
      after.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: leave `hot` to the op's own end → the late load lands with no waiter and the page reads it cold.
  it('Work: a plan whose cold load is too slow is 503 load-slow; when the late load lands the conversation reads hot', async () => {
    const fake = fakeService('work-workflow-run', {
      timings: { loadSoftMs: 300 },
      mutate: (scenario) => {
        scenario.prompts = [];
        scenario.faults = { delayMs: { 'session/load': 1_500 } };
      },
    });
    try {
      const project = join(fake.root, 'home', 'code', 'app');
      mkdirSync(project, { recursive: true });
      const { conversation } = await fake.service.create({ mode: 'work', text: 'first', cwd: project });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      w.off();
      await fake.service.close();
      const next = fake.restart();
      const after = watch(next, id);
      await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
      expect(after.state().conversation.hot).not.toBe(true);
      const entries = after.state().entries;
      const at = userAt(entries, 'first');
      expect(await refusal(next.rewindPlan(id, at, entries[at]!.ts))).toMatchObject({
        code: 'E_UNAVAILABLE',
        reason: 'load-slow',
      });
      expect(after.state().conversation.hot).not.toBe(true);
      await until(() => after.state().conversation.hot === true, 8_000);
      expect(fake.records().filter((r) => r.kind === 'rewind')).toEqual([]);
      after.off();
    } finally {
      await fake.close();
    }
  }, 30_000);
});

describe('Fork through the service', () => {
  const forks = (fake: FakeService): Json[] =>
    fake
      .records()
      .filter((r) => r.kind === 'fork')
      .map((r) => r.params as Json);
  const typed = (entries: readonly TranscriptEntry[]): string[] =>
    entries.flatMap((e) => (e.kind === 'user' ? [e.text] : []));

  it('Chat: the copy is on disk as a fork, opens with its origin, writes no index line; a message forks with its answer', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'two');
      await idle(w, 2);
      const changed: string[] = [];
      const offHost = fake.service.watchHost((event) => {
        if (event.type === 'conversations-changed') changed.push(event.type);
      });

      const whole = await fake.service.fork(id);
      expect(whole.id).not.toBe(id);
      expect(changed).toHaveLength(1);
      expect(forks(fake)).toHaveLength(1);
      expect(forks(fake)[0]).not.toHaveProperty('targetPromptIndex');
      const row = fake.reader.detail(whole.id)!.session;
      expect(row).toMatchObject({ kind: 'fork', parentId: id, origin: 'chat' });
      expect(typed(fake.reader.detail(whole.id)!.transcript)).toEqual(['one', 'two']);
      const c = watch(fake.service, whole.id);
      await until(() => c.current() !== null);
      expect(c.state().conversation.forkOf).toEqual({ id, title: conversation.title });
      expect(c.state().conversation.permissionMode).toBe('ask');
      await fake.service.prompt(whole.id, 'three');
      await idle(c, 3);
      expect(fake.reader.detail(whole.id)!.session).toMatchObject({ kind: 'fork', parentId: id });
      c.off();

      const first = userAt(w.state().entries, 'one');
      const partial = await fake.service.fork(id, entryAt(w.state().entries, first));
      expect(forks(fake)[1]).toMatchObject({ targetPromptIndex: 0 });
      expect(typed(fake.reader.detail(partial.id)!.transcript)).toEqual(['one']);
      expect(fake.reader.detail(partial.id)!.transcript.some((e) => e.kind === 'assistant')).toBe(true);
      expect(await refusal(fake.service.fork(id, entryAt(w.state().entries, first + 1)))).toMatchObject({
        code: 'E_CONFLICT',
        reason: 'fork-not-point',
      });
      expect(forks(fake)).toHaveLength(2);
      // Mutation: send the rewind's exclusive index (one less) → the second message's fork names prompt 0.
      const upToTwo = await fake.service.fork(id, entryAt(w.state().entries, userAt(w.state().entries, 'two')));
      expect(forks(fake)[2]).toMatchObject({ targetPromptIndex: 1 });
      expect(typed(fake.reader.detail(upToTwo.id)!.transcript)).toEqual(['one', 'two']);
      expect(() => readFileSync(fake.paths.conversations, 'utf8')).toThrow();

      // Deleting the original: the copy held here now reads as a fork of a deleted conversation.
      const held = watch(fake.service, partial.id);
      await until(() => held.current() !== null);
      await fake.service.delete(id);
      await until(() => held.state().conversation.forkOf?.title === null);
      expect(held.state().conversation.forkOf).toEqual({ id, title: null });
      held.off();
      offHost();
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('after a Control Panel restart a cold conversation forks with the agent started and nothing loaded', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'two');
      await idle(w, 2);
      w.off();
      await fake.service.close();
      const next = fake.restart();
      const after = watch(next, id);
      await until(() => after.current() !== null && after.state().entries.length > 0, 8_000);
      const copy = await next.fork(id, entryAt(after.state().entries, userAt(after.state().entries, 'two')));
      const sent = fake.sent().map((m) => m.method);
      expect(sent.slice(sent.lastIndexOf('initialize'))).not.toContain('session/load');
      const params = forks(fake)[0]!;
      expect(params).toMatchObject({ sourceSessionId: id, targetPromptIndex: 1 });
      expect(params.newCwd).toBe(params.sourceCwd);
      expect(fake.reader.detail(copy.id)!.session).toMatchObject({ kind: 'fork', parentId: id });
      expect(typed(fake.reader.detail(copy.id)!.transcript)).toEqual(['one', 'two']);
      after.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('refuses while an answer runs, and an agent error is the route error; nothing is copied', async () => {
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scenario.prompts = [
          { match: 'slow', steps: [{ sleepMs: 1_500 }, { text: 'late' }, { end: { stopReason: 'end_turn' } }] },
        ];
        scenario.fork = { rpcError: true };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      await fake.service.prompt(id, 'slow');
      expect(await refusal(fake.service.fork(id))).toMatchObject({ reason: 'conversation-busy' });
      expect(forks(fake)).toEqual([]);
      await idle(w, 2);
      expect(await refusal(fake.service.fork(id))).toMatchObject({ reason: 'agent-protocol-error' });
      expect(forks(fake)).toHaveLength(1);
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: drop the queue's drain at the fork's end → the queued row waits for the user.
  it('a row queued while the copy is made goes to the original once the fork ends', async () => {
    const gateDir = mkdtempSync(join(tmpdir(), 'mlx-fork-gate-'));
    const go = join(gateDir, 'go');
    const fake = fakeService('chat-basic', { mutate: (scenario) => void (scenario.fork = { holdOn: go }) });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      const forking = fake.service.fork(id);
      await until(() => forks(fake).length === 1);
      expect(await fake.service.prompt(id, 'two', undefined, undefined, 'queue')).toHaveProperty('queued');
      expect(await refusal(fake.service.prompt(id, 'three'))).toMatchObject({ reason: 'conversation-busy' });
      writeFileSync(go, '');
      const copy = await forking;
      await idle(w, 2);
      expect(typed(w.state().entries)).toEqual(['one', 'two']);
      expect(typed(fake.reader.detail(copy.id)!.transcript)).toEqual(['one']);
      w.off();
    } finally {
      await fake.close();
      rmSync(gateDir, { recursive: true, force: true });
    }
  }, 30_000);

  async function workConversation(fake: FakeService): Promise<{ id: string; w: Watched; project: string }> {
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const { conversation } = await fake.service.create({ mode: 'work', text: 'first', cwd: project });
    const w = watch(fake.service, conversation.id);
    await idle(w, 1);
    return { id: conversation.id, w, project };
  }

  const indexed = (fake: FakeService): string[] =>
    (JSON.parse(readFileSync(fake.paths.conversations, 'utf8')) as { work: Array<{ id: string }> }).work.map(
      (e) => e.id,
    );

  it('Work: the index gains the copy, which opens as a Work fork in the same folder', async () => {
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => void (scenario.prompts = []) });
    try {
      const { id, w, project } = await workConversation(fake);
      const copy = await fake.service.fork(id);
      expect(indexed(fake)).toEqual(expect.arrayContaining([id, copy.id]));
      expect(forks(fake)[0]).toMatchObject({ sourceCwd: realpathSync(project), newCwd: realpathSync(project) });
      const c = watch(fake.service, copy.id);
      await until(() => c.current() !== null);
      expect(c.state().conversation).toMatchObject({
        mode: 'work',
        cwd: realpathSync(project),
        permissionMode: 'ask',
        forkOf: { id, title: expect.any(String) },
      });
      c.off();
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: drop the delete after a failed index write → the copy stays in the agent home.
  it('Work: a failed index write deletes the copy through the agent and refuses the fork', async () => {
    const fake = fakeService('work-workflow-run', { mutate: (scenario) => void (scenario.prompts = []) });
    try {
      const { id, w } = await workConversation(fake);
      chmodSync(fake.paths.root, 0o500);
      let refused: unknown = null;
      try {
        await fake.service.fork(id);
      } catch (error) {
        refused = error;
      } finally {
        chmodSync(fake.paths.root, 0o700);
      }
      expect(refused).not.toBeNull();
      const made = forks(fake);
      expect(made).toHaveLength(1);
      const deleted = fake
        .sent()
        .filter((m) => m.method === '_x.ai/session/delete')
        .map((m) => (m.params as Json).sessionId);
      expect(deleted).toHaveLength(1);
      expect(deleted[0]).not.toBe(id);
      expect(indexed(fake)).toEqual([id]);
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  // Mutation: drop `onForkLanded` in the service's listener → the late copy never gets its index line.
  it('Work: a fork past its soft deadline lands when its reply comes: the index gains the copy and the lists move', async () => {
    const gateDir = mkdtempSync(join(tmpdir(), 'mlx-fork-gate-'));
    const go = join(gateDir, 'go');
    const fake = fakeService('work-workflow-run', {
      timings: { forkMs: 300 },
      mutate: (scenario) => {
        scenario.prompts = [];
        scenario.fork = { holdOn: go };
      },
    });
    try {
      const { id, w } = await workConversation(fake);
      let changed = 0;
      const offHost = fake.service.watchHost((event) => {
        if (event.type === 'conversations-changed') changed += 1;
      });
      expect(await refusal(fake.service.fork(id))).toMatchObject({ reason: 'agent-slow' });
      expect(indexed(fake)).toEqual([id]);
      const before = changed;
      writeFileSync(go, '');
      await until(() => indexed(fake).length === 2, 8_000);
      const copy = indexed(fake).find((entry) => entry !== id)!;
      await until(() => changed > before);
      const c = watch(fake.service, copy);
      await until(() => c.current() !== null);
      expect(c.state().conversation).toMatchObject({ mode: 'work', forkOf: { id, title: expect.any(String) } });
      c.off();
      offHost();
      w.off();
    } finally {
      await fake.close();
      rmSync(gateDir, { recursive: true, force: true });
    }
  }, 30_000);

  // Mutation: drop the fork and recap checks of `opWall` → each refusal reads `conversation-busy`; drop the recap wait
  // from the queue's `free()` → the queued row is sent into the recap and cancels it.
  it('a fork or a recap that runs refuses a fork, a compact and a rewind with its own reason; a queued row waits for it', async () => {
    const gateDir = mkdtempSync(join(tmpdir(), 'mlx-op-gate-'));
    const forkGo = join(gateDir, 'fork');
    const recapGo = join(gateDir, 'recap');
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scenario.fork = { holdOn: forkGo };
        scenario.recap = { holdOn: recapGo, summary: 'Where we are.' };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      const at = userAt(w.state().entries, 'one');
      const opOf = async (): Promise<string | null | undefined> =>
        (await fake.service.state()).active.find((a) => a.conversationId === id)?.op;
      const walls = async (reason: string): Promise<void> => {
        for (const run of [
          fake.service.fork(id),
          fake.service.compact(id),
          fake.service.rewindPlan(id, at, tsAt(w, at)),
        ])
          expect(await refusal(run)).toMatchObject({ code: 'E_CONFLICT', reason });
      };

      const forking = fake.service.fork(id);
      await until(() => forks(fake).length === 1);
      expect(await opOf()).toBe('fork');
      await walls('fork-running');
      writeFileSync(forkGo, '');
      await forking;
      expect(await opOf()).toBeNull();

      expect(await fake.service.recap(id)).toEqual({});
      await until(() => w.current()?.recap != null);
      expect(await opOf()).toBe('recap');
      await walls('recap-running');
      expect(await fake.service.prompt(id, 'two', undefined, undefined, 'queue')).toHaveProperty('queued');
      // Send next while it is written: the row goes once it ends.
      expect(await fake.service.queueRelease(id)).toEqual({});
      await new Promise((resolve) => setTimeout(resolve, 200));
      expect(typed(w.state().entries)).toEqual(['one']);
      expect(await opOf()).toBe('recap');
      expect(forks(fake)).toHaveLength(1);
      writeFileSync(recapGo, '');
      await idle(w, 2);
      expect(typed(w.state().entries)).toEqual(['one', 'two']);
      expect(w.state().entries.filter((e) => e.kind === 'recap')).toEqual([
        { kind: 'recap', ts: expect.any(Number), text: 'Where we are.' },
      ]);
      w.off();
    } finally {
      await fake.close();
      rmSync(gateDir, { recursive: true, force: true });
    }
  }, 30_000);

  // Mutation: drop `originChanged` from the service's `onTitle` → the copy keeps the old name of its original.
  it('the agent naming the original renames it on the copies this window holds', async () => {
    const plain = [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }];
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scenario.prompts = [
          { match: 'quiet', noTitle: true, steps: plain },
          { match: '', steps: plain },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'quiet one' });
      const id = conversation.id;
      const w = watch(fake.service, id);
      await idle(w, 1);
      const copy = await fake.service.fork(id);
      const c = watch(fake.service, copy.id);
      await until(() => c.current() !== null);
      const before = c.state().conversation.forkOf;
      expect(before?.id).toBe(id);
      expect(before?.title).not.toBe('Plan the trip to Kyoto');
      await fake.service.prompt(id, 'Plan the trip to Kyoto');
      await idle(w, 2);
      await until(() => w.state().conversation.title === 'Plan the trip to Kyoto');
      await until(() => c.state().conversation.forkOf?.title === 'Plan the trip to Kyoto');
      expect(c.state().conversation.forkOf).toEqual({ id, title: 'Plan the trip to Kyoto' });
      c.off();
      w.off();
    } finally {
      await fake.close();
    }
  }, 30_000);
});
