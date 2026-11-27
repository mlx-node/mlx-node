/**
 * History ops on the scripted fake agent: the busy gates they share (`session.op`, `engineBusy`), the one load path
 * (`ensureLoaded`), and Compact now: its request, its wait line from the call to the reply, its end on the stream,
 * Stop with no turn, and the walls (a turn, a workflow run, another op).
 */

import { mkdtempSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { TranscriptReducer } from '../src/agent-sessions/transcript.js';
import { ACP_TIMEOUTS, AcpClient } from '../src/chat/acp/client.js';
import { AgentHostError } from '../src/chat/agent/agent-host.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ConversationEvent } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import {
  fakeHost,
  fakeService,
  noProviders,
  until,
  type FakeHost,
  type FakeHostOptions,
  type Json,
} from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const gates: string[] = [];
afterEach(() => {
  for (const dir of gates.splice(0)) rmSync(dir, { recursive: true, force: true });
});

/** A file path that does not exist until the test writes it: the fake agent holds the compaction until then. */
function gate(): string {
  const dir = mkdtempSync(join(tmpdir(), 'mlx-compact-gate-'));
  gates.push(dir);
  return join(dir, 'go');
}

function compactNow(options: Json): (scenario: Json) => void {
  return (scenario) => {
    scenario.compactNow = options;
  };
}

async function withHost(name: string, run: (fake: FakeHost) => Promise<void>, options: FakeHostOptions = {}) {
  const fake = await fakeHost(name, options);
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

function asked(fake: FakeHost, method: string): Json[] {
  return fake.sent().filter((m) => m.method === method);
}

function codeOf(run: () => unknown): string | null {
  try {
    run();
  } catch (error) {
    return error instanceof AgentHostError ? error.code : String(error);
  }
  return null;
}

describe('Compact now on the host', () => {
  it('one compact request with no deadline; busy from the call to the reply; one end; the mark comes live', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const run = fake.host.compact(id, { modelId: MODEL }, 41_000);
        expect(fake.events.compactions).toEqual([
          { sessionId: id, compaction: { phase: 'running', startedAt: expect.any(Number), tokensBefore: 41_000 } },
        ]);
        expect(fake.host.isBusy(id)).toBe(true);
        expect(fake.host.engineBusy(id)).toBe(true);
        expect(fake.host.isWorking(id)).toBe(true);
        expect(fake.host.modeState()).toMatchObject({ turnsInFlight: 0, background: 1 });
        expect(fake.host.active()).toEqual([
          {
            sessionId: id,
            busy: false,
            waiting: false,
            asks: 0,
            stopping: false,
            background: true,
            helpers: false,
            tasks: 0,
            op: 'compact',
          },
        ]);
        expect(fake.host.compaction(id)).toMatchObject({ phase: 'running', tokensBefore: 41_000 });
        expect(codeOf(() => fake.host.startTurn(id, 'hi', { modelId: MODEL }))).toBe('conversation-busy');
        expect(codeOf(() => fake.host.compact(id, { modelId: MODEL }, null))).toBe('conversation-busy');
        await expect(fake.host.setModel(id, 'qwen3.5-9b')).rejects.toMatchObject({ code: 'conversation-busy' });
        await expect(fake.host.deleteSession(id, '')).rejects.toMatchObject({ code: 'conversation-busy' });
        await until(() => asked(fake, '_x.ai/compact_conversation').length === 1);
        expect(asked(fake, '_x.ai/compact_conversation')[0]!.params).toEqual({ sessionId: id });
        // Held by the agent: no deadline ends it, and the line has not come.
        await sleep(300);
        expect(fake.host.isBusy(id)).toBe(true);
        expect(fake.events.compactions).toHaveLength(1);
        writeFileSync(go, '');
        expect(await run.done).toBe('compacted');
        expect(fake.events.compactions.slice(1)).toEqual([{ sessionId: id, compaction: null, end: 'compacted' }]);
        expect(fake.events.timeline.slice(-2)).toEqual(['auto_compact_completed', 'compaction end']);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.compaction(id)).toBeNull();
        expect(fake.host.modeState()).toMatchObject({ background: 0 });
        const reducer = new TranscriptReducer();
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        expect(reducer.snapshot().entries.filter((e) => e.kind === 'compaction')).toEqual([
          { kind: 'compaction', ts: expect.any(Number), outcome: 'done', tokensBefore: 41_000, tokensAfter: 9_000 },
        ]);
        expect(fake.events.turns).toEqual([]);
        expect(fake.events.ends).toEqual([]);
        // The conversation takes a prompt again.
        expect((await fake.host.startTurn(id, 'hi', { modelId: MODEL }).done).outcome).toBe('completed');
      },
      { mutate: compactNow({ holdOn: go }) },
    );
  }, 20_000);

  it('Stop sends `session/cancel` with no turn; the state shows stopping; the end is `stopped` and no mark', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const run = fake.host.compact(id, { modelId: MODEL }, null);
        await until(() => asked(fake, '_x.ai/compact_conversation').length === 1);
        expect(fake.host.cancel(id)).toBe(true);
        expect(fake.events.compactions.at(-1)).toEqual({
          sessionId: id,
          compaction: { phase: 'stopping', startedAt: expect.any(Number) },
        });
        expect(fake.host.cancel(id)).toBe(true);
        expect(await run.done).toBe('stopped');
        expect(asked(fake, 'session/cancel')).toEqual([
          { jsonrpc: '2.0', method: 'session/cancel', params: { sessionId: id } },
        ]);
        expect(fake.events.compactions.at(-1)).toEqual({ sessionId: id, compaction: null, end: 'stopped' });
        expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'auto_compact_completed')).toBe(false);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.cancel(id)).toBe(false);
        expect(fake.host.modeState().state).toBe('ready');
      },
      { mutate: compactNow({ holdOn: join(tmpdir(), 'never-written-compact-gate') }) },
    );
  }, 20_000);

  it('a failed run ends `failed`, a cancelled one `stopped`; an agent exit during the run ends it `failed`', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        expect(await fake.host.compact(id, { modelId: MODEL }, null).done).toBe('failed');
        expect(fake.events.compactions.at(-1)).toEqual({ sessionId: id, compaction: null, end: 'failed' });
        expect(fake.host.isBusy(id)).toBe(false);
      },
      { mutate: compactNow({ fail: 'compact failed: the model said no' }) },
    );
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        // `data.kind` alone tells a cancel from a failure: no Stop was pressed here.
        expect(await fake.host.compact(id, { modelId: MODEL }, null).done).toBe('stopped');
        expect(asked(fake, 'session/cancel')).toEqual([]);
        expect(fake.events.compactions.at(-1)).toEqual({ sessionId: id, compaction: null, end: 'stopped' });
      },
      { mutate: compactNow({ cancelled: true }) },
    );
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const run = fake.host.compact(id, { modelId: MODEL }, null);
        await until(() => asked(fake, '_x.ai/compact_conversation').length === 1);
        const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
        process.kill(pid, 'SIGKILL');
        expect(await run.done).toBe('failed');
        expect(fake.events.compactions.filter((c) => c.compaction === null)).toEqual([
          { sessionId: id, compaction: null, end: 'failed' },
        ]);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.compaction(id)).toBeNull();
      },
      { mutate: compactNow({ holdOn: join(tmpdir(), 'never-written-compact-gate') }) },
    );
  }, 20_000);

  it('a cold conversation loads first (`session/load`, then the request); a Stop during the load sends nothing', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.restart();
        expect(fake.host.isHot(id)).toBe(false);
        expect(await fake.host.compact(id, { modelId: MODEL }, null).done).toBe('compacted');
        const methods = fake
          .sent()
          .map((m) => m.method)
          .filter((m) => m === 'session/load' || m === '_x.ai/compact_conversation');
        expect(methods).toEqual(['session/load', '_x.ai/compact_conversation']);
        expect(fake.host.isHot(id)).toBe(true);

        await fake.host.restart();
        const run = fake.host.compact(id, { modelId: MODEL }, null);
        await until(() => asked(fake, 'session/load').length === 2);
        expect(fake.host.cancel(id)).toBe(true);
        expect(await run.done).toBe('stopped');
        expect(asked(fake, '_x.ai/compact_conversation')).toHaveLength(1);
        expect(asked(fake, 'session/cancel')).toEqual([]);
        expect(fake.events.compactions.at(-1)).toEqual({ sessionId: id, compaction: null, end: 'stopped' });
      },
      { mutate: (scenario) => (scenario.faults = { delayMs: { 'session/load': 400 } }) },
    );
  }, 20_000);

  it('a turn or another op refuses it at once; while it runs the idle stop waits', async () => {
    const go = gate();
    await withHost(
      'slow-stream',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const turn = fake.host.startTurn(id, 'go', { modelId: MODEL });
        expect(codeOf(() => fake.host.compact(id, { modelId: MODEL }, null))).toBe('conversation-busy');
        expect(fake.events.compactions).toEqual([]);
        expect((await turn.done).outcome).toBe('completed');

        const run = fake.host.compact(id, { modelId: MODEL }, null);
        await sleep(500);
        expect(fake.host.modeState().state).toBe('ready');
        writeFileSync(go, '');
        expect(await run.done).toBe('compacted');
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { timings: { idleStopMs: 150 }, mutate: compactNow({ holdOn: go }) },
    );
  }, 20_000);

  it('a live workflow run refuses it (engine work); its own end frees it', async () => {
    const complete = gate();
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const turn = fake.host.startTurn(id, 'p1', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        fake.host.answer(id, fake.events.interactions[0]!.interaction.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.host.runs(id)).toHaveLength(1);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.engineBusy(id)).toBe(true);
        expect(codeOf(() => fake.host.compact(id, { modelId: MODEL, cwd: fake.project }, null))).toBe(
          'conversation-busy',
        );
        writeFileSync(complete, '');
        await until(() => fake.events.ends.length === 2, 5_000);
        expect(fake.host.engineBusy(id)).toBe(false);
        expect(await fake.host.compact(id, { modelId: MODEL, cwd: fake.project }, null).done).toBe('compacted');
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: (scenario) => {
          const prompts = scenario.prompts as Json[];
          const steps = prompts[0]!.steps as Json[];
          steps[1] = { workflow: { ...(steps[1]!.workflow as Json), ask: false, completeOn: complete } };
        },
      },
    );
  }, 20_000);
});

describe('Compact now through the service', () => {
  it('answers at once; the page sees the wait line with the context it had, Stop, and the end on the stream', async () => {
    const go = gate();
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        compactNow({ holdOn: go, tokensBefore: 4_321 })(scenario);
        scenario.prompts = [
          {
            match: '',
            steps: [
              { text: 'Hello! How can I help?', chunk: 7, totalTokens: 4_321 },
              { turnCompleted: { inputTokens: 400, outputTokens: 30 } },
              { end: { stopReason: 'end_turn' } },
            ],
          },
        ];
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
      const id = conversation.id;
      const events: ConversationEvent[] = [];
      let state: ConversationState | null = null;
      const detach = fake.service.watchConversation(id, (event) => {
        events.push(event);
        const next = applyConversationEvent(state, event);
        if (next !== 'resync') state = next;
      });
      await until(() => state !== null && state.turns.length === 1, 5_000);
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      expect(await fake.service.compact(id)).toEqual({});
      expect(fake.service.busyConversations()).toEqual([id]);
      await until(() => state!.compaction !== null);
      expect(state!.contextUsed).toBe(4_321);
      expect(state!.compaction).toEqual({ phase: 'running', startedAt: expect.any(Number), tokensBefore: 4_321 });
      const busy = await fake.service.prompt(id, 'again').catch((error: unknown) => routeError('chat', error));
      expect(busy).toMatchObject({ code: 'E_CONFLICT', reason: 'conversation-busy' });
      expect(routeError('chat', await fake.service.compact(id).catch((e: unknown) => e))).toMatchObject({
        reason: 'conversation-busy',
      });
      writeFileSync(go, '');
      await until(() => events.some((e) => e.type === 'compaction' && e.end !== undefined), 5_000);
      expect(
        events.filter((e) => e.type === 'compaction').map((e) => (e.type === 'compaction' ? e.end : null)),
      ).toEqual([undefined, 'compacted']);
      expect(state!.compaction).toBeNull();
      expect(state!.entries.filter((e) => e.kind === 'compaction')).toEqual([
        { kind: 'compaction', ts: expect.any(Number), outcome: 'done', tokensBefore: 4_321, tokensAfter: 9_000 },
      ]);
      expect(fake.service.busyConversations()).toEqual([]);
      detach();
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('refuses with inference-stopped while the engine is stopped: nothing starts and nothing goes to the agent', async () => {
    const fake = fakeService('chat-basic');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
      const id = conversation.id;
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      const events: ConversationEvent[] = [];
      const detach = fake.service.watchConversation(id, (event) => events.push(event));
      const endpoint = fake.endpoint.current;
      fake.endpoint.current = null;
      const stopped = routeError('chat', await fake.service.compact(id).catch((e: unknown) => e));
      expect([stopped.code, stopped.reason]).toEqual(['E_UNAVAILABLE', 'inference-stopped']);
      expect(events.filter((e) => e.type === 'compaction')).toEqual([]);
      expect(fake.service.busyConversations()).toEqual([]);
      fake.endpoint.current = endpoint;
      expect(await fake.service.compact(id)).toEqual({});
      await until(() => events.some((e) => e.type === 'compaction' && e.end !== undefined), 5_000);
      expect(events.filter((e) => e.type === 'compaction').at(-1)).toMatchObject({
        compaction: null,
        end: 'compacted',
      });
      detach();
    } finally {
      await fake.close();
    }
  }, 20_000);
});

describe('Rewind on the host', () => {
  const rewound = (fake: FakeHost): Json[] => fake.records().filter((r) => r.kind === 'rewind');
  const pids = (fake: FakeHost): unknown[] =>
    fake
      .records()
      .filter((r) => r.kind === 'start')
      .map((r) => r.pid);
  const NOTHING = { tasks: [], schedules: [], runs: [], files: [] };

  it('mode and force always go; busy from the call to the reply; the agent keeps prompts before the target', async () => {
    await withHost('chat-basic', async (fake) => {
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
      for (const text of ['one', 'two']) {
        expect((await fake.host.startTurn(id, text, { modelId: MODEL }).done).outcome).toBe('completed');
      }
      const run = fake.host.rewind(id, { modelId: MODEL }, { target: 1, mode: 'conversation', ...NOTHING });
      expect(fake.host.isBusy(id)).toBe(true);
      expect(fake.host.engineBusy(id)).toBe(true);
      expect(codeOf(() => fake.host.startTurn(id, 'hi', { modelId: MODEL }))).toBe('conversation-busy');
      expect(codeOf(() => fake.host.compact(id, { modelId: MODEL }, null))).toBe('conversation-busy');
      expect(
        codeOf(() => fake.host.rewind(id, { modelId: MODEL }, { target: 0, mode: 'conversation', ...NOTHING })),
      ).toBe('conversation-busy');
      expect(fake.host.cancel(id)).toBe(false);
      expect(await run.done).toEqual({ outcome: 'done', restored: 0 });
      expect(asked(fake, '_x.ai/rewind/execute').map((m) => m.params)).toEqual([
        { sessionId: id, targetPromptIndex: 1, force: true, mode: 'conversation_only' },
      ]);
      expect(rewound(fake)).toEqual([
        { kind: 'rewind', target: 1, force: true, mode: 'conversation_only', hasMode: true, hasForce: true },
      ]);
      expect(fake.host.isBusy(id)).toBe(false);
      expect(fake.host.modeState()).toMatchObject({ background: 0 });
      // The agent's prompt index is the target now: a rewind to 1 again is past it.
      expect(
        await fake.host.rewind(id, { modelId: MODEL }, { target: 1, mode: 'conversation', ...NOTHING }).done,
      ).toEqual({ outcome: 'refused' });
    });
  }, 20_000);

  // Mutation: give the execute call a hard deadline → the held commit kills the agent (two starts).
  it('a held commit is never killed: the op stays until the late reply; the agent exit ends it `stopped`', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const run = fake.host.rewind(id, { modelId: MODEL }, { target: 0, mode: 'conversation', ...NOTHING });
        await until(() => asked(fake, '_x.ai/rewind/execute').length === 1);
        await sleep(1_200);
        expect(fake.host.isBusy(id)).toBe(true);
        expect(fake.host.modeState().state).toBe('ready');
        expect(pids(fake)).toHaveLength(1);
        writeFileSync(go, '');
        expect(await run.done).toEqual({ outcome: 'done', restored: 0 });
        expect(fake.host.isBusy(id)).toBe(false);
      },
      { mutate: (scenario) => void (scenario.rewind = { holdOn: go }) },
    );
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const held = fake.host.rewind(id, { modelId: MODEL }, { target: 0, mode: 'conversation', ...NOTHING });
        await until(() => asked(fake, '_x.ai/rewind/execute').length === 1);
        await fake.host.restart();
        expect(await held.done).toEqual({ outcome: 'stopped' });
        expect(fake.host.isBusy(id)).toBe(false);
      },
      { mutate: (scenario) => void (scenario.rewind = { holdOn: join(tmpdir(), 'never-written-rewind-gate') }) },
    );
  }, 20_000);

  it('the agent saying no ends it `refused`; an error reply too', async () => {
    for (const rewind of [{ refuse: true }, { rpcError: true }]) {
      await withHost(
        'chat-basic',
        async (fake) => {
          const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
          await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
          expect(
            await fake.host.rewind(id, { modelId: MODEL }, { target: 0, mode: 'conversation', ...NOTHING }).done,
          ).toEqual({ outcome: 'refused' });
          expect(fake.host.isBusy(id)).toBe(false);
        },
        { mutate: (scenario) => void (scenario.rewind = rewind) },
      );
    }
  }, 20_000);

  it('a turn refuses it at once; a cold conversation loads first', async () => {
    await withHost('chat-basic', async (fake) => {
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
      await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
      const turn = fake.host.startTurn(id, 'two', { modelId: MODEL });
      expect(
        codeOf(() => fake.host.rewind(id, { modelId: MODEL }, { target: 0, mode: 'conversation', ...NOTHING })),
      ).toBe('conversation-busy');
      await turn.done;
      await fake.host.restart();
      expect(fake.host.isHot(id)).toBe(false);
      const run = fake.host.rewind(id, { modelId: MODEL }, { target: 1, mode: 'conversation', ...NOTHING });
      expect(await run.done).toEqual({ outcome: 'done', restored: 0 });
      const methods = fake.sent().map((m) => m.method);
      expect(methods.lastIndexOf('session/load')).toBeLessThan(methods.lastIndexOf('_x.ai/rewind/execute'));
      expect(methods.lastIndexOf('session/load')).toBeGreaterThan(-1);
    });
  }, 20_000);
});

describe('Rewind dry run on the host (#7 files)', () => {
  const EDIT = {
    toolName: 'search_replace',
    file_path: 'src/a.ts',
    before: 'a1\n',
    old_string: 'a1',
    new_string: 'a2',
  };
  const options = (rewind: Json = {}): FakeHostOptions => ({
    mode: 'work',
    providers: noProviders(),
    mutate: (scenario) => {
      scenario.prompts = [
        {
          match: 'edit',
          steps: [{ edit: { ...EDIT, ask: false } }, { text: 'Edited.' }, { end: { stopReason: 'end_turn' } }],
        },
      ];
      scenario.rewind = rewind;
    },
  });

  // Mutation: set `session.op` for the dry run → the conversation reads busy while it runs.
  it('force false and mode all always go; it sets no op; the files come back; a target past the prompts is refused', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const load = { modelId: MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        for (const text of ['first', 'edit']) {
          expect((await fake.host.startTurn(id, text, load).done).outcome).toBe('completed');
        }
        const run = fake.host.rewindPreview(id, load, 1);
        await until(() => asked(fake, '_x.ai/rewind/execute').length === 1);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.opOf(id)).toBeNull();
        expect(await run).toEqual({ files: [{ path: 'src/a.ts', state: 'modified' }] });
        expect(asked(fake, '_x.ai/rewind/execute').map((m) => m.params)).toEqual([
          { sessionId: id, targetPromptIndex: 1, force: false, mode: 'all' },
        ]);
        expect(await fake.host.rewindPreview(id, load, 2)).toEqual({ refused: true });
        expect(await fake.host.rewindPreview(id, load, 0)).toEqual({
          files: [{ path: 'src/a.ts', state: 'modified' }],
        });
        // A dry run changes nothing: the commit still finds the file, the one the plan showed.
        const files = [{ path: 'src/a.ts', state: 'modified' as const }];
        const done = await fake.host.rewind(id, load, {
          target: 1,
          mode: 'files',
          tasks: [],
          schedules: [],
          runs: [],
          files,
        }).done;
        expect(done).toEqual({ outcome: 'done', restored: 1 });
      },
      options({ previewMs: 300, changed: { 'src/a.ts': 'modified_externally' } }),
    );
  }, 20_000);

  // Mutation: read every conflict as `modified` → the deleted and made-again files lose their state.
  it('each conflict type of the agent keeps its state; one this app does not know reads as changed', async () => {
    const paths = ['src/a.ts', 'src/b.ts', 'src/c.ts', 'src/d.ts'];
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const load = { modelId: MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        expect((await fake.host.startTurn(id, 'edit', load).done).outcome).toBe('completed');
        expect(await fake.host.rewindPreview(id, load, 0)).toEqual({
          files: [
            { path: 'src/a.ts', state: 'modified' },
            { path: 'src/b.ts', state: 'deleted' },
            { path: 'src/c.ts', state: 'created' },
            { path: 'src/d.ts', state: 'modified' },
          ],
        });
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: (scenario) => {
          scenario.prompts = [
            {
              match: 'edit',
              steps: [
                ...paths.map((file_path) => ({ edit: { ...EDIT, file_path, ask: false } })),
                { text: 'Edited.' },
                { end: { stopReason: 'end_turn' } },
              ],
            },
          ];
          scenario.rewind = {
            changed: {
              'src/a.ts': 'modified_externally',
              'src/b.ts': 'deleted_externally',
              'src/c.ts': 'created_externally',
              'src/d.ts': 'content_mismatch',
            },
          };
        },
      },
    );
  }, 20_000);

  // Mutation: let a dry run through while a turn runs → no refusal.
  it('a turn refuses it at once; a cold conversation loads first; a missed deadline is agent-slow and the agent lives', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const load = { modelId: MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        await fake.host.startTurn(id, 'first', load).done;
        const turn = fake.host.startTurn(id, 'edit', load);
        await expect(fake.host.rewindPreview(id, load, 0)).rejects.toMatchObject({ code: 'conversation-busy' });
        await turn.done;
        await fake.host.restart();
        expect(fake.host.isHot(id)).toBe(false);
        await expect(fake.host.rewindPreview(id, load, 1)).rejects.toMatchObject({ code: 'agent-slow' });
        const methods = fake.sent().map((m) => m.method);
        expect(methods.lastIndexOf('session/load')).toBeLessThan(methods.lastIndexOf('_x.ai/rewind/execute'));
        expect(methods.lastIndexOf('session/load')).toBeGreaterThan(-1);
        await sleep(600);
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.host.isHot(id)).toBe(true);
      },
      { ...options({ previewMs: 500 }), timings: { rewindPreviewMs: 100 } },
    );
  }, 20_000);
});

describe('Fork on the host', () => {
  const forked = (fake: FakeHost): Json[] => asked(fake, '_x.ai/session/fork').map((m) => m.params as Json);
  const pids = (fake: FakeHost): unknown[] =>
    fake
      .records()
      .filter((r) => r.kind === 'start')
      .map((r) => r.pid);

  it('both folders the same, no model, the target only when given; busy while it runs; the copy loads and answers', async () => {
    await withHost('chat-basic', async (fake) => {
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
      for (const text of ['one', 'two']) {
        expect((await fake.host.startTurn(id, text, { modelId: MODEL }).done).outcome).toBe('completed');
      }
      const whole = fake.host.fork(id, { modelId: MODEL }, null);
      expect(fake.host.isBusy(id)).toBe(true);
      expect(fake.host.engineBusy(id)).toBe(true);
      expect(codeOf(() => fake.host.startTurn(id, 'hi', { modelId: MODEL }))).toBe('conversation-busy');
      expect(codeOf(() => fake.host.compact(id, { modelId: MODEL }, null))).toBe('conversation-busy');
      expect(fake.host.cancel(id)).toBe(false);
      const copy = await whole;
      expect(copy).not.toBe(id);
      expect(fake.host.isBusy(id)).toBe(false);
      expect(fake.host.modeState()).toMatchObject({ background: 0 });
      const first = forked(fake)[0]!;
      expect(Object.keys(first).sort()).toEqual(['newCwd', 'sourceCwd', 'sourceSessionId']);
      expect(first.sourceSessionId).toBe(id);
      expect(typeof first.sourceCwd).toBe('string');
      expect(first.newCwd).toBe(first.sourceCwd);

      const partial = await fake.host.fork(id, { modelId: MODEL }, 0);
      expect(forked(fake)[1]).toEqual({ ...first, targetPromptIndex: 0 });
      // The copy is not loaded by the fork; its first prompt loads it, and the agent knows its two prompts.
      const methods = fake.sent().map((m) => m.method);
      expect(methods.slice(methods.lastIndexOf('_x.ai/session/fork'))).not.toContain('session/load');
      expect((await fake.host.startTurn(copy, 'three', { modelId: MODEL }).done).outcome).toBe('completed');
      const load = fake.sent().find((m) => m.method === 'session/load' && (m.params as Json).sessionId === copy);
      expect(load).toBeDefined();
      expect((await fake.host.startTurn(partial, 'again', { modelId: MODEL }).done).outcome).toBe('completed');
    });
  }, 20_000);

  it('starts a stopped agent and loads nothing; an error reply is a protocol error', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        await fake.host.restart();
        expect(fake.host.modeState().state).not.toBe('ready');
        const sentBefore = fake.sent().length;
        await expect(fake.host.fork(id, { modelId: MODEL }, null)).rejects.toMatchObject({
          code: 'agent-protocol-error',
        });
        expect(pids(fake)).toHaveLength(2);
        const after = fake
          .sent()
          .slice(sentBefore)
          .map((m) => m.method);
        expect(after).toContain('_x.ai/session/fork');
        expect(after).not.toContain('session/load');
        expect(fake.host.isBusy(id)).toBe(false);
      },
      { mutate: (scenario) => void (scenario.fork = { rpcError: true }) },
    );
  }, 20_000);

  // Mutation: send the fork with `call` (a hard `ACP_TIMEOUTS` deadline) → the miss kills the agent (two starts).
  // Mutation: drop the `late` hand-off → no copy is told and the late copy never lands.
  it('a missed reply is agent-slow: the agent stays, the op ends, and the late reply names its copy', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        await expect(fake.host.fork(id, { modelId: MODEL }, null)).rejects.toMatchObject({ code: 'agent-slow' });
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.events.forks).toEqual([]);
        writeFileSync(go, '');
        await until(() => fake.events.forks.length === 1);
        const copy = fake.events.forks[0]!.copyId;
        expect(fake.events.forks).toEqual([{ sessionId: id, copyId: copy }]);
        expect(copy).not.toBe(id);
        expect(pids(fake)).toHaveLength(1);
        expect(fake.host.modeState().state).toBe('ready');
        expect((await fake.host.startTurn(id, 'two', { modelId: MODEL }).done).outcome).toBe('completed');
      },
      { timings: { forkMs: 300 }, mutate: (scenario) => void (scenario.fork = { holdOn: go }) },
    );
  }, 20_000);

  it('a turn or another op refuses it; nothing is sent', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const turn = fake.host.startTurn(id, 'two', { modelId: MODEL });
        await expect(fake.host.fork(id, { modelId: MODEL }, null)).rejects.toMatchObject({
          code: 'conversation-busy',
        });
        await turn.done;
        const run = fake.host.compact(id, { modelId: MODEL }, null);
        await expect(fake.host.fork(id, { modelId: MODEL }, null)).rejects.toMatchObject({
          code: 'conversation-busy',
        });
        writeFileSync(go, '');
        expect(await run.done).toBe('compacted');
        expect(forked(fake)).toEqual([]);
      },
      { mutate: compactNow({ holdOn: go }) },
    );
  }, 20_000);
});

describe('Rewind on the host: the removed part’s background work (C13)', () => {
  const WORK_MODEL = MODEL;

  function steps(words: Record<string, Json>): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = Object.entries(words).map(([match, step]) => ({
        match,
        steps: [step, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }],
      }));
    };
  }

  async function work(fake: FakeHost): Promise<string> {
    const { sessionId: id } = await fake.host.createSession({ modelId: WORK_MODEL, cwd: fake.project });
    for (const text of ['first', 'cmd', 'sched']) {
      expect((await fake.host.startTurn(id, text, { modelId: WORK_MODEL, cwd: fake.project }).done).outcome).toBe(
        'completed',
      );
    }
    return id;
  }

  // Mutation: send the kills with `clientUi` → the teardown source check fails.
  it('stops the commands (teardown) and deletes the schedules first, then sends the rewind', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.liveActivity(id).tasks.length === 1);
        const run = fake.host.rewind(
          id,
          { modelId: WORK_MODEL, cwd: fake.project },
          { target: 1, mode: 'conversation', tasks: ['t1'], schedules: ['s1'], runs: [], files: [] },
        );
        expect(await run.done).toEqual({ outcome: 'done', restored: 0 });
        const order = fake
          .sent()
          .map((m) => m.method)
          .filter((m) => m === '_x.ai/task/kill' || m === '_x.ai/scheduler/delete' || m === '_x.ai/rewind/execute');
        expect(order.at(-1)).toBe('_x.ai/rewind/execute');
        expect(order.slice(0, 2).sort()).toEqual(['_x.ai/scheduler/delete', '_x.ai/task/kill']);
        expect(asked(fake, '_x.ai/task/kill').map((m) => m.params)).toEqual([
          { sessionId: id, taskId: 't1', source: 'teardown' },
        ]);
        expect(asked(fake, '_x.ai/scheduler/delete').map((m) => m.params)).toEqual([{ sessionId: id, taskId: 's1' }]);
        await until(() => fake.host.liveActivity(id).tasks.length === 0);
        // A teardown kill wakes nothing.
        await sleep(300);
        expect(fake.events.turns.filter((t) => t.turn?.origin === 'wake')).toEqual([]);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: steps({ cmd: { background: { taskId: 't1', wake: false } }, sched: { schedule: { taskId: 's1' } } }),
      },
    );
  }, 20_000);

  // Mutation: stop only the commands and schedules → the paused run stays, and the rewound conversation can resume it.
  it('stops the paused runs the removed part launched, by you, before the rewind; a refused stop refuses it (T39)', async () => {
    const paused = { workflow: { runId: 'wf_1', ask: false, completeAfterMs: 0, status: 'user_paused', wake: false } };
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.runs(id)[0]?.status === 'user_paused');
        const run = fake.host.rewind(
          id,
          { modelId: WORK_MODEL, cwd: fake.project },
          { target: 1, mode: 'conversation', tasks: [], schedules: [], runs: ['wf_1'], files: [] },
        );
        expect(await run.done).toEqual({ outcome: 'done', restored: 0 });
        const order = fake
          .sent()
          .map((m) => m.method)
          .filter((m) => m === '_x.ai/workflows/stop' || m === '_x.ai/rewind/execute');
        expect(order).toEqual(['_x.ai/workflows/stop', '_x.ai/rewind/execute']);
        expect(asked(fake, '_x.ai/workflows/stop').map((m) => m.params)).toEqual([{ sessionId: id, runId: 'wf_1' }]);
        await until(() => fake.host.runs(id)[0]?.status === 'cancelled');
        expect(fake.host.runs(id)[0]).toMatchObject({ live: false, by: 'you' });
      },
      { mode: 'work', providers: noProviders(), mutate: steps({ cmd: paused }) },
    );
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.runs(id)[0]?.status === 'user_paused');
        const run = fake.host.rewind(
          id,
          { modelId: WORK_MODEL, cwd: fake.project },
          { target: 1, mode: 'conversation', tasks: [], schedules: [], runs: ['wf_1'], files: [] },
        );
        expect(await run.done).toEqual({ outcome: 'tasks' });
        expect(asked(fake, '_x.ai/rewind/execute')).toEqual([]);
        expect(fake.host.runs(id)[0]).toMatchObject({ status: 'user_paused', stopping: false });
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: steps({ cmd: { workflow: { ...paused.workflow, stopFails: true } } }),
      },
    );
  }, 30_000);

  // Mutation: drop the dry run after the stops → force true goes and the rewind ends `done`.
  it('a file that changes while the removed part’s work stops ends it `changed` after the stops', async () => {
    const hold = gate();
    const changedFile = gate();
    const edit = {
      toolName: 'search_replace',
      file_path: 'src/a.ts',
      before: 'a1\n',
      old_string: 'a1',
      new_string: 'a2',
    };
    const paused = { runId: 'wf_1', ask: false, completeAfterMs: 0, status: 'user_paused', wake: false, stopOn: hold };
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const load = { modelId: WORK_MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        for (const text of ['first', 'edit', 'cmd']) {
          expect((await fake.host.startTurn(id, text, load).done).outcome).toBe('completed');
        }
        await until(() => fake.host.runs(id)[0]?.status === 'user_paused');
        const run = fake.host.rewind(id, load, {
          target: 1,
          mode: 'all',
          tasks: [],
          schedules: [],
          runs: ['wf_1'],
          files: [{ path: 'src/a.ts', state: 'clean' }],
        });
        await until(() => fake.records().some((r) => r.kind === 'stop'));
        writeFileSync(changedFile, JSON.stringify({ 'src/a.ts': 'modified_externally' }));
        writeFileSync(hold, '');
        expect(await run.done).toEqual({ outcome: 'changed' });
        const rewinds = fake.records().filter((r) => r.kind === 'rewind');
        expect(rewinds.map((r) => r.force)).toEqual([false, false]);
        expect(asked(fake, '_x.ai/workflows/stop')).toHaveLength(1);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: (scenario) => {
          scenario.prompts = [
            {
              match: 'edit',
              steps: [{ edit: { ...edit, ask: false } }, { text: 'Edited.' }, { end: { stopReason: 'end_turn' } }],
            },
            {
              match: 'cmd',
              steps: [{ workflow: paused }, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }],
            },
          ];
          scenario.rewind = { changedFile };
        },
      },
    );
  }, 20_000);

  it('a stop that fails refuses the rewind: nothing is rewound; files only stops nothing', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        const run = fake.host.rewind(
          id,
          { modelId: WORK_MODEL, cwd: fake.project },
          { target: 1, mode: 'conversation', tasks: ['t1'], schedules: [], runs: [], files: [] },
        );
        expect(await run.done).toEqual({ outcome: 'tasks' });
        expect(asked(fake, '_x.ai/rewind/execute')).toEqual([]);
        expect(fake.host.isBusy(id)).toBe(false);
        const files = fake.host.rewind(
          id,
          { modelId: WORK_MODEL, cwd: fake.project },
          { target: 1, mode: 'files', tasks: ['t1'], schedules: ['s1'], runs: [], files: [] },
        );
        expect(await files.done).toEqual({ outcome: 'done', restored: 0 });
        expect(asked(fake, '_x.ai/task/kill')).toHaveLength(1);
        expect(asked(fake, '_x.ai/scheduler/delete')).toEqual([]);
        expect(
          asked(fake, '_x.ai/rewind/execute').map((m) => [(m.params as Json).mode, (m.params as Json).force]),
        ).toEqual([
          ['files_only', false],
          ['files_only', true],
        ]);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: (scenario) => {
          steps({ cmd: { background: { taskId: 't1', wake: false } }, sched: { schedule: { taskId: 's1' } } })(
            scenario,
          );
          scenario.faults = { killReply: 'error' };
        },
      },
    );
  }, 20_000);

  // Mutation: drop the `lostTasks()` wall in `rewind()` → the load restarts the agent and stops the command.
  it('the engine restarted under a live command: rewind is refused, nothing is sent and the agent stays', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.liveActivity(id).tasks.length === 1);
        const engine = fake.endpoint.current!;
        fake.endpoint.current = { ...engine, pid: engine.pid + 1 };
        await fake.host.checkEndpoint();
        await until(() => fake.host.modeState().engineLost === true);
        expect(
          codeOf(() =>
            fake.host.rewind(
              id,
              { modelId: WORK_MODEL, cwd: fake.project },
              { target: 1, mode: 'conversation', tasks: ['t1'], schedules: [], runs: [], files: [] },
            ),
          ),
        ).toBe('engine-lost');
        await sleep(200);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(asked(fake, '_x.ai/task/kill')).toEqual([]);
        expect(asked(fake, '_x.ai/rewind/execute')).toEqual([]);
        expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(1);
        expect(fake.host.liveActivity(id).tasks).toHaveLength(1);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: steps({ cmd: { background: { taskId: 't1', wake: false } }, sched: { schedule: { taskId: 's1' } } }),
      },
    );
  }, 20_000);

  // Mutation: drop the `lostTasks()` wall in `fork()` → the stale agent is restarted and the command stops.
  it('the engine restarted under a live command: fork is refused, nothing is sent and the agent stays', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.liveActivity(id).tasks.length === 1);
        const engine = fake.endpoint.current!;
        fake.endpoint.current = { ...engine, pid: engine.pid + 1 };
        await fake.host.checkEndpoint();
        await until(() => fake.host.modeState().engineLost === true);
        await expect(fake.host.fork(id, { modelId: WORK_MODEL, cwd: fake.project }, null)).rejects.toMatchObject({
          code: 'engine-lost',
        });
        await sleep(200);
        expect(fake.host.isBusy(id)).toBe(false);
        expect(asked(fake, '_x.ai/session/fork')).toEqual([]);
        expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(1);
        expect(fake.host.liveActivity(id).tasks).toHaveLength(1);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: steps({ cmd: { background: { taskId: 't1', wake: false } }, sched: { schedule: { taskId: 's1' } } }),
      },
    );
  }, 20_000);

  // Mutation: drop the `lostTasks()` wall in `recap()` → the load restarts the stale agent and the command stops.
  it('the engine restarted under a live command: recap is refused, nothing is sent and the agent stays', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.liveActivity(id).tasks.length === 1);
        // A live command alone does not turn a recap off.
        expect(await fake.host.recap(id, { modelId: WORK_MODEL, cwd: fake.project }).done).toBe('shown');
        const engine = fake.endpoint.current!;
        fake.endpoint.current = { ...engine, pid: engine.pid + 1 };
        await fake.host.checkEndpoint();
        await until(() => fake.host.modeState().engineLost === true);
        expect(codeOf(() => fake.host.recap(id, { modelId: WORK_MODEL, cwd: fake.project }))).toBe('engine-lost');
        await sleep(200);
        expect(asked(fake, '_x.ai/recap')).toHaveLength(1);
        expect(fake.host.recapState(id)).toBeNull();
        expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(1);
        expect(fake.host.liveActivity(id).tasks).toHaveLength(1);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: steps({ cmd: { background: { taskId: 't1', wake: false } }, sched: { schedule: { taskId: 's1' } } }),
      },
    );
  }, 20_000);

  it('Work: a fork names the conversation’s folder as both folders; a live command does not refuse it', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const id = await work(fake);
        await until(() => fake.host.liveActivity(id).tasks.length === 1);
        const copy = await fake.host.fork(id, { modelId: WORK_MODEL, cwd: fake.project }, 1);
        const params = asked(fake, '_x.ai/session/fork')[0]!.params as Json;
        expect(params).toEqual({
          sourceSessionId: id,
          sourceCwd: realpathSync(fake.project),
          newCwd: realpathSync(fake.project),
          targetPromptIndex: 1,
        });
        expect(fake.host.liveActivity(id).tasks).toHaveLength(1);
        const disk = readdirSync(join(fake.agentHome, 'sessions'), { recursive: true, encoding: 'utf8' });
        expect(disk.some((f) => f.endsWith(join(copy, 'summary.json')))).toBe(true);
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: steps({ cmd: { background: { taskId: 't1', wake: false } }, sched: { schedule: { taskId: 's1' } } }),
      },
    );
  }, 20_000);
});

describe('Recap on the host', () => {
  const recapsAsked = (fake: FakeHost): Json[] => asked(fake, '_x.ai/recap').map((m) => m.params as Json);
  const starts = (fake: FakeHost): number => fake.records().filter((r) => r.kind === 'start').length;
  const recapLines = (fake: FakeHost): string[] =>
    fake.events.updates
      .map((u) => String(u.line.update.sessionUpdate))
      .filter((kind) => kind === 'session_recap' || kind === 'session_recap_unavailable');
  function recap(options: Json): (scenario: Json) => void {
    return (scenario) => {
      scenario.recap = options;
    };
  }

  // Mutation: send `auto: true` → the request check fails; drop the tee of `session_recap` → the wait ends `none`.
  it('`auto: false` goes; the wait shows from the call; its line ends it `shown` and goes on to the reducer', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const run = fake.host.recap(id, { modelId: MODEL });
        expect(fake.events.recaps).toEqual([{ sessionId: id, recap: { startedAt: expect.any(Number) } }]);
        expect(fake.host.recapState(id)).toEqual({ startedAt: expect.any(Number) });
        // A recap only reads: the conversation is not busy, and a second one waits for the first.
        expect(fake.host.isBusy(id)).toBe(false);
        expect(codeOf(() => fake.host.recap(id, { modelId: MODEL }))).toBe('conversation-busy');
        await until(() => recapsAsked(fake).length === 1);
        expect(recapsAsked(fake)).toEqual([{ sessionId: id, auto: false }]);
        writeFileSync(go, '');
        expect(await run.done).toBe('shown');
        expect(fake.events.recaps.at(-1)).toEqual({ sessionId: id, recap: null, end: 'shown' });
        expect(fake.host.recapState(id)).toBeNull();
        expect(recapLines(fake)).toEqual(['session_recap']);
        const reducer = new TranscriptReducer();
        for (const u of fake.events.updates.filter((u) => u.sessionId === id)) reducer.apply(u.line);
        expect(reducer.snapshot().entries.filter((e) => e.kind === 'recap')).toEqual([
          { kind: 'recap', ts: expect.any(Number), text: 'Where we are.' },
        ]);
      },
      { mutate: recap({ holdOn: go, summary: 'Where we are.' }) },
    );
  }, 20_000);

  it('recaps turned off in the agent: `off`; nothing more comes', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        expect(await fake.host.recap(id, { modelId: MODEL }).done).toBe('off');
        expect(fake.events.recaps.at(-1)).toEqual({ sessionId: id, recap: null, end: 'off' });
        expect(recapLines(fake)).toEqual([]);
      },
      { mutate: recap({ disabled: true }) },
    );
  }, 20_000);

  it('no finished answer: the agent writes none (`session_recap_unavailable`), the wait ends `none`', async () => {
    await withHost('chat-basic', async (fake) => {
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
      expect(await fake.host.recap(id, { modelId: MODEL }).done).toBe('none');
      expect(recapLines(fake)).toEqual(['session_recap_unavailable']);
      expect(fake.host.recapState(id)).toBeNull();
    });
  }, 20_000);

  // Mutation: send the recap with `call` (a hard deadline) → a missed ack kills the agent (two starts).
  it('no line within the watchdog, or no ack in time: `none`, and the agent lives', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        expect(await fake.host.recap(id, { modelId: MODEL }).done).toBe('none');
        expect(starts(fake)).toBe(1);
        expect(fake.host.modeState().state).toBe('ready');
        expect((await fake.host.startTurn(id, 'two', { modelId: MODEL }).done).outcome).toBe('completed');
      },
      { timings: { recapMs: 300 }, mutate: recap({ silent: true }) },
    );
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        expect(await fake.host.recap(id, { modelId: MODEL }).done).toBe('none');
        expect(starts(fake)).toBe(1);
        expect(fake.host.modeState().state).toBe('ready');
      },
      { timings: { recapAckMs: 300 }, mutate: recap({ noAck: true, silent: true }) },
    );
  }, 20_000);

  // Mutation: drop the `cancelled` end in `startTurn` → the wait runs on beside the answer and ends `none`.
  it('a typed message ends the wait `cancelled`; the agent drops the recap and no row comes', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const run = fake.host.recap(id, { modelId: MODEL });
        await until(() => recapsAsked(fake).length === 1);
        const turn = fake.host.startTurn(id, 'two', { modelId: MODEL });
        expect(await run.done).toBe('cancelled');
        expect(fake.events.recaps.at(-1)).toEqual({ sessionId: id, recap: null, end: 'cancelled' });
        await turn.done;
        writeFileSync(go, '');
        await until(() => recapLines(fake).length === 1, 5_000);
        expect(recapLines(fake)).toEqual(['session_recap_unavailable']);
        expect(fake.events.recaps.filter((r) => r.recap === null)).toHaveLength(1);
      },
      { mutate: recap({ holdOn: go }) },
    );
  }, 20_000);

  // Mutation: leave the wait to the delete's `drop` (it only clears the watchdog) → the wait never ends.
  it('a delete while it is written ends the wait `cancelled`; nothing more is asked for the session', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const run = fake.host.recap(id, { modelId: MODEL });
        await until(() => recapsAsked(fake).length === 1);
        await fake.host.deleteSession(id, '');
        expect(await Promise.race([run.done, sleep(1_000).then(() => 'still waiting')])).toBe('cancelled');
        expect(fake.events.recaps.filter((r) => r.recap === null)).toEqual([
          { sessionId: id, recap: null, end: 'cancelled' },
        ]);
        expect(fake.host.recapState(id)).toBeNull();
        writeFileSync(go, '');
        await sleep(200);
        expect(recapsAsked(fake)).toHaveLength(1);
      },
      { mutate: recap({ holdOn: go }) },
    );
  }, 20_000);

  it('an error answer is `failed`; the agent exiting during the wait ends it `failed`', async () => {
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        expect(await fake.host.recap(id, { modelId: MODEL }).done).toBe('failed');
      },
      { mutate: recap({ rpcError: true }) },
    );
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const run = fake.host.recap(id, { modelId: MODEL });
        await until(() => recapsAsked(fake).length === 1);
        const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
        process.kill(pid, 'SIGKILL');
        expect(await run.done).toBe('failed');
        expect(fake.events.recaps.filter((r) => r.recap === null)).toEqual([
          { sessionId: id, recap: null, end: 'failed' },
        ]);
      },
      { mutate: recap({ holdOn: go }) },
    );
  }, 20_000);

  it('a turn or a history op refuses it; a cold conversation loads first', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const turn = fake.host.startTurn(id, 'two', { modelId: MODEL });
        expect(codeOf(() => fake.host.recap(id, { modelId: MODEL }))).toBe('conversation-busy');
        await turn.done;
        const run = fake.host.compact(id, { modelId: MODEL }, null);
        expect(codeOf(() => fake.host.recap(id, { modelId: MODEL }))).toBe('conversation-busy');
        writeFileSync(go, '');
        expect(await run.done).toBe('compacted');
        expect(recapsAsked(fake)).toEqual([]);
        await fake.host.restart();
        const before = fake.sent().length;
        expect(await fake.host.recap(id, { modelId: MODEL }).done).toBe('shown');
        const methods = fake
          .sent()
          .slice(before)
          .map((m) => m.method);
        expect(methods.indexOf('session/load')).toBeGreaterThan(-1);
        expect(methods.indexOf('session/load')).toBeLessThan(methods.indexOf('_x.ai/recap'));
      },
      { mutate: compactNow({ holdOn: go }) },
    );
  }, 20_000);

  // Mutation: drop `s.recap` from `hasWork` → the stale restart stops the agent and the wait ends `failed`; from
  // `holdsIdle` → the idle stop does.
  it('the wait counts as work: the idle stop and the stale restart wait for it; history ops are refused while it runs', async () => {
    const go = gate();
    await withHost(
      'chat-basic',
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        await fake.host.startTurn(id, 'one', { modelId: MODEL }).done;
        const run = fake.host.recap(id, { modelId: MODEL });
        await until(() => recapsAsked(fake).length === 1);
        expect(fake.host.opOf(id)).toBe('recap');
        expect(fake.host.active()).toEqual([expect.objectContaining({ sessionId: id, op: 'recap' })]);
        await sleep(500);
        expect(fake.host.modeState().state).toBe('ready');
        fake.host.markStale();
        await sleep(300);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true });
        expect(starts(fake)).toBe(1);
        expect(codeOf(() => fake.host.compact(id, { modelId: MODEL }, null))).toBe('conversation-busy');
        expect(
          codeOf(() =>
            fake.host.rewind(
              id,
              { modelId: MODEL },
              { target: 0, mode: 'conversation', tasks: [], schedules: [], runs: [], files: [] },
            ),
          ),
        ).toBe('conversation-busy');
        await expect(fake.host.fork(id, { modelId: MODEL }, null)).rejects.toMatchObject({ code: 'conversation-busy' });
        writeFileSync(go, '');
        expect(await run.done).toBe('shown');
        expect(fake.host.opOf(id)).toBeNull();
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { timings: { idleStopMs: 150 }, mutate: recap({ holdOn: go }) },
    );
  }, 20_000);

  // Mutation: gate the recap on `engineBusy` (as rewind, compact and fork are) → a live workflow run refuses it.
  it('a live workflow run does not refuse it: a recap only reads', async () => {
    const complete = gate();
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const load = { modelId: MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        const turn = fake.host.startTurn(id, 'p1', load);
        await until(() => fake.events.interactions.length === 1);
        fake.host.answer(id, fake.events.interactions[0]!.interaction.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.host.engineBusy(id)).toBe(true);
        expect(await fake.host.recap(id, load).done).toBe('shown');
        expect(fake.host.runs(id)).toHaveLength(1);
        writeFileSync(complete, '');
      },
      {
        mode: 'work',
        providers: noProviders(),
        mutate: (scenario) => {
          const prompts = scenario.prompts as Json[];
          const steps = prompts[0]!.steps as Json[];
          steps[1] = { workflow: { ...(steps[1]!.workflow as Json), ask: false, completeOn: complete } };
        },
      },
    );
  }, 20_000);
});

describe('Recap through the service', () => {
  it('answers at once; the wait and its end ride the stream; the row comes as an entry; a busy one is refused', async () => {
    const go = gate();
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scenario.recap = { holdOn: go, summary: 'The fix is in. Next: the docs.' };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
      const id = conversation.id;
      const events: ConversationEvent[] = [];
      let state: ConversationState | null = null;
      const detach = fake.service.watchConversation(id, (event) => {
        events.push(event);
        const next = applyConversationEvent(state, event);
        if (next !== 'resync') state = next;
      });
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      expect(await fake.service.recap(id)).toEqual({});
      await until(() => state !== null && state.recap !== null);
      expect(state!.recap).toEqual({ startedAt: expect.any(Number) });
      expect(fake.service.busyConversations()).toEqual([]);
      expect(routeError('chat', await fake.service.recap(id).catch((e: unknown) => e))).toMatchObject({
        reason: 'conversation-busy',
      });
      writeFileSync(go, '');
      await until(() => events.some((e) => e.type === 'recap' && e.end !== undefined), 5_000);
      expect(events.filter((e) => e.type === 'recap').map((e) => (e.type === 'recap' ? e.end : null))).toEqual([
        undefined,
        'shown',
      ]);
      expect(state!.recap).toBeNull();
      expect(state!.recapEnd).toEqual({ outcome: 'shown' });
      await until(() => state!.entries.some((e) => e.kind === 'recap'), 5_000);
      expect(state!.entries.filter((e) => e.kind === 'recap')).toEqual([
        { kind: 'recap', ts: expect.any(Number), text: 'The fix is in. Next: the docs.' },
      ]);
      detach();
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('a page that attaches while it is written gets the wait in its sync; the engine stopped refuses it', async () => {
    const go = gate();
    const fake = fakeService('chat-basic', {
      mutate: (scenario) => {
        scenario.recap = { holdOn: go };
      },
    });
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
      const id = conversation.id;
      await until(() => !fake.service.busyConversations().includes(id), 5_000);
      const endpoint = fake.endpoint.current;
      fake.endpoint.current = null;
      const stopped = routeError('chat', await fake.service.recap(id).catch((e: unknown) => e));
      expect([stopped.code, stopped.reason]).toEqual(['E_UNAVAILABLE', 'inference-stopped']);
      fake.endpoint.current = endpoint;
      expect(await fake.service.recap(id)).toEqual({});
      const events: ConversationEvent[] = [];
      const detach = fake.service.watchConversation(id, (event) => events.push(event));
      await until(() => events.length > 0);
      expect(events[0]).toMatchObject({ type: 'sync', recap: { startedAt: expect.any(Number) } });
      writeFileSync(go, '');
      await until(() => events.some((e) => e.type === 'recap' && e.end === 'shown'), 5_000);
      detach();
    } finally {
      await fake.close();
    }
  }, 20_000);
});

describe('A cold load for an op, a recap or a Delete is soft', () => {
  const starts = (fake: FakeHost): number => fake.records().filter((r) => r.kind === 'start').length;
  const sentTo = (fake: FakeHost, ids: readonly string[], method: string): Json[] =>
    fake.sent().filter((m) => m.method === method && ids.includes((m.params as Json).sessionId as string));
  const codeOrOk = (run: Promise<unknown>): Promise<unknown> =>
    run.then(
      () => 'ok',
      (error: unknown) => (error instanceof AgentHostError ? error.code : String(error)),
    );

  // Mutation: load every caller with the turn's hard path → no op ends `load-slow` within the soft deadline.
  // Mutation: let the soft wait's miss close the agent → a second process starts and the command is gone.
  it('a slow session/load ends each op load-slow; the agent and another conversation’s command live on; the late load lands and runs nothing', async () => {
    await withHost(
      'work-workflow-run',
      async (fake) => {
        const load = { modelId: MODEL, cwd: fake.project };
        const cold: string[] = [];
        for (let i = 0; i < 5; i++) {
          const { sessionId } = await fake.host.createSession(load);
          expect((await fake.host.startTurn(sessionId, 'first', load).done).outcome).toBe('completed');
          cold.push(sessionId);
        }
        await fake.host.restart();
        const { sessionId: live } = await fake.host.createSession(load);
        expect((await fake.host.startTurn(live, 'cmd', load).done).outcome).toBe('completed');
        await until(() => fake.host.liveActivity(live).tasks.length === 1);
        const before = starts(fake);
        const [plan, recap, compact, schedule, rewind] = cold as [string, string, string, string, string];
        const ends = await Promise.all([
          codeOrOk(fake.host.rewindPreview(plan, load, 0)),
          fake.host.recap(recap, load).done,
          fake.host.compact(compact, load, null).done,
          codeOrOk(fake.host.deleteSchedule(schedule, 's1', load)),
          fake.host.rewind(rewind, load, {
            target: 0,
            mode: 'conversation',
            tasks: [],
            schedules: [],
            runs: [],
            files: [],
          }).done,
        ]);
        expect(ends).toEqual([
          'load-slow',
          'load-slow',
          'load-slow',
          'load-slow',
          { outcome: 'failed', failure: 'load-slow' },
        ]);
        expect(fake.events.recaps.at(-1)).toEqual({ sessionId: recap, recap: null, end: 'load-slow' });
        expect(fake.events.compactions.at(-1)).toEqual({ sessionId: compact, compaction: null, end: 'load-slow' });
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.host.liveActivity(live).tasks.map((task) => task.id)).toEqual(['t1']);
        for (const id of cold) expect(fake.host.isBusy(id)).toBe(false);

        await until(() => cold.every((id) => fake.host.isHot(id)), 8_000);
        expect(starts(fake)).toBe(before);
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.host.liveActivity(live).tasks.map((task) => task.id)).toEqual(['t1']);
        for (const method of [
          '_x.ai/rewind/execute',
          '_x.ai/recap',
          '_x.ai/compact_conversation',
          '_x.ai/scheduler/delete',
        ]) {
          expect(sentTo(fake, cold, method)).toEqual([]);
        }
        // Loaded now: the next op runs at once, with no second load.
        const loads = sentTo(fake, [compact], 'session/load').length;
        expect(await fake.host.compact(compact, load, null).done).toBe('compacted');
        expect(sentTo(fake, [compact], 'session/load')).toHaveLength(loads);
      },
      {
        mode: 'work',
        providers: noProviders(),
        timings: { loadSoftMs: 300 },
        mutate: (scenario) => {
          scenario.prompts = [
            {
              match: 'cmd',
              steps: [
                { background: { taskId: 't1', wake: false } },
                { text: 'Started.' },
                { end: { stopReason: 'end_turn' } },
              ],
            },
          ];
          scenario.faults = { delayMs: { 'session/load': 1_500 } };
        },
      },
    );
  }, 40_000);

  // Mutation: drop the `harden` of a turn that waits on a soft load → the turn's load has no deadline.
  // Mutation: keep the guard's read soft once a turn waits → its `_x.ai/session/info` goes through `requestSoft`.
  it('a turn that waits on a soft load gives its session/load the hard deadline of a turn; a turn’s own load is hard', async () => {
    const harden = vi.spyOn(AcpClient.prototype, 'harden');
    const soft = vi.spyOn(AcpClient.prototype, 'requestSoft');
    const softReads = (): unknown[] => soft.mock.calls.filter(([method]) => method === '_x.ai/session/info');
    try {
      await withHost(
        'chat-basic',
        async (fake) => {
          const load = { modelId: MODEL };
          const { sessionId: id } = await fake.host.createSession(load);
          expect((await fake.host.startTurn(id, 'one', load).done).outcome).toBe('completed');
          await fake.host.restart();
          const recap = fake.host.recap(id, load);
          await until(() => asked(fake, 'session/load').length === 1);
          const loadId = asked(fake, 'session/load')[0]!.id;
          expect(harden).not.toHaveBeenCalled();
          const turn = fake.host.startTurn(id, 'two', load);
          expect(await recap.done).toBe('cancelled');
          expect((await turn.done).outcome).toBe('completed');
          expect(harden.mock.calls).toEqual([[loadId, ACP_TIMEOUTS['session/load']]]);
          expect(asked(fake, 'session/load')).toHaveLength(1);
          expect(softReads()).toEqual([]);

          await fake.host.restart();
          expect((await fake.host.startTurn(id, 'three', load).done).outcome).toBe('completed');
          expect(harden).toHaveBeenCalledTimes(1);
          expect(softReads()).toEqual([]);

          // With no turn waiting, the guard's read of a soft load is soft.
          await fake.host.restart();
          expect(await fake.host.compact(id, load, null).done).toBe('compacted');
          expect(softReads()).toEqual([['_x.ai/session/info', { sessionId: id }, ACP_TIMEOUTS['_x.ai/session/info']]]);
        },
        {
          timings: { loadSoftMs: 10_000 },
          mutate: (scenario) => (scenario.faults = { delayMs: { 'session/load': 600 } }),
        },
      );
    } finally {
      harden.mockRestore();
      soft.mockRestore();
    }
  }, 30_000);

  // Mutation: harden only a request already sent → a turn that joins before it goes waits with no deadline.
  it('a turn that joins a soft load before its request went: the request goes with the hard deadline', async () => {
    const harden = vi.spyOn(AcpClient.prototype, 'harden');
    try {
      await withHost(
        'work-workflow-run',
        async (fake) => {
          const load = { modelId: MODEL, cwd: fake.project };
          const { sessionId: id } = await fake.host.createSession(load);
          expect((await fake.host.startTurn(id, 'one', load).done).outcome).toBe('completed');
          await fake.host.restart();
          const plan = fake.host.rewindPreview(id, load, 0).catch(() => null);
          const turn = fake.host.startTurn(id, 'two', load);
          expect((await turn.done).outcome).toBe('completed');
          await plan;
          const loads = asked(fake, 'session/load');
          expect(loads).toHaveLength(1);
          expect(harden.mock.calls).toEqual([[loads[0]!.id, ACP_TIMEOUTS['session/load']]]);
        },
        {
          mode: 'work',
          providers: noProviders(),
          timings: { loadSoftMs: 10_000 },
          mutate: (scenario) => {
            scenario.prompts = [{ match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] }];
            scenario.faults = { delayMs: { 'session/load': 300 } };
          },
        },
      );
    } finally {
      harden.mockRestore();
    }
  }, 30_000);
});
