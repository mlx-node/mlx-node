/**
 * The thinking level on the host, against the fake agent (which plays the agent's `reasoning_effort` option: its
 * initialize list, the `configOptions` of a session reply, `session/set_config_option`, the live `config_option_update`
 * and the process default). The set call goes out after the load and before the prompt, once per change, never when
 * the session holds the level; a stored level the model does not list steps down as the page shows it; a refusal or a
 * miss gives the `thinking-not-set` notice and the answer still runs; a later set that took, or a session that holds
 * the level, clears it.
 */

import { describe, expect, it } from 'vite-plus/test';

import { endpointModel, fakeHost, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
/** A model whose engine row lists no levels. */
const PLAIN = 'qwen3.5-9b';
/** A second model with the same levels. */
const OTHER = 'qwen3.5-27b';
/** A qwen3_5 row's `reasoningEfforts` as the engine writes them (`thinking-levels.test.ts`). */
const QWEN_EFFORTS = [
  { value: 'none', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High', default: true },
];

function scripted(scenario: Json): void {
  scenario.initialize = {
    models: [
      { modelId: MODEL, totalContextTokens: 32768, reasoningEfforts: QWEN_EFFORTS },
      { modelId: PLAIN, totalContextTokens: 32768 },
      { modelId: OTHER, totalContextTokens: 32768, reasoningEfforts: QWEN_EFFORTS },
    ],
  };
  scenario.prompts = [{ match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] }];
}

async function withHost(
  run: (fake: FakeHost) => Promise<void>,
  options: FakeHostOptions & { also?: (scenario: Json) => void } = {},
): Promise<void> {
  const { also, ...rest } = options;
  const fake = await fakeHost(rest.mode === 'chat' ? 'chat-basic' : 'work-length', {
    mode: 'work',
    endpointModels: [MODEL, PLAIN, OTHER].map((name) => endpointModel(name, `/models/${name}`)),
    ...rest,
    mutate: (scenario) => {
      scripted(scenario);
      also?.(scenario);
    },
  });
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

/** The session methods the fake agent got, in order. */
const order = (fake: FakeHost): string[] =>
  fake
    .sent()
    .map((m) => String(m.method))
    .filter((m) => ['session/new', 'session/load', 'session/set_config_option', 'session/prompt'].includes(m));

const sets = (fake: FakeHost): Json[] =>
  fake
    .sent()
    .filter((m) => m.method === 'session/set_config_option')
    .map((m) => m.params as Json);

describe('Thinking level on the host', () => {
  it('lists the levels of each model as the agent lists them', async () => {
    await withHost(async (fake) => {
      await fake.host.start();
      expect(fake.host.models()).toEqual([
        {
          modelId: MODEL,
          contextTokens: 32768,
          thinking: { levels: ['none', 'low', 'medium', 'high'], default: 'high' },
        },
        { modelId: PLAIN, contextTokens: 32768, thinking: null },
        {
          modelId: OTHER,
          contextTokens: 32768,
          thinking: { levels: ['none', 'low', 'medium', 'high'], default: 'high' },
        },
      ]);
    });
  }, 20_000);

  it('sets the level after the load and before the prompt, once per change, and again after a load', async () => {
    await withHost(async (fake) => {
      const cwd = fake.project;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd });
      const low = { modelId: MODEL, cwd, thinking: 'low' as const };
      expect((await fake.host.startTurn(id, 'one', low).done).outcome).toBe('completed');
      expect(order(fake)).toEqual(['session/new', 'session/set_config_option', 'session/prompt']);
      expect(sets(fake)).toEqual([{ sessionId: id, configId: 'reasoning_effort', value: 'low' }]);

      // The session holds it: no call.
      expect((await fake.host.startTurn(id, 'two', low).done).outcome).toBe('completed');
      expect(sets(fake)).toHaveLength(1);

      // No level picked: the model's default goes back.
      expect((await fake.host.startTurn(id, 'three', { modelId: MODEL, cwd }).done).outcome).toBe('completed');
      expect(sets(fake).map((p) => p.value)).toEqual(['low', 'high']);

      // A new process forgot it: the load reply says the default, so the level goes again before the prompt.
      await fake.host.restart();
      expect((await fake.host.startTurn(id, 'four', low).done).outcome).toBe('completed');
      expect(order(fake).slice(-3)).toEqual(['session/load', 'session/set_config_option', 'session/prompt']);
      expect(sets(fake).map((p) => p.value)).toEqual(['low', 'high', 'low']);

      // The live lines of a set are the host's alone: none reaches the reducer.
      const kinds = fake.events.updates.map((u) => u.line.update.sessionUpdate);
      expect(kinds).not.toContain('config_option_update');
      expect(kinds).not.toContain('model_changed');
      expect(fake.events.notices).toEqual([]);
    });
  }, 20_000);

  it('a session that starts at the level another conversation left as the process default gets its own', async () => {
    await withHost(async (fake) => {
      const cwd = fake.project;
      const a = await fake.host.createSession({ modelId: MODEL, cwd });
      await fake.host.startTurn(a.sessionId, 'one', { modelId: MODEL, cwd, thinking: 'none' }).done;
      const b = await fake.host.createSession({ modelId: MODEL, cwd });
      expect((await fake.host.startTurn(b.sessionId, 'two', { modelId: MODEL, cwd }).done).outcome).toBe('completed');
      expect(sets(fake)).toEqual([
        { sessionId: a.sessionId, configId: 'reasoning_effort', value: 'none' },
        { sessionId: b.sessionId, configId: 'reasoning_effort', value: 'high' },
      ]);
    });
  }, 20_000);

  it('a stored level the model does not list steps down to the one the page shows', async () => {
    await withHost(async (fake) => {
      const cwd = fake.project;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd });
      const max = { modelId: MODEL, cwd, thinking: 'max' as const };
      // Max on a model of Off · Low · Medium · High is High, which the session already holds: no call, no notice.
      expect((await fake.host.startTurn(id, 'one', max).done).outcome).toBe('completed');
      expect(sets(fake)).toEqual([]);
      await fake.host.startTurn(id, 'two', { modelId: MODEL, cwd, thinking: 'low' }).done;
      expect((await fake.host.startTurn(id, 'three', max).done).outcome).toBe('completed');
      expect(sets(fake).map((p) => p.value)).toEqual(['low', 'high']);
      expect(fake.events.notices).toEqual([]);
    });
  }, 20_000);

  it('a refused set gives the notice; a later set that took clears it', async () => {
    await withHost(
      async (fake) => {
        const cwd = fake.project;
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd });
        const low = { modelId: MODEL, cwd, thinking: 'low' as const };
        expect((await fake.host.startTurn(id, 'one', low).done).outcome).toBe('completed');
        expect(fake.events.notices).toEqual([{ sessionId: id, notice: { code: 'thinking-not-set', tone: 'warning' } }]);
        expect(fake.events.cleared).toEqual([]);
        expect((await fake.host.startTurn(id, 'two', low).done).outcome).toBe('completed');
        expect(sets(fake).map((p) => p.value)).toEqual(['low', 'low']);
        expect(fake.events.cleared).toEqual([{ sessionId: id, code: 'thinking-not-set' }]);
      },
      { also: (scenario) => (scenario.session = { ...(scenario.session as Json), refuseEffort: 1 }) },
    );
  }, 20_000);

  it('a refusal gives the notice and the answer still runs', async () => {
    await withHost(
      async (fake) => {
        const cwd = fake.project;
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd });
        const turn = fake.host.startTurn(id, 'one', { modelId: MODEL, cwd, thinking: 'low' });
        expect((await turn.done).outcome).toBe('completed');
        expect(order(fake)).toEqual(['session/new', 'session/set_config_option', 'session/prompt']);
        expect(fake.events.notices.map((n) => n.notice.code)).toEqual(['thinking-not-set']);
        expect(fake.events.cleared).toEqual([]);
      },
      { also: (scenario) => (scenario.session = { ...(scenario.session as Json), refuseEffort: true }) },
    );
  }, 20_000);

  it('a set past its soft deadline gives the notice, the answer goes on, and the late OK records the level', async () => {
    await withHost(
      async (fake) => {
        const cwd = fake.project;
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd });
        const low = { modelId: MODEL, cwd, thinking: 'low' as const };
        expect((await fake.host.startTurn(id, 'one', low).done).outcome).toBe('completed');
        expect(fake.events.notices.map((n) => n.notice.code)).toEqual(['thinking-not-set']);
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.events.cleared).toEqual([]);
        // The late reply lands: the next answer needs no call, and runs at the level, so the notice goes.
        await new Promise((resolve) => setTimeout(resolve, 700));
        expect((await fake.host.startTurn(id, 'two', low).done).outcome).toBe('completed');
        expect(sets(fake)).toHaveLength(1);
        expect(fake.events.cleared).toEqual([{ sessionId: id, code: 'thinking-not-set' }]);
      },
      {
        timings: { effortSetMs: 100 },
        also: (scenario) => (scenario.faults = { delayMs: { 'session/set_config_option': 500 } }),
      },
    );
  }, 20_000);

  it('a model change: the agent says what it kept, a model with no levels gets no call', async () => {
    await withHost(async (fake) => {
      const cwd = fake.project;
      const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd });
      await fake.host.startTurn(id, 'one', { modelId: MODEL, cwd, thinking: 'low' }).done;
      // The new model lists the level: the agent keeps it, its push says so, and no call goes.
      await fake.host.setModel(id, OTHER);
      expect((await fake.host.startTurn(id, 'kept', { modelId: OTHER, cwd, thinking: 'low' }).done).outcome).toBe(
        'completed',
      );
      expect(sets(fake)).toHaveLength(1);
      await fake.host.setModel(id, PLAIN);
      expect((await fake.host.startTurn(id, 'two', { modelId: PLAIN, cwd, thinking: 'low' }).done).outcome).toBe(
        'completed',
      );
      expect(sets(fake)).toHaveLength(1);
      // Back on a model with levels the agent starts it at its default (its push says so): low goes again.
      await fake.host.setModel(id, MODEL);
      await fake.host.startTurn(id, 'three', { modelId: MODEL, cwd, thinking: 'low' }).done;
      expect(sets(fake).map((p) => p.value)).toEqual(['low', 'low']);
      expect(fake.events.notices).toEqual([]);
    });
  }, 20_000);

  it('Chat sets the level too', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        expect((await fake.host.startTurn(id, 'hi', { modelId: MODEL, thinking: 'none' }).done).outcome).toBe(
          'completed',
        );
        expect(sets(fake)).toEqual([{ sessionId: id, configId: 'reasoning_effort', value: 'none' }]);
      },
      { mode: 'chat' },
    );
  }, 20_000);
});
