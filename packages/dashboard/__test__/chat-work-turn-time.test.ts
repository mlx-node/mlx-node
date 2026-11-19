/**
 * How long a turn may run, on the scripted fake agent with a fake clock: a Work turn has no time limit (it ends when
 * the agent ends it, at Stop, or when the engine or the agent is gone); a Chat turn still stops at its cap, and the
 * time the agent compacts the conversation is not counted against it.
 */

import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CHAT_TURN_MAX_MS } from '../src/chat/agent/agent-host.js';
import { fakeHost, noProviders, until, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const DAY_MS = 86_400_000;

/** One prompt: a first text, a real wait of the agent, a last text, the end. */
function slowTurn(sleepMs: number): (scenario: Json) => void {
  return (scenario) => {
    scenario.prompts = [
      {
        match: '',
        steps: [
          { text: 'Working on it', chunk: 5 },
          { sleepMs },
          { text: ' and done.', chunk: 5 },
          { end: { stopReason: 'end_turn' } },
        ],
      },
    ];
  };
}

async function withHost(name: string, options: FakeHostOptions, run: (fake: FakeHost) => Promise<void>) {
  const fake = await fakeHost(name, { providers: noProviders(), ...options });
  try {
    await run(fake);
  } finally {
    vi.useRealTimers();
    await fake.close();
  }
}

function texts(fake: FakeHost): string {
  return fake.events.updates
    .map((u) => {
      const content = u.line.update.content as Json | undefined;
      return typeof content?.text === 'string' ? content.text : '';
    })
    .join('');
}

async function running(fake: FakeHost, id: string, cwd?: string) {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
  const turn = fake.host.startTurn(id, 'go', { modelId: MODEL, ...(cwd === undefined ? {} : { cwd }) });
  await until(() => texts(fake).includes('Working'), 5_000);
  return turn;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('a Work turn', () => {
  it('runs past a day of the clock and ends when the agent ends it', async () => {
    await withHost('work-permission', { mode: 'work', mutate: slowTurn(1_500) }, async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = await running(fake, sessionId, fake.project);
      vi.advanceTimersByTime(3_600_001);
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(fake.host.modeState().turnsInFlight).toBe(1);
      vi.advanceTimersByTime(DAY_MS);
      const end = await turn.done;
      expect(end.outcome).toBe('completed');
      expect(texts(fake)).toContain('and done.');
    });
  }, 20_000);

  it('Stop still ends it as cancelled', async () => {
    await withHost('work-permission', { mode: 'work', mutate: slowTurn(5_000) }, async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = await running(fake, sessionId, fake.project);
      vi.advanceTimersByTime(DAY_MS);
      fake.host.cancel(sessionId);
      expect((await turn.done).outcome).toBe('cancelled');
    });
  }, 20_000);

  it('the engine gone still ends it as inference-lost', async () => {
    await withHost('work-permission', { mode: 'work', mutate: slowTurn(5_000) }, async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = await running(fake, sessionId, fake.project);
      vi.advanceTimersByTime(DAY_MS);
      fake.endpoint.current = null;
      await fake.host.checkEndpoint();
      expect((await turn.done).outcome).toBe('inference-lost');
    });
  }, 20_000);

  it('an agent crash still ends it as agent-crashed', async () => {
    const crash = (scenario: Json): void => {
      scenario.prompts = [
        {
          match: '',
          steps: [{ text: 'Working on it', chunk: 5 }, { sleepMs: 800 }, { exit: 9 }],
        },
      ];
    };
    await withHost('work-permission', { mode: 'work', mutate: crash }, async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = await running(fake, sessionId, fake.project);
      vi.advanceTimersByTime(DAY_MS);
      const end = await turn.done;
      expect([end.outcome, end.failure?.code]).toEqual(['agent-crashed', 'agent-crashed']);
    });
  }, 20_000);
});

describe('a Chat turn', () => {
  it('still stops at its cap as turn-timeout', async () => {
    await withHost('cancel', { mode: 'chat' }, async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'], shouldAdvanceTime: true });
      const turn = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'), 5_000);
      vi.advanceTimersByTime(CHAT_TURN_MAX_MS - 1_000);
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(fake.host.modeState().turnsInFlight).toBe(1);
      vi.advanceTimersByTime(1_001);
      expect((await turn.done).outcome).toBe('turn-timeout');
    });
  }, 20_000);

  it('a compaction holds the cap; the cap runs on with the time that was left', async () => {
    let clock = 1_000_000;
    const compacting = (scenario: Json): void => {
      scenario.prompts = [
        {
          match: '',
          steps: [
            { text: 'Working on it', chunk: 5 },
            { sleepMs: 300 },
            { compact: { ms: 1_500 } },
            { text: ' and done.', chunk: 5 },
            { sleepMs: 20_000 },
            { end: { stopReason: 'end_turn' } },
          ],
        },
      ];
    };
    await withHost('cancel', { mode: 'chat', mutate: compacting, now: () => clock }, async (fake) => {
      const advance = (ms: number): void => {
        clock += ms;
        vi.advanceTimersByTime(ms);
      };
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const turn = await running(fake, sessionId);
      advance(CHAT_TURN_MAX_MS - 2_000);
      await until(() => fake.host.compaction(sessionId) !== null, 5_000);
      advance(DAY_MS);
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(fake.host.modeState().turnsInFlight).toBe(1);
      await until(() => fake.host.compaction(sessionId) === null, 5_000);
      advance(1_500);
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(fake.host.modeState().turnsInFlight).toBe(1);
      advance(600);
      expect((await turn.done).outcome).toBe('turn-timeout');
    });
  }, 20_000);

  it('a turn_completed line that ends an open compaction gives the cap back', async () => {
    const unclosed = (scenario: Json): void => {
      scenario.prompts = [
        {
          match: '',
          steps: [
            { text: 'Working on it', chunk: 5 },
            { compact: { outcome: 'none' } },
            { turnCompleted: {} },
            { sleepMs: 20_000 },
            { end: { stopReason: 'end_turn' } },
          ],
        },
      ];
    };
    await withHost('cancel', { mode: 'chat', mutate: unclosed }, async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const turn = await running(fake, sessionId);
      await until(() => fake.events.updates.some((u) => u.line.update.sessionUpdate === 'turn_completed'), 5_000);
      expect(fake.host.compaction(sessionId)).toBeNull();
      vi.advanceTimersByTime(CHAT_TURN_MAX_MS + 1);
      const ended = await Promise.race([turn.done, new Promise((resolve) => setTimeout(() => resolve(null), 3_000))]);
      expect(ended).toMatchObject({ outcome: 'turn-timeout' });
    });
  }, 20_000);
});
