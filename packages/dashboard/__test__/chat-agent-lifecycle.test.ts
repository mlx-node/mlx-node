/**
 * The agent host where timing decides the outcome: a reopen whose guard fails, calls that arrive while the process
 * stops or starts, a prompt during the create guard, a cancelled reopen, a web server that dies, and agent lines the
 * Control Panel cannot handle. All on the scripted fake agent.
 */

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { AgentHostError } from '../src/chat/agent/agent-host.js';
import { fakeHost, fakeService, until, type FakeHost, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';

async function withFake(
  name: string,
  run: (fake: FakeHost) => Promise<void>,
  options: Parameters<typeof fakeHost>[1] = {},
): Promise<void> {
  const fake = await fakeHost(name, options);
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

function rewriteScenario(fake: FakeHost, change: (scenario: Json) => void): void {
  const path = join(fake.root, 'scenario.json');
  const scenario = JSON.parse(readFileSync(path, 'utf8')) as Json;
  change(scenario);
  writeFileSync(path, JSON.stringify(scenario));
}

/** The session folders the fake agent wrote under the Chat agent home. */
function sessionDirs(fake: FakeHost): string[] {
  const root = join(fake.paths.chatAgent, 'sessions');
  if (!existsSync(root)) return [];
  return readdirSync(root).flatMap((cwd) => readdirSync(join(root, cwd)).map((id) => join(root, cwd, id)));
}

function pids(fake: FakeHost): number[] {
  return fake
    .records()
    .filter((r) => r.kind === 'start')
    .map((r) => r.pid as number);
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function methods(fake: FakeHost): unknown[] {
  return fake.sent().map((m) => m.method);
}

describe('a reopen whose guard fails', () => {
  it('keeps the saved conversation: the session is dropped from the table, never deleted through the agent', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        expect((await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done).outcome).toBe('completed');
        const [dir] = sessionDirs(fake);
        expect(dir).toBeDefined();
        await fake.host.restart();
        rewriteScenario(fake, (s) => {
          (s.session as Json).mcpReadyAfterMs = null;
        });
        const result = await fake.host.startTurn(sessionId, 'again', { modelId: MODEL }).done;
        expect(result).toMatchObject({ outcome: 'refused', failure: { code: 'profile-mismatch' } });
        expect(methods(fake)).toContain('session/load');
        expect(methods(fake)).not.toContain('_x.ai/session/delete');
        expect(existsSync(join(dir!, 'summary.json'))).toBe(true);
        expect(existsSync(join(dir!, 'updates.jsonl'))).toBe(true);
        expect(fake.host.isHot(sessionId)).toBe(false);
      },
      { timings: { mcpReadyMs: 300 } },
    );
  });
});

describe('calls while the process stops or starts', () => {
  it('a prompt during an idle stop waits for the old process and runs on a new one', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        await until(() => fake.host.models() === null);
        expect(alive(pids(fake)[0]!)).toBe(true);
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result.outcome).toBe('completed');
        expect(pids(fake)).toHaveLength(2);
        expect(methods(fake).filter((m) => m === 'session/load')).toHaveLength(1);
      },
      {
        timings: { idleStopMs: 200, stopEofMs: 3_000 },
        mutate: (s) => {
          s.faults = { exitAfterEofMs: 800 };
        },
      },
    );
  });

  it('a stale restart with no turn lets the next prompt start a new process', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        fake.host.markStale();
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result.outcome).toBe('completed');
        expect(pids(fake)).toHaveLength(2);
      },
      {
        timings: { stopEofMs: 3_000 },
        mutate: (s) => {
          s.faults = { exitAfterEofMs: 500 };
        },
      },
    );
  });

  it('a stale process is restarted for the prompt that finds it, when no other turn runs', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'one', { modelId: MODEL });
      fake.host.markStale();
      expect((await started.done).outcome).toBe('completed');
      expect(pids(fake)).toHaveLength(2);
      expect(methods(fake)).toContain('session/load');
    });
  });

  it('a replaced endpoint found before the prompt restarts the process and reloads the session', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      fake.endpoint.current = { ...fake.endpoint.current!, token: 'tok-REPLACED-0001' };
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result.outcome).toBe('completed');
      const records = fake.records();
      const starts = records.flatMap((r, i) => (r.kind === 'start' ? [i] : []));
      expect(starts).toHaveLength(2);
      const after = (method: string): number[] =>
        records.flatMap((r, i) => (r.kind === 'in' && (r.msg as Json).method === method ? [i] : []));
      expect(after('session/prompt')).toHaveLength(1);
      expect(after('session/prompt')[0]!).toBeGreaterThan(starts[1]!);
      expect(after('session/load').some((i) => i > starts[1]!)).toBe(true);
    });
  });

  it('an endpoint that changes again during the restart fails the turn after one restart', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const first = fake.endpoint.current!;
      let reads = 0;
      Object.defineProperty(fake.endpoint, 'current', {
        get: () => ({ ...first, token: `tok-MOVING-${String(++reads).padStart(4, '0')}` }),
      });
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result).toMatchObject({ failure: { code: 'inference-failed' } });
      expect(pids(fake)).toHaveLength(2);
      expect(fake.sent().filter((m) => m.method === 'session/prompt')).toHaveLength(0);
    });
  });

  it('a replaced endpoint found before the prompt while another turn runs refuses the prompt', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const a = await fake.host.createSession({ modelId: MODEL });
        const b = await fake.host.createSession({ modelId: MODEL });
        const slow = fake.host.startTurn(a.sessionId, 'slow', { modelId: MODEL });
        await until(() => fake.host.turnState(a.sessionId)?.phase === 'running');
        fake.endpoint.current = { ...fake.endpoint.current!, url: 'http://127.0.0.1:10', token: 'tok-REPLACED-0002' };
        const result = await fake.host.startTurn(b.sessionId, 'hi', { modelId: MODEL }).done;
        expect(result).toMatchObject({ failure: { code: 'inference-failed' } });
        const prompts = fake.sent().filter((m) => m.method === 'session/prompt');
        expect(prompts.map((m) => (m.params as Json).sessionId)).toEqual([a.sessionId]);
        fake.host.cancel(a.sessionId);
        await slow.done;
      },
      {
        mutate: (s) => {
          s.prompts = [
            {
              match: 'slow',
              steps: [{ text: 'Thinking', chunk: 3 }, { sleepMs: 5000 }, { end: { stopReason: 'end_turn' } }],
            },
            ...(s.prompts as Json[]),
          ];
        },
      },
    );
  });

  it('close in the same tick as a start leaves no process behind', async () => {
    const fake = await fakeHost('chat-basic');
    const started = fake.host.start().catch((e: unknown) => e);
    await fake.host.close();
    expect(await started).toBeInstanceOf(AgentHostError);
    expect(fake.host.modeState().state).toBe('stopped');
    for (const pid of pids(fake)) await until(() => !alive(pid));
    await fake.close();
  });

  it('close during a start stops the process that was starting', async () => {
    const fake = await fakeHost('chat-basic', {
      mutate: (s) => {
        s.faults = { delayMs: { initialize: 600 } };
      },
    });
    const started = fake.host.start().catch((e: unknown) => e);
    await until(() => pids(fake).length === 1);
    const [pid] = pids(fake);
    await fake.host.close();
    await started;
    expect(fake.host.modeState().state).toBe('stopped');
    await new Promise((resolve) => setTimeout(resolve, 700));
    expect(fake.host.modeState().state).toBe('stopped');
    expect(alive(pid!)).toBe(false);
    await fake.close();
  });

  it('restart during a start leaves the host stopped and the next use starts a new process', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const started = fake.host.start().catch((e: unknown) => e);
        await until(() => pids(fake).length === 1);
        const [pid] = pids(fake);
        await fake.host.restart();
        expect(await started).toBeInstanceOf(AgentHostError);
        expect(fake.host.modeState()).toMatchObject({ state: 'stopped', failure: null });
        expect(alive(pid!)).toBe(false);
        rewriteScenario(fake, (s) => {
          s.faults = {};
        });
        await fake.host.start();
        expect(fake.host.modeState().state).toBe('ready');
        expect(pids(fake)).toHaveLength(2);
      },
      {
        mutate: (s) => {
          s.faults = { delayMs: { initialize: 600 } };
        },
      },
    );
  });
});

describe('the create guard and reopen loads', () => {
  it('a prompt for a session whose create guard still runs is refused and never reaches the agent', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const created = fake.host.createSession({ modelId: MODEL });
        await until(() => sessionDirs(fake).length === 1);
        const id = sessionDirs(fake)[0]!.split('/').at(-1)!;
        // The fake writes the folder before its reply; the guard starts only when the host has the reply.
        await until(() => fake.host.sessionModel(id) !== null);
        expect(() => fake.host.startTurn(id, 'hi', { modelId: MODEL })).toThrow(AgentHostError);
        await expect(fake.host.rename(id, 'x')).rejects.toBeInstanceOf(AgentHostError);
        expect((await created).sessionId).toBe(id);
        expect(methods(fake)).not.toContain('session/prompt');
        expect((await fake.host.startTurn(id, 'hi', { modelId: MODEL }).done).outcome).toBe('completed');
        const order = methods(fake);
        expect(order.indexOf('_x.ai/session/info')).toBeLessThan(order.indexOf('session/prompt'));
      },
      { mutate: (s) => ((s.session as Json).mcpReadyAfterMs = 2_500) },
    );
  });

  it('a reopen cancelled while it loads shares its load with the next prompt', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      await fake.host.restart();
      rewriteScenario(fake, (s) => {
        s.faults = { delayMs: { 'session/load': 400 } };
      });
      const first = fake.host.startTurn(sessionId, 'one', { modelId: MODEL });
      await until(() => methods(fake).includes('session/load'));
      expect(fake.host.cancel(sessionId)).toBe(true);
      expect((await first.done).outcome).toBe('cancelled');
      const second = await fake.host.startTurn(sessionId, 'two', { modelId: MODEL }).done;
      expect(second.outcome).toBe('completed');
      expect(methods(fake).filter((m) => m === 'session/load')).toHaveLength(1);
    });
  });
});

describe('recorded agent shapes', () => {
  it('a web server that dies mid-turn: failed rows, the turn completes, and the next prompt reloads the session', async () => {
    await withFake('chat-web-server-died', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const result = await fake.host.startTurn(sessionId, 'search', { modelId: MODEL }).done;
      expect(result.outcome).toBe('completed');
      const failed = fake.events.updates.filter((u) => u.line.update.status === 'failed');
      expect(failed).toHaveLength(2);
      expect(fake.events.web).toEqual([]);
      await until(() => fake.host.modeState().state === 'stopped');
      expect(fake.host.isHot(sessionId)).toBe(false);
      rewriteScenario(fake, (s) => {
        s.prompts = [];
      });
      expect((await fake.host.startTurn(sessionId, 'again', { modelId: MODEL }).done).outcome).toBe('completed');
      const load = fake.sent().find((m) => m.method === 'session/load')!;
      expect(load.params).toMatchObject({ sessionId, mcpServers: [{ name: 'mlx_web' }] });
    });
  });

  it('a call the agent denies by policy is a failed row and the answer goes on', async () => {
    await withFake('chat-policy-denied', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result.outcome).toBe('completed');
      expect(fake.events.updates.some((u) => u.line.update.status === 'failed')).toBe(true);
      expect(fake.sent().some((m) => m.method === undefined && m.result !== undefined && m.id !== undefined)).toBe(
        false,
      );
    });
  });
});

describe('an agent line the Control Panel cannot handle', () => {
  it('stops the agent as a protocol error and leaves the Control Panel running', async () => {
    const fake = fakeService('deep-raw-input');
    try {
      const { conversation } = await fake.service.create({ mode: 'chat', text: 'hi' });
      const ends: string[] = [];
      const off = fake.service.watchConversation(conversation.id, (event) => {
        if (event.type === 'turn' && event.turn === null) ends.push(event.outcome ?? '');
      });
      await until(() => ends.length === 1, 10_000);
      off();
      expect(ends).toEqual(['agent-crashed']);
      const state = await fake.service.state();
      expect(state.modes.chat.failure?.code).toBe('agent-protocol-error');
    } finally {
      await fake.close();
    }
  }, 20_000);
});
