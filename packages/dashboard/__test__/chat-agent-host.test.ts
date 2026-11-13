/**
 * The agent host against the scripted fake agent: lifecycle, turns and their outcomes, both web call shapes, the
 * refusals that end a turn, crash and reopen, idle stop, stale restart, and a Work session.
 */

import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { AgentHostError, launchFromProbe } from '../src/chat/agent/agent-host.js';
import { CHAT_CONFIG_TOML, CHAT_REQUIREMENTS_TOML } from '../src/chat/agent/profiles.js';
import { REDACTED } from '../src/chat/redact.js';
import {
  endpointModel,
  fakeHost,
  INFERENCE_TOKEN,
  noProviders,
  until,
  type FakeHost,
  type Json,
} from './helpers/chat-fake.js';
import { gate } from './helpers/fake-engine.js';

const MODEL = 'qwen3.5-4b';

async function withFake(
  name: string,
  run: (fake: FakeHost) => Promise<void>,
  options: Parameters<typeof fakeHost>[1] = {},
) {
  const fake = await fakeHost(name, options);
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

/** Creates a Chat session and runs one turn to its end. */
async function turn(fake: FakeHost, text: string, sessionId?: string) {
  const id = sessionId ?? (await fake.host.createSession({ modelId: MODEL })).sessionId;
  const started = fake.host.startTurn(id, text, { modelId: MODEL });
  const result = await started.done;
  return { id, turnId: started.turnId, result };
}

function permissionAnswers(fake: FakeHost): unknown[] {
  return fake
    .sent()
    .filter(
      (m) =>
        m.method === undefined &&
        m.result !== undefined &&
        typeof m.result === 'object' &&
        'outcome' in (m.result as Json),
    )
    .map((m) => (m.result as Json).outcome);
}

function texts(fake: FakeHost): string {
  return fake.events.updates
    .map((u) => {
      const content = u.line.update.content as Json | undefined;
      return typeof content?.text === 'string' ? content.text : '';
    })
    .join('');
}

describe('launchFromProbe', () => {
  it('maps the probe to a launch or a typed failure', () => {
    expect(launchFromProbe({ source: 'none' })).toEqual({ failure: 'agent-missing' });
    expect(launchFromProbe({ source: 'none', managedTag: 'mlx-agent-v0.2.0' })).toEqual({ failure: 'agent-too-old' });
    expect(
      launchFromProbe({
        source: 'bundled',
        path: '/b',
        bundledTag: 'mlx-agent-v0.2.0',
        effectiveTag: 'mlx-agent-v0.2.0',
      }),
    ).toEqual({
      failure: 'agent-too-old',
    });
    expect(launchFromProbe({ source: 'managed', path: '/m', effectiveTag: 'mlx-agent-v0.3.3' })).toEqual({
      command: '/m',
      args: ['--disable-web-search', 'agent', '--no-leader', 'stdio'],
      source: 'managed',
      tag: 'mlx-agent-v0.3.3',
      versionVerified: true,
    });
    expect(launchFromProbe({ source: 'env', path: '/e' })).toMatchObject({
      command: '/e',
      versionVerified: false,
      tag: null,
    });
  });
});

describe('Chat host', () => {
  it('starts lazily with the Chat env and initialize params, and streams a turn', async () => {
    await withFake('chat-basic', async (fake) => {
      expect(fake.records()).toEqual([]);
      const { id, result } = await turn(fake, 'hello there');
      expect(result.outcome).toBe('completed');
      const start = fake.records().find((r) => r.kind === 'start')!;
      // macOS adds __CF_USER_TEXT_ENCODING to every process it starts.
      expect((start.envNames as string[]).filter((n) => n !== '__CF_USER_TEXT_ENCODING')).toEqual(
        [
          'DISABLE_TELEMETRY',
          'GROK_CLAUDE_HOOKS_ENABLED',
          'GROK_CLAUDE_MCPS_ENABLED',
          'GROK_CLAUDE_SKILLS_ENABLED',
          'GROK_CLI_CHAT_PROXY_BASE_URL',
          'GROK_CODEX_HOOKS_ENABLED',
          'GROK_CODEX_MCPS_ENABLED',
          'GROK_CONFIG',
          'GROK_CURSOR_HOOKS_ENABLED',
          'GROK_CURSOR_MCPS_ENABLED',
          'GROK_CURSOR_SKILLS_ENABLED',
          'GROK_DISABLE_AUTOUPDATER',
          'GROK_HOME',
          'GROK_IMAGE_EDIT',
          'GROK_IMAGE_GEN',
          'GROK_MODELS_BASE_URL',
          'GROK_SESSION_SEARCH',
          'GROK_TELEMETRY_ENABLED',
          'GROK_TELEMETRY_TRACE_UPLOAD',
          'GROK_TURN_SUMMARY',
          'GROK_VIDEO_GEN',
          'GROK_XAI_API_BASE_URL',
          'HOME',
          'LANG',
          'LOGNAME',
          'PATH',
          'SHELL',
          'TERM',
          'TMPDIR',
          'USER',
          'XAI_API_KEY',
        ].sort(),
      );
      expect(realpathSync(start.cwd as string)).toBe(realpathSync(fake.paths.chatWorkspace));
      expect(JSON.stringify(start.argv)).not.toContain(INFERENCE_TOKEN);
      const sent = fake.sent();
      expect(sent[0]).toMatchObject({ method: 'initialize', params: { _meta: { clientType: 'generic' } } });
      expect(sent.some((m) => m.method === 'authenticate')).toBe(false);
      const prompt = sent.find((m) => m.method === 'session/prompt')!;
      expect(prompt.params).toEqual({ sessionId: id, prompt: [{ type: 'text', text: 'hello there' }] });
      expect(texts(fake)).toContain('Hello! How can I help?');
      expect(fake.events.titles).toEqual([{ sessionId: id, title: 'hello there' }]);
      expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'model_changed')).toBe(false);
      expect(fake.events.turns.map((t) => t.turn?.phase ?? null)).toEqual(['starting', 'running', null]);
    });
  });

  it('writes the deny-rule files and removes the search index and prompt history before the spawn', async () => {
    await withFake('chat-basic', async (fake) => {
      const sessions = fake.paths.chatSessions;
      mkdirSync(join(sessions, 'some-cwd'), { recursive: true });
      for (const f of ['session_search.sqlite', 'session_search.sqlite-wal', 'session_search.sqlite-shm']) {
        writeFileSync(join(sessions, f), 'x');
      }
      writeFileSync(join(sessions, 'some-cwd', 'prompt_history.jsonl'), 'typed prompt\n');
      await fake.host.start();
      expect(readFileSync(fake.paths.chatConfig, 'utf8')).toBe(CHAT_CONFIG_TOML);
      expect(readFileSync(fake.paths.chatRequirements, 'utf8')).toBe(CHAT_REQUIREMENTS_TOML);
      for (const f of ['session_search.sqlite', 'session_search.sqlite-wal', 'session_search.sqlite-shm']) {
        expect(existsSync(join(sessions, f))).toBe(false);
      }
      expect(existsSync(join(sessions, 'some-cwd', 'prompt_history.jsonl'))).toBe(false);
      expect(existsSync(join(sessions, 'some-cwd'))).toBe(true);
    });
  });

  it('call shape A (use_tool) is approved and its result arrives as web data', async () => {
    await withFake('chat-web-search', async (fake) => {
      const { result } = await turn(fake, 'newest rust?');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'allow-once' }]);
      expect(fake.events.web.map((w) => w.result.kind)).toEqual(['search']);
      expect(texts(fake)).toContain('Rust 1.99 is the newest stable release');
    });
  });

  it('call shape B (direct) is approved, and a fetch of a search result address goes on', async () => {
    await withFake('chat-web-search-direct', async (fake) => {
      const { id, result } = await turn(fake, 'newest rust?');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual([
        { outcome: 'selected', optionId: 'allow-once' },
        { outcome: 'selected', optionId: 'allow-once' },
      ]);
      expect(fake.events.web.map((w) => w.result.kind)).toEqual(['search', 'page']);
      expect(fake.host.ledger(id)?.has('https://releases.rs')).toBe(true);
    });
  });

  it('chat-wrong-tool-name: the wrong guess is approved and the turn goes on to an answer', async () => {
    await withFake('chat-wrong-tool-name', async (fake) => {
      const { result } = await turn(fake, 'search please');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toHaveLength(2);
      expect(texts(fake)).toContain('Rust 1.99');
    });
  });

  it('chat-shell-attempt: rejected with reject-once, the turn ends tool-refused and nothing after it runs', async () => {
    await withFake('chat-shell-attempt', async (fake) => {
      const { result } = await turn(fake, 'run a command');
      expect(result.outcome).toBe('tool-refused');
      expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'reject-once' }]);
      expect(texts(fake)).not.toContain('never sent');
    });
  });

  it('chat-fetch-unlisted-url: rejected, the turn ends address-refused', async () => {
    await withFake('chat-fetch-unlisted-url', async (fake) => {
      const { result } = await turn(fake, 'what is new in rust');
      expect(result.outcome).toBe('address-refused');
      expect(permissionAnswers(fake)).toEqual([
        { outcome: 'selected', optionId: 'allow-once' },
        { outcome: 'selected', optionId: 'reject-once' },
      ]);
      expect(texts(fake)).not.toContain('never sent');
    });
  });

  it('chat-web-bad-input: an input text no reader can make an object of is approved; the tool error goes to the model', async () => {
    await withFake('chat-web-bad-input', async (fake) => {
      const { id, result } = await turn(fake, 'cut search');
      expect(result.outcome).toBe('completed');
      expect((await turn(fake, 'direct cut', id)).result.outcome).toBe('completed');
      expect((await turn(fake, 'cut read', id)).result.outcome).toBe('completed');
      expect((await turn(fake, 'python read', id)).result.outcome).toBe('completed');
      expect((await turn(fake, 'quoted search', id)).result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual(Array(5).fill({ outcome: 'selected', optionId: 'allow-once' }));
      for (const after of [
        'FCHK-AFTER-CUT-SEARCH',
        'FCHK-AFTER-DIRECT-CUT',
        'FCHK-AFTER-CUT-READ',
        'FCHK-AFTER-PYTHON-READ',
        'FCHK-AFTER-QUOTED-SEARCH',
      ])
        expect(texts(fake)).toContain(after);
    });
  });

  it('chat-web-bad-input: an input text that could still become an object the app cannot read ends the turn arguments-unreadable', async () => {
    await withFake('chat-web-bad-input', async (fake) => {
      const { id, result } = await turn(fake, 'duplicate url');
      expect(result.outcome).toBe('arguments-unreadable');
      expect((await turn(fake, 'smuggled tool', id)).result.outcome).toBe('arguments-unreadable');
      expect(permissionAnswers(fake)).toEqual(Array(2).fill({ outcome: 'selected', optionId: 'reject-once' }));
      expect(texts(fake)).not.toContain('never sent');
    });
  });

  it('chat-web-bad-input: a clean input text is checked against the ledger with the address the tool receives', async () => {
    await withFake('chat-web-bad-input', async (fake) => {
      const { id, result } = await turn(fake, 'clean read stranger');
      expect(result.outcome).toBe('address-refused');
      const listed = await turn(fake, 'clean read listed https://example.com/docs/', id);
      expect(listed.result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual([
        { outcome: 'selected', optionId: 'reject-once' },
        { outcome: 'selected', optionId: 'allow-once' },
      ]);
      expect(texts(fake)).toContain('FCHK-AFTER-LISTED');
    });
  });

  it('chat-fetch-listed-url: an address from the user message and one from a search result are fetched', async () => {
    await withFake('chat-fetch-listed-url', async (fake) => {
      const { result } = await turn(fake, 'Read https://example.com/docs/ and the rust releases.');
      expect(result.outcome).toBe('completed');
      expect(permissionAnswers(fake)).toEqual(Array(3).fill({ outcome: 'selected', optionId: 'allow-once' }));
    });
  });

  it('chat-use-tool-file-input: the tool_input_file and file forms and an extra key end the turn tool-refused', async () => {
    await withFake('chat-use-tool-file-input', async (fake) => {
      const { id, result } = await turn(fake, 'file form');
      expect(result.outcome).toBe('tool-refused');
      expect((await turn(fake, 'second form', id)).result.outcome).toBe('tool-refused');
      expect((await turn(fake, 'extra key', id)).result.outcome).toBe('tool-refused');
      expect(permissionAnswers(fake)).toEqual(Array(3).fill({ outcome: 'selected', optionId: 'reject-once' }));
    });
  });

  it('chat-enable-always-approve: the offered option is never chosen; the turn ends tool-refused', async () => {
    await withFake('chat-enable-always-approve', async (fake) => {
      const { result } = await turn(fake, 'search');
      expect(result.outcome).toBe('tool-refused');
      expect(permissionAnswers(fake)).toEqual([{ outcome: 'selected', optionId: 'reject-once' }]);
    });
  });

  it('chat-tool-loop: the ninth request is cancelled and the turn ends tool-loop', async () => {
    await withFake('chat-tool-loop', async (fake) => {
      const { id, result } = await turn(fake, 'loop');
      expect(result.outcome).toBe('tool-loop');
      const answers = permissionAnswers(fake);
      expect(answers).toHaveLength(9);
      expect(answers[8]).toEqual({ outcome: 'cancelled' });
      expect(fake.sent().some((m) => m.method === 'session/cancel' && (m.params as Json).sessionId === id)).toBe(true);
    });
  });

  it('cancel: the user stop ends the turn as cancelled', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'think long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'));
      expect(fake.host.cancel(sessionId)).toBe(true);
      expect((await started.done).outcome).toBe('cancelled');
      expect(fake.host.cancel(sessionId)).toBe(false);
    });
  });

  it('a second prompt while a turn runs is refused as conversation-busy', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'one', { modelId: MODEL });
      expect(() => fake.host.startTurn(sessionId, 'two', { modelId: MODEL })).toThrow(AgentHostError);
      fake.host.cancel(sessionId);
      await started.done;
    });
  });

  it('max-tokens ends as output-limit; an error text never reaches the result', async () => {
    await withFake('max-tokens', async (fake) => {
      expect((await turn(fake, 'long')).result.outcome).toBe('output-limit');
    });
    await withFake('error-with-marker', async (fake) => {
      const { result } = await turn(fake, 'boom');
      expect(result.outcome).toBe('model-error');
      expect(JSON.stringify(result)).not.toContain('marker-KEY');
      expect(JSON.stringify(fake.events)).not.toContain('marker-KEY');
    });
  });

  it('survives a crash: the turn ends agent-crashed, the session goes cold and reopens with a load', async () => {
    await withFake('crash-mid-turn', async (fake) => {
      const { id, result } = await turn(fake, 'crash now');
      expect(result).toMatchObject({ outcome: 'agent-crashed', failure: { code: 'agent-crashed' } });
      expect(fake.host.modeState()).toMatchObject({ state: 'crashed', failure: { code: 'agent-crashed' } });
      expect(fake.events.cold).toEqual([[id]]);
      expect(fake.host.isHot(id)).toBe(false);
    });
  });

  it('reopens a cold conversation: session/load resends everything with noReplay, the guard runs, no model entry', async () => {
    await withFake(
      'new-and-load-model-changed',
      async (fake) => {
        const { id } = await turn(fake, 'first');
        await until(() => fake.events.cold.length > 0);
        expect(fake.host.modeState().state).toBe('stopped');
        const second = fake.host.startTurn(id, 'second https://example.org/x', { modelId: MODEL });
        expect((await second.done).outcome).toBe('completed');
        const sent = fake.sent();
        const load = sent.find((m) => m.method === 'session/load')!;
        expect(load.params).toMatchObject({
          sessionId: id,
          mcpServers: [{ name: 'mlx_web' }],
          _meta: { noReplay: true, agentProfile: { name: 'mlx-chat' }, yoloMode: false },
        });
        expect(sent.filter((m) => m.method === '_x.ai/session/info')).toHaveLength(2);
        expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'model_changed')).toBe(false);
        expect(fake.host.ledger(id)?.has('https://example.org/x')).toBe(true);
      },
      { timings: { idleStopMs: 50 } },
    );
  });

  it('seeds the ledger of a cold conversation from its disk entries', async () => {
    await withFake('chat-fetch-listed-url', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      await fake.host.restart();
      const started = fake.host.startTurn(sessionId, 'again', {
        modelId: MODEL,
        entries: [{ kind: 'user', ts: 1, text: 'see https://example.com/docs', promptIndex: 0 }],
      });
      expect((await started.done).outcome).toBe('completed');
    });
  });

  it('renames and deletes a hot conversation through the agent; the session folder goes away', async () => {
    await withFake('chat-basic', async (fake) => {
      const { id } = await turn(fake, 'to be renamed');
      await fake.host.rename(id, 'New title');
      const cwd = realpathSync(fake.paths.chatWorkspace);
      const dir = join(fake.paths.chatSessions, encodeURIComponent(cwd), id);
      expect(JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8')).generated_title).toBe('New title');
      await fake.host.deleteSession(id, cwd);
      expect(existsSync(dir)).toBe(false);
      expect(fake.sent().find((m) => m.method === '_x.ai/session/delete')?.params).toEqual({ sessionId: id, cwd });
    });
  });

  it('switches the model between turns and confirms it with session info', async () => {
    await withFake('chat-basic', async (fake) => {
      const { id } = await turn(fake, 'hi');
      await fake.host.setModel(id, 'qwen3.5-9b');
      expect(fake.host.sessionModel(id)).toBe('qwen3.5-9b');
      await expect(fake.host.setModel(id, 'nope')).rejects.toMatchObject({ code: 'model-unavailable' });
    });
  });

  it('passes a real model switch during a turn and drops one to the current model', async () => {
    await withFake('model-switch-in-turn', async (fake) => {
      const { id } = await turn(fake, 'switch');
      const switches = fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'model_changed');
      expect(switches.map((u) => u.line.update.model_id)).toEqual(['qwen3.5-9b']);
      expect(fake.events.models).toEqual([{ sessionId: id, modelId: 'qwen3.5-9b' }]);
    });
  });

  it('stops after the idle time and marks its sessions cold', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { id } = await turn(fake, 'hi');
        await until(() => fake.host.modeState().state === 'stopped');
        expect(fake.events.cold).toEqual([[id]]);
      },
      { timings: { idleStopMs: 50 } },
    );
  });

  it('a stale host finishes the running turn on the old process, then restarts', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'));
      fake.host.markStale();
      expect(fake.host.modeState()).toMatchObject({ stale: true, state: 'ready' });
      fake.host.cancel(sessionId);
      await started.done;
      await until(() => fake.host.modeState().state === 'stopped');
      expect(fake.host.modeState().stale).toBe(false);
    });
  });

  it('a missing endpoint file during a turn cancels it as inference-lost', async () => {
    await withFake('cancel', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const started = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
      await until(() => texts(fake).includes('Thinking'));
      fake.endpoint.current = null;
      await fake.host.checkEndpoint();
      expect((await started.done).outcome).toBe('inference-lost');
    });
  });

  it('no endpoint file: the start fails as inference-stopped and nothing is spawned', async () => {
    await withFake('chat-basic', async (fake) => {
      fake.endpoint.current = null;
      await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({ code: 'inference-stopped' });
      expect(fake.records()).toEqual([]);
    });
  });

  it('a prompt reply that does not follow a cancel inside the grace time is a protocol error', async () => {
    await withFake(
      'hang-on-cancel',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'x', { modelId: MODEL });
        await until(() => texts(fake).includes('x'));
        fake.host.cancel(sessionId);
        expect((await started.done).outcome).toBe('agent-crashed');
        expect(fake.host.modeState()).toMatchObject({ state: 'crashed', failure: { code: 'agent-protocol-error' } });
      },
      { timings: { cancelReplyMs: 100 } },
    );
  });

  it('a few garbage lines are dropped; a flood of them is a protocol error', async () => {
    await withFake('garbage-stdout', async (fake) => {
      expect((await turn(fake, 'hi')).result.outcome).toBe('completed');
      expect(JSON.stringify(fake.events)).not.toContain('marker-SECRET');
    });
    await withFake('garbage-flood', async (fake) => {
      await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({ code: 'agent-protocol-error' });
    });
  });

  it('a flood stays a protocol error when the agent answers before its late exit', async () => {
    await withFake(
      'garbage-flood',
      async (fake) => {
        await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({
          code: 'agent-protocol-error',
        });
        expect(fake.sent().map((m) => m.method)).not.toContain('session/new');
      },
      {
        mutate: (scenario) => {
          (scenario.faults as Json).exitAfterTermMs = 150;
        },
      },
    );
  });

  it('three crashes inside the window fail the mode until restart', async () => {
    await withFake('crash-mid-turn', async (fake) => {
      for (let i = 0; i < 3; i++) await turn(fake, `crash ${i}`);
      expect(fake.host.modeState()).toMatchObject({ state: 'failed', failure: { code: 'agent-failed' } });
      await expect(fake.host.createSession({ modelId: MODEL })).rejects.toMatchObject({ code: 'agent-failed' });
      await fake.host.restart();
      expect(fake.host.modeState()).toMatchObject({ state: 'stopped', failure: null });
      await fake.host.createSession({ modelId: MODEL });
    });
  });

  it('close cancels the running turn as interrupted and stops the process', async () => {
    const fake = await fakeHost('cancel');
    const { sessionId } = await fake.host.createSession({ modelId: MODEL });
    const started = fake.host.startTurn(sessionId, 'long', { modelId: MODEL });
    await until(() => texts(fake).includes('Thinking'));
    await fake.close();
    expect((await started.done).outcome).toBe('interrupted');
  });
});

describe('Chat host: the model select call before the prompt', () => {
  const prompts = (fake: FakeHost): Json[] => fake.sent().filter((m) => m.method === 'session/prompt');
  const starts = (fake: FakeHost): number => fake.records().filter((r) => r.kind === 'start').length;
  const waits = (fake: FakeHost): Array<string | null> =>
    fake.events.turns.flatMap((t) => (t.turn === null ? [] : [`${t.turn.phase}:${t.turn.wait ?? '-'}`]));

  it('the select call ends before session/prompt is written; a resident model shows no wait', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      expect(fake.engine.selects()).toHaveLength(0);
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result).toEqual({ turnId: expect.any(String), outcome: 'completed' });
      expect(fake.engine.selects()).toHaveLength(1);
      expect(JSON.parse(fake.engine.selects()[0]!.body)).toMatchObject({ model: MODEL });
      expect(waits(fake)).toEqual(['starting:-', 'running:-']);
    });
  });

  it('beside a held answer on the same model the select returns, the prompt goes out, and the turn runs with no output', async () => {
    const held = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        const answer = fetch(`${fake.engine.url}/v1/messages`, { method: 'POST', body: '{}' });
        await until(() => fake.engine.requests.some((r) => r.path === '/v1/messages'));
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
        await until(() => prompts(fake).length === 1);
        expect(fake.engine.selects()).toHaveLength(1);
        expect(fake.engine.selects()[0]!.answered).toBe(true);
        expect(fake.host.turnState(sessionId)).toMatchObject({ phase: 'running', wait: null });
        expect(waits(fake)).toEqual(['starting:-', 'running:-']);
        expect(texts(fake)).toBe('');
        expect(fake.host.cancel(sessionId)).toBe(true);
        expect((await started.done).outcome).toBe('cancelled');
        held.open();
        expect((await answer).status).toBe(200);
      },
      {
        timings: { modelWaitShowMs: 20 },
        mutate: (scenario) => {
          (scenario.prompts as Json[])[0]!.steps = [{ sleepMs: 60_000 }, { end: { stopReason: 'end_turn' } }];
        },
        engine: (r) => (r.path === '/v1/messages' ? { until: held.promise, json: {} } : { json: { input_tokens: 1 } }),
      },
    );
  });

  it('a held load shows wait model, keeps the prompt back, and clears the wait before running', async () => {
    const hold = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
        await until(() => fake.host.turnState(sessionId)?.wait === 'model');
        expect(fake.host.turnState(sessionId)?.phase).toBe('starting');
        expect(prompts(fake)).toHaveLength(0);
        hold.open();
        expect((await started.done).outcome).toBe('completed');
        expect(prompts(fake)).toHaveLength(1);
        expect(waits(fake)).toEqual(['starting:-', 'starting:model', 'starting:-', 'running:-']);
      },
      {
        timings: { modelWaitShowMs: 20 },
        engine: (r) => (r.path.endsWith('count_tokens') ? { until: hold.promise, json: {} } : { json: {} }),
      },
    );
  });

  it('Stop during the load ends the turn cancelled and unsent: no session/cancel, no kill, the socket closes', async () => {
    const hold = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
        await until(() => fake.engine.selects().length === 1);
        expect(fake.host.cancel(sessionId)).toBe(true);
        expect(await started.done).toEqual({ turnId: started.turnId, outcome: 'cancelled', unsent: true });
        await until(() => fake.engine.selects()[0]!.closedEarly);
        hold.open();
        fake.engine.script(() => ({ json: {} }));
        expect((await fake.host.startTurn(sessionId, 'again', { modelId: MODEL }).done).outcome).toBe('completed');
        const methods = fake.sent().map((m) => m.method);
        expect(methods).not.toContain('session/cancel');
        expect(prompts(fake)).toHaveLength(1);
        expect(starts(fake)).toBe(1);
      },
      {
        timings: { cancelReplyMs: 1 },
        engine: (r) => (r.path.endsWith('count_tokens') ? { until: hold.promise, json: {} } : { json: {} }),
      },
    );
  });

  it('a failed load ends the turn model-load-failed with no prompt', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result).toEqual({
          turnId: expect.any(String),
          failure: { code: 'model-load-failed', message: 'The model did not load.' },
          unsent: true,
        });
        expect(prompts(fake)).toHaveLength(0);
      },
      { engine: () => ({ status: 500, json: {} }) },
    );
  });

  it('two conversations on two models: a model another select swapped out after its load still gets the prompt', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const one = await fake.host.createSession({ modelId: 'qwen3.5-4b' });
        const two = await fake.host.createSession({ modelId: 'qwen3.5-9b' });
        const results = await Promise.all([
          fake.host.startTurn(one.sessionId, 'one', { modelId: 'qwen3.5-4b' }).done,
          fake.host.startTurn(two.sessionId, 'two', { modelId: 'qwen3.5-9b' }).done,
        ]);
        expect(results.map((r) => r.outcome)).toEqual(['completed', 'completed']);
        expect(fake.engine.selects()).toHaveLength(2);
        expect(prompts(fake)).toHaveLength(2);
      },
      {
        endpointModels: [
          endpointModel('qwen3.5-4b', '/models/qwen3.5-4b'),
          endpointModel('qwen3.5-9b', '/models/qwen3.5-9b'),
        ],
        engine: () => ({ status: 400, json: {} }),
      },
    );
  });

  it('the agent exiting during the load ends the turn at once, agent-crashed and unsent', async () => {
    const hold = gate();
    await withFake(
      'chat-basic',
      async (fake) => {
        try {
          const { sessionId } = await fake.host.createSession({ modelId: MODEL });
          const started = fake.host.startTurn(sessionId, 'hi', { modelId: MODEL });
          await until(() => fake.engine.selects().length === 1);
          const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
          let ended: unknown = null;
          void started.done.then((result) => (ended = result));
          process.kill(pid, 'SIGKILL');
          await until(() => ended !== null, 5_000);
          expect(ended).toEqual({
            turnId: started.turnId,
            failure: { code: 'agent-crashed', message: expect.any(String) },
            unsent: true,
          });
          await until(() => fake.engine.selects()[0]!.closedEarly);
          expect(prompts(fake)).toHaveLength(0);
        } finally {
          hold.open();
        }
      },
      { engine: (r) => (r.path.endsWith('count_tokens') ? { until: hold.promise, json: {} } : { json: {} }) },
    );
  });

  it('Stop while the turn reads the endpoint sends no select call', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const endpoint = fake.endpoint.current;
      let stopOnRead = true;
      Object.defineProperty(fake.endpoint, 'current', {
        configurable: true,
        get: () => {
          if (stopOnRead && fake.host.turnState(sessionId) !== null) {
            stopOnRead = false;
            fake.host.cancel(sessionId);
          }
          return endpoint;
        },
      });
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result).toMatchObject({ outcome: 'cancelled', unsent: true });
      expect(stopOnRead).toBe(false);
      expect((await fake.host.startTurn(sessionId, 'again', { modelId: MODEL }).done).outcome).toBe('completed');
      expect(fake.engine.selects()).toHaveLength(1);
    });
  });

  it('a name the endpoint file does not list is model-unavailable with no engine request', async () => {
    await withFake('chat-basic', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      fake.endpoint.current = { ...fake.endpoint.current!, models: [endpointModel('other', '/models/other')] };
      const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
      expect(result.failure?.code).toBe('model-unavailable');
      expect(fake.engine.requests).toHaveLength(0);
      expect(prompts(fake)).toHaveLength(0);
    });
  });

  it('a name the engine does not serve is model-unavailable', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result.failure?.code).toBe('model-unavailable');
      },
      { engine: () => ({ status: 404, json: {} }) },
    );
  });

  it('a refused token restarts the agent once, then fails inference-failed', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const result = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(result.failure?.code).toBe('inference-failed');
        expect(fake.engine.selects()).toHaveLength(2);
        expect(starts(fake)).toBe(2);
        expect(prompts(fake)).toHaveLength(0);
      },
      { engine: () => ({ status: 401, json: {} }) },
    );
  });

  it('an engine that is gone is inference-failed, or inference-stopped when its file went away', async () => {
    await withFake(
      'chat-basic',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL });
        const first = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(first.failure?.code).toBe('inference-failed');
        const endpoint = fake.endpoint.current;
        Object.defineProperty(fake.endpoint, 'current', {
          configurable: true,
          get: () => (fake.engine.selects().length > 1 ? null : endpoint),
        });
        const second = await fake.host.startTurn(sessionId, 'hi', { modelId: MODEL }).done;
        expect(second.failure?.code).toBe('inference-stopped');
      },
      { engine: () => ({ drop: true }) },
    );
  });
});

describe('Work host', () => {
  const work = { mode: 'work' as const, providers: noProviders() };

  it('runs a Work session in the chosen folder with the parent env and a temp agent home', async () => {
    await withFake(
      'work-permission',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        expect(created).toMatchObject({ agentName: 'grok-build-plan', toolCount: 20, cwd: realpathSync(fake.project) });
        const newCall = fake.sent().find((m) => m.method === 'session/new')!;
        expect(newCall.params).toMatchObject({
          cwd: realpathSync(fake.project),
          mcpServers: [],
          _meta: { yoloMode: false, autoMode: false },
        });
        const start = fake.records().find((r) => r.kind === 'start')!;
        expect(start.envNames).toEqual(
          expect.arrayContaining(['GROK_HOME', 'HOME', 'PATH', 'XAI_API_KEY', 'GROK_RESUME_COMMAND']),
        );
        expect(start.envNames).not.toContain('ELECTRON_RUN_AS_NODE');
        const started = fake.host.startTurn(created.sessionId, 'list files', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        const { interaction } = fake.events.interactions[0]!;
        expect(interaction).toMatchObject({ kind: 'permission', access: 'bash', toolName: 'run_terminal_command' });
        expect(interaction.kind === 'permission' && interaction.options.map((o) => o.optionId)).toEqual([
          'always-allow',
          'allow-once',
          'reject-once',
        ]);
        expect(() => fake.host.answer(created.sessionId, interaction.id, 'reject-always')).toThrow();
        fake.host.answer(created.sessionId, interaction.id, 'allow-once');
        expect((await started.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: created.sessionId, id: interaction.id, reason: 'answered' }]);
      },
      work,
    );
  });

  it('Deny ends the turn as denied, never cancelled; enable-always-approve is never offered to the page', async () => {
    await withFake(
      'work-deny',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const started = fake.host.startTurn(created.sessionId, 'clean', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        const { interaction } = fake.events.interactions[0]!;
        expect(JSON.stringify(interaction)).not.toContain('enable-always-approve');
        fake.host.answer(created.sessionId, interaction.id, 'reject-once');
        expect((await started.done).outcome).toBe('denied');
      },
      work,
    );
  });

  it('a Work delete starts a stopped host and goes through the agent', async () => {
    await withFake(
      'work-plan',
      async (fake) => {
        await fake.host.deleteSession('01a1cb42-0000-7000-8000-000000000001', fake.project);
        const del = fake.sent().find((m) => m.method === '_x.ai/session/delete');
        expect(del?.params).toEqual({ sessionId: '01a1cb42-0000-7000-8000-000000000001', cwd: fake.project });
      },
      work,
    );
  });

  it('a provider key in a Work permission request is redacted before it reaches the page', async () => {
    const KEY = 'exa-KEY-feedfacecafebeef';
    const oneKey = noProviders({ workKeys: () => ({ EXA_API_KEY: KEY }), secretNeedles: () => [KEY] });
    await withFake(
      'work-permission',
      async (fake) => {
        const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        expect(fake.records().find((r) => r.kind === 'start')?.envNames).toContain('EXA_API_KEY');
        expect(JSON.stringify(fake.records().find((r) => r.kind === 'start'))).not.toContain(KEY);
        const started = fake.host.startTurn(created.sessionId, 'curl', { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length === 1);
        const { interaction } = fake.events.interactions[0]!;
        expect(JSON.stringify(interaction)).not.toContain(KEY);
        expect(JSON.stringify(interaction)).toContain(REDACTED);
        fake.host.answer(created.sessionId, interaction.id, 'reject-once');
        expect((await started.done).outcome).toBe('denied');
      },
      {
        mode: 'work',
        providers: oneKey,
        mutate: (s) => {
          const step = ((s.prompts as Json[])[0]!.steps as Json[])[0]!.permission as Json;
          step.title = `Execute \`curl -H x-api-key:${KEY}\``;
          step.rawInput = {
            variant: 'Bash',
            command: `curl -H x-api-key:${KEY} https://api.exa.ai`,
            description: 'call',
            is_background: false,
          };
        },
      },
    );
  });

  it('a Work folder that does not exist is cwd-invalid', async () => {
    await withFake(
      'work-plan',
      async (fake) => {
        await expect(
          fake.host.createSession({ modelId: MODEL, cwd: join(fake.root, 'missing') }),
        ).rejects.toMatchObject({
          code: 'cwd-invalid',
        });
      },
      work,
    );
  });
});
