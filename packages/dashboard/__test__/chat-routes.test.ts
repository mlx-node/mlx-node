/**
 * The Chat routes through the real runtime (worker thread included) on the scripted fake agent: the thread of each
 * route, the exact field lists (unknown fields refused), slash text that never reaches the agent, the size and id
 * rules, interactions refused for Chat, the error text law, and the `chat` stream of the runtime.
 */

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { MAX_FIELD_CHARS } from '../src/agent-sessions/transcript.js';
import type { SessionDetailResponse } from '../src/agent-sessions/types.js';
import { routeThreadFor } from '../src/api/dispatch.js';
import type { ApiResponse } from '../src/api/errors.js';
import { ROUTES } from '../src/api/routes.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import { guarded } from '../src/chat/route-errors.js';
import { isSlashText, permissionModeInput } from '../src/chat/route-input.js';
import type { ConversationEvent, ConversationInfo, Interaction } from '../src/chat/types.js';
import { createDashboardRuntime, type ChatRuntimeOptions, type DashboardRuntime } from '../src/runtime.js';
import { FAKE_AGENT, INFERENCE_TOKEN, readScenario, until, type Json } from './helpers/chat-fake.js';
import { startFakeEngine, UNLOAD_PATH, type FakeEngine } from './helpers/fake-engine.js';

const MARKER = 'marker-KEY-12345678';

let base: string;
let engine: FakeEngine;
let runtime: DashboardRuntime;
let recordTo: string;

function start(
  scenarioName: string,
  mutate?: (scenario: Json) => void,
  chatExtra: Partial<ChatRuntimeOptions> = {},
): void {
  recordTo = join(base, 'record.jsonl');
  const scenario = { ...readScenario(scenarioName), recordTo };
  mutate?.(scenario);
  const scenarioPath = join(base, `${scenarioName}.json`);
  writeFileSync(scenarioPath, JSON.stringify(scenario));
  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    tracesDir: join(base, 'traces'),
    modelsDir: join(base, 'models'),
    runtimeDir: join(base, 'runtime'),
    supervisorPath: join(base, 'supervisor.json'),
    agentHome: join(base, 'agent-home'),
    chat: {
      root: join(base, 'chat'),
      homeDir: join(base, 'home'),
      agentCommand: { command: process.execPath, args: [FAKE_AGENT, scenarioPath] },
      secrets: new MemorySecretStore(),
      readEndpoint: async () => ({
        version: 1,
        pid: process.pid,
        url: engine.url,
        token: INFERENCE_TOKEN,
        models: [{ name: 'qwen3.5-4b', path: '/models/qwen3.5-4b' }],
      }),
      timings: { stopEofMs: 300, stopTermMs: 300 },
      flushMs: 5,
      stateCoalesceMs: 5,
      ...chatExtra,
    },
  });
}

function sent(): Json[] {
  try {
    return readFileSync(recordTo, 'utf8')
      .split('\n')
      .filter((l) => l !== '')
      .map((l) => JSON.parse(l) as Json)
      .filter((r) => r.kind === 'in')
      .map((r) => r.msg as Json);
  } catch {
    return [];
  }
}

const call = (method: string, path: string, body?: unknown): Promise<ApiResponse> =>
  runtime.call({ method, path, ...(body === undefined ? {} : { body }) });

function failureOf(response: ApiResponse): { code: string; reason?: string; message: string } {
  if (response.ok) throw new Error(`expected a failure, got ${JSON.stringify(response.body)}`);
  return { code: response.code, message: response.message, ...(response.reason ? { reason: response.reason } : {}) };
}

async function created(text = 'hello there'): Promise<ConversationInfo> {
  const response = await call('POST', '/api/chat/conversations', { mode: 'chat', text });
  if (!response.ok) throw new Error(response.message);
  expect(response.status).toBe(201);
  return (response.body as { conversation: ConversationInfo }).conversation;
}

interface Watch {
  events: ConversationEvent[];
  /** Resolves once a turn of this stream ended, or the first `sync` shows no turn running. */
  idle(): Promise<void>;
  /** Resolves with the next `turn` event that ends a turn. */
  ended(): Promise<ConversationEvent>;
  off(): void;
}

function watch(id: string): Watch {
  const events: ConversationEvent[] = [];
  const off = runtime.watchChat({ scope: 'conversation', conversationId: id }, (e) =>
    events.push(e as ConversationEvent),
  );
  const endEvent = (): ConversationEvent | undefined => events.find((e) => e.type === 'turn' && e.turn === null);
  return {
    events,
    off,
    idle: () =>
      until(() => endEvent() !== undefined || events.some((e) => e.type === 'sync' && e.turn === null), 5_000),
    ended: async () => {
      await until(() => endEvent() !== undefined, 5_000);
      return endEvent()!;
    },
  };
}

/** Waits until the conversation runs no turn. */
async function settled(id: string): Promise<void> {
  const w = watch(id);
  try {
    await w.idle();
  } finally {
    w.off();
  }
}

beforeEach(async () => {
  base = mkdtempSync(join(tmpdir(), 'dash-chat-routes-'));
  engine = await startFakeEngine();
});

afterEach(async () => {
  await runtime?.close();
  await engine.close();
  rmSync(base, { recursive: true, force: true });
});

describe('Chat routes — threads', () => {
  it('runs the list on the worker and everything else with the transport', () => {
    const chat = ROUTES.filter((r) => r.segments[1] === 'chat').map(
      (r) => `${r.thread} ${r.method} /${r.segments.join('/')}`,
    );
    expect(chat).toEqual([
      'main GET /api/chat/state',
      'worker GET /api/chat/conversations',
      'main POST /api/chat/conversations',
      'main POST /api/chat/conversations/:id/prompt',
      'main POST /api/chat/conversations/:id/cancel',
      'main POST /api/chat/conversations/:id/interactions/:iid',
      'main POST /api/chat/conversations/:id/permission-mode',
      'main PATCH /api/chat/conversations/:id',
      'main DELETE /api/chat/conversations/:id',
      'main POST /api/chat/agent',
      'main POST /api/chat/engine/unload',
    ]);
    expect(routeThreadFor('POST', '/api/chat/conversations/abc/prompt')).toBe('main');
  });
});

describe('Chat routes — a conversation', () => {
  it('creates with 201, answers the prompt with 202, lists it, streams it, renames and deletes it', async () => {
    start('chat-basic');
    const conversation = await created();
    expect(conversation).toMatchObject({ mode: 'chat', hot: true, cwd: null });
    await settled(conversation.id);

    const w = watch(conversation.id);
    const prompt = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, { text: 'again' });
    expect(prompt.ok && prompt.status).toBe(202);
    expect(await w.ended()).toMatchObject({ outcome: 'completed' });
    expect(w.events[0]!.type).toBe('sync');
    w.off();

    const list = await call('GET', '/api/chat/conversations?mode=chat');
    expect(list.ok).toBe(true);
    const rows = (list as { body: { conversations: Array<{ id: string; mode: string }> } }).body.conversations;
    expect(rows.map((r) => [r.id, r.mode])).toEqual([[conversation.id, 'chat']]);
    const work = await call('GET', '/api/chat/conversations?mode=work');
    expect(work.ok && work.body).toEqual({ conversations: [] });

    const renamed = await call('PATCH', `/api/chat/conversations/${conversation.id}`, { title: 'Named' });
    expect(renamed.ok && (renamed.body as ConversationInfo).title).toBe('Named');
    const deleted = await call('DELETE', `/api/chat/conversations/${conversation.id}`);
    expect(deleted.ok && deleted.body).toEqual({});
    const after = await call('POST', `/api/chat/conversations/${conversation.id}/cancel`);
    expect(failureOf(after).code).toBe('E_NOT_FOUND');
  }, 20_000);

  it('serves the host state and the agent actions', async () => {
    start('chat-basic');
    const state = await call('GET', '/api/chat/state');
    expect(state.ok && state.body).toMatchObject({ inference: 'ready', defaultModelId: 'qwen3.5-4b' });
    const restart = await call('POST', '/api/chat/agent', { action: 'restart', mode: 'chat' });
    expect(restart.ok).toBe(true);
    const install = await call('POST', '/api/chat/agent', { action: 'install' });
    expect(install.ok && (install.body as { agent: { install: string } }).agent.install).toBe('failed');
    const hostEvents: unknown[] = [];
    const off = runtime.watchChat({ scope: 'host' }, (e) => hostEvents.push(e));
    await until(() => hostEvents.length > 0);
    expect(hostEvents[0]).toMatchObject({ type: 'state' });
    off();
  });
});

describe('Chat routes — engine unload', () => {
  const unloads = (): number => engine.requests.filter((r) => r.path === UNLOAD_PATH).length;

  it('unloads through the engine with the token in the header only, and names the model', async () => {
    start('chat-basic');
    engine.script((r) =>
      r.path === UNLOAD_PATH ? { json: { unloaded: ['qwen3.5-4b', 'mlx-local', 7] } } : { json: {} },
    );
    const response = await call('POST', '/api/chat/engine/unload');
    expect(response.ok && response.body).toEqual({ unloaded: ['qwen3.5-4b'] });
    const request = engine.requests.find((r) => r.path === UNLOAD_PATH)!;
    expect(request.headers['x-api-key']).toBe(INFERENCE_TOKEN);
    expect(request.body).toBe('');
    expect(failureOf(await call('POST', '/api/chat/engine/unload', { x: 1 })).code).toBe('E_BAD_REQUEST');
  });

  it('refuses while a turn runs, with no engine request', async () => {
    start('chat-basic', (s) => {
      s.prompts = [
        {
          match: 'slow',
          steps: [{ text: 'Thinking', chunk: 3 }, { sleepMs: 5000 }, { end: { stopReason: 'end_turn' } }],
        },
        ...(s.prompts as Json[]),
      ];
    });
    const conversation = await created('slow');
    const failure = failureOf(await call('POST', '/api/chat/engine/unload'));
    expect(failure).toMatchObject({ code: 'E_CONFLICT', reason: 'engine-busy' });
    expect(unloads()).toBe(0);
    await call('POST', `/api/chat/conversations/${conversation.id}/cancel`);
  });

  it('an engine that is busy is a 409; an engine that fails is inference-failed', async () => {
    start('chat-basic');
    engine.script(() => ({ status: 409, json: {} }));
    expect(failureOf(await call('POST', '/api/chat/engine/unload'))).toMatchObject({
      code: 'E_CONFLICT',
      reason: 'engine-busy',
    });
    engine.script(() => ({ status: 500, json: {} }));
    const failed = failureOf(await call('POST', '/api/chat/engine/unload'));
    expect(failed).toMatchObject({ code: 'E_UNAVAILABLE', reason: 'inference-failed' });
    expect(JSON.stringify(failed)).not.toContain(INFERENCE_TOKEN);
    expect(unloads()).toBe(2);
  });
});

describe('Chat routes — the input law', () => {
  it('refuses unknown fields, wrong types and fields the page may never send, on every route', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    const id = conversation.id;
    const cases: Array<[string, string, unknown]> = [
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', extra: 1 }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', _meta: { bash_command: 'id' } }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', mcpServers: [] }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', pluginDirs: ['/tmp'] }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', agentProfile: {} }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', sessionId: id }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', cwd: base }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 42 }],
      ['POST', '/api/chat/conversations', { mode: 'other', text: 'hi' }],
      ['POST', '/api/chat/conversations', { text: 'hi' }],
      ['POST', '/api/chat/conversations', ['hi']],
      ['POST', `/api/chat/conversations/${id}/prompt`, { text: 'hi', env: {} }],
      ['POST', `/api/chat/conversations/${id}/prompt`, { text: 'hi', prompt: [{ type: 'text', text: 'x' }] }],
      ['POST', `/api/chat/conversations/${id}/cancel`, { now: true }],
      ['PATCH', `/api/chat/conversations/${id}`, { title: 'a', modelId: 'qwen3.5-4b' }],
      ['PATCH', `/api/chat/conversations/${id}`, { name: 'a' }],
      ['DELETE', `/api/chat/conversations/${id}`, { force: true }],
      ['POST', '/api/chat/agent', { action: 'restart', extra: 1 }],
      ['POST', '/api/chat/agent', { action: 'upgrade' }],
      ['GET', '/api/chat/conversations?mode=all', undefined],
      ['GET', '/api/chat/conversations', undefined],
    ];
    for (const [method, path, body] of cases) {
      const failure = failureOf(await call(method, path, body));
      expect([method, path, failure.code]).toEqual([method, path, 'E_BAD_REQUEST']);
    }
    const prompts = sent().filter((m) => m.method === 'session/prompt');
    expect(prompts).toHaveLength(1);
  }, 20_000);

  it('never lets slash text reach the agent', async () => {
    start('slash-yolo');
    for (const text of ['/yolo', ' /yolo', '\n/always-approve on', '﻿/plugins add x', '　/workflow go']) {
      expect(isSlashText(text)).toBe(true);
      const failure = failureOf(await call('POST', '/api/chat/conversations', { mode: 'chat', text }));
      expect(failure).toEqual({
        code: 'E_BAD_REQUEST',
        reason: 'slash-text',
        message: 'Start the message with a word, not /.',
      });
    }
    const conversation = await created('a word first, then /yolo');
    await settled(conversation.id);
    const slash = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, { text: '/yolo' });
    expect(failureOf(slash).reason).toBe('slash-text');
    const prompts = sent().filter((m) => m.method === 'session/prompt');
    expect(prompts.map((m) => ((m.params as Json).prompt as Json[])[0]!.text)).toEqual(['a word first, then /yolo']);
    for (const m of prompts) {
      for (const block of (m.params as Json).prompt as Json[])
        expect(Object.keys(block).sort()).toEqual(['text', 'type']);
    }
  }, 20_000);

  it('answers empty, oversized, unknown and malformed ids with their codes', async () => {
    start('chat-basic');
    expect(failureOf(await call('POST', '/api/chat/conversations', { mode: 'chat', text: '  ' })).code).toBe(
      'E_BAD_REQUEST',
    );
    const big = 'x'.repeat(100_001);
    expect(failureOf(await call('POST', '/api/chat/conversations', { mode: 'chat', text: big })).code).toBe(
      'E_PAYLOAD_TOO_LARGE',
    );
    expect(failureOf(await call('POST', '/api/chat/conversations/no-such/prompt', { text: 'hi' })).code).toBe(
      'E_NOT_FOUND',
    );
    expect(failureOf(await call('POST', '/api/chat/conversations/a.b/cancel')).code).toBe('E_NOT_FOUND');
    expect(sent().filter((m) => m.method === 'session/prompt')).toEqual([]);
  });

  it('refuses interactions of a Chat conversation', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    const answer = await call('POST', `/api/chat/conversations/${conversation.id}/interactions/i-1`, {
      optionId: 'allow-once',
    });
    expect(failureOf(answer).code).toBe('E_NOT_FOUND');
  }, 20_000);
});

describe('Chat routes — the error text law', () => {
  it('replaces any unexpected error with the fixed sentence of its route and logs nothing', async () => {
    const spies = (['log', 'warn', 'error', 'info', 'debug'] as const).map((m) =>
      vi.spyOn(console, m).mockImplementation(() => {}),
    );
    try {
      const thrown = await guarded('chat', () => {
        throw new Error(`EACCES ${MARKER} /Users/someone/secret`);
      }).catch((e: unknown) => e as Error & { code: string });
      expect(thrown.message).toBe('The agent could not be reached.');
      expect(thrown.code).toBe('E_UNAVAILABLE');
      expect(JSON.stringify(thrown)).not.toContain(MARKER);
      expect(String((thrown as { cause?: unknown }).cause)).not.toContain(MARKER);
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });

  it('turns an agent error whose text holds a marker into a fixed outcome', async () => {
    start('error-with-marker');
    const conversation = await created();
    await settled(conversation.id);
    const w = watch(conversation.id);
    const prompt = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, { text: 'again' });
    expect(JSON.stringify(prompt)).not.toContain(MARKER);
    expect(await w.ended()).toMatchObject({ outcome: 'model-error' });
    w.off();
    expect(JSON.stringify(w.events)).not.toContain(MARKER);
  }, 20_000);
});

describe('Chat routes — Work', () => {
  function startWork(scenarioName: string): string {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start(scenarioName, undefined, { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } });
    return project;
  }

  async function createdWork(cwd: string, text: string): Promise<ConversationInfo> {
    const response = await call('POST', '/api/chat/conversations', { mode: 'work', text, cwd });
    if (!response.ok) throw new Error(response.message);
    expect(response.status).toBe(201);
    return (response.body as { conversation: ConversationInfo }).conversation;
  }

  function cards(w: { events: ConversationEvent[] }): Interaction[] {
    const seen = new Map<string, Interaction>();
    for (const e of w.events) {
      const list = e.type === 'interaction' ? [e.interaction] : e.type === 'sync' ? e.interactions : [];
      for (const card of list) if (!seen.has(card.id)) seen.set(card.id, card);
    }
    return [...seen.values()];
  }

  it('creates in a checked folder, answers a plan, lists, and deletes through the agent', async () => {
    const project = startWork('work-plan');
    const refused = await call('POST', '/api/chat/conversations', {
      mode: 'work',
      text: 'plan',
      cwd: join(base, 'home'),
    });
    expect([failureOf(refused).code, failureOf(refused).reason]).toEqual(['E_BAD_REQUEST', 'cwd-invalid']);

    const conversation = await createdWork(project, 'plan');
    expect(conversation).toMatchObject({ mode: 'work', cwd: realpathSync(project) });
    const w = watch(conversation.id);
    await until(() => cards(w).length === 1, 5_000);
    const plan = cards(w)[0]!;
    expect(plan.kind).toBe('plan');
    const path = `/api/chat/conversations/${conversation.id}/interactions/${plan.id}`;
    expect(failureOf(await call('POST', path, { optionId: 'allow-once' })).code).toBe('E_NOT_FOUND');
    expect(failureOf(await call('POST', path, { decision: 'maybe' })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path, { decision: 'approved', path: '/x' })).code).toBe('E_BAD_REQUEST');
    const answered = await call('POST', path, { decision: 'approved' });
    expect(answered.ok && answered.body).toEqual({});
    expect(await w.ended()).toMatchObject({ outcome: 'completed' });
    w.off();

    let rows: Array<{ id: string; mode: string }> = [];
    await until(() => {
      void call('GET', '/api/chat/conversations?mode=work').then((list) => {
        if (list.ok) rows = (list.body as { conversations: typeof rows }).conversations;
      });
      return rows.some((r) => r.id === conversation.id);
    }, 10_000);
    expect(rows.find((r) => r.id === conversation.id)?.mode).toBe('work');

    const deleted = await call('DELETE', `/api/chat/conversations/${conversation.id}`);
    expect(deleted.ok && deleted.body).toEqual({});
    expect(sent().find((m) => m.method === '_x.ai/session/delete')?.params).toEqual({
      sessionId: conversation.id,
      cwd: realpathSync(project),
    });
  }, 30_000);

  it('refuses an option the card did not list; Deny ends the turn as denied', async () => {
    const project = startWork('work-deny');
    const conversation = await createdWork(project, 'clean');
    const w = watch(conversation.id);
    await until(() => cards(w).length === 1, 5_000);
    const card = cards(w)[0]!;
    expect(card).toMatchObject({ kind: 'permission', command: 'rm -rf build', flags: ['deletes'] });
    const path = `/api/chat/conversations/${conversation.id}/interactions/${card.id}`;
    const yolo = await call('POST', path, { optionId: 'enable-always-approve' });
    expect([failureOf(yolo).code, failureOf(yolo).message]).toEqual(['E_BAD_REQUEST', 'That option was not offered.']);
    expect(failureOf(await call('POST', path, { decision: 'approved' })).code).toBe('E_NOT_FOUND');
    const deny = await call('POST', path, { optionId: 'reject-once' });
    expect(deny.ok).toBe(true);
    expect(await w.ended()).toMatchObject({ outcome: 'denied' });
    w.off();
  }, 20_000);

  describe('the Sessions routes redact a key a Work command printed', () => {
    const KEY = 'exa-KEY-printed-0123456789';

    /** A Work turn that prints through the card, then the detail read through the runtime once it is on disk. */
    async function printed(
      setOutput: (permission: Json) => void,
      prompt = 'print',
    ): Promise<{ id: string; detail: SessionDetailResponse }> {
      const home = join(base, 'home');
      const project = join(home, 'code', 'app');
      mkdirSync(project, { recursive: true });
      const secrets = new MemorySecretStore();
      await secrets.set('exa', KEY);
      start('work-key-printed', (s) => setOutput(((s.prompts as Json[])[0]!.steps as Json[])[0]!.permission as Json), {
        baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home },
        secrets,
      });
      await runtime.context.providers!.select({ search: 'exa' });
      const conversation = await createdWork(project, prompt);
      const w = watch(conversation.id);
      await until(() => cards(w).length === 1, 5_000);
      const path = `/api/chat/conversations/${conversation.id}/interactions/${cards(w)[0]!.id}`;
      expect((await call('POST', path, { optionId: 'allow-once' })).ok).toBe(true);
      expect(await w.ended()).toMatchObject({ outcome: 'completed' });
      w.off();
      const end = Date.now() + 10_000;
      for (;;) {
        const r = await call('GET', `/api/sessions/${conversation.id}`);
        const body = r.ok ? (r.body as SessionDetailResponse) : null;
        if (body?.transcript.some((e) => e.kind === 'tool' && e.output !== undefined)) {
          return { id: conversation.id, detail: body };
        }
        if (Date.now() > end) throw new Error('the tool output never reached the disk');
      }
    }

    /** What ChatService reads from disk: raw, the Chat address ledger is built from it. */
    function serviceRead(id: string): Promise<SessionDetailResponse | null> {
      const chat = runtime.context.chat as unknown as {
        opts: { readSession(id: string): Promise<SessionDetailResponse | null> };
      };
      return chat.opts.readSession(id);
    }

    async function otherRoutes(id: string): Promise<string> {
      const bodies = await Promise.all([
        call('GET', `/api/sessions/${id}/timeline?turn=1`),
        call('GET', '/api/sessions'),
        call('GET', '/api/chat/conversations?mode=work'),
      ]);
      return JSON.stringify(bodies);
    }

    it('a whole provider key, in the output and in the title the list routes carry', async () => {
      const { id, detail } = await printed(() => {}, `print ${KEY}`);
      expect(detail.session.title).toBe('print [redacted]');
      expect(JSON.stringify(detail)).not.toContain(KEY);
      expect(JSON.stringify(detail)).toContain('EXA_API_KEY=[redacted]');
      expect(await otherRoutes(id)).not.toContain(KEY);
      expect(JSON.stringify(await serviceRead(id))).toContain(`EXA_API_KEY=${KEY}`);
    }, 30_000);

    it('the whole inference token', async () => {
      const { id, detail } = await printed((p) => {
        p.outputEnv = 'XAI_API_KEY';
      });
      expect(JSON.stringify(detail)).not.toContain(INFERENCE_TOKEN);
      expect(JSON.stringify(detail)).toContain('XAI_API_KEY=[redacted]');
      expect(JSON.stringify(await serviceRead(id))).toContain(`XAI_API_KEY=${INFERENCE_TOKEN}`);
    }, 30_000);

    it('the start of a key at the 16 KB cut edge', async () => {
      const head = KEY.slice(0, 12);
      const { id, detail } = await printed((p) => {
        delete p.outputEnv;
        p.output = 'x'.repeat(MAX_FIELD_CHARS - head.length) + KEY + ' tail';
      });
      const tool = detail.transcript.find((e) => e.kind === 'tool');
      expect(tool?.kind === 'tool' && tool.truncated).toBe(true);
      expect(tool?.kind === 'tool' && tool.output?.endsWith('x')).toBe(true);
      const raw = (await serviceRead(id))?.transcript.find((e) => e.kind === 'tool');
      expect(raw?.kind === 'tool' && raw.output?.endsWith(head)).toBe(true);
    }, 30_000);
  });
});

describe('the permission mode input', () => {
  it('takes exactly {mode} with one of ask, auto, bypass', () => {
    for (const mode of ['ask', 'auto', 'bypass'] as const) expect(permissionModeInput({ mode })).toEqual({ mode });
    for (const body of [
      {},
      { mode: 'Bypass' },
      { mode: 'yolo' },
      { mode: true },
      { mode: ['bypass'] },
      { mode: 'bypass', on: true },
      { on: true },
      ['bypass'],
      'bypass',
      null,
    ]) {
      let code = 'ok';
      try {
        permissionModeInput(body);
      } catch (error) {
        code = (error as { code?: string }).code ?? 'thrown';
      }
      expect([body, code]).toEqual([body, 'E_BAD_REQUEST']);
    }
  });
});
