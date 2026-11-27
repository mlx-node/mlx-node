/**
 * The Chat routes through the real runtime (worker thread included) on the scripted fake agent: the thread of each
 * route, the exact field lists (unknown fields refused), slash text that never reaches the agent, the size and id
 * rules, interactions refused for Chat, the error text law, and the `chat` stream of the runtime.
 */

import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { MAX_FIELD_CHARS } from '../src/agent-sessions/transcript.js';
import type { SessionDetailResponse, TranscriptEntry } from '../src/agent-sessions/types.js';
import { routeThreadFor } from '../src/api/dispatch.js';
import type { ApiResponse } from '../src/api/errors.js';
import { ROUTES } from '../src/api/routes.js';
import { isRuleId } from '../src/chat/agent/never-rules.js';
import { promptParams } from '../src/chat/agent/profiles.js';
import { numberedBytes, pdfNote, type AttachmentInput } from '../src/chat/attachment-rule.js';
import { applyOp } from '../src/chat/conversation.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import { REDACTED } from '../src/chat/redact.js';
import { guarded } from '../src/chat/route-errors.js';
import {
  createConversationInput,
  forkInput,
  goalInput,
  interactionInput,
  pdfReadInput,
  pdfStageInput,
  isSlashText,
  MAX_FEEDBACK_CHARS,
  permissionModeInput,
  promptInput,
  queuePatchInput,
  queueSendInput,
  rewindInput,
  rewindPlanInput,
  stageParam,
} from '../src/chat/route-input.js';
import type { AgentWatchAnswer, ConversationEvent, ConversationInfo, Interaction } from '../src/chat/types.js';
import { createDashboardRuntime, type ChatRuntimeOptions, type DashboardRuntime } from '../src/runtime.js';
import { endpointModel, FAKE_AGENT, INFERENCE_TOKEN, readScenario, until, type Json } from './helpers/chat-fake.js';
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
      tempRoots: [],
      agentCommand: { command: process.execPath, args: [FAKE_AGENT, scenarioPath] },
      secrets: new MemorySecretStore(),
      readEndpoint: async () => ({
        version: 1,
        pid: process.pid,
        url: engine.url,
        token: INFERENCE_TOKEN,
        models: [endpointModel('qwen3.5-4b', '/models/qwen3.5-4b')],
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
      'main PATCH /api/chat/conversations/:id/queue/:qid',
      'main DELETE /api/chat/conversations/:id/queue/:qid',
      'main DELETE /api/chat/conversations/:id/queue',
      'main POST /api/chat/conversations/:id/queue/release',
      'main POST /api/chat/conversations/:id/queue/:qid/send',
      'main POST /api/chat/conversations/:id/compact',
      'main GET /api/chat/conversations/:id/context',
      'main POST /api/chat/conversations/:id/rewind/plan',
      'main POST /api/chat/conversations/:id/rewind',
      'main POST /api/chat/conversations/:id/fork',
      'main POST /api/chat/conversations/:id/recap',
      'main POST /api/chat/conversations/:id/goal',
      'main POST /api/chat/conversations/:id/pdf-read',
      'main POST /api/chat/conversations/:id/runs/:runId/stop',
      'main POST /api/chat/conversations/:id/tasks/:taskId/stop',
      'main POST /api/chat/conversations/:id/subagents/:childId/stop',
      'main POST /api/chat/conversations/:id/subagents/:childId/message',
      'main POST /api/chat/conversations/:id/subagents/:childId/watch',
      'main DELETE /api/chat/conversations/:id/subagents/:childId/watch',
      'main POST /api/chat/conversations/:id/schedules/:scheduleId/delete',
      'main POST /api/chat/conversations/:id/interactions/:iid',
      'main GET /api/chat/conversations/:id/never',
      'main DELETE /api/chat/conversations/:id/never/:ruleId',
      'main POST /api/chat/conversations/:id/permission-mode',
      'main POST /api/chat/drafts/:draft/permission-mode',
      'main POST /api/chat/pdf-stage',
      'main DELETE /api/chat/pdf-stage/:stage',
      'main GET /api/chat/work/hooks',
      'main GET /api/chat/work/git',
      'main POST /api/chat/work/git/switch',
      'main GET /api/chat/work/settings',
      'main PUT /api/chat/work/settings',
      'main PATCH /api/chat/conversations/:id',
      'main DELETE /api/chat/conversations/:id',
      'main POST /api/chat/agent',
      'main POST /api/chat/engine/unload',
    ]);
    expect(routeThreadFor('POST', '/api/chat/conversations/abc/prompt')).toBe('main');
  });
});

describe('Chat routes — the New Work screen', () => {
  it('the draft mode route takes exactly {mode, up, cwd?, noFolder?, temp?}; a move up names its place, a move down none', async () => {
    start('chat-basic');
    const folder = join(base, 'home', 'code', 'app');
    mkdirSync(folder, { recursive: true });
    const path = '/api/chat/drafts/d1/permission-mode';
    for (const body of [
      {},
      { mode: 'bypass', up: true },
      { mode: 'bypass', up: true, cwd: '' },
      { mode: 'bypass', up: true, cwd: 'x'.repeat(4_097) },
      { mode: 'ask', up: false, cwd: folder },
      { mode: 'ask', up: false, temp: '/tmp' },
      { mode: 'bypass', up: true, cwd: folder, draft: 'd2' },
      { mode: 'Bypass', up: true, cwd: folder },
      { mode: 'bypass', up: 'true', cwd: folder },
      { mode: 'bypass', cwd: folder },
      { mode: 'bypass', up: true, cwd: folder, noFolder: true },
      { mode: 'bypass', up: true, noFolder: false },
      { mode: 'ask', up: false, noFolder: true },
      null,
    ]) {
      expect([body, failureOf(await call('POST', path, body)).code]).toEqual([body, 'E_BAD_REQUEST']);
    }
    expect(
      failureOf(await call('POST', '/api/chat/drafts/bad%20id/permission-mode', { mode: 'ask', up: false })).code,
    ).toBe('E_NOT_FOUND');
    const home = await call('POST', path, { mode: 'bypass', up: true, cwd: join(base, 'home') });
    expect([failureOf(home).code, failureOf(home).reason]).toEqual(['E_BAD_REQUEST', 'cwd-invalid']);
    const up = await call('POST', path, { mode: 'bypass', up: true, cwd: folder });
    expect(up.ok && up.body).toEqual({ mode: 'bypass' });
    const down = await call('POST', path, { mode: 'ask', up: false });
    expect(down.ok && down.body).toEqual({ mode: 'ask' });
    // No folder: nothing on disk yet, so no folder is checked.
    const none = await call('POST', path, { mode: 'bypass', up: true, noFolder: true });
    expect(none.ok && none.body).toEqual({ mode: 'bypass' });
    expect(existsSync(join(base, 'home', 'Portato'))).toBe(false);
  });

  it('the hooks route reads the folder rule first, then the hooks', async () => {
    start('chat-basic');
    const folder = join(base, 'home', 'code', 'app');
    mkdirSync(folder, { recursive: true });
    const ok = await call('GET', `/api/chat/work/hooks?cwd=${encodeURIComponent(folder)}`);
    expect(ok.ok && ok.body).toEqual({ userHooks: false });
    for (const cwd of [join(base, 'home'), join(base, 'home', 'missing')]) {
      const refused = await call('GET', `/api/chat/work/hooks?cwd=${encodeURIComponent(cwd)}`);
      expect([cwd, failureOf(refused).reason]).toEqual([cwd, 'cwd-invalid']);
    }
    expect(failureOf(await call('GET', '/api/chat/work/hooks')).reason).toBe('cwd-invalid');
    // Mutation: the none read sent through the folder rule → the draft confirm of No folder never arms.
    const none = await call('GET', '/api/chat/work/hooks?noFolder=1');
    expect(none.ok && none.body).toEqual({ userHooks: false });
    expect(existsSync(join(base, 'home', 'Portato'))).toBe(false);
    expect(failureOf(await call('GET', '/api/chat/work/hooks?noFolder=1&cwd=%2Ftmp')).code).toBe('E_BAD_REQUEST');
  });
});

describe('Chat routes — the Work settings', () => {
  // Mutations: the parser takes an extra field or a value that is not a boolean; the value kept in memory only.
  it('takes exactly {claudeCode: boolean}, saves it and reads it back after a restart', async () => {
    start('chat-basic');
    const path = '/api/chat/work/settings';
    const read = async (): Promise<unknown> => {
      const reply = await call('GET', path);
      return reply.ok && reply.body;
    };
    expect(await read()).toEqual({ claudeCode: true });
    for (const body of [
      {},
      { claudeCode: 'false' },
      { claudeCode: null },
      { claudeCode: false, mode: 'work' },
      [],
      null,
    ]) {
      expect([body, failureOf(await call('PUT', path, body)).code]).toEqual([body, 'E_BAD_REQUEST']);
    }
    expect(await read()).toEqual({ claudeCode: true });
    const put = await call('PUT', path, { claudeCode: false });
    expect(put.ok && put.body).toEqual({ claudeCode: false });
    expect(JSON.parse(readFileSync(join(base, 'chat', 'work-settings.json'), 'utf8'))).toEqual({
      version: 1,
      claudeCode: false,
    });
    await runtime.close();
    start('chat-basic');
    expect(await read()).toEqual({ claudeCode: false });
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

  it('Compact now: 202 with no body once it started, its end on the stream; a body or a running turn is refused', async () => {
    start('chat-basic', (scenario) => {
      scenario.compactNow = { ms: 200 };
    });
    const conversation = await created();
    await settled(conversation.id);
    const path = `/api/chat/conversations/${conversation.id}/compact`;
    expect(failureOf(await call('POST', path, { force: true })).code).toBe('E_BAD_REQUEST');
    const w = watch(conversation.id);
    const response = await call('POST', path);
    expect(response.ok && response.status).toBe(202);
    expect(response.ok && response.body).toEqual({});
    expect(failureOf(await call('POST', path))).toMatchObject({ code: 'E_CONFLICT', reason: 'conversation-busy' });
    await until(() => w.events.some((e) => e.type === 'compaction' && e.end === 'compacted'), 5_000);
    expect(sent().filter((m) => m.method === '_x.ai/compact_conversation')).toHaveLength(1);
    w.off();
  }, 20_000);

  it('Recap: 202 with no body once it started, `auto: false` to the agent, its end and its row on the stream', async () => {
    start('chat-basic', (scenario) => {
      scenario.recap = { ms: 200, summary: 'Where things stand.' };
    });
    const conversation = await created();
    await settled(conversation.id);
    const path = `/api/chat/conversations/${conversation.id}/recap`;
    expect(failureOf(await call('POST', path, { auto: true })).code).toBe('E_BAD_REQUEST');
    const w = watch(conversation.id);
    const response = await call('POST', path);
    expect(response.ok && response.status).toBe(202);
    expect(response.ok && response.body).toEqual({});
    expect(failureOf(await call('POST', path))).toMatchObject({ code: 'E_CONFLICT', reason: 'conversation-busy' });
    await until(() => w.events.some((e) => e.type === 'recap' && e.end === 'shown'), 5_000);
    expect(
      sent()
        .filter((m) => m.method === '_x.ai/recap')
        .map((m) => m.params),
    ).toEqual([{ sessionId: conversation.id, auto: false }]);
    w.off();
  }, 20_000);

  it('Rewind: the plan answers 200 with its token; the commit 202 with no body, its end on the stream; bad bodies refused', async () => {
    start('chat-basic');
    const conversation = await created('one');
    await settled(conversation.id);
    const base = `/api/chat/conversations/${conversation.id}/rewind`;
    const shown = watch(conversation.id);
    await until(() => shown.events.some((e) => e.type === 'sync'), 5_000);
    const sync = shown.events.find((e) => e.type === 'sync');
    const ts = (at: number): number => (sync?.type === 'sync' ? sync.entries[at]!.ts : -1);
    shown.off();
    expect(failureOf(await call('POST', `${base}/plan`, { entry: 'x', ts: 0 })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', `${base}/plan`, { entry: 0 })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', `${base}/plan`, { entry: 1, ts: ts(1) }))).toMatchObject({
      code: 'E_CONFLICT',
      reason: 'rewind-not-point',
    });
    expect(failureOf(await call('POST', `${base}/plan`, { entry: 0, ts: ts(0) + 1 }))).toMatchObject({
      code: 'E_CONFLICT',
      reason: 'rewind-changed',
    });
    const plan = await call('POST', `${base}/plan`, { entry: 0, ts: ts(0) });
    expect(plan.ok && plan.status).toBe(200);
    const token = (plan as { body: { token: string } }).body.token;
    expect(plan.ok && plan.body).toMatchObject({
      removes: { messages: 1 },
      background: { commands: 0, schedules: 0, runs: 0 },
      files: [],
      filesMore: 0,
    });
    // Chat has no files: a file mode is a bad request, and no dry run went to the agent.
    for (const mode of ['all', 'files']) {
      expect(failureOf(await call('POST', base, { token, mode })).code).toBe('E_BAD_REQUEST');
    }
    expect(sent().filter((m) => m.method === '_x.ai/rewind/execute')).toEqual([]);
    const w = watch(conversation.id);
    const response = await call('POST', base, { token, mode: 'conversation' });
    expect(response.ok && response.status).toBe(202);
    expect(response.ok && response.body).toEqual({});
    await until(() => w.events.some((e) => e.type === 'rewind'), 5_000);
    expect(w.events.find((e) => e.type === 'rewind')).toMatchObject({
      end: { outcome: 'done', mode: 'conversation', restored: 0, entry: 0 },
    });
    expect(
      sent()
        .filter((m) => m.method === '_x.ai/rewind/execute')
        .map((m) => m.params),
    ).toEqual([{ sessionId: expect.any(String), targetPromptIndex: 0, force: true, mode: 'conversation_only' }]);
    w.off();
  }, 20_000);

  it('the Context panel: GET context answers the agent’s parts while the conversation is loaded', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    const response = await call('GET', `/api/chat/conversations/${conversation.id}/context`);
    expect(response.ok && response.status).toBe(200);
    expect(response.ok && response.body).toMatchObject({
      parts: { systemTokens: 432, toolCount: 2, toolTokens: 261, autoCompactPercent: 85, compactions: 0 },
    });
    expect(failureOf(await call('GET', '/api/chat/conversations/nope/context')).code).toBe('E_NOT_FOUND');
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

  it('reports agent.canInstall from whether installAgent is configured', async () => {
    start('chat-basic');
    const withoutInstaller = await call('GET', '/api/chat/state');
    expect(withoutInstaller.ok && withoutInstaller.body).toMatchObject({ agent: { canInstall: false } });
    await runtime.close();

    start('chat-basic', undefined, { installAgent: async () => {} });
    const withInstaller = await call('GET', '/api/chat/state');
    expect(withInstaller.ok && withInstaller.body).toMatchObject({ agent: { canInstall: true } });
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
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', plan: true }],
      ['POST', '/api/chat/conversations', { mode: 'chat', text: 'hi', plan: false }],
      ['POST', `/api/chat/conversations/${id}/prompt`, { text: 'hi', plan: true }],
      ['POST', `/api/chat/conversations/${id}/prompt`, { text: 'hi', plan: false }],
      ['POST', `/api/chat/conversations/${id}/prompt`, { text: 'hi', plan: 'yes' }],
      ['POST', '/api/chat/conversations', { mode: 'work', text: 'hi', cwd: base, plan: 1 }],
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
        message: 'No such command here.',
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

  // Mutation: the worker without the record file → the list row of a No folder conversation is not marked.
  it('a create with no folder runs in a new folder of ~/Portato, and the worker lists it as No folder', async () => {
    startWork('work-plan');
    const response = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'plan', noFolder: true });
    if (!response.ok) throw new Error(response.message);
    const conversation = (response.body as { conversation: ConversationInfo }).conversation;
    expect(conversation).toMatchObject({ mode: 'work', noFolder: true });
    expect(dirname(conversation.cwd!)).toBe(realpathSync(join(base, 'home', 'Portato')));
    const w = watch(conversation.id);
    await until(() => cards(w).length === 1, 5_000);
    const answer = `/api/chat/conversations/${conversation.id}/interactions/${cards(w)[0]!.id}`;
    expect((await call('POST', answer, { decision: 'approved' })).ok).toBe(true);
    expect(await w.ended()).toMatchObject({ outcome: 'completed' });
    w.off();
    let rows: Array<{ id: string; noFolder?: true }> = [];
    await until(() => {
      void call('GET', '/api/chat/conversations?mode=work').then((list) => {
        if (list.ok) rows = (list.body as { conversations: typeof rows }).conversations;
      });
      return rows.some((r) => r.id === conversation.id);
    }, 10_000);
    expect(rows.find((r) => r.id === conversation.id)?.noFolder).toBe(true);
  }, 30_000);

  /** Every transcript entry the stream showed: the `sync` list, then each pushed one. */
  function entries(w: { events: ConversationEvent[] }): Array<{ kind: string; mode?: string }> {
    const out: Array<{ kind: string; mode?: string }> = [];
    for (const e of w.events) {
      if (e.type === 'sync') out.splice(0, out.length, ...e.entries);
      if (e.type === 'ops') for (const op of e.ops) if (op.op === 'push') out.push(op.entry);
    }
    return out;
  }
  const planLines = (w: { events: ConversationEvent[] }) =>
    entries(w).flatMap((entry) => (entry.kind === 'plan-mode' ? [entry.mode] : []));
  const promptModes = () =>
    sent()
      .filter((m) => m.method === 'session/prompt')
      .map((m) => ((m.params as Json)._meta as Json | undefined)?.mode);
  const textOnly = (scenario: Json): void => {
    scenario.prompts = [{ match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] }];
  };

  it("plan: the first message and each prompt name their mode; the agent's mode lines reach the stream", async () => {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start('work-plan', textOnly, { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } });
    const made = await call('POST', '/api/chat/conversations', {
      mode: 'work',
      text: 'Plan it',
      cwd: project,
      plan: true,
    });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    await settled(id);
    const w = watch(id);
    await w.idle();
    expect(planLines(w)).toEqual(['plan']);
    // Already planning: the agent sends no mode line, and that is no sign of an agent without plan mode.
    const again = await call('POST', `/api/chat/conversations/${id}/prompt`, { text: 'Go on', plan: true });
    expect(again.ok).toBe(true);
    await settled(id);
    expect(planLines(w)).toEqual(['plan']);
    expect(w.events.some((e) => e.type === 'notice')).toBe(false);
    const next = await call('POST', `/api/chat/conversations/${id}/prompt`, { text: 'Build it', plan: false });
    expect(next.ok).toBe(true);
    await until(() => planLines(w).length === 2, 5_000);
    expect(planLines(w)).toEqual(['plan', 'default']);
    expect(promptModes()).toEqual(['plan', 'plan', 'agent']);
    expect(w.events.some((e) => e.type === 'notice')).toBe(false);
    w.off();
  }, 20_000);

  it('an agent that reads no `_meta.mode`: a stopped planned message says nothing; one that ends makes the notice; the next message clears it', async () => {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start(
      'work-plan',
      (scenario) => {
        textOnly(scenario);
        (scenario.prompts as Json[]).unshift({ match: 'Hold', steps: [{ sleepMs: 10_000 }, { text: 'ok' }] });
        scenario.noPlanMode = true;
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    const conversation = await createdWork(project, 'Build it');
    await settled(conversation.id);
    const w = watch(conversation.id);
    await w.idle();
    expect(w.events.some((e) => e.type === 'notice')).toBe(false);
    const ends = (): number => w.events.filter((e) => e.type === 'turn' && e.turn === null).length;
    // A stopped answer is no sign of an agent without plan mode.
    const held = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, {
      text: 'Hold it',
      plan: true,
    });
    expect(held.ok).toBe(true);
    await until(() => w.events.some((e) => e.type === 'turn' && e.turn?.phase === 'running'), 5_000);
    expect((await call('POST', `/api/chat/conversations/${conversation.id}/cancel`)).ok).toBe(true);
    await until(() => ends() === 1, 5_000);
    expect(w.events.find((e) => e.type === 'turn' && e.turn === null)).toMatchObject({ outcome: 'cancelled' });
    expect(w.events.some((e) => e.type === 'notice')).toBe(false);
    const asked = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, {
      text: 'Plan it',
      plan: true,
    });
    expect(asked.ok).toBe(true);
    await until(() => ends() === 2, 5_000);
    expect(w.events.find((e) => e.type === 'notice')).toMatchObject({
      notice: { code: 'no-plan-mode', tone: 'warning' },
    });
    expect(planLines(w)).toEqual([]);
    const again = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, { text: 'Go on' });
    expect(again.ok).toBe(true);
    await until(() => w.events.some((e) => e.type === 'notice-cleared'), 5_000);
    expect(w.events.find((e) => e.type === 'notice-cleared')).toMatchObject({ code: 'no-plan-mode' });
    await until(() => promptModes().length === 4, 5_000);
    expect(promptModes()).toEqual(['agent', 'plan', 'plan', 'agent']);
    w.off();
  }, 20_000);

  // Mutation: drop `no-model` from reachedAgent → the blocked message makes the notice.
  it('a planned message that no model answered says nothing about plan mode', async () => {
    const home = join(base, 'home');
    const project = join(home, 'code', 'blocked');
    mkdirSync(project, { recursive: true });
    start(
      'work-plan',
      (scenario) => {
        textOnly(scenario);
        (scenario.prompts as Json[]).unshift({ match: 'Blocked', noModel: 'No model is allowed.', steps: [] });
        scenario.noPlanMode = true;
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    const conversation = await createdWork(project, 'Build it');
    await settled(conversation.id);
    const w = watch(conversation.id);
    await w.idle();
    const blocked = await call('POST', `/api/chat/conversations/${conversation.id}/prompt`, {
      text: 'Blocked plan',
      plan: true,
    });
    expect(blocked.ok).toBe(true);
    await until(() => w.events.some((e) => e.type === 'turn' && e.turn === null), 5_000);
    expect(w.events.find((e) => e.type === 'turn' && e.turn === null)).toMatchObject({
      outcome: 'no-model',
      model: { previous: null, next: null, reason: 'No model is allowed.' },
    });
    await settled(conversation.id);
    expect(w.events.some((e) => e.type === 'notice')).toBe(false);
    w.off();
  }, 20_000);

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

  it('a deny with a note: a note with another option is refused; with reject-once the answer goes on', async () => {
    const project = startWork('work-deny');
    const conversation = await createdWork(project, 'clean');
    const w = watch(conversation.id);
    await until(() => cards(w).length === 1, 5_000);
    const path = `/api/chat/conversations/${conversation.id}/interactions/${cards(w)[0]!.id}`;
    expect(failureOf(await call('POST', path, { optionId: 'allow-once', note: 'use pnpm' })).code).toBe(
      'E_BAD_REQUEST',
    );
    expect(failureOf(await call('POST', path, { optionId: 'reject-once', note: 'x'.repeat(4_001) })).code).toBe(
      'E_BAD_REQUEST',
    );
    expect((await call('POST', path, { optionId: 'reject-once', note: 'use pnpm' })).ok).toBe(true);
    expect(await w.ended()).toMatchObject({ outcome: 'completed' });
    w.off();
  }, 20_000);

  it('never allow: the card answer, the rules of the folder, and their removal; Chat has none', async () => {
    const project = startWork('work-deny');
    const conversation = await createdWork(project, 'clean');
    const w = watch(conversation.id);
    await until(() => cards(w).length === 1, 5_000);
    const card = cards(w)[0]!;
    expect(card).toMatchObject({ never: { short: 'rm', exact: 'rm -rf build' } });
    const path = `/api/chat/conversations/${conversation.id}/interactions/${card.id}`;
    expect(failureOf(await call('POST', path, { never: 'all' })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path, { never: 'short', optionId: 'reject-once' })).code).toBe('E_BAD_REQUEST');
    const made = await call('POST', path, { never: 'exact' });
    if (!made.ok) throw new Error('the never answer failed');
    const never = (made.body as { never: { saved: true; ruleId: string; words: string } }).never;
    expect(never).toEqual({ saved: true, ruleId: expect.any(String), words: 'rm -rf build' });
    expect(isRuleId(never.ruleId)).toBe(true);
    expect(await w.ended()).toMatchObject({ outcome: 'completed' });
    const rules = `/api/chat/conversations/${conversation.id}/never`;
    const listed = await call('GET', rules);
    expect(listed.ok && listed.body).toEqual({
      folder: conversation.cwd,
      rules: [{ id: never.ruleId, words: 'rm -rf build', at: expect.any(Number) }],
    });
    expect(failureOf(await call('DELETE', `${rules}/n_${'0'.repeat(32)}`))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such rule.',
    });
    expect(failureOf(await call('DELETE', `${rules}/not.a.rule`)).code).toBe('E_NOT_FOUND');
    expect(failureOf(await call('POST', `${rules}`, {})).code).toBe('E_METHOD_NOT_ALLOWED');
    const gone = await call('DELETE', `${rules}/${never.ruleId}`);
    expect(gone.ok && gone.body).toEqual({});
    const after = await call('GET', rules);
    expect(after.ok && (after.body as { rules: unknown[] }).rules).toEqual([]);
    w.off();
  }, 20_000);

  it('never allow: a Chat conversation has no rules', async () => {
    start('chat-basic');
    const chat = await created();
    await settled(chat.id);
    expect(failureOf(await call('GET', `/api/chat/conversations/${chat.id}/never`)).code).toBe('E_NOT_FOUND');
    const rule = `n_${'0'.repeat(32)}`;
    expect(failureOf(await call('DELETE', `/api/chat/conversations/${chat.id}/never/${rule}`)).code).toBe(
      'E_NOT_FOUND',
    );
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

describe('Chat routes — attachments', () => {
  it('both routes pass checked attachments to the agent; a picture for a model that takes none is refused', async () => {
    start('chat-basic', undefined, {
      readEndpoint: async () => ({
        version: 1,
        pid: process.pid,
        url: engine.url,
        token: INFERENCE_TOKEN,
        models: [
          { ...endpointModel('qwen3.5-4b', '/models/qwen3.5-4b'), images: true },
          endpointModel('qwen3.5-9b', '/models/qwen3.5-9b'),
        ],
      }),
    });
    const file: AttachmentInput = { kind: 'file', name: 'dir/a#1.md', text: '# a\n' };
    const picture: AttachmentInput = { kind: 'image', mediaType: 'image/png', data: 'iVBORw0KGgoAAAANSUhEUg==' };
    const response = await call('POST', '/api/chat/conversations', {
      mode: 'chat',
      text: 'look',
      attachments: [file, picture],
    });
    if (!response.ok) throw new Error(response.message);
    const id = (response.body as { conversation: ConversationInfo }).conversation.id;
    await settled(id);
    const blind = await call('POST', '/api/chat/conversations', {
      mode: 'chat',
      text: 'look',
      modelId: 'qwen3.5-9b',
      attachments: [picture],
    });
    expect(failureOf(blind)).toEqual({
      code: 'E_BAD_REQUEST',
      reason: 'attachment-refused',
      message: 'The attached files were refused.',
    });
    const many = await call('POST', `/api/chat/conversations/${id}/prompt`, {
      text: 'again',
      attachments: [file, file, file, file, file],
    });
    expect(failureOf(many).reason).toBe('attachment-refused');
    const again = await call('POST', `/api/chat/conversations/${id}/prompt`, { text: 'again', attachments: [file] });
    expect(again.ok && again.status).toBe(202);
    await settled(id);
    const prompts = sent().filter((m) => m.method === 'session/prompt');
    expect(prompts.map((m) => m.params)).toEqual([
      promptParams(id, 'look', [{ ...file, name: 'a_1.md' }, picture]),
      promptParams(id, 'again', [{ ...file, name: 'a_1.md' }]),
    ]);
  }, 20_000);
});

describe('the PDF tool routes', () => {
  const jpeg = (n: number): string => Buffer.from([0xff, 0xd8, 0xff, 0xe0, n, 0]).toString('base64');
  const refused = { code: 'E_BAD_REQUEST', reason: 'attachment-refused', message: 'The attached files were refused.' };

  it('the stage route takes pages in order into a stage of their own; the delete route drops it; Read it now is 400 in Chat', async () => {
    start('chat-basic');
    const first = await call('POST', '/api/chat/pdf-stage', { stage: null, page: 1, pages: 3, data: jpeg(1) });
    if (!first.ok) throw new Error(first.message);
    const { stage } = first.body as { stage: string };
    const dir = join(base, 'chat', 'pdf-stage', stage);
    expect(existsSync(join(dir, 'page-001.jpg'))).toBe(true);
    const skip = await call('POST', '/api/chat/pdf-stage', { stage, page: 3, pages: 3, data: jpeg(3) });
    expect(failureOf(skip)).toEqual(refused);
    const other = await call('POST', '/api/chat/pdf-stage', {
      stage: 'b'.repeat(26),
      page: 2,
      pages: 3,
      data: jpeg(2),
    });
    expect(failureOf(other)).toEqual(refused);
    const second = await call('POST', '/api/chat/pdf-stage', { stage, page: 2, pages: 3, data: jpeg(2) });
    expect(second.ok && second.body).toEqual({ stage });
    expect(readFileSync(join(dir, 'page-002.jpg'))).toEqual(Buffer.from(jpeg(2), 'base64'));
    expect(failureOf(await call('DELETE', '/api/chat/pdf-stage/s1')).code).toBe('E_BAD_REQUEST');
    const dropped = await call('DELETE', `/api/chat/pdf-stage/${stage}`);
    expect(dropped.ok && dropped.body).toEqual({});
    expect(existsSync(dir)).toBe(false);
    const after = await call('POST', '/api/chat/pdf-stage', { stage, page: 3, pages: 3, data: jpeg(3) });
    expect(failureOf(after)).toEqual(refused);
    const chat = await created();
    await settled(chat.id);
    const inChat = await call('POST', `/api/chat/conversations/${chat.id}/pdf-read`, { pdf: 'k3f7a2c4-40' });
    expect(failureOf(inChat)).toEqual({
      code: 'E_BAD_REQUEST',
      message: 'The PDF tool is for Work conversations only.',
    });
  });

  it('Read it now in Work: 404 for a PDF no message of the conversation sent to the tool', async () => {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start('work-always', undefined, { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } });
    const response = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'hello', cwd: project });
    if (!response.ok) throw new Error(response.message);
    const work = (response.body as { conversation: ConversationInfo }).conversation;
    const unknown = await call('POST', `/api/chat/conversations/${work.id}/pdf-read`, { pdf: 'k3f7a2c4-40' });
    expect(failureOf(unknown)).toEqual({ code: 'E_NOT_FOUND', message: 'No such PDF.' });
    const bad = await call('POST', `/api/chat/conversations/${work.id}/pdf-read`, { pdf: 'k3f7a2c4-40', pages: '1' });
    expect(failureOf(bad).code).toBe('E_BAD_REQUEST');
  }, 20_000);
});

describe('the attachment input', () => {
  const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  const png = (bytes: number): string => {
    const data = new Uint8Array(bytes);
    data.set(PNG_HEAD);
    return Buffer.from(data).toString('base64');
  };
  const file = (text: string, name = 'a.txt') => ({ kind: 'file', name, text });
  const image = (data: string, mediaType = 'image/png') => ({ kind: 'image', mediaType, data });
  /** A file whose text, numbered as the agent renders it (`1→` + text), is `bytes` long. */
  const numbered = (bytes: number) => file('x'.repeat(bytes - 4));
  const reasonOf = (run: () => unknown): string => {
    try {
      run();
    } catch (error) {
      return (error as { reason?: string; code?: string }).reason ?? (error as { code?: string }).code ?? 'thrown';
    }
    return 'ok';
  };
  const create = (attachments: unknown, text = 'look') => createConversationInput({ mode: 'chat', text, attachments });
  const prompt = (attachments: unknown, text = 'look') => promptInput({ text, attachments });

  it('takes files and pictures on both routes, the name cleaned to its base; an empty list is no field', () => {
    const items = [file('hello\n', '../../etc/a#b.md'), image(png(64)), image('/9j/4AAQ', 'image/jpeg')];
    const expected = [
      { kind: 'file', name: 'a_b.md', text: 'hello\n' },
      { kind: 'image', mediaType: 'image/png', data: png(64) },
      { kind: 'image', mediaType: 'image/jpeg', data: '/9j/4AAQ' },
    ];
    expect(create(items)).toEqual({ mode: 'chat', text: 'look', attachments: expected });
    expect(prompt(items)).toEqual({ text: 'look', attachments: expected });
    expect(create([])).toEqual(createConversationInput({ mode: 'chat', text: 'look' }));
    expect(prompt([])).toEqual({ text: 'look' });
    expect(promptInput({ text: 'look' })).toEqual({ text: 'look' });
    expect(create([file('x', 'n'.repeat(255))]).attachments![0]).toMatchObject({ name: 'n'.repeat(120) });
  });

  it('takes the limits themselves: 4 items, 20,000 numbered bytes, 90,000 bytes in all, 1,000,000 picture bytes', () => {
    expect(
      reasonOf(() =>
        create([numbered(20_000), numbered(20_000), numbered(20_000), numbered(20_000)], 'y'.repeat(10_000)),
      ),
    ).toBe('ok');
    expect(reasonOf(() => prompt([image(png(1_000_000))]))).toBe('ok');
  });

  // Mutations: accept a 5th attachment → red; skip the first-bytes check of a picture → red.
  it('refuses every attachment past a limit as attachment-refused, on both routes', () => {
    const cases: Array<[string, unknown[], string?]> = [
      ['5 items', [file('a'), file('b'), file('c'), file('d'), file('e')]],
      ['an unknown key', [{ ...file('a'), path: '/etc/passwd' }]],
      ['a missing key', [{ kind: 'file', name: 'a.txt' }]],
      ['an unknown kind', [{ kind: 'audio', name: 'a', text: 'a' }]],
      ['not an object', ['a.txt']],
      ['an array item', [['file', 'a.txt', 'a']]],
      ['a name of 256 characters', [file('a', 'n'.repeat(256))]],
      ['an empty name', [file('a', '')]],
      ['a name that is not text', [{ kind: 'file', name: 7, text: 'a' }]],
      ['text with NUL', [file('a\0b')]],
      ['a lone surrogate', [file('a\ud800b')]],
      ['a blank text file', [file(' \n\t ')]],
      ['20,001 numbered bytes', [numbered(20_001)]],
      [
        'over 90,000 bytes in all',
        [numbered(20_000), numbered(20_000), numbered(20_000), numbered(20_000)],
        'y'.repeat(10_001),
      ],
      ['bad base64', [image('iVBORw0KGgo!AAAA')]],
      ['bad base64 length (len % 4 === 1)', [image('iVBORw0KGgoAA')]],
      ['too short', [image('iVB')]],
      ['1,000,001 picture bytes', [image(png(1_000_001))]],
      ['over the base64 length', [image(`${png(1_000_000)}AAAA`)]],
      ['PNG bytes sent as image/jpeg', [image(png(64), 'image/jpeg')]],
      ['JPEG bytes sent as image/png', [image('/9j/4AAQ')]],
      ['image/webp', [image('UklGRiQAAABXRUJQ', 'image/webp')]],
      ['data that is not text', [{ kind: 'image', mediaType: 'image/png', data: 7 }]],
    ];
    for (const [name, items, text] of cases) {
      expect([name, reasonOf(() => create(items, text))]).toEqual([name, 'attachment-refused']);
      expect([name, reasonOf(() => prompt(items, text))]).toEqual([name, 'attachment-refused']);
    }
    expect(reasonOf(() => create('a.txt'))).toBe('E_BAD_REQUEST');
    expect(reasonOf(() => prompt({ 0: file('a') }))).toBe('E_BAD_REQUEST');
  });

  describe('PDF pages', () => {
    const JPEG_HEAD = [0xff, 0xd8, 0xff, 0xe0];
    const jpeg = (bytes: number): string => {
      const data = new Uint8Array(bytes);
      data.set(JPEG_HEAD);
      return Buffer.from(data).toString('base64');
    };
    const pdf = (pages: number, bytes = 64, name = 'a.pdf') => ({
      kind: 'pdf',
      name,
      images: Array.from({ length: pages }, () => jpeg(bytes)),
    });
    const both = (items: unknown[], text?: string) => [
      reasonOf(() => create(items, text)),
      reasonOf(() => prompt(items, text)),
    ];

    it('takes a PDF as JPEG pages on both routes, the name cleaned to its base, up to the page and message limits', () => {
      const items = [pdf(2, 64, '../docs/a#1.pdf')];
      const expected = [{ kind: 'pdf', name: 'a_1.pdf', images: [jpeg(64), jpeg(64)] }];
      expect(create(items)).toEqual({ mode: 'chat', text: 'look', attachments: expected });
      expect(prompt(items)).toEqual({ text: 'look', attachments: expected });
      expect(both([pdf(20)])).toEqual(['ok', 'ok']);
      expect(both([pdf(12), pdf(8)])).toEqual(['ok', 'ok']);
      expect(both([pdf(1, 589_824)])).toEqual(['ok', 'ok']);
      const pictures = [image(png(1_000_000)), image(png(1_000_000)), image(png(1_000_000))];
      expect(both([...pictures, pdf(2, 589_824)])).toEqual(['ok', 'ok']);
      expect(both([file('hello\nPortato PDF note 1\n')])).toEqual(['ok', 'ok']);
    });

    it('counts each note in the bytes the message text and its files may fill', () => {
      const note = numberedBytes(pdfNote({ name: 'a.pdf', pages: 1, way: 'message', from: 1 }));
      const files = [numbered(20_000), numbered(20_000), numbered(20_000)];
      const room = 90_000 - 60_000 - note;
      expect(both([...files, pdf(1)], 'y'.repeat(room))).toEqual(['ok', 'ok']);
      expect(both([...files, pdf(1)], 'y'.repeat(room + 1))).toEqual(['attachment-refused', 'attachment-refused']);
    });

    // Mutations: the JPEG first-bytes check of a page dropped; a page that is not padded base64 without whitespace
    // taken; a text file that starts with the note's line taken → red.
    it('refuses a PDF past a limit, and a text file that starts like a note, as attachment-refused', () => {
      const cases: Array<[string, unknown[]]> = [
        ['an unknown key', [{ ...pdf(1), pages: 1 }]],
        ['a stage the host does not hold', [{ kind: 'pdf', name: 'a.pdf', stage: 's1', pages: 3 }]],
        ['no images', [{ kind: 'pdf', name: 'a.pdf' }]],
        ['images not a list', [{ kind: 'pdf', name: 'a.pdf', images: jpeg(64) }]],
        ['no pages', [pdf(0)]],
        ['21 images', [pdf(21)]],
        ['21 pages in two PDFs', [pdf(12), pdf(9)]],
        ['a PNG page', [{ kind: 'pdf', name: 'a.pdf', images: [jpeg(64), png(64)] }]],
        ['a page that is not text', [{ kind: 'pdf', name: 'a.pdf', images: [7] }]],
        ['a page that is not base64', [{ kind: 'pdf', name: 'a.pdf', images: ['/9j/4AA!'] }]],
        ['a 589,825-byte page', [pdf(1, 589_825)]],
        ['a page over the base64 length', [{ kind: 'pdf', name: 'a.pdf', images: [`${jpeg(589_824)}AAAA`] }]],
        ['a page with whitespace', [{ kind: 'pdf', name: 'a.pdf', images: [jpeg(64) + ' '.repeat(1000)] }]],
        ['an unpadded page', [{ kind: 'pdf', name: 'a.pdf', images: [jpeg(64).replaceAll('=', '')] }]],
        ['an empty name', [pdf(1, 64, '')]],
        ['a name of 256 characters', [pdf(1, 64, 'n'.repeat(256))]],
        [
          'over 6 MiB of pictures and pages',
          [image(png(1_000_000)), image(png(1_000_000)), image(png(1_000_000)), pdf(4, 589_824)],
        ],
        ['a text file that starts with the note line', [file('Portato PDF note 1\nname: a.pdf\n')]],
        ['the same with CRLF', [file('Portato PDF note 1\r\nname: a.pdf\r\n')]],
        ['the note line alone', [file('Portato PDF note 1')]],
      ];
      for (const [name, items] of cases) {
        expect([name, ...both(items)]).toEqual([name, 'attachment-refused', 'attachment-refused']);
      }
    });
  });

  describe('a PDF for the tool', () => {
    const STAGE = 'k3f7a2c4abcdefghijklmnopqr';
    const tool = (over: Record<string, unknown> = {}) => ({
      kind: 'pdf',
      name: 'docs/report#1.pdf',
      stage: STAGE,
      pages: 212,
      ...over,
    });
    const jpeg = (bytes: number): string => {
      const data = new Uint8Array(bytes);
      data.set([0xff, 0xd8, 0xff, 0xe0]);
      return Buffer.from(data).toString('base64');
    };
    const both = (items: unknown[]) => [reasonOf(() => create(items)), reasonOf(() => prompt(items))];

    // Mutation: two PDFs for the tool in one message taken → red.
    it('takes a stage id and its page count on both routes, the name cleaned; one per message', () => {
      const expected = [{ kind: 'pdf', name: 'report_1.pdf', stage: STAGE, pages: 212 }];
      expect(create([tool()])).toEqual({ mode: 'chat', text: 'look', attachments: expected });
      expect(prompt([tool()])).toEqual({ text: 'look', attachments: expected });
      expect(
        both([tool({ pages: 320 }), { kind: 'pdf', name: 'b.pdf', images: Array<string>(20).fill(jpeg(64)) }]),
      ).toEqual(['ok', 'ok']);
      const cases: Array<[string, unknown[]]> = [
        ['two PDFs for the tool', [tool(), tool({ stage: 'b'.repeat(26) })]],
        ['a key more', [tool({ images: [] })]],
        ['no pages', [{ kind: 'pdf', name: 'a.pdf', stage: STAGE }]],
        ['0 pages', [tool({ pages: 0 })]],
        ['321 pages', [tool({ pages: 321 })]],
        ['2.5 pages', [tool({ pages: 2.5 })]],
        ['pages as text', [tool({ pages: '212' })]],
        ['a stage of 25 letters', [tool({ stage: STAGE.slice(1) })]],
        ['a stage with a capital', [tool({ stage: STAGE.toUpperCase() })]],
        ['a stage with a path', [tool({ stage: `../${STAGE.slice(3)}` })]],
        ['an empty name', [tool({ name: '' })]],
      ];
      for (const [name, items] of cases) {
        expect([name, ...both(items)]).toEqual([name, 'attachment-refused', 'attachment-refused']);
      }
    });

    // Mutations: the page check dropped (a PNG page, a page past its limit taken); page 2 with no stage taken → red.
    it('a page sent to the stage route is exactly {stage, page, pages, data}: JPEG, padded base64, in its limits', () => {
      expect(pdfStageInput({ stage: null, page: 1, pages: 3, data: jpeg(64) })).toEqual({
        stage: null,
        page: 1,
        pages: 3,
        data: Buffer.from(jpeg(64), 'base64'),
      });
      expect(pdfStageInput({ stage: STAGE, page: 3, pages: 3, data: jpeg(589_824) }).page).toBe(3);
      const cases: Array<[string, unknown]> = [
        ['no stage key', { page: 1, pages: 3, data: jpeg(64) }],
        ['a key more', { stage: null, page: 1, pages: 3, data: jpeg(64), name: 'a.pdf' }],
        ['page 2 with no stage', { stage: null, page: 2, pages: 3, data: jpeg(64) }],
        ['a page past the count', { stage: STAGE, page: 4, pages: 3, data: jpeg(64) }],
        ['page 0', { stage: STAGE, page: 0, pages: 3, data: jpeg(64) }],
        ['321 pages', { stage: null, page: 1, pages: 321, data: jpeg(64) }],
        ['a stage that is not an id', { stage: 's1', page: 2, pages: 3, data: jpeg(64) }],
        ['a PNG page', { stage: null, page: 1, pages: 3, data: png(64) }],
        ['a 589,825-byte page', { stage: null, page: 1, pages: 3, data: jpeg(589_825) }],
        ['an unpadded page', { stage: null, page: 1, pages: 3, data: jpeg(64).replaceAll('=', '') }],
        ['a page that is not text', { stage: null, page: 1, pages: 3, data: 7 }],
        ['not an object', [null, 1, 3, jpeg(64)]],
      ];
      for (const [name, body] of cases) {
        expect([name, reasonOf(() => pdfStageInput(body))]).toEqual([name, 'attachment-refused']);
      }
    });

    it('the stage of the delete route is a stage id; Read it now takes exactly {pdf} with a PDF id', () => {
      expect(stageParam(STAGE)).toBe(STAGE);
      for (const bad of [undefined, '', 's1', STAGE.toUpperCase(), `${STAGE}a`]) {
        expect([bad, reasonOf(() => stageParam(bad))]).toEqual([bad, 'E_BAD_REQUEST']);
      }
      expect(pdfReadInput({ pdf: 'k3f7a2c4-212' })).toEqual({ pdf: 'k3f7a2c4-212' });
      for (const body of [
        {},
        { pdf: 'k3f7a2c4-321' },
        { pdf: '../x-12' },
        { pdf: 7 },
        { pdf: 'k3f7a2c4-12', question: 'q' },
      ]) {
        expect([body, reasonOf(() => pdfReadInput(body))]).toEqual([body, 'E_BAD_REQUEST']);
      }
    });
  });

  it('accepts wrapped, unpadded, and URL-safe base64 images on both routes', () => {
    const unpaddedPng = png(64).replaceAll('=', '');
    const wrappedPng = `  ${unpaddedPng.slice(0, 12)}\r\n${unpaddedPng.slice(12)} `;
    const urlSafeJpeg = '_9j_4AAQ';
    expect(reasonOf(() => create([image(unpaddedPng), image(wrappedPng), image(urlSafeJpeg, 'image/jpeg')]))).toBe(
      'ok',
    );
    expect(reasonOf(() => prompt([image(unpaddedPng), image(wrappedPng), image(urlSafeJpeg, 'image/jpeg')]))).toBe(
      'ok',
    );
  });
});

describe('the permission mode input', () => {
  it('takes exactly {mode, up, temp?}: one of ask, auto, bypass, whether it is a move up, the temp folder named', () => {
    for (const mode of ['ask', 'auto', 'bypass'] as const)
      for (const up of [true, false]) expect(permissionModeInput({ mode, up })).toEqual({ mode, up, temp: null });
    expect(permissionModeInput({ mode: 'auto', up: true, temp: '/tmp' })).toEqual({
      mode: 'auto',
      up: true,
      temp: '/tmp',
    });
    for (const body of [
      {},
      { mode: 'Bypass', up: true },
      { mode: 'yolo', up: true },
      { mode: true, up: true },
      { mode: ['bypass'], up: true },
      { mode: 'bypass', up: true, on: true },
      { on: true },
      { on: true, up: true },
      { mode: 'bypass' },
      { mode: 'ask' },
      { up: true },
      { mode: 'bypass', up: 'true' },
      { mode: 'bypass', up: 1 },
      { mode: 'bypass', up: null },
      { mode: 'bypass', up: [true] },
      { mode: 'auto', up: true, temp: '' },
      { mode: 'auto', up: true, temp: true },
      { mode: 'auto', up: true, temp: null },
      { mode: 'auto', up: true, temp: 'x'.repeat(4097) },
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

describe('the rewind inputs', () => {
  // Mutation: let `rewindInput` take any mode → the agent's own `conversation_only` passes.
  // Mutation: let the plan go without `ts` → a plan names an index the page may no longer show there.
  it('the plan takes {entry, ts} (a whole number from 0 and the entry time); the commit takes {token, mode} of the three modes', () => {
    expect(rewindPlanInput({ entry: 0, ts: 1_700_000_000_000 })).toEqual({ entry: 0, ts: 1_700_000_000_000 });
    expect(rewindPlanInput({ entry: 12, ts: 0 })).toEqual({ entry: 12, ts: 0 });
    for (const mode of ['all', 'conversation', 'files'] as const) {
      expect(rewindInput({ token: '3:12', mode })).toEqual({ token: '3:12', mode });
    }
    const refused = (read: (body: unknown) => unknown, body: unknown): string => {
      try {
        read(body);
        return 'ok';
      } catch (error) {
        return (error as { code?: string }).code ?? 'thrown';
      }
    };
    for (const body of [
      { entry: -1, ts: 1 },
      { entry: 1.5, ts: 1 },
      { entry: '1', ts: 1 },
      { entry: 2 ** 60, ts: 1 },
      { entry: 1 },
      { entry: 1, ts: '1' },
      { entry: 1, ts: Number.NaN },
      {},
      { entry: 1, ts: 1, x: 1 },
      null,
    ]) {
      expect([body, refused(rewindPlanInput, body)]).toEqual([body, 'E_BAD_REQUEST']);
    }
    for (const body of [
      { token: 't', mode: 'conversation_only' },
      { token: 't', mode: 'files_only' },
      { token: 't', mode: 'ALL' },
      { token: '', mode: 'conversation' },
      { token: 'x'.repeat(65), mode: 'conversation' },
      { token: 't' },
      { mode: 'conversation' },
      { token: 't', mode: 'conversation', dryRun: true },
      { token: 1, mode: 'conversation' },
    ]) {
      expect([body, refused(rewindInput, body)]).toEqual([body, 'E_BAD_REQUEST']);
    }
  });
});

describe('the fork input', () => {
  // Mutation: make `entry` required → the whole-conversation fork body `{}` is refused.
  // Mutation: let `{entry}` through without its time → the message is named by its place only.
  it('takes {} (the whole conversation) or {entry, ts} (a whole number from 0 and the entry’s time), nothing else', () => {
    expect(forkInput({})).toEqual({});
    expect(forkInput({ entry: 0, ts: 1_700_000_000_000 })).toEqual({ at: { entry: 0, ts: 1_700_000_000_000 } });
    expect(forkInput({ entry: 7, ts: 5 })).toEqual({ at: { entry: 7, ts: 5 } });
    for (const body of [
      { entry: 1 },
      { ts: 5 },
      { entry: -1, ts: 5 },
      { entry: 1.5, ts: 5 },
      { entry: '1', ts: 5 },
      { entry: 1, ts: Number.NaN },
      { entry: 1, ts: '5' },
      { entry: 1, ts: 5, target: 1 },
      { title: 'x' },
      null,
    ]) {
      let code = 'ok';
      try {
        forkInput(body);
      } catch (error) {
        code = (error as { code?: string }).code ?? 'thrown';
      }
      expect([body, code]).toEqual([body, 'E_BAD_REQUEST']);
    }
  });
});

describe('the goal input', () => {
  // Mutation: let any string through as the action → `{action: 'status'}` is accepted.
  it('takes exactly {action: pause | resume | clear}', () => {
    for (const action of ['pause', 'resume', 'clear'] as const) expect(goalInput({ action })).toEqual({ action });
    for (const body of [
      {},
      { action: 'status' },
      { action: '/goal pause' },
      { action: 'Pause' },
      { action: 1 },
      { action: 'pause', text: '/goal pause' },
      null,
    ]) {
      let code = 'ok';
      try {
        goalInput(body);
      } catch (error) {
        code = (error as { code?: string }).code ?? 'thrown';
      }
      expect([body, code]).toEqual([body, 'E_BAD_REQUEST']);
    }
  });
});

describe('the interaction answer input', () => {
  it('takes {optionId}, {optionId: reject-once, note} with at most 4,000 characters, or {decision, feedback?}', () => {
    expect(interactionInput({ optionId: 'allow-once' })).toEqual({ optionId: 'allow-once' });
    expect(interactionInput({ optionId: 'reject-once', note: 'use pnpm, not npm' })).toEqual({
      optionId: 'reject-once',
      note: 'use pnpm, not npm',
    });
    expect(interactionInput({ optionId: 'reject-once', note: '  ' })).toEqual({ optionId: 'reject-once', note: '  ' });
    const longest = 'x'.repeat(MAX_FEEDBACK_CHARS);
    expect(interactionInput({ optionId: 'reject-once', note: longest })).toEqual({
      optionId: 'reject-once',
      note: longest,
    });
    expect(interactionInput({ decision: 'rejected', feedback: 'more tests' })).toEqual({
      decision: 'rejected',
      feedback: 'more tests',
    });
    expect(interactionInput({ never: 'short' })).toEqual({ never: 'short' });
    expect(interactionInput({ never: 'exact' })).toEqual({ never: 'exact' });
    for (const body of [
      { never: 'all' },
      { never: 1 },
      { never: 'short', optionId: 'reject-once' },
      { never: 'short', note: 'x' },
      { optionId: 'reject-once', note: 'x'.repeat(MAX_FEEDBACK_CHARS + 1) },
      { optionId: 'allow-once', note: 'use pnpm' },
      { optionId: 'always-allow', note: 'use pnpm' },
      { optionId: 'reject-once', note: 7 },
      { optionId: 'reject-once', note: null },
      { optionId: 'reject-once', note: '\ud800' },
      { optionId: 'reject-once', note: 'x', feedback: 'y' },
      { decision: 'rejected', note: 'x' },
      { note: 'x' },
    ]) {
      let code = 'ok';
      try {
        interactionInput(body);
      } catch (error) {
        code = (error as { code?: string }).code ?? 'thrown';
      }
      expect([body, code]).toEqual([body, 'E_BAD_REQUEST']);
    }
  });
});

describe('the queue routes', () => {
  it('prompt takes whenBusy "queue" or "now"; a row patch is exactly {text} or {editing}; Send now exactly {how: "now"}', () => {
    expect(promptInput({ text: 'hi', whenBusy: 'queue' })).toEqual({ text: 'hi', whenBusy: 'queue' });
    expect(promptInput({ text: 'hi', whenBusy: 'now' })).toEqual({ text: 'hi', whenBusy: 'now' });
    expect(promptInput({ text: 'hi' })).toEqual({ text: 'hi' });
    for (const whenBusy of ['later', 'Now', '']) {
      expect(() => promptInput({ text: 'hi', whenBusy })).toThrow('Field "whenBusy" must be "queue" or "now".');
    }
    expect(queueSendInput({ how: 'now' })).toEqual({ how: 'now' });
    for (const body of [{}, { how: 'later' }, { how: 'now', text: 'x' }, { how: true }, ['now'], null]) {
      expect(() => queueSendInput(body)).toThrow();
    }
    expect(() => promptInput({ text: 'hi', whenBusy: true })).toThrow('wrong type');
    expect(queuePatchInput({ text: 'new words' })).toEqual({ text: 'new words' });
    expect(queuePatchInput({ editing: true })).toEqual({ editing: true });
    expect(queuePatchInput({ editing: false })).toEqual({ editing: false });
    const refused: unknown[] = [
      {},
      { text: 'a', editing: true },
      { editing: 'yes' },
      { text: 3 },
      { text: '   ' },
      { text: 'a', plan: true },
      { how: 'now' },
      ['a'],
    ];
    for (const body of refused) {
      let code: string | undefined;
      try {
        queuePatchInput(body);
      } catch (error) {
        code = (error as { code?: string }).code;
      }
      expect([body, code]).toEqual([body, 'E_BAD_REQUEST']);
    }
    expect(() => queuePatchInput({ text: '/yolo' })).toThrow('No such command here.');
  });

  it('queues with 202 {queued} while an answer runs; edits, removes, clears and releases through their routes', async () => {
    const gate = join(base, 'gate');
    start('chat-basic', (scenario) => {
      scenario.prompts = [
        { match: 'hold', steps: [{ holdOn: gate }, { text: 'ok', chunk: 2 }, { end: { stopReason: 'end_turn' } }] },
        { match: '', steps: [{ text: 'ok', chunk: 2 }, { end: { stopReason: 'end_turn' } }] },
      ];
    });
    const conversation = await created('hold on');
    const id = conversation.id;
    const w = watch(id);
    await until(
      () => w.events.some((e) => (e.type === 'turn' || e.type === 'sync') && e.turn?.phase === 'running'),
      5_000,
    );
    const path = `/api/chat/conversations/${id}`;
    const queuedOf = async (text: string): Promise<string> => {
      const response = await call('POST', `${path}/prompt`, { text, whenBusy: 'queue' });
      expect(response.ok && response.status).toBe(202);
      return (response as { body: { queued: string } }).body.queued;
    };
    const a = await queuedOf('row a');
    const b = await queuedOf('row b');
    expect(failureOf(await call('POST', `${path}/prompt`, { text: 'no queue' }))).toMatchObject({
      code: 'E_CONFLICT',
      reason: 'conversation-busy',
    });
    const lastQueue = () => {
      const events = w.events.filter((e) => e.type === 'queue');
      const last = events.at(-1);
      return last?.type === 'queue' ? last.queue : null;
    };
    expect(lastQueue()?.items.map((item) => item.text)).toEqual(['row a', 'row b']);

    for (const [method, route, body] of [
      ['PATCH', `${path}/queue/${a}`, { text: 'a', extra: 1 }],
      ['PATCH', `${path}/queue/${a}`, { how: 'now' }],
      ['DELETE', `${path}/queue/${a}`, { force: true }],
      ['DELETE', `${path}/queue`, { all: true }],
      ['POST', `${path}/queue/release`, { now: true }],
    ] as const) {
      expect([method, route, failureOf(await call(method, route, body)).code]).toEqual([
        method,
        route,
        'E_BAD_REQUEST',
      ]);
    }
    expect(failureOf(await call('PATCH', `${path}/queue/no-such-row`, { text: 'x' })).code).toBe('E_NOT_FOUND');
    expect(failureOf(await call('DELETE', `${path}/queue/bad id`)).code).toBe('E_NOT_FOUND');

    const edited = await call('PATCH', `${path}/queue/${a}`, { text: 'row a, edited' });
    expect(edited.ok && edited.body).toEqual({});
    const removed = await call('DELETE', `${path}/queue/${b}`);
    expect(removed.ok && removed.body).toEqual({});
    expect(lastQueue()?.items.map((item) => item.text)).toEqual(['row a, edited']);

    const cancelled = await call('POST', `${path}/cancel`);
    expect(cancelled.ok && cancelled.body).toEqual({ cancelled: true });
    await w.ended();
    await until(() => lastQueue()?.held !== null, 5_000);
    expect(lastQueue()?.held).toEqual({ why: 'stopped' });
    const released = await call('POST', `${path}/queue/release`);
    expect(released.ok && released.body).toEqual({});
    await until(() => sent().filter((m) => m.method === 'session/prompt').length === 2, 5_000);
    const texts = sent()
      .filter((m) => m.method === 'session/prompt')
      .map((m) => ((m.params as Json).prompt as Json[])[0]!.text);
    expect(texts).toEqual(['hold on', 'row a, edited']);
    await until(() => lastQueue()?.items.length === 0, 5_000);
    await settled(id);

    const cleared = await call('DELETE', `${path}/queue`);
    expect(cleared.ok && cleared.body).toEqual({});
    w.off();
  }, 20_000);

  it('Send now: exactly {how: "now"}; Chat answers 409 and sends nothing; a gone row is 404', async () => {
    const gate = join(base, 'gate');
    start('chat-basic', (scenario) => {
      scenario.prompts = [
        { match: 'hold', steps: [{ holdOn: gate }, { text: 'ok', chunk: 2 }, { end: { stopReason: 'end_turn' } }] },
        { match: '', steps: [{ text: 'ok', chunk: 2 }, { end: { stopReason: 'end_turn' } }] },
      ];
    });
    const conversation = await created('hold on');
    const id = conversation.id;
    const w = watch(id);
    await until(
      () => w.events.some((e) => (e.type === 'turn' || e.type === 'sync') && e.turn?.phase === 'running'),
      5_000,
    );
    const path = `/api/chat/conversations/${id}`;
    const queued = await call('POST', `${path}/prompt`, { text: 'row a', whenBusy: 'queue' });
    const a = (queued as { body: { queued: string } }).body.queued;
    for (const body of [undefined, {}, { how: 'later' }, { how: 'now', extra: 1 }]) {
      expect([body, failureOf(await call('POST', `${path}/queue/${a}/send`, body)).code]).toEqual([
        body,
        'E_BAD_REQUEST',
      ]);
    }
    expect(failureOf(await call('POST', `${path}/queue/${a}/send`, { how: 'now' }))).toMatchObject({
      code: 'E_CONFLICT',
      reason: 'steer-work-only',
    });
    expect(failureOf(await call('POST', `${path}/prompt`, { text: 'now', whenBusy: 'now' }))).toMatchObject({
      code: 'E_CONFLICT',
      reason: 'steer-work-only',
    });
    expect(failureOf(await call('POST', `${path}/queue/bad id/send`, { how: 'now' })).code).toBe('E_NOT_FOUND');
    expect(sent().some((m) => m.method === '_x.ai/interject')).toBe(false);
    await call('POST', `${path}/cancel`);
    await w.ended();
    w.off();
  }, 20_000);
});

describe('the task stop route (T1)', () => {
  function startTasks(faults: Json = {}): string {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start(
      'work-plan',
      (scenario) => {
        scenario.prompts = [
          {
            match: 'Start',
            steps: [{ background: { wake: false } }, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }],
          },
          { match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] },
        ];
        scenario.faults = { ...(scenario.faults as Json | undefined), ...faults };
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    return project;
  }

  /** The task id on the stream's entries, once the background step named it. */
  function taskOf(events: readonly ConversationEvent[]): string | null {
    for (const e of events) {
      const list =
        e.type === 'sync'
          ? e.entries
          : e.type === 'ops'
            ? e.ops.flatMap((op) => ('entry' in op ? [op.entry] : []))
            : [];
      for (const entry of list) if (entry.kind === 'tool' && entry.task !== undefined) return entry.task.id;
    }
    return null;
  }

  // Mutation: answer `not-live` as 503 → the page says "Could not stop" for a command that already ended.
  it('200 sends the kill and the end arrives; 404 for a task that is not live or a bad id; a body is refused', async () => {
    const project = startTasks();
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Start it', cwd: project });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => taskOf(w.events) !== null, 5_000);
    await settled(id);
    const task = taskOf(w.events)!;
    const path = `/api/chat/conversations/${id}/tasks`;

    expect(failureOf(await call('POST', `${path}/${task}/stop`, { now: true })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', `${path}/no-such-task/stop`))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such task.',
    });
    expect(failureOf(await call('POST', `${path}/bad.id/stop`)).code).toBe('E_NOT_FOUND');
    expect(sent().some((m) => m.method === '_x.ai/task/kill')).toBe(false);

    const stopped = await call('POST', `${path}/${task}/stop`);
    expect(stopped.ok && stopped.body).toEqual({});
    expect(sent().find((m) => m.method === '_x.ai/task/kill')?.params).toEqual({
      sessionId: id,
      taskId: task,
      source: 'clientUi',
    });
    await until(
      () =>
        w.events.some(
          (e) =>
            e.type === 'ops' &&
            e.ops.some((op) => 'entry' in op && op.entry.kind === 'tool' && op.entry.task?.killed === true),
        ),
      5_000,
    );
    expect(w.events.some((e) => e.type === 'activity' && e.live.tasks.some((t) => t.stopping))).toBe(true);
    expect(failureOf(await call('POST', `${path}/${task}/stop`)).code).toBe('E_NOT_FOUND');
    w.off();
  }, 30_000);

  // Mutation: leave the tasks out of `sync.live` → a reopen shows Stop again on a task whose stop is out.
  it('a reopen while the stop waits for the end line reads stopping from sync', async () => {
    const project = startTasks({ killReply: 'already_exited' });
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Start it', cwd: project });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => taskOf(w.events) !== null, 5_000);
    await settled(id);
    const task = taskOf(w.events)!;
    w.off();
    const stopped = await call('POST', `/api/chat/conversations/${id}/tasks/${task}/stop`);
    expect(stopped.ok && stopped.body).toEqual({});
    const again = watch(id);
    await until(() => again.events.some((e) => e.type === 'sync'), 5_000);
    const sync = again.events.find((e) => e.type === 'sync');
    expect(sync?.type === 'sync' ? sync.live.tasks : null).toEqual([
      { id: task, kind: 'command', stopping: true, events: 0, recent: [] },
    ]);
    again.off();
  }, 30_000);

  it('a Chat conversation has no tasks: 404', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    expect(failureOf(await call('POST', `/api/chat/conversations/${conversation.id}/tasks/task-1/stop`))).toMatchObject(
      { code: 'E_NOT_FOUND', message: 'No such task.' },
    );
    expect(sent().some((m) => m.method === '_x.ai/task/kill')).toBe(false);
  }, 30_000);

  it('503 when the agent answers with an error, and the task can be stopped again', async () => {
    const project = startTasks({ killReply: 'error' });
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Start it', cwd: project });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => taskOf(w.events) !== null, 5_000);
    await settled(id);
    const failed = await call('POST', `/api/chat/conversations/${id}/tasks/${taskOf(w.events)!}/stop`);
    expect(failureOf(failed)).toMatchObject({ code: 'E_UNAVAILABLE', message: 'The agent could not be reached.' });
    const lastLive = [...w.events].reverse().find((e) => e.type === 'activity');
    expect(lastLive?.type === 'activity' && lastLive.live.tasks.map((t) => t.stopping)).toEqual([false]);
    w.off();
  }, 30_000);
});

describe('the subagent routes: Stop and Message', () => {
  const CHILD = 'helper-1';

  function startHelper(faults: Json = {}): string {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start(
      'work-plan',
      (scenario) => {
        scenario.prompts = [
          {
            match: 'Start',
            steps: [
              { helper: { child: CHILD, wake: false } },
              { text: 'Started.' },
              { end: { stopReason: 'end_turn' } },
            ],
          },
          { match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] },
        ];
        scenario.faults = { ...(scenario.faults as Json | undefined), ...faults };
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    return project;
  }

  /** A Work conversation whose first turn started the helper, with its stream. */
  async function running(project: string): Promise<{ id: string; w: Watch }> {
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Start it', cwd: project });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => w.events.some((e) => e.type === 'activity' && e.live.agents.length > 0), 5_000);
    await settled(id);
    return { id, w };
  }

  const subagentsOf = (events: readonly ConversationEvent[]): unknown => {
    const last = [...events].reverse().find((e) => e.type === 'activity' || e.type === 'sync');
    return last?.type === 'activity' || last?.type === 'sync' ? last.live.agents : null;
  };

  // Mutation: answer `not-live` as 503 → the page says "Could not stop" for a subagent that already ended.
  // Mutation: drop `subagents` from the conversation's copy of the live facts → the page never shows Stop.
  it('stop: 200 sends the cancel and the end arrives; 404 for a subagent not running here or a bad id; a body is refused', async () => {
    const { id, w } = await running(startHelper());
    expect(subagentsOf(w.events)).toEqual([
      { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'on' },
    ]);
    const path = `/api/chat/conversations/${id}/subagents`;
    expect(failureOf(await call('POST', `${path}/${CHILD}/stop`, { now: true })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', `${path}/stranger/stop`))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such subagent.',
    });
    expect(failureOf(await call('POST', `${path}/bad.id/stop`)).code).toBe('E_NOT_FOUND');
    expect(sent().some((m) => m.method === '_x.ai/subagent/cancel')).toBe(false);

    const stopped = await call('POST', `${path}/${CHILD}/stop`);
    expect(stopped.ok && stopped.body).toEqual({});
    expect(sent().find((m) => m.method === '_x.ai/subagent/cancel')?.params).toEqual({ subagentId: CHILD });
    expect(w.events.some((e) => e.type === 'activity' && e.live.agents.some((s) => s.stopping))).toBe(true);
    await until(
      () =>
        w.events.some(
          (e) =>
            e.type === 'ops' &&
            e.ops.some((op) => 'entry' in op && op.entry.kind === 'subagent' && op.entry.status === 'cancelled'),
        ),
      5_000,
    );
    await until(() => JSON.stringify(subagentsOf(w.events)) === '[]', 5_000);
    expect(failureOf(await call('POST', `${path}/${CHILD}/stop`)).code).toBe('E_NOT_FOUND');
    w.off();
  }, 30_000);

  // Mutation: let a blank text through → an empty message goes to the agent, which answers `limit`.
  // Mutation: answer `gone` as a result → the page keeps the popover open for a subagent that ended.
  it('message: 200 names the result; 400 for a blank or wrong body; 404 when it is not running here', async () => {
    const { id, w } = await running(startHelper({ messageReplies: ['not_active'] }));
    const path = `/api/chat/conversations/${id}/subagents/${CHILD}/message`;
    expect(failureOf(await call('POST', path, { text: '  \n ' })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path, { text: 'hi', queue: true })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path)).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path, { text: 'x'.repeat(100_001) })).code).toBe('E_PAYLOAD_TOO_LARGE');
    expect(sent().some((m) => m.method === '_x.ai/subagent/message')).toBe(false);

    const notReady = await call('POST', path, { text: '/compact is plain text here' });
    expect(notReady.ok && notReady.body).toEqual({ result: 'not-ready' });
    const ok = await call('POST', path, { text: '/compact is plain text here' });
    expect(ok.ok && ok.body).toEqual({ result: 'sent' });
    expect(
      sent()
        .filter((m) => m.method === '_x.ai/subagent/message')
        .map((m) => m.params),
    ).toEqual([
      {
        sessionId: id,
        agentAddress: `addr-${CHILD}`,
        content: [{ type: 'text', text: '/compact is plain text here' }],
      },
      {
        sessionId: id,
        agentAddress: `addr-${CHILD}`,
        content: [{ type: 'text', text: '/compact is plain text here' }],
      },
    ]);
    expect(
      failureOf(await call('POST', `/api/chat/conversations/${id}/subagents/stranger/message`, { text: 'hi' })),
    ).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such subagent.',
    });
    w.off();
  }, 30_000);

  it('a Chat conversation has no subagents: 404 for both, and nothing is sent', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    const path = `/api/chat/conversations/${conversation.id}/subagents/${CHILD}`;
    expect(failureOf(await call('POST', `${path}/stop`))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such subagent.',
    });
    expect(failureOf(await call('POST', `${path}/message`, { text: 'hi' }))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such subagent.',
    });
    expect(sent().some((m) => m.method === '_x.ai/subagent/cancel' || m.method === '_x.ai/subagent/message')).toBe(
      false,
    );
  }, 30_000);

  it('503 when the cancel gets an error reply; the subagent can be stopped again', async () => {
    const { id, w } = await running(startHelper({ cancelReply: 'error' }));
    const failed = await call('POST', `/api/chat/conversations/${id}/subagents/${CHILD}/stop`);
    expect(failureOf(failed)).toMatchObject({ code: 'E_UNAVAILABLE', message: 'The agent could not be reached.' });
    expect(subagentsOf(w.events)).toEqual([
      { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'on' },
    ]);
    w.off();
  }, 30_000);
});

describe('the subagent watch routes: the Agent view', () => {
  const CHILD = 'helper-watched';

  type AgentEvent = Extract<ConversationEvent, { type: 'agent-entries' }>;
  const agentEvents = (events: readonly ConversationEvent[]): AgentEvent[] =>
    events.filter((e): e is AgentEvent => e.type === 'agent-entries');

  /** The list the window holds: the watch's answer, then each of its events in order. */
  function windowList(answer: AgentWatchAnswer, events: readonly ConversationEvent[]): TranscriptEntry[] {
    let entries = [...answer.entries];
    for (const event of agentEvents(events)) {
      if (event.window !== 'win-1' || event.childId !== CHILD) continue;
      if (event.stored !== undefined) entries = [...event.stored];
      for (const op of event.ops) applyOp(entries, [], op);
    }
    return entries;
  }

  // Mutation: drop the child's lines in the host (no `watches.line`) → the message streamed after the watch never shows.
  // Mutation: no end swap at the agent's end (no `agentsSettled` after the tracker) → the stored list never comes.
  it('answers the file and later lines live, redacted, then the stored list once the agent ended', async () => {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    const more = join(base, 'more.gate');
    const end = join(base, 'end.gate');
    start(
      'work-plan',
      (scenario) => {
        scenario.persist = true;
        scenario.prompts = [
          {
            match: 'Start',
            steps: [
              {
                helper: {
                  child: CHILD,
                  wake: false,
                  lines: [
                    { prompt: 'Find the flaky test' },
                    { thought: 'Looking at the tests', chunks: 3 },
                    { tool: 'call-1', title: 'Read src/a.test.ts' },
                  ],
                  more: [{ message: `The token is ${INFERENCE_TOKEN}. Found it.`, chunks: 5 }],
                  moreOn: more,
                  endOn: end,
                  endLagMs: 400,
                },
              },
              { text: 'Started.' },
              { end: { stopReason: 'end_turn' } },
            ],
          },
          { match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Start it', cwd: project });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => w.events.some((e) => e.type === 'activity' && e.live.agents.length > 0), 5_000);
    await settled(id);
    const path = `/api/chat/conversations/${id}/subagents/${CHILD}/watch`;

    expect(failureOf(await call('POST', path)).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path, { window: 'not an id' })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', path, { window: 'win-1', extra: 1 })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('DELETE', path)).code).toBe('E_BAD_REQUEST');
    expect(
      failureOf(await call('POST', `/api/chat/conversations/${id}/subagents/stranger/watch`, { window: 'win-1' })),
    ).toMatchObject({ code: 'E_NOT_FOUND', message: 'No such subagent.' });

    const first = await call('POST', path, { window: 'win-1' });
    if (!first.ok) throw new Error(first.message);
    const answer = first.body as AgentWatchAnswer;
    expect({ childId: answer.childId, attempts: answer.attempts, live: answer.live }).toEqual({
      childId: CHILD,
      attempts: [CHILD],
      live: true,
    });
    expect(answer.entries.map((e) => e.kind)).toEqual(['user', 'thinking', 'tool']);
    expect(answer.entries[1]?.kind === 'thinking' && answer.entries[1].text).toBe('Looking at the tests');

    // Another conversation cannot close this window's view of an agent it did not start.
    const other = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Hello', cwd: project });
    if (!other.ok) throw new Error(other.message);
    const otherId = (other.body as { conversation: ConversationInfo }).conversation.id;
    await settled(otherId);
    expect(
      failureOf(await call('DELETE', `/api/chat/conversations/${otherId}/subagents/${CHILD}/watch?window=win-1`)),
    ).toMatchObject({ code: 'E_NOT_FOUND', message: 'No such subagent.' });

    writeFileSync(more, '');
    await until(() => windowList(answer, w.events).some((e) => e.kind === 'assistant'), 5_000);
    await until(() => {
      const said = windowList(answer, w.events).find((e) => e.kind === 'assistant');
      return said?.kind === 'assistant' && said.text.endsWith('Found it.');
    }, 5_000);
    const said = windowList(answer, w.events).find((e) => e.kind === 'assistant');
    expect(said?.kind === 'assistant' && said.text).toBe(`The token is ${REDACTED}. Found it.`);

    writeFileSync(end, '');
    await until(() => agentEvents(w.events).some((e) => e.stored !== undefined), 10_000);
    const stored = agentEvents(w.events).find((e) => e.stored !== undefined)!;
    expect(stored.window).toBe('win-1');
    expect(stored.stored!.map((e) => e.kind)).toEqual(['user', 'thinking', 'tool', 'assistant', 'turn-end']);
    expect(JSON.stringify(w.events)).not.toContain(INFERENCE_TOKEN);

    const again = await call('POST', path, { window: 'win-1' });
    if (!again.ok) throw new Error(again.message);
    const reopened = again.body as AgentWatchAnswer;
    expect(reopened.live).toBe(false);
    expect(reopened.entries).toEqual(stored.stored);
    const closed = await call('DELETE', `${path}?window=win-1`);
    expect(closed.ok && closed.body).toEqual({});
    w.off();
  }, 30_000);

  it('a Chat conversation has no agents to watch: 404', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    const path = `/api/chat/conversations/${conversation.id}/subagents/${CHILD}/watch`;
    expect(failureOf(await call('POST', path, { window: 'win-1' }))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such subagent.',
    });
  }, 30_000);
});

describe('the schedule delete route (T3)', () => {
  function startSchedule(faults: Json = {}): string {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start(
      'work-plan',
      (scenario) => {
        scenario.persist = true;
        scenario.prompts = [
          {
            match: 'Schedule',
            steps: [{ schedule: { taskId: 's1' } }, { text: 'Scheduled.' }, { end: { stopReason: 'end_turn' } }],
          },
          { match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] },
        ];
        scenario.faults = { ...(scenario.faults as Json | undefined), ...faults };
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    return project;
  }

  async function scheduledConversation(project: string): Promise<{ id: string; w: Watch }> {
    const made = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'Schedule it', cwd: project });
    if (!made.ok) throw new Error(made.message);
    const id = (made.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => scheduleEnded(w.events) === false, 5_000);
    await settled(id);
    return { id, w };
  }

  /** The schedule row's state on the stream: null before it shows, then whether it ended. */
  function scheduleEnded(events: readonly ConversationEvent[]): boolean | null {
    let ended: boolean | null = null;
    for (const e of events) {
      const list =
        e.type === 'sync'
          ? e.entries
          : e.type === 'ops'
            ? e.ops.flatMap((op) => ('entry' in op ? [op.entry] : []))
            : [];
      for (const entry of list) {
        if (entry.kind === 'tool' && entry.schedule?.id === 's1') ended = entry.schedule.endTs !== undefined;
      }
    }
    return ended;
  }

  const deletes = (): Json[] => sent().filter((m) => m.method === '_x.ai/scheduler/delete');

  // Mutation: answer `gone` as 503 → the page says "Could not delete" for a schedule the agent already removed.
  it('200 deletes it and its row ends; 404 for a schedule the thread does not list, a bad id, or once gone', async () => {
    const project = startSchedule();
    const { id, w } = await scheduledConversation(project);
    const path = `/api/chat/conversations/${id}/schedules`;
    expect(failureOf(await call('POST', `${path}/s1/delete`, { now: true })).code).toBe('E_BAD_REQUEST');
    expect(failureOf(await call('POST', `${path}/no-such-schedule/delete`))).toMatchObject({
      code: 'E_NOT_FOUND',
      message: 'No such schedule.',
    });
    expect(failureOf(await call('POST', `${path}/bad.id/delete`)).code).toBe('E_NOT_FOUND');
    expect(deletes()).toEqual([]);

    const deleted = await call('POST', `${path}/s1/delete`);
    expect(deleted.ok && deleted.body).toEqual({});
    expect(deletes().map((m) => m.params)).toEqual([{ sessionId: id, taskId: 's1' }]);
    await until(() => scheduleEnded(w.events) === true, 5_000);
    expect(w.events.some((e) => e.type === 'activity' && e.live.schedules.some((s) => s.deleting === 'removing'))).toBe(
      true,
    );
    expect(failureOf(await call('POST', `${path}/s1/delete`)).code).toBe('E_NOT_FOUND');
    w.off();
  }, 30_000);

  // Mutation: skip the load for a conversation the agent does not hold → the agent answers "session not found" (503).
  it('a paused schedule: the conversation loads first, then the schedule is deleted', async () => {
    const project = startSchedule();
    const { id, w } = await scheduledConversation(project);
    w.off();
    expect((await call('POST', '/api/chat/agent', { action: 'restart', mode: 'work' })).ok).toBe(true);
    const again = watch(id);
    await until(() => again.events.some((e) => e.type === 'sync' && !e.conversation.hot), 5_000);
    const deleted = await call('POST', `/api/chat/conversations/${id}/schedules/s1/delete`);
    expect(deleted.ok && deleted.body).toEqual({});
    const order = sent()
      .map((m) => m.method)
      .filter((m) => m === 'session/load' || m === '_x.ai/scheduler/delete');
    expect(order).toEqual(['session/load', '_x.ai/scheduler/delete']);
    expect(
      again.events.some((e) => e.type === 'activity' && e.live.schedules.some((s) => s.deleting === 'opening')),
    ).toBe(true);
    await until(() => scheduleEnded(again.events) === true, 5_000);
    await until(() => again.events.some((e) => e.type === 'conversation' && e.conversation.hot), 5_000);
    again.off();
  }, 30_000);

  it('a Chat conversation has no schedules: 404', async () => {
    start('chat-basic');
    const conversation = await created();
    await settled(conversation.id);
    expect(
      failureOf(await call('POST', `/api/chat/conversations/${conversation.id}/schedules/s1/delete`)),
    ).toMatchObject({ code: 'E_NOT_FOUND', message: 'No such schedule.' });
    expect(deletes()).toEqual([]);
  }, 30_000);

  it('503 when the agent answers with an error, and the row is back', async () => {
    const project = startSchedule({ scheduleDeleteReply: 'error' });
    const { id, w } = await scheduledConversation(project);
    const failed = await call('POST', `/api/chat/conversations/${id}/schedules/s1/delete`);
    expect(failureOf(failed)).toMatchObject({ code: 'E_UNAVAILABLE', message: 'The agent could not be reached.' });
    const lastLive = [...w.events].reverse().find((e) => e.type === 'activity');
    expect(lastLive?.type === 'activity' && lastLive.live.schedules).toEqual([
      { id: 's1', deleting: null, firing: null },
    ]);
    expect(scheduleEnded(w.events)).toBe(false);
    w.off();
  }, 30_000);
});

describe('Chat routes — slash commands', () => {
  const refusal = async (path: string, text: string): Promise<[string, string | undefined]> => {
    const response = await call('POST', path, { text });
    return [text, response.ok ? `ok ${response.status}` : failureOf(response).reason];
  };

  it('Work sends its built-ins and the listed skills as typed; app, blocked, hidden, unknown and unlisted are refused', async () => {
    const home = join(base, 'home');
    const project = join(home, 'code', 'app');
    mkdirSync(project, { recursive: true });
    start(
      'work-length',
      (scenario) => {
        scenario.prompts = [{ match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] }];
        scenario.session = {
          ...(scenario.session as Json),
          commands: { skills: [{ name: 'commit', description: 'Write a commit message' }] },
        };
      },
      { baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home } },
    );
    const createdReply = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'hello', cwd: project });
    if (!createdReply.ok) throw new Error(createdReply.message);
    const id = (createdReply.body as { conversation: ConversationInfo }).conversation.id;
    await settled(id);
    const path = `/api/chat/conversations/${id}/prompt`;
    for (const text of [
      '/compact',
      '/context now',
      '/rewind',
      '/fork',
      '/recap',
      '/yolo',
      '/status',
      '/hooks-add x',
      '/nothere',
      '/a/b',
    ]) {
      expect(await refusal(path, text)).toEqual([text, 'slash-text']);
    }
    expect(await refusal(path, '/goal ship it')).toEqual(['/goal ship it', 'ok 202']);
    await settled(id);
    expect(await refusal(path, '/commit fix')).toEqual(['/commit fix', 'ok 202']);
    await settled(id);
    const prompts = sent()
      .filter((m) => m.method === 'session/prompt')
      .map((m) => ((m.params as Json).prompt as Json[])[0]!.text);
    expect(prompts).toEqual(['hello', '/goal ship it', '/commit fix']);
  }, 30_000);

  it('Chat sends no agent command, and a refused first message makes no conversation', async () => {
    start('chat-basic');
    const conversation = await created('hello');
    await settled(conversation.id);
    const path = `/api/chat/conversations/${conversation.id}/prompt`;
    for (const text of ['/goal ship it', '/loop 5m check', '/compact', '/workflow runs']) {
      expect(await refusal(path, text)).toEqual([text, 'slash-text']);
    }
    const first = await call('POST', '/api/chat/conversations', { mode: 'chat', text: '/goal x' });
    expect(failureOf(first).reason).toBe('slash-text');
    const list = await call('GET', '/api/chat/conversations?mode=chat');
    expect(list.ok && (list.body as { conversations: unknown[] }).conversations).toHaveLength(1);
    expect(sent().filter((m) => m.method === 'session/new')).toHaveLength(1);
  }, 30_000);
});
