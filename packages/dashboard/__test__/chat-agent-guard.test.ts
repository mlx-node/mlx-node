/**
 * The self-check and the post-create guard as pure functions, then every guard scenario against the scripted fake
 * agent: each must be refused before any prompt reaches the agent.
 */

import { describe, expect, it } from 'vite-plus/test';

import { readInitialize } from '../src/chat/acp/messages.js';
import { AgentHostError } from '../src/chat/agent/agent-host.js';
import { categoryCount, checkSession, leadingInteger, selfCheck, type GuardInput } from '../src/chat/agent/guard.js';
import { fakeHost, until } from './helpers/chat-fake.js';

const goodInfo = {
  agentName: 'mlx-chat',
  model: 'm',
  apiBackend: 'Messages',
  toolDefinitionsCount: 2,
  systemPromptTokens: 1,
  toolDefinitionsTokens: 1,
  usageCategories: [
    { label: 'Skills', detail: '1 skill' },
    { label: 'Workflows', detail: '1 workflow' },
    { label: 'MCP servers', detail: '1 server' },
  ],
};

function input(over: Partial<GuardInput> = {}): GuardInput {
  return {
    mode: 'chat',
    reply: { sessionId: 's', currentModelId: 'm', cwd: '/w' },
    info: goodInfo,
    mcpReady: true,
    requestedModelId: 'm',
    cwd: '/w',
    ...over,
  };
}

describe('leadingInteger', () => {
  it('reads the leading digits by hand', () => {
    expect(leadingInteger('1 skill')).toBe(1);
    expect(leadingInteger('12 servers')).toBe(12);
    expect(leadingInteger('skill')).toBeNull();
    expect(leadingInteger('')).toBeNull();
    expect(categoryCount({ ...goodInfo, usageCategories: [] }, 'Skills')).toBe(0);
  });
});

describe('checkSession', () => {
  it('passes the recorded Chat session', () => {
    expect(checkSession(input())).toEqual({ ok: true, agentName: 'mlx-chat', toolCount: 2 });
  });

  it.each([
    ['row 1: info missing', { info: null }, 1],
    ['row 1: another agent', { info: { ...goodInfo, agentName: 'grok-build-plan' } }, 1],
    ['row 2: tool count', { info: { ...goodInfo, toolDefinitionsCount: 41 } }, 2],
    [
      'row 3: two skills',
      {
        info: {
          ...goodInfo,
          usageCategories: [
            { label: 'Skills', detail: '2 skills' },
            { label: 'MCP servers', detail: '1 server' },
          ],
        },
      },
      3,
    ],
    ['row 3: no skill', { info: { ...goodInfo, usageCategories: [{ label: 'MCP servers', detail: '1 server' }] } }, 3],
    ['row 4: never ready', { mcpReady: false }, 4],
    ['row 4: no MCP entry', { info: { ...goodInfo, usageCategories: [{ label: 'Skills', detail: '1 skill' }] } }, 4],
    ['row 5: model fell back', { reply: { sessionId: 's', currentModelId: 'other', cwd: '/w' } }, 5],
    ['row 6: other cwd', { reply: { sessionId: 's', currentModelId: 'm', cwd: '/elsewhere' } }, 6],
    ['row 7: backend', { info: { ...goodInfo, apiBackend: 'ChatCompletions' } }, 7],
  ] as const)('%s', (_name, over, row) => {
    const result = checkSession(input(over as Partial<GuardInput>));
    expect(result).toMatchObject({ ok: false, row });
    if (!result.ok) {
      expect(result.reason).toBe(row === 5 ? 'model-unavailable' : row === 6 ? 'cwd-mismatch' : 'profile-mismatch');
    }
  });

  it('Work expects any agent but mlx-chat and does not compare tools, skills or MCP servers', () => {
    const work = { ...goodInfo, agentName: 'grok-build-plan', toolDefinitionsCount: 20, usageCategories: [] };
    expect(checkSession(input({ mode: 'work', info: work, mcpReady: false }))).toMatchObject({
      ok: true,
      toolCount: 20,
    });
    expect(checkSession(input({ mode: 'work', info: goodInfo }))).toMatchObject({ ok: false, row: 1 });
  });
});

describe('selfCheck', () => {
  const good = {
    protocolVersion: 1,
    agentCapabilities: { loadSession: true, promptCapabilities: { embeddedContext: true } },
    _meta: { 'x.ai/pluginDirs': true, agentVersion: '1.0.45', modelState: { availableModels: [{ modelId: 'm' }] } },
  };

  it('passes the recorded initialize reply', () => {
    expect(selfCheck(readInitialize(good))).toMatchObject({ ok: true, agentVersion: '1.0.45' });
  });

  it('fails each row', () => {
    expect(selfCheck(readInitialize({ ...good, protocolVersion: 2 }))).toEqual({ ok: false, row: 1 });
    expect(selfCheck(readInitialize({ ...good, agentCapabilities: {} }))).toEqual({ ok: false, row: 2 });
    expect(selfCheck(readInitialize({ ...good, _meta: { ...good._meta, 'x.ai/pluginDirs': false } }))).toEqual({
      ok: false,
      row: 3,
    });
    expect(selfCheck(readInitialize({ ...good, _meta: { ...good._meta, agentVersion: 1 } }))).toEqual({
      ok: false,
      row: 4,
    });
    expect(
      selfCheck(readInitialize({ ...good, _meta: { ...good._meta, modelState: { availableModels: [] } } })),
    ).toEqual({
      ok: false,
      row: 5,
    });
  });
});

describe('guard scenarios against the fake agent', () => {
  it.each([
    ['wrong-agent-name', 'profile-mismatch'],
    ['tool-count-41', 'profile-mismatch'],
    ['info-missing', 'profile-mismatch'],
    ['mcp-never-ready', 'profile-mismatch'],
    ['model-fallback', 'model-unavailable'],
    ['cwd-other', 'cwd-mismatch'],
  ] as const)('%s is refused with %s, removed through the agent, and never prompted', async (scenario, code) => {
    const fake = await fakeHost(scenario, { timings: { mcpReadyMs: 300 } });
    try {
      const error = await fake.host.createSession({ modelId: 'qwen3.5-4b' }).catch((e: unknown) => e);
      expect(error).toBeInstanceOf(AgentHostError);
      expect((error as AgentHostError).code).toBe(code);
      await until(() => fake.sent().some((m) => m.method === '_x.ai/session/delete'));
      const methods = fake.sent().map((m) => m.method);
      expect(methods).toContain('session/new');
      expect(methods).not.toContain('session/prompt');
    } finally {
      await fake.close();
    }
  });

  it('mcp-ready-late waits for the ready status and passes', async () => {
    const fake = await fakeHost('mcp-ready-late');
    try {
      const created = await fake.host.createSession({ modelId: 'qwen3.5-4b' });
      expect(created.agentName).toBe('mlx-chat');
      const info = fake.sent().filter((m) => m.method === '_x.ai/session/info');
      expect(info).toHaveLength(1);
    } finally {
      await fake.close();
    }
  });

  it('three profile mismatches in a row block the mode as agent-incompatible', async () => {
    const fake = await fakeHost('wrong-agent-name');
    try {
      for (let i = 0; i < 3; i++) await fake.host.createSession({ modelId: 'qwen3.5-4b' }).catch(() => {});
      expect(fake.host.modeState()).toMatchObject({ state: 'blocked', failure: { code: 'agent-incompatible' } });
      const error = await fake.host.createSession({ modelId: 'qwen3.5-4b' }).catch((e: unknown) => e);
      expect((error as AgentHostError).code).toBe('agent-incompatible');
    } finally {
      await fake.close();
    }
  });

  it('a self-check failure blocks the mode before any session (no-plugin-dirs)', async () => {
    const fake = await fakeHost('no-plugin-dirs');
    try {
      const error = await fake.host.createSession({ modelId: 'qwen3.5-4b' }).catch((e: unknown) => e);
      expect((error as AgentHostError).code).toBe('agent-incompatible');
      expect(fake.host.modeState().state).toBe('blocked');
      expect(fake.sent().map((m) => m.method)).toEqual(['initialize']);
    } finally {
      await fake.close();
    }
  });

  it('an agent with no models reports no-models when the endpoint lists none', async () => {
    const fake = await fakeHost('no-models', { endpointModels: [] });
    try {
      const error = await fake.host.createSession({ modelId: 'qwen3.5-4b' }).catch((e: unknown) => e);
      expect((error as AgentHostError).code).toBe('no-models');
    } finally {
      await fake.close();
    }
  });

  it('a model the process does not list is refused before any session call', async () => {
    const fake = await fakeHost('chat-basic');
    try {
      const error = await fake.host.createSession({ modelId: 'no-such-model' }).catch((e: unknown) => e);
      expect((error as AgentHostError).code).toBe('model-unavailable');
      expect(fake.sent().map((m) => m.method)).toEqual(['initialize']);
    } finally {
      await fake.close();
    }
  });
});
