import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';

import type { ChatConfig, ChatResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { launchClaudeHostOptions } from '../../packages/cli/src/commands/launch-claude/index.js';
import { parseServeArgs } from '../../packages/cli/src/commands/serve.js';
import { CHAT_FAMILY_IDS, THINKING_LEVELS, type ThinkingLevel } from '../../packages/lm/src/family-data.js';
import { createHandler, type HandlerOptions, type PublicModelEntry } from '../../packages/server/src/handler.js';
import { AGENT_HOST_THINKING } from '../../packages/server/src/host/agent-thinking.js';
import { thinkingFor } from '../../packages/server/src/host/thinking-levels.js';
import { mapAnthropicRequest } from '../../packages/server/src/mappers/anthropic-request.js';
import { ModelRegistry } from '../../packages/server/src/registry.js';
import type { AnthropicMessagesRequest } from '../../packages/server/src/types-anthropic.js';

/** Every value the engine takes in `extra_body.reasoning_effort`. */
const WIRE = ['none', 'low', 'medium', 'high', 'xhigh', 'max'] as const;
/** What agent 0.3.4 sends in `output_config.effort` (`to_messages_api`: none and minimal send nothing). */
const OUTPUT_EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'] as const;

const request = (extra: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: 'local',
  max_tokens: 512,
  messages: [{ role: 'user', content: 'hi' }],
  ...extra,
});
const withEffort = (effort: string, extra: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest =>
  request({ ...extra, extra_body: { reasoning_effort: effort } });
/** What agent 0.3.4 adds for a session effort: an adaptive thought and `output_config.effort`. */
const agentEffort = (effort: string): Partial<AnthropicMessagesRequest> =>
  ({
    thinking: { type: 'adaptive', display: 'summarized' },
    output_config: { effort },
  }) as unknown as Partial<AnthropicMessagesRequest>;
/** What agent 0.3.5 sends for a session level (`build_messages_request`): 0.3.4's two fields plus `extra_body.reasoning_effort`; Off sends only that one. */
const agent035 = (level: ThinkingLevel): Partial<AnthropicMessagesRequest> =>
  level === 'none'
    ? { extra_body: { reasoning_effort: level } }
    : { ...agentEffort(level), extra_body: { reasoning_effort: level } };
/** The exact bytes agent 0.3.5 sends for Off and Low: golden in the fork's `messages_tests.rs` (`MLX_NODE_GOLDEN_BODIES`); change both together. */
const AGENT_035_BYTES = {
  none: '{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"extra_body":{"reasoning_effort":"none"}}',
  low: '{"model":"q35","messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral"}}]}],"max_tokens":64,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"low"},"extra_body":{"reasoning_effort":"low"}}',
} as const;
const configJson = (req: AnthropicMessagesRequest, family?: string): string =>
  JSON.stringify(mapAnthropicRequest(req, family).config);

describe('Messages effort: the mapper', () => {
  it('output_config.effort is never read: the config is byte-equal to the same request without it, levels on or off', () => {
    for (const family of [undefined, ...CHAT_FAMILY_IDS]) {
      for (const effort of OUTPUT_EFFORTS) {
        expect(configJson(request(agentEffort(effort)), family), `${family} ${effort}`).toBe(
          configJson(request(), family),
        );
        for (const wire of WIRE) {
          expect(configJson(withEffort(wire, agentEffort(effort)), family)).toBe(configJson(withEffort(wire), family));
        }
      }
    }
  });

  it('without levels, extra_body.reasoning_effort gives the config it gave before levels, for every value', () => {
    for (const wire of WIRE) {
      expect(configJson(withEffort(wire))).toBe(
        JSON.stringify({ reportPerformance: true, reasoningEffort: wire, maxNewTokens: 512 }),
      );
    }
    expect(configJson(request())).toBe(JSON.stringify({ reportPerformance: true, maxNewTokens: 512 }));
  });

  it("with levels, each family's listed level maps through its row; any other value maps as without levels", () => {
    for (const family of CHAT_FAMILY_IDS) {
      for (const wire of WIRE) {
        const setting = thinkingFor(family, wire);
        const want =
          setting === null
            ? { reportPerformance: true, reasoningEffort: wire, maxNewTokens: 512 }
            : { reportPerformance: true, ...setting, maxNewTokens: 512 };
        expect(mapAnthropicRequest(withEffort(wire), family).config, `${family} ${wire}`).toEqual(want);
      }
    }
    expect(mapAnthropicRequest(withEffort('low'), 'qwen3_5').config).toEqual({
      reportPerformance: true,
      reasoningEffort: 'low',
      thinkingTokenBudget: 1024,
      maxNewTokens: 512,
    });
    expect(mapAnthropicRequest(withEffort('low'), 'qwen3_5_moe').config).toEqual({
      reportPerformance: true,
      thinkingTokenBudget: 1024,
      maxNewTokens: 512,
    });
    expect(mapAnthropicRequest(withEffort('high'), 'qwen3_5').config).toEqual({
      reportPerformance: true,
      maxNewTokens: 512,
    });
  });

  it("a request's own budget wins over a level's budget", () => {
    const own = request({ extra_body: { reasoning_effort: 'low', thinking_budget: 2048 } });
    expect(mapAnthropicRequest(own, 'qwen3_5').config.thinkingTokenBudget).toBe(2048);
    const enabled = withEffort('low', { thinking: { type: 'enabled', budget_tokens: 3000 } });
    expect(mapAnthropicRequest(enabled, 'lfm2').config.thinkingTokenBudget).toBe(3000);
  });

  it('a value outside the engine set is refused, levels on or off; minimal included', () => {
    for (const family of [undefined, 'qwen3_5', 'gemma4']) {
      for (const bad of ['minimal', 'bogus', 'HIGH', '']) {
        expect(() => mapAnthropicRequest(withEffort(bad), family), `${family} ${bad}`).toThrow(
          'Invalid extra_body.reasoning_effort',
        );
      }
    }
  });

  it('a forced tool call still turns thinking off over any level', () => {
    const forced = withEffort('high', {
      tools: [{ name: 't', input_schema: { type: 'object', properties: {} } }],
      tool_choice: { type: 'any' },
    });
    expect(mapAnthropicRequest(forced, 'qwen3_5').config).toMatchObject({
      reasoningEffort: 'none',
      thinkingTokenBudget: 0,
    });
  });
});

const RESULT = {
  text: 'ok',
  toolCalls: [],
  thinkingEnabled: true,
  numTokens: 1,
  promptTokens: 1,
  reasoningTokens: 0,
  finishReason: 'stop',
  rawText: 'ok',
  cachedTokens: 0,
} as unknown as ChatResult;

function recordingModel(configs: ChatConfig[]): SessionCapableModel {
  const record = (config: unknown): ChatResult => {
    configs.push({ ...(config as ChatConfig) });
    return RESULT;
  };
  return {
    chatSessionStart: async (_messages: unknown, config: unknown) => record(config),
    chatSessionContinue: async (_messages: unknown, config: unknown) => record(config),
    chatSessionContinueTool: async (_messages: unknown, config: unknown) => record(config),
    resetCaches: async () => undefined,
  } as unknown as SessionCapableModel;
}

/** A model that renders its prompt and has a window, so a generation renders it too (the context clamp). */
function renderingModel(renders: unknown[][]): SessionCapableModel {
  return {
    ...recordingModel([]),
    applyChatTemplate: async (...args: unknown[]) => {
      renders.push(args.slice(2));
      return new Uint32Array(8);
    },
    contextLimits: () => ({
      trainedWindowTokens: 65_536,
      effectiveWindowTokens: 65_536,
      pagedBlockCapacity: 4096,
      pagedBlockSize: 16,
    }),
  } as unknown as SessionCapableModel;
}

const ROWS: PublicModelEntry[] = [
  { id: 'q35', object: 'model', created: 0, owned_by: 'mlx-node', api_backend: 'messages', model_family: 'qwen3_5' },
  { id: 'g4', object: 'model', created: 0, owned_by: 'mlx-node', api_backend: 'messages', model_family: 'gemma4' },
  { id: 'lfm', object: 'model', created: 0, owned_by: 'mlx-node', api_backend: 'messages', model_family: 'lfm2' },
  {
    id: 'moe',
    object: 'model',
    created: 0,
    owned_by: 'mlx-node',
    api_backend: 'messages',
    model_family: 'qwen3_5_moe',
  },
];

const AGENT_HOST: HandlerOptions = {
  thinkingTokenBudget: AGENT_HOST_THINKING.tokenBudget,
  thinkingCycleGuard: AGENT_HOST_THINKING.cycleGuard,
  thinkingEndText: AGENT_HOST_THINKING.endText,
};

const servers: Server[] = [];
afterEach(async () => {
  for (const server of servers.splice(0)) await new Promise<void>((done) => server.close(() => done()));
});

interface Served {
  post: (body: Record<string, unknown>) => Promise<{ status: number; text: string }>;
  /** The config the request dispatched to its model. */
  send: (body: Record<string, unknown>) => Promise<ChatConfig>;
  /** `send` with the body bytes as given. */
  sendBytes: (body: string) => Promise<ChatConfig>;
}

/** A server over the three listed models and one loaded name `/v1/models` does not list (`alias`). */
async function serve(options: HandlerOptions): Promise<Served> {
  const registry = new ModelRegistry();
  const configs: ChatConfig[] = [];
  for (const row of ROWS) registry.register(row.id, recordingModel(configs));
  registry.register('alias', recordingModel(configs));
  const handler = createHandler(registry, { listModels: () => ROWS, ...options });
  const server = createServer((req, res) => void handler(req, res));
  servers.push(server);
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', () => done()));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const postBytes = async (body: string) => {
    const res = await fetch(`${url}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
    });
    return { status: res.status, text: await res.text() };
  };
  const post = (body: Record<string, unknown>) =>
    postBytes(JSON.stringify({ max_tokens: 64, messages: [{ role: 'user', content: 'hi' }], ...body }));
  const dispatched = async (send: () => Promise<{ status: number; text: string }>) => {
    const before = configs.length;
    const { status, text } = await send();
    expect(status, text).toBe(200);
    expect(configs.length).toBe(before + 1);
    return configs[before]!;
  };
  return {
    post,
    send: (body) => dispatched(() => post(body)),
    sendBytes: (body) => dispatched(() => postBytes(body)),
  };
}

type Rendered = (
  path: '/v1/messages' | '/v1/messages/count_tokens',
  body: Record<string, unknown>,
) => Promise<unknown[]>;

/**
 * A server over the listed models that render; a request answers what it rendered with: tools, thinking, effort. A
 * generation renders twice (the preflight, then the send), the same both times.
 */
async function serveRendering(options: HandlerOptions): Promise<Rendered> {
  const registry = new ModelRegistry();
  const renders: unknown[][] = [];
  for (const row of ROWS) registry.register(row.id, renderingModel(renders));
  const handler = createHandler(registry, { listModels: () => ROWS, ...options });
  const server = createServer((req, res) => void handler(req, res));
  servers.push(server);
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', () => done()));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  return async (path, body) => {
    const before = renders.length;
    const res = await fetch(`${url}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ max_tokens: 64, messages: [{ role: 'user', content: 'hi' }], ...body }),
    });
    const text = await res.text();
    expect(res.status, text).toBe(200);
    const own = renders.slice(before);
    expect(own.length, path).toBe(path === '/v1/messages' ? 2 : 1);
    for (const render of own) expect(render, path).toEqual(own[0]);
    return own[0]!;
  };
}

/** What agent 0.3.5 sends for a compaction summary (`compaction_thinking_budget`): its own budget, none at Off. */
const agent035Compaction = (level: ThinkingLevel, budget: number): Partial<AnthropicMessagesRequest> =>
  level === 'none'
    ? agent035(level)
    : ({
        thinking: { type: 'enabled', budget_tokens: budget },
        output_config: { effort: level },
        extra_body: { reasoning_effort: level },
      } as unknown as Partial<AnthropicMessagesRequest>);

/** The fields thinking moves; the rest of a dispatched config carries per-turn ids. */
const thinkingOf = (config: ChatConfig) => ({
  effort: config.reasoningEffort,
  budget: config.thinkingTokenBudget,
  guard: config.thinkingCycleGuard,
  endText: config.thinkingEndText,
});

describe('Messages effort: the server', () => {
  it('with levels (the agent hosts), the family table reaches the model under the host thinking', async () => {
    const { send } = await serve({ ...AGENT_HOST, thinkingLevels: true });
    const host = { guard: true, endText: AGENT_HOST_THINKING.endText };
    const effort = (model: string, value: string) =>
      send({ model, extra_body: { reasoning_effort: value } }).then(thinkingOf);
    expect(await effort('q35', 'none')).toEqual({
      effort: 'none',
      budget: undefined,
      guard: undefined,
      endText: undefined,
    });
    expect(await effort('q35', 'low')).toEqual({ effort: 'low', budget: 1024, ...host });
    expect(await effort('q35', 'medium')).toEqual({ effort: 'medium', budget: 4096, ...host });
    expect(await effort('q35', 'high')).toEqual({ effort: undefined, budget: 12_000, ...host });
    expect(thinkingOf(await send({ model: 'q35' }))).toEqual(await effort('q35', 'high'));
    expect(await effort('moe', 'low')).toEqual({ effort: undefined, budget: 1024, ...host });
    expect(await effort('moe', 'medium')).toEqual({ effort: undefined, budget: 4096, ...host });
    expect(await effort('q35', 'max')).toEqual({ effort: 'max', budget: 12_000, ...host });
    expect(await effort('lfm', 'low')).toEqual({ effort: 'low', budget: 256, ...host });
    expect(await effort('g4', 'low')).toEqual({ effort: 'low', budget: 12_000, ...host });
    expect(await effort('alias', 'low')).toEqual({ effort: 'low', budget: 12_000, ...host });
  });

  it('without levels (`mlx serve` by hand, `mlx launch claude`), every value reaches the model as it did before', async () => {
    expect('thinkingLevels' in parseServeArgs([])).toBe(false);
    expect('thinkingLevels' in launchClaudeHostOptions({})).toBe(false);
    const bare = (await serve({})).send;
    const host = (await serve(AGENT_HOST)).send;
    for (const model of ['q35', 'g4', 'lfm']) {
      for (const value of WIRE) {
        expect(thinkingOf(await bare({ model, extra_body: { reasoning_effort: value } })), `${model} ${value}`).toEqual(
          {
            effort: value,
            budget: undefined,
            guard: undefined,
            endText: undefined,
          },
        );
        expect(thinkingOf(await host({ model, extra_body: { reasoning_effort: value } }))).toEqual(
          value === 'none'
            ? { effort: 'none', budget: undefined, guard: undefined, endText: undefined }
            : { effort: value, budget: 12_000, guard: true, endText: AGENT_HOST_THINKING.endText },
        );
      }
    }
  });

  it('output_config.effort reaches no model: the dispatched thinking equals the same request without it', async () => {
    for (const options of [{}, AGENT_HOST, { ...AGENT_HOST, thinkingLevels: true }]) {
      const { send } = await serve(options);
      for (const model of ['q35', 'g4', 'lfm']) {
        const plain = thinkingOf(await send({ model }));
        for (const effort of OUTPUT_EFFORTS) {
          expect(thinkingOf(await send({ model, ...agentEffort(effort) })), `${model} ${effort}`).toEqual(plain);
        }
      }
    }
  });

  it('the body agent 0.3.5 sends maps by its extra_body alone, levels on or off; Off adds no thought', async () => {
    for (const options of [{}, AGENT_HOST, { ...AGENT_HOST, thinkingLevels: true }]) {
      const { send } = await serve(options);
      for (const model of ['q35', 'g4', 'lfm']) {
        for (const level of THINKING_LEVELS) {
          expect(thinkingOf(await send({ model, ...agent035(level) })), `${model} ${level}`).toEqual(
            thinkingOf(await send({ model, extra_body: { reasoning_effort: level } })),
          );
        }
      }
    }
    const { send } = await serve({ ...AGENT_HOST, thinkingLevels: true });
    expect(thinkingOf(await send({ model: 'q35', ...agent035('low') }))).toEqual({
      effort: 'low',
      budget: 1024,
      guard: true,
      endText: AGENT_HOST_THINKING.endText,
    });
    expect(thinkingOf(await send({ model: 'q35', ...agent035('none') }))).toEqual({
      effort: 'none',
      budget: undefined,
      guard: undefined,
      endText: undefined,
    });
  });

  it("agent 0.3.5's own bytes: the fixture above is that body, and the server maps it by its extra_body alone", async () => {
    for (const [level, bytes] of Object.entries(AGENT_035_BYTES) as [ThinkingLevel, string][]) {
      expect(JSON.parse(bytes), level).toEqual({
        model: 'q35',
        max_tokens: 64,
        messages: [{ role: 'user', content: [{ type: 'text', text: 'hi', cache_control: { type: 'ephemeral' } }] }],
        ...agent035(level),
      });
    }
    for (const options of [{}, AGENT_HOST, { ...AGENT_HOST, thinkingLevels: true }]) {
      const { send, sendBytes } = await serve(options);
      for (const [level, bytes] of Object.entries(AGENT_035_BYTES)) {
        expect(thinkingOf(await sendBytes(bytes)), level).toEqual(
          thinkingOf(await send({ model: 'q35', extra_body: { reasoning_effort: level } })),
        );
      }
    }
    const { sendBytes } = await serve({ ...AGENT_HOST, thinkingLevels: true });
    expect(thinkingOf(await sendBytes(AGENT_035_BYTES.low))).toEqual({
      effort: 'low',
      budget: 1024,
      guard: true,
      endText: AGENT_HOST_THINKING.endText,
    });
    expect(thinkingOf(await sendBytes(AGENT_035_BYTES.none))).toEqual({
      effort: 'none',
      budget: undefined,
      guard: undefined,
      endText: undefined,
    });
  });

  it('an unknown value is a 400 with levels on or off', async () => {
    for (const options of [{}, { ...AGENT_HOST, thinkingLevels: true }]) {
      const { post } = await serve(options);
      for (const bad of ['minimal', 'bogus']) {
        const { status, text } = await post({ model: 'q35', extra_body: { reasoning_effort: bad } });
        expect(status).toBe(400);
        expect(text).toContain('Invalid extra_body.reasoning_effort');
      }
    }
  });

  it('every level the engine offers is one the agent can send', () => {
    expect([...THINKING_LEVELS]).toEqual(['none', 'low', 'medium', 'high', 'max']);
  });

  it("the agent route: a level's budget comes before the host budget, a compaction's own budget before the level", async () => {
    const { send } = await serve({ ...AGENT_HOST, thinkingLevels: true });
    const host = { guard: true, endText: AGENT_HOST_THINKING.endText };
    const turn = (model: string, level: ThinkingLevel) => send({ model, ...agent035(level) }).then(thinkingOf);
    const compaction = (model: string, level: ThinkingLevel) =>
      send({ model, ...agent035Compaction(level, 6000) }).then(thinkingOf);
    const off = { effort: 'none', budget: undefined, guard: undefined, endText: undefined };
    expect(await turn('q35', 'none')).toEqual(off);
    expect(await turn('q35', 'low')).toEqual({ effort: 'low', budget: 1024, ...host });
    expect(await turn('q35', 'medium')).toEqual({ effort: 'medium', budget: 4096, ...host });
    expect(await turn('q35', 'high')).toEqual({ effort: undefined, budget: 12_000, ...host });
    expect(await turn('lfm', 'low')).toEqual({ effort: 'low', budget: 256, ...host });
    expect(await turn('lfm', 'high')).toEqual({ effort: undefined, budget: 12_000, ...host });
    expect(await compaction('q35', 'none')).toEqual(off);
    for (const level of ['low', 'medium', 'high'] as const) {
      expect(await compaction('q35', level), level).toEqual({
        effort: level === 'high' ? undefined : level,
        budget: 6000,
        ...host,
      });
    }
    expect(await compaction('lfm', 'low')).toEqual({ effort: 'low', budget: 6000, ...host });
  });
});

describe('Messages effort: count_tokens', () => {
  it('counts the prompt /v1/messages renders for every value, levels on or off; without levels, the value as sent', async () => {
    for (const options of [{}, AGENT_HOST, { ...AGENT_HOST, thinkingLevels: true }]) {
      const rendered = await serveRendering(options);
      for (const model of ['q35', 'g4', 'lfm']) {
        for (const value of WIRE) {
          const body = { model, extra_body: { reasoning_effort: value } };
          const counted = await rendered('/v1/messages/count_tokens', body);
          expect(counted, `${model} ${value}`).toEqual(await rendered('/v1/messages', body));
          if (!('thinkingLevels' in options)) {
            expect(counted).toEqual([null, value !== 'none', value]);
          }
        }
      }
    }
    const levels = await serveRendering({ ...AGENT_HOST, thinkingLevels: true });
    const counted = (model: string, value: string) =>
      levels('/v1/messages/count_tokens', { model, extra_body: { reasoning_effort: value } });
    expect(await counted('q35', 'low')).toEqual([null, true, 'low']);
    expect(await counted('q35', 'medium')).toEqual([null, true, 'medium']);
    expect(await counted('q35', 'high')).toEqual([null, null, null]);
    for (const value of ['low', 'medium', 'high'] as const) {
      expect(await counted('moe', value), value).toEqual([null, null, null]);
    }
  });
});
