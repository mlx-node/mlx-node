import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { Api, AssistantMessage, Message, Model } from '@earendil-works/pi-ai';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext, InlineExtension, ToolDefinition } from '@earendil-works/pi-coding-agent';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  createProductionSession,
  createSubagentExtension,
  discoverSubagents,
  scaleSubagentCompactionSettings,
  type InProcessSubagentSession,
  type SubagentSessionCreateOptions,
} from '../src/extensions/subagent.js';

/** The parent's in-process mlx provider config the subagent inherits. */
type MlxProviderConfig = SubagentSessionCreateOptions['mlxProviderConfig'];

const PARENT_MODEL = { provider: 'mlx', id: 'agents-a1' } as NonNullable<ExtensionContext['model']>;
const ALT_MODEL = { provider: 'mlx', id: 'alternate' } as NonNullable<ExtensionContext['model']>;
const MLX_PROVIDER_CONFIG: MlxProviderConfig = {
  api: 'mlx',
  baseUrl: 'mlx://local',
  apiKey: 'mlx-local',
  streamSimple: () => createAssistantMessageEventStream(),
  models: [],
};
const MODEL_REGISTRY = {
  find(provider: string, id: string) {
    if (provider !== 'mlx') return undefined;
    if (id === PARENT_MODEL.id) return PARENT_MODEL;
    if (id === ALT_MODEL.id) return ALT_MODEL;
    return undefined;
  },
  getRegisteredProviderConfig(provider: string) {
    return provider === 'mlx' ? MLX_PROVIDER_CONFIG : undefined;
  },
} as unknown as ExtensionContext['modelRegistry'];

function assistantMessage(text: string, stopReason = 'stop'): Message {
  return {
    role: 'assistant',
    content: [{ type: 'text', text }],
    usage: {
      input: 10,
      output: 2,
      cacheRead: 3,
      cacheWrite: 0,
      totalTokens: 15,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    provider: 'mlx',
    model: 'agents-a1',
    timestamp: Date.now(),
  } as Message;
}

class FakeSession implements InProcessSubagentSession {
  readonly prompts: string[] = [];
  abortCalls = 0;
  disposed = false;
  private readonly listeners = new Set<(event: unknown) => void>();
  private resolvePrompt?: () => void;
  private rejectPrompt?: (error: Error) => void;

  constructor(private readonly automaticReply?: (prompt: string) => string) {}

  subscribe(listener: (event: unknown) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  prompt(text: string): Promise<void> {
    this.prompts.push(text);
    const pending = new Promise<void>((resolve, reject) => {
      this.resolvePrompt = resolve;
      this.rejectPrompt = reject;
    });
    if (this.automaticReply) queueMicrotask(() => this.finish(this.automaticReply!(text)));
    return pending;
  }

  async abort(): Promise<void> {
    this.abortCalls++;
    this.resolvePrompt?.();
  }

  dispose(): void {
    this.disposed = true;
    this.listeners.clear();
  }

  finish(text = 'done', stopReason = 'stop'): void {
    const message = assistantMessage(text, stopReason);
    for (const listener of this.listeners) listener({ type: 'message_end', message });
    this.resolvePrompt?.();
  }

  fail(error: Error): void {
    this.rejectPrompt?.(error);
  }
}

function captureTool(extension: InlineExtension): ToolDefinition {
  if (typeof extension === 'function') throw new Error('expected named extension');
  let captured: ToolDefinition | undefined;
  const pi = {
    registerTool(tool: ToolDefinition): void {
      captured = tool;
    },
  } as unknown as ExtensionAPI;
  void extension.factory(pi);
  if (!captured) throw new Error('subagent tool was not registered');
  return captured;
}

function context(extra: Partial<ExtensionContext> = {}): ExtensionContext {
  return {
    cwd: '/repo',
    hasUI: true,
    model: PARENT_MODEL,
    modelRegistry: MODEL_REGISTRY,
    ui: { confirm: async () => true },
    ...extra,
  } as unknown as ExtensionContext;
}

const savedAgentDir = process.env['PI_CODING_AGENT_DIR'];

afterEach(() => {
  if (savedAgentDir === undefined) delete process.env['PI_CODING_AGENT_DIR'];
  else process.env['PI_CODING_AGENT_DIR'] = savedAgentDir;
});

describe('mlx subagent extension', () => {
  it('scales pi compaction defaults to a small physical context window', () => {
    expect(
      scaleSubagentCompactionSettings(32_768, {
        enabled: true,
        reserveTokens: 16_384,
        keepRecentTokens: 20_000,
      }),
    ).toEqual({ enabled: true, reserveTokens: 8_192, keepRecentTokens: 16_384 });
  });

  it('preserves smaller compaction settings and the enabled flag', () => {
    expect(
      scaleSubagentCompactionSettings(32_768, {
        enabled: false,
        reserveTokens: 2_048,
        keepRecentTokens: 4_096,
      }),
    ).toEqual({ enabled: false, reserveTokens: 2_048, keepRecentTokens: 4_096 });
  });

  it('keeps rounded compaction budgets inside three quarters of the physical window', () => {
    const scaled = scaleSubagentCompactionSettings(101, {
      enabled: true,
      reserveTokens: Number.POSITIVE_INFINITY,
      keepRecentTokens: Number.POSITIVE_INFINITY,
    });
    expect(scaled).toEqual({ enabled: true, reserveTokens: 25, keepRecentTokens: 50 });
    expect(scaled.reserveTokens + scaled.keepRecentTokens).toBeLessThanOrEqual(Math.floor((101 * 3) / 4));
  });

  it('rejects an invalid physical context window', () => {
    expect(() =>
      scaleSubagentCompactionSettings(0, {
        enabled: true,
        reserveTokens: 16_384,
        keepRecentTokens: 20_000,
      }),
    ).toThrow('invalid context window');
  });

  it('discovers built-ins and lets user definitions override them', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-test-'));
    try {
      process.env.PI_CODING_AGENT_DIR = root;
      await mkdir(join(root, 'agents'));
      await writeFile(
        join(root, 'agents', 'worker.md'),
        '---\nname: worker\ndescription: custom worker\ntools: read\n---\nCustom prompt.\n',
      );
      const { agents } = discoverSubagents('/repo', 'user');
      expect(agents.map((agent) => agent.name)).toEqual(['scout', 'planner', 'reviewer', 'worker']);
      expect(agents.find((agent) => agent.name === 'worker')).toMatchObject({
        source: 'user',
        description: 'custom worker',
        tools: ['read'],
      });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('creates an isolated session with the parent model registry and requested role configuration', async () => {
    const calls: SubagentSessionCreateOptions[] = [];
    const sessions: FakeSession[] = [];
    const tool = captureTool(
      createSubagentExtension({
        async createSession(options) {
          calls.push(options);
          const session = new FakeSession(() => 'finished');
          sessions.push(session);
          return session;
        },
      }),
    );

    const result = await tool.execute(
      'call-1',
      { agent: 'scout', task: 'inspect the code', cwd: '/repo/nested' },
      undefined,
      undefined,
      context(),
    );

    expect(result.content).toEqual([{ type: 'text', text: 'finished' }]);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({
      cwd: '/repo/nested',
      model: PARENT_MODEL,
      mlxProviderConfig: MLX_PROVIDER_CONFIG,
    });
    expect(calls[0]!.tools).toEqual(['read', 'grep', 'find', 'ls', 'bash']);
    expect(calls[0]!.systemPrompt).toContain('You are a scout');
    expect(sessions[0]!.prompts).toEqual(['Task: inspect the code']);
    expect(sessions[0]!.disposed).toBe(true);
    expect((result as typeof result & { details: { results: unknown[] } }).details.results).toHaveLength(1);
  });

  it('resolves an agent-specific mlx model through the shared parent registry', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-model-'));
    try {
      process.env.PI_CODING_AGENT_DIR = root;
      await mkdir(join(root, 'agents'));
      await writeFile(
        join(root, 'agents', 'alternate.md'),
        '---\nname: alternate\ndescription: alternate model\nmodel: mlx/alternate\n---\nUse the alternate.\n',
      );
      let created: SubagentSessionCreateOptions | undefined;
      const tool = captureTool(
        createSubagentExtension({
          async createSession(options) {
            created = options;
            return new FakeSession(() => 'alternate done');
          },
        }),
      );
      await tool.execute('call-model', { agent: 'alternate', task: 'work' }, undefined, undefined, context());
      expect(created?.model).toBe(ALT_MODEL);
      expect(created?.mlxProviderConfig).toBe(MLX_PROVIDER_CONFIG);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('fails closed when the parent has no registered mlx provider', async () => {
    let created = false;
    const tool = captureTool(
      createSubagentExtension({
        async createSession() {
          created = true;
          return new FakeSession(() => 'unexpected');
        },
      }),
    );
    // Parent resolves the mlx model but exposes no registered mlx provider config,
    // so the subagent cannot inherit the shared MlxModelHost and must fail closed.
    const noProviderRegistry = {
      find(provider: string, id: string) {
        if (provider !== 'mlx') return undefined;
        return id === PARENT_MODEL.id ? PARENT_MODEL : undefined;
      },
      getRegisteredProviderConfig() {
        return undefined;
      },
    } as unknown as ExtensionContext['modelRegistry'];

    const result = await tool.execute(
      'call-no-mlx-provider',
      { agent: 'worker', task: 'work' },
      undefined,
      undefined,
      context({ modelRegistry: noProviderRegistry }),
    );

    expect(created).toBe(false);
    expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toContain("no 'mlx' provider registered");
  });

  it('drives a fresh runtime through createProductionSession and round-trips streamSimple', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-runtime-'));
    process.env.PI_CODING_AGENT_DIR = root;
    const SCRIPTED_TEXT = 'scripted subagent reply';
    let streamCalls = 0;
    const seamModel: NonNullable<ExtensionContext['model']> = {
      id: 'seam-model',
      name: 'seam-model',
      api: 'mlx',
      provider: 'mlx',
      baseUrl: 'mlx://local',
      reasoning: false,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
    };
    const mlxProviderConfig: MlxProviderConfig = {
      api: 'mlx',
      baseUrl: 'mlx://local',
      apiKey: 'mlx-local',
      // Dispatch through a FRESH runtime proves provider-id (not object-identity)
      // routing to the shared host's streamSimple. Script a minimal terminal turn.
      streamSimple: (streamedModel: Model<Api>) => {
        streamCalls++;
        const stream = createAssistantMessageEventStream();
        const message: AssistantMessage = {
          role: 'assistant',
          content: [{ type: 'text', text: SCRIPTED_TEXT }],
          api: streamedModel.api,
          provider: streamedModel.provider,
          model: streamedModel.id,
          usage: {
            input: 1,
            output: 1,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 2,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: 'stop',
          timestamp: Date.now(),
        };
        stream.push({ type: 'start', partial: message });
        stream.push({ type: 'text_start', contentIndex: 0, partial: message });
        stream.push({ type: 'text_delta', contentIndex: 0, delta: SCRIPTED_TEXT, partial: message });
        stream.push({ type: 'text_end', contentIndex: 0, content: SCRIPTED_TEXT, partial: message });
        stream.push({ type: 'done', reason: 'stop', message });
        stream.end();
        return stream;
      },
      models: [
        {
          id: 'seam-model',
          name: 'seam-model',
          api: 'mlx',
          baseUrl: 'mlx://local',
          reasoning: false,
          input: ['text'],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 4096,
          maxTokens: 1024,
        },
      ],
    };

    let session: InProcessSubagentSession | undefined;
    try {
      session = await createProductionSession({
        cwd: root,
        model: seamModel,
        mlxProviderConfig,
        tools: ['read'],
        systemPrompt: 'You are a runtime-seam probe.',
      });
      const assistantTexts: string[] = [];
      const unsubscribe = session.subscribe((raw) => {
        const event = raw as { type?: string; message?: Message };
        if (event.type !== 'message_end' || event.message?.role !== 'assistant') return;
        const content = event.message.content;
        const text = Array.isArray(content) ? content.find((part) => part.type === 'text') : undefined;
        if (text?.type === 'text') assistantTexts.push(text.text);
      });
      await session.prompt('probe the seam');
      unsubscribe();

      // One dispatch through the fresh runtime, and the scripted reply round-trips
      // back out of the real AgentSession — proving shared-host wiring end to end.
      expect(streamCalls).toBe(1);
      expect(assistantTexts).toContain(SCRIPTED_TEXT);
    } finally {
      session?.dispose();
      await rm(root, { recursive: true, force: true });
    }
  }, 20_000);

  it('propagates abort to the in-process session and disposes it', async () => {
    const session = new FakeSession();
    const controller = new AbortController();
    const tool = captureTool(createSubagentExtension({ createSession: async () => session }));
    const pending = tool.execute(
      'call-abort',
      { agent: 'worker', task: 'work' },
      controller.signal,
      undefined,
      context(),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();
    const result = await pending;

    expect(session.abortCalls).toBe(1);
    expect(session.disposed).toBe(true);
    expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toContain('Subagent was aborted');
  });

  it('does not prompt a newly created session when the call was already aborted', async () => {
    const session = new FakeSession();
    const controller = new AbortController();
    controller.abort();
    const tool = captureTool(createSubagentExtension({ createSession: async () => session }));
    const result = await tool.execute(
      'call-pre-abort',
      { agent: 'worker', task: 'must not run' },
      controller.signal,
      undefined,
      context(),
    );
    expect(session.prompts).toEqual([]);
    expect(session.abortCalls).toBe(1);
    expect(session.disposed).toBe(true);
    expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
  });

  it('runs at most four task loops concurrently and starts queued work as slots finish', async () => {
    const sessions: FakeSession[] = [];
    const tool = captureTool(
      createSubagentExtension({
        async createSession() {
          const session = new FakeSession();
          sessions.push(session);
          return session;
        },
      }),
    );
    const tasks = Array.from({ length: 6 }, (_, index) => ({ agent: 'scout', task: `task ${index + 1}` }));
    const pending = tool.execute('call-parallel', { tasks }, undefined, undefined, context());

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(sessions).toHaveLength(4);
    sessions[0]!.finish('one');
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(sessions).toHaveLength(5);
    sessions[1]!.finish('two');
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(sessions).toHaveLength(6);
    for (const session of sessions.slice(2)) session.finish('done');

    const result = await pending;
    expect((result.content[0] as { text: string }).text).toContain('6/6 succeeded');
    expect(sessions.every((session) => session.disposed)).toBe(true);
  });

  it('caps a concurrent request at eight tasks before creating sessions', async () => {
    let created = 0;
    const tool = captureTool(
      createSubagentExtension({
        async createSession() {
          created++;
          return new FakeSession(() => 'done');
        },
      }),
    );
    const tasks = Array.from({ length: 9 }, (_, index) => ({ agent: 'scout', task: `task ${index + 1}` }));
    const result = await tool.execute('call-too-many', { tasks }, undefined, undefined, context());
    expect(created).toBe(0);
    expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toContain('max is 8');
  });

  it('feeds successful chain output into the next independent session', async () => {
    const prompts: string[] = [];
    const tool = captureTool(
      createSubagentExtension({
        async createSession() {
          return new FakeSession((prompt) => {
            prompts.push(prompt);
            return prompts.length === 1 ? 'first result' : 'second result';
          });
        },
      }),
    );
    const result = await tool.execute(
      'call-chain',
      {
        chain: [
          { agent: 'scout', task: 'inspect' },
          { agent: 'planner', task: 'plan from {previous}' },
        ],
      },
      undefined,
      undefined,
      context(),
    );
    expect(prompts).toEqual(['Task: inspect', 'Task: plan from first result']);
    expect(result.content).toEqual([{ type: 'text', text: 'second result' }]);
  });

  it('substitutes prior chain output literally when it contains JavaScript replacement tokens', async () => {
    const previousOutput = "literal $& $$ $` $'";
    const prompts: string[] = [];
    const tool = captureTool(
      createSubagentExtension({
        async createSession() {
          return new FakeSession((prompt) => {
            prompts.push(prompt);
            return prompts.length === 1 ? previousOutput : 'done';
          });
        },
      }),
    );

    await tool.execute(
      'call-chain-literal',
      {
        chain: [
          { agent: 'scout', task: 'inspect' },
          { agent: 'planner', task: 'plan from {previous}; verify {previous}' },
        ],
      },
      undefined,
      undefined,
      context(),
    );

    expect(prompts).toEqual(['Task: inspect', `Task: plan from ${previousOutput}; verify ${previousOutput}`]);
  });

  it('rejects a user agent that explicitly requests a cloud provider', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-test-'));
    try {
      process.env.PI_CODING_AGENT_DIR = root;
      await mkdir(join(root, 'agents'));
      await writeFile(
        join(root, 'agents', 'cloud.md'),
        '---\nname: cloud\ndescription: cloud agent\nmodel: anthropic/claude\n---\nNo.\n',
      );
      let created = false;
      const tool = captureTool(
        createSubagentExtension({
          async createSession() {
            created = true;
            return new FakeSession(() => 'unexpected');
          },
        }),
      );
      const result = await tool.execute(
        'call-cloud',
        { agent: 'cloud', task: 'work' },
        undefined,
        undefined,
        context(),
      );
      expect(created).toBe(false);
      expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
      expect((result.content[0] as { text: string }).text).toContain('non-local model');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('always confirms a requested project agent and cannot be bypassed by tool arguments', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-project-'));
    try {
      await mkdir(join(root, '.pi', 'agents'), { recursive: true });
      await writeFile(
        join(root, '.pi', 'agents', 'worker.md'),
        '---\nname: worker\ndescription: project worker\n---\nProject-controlled prompt.\n',
      );
      let created = false;
      let confirmations = 0;
      const tool = captureTool(
        createSubagentExtension({
          async createSession() {
            created = true;
            return new FakeSession(() => 'unexpected');
          },
        }),
      );
      const result = await tool.execute(
        'call-project',
        {
          agent: 'worker',
          task: 'work',
          agentScope: 'both',
          confirmProjectAgents: false,
        } as never,
        undefined,
        undefined,
        context({
          cwd: root,
          ui: {
            confirm: async () => {
              confirmations++;
              return false;
            },
          } as never,
        }),
      );
      expect(confirmations).toBe(1);
      expect(created).toBe(false);
      expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('encodes control bytes in project-agent names and source paths before confirmation', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-project-sanitize-'));
    const projectRoot = join(root, 'repo\u001b[31m');
    const agentName = 'worker\u001b[2J\u009b31m';
    try {
      await mkdir(join(projectRoot, '.pi', 'agents'), { recursive: true });
      await writeFile(
        join(projectRoot, '.pi', 'agents', 'worker.md'),
        `---\nname: ${agentName}\ndescription: project worker\n---\nProject-controlled prompt.\n`,
      );
      let confirmationDetail = '';
      const tool = captureTool(createSubagentExtension());
      await tool.execute(
        'call-project-controls',
        { agent: agentName, task: 'work', agentScope: 'project' },
        undefined,
        undefined,
        context({
          cwd: projectRoot,
          ui: {
            confirm: async (_title: string, detail: string) => {
              confirmationDetail = detail;
              return false;
            },
          } as never,
        }),
      );

      expect(confirmationDetail).not.toContain('\u001b');
      expect(confirmationDetail).not.toContain('\u009b');
      expect(confirmationDetail).toContain('Agents: worker\\x1b[2J\\x9b31m');
      expect(confirmationDetail).toContain(`Source: ${join(projectRoot.replace('\u001b', '\\x1b'), '.pi', 'agents')}`);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('caps oversized project-agent confirmation details with a visible marker', async () => {
    const root = await mkdtemp(join(tmpdir(), 'mlx-subagent-project-truncate-'));
    const agentName = `project-worker-${'x'.repeat(600)}-hidden-tail`;
    try {
      await mkdir(join(root, '.pi', 'agents'), { recursive: true });
      await writeFile(
        join(root, '.pi', 'agents', 'worker.md'),
        `---\nname: ${agentName}\ndescription: project worker\n---\nProject-controlled prompt.\n`,
      );
      let confirmationDetail = '';
      const tool = captureTool(createSubagentExtension());
      await tool.execute(
        'call-project-long-name',
        { agent: agentName, task: 'work', agentScope: 'project' },
        undefined,
        undefined,
        context({
          cwd: root,
          ui: {
            confirm: async (_title: string, detail: string) => {
              confirmationDetail = detail;
              return false;
            },
          } as never,
        }),
      );

      expect(confirmationDetail).toContain('… [truncated]');
      expect(confirmationDetail).not.toContain('hidden-tail');
      expect(confirmationDetail.length).toBeLessThan(520);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
