import { EventEmitter } from 'node:events';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PassThrough } from 'node:stream';

import type { ExtensionAPI, ExtensionContext, InlineExtension, ToolDefinition } from '@earendil-works/pi-coding-agent';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createSubagentExtension, discoverSubagents } from '../src/extensions/subagent.js';

class FakeChild extends EventEmitter {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  exitCode: number | null = null;
  signalCode: NodeJS.Signals | null = null;
  readonly kills: NodeJS.Signals[] = [];

  kill(signal: NodeJS.Signals = 'SIGTERM'): boolean {
    this.kills.push(signal);
    return true;
  }

  finish(text = 'done', code = 0): void {
    const message = {
      role: 'assistant',
      content: [{ type: 'text', text }],
      usage: {
        input: 10,
        output: 2,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 12,
        cost: { total: 0 },
      },
      stopReason: 'stop',
      model: 'local-model',
    };
    this.stdout.end(`${JSON.stringify({ type: 'message_end', message })}\n`);
    this.exitCode = code;
    this.emit('close', code);
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
    model: { provider: 'mlx', id: 'agents-a1' },
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

  it('re-enters mlx with the same models dir/model and a child-only safety environment', async () => {
    const calls: Array<{ command: string; args: string[]; env: NodeJS.ProcessEnv }> = [];
    const child = new FakeChild();
    const tool = captureTool(
      createSubagentExtension({
        modelsDir: '/models/custom',
        invocation: { command: '/node', args: ['/mlx/cli.js', 'agent'] },
        spawnChild(command, args, options) {
          calls.push({ command, args, env: options.env });
          queueMicrotask(() => child.finish('finished'));
          return child as never;
        },
      }),
    );

    const result = await tool.execute('call-1', { agent: 'worker', task: 'do work' }, undefined, undefined, context());
    expect(result.content).toEqual([{ type: 'text', text: 'finished' }]);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.command).toBe('/node');
    expect(calls[0]!.args).toEqual(
      expect.arrayContaining([
        '/mlx/cli.js',
        'agent',
        '--models-dir',
        '/models/custom',
        '--mode',
        'json',
        '--no-session',
        '--no-extensions',
        '--no-approve',
        '--model',
        'mlx/agents-a1',
      ]),
    );
    expect(calls[0]!.env['MLX_AGENT_SUBAGENT_CHILD']).toBe('1');
    expect(calls[0]!.env['MLX_AGENT_AUTO_APPROVE']).toBe('1');
  });

  it('derives a real child invocation from the current Node/oxnode CLI process', async () => {
    const calls: Array<{ command: string; args: string[] }> = [];
    const child = new FakeChild();
    const tool = captureTool(
      createSubagentExtension({
        modelsDir: '/models',
        spawnChild(command, args) {
          calls.push({ command, args });
          queueMicrotask(() => child.finish());
          return child as never;
        },
      }),
    );
    await tool.execute('call-route', { agent: 'scout', task: 'inspect' }, undefined, undefined, context());

    expect(calls[0]!.command).toBe(process.execPath);
    const scriptIndex = calls[0]!.args.indexOf(process.argv[1]!);
    expect(scriptIndex).toBeGreaterThanOrEqual(0);
    expect(calls[0]!.args[scriptIndex + 1]).toBe('agent');
  });

  it('propagates abort to the child process', async () => {
    const child = new FakeChild();
    const controller = new AbortController();
    const tool = captureTool(
      createSubagentExtension({
        modelsDir: '/models',
        invocation: { command: 'mlx', args: ['agent'] },
        spawnChild() {
          return child as never;
        },
      }),
    );
    const pending = tool.execute(
      'call-abort',
      { agent: 'worker', task: 'work' },
      controller.signal,
      undefined,
      context(),
    );
    controller.abort();
    expect(child.kills).toEqual(['SIGTERM']);
    child.finish('', 1);
    const result = await pending;
    expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
    expect((result.content[0] as { text: string }).text).toContain('Subagent was aborted');
  });

  it('serializes parallel-shaped tasks so only one model-owning child is alive', async () => {
    const children: FakeChild[] = [];
    const tool = captureTool(
      createSubagentExtension({
        modelsDir: '/models',
        invocation: { command: 'mlx', args: ['agent'] },
        spawnChild() {
          const child = new FakeChild();
          children.push(child);
          return child as never;
        },
      }),
    );

    const pending = tool.execute(
      'call-2',
      {
        tasks: [
          { agent: 'scout', task: 'one' },
          { agent: 'scout', task: 'two' },
        ],
      },
      undefined,
      undefined,
      context(),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(children).toHaveLength(1);
    children[0]!.finish('one');
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(children).toHaveLength(2);
    children[1]!.finish('two');
    const result = await pending;
    expect(result.content[0]).toMatchObject({ type: 'text' });
    expect((result.content[0] as { text: string }).text).toContain('2/2 succeeded');
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
      let spawned = false;
      const tool = captureTool(
        createSubagentExtension({
          modelsDir: '/models',
          spawnChild() {
            spawned = true;
            return new FakeChild() as never;
          },
        }),
      );
      const result = await tool.execute('call-3', { agent: 'cloud', task: 'work' }, undefined, undefined, context());
      expect(spawned).toBe(false);
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
      let spawned = false;
      let confirmations = 0;
      const tool = captureTool(
        createSubagentExtension({
          modelsDir: '/models',
          spawnChild() {
            spawned = true;
            return new FakeChild() as never;
          },
        }),
      );
      const result = await tool.execute(
        'call-project',
        {
          agent: 'worker',
          task: 'work',
          agentScope: 'both',
          // Deliberately exercise the upstream bypass even though mlx omits it
          // from the schema: the execution path must ignore hostile extras.
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
      expect(spawned).toBe(false);
      expect((result as typeof result & { isError?: boolean }).isError).toBe(true);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
