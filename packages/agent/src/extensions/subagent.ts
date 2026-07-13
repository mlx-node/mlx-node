/**
 * MLX port of pi's official `examples/extensions/subagent` extension.
 *
 * The upstream example starts a fresh pi process per task. We preserve that
 * isolation, JSONL protocol, agent discovery, and single/parallel/chain modes,
 * while routing children back through `mlx agent` so the local provider is
 * registered. MLX-specific safety differences are intentionally small:
 *
 * - at most one child runs at once (each process owns a model + KV pool),
 * - children inherit the parent's current local model and models directory,
 * - external extensions/project resources and recursive subagents are disabled,
 * - the permission gate approves the delegation before child-only tool access.
 *
 * Upstream source: @earendil-works/pi-coding-agent 0.80.6,
 * examples/extensions/subagent (MIT).
 */

import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import * as path from 'node:path';

import type { Message } from '@earendil-works/pi-ai';
import { StringEnum } from '@earendil-works/pi-ai';
import type { ExtensionAPI, InlineExtension } from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';

const MAX_PARALLEL_TASKS = 8;
/** Local models are process-owned; do not multiply model/KV allocations. */
const MAX_CONCURRENCY = 1;
const PER_TASK_OUTPUT_CAP = 50 * 1024;
const CHILD_ENV = 'MLX_AGENT_SUBAGENT_CHILD';

type AgentScope = 'user' | 'project' | 'both';
type AgentSource = 'builtin' | 'user' | 'project' | 'unknown';

export interface SubagentConfig {
  name: string;
  description: string;
  tools?: string[];
  model?: string;
  systemPrompt: string;
  source: Exclude<AgentSource, 'unknown'>;
  filePath: string;
}

interface UsageStats {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cost: number;
  contextTokens: number;
  turns: number;
}

interface SingleResult {
  agent: string;
  agentSource: AgentSource;
  task: string;
  exitCode: number;
  messages: Message[];
  stderr: string;
  usage: UsageStats;
  model?: string;
  stopReason?: string;
  errorMessage?: string;
  step?: number;
}

interface SubagentDetails {
  mode: 'single' | 'parallel' | 'chain';
  agentScope: AgentScope;
  projectAgentsDir: string | null;
  results: SingleResult[];
}

interface SpawnedChild {
  stdout: NodeJS.ReadableStream;
  stderr: NodeJS.ReadableStream;
  exitCode: number | null;
  signalCode: NodeJS.Signals | null;
  kill(signal?: NodeJS.Signals): boolean;
  once(event: 'close', listener: (code: number | null) => void): this;
  once(event: 'error', listener: (error: Error) => void): this;
}

export interface SubagentSpawnOptions {
  cwd: string;
  env: NodeJS.ProcessEnv;
}

export interface SubagentExtensionOptions {
  modelsDir: string;
  /** Test seam. Production re-enters the current mlx CLI. */
  spawnChild?: (command: string, args: string[], options: SubagentSpawnOptions) => SpawnedChild;
  /** Test/programmatic seam. Production derives the current CLI invocation. */
  invocation?: { command: string; args: string[] };
}

const BUILTIN_AGENTS: readonly SubagentConfig[] = [
  {
    name: 'scout',
    description: 'Fast codebase recon that returns compressed context for handoff to other agents',
    tools: ['read', 'grep', 'find', 'ls', 'bash'],
    systemPrompt: `You are a scout. Quickly investigate a codebase and return structured findings that another agent can use without re-reading everything.

Follow imports, read critical sections, and report exact file paths and line ranges. Summarize the architecture, key types/functions, and where the next agent should start. Do not modify files.`,
    source: 'builtin',
    filePath: '<builtin:scout>',
  },
  {
    name: 'planner',
    description: 'Creates implementation plans from context and requirements',
    tools: ['read', 'grep', 'find', 'ls'],
    systemPrompt: `You are a planning specialist. Produce a concrete implementation plan with the goal, numbered steps, files to modify, new files, and risks. You must not make changes.`,
    source: 'builtin',
    filePath: '<builtin:planner>',
  },
  {
    name: 'reviewer',
    description: 'Code review specialist for quality and security analysis',
    tools: ['read', 'grep', 'find', 'ls', 'bash'],
    systemPrompt: `You are a senior code reviewer. Review the relevant diff and code for correctness, security, and maintainability. Keep bash read-only. Report concrete findings with file paths and line numbers, then give a concise verdict. Do not modify files.`,
    source: 'builtin',
    filePath: '<builtin:reviewer>',
  },
  {
    name: 'worker',
    description: 'General-purpose subagent with full capabilities and isolated context',
    systemPrompt: `You are a worker agent with full capabilities. Work autonomously to complete the delegated task. Report what changed, exact files changed, tests run, and anything the parent agent must know.`,
    source: 'builtin',
    filePath: '<builtin:worker>',
  },
];

function emptyUsage(): UsageStats {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, contextTokens: 0, turns: 0 };
}

function agentDir(): string {
  const configured = process.env['PI_CODING_AGENT_DIR'];
  if (!configured) return path.join(homedir(), '.mlx-node', 'agent');
  if (configured === '~') return homedir();
  if (configured.startsWith('~/')) return path.join(homedir(), configured.slice(2));
  return configured;
}

function parseAgentFile(filePath: string, source: 'user' | 'project'): SubagentConfig | undefined {
  let content: string;
  try {
    content = fs.readFileSync(filePath, 'utf8');
  } catch {
    return undefined;
  }
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(content);
  if (!match) return undefined;
  const frontmatter: Record<string, string> = {};
  for (const line of match[1]!.split(/\r?\n/)) {
    const separator = line.indexOf(':');
    if (separator <= 0) continue;
    frontmatter[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }
  if (!frontmatter['name'] || !frontmatter['description']) return undefined;
  const tools = frontmatter['tools']
    ?.split(',')
    .map((tool) => tool.trim())
    .filter(Boolean);
  return {
    name: frontmatter['name'],
    description: frontmatter['description'],
    tools: tools?.length ? tools : undefined,
    model: frontmatter['model'] || undefined,
    systemPrompt: match[2]!,
    source,
    filePath,
  };
}

function loadAgentsFromDir(dir: string, source: 'user' | 'project'): SubagentConfig[] {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter((entry) => entry.name.endsWith('.md') && (entry.isFile() || entry.isSymbolicLink()))
    .map((entry) => parseAgentFile(path.join(dir, entry.name), source))
    .filter((agent): agent is SubagentConfig => agent !== undefined);
}

function findProjectAgentsDir(cwd: string): string | null {
  let current = path.resolve(cwd);
  while (true) {
    const candidate = path.join(current, '.pi', 'agents');
    try {
      if (fs.statSync(candidate).isDirectory()) return candidate;
    } catch {
      // Continue toward the filesystem root.
    }
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

export function discoverSubagents(
  cwd: string,
  scope: AgentScope,
): {
  agents: SubagentConfig[];
  projectAgentsDir: string | null;
} {
  const projectAgentsDir = findProjectAgentsDir(cwd);
  const users = scope === 'project' ? [] : loadAgentsFromDir(path.join(agentDir(), 'agents'), 'user');
  const projects = scope === 'user' || !projectAgentsDir ? [] : loadAgentsFromDir(projectAgentsDir, 'project');
  const agents = new Map<string, SubagentConfig>();
  if (scope !== 'project') for (const agent of BUILTIN_AGENTS) agents.set(agent.name, agent);
  for (const agent of users) agents.set(agent.name, agent);
  for (const agent of projects) agents.set(agent.name, agent);
  return { agents: [...agents.values()], projectAgentsDir };
}

function getFinalOutput(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message?.role !== 'assistant' || !Array.isArray(message.content)) continue;
    const text = message.content.find((part) => part.type === 'text');
    if (text?.type === 'text') return text.text;
  }
  return '';
}

function isFailed(result: SingleResult): boolean {
  return result.exitCode !== 0 || result.stopReason === 'error' || result.stopReason === 'aborted';
}

function resultOutput(result: SingleResult): string {
  return (
    result.errorMessage || (isFailed(result) ? result.stderr : '') || getFinalOutput(result.messages) || '(no output)'
  );
}

function truncateOutput(output: string): string {
  const bytes = Buffer.byteLength(output);
  if (bytes <= PER_TASK_OUTPUT_CAP) return output;
  let truncated = output.slice(0, PER_TASK_OUTPUT_CAP);
  while (Buffer.byteLength(truncated) > PER_TASK_OUTPUT_CAP) truncated = truncated.slice(0, -1);
  return `${truncated}\n\n[Output truncated; full output remains in tool details.]`;
}

function normalizedModel(model: string): string | undefined {
  if (model.startsWith('mlx/')) return model;
  // A bare name is a local discovered model id. A provider/name pair is not.
  if (!model.includes('/')) return `mlx/${model}`;
  return undefined;
}

function defaultInvocation(): { command: string; args: string[] } {
  const currentScript = process.argv[1];
  if (currentScript && fs.existsSync(currentScript)) {
    return { command: process.execPath, args: [...process.execArgv, currentScript, 'agent'] };
  }
  return { command: 'mlx', args: ['agent'] };
}

async function mapWithConcurrencyLimit<T, R>(items: T[], fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results = Array.from<R>({ length: items.length });
  let next = 0;
  const workers = Array.from({ length: Math.min(MAX_CONCURRENCY, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index]!, index);
    }
  });
  await Promise.all(workers);
  return results;
}

function productionSpawn(command: string, args: string[], options: SubagentSpawnOptions): SpawnedChild {
  return spawn(command, args, { ...options, shell: false, stdio: ['ignore', 'pipe', 'pipe'] }) as SpawnedChild;
}

async function runSingleAgent(
  options: SubagentExtensionOptions,
  defaultCwd: string,
  agents: SubagentConfig[],
  agentName: string,
  task: string,
  cwd: string | undefined,
  parentModel: string | undefined,
  step: number | undefined,
  signal: AbortSignal | undefined,
  onUpdate: ((result: { content: { type: 'text'; text: string }[]; details: SubagentDetails }) => void) | undefined,
  makeDetails: (results: SingleResult[]) => SubagentDetails,
): Promise<SingleResult> {
  const agent = agents.find((candidate) => candidate.name === agentName);
  if (!agent) {
    return {
      agent: agentName,
      agentSource: 'unknown',
      task,
      exitCode: 1,
      messages: [],
      stderr: `Unknown agent: ${agentName}. Available: ${agents.map((a) => a.name).join(', ') || 'none'}`,
      usage: emptyUsage(),
      step,
    };
  }

  const requestedModel = agent.model ? normalizedModel(agent.model) : parentModel;
  if (agent.model && !requestedModel) {
    return {
      agent: agent.name,
      agentSource: agent.source,
      task,
      exitCode: 1,
      messages: [],
      stderr: `Agent ${agent.name} requested non-local model ${agent.model}; mlx subagents only accept mlx/<model>.`,
      usage: emptyUsage(),
      step,
    };
  }

  const args = [
    '--models-dir',
    options.modelsDir,
    '--mode',
    'json',
    '-p',
    '--no-session',
    '--no-extensions',
    '--no-approve',
  ];
  if (requestedModel) args.push('--model', requestedModel);
  if (agent.tools?.length) args.push('--tools', agent.tools.join(','));

  let promptDir: string | undefined;
  const result: SingleResult = {
    agent: agent.name,
    agentSource: agent.source,
    task,
    exitCode: 0,
    messages: [],
    stderr: '',
    usage: emptyUsage(),
    model: requestedModel,
    step,
  };
  const emitUpdate = () =>
    onUpdate?.({
      content: [{ type: 'text', text: getFinalOutput(result.messages) || '(running...)' }],
      details: makeDetails([result]),
    });

  try {
    if (agent.systemPrompt.trim()) {
      promptDir = fs.mkdtempSync(path.join(tmpdir(), 'mlx-subagent-'));
      const promptPath = path.join(promptDir, `prompt-${agent.name.replace(/[^\w.-]+/g, '_')}.md`);
      fs.writeFileSync(promptPath, agent.systemPrompt, { encoding: 'utf8', mode: 0o600 });
      args.push('--append-system-prompt', promptPath);
    }
    args.push(`Task: ${task}`);

    const invocation = options.invocation ?? defaultInvocation();
    const childArgs = [...invocation.args, ...args];
    const child = (options.spawnChild ?? productionSpawn)(invocation.command, childArgs, {
      cwd: cwd ?? defaultCwd,
      env: {
        ...process.env,
        [CHILD_ENV]: '1',
        // The parent permission gate approved this delegated capability. Keep
        // auto-approval scoped to the child instead of weakening the parent.
        MLX_AGENT_AUTO_APPROVE: '1',
      },
    });

    let buffer = '';
    let aborted = false;
    let settled = false;
    let killTimer: ReturnType<typeof setTimeout> | undefined;
    const processLine = (line: string) => {
      if (!line.trim()) return;
      let event: Record<string, unknown>;
      try {
        event = JSON.parse(line) as Record<string, unknown>;
      } catch {
        return;
      }
      if ((event['type'] === 'message_end' || event['type'] === 'tool_result_end') && event['message']) {
        const message = event['message'] as Message;
        result.messages.push(message);
        if (event['type'] === 'message_end' && message.role === 'assistant') {
          result.usage.turns++;
          const usage = message.usage;
          result.usage.input += usage?.input ?? 0;
          result.usage.output += usage?.output ?? 0;
          result.usage.cacheRead += usage?.cacheRead ?? 0;
          result.usage.cacheWrite += usage?.cacheWrite ?? 0;
          result.usage.cost += usage?.cost?.total ?? 0;
          result.usage.contextTokens = usage?.totalTokens ?? 0;
          result.model ??= message.model;
          result.stopReason = message.stopReason;
          result.errorMessage = message.errorMessage;
        }
        emitUpdate();
      }
    };
    child.stdout.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) processLine(line);
    });
    child.stderr.on('data', (chunk) => {
      result.stderr += chunk.toString();
    });

    const abort = () => {
      aborted = true;
      child.kill('SIGTERM');
      killTimer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
      }, 5000);
      killTimer.unref?.();
    };
    if (signal?.aborted) abort();
    else signal?.addEventListener('abort', abort, { once: true });

    result.exitCode = await new Promise<number>((resolve) => {
      const finish = (code: number) => {
        if (settled) return;
        settled = true;
        resolve(code);
      };
      child.once('close', (code) => {
        if (buffer.trim()) processLine(buffer);
        finish(code ?? 1);
      });
      child.once('error', (error) => {
        result.stderr += error.message;
        finish(1);
      });
    });
    signal?.removeEventListener('abort', abort);
    if (killTimer) clearTimeout(killTimer);
    if (aborted) {
      result.stopReason = 'aborted';
      result.errorMessage = 'Subagent was aborted';
    }
    return result;
  } finally {
    if (promptDir) fs.rmSync(promptDir, { recursive: true, force: true });
  }
}

const TaskItem = Type.Object({
  agent: Type.String({ description: 'Name of the agent to invoke' }),
  task: Type.String({ description: 'Task to delegate' }),
  cwd: Type.Optional(Type.String({ description: 'Working directory for the child process' })),
});

const Params = Type.Object({
  agent: Type.Optional(Type.String()),
  task: Type.Optional(Type.String()),
  cwd: Type.Optional(Type.String()),
  tasks: Type.Optional(Type.Array(TaskItem)),
  chain: Type.Optional(Type.Array(TaskItem)),
  agentScope: Type.Optional(StringEnum(['user', 'project', 'both'] as const, { default: 'user' })),
});

export function createSubagentExtension(options: SubagentExtensionOptions): InlineExtension {
  return {
    name: 'mlx-subagent',
    factory: (pi: ExtensionAPI) => {
      pi.registerTool({
        name: 'subagent',
        label: 'Subagent',
        description:
          'Delegate one task, a sequential chain, or parallel-shaped tasks to isolated local mlx agents. ' +
          'Built-ins: scout, planner, reviewer, worker. Local safety serializes all child processes.',
        promptSnippet: 'Delegate isolated research, planning, review, or implementation with the subagent tool.',
        promptGuidelines: [
          'Use subagents for bounded work that benefits from an isolated context.',
          'Although the tool accepts a tasks array, local mlx children run one at a time to avoid duplicate model/KV pressure.',
        ],
        parameters: Params,
        executionMode: 'sequential',
        async execute(_id, params, signal, onUpdate, ctx) {
          const scope: AgentScope = params.agentScope ?? 'user';
          const discovery = discoverSubagents(ctx.cwd, scope);
          const hasSingle = Boolean(params.agent && params.task);
          const hasParallel = Boolean(params.tasks?.length);
          const hasChain = Boolean(params.chain?.length);
          const modeCount = Number(hasSingle) + Number(hasParallel) + Number(hasChain);
          const mode: SubagentDetails['mode'] = hasChain ? 'chain' : hasParallel ? 'parallel' : 'single';
          const makeDetails = (results: SingleResult[]): SubagentDetails => ({
            mode,
            agentScope: scope,
            projectAgentsDir: discovery.projectAgentsDir,
            results,
          });
          if (modeCount !== 1) {
            return {
              content: [{ type: 'text', text: 'Provide exactly one mode: agent+task, tasks, or chain.' }],
              details: makeDetails([]),
              isError: true,
            };
          }

          const requested = new Set<string>();
          if (params.agent) requested.add(params.agent);
          for (const item of params.tasks ?? []) requested.add(item.agent);
          for (const item of params.chain ?? []) requested.add(item.agent);
          const projectAgents = [...requested]
            .map((name) => discovery.agents.find((agent) => agent.name === name))
            .filter((agent): agent is SubagentConfig => agent?.source === 'project');
          if (projectAgents.length) {
            if (!ctx.hasUI) {
              return {
                content: [{ type: 'text', text: 'Project-local subagents require interactive confirmation.' }],
                details: makeDetails([]),
                isError: true,
              };
            }
            const approved = await ctx.ui.confirm(
              'Run project-local agents?',
              `Agents: ${projectAgents.map((agent) => agent.name).join(', ')}\nSource: ${discovery.projectAgentsDir}`,
            );
            if (!approved) {
              return {
                content: [{ type: 'text', text: 'Canceled by user.' }],
                details: makeDetails([]),
                isError: true,
              };
            }
          }

          const parentModel = ctx.model?.provider === 'mlx' ? `mlx/${ctx.model.id}` : undefined;
          const run = (
            agent: string,
            task: string,
            cwd: string | undefined,
            step: number | undefined,
            update = onUpdate,
          ) =>
            runSingleAgent(
              options,
              ctx.cwd,
              discovery.agents,
              agent,
              task,
              cwd,
              parentModel,
              step,
              signal,
              update,
              makeDetails,
            );

          if (params.chain?.length) {
            const results: SingleResult[] = [];
            let previous = '';
            for (let i = 0; i < params.chain.length; i++) {
              const item = params.chain[i]!;
              const result = await run(item.agent, item.task.replace(/\{previous\}/g, previous), item.cwd, i + 1);
              results.push(result);
              if (isFailed(result)) {
                return {
                  content: [{ type: 'text', text: `Chain stopped at step ${i + 1}: ${resultOutput(result)}` }],
                  details: makeDetails(results),
                  isError: true,
                };
              }
              previous = getFinalOutput(result.messages);
            }
            return { content: [{ type: 'text', text: previous || '(no output)' }], details: makeDetails(results) };
          }

          if (params.tasks?.length) {
            if (params.tasks.length > MAX_PARALLEL_TASKS) {
              return {
                content: [
                  { type: 'text', text: `Too many tasks (${params.tasks.length}); max is ${MAX_PARALLEL_TASKS}.` },
                ],
                details: makeDetails([]),
                isError: true,
              };
            }
            const results = await mapWithConcurrencyLimit(params.tasks, (item) =>
              run(item.agent, item.task, item.cwd, undefined, undefined),
            );
            const summaries = results.map(
              (result) =>
                `### [${result.agent}] ${isFailed(result) ? 'failed' : 'completed'}\n\n${truncateOutput(resultOutput(result))}`,
            );
            const success = results.filter((result) => !isFailed(result)).length;
            return {
              content: [
                {
                  type: 'text',
                  text: `Parallel-shaped queue: ${success}/${results.length} succeeded\n\n${summaries.join('\n\n---\n\n')}`,
                },
              ],
              details: makeDetails(results),
              isError: success !== results.length,
            };
          }

          const result = await run(params.agent!, params.task!, params.cwd, undefined);
          return {
            content: [{ type: 'text', text: resultOutput(result) }],
            details: makeDetails([result]),
            isError: isFailed(result),
          };
        },
      });
    },
  };
}

export function isSubagentChild(): boolean {
  return process.env[CHILD_ENV] === '1';
}
