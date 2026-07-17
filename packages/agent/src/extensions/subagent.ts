/**
 * MLX port of pi's official `examples/extensions/subagent` extension.
 *
 * The upstream example starts a fresh pi process per task. MLX instead creates
 * an in-process `AgentSession` for each task so all delegated sessions reuse
 * the parent's registered provider and its single `MlxModelHost`. Each task
 * still has isolated conversation/compaction state and its own cwd/tools:
 *
 * - at most four task loops run concurrently; the host serializes inference,
 * - sessions inherit the parent's current local model/model registry,
 * - context files and skills load, while extensions/prompts/themes do not,
 * - the parent permission gate approves delegated tool access once up front.
 *
 * Upstream source: @earendil-works/pi-coding-agent 0.80.6,
 * examples/extensions/subagent (MIT).
 */

import * as fs from 'node:fs';
import { homedir } from 'node:os';
import * as path from 'node:path';

import type { Message } from '@earendil-works/pi-ai';
import { StringEnum } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext, InlineExtension } from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';

import { sanitizeApprovalDetail } from './approval-detail.js';

const MAX_PARALLEL_TASKS = 8;
/** Tool loops can overlap; the shared `MlxModelHost` serializes model calls. */
const MAX_CONCURRENCY = 4;
const PER_TASK_OUTPUT_CAP = 50 * 1024;

type AgentScope = 'user' | 'project' | 'both';
type AgentSource = 'builtin' | 'user' | 'project' | 'unknown';
export type SubagentMode = 'single' | 'parallel' | 'chain';

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
  mode: SubagentMode;
  agentScope: AgentScope;
  projectAgentsDir: string | null;
  results: SingleResult[];
}

export interface InProcessSubagentSession {
  subscribe(listener: (event: unknown) => void): () => void;
  prompt(text: string): Promise<void>;
  abort(): Promise<void>;
  dispose(): void;
}

export interface SubagentSessionCreateOptions {
  cwd: string;
  model: NonNullable<ExtensionContext['model']>;
  modelRegistry: ExtensionContext['modelRegistry'];
  tools?: string[];
  systemPrompt: string;
}

export interface SubagentExtensionOptions {
  /** Test/programmatic seam. Production uses pi's in-process SDK. */
  createSession?: (options: SubagentSessionCreateOptions) => Promise<InProcessSubagentSession>;
}

export interface SubagentCompactionSettings {
  enabled: boolean;
  reserveTokens: number;
  keepRecentTokens: number;
}

const COMPACTION_RESERVE_WINDOW_FRACTION = 0.25;
const COMPACTION_KEEP_RECENT_WINDOW_FRACTION = 0.5;
const COMPACTION_TOTAL_WINDOW_FRACTION = 0.75;

/**
 * Fit pi's compaction budgets to the model's effective physical context.
 *
 * Pi's defaults (16,384 reserved + 20,000 recent) assume a context larger
 * than some dynamically sized MLX KV pools. Preserve explicitly smaller user
 * values, but keep the summary reserve at most 25%, retained history at most
 * 50%, and their combined budget at most 75% of the usable window.
 */
export function scaleSubagentCompactionSettings(
  contextWindow: number,
  current: SubagentCompactionSettings,
): SubagentCompactionSettings {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) {
    throw new RangeError(`Subagent model reported an invalid context window: ${contextWindow}`);
  }

  const window = Math.floor(contextWindow);
  const reserveCap = Math.floor(window * COMPACTION_RESERVE_WINDOW_FRACTION);
  const keepRecentCap = Math.floor(window * COMPACTION_KEEP_RECENT_WINDOW_FRACTION);
  const totalCap = Math.floor(window * COMPACTION_TOTAL_WINDOW_FRACTION);
  const capConfiguredValue = (value: number, cap: number): number => {
    if (!Number.isFinite(value)) return cap;
    return Math.min(Math.max(0, Math.floor(value)), cap);
  };

  const reserveTokens = capConfiguredValue(current.reserveTokens, reserveCap);
  const keepRecentTokens = Math.min(
    capConfiguredValue(current.keepRecentTokens, keepRecentCap),
    Math.max(0, totalCap - reserveTokens),
  );
  return { enabled: current.enabled, reserveTokens, keepRecentTokens };
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

function normalizedModelId(model: string): string | undefined {
  if (model.startsWith('mlx/')) return model.slice('mlx/'.length) || undefined;
  // A bare name is a local discovered model id. A provider/name pair is not.
  if (!model.includes('/')) return model;
  return undefined;
}

interface SubagentRequestShape {
  agent?: string;
  task?: string;
  tasks?: unknown[];
  chain?: unknown[];
}

/**
 * Select the mode with the same precedence used by execution. Permission UI
 * imports this helper so a stray top-level field can never hide the queued
 * agents that will actually receive delegated tool access.
 */
export function normalizeSubagentMode(input: SubagentRequestShape): {
  mode: SubagentMode;
  modeCount: number;
} {
  const hasSingle = Boolean(input.agent && input.task);
  const hasParallel = Boolean(input.tasks?.length);
  const hasChain = Boolean(input.chain?.length);
  return {
    mode: hasChain ? 'chain' : hasParallel ? 'parallel' : 'single',
    modeCount: Number(hasSingle) + Number(hasParallel) + Number(hasChain),
  };
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

async function createProductionSession(options: SubagentSessionCreateOptions): Promise<InProcessSubagentSession> {
  // `runAgent()` seeds pi's config environment before the extension can execute,
  // so defer the runtime import until now instead of importing pi at module load.
  const { createAgentSession, DefaultResourceLoader, getAgentDir, SessionManager, SettingsManager } =
    await import('@earendil-works/pi-coding-agent');
  const agentDir = getAgentDir();
  const settingsManager = SettingsManager.create(options.cwd, agentDir);
  settingsManager.applyOverrides({
    compaction: scaleSubagentCompactionSettings(options.model.contextWindow, settingsManager.getCompactionSettings()),
  });
  const resourceLoader = new DefaultResourceLoader({
    cwd: options.cwd,
    agentDir,
    settingsManager,
    // Subagents keep project/user context files and skills, but cannot load the
    // subagent extension recursively or introduce unrelated prompt/theme state.
    noExtensions: true,
    noPromptTemplates: true,
    noThemes: true,
    appendSystemPromptOverride: (base) => (options.systemPrompt.trim() ? [...base, options.systemPrompt] : base),
  });
  await resourceLoader.reload();
  const { session } = await createAgentSession({
    cwd: options.cwd,
    agentDir,
    model: options.model,
    modelRegistry: options.modelRegistry,
    tools: options.tools,
    resourceLoader,
    sessionManager: SessionManager.inMemory(options.cwd),
    settingsManager,
  });
  return session;
}

async function runSingleAgent(
  options: SubagentExtensionOptions,
  defaultCwd: string,
  agents: SubagentConfig[],
  agentName: string,
  task: string,
  cwd: string | undefined,
  context: ExtensionContext,
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

  const requestedModelId = agent.model ? normalizedModelId(agent.model) : undefined;
  if (agent.model && !requestedModelId) {
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

  const model = requestedModelId ? context.modelRegistry.find('mlx', requestedModelId) : context.model;
  if (!model || model.provider !== 'mlx') {
    const requested = requestedModelId ? `mlx/${requestedModelId}` : 'the parent mlx model';
    return {
      agent: agent.name,
      agentSource: agent.source,
      task,
      exitCode: 1,
      messages: [],
      stderr: `Agent ${agent.name} could not resolve ${requested} in the parent model registry.`,
      usage: emptyUsage(),
      step,
    };
  }

  const result: SingleResult = {
    agent: agent.name,
    agentSource: agent.source,
    task,
    exitCode: 0,
    messages: [],
    stderr: '',
    usage: emptyUsage(),
    model: `mlx/${model.id}`,
    step,
  };
  const emitUpdate = () =>
    onUpdate?.({
      content: [{ type: 'text', text: getFinalOutput(result.messages) || '(running...)' }],
      details: makeDetails([result]),
    });

  let session: InProcessSubagentSession | undefined;
  let unsubscribe: (() => void) | undefined;
  let aborted = signal?.aborted ?? false;
  let abortPromise: Promise<void> | undefined;
  const abort = () => {
    aborted = true;
    if (session) abortPromise = session.abort().catch(() => undefined);
  };
  if (!signal?.aborted) signal?.addEventListener('abort', abort, { once: true });

  try {
    session = await (options.createSession ?? createProductionSession)({
      cwd: cwd ?? defaultCwd,
      model,
      modelRegistry: context.modelRegistry,
      tools: agent.tools,
      systemPrompt: agent.systemPrompt,
    });
    if (aborted) {
      abort();
      await abortPromise;
      result.exitCode = 1;
      result.stopReason = 'aborted';
      result.errorMessage = 'Subagent was aborted';
      return result;
    }

    unsubscribe = session.subscribe((rawEvent) => {
      const event = rawEvent as { type?: unknown; message?: unknown };
      if (event.type !== 'message_end' || !event.message) return;
      const message = event.message as Message;
      result.messages.push(message);
      if (message.role === 'assistant') {
        result.usage.turns++;
        const usage = message.usage;
        result.usage.input += usage?.input ?? 0;
        result.usage.output += usage?.output ?? 0;
        result.usage.cacheRead += usage?.cacheRead ?? 0;
        result.usage.cacheWrite += usage?.cacheWrite ?? 0;
        result.usage.cost += usage?.cost?.total ?? 0;
        result.usage.contextTokens = usage?.totalTokens ?? 0;
        result.model = message.provider && message.model ? `${message.provider}/${message.model}` : result.model;
        result.stopReason = message.stopReason;
        result.errorMessage = message.errorMessage;
      }
      emitUpdate();
    });

    try {
      await session.prompt(`Task: ${task}`);
      await abortPromise;
    } catch (error) {
      result.exitCode = 1;
      result.stderr = error instanceof Error ? error.message : String(error);
    }
    if (aborted) {
      result.exitCode = 1;
      result.stopReason = 'aborted';
      result.errorMessage = 'Subagent was aborted';
    } else if (result.stopReason === 'error') {
      result.exitCode = 1;
    }
    return result;
  } catch (error) {
    result.exitCode = 1;
    result.stderr = error instanceof Error ? error.message : String(error);
    if (aborted) {
      result.stopReason = 'aborted';
      result.errorMessage = 'Subagent was aborted';
    }
    return result;
  } finally {
    signal?.removeEventListener('abort', abort);
    unsubscribe?.();
    session?.dispose();
  }
}

const TaskItem = Type.Object({
  agent: Type.String({ description: 'Name of the agent to invoke' }),
  task: Type.String({ description: 'Task to delegate' }),
  cwd: Type.Optional(Type.String({ description: 'Working directory for the isolated session' })),
});

const Params = Type.Object({
  agent: Type.Optional(Type.String()),
  task: Type.Optional(Type.String()),
  cwd: Type.Optional(Type.String()),
  tasks: Type.Optional(Type.Array(TaskItem)),
  chain: Type.Optional(Type.Array(TaskItem)),
  agentScope: Type.Optional(StringEnum(['user', 'project', 'both'] as const, { default: 'user' })),
});

export function createSubagentExtension(options: SubagentExtensionOptions = {}): InlineExtension {
  return {
    name: 'mlx-subagent',
    factory: (pi: ExtensionAPI) => {
      pi.registerTool({
        name: 'subagent',
        label: 'Subagent',
        description:
          'Delegate one task, a sequential chain, or concurrent tasks to isolated in-process mlx agent sessions. ' +
          'Built-ins: scout, planner, reviewer, worker. Sessions share one model host and KV pool.',
        promptSnippet: 'Delegate isolated research, planning, review, or implementation with the subagent tool.',
        promptGuidelines: [
          'Use subagents for bounded work that benefits from an isolated context.',
          'A tasks array runs up to four independent tool loops concurrently; shared mlx inference remains serialized.',
        ],
        parameters: Params,
        executionMode: 'sequential',
        async execute(_id, params, signal, onUpdate, ctx) {
          const scope: AgentScope = params.agentScope ?? 'user';
          const discovery = discoverSubagents(ctx.cwd, scope);
          const { mode, modeCount } = normalizeSubagentMode(params);
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
          if (mode === 'single' && params.agent) requested.add(params.agent);
          if (mode === 'parallel') for (const item of params.tasks ?? []) requested.add(item.agent);
          if (mode === 'chain') for (const item of params.chain ?? []) requested.add(item.agent);
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
              sanitizeApprovalDetail(
                `Agents: ${projectAgents.map((agent) => agent.name).join(', ')}\nSource: ${discovery.projectAgentsDir}`,
              ),
            );
            if (!approved) {
              return {
                content: [{ type: 'text', text: 'Canceled by user.' }],
                details: makeDetails([]),
                isError: true,
              };
            }
          }

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
              ctx,
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
              const result = await run(
                item.agent,
                item.task.replace(/\{previous\}/g, () => previous),
                item.cwd,
                i + 1,
              );
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
                  text: `Concurrent tasks: ${success}/${results.length} succeeded\n\n${summaries.join('\n\n---\n\n')}`,
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
