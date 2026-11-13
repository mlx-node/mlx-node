/**
 * An `AgentHost` wired to the scripted fake agent in a throwaway root under the OS temp dir. No network, no real
 * agent, no real home: the Work agent home and the Work process home are temp folders too.
 */

import {
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgentSessionsReader } from '../../src/agent-sessions/reader.js';
import type { UpdateLine } from '../../src/agent-sessions/transcript.js';
import {
  AgentHost,
  type AgentHostListener,
  type AgentHostOptions,
  type HostTimings,
  type TurnResult,
} from '../../src/chat/agent/agent-host.js';
import type { InferenceEndpoint } from '../../src/chat/agent/endpoint.js';
import { readWorkIndexSync, workIds } from '../../src/chat/index-file.js';
import { chatPaths, type ChatPaths } from '../../src/chat/paths.js';
import { MemorySecretStore, type SecretStore } from '../../src/chat/providers/secrets.js';
import { ProvidersService } from '../../src/chat/providers/service.js';
import type { ProviderSource } from '../../src/chat/providers/source.js';
import { ChatService, type ChatServiceOptions } from '../../src/chat/service.js';
import type { AgentModeState, ChatMode, ConversationNotice, Interaction, TurnState } from '../../src/chat/types.js';
import type { WebResult } from '../../src/chat/web/result-text.js';
import { startFakeEngine, type EngineScript, type FakeEngine } from './fake-engine.js';

export const FAKE_AGENT = fileURLToPath(new URL('./fake-agent.mjs', import.meta.url));
export const SCENARIOS = fileURLToPath(new URL('../fixtures/chat-scenarios/', import.meta.url));
export const INFERENCE_TOKEN = 'tok-INFERENCE-0123456789abcdef';

export type Json = Record<string, unknown>;

export interface HostEvents {
  states: AgentModeState[];
  updates: Array<{ sessionId: string; line: UpdateLine }>;
  titles: Array<{ sessionId: string; title: string }>;
  turns: Array<{ sessionId: string; turn: TurnState | null }>;
  ends: Array<{ sessionId: string; result: TurnResult }>;
  interactions: Array<{ sessionId: string; interaction: Interaction }>;
  closed: Array<{ sessionId: string; id: string; reason: string }>;
  cold: string[][];
  web: Array<{ sessionId: string; result: WebResult }>;
  models: Array<{ sessionId: string; modelId: string }>;
  notices: Array<{ sessionId: string; notice: ConversationNotice }>;
  /** Work: sessions that passed their guard after `session/new` or `session/load`. */
  ready: string[];
}

/** An endpoint file model that is its own base, as an MLX folder with no build words in its name. */
export function endpointModel(name: string, path: string): InferenceEndpoint['models'][number] {
  return { name, path, base: name, variant: 'MLX', format: 'mlx' };
}

export interface FakeHostOptions {
  mode?: ChatMode;
  timings?: Partial<HostTimings>;
  /** The `models` of the endpoint file (default: one model). */
  endpointModels?: InferenceEndpoint['models'];
  providers?: ProviderSource;
  mutate?: (scenario: Json) => void;
  /** How the fake engine answers (default: 200 to every select call). */
  engine?: EngineScript;
  /** Work: the permission mode, as the service passes it. */
  permission?: AgentHostOptions['permission'];
  onModeAllowed?: AgentHostListener['onModeAllowed'];
  onModeMarked?: AgentHostListener['onModeMarked'];
  onModeWithdrawn?: AgentHostListener['onModeWithdrawn'];
  /** Added to the Work environment (`CDPATH`, …). */
  env?: Record<string, string>;
  spawn?: AgentHostOptions['spawn'];
  /** Work: the temp roots Auto covers beside the Work folder, and whose files are yours there. */
  tempRoots?: AgentHostOptions['tempRoots'];
  uid?: number;
  login?: string;
}

export interface FakeHost {
  host: AgentHost;
  root: string;
  paths: ChatPaths;
  /** Work: a project folder inside the temp root. */
  project: string;
  /** Work: the temp agent home. */
  agentHome: string;
  events: HostEvents;
  endpoint: { current: InferenceEndpoint | null };
  /** The engine the endpoint names. */
  engine: FakeEngine;
  /** Every message the fake agent received, in order. */
  sent(): Json[];
  /** Every line the fake recorded (start lines included). */
  records(): Json[];
  close(): Promise<void>;
}

/** A provider source with no provider: the web tools answer `not-configured`, Work gets no key. */
export function noProviders(overrides: Partial<ProviderSource> = {}): ProviderSource {
  return {
    chatWeb: () => ({ searchProvider: 'none', searchKey: '', fetchProvider: 'none', fetchKey: '' }),
    workKeys: () => ({}),
    workPluginDirs: () => [],
    secretNeedles: () => [],
    keyNeedles: () => [],
    markKeyCheck: () => {},
    settled: () => Promise.resolve(),
    chatSkillReady: () => Promise.resolve(true),
    summary: () => ({ search: null, searchCheck: null, firecrawl: false, fetch: null, applyPending: false }),
    ...overrides,
  };
}

/** A fetch that never reaches the network: unit tests have none. */
export const NO_NETWORK: typeof fetch = () => Promise.reject(new Error('no network in tests'));

/** The real providers service on a memory store, with no network and an empty `PATH`. */
export function memoryProviders(
  paths: ChatPaths,
  options: { secrets?: SecretStore; skillsDir?: string; fetch?: typeof fetch; workPath?: string } = {},
): ProvidersService {
  return new ProvidersService({
    paths,
    secrets: options.secrets ?? new MemorySecretStore(),
    skillsDir: options.skillsDir,
    fetch: options.fetch ?? NO_NETWORK,
    workPath: () => options.workPath ?? '',
  });
}

export function readScenario(name: string): Json {
  return JSON.parse(readFileSync(join(SCENARIOS, `${name}.json`), 'utf8')) as Json;
}

export async function fakeHost(name: string, options: FakeHostOptions = {}): Promise<FakeHost> {
  const root = mkdtempSync(join(tmpdir(), 'mlx-chat-test-'));
  const paths = chatPaths(join(root, 'chat'));
  const project = join(root, 'project');
  const agentHome = join(root, 'agent-home');
  const userHome = join(root, 'home');
  for (const dir of [project, agentHome, userHome]) mkdirSync(dir, { recursive: true });
  const recordTo = join(root, 'record.jsonl');
  const scenario = { ...readScenario(name), recordTo };
  options.mutate?.(scenario);
  const scenarioPath = join(root, 'scenario.json');
  writeFileSync(scenarioPath, JSON.stringify(scenario));
  const engine = await startFakeEngine(options.engine);
  const endpoint: { current: InferenceEndpoint | null } = {
    current: {
      version: 1,
      pid: process.pid,
      url: engine.url,
      token: INFERENCE_TOKEN,
      models: options.endpointModels ?? [endpointModel('qwen3.5-4b', '/models/qwen3.5-4b')],
    },
  };
  const events: HostEvents = {
    states: [],
    updates: [],
    titles: [],
    turns: [],
    ends: [],
    interactions: [],
    closed: [],
    cold: [],
    web: [],
    models: [],
    notices: [],
    ready: [],
  };
  const mode = options.mode ?? 'chat';
  const host = new AgentHost({
    mode,
    paths,
    resolveLaunch: () => ({
      command: process.execPath,
      args: [FAKE_AGENT, scenarioPath],
      source: 'test',
      tag: null,
      versionVerified: false,
    }),
    readEndpoint: async () => endpoint.current,
    providers: options.providers ?? noProviders(),
    clientVersion: '0.0.0-test',
    chatPrompt: 'TEST CHAT PROMPT',
    webServer: { command: process.execPath, script: join(root, 'no-server.js') },
    baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: userHome, ...options.env },
    agentHome,
    homeDir: userHome,
    username: 'tester',
    timings: { stopEofMs: 300, stopTermMs: 300, ...options.timings },
    ...(options.permission === undefined ? {} : { permission: options.permission }),
    ...(options.spawn === undefined ? {} : { spawn: options.spawn }),
    tempRoots: options.tempRoots ?? [],
    ...(options.uid === undefined ? {} : { uid: options.uid }),
    ...(options.login === undefined ? {} : { login: options.login }),
    listener: {
      onSessionReady: (sessionId) => events.ready.push(sessionId),
      ...(options.onModeAllowed === undefined ? {} : { onModeAllowed: options.onModeAllowed }),
      ...(options.onModeMarked === undefined ? {} : { onModeMarked: options.onModeMarked }),
      ...(options.onModeWithdrawn === undefined ? {} : { onModeWithdrawn: options.onModeWithdrawn }),
      onState: (s) => events.states.push(s),
      onUpdate: (sessionId, line) => events.updates.push({ sessionId, line }),
      onTitle: (sessionId, title) => events.titles.push({ sessionId, title }),
      onTurn: (sessionId, turn) => events.turns.push({ sessionId, turn }),
      onTurnEnd: (sessionId, result) => events.ends.push({ sessionId, result }),
      onInteraction: (sessionId, interaction) => events.interactions.push({ sessionId, interaction }),
      onInteractionClosed: (sessionId, id, reason) => events.closed.push({ sessionId, id, reason }),
      onCold: (ids) => events.cold.push(ids),
      onWebResult: (sessionId, result) => events.web.push({ sessionId, result }),
      onModel: (sessionId, modelId) => events.models.push({ sessionId, modelId }),
      onNotice: (sessionId, notice) => events.notices.push({ sessionId, notice }),
    },
  });
  const records = (): Json[] => {
    try {
      return readFileSync(recordTo, 'utf8')
        .split('\n')
        .filter((l) => l !== '')
        .map((l) => JSON.parse(l) as Json);
    } catch {
      return [];
    }
  };
  return {
    host,
    root,
    paths,
    project,
    agentHome,
    events,
    endpoint,
    engine,
    records,
    sent: () =>
      records()
        .filter((r) => r.kind === 'in')
        .map((r) => r.msg as Json),
    close: async () => {
      await host.close();
      await engine.close();
      rmSync(root, { recursive: true, force: true });
    },
  };
}

/** Waits until `check` holds, polling every 10 ms, at most `ms`. */
export async function until(check: () => boolean, ms = 3_000): Promise<void> {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error('condition not reached');
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

export interface FakeServiceOptions {
  mutate?: (scenario: Json) => void;
  /** Built over the service's own paths; default: the memory store with no key. */
  providers?: (paths: ChatPaths) => ProvidersService;
  timings?: Partial<HostTimings>;
  /** The Chat skills folder (default: the package's own). */
  skillsDir?: string;
  createBudgetMs?: number;
  /** Delays every disk read of a session, as a busy database worker would. */
  readSessionDelayMs?: number;
  /** Awaited before every disk read of a session. */
  readSessionHold?: () => Promise<void>;
  /** How long Work questions wait for a page before they are denied. */
  pageGoneMs?: number;
  needlesWaitMs?: number;
  /** How the fake engine answers (default: 200 to every select call). */
  engine?: EngineScript;
  /** The `models` of the endpoint file (default: two models). */
  endpointModels?: InferenceEndpoint['models'];
  live?: ChatServiceOptions['live'];
  /** The temp roots Auto covers (default: none); `'default'` leaves the service's own default. */
  tempRoots?: ChatServiceOptions['tempRoots'] | 'default';
}

export interface FakeService {
  service: ChatService;
  root: string;
  paths: ChatPaths;
  endpoint: { current: InferenceEndpoint | null };
  /** The engine the endpoint names; started before the first endpoint read answers. */
  engine(): Promise<FakeEngine>;
  /** The Chat agent home's reader, as the database worker has it. */
  reader: AgentSessionsReader;
  sent(): Json[];
  records(): Json[];
  /** A new service over the same folders: a Control Panel restart. The old one must be closed first. */
  restart(): ChatService;
  close(): Promise<void>;
}

/** A `ChatService` on the fake agent in a throwaway root; sessions are read from disk as the worker reads them. */
export function fakeService(name: string, options: FakeServiceOptions = {}): FakeService {
  const root = mkdtempSync(join(tmpdir(), 'mlx-chat-svc-'));
  const paths = chatPaths(join(root, 'chat'));
  const userHome = join(root, 'home');
  mkdirSync(userHome, { recursive: true });
  const recordTo = join(root, 'record.jsonl');
  const scenario = { ...readScenario(name), recordTo };
  options.mutate?.(scenario);
  const scenarioPath = join(root, 'scenario.json');
  writeFileSync(scenarioPath, JSON.stringify(scenario));
  const unbound = 'http://127.0.0.1:9';
  const endpoint: { current: InferenceEndpoint | null } = {
    current: {
      version: 1,
      pid: process.pid,
      url: unbound,
      token: INFERENCE_TOKEN,
      models: options.endpointModels ?? [
        endpointModel('qwen3.5-4b', '/models/qwen3.5-4b'),
        endpointModel('qwen3.5-9b', '/models/qwen3.5-9b'),
      ],
    },
  };
  const engine = startFakeEngine(options.engine).then((started) => {
    if (endpoint.current?.url === unbound) endpoint.current = { ...endpoint.current, url: started.url };
    return started;
  });
  const reader = new AgentSessionsReader({ agentHome: paths.chatAgent, origin: 'chat' });
  const workReader = new AgentSessionsReader({ agentHome: join(root, 'agent-home') });
  const providersOf = options.providers ?? ((p: ChatPaths) => memoryProviders(p, { skillsDir: options.skillsDir }));
  const make = (): ChatService =>
    new ChatService({
      paths,
      readEndpoint: async () => {
        await engine;
        return endpoint.current;
      },
      ...(options.live === undefined ? {} : { live: options.live }),
      resolveAgent: () => ({
        launch: {
          command: process.execPath,
          args: [FAKE_AGENT, scenarioPath],
          source: 'test',
          tag: null,
          versionVerified: false,
        },
        source: 'test',
        tag: null,
      }),
      readSession: async (id) => {
        if (options.readSessionDelayMs !== undefined) {
          await new Promise((resolve) => setTimeout(resolve, options.readSessionDelayMs));
        }
        await options.readSessionHold?.();
        const chat = reader.detail(id);
        if (chat !== null) return chat;
        // As the database worker: a Work row is a session of the Work agent home named in the Work index.
        const work = workReader.detail(id);
        if (work === null || !workIds(readWorkIndexSync(paths.conversations)).has(id)) return null;
        return { ...work, session: { ...work.session, origin: 'work' } };
      },
      providers: providersOf(paths),
      clientVersion: '0.0.0-test',
      webServer: { command: process.execPath, script: join(root, 'no-server.js') },
      skillsDir: options.skillsDir,
      baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: userHome },
      agentHome: join(root, 'agent-home'),
      workAgentHome: join(root, 'agent-home'),
      homeDir: userHome,
      ...(options.tempRoots === 'default' ? {} : { tempRoots: options.tempRoots ?? [] }),
      username: 'tester',
      pageGoneMs: options.pageGoneMs,
      needlesWaitMs: options.needlesWaitMs,
      timings: { stopEofMs: 300, stopTermMs: 300, ...options.timings },
      flushMs: 5,
      stateCoalesceMs: 5,
      createBudgetMs: options.createBudgetMs,
    });
  const records = (): Json[] => {
    try {
      return readFileSync(recordTo, 'utf8')
        .split('\n')
        .filter((l) => l !== '')
        .map((l) => JSON.parse(l) as Json);
    } catch {
      return [];
    }
  };
  const fake: FakeService = {
    service: make(),
    root,
    paths,
    endpoint,
    engine: () => engine,
    reader,
    records,
    sent: () =>
      records()
        .filter((r) => r.kind === 'in')
        .map((r) => r.msg as Json),
    restart: () => {
      fake.service = make();
      return fake.service;
    },
    close: async () => {
      await fake.service.close();
      await (await engine).close();
      rmSync(root, { recursive: true, force: true });
    },
  };
  return fake;
}

/** Sets every file and folder under `dir` to `when`: the session writers refuse files written in the last 30 s. */
export function ageTree(dir: string, when: number): void {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    const st = lstatSync(path);
    if (st.isSymbolicLink()) continue;
    if (st.isDirectory()) ageTree(path, when);
    utimesSync(path, new Date(when), new Date(when));
  }
}
