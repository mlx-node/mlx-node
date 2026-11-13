/**
 * One agent process per mode, shared by all conversations of that mode: lazy start, start-up self-check, the
 * post-create guard, turns with their caps, permission answers, idle stop, stale restart and crash handling.
 *
 * Every request field is built here. Agent text (stderr, error messages, update payloads) is never logged and never
 * becomes a failure message: failures carry fixed text per code.
 */

import { randomUUID } from 'node:crypto';
import { realpath } from 'node:fs/promises';
import { homedir, userInfo } from 'node:os';

import { compareReleaseTags, MLX_AGENT_MIN_RELEASE, type AgentInstallProbe } from '@mlx-node/agent/binary';

import type { UpdateLine } from '../../agent-sessions/transcript.js';
import type { TranscriptEntry } from '../../agent-sessions/types.js';
import { ACP_TIMEOUTS, AcpRpcError, type ProtocolErrorReason } from '../acp/client.js';
import {
  readInitialize,
  readPermissionRequest,
  readPromptResult,
  readServerStatus,
  readSessionInfo,
  readSessionReply,
  readSessionUpdate,
  readTitle,
  type PermissionRequest,
  type SessionInfoFacts,
} from '../acp/messages.js';
import { AgentProcess, spawnAgent, type AgentExit, type SpawnAgent } from '../acp/process.js';
import {
  ensureChatDirs,
  pluginDir,
  removeChatPromptHistory,
  removeChatSearchIndex,
  writeFileAtomic,
  writeWorkGuard,
  type ChatPaths,
} from '../paths.js';
import type { ProviderSource } from '../providers/source.js';
import type { ProviderId } from '../providers/types.js';
import { needleList } from '../redact.js';
import type {
  AgentModeState,
  AutoAsk,
  PermissionMark,
  PermissionMode,
  ChatFailure,
  ChatFailureCode,
  ChatMode,
  ConversationNotice,
  Interaction,
  PermissionInteraction,
  TurnOutcome,
  TurnState,
} from '../types.js';
import { parseWebResultText, type WebResult } from '../web/result-text.js';
import { editLanding, type WorkFolderCheck } from '../work-folder.js';
import { AddressLedger } from './address-ledger.js';
import { foldName } from './auto-scan.js';
import {
  autoFacts,
  changesFolder,
  decideAuto,
  helperStart,
  staysBehind,
  type AutoAskDetail,
  type AutoDecision,
  type AutoFacts,
} from './auto.js';
import { agentPath, type WorkPlace } from './command-flags.js';
import { endpointFingerprint, sameEndpoint, type EndpointFingerprint, type EndpointReader } from './endpoint.js';
import { chatAgentEnv, workAgentEnv } from './env.js';
import { checkSession, selfCheck, type GuardResult } from './guard.js';
import { liveFilter } from './live-filter.js';
import {
  decideChatPermission,
  planAnswer,
  planInteraction,
  readPlanRequest,
  requestTarget,
  accessOf,
  causeOutcome,
  showableWhole,
  shownCut,
  turnOutcome,
  workInteraction,
  type PermissionAnswer,
  type TurnEndCause,
} from './permissions.js';
import {
  AGENT_ARGS,
  ALLOW_EDITS_SESSION_OPTION_ID,
  ALLOW_ONCE_OPTION_ID,
  ALWAYS_ALLOW_OPTION_ID,
  CHAT_CONFIG_TOML,
  CHAT_REQUIREMENTS_TOML,
  CHAT_SKILL_PLUGIN,
  type ChatWebEnv,
  chatSessionParams,
  initializeParams,
  loadSessionParams,
  NO_WEB,
  promptParams,
  REJECT_ONCE_OPTION_ID,
  WEB_SERVER_NAME,
  WEB_TOOL_KEYS,
  workGuardHookName,
  workGuardManifest,
  workGuardName,
  workSessionParams,
} from './profiles.js';
import { tempRoots, type AllowedRoot } from './roots.js';
import { MODEL_SELECT_MAX_MS, selectModel, type SelectAnswer, type SelectTarget } from './select-model.js';
import { applyWebResult, readWebToolResult } from './web-results.js';

export const AGENT_IDLE_STOP_MS = 600_000;
/** Chat only: a Work turn has no time limit. */
export const CHAT_TURN_MAX_MS = 600_000;
/** After `session/cancel` the prompt reply must come inside this, else the agent broke the protocol. */
export const CANCEL_REPLY_MS = 5_000;
/** Chat guard: how long to wait for `server_status ready` of `mlx_web` after the session reply. */
export const MCP_READY_MS = 10_000;
export const CRASH_WINDOW_MS = 300_000;
export const CRASHES_TO_FAIL = 3;
export const MISMATCHES_TO_BLOCK = 3;
/** A select call with no answer after this shows `wait: 'model'`: a resident model answers in 1–2 ms. */
export const MODEL_WAIT_SHOW_MS = 400;

/** The fixed text of every failure code. Never agent or provider text. */
export const FAILURE_MESSAGES: Readonly<Record<ChatFailureCode, string>> = {
  'inference-stopped': 'The inference engine is not running.',
  'inference-failed': 'The inference engine did not answer.',
  'no-models': 'No model yet.',
  'agent-missing': 'The agent is not installed.',
  'agent-too-old': 'The agent needs an update.',
  'agent-incompatible': 'This agent build does not match this app.',
  'agent-protocol-error': 'The agent broke the protocol and was stopped.',
  'agent-crashed': 'The agent stopped. Your conversation is saved.',
  'agent-failed': 'The agent keeps stopping.',
  'agent-slow': 'The agent did not answer in time.',
  'profile-mismatch': 'Chat could not be started safely.',
  'model-unavailable': 'That model is not available.',
  'model-load-failed': 'The model did not load.',
  'cwd-invalid': 'That folder cannot be used.',
  'cwd-mismatch': 'The agent opened another folder.',
  'conversation-busy': 'An answer is still running.',
  'engine-busy': 'The engine is busy: an answer or a model load runs.',
  'slash-text': 'Start the message with a word, not /.',
  'keychain-unavailable': 'The Keychain cannot be reached. Keys cannot be saved.',
  'skill-bundle-corrupt': "The app's skill files are damaged. Reinstall the app.",
};

export function chatFailure(code: ChatFailureCode): ChatFailure {
  return { code, message: FAILURE_MESSAGES[code] };
}

/** A typed failure of a host call. Its message is the fixed text of its code. */
export class AgentHostError extends Error {
  readonly code: ChatFailureCode;

  constructor(code: ChatFailureCode) {
    super(FAILURE_MESSAGES[code]);
    this.code = code;
    this.name = 'AgentHostError';
  }
}

/** An interaction answer that names no open interaction, or an option it did not offer. */
export class InteractionError extends Error {
  readonly kind: 'not-found' | 'not-offered';

  constructor(kind: 'not-found' | 'not-offered') {
    super(kind === 'not-found' ? 'No such request.' : 'That option was not offered.');
    this.kind = kind;
    this.name = 'InteractionError';
  }
}

/** How to start the agent: the resolved binary with {@link AGENT_ARGS}, or a test command (no version check). */
export interface AgentLaunch {
  command: string;
  args: readonly string[];
  /** Added to the spawn env (tests only). */
  env?: Readonly<Record<string, string>>;
  source: string;
  tag: string | null;
  /** False for an env or repo build, which has no release tag. */
  versionVerified: boolean;
}

export type LaunchResolution = AgentLaunch | { failure: 'agent-missing' | 'agent-too-old' };

/** The binary probe → a launch: a managed or bundled binary below the minimum release is too old. */
export function launchFromProbe(probe: AgentInstallProbe): LaunchResolution {
  if (probe.source === 'none' || probe.path === undefined) {
    const old =
      (probe.managedTag !== undefined && compareReleaseTags(probe.managedTag, MLX_AGENT_MIN_RELEASE) < 0) ||
      (probe.bundledTag !== undefined && compareReleaseTags(probe.bundledTag, MLX_AGENT_MIN_RELEASE) < 0);
    return { failure: old ? 'agent-too-old' : 'agent-missing' };
  }
  if (probe.source === 'env' || probe.source === 'vendored') {
    return { command: probe.path, args: AGENT_ARGS, source: probe.source, tag: null, versionVerified: false };
  }
  const tag = probe.effectiveTag ?? null;
  if (tag === null || compareReleaseTags(tag, MLX_AGENT_MIN_RELEASE) < 0) return { failure: 'agent-too-old' };
  return { command: probe.path, args: AGENT_ARGS, source: probe.source, tag, versionVerified: true };
}

export interface HostTimings {
  idleStopMs: number;
  chatTurnMaxMs: number;
  cancelReplyMs: number;
  mcpReadyMs: number;
  stopEofMs: number;
  stopTermMs: number;
  crashWindowMs: number;
  /** The ceiling of the model select call (the whole load). */
  modelSelectMaxMs: number;
  modelWaitShowMs: number;
}

const DEFAULT_TIMINGS: HostTimings = {
  idleStopMs: AGENT_IDLE_STOP_MS,
  chatTurnMaxMs: CHAT_TURN_MAX_MS,
  cancelReplyMs: CANCEL_REPLY_MS,
  mcpReadyMs: MCP_READY_MS,
  stopEofMs: 1_500,
  stopTermMs: 1_000,
  crashWindowMs: CRASH_WINDOW_MS,
  modelSelectMaxMs: MODEL_SELECT_MAX_MS,
  modelWaitShowMs: MODEL_WAIT_SHOW_MS,
};

export interface TurnResult {
  turnId: string;
  /** Unset when the turn failed before the prompt reached the agent. */
  outcome?: TurnOutcome;
  failure?: ChatFailure;
  /** The turn ended before `session/prompt` was written to the agent. */
  unsent?: true;
}

export interface AgentHostListener {
  onState?(state: AgentModeState): void;
  /** A live update line that passed the live filter, for the transcript reducer. */
  onUpdate?(sessionId: string, line: UpdateLine): void;
  onTitle?(sessionId: string, title: string): void;
  /** The agent switched the model during a turn; `ConversationInfo.modelId` follows. */
  onModel?(sessionId: string, modelId: string): void;
  onTurn?(sessionId: string, turn: TurnState | null): void;
  onTurnEnd?(sessionId: string, result: TurnResult): void;
  onInteraction?(sessionId: string, interaction: Interaction): void;
  onInteractionClosed?(sessionId: string, id: string, reason: 'answered' | 'cancelled' | 'expired'): void;
  /** These sessions are no longer loaded in a running process. */
  onCold?(sessionIds: string[]): void;
  /** A parsed `mlx_web` result of a Chat turn. */
  onWebResult?(sessionId: string, result: WebResult): void;
  /** Work: a provider key appeared in tool output. */
  onNotice?(sessionId: string, notice: ConversationNotice): void;
  /** Work: a session passed its guard after `session/new` or `session/load`. */
  onSessionReady?(sessionId: string): void;
  /**
   * Work: Auto or Bypass allows this call. The history line is written before the agent gets the answer; false (not
   * written) asks the mode again, and the request is decided under it.
   */
  onModeAllowed?(sessionId: string, mark: AllowedMark): Promise<boolean>;
  /** Work: the call whose history line was written gets `allow-once` now, in this tick. */
  onModeMarked?(sessionId: string, mark: AllowedMark): void;
  /** Work: the call whose history line was written gets no allow (Stop, the mode moved, Auto's late checks). */
  onModeWithdrawn?(sessionId: string, mark: AllowedMark): void;
}

export interface AgentHostOptions {
  mode: ChatMode;
  paths: ChatPaths;
  resolveLaunch: () => LaunchResolution;
  readEndpoint: EndpointReader;
  providers: ProviderSource;
  clientVersion: string;
  /** Chat: the system prompt (`chatPrompt(skill)`). */
  chatPrompt?: string;
  /** Chat: how the agent starts the web tool server. */
  webServer?: { command: string; script: string };
  extraCaCerts?: string;
  /** Work: the Control Panel's environment. */
  baseEnv?: NodeJS.ProcessEnv;
  /** Work: `GROK_HOME` when the runtime's agent home is not the default. */
  agentHome?: string | null;
  /** Work: process cwd (the user's home). */
  homeDir?: string;
  /** Work: the folder rule, applied again right before `session/new` and `session/load`. */
  checkCwd?: (path: string) => Promise<WorkFolderCheck>;
  /** Redaction needles for the cards: provider keys (current and retired) and the inference token. */
  needles?: () => string[];
  /** Work: the permission mode of a session, read in the tick that acts. Without it every request is a card. */
  permission?: { modeOf(sessionId: string): PermissionMode; watched(): boolean };
  /** Work: the temp roots Auto covers beside the Work folder; none by default. */
  tempRoots?: readonly AllowedRoot[];
  /** Work: whose files "yours" means in a temp root; default the process uid. */
  uid?: number;
  /** Work: the login name, for the private-folder names of a temp root; default the process user. */
  login?: string;
  /** Work: real paths of the folders the Work folder rule refuses (app and agent data). */
  refused?: readonly string[];
  username?: string;
  now?: () => number;
  spawn?: SpawnAgent;
  timings?: Partial<HostTimings>;
  listener?: AgentHostListener;
}

interface OpenInteraction {
  interaction: Interaction;
  /** Answers the agent's request with this JSON-RPC result. */
  settle: (result: unknown) => void;
  /** The result that withdraws the request. */
  withdrawn: unknown;
  /** A helper start that can work in another folder: once allowed, Auto asks for every call of the session. */
  helper: boolean;
  /** The call can go on after it reports its end ({@link staysBehind}). */
  stays: boolean;
}

type AllowedMark = Extract<PermissionMark, { kind: 'allowed' }>;

/** A request whose mode changed during a wait: it is decided again under the mode it has now. */
const MOVED: unique symbol = Symbol('moved');
const MODE_ROUNDS = 8;
const ALLOW_ONCE_ANSWER = { outcome: { outcome: 'selected', optionId: ALLOW_ONCE_OPTION_ID } };

const PERMISSION_CANCELLED = { outcome: { outcome: 'cancelled' } };
const PLAN_CANCELLED = { outcome: 'cancelled' };

interface TurnRecord {
  turnId: string;
  phase: TurnState['phase'];
  startedAt: number;
  wait: TurnState['wait'];
  /** The model select call in flight. */
  select: AbortController | null;
  recorded: TurnEndCause | null;
  /** Calls allowed in this turn since the user's last allow; Auto counts a call at its decision. */
  approvals: number;
  /** Goes up at each allow by the user: a count given back after it would be counted twice. */
  approvalEpoch: number;
  /** Chat only. */
  capTimer: ReturnType<typeof setTimeout> | null;
  graceTimer: ReturnType<typeof setTimeout> | null;
  /** Ends the turn before the prompt was sent (cancel during start). */
  abort: ((result: TurnResult) => void) | null;
}

interface HostSession {
  id: string;
  cwd: string;
  modelId: string;
  /** The guard passed in this process. */
  guarded: boolean;
  /** Loaded in the running process. */
  hot: boolean;
  /** `session/new` replied and its guard is still running: no prompt, rename or model change yet. */
  creating: boolean;
  /** The one `session/load` and guard in flight; later turns wait for it. */
  loading: Promise<GuardResult> | null;
  ledger: AddressLedger;
  /** The `mlx_web` keys this session's server was started with: a key failure it reports is about these keys. */
  web: ChatWebEnv;
  turn: TurnRecord | null;
  interactions: Map<string, OpenInteraction>;
  /** Work: providers whose key was already reported printed in this session. */
  printed: Set<ProviderId>;
  /** Work: the user allowed a helper that can work in another folder, in this process. */
  elsewhere: boolean;
  /** Work: the user picked "allow all edits" in this process: the agent no longer asks for edits. */
  editsGranted: boolean;
  /** Work: the user picked an "always" option: the agent may run calls with no request. */
  granted: boolean;
}

/**
 * Work: the calls of one folder that run now, by session and call id, and whether each may change the folder; `gen`
 * goes up when such a call is allowed and when it ends. Facts read before a change are stale after it.
 */
interface FolderCalls {
  gen: number;
  busy: Map<string, FolderCall>;
}

/**
 * `changes`: it may change files, links or names. `wide`: it may do so in any folder (a call the user or Bypass
 * allowed, or one the agent runs on an "always" grant); a call Auto allowed stays in its folder, so it reaches only
 * folders that hold it or that it holds. `stays`: a Bypass call that can go on after it reports its end; it leaves
 * only when the process exits.
 */
interface FolderCall {
  changes: boolean;
  wide: boolean;
  stays?: true;
  /** The keys of the temp root records the call also sits in. */
  reach?: string[];
}

/** The `gen` of the Work folder and of each temp root at a request's entry. */
interface Seen {
  work: number;
  temps: Map<string, number>;
}

interface Running {
  process: AgentProcess;
  fingerprint: EndpointFingerprint;
  agentVersion: string;
  models: Array<{ modelId: string; contextTokens: number | null }>;
  embeddedContext: boolean;
  launch: AgentLaunch;
  /** Work: the hook name of this start's guard; a shell request must name exactly this one. */
  guardHook: string | null;
}

export interface CreateInput {
  modelId: string;
  /** Work: the checked folder. Chat always uses its fixed workspace. */
  cwd?: string;
}

export interface LoadInput {
  modelId: string;
  /** Work: the folder from the session's summary, checked by the caller. */
  cwd?: string;
  /** The conversation's disk entries: they seed the Chat address ledger. */
  entries?: readonly TranscriptEntry[];
}

export interface CreatedSession {
  sessionId: string;
  modelId: string;
  cwd: string;
  agentName: string;
  toolCount: number | null;
}

type ExitWaiter = (exit: AgentExit) => void;

export class AgentHost {
  readonly mode: ChatMode;
  private readonly opts: AgentHostOptions;
  private readonly timings: HostTimings;
  private readonly now: () => number;
  private state: AgentModeState['state'] = 'stopped';
  private stale = false;
  private failure: ChatFailure | null = null;
  private running: Running | null = null;
  /** A process being stopped: it holds no new work, and the next start waits for its exit. */
  private retiring: Running | null = null;
  private starting: Promise<Running> | null = null;
  /** The process of the launch in flight, before its self-check passed. */
  private launchingProcess: AgentProcess | null = null;
  /** Raised by close and restart: a launch that started before is stopped when it finishes. */
  private generation = 0;
  private closed = false;
  private readonly sessions = new Map<string, HostSession>();
  /** Every `server_status ready` seen, keyed by session id, from before the session reply on. */
  private readonly readyServers = new Map<string, Set<string>>();
  private readonly readyWaiters = new Map<string, Array<() => void>>();
  private readonly exitWaiters = new Set<ExitWaiter>();
  /**
   * Work: by folder (its path with each part folded), the calls that run there. Every session and turn on the folder
   * shares it, and neither a turn's end nor Stop clears it: a call leaves only when it reports an end or the process
   * exits.
   */
  private readonly folders = new Map<string, FolderCalls>();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private crashes: number[] = [];
  private mismatches = 0;

  constructor(options: AgentHostOptions) {
    this.mode = options.mode;
    this.opts = options;
    this.timings = { ...DEFAULT_TIMINGS, ...options.timings };
    this.now = options.now ?? Date.now;
  }

  modeState(): AgentModeState {
    let turnsInFlight = 0;
    for (const s of this.sessions.values()) if (s.turn !== null) turnsInFlight++;
    return {
      state: this.state,
      stale: this.stale,
      failure: this.failure,
      agentVersion: this.running?.agentVersion ?? null,
      turnsInFlight,
    };
  }

  /** The models of the running process, or null when it is not running. */
  models(): Array<{ modelId: string; contextTokens: number | null }> | null {
    return this.running === null ? null : this.running.models.map((m) => ({ ...m }));
  }

  launchInfo(): AgentLaunch | null {
    return this.running?.launch ?? null;
  }

  isHot(sessionId: string): boolean {
    return this.sessions.get(sessionId)?.hot === true;
  }

  /** Resolves once a process being stopped has exited, so `isHot` answers for the process that will take the call. */
  async settled(): Promise<void> {
    if (this.retiring !== null) await this.retiring.process.exited;
  }

  isBusy(sessionId: string): boolean {
    return this.sessions.get(sessionId)?.turn != null;
  }

  turnState(sessionId: string): TurnState | null {
    const t = this.sessions.get(sessionId)?.turn;
    return t == null ? null : { turnId: t.turnId, phase: t.phase, startedAt: t.startedAt, wait: t.wait };
  }

  openInteractions(sessionId: string): Interaction[] {
    const s = this.sessions.get(sessionId);
    return s === undefined ? [] : [...s.interactions.values()].map((o) => o.interaction);
  }

  /** Work: "allow all edits" was picked in this session's process. */
  editsGranted(sessionId: string): boolean {
    return this.sessions.get(sessionId)?.editsGranted === true;
  }

  sessionModel(sessionId: string): string | null {
    return this.sessions.get(sessionId)?.modelId ?? null;
  }

  /** The Chat address ledger of a session the host knows. */
  ledger(sessionId: string): AddressLedger | null {
    return this.sessions.get(sessionId)?.ledger ?? null;
  }

  /** The host's sessions with their busy and waiting flags. */
  active(): Array<{ sessionId: string; busy: boolean; waiting: boolean }> {
    return [...this.sessions.values()]
      .filter((s) => s.hot || s.turn !== null)
      .map((s) => ({ sessionId: s.id, busy: s.turn !== null, waiting: s.interactions.size > 0 }));
  }

  /** A config change (endpoint, providers, models, binary). The host restarts as soon as no turn is in flight. */
  markStale(): void {
    if (this.running === null && this.starting === null) return;
    this.stale = true;
    this.emitState();
    this.restartIfIdle();
  }

  /** Leaves `failed` and `blocked`: stops the process; the next use starts a new one. */
  async restart(): Promise<void> {
    this.crashes = [];
    this.mismatches = 0;
    this.failure = null;
    await this.abandonLaunch();
    await this.stopProcess();
    this.state = 'stopped';
    this.emitState();
  }

  /**
   * Re-reads the endpoint while turns run (on runtime events). A missing file cancels every running turn as
   * `inference-lost`; a changed one marks the host stale.
   */
  async checkEndpoint(): Promise<void> {
    const running = this.running;
    if (running === null) return;
    const endpoint = await this.opts.readEndpoint();
    if (this.running !== running) return;
    if (endpoint === null) {
      for (const s of this.sessions.values()) if (s.turn !== null) this.cancelTurn(s, 'inference-lost');
      this.markStale();
      return;
    }
    if (!sameEndpoint(endpointFingerprint(endpoint), running.fingerprint)) this.markStale();
  }

  /** Starts the process if needed. Resolves when the self-check passed. */
  async start(): Promise<void> {
    await this.ensureRunning();
  }

  async createSession(input: CreateInput): Promise<CreatedSession> {
    await this.opts.providers.settled();
    const running = await this.ensureRunning();
    this.clearIdle();
    if (!running.models.some((m) => m.modelId === input.modelId)) throw new AgentHostError('model-unavailable');
    const cwd = await this.sessionCwd(input.cwd);
    const web = this.sessionWeb();
    const params = this.newParams(cwd, input.modelId, web);
    const reply = readSessionReply(await this.call(running, 'session/new', params, ACP_TIMEOUTS['session/new']));
    if (reply.sessionId === null) throw new AgentHostError('agent-protocol-error');
    const session = this.register(reply.sessionId, cwd, input.modelId, false);
    session.web = web;
    session.creating = true;
    let guard: GuardResult;
    try {
      guard = await this.guard(running, session, reply, input.modelId, true);
    } finally {
      session.creating = false;
    }
    if (!guard.ok) throw new AgentHostError(guard.reason);
    if (this.mode === 'work') this.opts.listener?.onSessionReady?.(session.id);
    this.scheduleIdle();
    return {
      sessionId: session.id,
      modelId: input.modelId,
      cwd,
      agentName: guard.agentName,
      toolCount: guard.toolCount,
    };
  }

  /**
   * Starts a turn. Synchronous checks throw (`conversation-busy`); loading a cold session, the guard and the model
   * call run in the background, and `done` resolves when the turn ended.
   */
  startTurn(sessionId: string, text: string, load: LoadInput): { turnId: string; done: Promise<TurnResult> } {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (existing?.turn != null || existing?.creating === true) throw new AgentHostError('conversation-busy');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    if (!session.guarded) this.seedLedger(session, load.entries);
    const turn: TurnRecord = {
      turnId: randomUUID(),
      phase: 'starting',
      startedAt: this.now(),
      wait: null,
      select: null,
      recorded: null,
      approvals: 0,
      approvalEpoch: 0,
      capTimer: null,
      graceTimer: null,
      abort: null,
    };
    session.turn = turn;
    this.clearIdle();
    this.emitTurn(session);
    this.emitState();
    const done = this.runTurn(session, turn, text, load).then((result) => {
      this.finishTurn(session, turn, result);
      return result;
    });
    return { turnId: turn.turnId, done };
  }

  /** The user's stop. Open permission requests are answered `cancelled` first. */
  cancel(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);
    if (session?.turn == null) return false;
    this.cancelTurn(session, 'cancelled');
    return true;
  }

  /** Work: the user's answer to a permission request; the option must be one the interaction offered. */
  answer(sessionId: string, interactionId: string, optionId: string): void {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined || open.interaction.kind !== 'permission') {
      throw new InteractionError('not-found');
    }
    if (!open.interaction.options.some((o) => o.optionId === optionId)) throw new InteractionError('not-offered');
    if (this.modeOf(sessionId) !== 'ask' && optionId !== REJECT_ONCE_OPTION_ID && optionId !== ALLOW_ONCE_OPTION_ID) {
      throw new InteractionError('not-offered');
    }
    if (optionId === REJECT_ONCE_OPTION_ID && session.turn !== null && session.turn.recorded === null) {
      session.turn.recorded = 'denied';
    }
    if (optionId !== REJECT_ONCE_OPTION_ID) {
      if (session.turn !== null) {
        session.turn.approvals = 0;
        session.turn.approvalEpoch += 1;
      }
      if (optionId === ALLOW_EDITS_SESSION_OPTION_ID) session.editsGranted = true;
      if (optionId === ALLOW_EDITS_SESSION_OPTION_ID || optionId === ALWAYS_ALLOW_OPTION_ID) session.granted = true;
      if (open.helper) session.elsewhere = true;
      this.callStarted(session, open.interaction.toolCallId, { changes: true, wide: true });
    }
    this.settleInteraction(session, interactionId, { outcome: { outcome: 'selected', optionId } }, 'answered');
  }

  /**
   * Work, the move to Bypass: an open permission card that offers `allow-once` is answered with it, as Bypass answers
   * a new request. The caller wrote its history line first. False when the card is gone or offers no `allow-once`.
   */
  allowOpen(sessionId: string, interactionId: string): boolean {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined || open.interaction.kind !== 'permission') return false;
    if (!open.interaction.options.some((o) => o.optionId === ALLOW_ONCE_OPTION_ID)) return false;
    if (open.helper) session.elsewhere = true;
    this.callStarted(session, open.interaction.toolCallId, {
      changes: true,
      wide: true,
      ...(open.stays ? { stays: true as const } : {}),
    });
    this.settleInteraction(session, interactionId, ALLOW_ONCE_ANSWER, 'answered');
    return true;
  }

  /** Work, the move to Bypass: an open card that offers neither `allow-once` nor `reject-once` is withdrawn. */
  withdrawOpen(sessionId: string, interactionId: string): boolean {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined || open.interaction.kind !== 'permission') return false;
    if (
      open.interaction.options.some((o) => o.optionId === ALLOW_ONCE_OPTION_ID || o.optionId === REJECT_ONCE_OPTION_ID)
    ) {
      return false;
    }
    if (session.turn !== null && session.turn.recorded === null) session.turn.recorded = 'denied';
    this.settleInteraction(session, interactionId, open.withdrawn, 'cancelled');
    return true;
  }

  /**
   * Work: the user's answer to a plan. A rejection carries the user's feedback to the agent, except for a plan with no
   * text: with no plan file the agent drops the feedback, so none is sent.
   */
  answerPlan(sessionId: string, interactionId: string, decision: 'approved' | 'rejected', feedback?: string): void {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined || open.interaction.kind !== 'plan') {
      throw new InteractionError('not-found');
    }
    const noText = open.interaction.plan.trim() === '';
    this.settleInteraction(session, interactionId, planAnswer(decision, noText ? undefined : feedback), 'answered');
  }

  /**
   * Nobody can answer (the window is gone): every open permission request is denied, which ends its turn as
   * `denied`, and every open plan is withdrawn.
   */
  denyOpen(): void {
    for (const session of this.sessions.values()) {
      for (const [id, open] of session.interactions) {
        if (open.interaction.kind === 'permission') {
          const deny = open.interaction.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID);
          if (session.turn !== null && session.turn.recorded === null) session.turn.recorded = 'denied';
          const result = deny ? { outcome: { outcome: 'selected', optionId: REJECT_ONCE_OPTION_ID } } : open.withdrawn;
          this.settleInteraction(session, id, result, 'expired');
        } else {
          this.settleInteraction(session, id, open.withdrawn, 'expired');
        }
      }
    }
  }

  /** True while any session waits on the user. */
  hasOpenInteractions(): boolean {
    for (const s of this.sessions.values()) if (s.interactions.size > 0) return true;
    return false;
  }

  private settleInteraction(
    session: HostSession,
    interactionId: string,
    result: unknown,
    reason: 'answered' | 'cancelled' | 'expired',
  ): void {
    const open = session.interactions.get(interactionId);
    if (open === undefined) return;
    session.interactions.delete(interactionId);
    open.settle(result);
    this.opts.listener?.onInteractionClosed?.(session.id, interactionId, reason);
    this.emitState();
  }

  async rename(sessionId: string, title: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (session === undefined || !session.guarded || running === null) throw new AgentHostError('agent-crashed');
    if (session.turn !== null || session.loading !== null) throw new AgentHostError('conversation-busy');
    const result = await this.call(
      running,
      '_x.ai/session/rename',
      { sessionId, title, cwd: session.cwd },
      ACP_TIMEOUTS['_x.ai/session/rename'],
    );
    if (!isSuccess(result)) throw new AgentHostError('agent-protocol-error');
  }

  /**
   * `_x.ai/session/delete`; starts the process when it is stopped (a Work delete always goes through the agent). A
   * Chat session not loaded in this process is in the fixed Chat workspace.
   */
  async deleteSession(sessionId: string, cwd: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session?.turn != null) throw new AgentHostError('conversation-busy');
    const running = await this.ensureRunning();
    const sessionCwd = session?.cwd || (this.mode === 'chat' ? await this.sessionCwd(undefined) : cwd);
    const result = await this.call(
      running,
      '_x.ai/session/delete',
      { sessionId, cwd: sessionCwd },
      ACP_TIMEOUTS['_x.ai/session/delete'],
    );
    if (!isSuccess(result)) throw new AgentHostError('agent-protocol-error');
    this.drop(sessionId);
    this.scheduleIdle();
  }

  /** `session/set_model` between turns, confirmed by `_x.ai/session/info`. */
  async setModel(sessionId: string, modelId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (session === undefined || !session.guarded || running === null) throw new AgentHostError('agent-crashed');
    if (session.turn !== null || session.loading !== null) throw new AgentHostError('conversation-busy');
    if (!running.models.some((m) => m.modelId === modelId)) throw new AgentHostError('model-unavailable');
    try {
      await this.call(running, 'session/set_model', { sessionId, modelId }, ACP_TIMEOUTS['session/set_model']);
    } catch (error) {
      if (error instanceof AcpRpcError) throw new AgentHostError('model-unavailable');
      throw error;
    }
    const info = readSessionInfo(
      await this.call(running, '_x.ai/session/info', { sessionId }, ACP_TIMEOUTS['_x.ai/session/info']),
    );
    if (info?.model !== modelId) throw new AgentHostError('model-unavailable');
    session.modelId = modelId;
  }

  /** Control Panel exit: cancel every turn, then the stop ladder. The host takes no more work. */
  async close(): Promise<void> {
    this.closed = true;
    this.clearIdle();
    for (const s of this.sessions.values()) if (s.turn !== null) this.cancelTurn(s, 'app-closed');
    await Promise.all([this.abandonLaunch(), this.stopProcess()]);
    this.state = 'stopped';
  }

  // ── process lifecycle ──────────────────────────────────────────────────────────────────────────────────────────

  /** `caller`: the turn asking, which does not count as a turn in flight on a stale process. */
  private async ensureRunning(caller?: TurnRecord): Promise<Running> {
    for (;;) {
      if (this.closed) throw new AgentHostError('agent-crashed');
      if (this.state === 'failed') throw new AgentHostError('agent-failed');
      if (this.state === 'blocked' && this.failure?.code === 'agent-incompatible') {
        throw new AgentHostError('agent-incompatible');
      }
      if (this.running !== null && !this.stale) return this.running;
      if (this.running !== null && this.stale && !this.hasTurns(caller)) await this.stopProcess();
      if (this.running !== null) return this.running;
      if (this.retiring !== null) {
        await this.retiring.process.exited;
        continue;
      }
      if (this.starting === null) {
        this.starting = this.launch().finally(() => {
          this.starting = null;
        });
      }
      return this.starting;
    }
  }

  private async launch(): Promise<Running> {
    const generation = this.generation;
    this.state = 'starting';
    this.failure = null;
    this.stale = false;
    this.emitState();
    try {
      let running: Running | null = null;
      for (let attempt = 0; running === null; attempt++) {
        if (generation !== this.generation) throw new AgentHostError('agent-crashed');
        running = await this.spawnOnce(attempt);
      }
      if (generation !== this.generation) {
        await running.process.stop(this.timings.stopEofMs, this.timings.stopTermMs);
        throw new AgentHostError('agent-crashed');
      }
      this.running = running;
      this.state = 'ready';
      this.emitState();
      return running;
    } catch (error) {
      if (generation !== this.generation)
        throw error instanceof AgentHostError ? error : new AgentHostError('agent-crashed');
      const code = error instanceof AgentHostError ? error.code : 'agent-crashed';
      const blocking = code === 'agent-incompatible';
      this.state = blocking ? 'blocked' : code === 'agent-protocol-error' ? 'crashed' : 'stopped';
      this.failure = chatFailure(code);
      this.emitState();
      throw error instanceof AgentHostError ? error : new AgentHostError(code);
    }
  }

  /** One spawn and self-check; null asks for one more attempt (ENOENT re-probe, or a changed endpoint). */
  private async spawnOnce(attempt: number): Promise<Running | null> {
    const launch = this.opts.resolveLaunch();
    if ('failure' in launch) throw new AgentHostError(launch.failure);
    const endpoint = await this.opts.readEndpoint();
    if (endpoint === null) throw new AgentHostError('inference-stopped');
    const guardName = this.mode === 'work' ? workGuardName() : null;
    if (guardName === null) await this.prepareChatHome();
    else await writeWorkGuard(this.opts.paths, workGuardManifest(guardName));
    const env =
      this.mode === 'chat'
        ? chatAgentEnv({
            paths: this.opts.paths,
            endpoint,
            username: this.opts.username ?? userInfo().username,
          })
        : workAgentEnv({
            base: this.opts.baseEnv ?? process.env,
            endpoint,
            agentHome: this.opts.agentHome ?? null,
            providerKeys: this.opts.providers.workKeys(),
          });
    if (launch.env !== undefined) Object.assign(env, launch.env);
    const cwd = this.mode === 'chat' ? this.opts.paths.chatWorkspace : (this.opts.homeDir ?? homedir());
    const fingerprint = endpointFingerprint(endpoint);
    let protocolError = false;
    let initialized = false;
    const proc = new AgentProcess(
      { command: launch.command, args: launch.args, env, cwd },
      {
        onNotification: (method, params) => this.onNotification(proc, method, params),
        onRequest: (method, params) => this.onRequest(proc, method, params),
        onProtocolError: (_reason: ProtocolErrorReason) => {
          protocolError = true;
          if (initialized) this.failure = chatFailure('agent-protocol-error');
        },
        onExit: (exit) => this.onExit(proc, exit),
      },
      this.opts.spawn ?? spawnAgent,
    );
    this.launchingProcess = proc;
    try {
      let facts;
      try {
        facts = readInitialize(
          await proc.client.request('initialize', initializeParams(this.opts.clientVersion), ACP_TIMEOUTS.initialize),
        );
      } catch (error) {
        const exit = await proc.kill(this.timings.stopTermMs);
        if (exit.errorCode === 'ENOENT' && attempt === 0) return null;
        if (protocolError || error instanceof AcpRpcError) throw new AgentHostError('agent-protocol-error');
        throw new AgentHostError(exit.errorCode === 'ENOENT' ? 'agent-missing' : 'agent-crashed');
      }
      const check = selfCheck(facts);
      if (!check.ok) {
        await proc.stop(this.timings.stopEofMs, this.timings.stopTermMs);
        if (check.row !== 5) throw new AgentHostError('agent-incompatible');
        const now = await this.opts.readEndpoint();
        if (now !== null && !sameEndpoint(endpointFingerprint(now), fingerprint) && attempt === 0) return null;
        throw new AgentHostError(endpoint.models.length === 0 ? 'no-models' : 'inference-failed');
      }
      initialized = true;
      return {
        process: proc,
        fingerprint,
        agentVersion: check.agentVersion,
        models: check.models,
        embeddedContext: check.embeddedContext,
        launch,
        guardHook: guardName === null ? null : workGuardHookName(guardName),
      };
    } finally {
      if (this.launchingProcess === proc) this.launchingProcess = null;
    }
  }

  private async prepareChatHome(): Promise<void> {
    const paths = this.opts.paths;
    await ensureChatDirs(paths);
    await writeFileAtomic(paths.chatConfig, CHAT_CONFIG_TOML);
    await writeFileAtomic(paths.chatRequirements, CHAT_REQUIREMENTS_TOML);
    await removeChatSearchIndex(paths);
    await removeChatPromptHistory(paths);
  }

  private onExit(proc: AgentProcess, exit: AgentExit): void {
    const protocolError = exit.protocolError;
    for (const waiter of this.exitWaiters) waiter(exit);
    if (this.running?.process === proc) this.running = null;
    else if (this.retiring?.process === proc) this.retiring = null;
    else return;
    for (const folder of this.folders.values()) {
      folder.busy.clear();
      folder.gen += 1;
    }
    const cold: string[] = [];
    for (const s of this.sessions.values()) {
      if (s.hot) cold.push(s.id);
      s.hot = false;
      s.guarded = false;
    }
    this.readyServers.clear();
    this.clearIdle();
    if (exit.asked && !protocolError) {
      if (this.state !== 'blocked' && this.state !== 'failed') this.state = 'stopped';
    } else {
      const now = this.now();
      this.crashes = this.crashes.filter((t) => now - t < this.timings.crashWindowMs);
      this.crashes.push(now);
      if (this.crashes.length >= CRASHES_TO_FAIL) {
        this.state = 'failed';
        this.failure = chatFailure('agent-failed');
      } else {
        this.state = 'crashed';
        this.failure = chatFailure(protocolError ? 'agent-protocol-error' : 'agent-crashed');
      }
    }
    if (cold.length > 0) this.opts.listener?.onCold?.(cold);
    this.emitState();
  }

  /** The stop ladder. The process leaves `running` at once, so no new call reaches it while it exits. */
  private async stopProcess(): Promise<void> {
    const running = this.running;
    if (running === null) {
      if (this.retiring !== null) await this.retiring.process.exited;
      return;
    }
    this.running = null;
    this.retiring = running;
    for (const s of this.sessions.values()) this.closeInteractions(s, 'cancelled');
    await running.process.stop(this.timings.stopEofMs, this.timings.stopTermMs);
  }

  /** Stops the process of a launch in flight and waits until that launch settled. */
  private async abandonLaunch(): Promise<void> {
    this.generation += 1;
    const starting = this.starting;
    const proc = this.launchingProcess;
    if (proc !== null) await proc.stop(this.timings.stopEofMs, this.timings.stopTermMs);
    if (starting !== null) await starting.catch(() => {});
  }

  private hasTurns(except?: TurnRecord): boolean {
    for (const s of this.sessions.values()) if (s.turn !== null && s.turn !== except) return true;
    return false;
  }

  private restartIfIdle(): void {
    if (!this.stale || this.hasTurns() || this.running === null) return;
    void this.stopProcess().then(() => {
      if (this.running === null && this.starting === null && !this.closed) {
        this.stale = false;
        this.emitState();
      }
    });
  }

  private clearIdle(): void {
    if (this.idleTimer !== null) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  private scheduleIdle(): void {
    this.clearIdle();
    if (this.running === null || this.hasTurns() || this.closed) return;
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      if (!this.hasTurns()) void this.stopProcess();
    }, this.timings.idleStopMs);
    this.idleTimer.unref?.();
  }

  // ── sessions ───────────────────────────────────────────────────────────────────────────────────────────────────

  private register(id: string, cwd: string, modelId: string, hot = true): HostSession {
    const session: HostSession = {
      id,
      cwd,
      modelId,
      guarded: false,
      hot,
      creating: false,
      loading: null,
      ledger: new AddressLedger(),
      web: NO_WEB,
      turn: null,
      interactions: new Map(),
      printed: new Set(),
      elsewhere: false,
      editsGranted: false,
      granted: false,
    };
    this.sessions.set(id, session);
    return session;
  }

  private drop(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (session !== undefined) this.closeInteractions(session, 'cancelled');
    this.sessions.delete(sessionId);
    this.readyServers.delete(sessionId);
    this.readyWaiters.delete(sessionId);
  }

  private seedLedger(session: HostSession, entries: readonly TranscriptEntry[] | undefined): void {
    if (this.mode !== 'chat' || entries === undefined) return;
    for (const entry of entries) {
      if (entry.kind === 'user') session.ledger.addUserText(entry.text);
      else if (entry.kind === 'tool' && entry.output !== undefined && isWebToolName(entry.name)) {
        const result = parseWebResultText(entry.output);
        if (result?.kind === 'search') session.ledger.addSearchResults(result.results.map((r) => r.url));
      }
    }
  }

  private async sessionCwd(requested: string | undefined): Promise<string> {
    if (this.mode === 'chat') return realpath(this.opts.paths.chatWorkspace);
    if (requested === undefined || requested === '') throw new AgentHostError('cwd-invalid');
    const check = this.opts.checkCwd;
    if (check !== undefined) {
      const folder = await check(requested).catch(() => null);
      if (folder === null || !folder.ok) throw new AgentHostError('cwd-invalid');
      return folder.path;
    }
    try {
      return await realpath(requested);
    } catch {
      throw new AgentHostError('cwd-invalid');
    }
  }

  private sessionWeb(): ChatWebEnv {
    return this.mode === 'chat' ? this.opts.providers.chatWeb() : NO_WEB;
  }

  private newParams(cwd: string, modelId: string, web: ChatWebEnv): Record<string, unknown> {
    if (this.mode === 'work') {
      return workSessionParams({
        cwd,
        modelId,
        guardDir: this.opts.paths.workGuard,
        pluginDirs: this.opts.providers.workPluginDirs(),
      });
    }
    if (this.opts.chatPrompt === undefined || this.opts.webServer === undefined) {
      throw new AgentHostError('skill-bundle-corrupt');
    }
    return chatSessionParams({
      cwd,
      modelId,
      webServer: this.opts.webServer,
      web,
      extraCaCerts: this.opts.extraCaCerts,
      prompt: this.opts.chatPrompt,
      pluginDir: pluginDir(this.opts.paths, CHAT_SKILL_PLUGIN),
    });
  }

  /**
   * Rows 1–7 of the guard. A failure drops the session from the table. Only a session `session/new` just made is also
   * removed through the agent (best effort): a loaded one is the user's saved conversation, and a reopen can fail for
   * reasons that say nothing about it (a slow `mlx_web` start, a changed binary).
   */
  private async guard(
    running: Running,
    session: HostSession,
    reply: ReturnType<typeof readSessionReply>,
    modelId: string,
    fresh: boolean,
  ): Promise<GuardResult> {
    let mcpReady = false;
    if (this.mode === 'chat') mcpReady = await this.waitReady(session.id);
    let info: SessionInfoFacts | null = null;
    try {
      info = readSessionInfo(
        await this.call(running, '_x.ai/session/info', { sessionId: session.id }, ACP_TIMEOUTS['_x.ai/session/info']),
      );
    } catch (error) {
      if (!(error instanceof AcpRpcError)) {
        this.drop(session.id);
        throw error;
      }
    }
    const result = checkSession({
      mode: this.mode,
      reply,
      info,
      mcpReady,
      requestedModelId: modelId,
      cwd: session.cwd,
    });
    if (result.ok) {
      this.mismatches = 0;
      session.guarded = true;
      session.hot = true;
      return result;
    }
    this.drop(session.id);
    if (fresh && this.running === running) {
      running.process.client
        .request(
          '_x.ai/session/delete',
          { sessionId: session.id, cwd: session.cwd },
          ACP_TIMEOUTS['_x.ai/session/delete'],
        )
        .catch(() => {});
    }
    if (result.reason === 'profile-mismatch') {
      this.mismatches += 1;
      if (this.mismatches >= MISMATCHES_TO_BLOCK) {
        this.failure = chatFailure('agent-incompatible');
        this.state = 'blocked';
        this.emitState();
        void this.stopProcess();
      }
    }
    return result;
  }

  private waitReady(sessionId: string): Promise<boolean> {
    if (this.readyServers.get(sessionId)?.has(WEB_SERVER_NAME)) return Promise.resolve(true);
    return new Promise((resolve) => {
      let finished = false;
      const finish = (value: boolean): void => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        this.exitWaiters.delete(onExit);
        resolve(value);
      };
      const timer = setTimeout(() => finish(false), this.timings.mcpReadyMs);
      const onExit = (): void => finish(false);
      this.exitWaiters.add(onExit);
      const waiters = this.readyWaiters.get(sessionId) ?? [];
      waiters.push(() => finish(true));
      this.readyWaiters.set(sessionId, waiters);
    });
  }

  private async load(running: Running, session: HostSession, load: LoadInput): Promise<GuardResult> {
    const cwd = await this.sessionCwd(this.mode === 'work' ? load.cwd : undefined);
    session.cwd = cwd;
    session.modelId = load.modelId;
    if (!running.models.some((m) => m.modelId === load.modelId)) throw new AgentHostError('model-unavailable');
    const web = this.sessionWeb();
    session.web = web;
    const params = loadSessionParams(this.newParams(cwd, load.modelId, web), session.id);
    const reply = readSessionReply(await this.call(running, 'session/load', params, ACP_TIMEOUTS['session/load']));
    const guard = await this.guard(running, session, { ...reply, sessionId: session.id }, load.modelId, false);
    if (guard.ok && this.mode === 'work') {
      session.elsewhere = false;
      session.editsGranted = false;
      this.opts.listener?.onSessionReady?.(session.id);
    }
    return guard;
  }

  private loadOnce(running: Running, session: HostSession, load: LoadInput): Promise<GuardResult> {
    if (session.loading === null) {
      const loading = this.load(running, session, load).finally(() => {
        if (session.loading === loading) session.loading = null;
      });
      session.loading = loading;
    }
    return session.loading;
  }

  // ── turns ──────────────────────────────────────────────────────────────────────────────────────────────────────

  private async runTurn(session: HostSession, turn: TurnRecord, text: string, load: LoadInput): Promise<TurnResult> {
    const aborted = new Promise<TurnResult>((resolve) => {
      turn.abort = resolve;
    });
    const unsent = (end: { outcome?: TurnOutcome; failure?: ChatFailureCode }): TurnResult => ({
      turnId: turn.turnId,
      ...(end.outcome === undefined ? {} : { outcome: end.outcome }),
      ...(end.failure === undefined ? {} : { failure: chatFailure(end.failure) }),
      unsent: true,
    });
    const prepare = async (): Promise<Running | TurnResult> => {
      try {
        await this.opts.providers.settled();
        for (let attempt = 0; ; attempt++) {
          const running = await this.ensureRunning(turn);
          if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
          if (!session.guarded || !this.sessions.has(session.id)) {
            if (!this.sessions.has(session.id)) this.sessions.set(session.id, session);
            const guard = await this.loadOnce(running, session, load);
            if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
            if (!guard.ok) return unsent({ outcome: 'refused', failure: guard.reason });
          }
          const endpoint = await this.opts.readEndpoint();
          if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
          if (endpoint === null) return unsent({ failure: 'inference-stopped' });
          if (sameEndpoint(endpointFingerprint(endpoint), running.fingerprint)) {
            const picked = await this.selectFor(running, session, turn, endpoint);
            if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
            if (!running.process.running) return unsent({ failure: 'agent-crashed' });
            if (picked === 'selected') return running;
            if (picked === 'unknown') return unsent({ failure: 'model-unavailable' });
            if (picked === 'socket') {
              const now = await this.opts.readEndpoint();
              return unsent({ failure: now === null ? 'inference-stopped' : 'inference-failed' });
            }
            if (picked !== 'auth') return unsent({ failure: 'model-load-failed' });
          }
          // The old process holds the old address or token: the prompt never goes to it.
          this.markStale();
          if (attempt > 0 || this.hasTurns(turn)) return unsent({ failure: 'inference-failed' });
        }
      } catch (error) {
        return unsent({ failure: error instanceof AgentHostError ? error.code : 'agent-crashed' });
      }
    };
    const ready = await Promise.race([prepare(), aborted]);
    if (!('process' in ready)) return ready;
    if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
    const running = ready;
    if (this.mode === 'chat') session.ledger.addUserText(text);
    turn.abort = null;
    turn.phase = 'running';
    this.emitTurn(session);
    if (this.mode === 'chat') {
      turn.capTimer = setTimeout(() => this.cancelTurn(session, 'turn-timeout'), this.timings.chatTurnMaxMs);
    }
    let exited = false;
    const onExit: ExitWaiter = () => {
      exited = true;
    };
    this.exitWaiters.add(onExit);
    try {
      const result = await running.process.client.request('session/prompt', promptParams(session.id, text), null);
      return {
        turnId: turn.turnId,
        outcome: turnOutcome({ mode: this.mode, result: readPromptResult(result), recorded: turn.recorded }),
      };
    } catch (error) {
      if (error instanceof AcpRpcError) {
        return {
          turnId: turn.turnId,
          outcome: turnOutcome({
            mode: this.mode,
            error: { code: error.code, errorKind: error.errorKind },
            recorded: turn.recorded,
          }),
        };
      }
      if (exited || !running.process.running || running.process.brokeProtocol) {
        return { turnId: turn.turnId, outcome: 'agent-crashed', failure: chatFailure('agent-crashed') };
      }
      return { turnId: turn.turnId, outcome: 'model-error' };
    } finally {
      this.exitWaiters.delete(onExit);
    }
  }

  private finishTurn(session: HostSession, turn: TurnRecord, result: TurnResult): void {
    if (turn.capTimer !== null) clearTimeout(turn.capTimer);
    if (turn.graceTimer !== null) clearTimeout(turn.graceTimer);
    if (session.turn === turn) session.turn = null;
    this.closeInteractions(session, 'cancelled');
    this.opts.listener?.onTurn?.(session.id, null);
    this.opts.listener?.onTurnEnd?.(session.id, result);
    this.emitState();
    if (this.stale) this.restartIfIdle();
    else this.scheduleIdle();
  }

  private cancelTurn(session: HostSession, cause: TurnEndCause): void {
    const turn = session.turn;
    if (turn === null) return;
    if (turn.recorded === null) turn.recorded = cause;
    this.closeInteractions(session, 'cancelled');
    if (turn.abort !== null) {
      turn.select?.abort();
      turn.abort({ turnId: turn.turnId, outcome: causeOutcome(turn.recorded), unsent: true });
      turn.abort = null;
      return;
    }
    if (turn.phase === 'cancelling') return;
    turn.phase = 'cancelling';
    this.emitTurn(session);
    const running = this.running;
    if (running === null) return;
    try {
      running.process.client.notify('session/cancel', { sessionId: session.id });
    } catch {
      return;
    }
    turn.graceTimer = setTimeout(() => {
      if (session.turn === turn && this.running === running) {
        void running.process.kill(this.timings.stopTermMs, true);
      }
    }, this.timings.cancelReplyMs);
  }

  /**
   * Asks the engine to put the session's model in memory. The agent process exiting during the call aborts it: the
   * prompt can no longer go to that process.
   */
  private async selectFor(
    running: Running,
    session: HostSession,
    turn: TurnRecord,
    endpoint: SelectTarget,
  ): Promise<SelectAnswer> {
    const select = new AbortController();
    turn.select = select;
    const live = (): boolean => session.turn === turn && turn.recorded === null && turn.select === select;
    const show = setTimeout(() => {
      if (!live()) return;
      turn.wait = 'model';
      this.emitTurn(session);
    }, this.timings.modelWaitShowMs);
    const onExit: ExitWaiter = () => {
      if (!running.process.running) select.abort();
    };
    this.exitWaiters.add(onExit);
    try {
      if (!running.process.running) return 'aborted';
      return await selectModel(endpoint, session.modelId, {
        maxMs: this.timings.modelSelectMaxMs,
        signal: select.signal,
      });
    } finally {
      this.exitWaiters.delete(onExit);
      clearTimeout(show);
      const shown = turn.wait !== null;
      if (live() && shown) {
        turn.wait = null;
        this.emitTurn(session);
      }
      turn.wait = null;
      turn.select = null;
    }
  }

  private closeInteractions(session: HostSession, reason: 'cancelled' | 'expired'): void {
    for (const [id, open] of session.interactions) {
      session.interactions.delete(id);
      open.settle(open.withdrawn);
      this.opts.listener?.onInteractionClosed?.(session.id, id, reason);
    }
  }

  // ── agent → client ─────────────────────────────────────────────────────────────────────────────────────────────

  private onNotification(proc: AgentProcess, method: string, params: unknown): void {
    if (this.running !== null && this.running.process !== proc) return;
    if (method === '_x.ai/mcp/server_status') {
      const status = readServerStatus(params);
      if (status === null) return;
      if (status.status !== 'ready') {
        this.webServerLost(status.sessionId, status.name);
        return;
      }
      let set = this.readyServers.get(status.sessionId);
      if (set === undefined) {
        set = new Set();
        this.readyServers.set(status.sessionId, set);
      }
      set.add(status.name);
      if (status.name === WEB_SERVER_NAME) {
        const waiters = this.readyWaiters.get(status.sessionId) ?? [];
        this.readyWaiters.delete(status.sessionId);
        for (const w of waiters) w();
      }
      return;
    }
    const message = readSessionUpdate(method, params);
    if (message === null) return;
    const session = this.sessions.get(message.sessionId);
    if (session === undefined) return;
    if (this.mode === 'work') this.trackCall(session, message.update);
    const title = readTitle(message.update);
    if (title !== null) {
      this.opts.listener?.onTitle?.(session.id, title);
      return;
    }
    const line: UpdateLine = { xai: message.xai, update: message.update, meta: message.meta };
    const turnRunning = session.turn !== null && session.turn.phase !== 'starting';
    if (liveFilter(line, { turnRunning, modelId: session.modelId }) === 'drop') return;
    if (line.xai && line.update.sessionUpdate === 'model_changed' && typeof line.update.model_id === 'string') {
      session.modelId = line.update.model_id;
      this.opts.listener?.onModel?.(session.id, session.modelId);
    }
    this.opts.listener?.onUpdate?.(session.id, line);
    if (this.mode === 'work') this.checkPrinted(session, line.update);
    if (this.mode === 'chat') {
      const result = readWebToolResult(line.update);
      if (result !== null) {
        applyWebResult(result, {
          ledger: session.ledger,
          onKeyCheck: (provider, check) =>
            this.opts.providers.markKeyCheck(provider, check, keyOf(session.web, provider)),
        });
        this.opts.listener?.onWebResult?.(session.id, result);
      }
    }
  }

  /**
   * Work: a provider key (current or retired) in a tool call or its output. The agent has already stored it; the
   * page is told once per provider, and the conversation's redaction keeps it out of what the page is sent.
   */
  private checkPrinted(session: HostSession, update: Record<string, unknown>): void {
    const kind = update.sessionUpdate;
    if (kind !== 'tool_call' && kind !== 'tool_call_update') return;
    for (const { provider, key } of this.opts.providers.keyNeedles()) {
      if (key === '' || session.printed.has(provider) || !holdsText(update, key, 0)) continue;
      session.printed.add(provider);
      this.opts.listener?.onNotice?.(session.id, { code: 'key-printed', provider });
    }
  }

  /**
   * `mlx_web` of a Chat session stopped after it was ready. The agent never restarts it inside the process, so every
   * later web call of that session would fail: the host goes stale, and the new process gets `mlx_web` back at load.
   */
  private webServerLost(sessionId: string, name: string): void {
    if (this.mode !== 'chat' || name !== WEB_SERVER_NAME) return;
    const ready = this.readyServers.get(sessionId);
    if (ready?.has(name) !== true) return;
    ready.delete(name);
    this.markStale();
  }

  private async onRequest(proc: AgentProcess, method: string, params: unknown): Promise<{ result: unknown } | null> {
    if (this.running?.process !== proc) return { result: { outcome: { outcome: 'cancelled' } } };
    if (method === 'session/request_permission') {
      const req = readPermissionRequest(params);
      if (req === null) {
        this.cancelUnknownTurns();
        return { result: { outcome: { outcome: 'cancelled' } } };
      }
      if (this.mode === 'chat') return { result: { outcome: this.chatPermission(req) } };
      return { result: await this.workPermission(proc, req) };
    }
    if (method === '_x.ai/exit_plan_mode') {
      return { result: this.mode === 'work' ? await this.workPlan(proc, params) : PLAN_CANCELLED };
    }
    if (method === '_x.ai/ask_user_question') return { result: { outcome: 'cancelled' } };
    return null;
  }

  /** A permission request the reader could not parse: fail closed on every running turn. */
  private cancelUnknownTurns(): void {
    for (const s of this.sessions.values()) if (s.turn !== null) this.cancelTurnLater(s, 'tool-refused');
  }

  private cancelTurnLater(session: HostSession, cause: TurnEndCause): void {
    if (session.turn !== null && session.turn.recorded === null) session.turn.recorded = cause;
    setImmediate(() => this.cancelTurn(session, cause));
  }

  private chatPermission(req: PermissionRequest): PermissionAnswer {
    const session = this.sessions.get(req.sessionId);
    const decision = decideChatPermission(req, {
      sessionGuarded: session?.guarded === true && session.turn !== null,
      approvals: session?.turn?.approvals ?? 0,
      ledger: session?.ledger ?? new AddressLedger(),
    });
    if (decision.kind === 'approve') {
      if (session?.turn != null) session.turn.approvals += 1;
      return decision.answer;
    }
    if (session?.turn != null) {
      if (session.turn.recorded === null) session.turn.recorded = decision.cause;
      if (decision.cancelTurn) this.cancelTurnLater(session, decision.cause);
    } else if (decision.cancelTurn && this.running !== null) {
      const running = this.running;
      setImmediate(() => {
        try {
          running.process.client.notify('session/cancel', { sessionId: req.sessionId });
        } catch {
          // The process is gone.
        }
      });
    }
    return decision.answer;
  }

  /**
   * Work: the request is decided by the conversation's permission mode ({@link openWorkCard}). A request that asks
   * after Stop, after its turn ended, or from a process that is gone is answered `cancelled`.
   */
  private workPermission(proc: AgentProcess, req: PermissionRequest): Promise<unknown> {
    const session = this.sessions.get(req.sessionId);
    const turn = session?.turn ?? null;
    if (session === undefined || turn === null || !this.asks(proc, session, turn)) {
      return Promise.resolve(PERMISSION_CANCELLED);
    }
    const place = { cwd: session.cwd, home: this.opts.homeDir ?? homedir() };
    const seen: Seen = { work: this.folderCalls(session.cwd).gen, temps: new Map() };
    for (const top of tempRoots(this.gatePlace(place)).map((r) => r.top)) {
      seen.temps.set(top, this.folderCalls(top).gen);
    }
    const path = requestTarget(req).path;
    if (path === null) return this.openWorkCard(proc, session, turn, req, place, null, seen);
    return editLanding(agentPath(path, place)).then(
      (landing) => this.openWorkCard(proc, session, turn, req, place, landing, seen),
      () => this.openWorkCard(proc, session, turn, req, place, null, seen),
    );
  }

  /**
   * A card opens only for the turn and process that asked, while that turn still runs: a request that resolves after
   * Stop, after the turn ended, or from a process that is gone is answered `cancelled`.
   */
  private asks(proc: AgentProcess, session: HostSession, turn: TurnRecord): boolean {
    return (
      session.guarded &&
      this.sessions.get(session.id) === session &&
      session.turn === turn &&
      turn.phase !== 'cancelling' &&
      turn.recorded === null &&
      this.running?.process === proc
    );
  }

  /** The permission mode of a session now; Ask when the host has no mode source. */
  private modeOf(sessionId: string): PermissionMode {
    const mode = this.opts.permission?.modeOf(sessionId);
    return mode === 'auto' || mode === 'bypass' ? mode : 'ask';
  }

  /**
   * The round loop. Each round reads the mode in the tick that acts on it: Ask opens the card, Auto runs the gate,
   * Bypass allows. A round whose mode moved during a wait (disk facts, history line) ends `moved`, and the next round
   * decides under the new mode, so a request never opens a card or gets an allow under a mode that is gone.
   */
  private async openWorkCard(
    proc: AgentProcess,
    session: HostSession,
    turn: TurnRecord,
    req: PermissionRequest,
    place: WorkPlace,
    landing: string | null,
    seen: Seen,
  ): Promise<unknown> {
    for (let round = 0; round < MODE_ROUNDS; round++) {
      if (!this.asks(proc, session, turn)) return PERMISSION_CANCELLED;
      const mode = this.modeOf(session.id);
      if (mode !== 'ask' && !offersOnce(req)) {
        turn.recorded = 'denied';
        return PERMISSION_CANCELLED;
      }
      if (mode === 'bypass') {
        const answer = await this.allowAll(proc, session, turn, req, place, landing);
        if (answer !== MOVED) return answer;
        continue;
      }
      if (!showableWhole(req)) return this.tooLong(turn, req);
      const interaction = workInteraction(randomUUID(), req, this.cardNeedles(), place, landing);
      if (mode === 'ask') return this.workCard(session, req, interaction);
      const answer = await this.autoAnswer(proc, session, turn, req, place, landing, seen, interaction);
      if (answer !== MOVED) return answer;
    }
    if (!this.asks(proc, session, turn)) return PERMISSION_CANCELLED;
    if (!showableWhole(req)) return this.tooLong(turn, req);
    return this.workCard(session, req, workInteraction(randomUUID(), req, this.cardNeedles(), place, landing));
  }

  /** A command or path too long to show whole is denied without asking: the card must show exactly what would run. */
  private tooLong(turn: TurnRecord, req: PermissionRequest): unknown {
    turn.recorded = 'denied';
    const deny = req.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID && o.kind === 'reject_once');
    return deny ? { outcome: { outcome: 'selected', optionId: REJECT_ONCE_OPTION_ID } } : PERMISSION_CANCELLED;
  }

  /** The card, opened in this tick; the user's answer settles it. */
  private workCard(session: HostSession, req: PermissionRequest, interaction: PermissionInteraction): Promise<unknown> {
    return this.openInteraction(session, interaction, PERMISSION_CANCELLED, helperStart(req), staysBehind(req));
  }

  /**
   * Bypass: `allow-once` for any request, after its history line. No gate, no count, no folder limit, no window
   * check. With no `allow-once` offered the card opens. The mode is read again after the line: a move away ends the
   * round `moved`, and a line that was written then is withdrawn.
   */
  private async allowAll(
    proc: AgentProcess,
    session: HostSession,
    turn: TurnRecord,
    req: PermissionRequest,
    place: WorkPlace,
    landing: string | null,
  ): Promise<unknown> {
    if (!req.options.some((o) => o.optionId === ALLOW_ONCE_OPTION_ID && o.kind === 'allow_once')) {
      if (!showableWhole(req)) return this.tooLong(turn, req);
      return this.workCard(session, req, workInteraction(randomUUID(), req, this.cardNeedles(), place, landing));
    }
    const shown = showableWhole(req)
      ? { ...workInteraction(randomUUID(), req, this.cardNeedles(), place, landing), options: [] }
      : null;
    const mark: AllowedMark = {
      kind: 'allowed',
      at: this.now(),
      mode: 'bypass',
      toolCallId: req.toolCallId,
      access: accessOf(req),
      card: shown,
    };
    const written = await this.writeLine(session, mark);
    if (!this.asks(proc, session, turn)) {
      if (written) this.withdraw(session, mark);
      return PERMISSION_CANCELLED;
    }
    if (!written || this.modeOf(session.id) !== 'bypass') {
      if (written) this.withdraw(session, mark);
      return MOVED;
    }
    if (helperStart(req)) session.elsewhere = true;
    this.callStarted(session, req.toolCallId, {
      changes: true,
      wide: true,
      ...(staysBehind(req) ? { stays: true as const } : {}),
    });
    this.opts.listener?.onModeMarked?.(session.id, mark);
    return ALLOW_ONCE_ANSWER;
  }

  /** The history line of an allowed call; false when it was not written. */
  private async writeLine(session: HostSession, mark: AllowedMark): Promise<boolean> {
    try {
      return (await this.opts.listener?.onModeAllowed?.(session.id, mark)) === true;
    } catch {
      return false;
    }
  }

  private withdraw(session: HostSession, mark: AllowedMark): void {
    this.opts.listener?.onModeWithdrawn?.(session.id, mark);
  }

  private folderCalls(cwd: string): FolderCalls {
    const key = folderKey(cwd);
    let folder = this.folders.get(key);
    if (folder === undefined) {
      folder = { gen: 0, busy: new Map() };
      this.folders.set(key, folder);
    }
    return folder;
  }

  /**
   * Raises `gen` of the session's folder, of every folder a change there can reach (all of them for a wide one), and of
   * every folder that holds or lies in a temp root the call reaches.
   */
  private raise(session: HostSession, wide: boolean, reach: readonly string[] = []): void {
    const own = folderKey(session.cwd);
    this.folderCalls(session.cwd);
    const temps = reach.map((top) => folderKey(top));
    for (const [key, folder] of this.folders) {
      if (wide || nestedFolders(key, own) || temps.some((t) => nestedFolders(key, t))) folder.gen += 1;
    }
  }

  /** A call of the session's folder (and of each temp root it reaches) starts running; one that may change it raises `gen`. */
  private callStarted(session: HostSession, toolCallId: string, call: FolderCall): void {
    const id = callKey(session.id, toolCallId);
    this.folderCalls(session.cwd).busy.set(id, call);
    for (const top of call.reach ?? []) this.folderCalls(top).busy.set(id, call);
    if (call.changes) this.raise(session, call.wide, call.reach);
  }

  /**
   * A call leaves the folder (it ended, or its allow was given back). `gen` goes up when it could change the folder,
   * and for an unknown call of a session the agent may run calls for without asking.
   */
  private callEnded(session: HostSession, toolCallId: string): void {
    const folder = this.folderCalls(session.cwd);
    const key = callKey(session.id, toolCallId);
    const call = folder.busy.get(key);
    if (call?.stays === true) {
      this.raise(session, call.wide, call.reach);
      return;
    }
    folder.busy.delete(key);
    for (const top of call?.reach ?? []) this.folderCalls(top).busy.delete(key);
    if (call?.changes === true) this.raise(session, call.wide, call.reach);
    else if (call === undefined && session.granted) this.raise(session, true);
  }

  /**
   * Another call runs that this one could race: in this folder, a folder that holds it or one it holds, either may
   * change the folder; anywhere else, the other may change any folder.
   */
  private folderBusy(session: HostSession, toolCallId: string, changes: boolean, reach: readonly string[]): boolean {
    const own = callKey(session.id, toolCallId);
    const place = folderKey(session.cwd);
    const temps = reach.map((top) => folderKey(top));
    for (const [key, folder] of this.folders) {
      const near = nestedFolders(key, place) || temps.some((t) => nestedFolders(key, t));
      for (const [id, other] of folder.busy) {
        if (id === own) continue;
        if (other.wide || (near && (other.changes || changes))) return true;
      }
    }
    return false;
  }

  /** The place the Auto gate judges: the Work folder, the temp roots and whose files are yours. */
  private gatePlace(place: WorkPlace): WorkPlace {
    return {
      ...place,
      temp: this.opts.tempRoots ?? [],
      uid: this.opts.uid ?? process.getuid?.() ?? -1,
      login: this.opts.login ?? processLogin(),
      refused: this.opts.refused ?? [],
    };
  }

  /**
   * Work: an ended call leaves its folder. In a session with an "always" grant the agent runs calls with no request,
   * so a new call that is not a read joins the folder as one that may change it.
   */
  private trackCall(session: HostSession, update: Record<string, unknown>): void {
    const ended = endedCall(update);
    if (ended !== null) {
      this.callEnded(session, ended);
      return;
    }
    if (!session.granted || update.sessionUpdate !== 'tool_call' || typeof update.toolCallId !== 'string') return;
    const kind = update.kind;
    if (kind === 'read' || kind === 'search' || kind === 'think' || kind === 'fetch') return;
    if (!this.folderCalls(session.cwd).busy.has(callKey(session.id, update.toolCallId))) {
      this.callStarted(session, update.toolCallId, { changes: true, wide: true });
    }
  }

  /**
   * Auto: the app's own `allow-once` when every rule of the gate passes and the history line is written; else the
   * card with the reason, opened in the tick of the last mode check. `MOVED` when the mode is no longer Auto at a
   * check. A call is counted in the same tick as its decision, so requests that arrive together cannot pass the cap;
   * a call that does not get its allow gives the count back. Any error asks.
   */
  private async autoAnswer(
    proc: AgentProcess,
    session: HostSession,
    turn: TurnRecord,
    req: PermissionRequest,
    place: WorkPlace,
    landing: string | null,
    seen: Seen,
    interaction: PermissionInteraction,
  ): Promise<unknown> {
    let decision: AutoDecision;
    let changes = true;
    let stale = false;
    let facts: AutoFacts | null = null;
    const gate = this.gatePlace(place);
    const folder = this.folderCalls(session.cwd);
    const watched = (): boolean => this.opts.permission?.watched() === true;
    /** The gens of the Work folder and of each temp root the call reaches, as they are now. */
    const gens = (reach: readonly string[]): number[] => [folder.gen, ...reach.map((top) => this.folderCalls(top).gen)];
    const moved = (reach: readonly string[]): boolean =>
      folder.gen !== seen.work || reach.some((top) => this.folderCalls(top).gen !== (seen.temps.get(top) ?? -1));
    try {
      facts = await autoFacts(req, gate, landing, { cdpath: this.cdpath() });
      if (!this.asks(proc, session, turn)) return PERMISSION_CANCELLED;
      if (moved(facts.reach)) {
        // A call that may change the folder started or ended while the facts were read: read them once more.
        const again = gens(facts.reach);
        const path = requestTarget(req).path;
        if (path !== null && (await editLanding(agentPath(path, place)).catch(() => null)) !== landing) stale = true;
        const reach = facts.reach;
        facts = await autoFacts(req, gate, landing, { cdpath: this.cdpath() });
        if (!this.asks(proc, session, turn)) return PERMISSION_CANCELLED;
        if (facts.reach.some((top) => !reach.includes(top))) stale = true;
        else if (gens(reach).some((gen, k) => gen !== again[k])) stale = true;
      }
      changes = changesFolder(facts, interaction.access);
      decision = decideAuto({
        auto: this.modeOf(session.id) === 'auto',
        watched: watched(),
        request: req,
        interaction,
        guardHook: this.running?.guardHook ?? null,
        elsewhere: session.elsewhere,
        facts,
        allowed: turn.approvals,
      });
    } catch {
      decision = { allow: false, why: 'unread' };
    }
    const asking = (why: AutoAsk, found?: AutoAskDetail): Promise<unknown> => {
      interaction.autoAsk = why;
      if (found !== undefined) {
        interaction.autoAskDetail = {
          part: shownCut(found.part, AUTO_PART_MAX, this.cardNeedles()),
          more: found.more,
          cause: found.cause,
        };
      }
      return this.workCard(session, req, interaction);
    };
    if (this.modeOf(session.id) !== 'auto') return MOVED;
    if (!decision.allow) return asking(decision.why ?? 'unread', decision.detail);
    if (interaction.access !== 'bash' && interaction.access !== 'edit') return asking('not-covered');
    const reach = facts?.reach ?? tempRoots(gate).map((r) => r.top);
    if (stale || this.folderBusy(session, req.toolCallId, changes, reach)) return asking('unknown-target');
    this.callStarted(session, req.toolCallId, { changes, wide: false, ...(reach.length > 0 ? { reach } : {}) });
    const mine = gens(reach);
    const epoch = turn.approvalEpoch;
    turn.approvals += 1;
    const giveBack = (): void => {
      if (turn.approvalEpoch === epoch && turn.approvals > 0) turn.approvals -= 1;
      this.callEnded(session, req.toolCallId);
    };
    const mark: AllowedMark = {
      kind: 'allowed',
      at: this.now(),
      mode: 'auto',
      toolCallId: req.toolCallId,
      access: interaction.access,
      card: { ...interaction, options: [] },
      ...(facts?.touched === true ? { temp: true as const } : {}),
    };
    const written = await this.writeLine(session, mark);
    if (!this.asks(proc, session, turn)) {
      giveBack();
      if (written) this.withdraw(session, mark);
      return PERMISSION_CANCELLED;
    }
    if (!written || this.modeOf(session.id) !== 'auto') {
      giveBack();
      if (written) this.withdraw(session, mark);
      return MOVED;
    }
    if (!watched()) {
      giveBack();
      this.withdraw(session, mark);
      return asking('no-window');
    }
    if (gens(reach).some((gen, k) => gen !== mine[k])) {
      giveBack();
      this.withdraw(session, mark);
      return asking('unknown-target');
    }
    this.opts.listener?.onModeMarked?.(session.id, mark);
    return ALLOW_ONCE_ANSWER;
  }

  /** The Work environment sets `CDPATH`. */
  private cdpath(): boolean {
    const value = (this.opts.baseEnv ?? process.env).CDPATH;
    return value !== undefined && value !== '';
  }

  /** Work: `_x.ai/exit_plan_mode` becomes a plan card; anything that cannot be shown is answered `cancelled`. */
  private workPlan(proc: AgentProcess, params: unknown): Promise<unknown> {
    const req = readPlanRequest(params);
    const session = req === null ? undefined : this.sessions.get(req.sessionId);
    const turn = session?.turn ?? null;
    if (req === null || session === undefined || turn === null || !this.asks(proc, session, turn)) {
      return Promise.resolve(PLAN_CANCELLED);
    }
    const interaction = planInteraction(randomUUID(), req, this.cardNeedles());
    return this.openInteraction(session, interaction, PLAN_CANCELLED, false);
  }

  /** Every needle is known before a card text is cut: a key cut in two would no longer match. */
  private cardNeedles(): string[] {
    return needleList((this.opts.needles ?? (() => this.opts.providers.secretNeedles()))());
  }

  private openInteraction(
    session: HostSession,
    interaction: Interaction,
    withdrawn: unknown,
    helper: boolean,
    stays = false,
  ): Promise<unknown> {
    return new Promise<unknown>((resolve) => {
      session.interactions.set(interaction.id, { interaction, settle: resolve, withdrawn, helper, stays });
      this.opts.listener?.onInteraction?.(session.id, interaction);
      this.emitState();
    });
  }

  // ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────

  private async call(running: Running, method: string, params: unknown, timeoutMs: number): Promise<unknown> {
    try {
      return await running.process.client.request(method, params, timeoutMs);
    } catch (error) {
      if (error instanceof AcpRpcError) throw error;
      const broke = running.process.brokeProtocol || running.process.running;
      throw new AgentHostError(broke ? 'agent-protocol-error' : 'agent-crashed');
    }
  }

  private emitTurn(session: HostSession): void {
    const t = session.turn;
    this.opts.listener?.onTurn?.(
      session.id,
      t === null ? null : { turnId: t.turnId, phase: t.phase, startedAt: t.startedAt, wait: t.wait },
    );
  }

  private emitState(): void {
    this.opts.listener?.onState?.(this.modeState());
  }
}

/** The request offers `allow-once` or `reject-once`, each with its own kind. */
function offersOnce(req: PermissionRequest): boolean {
  return req.options.some(
    (o) =>
      (o.optionId === ALLOW_ONCE_OPTION_ID && o.kind === 'allow_once') ||
      (o.optionId === REJECT_ONCE_OPTION_ID && o.kind === 'reject_once'),
  );
}

/** Characters of the asking part on a card. */
const AUTO_PART_MAX = 60;

/** The login name of the process, or undefined when the system has no entry for it. */
function processLogin(): string | undefined {
  try {
    return userInfo().username;
  } catch {
    return undefined;
  }
}

function callKey(sessionId: string, toolCallId: string): string {
  return `${sessionId}\n${toolCallId}`;
}

/** A Work folder's key: its real path with each part folded as the disk compares names. */
function folderKey(cwd: string): string {
  return cwd
    .split('/')
    .map((p) => foldName(p))
    .join('/');
}

/** Two folder keys that are the same folder, or one holds the other. */
function nestedFolders(a: string, b: string): boolean {
  return a === b || a.startsWith(b.endsWith('/') ? b : `${b}/`) || b.startsWith(a.endsWith('/') ? a : `${a}/`);
}

/** The id of a tool call this update reports as ended (`completed` or `failed`), else null. */
function endedCall(update: Record<string, unknown>): string | null {
  if (update.sessionUpdate !== 'tool_call_update' && update.sessionUpdate !== 'tool_call') return null;
  if (update.status !== 'completed' && update.status !== 'failed') return null;
  return typeof update.toolCallId === 'string' ? update.toolCallId : null;
}

/** The key a session's `mlx_web` holds for `provider`, or '' when it holds none. */
function keyOf(web: ChatWebEnv, provider: ProviderId): string {
  if (web.searchProvider === provider) return web.searchKey;
  if (web.fetchProvider === provider) return web.fetchKey;
  return '';
}

function isWebToolName(name: string): boolean {
  return name === 'use_tool' || (WEB_TOOL_KEYS as readonly string[]).includes(name);
}

/** Whether any string inside `value` holds `needle` (bounded depth: agent JSON is untrusted). */
function holdsText(value: unknown, needle: string, depth: number): boolean {
  if (typeof value === 'string') return value.includes(needle);
  if (depth > 32 || typeof value !== 'object' || value === null) return false;
  for (const item of Array.isArray(value) ? value : Object.values(value)) {
    if (holdsText(item, needle, depth + 1)) return true;
  }
  return false;
}

function isSuccess(result: unknown): boolean {
  return typeof result === 'object' && result !== null && (result as Record<string, unknown>).success === true;
}
