/**
 * Chat and Work for the Control Panel: the two agent hosts, the conversations they serve, the host and conversation
 * streams, and the route operations. mlx-node writes metadata only; conversation text is written by the agent into
 * its own home. Every failure a route returns is a typed code with fixed text.
 */

import { existsSync } from 'node:fs';
import { mkdir, readdir, stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MLX_AGENT_MIN_RELEASE } from '@mlx-node/agent/binary';
import { agentConfigDir } from '@mlx-node/agent/paths';

import { AgentSessionsReader } from '../agent-sessions/reader.js';
import type { SessionDetailResponse } from '../agent-sessions/types.js';
import { deleteSession, renameSession, SessionWriteError, validateRenameTitle } from '../agent-sessions/writes.js';
import { ApiError } from '../api/errors.js';
import type { LiveEvent, RuntimeLive, RuntimeLiveModel } from '../live.js';
import type { ChatScope } from '../rpc/protocol.js';
import {
  AgentHost,
  AgentHostError,
  chatFailure,
  type AgentHostListener,
  type HostTimings,
  type LaunchResolution,
  type TurnResult,
} from './agent/agent-host.js';
import { findUserHooks } from './agent/auto.js';
import type { EndpointReader, InferenceEndpoint } from './agent/endpoint.js';
import { safeReader } from './agent/endpoint.js';
import { ALLOW_ONCE_OPTION_ID, chatPrompt, REJECT_ONCE_OPTION_ID } from './agent/profiles.js';
import { UNLOAD_MAX_MS, unloadModels } from './agent/select-model.js';
import { Conversation, type ConversationSink } from './conversation.js';
import { readWorkIndexSync, writeWorkIndex, type WorkIndex } from './index-file.js';
import { removeChatPromptHistory, type ChatPaths } from './paths.js';
import { PermissionLog, markOf, settingsPrint, type PermissionLine, type SettingsPlace } from './permission-log.js';
import type { ProvidersService } from './providers/service.js';
import type { ProviderSource } from './providers/source.js';
import { needleList } from './redact.js';
import { chatRouteError } from './route-errors.js';
import type {
  AgentActionInput,
  CreateConversationInput,
  InteractionAnswerInput,
  PatchConversationInput,
} from './route-input.js';
import { defaultSkillsDir, readChatSkill } from './skills/manifest.js';
import { agentTitle, promptTitle, resolveTitle } from './titles.js';
import type {
  AgentModeState,
  PermissionMark,
  PermissionMode,
  ChatHostEvent,
  ChatHostState,
  ChatMode,
  ConversationEvent,
  ConversationInfo,
  CwdProblem,
  Interaction,
} from './types.js';
import { checkWorkFolder, cwdError, type WorkFolderRules } from './work-folder.js';

/** create: agent start + `session/new` + guard must fit in this, inside the RPC client's 30 s deadline. */
export const CREATE_BUDGET_MS = 20_000;
/** The host stream sends at most one `state` per this. */
export const STATE_COALESCE_MS = 100;
/** With no window attached for this long, open Work requests are denied: nobody is left to answer them. */
export const PAGE_GONE_MS = 60_000;
/** Session text routes wait this long at most for the Keychain before they redact with what is known. */
export const NEEDLES_WAIT_MS = 10_000;

const MODE_RANK: Readonly<Record<PermissionMode, number>> = { ask: 0, auto: 1, bypass: 2 };

type AllowedMark = Extract<PermissionMark, { kind: 'allowed' }>;

function modeChanged(): ApiError {
  return new ApiError('E_CONFLICT', 'The permission mode could not be changed.', 'mode-changed');
}

/** Where the agent binary comes from, for the host state and for each spawn. */
export interface AgentResolution {
  launch: LaunchResolution;
  source: string;
  tag: string | null;
}

export interface ChatServiceOptions {
  paths: ChatPaths;
  readEndpoint: EndpointReader;
  resolveAgent: () => AgentResolution;
  /** A session's detail read from disk (the database worker's reader): both homes, origin tagged. */
  readSession: (id: string) => Promise<SessionDetailResponse | null>;
  /** The provider settings, keys and skill folders; the providers routes use the same instance. */
  providers: ProvidersService;
  live?: { subscribe(listener: (event: LiveEvent) => void): () => void; readNow(): Promise<RuntimeLive> };
  clientVersion: string;
  /** How the agent starts the web tool server. Default: this Node binary and the server entry next to this file. */
  webServer?: { command: string; script: string };
  skillsDir?: string;
  /** Runs `mlx agent update`; without it the install action reports `failed`. */
  installAgent?: () => Promise<void>;
  extraCaCerts?: string;
  /** Work: the Control Panel environment and the agent home when it is not the default. */
  baseEnv?: NodeJS.ProcessEnv;
  agentHome?: string | null;
  /** The agent home Work sessions live in (default `~/.mlx-agent`): cold renames and the Work folder rule. */
  workAgentHome?: string;
  homeDir?: string;
  username?: string;
  pageGoneMs?: number;
  needlesWaitMs?: number;
  timings?: Partial<HostTimings>;
  createBudgetMs?: number;
  flushMs?: number;
  stateCoalesceMs?: number;
  unloadMaxMs?: number;
  now?: () => number;
}

/** The web server entry: the built `server.js` when it exists, else the source bootstrap. */
export function webServerScript(): string {
  const built = new URL('./web/server.js', import.meta.url);
  if (existsSync(fileURLToPath(built))) return fileURLToPath(built);
  return fileURLToPath(new URL('./web/server-ts-bootstrap.ts', import.meta.url));
}

function epochOf(iso: string): number {
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : 0;
}

interface PendingAttach {
  cancelled: boolean;
  detach: (() => void) | null;
}

export class ChatService {
  private readonly opts: ChatServiceOptions;
  private readonly now: () => number;
  private readonly readEndpoint: EndpointReader;
  private readonly providers: ProviderSource;
  private readonly hosts: Record<ChatMode, AgentHost>;
  private readonly chatReader: AgentSessionsReader;
  private readonly workReader: AgentSessionsReader;
  private readonly folderRules: WorkFolderRules;
  private indexChain: Promise<void> = Promise.resolve();
  private pageGoneTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly conversations = new Map<string, Conversation>();
  private readonly resolving = new Map<string, Promise<Conversation | null>>();
  /** Conversations a delete holds: no prompt, rename or model change starts while the delete awaits. */
  private readonly deleting = new Set<string>();
  private readonly hostSinks = new Set<(event: ChatHostEvent) => void>();
  private readonly chatSkillOk: boolean;
  private install: ChatHostState['agent']['install'] = 'idle';
  private inferenceToken: string | null = null;
  private stateTimer: ReturnType<typeof setTimeout> | null = null;
  private liveOff: (() => void) | null = null;
  private readonly sizes = new Map<string, Promise<number | null>>();
  /** The last unload: a heartbeat of that engine written before `at` still names the models it dropped. */
  private unloaded: { pid: number; names: ReadonlySet<string>; at: number } | null = null;
  private closed = false;
  /** Work conversations not in Ask, with their mode. Memory only: never read from any file. */
  private readonly permissionModes = new Map<string, 'auto' | 'bypass'>();
  /** Goes up at each move down: a move up that was sent before it is refused at its next check. */
  private readonly permissionEpoch = new Map<string, number>();
  /** The targets of the moves up sent and not yet settled, per conversation. */
  private readonly raising = new Map<string, PermissionMode[]>();
  /** The mode of the last mode line this run wrote: the `from` of the next one. */
  private readonly logged = new Map<string, PermissionMode>();
  /** Per conversation not in Ask: the settings print the next check compares with. */
  private readonly settingsBase = new Map<string, string>();
  private readonly permissionLog: PermissionLog;
  /** Moves up, and the record of every move, one at a time. */
  private permissionChain: Promise<unknown> = Promise.resolve();
  private readonly home: string;
  private readonly workHome: string;

  constructor(options: ChatServiceOptions) {
    this.opts = options;
    this.now = options.now ?? Date.now;
    this.readEndpoint = safeReader(options.readEndpoint);
    this.providers = options.providers;
    options.providers.attach({
      changed: () => {
        this.hosts.chat.markStale();
        this.hosts.work.markStale();
        this.scheduleState();
      },
      keysChanged: () => {
        // A transcript seeded while the Keychain had not answered was sent with this key in clear.
        for (const conversation of this.conversations.values()) {
          if (conversation.listeners === 0) continue;
          const host = this.hosts[conversation.info.mode];
          const id = conversation.info.id;
          conversation.resync({ turn: host.turnState(id), interactions: host.openInteractions(id) });
        }
      },
      applyPending: () =>
        (['chat', 'work'] as const).some((mode) => {
          const state = this.hosts[mode].modeState();
          return state.stale && state.turnsInFlight > 0;
        }),
    });
    const skillsDir = options.skillsDir ?? defaultSkillsDir();
    let prompt: string | undefined;
    try {
      prompt = chatPrompt(readChatSkill(skillsDir));
    } catch {
      prompt = undefined;
    }
    this.chatSkillOk = prompt !== undefined;
    this.chatReader = new AgentSessionsReader({ agentHome: options.paths.chatAgent, origin: 'chat' });
    const home = options.homeDir ?? homedir();
    const workHome = options.workAgentHome ?? agentConfigDir();
    this.home = home;
    this.workHome = workHome;
    this.permissionLog = new PermissionLog(options.paths.permissionLog);
    this.workReader = new AgentSessionsReader({ agentHome: workHome });
    this.folderRules = {
      home,
      refused: [
        options.paths.root,
        join(home, '.mlx-node'),
        join(homedir(), '.mlx-node'),
        workHome,
        join(home, '.mlx-agent'),
        join(home, 'Library', 'Keychains'),
      ],
    };
    const common = {
      paths: options.paths,
      resolveLaunch: () => this.opts.resolveAgent().launch,
      readEndpoint: this.readEndpoint,
      providers: this.providers,
      clientVersion: options.clientVersion,
      now: this.now,
      timings: options.timings,
      needles: () => this.needles(),
    };
    this.hosts = {
      chat: new AgentHost({
        ...common,
        mode: 'chat',
        chatPrompt: prompt,
        webServer: options.webServer ?? { command: process.execPath, script: webServerScript() },
        extraCaCerts: options.extraCaCerts,
        username: options.username,
        listener: this.listenerFor('chat'),
      }),
      work: new AgentHost({
        ...common,
        mode: 'work',
        baseEnv: options.baseEnv,
        agentHome: options.agentHome ?? null,
        homeDir: options.homeDir,
        checkCwd: (path) => checkWorkFolder(path, this.folderRules),
        permission: { modeOf: (id) => this.modeOf(id), watched: () => this.hostSinks.size > 0 },
        listener: this.listenerFor('work'),
      }),
    };
  }

  // ── routes ───────────────────────────────────────────────────────────────────────────────────────────────────────

  async state(): Promise<ChatHostState> {
    return this.computeState();
  }

  /**
   * Every redaction needle, once the Keychain and the endpoint answered or {@link NEEDLES_WAIT_MS} passed: a route
   * that sends session text must answer inside the RPC deadline, and a later read redacts what was not known yet.
   */
  async redactionNeedles(): Promise<string[]> {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const budget = new Promise<void>((resolve) => {
      timer = setTimeout(resolve, this.opts.needlesWaitMs ?? NEEDLES_WAIT_MS);
    });
    try {
      await Promise.race([this.loadNeedles(), budget]);
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
    return needleList(this.needles());
  }

  async create(input: CreateConversationInput): Promise<{ conversation: ConversationInfo; turnId: string }> {
    this.assertOpen();
    const mode = input.mode;
    let cwd: string | undefined;
    if (mode === 'work') {
      const folder = await checkWorkFolder(input.cwd ?? '', this.folderRules);
      if (!folder.ok) throw cwdError(folder.problem);
      cwd = folder.path;
    } else {
      await this.assertChatReady();
    }
    const endpoint = await this.endpoint();
    if (endpoint === null) throw chatRouteError('inference-stopped');
    const modelId = input.modelId ?? (await this.state()).defaultModelId ?? undefined;
    if (modelId === undefined) throw chatRouteError('no-models');
    const host = this.hosts[mode];
    const created = host.createSession(cwd === undefined ? { modelId } : { modelId, cwd });
    let timer: ReturnType<typeof setTimeout> | null = null;
    const budget = new Promise<'slow'>((resolve) => {
      timer = setTimeout(() => resolve('slow'), this.opts.createBudgetMs ?? CREATE_BUDGET_MS);
    });
    let session;
    try {
      const raced = await Promise.race([created, budget]);
      if (raced === 'slow') {
        created.then(
          (late) => host.deleteSession(late.sessionId, late.cwd).catch(() => {}),
          () => {},
        );
        throw chatRouteError('agent-slow');
      }
      session = raced;
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
    const at = this.now();
    if (mode === 'work') {
      try {
        await this.indexWrite((index) => index.work.push({ id: session.sessionId, createdAt: at }), session.sessionId);
      } catch (error) {
        await host.deleteSession(session.sessionId, session.cwd).catch(() => {});
        throw error;
      }
    }
    const conversation = this.newConversation({
      id: session.sessionId,
      mode,
      title: promptTitle(input.text),
      cwd: mode === 'work' ? session.cwd : null,
      modelId: session.modelId,
      createdAt: at,
      updatedAt: at,
      hot: true,
      permissionMode: 'ask',
    });
    conversation.marksLoaded = true;
    conversation.seed([], [], null);
    this.conversations.set(conversation.info.id, conversation);
    const { turnId } = this.startTurn(conversation, input.text);
    this.emitHost({ type: 'conversations-changed' });
    return { conversation: conversation.publicInfo(), turnId };
  }

  async prompt(id: string, text: string): Promise<{ turnId: string }> {
    this.assertOpen();
    const conversation = await this.known(id);
    const mode = conversation.info.mode;
    let cwd: string | undefined;
    if (mode === 'work') {
      const problem = await this.folderState(conversation);
      if (problem !== null) throw cwdError(problem);
      cwd = conversation.info.cwd ?? undefined;
    } else {
      await this.assertChatReady();
    }
    const host = this.hosts[mode];
    if (host.isBusy(id)) throw chatRouteError('conversation-busy');
    if (!conversation.hasBase()) await this.loadBase(conversation);
    if (host.isBusy(id)) throw chatRouteError('conversation-busy');
    await this.endpoint();
    return { turnId: this.startTurn(conversation, text, cwd).turnId };
  }

  async cancel(id: string): Promise<{ cancelled: boolean }> {
    const conversation = await this.known(id);
    return { cancelled: this.hosts[conversation.info.mode].cancel(id) };
  }

  async answer(id: string, interactionId: string, input: InteractionAnswerInput): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    if (conversation.info.mode === 'chat') throw ApiError.notFound('No such request.');
    if ('optionId' in input) this.hosts.work.answer(id, interactionId, input.optionId);
    else this.hosts.work.answerPlan(id, interactionId, input.decision, input.feedback);
    return {};
  }

  /** The permission mode of a conversation now. */
  modeOf(id: string): PermissionMode {
    return this.permissionModes.get(id) ?? 'ask';
  }

  /**
   * The permission mode of a Work conversation, from the user's pick in the app's window: the only way up. A move
   * down is applied in this tick and is never queued; a move up runs on the permission chain and is refused when a
   * move down, or a pick of a lower mode while it waits, came after it was sent.
   */
  setPermissionMode(id: string, target: PermissionMode): Promise<{ mode: PermissionMode }> {
    if (MODE_RANK[target] < MODE_RANK[this.modeOf(id)]) return this.moveDown(id, target);
    const pending = this.raising.get(id) ?? [];
    if (pending.some((up) => MODE_RANK[up] > MODE_RANK[target])) {
      // A pick at or below the mode while a higher move up waits calls that move back: it is refused at its next check.
      this.permissionEpoch.set(id, this.epochOf(id) + 1);
      if (MODE_RANK[target] === MODE_RANK[this.modeOf(id)]) return this.stay(id);
    }
    const epoch = this.epochOf(id);
    pending.push(target);
    this.raising.set(id, pending);
    const settled = (): void => {
      const at = pending.indexOf(target);
      if (at >= 0) pending.splice(at, 1);
      if (pending.length === 0 && this.raising.get(id) === pending) this.raising.delete(id);
    };
    const run = (): Promise<{ mode: PermissionMode }> => this.raise(id, target, epoch).finally(settled);
    const next = this.permissionChain.then(run, run);
    this.permissionChain = next.catch(() => {});
    return next;
  }

  private async stay(id: string): Promise<{ mode: PermissionMode }> {
    const conversation = this.conversations.get(id);
    if (conversation === undefined || conversation.info.mode !== 'work' || this.deleting.has(id)) {
      throw ApiError.notFound('No such conversation.');
    }
    return { mode: this.modeOf(id) };
  }

  private epochOf(id: string): number {
    return this.permissionEpoch.get(id) ?? 0;
  }

  private async moveDown(id: string, target: PermissionMode): Promise<{ mode: PermissionMode }> {
    const conversation = this.conversations.get(id);
    if (conversation === undefined || conversation.info.mode !== 'work' || this.deleting.has(id)) {
      throw ApiError.notFound('No such conversation.');
    }
    this.lower(conversation, target, 'user');
    return { mode: target };
  }

  private async raise(id: string, target: PermissionMode, epoch: number): Promise<{ mode: PermissionMode }> {
    this.assertOpen();
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.badRequest('Permission mode is for Work only.');
    const gone = (): boolean => this.deleting.has(id) || this.conversations.get(id) !== conversation;
    if (gone()) throw ApiError.notFound('No such conversation.');
    if (this.epochOf(id) !== epoch) throw modeChanged();
    const from = this.modeOf(id);
    if (MODE_RANK[target] <= MODE_RANK[from]) return { mode: from };
    const gated = from === 'ask' && target === 'auto';
    const cardOpen = (): ApiError | null =>
      gated && this.hosts.work.openInteractions(id).length > 0
        ? new ApiError('E_CONFLICT', 'Answer the open request first.', 'card-open')
        : null;
    const open = cardOpen();
    if (open !== null) throw open;
    const problem = await this.folderState(conversation);
    if (problem !== null) throw cwdError(problem);
    const place = this.settingsPlace(conversation);
    const [print, hooks] = await Promise.all([
      settingsPrint(place),
      from === 'ask' ? findUserHooks(place) : Promise.resolve(conversation.info.userHooks === true),
    ]);
    if (gone()) throw ApiError.notFound('No such conversation.');
    if (this.epochOf(id) !== epoch) throw modeChanged();
    const late = cardOpen();
    if (late !== null) throw late;
    if (from !== 'ask') await this.settingsChanged(conversation, print);
    if (gone()) throw ApiError.notFound('No such conversation.');
    if (from === 'ask') this.setUserHooks(conversation, hooks);
    const line: PermissionLine = {
      kind: 'mode',
      at: this.now(),
      mode: target,
      from: this.logged.get(id) ?? 'ask',
      by: 'user',
      print,
    };
    if (target === 'auto' && this.hosts.work.editsGranted(id)) line.editsUnasked = true;
    if (hooks) line.userHooks = true;
    try {
      await this.permissionLog.append(id, line);
    } catch {
      if (gone()) throw ApiError.notFound('No such conversation.');
      throw new ApiError('E_UNAVAILABLE', 'The permission mode could not be changed.');
    }
    if (gone()) throw ApiError.notFound('No such conversation.');
    this.logged.set(id, target);
    if (this.epochOf(id) !== epoch) {
      conversation.permissionMark(markOf(line));
      throw modeChanged();
    }
    const opened = cardOpen();
    if (opened !== null) {
      // A card opened while the line was written: it offers "always" answers, so Auto is refused and the record closed.
      const back: PermissionLine = { kind: 'mode', at: this.now(), mode: 'ask', from: target, by: 'user' };
      this.logged.set(id, 'ask');
      conversation.permissionMark(markOf(line));
      conversation.permissionMark(markOf(back));
      await this.permissionLog.append(id, back).catch(() => {});
      throw opened;
    }
    this.permissionModes.set(id, target === 'auto' ? 'auto' : 'bypass');
    this.settingsBase.set(id, print);
    conversation.setInfo({ permissionMode: target });
    conversation.permissionMark(markOf(line));
    this.scheduleState();
    if (target === 'bypass') {
      const cards = this.hosts.work.openInteractions(id);
      await this.allowOpenCards(conversation, cards, epoch);
    }
    return { mode: this.modeOf(id) };
  }

  /**
   * The move to Bypass answers each open permission card it covers, one at a time: its history line, then the
   * reply, then its mark. A card that offers neither `allow-once` nor `reject-once` is withdrawn; a plan card and a
   * card with `reject-once` only stay. A failed line moves the conversation to Ask; a move down stops the loop.
   */
  private async allowOpenCards(conversation: Conversation, cards: Interaction[], epoch: number): Promise<void> {
    const id = conversation.info.id;
    const host = this.hosts.work;
    for (const card of cards) {
      if (card.kind !== 'permission') continue;
      const once = card.options.some((o) => o.optionId === ALLOW_ONCE_OPTION_ID);
      if (!once && !card.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID)) {
        host.withdrawOpen(id, card.id);
        continue;
      }
      if (!once) continue;
      const mark: AllowedMark = {
        kind: 'allowed',
        at: this.now(),
        mode: 'bypass',
        toolCallId: card.toolCallId,
        access: card.access,
        card: { ...card, options: [] },
      };
      try {
        await this.permissionLog.append(id, {
          kind: 'allowed',
          at: mark.at,
          mode: 'bypass',
          toolCallId: mark.toolCallId,
          access: mark.access,
        });
      } catch {
        if (this.modeOf(id) !== 'ask') this.lower(conversation, 'ask', 'record-failed');
        return;
      }
      if (this.epochOf(id) !== epoch || this.modeOf(id) !== 'bypass') {
        this.modeWithdrawn(id, mark);
        return;
      }
      if (host.allowOpen(id, card.id)) conversation.permissionMark(mark);
      else this.modeWithdrawn(id, mark);
    }
  }

  /**
   * A move down, by the user or by the app (`record-failed`): memory, the epoch, the info and the host state in this
   * tick, so the next request is decided under the new mode at once; the settings check, the mode line and its mark
   * follow on the permission chain (a failed line is ignored).
   */
  private lower(conversation: Conversation, target: PermissionMode, by: 'user' | 'record-failed'): void {
    const id = conversation.info.id;
    this.permissionEpoch.set(id, this.epochOf(id) + 1);
    if (target === 'ask') this.permissionModes.delete(id);
    else this.permissionModes.set(id, target === 'auto' ? 'auto' : 'bypass');
    conversation.setInfo({ permissionMode: target });
    this.scheduleState();
    const run = async (): Promise<void> => {
      const print = await settingsPrint(this.settingsPlace(conversation)).catch(() => null);
      if (print !== null) await this.settingsChanged(conversation, print);
      if (this.modeOf(id) === 'ask') this.settingsBase.delete(id);
      const line: PermissionLine = {
        kind: 'mode',
        at: this.now(),
        mode: target,
        from: this.logged.get(id) ?? 'ask',
        by,
      };
      if (print !== null) line.print = print;
      if (this.conversations.get(id) === conversation) this.logged.set(id, target);
      conversation.permissionMark(markOf(line));
      await this.permissionLog.append(id, line).catch(() => {});
    };
    const next = this.permissionChain.then(run, run);
    this.permissionChain = next.catch(() => {});
  }

  /** The settings print differs from the base: a `settings` mark, and the new print is the next base. */
  private async settingsChanged(conversation: Conversation, print: string): Promise<void> {
    const id = conversation.info.id;
    const base = this.settingsBase.get(id);
    if (base === undefined || print === base) return;
    this.settingsBase.set(id, print);
    const line: PermissionLine = { kind: 'settings', at: this.now(), print };
    conversation.permissionMark(markOf(line));
    await this.permissionLog.append(id, line).catch(() => {});
  }

  private async compareSettings(conversation: Conversation): Promise<void> {
    if (!this.settingsBase.has(conversation.info.id)) return;
    await this.settingsChanged(conversation, await settingsPrint(this.settingsPlace(conversation)));
  }

  /**
   * Auto or Bypass allowed a call: its history line first. A line that was not written moves the conversation to
   * Ask, and the host decides the request again under it. The thread mark follows from the host in the tick of the
   * reply (`modeMarked`).
   */
  private async modeAllowed(id: string, mark: AllowedMark): Promise<boolean> {
    const conversation = this.conversations.get(id);
    if (conversation === undefined || conversation.info.mode !== 'work' || this.modeOf(id) !== mark.mode) return false;
    try {
      await this.permissionLog.append(id, {
        kind: 'allowed',
        at: mark.at,
        mode: mark.mode,
        toolCallId: mark.toolCallId,
        access: mark.access,
      });
    } catch {
      if (this.modeOf(id) !== 'ask' && this.conversations.get(id) === conversation) {
        this.lower(conversation, 'ask', 'record-failed');
      }
      return false;
    }
    return this.conversations.get(id) === conversation;
  }

  private modeMarked(id: string, mark: AllowedMark): void {
    const conversation = this.conversations.get(id);
    if (conversation !== undefined && conversation.info.mode === 'work') conversation.permissionMark(mark);
  }

  /** A written allowed line whose call got no allow: a `withdrawn` line, best effort, so a reload does not list it. */
  private modeWithdrawn(id: string, mark: AllowedMark): void {
    if (this.conversations.get(id)?.info.mode !== 'work') return;
    void this.permissionLog
      .append(id, { kind: 'withdrawn', at: this.now(), of: mark.at, toolCallId: mark.toolCallId })
      .catch(() => {});
  }

  /** A Work session (re)started: the settings check while the mode is not Ask, and the user's hooks again. */
  private async sessionReady(id: string): Promise<void> {
    const conversation = this.conversations.get(id);
    if (conversation === undefined || conversation.info.mode !== 'work') return;
    if (this.modeOf(id) !== 'ask') await this.compareSettings(conversation);
    await this.refreshUserHooks(conversation);
  }

  private async refreshUserHooks(conversation: Conversation): Promise<void> {
    if (conversation.info.mode !== 'work' || conversation.info.cwd === null) return;
    const found = await findUserHooks(this.settingsPlace(conversation)).catch(() => true);
    this.setUserHooks(conversation, found);
  }

  private setUserHooks(conversation: Conversation, found: boolean): void {
    if ((conversation.info.userHooks === true) === found) return;
    const info = { ...conversation.info };
    delete info.userHooks;
    conversation.info = info;
    conversation.setInfo(found ? { userHooks: true } : {});
  }

  private settingsPlace(conversation: Conversation): SettingsPlace {
    return { home: this.home, agentHome: this.workHome, cwd: conversation.info.cwd ?? this.home };
  }

  /**
   * The record of a Work conversation, read once per app run. A last mode line other than Ask is history from an
   * earlier run: the settings are compared with the last print and a line to Ask (`restart`) is added. Nothing here
   * can move the mode: it is never read from the file.
   */
  private async loadMarks(conversation: Conversation): Promise<void> {
    if (conversation.info.mode !== 'work' || conversation.marksLoaded) return;
    conversation.marksLoaded = true;
    const id = conversation.info.id;
    const lines = await this.permissionLog.read(id);
    for (const line of lines) conversation.permissionMark(markOf(line));
    let last: PermissionMode = 'ask';
    let base: string | undefined;
    for (const line of lines) {
      if (line.kind === 'mode') last = line.mode;
      if (line.kind !== 'allowed' && line.print !== undefined) base = line.print;
    }
    if (last === 'ask' || this.modeOf(id) !== 'ask') return;
    if (base !== undefined) {
      const print = await settingsPrint(this.settingsPlace(conversation));
      if (print !== base) {
        const changed: PermissionLine = { kind: 'settings', at: this.now(), print };
        conversation.permissionMark(markOf(changed));
        await this.permissionLog.append(id, changed).catch(() => {});
      }
    }
    const restart: PermissionLine = { kind: 'mode', at: this.now(), mode: 'ask', from: last, by: 'restart' };
    this.logged.set(id, 'ask');
    conversation.permissionMark(markOf(restart));
    await this.permissionLog.append(id, restart).catch(() => {});
  }

  async patch(id: string, input: PatchConversationInput): Promise<ConversationInfo> {
    const conversation = await this.known(id);
    const mode = conversation.info.mode;
    const host = this.hosts[mode];
    await host.settled();
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    if ('title' in input) {
      const title = validateRenameTitle(input.title);
      if (host.isHot(id)) await host.rename(id, title);
      else renameSession(mode === 'chat' ? this.chatReader : this.workReader, id, title);
      conversation.setInfo({ title, updatedAt: this.now() });
      this.emitHost({ type: 'conversations-changed' });
      return conversation.publicInfo();
    }
    const modelId = input.modelId;
    if (host.isHot(id)) {
      await host.setModel(id, modelId);
    } else {
      const endpoint = await this.endpoint();
      if (this.deleting.has(id)) throw chatRouteError('conversation-busy');
      const available = host.models()?.map((m) => m.modelId) ?? endpoint?.models.map((m) => m.name) ?? [];
      if (!available.includes(modelId)) throw chatRouteError('model-unavailable');
    }
    if (modelId !== conversation.info.modelId) {
      if (!conversation.hasBase()) await this.loadBase(conversation);
      conversation.modelSwitch(modelId);
      conversation.setInfo({ modelId });
    }
    return conversation.publicInfo();
  }

  async delete(id: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    if (conversation.info.mode === 'work') return this.deleteWork(conversation);
    const host = this.hosts.chat;
    await host.settled();
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    this.deleting.add(id);
    try {
      if (host.isHot(id) || (await this.agentCanStart(host))) await host.deleteSession(id, '');
      else this.deleteCold(id);
      await removeChatPromptHistory(this.opts.paths);
      conversation.gone('deleted');
      this.conversations.delete(id);
      this.emitHost({ type: 'conversations-changed' });
      return {};
    } finally {
      this.deleting.delete(id);
    }
  }

  /**
   * Frees the engine's memory: the resident chat model is unloaded, and it loads again on the next message. Refused
   * while an answer runs in Chat or Work or the engine loads a model.
   */
  async unloadModels(): Promise<{ unloaded: string[] }> {
    this.assertOpen();
    const endpoint = await this.endpoint();
    if (endpoint === null) throw chatRouteError('inference-stopped');
    const loading = await this.engineLoading(endpoint.pid);
    const turns = this.hosts.chat.modeState().turnsInFlight + this.hosts.work.modeState().turnsInFlight;
    if (loading !== null || turns > 0) throw chatRouteError('engine-busy');
    const answer = await unloadModels(endpoint, { maxMs: this.opts.unloadMaxMs ?? UNLOAD_MAX_MS });
    if (answer === 'busy') throw chatRouteError('engine-busy');
    if (typeof answer === 'string') throw chatRouteError('inference-failed');
    this.unloaded = { pid: endpoint.pid, names: new Set(answer.unloaded), at: this.now() };
    this.scheduleState();
    return answer;
  }

  async agentAction(input: AgentActionInput): Promise<ChatHostState> {
    this.assertOpen();
    if (input.action === 'restart') {
      const modes: ChatMode[] = input.mode === undefined ? ['chat', 'work'] : [input.mode];
      for (const mode of modes) await this.hosts[mode].restart();
    } else if (this.install !== 'running') {
      const run = this.opts.installAgent;
      this.install = run === undefined ? 'failed' : 'running';
      if (run !== undefined) {
        void run().then(
          () => this.installDone('idle'),
          () => this.installDone('failed'),
        );
      }
    }
    this.scheduleState();
    return this.state();
  }

  /**
   * A Work delete always goes through the agent, which also drops the session from its search index; the session
   * writer is never used for a Work session. A host that cannot start deletes nothing.
   */
  private async deleteWork(conversation: Conversation): Promise<Record<string, never>> {
    const id = conversation.info.id;
    const host = this.hosts.work;
    await host.settled();
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    this.deleting.add(id);
    try {
      await host.deleteSession(id, conversation.info.cwd ?? '');
      await this.indexWrite((index) => {
        index.work = index.work.filter((entry) => entry.id !== id);
      });
      this.permissionModes.delete(id);
      this.permissionEpoch.delete(id);
      this.logged.delete(id);
      this.settingsBase.delete(id);
      await this.permissionLog.remove(id).catch(() => {});
      conversation.gone('deleted');
      this.conversations.delete(id);
      this.emitHost({ type: 'conversations-changed' });
      return {};
    } finally {
      this.deleting.delete(id);
    }
  }

  // ── streams ──────────────────────────────────────────────────────────────────────────────────────────────────────

  watch(scope: ChatScope, send: (event: ChatHostEvent | ConversationEvent) => void): () => void {
    if (scope.scope === 'host') return this.watchHost(send);
    return this.watchConversation(scope.conversationId, send);
  }

  watchHost(send: (event: ChatHostEvent) => void): () => void {
    let live = true;
    let started = false;
    const sink = (event: ChatHostEvent): void => {
      if (live && started) send(event);
    };
    this.hostSinks.add(sink);
    this.watchPage();
    this.updateLive();
    void this.state().then(
      (state) => {
        if (!live) return;
        started = true;
        send({ type: 'state', state });
      },
      () => {},
    );
    return () => {
      live = false;
      this.hostSinks.delete(sink);
      this.watchPage();
      this.updateLive();
    };
  }

  watchConversation(id: string, send: ConversationSink): () => void {
    const pending: PendingAttach = { cancelled: false, detach: null };
    const attach = async (): Promise<void> => {
      let conversation: Conversation | null;
      try {
        conversation = await this.resolve(id);
        if (conversation !== null && !conversation.hasBase()) await this.loadBase(conversation);
        if (conversation?.info.mode === 'work') {
          await this.folderState(conversation);
          await this.refreshUserHooks(conversation);
        }
      } catch {
        conversation = null;
      }
      if (pending.cancelled) return;
      if (conversation === null) {
        send({ type: 'gone', seq: 1, reason: 'unknown' });
        return;
      }
      const host = this.hosts[conversation.info.mode];
      conversation.info = { ...conversation.info, hot: host.isHot(id) };
      pending.detach = conversation.attach(send, {
        turn: host.turnState(id),
        interactions: host.openInteractions(id),
      });
    };
    void attach();
    return () => {
      pending.cancelled = true;
      pending.detach?.();
    };
  }

  /** A live watcher event: while turns run, the endpoint is re-read; the host state follows the engine. */
  onRuntimeEvent(): void {
    void this.hosts.chat.checkEndpoint();
    void this.hosts.work.checkEndpoint();
    this.scheduleState();
  }

  /** Control Panel exit: open requests are answered, every turn is cancelled, then the stop ladder. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.stateTimer !== null) clearTimeout(this.stateTimer);
    this.stateTimer = null;
    if (this.pageGoneTimer !== null) clearTimeout(this.pageGoneTimer);
    this.pageGoneTimer = null;
    this.liveOff?.();
    this.liveOff = null;
    await Promise.all([this.hosts.chat.close(), this.hosts.work.close()]);
    for (const conversation of this.conversations.values()) conversation.flush();
    this.hostSinks.clear();
  }

  // ── internals ────────────────────────────────────────────────────────────────────────────────────────────────────

  /**
   * A cold Chat delete goes through the agent when it runs or can start: the session writer refuses a session the
   * agent wrote in the last 30 s (right after a restart or an idle stop), the agent does not.
   */
  private async agentCanStart(host: AgentHost): Promise<boolean> {
    try {
      await host.start();
      return true;
    } catch {
      return false;
    }
  }

  /** The fallback when the agent cannot start: the session writer, whose recent-write refusal is a typed busy. */
  private deleteCold(id: string): void {
    try {
      deleteSession(this.chatReader, id);
    } catch (error) {
      if (error instanceof SessionWriteError && error.kind === 'conflict') throw chatRouteError('conversation-busy');
      throw error;
    }
  }

  /**
   * The stored folder of a Work conversation against the folder rule; a problem is kept on the conversation (the page
   * shows why it cannot continue) and returned. The folder can go away between turns; nothing here throws.
   */
  private async folderState(conversation: Conversation): Promise<CwdProblem | null> {
    const stored = conversation.info.cwd;
    const check = stored === null ? null : await checkWorkFolder(stored, this.folderRules);
    const problem: CwdProblem | null = check === null ? 'missing' : check.ok ? null : check.problem;
    if ((conversation.info.cwdProblem ?? null) !== problem) {
      const info = { ...conversation.info };
      delete info.cwdProblem;
      conversation.info = info;
      conversation.setInfo(problem === null ? {} : { cwdProblem: problem });
    }
    return problem;
  }

  /**
   * One change of `conversations.json`, after every earlier one. Ids with no session in the Work agent home are
   * dropped, except `keep` (a session the agent made a moment ago) and the conversations this service holds.
   */
  private indexWrite(change: (index: WorkIndex) => void, keep?: string): Promise<void> {
    const run = async (): Promise<void> => {
      await mkdir(this.opts.paths.root, { recursive: true, mode: 0o700 });
      const index = readWorkIndexSync(this.opts.paths.conversations);
      change(index);
      const onDisk = this.workReader.scan().sessions;
      index.work = index.work.filter((e) => e.id === keep || onDisk.has(e.id) || this.conversations.has(e.id));
      await writeWorkIndex(this.opts.paths.conversations, index);
    };
    const next = this.indexChain.then(run, run);
    this.indexChain = next.catch(() => {});
    return next;
  }

  /** Open Work requests while no window holds the host stream: denied once {@link PAGE_GONE_MS} passed. */
  private watchPage(): void {
    const waiting = this.hostSinks.size === 0 && !this.closed && this.hosts.work.hasOpenInteractions();
    if (!waiting) {
      if (this.pageGoneTimer !== null) clearTimeout(this.pageGoneTimer);
      this.pageGoneTimer = null;
      return;
    }
    if (this.pageGoneTimer !== null) return;
    this.pageGoneTimer = setTimeout(() => {
      this.pageGoneTimer = null;
      if (this.hostSinks.size === 0) this.hosts.work.denyOpen();
    }, this.opts.pageGoneMs ?? PAGE_GONE_MS);
    this.pageGoneTimer.unref?.();
  }

  private assertOpen(): void {
    if (this.closed) throw chatRouteError('agent-crashed');
  }

  private async assertChatReady(): Promise<void> {
    if (!this.chatSkillOk || !(await this.ensureSkills())) throw chatRouteError('skill-bundle-corrupt');
  }

  /** The start-up skill sync ran; true when the Chat skill is installed and verified. */
  private ensureSkills(): Promise<boolean> {
    return this.providers.chatSkillReady().catch(() => false);
  }

  private async endpoint(): Promise<InferenceEndpoint | null> {
    const endpoint = await this.readEndpoint();
    if (endpoint !== null) this.inferenceToken = endpoint.token;
    return endpoint;
  }

  /** A transcript read from disk may hold the inference token or a provider key: both are known before the seed. */
  private async loadNeedles(): Promise<void> {
    await Promise.all([this.endpoint(), this.providers.settled().catch(() => {})]);
  }

  private needles(): string[] {
    const needles = [...this.providers.secretNeedles()];
    if (this.inferenceToken !== null) needles.push(this.inferenceToken);
    return needles;
  }

  private newConversation(info: ConversationInfo): Conversation {
    return new Conversation(info, { needles: () => this.needles(), flushMs: this.opts.flushMs });
  }

  /** A conversation this service knows: in its table, or a session of the Chat home (or a Work session) on disk. */
  private async known(id: string): Promise<Conversation> {
    const conversation = await this.resolve(id);
    if (conversation === null) throw ApiError.notFound('No such conversation.');
    return conversation;
  }

  private resolve(id: string): Promise<Conversation | null> {
    const known = this.conversations.get(id);
    if (known !== undefined) return Promise.resolve(known);
    let pending = this.resolving.get(id);
    if (pending === undefined) {
      pending = this.readConversation(id).finally(() => this.resolving.delete(id));
      this.resolving.set(id, pending);
    }
    return pending;
  }

  private async readConversation(id: string): Promise<Conversation | null> {
    const [detail] = await Promise.all([this.opts.readSession(id), this.loadNeedles()]);
    if (detail === null) return null;
    const origin = detail.session.origin;
    if (origin !== 'chat' && origin !== 'work') return null;
    const existing = this.conversations.get(id);
    if (existing !== undefined) return existing;
    const conversation = this.newConversation(infoOf(detail, origin, this.hosts[origin].isHot(id)));
    conversation.seed(detail.transcript, detail.turns, detail.transcriptError ?? null);
    await this.loadMarks(conversation);
    const raced = this.conversations.get(id);
    if (raced !== undefined) return raced;
    this.conversations.set(id, conversation);
    return conversation;
  }

  private async loadBase(conversation: Conversation): Promise<void> {
    const [detail] = await Promise.all([this.opts.readSession(conversation.info.id), this.loadNeedles()]);
    if (conversation.hasBase()) return;
    if (detail === null) {
      conversation.seed([], [], null);
      return;
    }
    conversation.info = { ...conversation.info, title: agentTitle(detail.session.title) || conversation.info.title };
    conversation.seed(detail.transcript, detail.turns, detail.transcriptError ?? null);
  }

  private startTurn(conversation: Conversation, text: string, cwd?: string): { turnId: string } {
    const id = conversation.info.id;
    const host = this.hosts[conversation.info.mode];
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    // A delete that finished while the prompt read the disk: the agent would load the deleted session back.
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    const entries = conversation.diskEntries();
    let started;
    try {
      started = host.startTurn(id, text, {
        modelId: conversation.info.modelId,
        ...(cwd === undefined ? {} : { cwd }),
        ...(entries === undefined ? {} : { entries }),
      });
    } catch (error) {
      if (error instanceof AgentHostError) throw chatRouteError(error.code);
      throw error;
    }
    conversation.userPrompt(text);
    void started.done.catch(() => {});
    this.updateLive();
    return { turnId: started.turnId };
  }

  private listenerFor(mode: ChatMode): AgentHostListener {
    const conv = (id: string): Conversation | undefined => {
      const c = this.conversations.get(id);
      return c?.info.mode === mode ? c : undefined;
    };
    return {
      onState: () => this.scheduleState(),
      onUpdate: (id, line) => conv(id)?.applyLine(line),
      onTitle: (id, title) => {
        const c = conv(id);
        const named = resolveTitle(title, null);
        if (c === undefined || named === '') return;
        c.setInfo({ title: named });
        this.emitHost({ type: 'conversations-changed' });
      },
      onModel: (id, modelId) => conv(id)?.setInfo({ modelId }),
      onTurn: (id, turn) => {
        const c = conv(id);
        if (c === undefined || turn === null) return;
        if (turn.phase === 'running' && !c.info.hot && this.hosts[mode].isHot(id)) c.setInfo({ hot: true });
        c.turnChanged(turn);
      },
      onTurnEnd: (id, result: TurnResult) => {
        const c = conv(id);
        if (c !== undefined) {
          c.turnEnded(result);
          c.info = { ...c.info, updatedAt: this.now(), hot: this.hosts[mode].isHot(id) };
          this.emitHost({ type: 'conversations-changed' });
        }
        this.updateLive();
        this.scheduleState();
      },
      onInteraction: (id, interaction) => {
        conv(id)?.interaction(interaction);
        this.watchPage();
      },
      onInteractionClosed: (id, interactionId, reason) => {
        conv(id)?.interactionClosed(interactionId, reason);
        this.watchPage();
      },
      onNotice: (id, notice) => conv(id)?.notice(notice),
      onSessionReady: (id) => void this.sessionReady(id),
      onModeAllowed: (id, mark) => this.modeAllowed(id, mark),
      onModeMarked: (id, mark) => this.modeMarked(id, mark),
      onModeWithdrawn: (id, mark) => this.modeWithdrawn(id, mark),
      onCold: (ids) => {
        for (const id of ids) {
          const c = conv(id);
          if (c !== undefined) void this.goCold(c);
        }
      },
    };
  }

  /** Hot → cold: attached pages get a fresh `sync` read from disk; with none attached the base is dropped. */
  private async goCold(conversation: Conversation): Promise<void> {
    conversation.info = { ...conversation.info, hot: false };
    if (conversation.listeners === 0) {
      conversation.dropBase();
      return;
    }
    conversation.flush();
    const version = conversation.baseVersion;
    const detail = await this.opts.readSession(conversation.info.id).catch(() => null);
    const host = this.hosts[conversation.info.mode];
    const id = conversation.info.id;
    if (host.isHot(id)) return;
    // A prompt sent during the read is in the base and not on disk yet: a reseed would drop it.
    const reseed = detail !== null && !host.isBusy(id) && conversation.baseVersion === version;
    if (reseed) conversation.seed(detail.transcript, detail.turns, detail.transcriptError ?? null);
    conversation.resync({
      turn: host.turnState(conversation.info.id),
      interactions: host.openInteractions(conversation.info.id),
    });
  }

  private installDone(result: 'idle' | 'failed'): void {
    this.install = result;
    this.scheduleState();
  }

  /** The live watcher is open while a turn runs or a host stream is attached. */
  private updateLive(): void {
    const live = this.opts.live;
    if (live === undefined || this.closed) return;
    const wanted =
      this.hostSinks.size > 0 ||
      this.hosts.chat.modeState().turnsInFlight > 0 ||
      this.hosts.work.modeState().turnsInFlight > 0;
    if (wanted && this.liveOff === null) {
      this.liveOff = live.subscribe((event) => {
        if (event.type === 'runtime') this.onRuntimeEvent();
      });
    } else if (!wanted && this.liveOff !== null) {
      this.liveOff();
      this.liveOff = null;
    }
  }

  private emitHost(event: ChatHostEvent): void {
    for (const sink of Array.from(this.hostSinks)) sink(event);
  }

  private scheduleState(): void {
    if (this.closed || this.hostSinks.size === 0 || this.stateTimer !== null) return;
    this.stateTimer = setTimeout(() => {
      this.stateTimer = null;
      void this.state().then(
        (state) => this.emitHost({ type: 'state', state }),
        () => {},
      );
    }, this.opts.stateCoalesceMs ?? STATE_COALESCE_MS);
  }

  private modeState(mode: ChatMode): AgentModeState {
    const state = this.hosts[mode].modeState();
    if (mode === 'chat' && !this.chatSkillOk) {
      return { ...state, state: 'blocked', failure: chatFailure('skill-bundle-corrupt') };
    }
    return state;
  }

  private async readLive(): Promise<RuntimeLive | null> {
    try {
      return (await this.opts.live?.readNow()) ?? null;
    } catch {
      return null;
    }
  }

  /** The load the app's engine runs now, from its heartbeat. */
  private async engineLoading(pid: number): Promise<RuntimeLive['processes'][number]['loading']> {
    const live = await this.readLive();
    return live?.processes.find((p) => p.status === 'running' && p.pid === pid && p.loading !== null)?.loading ?? null;
  }

  private async computeState(): Promise<ChatHostState> {
    const endpoint = await this.endpoint();
    const live = await this.readLive();
    // Only the endpoint file's engine: the app never selects on a terminal `mlx serve` or `mlx launch` engine.
    const running = live?.processes.filter((p) => p.status === 'running' && p.pid === endpoint?.pid) ?? [];
    const loading = running.find((p) => p.loading !== null)?.loading ?? null;
    const supervisor = live?.supervisor?.state;
    const inference: ChatHostState['inference'] =
      endpoint !== null
        ? loading !== null
          ? 'loading'
          : 'ready'
        : supervisor === 'starting' || supervisor === 'restarting'
          ? 'starting'
          : 'stopped';
    const known = this.hosts.chat.models() ?? this.hosts.work.models() ?? [];
    const models = (endpoint?.models ?? []).map((m) => ({
      id: m.name,
      contextTokens: known.find((k) => k.modelId === m.name)?.contextTokens ?? null,
    }));
    const inMemory = new Map<string, RuntimeLiveModel>();
    const unloaded = this.unloaded;
    for (const process of running) {
      const stale = unloaded !== null && process.pid === unloaded.pid && process.ts < unloaded.at;
      if (unloaded !== null && process.pid === unloaded.pid && !stale) this.unloaded = null;
      for (const model of process.models)
        if (!stale || !unloaded.names.has(model.name)) inMemory.set(model.name, model);
    }
    const resident: ChatHostState['resident'] = [];
    const answers: ChatHostState['answers'] = [];
    for (const m of endpoint?.models ?? []) {
      const live = inMemory.get(m.name);
      if (live === undefined) continue;
      resident.push({ modelId: m.name, sizeBytes: await this.diskSize(m.path) });
      let lastStart: number | null = null;
      for (const process of running)
        for (const turn of process.turns)
          if (turn.model === m.name && (lastStart === null || turn.startedAt > lastStart)) lastStart = turn.startedAt;
      answers.push({
        modelId: m.name,
        inFlight: live.inFlight,
        maxConcurrent: live.maxConcurrent,
        queued: live.queued,
        lastStart,
      });
    }
    const agent = this.agentState();
    const active: ChatHostState['active'] = [];
    for (const mode of ['chat', 'work'] as const) {
      for (const a of this.hosts[mode].active()) {
        active.push({ conversationId: a.sessionId, mode, busy: a.busy, waiting: a.waiting });
      }
    }
    return {
      inference,
      loading: loading === null ? null : { modelId: loading.model, startedAt: loading.startedAt },
      resident,
      answers,
      models,
      defaultModelId: resident[0]?.modelId ?? models[0]?.id ?? null,
      agent,
      modes: { chat: this.modeState('chat'), work: this.modeState('work') },
      providers: this.providers.summary(),
      active,
      permissionRaised: [...this.permissionModes].map(([id, mode]) => ({ id, mode })),
    };
  }

  /** The size on disk of a model path, read once per path; null when it cannot be read. */
  private diskSize(path: string): Promise<number | null> {
    let size = this.sizes.get(path);
    if (size === undefined) {
      size = pathSize(path, 0).catch(() => null);
      this.sizes.set(path, size);
    }
    return size;
  }

  private agentState(): ChatHostState['agent'] {
    let resolved: AgentResolution;
    try {
      resolved = this.opts.resolveAgent();
    } catch {
      resolved = { launch: { failure: 'agent-missing' }, source: 'none', tag: null };
    }
    const launch = resolved.launch;
    const failure = 'failure' in launch ? launch.failure : null;
    return {
      binary: failure === 'agent-missing' ? 'missing' : failure === 'agent-too-old' ? 'too-old' : 'ok',
      source: resolved.source,
      tag: resolved.tag,
      minTag: MLX_AGENT_MIN_RELEASE,
      versionVerified: 'failure' in launch ? false : launch.versionVerified,
      install: this.install,
    };
  }
}

/** Bytes of a file, or of every file under a folder (links are followed, two levels at most). */
async function pathSize(path: string, depth: number): Promise<number> {
  const info = await stat(path);
  if (!info.isDirectory()) return info.size;
  if (depth >= 2) return 0;
  let total = 0;
  for (const name of await readdir(path)) total += await pathSize(join(path, name), depth + 1);
  return total;
}

function infoOf(detail: SessionDetailResponse, mode: ChatMode, hot: boolean): ConversationInfo {
  const s = detail.session;
  return {
    id: s.id,
    mode,
    title: agentTitle(s.title),
    cwd: mode === 'work' ? s.cwd : null,
    modelId: s.model ?? '',
    createdAt: epochOf(s.createdAt),
    updatedAt: epochOf(s.lastActiveAt),
    hot,
    permissionMode: 'ask',
  };
}
