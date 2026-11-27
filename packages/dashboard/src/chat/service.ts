/**
 * Chat and Work for the Control Panel: the two agent hosts, the conversations they serve, the host and conversation
 * streams, and the route operations. mlx-node writes metadata only; conversation text is written by the agent into
 * its own home. Every failure a route returns is a typed code with fixed text.
 */

import { existsSync } from 'node:fs';
import { mkdir, readdir, stat } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MLX_AGENT_MIN_RELEASE } from '@mlx-node/agent/binary';
import { agentConfigDir } from '@mlx-node/agent/paths';
import { ggufSplitPartNames, parseGgufFileName } from '@mlx-node/lm/model-discovery';
import { levelsFor } from '@mlx-node/server/host/thinking-levels';

import { AgentSessionsReader } from '../agent-sessions/reader.js';
import type { UpdateLine } from '../agent-sessions/transcript.js';
import type { SessionDetailResponse, TranscriptEntry } from '../agent-sessions/types.js';
import { deleteSession, renameSession, SessionWriteError, validateRenameTitle } from '../agent-sessions/writes.js';
import { ApiError } from '../api/errors.js';
import type { LiveEvent, RuntimeLive, RuntimeLiveModel } from '../live.js';
import type { ChatScope } from '../rpc/protocol.js';
import type { SessionInfoFacts } from './acp/messages.js';
import {
  AgentHost,
  AgentHostError,
  chatFailure,
  type AgentHostListener,
  type HostTimings,
  type LaunchResolution,
  type McpClosed,
  type NoteDeny,
  type RewindHostEnd,
  type RewindPreview,
  type RuleDeny,
  type TurnResult,
} from './agent/agent-host.js';
import { findUserHooks } from './agent/auto.js';
import type { EndpointReader, InferenceEndpoint } from './agent/endpoint.js';
import { safeReader } from './agent/endpoint.js';
import { folderKey, NeverRules, neverNote } from './agent/never-rules.js';
import {
  HOST_START_FAMILIES,
  pdfToolBlocked,
  pdfToolShadowed,
  READ_PDF_WORKFLOW,
  readPdfCommand,
} from './agent/pdf-tool.js';
import type { PdfFacts } from './agent/permissions.js';
import { ALLOW_ONCE_OPTION_ID, chatPrompt, REJECT_ONCE_OPTION_ID } from './agent/profiles.js';
import { tempRoots, type AllowedRoot } from './agent/roots.js';
import { UNLOAD_MAX_MS, unloadModels } from './agent/select-model.js';
import { visibleText } from './agent/visible-text.js';
import {
  attachmentView,
  type AttachmentInput,
  CHAT_IMAGES_MAX,
  CHAT_PICTURES_MAX_BASE64,
  imageCount,
  isToolPdf,
  pdfToolId,
  type PdfStartBy,
  storedPictureChars,
} from './attachment-rule.js';
import { checkCommand } from './commands.js';
import { ComposerStore } from './composer-store.js';
import { Conversation, type ConversationSink, type HostView } from './conversation.js';
import { PermissionDraft } from './draft-mode.js';
import { readWorkIndexSync, writeWorkIndex, type WorkIndex } from './index-file.js';
import { readModelWindowsSync, writeModelWindows } from './model-windows.js';
import { removeChatPromptHistory, type ChatPaths } from './paths.js';
import { PdfStore, readPdfMeta, removeSessionPdfs } from './pdf-store.js';
import {
  changedSettings,
  markOf,
  PermissionLog,
  settingsPrints,
  storedServer,
  type PermissionLine,
  type SettingsPlace,
  type SettingsPrints,
} from './permission-log.js';
import type { ProvidersService } from './providers/service.js';
import type { ProviderSource } from './providers/source.js';
import { ConversationQueue, QUEUE_FULL_MESSAGE } from './queue.js';
import { needleList, redactText } from './redact.js';
import { answerEnd, planFiles, rewindCounts, targetOf } from './rewind-target.js';
import { chatRouteError, folderNotMadeError, routeError, workGitError } from './route-errors.js';
import {
  type AgentActionInput,
  checkAttachments,
  type CreateConversationInput,
  type InteractionAnswerInput,
  isSlashText,
  type PatchConversationInput,
  type PdfStageInput,
  PLAN_WORK_ONLY,
  type QueuePatchInput,
  type RewindInput,
} from './route-input.js';
import { defaultSkillsDir, readChatSkill } from './skills/manifest.js';
import { stepLevel } from './thinking-levels.js';
import { agentTitle, promptTitle, resolveTitle } from './titles.js';
import {
  CONTINUE_PROMPT,
  type AgentModeState,
  type AgentWatchAnswer,
  type AppEnd,
  type PermissionMark,
  type PermissionMode,
  type ChatHostEvent,
  type ChatHostState,
  type ChatMode,
  type CompactOutcome,
  type ConversationContext,
  type ConversationEvent,
  type ConversationInfo,
  type CwdProblem,
  type GoalAction,
  type Interaction,
  type InteractionAnswerResult,
  type ModelThinking,
  type NeverRulesResponse,
  type QueueHold,
  type RewindMode,
  type RewindPlan,
  type SubagentMessageResult,
  type ThinkingLevel,
  type TurnOutcome,
  type WorkGit,
  type WorkSwitchReply,
} from './types.js';
import { checkWorkFolder, cwdError, realFolder, resolveTempRoot, type WorkFolderRules } from './work-folder.js';
import {
  BRANCHES_SHOWN,
  GIT_DEPS,
  gitEnv,
  orderBranches,
  overlapCount,
  readRepo,
  repoKey,
  resolveGit,
  switchBranch,
  type GitDeps,
  type GitRead,
  type SwitchRun,
} from './work-git.js';
import { makeScratch, removeIfEmpty, ScratchFolders, scratchRoot, stampName } from './work-scratch.js';
import { readWorkSettingsSync, type WorkSettings, writeWorkSettings } from './work-settings.js';

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

/**
 * The Context panel's answer from the agent's facts (null: not loaded, or the read failed) and the stored use. The
 * parts need every number; an agent whose session gave no answer sends zeros with no window, which is no parts.
 */
export function conversationContext(info: SessionInfoFacts | null, stored: number | null): ConversationContext {
  if (
    info === null ||
    info.total === null ||
    info.total === 0 ||
    info.systemPromptTokens === null ||
    info.toolDefinitionsCount === null ||
    info.toolDefinitionsTokens === null ||
    info.messageTokens === null ||
    info.autoCompactThresholdPercent === null ||
    info.compactionCount === null
  ) {
    return { used: stored, parts: null };
  }
  return {
    used: info.used ?? stored,
    parts: {
      systemTokens: info.systemPromptTokens,
      toolCount: info.toolDefinitionsCount,
      toolTokens: info.toolDefinitionsTokens,
      messageTokens: info.messageTokens,
      autoCompactPercent: info.autoCompactThresholdPercent,
      compactions: info.compactionCount,
    },
  };
}

const QUEUE_ROW_GONE = 'No such queued message.';
const STEER_WORK_ONLY = 'Send now is for Work conversations only.';
const GOAL_WORK_ONLY = 'A goal is for Work conversations only.';
const PDF_READ_WORK_ONLY = 'The PDF tool is for Work conversations only.';
const PDF_GONE = 'No such PDF.';
/** Ends of a turn after which the host starts a PDF read the model did not start. */
const HOST_START_OUTCOMES: ReadonlySet<TurnOutcome> = new Set(['completed', 'output-limit', 'tool-loop']);
const STEER_NOT_NOW = 'No answer takes it now. The message stays queued.';
const STEER_BUSY = 'The answer takes no message at this step.';
const STEER_LATE = 'The agent did not answer in time. The message stays queued.';
const REWIND_CHANGED = 'The conversation changed. Open Rewind again.';
const REWIND_NOT_POINT = 'This message is not a rewind point.';
const REWIND_GOAL = 'A goal started after this message. Clear it first.';
const REWIND_CHAT_FILES = 'Chat rewinds the conversation only.';
const FORK_NOT_POINT = 'This message is not a fork point.';
const FORK_CHANGED = 'The conversation changed. Pick the message again.';
const FORK_RUNNING = 'A fork of this conversation is already running.';
const RECAP_RUNNING = 'A recap is being written. Wait for it.';
/** Lines one new conversation keeps while its create is in flight. */
const EARLY_LINES_MAX = 64;

/** A rewind the page was shown: the commit runs it only while the conversation is still as it was read. */
interface RewindHeld {
  token: string;
  /** The page's entry, and the time of that entry: the first one the rewind removes. */
  entry: number;
  from: number;
  target: number;
  tasks: string[];
  schedules: string[];
  runs: string[];
  /** Work: the agent's dry run the page was shown, paths as the agent sent them. */
  files: Extract<RewindPreview, { files: unknown }>['files'];
}

/** What a message typed with `whenBusy: 'now'` became: sent to the running answer, queued, or a new answer. */
export type PromptReply = { turnId: string } | { queued: string; steer?: 'refused' | 'late' } | { steered: true };

/** A check that failed when the queue's next row was due: its fixed sentence and its reason code. */
function blockedHold(error: unknown): QueueHold {
  const api = routeError('chat', error);
  return { why: 'blocked', reason: api.message, code: api.reason ?? null };
}

/**
 * A rewind plan of `entry` on the conversation as it was read: a prompt, a line or a cut since then changes it. It
 * names the entry, so a commit never runs the plan another window made of another message.
 */
function rewindToken(conversation: Conversation, entry: number): string {
  return `${conversation.baseVersion}:${conversation.shownEntries().length}:${entry}`;
}

/** What the host holds live for a conversation, for a page's `sync`. */
/** The transcript lists the schedule and no removal of it. */
function listsSchedule(entries: readonly TranscriptEntry[], scheduleId: string): boolean {
  return entries.some(
    (entry) => entry.kind === 'tool' && entry.schedule?.id === scheduleId && entry.schedule.endTs === undefined,
  );
}

function hostView(host: AgentHost, id: string): HostView {
  return {
    turn: host.turnState(id),
    interactions: host.openInteractions(id),
    compaction: host.compaction(id),
    hookRun: host.hookRun(id),
    recap: host.recapState(id),
    runs: host.runs(id),
    live: host.liveActivity(id),
    commands: host.commandsOf(id),
  };
}

/** A temp root as people type it: its last spelling. */
function typedTop(root: AllowedRoot | undefined): string | null {
  return root === undefined ? null : (root.spellings[root.spellings.length - 1] ?? root.top);
}

function modeChanged(): ApiError {
  return new ApiError('E_CONFLICT', 'The permission mode could not be changed.', 'mode-changed');
}

/**
 * A message with pictures or PDF pages is refused once the chat's stored ones and its own pass `maxBase64`, or the
 * images the chat holds and its own pass {@link CHAT_IMAGES_MAX}.
 */
function checkPictureRoom(
  conversation: Conversation,
  attachments: readonly AttachmentInput[],
  maxBase64: number = CHAT_PICTURES_MAX_BASE64,
): void {
  const added = attachments.map(attachmentView);
  const images = imageCount(added);
  if (images === 0) return;
  if (storedPictureChars([...conversation.userAttachments(), ...added]) > maxBase64) {
    throw chatRouteError('pictures-full');
  }
  if (conversation.heldImages() + images > CHAT_IMAGES_MAX) throw chatRouteError('pictures-full');
}

/** The agent took the prompt and ran the turn to an end of its own: no Stop, no exit, no crash. */
function reachedAgent(result: TurnResult): boolean {
  const outcome = result.outcome;
  return (
    result.unsent !== true &&
    outcome !== undefined &&
    outcome !== 'cancelled' &&
    outcome !== 'interrupted' &&
    outcome !== 'agent-crashed' &&
    outcome !== 'no-model'
  );
}

function isDecisionModel(model: InferenceEndpoint['models'][number]): boolean {
  return model.kind === 'decision' || model.modelType === 'clairic' || model.modelType === 'clef';
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
  /** The size on disk of a model path. Default: the files on disk (every part of a split GGUF). */
  modelSize?: (path: string) => Promise<number | null>;
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
  /** Work: the Control Panel environment. */
  baseEnv?: NodeJS.ProcessEnv;
  /** Work: the login shell's `PATH` (`resolveLoginPath`), started with the host; none outside the app. */
  workLoginPath?: Promise<string | null>;
  /**
   * The agent home of Work (default `agentConfigDir()`): `GROK_HOME` of the Work agent, the home its sessions are read
   * from, cold renames and the Work folder rule.
   */
  workAgentHome?: string;
  homeDir?: string;
  /** Work: the temp roots Auto covers beside the Work folder. Default: `/tmp` when `resolveTempRoot` accepts it. */
  tempRoots?: readonly AllowedRoot[];
  username?: string;
  pageGoneMs?: number;
  needlesWaitMs?: number;
  timings?: Partial<HostTimings>;
  createBudgetMs?: number;
  flushMs?: number;
  stateCoalesceMs?: number;
  unloadMaxMs?: number;
  /** Override for the per-chat base64 picture cap (default {@link CHAT_PICTURES_MAX_BASE64}). */
  maxPicturesBase64?: number;
  now?: () => number;
  /** How earlier Control Panel lives ended (MAIN's ledger), carried in the host state. */
  appEnds?: Promise<readonly AppEnd[]>;
  /** Called when a turn starts or ends. */
  onBusyChange?: () => void;
  /** Tests: how the branch tray finds and runs git. */
  gitDeps?: Partial<GitDeps>;
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
  /** {@link workSessionDir}'s finds, by session id. */
  private readonly workDirs = new Map<string, string>();
  private readonly folderRules: WorkFolderRules;
  /** The temp roots Auto covers beside the Work folder. */
  private readonly tempRoots: readonly AllowedRoot[];
  private indexChain: Promise<void> = Promise.resolve();
  private pageGoneTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly conversations = new Map<string, Conversation>();
  private readonly resolving = new Map<string, Promise<Conversation | null>>();
  /** Conversations a delete holds: no prompt, rename or model change starts while the delete awaits. */
  private readonly deleting = new Set<string>();
  /**
   * Work: the turn whose message asked for plan mode while the agent was not in it, and the mode moves the
   * conversation had then. A turn that reached the agent and ends with no move: the agent has no plan mode.
   */
  private readonly planAsked = new Map<string, { turnId: string; moves: number }>();
  private readonly hostSinks = new Set<(event: ChatHostEvent) => void>();
  private readonly chatSkillOk: boolean;
  private install: ChatHostState['agent']['install'] = 'idle';
  private inferenceToken: string | null = null;
  private stateTimer: ReturnType<typeof setTimeout> | null = null;
  private liveOff: (() => void) | null = null;
  /** Sizes on disk by model path, for the engine process `sizesPid`: another process may see other files. */
  private readonly sizes = new Map<string, Promise<number | null>>();
  private sizesPid: number | null = null;
  /** The window per model from the last list an agent process gave; on disk, as no agent runs after an idle stop. */
  private readonly windows: Map<string, number>;
  private windowsWrite: Promise<void> = Promise.resolve();
  /** The thinking levels per model from the last list an agent process gave; memory only. */
  private readonly levels = new Map<string, ModelThinking | null>();
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
  private readonly settingsBase = new Map<string, SettingsPrints>();
  private readonly permissionLog: PermissionLog;
  /** Work: the never-allow rules of every Work folder, read once. */
  private readonly never: NeverRules;
  /** The queues that hold rows or a hold, by conversation; read from the composer store once per app run. */
  private readonly queues = new Map<string, ConversationQueue>();
  private readonly composerStore: ComposerStore;
  /** The last rewind plan per conversation, until its commit or the next plan. */
  private readonly rewindPlans = new Map<string, RewindHeld>();
  /** Conversations whose queue is sending its next row now (its checks run). */
  /** Conversations whose drain checks a row now; `again`: a drain was asked for meanwhile, `user` by Send next. */
  private readonly draining = new Map<string, { again: boolean; user: boolean }>();
  /** `create` calls in flight, and the lines of a new session the service does not hold yet, by session id. */
  private creates = 0;
  private readonly early = new Map<string, UpdateLine[]>();
  /** Moves up, and the record of every move, one at a time. */
  private permissionChain: Promise<unknown> = Promise.resolve();
  /** The mode picked on the New Work screen, before its conversation exists. */
  private readonly draft = new PermissionDraft();
  private readonly home: string;
  private readonly workHome: string;
  /** Where Work without a folder makes each conversation's own folder. */
  private readonly scratch: string;
  /** The folders Work without a folder made: only these are marked and removed. */
  private readonly scratchFolders: ScratchFolders;
  /** The folders a create or a fork is opening, with how many: a delete keeps them. */
  private readonly opening = new Map<string, number>();
  /** The folders of forks past their soft deadline, by origin: held until the late copy landed. */
  private readonly lateForkFolders = new Map<string, string[]>();
  /** Read once at start; a failed read is no ends. */
  private readonly appEnds: Promise<readonly AppEnd[]>;
  /** The app-wide Work settings: read once at start, then each save. */
  private workSettings: WorkSettings;
  /** Saves of the Work settings, one at a time. */
  private workSettingsChain: Promise<unknown> = Promise.resolve();
  private readonly gitDeps: GitDeps;
  /** The git binary found, kept for the app run; a miss is looked for again at the next read. */
  private gitBin: string | null = null;
  private gitFinding: Promise<string | null> | null = null;
  /** One chain of git commands per work tree (`repoKey`, folded): reads and switches never overlap. */
  private readonly gitChains = new Map<string, Promise<unknown>>();
  /** The work tree tops where `git switch` runs now: no Work conversation starts or sends in them. */
  private readonly switching: string[] = [];
  /** The page files of the PDF tool while the page sends them. */
  private readonly pdfStore: PdfStore;
  /** Work, before its first agent start: whether no file of the person's holds a PDF tool name (read once). */
  private pdfToolChecked: Promise<boolean> | null = null;
  /**
   * The host's own start of a PDF read, the newest by conversation, from the turn `turnId` that sent the PDF: the turn
   * it started, null when it started none.
   */
  private readonly hostStarts = new Map<
    string,
    { pdf: string; turnId: string; started: Promise<{ turnId: string } | null> }
  >();

  constructor(options: ChatServiceOptions) {
    this.opts = options;
    this.gitDeps = { ...GIT_DEPS, ...options.gitDeps };
    this.windows = readModelWindowsSync(options.paths.modelWindows);
    this.workSettings = readWorkSettingsSync(options.paths.workSettings);
    this.appEnds = (options.appEnds ?? Promise.resolve([])).catch(() => []);
    this.now = options.now ?? Date.now;
    this.pdfStore = new PdfStore(options.paths.pdfStage, this.now);
    void this.pdfStore.sweep().catch(() => {});
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
          conversation.resync(hostView(host, id));
        }
      },
      applyPending: () =>
        (['chat', 'work'] as const).some((mode) => {
          const state = this.hosts[mode].modeState();
          return state.stale && (state.turnsInFlight > 0 || state.background > 0 || state.tasks > 0);
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
    this.scratch = scratchRoot(home);
    this.scratchFolders = new ScratchFolders(options.paths.scratchFolders);
    this.permissionLog = new PermissionLog(options.paths.permissionLog);
    this.never = new NeverRules(options.paths.neverAllow, { now: this.now });
    this.composerStore = new ComposerStore(options.paths.composer);
    const appTemps = [tmpdir(), (options.baseEnv ?? process.env).TMPDIR].filter(
      (t): t is string => t !== undefined && t !== '',
    );
    this.tempRoots = options.tempRoots ?? resolveTempRoot('/tmp', home, appTemps);
    this.workReader = new AgentSessionsReader({ agentHome: workHome });
    this.folderRules = {
      home,
      refused: [
        options.paths.root,
        join(home, '.mlx-node'),
        workHome,
        join(home, '.mlx-agent'),
        join(home, 'Library', 'Keychains'),
      ],
    };
    const common = {
      paths: options.paths,
      agentHome: workHome,
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
        loginPath: options.workLoginPath,
        claudeCode: () => this.workSettings.claudeCode,
        homeDir: options.homeDir,
        tempRoots: this.tempRoots,
        refused: this.folderRules.refused.map((f) => realFolder(f)),
        checkCwd: (path) => checkWorkFolder(path, this.folderRules),
        permission: { modeOf: (id) => this.modeOf(id), watched: () => this.hostSinks.size > 0 },
        never: this.never,
        sessionDir: (id) => this.workSessionDir(id),
        pdfFacts: (sessionId, pdf) => this.pdfFacts(sessionId, pdf),
        listener: this.listenerFor('work'),
      }),
    };
  }

  /** Work: `childId` is an agent the conversation started: in the running process, or by its stored link. */
  private ownsChild(id: string, childId: string): boolean {
    return this.hosts.work.ownsChild(id, childId) || this.workReader.scan().parentOf.get(childId)?.parentId === id;
  }

  /** Where the Work agent stores a session, as its reader finds it; kept once found, since it never moves. */
  private workSessionDir(id: string): string | null {
    const kept = this.workDirs.get(id);
    if (kept !== undefined) return kept;
    const dir = this.workReader.scan().sessions.get(id)?.dir ?? null;
    if (dir !== null) this.workDirs.set(id, dir);
    return dir;
  }

  // ── routes ───────────────────────────────────────────────────────────────────────────────────────────────────────

  async state(): Promise<ChatHostState> {
    return this.computeState();
  }

  /**
   * The conversations with a turn running, both modes, wake turns, workflow runs, helpers, commands and monitors
   * included: quitting stops them. Synchronous: no worker, no fs.
   */
  busyConversations(): string[] {
    const out = new Set<string>();
    for (const mode of ['chat', 'work'] as const) {
      for (const a of this.hosts[mode].active()) if (a.busy || a.background || a.tasks > 0) out.add(a.sessionId);
    }
    return [...out];
  }

  /** {@link busyConversations} less the turns already stopping for another cause (a Stop, a cap): a close keeps it. */
  closingConversations(): string[] {
    const out = new Set<string>();
    for (const mode of ['chat', 'work'] as const) {
      for (const a of this.hosts[mode].active()) if ((a.busy && !a.stopping) || a.background) out.add(a.sessionId);
    }
    return [...out];
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
    // Before the session exists: a refused command must not leave a conversation behind.
    if (checkCommand(mode, input.text, null) === 'refused') throw chatRouteError('slash-text');
    const none = mode === 'work' && input.noFolder === true;
    let cwd: string | undefined;
    if (mode === 'work' && !none) {
      const folder = await checkWorkFolder(input.cwd ?? '', this.folderRules);
      if (!folder.ok) throw cwdError(folder.problem);
      cwd = folder.path;
      this.hold(cwd);
    } else if (mode === 'chat') {
      await this.assertChatReady();
    }
    const slow: { settled: Promise<void> | null } = { settled: null };
    try {
      const endpoint = await this.endpoint();
      if (endpoint === null) throw chatRouteError('inference-stopped');
      const modelId = input.modelId ?? (await this.state()).defaultModelId ?? undefined;
      if (modelId === undefined) throw chatRouteError('no-models');
      await this.checkImages(mode, modelId, input.attachments);
      for (const item of input.attachments ?? []) {
        if (isToolPdf(item) && !this.pdfStore.complete(item.stage, item.pages)) {
          throw chatRouteError('attachment-refused');
        }
      }
      if (none) {
        cwd = await this.newScratch();
        this.hold(cwd);
      }
      this.creates += 1;
      try {
        return await this.createIn(mode, input, modelId, cwd, (settled) => {
          slow.settled = settled;
        });
      } finally {
        this.creates -= 1;
        if (this.creates === 0) this.early.clear();
      }
    } finally {
      if (cwd !== undefined) {
        const folder = cwd;
        // A session that answers past the budget may still start in the folder: it is kept until that settles.
        if (slow.settled === null) await this.leaveFolder(folder, none);
        else void slow.settled.then(() => this.leaveFolder(folder, none)).catch(() => {});
      }
    }
  }

  /**
   * A new empty folder under the scratch root, recorded before the agent gets it and checked by the folder rule; it
   * is removed when the record cannot be written or the rule refuses it.
   */
  private async newScratch(): Promise<string> {
    const made = await makeScratch(this.scratch, this.now());
    if (made === null) throw folderNotMadeError();
    try {
      await this.scratchFolders.add(made);
    } catch {
      await removeIfEmpty(made);
      throw folderNotMadeError();
    }
    const folder = await checkWorkFolder(made, this.folderRules);
    if (!folder.ok) {
      await this.removeScratch(made);
      throw cwdError(folder.problem);
    }
    return folder.path;
  }

  private hold(cwd: string): void {
    this.opening.set(cwd, (this.opening.get(cwd) ?? 0) + 1);
  }

  private release(cwd: string): void {
    const left = (this.opening.get(cwd) ?? 0) - 1;
    if (left > 0) this.opening.set(cwd, left);
    else this.opening.delete(cwd);
  }

  /**
   * The end of a create's hold on its folder. A folder made for it that no conversation has is removed when empty:
   * once the conversation is stored it is its folder for life.
   */
  private async leaveFolder(cwd: string, made: boolean): Promise<void> {
    this.release(cwd);
    if (made && !this.cwdInUse(cwd)) await this.removeScratch(cwd);
  }

  /** Removes a folder the app made when it is empty, and its record with it. */
  private async removeScratch(cwd: string): Promise<void> {
    if (await removeIfEmpty(cwd)) await this.scratchFolders.drop(cwd);
  }

  /** A conversation this service holds runs in `cwd`. */
  private holdsCwd(cwd: string, except?: string): boolean {
    for (const c of this.conversations.values()) {
      if (c.info.mode === 'work' && c.info.cwd === cwd && c.info.id !== except) return true;
    }
    return false;
  }

  /** A Work folder the app made for a conversation started without a folder, and still the folder it made. */
  private isScratchFolder(cwd: string): boolean {
    return this.scratchFolders.owns(cwd);
  }

  /** Where a none pick is read before its folder exists: a child of the scratch root that is not made. */
  private scratchStandIn(): string {
    return join(realFolder(this.scratch), stampName(this.now()));
  }

  /** The lines of a session the service does not hold yet, while a create is in flight: the newest are dropped. */
  private keepEarly(id: string, line: UpdateLine): void {
    const kept = this.early.get(id) ?? [];
    if (kept.length >= EARLY_LINES_MAX) return;
    kept.push(line);
    this.early.set(id, kept);
  }

  /** `onSlow`: past the budget, the late session's end (its delete included) is handed there before `agent-slow`. */
  private async createIn(
    mode: ChatMode,
    input: CreateConversationInput,
    modelId: string,
    cwd: string | undefined,
    onSlow: (settled: Promise<void>) => void,
  ): Promise<{ conversation: ConversationInfo; turnId: string }> {
    const host = this.hosts[mode];
    if (cwd !== undefined) this.assertNotSwitching(cwd);
    const created = host.createSession(cwd === undefined ? { modelId } : { modelId, cwd });
    let timer: ReturnType<typeof setTimeout> | null = null;
    const budget = new Promise<'slow'>((resolve) => {
      timer = setTimeout(() => resolve('slow'), this.opts.createBudgetMs ?? CREATE_BUDGET_MS);
    });
    let session;
    try {
      const raced = await Promise.race([created, budget]);
      if (raced === 'slow') {
        onSlow(
          created.then(
            (late) => host.deleteSession(late.sessionId, late.cwd).catch(() => {}),
            () => {},
          ),
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
      await this.never.ready();
    }
    const rules = mode === 'work' ? this.never.rulesOf(session.cwd).length : 0;
    const thinking = this.levelOn(input.thinking, session.modelId);
    if (thinking !== undefined) this.composerStore.setThinking(session.sessionId, thinking).catch(() => {});
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
      ...(rules > 0 ? { neverRules: rules } : {}),
      ...(thinking === undefined ? {} : { thinking }),
    });
    conversation.marksLoaded = true;
    conversation.seed([], [], null);
    this.conversations.set(conversation.info.id, conversation);
    // Lines the agent sent before the service held the conversation (a `session_start` hook's run), in order.
    for (const line of this.early.get(session.sessionId) ?? []) conversation.applyLine(line);
    this.early.delete(session.sessionId);
    const place = input.noFolder === true ? null : cwd;
    const grant =
      mode === 'work' && place !== undefined ? this.draft.claim(input.draft, place, session.sessionId) : null;
    if (input.draft !== undefined) this.scheduleState();
    if (grant !== null) await this.applyDraft(conversation, grant);
    let turnId: string;
    try {
      await this.placePdfs(conversation, input.attachments ?? []);
      turnId = this.startTurn(conversation, input.text, undefined, input.attachments, input.plan).turnId;
    } catch (error) {
      if (this.modeOf(session.sessionId) !== 'ask') this.lower(conversation, 'ask', 'user');
      throw error;
    }
    this.emitHost({ type: 'conversations-changed' });
    return { conversation: conversation.publicInfo(), turnId };
  }

  /**
   * The mode picked on the New screen, for the conversation just made from it: the same move up as the conversation's
   * own route, before its first turn. A failure leaves the conversation in Ask; a mode line it wrote is closed.
   */
  private async applyDraft(conversation: Conversation, grant: 'auto' | 'bypass'): Promise<void> {
    const id = conversation.info.id;
    const epoch = this.epochOf(id);
    const temp = grant === 'auto' ? this.tempOf(conversation) : null;
    const run = (): Promise<{ mode: PermissionMode }> => this.raise(id, grant, epoch, temp);
    const next = this.permissionChain.then(run, run);
    this.permissionChain = next.catch(() => {});
    try {
      await next;
    } catch {
      const open = this.conversations.get(id) === conversation;
      if (open && this.modeOf(id) === 'ask' && (this.logged.get(id) ?? 'ask') !== 'ask') {
        this.lower(conversation, 'ask', 'user');
      }
    }
  }

  /**
   * The mode picked on the New Work screen, kept for `draft` and its folder until the first message makes the
   * conversation (`create` applies it there). The same laws as a conversation's mode: only a confirm's `up` moves up,
   * a move down is applied in this tick and calls back a move up that waits. Once the draft made its conversation, a
   * move down goes to that conversation and a move up is refused.
   */
  async setDraftPermissionMode(
    draft: string,
    target: PermissionMode,
    up: boolean,
    cwd: string | null,
    temp: string | null,
  ): Promise<{ mode: PermissionMode }> {
    this.assertOpen();
    const spent = this.draft.spent(draft);
    if (spent !== null) {
      if (up) throw modeChanged();
      this.permissionEpoch.set(spent, this.epochOf(spent) + 1);
      const conversation = this.conversations.get(spent);
      if (conversation !== undefined && MODE_RANK[target] < MODE_RANK[this.modeOf(spent)]) {
        this.lower(conversation, target, 'user');
      }
      return { mode: this.modeOf(spent) };
    }
    this.draft.epoch += 1;
    const epoch = this.draft.epoch;
    const mode = this.draft.modeOf(draft);
    if (MODE_RANK[target] < MODE_RANK[mode]) {
      this.draft.lower(draft, target === 'auto' ? 'auto' : 'ask');
      this.scheduleState();
      return { mode: target };
    }
    if (target === mode) return { mode };
    if (!up) throw modeChanged();
    let path: string | null = null;
    if (cwd !== null) {
      const folder = await checkWorkFolder(cwd, this.folderRules);
      if (!folder.ok) throw cwdError(folder.problem);
      path = folder.path;
    }
    this.assertOpen();
    if (this.draft.epoch !== epoch) throw modeChanged();
    if (target === 'auto') {
      const place = path ?? this.scratchStandIn();
      const covered = typedTop(tempRoots({ cwd: place, home: this.home, temp: this.tempRoots })[0]);
      if (covered !== null && temp !== covered) {
        throw new ApiError('E_CONFLICT', 'The Auto text did not name the temp folder.', 'temp-unnamed');
      }
    }
    this.draft.set(draft, target === 'auto' ? 'auto' : 'bypass', path);
    this.scheduleState();
    return { mode: target };
  }

  getWorkSettings(): WorkSettings {
    return { ...this.workSettings };
  }

  /**
   * Saves the Work settings. A change marks the Work host stale: an idle agent stops at once, a running answer keeps
   * the agent it started with, and the next start reads the new value. Chat is not touched.
   */
  async setWorkSettings(next: WorkSettings): Promise<WorkSettings> {
    this.assertOpen();
    const run = this.workSettingsChain.then(async () => {
      if (next.claudeCode !== this.workSettings.claudeCode) {
        await writeWorkSettings(this.opts.paths.workSettings, next);
        this.workSettings = { ...next };
        this.hosts.work.markStale();
      }
      return { ...this.workSettings };
    });
    this.workSettingsChain = run.catch(() => {});
    return run;
  }

  /** Whether the user has tool hooks for a Work folder, or (null) for a folder of the scratch root; a read error says yes. */
  async workHooks(cwd: string | null): Promise<{ userHooks: boolean }> {
    let path = this.scratchStandIn();
    if (cwd !== null) {
      const folder = await checkWorkFolder(cwd, this.folderRules);
      if (!folder.ok) throw cwdError(folder.problem);
      path = folder.path;
    }
    const place: SettingsPlace = { home: this.home, agentHome: this.workHome, cwd: path };
    return { userHooks: await findUserHooks(place).catch(() => true) };
  }

  /** The New screen's branch tray: the work tree of a Work folder, its branches and its uncommitted change count. */
  async workGit(cwd: string): Promise<WorkGit> {
    const folder = await checkWorkFolder(cwd, this.folderRules);
    if (!folder.ok) throw cwdError(folder.problem);
    const key = await repoKey(folder.path);
    return this.inGitChain(key, async () => {
      const env = await this.gitEnv();
      const bin = await this.gitBinary(env);
      if (bin === null) return { kind: 'no-git' };
      return this.gitView(await readRepo(bin, folder.path, env, this.gitDeps));
    });
  }

  /**
   * Switches the folder's work tree to a local branch the read lists, in the work tree's chain: the folder rule, then
   * no working Work conversation and no create in it, then the list and the change count again; a folder with
   * uncommitted changes, or whose count is unknown, never switches. Git's refusal is an outcome, not an error.
   */
  async switchWorkBranch(input: { cwd: string; branch: string }): Promise<WorkSwitchReply> {
    const folder = await checkWorkFolder(input.cwd, this.folderRules);
    if (!folder.ok) throw cwdError(folder.problem);
    const key = await repoKey(folder.path);
    return this.inGitChain(key, async () => {
      if (overlapCount(key, this.busyFolders()) > 0) throw workGitError('folder-busy');
      const env = await this.gitEnv();
      const bin = await this.gitBinary(env);
      if (bin === null) throw workGitError('git-unavailable');
      const read = await readRepo(bin, folder.path, env, this.gitDeps);
      if (read.kind !== 'repo') throw workGitError('git-unavailable');
      if (overlapCount(read.top, this.busyFolders()) > 0) throw workGitError('folder-busy');
      if (!read.refs.some((ref) => ref.name === input.branch)) throw workGitError('branch-unknown');
      if ('branch' in read.head && read.head.branch === input.branch) {
        return { outcome: 'switched', git: this.gitView(read) };
      }
      if (read.changes === null) throw workGitError('status-unknown');
      if (read.changes > 0) throw workGitError('dirty');
      this.switching.push(read.top);
      let run: SwitchRun;
      try {
        run = await switchBranch(bin, read.top, input.branch, env, this.gitDeps);
      } finally {
        this.switching.splice(this.switching.indexOf(read.top), 1);
      }
      const after = this.gitView(await readRepo(bin, folder.path, env, this.gitDeps));
      return { ...run, git: after };
    });
  }

  private async gitEnv(): Promise<Record<string, string>> {
    const loginPath = (await this.opts.workLoginPath) ?? null;
    return gitEnv(this.opts.baseEnv ?? process.env, this.home, loginPath, this.gitDeps.platform);
  }

  private async gitBinary(env: Readonly<Record<string, string>>): Promise<string | null> {
    if (this.gitBin !== null) return this.gitBin;
    const finding = (this.gitFinding ??= resolveGit(env, this.gitDeps).finally(() => {
      if (this.gitFinding === finding) this.gitFinding = null;
    }));
    const found = await finding;
    if (found !== null) this.gitBin = found;
    return found;
  }

  /**
   * The folders a branch switch must leave alone: each working Work conversation's, and each a create or a fork holds
   * (the agent opens it before its session counts as working).
   */
  private busyFolders(): string[] {
    return [...this.hosts.work.workingFolders(), ...this.opening.keys()];
  }

  /** Work: refused while `git switch` runs in a work tree that holds `cwd` or lies in it. */
  private assertNotSwitching(cwd: string): void {
    if (overlapCount(cwd, this.switching) > 0) throw workGitError('folder-busy');
  }

  private inGitChain<T>(folder: string, run: () => Promise<T>): Promise<T> {
    const key = folderKey(folder);
    const next = (this.gitChains.get(key) ?? Promise.resolve()).then(run, run);
    const tail = next.then(
      () => undefined,
      () => undefined,
    );
    this.gitChains.set(key, tail);
    void tail.then(() => {
      if (this.gitChains.get(key) === tail) this.gitChains.delete(key);
    });
    return next;
  }

  private gitView(read: GitRead): WorkGit {
    if (read.kind !== 'repo') return { kind: read.kind };
    const current = 'branch' in read.head ? read.head.branch : null;
    const ordered = orderBranches(read.refs, current);
    return {
      kind: 'repo',
      top: read.top,
      head: read.head,
      branches: ordered.slice(0, BRANCHES_SHOWN).map((ref) => ({
        name: ref.name,
        at: ref.at,
        current: ref.name === current,
        elsewhere: ref.elsewhere,
      })),
      total: read.refs.length,
      busy: overlapCount(read.top, this.hosts.work.workingFolders()),
      changes: read.changes,
    };
  }

  /**
   * `plan` (Work only): the message asks for plan mode (`true`) or leaves it (`false`). `whenBusy: 'queue'`: while
   * an answer, a wake, a history op or a recap runs the message joins the queue (`{queued}`) instead of the busy
   * refusal; an idle conversation starts the turn as without it, and a held queue stays held under it. One queued
   * while a history op or a recap runs goes when that ends, as it would have gone at once, though older rows are held
   * (`QueueRow.unheld`). `whenBusy:
   * 'now'` (Work, text only, ⌘Enter): while an answer runs the text goes to it as a Send now (`{steered}`); a refused
   * or late one joins the queue and says so (`steer`).
   */
  async prompt(
    id: string,
    text: string,
    attachments?: readonly AttachmentInput[],
    plan?: boolean,
  ): Promise<{ turnId: string }>;
  async prompt(
    id: string,
    text: string,
    attachments: readonly AttachmentInput[] | undefined,
    plan: boolean | undefined,
    whenBusy: 'queue' | 'now',
  ): Promise<PromptReply>;
  async prompt(
    id: string,
    text: string,
    attachments: readonly AttachmentInput[] = [],
    plan?: boolean,
    whenBusy?: 'queue' | 'now',
  ): Promise<PromptReply> {
    this.assertOpen();
    const conversation = await this.known(id);
    const mode = conversation.info.mode;
    if (plan !== undefined && mode !== 'work') throw ApiError.badRequest(PLAN_WORK_ONLY);
    if (whenBusy === 'now' && mode !== 'work') throw ApiError.conflict(STEER_WORK_ONLY, 'steer-work-only');
    const host = this.hosts[mode];
    await this.checkSlash(conversation, text);
    const cwd = await this.sendChecks(conversation, attachments, whenBusy === undefined);
    // A typed message waits for a recap being written; without `whenBusy` it starts, and the agent drops the recap.
    if (whenBusy !== undefined && host.recapState(id) !== null) {
      return this.enqueue(conversation, text, attachments, plan);
    }
    if (host.isBusy(id) || (whenBusy === 'queue' && host.goalRuns(id))) {
      if (whenBusy === undefined) throw chatRouteError('conversation-busy');
      if (whenBusy === 'queue' || attachments.length > 0 || isSlashText(text)) {
        return this.enqueue(conversation, text, attachments, plan);
      }
      const steered = await this.steerText(conversation, text, plan);
      if (steered !== null) return steered;
    }
    try {
      return { turnId: this.startTurn(conversation, text, cwd, attachments, plan).turnId };
    } catch (error) {
      if (whenBusy !== undefined && host.isBusy(id)) return this.enqueue(conversation, text, attachments, plan);
      throw error;
    }
  }

  /**
   * The allow rule for slash text (`commands.ts`), refused as `slash-text` before the message is sent or queued. A
   * name only a listed skill or workflow allows waits for the list the pull after a load brings; with no list (a
   * cold conversation) the host checks it once the session is loaded.
   */
  private async checkSlash(conversation: Conversation, text: string): Promise<void> {
    const mode = conversation.info.mode;
    if (checkCommand(mode, text, null) === 'plain') return;
    const rows = await this.hosts[mode].agentCommands(conversation.info.id);
    if (checkCommand(mode, text, rows) === 'refused') throw chatRouteError('slash-text');
  }

  /**
   * What a message must pass before its turn starts, sent by the user or by the queue: pictures, the Work folder or
   * the Chat skill, the stored transcript, the picture room, a fresh read of the engine, last no branch switch in the
   * Work folder. `busyFails`: a busy conversation is refused before the transcript is read. Returns the Work folder.
   */
  private async sendChecks(
    conversation: Conversation,
    attachments: readonly AttachmentInput[],
    busyFails: boolean,
  ): Promise<string | undefined> {
    const mode = conversation.info.mode;
    await this.checkImages(mode, conversation.info.modelId, attachments);
    let cwd: string | undefined;
    if (mode === 'work') {
      const problem = await this.folderState(conversation);
      if (problem !== null) throw cwdError(problem);
      cwd = conversation.info.cwd ?? undefined;
    } else {
      await this.assertChatReady();
    }
    const host = this.hosts[mode];
    if (busyFails && host.isBusy(conversation.info.id)) throw chatRouteError('conversation-busy');
    if (!conversation.hasBase()) await this.loadBase(conversation);
    checkPictureRoom(conversation, attachments, this.opts.maxPicturesBase64 ?? CHAT_PICTURES_MAX_BASE64);
    if (busyFails && host.isBusy(conversation.info.id)) throw chatRouteError('conversation-busy');
    await this.endpoint();
    if (cwd !== undefined) this.assertNotSwitching(cwd);
    await this.placePdfs(conversation, attachments);
    return cwd;
  }

  /**
   * Moves the stage of each PDF for the tool to the conversation's page folders; a stage already moved for it stays
   * there. A stage that is not complete, or is another conversation's, is `attachment-refused`.
   */
  private async placePdfs(conversation: Conversation, attachments: readonly AttachmentInput[]): Promise<void> {
    for (const item of attachments) {
      if (!isToolPdf(item)) continue;
      const id = await this.pdfStore.place(
        item.stage,
        item.pages,
        this.opts.paths.workPdf,
        conversation.info.id,
        item.name,
      );
      if (id === null) throw chatRouteError('attachment-refused');
    }
  }

  /**
   * A picture goes only to a Chat conversation, a PDF to either mode (one for the PDF tool to Work while the tool is
   * there), and both only when the model row takes images
   * (`images`, as the host state lists it). The page blocks it first; this settles a model change that landed before
   * this check. A change still in flight is caught by the engine's 400 and the agent's retry without the images.
   */
  private async checkImages(
    mode: ChatMode,
    modelId: string,
    attachments: readonly AttachmentInput[] = [],
  ): Promise<void> {
    const pictures = attachments.some((a) => a.kind === 'image');
    if (!pictures && !attachments.some((a) => a.kind === 'pdf')) return;
    if (pictures && mode === 'work') throw chatRouteError('attachment-refused');
    if (attachments.some(isToolPdf) && (mode !== 'work' || !(await this.pdfToolWork()))) {
      throw chatRouteError('attachment-refused');
    }
    const row = (await this.state()).models.find((m) => m.id === modelId);
    if (row?.images !== true) throw chatRouteError('attachment-refused');
  }

  /**
   * Work: a press on the goal strip. Pause while an answer, a wake or a history op runs is Stop (the agent pauses the
   * goal itself); else each is the agent's own `/goal pause|resume|clear`, sent through the checks of a typed message
   * (the page never sends that text). Resume and Clear while the conversation is busy: 409 `conversation-busy`.
   */
  async goal(id: string, action: GoalAction): Promise<{ turnId: string } | { stopped: true }> {
    this.assertOpen();
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.badRequest(GOAL_WORK_ONLY);
    if (action === 'pause' && this.hosts.work.isBusy(id)) {
      await this.cancel(id);
      return { stopped: true };
    }
    return this.prompt(id, `/goal ${action}`);
  }

  /**
   * A page of a PDF for the PDF tool, sent as the page draws it. `{stage}`: the stage that holds it. A page out of
   * order or of another count is `attachment-refused`.
   */
  async stagePdf(input: PdfStageInput): Promise<{ stage: string }> {
    this.assertOpen();
    let stage: string | null;
    try {
      stage = await this.pdfStore.stagePage(input.stage, input.page, input.pages, input.data);
    } catch {
      throw new ApiError('E_UNAVAILABLE', 'The page could not be saved.', 'pdf-save');
    }
    if (stage === null) throw chatRouteError('attachment-refused');
    return { stage };
  }

  /** A PDF's chip was removed before Send: its stage goes. */
  async dropPdfStage(stage: string): Promise<Record<string, never>> {
    await this.pdfStore.drop(stage);
    return {};
  }

  /**
   * Work: Read it now. The agent starts `read-pdf` on the PDF `pdf` of a message of this conversation by its own
   * `/workflow` command, with the message's text as the question: no model call, no permission request. 404 for a PDF
   * no message of the conversation sent to the tool; `conversation-busy` while it works or a run of the tool is not
   * over. Once the turn that sent the PDF ended, while the host decides its own start and while the turn of that start
   * runs, the answer is that turn.
   */
  async readPdf(id: string, pdf: string): Promise<{ turnId: string }> {
    this.assertOpen();
    const mark = this.hostStarts.get(id);
    if (mark?.pdf === pdf && this.hosts.work.turnState(id)?.turnId !== mark.turnId) {
      const started = await mark.started;
      if (started !== null && this.hosts.work.turnState(id)?.turnId === started.turnId) return started;
    }
    return this.startPdfRead(id, pdf, 'you');
  }

  private async startPdfRead(id: string, pdf: string, by: PdfStartBy): Promise<{ turnId: string }> {
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.badRequest(PDF_READ_WORK_ONLY);
    if (!conversation.hasBase()) await this.loadBase(conversation);
    const sent = conversation.toolPdf(pdf);
    const facts = sent === null ? null : await this.pdfFacts(id, pdf);
    if (sent === null || facts === null || facts.session !== id) throw ApiError.notFound(PDF_GONE);
    if (!facts.images || !facts.tool) throw chatRouteError('attachment-refused');
    const text = readPdfCommand(pdf, sent.text, sent.name, by);
    await this.checkSlash(conversation, text);
    const cwd = await this.sendChecks(conversation, [], true);
    // Checked in the tick that starts the turn: a run that started, or a move down to Ask, during the awaits counts.
    if (this.hosts.work.runs(id).some((r) => r.name === READ_PDF_WORKFLOW && (r.live || r.waiting))) {
      throw chatRouteError('conversation-busy');
    }
    if (by === 'app' && this.modeOf(id) === 'ask') throw chatRouteError('conversation-busy');
    return { turnId: this.startTurn(conversation, text, cwd).turnId };
  }

  /** {@link hostStartPdf} once the turn `turn` that sent `pdf` ends, kept in {@link hostStarts} from its start. */
  private hostStart(
    conversation: Conversation,
    turn: { turnId: string; done: Promise<TurnResult> },
    pdf: string,
  ): void {
    const id = conversation.info.id;
    const started = turn.done.then((end) => this.hostStartPdf(conversation, end.outcome, pdf)).catch(() => null);
    const mark = { pdf, turnId: turn.turnId, started };
    this.hostStarts.set(id, mark);
    void started.then((made) => {
      if (made === null && this.hostStarts.get(id) === mark) this.hostStarts.delete(id);
    });
  }

  /**
   * Work in Auto or Bypass, a model of {@link HOST_START_FAMILIES}: the turn that sent the PDF `pdf` to the tool ended
   * and no `read-pdf` call of the model named it, whatever its answer, so the host starts it as Read it now does, in a
   * fork too (the PDF was sent there). Not in Ask, not after a Stop, a deny or a failed turn, and not while another turn
   * or a run of the tool goes on (Read it now stays offered).
   */
  private async hostStartPdf(
    conversation: Conversation,
    outcome: TurnOutcome | undefined,
    pdf: string,
  ): Promise<{ turnId: string } | null> {
    if (outcome === undefined || !HOST_START_OUTCOMES.has(outcome)) return null;
    const id = conversation.info.id;
    if (this.conversations.get(id) !== conversation) return null;
    if (this.modeOf(id) === 'ask') return null;
    const family = (await this.state()).models.find((m) => m.id === conversation.info.modelId)?.family;
    if (family === null || family === undefined || !HOST_START_FAMILIES.has(family)) return null;
    if (conversation.pdfCalled(pdf) !== false) return null;
    return this.startPdfRead(id, pdf, 'app');
  }

  /** Work: the PDF tool is there, by the last agent start, else by the files in the agent home. */
  private pdfToolWork(): Promise<boolean> {
    const started = this.hosts.work.pdfTool();
    if (started !== null) return Promise.resolve(started);
    this.pdfToolChecked ??= pdfToolBlocked(this.workHome).then(
      (blocked) => !blocked,
      () => false,
    );
    return this.pdfToolChecked;
  }

  /**
   * Work: the page folder `pdf` names, whether the model of conversation `sessionId` sees images, and whether the
   * agent finds the app's PDF tool by its names there.
   */
  private async pdfFacts(sessionId: string, pdf: string): Promise<PdfFacts | null> {
    const meta = await readPdfMeta(this.opts.paths.workPdf, pdf);
    const conversation = this.conversations.get(sessionId);
    if (meta === null || conversation === undefined) return null;
    const modelId = conversation.info.modelId;
    const images = (await this.state()).models.find((m) => m.id === modelId)?.images === true;
    const cwd = conversation.info.cwd;
    const tool =
      cwd !== null &&
      (await this.pdfToolWork()) &&
      !(await pdfToolShadowed(cwd, this.workHome, this.home).catch(() => true));
    return { id: pdf, session: meta.session, name: meta.name, pages: meta.pages, images, tool };
  }

  /** The user's Stop: the queue is held, so nothing runs on by itself after it. */
  async cancel(id: string): Promise<{ cancelled: boolean }> {
    const conversation = await this.known(id);
    const cancelled = this.hosts[conversation.info.mode].cancel(id);
    if (cancelled) this.holdQueue(conversation, { why: 'stopped' });
    return { cancelled };
  }

  /**
   * Work: a queued row's Send now. The row leaves the list once the agent took it, and its text shows in the running
   * answer. 409 while no prompt is open, the row stays queued: `steer-not-now` when no answer runs (the list sends
   * the row by itself), `steer-busy` when the answer runs at a step that takes no message (its start, the gap of an
   * automatic continue, a history op, the agent's refusal), `steer-late` while an earlier Send now of the row waits for
   * its reply. 503 `steer-late` when the agent did not answer in time: the row stays, the list sends no row meanwhile,
   * and a late yes still moves it into the answer.
   */
  async queueSend(id: string, rowId: string): Promise<{ steered: true }> {
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.conflict(STEER_WORK_ONLY, 'steer-work-only');
    const queue = this.queues.get(id);
    const row = queue?.row(rowId);
    if (queue === undefined || row === undefined) throw ApiError.notFound(QUEUE_ROW_GONE);
    if (row.attachments.length > 0)
      throw ApiError.conflict('A message with files runs as its own turn.', 'steer-files');
    if (isSlashText(row.text)) throw ApiError.conflict('A command runs as its own turn.', 'steer-slash');
    if (queue.isSteering(rowId)) throw ApiError.conflict(STEER_LATE, 'steer-late');
    const text = row.text;
    queue.setSteering(rowId, true);
    const outcome = await this.hosts.work.steer(id, text);
    if (outcome === 'sent') {
      this.steered(conversation, queue, rowId, text);
      return { steered: true };
    }
    if (outcome === 'not-now') {
      queue.setSteering(rowId, false);
      const answering = this.hosts.work.isBusy(id);
      void this.drainQueue(id, false);
      if (answering) throw ApiError.conflict(STEER_BUSY, 'steer-busy');
      throw ApiError.conflict(STEER_NOT_NOW, 'steer-not-now');
    }
    this.lateSteer(conversation, queue, rowId, text, outcome.late);
    throw new ApiError('E_UNAVAILABLE', STEER_LATE, 'steer-late');
  }

  /** A row's new text (its edit ends), or a page opening or closing it for edit. 404 for a row that is gone. */
  async queueEdit(id: string, rowId: string, input: QueuePatchInput): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const queue = this.queues.get(id);
    const row = queue?.row(rowId);
    if (queue === undefined || row === undefined) throw ApiError.notFound(QUEUE_ROW_GONE);
    if ('text' in input) {
      checkAttachments(row.attachments, input.text);
      queue.edit(rowId, input.text);
    } else {
      queue.setEditing(rowId, input.editing);
    }
    this.queueChanged(conversation, queue);
    return {};
  }

  async queueRemove(id: string, rowId: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const queue = this.queues.get(id);
    if (queue === undefined || !queue.remove(rowId)) throw ApiError.notFound(QUEUE_ROW_GONE);
    this.queueChanged(conversation, queue);
    return {};
  }

  async queueClear(id: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const queue = this.queues.get(id);
    if (queue === undefined || (queue.size === 0 && queue.held === null)) return {};
    queue.clear();
    this.queueChanged(conversation, queue);
    return {};
  }

  /**
   * Send next: the hold ends and the first row goes now when the conversation is free, else when it is. The press is
   * the user's yes, so the engine-lost wall and a due wake do not keep that row back.
   */
  async queueRelease(id: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const queue = this.queues.get(id);
    if (queue === undefined) return {};
    if (queue.release()) this.publishQueue(conversation, queue, false);
    void this.drainQueue(id, true);
    return {};
  }

  /**
   * Compact now. Answers once the run started (a cold conversation loads inside it, and a local model can take
   * minutes); its state and its end ride the conversation's `compaction` events. Refused while a turn, a wake, a
   * workflow run, another history op or a recap is live, and while the engine is stopped.
   */
  async compact(id: string): Promise<Record<string, never>> {
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
    await host.settled();
    this.opWall(host, id);
    if (!conversation.hasBase()) await this.loadBase(conversation);
    if ((await this.endpoint()) === null) throw chatRouteError('inference-stopped');
    this.opWall(host, id);
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    const entries = conversation.diskEntries();
    let run: { done: Promise<CompactOutcome> };
    try {
      run = host.compact(
        id,
        {
          modelId: conversation.info.modelId,
          ...(cwd === undefined ? {} : { cwd }),
          ...(entries === undefined ? {} : { entries }),
        },
        conversation.contextUsed(),
      );
    } catch (error) {
      if (error instanceof AgentHostError) throw this.opRefusal(host, id, error);
      throw error;
    }
    this.updateLive();
    this.busyChanged();
    void run.done.then((end) => {
      this.updateLive();
      this.scheduleState();
      this.busyChanged();
      this.answerEnded(conversation, end);
    });
    return {};
  }

  /**
   * Recap: the agent writes where things stand. Answers once the wait started (a cold conversation loads inside it, and
   * the recap is a model call on the local engine); its state and its end ride the conversation's `recap` events.
   * Refused while a turn, a wake, a history op or another recap is live, and while the engine is stopped. A workflow
   * run, a helper or a command does not refuse it: it only reads.
   */
  async recap(id: string): Promise<Record<string, never>> {
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
    await host.settled();
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    if (!conversation.hasBase()) await this.loadBase(conversation);
    if ((await this.endpoint()) === null) throw chatRouteError('inference-stopped');
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    const entries = conversation.diskEntries();
    try {
      host.recap(id, {
        modelId: conversation.info.modelId,
        ...(cwd === undefined ? {} : { cwd }),
        ...(entries === undefined ? {} : { entries }),
      });
    } catch (error) {
      if (error instanceof AgentHostError) throw chatRouteError(error.code);
      throw error;
    }
    return {};
  }

  /**
   * Rewind to here, step one: what a rewind to the page's `entry` removes, read against the disk now (the entry's
   * prompt index). Chat never starts or loads the agent. Work asks the agent for its dry run (the session loaded
   * first when it is cold): the files a restore puts back. 409 `rewind-changed` when the entry there now is not the
   * page's (`ts`, the time of the entry the page holds), `rewind-not-point` for a message the agent never ran (or
   * any other row), `fork-running` or `recap-running` while a fork or a recap of it runs, `conversation-busy` while a
   * turn, engine work or another op is live, `engine-lost` while the engine restarted under commands; Work also the
   * folder's errors, `inference-stopped`, and 503 `agent-slow` for a dry run with no answer in time.
   */
  async rewindPlan(id: string, entry: number, ts: number): Promise<RewindPlan> {
    this.assertOpen();
    const conversation = await this.known(id);
    const host = this.hosts[conversation.info.mode];
    await host.settled();
    this.opWall(host, id);
    if (host.engineLostTasks()) throw chatRouteError('engine-lost');
    if (!conversation.hasBase()) await this.loadBase(conversation);
    const detail = await this.opts.readSession(id);
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    this.opWall(host, id);
    const view = conversation.shownEntries();
    if (view[entry]?.ts !== ts) throw ApiError.conflict(REWIND_CHANGED, 'rewind-changed');
    const needles = needleList(this.needles());
    const target = targetOf(
      view,
      detail?.transcript ?? [],
      entry,
      (text) => redactText(text, needles),
      conversation.unsentPrompts(),
    );
    if ('refused' in target) throw ApiError.conflict(REWIND_NOT_POINT, 'rewind-not-point');
    // The agent holds one goal, the last row's: cutting that row while it is open would leave it with no strip.
    const goal = view.findLastIndex((e) => e.kind === 'goal');
    const open = view[goal];
    if (goal >= entry && open?.kind === 'goal' && open.status !== 'complete' && open.status !== 'cleared')
      throw ApiError.conflict(REWIND_GOAL, 'rewind-goal');
    const live = new Set(host.liveActivity(id).tasks.map((task) => task.id));
    const counts = rewindCounts(view, entry, live, host.stoppableRuns(id));
    const token = rewindToken(conversation, entry);
    const files = conversation.info.mode === 'work' ? await this.rewindFiles(conversation, target.target) : [];
    const listed = planFiles(files, (path) => visibleText(redactText(path, needles)));
    this.rewindPlans.set(id, {
      token,
      entry,
      from: view[entry]!.ts,
      target: target.target,
      tasks: counts.tasks,
      schedules: counts.schedules,
      runs: counts.runs,
      files,
    });
    return {
      token,
      removes: { messages: counts.messages, steps: counts.steps },
      background: { commands: counts.tasks.length, schedules: counts.schedules.length, runs: counts.runs.length },
      helpers: counts.helpers,
      attachments: counts.attachments,
      ...listed,
    };
  }

  /** Work: the agent's dry run of a rewind to prompt `target`, with the checks of a commit before it starts the agent. */
  private async rewindFiles(
    conversation: Conversation,
    target: number,
  ): Promise<Extract<RewindPreview, { files: unknown }>['files']> {
    const id = conversation.info.id;
    const problem = await this.folderState(conversation);
    if (problem !== null) throw cwdError(problem);
    const host = this.hosts.work;
    if ((await this.endpoint()) === null) throw chatRouteError('inference-stopped');
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    const entries = conversation.diskEntries();
    let preview: RewindPreview;
    try {
      preview = await host.rewindPreview(
        id,
        {
          modelId: conversation.info.modelId,
          ...(conversation.info.cwd === null ? {} : { cwd: conversation.info.cwd }),
          ...(entries === undefined ? {} : { entries }),
        },
        target,
      );
    } catch (error) {
      if (error instanceof AgentHostError) throw this.opRefusal(host, id, error);
      throw error;
    }
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    if (!conversation.info.hot && host.isHot(id)) conversation.setInfo({ hot: true });
    if ('refused' in preview) throw ApiError.conflict(REWIND_NOT_POINT, 'rewind-not-point');
    return preview.files;
  }

  /**
   * Rewind to here, step two: the plan's rewind, while the conversation is as the plan read it (else 409
   * `rewind-changed`). Answers once the op started; it ends on the conversation's stream (`rewind`), after the cut's
   * `sync` and, in Work, the rule's mark. Refused like the plan, and while the engine is stopped. Chat has no files: a
   * mode other than `conversation` is a bad request.
   */
  async rewind(id: string, input: RewindInput): Promise<Record<string, never>> {
    this.assertOpen();
    const conversation = await this.known(id);
    const mode = conversation.info.mode;
    if (mode === 'chat' && input.mode !== 'conversation') throw ApiError.badRequest(REWIND_CHAT_FILES);
    const plan = this.rewindPlans.get(id);
    const current = (): boolean =>
      plan !== undefined && plan.token === input.token && rewindToken(conversation, plan.entry) === plan.token;
    if (!current()) throw ApiError.conflict(REWIND_CHANGED, 'rewind-changed');
    let cwd: string | undefined;
    if (mode === 'work') {
      const problem = await this.folderState(conversation);
      if (problem !== null) throw cwdError(problem);
      cwd = conversation.info.cwd ?? undefined;
    } else {
      await this.assertChatReady();
    }
    const host = this.hosts[mode];
    await host.settled();
    if ((await this.endpoint()) === null) throw chatRouteError('inference-stopped');
    this.opWall(host, id);
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    if (!current() || plan === undefined) throw ApiError.conflict(REWIND_CHANGED, 'rewind-changed');
    const entries = conversation.diskEntries();
    let run: { done: Promise<RewindHostEnd> };
    try {
      run = host.rewind(
        id,
        {
          modelId: conversation.info.modelId,
          ...(cwd === undefined ? {} : { cwd }),
          ...(entries === undefined ? {} : { entries }),
        },
        {
          target: plan.target,
          mode: input.mode,
          tasks: plan.tasks,
          schedules: plan.schedules,
          runs: plan.runs,
          files: plan.files,
        },
      );
    } catch (error) {
      if (error instanceof AgentHostError) throw this.opRefusal(host, id, error);
      throw error;
    }
    this.rewindPlans.delete(id);
    this.updateLive();
    this.busyChanged();
    void run.done.then((end) => this.rewindEnded(conversation, plan, input.mode, end));
    return {};
  }

  /**
   * Fork: the agent copies the conversation on disk; the page opens the copy as any stored conversation (its first
   * message loads it). With `at`, up to and including the answer of the message at `at.entry`: the rewind's mapping to
   * the agent's prompt index, then the automatic continues of that answer. Work writes the copy's index line; a failed
   * write deletes the copy. Never loads the conversation. 409 `fork-changed` when the entry there now is not the page's
   * (`at.ts`, the time of the entry the page holds), `fork-not-point` for a message the agent never ran,
   * `fork-running` while a fork of it runs, `recap-running` while a recap is written, `conversation-busy` while a turn,
   * engine work or another op is live, `engine-lost` while the engine restarted under commands. A missed reply is
   * `agent-slow`, and the lists are read again; a reply that comes later lands the copy as an answered one does. A row
   * queued while the copy was made goes once the op ends, however it ended: the original did not change.
   */
  async fork(id: string, at?: { entry: number; ts: number }): Promise<{ id: string }> {
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
    await host.settled();
    this.opWall(host, id);
    if (host.engineLostTasks()) throw chatRouteError('engine-lost');
    let target: number | null = null;
    if (at !== undefined) {
      if (!conversation.hasBase()) await this.loadBase(conversation);
      const detail = await this.opts.readSession(id);
      if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
      const view = conversation.shownEntries();
      if (view[at.entry]?.ts !== at.ts) throw ApiError.conflict(FORK_CHANGED, 'fork-changed');
      const disk = detail?.transcript ?? [];
      const needles = needleList(this.needles());
      const found = targetOf(view, disk, at.entry, (text) => redactText(text, needles), conversation.unsentPrompts());
      if ('refused' in found) throw ApiError.conflict(FORK_NOT_POINT, 'fork-not-point');
      target = answerEnd(disk, found.target);
    }
    this.opWall(host, id);
    if (cwd !== undefined) this.hold(cwd);
    let late = false;
    try {
      let copy: string;
      try {
        copy = await host.fork(
          id,
          { modelId: conversation.info.modelId, ...(cwd === undefined ? {} : { cwd }) },
          target,
        );
      } catch (error) {
        if (!(error instanceof AgentHostError)) throw error;
        if (error.code === 'agent-slow') {
          // Held until the late copy lands (`lateFork`); a late reply that names no copy keeps it for the app run.
          if (cwd !== undefined) {
            this.lateForkFolders.set(id, [...(this.lateForkFolders.get(id) ?? []), cwd]);
            late = true;
          }
          this.emitHost({ type: 'conversations-changed' });
        }
        throw this.opRefusal(host, id, error);
      } finally {
        void this.drainQueue(id, false);
      }
      await this.forkLanded(mode, copy, cwd, conversation.info.thinking);
      return { id: copy };
    } finally {
      if (cwd !== undefined && !late) this.release(cwd);
    }
  }

  /**
   * Work writes the copy's index line (a failed write deletes the copy); the copy keeps the original's thinking level
   * and starts with no queue; every list is read again.
   */
  private async forkLanded(
    mode: ChatMode,
    copy: string,
    cwd: string | undefined,
    thinking: ThinkingLevel | undefined,
  ): Promise<void> {
    if (mode === 'work') {
      try {
        await this.indexWrite((index) => index.work.push({ id: copy, createdAt: this.now() }), copy);
      } catch (error) {
        await this.hosts.work.deleteSession(copy, cwd ?? '').catch(() => {});
        throw error;
      }
    }
    if (thinking !== undefined) await this.composerStore.setThinking(copy, thinking).catch(() => {});
    this.emitHost({ type: 'conversations-changed' });
  }

  /**
   * A fork's reply that came after its soft deadline: the copy lands as an answered fork's does, in the folder the fork
   * held (the original may be gone), which it lets go once the copy is in the index.
   */
  private lateFork(mode: ChatMode, from: string, copy: string): void {
    const held = this.lateForkFolders.get(from) ?? [];
    const folder = held.shift();
    if (held.length === 0) this.lateForkFolders.delete(from);
    if (this.closed) return;
    const original = this.conversations.get(from)?.info;
    const cwd = mode === 'work' ? (folder ?? original?.cwd ?? undefined) : undefined;
    void this.forkLanded(mode, copy, cwd, original?.thinking)
      .catch(() => {})
      .then(() => {
        if (folder !== undefined) this.release(folder);
      });
  }

  /**
   * A history op may not start: a fork or a recap of the conversation runs (each its own reason), else a turn, engine
   * work, another op or a delete (`conversation-busy`).
   */
  private opWall(host: AgentHost, id: string): void {
    const op = host.opOf(id);
    if (op === 'fork') throw ApiError.conflict(FORK_RUNNING, 'fork-running');
    if (op === 'recap') throw ApiError.conflict(RECAP_RUNNING, 'recap-running');
    if (host.engineBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
  }

  /** The route error of a host op that refused: a busy one says what holds the conversation. */
  private opRefusal(host: AgentHost, id: string, error: AgentHostError): ApiError {
    if (error.code === 'conversation-busy') {
      try {
        this.opWall(host, id);
      } catch (wall) {
        if (wall instanceof ApiError) return wall;
      }
    }
    return chatRouteError(error.code);
  }

  /**
   * The Context panel: the agent's split of the context while the conversation is loaded in the running agent, else
   * the last use stored. Never starts or loads the agent; answers while the conversation works too.
   */
  async context(id: string): Promise<ConversationContext> {
    this.assertOpen();
    const conversation = await this.known(id);
    if (!conversation.hasBase()) await this.loadBase(conversation);
    const info = await this.hosts[conversation.info.mode].contextInfo(id);
    return conversationContext(info, conversation.contextUsed());
  }

  /** Work: the user's Stop of one workflow run. */
  async stopRun(id: string, runId: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const stop = conversation.info.mode === 'work' ? await this.hosts.work.stopRun(id, runId) : 'not-live';
    if (stop === 'not-live') throw ApiError.notFound('No such run.');
    if (stop === 'rejected') throw ApiError.conflict('The run could not be stopped.');
    if (stop === 'failed') throw ApiError.unavailable('The agent could not be reached.');
    return {};
  }

  /**
   * Work: the user's Stop of one background command or monitor. 404 when the conversation holds no such live task
   * (not loaded in the running agent, ended, or a Chat conversation), 503 when the agent gives no answer in time.
   */
  async stopTask(id: string, taskId: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const stop = conversation.info.mode === 'work' ? await this.hosts.work.stopTask(id, taskId) : 'not-live';
    if (stop === 'not-live') throw ApiError.notFound('No such task.');
    if (stop === 'failed') throw ApiError.unavailable('The agent could not be reached.');
    return {};
  }

  /**
   * Work: the user's Stop of one of the conversation's own subagents, by its child session id. 404 when the running
   * agent holds no such running subagent for it (not loaded, ended, or a Chat conversation), 503 when the agent gives no
   * answer in time. Its end rides the conversation stream.
   */
  async stopSubagent(id: string, childId: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const stop = conversation.info.mode === 'work' ? await this.hosts.work.stopSubagent(id, childId) : 'not-live';
    if (stop === 'not-live') throw ApiError.notFound('No such subagent.');
    if (stop === 'failed') throw ApiError.unavailable('The agent could not be reached.');
    return {};
  }

  /**
   * Work: the user's message to one of the conversation's own running subagents, read at its next step. 404 when the
   * running agent holds no such running subagent for it, or the agent refused it as gone; every other end is a result.
   */
  async messageSubagent(id: string, childId: string, text: string): Promise<{ result: SubagentMessageResult }> {
    const conversation = await this.known(id);
    const result =
      conversation.info.mode === 'work' ? await this.hosts.work.messageSubagent(id, childId, text) : 'gone';
    if (result === 'gone') throw ApiError.notFound('No such subagent.');
    return { result };
  }

  /**
   * Work: the Agent view of `window` on one agent the conversation started (`childId`, the attempt it shows). While the
   * agent runs, the window's `agent-entries` events on the conversation stream carry its later ops and, at its end, the
   * stored list. 404 for an agent that is not the conversation's own (a nested one included), and for a Chat
   * conversation; 409 for a third live view in one window.
   */
  async watchSubagent(id: string, childId: string, window: string): Promise<AgentWatchAnswer> {
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work' || !this.ownsChild(id, childId)) throw ApiError.notFound('No such subagent.');
    const answer = await this.hosts.work.watchAgent(id, childId, window);
    if (answer === 'too-many') throw ApiError.conflict('Two agents are open in this window. Close one first.');
    return answer;
  }

  /**
   * Work: the window's view of the agent closed. Answered the same when no view was open; 404, with the view left as it
   * is, for an agent that is not the conversation's own.
   */
  async unwatchSubagent(id: string, childId: string, window: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work' || !this.ownsChild(id, childId)) throw ApiError.notFound('No such subagent.');
    this.hosts.work.unwatchAgent(childId, window);
    return {};
  }

  /**
   * Work: the user's Delete of one schedule the conversation's transcript lists. A conversation the running agent does
   * not hold is loaded first (its folder and the engine must be there). 404 for a schedule the transcript does not list
   * as live or the agent no longer holds, and for a Chat conversation; 409 `engine-lost` while the engine restarted
   * under commands or monitors; 503 when the agent gives no answer in time. The row leaves on the conversation stream.
   */
  async deleteSchedule(id: string, scheduleId: string): Promise<Record<string, never>> {
    this.assertOpen();
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.notFound('No such schedule.');
    const host = this.hosts.work;
    await host.settled();
    if (this.deleting.has(id)) throw chatRouteError('conversation-busy');
    if (host.engineLostTasks()) throw chatRouteError('engine-lost');
    if (!conversation.hasBase()) await this.loadBase(conversation);
    if (!listsSchedule(conversation.shownEntries(), scheduleId)) throw ApiError.notFound('No such schedule.');
    if (!host.isHot(id)) {
      const problem = await this.folderState(conversation);
      if (problem !== null) throw cwdError(problem);
      if ((await this.endpoint()) === null) throw chatRouteError('inference-stopped');
    }
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    if (this.deleting.has(id)) throw chatRouteError('conversation-busy');
    const cwd = conversation.info.cwd ?? undefined;
    const entries = conversation.diskEntries();
    const end = await host.deleteSchedule(id, scheduleId, {
      modelId: conversation.info.modelId,
      ...(cwd === undefined ? {} : { cwd }),
      ...(entries === undefined ? {} : { entries }),
    });
    if (!conversation.info.hot && host.isHot(id)) conversation.setInfo({ hot: true });
    if (end === 'gone') throw ApiError.notFound('No such schedule.');
    if (end === 'failed') throw ApiError.unavailable('The agent could not be reached.');
    return {};
  }

  /** An MCP form the host refused comes back as `fields`; its card stays open. */
  async answer(id: string, interactionId: string, input: InteractionAnswerInput): Promise<InteractionAnswerResult> {
    const conversation = await this.known(id);
    if (conversation.info.mode === 'chat') throw ApiError.notFound('No such request.');
    if ('never' in input) return this.answerNever(conversation, interactionId, input.never);
    if ('mcp' in input) {
      const fields = this.hosts.work.answerElicit(id, interactionId, input);
      return fields === null ? {} : { fields };
    }
    if ('answers' in input || 'reply' in input) {
      this.hosts.work.answerQuestion(id, interactionId, input);
      return {};
    }
    if (!('optionId' in input)) {
      this.hosts.work.answerPlan(id, interactionId, input.decision, input.feedback);
      return {};
    }
    const denied = this.hosts.work.answer(id, interactionId, input.optionId, input.note);
    if (denied !== null && input.note !== undefined) await this.noteDenied(conversation, denied, input.note);
    return {};
  }

  /**
   * "Never allow" on a command card: the call is denied with the rule's note now (its mark and note row in that tick),
   * then the rule file is written. The rule's words are not in the reply's history line, only its id.
   */
  private async answerNever(
    conversation: Conversation,
    interactionId: string,
    scope: 'short' | 'exact',
  ): Promise<InteractionAnswerResult> {
    const { denied, saved } = this.hosts.work.answerNever(conversation.info.id, interactionId, scope);
    const logged = this.ruleDenied(conversation.info.id, denied);
    const kept = await saved;
    const cwd = conversation.info.cwd;
    if (cwd !== null) this.rulesChanged(cwd, kept || denied.ruleId === null ? null : denied.ruleId);
    await logged;
    if (!kept || denied.ruleId === null) return { never: { saved: false } };
    return { never: { saved: true, ruleId: denied.ruleId, words: redactText(denied.words, this.needles()) } };
  }

  /**
   * A call a never-allow rule denied: its mark, and for the conversation's own call the note row of what the agent was
   * told, in this tick; then its history line (the rule's id, never its words). A line that was not written is not an
   * error for the user.
   */
  private async ruleDenied(id: string, denied: RuleDeny): Promise<void> {
    const at = this.now();
    const child = denied.child === undefined ? {} : { child: denied.child };
    const conversation = this.conversations.get(id);
    const cwd = conversation?.info.cwd ?? null;
    conversation?.permissionMark({
      kind: 'denied',
      at,
      toolCallId: denied.toolCallId,
      access: denied.access,
      by: 'rule',
      ...(denied.ruleId === null
        ? { words: null }
        : { ruleId: denied.ruleId, words: cwd === null ? null : this.never.wordsOf(cwd, denied.ruleId) }),
      card: denied.card,
      ...child,
    });
    if (denied.child === undefined && denied.noted) conversation?.noteSent(neverNote(denied.words));
    if (denied.ruleId === null) return;
    await this.permissionLog
      .append(id, {
        kind: 'denied',
        at,
        toolCallId: denied.toolCallId,
        access: denied.access,
        by: 'rule',
        ruleId: denied.ruleId,
        ...child,
      })
      .catch(() => {});
  }

  /**
   * The rules of a folder changed: every loaded conversation of it gets the new count, and a rule that is gone makes
   * its marks read "rule removed".
   */
  private rulesChanged(cwd: string, gone: string | null): void {
    const key = folderKey(cwd);
    for (const conversation of this.conversations.values()) {
      const own = conversation.info.cwd;
      if (conversation.info.mode !== 'work' || own === null || folderKey(own) !== key) continue;
      if (gone !== null) conversation.ruleGone(gone);
      this.setRuleCount(conversation, this.never.rulesOf(own).length);
    }
  }

  private setRuleCount(conversation: Conversation, count: number): void {
    if ((conversation.info.neverRules ?? 0) === count) return;
    const info = { ...conversation.info };
    delete info.neverRules;
    conversation.info = info;
    conversation.setInfo(count === 0 ? {} : { neverRules: count });
  }

  /** Work: the never-allow rules of the conversation's folder, oldest first. 404 for a Chat conversation. */
  async neverRules(id: string): Promise<NeverRulesResponse> {
    const conversation = await this.known(id);
    const cwd = conversation.info.cwd;
    if (conversation.info.mode !== 'work' || cwd === null) throw ApiError.notFound('No such conversation.');
    await this.never.ready();
    const needles = this.needles();
    return {
      folder: cwd,
      rules: this.never.rulesOf(cwd).map((rule) => ({
        id: rule.id,
        words: redactText(rule.words.join(' '), needles),
        at: rule.at,
      })),
    };
  }

  /** Work: removes a never-allow rule of the conversation's folder; its marks read "rule removed" from now on. */
  async removeNeverRule(id: string, ruleId: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    const cwd = conversation.info.cwd;
    if (conversation.info.mode !== 'work' || cwd === null) throw ApiError.notFound('No such conversation.');
    await this.never.ready();
    const removed = await this.never.remove(cwd, ruleId);
    if (removed === 'not-found') throw ApiError.notFound('No such rule.');
    if (removed === 'failed') throw ApiError.internal('The rule was not removed.');
    this.rulesChanged(cwd, ruleId);
    return {};
  }

  /**
   * A call denied with a note: its mark, and the note row for the conversation's own call (a helper's note is in the
   * helper's session), in the tick of the reply so the row sits under its call; then its history line. The agent
   * sends no live copy of the note. A line that was not written is not an error for the user.
   */
  private async noteDenied(conversation: Conversation, denied: NoteDeny, note: string): Promise<void> {
    const at = this.now();
    const child = denied.child === undefined ? {} : { child: denied.child };
    conversation.permissionMark({
      kind: 'denied',
      at,
      toolCallId: denied.toolCallId,
      access: denied.access,
      by: 'note',
      card: denied.card,
      ...child,
    });
    if (denied.child === undefined) conversation.noteSent(note);
    await this.permissionLog
      .append(conversation.info.id, {
        kind: 'denied',
        at,
        toolCallId: denied.toolCallId,
        access: denied.access,
        by: 'note',
        ...child,
      })
      .catch(() => {});
  }

  /** An MCP card closed, or a page opened from one is done: its line, live now and in the history file (never a typed value). */
  private mcpClosed(id: string, closed: McpClosed): void {
    const line = {
      kind: 'mcp' as const,
      at: this.now(),
      server: storedServer(closed.server),
      mode: closed.mode,
      outcome: closed.outcome,
    };
    this.conversations.get(id)?.permissionMark(line);
    void this.permissionLog.append(id, line).catch(() => {});
  }

  /** The permission mode of a conversation now. */
  modeOf(id: string): PermissionMode {
    return this.permissionModes.get(id) ?? 'ask';
  }

  /**
   * The permission mode of a Work conversation, from the user's pick in the app's window: the only way up, and only
   * when the page sent it as a move up (`up`: its confirm). A move down is applied in this tick and is never queued; a
   * move up runs on the permission chain and is refused when a move down, or a pick of a lower mode while it waits,
   * came after it was sent. A target below the mode is a move down whatever `up` says. `temp`: the temp folder the
   * confirm named; a move up to Auto whose confirm did not name the temp folder Auto covers is refused.
   */
  setPermissionMode(
    id: string,
    target: PermissionMode,
    up: boolean,
    temp: string | null = null,
  ): Promise<{ mode: PermissionMode }> {
    if (MODE_RANK[target] < MODE_RANK[this.modeOf(id)]) return this.moveDown(id, target);
    const pending = this.raising.get(id) ?? [];
    if (!up && MODE_RANK[target] > MODE_RANK[this.modeOf(id)]) return this.notUp(id, pending.length > 0);
    if (pending.some((raised) => MODE_RANK[raised] > MODE_RANK[target])) {
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
    const run = (): Promise<{ mode: PermissionMode }> => this.raise(id, target, epoch, temp).finally(settled);
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

  /**
   * A pick sent as a move down to a mode above the conversation's: the page still showed an older, higher mode. It
   * never moves the mode up; like a move down it calls back a move up that waits (in this tick), then it is 409.
   */
  private async notUp(id: string, raising: boolean): Promise<never> {
    if (raising) this.permissionEpoch.set(id, this.epochOf(id) + 1);
    this.assertOpen();
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.badRequest('Permission mode is for Work only.');
    if (this.deleting.has(id) || this.conversations.get(id) !== conversation) {
      throw ApiError.notFound('No such conversation.');
    }
    throw modeChanged();
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

  private async raise(
    id: string,
    target: PermissionMode,
    epoch: number,
    temp: string | null,
  ): Promise<{ mode: PermissionMode }> {
    this.assertOpen();
    const conversation = await this.known(id);
    if (conversation.info.mode !== 'work') throw ApiError.badRequest('Permission mode is for Work only.');
    const gone = (): boolean => this.deleting.has(id) || this.conversations.get(id) !== conversation;
    if (gone()) throw ApiError.notFound('No such conversation.');
    if (this.epochOf(id) !== epoch) throw modeChanged();
    const from = this.modeOf(id);
    if (MODE_RANK[target] <= MODE_RANK[from]) return { mode: from };
    const covered = target === 'auto' ? this.tempOf(conversation) : null;
    if (covered !== null && temp !== covered) {
      throw new ApiError('E_CONFLICT', 'The Auto text did not name the temp folder.', 'temp-unnamed');
    }
    const gated = from === 'ask' && target === 'auto';
    // Only permission and plan cards block: a question offers no "always" answer.
    const cardOpen = (): ApiError | null =>
      gated && this.hosts.work.openInteractions(id).some((item) => item.kind === 'permission' || item.kind === 'plan')
        ? new ApiError('E_CONFLICT', 'Answer the open request first.', 'card-open')
        : null;
    const open = cardOpen();
    if (open !== null) throw open;
    const problem = await this.folderState(conversation);
    if (problem !== null) throw cwdError(problem);
    const place = this.settingsPlace(conversation);
    const [prints, hooks] = await Promise.all([
      settingsPrints(place),
      from === 'ask' ? findUserHooks(place) : Promise.resolve(conversation.info.userHooks === true),
    ]);
    if (gone()) throw ApiError.notFound('No such conversation.');
    if (this.epochOf(id) !== epoch) throw modeChanged();
    const late = cardOpen();
    if (late !== null) throw late;
    if (from !== 'ask') await this.settingsChanged(conversation, prints);
    if (gone()) throw ApiError.notFound('No such conversation.');
    if (from === 'ask') this.setUserHooks(conversation, hooks);
    const line: PermissionLine = {
      kind: 'mode',
      at: this.now(),
      mode: target,
      from: this.logged.get(id) ?? 'ask',
      by: 'user',
      prints,
    };
    if (target === 'auto' && this.hosts.work.editsGranted(id)) line.editsUnasked = true;
    if (hooks) line.userHooks = true;
    if (covered !== null) line.temp = true;
    try {
      await this.permissionLog.append(id, line);
    } catch {
      if (gone()) throw ApiError.notFound('No such conversation.');
      throw new ApiError('E_UNAVAILABLE', 'The permission mode could not be changed.');
    }
    if (gone()) throw ApiError.notFound('No such conversation.');
    this.logged.set(id, target);
    if (this.epochOf(id) !== epoch) {
      conversation.permissionMark(markOf(line, place));
      throw modeChanged();
    }
    const opened = cardOpen();
    if (opened !== null) {
      // A card opened while the line was written: it offers "always" answers, so Auto is refused and the record closed.
      const back: PermissionLine = { kind: 'mode', at: this.now(), mode: 'ask', from: target, by: 'user' };
      this.logged.set(id, 'ask');
      conversation.permissionMark(markOf(line, place));
      conversation.permissionMark(markOf(back, place));
      await this.permissionLog.append(id, back).catch(() => {});
      throw opened;
    }
    this.permissionModes.set(id, target === 'auto' ? 'auto' : 'bypass');
    this.settingsBase.set(id, prints);
    conversation.setInfo({ permissionMode: target });
    conversation.permissionMark(markOf(line, place));
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
        ...host.openChild(id, card.id),
      };
      try {
        await this.permissionLog.append(id, {
          kind: 'allowed',
          at: mark.at,
          mode: 'bypass',
          toolCallId: mark.toolCallId,
          access: mark.access,
          ...(mark.child !== undefined ? { child: mark.child } : {}),
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
      const place = this.settingsPlace(conversation);
      const prints = await settingsPrints(place).catch(() => null);
      if (prints !== null) await this.settingsChanged(conversation, prints);
      if (this.modeOf(id) === 'ask') this.settingsBase.delete(id);
      const line: PermissionLine = {
        kind: 'mode',
        at: this.now(),
        mode: target,
        from: this.logged.get(id) ?? 'ask',
        by,
      };
      if (prints !== null) line.prints = prints;
      if (target === 'auto' && this.tempOf(conversation) !== null) line.temp = true;
      if (this.conversations.get(id) === conversation) this.logged.set(id, target);
      conversation.permissionMark(markOf(line, place));
      await this.permissionLog.append(id, line).catch(() => {});
    };
    const next = this.permissionChain.then(run, run);
    this.permissionChain = next.catch(() => {});
  }

  /** A print differs from the base: a `settings` mark that names those files, and the prints are the next base. */
  private async settingsChanged(conversation: Conversation, prints: SettingsPrints): Promise<void> {
    const id = conversation.info.id;
    const base = this.settingsBase.get(id);
    if (base === undefined) return;
    const changed = changedSettings(base, prints);
    if (changed.length === 0) return;
    this.settingsBase.set(id, prints);
    const line: PermissionLine = { kind: 'settings', at: this.now(), prints, changed };
    conversation.permissionMark(markOf(line, this.settingsPlace(conversation)));
    await this.permissionLog.append(id, line).catch(() => {});
  }

  private async compareSettings(conversation: Conversation): Promise<void> {
    if (!this.settingsBase.has(conversation.info.id)) return;
    await this.settingsChanged(conversation, await settingsPrints(this.settingsPlace(conversation)));
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
        ...(mark.temp === true ? { temp: true as const } : {}),
        ...(mark.child !== undefined ? { child: mark.child } : {}),
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

  /**
   * A Work session (re)started: it reads hot (a soft load that landed after its op gave up has no op to say so), the
   * settings check while the mode is not Ask, and the user's hooks again.
   */
  private async sessionReady(id: string): Promise<void> {
    const conversation = this.conversations.get(id);
    if (conversation === undefined || conversation.info.mode !== 'work') return;
    if (!conversation.info.hot && this.hosts.work.isHot(id)) conversation.setInfo({ hot: true });
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

  /** The temp folder Auto covers in this conversation, as people type it; null when its Work folder holds it. */
  private tempOf(conversation: Conversation): string | null {
    const cwd = conversation.info.cwd;
    if (cwd === null) return null;
    return typedTop(tempRoots({ cwd, home: this.home, temp: this.tempRoots })[0]);
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
    const [lines] = await Promise.all([this.permissionLog.read(id), this.never.ready()]);
    const place = this.settingsPlace(conversation);
    const cwd = conversation.info.cwd;
    const words = (ruleId: string): string | null => (cwd === null ? null : this.never.wordsOf(cwd, ruleId));
    for (const line of lines) conversation.permissionMark(markOf(line, place, words));
    const rules = cwd === null ? 0 : this.never.rulesOf(cwd).length;
    if (rules > 0) conversation.info = { ...conversation.info, neverRules: rules };
    let last: PermissionMode = 'ask';
    let base: SettingsPrints | undefined;
    for (const line of lines) {
      if (line.kind === 'mode') last = line.mode;
      if ((line.kind === 'mode' || line.kind === 'settings') && line.prints !== undefined) base = line.prints;
    }
    if (last === 'ask' || this.modeOf(id) !== 'ask') return;
    if (base !== undefined) {
      const prints = await settingsPrints(place);
      const keys = changedSettings(base, prints);
      if (keys.length > 0) {
        const changed: PermissionLine = { kind: 'settings', at: this.now(), prints, changed: keys };
        conversation.permissionMark(markOf(changed, place));
        await this.permissionLog.append(id, changed).catch(() => {});
      }
    }
    const restart: PermissionLine = { kind: 'mode', at: this.now(), mode: 'ask', from: last, by: 'restart' };
    this.logged.set(id, 'ask');
    conversation.permissionMark(markOf(restart, place));
    await this.permissionLog.append(id, restart).catch(() => {});
  }

  async patch(id: string, input: PatchConversationInput): Promise<ConversationInfo> {
    const conversation = await this.known(id);
    const mode = conversation.info.mode;
    const host = this.hosts[mode];
    if ('thinking' in input) return this.setThinking(conversation, input.thinking);
    await host.settled();
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    // A workflow run's helpers and a subagent use the conversation's model: a change waits for them. A command does not.
    if (!('title' in input) && host.engineBusy(id)) throw chatRouteError('conversation-busy');
    if ('title' in input) {
      const title = validateRenameTitle(input.title);
      if (host.isHot(id)) await host.rename(id, title);
      else renameSession(mode === 'chat' ? this.chatReader : this.workReader, id, title);
      conversation.setInfo({ title, updatedAt: this.now() });
      this.originChanged(id, title);
      this.emitHost({ type: 'conversations-changed' });
      return conversation.publicInfo();
    }
    const modelId = input.modelId;
    if (host.isHot(id)) {
      await host.setModel(id, modelId);
    } else {
      const endpoint = await this.endpoint();
      if (this.deleting.has(id)) throw chatRouteError('conversation-busy');
      const available =
        host.models()?.map((m) => m.modelId) ??
        endpoint?.models.filter((m) => !isDecisionModel(m)).map((m) => m.name) ??
        [];
      if (!available.includes(modelId)) throw chatRouteError('model-unavailable');
    }
    if (modelId !== conversation.info.modelId) {
      if (!conversation.hasBase()) await this.loadBase(conversation);
      conversation.modelSwitch(modelId);
      const level = conversation.info.thinking;
      const stepped = this.levelOn(level, modelId);
      if (stepped !== level && stepped !== undefined) {
        await this.composerStore.setThinking(id, stepped).catch(() => {});
        conversation.setInfo({ modelId, thinking: stepped });
      } else {
        conversation.setInfo({ modelId });
      }
      conversation.clearNotice('thinking-not-set');
    }
    return conversation.publicInfo();
  }

  /**
   * The conversation's thinking level, stored in its composer file. Taken while an answer runs too: the host sets it
   * before the next answer.
   */
  private async setThinking(conversation: Conversation, level: ThinkingLevel): Promise<ConversationInfo> {
    const id = conversation.info.id;
    if (this.deleting.has(id)) throw chatRouteError('conversation-busy');
    await this.composerStore.setThinking(id, level);
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    conversation.setInfo({ thinking: level });
    conversation.clearNotice('thinking-not-set');
    return conversation.publicInfo();
  }

  /**
   * The level a conversation keeps on `modelId`: stepped to one the model lists (nearest below, else its default)
   * when an agent listed the model's levels; as it is otherwise.
   */
  private levelOn(level: ThinkingLevel | undefined, modelId: string): ThinkingLevel | undefined {
    const thinking = this.levels.get(modelId);
    return level === undefined || thinking === undefined || thinking === null ? level : stepLevel(level, thinking);
  }

  async delete(id: string): Promise<Record<string, never>> {
    const conversation = await this.known(id);
    if (conversation.info.mode === 'work') return this.deleteWork(conversation);
    const host = this.hosts.chat;
    await host.settled();
    if (host.isWorking(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    this.deleting.add(id);
    try {
      if (host.isHot(id) || (await this.agentCanStart(host))) await host.deleteSession(id, '');
      else {
        this.deleteCold(id);
        host.forgetSession(id);
      }
      await removeChatPromptHistory(this.opts.paths);
      await this.forgetQueue(id);
      conversation.gone('deleted');
      this.conversations.delete(id);
      this.hostStarts.delete(id);
      this.originChanged(id, null);
      this.emitHost({ type: 'conversations-changed' });
      return {};
    } finally {
      this.deleting.delete(id);
    }
  }

  /**
   * Frees the engine's memory: the resident chat model is unloaded, and it loads again on the next message. Refused
   * while an answer, a wake turn, a workflow run or a helper is live in Chat or Work, or the engine loads a model. A
   * command or monitor does not use the engine and does not refuse it.
   */
  async unloadModels(): Promise<{ unloaded: string[] }> {
    this.assertOpen();
    const endpoint = await this.endpoint();
    if (endpoint === null) throw chatRouteError('inference-stopped');
    const loading = await this.engineLoading(endpoint.pid);
    const turns = (['chat', 'work'] as const).reduce((n, mode) => {
      const state = this.hosts[mode].modeState();
      return n + state.turnsInFlight + state.background;
    }, 0);
    if (loading.length > 0 || turns > 0) throw chatRouteError('engine-busy');
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
    if (host.isWorking(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    this.deleting.add(id);
    try {
      await host.deleteSession(id, conversation.info.cwd ?? '');
      await this.indexWrite((index) => {
        index.work = index.work.filter((entry) => entry.id !== id);
      });
      this.permissionModes.delete(id);
      this.permissionEpoch.delete(id);
      this.draft.forget(id);
      this.logged.delete(id);
      this.settingsBase.delete(id);
      await this.permissionLog.remove(id).catch(() => {});
      await removeSessionPdfs(this.opts.paths.workPdf, id).catch(() => {});
      await this.forgetQueue(id);
      conversation.gone('deleted');
      this.conversations.delete(id);
      this.hostStarts.delete(id);
      this.originChanged(id, null);
      const cwd = conversation.info.cwd;
      if (cwd !== null && this.isScratchFolder(cwd) && !this.cwdInUse(cwd, id)) await this.removeScratch(cwd);
      this.emitHost({ type: 'conversations-changed' });
      return {};
    } finally {
      this.deleting.delete(id);
    }
  }

  /**
   * Another Work conversation runs in `cwd`: one this service holds, one a create or a fork is opening, or a session
   * of the Work index on disk.
   */
  private cwdInUse(cwd: string, except?: string): boolean {
    if (this.holdsCwd(cwd, except) || (this.opening.get(cwd) ?? 0) > 0) return true;
    const sessions = this.workReader.scan().sessions;
    for (const entry of readWorkIndexSync(this.opts.paths.conversations).work) {
      if (entry.id !== except && sessions.get(entry.id)?.summary.cwd === cwd) return true;
    }
    return false;
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
      // A pick on the New screen lives in a window: with no window left it goes.
      if (this.hostSinks.size === 0) this.draft.clear();
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
      pending.detach = conversation.attach(send, hostView(host, id));
    };
    void attach();
    return () => {
      pending.cancelled = true;
      pending.detach?.();
      // A row left open for edit by the last page would keep the queue from sending for good.
      const conversation = this.conversations.get(id);
      const queue = this.queues.get(id);
      if (conversation !== undefined && queue !== undefined && conversation.listeners === 0 && queue.endEdits()) {
        this.queueChanged(conversation, queue);
      }
      if (conversation !== undefined && conversation.listeners === 0) this.hosts.work.unwatchAgents(id);
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
    this.draft.clear();
    await Promise.all([this.hosts.chat.close(), this.hosts.work.close()]);
    for (const conversation of this.conversations.values()) conversation.flush();
    this.hostSinks.clear();
    await this.windowsWrite;
    await this.composerStore.settled();
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

  /** The forks this service holds of conversation `id` name its new title, or (null) that it was deleted. */
  private originChanged(id: string, title: string | null): void {
    for (const c of this.conversations.values()) {
      if (c.info.forkOf?.id === id) c.setInfo({ forkOf: { id, title } });
    }
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
    return new Conversation(info, {
      needles: () => this.needles(),
      flushMs: this.opts.flushMs,
      noFolder: (cwd) => this.isScratchFolder(cwd),
    });
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
    conversation.seed(detail.transcript, detail.turns, detail.transcriptError ?? null, detail.contextUsed ?? null);
    await this.loadMarks(conversation);
    await this.loadQueue(conversation);
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
    conversation.seed(detail.transcript, detail.turns, detail.transcriptError ?? null, detail.contextUsed ?? null);
  }

  // ── the queue ────────────────────────────────────────────────────────────────────────────────────────────────────

  /** The conversation's queue, made empty on first use. */
  private queueOf(id: string): ConversationQueue {
    let queue = this.queues.get(id);
    if (queue === undefined) {
      queue = new ConversationQueue();
      this.queues.set(id, queue);
    }
    return queue;
  }

  /**
   * The rows a conversation left queued in an earlier app run, held: nothing typed then is sent without the user. The
   * file's thinking level becomes the conversation's.
   */
  private async loadQueue(conversation: Conversation): Promise<void> {
    const id = conversation.info.id;
    const record = await this.composerStore.read(id);
    if (!this.queues.has(id) && record.queue.length > 0) {
      this.queues.set(id, new ConversationQueue(record.queue, { why: 'restart' }));
    }
    if (record.thinking !== undefined) conversation.info = { ...conversation.info, thinking: record.thinking };
    const queue = this.queues.get(id);
    if (queue !== undefined) conversation.queueChanged(queue.view());
  }

  /** A deleted conversation's queue and its composer file go with it. */
  private async forgetQueue(id: string): Promise<void> {
    this.queues.delete(id);
    await this.composerStore.remove(id).catch(() => {});
  }

  private enqueue(
    conversation: Conversation,
    text: string,
    attachments: readonly AttachmentInput[],
    plan: boolean | undefined,
  ): { queued: string } {
    const id = conversation.info.id;
    if (this.deleting.has(id)) throw chatRouteError('conversation-busy');
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    const queue = this.queueOf(id);
    const row = queue.add({
      text,
      attachments: [...attachments],
      ...(plan === undefined ? {} : { plan }),
      queuedAt: this.now(),
      ...(this.hosts[conversation.info.mode].opOf(id) === null ? {} : { unheld: true as const }),
    });
    if (row === null) throw new ApiError('E_CONFLICT', QUEUE_FULL_MESSAGE, 'queue-full');
    this.publishQueue(conversation, queue, true);
    return { queued: row.id };
  }

  /**
   * ⌘Enter while busy: the text goes to the running answer. Null when no prompt took it and the conversation is idle
   * now (the caller starts a new answer); a refused or late one while busy joins the queue. A steer needs no place in
   * the list: with the list full, a late reply is waited for, and a refusal while busy is 409 `steer-busy`.
   */
  private async steerText(
    conversation: Conversation,
    text: string,
    plan: boolean | undefined,
  ): Promise<PromptReply | null> {
    const id = conversation.info.id;
    const outcome = await this.hosts.work.steer(id, text);
    if (outcome === 'sent') return this.steerTaken(conversation, text);
    if (outcome === 'not-now') return this.steerRefused(conversation, text, plan);
    if (this.queues.get(id)?.full === true) {
      return (await outcome.late) ? this.steerTaken(conversation, text) : this.steerRefused(conversation, text, plan);
    }
    const queued = this.enqueue(conversation, text, [], plan);
    const queue = this.queueOf(id);
    queue.setSteering(queued.queued, true);
    this.lateSteer(conversation, queue, queued.queued, text, outcome.late);
    return { ...queued, steer: 'late' };
  }

  private steerTaken(conversation: Conversation, text: string): PromptReply {
    if (this.conversations.get(conversation.info.id) === conversation) conversation.steerSent(text);
    return { steered: true };
  }

  /** ⌘Enter text no prompt took: null when no answer runs (a new one starts), else it joins the queue. */
  private steerRefused(conversation: Conversation, text: string, plan: boolean | undefined): PromptReply | null {
    const id = conversation.info.id;
    if (!this.hosts.work.isBusy(id)) return null;
    if (this.queues.get(id)?.full === true) throw ApiError.conflict(STEER_BUSY, 'steer-busy');
    return { ...this.enqueue(conversation, text, [], plan), steer: 'refused' };
  }

  /** The agent took a row's Send now: its text shows in the answer and the row leaves. */
  private steered(conversation: Conversation, queue: ConversationQueue, rowId: string, text: string): void {
    const id = conversation.info.id;
    if (this.conversations.get(id) !== conversation) return;
    conversation.steerSent(text);
    if (this.queues.get(id) === queue) {
      // A row edited while the reply was late holds words the agent never got: it stays as a message of its own.
      if (queue.row(rowId)?.text === text) {
        queue.remove(rowId);
        this.publishQueue(conversation, queue, true);
      } else {
        queue.setSteering(rowId, false);
      }
    }
    void this.drainQueue(id, false);
  }

  /** A Send now with no reply in time: the row waits for it; a late yes moves it into the answer, a no frees it. */
  private lateSteer(
    conversation: Conversation,
    queue: ConversationQueue,
    rowId: string,
    text: string,
    late: Promise<boolean>,
  ): void {
    void late.then((took) => {
      if (took) {
        this.steered(conversation, queue, rowId, text);
        return;
      }
      if (queue.setSteering(rowId, false)) void this.drainQueue(conversation.info.id, false);
    });
  }

  /** The page gets the new view; `store`: the rows changed, so the composer file is written again. */
  private publishQueue(conversation: Conversation, queue: ConversationQueue, store: boolean): void {
    conversation.queueChanged(queue.view());
    if (store) {
      const id = conversation.info.id;
      // The rows stay in memory when the write fails: only a restart would lose them.
      this.composerStore.write(id, { queue: queue.stored() }).catch(() => {});
    }
  }

  /** A route changed the rows or an edit: published, stored, and the next row goes if it is due. */
  private queueChanged(conversation: Conversation, queue: ConversationQueue): void {
    this.publishQueue(conversation, queue, true);
    void this.drainQueue(conversation.info.id, false);
  }

  private holdQueue(conversation: Conversation, hold: QueueHold): void {
    const queue = this.queues.get(conversation.info.id);
    if (queue !== undefined && queue.holdFor(hold)) this.publishQueue(conversation, queue, false);
  }

  /**
   * A rewind ended. Done or partial: the thread is cut before the plan's entry (not for files only), Chat's address
   * ledger follows the kept entries, Work stores the rule; then the page hears the end (a partial one names the files
   * not put back), and the context is read again. The queue sends its next row after a done rewind and holds after any
   * other end.
   */
  private async rewindEnded(conversation: Conversation, plan: RewindHeld, mode: RewindMode, end: RewindHostEnd) {
    const id = conversation.info.id;
    const host = this.hosts[conversation.info.mode];
    this.updateLive();
    this.scheduleState();
    this.busyChanged();
    if (end.outcome !== 'done' && end.outcome !== 'partial') {
      conversation.rewound({
        outcome: end.outcome === 'failed' && end.failure === 'load-slow' ? 'load-slow' : end.outcome,
      });
      this.answerEnded(conversation, 'failed');
      return;
    }
    if (mode !== 'files') {
      const kept = conversation.cut(plan.entry, hostView(host, id));
      host.rebuildLedger(id, kept);
    }
    if (conversation.info.mode === 'work') {
      const line: PermissionLine = {
        kind: 'rewind',
        at: this.now(),
        mode,
        files: end.restored,
        ...(mode === 'files' ? {} : { from: plan.from }),
      };
      conversation.permissionMark(markOf(line, this.settingsPlace(conversation)));
      await this.permissionLog.append(id, line).catch(() => {});
    }
    if (end.outcome === 'done') {
      conversation.rewound({ outcome: 'done', mode, restored: end.restored, entry: plan.entry });
    } else {
      const needles = needleList(this.needles());
      const failed = planFiles(
        end.failed.map((path) => ({ path, state: 'clean' as const })),
        (path) => visibleText(redactText(path, needles)),
      );
      conversation.rewound({
        outcome: 'partial',
        mode,
        restored: end.restored,
        entry: plan.entry,
        failed: failed.files.map((file) => file.path),
        failedMore: failed.filesMore,
      });
    }
    conversation.info = { ...conversation.info, updatedAt: this.now() };
    this.emitHost({ type: 'conversations-changed' });
    const info = await host.contextInfo(id);
    if (info?.used != null && !host.isBusy(id)) conversation.contextRead(info.used);
    this.answerEnded(conversation, end.outcome === 'done' ? 'completed' : 'failed');
  }

  /** An answer, a wake or a manual compaction ended: a completed one sends the next row, any other holds the list. */
  private answerEnded(
    conversation: Conversation,
    outcome: TurnOutcome | 'compacted' | 'stopped' | 'failed' | 'load-slow' | undefined,
  ): void {
    const queue = this.queues.get(conversation.info.id);
    if (queue === undefined || queue.size === 0) return;
    if (outcome === 'completed' || outcome === 'compacted') void this.drainQueue(conversation.info.id, false);
    else this.holdQueue(conversation, { why: outcome === 'cancelled' || outcome === 'stopped' ? 'stopped' : 'failed' });
  }

  /**
   * Sends the queue's next row through the checks of a typed message, when the conversation is free: no turn, wake,
   * history op or recap, no wake due, no row open for edit, not held. A failed check holds the list with its sentence
   * and the row stays. `user` (Send next): a due wake and the engine-lost wall do not hold it back, as for a
   * message the user types. A conversation busy again by the time the checks end keeps the row for that answer's end.
   * A drain asked for while one checks its row runs once that check ended. The row stays in the queue, out of the list,
   * until its prompt went to the agent.
   */
  private async drainQueue(id: string, user: boolean): Promise<void> {
    if (this.closed) return;
    const asked = this.draining.get(id);
    if (asked !== undefined) {
      asked.again = true;
      asked.user ||= user;
      return;
    }
    const conversation = this.conversations.get(id);
    const queue = this.queues.get(id);
    if (conversation === undefined || queue === undefined) return;
    const host = this.hosts[conversation.info.mode];
    const free = (): boolean =>
      !this.closed &&
      !host.isBusy(id) &&
      host.recapState(id) === null &&
      !this.deleting.has(id) &&
      (user || (!host.wakeDue(id) && !host.goalRuns(id)));
    const row = queue.next();
    if (row === null || !free()) return;
    if (!user && conversation.info.mode === 'work' && host.engineLostTasks()) {
      this.holdQueue(conversation, blockedHold(chatRouteError('engine-lost')));
      return;
    }
    const checking = { again: false, user: false };
    this.draining.set(id, checking);
    let cwd: string | undefined;
    try {
      cwd = await this.sendChecks(conversation, row.attachments, false);
    } catch (error) {
      if (queue.next() === row) this.holdQueue(conversation, blockedHold(error));
      return;
    } finally {
      this.draining.delete(id);
      if (checking.again) queueMicrotask(() => void this.drainQueue(id, checking.user));
    }
    if (queue.next() !== row || !free() || this.conversations.get(id) !== conversation) return;
    let started: { turnId: string };
    try {
      started = this.startTurn(conversation, row.text, cwd, row.attachments, row.plan);
    } catch (error) {
      if (!host.isBusy(id)) this.holdQueue(conversation, blockedHold(error));
      return;
    }
    queue.sending(row.id, started.turnId);
    this.publishQueue(conversation, queue, false);
  }

  /** The turn of the row being sent: its prompt went to the agent (`sent`) and the row leaves, or it never did. */
  private rowStarted(conversation: Conversation, turnId: string, sent: boolean): void {
    const queue = this.queues.get(conversation.info.id);
    if (queue === undefined) return;
    if (sent ? queue.sent(turnId) : queue.notSent(turnId)) this.publishQueue(conversation, queue, sent);
  }

  private startTurn(
    conversation: Conversation,
    text: string,
    cwd?: string,
    attachments: readonly AttachmentInput[] = [],
    plan?: boolean,
  ): { turnId: string } {
    const id = conversation.info.id;
    const host = this.hosts[conversation.info.mode];
    if (host.isBusy(id) || this.deleting.has(id)) throw chatRouteError('conversation-busy');
    // A delete that finished while the prompt read the disk: the agent would load the deleted session back.
    if (this.conversations.get(id) !== conversation) throw ApiError.notFound('No such conversation.');
    const entries = conversation.diskEntries();
    let started;
    try {
      started = host.startTurn(
        id,
        text,
        {
          modelId: conversation.info.modelId,
          ...(cwd === undefined ? {} : { cwd }),
          ...(entries === undefined ? {} : { entries }),
          ...(conversation.info.thinking === undefined ? {} : { thinking: conversation.info.thinking }),
        },
        attachments,
        { plan: plan === true },
      );
    } catch (error) {
      if (error instanceof AgentHostError) throw chatRouteError(error.code);
      throw error;
    }
    conversation.clearNotice('no-plan-mode');
    conversation.clearNotice('question-too-long');
    const before = plan === true ? conversation.modeState() : null;
    if (before !== null && before.mode !== 'plan') {
      this.planAsked.set(id, { turnId: started.turnId, moves: before.moves });
    } else {
      this.planAsked.delete(id);
    }
    conversation.userPrompt(text, attachments.map(attachmentView));
    void started.done.catch(() => {});
    const toolPdf = conversation.info.mode === 'work' ? attachments.find(isToolPdf) : undefined;
    if (toolPdf !== undefined) {
      this.hostStart(conversation, started, pdfToolId(toolPdf.stage, toolPdf.pages));
    }
    this.updateLive();
    this.busyChanged();
    return { turnId: started.turnId };
  }

  private listenerFor(mode: ChatMode): AgentHostListener {
    const conv = (id: string): Conversation | undefined => {
      const c = this.conversations.get(id);
      return c?.info.mode === mode ? c : undefined;
    };
    return {
      onState: () => {
        this.rememberWindows(this.hosts[mode].models());
        this.rememberLevels(this.hosts[mode].models());
        this.scheduleState();
      },
      onUpdate: (id, line) => {
        const c = conv(id);
        if (c !== undefined) c.applyLine(line);
        else if (this.creates > 0) this.keepEarly(id, line);
        // The goal paused, blocked, ran out or ended with no answer running: the rows that waited for it go.
        if (
          c !== undefined &&
          line.xai &&
          line.update.sessionUpdate === 'goal_updated' &&
          !this.hosts[mode].goalRuns(id)
        )
          queueMicrotask(() => void this.drainQueue(id, false));
      },
      onTitle: (id, title) => {
        const c = conv(id);
        const named = resolveTitle(title, null);
        if (c === undefined || named === '') return;
        c.setInfo({ title: named });
        this.originChanged(id, named);
        this.emitHost({ type: 'conversations-changed' });
      },
      onModel: (id, modelId) => conv(id)?.setInfo({ modelId }),
      onCompaction: (id, compaction, end) => conv(id)?.compactionChanged(compaction, end),
      onHookRun: (id, hookRun) => conv(id)?.hookRunChanged(hookRun),
      onRecap: (id, recap, end) => {
        conv(id)?.recapChanged(recap, end);
        // Rows queued while it was written go once it ended, however it ended.
        if (recap === null && conv(id) !== undefined) queueMicrotask(() => void this.drainQueue(id, false));
      },
      onAutoContinue: (id) => conv(id)?.userPrompt(CONTINUE_PROMPT),
      onTurn: (id, turn) => {
        if (turn?.origin === 'wake') {
          this.updateLive();
          this.busyChanged();
        }
        const c = conv(id);
        if (c === undefined || turn === null) return;
        if (turn.phase === 'running') this.rowStarted(c, turn.turnId, true);
        if (turn.phase === 'running' && !c.info.hot && this.hosts[mode].isHot(id)) c.setInfo({ hot: true });
        c.turnChanged(turn);
      },
      onRunState: (id, state, fileTs) => conv(id)?.runState(state, fileTs),
      onRuns: (id, runs) => {
        conv(id)?.runsChanged(runs);
        this.updateLive();
        this.scheduleState();
        this.busyChanged();
      },
      onWork: () => {
        this.updateLive();
        this.scheduleState();
        this.busyChanged();
      },
      onActivity: (id, live) => conv(id)?.activityChanged(live),
      onAgentEntries: (id, event) => conv(id)?.agentEntries(event),
      onTurnEnd: (id, result: TurnResult) => {
        const c = conv(id);
        const asked = this.planAsked.get(id);
        if (asked?.turnId === result.turnId) {
          this.planAsked.delete(id);
          if (c !== undefined && reachedAgent(result) && c.modeState()?.moves === asked.moves) {
            c.notice({ code: 'no-plan-mode', tone: 'warning' });
          }
        }
        if (c !== undefined) {
          this.rowStarted(c, result.turnId, result.unsent !== true);
          // A hook blocked the prompt: the agent stores none of it, so it leaves the thread and its text goes back to
          // the message box (not an automatic `continue`, which nobody typed).
          const blocked = result.outcome === 'hook-stopped' ? c.typedPrompt() : null;
          if (result.unsent === true) c.promptUnsent();
          c.turnEnded(blocked === null || blocked.continued === true ? result : { ...result, blocked: blocked.text });
          if (blocked !== null) c.dropTypedPrompt(hostView(this.hosts[mode], id));
          c.info = { ...c.info, updatedAt: this.now(), hot: this.hosts[mode].isHot(id) };
          this.emitHost({ type: 'conversations-changed' });
        }
        this.updateLive();
        this.scheduleState();
        this.busyChanged();
        // After the host's own end steps of this turn: the next row's turn starts from a settled host.
        if (c !== undefined) queueMicrotask(() => this.answerEnded(c, result.outcome));
      },
      // The rows that waited for the wake go as after a turn's end; a held list stays held.
      onWakeDropped: (id) => {
        if (conv(id) !== undefined) queueMicrotask(() => void this.drainQueue(id, false));
      },
      onInteraction: (id, interaction) => {
        conv(id)?.interaction(interaction);
        this.watchPage();
        this.scheduleState();
      },
      onInteractionClosed: (id, interactionId, reason) => {
        conv(id)?.interactionClosed(interactionId, reason);
        this.watchPage();
        this.scheduleState();
      },
      onMcpClosed: (id, closed) => this.mcpClosed(id, closed),
      onMcpDone: (id, server) => this.mcpClosed(id, { server, mode: 'url', outcome: 'completed' }),
      onNotice: (id, notice) => conv(id)?.notice(notice),
      onNoticeCleared: (id, code) => conv(id)?.clearNotice(code),
      onSessionReady: (id) => void this.sessionReady(id),
      onCommands: (id, commands) => conv(id)?.commandsChanged(commands),
      onForkLanded: (from, copy) => this.lateFork(mode, from, copy),
      onModeAllowed: (id, mark) => this.modeAllowed(id, mark),
      onModeMarked: (id, mark) => this.modeMarked(id, mark),
      onModeWithdrawn: (id, mark) => this.modeWithdrawn(id, mark),
      onRuleDenied: (id, denied) => void this.ruleDenied(id, denied),
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
    if (reseed)
      conversation.seed(detail.transcript, detail.turns, detail.transcriptError ?? null, detail.contextUsed ?? null);
    conversation.resync(hostView(host, id));
  }

  private installDone(result: 'idle' | 'failed'): void {
    this.install = result;
    this.scheduleState();
  }

  /**
   * The live watcher (it feeds the endpoint check) is open while a turn, wake, run, helper, command or monitor is
   * live, or a host stream is attached.
   */
  private updateLive(): void {
    const live = this.opts.live;
    if (live === undefined || this.closed) return;
    const wanted =
      this.hostSinks.size > 0 ||
      (['chat', 'work'] as const).some((mode) => {
        const state = this.hosts[mode].modeState();
        return state.turnsInFlight > 0 || state.background > 0 || state.tasks > 0;
      });
    if (wanted && this.liveOff === null) {
      this.liveOff = live.subscribe((event) => {
        if (event.type === 'runtime') this.onRuntimeEvent();
      });
    } else if (!wanted && this.liveOff !== null) {
      this.liveOff();
      this.liveOff = null;
    }
  }

  private busyChanged(): void {
    try {
      this.opts.onBusyChange?.();
    } catch {
      // A notice to MAIN never changes a turn.
    }
  }

  /** A running agent's list replaces each listed model's levels; other models keep theirs. */
  private rememberLevels(models: ReturnType<AgentHost['models']>): void {
    if (models === null) return;
    for (const m of models) this.levels.set(m.modelId, m.thinking);
  }

  /**
   * A served model's levels: an agent's list, else the levels of its family, which the engine lists as the model's
   * `reasoningEfforts` (the endpoint's `modelType` is the row's `model_family`); none when the endpoint names no family.
   */
  private thinkingOf(model: InferenceEndpoint['models'][number]): ModelThinking | null | undefined {
    if (this.levels.has(model.name)) return this.levels.get(model.name);
    return model.modelType === undefined ? undefined : levelsFor(model.modelType);
  }

  /** A running agent's list replaces each listed model's window (a null drops it); other models keep theirs. */
  private rememberWindows(models: ReturnType<AgentHost['models']>): void {
    if (models === null) return;
    let changed = false;
    for (const { modelId, contextTokens } of models) {
      if (contextTokens === null ? !this.windows.has(modelId) : this.windows.get(modelId) === contextTokens) continue;
      if (contextTokens === null) this.windows.delete(modelId);
      else this.windows.set(modelId, contextTokens);
      changed = true;
    }
    if (!changed) return;
    const path = this.opts.paths.modelWindows;
    const root = this.opts.paths.root;
    this.windowsWrite = this.windowsWrite
      .then(async () => {
        await mkdir(root, { recursive: true, mode: 0o700 });
        await writeModelWindows(path, this.windows);
      })
      .catch(() => {});
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

  /** The loads the app's engine runs now, from its heartbeat. */
  private async engineLoading(pid: number): Promise<RuntimeLive['processes'][number]['loading']> {
    const live = await this.readLive();
    return live?.processes.find((p) => p.status === 'running' && p.pid === pid)?.loading ?? [];
  }

  private async computeState(): Promise<ChatHostState> {
    const endpoint = await this.endpoint();
    const live = await this.readLive();
    // Only the endpoint file's engine: the app never selects on a terminal `mlx serve` or `mlx launch` engine.
    const running = live?.processes.filter((p) => p.status === 'running' && p.pid === endpoint?.pid) ?? [];
    const loading = running.flatMap((p) => p.loading).sort((a, b) => a.startedAt - b.startedAt);
    const supervisor = live?.supervisor?.state;
    const inference: ChatHostState['inference'] =
      endpoint !== null
        ? loading.length > 0
          ? 'loading'
          : 'ready'
        : supervisor === 'starting' || supervisor === 'restarting'
          ? 'starting'
          : 'stopped';
    const served = (endpoint?.models ?? []).filter((m) => !isDecisionModel(m));
    if (endpoint !== null && endpoint.pid !== this.sizesPid) {
      this.sizes.clear();
      this.sizesPid = endpoint.pid;
    }
    const sizes = await Promise.all(served.map((m) => this.diskSize(m.path)));
    const inMemory = new Map<string, RuntimeLiveModel>();
    const unloaded = this.unloaded;
    for (const process of running) {
      const stale = unloaded !== null && process.pid === unloaded.pid && process.ts < unloaded.at;
      if (unloaded !== null && process.pid === unloaded.pid && !stale) this.unloaded = null;
      for (const model of process.models)
        if (!stale || !unloaded.names.has(model.name)) inMemory.set(model.name, model);
    }
    const models = served.map((m, index) => {
      const thinking = this.thinkingOf(m);
      return {
        id: m.name,
        base: m.base,
        variant: m.variant,
        format: m.format,
        contextTokens: this.windows.get(m.name) ?? null,
        sizeBytes: sizes[index]!,
        images: inMemory.get(m.name)?.images ?? m.images ?? false,
        family: m.modelType ?? null,
        ...(thinking === undefined ? {} : { thinking }),
      };
    });
    const resident: ChatHostState['resident'] = [];
    const answers: ChatHostState['answers'] = [];
    for (const [index, m] of served.entries()) {
      const live = inMemory.get(m.name);
      if (live === undefined) continue;
      resident.push({ modelId: m.name, sizeBytes: sizes[index]! });
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
    const requests: ChatHostState['requests'] = running
      .flatMap((p) => p.turns)
      .sort((a, b) => a.startedAt - b.startedAt)
      .map((turn) => ({
        session: turn.clientSession,
        modelId: turn.model,
        phase: turn.phase,
        startedAt: turn.startedAt,
        prefillDone: turn.prefillDone,
        prefillTotal: turn.prefillTotal,
        generated: turn.generatedTokens,
      }));
    const agent = this.agentState();
    const appEnds = [...(await this.appEnds)];
    const active: ChatHostState['active'] = [];
    for (const mode of ['chat', 'work'] as const) {
      for (const a of this.hosts[mode].active()) {
        active.push({
          conversationId: a.sessionId,
          mode,
          busy: a.busy,
          waiting: a.waiting,
          asks: a.asks,
          background: a.background,
          helpers: a.helpers,
          tasks: a.tasks,
          op: a.op,
        });
      }
    }
    return {
      inference,
      loading: loading.map((load) => ({
        modelId: load.model,
        startedAt: load.startedAt,
        phase: load.phase,
        stepsDone: load.stepsDone,
        steps: load.steps,
      })),
      resident,
      answers,
      requests,
      models,
      defaultModelId: resident[0]?.modelId ?? models[0]?.id ?? null,
      agent,
      modes: { chat: this.modeState('chat'), work: this.modeState('work') },
      providers: this.providers.summary(),
      active,
      permissionRaised: [...this.permissionModes].map(([id, mode]) => ({ id, mode })),
      permissionDraft: this.draft.shown(),
      tempRoot: typedTop(this.tempRoots[0]),
      tempTop: this.tempRoots[0]?.top ?? null,
      scratchRoot: this.scratch,
      pdfTool: { chat: false, work: await this.pdfToolWork() },
      appEnds,
    };
  }

  /** The size on disk of a model path, read once per path; null when it cannot be read. */
  private diskSize(path: string): Promise<number | null> {
    let size = this.sizes.get(path);
    if (size === undefined) {
      size = (this.opts.modelSize ?? modelSize)(path).catch(() => null);
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
      canInstall: this.opts.installAgent !== undefined,
    };
  }
}

/**
 * Bytes of a model path: every part the loader opens for a split GGUF, else {@link pathSize}. Null for a split set the
 * loader cannot open, so it is never the smallest build.
 */
async function modelSize(path: string): Promise<number | null> {
  const name = basename(path);
  const split = parseGgufFileName(name);
  if (split === undefined || split.stem.length === name.length - '.gguf'.length) return pathSize(path, 0);
  const dir = dirname(path);
  const parts = ggufSplitPartNames(name, await readdir(dir));
  if (parts === undefined) return null;
  const infos = await Promise.all(parts.map((part) => stat(join(dir, part))));
  if (!infos.every((info) => info.isFile())) return null;
  return infos.reduce((total, info) => total + info.size, 0);
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
    ...(s.kind === 'fork' && s.parent !== null
      ? { forkOf: { id: s.parent.id, title: s.parent.title === null ? null : agentTitle(s.parent.title) } }
      : {}),
  };
}
