/**
 * One agent process per mode, shared by all conversations of that mode: lazy start, start-up self-check, the
 * post-create guard, turns with their caps, permission answers, idle stop, stale restart and crash handling.
 *
 * Every request field is built here. Agent text (stderr, error messages, update payloads) is never logged and never
 * becomes a failure message: failures carry fixed text per code.
 */

import { randomUUID } from 'node:crypto';
import { unlinkSync } from 'node:fs';
import { realpath } from 'node:fs/promises';
import { homedir, userInfo } from 'node:os';
import { basename, join } from 'node:path';

import { compareReleaseTags, MLX_AGENT_MIN_RELEASE, type AgentInstallProbe } from '@mlx-node/agent/binary';

import { readRunState, type RunState } from '../../agent-sessions/run-state.js';
import {
  findRun,
  NOTICE_TEXT_CHARS,
  runsNamed,
  truncateField,
  WAKE_PROMPT_ID,
  workflowCommand,
  type UpdateLine,
} from '../../agent-sessions/transcript.js';
import type { TranscriptEntry } from '../../agent-sessions/types.js';
import {
  ACP_TIMEOUTS,
  AcpRpcError,
  AcpSoftTimeoutError,
  METHOD_NOT_FOUND,
  type AcpClient,
  type ProtocolErrorReason,
} from '../acp/client.js';
import {
  isRecord,
  readAskSignal,
  readCommandList,
  readCommandsUpdate,
  readCompaction,
  readEffortConfig,
  readForkResult,
  readHookRun,
  readInitialize,
  readInterject,
  readModelSwitch,
  readMonitorEvent,
  readPermissionRequest,
  readPromptResult,
  readRecapAck,
  readRewindResult,
  readRunningSubagents,
  readScheduleDelete,
  readServerStatus,
  readSessionInfo,
  readSessionReply,
  readSessionUpdate,
  readSpawnedChild,
  readSubagentCancel,
  readSubagentMessage,
  readSubagentProgress,
  readTitle,
  readWorkflowCall,
  readWorkflowStop,
  readTaskKill,
  readWorkflowUpdate,
  type CompactionUpdate,
  type InitializeFacts,
  type PermissionRequest,
  type SessionInfoFacts,
  type WorkflowUpdate,
} from '../acp/messages.js';
import { AgentProcess, spawnAgent, type AgentExit, type SpawnAgent } from '../acp/process.js';
import type { AttachmentInput } from '../attachment-rule.js';
import {
  allowedCommands,
  checkCommand,
  readCommands,
  withoutAppWorkflow,
  type AgentCommand,
  type AppWorkflow,
} from '../commands.js';
import {
  ensureChatDirs,
  ensureWorkProcessDir,
  pluginDir,
  removeChatPromptHistory,
  removeChatSearchIndex,
  removeStaleLedgers,
  writeFileAtomic,
  writeWorkGuard,
  type ChatPaths,
} from '../paths.js';
import type { ProviderSource } from '../providers/source.js';
import type { ProviderId } from '../providers/types.js';
import { needleList } from '../redact.js';
import { stepLevel } from '../thinking-levels.js';
import { EFFORT_CONFIG_ID, modelThinking } from '../thinking.js';
import {
  CONTINUE_PROMPT,
  type AgentModeState,
  type AgentWatchAnswer,
  type AutoAsk,
  type CompactionState,
  type CompactOutcome,
  type HookRunState,
  type PermissionMark,
  type PermissionMode,
  type ChatFailure,
  type ChatFailureCode,
  type ChatMode,
  type CommandView,
  type ConversationNotice,
  type Interaction,
  type LiveActivity,
  type McpFieldError,
  type McpInteraction,
  type ModelSwitch,
  type ModelThinking,
  type PermissionInteraction,
  type RecapEnd,
  type RecapState,
  type RewindFileState,
  type RewindMode,
  type RunBy,
  type RunView,
  SUBAGENT_MESSAGE_MAX_BYTES,
  type SubagentMessageResult,
  type ThinkingLevel,
  type TurnOutcome,
  type TurnState,
} from '../types.js';
import { parseWebResultText, type WebResult } from '../web/result-text.js';
import { editLanding, type WorkFolderCheck } from '../work-folder.js';
import { ActivityTracker, type ScheduleDeleting, wakeDueOf, type WakeDue } from './activity-tracker.js';
import { AddressLedger, writeLedgerFile } from './address-ledger.js';
import { type AgentEntriesEvent, AgentWatches, readAgentUpdates } from './agent-watch.js';
import { foldName } from './auto-scan.js';
import {
  autoFacts,
  changesFolder,
  decideAuto,
  staysBehind,
  type AutoAskDetail,
  type AutoDecision,
  type AutoFacts,
} from './auto.js';
import { agentPath, type WorkPlace } from './command-flags.js';
import {
  acceptReply,
  LINK_ACCEPT,
  linkInteraction,
  MCP_CANCEL,
  MCP_DECLINE,
  mcpInteraction,
  parseForm,
  readElicitComplete,
  readElicitRequest,
  validateForm,
  type FieldSpec,
  type McpAnswerInput,
} from './elicit.js';
import { endpointFingerprint, sameEndpoint, type EndpointFingerprint, type EndpointReader } from './endpoint.js';
import { chatAgentEnv, workAgentEnv } from './env.js';
import { checkSession, selfCheck, type GuardResult } from './guard.js';
import { liveFilter } from './live-filter.js';
import {
  match as neverMatch,
  NEVER_PER_TURN_MAX,
  neverNote,
  offer as neverOffer,
  ruleText,
  type NeverOffer,
  type NeverRules,
} from './never-rules.js';
import { appReadPdfPaths, installPdfTool, READ_PDF_WORKFLOW } from './pdf-tool.js';
import {
  decideChatPermission,
  noteAnswer,
  pdfLaunchRow,
  planAnswer,
  planInteraction,
  readPdfLaunch,
  readPlanRequest,
  requestTarget,
  accessOf,
  causeOutcome,
  shellCommand,
  showableWhole,
  shownCut,
  turnOutcome,
  workInteraction,
  type PdfFacts,
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
  type PromptMeta,
  promptParams,
  REJECT_ONCE_OPTION_ID,
  WEB_SERVER_NAME,
  WEB_TOOL_KEYS,
  workGuardHookName,
  workGuardManifest,
  workGuardName,
  workSessionParams,
} from './profiles.js';
import {
  questionInteraction,
  questionReply,
  readQuestionRequest,
  type QuestionAnswerInput,
  type QuestionRequest,
} from './questions.js';
import { below, isWinPath, tempRoots, type AllowedRoot } from './roots.js';
import { MODEL_SELECT_MAX_MS, selectModel, type SelectAnswer, type SelectTarget } from './select-model.js';
import { applyWebResult, readWebToolResult } from './web-results.js';

export const AGENT_IDLE_STOP_MS = 600_000;
/** Chat only: a Work turn has no time limit. The time the agent compacts the conversation is not counted. */
export const CHAT_TURN_MAX_MS = 600_000;

/** The end of a Stop run: `failed` when the reply did not come or did not read. */
export type RunStop = 'stopped' | 'rejected' | 'failed' | 'not-live';

/**
 * How a Stop of a command or monitor went. `stopped`: the agent killed it, or it had already exited (its end line
 * follows); `not-live`: the conversation holds no such live task, or the agent no longer knows it; `failed`: no reply
 * in time, or an error reply.
 */
export type TaskStop = 'stopped' | 'not-live' | 'failed';

/**
 * How a Stop of a subagent ended: `stopped` (its end line follows), `not-live` (no such running helper here, or the
 * agent says it already ended or knows none), `failed` (an error reply or none in time).
 */
export type SubagentStop = 'stopped' | 'not-live' | 'failed';

/** How a message to a subagent ended; `gone` when the host holds no such running helper or the agent refused it. */
export type SubagentMessage = SubagentMessageResult | 'gone';

/**
 * How a Send now went. `sent`: the agent took the message for the running answer; `not-now`: no prompt is open (a
 * turn starting, a gap between prompts, a history op, an idle session, Chat) or the agent refused it, and nothing
 * stays in flight; `late`: no reply in time, and `late` settles true when the agent took it after all.
 */
export type SteerOutcome = 'sent' | 'not-now' | { late: Promise<boolean> };

/** Who stops a task: the user's Stop (`clientUi`: the model is told in a wake turn) or a rewind (`teardown`: no wake). */
export type TaskKillSource = 'clientUi' | 'teardown';

/**
 * How a Delete of a schedule went. `deleted`: the agent deleted it (its `scheduled_task_deleted` follows or came);
 * `gone`: the agent holds no such schedule; `failed`: no reply in time, or an error reply.
 */
export type ScheduleDelete = 'deleted' | 'gone' | 'failed';

/** The user's Delete of one schedule in flight: it holds the process, and the page reads its phase. */
interface DeleteRecord {
  phase: ScheduleDeleting;
}

/** After `session/cancel` the prompt reply must come inside this, else the agent broke the protocol. */
export const CANCEL_REPLY_MS = 5_000;
/** Chat guard: how long to wait for `server_status ready` of `mlx_web` after the session reply. */
export const MCP_READY_MS = 10_000;
export const CRASH_WINDOW_MS = 300_000;
export const CRASHES_TO_FAIL = 3;
export const MISMATCHES_TO_BLOCK = 3;
/** A select call with no answer after this shows `wait: 'model'`: a resident model answers in 1–2 ms. */
export const MODEL_WAIT_SHOW_MS = 400;
/** The idle guard's soft deadline for `_x.ai/subagent/list_running`: a miss keeps the helpers until the next timer. */
export const LIST_RUNNING_MS = 5_000;
/** The page gets a session's live facts at most once in this time: eight helpers tick four times a second in all. */
export const ACTIVITY_EVERY_MS = 1_000;
/**
 * The soft deadline of `_x.ai/task/kill`: a miss leaves the agent and its other commands alive, and the Stop fails.
 * A hard deadline would kill the process, and with it the very commands the user did not stop.
 */
export const TASK_KILL_MS = 10_000;
/** The soft deadline of `_x.ai/interject`: the agent answers at once; a miss keeps the agent and the row stays queued. */
export const STEER_MS = 5_000;
/** The soft deadline of `_x.ai/subagent/cancel`: a miss leaves the agent and the helper alive, and the Stop fails. */
export const SUBAGENT_CANCEL_MS = 10_000;
/** The soft deadline of `_x.ai/subagent/message`: a miss leaves the agent alive, and the message may have arrived. */
export const SUBAGENT_MESSAGE_MS = 10_000;
/**
 * A stopped run reads Stopping… at most this long when its drain update never comes: the agent waits 20 s for its
 * agents, and its watcher makes the run `interrupted` at 25 s.
 */
export const RUN_STOP_DRAIN_MS = 30_000;
/** A run's `state.json` is read this long after its last new revision: a burst of updates gives one read. */
export const RUN_STATE_READ_MS = 500;
/** The soft deadline of the Context panel's `_x.ai/session/info`: a miss shows the panel without the parts. */
export const CONTEXT_INFO_MS = 5_000;
/** The soft deadline of `_x.ai/commands/list` after a load: a miss keeps the static list. */
export const COMMANDS_LIST_MS = 10_000;
/** The soft deadline of the thinking level's `session/set_config_option`: a miss gives a notice, the turn goes on. */
export const EFFORT_SET_MS = 10_000;
/** The soft deadline of `_x.ai/scheduler/delete`: a miss leaves the agent and the schedule alive. */
export const SCHEDULE_DELETE_MS = 10_000;
/**
 * The soft deadline of `_x.ai/session/fork`: the agent copies the session on a blocking thread, well under a second for
 * a normal one. A miss leaves the agent alive; a late reply that names the copy still lands it.
 */
export const FORK_MS = 60_000;
/** The soft deadline of the `_x.ai/recap` ack: the agent acks at once and writes the recap after. */
export const RECAP_ACK_MS = 10_000;
/** How long the host waits for the recap's line: the agent's own model call stops at 120 s at most. */
export const RECAP_WAIT_MS = 120_000;
/**
 * The soft deadline of a cold load (`session/load` and its guard) for a history op, a recap or a schedule's Delete: a
 * miss ends that one `load-slow`, the agent and its background work live on, and the late load lands as a load does.
 */
export const LOAD_SOFT_MS = ACP_TIMEOUTS['session/load'];

/** The agent's name of each rewind mode. */
const REWIND_MODES: Readonly<Record<RewindMode, string>> = {
  all: 'all',
  conversation: 'conversation_only',
  files: 'files_only',
};

/** The soft deadline of a rewind's dry run: the agent reads the files it wrote and answers. */
export const REWIND_PREVIEW_MS = 20_000;

/**
 * A rewind's dry run: the files a restore to the target puts back, as the agent sent them (paths relative to the
 * folder, not yet redacted), or `refused` when the agent holds no such prompt.
 */
export type RewindPreview = { files: Array<{ path: string; state: RewindFileState }> } | { refused: true };

/** What a rewind ends with first, and the prompt it goes back to: the state before prompt `target` ran. */
export interface RewindInput {
  target: number;
  mode: RewindMode;
  /** The live commands and monitors the removed part started: stopped as `teardown` before the rewind. */
  tasks: readonly string[];
  /** The schedules the removed part made: deleted before the rewind. */
  schedules: readonly string[];
  /** The workflow runs the removed part launched that still wait or run: stopped (as the user's Stop) before it. */
  runs: readonly string[];
  /** The plan's dry run as the user confirmed it (paths as the agent sent them); none in Chat. */
  files: ReadonlyArray<{ path: string; state: RewindFileState }>;
}

/**
 * How a rewind on the host ended. `done`: the agent rewound (`restored` files put back). `partial`: it rewound, but
 * could not write the `failed` files (paths as the agent sent them) and dropped its copies of them. `changed`: a file
 * the plan showed clean, or did not show, changed outside the agent since: nothing was rewound. `tasks`: a stop or
 * delete of the removed part's work failed, and no rewind was sent. `refused`: the agent said no. `stopped`: the agent
 * exited before its reply. `failed`: the agent could not be started or loaded (`failure`).
 */
export type RewindHostEnd =
  | { outcome: 'done'; restored: number }
  | { outcome: 'partial'; restored: number; failed: string[] }
  | { outcome: 'changed' | 'tasks' | 'refused' | 'stopped' }
  | { outcome: 'failed'; failure: ChatFailureCode };

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
  'load-slow': 'The agent is slow to open this conversation. Try again.',
  'profile-mismatch': 'Chat could not be started safely.',
  'model-unavailable': 'That model is not available.',
  'model-load-failed': 'The model did not load.',
  'cwd-invalid': 'That folder cannot be used.',
  'cwd-mismatch': 'The agent opened another folder.',
  'conversation-busy': 'An answer or a workflow is still running.',
  'engine-busy': 'The engine is busy: an answer, a workflow or a model load runs.',
  'engine-lost': 'The engine restarted. Your next message restarts the agent.',
  'slash-text': 'No such command here.',
  'attachment-refused': 'The attached files were refused.',
  'pictures-full': 'This chat has too many pictures to keep. Start a new chat to send more.',
  'keychain-unavailable': 'The Keychain cannot be reached. Keys cannot be saved.',
  'skill-bundle-corrupt': "The app's skill files are damaged. Reinstall the app.",
};

/** The agent's words on a model it could not use, cut at {@link NOTICE_TEXT_CHARS} with `…`. */
function cutReason(reason: string): string {
  const cut = truncateField(reason.trim(), NOTICE_TEXT_CHARS);
  return cut.truncated ? `${cut.text}…` : cut.text;
}

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
  listRunningMs: number;
  activityEveryMs: number;
  /** The soft deadline of `_x.ai/task/kill`. */
  taskKillMs: number;
  /** The soft deadline of `_x.ai/interject`. */
  steerMs: number;
  /** The soft deadline of `_x.ai/subagent/cancel`. */
  subagentCancelMs: number;
  /** The soft deadline of `_x.ai/subagent/message`. */
  subagentMessageMs: number;
  /** {@link RUN_STOP_DRAIN_MS}. */
  stopDrainMs: number;
  /** {@link RUN_STATE_READ_MS}. */
  runStateMs: number;
  /** The soft deadline of the Context panel's `_x.ai/session/info`. */
  contextInfoMs: number;
  /** The soft deadline of `_x.ai/commands/list`. */
  commandsListMs: number;
  /** The soft deadline of the thinking level's `session/set_config_option`. */
  effortSetMs: number;
  /** The soft deadline of `_x.ai/scheduler/delete`. */
  scheduleDeleteMs: number;
  /** The soft deadline of `_x.ai/session/fork`. */
  forkMs: number;
  /** The soft deadline of a rewind's dry run (`_x.ai/rewind/execute`, `force: false`). */
  rewindPreviewMs: number;
  /** The soft deadline of the `_x.ai/recap` ack. */
  recapAckMs: number;
  /** How long a recap may take before the wait ends with `none`. */
  recapMs: number;
  /** {@link LOAD_SOFT_MS}. */
  loadSoftMs: number;
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
  listRunningMs: LIST_RUNNING_MS,
  activityEveryMs: ACTIVITY_EVERY_MS,
  taskKillMs: TASK_KILL_MS,
  steerMs: STEER_MS,
  subagentCancelMs: SUBAGENT_CANCEL_MS,
  subagentMessageMs: SUBAGENT_MESSAGE_MS,
  stopDrainMs: RUN_STOP_DRAIN_MS,
  runStateMs: RUN_STATE_READ_MS,
  contextInfoMs: CONTEXT_INFO_MS,
  commandsListMs: COMMANDS_LIST_MS,
  effortSetMs: EFFORT_SET_MS,
  scheduleDeleteMs: SCHEDULE_DELETE_MS,
  forkMs: FORK_MS,
  rewindPreviewMs: REWIND_PREVIEW_MS,
  recapAckMs: RECAP_ACK_MS,
  recapMs: RECAP_WAIT_MS,
  loadSoftMs: LOAD_SOFT_MS,
};

export interface TurnResult {
  turnId: string;
  /** Unset when the turn failed before the prompt reached the agent. */
  outcome?: TurnOutcome;
  failure?: ChatFailure;
  /** The turn ended before `session/prompt` was written to the agent. */
  unsent?: true;
  /** Work, an `output-limit` end: length stops in a row with no tool call between. */
  lengthStops?: number;
  /** Automatic continues sent in this turn; set when one was sent. */
  continues?: number;
  /** Work, an `output-limit` end: why no automatic continue was sent. */
  noContinue?: NoContinue;
  /** What the host recorded after the model's length reply. */
  late?: TurnEndCause;
  /** The agent could not use the conversation's model during this turn (live only). */
  model?: ModelSwitch;
}

/**
 * Why a Work turn that ended `output-limit` was not continued: a second length stop with no tool call between, the
 * user's Stop, the engine's endpoint file gone, the agent gone, or the host stale (the engine's address or token
 * changed).
 */
export type NoContinue = 'second-length' | 'stop' | 'endpoint' | 'agent-exit' | 'host-stale';

export interface AgentHostListener {
  onState?(state: AgentModeState): void;
  /** A live update line that passed the live filter, for the transcript reducer. */
  onUpdate?(sessionId: string, line: UpdateLine): void;
  onTitle?(sessionId: string, title: string): void;
  /** The agent switched the model during a turn; `ConversationInfo.modelId` follows. */
  onModel?(sessionId: string, modelId: string): void;
  /** A compaction of the conversation started or changed (its state) or ended (null; `end` for a manual run). */
  onCompaction?(sessionId: string, compaction: CompactionState | null, end?: CompactOutcome): void;
  /** The agent started waiting on a batch of hooks (its state) or stopped (null). */
  onHookRun?(sessionId: string, hookRun: HookRunState | null): void;
  /** A recap the user asked for started (its state) or ended (null, with how). */
  onRecap?(sessionId: string, recap: RecapState | null, end?: RecapEnd): void;
  /** The typed turn, or the wake turn the agent started itself (`origin: 'wake'`); null when neither runs. */
  onTurn?(sessionId: string, turn: TurnState | null): void;
  /** Work: the session's live workflow runs changed. */
  onRuns?(sessionId: string, runs: RunView[]): void;
  /** Work: a run's `state.json`, read after its new revisions: its history and step details for the page. */
  onRunState?(sessionId: string, state: RunState, fileTs: number): void;
  /** Work: a helper, a command or monitor, or a schedule of the session started or ended. */
  onWork?(sessionId: string): void;
  /** Work: the session's live facts changed (a helper's progress); at most once per {@link ACTIVITY_EVERY_MS}. */
  onActivity?(sessionId: string, live: LiveActivity): void;
  /** Work: an Agent view of a window got ops, or the stored list at its agent's end. */
  onAgentEntries?(sessionId: string, event: AgentEntriesEvent): void;
  onTurnEnd?(sessionId: string, result: TurnResult): void;
  /** Work: the agent process exited while a wake of the session was due: that wake never comes, and no turn ended. */
  onWakeDropped?(sessionId: string): void;
  /** Work: the automatic `continue` after a length stop was written to the agent; `n` counts them in the turn. */
  onAutoContinue?(sessionId: string, n: number): void;
  onInteraction?(sessionId: string, interaction: Interaction): void;
  onInteractionClosed?(sessionId: string, id: string, reason: 'answered' | 'cancelled' | 'expired'): void;
  /** Work: an MCP card closed, by any way: what its history line holds (never a typed value). */
  onMcpClosed?(sessionId: string, line: McpClosed): void;
  /** Work: the server of a page the user opened from a link card says it is done; `server`: its own name. */
  onMcpDone?(sessionId: string, server: string): void;
  /** These sessions are no longer loaded in a running process. */
  onCold?(sessionIds: string[]): void;
  /** A parsed `mlx_web` result of a Chat turn. */
  onWebResult?(sessionId: string, result: WebResult): void;
  /** Work: a provider key appeared in tool output. */
  onNotice?(sessionId: string, notice: ConversationNotice): void;
  /** Every notice of this code is gone (a later thinking-level set succeeded). */
  onNoticeCleared?(sessionId: string, code: ConversationNotice['code']): void;
  /** Work: a session passed its guard after `session/new` or `session/load`. */
  onSessionReady?(sessionId: string): void;
  /** The commands the loaded session may send changed (the agent's push, or the pull after a load); null: it exited. */
  onCommands?(sessionId: string, commands: CommandView[] | null): void;
  /** A fork whose reply came after its soft deadline made `copyId` of `sessionId`. */
  onForkLanded?(sessionId: string, copyId: string): void;
  /**
   * Work: Auto or Bypass allows this call. The history line is written before the agent gets the answer; false (not
   * written) asks the mode again, and the request is decided under it.
   */
  onModeAllowed?(sessionId: string, mark: AllowedMark): Promise<boolean>;
  /** Work: the call whose history line was written gets `allow-once` now, in this tick. */
  onModeMarked?(sessionId: string, mark: AllowedMark): void;
  /** Work: the call whose history line was written gets no allow (Stop, the mode moved, Auto's late checks). */
  onModeWithdrawn?(sessionId: string, mark: AllowedMark): void;
  /** Work: a never-allow rule denied a call with no card; `noted`: the agent got the rule's note and goes on. */
  onRuleDenied?(sessionId: string, denied: RuleDeny): void;
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
  /** Work: the login shell's `PATH`, read once at host start (`resolveLoginPath`); a start waits for it. */
  loginPath?: Promise<string | null>;
  /** Work: whether the user's Claude Code setup loads, read at each start; on when absent. */
  claudeCode?: () => boolean;
  /** `GROK_HOME` of the Work agent: the runtime's agent home. Chat's is `paths.chatAgent`. */
  agentHome: string;
  /** Work: the folder the agent stores a session in, as the sessions reader finds it; null when it finds none. */
  sessionDir?: (sessionId: string) => string | null;
  /** Work: the user's home. */
  homeDir?: string;
  /** Work: the folder rule, applied again right before `session/new` and `session/load`. */
  checkCwd?: (path: string) => Promise<WorkFolderCheck>;
  /** Redaction needles for the cards: provider keys (current and retired) and the inference token. */
  needles?: () => string[];
  /** Work: the permission mode of a session, read in the tick that acts. Without it every request is a card. */
  permission?: { modeOf(sessionId: string): PermissionMode; watched(): boolean };
  /** Work: the temp roots Auto covers beside the Work folder; none by default. */
  tempRoots?: readonly AllowedRoot[];
  /** Work: the never-allow rules of the Work folders; without them no rule is checked and no card offers one. */
  never?: NeverRules;
  /** Work: what a `read-pdf` launch's `pdf` is, read for the session that asks; null when it is no PDF of the app. */
  pdfFacts?: (sessionId: string, pdf: string) => Promise<PdfFacts | null>;
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

/** An MCP card that closed: the server's own name, the request's mode, and how it ended. */
export interface McpClosed {
  server: string;
  mode: Extract<PermissionMark, { kind: 'mcp' }>['mode'];
  outcome: Extract<PermissionMark, { kind: 'mcp' }>['outcome'];
}

/** Who a request is answered for: the typed turn, a live workflow run, or the wake turn. */
type Owner = { kind: 'turn'; turn: TurnRecord } | { kind: 'run'; runId: string } | { kind: 'wake'; wake: WakeRecord };

interface OpenInteraction {
  interaction: Interaction;
  /** Answers the agent's request with this JSON-RPC result. */
  settle: (result: unknown) => void;
  /** The result that withdraws the request. */
  withdrawn: unknown;
  owner: Owner;
  /** The session whose call this is: the requester (a helper) or the session itself. */
  callSession: string;
  /** The agent's own tool call id, as its `interaction_resolved` names it. */
  callId: string;
  /** The call can go on after it reports its end ({@link staysBehind}). */
  stays: boolean;
}

/** A turn the agent started with no prompt open: from its first content update to its `turn_completed`. */
interface WakeRecord {
  turnId: string;
  startedAt: number;
  phase: 'running' | 'cancelling';
  /** The run whose wake the host took this one for; null for any other wake. */
  runId: string | null;
}

/**
 * A run's wake the agent queued and has not started (grok `run_loop.rs` `WorkflowCompletionTurn`). The agent queues
 * one at a time: while it is queued or runs, another run's end gets none, and that wake's turn reports it.
 */
interface RunWake {
  runId: string;
  /** The run's newest revision: the agent names the wake by the revision its last update had. */
  revision: number;
  /** The agent's stamp of the update that made the run reportable; null when its line had none. */
  since: number | null;
  /** A command's or helper's wake was due before it: the agent starts that one first. */
  behind: boolean;
}

/** Calls Auto allowed for an owner since the user's last allow; `approvalEpoch` goes up at each allow by the user. */
interface ApprovalCount {
  approvals: number;
  approvalEpoch: number;
}

type AllowedMark = Extract<PermissionMark, { kind: 'allowed' }>;

/** A call denied with a note: what its history line and its mark hold. `child`: the helper session that asked. */
export interface NoteDeny {
  toolCallId: string;
  access: PermissionInteraction['access'];
  /** The card as it showed, `options: []`. */
  card: PermissionInteraction;
  child?: string;
}

/**
 * A call a never-allow rule denied, or "Never allow" on its card: the rule (null when the folder already holds the most
 * rules), the words its note named, and whether the agent got that note (a plain deny ends the answer).
 */
export interface RuleDeny extends Omit<NoteDeny, 'card'> {
  /** The card as it showed or would have shown, `options: []`; null for a command too long to show. */
  card: PermissionInteraction | null;
  ruleId: string | null;
  words: string;
  noted: boolean;
}

/** An operation on a conversation's history that runs with no turn. */
export type HistoryOp = 'rewind' | 'compact' | 'fork';

/** A history op in flight. `sent`: its request went to the agent. `stopping`: the user pressed Stop. */
interface OpRecord {
  kind: HistoryOp;
  sent: boolean;
  stopping: boolean;
}

/** A session loaded in the running process and past its guard, or the code of the guard that refused it. */
type Loaded = { running: Running } | { refused: ChatFailureCode };

/** What asks to start or load: on a stale process it does not count as work in flight. */
type Caller = TurnRecord | OpRecord | RecapWait | DeleteRecord;

/** A request whose mode changed during a wait: it is decided again under the mode it has now. */
const MOVED: unique symbol = Symbol('moved');
const MODE_ROUNDS = 8;
const ALLOW_ONCE_ANSWER = { outcome: { outcome: 'selected', optionId: ALLOW_ONCE_OPTION_ID } };

const PERMISSION_CANCELLED = { outcome: { outcome: 'cancelled' } };
const PLAN_CANCELLED = { outcome: 'cancelled' };
/** The agent reads it as "no answer: go on with your best judgment". */
const QUESTION_CANCELLED = { outcome: 'cancelled' };
/** Question calls one answer may make; the next is answered "no answer" with no card. */
export const ASKS_PER_TURN_MAX = 3;
/** Opened link requests a session keeps waiting for their server's "done". */
const LINKS_OPENED_MAX = 32;

interface TurnRecord {
  turnId: string;
  phase: TurnState['phase'];
  startedAt: number;
  wait: TurnState['wait'];
  /** The model select call in flight. */
  select: AbortController | null;
  recorded: TurnEndCause | null;
  /** The agent's `model_auto_switched` during this turn. */
  model: ModelSwitch | null;
  /** Calls allowed in this turn since the user's last allow; Auto counts a call at its decision. */
  approvals: number;
  /** Goes up at each allow by the user: a count given back after it would be counted twice. */
  approvalEpoch: number;
  /** Chat only. */
  capTimer: ReturnType<typeof setTimeout> | null;
  /** Chat only: the cap's time left when `capTimer` was armed at `capFrom`, or since a compaction paused it. */
  capLeft: number | null;
  capFrom: number;
  graceTimer: ReturnType<typeof setTimeout> | null;
  /** Ends the turn before the prompt was sent (cancel during start). */
  abort: ((result: TurnResult) => void) | null;
  /** Work: a length reply came and no prompt is in flight; the host decides whether to continue. */
  between: boolean;
  /** Wakes the wait for the agent's end line between two prompts. */
  gap: (() => void) | null;
  /** A live tool call of the session since the last prompt was written. */
  toolSeen: boolean;
  /** The agent's `turn_completed` of the last prompt arrived. */
  endSeen: boolean;
  /** The last `session/prompt` written: its client and request id. */
  prompt: { client: AcpClient; id: number } | null;
  /** Length stops in a row with no tool call between. */
  lengthStops: number;
  /** Automatic continues sent. */
  continues: number;
  /** Work: the prompt is a typed `/workflow pause|resume|stop <name>`. */
  runCommand: ReturnType<typeof workflowCommand>;
}

/** A workflow run the host holds; one object from its first `active` until it leaves the ended list. */
interface RunRecord {
  name: string;
  status: string;
  stopping: boolean;
  /** Who sent the stop in flight. */
  stopBy: 'you' | 'engine' | null;
  stopTimer: ReturnType<typeof setTimeout> | null;
  by: RunBy | null;
  /** The pause or stop of a `workflow` call of the main agent that names this run and has not ended. */
  toolVerb: 'stop' | 'pause' | null;
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
  /** The `session/load` of a soft load while it has no reply: a turn that waits on it gives it the hard deadline. */
  loadCall: { client: AcpClient; id: number } | null;
  /** A turn waits on the load in flight: a soft `session/load` sent after that gets the hard deadline as it goes. */
  loadHard: boolean;
  ledger: AddressLedger;
  /** Chat: where the host keeps `ledger` for this session's `mlx_web` server; null in Work. */
  ledgerFile: string | null;
  /** The last write of `ledgerFile` failed: the file may miss members, so the next permission request writes it. */
  ledgerDirty: boolean;
  /** The `mlx_web` keys this session's server was started with: a key failure it reports is about these keys. */
  web: ChatWebEnv;
  turn: TurnRecord | null;
  interactions: Map<string, OpenInteraction>;
  /** Work: providers whose key was already reported printed in this session. */
  printed: Set<ProviderId>;
  /** Work: runs whose last `workflow_updated` is `active`, by run id. */
  runs: Map<string, RunRecord>;
  /** Work: the newest {@link ENDED_RUNS_KEPT} runs that left `active` in this process, oldest first. */
  endedRuns: Map<string, RunRecord>;
  /** Every run of this process by id: its name. */
  runNames: Map<string, string>;
  /** The main agent's `workflow` calls that have not ended, by tool call id. */
  runCalls: Map<string, { verb: 'stop' | 'pause' | null; runId: string | null }>;
  /** The highest `workflow_updated` revision seen per run: a lower one changes nothing. */
  revisions: Map<string, number>;
  wake: WakeRecord | null;
  /**
   * A wake the agent will start and has not started yet: a command, monitor or helper ended with `will_wake`. A stale
   * restart now would drop the wake, so it waits for the wake's end (or the idle stop when the agent sends none).
   */
  wakeDue: WakeDue | null;
  /** The run wake the agent queued; held the same way, and it names the wake row before its `turn_completed` does. */
  runWake: RunWake | null;
  /** Work: the pending `state.json` read of each run, by run id. */
  runStateTimers: Map<string, ReturnType<typeof setTimeout>>;
  /** Work: the helpers, commands, monitors and schedules that run in the process now. */
  tracker: ActivityTracker;
  /** The live facts changed and go to the page when this fires. */
  activityTimer: ReturnType<typeof setTimeout> | null;
  /** Auto's count for the asks of the runs and the wake turn. */
  bg: ApprovalCount;
  /** Work: the user picked "allow all edits" in this process: the agent no longer asks for edits. */
  editsGranted: boolean;
  /** Work: the user picked an "always" option: the agent may run calls with no request. */
  granted: boolean;
  /** The agent compacts the conversation now. */
  compaction: CompactionState | null;
  /**
   * The batch of hooks the agent waits on now: its `hook_execution` names the same event, tool and prompt id.
   * `agentMs`: the agent's stamp of its start; a text chunk stamped later is the model's next output.
   */
  hookRun: { state: HookRunState; promptId: string | null; agentMs: number | null } | null;
  /** A history op runs: the session is busy (no prompt) and holds the engine. */
  op: OpRecord | null;
  /** The agent's command list for this load, from its push or the pull after the load; null until one came. */
  commands: AgentCommand[] | null;
  /** Work: the pull after the load; settles (never rejects) when it answered or missed. */
  commandsPull: Promise<void> | null;
  /** A recap the user asked for is being written; null while none is. */
  recap: RecapWait | null;
  /** Work: the user's Deletes of schedules in flight, by schedule id. */
  deleting: Map<string, DeleteRecord>;
  /** Work: the goal's status as the agent's last `goal_updated` line of this process said; null with none. */
  goal: string | null;
  /**
   * The `reasoning_effort` option ids the agent offers the session and the one it holds, as its last session reply,
   * `config_option_update` or set reply said; null when not known (none yet, or a model change since).
   */
  efforts: string[] | null;
  effort: string | null;
}

/** The wait for one recap: its start, the watchdog, and the end its `done` resolves with. */
interface RecapWait {
  startedAt: number;
  timer: ReturnType<typeof setTimeout> | null;
  settle: (end: RecapEnd) => void;
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
  models: NonNullable<InitializeFacts['models']>;
  embeddedContext: boolean;
  launch: AgentLaunch;
  /** Work: the hook name of this start's guard; a shell request must name exactly this one. */
  guardHook: string | null;
  /** Work: this process answered `_x.ai/subagent/message` as not found (an agent without the feature on). */
  messagesOff: boolean;
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
  /** The conversation's thinking level, set in the agent before a prompt; absent: the model's default. */
  thinking?: ThinkingLevel;
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
  /** `session/new` calls in flight: a `session_start` hook line of a session not known yet belongs to one of them. */
  private sessionsNew = 0;
  /** Sessions not known yet whose `session_start` batch was handed on: the batch's notes follow it. */
  private readonly startedEarly = new Set<string>();
  /**
   * The engine restarted under the running process: it can never reach the engine again. Its commands and monitors
   * run on until the user's next message, which restarts it.
   */
  private engineLost = false;
  private failure: ChatFailure | null = null;
  /** Work: the PDF tool's files were written at the last start (false: a file of the person's holds a name); null before one. */
  private pdfToolOpen: boolean | null = null;
  /** Work: the app's `read-pdf`, as the last agent start found it in the agent home; null when that file is not the app's. */
  private appWorkflow: AppWorkflow | null = null;
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
  /** Work: child session id → its parent, from the parent's `subagent_spawned`; until the process exits. */
  private readonly children = new Map<string, HostSession>();
  /** Work: each running agent's own live lines, and the Agent views that watch one. */
  private readonly watches: AgentWatches | null;
  /**
   * Work permission requests from their arrival to their answer, by asking session and tool call id. `gone`: the agent
   * stopped waiting before the card opened (Auto's checks, an edit's landing), so the card must not open.
   */
  private readonly deciding = new Map<string, { gone: boolean }>();
  /** Work: question calls of each typed turn or wake turn so far. */
  private readonly questionCalls = new WeakMap<TurnRecord | WakeRecord, number>();
  /** Work: the agent's own request behind each open question card; its texts are the reply's keys. */
  private readonly questionRequests = new WeakMap<Interaction, QuestionRequest>();
  /** Work: the words a never-allow answer to an open command card would keep, as the command holds them. */
  private readonly neverOffers = new WeakMap<Interaction, NeverOffer>();
  /** Work: never-allow rule hits of each typed turn, wake turn or run so far. */
  private readonly neverHits = new WeakMap<object, number>();
  /**
   * Work: each open MCP card's server name and form (the option values the server reads stay here), or, for a link,
   * its `elicitationId`.
   */
  private readonly mcpForms = new WeakMap<
    Interaction,
    { server: string; specs: FieldSpec[] | null; elicitationId: string | null }
  >();
  /**
   * Work: by session, the link requests whose page the user opened in this process's life, oldest first, each by its
   * server's own name and `elicitationId` (two servers may use the same id). Only these get a "finished in the
   * browser" line, once each.
   */
  private readonly mcpOpened = new Map<string, Array<{ server: string; elicitationId: string }>>();
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private crashes: number[] = [];
  private mismatches = 0;

  constructor(options: AgentHostOptions) {
    this.mode = options.mode;
    this.opts = options;
    this.timings = { ...DEFAULT_TIMINGS, ...options.timings };
    this.now = options.now ?? Date.now;
    this.watches =
      options.mode === 'work'
        ? new AgentWatches({
            read: (childId) => readAgentUpdates(this.opts.sessionDir?.(childId) ?? null),
            needles: () => this.opts.needles?.() ?? [],
            send: (sessionId, event) => this.opts.listener?.onAgentEntries?.(sessionId, event),
            now: () => this.now(),
          })
        : null;
  }

  modeState(): AgentModeState {
    let turnsInFlight = 0;
    let background = 0;
    let tasks = 0;
    for (const s of this.sessions.values()) {
      if (s.turn !== null) turnsInFlight++;
      if (hasBackground(s)) background++;
      tasks += s.tracker.taskCount();
    }
    return {
      state: this.state,
      stale: this.stale,
      failure: this.failure,
      agentVersion: this.running?.agentVersion ?? null,
      turnsInFlight,
      background,
      tasks,
      engineLost: this.engineLost,
    };
  }

  /** Work: the PDF tool is there, as the last agent start wrote its files; null before the first start. */
  pdfTool(): boolean | null {
    return this.pdfToolOpen;
  }

  /** The models of the running process, or null when it is not running. `thinking`: the levels it lists for the model. */
  models(): Array<{ modelId: string; contextTokens: number | null; thinking: ModelThinking | null }> | null {
    const running = this.running;
    if (running === null) return null;
    return running.models.map((m) => ({
      modelId: m.modelId,
      contextTokens: m.contextTokens,
      thinking: modelThinking(m.efforts),
    }));
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

  /** A typed turn, a wake turn or a history op runs: no prompt, rename or reseed. A workflow run alone is not busy. */
  isBusy(sessionId: string): boolean {
    const s = this.sessions.get(sessionId);
    return s !== undefined && (s.turn !== null || s.wake !== null || s.op !== null);
  }

  /** A typed turn, engine work or a history op: no history op and no model change. */
  engineBusy(sessionId: string): boolean {
    const s = this.sessions.get(sessionId);
    return s !== undefined && (s.turn !== null || hasBackground(s));
  }

  /** {@link engineBusy}, or a command or monitor runs: no delete. */
  isWorking(sessionId: string): boolean {
    const s = this.sessions.get(sessionId);
    return this.engineBusy(sessionId) || (s !== undefined && processWork(s));
  }

  /**
   * Work: the folder of each session that works now: a turn, a wake, a run, a helper, a history op, a command or a
   * monitor runs, its goal runs, or it is being created.
   */
  workingFolders(): string[] {
    const out: string[] = [];
    for (const s of this.sessions.values()) {
      if (this.isWorking(s.id) || this.goalRuns(s.id) || s.creating) out.push(s.cwd);
    }
    return out;
  }

  /** Work: the session's workflow runs of the running process: the live ones, then the newest ended ones. */
  runs(sessionId: string): RunView[] {
    const s = this.sessions.get(sessionId);
    return s === undefined ? [] : runViews(s);
  }

  /** Work: the session's runs a Stop can reach now: the live ones, and the ended ones that wait and the agent stops. */
  stoppableRuns(sessionId: string): Set<string> {
    const s = this.sessions.get(sessionId);
    const out = new Set<string>();
    if (s === undefined) return out;
    for (const runId of s.runs.keys()) out.add(runId);
    for (const [runId, run] of s.endedRuns) if (STOPPABLE_WAITS.has(run.status)) out.add(runId);
    return out;
  }

  /** A run, a command, a monitor or a helper of the session ended with a wake the agent has not started yet. */
  wakeDue(sessionId: string): boolean {
    const s = this.sessions.get(sessionId);
    return s !== undefined && (s.wakeDue !== null || s.runWake !== null);
  }

  /**
   * Work: the session's goal is `active` in the running process: the agent starts its next round by itself, so a
   * queued message waits for the goal to pause.
   */
  goalRuns(sessionId: string): boolean {
    const s = this.sessions.get(sessionId);
    return s !== undefined && s.hot && s.goal === 'active';
  }

  /** The engine restarted under live commands or monitors: only a message the user sends may restart the agent. */
  engineLostTasks(): boolean {
    return this.lostTasks();
  }

  /** Work: the live facts of the session's background work, for a page's `sync`. */
  liveActivity(sessionId: string): LiveActivity {
    const session = this.sessions.get(sessionId);
    return session === undefined
      ? { helpers: [], agents: [], tasks: [], schedules: [], reports: [], stoppedByYou: [] }
      : this.liveOf(session);
  }

  /**
   * Whether the typed turn's prompt is not answered yet: a turn starting, or its `session/prompt` with no reply. Read
   * per stdout line, so a line after the reply in the same chunk already reads false.
   */
  promptOpen(sessionId: string): boolean {
    const turn = this.sessions.get(sessionId)?.turn;
    if (turn == null) return false;
    if (turn.prompt === null) return !turn.between;
    return turn.prompt.client.isPending(turn.prompt.id);
  }

  turnState(sessionId: string): TurnState | null {
    const s = this.sessions.get(sessionId);
    return s === undefined ? null : shownTurn(s);
  }

  openInteractions(sessionId: string): Interaction[] {
    const s = this.sessions.get(sessionId);
    return s === undefined ? [] : [...s.interactions.values()].map((o) => o.interaction);
  }

  /** Work: the helper session whose call an open card is; none for the session's own call or a card that is gone. */
  openChild(sessionId: string, interactionId: string): { child?: string } {
    const s = this.sessions.get(sessionId);
    const open = s?.interactions.get(interactionId);
    return s === undefined || open === undefined ? {} : childOf(s, open.callSession);
  }

  /** Work: "allow all edits" was picked in this session's process. */
  editsGranted(sessionId: string): boolean {
    return this.sessions.get(sessionId)?.editsGranted === true;
  }

  compaction(sessionId: string): CompactionState | null {
    const state = this.sessions.get(sessionId)?.compaction;
    return state == null ? null : { ...state };
  }

  hookRun(sessionId: string): HookRunState | null {
    const run = this.sessions.get(sessionId)?.hookRun;
    return run == null ? null : { ...run.state };
  }

  /** A recap the user asked for is being written in the session; null while none is. */
  recapState(sessionId: string): RecapState | null {
    const wait = this.sessions.get(sessionId)?.recap;
    return wait == null ? null : { startedAt: wait.startedAt };
  }

  sessionModel(sessionId: string): string | null {
    return this.sessions.get(sessionId)?.modelId ?? null;
  }

  /** The Chat address ledger of a session the host knows. */
  ledger(sessionId: string): AddressLedger | null {
    return this.sessions.get(sessionId)?.ledger ?? null;
  }

  /**
   * The host's sessions with their flags. `busy`: a typed turn runs; `stopping`: it ends for a recorded cause;
   * `asks`: open requests; `background`: engine work (a workflow run, a wake turn, a helper) or a history op is live;
   * `helpers`: a subagent of the conversation's own runs; `tasks`: its live commands and monitors; `op`: the history op
   * or the recap that runs now.
   */
  active(): Array<{
    sessionId: string;
    busy: boolean;
    waiting: boolean;
    asks: number;
    stopping: boolean;
    background: boolean;
    helpers: boolean;
    tasks: number;
    op: HistoryOp | 'recap' | null;
  }> {
    return [...this.sessions.values()]
      .filter((s) => s.hot || s.turn !== null || hasBackground(s) || s.recap !== null)
      .map((s) => ({
        sessionId: s.id,
        busy: s.turn !== null,
        waiting: s.interactions.size > 0,
        asks: s.interactions.size,
        stopping: s.turn !== null && s.turn.recorded !== null,
        background: hasBackground(s),
        helpers: s.tracker.ownHelperCount() > 0,
        tasks: s.tracker.taskCount(),
        op: opOf(s),
      }));
  }

  /** The history op or the recap of the session that runs now, or null. */
  opOf(sessionId: string): HistoryOp | 'recap' | null {
    const s = this.sessions.get(sessionId);
    return s === undefined ? null : opOf(s);
  }

  /**
   * A config change (endpoint, providers, models, binary). The host restarts as soon as no turn, wake, workflow run,
   * helper, command or monitor is live.
   */
  markStale(): void {
    if (this.running === null && this.starting === null) return;
    this.stale = true;
    for (const s of this.sessions.values()) s.turn?.gap?.();
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
   * `inference-lost`; a missing or changed one stops every workflow run (the process holds the old address and
   * token, so its helpers can never reach the engine again) and marks the engine lost.
   */
  async checkEndpoint(): Promise<void> {
    const running = this.running;
    if (running === null) return;
    const endpoint = await this.opts.readEndpoint();
    if (this.running !== running) return;
    if (endpoint === null) {
      for (const s of this.sessions.values()) if (s.turn !== null) this.cancelTurn(s, 'inference-lost');
      this.stopAllRuns();
      this.engineMoved();
      return;
    }
    if (!sameEndpoint(endpointFingerprint(endpoint), running.fingerprint)) {
      this.stopAllRuns();
      this.engineMoved();
    }
  }

  /** The running process holds an engine address or token that is gone: stale, and its commands wait for a message. */
  private engineMoved(): void {
    if (this.running !== null) this.engineLost = true;
    this.markStale();
  }

  /**
   * Work: the user's Stop of one workflow run, through `_x.ai/workflows/stop`: a live run, or one that waits on the
   * person and the agent can stop (it refuses a run out of agents). `not-live` for any other. Its open cards are
   * answered `cancelled` at once; the run leaves on its terminal `workflow_updated`, and a rejected or failed stop gives
   * it back.
   */
  async stopRun(sessionId: string, runId: string): Promise<RunStop> {
    return this.stopRunAs(sessionId, runId, 'you');
  }

  private async stopRunAs(sessionId: string, runId: string, by: 'you' | 'engine'): Promise<RunStop> {
    const session = this.sessions.get(sessionId);
    const live = session?.runs.get(runId);
    const ended = session?.endedRuns.get(runId);
    const run = live ?? (ended !== undefined && STOPPABLE_WAITS.has(ended.status) ? ended : undefined);
    const running = this.running;
    if (session === undefined || run === undefined || running === null) return 'not-live';
    if (!run.stopping) run.stopBy = by;
    run.stopping = true;
    this.armStopTimer(session, runId, run);
    this.closeOwned(session, (o) => o.kind === 'run' && o.runId === runId, 'cancelled');
    this.emitRuns(session);
    let outcome: 'stopped' | 'rejected' | null = null;
    try {
      outcome = readWorkflowStop(
        await this.call(running, '_x.ai/workflows/stop', { sessionId, runId }, ACP_TIMEOUTS['_x.ai/workflows/stop']),
      );
    } catch {
      outcome = null;
    }
    const held =
      live !== undefined
        ? session.runs.get(runId) === run
        : session.endedRuns.get(runId) === run && run.status !== 'cancelled';
    if (outcome === 'stopped' || !held) return 'stopped';
    endStopping(run);
    this.emitRuns(session);
    return outcome ?? 'failed';
  }

  /** `stopping` ends at the drain update; this ends it when that never comes. */
  private armStopTimer(session: HostSession, runId: string, run: RunRecord): void {
    if (run.stopTimer !== null) clearTimeout(run.stopTimer);
    run.stopTimer = setTimeout(() => {
      run.stopTimer = null;
      const held = session.runs.get(runId) === run || session.endedRuns.get(runId) === run;
      if (!run.stopping || !held || this.sessions.get(session.id) !== session) return;
      endStopping(run);
      this.emitRuns(session);
    }, this.timings.stopDrainMs);
    run.stopTimer.unref?.();
  }

  /**
   * Work: a Stop of one command or monitor, through `_x.ai/task/kill` (soft). The task reads `stopping` from the press
   * until its end line, or until the stop fails. `not-live` when the conversation is not loaded in the running process
   * or holds no such live task; a task the agent no longer knows leaves the live facts. `teardown` is the rewind's:
   * the agent counts its result as delivered and starts no wake turn.
   */
  async stopTask(sessionId: string, taskId: string, source: TaskKillSource = 'clientUi'): Promise<TaskStop> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (session === undefined || running === null || !session.hot || !session.tracker.hasTask(taskId))
      return 'not-live';
    if (session.tracker.setStopping(taskId, true)) this.activityNow(session);
    let outcome: ReturnType<typeof readTaskKill> = null;
    try {
      outcome = readTaskKill(
        await running.process.client.requestSoft(
          '_x.ai/task/kill',
          { sessionId, taskId, source },
          this.timings.taskKillMs,
        ),
      );
    } catch {
      outcome = null;
    }
    if (outcome === 'killed' || outcome === 'already_exited') return 'stopped';
    const same = this.running === running && this.sessions.get(sessionId) === session;
    if (outcome === 'not_found') {
      if (same && session.tracker.dropTask(taskId)) {
        this.activityNow(session);
        this.workChanged(session);
      }
      return 'not-live';
    }
    if (same && session.tracker.setStopping(taskId, false)) this.activityNow(session);
    return 'failed';
  }

  /**
   * Work: a Stop of one agent (`childId`, any attempt's id): a subagent of the conversation or an agent of a workflow
   * run, through `_x.ai/subagent/cancel` (soft) with its newest attempt's id. The run's script gets a cancelled result
   * for it and goes on. The agent reads `stopping` from the press until its `subagent_finished`, or until the stop
   * fails; a stop the agent took makes it "Stopped by you". `not-live` when the conversation is not loaded in the
   * running process or holds no such running agent; one the agent says already ended or does not know leaves the live
   * facts (no end line is coming).
   */
  async stopSubagent(sessionId: string, childId: string): Promise<SubagentStop> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    const helper = session?.tracker.agent(childId) ?? null;
    if (session === undefined || running === null || !session.hot || helper === null) return 'not-live';
    if (session.tracker.setHelperStopping(helper.childId, true)) this.activityNow(session);
    let outcome: ReturnType<typeof readSubagentCancel> = null;
    try {
      outcome = readSubagentCancel(
        await running.process.client.requestSoft(
          '_x.ai/subagent/cancel',
          { subagentId: helper.subagentId },
          this.timings.subagentCancelMs,
        ),
      );
    } catch {
      outcome = null;
    }
    if (outcome === 'cancelled') {
      if (this.sessions.get(sessionId) === session && session.tracker.stopTaken(helper.childId)) {
        this.activityNow(session);
      }
      return 'stopped';
    }
    const same = this.running === running && this.sessions.get(sessionId) === session;
    if (outcome === 'already_finished' || outcome === 'not_found') {
      if (same && session.tracker.dropHelper(helper.childId, helper.subagentId)) {
        this.agentsSettled(session);
        this.activityNow(session);
        this.workChanged(session);
      }
      return 'not-live';
    }
    if (same && session.tracker.setHelperStopping(helper.childId, false)) this.activityNow(session);
    return 'failed';
  }

  /**
   * Work: the user's message to one of the conversation's own subagents, steered into its running turn through
   * `_x.ai/subagent/message` (soft, no `queue`). Sent only while the tracker holds the helper as running and not
   * stopping: to one that ended the agent would start a new turn of it whose result reaches no one. `gone` when it does
   * not, and for the agent's `rejected` and `channel_closed`. A process that answers the method as not found takes no
   * message again (`off`). No reply in time is `unsure`: the message may have arrived, and nothing is sent again.
   */
  async messageSubagent(sessionId: string, childId: string, text: string): Promise<SubagentMessage> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    const helper = session?.tracker.agent(childId) ?? null;
    if (session === undefined || running === null || !session.hot || helper === null || helper.stopping) return 'gone';
    if (running.messagesOff) return 'off';
    if (helper.address === null) return 'error';
    if (Buffer.byteLength(text, 'utf8') > SUBAGENT_MESSAGE_MAX_BYTES) return 'too-long';
    let reply: unknown;
    try {
      reply = await running.process.client.requestSoft(
        '_x.ai/subagent/message',
        { sessionId, agentAddress: helper.address, content: [{ type: 'text', text }] },
        this.timings.subagentMessageMs,
      );
    } catch (error) {
      if (error instanceof AcpSoftTimeoutError) return 'unsure';
      if (error instanceof AcpRpcError && error.code === METHOD_NOT_FOUND) {
        if (this.running === running && !running.messagesOff) {
          running.messagesOff = true;
          for (const s of this.sessions.values()) if (s.tracker.helperCount() > 0) this.activityNow(s);
        }
        return 'off';
      }
      return 'error';
    }
    return messageResult(readSubagentMessage(reply));
  }

  /** Work: `childId` is an agent the session started in the running process (its own child, never a nested one). */
  ownsChild(sessionId: string, childId: string): boolean {
    const parent = this.children.get(childId);
    return parent !== undefined && this.sessions.get(sessionId) === parent;
  }

  /**
   * Work: the Agent view of `window` on one agent of the session (`childId`, the attempt it shows): its file, then its
   * kept live lines past the seam, redacted. While it runs, the window gets its later lines and, at its end, the stored
   * list ({@link AgentHostListener.onAgentEntries}); an agent that ended is one read. `too-many` when the window holds
   * two other live views. The caller checks that the agent is the session's.
   */
  async watchAgent(sessionId: string, childId: string, window: string): Promise<AgentWatchAnswer | 'too-many'> {
    if (this.watches === null) return { childId, attempts: [childId], entries: [], live: false };
    const session = this.sessions.get(sessionId);
    const running = (): boolean =>
      session !== undefined &&
      this.sessions.get(sessionId) === session &&
      this.children.get(childId) === session &&
      session.tracker.agent(childId)?.childId === childId;
    const attempts = session?.tracker.attempts(childId) ?? [childId];
    const start = await this.watches.watch(sessionId, childId, window, running);
    if (start === 'too-many') return start;
    return { childId, attempts, entries: start.entries, live: start.live };
  }

  /** Work: the window's view of the agent closed; nothing more is sent for it. */
  unwatchAgent(childId: string, window: string): void {
    this.watches?.unwatch(childId, window);
  }

  /** Work: no page shows the conversation now, so none of its Agent views gets more lines. */
  unwatchAgents(sessionId: string): void {
    this.watches?.unwatchSession(sessionId);
  }

  /**
   * Work: the user's Delete of one schedule, through `_x.ai/scheduler/delete` (soft), the conversation loaded first
   * when the running agent does not hold it (a paused schedule; the agent fires an overdue one as the session loads).
   * The schedule reads `opening` from the press while it loads, then `removing` until the reply. `deleted` and `gone`
   * take it out of the live facts (the stored `scheduled_task_deleted` ends its row). Throws `engine-lost` while the
   * engine restarted under commands or monitors (a load would restart the agent and stop them),
   * `conversation-busy` while a Delete of it is out, the start, load and guard failures of a turn, and `load-slow` for
   * a cold load past its soft deadline.
   */
  async deleteSchedule(sessionId: string, scheduleId: string, load: LoadInput): Promise<ScheduleDelete> {
    if (this.closed) throw new AgentHostError('agent-crashed');
    if (this.mode !== 'work') return 'gone';
    if (this.lostTasks()) throw new AgentHostError('engine-lost');
    const existing = this.sessions.get(sessionId);
    if (existing?.creating === true || existing?.deleting.has(scheduleId) === true) {
      throw new AgentHostError('conversation-busy');
    }
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    const loaded = session.guarded && session.hot && this.running !== null && !this.stale;
    const record: DeleteRecord = { phase: loaded ? 'removing' : 'opening' };
    session.deleting.set(scheduleId, record);
    this.clearIdle();
    this.activityNow(session);
    try {
      await this.opts.providers.settled();
      const ready = await this.ensureLoaded(session, load, record, () => false);
      if (ready === null) return 'failed';
      if ('refused' in ready) throw new AgentHostError(ready.refused);
      if (record.phase !== 'removing') {
        record.phase = 'removing';
        this.activityNow(session);
      }
      const end = await this.requestScheduleDelete(ready.running, sessionId, scheduleId);
      if (end !== 'failed' && this.running === ready.running && session.tracker.dropSchedule(scheduleId)) {
        this.workChanged(session);
      }
      return end;
    } finally {
      if (session.deleting.get(scheduleId) === record) session.deleting.delete(scheduleId);
      this.activityNow(session);
      this.idleStep();
    }
  }

  /**
   * Work: Send now. The text goes to the running answer through `_x.ai/interject` (soft), and the agent reads it at
   * its next step. Only while a prompt of the session is open ({@link steerable}): sent with none, the agent would run
   * it as a turn the host did not start. A Stop before the model read it drops it in the agent.
   */
  async steer(sessionId: string, text: string): Promise<SteerOutcome> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (this.mode !== 'work' || session === undefined || running === null || !steerable(session, running)) {
      return 'not-now';
    }
    let reply: unknown;
    try {
      reply = await running.process.client.requestSoft(
        '_x.ai/interject',
        { sessionId, text, interjectionId: randomUUID() },
        this.timings.steerMs,
      );
    } catch (error) {
      if (error instanceof AcpSoftTimeoutError) return { late: error.late.then(readInterject, () => false) };
      return 'not-now';
    }
    return readInterject(reply) ? 'sent' : 'not-now';
  }

  private stopAllRuns(): void {
    for (const s of this.sessions.values())
      for (const runId of s.runs.keys()) void this.stopRunAs(s.id, runId, 'engine');
  }

  /** Starts the process if needed. Resolves when the self-check passed. */
  async start(): Promise<void> {
    await this.ensureRunning();
  }

  async createSession(input: CreateInput): Promise<CreatedSession> {
    await this.opts.providers.settled();
    const running = await this.ensureRunning(undefined, true);
    this.clearIdle();
    if (!running.models.some((m) => m.modelId === input.modelId)) throw new AgentHostError('model-unavailable');
    const cwd = await this.sessionCwd(input.cwd);
    const web = this.sessionWeb();
    const ledgerFile = this.newLedgerFile();
    let reply: ReturnType<typeof readSessionReply>;
    saveLedgerFile(ledgerFile, []);
    this.sessionsNew += 1;
    try {
      const params = this.newParams(cwd, input.modelId, web, ledgerFile);
      reply = readSessionReply(await this.call(running, 'session/new', params, ACP_TIMEOUTS['session/new']));
      if (reply.sessionId === null) throw new AgentHostError('agent-protocol-error');
    } catch (error) {
      removeLedgerFile(ledgerFile);
      throw error;
    } finally {
      this.sessionsNew -= 1;
      if (this.sessionsNew === 0) this.startedEarly.clear();
    }
    const session = this.register(reply.sessionId, cwd, input.modelId, false, ledgerFile);
    session.web = web;
    session.efforts = reply.efforts;
    session.effort = reply.effort;
    session.creating = true;
    let guard: GuardResult;
    try {
      guard = await this.guard(running, session, reply, input.modelId, true);
    } finally {
      session.creating = false;
    }
    if (!guard.ok) throw new AgentHostError(guard.reason);
    if (this.mode === 'work') {
      this.pullCommands(running, session);
      this.opts.listener?.onSessionReady?.(session.id);
    }
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
   * call run in the background, and `done` resolves when the turn ended. Work: the turn's first prompt names its mode,
   * `plan` when `opts.plan`, else `agent`; an automatic `continue` names none, so the agent keeps the mode it has.
   * `opts.sendable`, read in the tick the prompt would go: false ends the turn unsent, with no outcome.
   */
  startTurn(
    sessionId: string,
    text: string,
    load: LoadInput,
    attachments: readonly AttachmentInput[] = [],
    opts: { plan?: boolean; sendable?: () => boolean } = {},
  ): { turnId: string; done: Promise<TurnResult> } {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (this.isBusy(sessionId) || existing?.creating === true) throw new AgentHostError('conversation-busy');
    if (checkCommand(this.mode, text, existing?.commands ?? null) === 'refused') throw new AgentHostError('slash-text');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    if (!session.guarded) this.seedLedger(session, load.entries);
    const turn: TurnRecord = {
      turnId: randomUUID(),
      phase: 'starting',
      startedAt: this.now(),
      wait: null,
      select: null,
      recorded: null,
      model: null,
      approvals: 0,
      approvalEpoch: 0,
      capTimer: null,
      capLeft: null,
      capFrom: 0,
      graceTimer: null,
      abort: null,
      between: false,
      gap: null,
      toolSeen: false,
      endSeen: false,
      prompt: null,
      lengthStops: 0,
      continues: 0,
      runCommand: this.mode === 'work' ? workflowCommand(text) : null,
    };
    session.turn = turn;
    // The agent drops a recap it is writing when a typed prompt comes.
    if (session.recap !== null) this.endRecap(session, session.recap, 'cancelled');
    this.clearIdle();
    this.emitTurn(session);
    this.emitState();
    const meta: PromptMeta | undefined =
      this.mode === 'work' ? { mode: opts.plan === true ? 'plan' : 'agent' } : undefined;
    const done = this.runTurn(session, turn, text, load, attachments, meta, opts.sendable).then((ended) => {
      const result = turn.model === null ? ended : { ...ended, model: turn.model };
      this.finishTurn(session, turn, result);
      return result;
    });
    return { turnId: turn.turnId, done };
  }

  /**
   * The user's stop of the typed turn, else of the wake turn, else of a manual compaction. Open requests of that turn
   * are answered `cancelled` first. A wake and a compaction get `session/cancel` and no grace kill: a kill would also
   * end the workflow runs and the background commands.
   */
  cancel(sessionId: string): boolean {
    const session = this.sessions.get(sessionId);
    if (session === undefined) return false;
    if (session.turn !== null) {
      this.cancelTurn(session, 'cancelled');
      return true;
    }
    const wake = session.wake;
    if (wake === null) return this.stopCompact(session);
    if (wake.phase === 'cancelling') return true;
    wake.phase = 'cancelling';
    this.closeOwned(session, (o) => o.kind === 'wake', 'cancelled');
    this.emitTurn(session);
    const running = this.running;
    if (running !== null) {
      try {
        running.process.client.notify('session/cancel', { sessionId: session.id });
      } catch {
        // The process is gone: the wake ends at its exit.
      }
    }
    return true;
  }

  /**
   * Work: the user's answer to a permission request; the option must be one the interaction offered. A deny with a
   * note that is not blank sends the note as `_meta.followup_message`: the agent skips the call, reads the note as the
   * user's message and goes on, so the turn is not marked denied. Returns what the history records of that deny.
   */
  answer(sessionId: string, interactionId: string, optionId: string, note?: string): NoteDeny | null {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined || open.interaction.kind !== 'permission') {
      throw new InteractionError('not-found');
    }
    if (!open.interaction.options.some((o) => o.optionId === optionId)) throw new InteractionError('not-offered');
    if (this.modeOf(sessionId) !== 'ask' && optionId !== REJECT_ONCE_OPTION_ID && optionId !== ALLOW_ONCE_OPTION_ID) {
      throw new InteractionError('not-offered');
    }
    if (note !== undefined && optionId !== REJECT_ONCE_OPTION_ID) throw new InteractionError('not-offered');
    const withNote = note === undefined ? null : noteAnswer(optionId, note);
    if (withNote !== null) {
      const card = open.interaction;
      if (!card.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID && o.kind === 'reject_once')) {
        throw new InteractionError('not-offered');
      }
      const facts: NoteDeny = {
        toolCallId: card.toolCallId,
        access: card.access,
        card: { ...card, options: [] },
        ...childOf(session, open.callSession),
      };
      this.settleInteraction(session, interactionId, withNote, 'answered');
      return facts;
    }
    if (optionId === REJECT_ONCE_OPTION_ID) this.recordDenied(session, open.owner);
    if (optionId !== REJECT_ONCE_OPTION_ID) {
      const count = this.countOf(session, open.owner);
      if (count !== null) {
        count.approvals = 0;
        count.approvalEpoch += 1;
      }
      if (optionId === ALLOW_EDITS_SESSION_OPTION_ID) session.editsGranted = true;
      if (optionId === ALLOW_EDITS_SESSION_OPTION_ID || optionId === ALWAYS_ALLOW_OPTION_ID) session.granted = true;
      this.callStarted(session, open.callSession, open.interaction.toolCallId, { changes: true, wide: true });
    }
    this.settleInteraction(session, interactionId, { outcome: { outcome: 'selected', optionId } }, 'answered');
    return null;
  }

  /**
   * Work: "Never allow" on a command card. The rule (the card's `short` or `exact` words) is kept for the Work folder
   * from now on; this call is denied with the rule's note in this tick, so the agent goes on and the turn is not marked
   * denied. `saved` resolves false when the rule was not kept (the file was not written, or the folder is full); the
   * call is denied all the same. Only a card that offered a rule may take one.
   */
  answerNever(
    sessionId: string,
    interactionId: string,
    scope: 'short' | 'exact',
  ): { denied: RuleDeny; saved: Promise<boolean> } {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    const never = this.opts.never;
    if (session === undefined || open === undefined || never === undefined) throw new InteractionError('not-found');
    const card = open.interaction;
    const offered = this.neverOffers.get(card);
    if (card.kind !== 'permission' || offered === undefined) throw new InteractionError('not-offered');
    const words = scope === 'short' ? offered.short : offered.exact;
    const answer = noteAnswer(REJECT_ONCE_OPTION_ID, neverNote(ruleText(words)));
    if (answer === null) throw new InteractionError('not-offered');
    const { rule, saved } = never.add(session.cwd, words);
    const denied: RuleDeny = {
      toolCallId: card.toolCallId,
      access: card.access,
      card: { ...card, options: [] },
      ...childOf(session, open.callSession),
      ruleId: rule?.id ?? null,
      words: ruleText(words),
      noted: true,
    };
    this.settleInteraction(session, interactionId, answer, 'answered');
    return { denied, saved };
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
    this.callStarted(session, open.callSession, open.interaction.toolCallId, {
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
    this.recordDenied(session, open.owner);
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
   * Work: the user's answer to a question card, by option index; the reply names the agent's own texts. An answer
   * that does not fit the card (or a body for another kind of card) is `not-offered`.
   */
  answerQuestion(sessionId: string, interactionId: string, input: QuestionAnswerInput): void {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined) throw new InteractionError('not-found');
    const req = this.questionRequests.get(open.interaction);
    if (open.interaction.kind !== 'question' || req === undefined) throw new InteractionError('not-offered');
    const reply = questionReply(req, input);
    if (reply === null) throw new InteractionError('not-offered');
    this.settleInteraction(session, interactionId, reply, 'answered');
  }

  /**
   * Work: the user's answer to an MCP card. Decline goes as it is. A link's accept (the user opens the page) carries no
   * values, and its id is kept for the server's "done". A form's accept is checked against the form first: values
   * that break its rules come back by field and the card stays open; values that do not fit the card (a count other
   * than the shown fields, a wrong shape, an unusable form) are `not-offered`. No error carries a typed value.
   */
  answerElicit(sessionId: string, interactionId: string, input: McpAnswerInput): Array<McpFieldError | null> | null {
    const session = this.sessions.get(sessionId);
    const open = session?.interactions.get(interactionId);
    if (session === undefined || open === undefined) throw new InteractionError('not-found');
    const form = this.mcpForms.get(open.interaction);
    if (open.interaction.kind !== 'mcp-input' || form === undefined) throw new InteractionError('not-offered');
    if (input.mcp === 'decline') {
      this.settleInteraction(session, interactionId, MCP_DECLINE, 'answered');
      return null;
    }
    if (open.interaction.unusable !== undefined) throw new InteractionError('not-offered');
    if (form.elicitationId !== null) {
      if ((input.values ?? []).length > 0) throw new InteractionError('not-offered');
      this.linkOpened(session.id, form.elicitationId, form.server);
      this.settleInteraction(session, interactionId, LINK_ACCEPT, 'answered');
      return null;
    }
    if (form.specs === null) throw new InteractionError('not-offered');
    const checked = validateForm(form.specs, input.values ?? []);
    if (checked === null) throw new InteractionError('not-offered');
    if ('errors' in checked) return checked.errors;
    this.settleInteraction(session, interactionId, acceptReply(checked.content), 'answered');
    return null;
  }

  /**
   * Nobody can answer (the window is gone): every open permission request is denied, which ends a typed turn that
   * owns it as `denied`, and every open plan is withdrawn.
   */
  denyOpen(): void {
    for (const session of this.sessions.values()) {
      for (const [id, open] of session.interactions) {
        if (open.interaction.kind === 'permission') {
          const deny = open.interaction.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID);
          this.recordDenied(session, open.owner);
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
    this.mcpClosed(session, open.interaction, result, reason);
    this.emitState();
  }

  /** An MCP card closed: its history line, by the reply it got (an accept or a decline), else by why it closed. */
  private mcpClosed(
    session: HostSession,
    interaction: Interaction,
    result: unknown,
    reason: 'answered' | 'cancelled' | 'expired',
  ): void {
    const form = this.mcpForms.get(interaction);
    if (interaction.kind !== 'mcp-input' || form === undefined) return;
    const sent = result === MCP_DECLINE ? 'decline' : reason === 'answered' && result !== MCP_CANCEL ? 'accept' : null;
    const outcome = sent ?? (reason === 'expired' ? 'expired' : 'cancel');
    this.opts.listener?.onMcpClosed?.(session.id, { server: form.server, mode: interaction.mode, outcome });
  }

  /** The user opened a link card's page: its server may now say it is done. The oldest ids go past a few dozen. */
  private linkOpened(sessionId: string, elicitationId: string, server: string): void {
    const opened = (this.mcpOpened.get(sessionId) ?? []).filter(
      (o) => o.elicitationId !== elicitationId || o.server !== server,
    );
    opened.push({ server, elicitationId });
    this.mcpOpened.set(sessionId, opened.slice(-LINKS_OPENED_MAX));
  }

  /**
   * `_x.ai/mcp/elicit_complete`: a "finished in the browser" line, only for a page the user opened from this app in
   * this process's life, once, and only from the server that asked (any server that opened that id, oldest first, when
   * the agent names none).
   */
  private linkDone(params: unknown): void {
    const done = readElicitComplete(params);
    const opened = done === null ? undefined : this.mcpOpened.get(done.sessionId);
    if (done === null || opened === undefined) return;
    const at = opened.findIndex(
      (o) => o.elicitationId === done.elicitationId && (done.server === null || o.server === done.server),
    );
    if (at === -1) return;
    const { server } = opened[at]!;
    opened.splice(at, 1);
    this.opts.listener?.onMcpDone?.(done.sessionId, server);
  }

  async rename(sessionId: string, title: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (session === undefined || !session.guarded || running === null) throw new AgentHostError('agent-crashed');
    if (this.isBusy(sessionId) || session.loading !== null) throw new AgentHostError('conversation-busy');
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
    // A schedule's Delete may be loading the session: a load beside the agent's delete can write it back.
    if (this.isWorking(sessionId) || (session?.deleting.size ?? 0) > 0) throw new AgentHostError('conversation-busy');
    const running = await this.ensureRunning();
    const sessionCwd = session?.cwd || (this.mode === 'chat' ? await this.sessionCwd(undefined) : cwd);
    const result = await this.call(
      running,
      '_x.ai/session/delete',
      { sessionId, cwd: sessionCwd },
      ACP_TIMEOUTS['_x.ai/session/delete'],
    );
    if (!isSuccess(result)) throw new AgentHostError('agent-protocol-error');
    this.recapDeleted(sessionId);
    this.drop(sessionId);
    this.scheduleIdle();
  }

  /** After a delete outside the agent (the session writer): the host forgets the session and its ledger file. */
  forgetSession(sessionId: string): void {
    this.recapDeleted(sessionId);
    this.drop(sessionId);
  }

  /** `session/set_model` between turns, confirmed by `_x.ai/session/info`. */
  async setModel(sessionId: string, modelId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (session === undefined || !session.guarded || running === null) throw new AgentHostError('agent-crashed');
    if (this.engineBusy(sessionId) || session.loading !== null) throw new AgentHostError('conversation-busy');
    if (!running.models.some((m) => m.modelId === modelId)) throw new AgentHostError('model-unavailable');
    // The agent keeps or drops the level by the new model's list; its `config_option_update` says which.
    session.efforts = null;
    session.effort = null;
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

  /**
   * The Context panel's facts: `_x.ai/session/info` for a session loaded and guarded in the running process, under a
   * soft deadline (a miss leaves the agent alive). Null when it is not loaded (a panel never starts or loads the agent)
   * and when the read fails. Answers while a turn runs too.
   */
  async contextInfo(sessionId: string): Promise<SessionInfoFacts | null> {
    const session = this.sessions.get(sessionId);
    const running = this.running;
    if (session === undefined || running === null || !session.hot || !session.guarded) return null;
    try {
      return readSessionInfo(
        await running.process.client.requestSoft('_x.ai/session/info', { sessionId }, this.timings.contextInfoMs),
      );
    } catch {
      return null;
    }
  }

  /** The commands the session may send, as the page lists them; null while its session holds no list. */
  commandsOf(sessionId: string): CommandView[] | null {
    const commands = this.sessions.get(sessionId)?.commands ?? null;
    return commands === null ? null : allowedCommands(this.mode, commands);
  }

  /** The agent's command list of the session for the allow rule, once the pull after its load settled; null: none. */
  async agentCommands(sessionId: string): Promise<AgentCommand[] | null> {
    const session = this.sessions.get(sessionId);
    if (session === undefined) return null;
    if (session.commandsPull !== null) await session.commandsPull;
    return session.commands;
  }

  /** Work: `_x.ai/commands/list` after a load, soft; a miss keeps what the session has. */
  private pullCommands(running: Running, session: HostSession): void {
    session.commandsPull = running.process.client
      .requestSoft('_x.ai/commands/list', { sessionId: session.id }, this.timings.commandsListMs)
      .then(
        (result) => {
          const rows = readCommandList(result);
          if (rows !== null && this.running === running && this.sessions.get(session.id) === session) {
            this.setCommands(session, readCommands(withoutAppWorkflow(rows, this.appWorkflow)));
          }
        },
        () => {},
      );
  }

  /**
   * Before a prompt: the conversation's thinking level, stepped to one the model lists as the page shows it (else the
   * model's default), goes to the agent when the session holds another, through `session/set_config_option` (soft).
   * Nothing for a model with no levels. A refusal, a miss or a level the session does not offer gives the
   * `thinking-not-set` notice and the turn goes on; a late OK still records the level. A set that took, or a session
   * that already holds the level, clears the notice.
   */
  private async applyEffort(running: Running, session: HostSession, level: ThinkingLevel | undefined): Promise<void> {
    const thinking = modelThinking(running.models.find((m) => m.modelId === session.modelId)?.efforts ?? null);
    if (thinking === null) return;
    const wanted = stepLevel(level, thinking);
    if (wanted === session.effort) {
      this.opts.listener?.onNoticeCleared?.(session.id, 'thinking-not-set');
      return;
    }
    if (session.efforts !== null && !session.efforts.includes(wanted)) {
      this.effortMissed(session);
      return;
    }
    const took = (reply: unknown): boolean => {
      if (this.running !== running || this.sessions.get(session.id) !== session) return false;
      const now = readEffortConfig(isRecord(reply) ? reply.configOptions : null);
      session.efforts = now?.ids ?? session.efforts;
      session.effort = now?.current ?? wanted;
      return session.effort === wanted;
    };
    try {
      const reply = await running.process.client.requestSoft(
        'session/set_config_option',
        { sessionId: session.id, configId: EFFORT_CONFIG_ID, value: wanted },
        this.timings.effortSetMs,
      );
      if (took(reply)) this.opts.listener?.onNoticeCleared?.(session.id, 'thinking-not-set');
      else this.effortMissed(session);
    } catch (error) {
      if (error instanceof AcpSoftTimeoutError) void error.late.then(took, () => false);
      this.effortMissed(session);
    }
  }

  private effortMissed(session: HostSession): void {
    if (this.sessions.get(session.id) !== session) return;
    this.opts.listener?.onNotice?.(session.id, { code: 'thinking-not-set', tone: 'warning' });
  }

  /** The session's list from the agent; a Work page hears of it when the rows it may send changed (Chat sends none). */
  private setCommands(session: HostSession, commands: AgentCommand[]): void {
    const before = session.commands === null ? null : JSON.stringify(allowedCommands(this.mode, session.commands));
    session.commands = commands;
    const view = allowedCommands(this.mode, commands);
    if (this.mode === 'work' && JSON.stringify(view) !== before) this.opts.listener?.onCommands?.(session.id, view);
  }

  /**
   * Compact now: `_x.ai/compact_conversation` for the session, loaded first when it is cold. Throws `conversation-busy`
   * while a turn, engine work or another op is live. The wait line shows from this call on (a manual run sends no
   * `auto_compact_started`). No deadline: the run ends by its reply, Stop or the agent's exit, and `done` gives how; a
   * start, load, guard or engine failure before the request is `failed`, a cold load past its soft deadline
   * `load-slow`. Throws `engine-lost` while the engine
   * restarted under commands or monitors: loading would restart the agent and stop them.
   */
  compact(sessionId: string, load: LoadInput, tokensBefore: number | null): { done: Promise<CompactOutcome> } {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (this.engineBusy(sessionId) || existing?.creating === true || (existing?.recap ?? null) !== null) {
      throw new AgentHostError('conversation-busy');
    }
    if (this.lostTasks()) throw new AgentHostError('engine-lost');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    if (!session.guarded) this.seedLedger(session, load.entries);
    const op: OpRecord = { kind: 'compact', sent: false, stopping: false };
    session.op = op;
    this.clearIdle();
    session.compaction = {
      phase: 'running',
      startedAt: this.now(),
      ...(tokensBefore === null ? {} : { tokensBefore }),
    };
    this.opts.listener?.onCompaction?.(session.id, { ...session.compaction });
    this.emitState();
    const done = this.runCompact(session, op, load).then((end) => {
      if (session.op === op) session.op = null;
      this.endCompaction(session, end);
      this.emitState();
      this.idleStep();
      return end;
    });
    return { done };
  }

  /**
   * Rewind: `_x.ai/rewind/execute` with `force` and `mode` always sent, the session loaded first when it is cold. A
   * rewind that puts files back first asks for the dry run again: the agent's commit overwrites every file it finds
   * changed, so one that changed since the plan without the plan showing it so ends it `changed`. A rewind that removes
   * conversation then stops the removed part's commands and monitors (`teardown`: no wake) and deletes its schedules;
   * one that fails refuses the rewind. One that does both and stopped anything asks for the dry run once more after the
   * stops, as the stopped work ran until then. A file of the dry run the commit did not put back ends it `partial`.
   * The dry run and the commit have no deadline and are never killed by the host: a kill mid-restore leaves the files
   * half put back. The op holds the session until the reply or the agent's exit. Throws `conversation-busy` while a
   * turn, engine work or another op is live, and `engine-lost` while the engine restarted under commands or monitors.
   */
  rewind(sessionId: string, load: LoadInput, input: RewindInput): { done: Promise<RewindHostEnd> } {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (this.engineBusy(sessionId) || existing?.creating === true || (existing?.recap ?? null) !== null) {
      throw new AgentHostError('conversation-busy');
    }
    if (this.lostTasks()) throw new AgentHostError('engine-lost');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    if (!session.guarded) this.seedLedger(session, load.entries);
    const op: OpRecord = { kind: 'rewind', sent: false, stopping: false };
    session.op = op;
    this.clearIdle();
    this.emitState();
    const done = this.runRewind(session, op, load, input).then((end) => {
      if (session.op === op) session.op = null;
      this.emitState();
      this.idleStep();
      return end;
    });
    return { done };
  }

  /**
   * Work: a rewind's dry run, `_x.ai/rewind/execute` with `force: false` and `mode: all` always sent, the session loaded
   * first when it is cold. The agent changes nothing and names the files a restore to `target` puts back, each clean or
   * changed outside it since it wrote it. It sets no op: a message sent meanwhile makes the plan stale instead. Soft: a
   * miss throws `agent-slow`, the agent lives and the late reply is dropped; a cold load past its soft deadline throws
   * `load-slow`. Throws like {@link rewind}, and `agent-protocol-error` for an error reply.
   */
  async rewindPreview(sessionId: string, load: LoadInput, target: number): Promise<RewindPreview> {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (this.engineBusy(sessionId) || existing?.creating === true || (existing?.recap ?? null) !== null) {
      throw new AgentHostError('conversation-busy');
    }
    if (this.lostTasks()) throw new AgentHostError('engine-lost');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    const caller: OpRecord = { kind: 'rewind', sent: false, stopping: false };
    this.clearIdle();
    try {
      await this.opts.providers.settled();
      const loaded = await this.ensureLoaded(session, load, caller, () => false);
      if (loaded === null) throw new AgentHostError('agent-crashed');
      if ('refused' in loaded) throw new AgentHostError(loaded.refused);
      let reply: unknown;
      try {
        reply = await loaded.running.process.client.requestSoft(
          '_x.ai/rewind/execute',
          { sessionId: session.id, targetPromptIndex: target, force: false, mode: REWIND_MODES.all },
          this.timings.rewindPreviewMs,
        );
      } catch (error) {
        if (error instanceof AcpSoftTimeoutError) {
          error.late.catch(() => {});
          throw new AgentHostError('agent-slow');
        }
        if (error instanceof AcpRpcError) throw new AgentHostError('agent-protocol-error');
        throw new AgentHostError('agent-crashed');
      }
      const preview = readRewindResult(reply);
      if (preview === null) throw new AgentHostError('agent-protocol-error');
      // A dry run names an error for a target past the agent's prompts, and for conflicts (which it lists).
      if (preview.error && preview.files.every((file) => file.state === 'clean')) return { refused: true };
      return { files: preview.files };
    } finally {
      this.idleStep();
    }
  }

  /**
   * Fork: `_x.ai/session/fork` copies the session on disk into a new one that is not loaded (its summary names it a
   * `fork` of this one); with `target`, up to and including that prompt and its answer. Both folders are the session's
   * own and no model is named. Starts a stopped agent and loads nothing. Resolves to the copy's session id. Soft: a
   * miss throws `agent-slow` and leaves the agent alive; a late reply that names a copy goes to `onForkLanded`. Throws
   * `conversation-busy` while a turn, engine work, another op or a recap is live, `engine-lost` while the engine
   * restarted under commands or monitors, `agent-protocol-error` for an error reply.
   */
  async fork(sessionId: string, load: LoadInput, target: number | null): Promise<string> {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (this.engineBusy(sessionId) || existing?.creating === true || (existing?.recap ?? null) !== null) {
      throw new AgentHostError('conversation-busy');
    }
    if (this.lostTasks()) throw new AgentHostError('engine-lost');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    const op: OpRecord = { kind: 'fork', sent: false, stopping: false };
    session.op = op;
    this.clearIdle();
    this.emitState();
    try {
      await this.opts.providers.settled();
      const running = await this.ensureRunning(op);
      const cwd = session.cwd !== '' ? session.cwd : await this.sessionCwd(this.mode === 'chat' ? undefined : '');
      const params = {
        sourceSessionId: sessionId,
        sourceCwd: cwd,
        newCwd: cwd,
        ...(target === null ? {} : { targetPromptIndex: target }),
      };
      op.sent = true;
      let reply: unknown;
      try {
        reply = await running.process.client.requestSoft('_x.ai/session/fork', params, this.timings.forkMs);
      } catch (error) {
        if (error instanceof AcpSoftTimeoutError) {
          error.late.then(
            (late) => {
              const landed = readForkResult(late);
              if (landed !== null && !this.closed) this.opts.listener?.onForkLanded?.(sessionId, landed.sessionId);
            },
            () => {},
          );
          throw new AgentHostError('agent-slow');
        }
        if (error instanceof AcpRpcError) throw new AgentHostError('agent-protocol-error');
        throw new AgentHostError('agent-crashed');
      }
      const copy = readForkResult(reply);
      if (copy === null) throw new AgentHostError('agent-protocol-error');
      return copy.sessionId;
    } finally {
      if (session.op === op) session.op = null;
      this.emitState();
      this.idleStep();
    }
  }

  /**
   * Recap: `_x.ai/recap` with `auto: false` (the app never asks for an automatic one), the session loaded first when it
   * is cold. The agent acks at once and later sends one `session_recap` (the row) or `session_recap_unavailable`. The
   * wait shows from this call on and `done` gives how it ended: `shown`; `none` (unavailable, no ack in time, or no
   * line within {@link RECAP_WAIT_MS}); `off` (recaps are off in the agent's settings); `cancelled` (a typed message
   * started a turn, which makes the agent drop the recap); `failed` (the start, load, guard or ack failed, or the agent
   * exited); `load-slow` (the cold load passed its soft deadline). Throws `conversation-busy` while a turn, a wake, a history op or another recap is live, and `engine-lost`
   * while the engine restarted under commands or monitors: loading would restart the agent and stop them.
   */
  recap(sessionId: string, load: LoadInput): { done: Promise<RecapEnd> } {
    if (this.closed) throw new AgentHostError('agent-crashed');
    const existing = this.sessions.get(sessionId);
    if (this.isBusy(sessionId) || existing?.creating === true || (existing?.recap ?? null) !== null) {
      throw new AgentHostError('conversation-busy');
    }
    if (this.lostTasks()) throw new AgentHostError('engine-lost');
    const session = existing ?? this.register(sessionId, load.cwd ?? '', load.modelId, false);
    if (!session.guarded) this.seedLedger(session, load.entries);
    let settle: (end: RecapEnd) => void = () => {};
    const done = new Promise<RecapEnd>((resolve) => {
      settle = resolve;
    });
    const wait: RecapWait = { startedAt: this.now(), timer: null, settle };
    session.recap = wait;
    this.clearIdle();
    this.opts.listener?.onRecap?.(session.id, { startedAt: wait.startedAt });
    void this.runRecap(session, wait, load);
    return { done };
  }

  /** Chat: the address ledger again from the entries a rewind kept: the cut part's addresses are no longer allowed. */
  rebuildLedger(sessionId: string, entries: readonly TranscriptEntry[]): void {
    const session = this.sessions.get(sessionId);
    if (this.mode !== 'chat' || session === undefined) return;
    session.ledger = new AddressLedger();
    this.seedLedger(session, entries);
  }

  /** Control Panel exit: cancel every turn, then the stop ladder. The host takes no more work. */
  async close(): Promise<void> {
    this.closed = true;
    this.clearIdle();
    this.watches?.clear();
    for (const s of this.sessions.values()) {
      if (s.activityTimer !== null) clearTimeout(s.activityTimer);
      s.activityTimer = null;
      if (s.recap?.timer != null) clearTimeout(s.recap.timer);
      clearStopTimers(s);
      clearRunStateTimers(s);
      if (s.turn !== null) this.cancelTurn(s, 'app-closed');
    }
    await Promise.all([this.abandonLaunch(), this.stopProcess()]);
    for (const s of this.sessions.values()) removeLedgerFile(s.ledgerFile);
    this.state = 'stopped';
  }

  // ── process lifecycle ──────────────────────────────────────────────────────────────────────────────────────────

  /**
   * `caller`: the turn, op or recap asking, which does not count as work in flight on a stale process. `sends`: the
   * user's message asks; with the engine lost, its commands and monitors do not hold the old process.
   */
  private async ensureRunning(caller?: Caller, sends = false): Promise<Running> {
    for (;;) {
      if (this.closed) throw new AgentHostError('agent-crashed');
      if (this.state === 'failed') throw new AgentHostError('agent-failed');
      if (this.state === 'blocked' && this.failure?.code === 'agent-incompatible') {
        throw new AgentHostError('agent-incompatible');
      }
      if (this.running !== null && !this.stale) return this.running;
      if (this.running !== null && this.stale && !this.hasWork(caller, !(sends && this.engineLost))) {
        await this.stopProcess();
      }
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
    this.engineLost = false;
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
    const loginPath = this.mode === 'work' ? await this.opts.loginPath : undefined;
    const endpoint = await this.opts.readEndpoint();
    if (endpoint === null) throw new AgentHostError('inference-stopped');
    const guardName = this.mode === 'work' ? workGuardName() : null;
    if (guardName === null) await this.prepareChatHome();
    else {
      await writeWorkGuard(this.opts.paths, workGuardManifest(guardName));
      await ensureWorkProcessDir(this.opts.paths);
      this.pdfToolOpen = await installPdfTool(this.opts.agentHome, this.opts.paths.workPdf).catch(() => false);
      const appPaths = await appReadPdfPaths(this.opts.agentHome).catch(() => []);
      this.appWorkflow = appPaths.length === 0 ? null : { name: READ_PDF_WORKFLOW, paths: appPaths };
    }
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
            agentHome: this.opts.agentHome,
            providerKeys: this.opts.providers.workKeys(),
            loginPath,
            claudeCode: this.opts.claudeCode?.() ?? true,
          });
    if (launch.env !== undefined) Object.assign(env, launch.env);
    // Never `$HOME`: the agent reads its process cwd as a project, and `$HOME/.grok` is grok-build's own home.
    const cwd = this.mode === 'chat' ? this.opts.paths.chatWorkspace : this.opts.paths.workProcess;
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
        messagesOff: false,
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
    await removeStaleLedgers(paths, (name) => this.knowsLedger(name));
  }

  /** A ledger file name of a session the host knows. */
  private knowsLedger(name: string): boolean {
    for (const s of this.sessions.values()) if (s.ledgerFile !== null && basename(s.ledgerFile) === name) return true;
    return false;
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
    const dropped: string[] = [];
    this.children.clear();
    const crashed = !exit.asked || protocolError;
    if (this.running === null) this.engineLost = false;
    for (const s of this.sessions.values()) {
      this.endCompaction(s);
      this.endHookRun(s);
      if (s.recap !== null) this.endRecap(s, s.recap, 'failed');
      this.closeOwned(s, (o) => o.kind !== 'turn', 'cancelled');
      s.revisions.clear();
      if (s.wakeDue !== null || s.runWake !== null) dropped.push(s.id);
      s.wakeDue = null;
      s.runWake = null;
      clearRunStateTimers(s);
      if (s.tracker.tickedCount() > 0 || s.tracker.ownHelperCount() > 0 || s.tracker.taskCount() > 0)
        this.activityChanged(s);
      s.tracker.clear();
      this.agentsSettled(s);
      s.runNames.clear();
      s.runCalls.clear();
      if (s.runs.size > 0 || s.endedRuns.size > 0) {
        clearStopTimers(s);
        s.runs.clear();
        s.endedRuns.clear();
        this.emitRuns(s);
      }
      if (s.wake !== null) {
        this.endWake(
          s,
          crashed
            ? { turnId: s.wake.turnId, outcome: 'agent-crashed', failure: chatFailure('agent-crashed') }
            : { turnId: s.wake.turnId, outcome: 'interrupted' },
        );
      }
      if (s.hot) cold.push(s.id);
      s.hot = false;
      s.guarded = false;
      if (s.commands !== null) this.opts.listener?.onCommands?.(s.id, null);
      s.commands = null;
      s.commandsPull = null;
      s.goal = null;
    }
    this.readyServers.clear();
    this.mcpOpened.clear();
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
    for (const id of dropped) this.opts.listener?.onWakeDropped?.(id);
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

  /**
   * A typed turn, a history op, a recap or a schedule's Delete other than `except`, engine work, or (with `tasks`) a
   * command or monitor is live: the process must keep running.
   */
  private hasWork(except?: Caller, tasks = true): boolean {
    for (const s of this.sessions.values()) {
      if ((s.turn !== null && s.turn !== except) || engineWork(s) || (s.op !== null && s.op !== except)) return true;
      if (s.recap !== null && s.recap !== except) return true;
      for (const record of s.deleting.values()) if (record !== except) return true;
      if (tasks && processWork(s)) return true;
    }
    return false;
  }

  /**
   * What holds off the idle stop, helpers left out (the timer's guard asks about them): {@link hasWork}, a schedule, or
   * a run that waits on the person (only this process can resume it). Neither holds a stale process: the restart does
   * not wait for them, and the timer's guard must still ask.
   */
  private holdsIdle(): boolean {
    for (const s of this.sessions.values()) {
      if (s.turn !== null || s.wake !== null || s.runs.size > 0 || s.op !== null || s.recap !== null) return true;
      if (s.deleting.size > 0 || processWork(s)) return true;
      if (!this.stale && (s.tracker.scheduleCount() > 0 || runsWait(s))) return true;
    }
    return false;
  }

  private helpersLive(): boolean {
    for (const s of this.sessions.values()) if (s.tracker.helperCount() > 0) return true;
    return false;
  }

  /** The engine restarted under live commands or monitors: only the user's message may restart the agent now. */
  private lostTasks(): boolean {
    if (!this.engineLost) return false;
    for (const s of this.sessions.values()) if (processWork(s)) return true;
    return false;
  }

  /** After a turn, a wake, a run, a helper, a command or a schedule ended: the stale restart, else the idle timer. */
  private idleStep(): void {
    if (this.stale) this.restartIfIdle();
    else this.scheduleIdle();
  }

  private restartIfIdle(): void {
    if (!this.stale || this.running === null) return;
    if (this.hasWork()) {
      // Only helpers hold it: the idle timer's guard drops the ones the agent no longer runs.
      this.scheduleIdle();
      return;
    }
    for (const s of this.sessions.values()) {
      if (s.wakeDue !== null || s.runWake !== null) {
        this.scheduleIdle();
        return;
      }
    }
    void this.stopIdle();
  }

  /** Stops an idle process; a stale host is fresh once it is gone. */
  private async stopIdle(): Promise<void> {
    await this.stopProcess();
    if (this.stale && this.running === null && this.starting === null && !this.closed) {
      this.stale = false;
      this.emitState();
    }
  }

  private clearIdle(): void {
    if (this.idleTimer !== null) clearTimeout(this.idleTimer);
    this.idleTimer = null;
  }

  private scheduleIdle(): void {
    this.clearIdle();
    if (this.running === null || this.holdsIdle() || this.closed) return;
    this.idleTimer = setTimeout(() => {
      this.idleTimer = null;
      void this.idleFired();
    }, this.timings.idleStopMs);
    this.idleTimer.unref?.();
  }

  /**
   * The idle timer: stops the process when nothing holds it. When only helpers do, `_x.ai/subagent/list_running`
   * (soft) drops the ones the agent no longer runs, whose end line was lost; a helper it still lists, one that started
   * after the request went, or a missed reply, keeps the process until the next timer.
   */
  private async idleFired(): Promise<void> {
    const running = this.running;
    if (running === null || this.closed || this.holdsIdle()) return;
    if (this.helpersLive()) {
      for (const s of this.sessions.values()) {
        if (s.tracker.helperCount() === 0) continue;
        const asked = s.tracker.helperMark();
        let listed: Set<string> | null;
        try {
          listed = readRunningSubagents(
            await running.process.client.requestSoft(
              '_x.ai/subagent/list_running',
              { sessionId: s.id },
              this.timings.listRunningMs,
            ),
          );
        } catch {
          listed = null;
        }
        if (this.running !== running || this.closed) return;
        const ticked = s.tracker.tickedCount();
        const agents = s.tracker.agentsVersion();
        if (listed !== null && this.sessions.get(s.id) === s && s.tracker.keepListed(listed, asked, runsHeld(s))) {
          this.agentsSettled(s);
          if (s.tracker.tickedCount() !== ticked || s.tracker.agentsVersion() !== agents) this.activityChanged(s);
          this.opts.listener?.onWork?.(s.id);
          this.emitState();
        }
      }
      if (this.holdsIdle() || this.helpersLive()) {
        this.idleStep();
        return;
      }
    }
    void this.stopIdle();
  }

  // ── sessions ───────────────────────────────────────────────────────────────────────────────────────────────────

  private register(
    id: string,
    cwd: string,
    modelId: string,
    hot = true,
    ledgerFile: string | null = this.newLedgerFile(),
  ): HostSession {
    const session: HostSession = {
      id,
      cwd,
      modelId,
      guarded: false,
      hot,
      creating: false,
      loading: null,
      loadCall: null,
      loadHard: false,
      ledger: new AddressLedger(),
      ledgerFile,
      ledgerDirty: false,
      web: NO_WEB,
      turn: null,
      interactions: new Map(),
      printed: new Set(),
      runs: new Map(),
      endedRuns: new Map(),
      runNames: new Map(),
      runCalls: new Map(),
      revisions: new Map(),
      wake: null,
      wakeDue: null,
      runWake: null,
      runStateTimers: new Map(),
      tracker: new ActivityTracker(),
      activityTimer: null,
      bg: { approvals: 0, approvalEpoch: 0 },
      editsGranted: false,
      granted: false,
      compaction: null,
      hookRun: null,
      op: null,
      commands: null,
      commandsPull: null,
      recap: null,
      deleting: new Map(),
      goal: null,
      efforts: null,
      effort: null,
    };
    this.sessions.set(id, session);
    return session;
  }

  private drop(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (session !== undefined) {
      this.closeInteractions(session, 'cancelled');
      if (session.activityTimer !== null) clearTimeout(session.activityTimer);
      if (session.recap?.timer != null) clearTimeout(session.recap.timer);
      clearStopTimers(session);
      clearRunStateTimers(session);
    }
    for (const [child, parent] of this.children) if (parent === session) this.children.delete(child);
    this.watches?.dropSession(sessionId);
    this.sessions.delete(sessionId);
    this.readyServers.delete(sessionId);
    this.mcpOpened.delete(sessionId);
    this.readyWaiters.delete(sessionId);
    if (session !== undefined) removeLedgerFile(session.ledgerFile);
  }

  /** Chat: a new ledger file path; the file is written before the session's `mlx_web` can start. */
  private newLedgerFile(): string | null {
    return this.mode === 'chat' ? join(this.opts.paths.webLedger, `${randomUUID()}.json`) : null;
  }

  /**
   * Writes the session's ledger file. Synchronous: the agent sends a call's result before the next call's permission
   * request, and both are read in order, so the file holds a page's links before a call that follows one can run. A
   * failed write leaves the older file (or none), which allows fewer addresses, never more.
   */
  private saveLedger(session: HostSession): void {
    session.ledgerDirty = !saveLedgerFile(session.ledgerFile, session.ledger.members());
  }

  private seedLedger(session: HostSession, entries: readonly TranscriptEntry[] | undefined): void {
    if (this.mode !== 'chat' || entries === undefined) return;
    for (const entry of entries) {
      if (entry.kind === 'user') session.ledger.addUserText(entry.text);
      else if (entry.kind === 'tool' && entry.output !== undefined && isWebToolName(entry.name)) {
        const result = parseWebResultText(entry.output);
        if (result !== null) applyWebResult(result, { ledger: session.ledger });
      }
    }
    this.saveLedger(session);
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

  private newParams(cwd: string, modelId: string, web: ChatWebEnv, ledgerFile: string | null): Record<string, unknown> {
    if (this.mode === 'work') {
      return workSessionParams({
        cwd,
        modelId,
        guardDir: this.opts.paths.workGuard,
        pluginDirs: this.opts.providers.workPluginDirs(),
      });
    }
    if (this.opts.chatPrompt === undefined || this.opts.webServer === undefined || ledgerFile === null) {
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
      ledgerFile,
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
    soft = false,
  ): Promise<GuardResult> {
    let mcpReady = false;
    if (this.mode === 'chat') mcpReady = await this.waitReady(session.id);
    let info: SessionInfoFacts | null = null;
    try {
      const params = { sessionId: session.id };
      const ms = ACP_TIMEOUTS['_x.ai/session/info'];
      info = readSessionInfo(
        soft && !session.loadHard
          ? await this.softLoadCall(running, '_x.ai/session/info', params, ms)
          : await this.call(running, '_x.ai/session/info', params, ms),
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

  /**
   * `session/load` and the guard. Hard (a turn's): each call's {@link ACP_TIMEOUTS} miss stops the agent. Soft: the load
   * has no deadline of its own (its waiter's {@link softWait} ends the op) and the guard's read is soft, until a turn
   * waits on it (`loadHard`): from then on each call is a turn's.
   */
  private async load(running: Running, session: HostSession, load: LoadInput, soft: boolean): Promise<GuardResult> {
    const cwd = await this.sessionCwd(this.mode === 'work' ? load.cwd : undefined);
    session.cwd = cwd;
    session.modelId = load.modelId;
    if (!running.models.some((m) => m.modelId === load.modelId)) throw new AgentHostError('model-unavailable');
    const web = this.sessionWeb();
    session.web = web;
    this.saveLedger(session);
    const params = loadSessionParams(this.newParams(cwd, load.modelId, web, session.ledgerFile), session.id);
    let answer: unknown;
    if (soft) {
      const call = running.process.client.start('session/load', params, null);
      if (session.loadHard) running.process.client.harden(call.id, ACP_TIMEOUTS['session/load']);
      session.loadCall = { client: running.process.client, id: call.id };
      try {
        answer = await this.replyOf(running, call.reply);
      } finally {
        if (session.loadCall?.id === call.id) session.loadCall = null;
      }
    } else {
      answer = await this.call(running, 'session/load', params, ACP_TIMEOUTS['session/load']);
    }
    const reply = readSessionReply(answer);
    session.efforts = reply.efforts;
    session.effort = reply.effort;
    const guard = await this.guard(running, session, { ...reply, sessionId: session.id }, load.modelId, false, soft);
    if (guard.ok && this.mode === 'work') {
      session.editsGranted = false;
      this.pullCommands(running, session);
      this.opts.listener?.onSessionReady?.(session.id);
    }
    return guard;
  }

  private loadOnce(running: Running, session: HostSession, load: LoadInput, soft: boolean): Promise<GuardResult> {
    if (session.loading === null) {
      const loading = this.load(running, session, load, soft).finally(() => {
        if (session.loading !== loading) return;
        session.loading = null;
        session.loadHard = false;
      });
      session.loading = loading;
    }
    return session.loading;
  }

  /**
   * The head of a turn, of every history op but a fork, of a recap and of a schedule's Delete: start the agent (a stale one is restarted when
   * nothing but `caller` runs), then load the session and pass its guard when this process has not. Null when `gone`
   * turned true between the steps. A turn's load is hard, as it always was. Any other caller's is soft: past
   * {@link HostTimings.loadSoftMs} it throws `load-slow` and the agent, with every command and run it holds, lives on;
   * the load goes on and its late reply loads the session, and nothing of the caller runs after it.
   */
  private async ensureLoaded(
    session: HostSession,
    load: LoadInput,
    caller: Caller,
    gone: () => boolean,
  ): Promise<Loaded | null> {
    const turn = 'turnId' in caller;
    const running = await this.ensureRunning(caller, turn);
    if (gone()) return null;
    if (session.guarded && this.sessions.has(session.id)) return { running };
    if (!this.sessions.has(session.id)) this.sessions.set(session.id, session);
    const loading = this.loadOnce(running, session, load, !turn);
    if (turn) {
      session.loadHard = true;
      session.loadCall?.client.harden(session.loadCall.id, ACP_TIMEOUTS['session/load']);
    }
    const guard = turn ? await loading : await this.softWait(loading);
    if (gone()) return null;
    return guard.ok ? { running } : { refused: guard.reason };
  }

  /** `loading` within {@link HostTimings.loadSoftMs}, else `load-slow`: the load runs on with no waiter. */
  private softWait(loading: Promise<GuardResult>): Promise<GuardResult> {
    return new Promise<GuardResult>((resolve, reject) => {
      const timer = setTimeout(() => {
        loading.catch(() => {});
        reject(new AgentHostError('load-slow'));
      }, this.timings.loadSoftMs);
      loading.then(
        (guard) => {
          clearTimeout(timer);
          resolve(guard);
        },
        (error: unknown) => {
          clearTimeout(timer);
          reject(error);
        },
      );
    });
  }

  // ── turns ──────────────────────────────────────────────────────────────────────────────────────────────────────

  private async runTurn(
    session: HostSession,
    turn: TurnRecord,
    text: string,
    load: LoadInput,
    attachments: readonly AttachmentInput[],
    meta: PromptMeta | undefined,
    sendable?: () => boolean,
  ): Promise<TurnResult> {
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
          const loaded = await this.ensureLoaded(session, load, turn, () => turn.recorded !== null);
          if (loaded === null || turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded!) });
          if ('refused' in loaded) return unsent({ outcome: 'refused', failure: loaded.refused });
          let command = checkCommand(this.mode, text, session.commands);
          if (command === 'needs-list') {
            command = checkCommand(this.mode, text, await this.agentCommands(session.id));
            if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
          }
          if (command === 'refused' || command === 'needs-list') {
            return unsent({ outcome: 'refused', failure: 'slash-text' });
          }
          const running = loaded.running;
          await this.applyEffort(running, session, load.thinking);
          if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
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
          this.engineMoved();
          if (attempt > 0 || this.hasWork(turn, !this.engineLost)) return unsent({ failure: 'inference-failed' });
        }
      } catch (error) {
        return unsent({ failure: error instanceof AgentHostError ? error.code : 'agent-crashed' });
      }
    };
    const ready = await Promise.race([prepare(), aborted]);
    if (!('process' in ready)) return ready;
    if (turn.recorded !== null) return unsent({ outcome: causeOutcome(turn.recorded) });
    if (sendable?.() === false) return unsent({});
    const running = ready;
    // An agent that does not take embedded resources would drop the file text and the PDF notes.
    if (running.embeddedContext !== true && attachments.some((a) => a.kind === 'file' || a.kind === 'pdf')) {
      return unsent({ failure: 'agent-incompatible' });
    }
    if (this.mode === 'chat') {
      session.ledger.addUserText(text);
      this.saveLedger(session);
    }
    turn.abort = null;
    turn.phase = 'running';
    this.emitTurn(session);
    if (this.mode === 'chat') this.armCap(session, turn, this.timings.chatTurnMaxMs);
    let exited = false;
    const onExit: ExitWaiter = () => {
      exited = true;
      turn.gap?.();
    };
    this.exitWaiters.add(onExit);
    try {
      let prompt = text;
      let blocks = attachments;
      let promptMeta = meta;
      for (;;) {
        turn.between = false;
        turn.toolSeen = false;
        turn.endSeen = false;
        const sent = running.process.client.start(
          'session/prompt',
          promptParams(session.id, prompt, blocks, promptMeta),
          null,
        );
        turn.prompt = { client: running.process.client, id: sent.id };
        if (turn.continues > 0) this.opts.listener?.onAutoContinue?.(session.id, turn.continues);
        const end = await this.promptEnd(sent.reply, turn, running, () => exited);
        if (end.outcome !== 'output-limit' || this.mode !== 'work') return this.withContinues(turn, end);
        turn.between = true;
        const recordedAtReply = turn.recorded;
        turn.lengthStops = turn.toolSeen ? 1 : turn.lengthStops + 1;
        if (turn.lengthStops >= 2) return this.lengthEnd(turn, end, 'second-length', recordedAtReply);
        const block = await this.continueGate(turn, running, () => exited);
        if (block !== null) return this.lengthEnd(turn, end, block, recordedAtReply);
        turn.continues += 1;
        this.emitTurn(session);
        prompt = CONTINUE_PROMPT;
        blocks = [];
        promptMeta = undefined;
      }
    } finally {
      turn.between = false;
      turn.gap = null;
      this.exitWaiters.delete(onExit);
    }
  }

  /** The end of one `session/prompt`: the reply, or the error it got. */
  private async promptEnd(
    sent: Promise<unknown>,
    turn: TurnRecord,
    running: Running,
    exited: () => boolean,
  ): Promise<TurnResult> {
    try {
      const result = await sent;
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
      if (exited() || !running.process.running || running.process.brokeProtocol) {
        return { turnId: turn.turnId, outcome: 'agent-crashed', failure: chatFailure('agent-crashed') };
      }
      return { turnId: turn.turnId, outcome: 'model-error' };
    }
  }

  private withContinues(turn: TurnRecord, end: TurnResult): TurnResult {
    return turn.continues > 0 ? { ...end, continues: turn.continues } : end;
  }

  /** A Work turn that ends on its length reply: the count, and why no continue went out. */
  private lengthEnd(
    turn: TurnRecord,
    end: TurnResult,
    noContinue: NoContinue,
    recordedAtReply: TurnEndCause | null,
  ): TurnResult {
    return {
      ...this.withContinues(turn, end),
      lengthStops: turn.lengthStops,
      noContinue,
      ...(recordedAtReply === null && turn.recorded !== null ? { late: turn.recorded } : {}),
    };
  }

  /**
   * Work, after a length reply: null when the automatic continue may go out. It waits for the agent's own end line of
   * the reply (no timer), then re-reads the endpoint. Woken by that line, the agent's exit, Stop and a stale host.
   */
  private async continueGate(turn: TurnRecord, running: Running, exited: () => boolean): Promise<NoContinue | null> {
    const check = (): NoContinue | null => {
      if (turn.recorded !== null) return turn.recorded === 'inference-lost' ? 'endpoint' : 'stop';
      if (exited() || !running.process.running || running.process.brokeProtocol) return 'agent-exit';
      if (this.stale || this.running !== running) return 'host-stale';
      return null;
    };
    let block = check();
    if (block !== null) return block;
    if (!turn.endSeen) {
      await new Promise<void>((resolve) => {
        turn.gap = resolve;
      });
      turn.gap = null;
      block = check();
      if (block !== null) return block;
    }
    try {
      await this.checkEndpoint();
    } catch {
      return 'endpoint';
    }
    return check();
  }

  private finishTurn(session: HostSession, turn: TurnRecord, result: TurnResult): void {
    if (turn.capTimer !== null) clearTimeout(turn.capTimer);
    if (turn.graceTimer !== null) clearTimeout(turn.graceTimer);
    if (session.turn === turn) session.turn = null;
    this.endCompaction(session);
    this.endHookRun(session);
    this.closeOwned(session, (o) => o.kind === 'turn' && o.turn === turn, 'cancelled');
    this.opts.listener?.onTurn?.(session.id, null);
    this.opts.listener?.onTurnEnd?.(session.id, result);
    // A wake whose first lines came in the stdout chunk of this turn's reply shows from here.
    if (session.wake !== null && session.turn === null) this.emitTurn(session);
    this.emitState();
    this.idleStep();
  }

  private armCap(session: HostSession, turn: TurnRecord, ms: number): void {
    turn.capLeft = ms;
    turn.capFrom = this.now();
    turn.capTimer = setTimeout(() => this.cancelTurn(session, 'turn-timeout'), ms);
  }

  /** A compaction holds the Chat cap: the time left is kept, and the cap runs on from it when the compaction ends. */
  private pauseCap(turn: TurnRecord): void {
    if (turn.capTimer === null || turn.capLeft === null) return;
    clearTimeout(turn.capTimer);
    turn.capTimer = null;
    turn.capLeft = Math.max(0, turn.capLeft - (this.now() - turn.capFrom));
  }

  private resumeCap(session: HostSession, turn: TurnRecord): void {
    if (turn.capTimer !== null || turn.capLeft === null || turn.phase !== 'running') return;
    this.armCap(session, turn, turn.capLeft);
  }

  private onCompaction(session: HostSession, update: CompactionUpdate): void {
    const turn = session.turn;
    if (update.kind !== 'started') {
      if (turn !== null) this.resumeCap(session, turn);
      this.endCompaction(session);
      return;
    }
    if (turn !== null) this.pauseCap(turn);
    this.endHookRun(session);
    session.compaction = {
      phase: 'running',
      startedAt: this.now(),
      ...(update.tokensUsed === null ? {} : { tokensBefore: update.tokensUsed }),
    };
    this.opts.listener?.onCompaction?.(session.id, { ...session.compaction });
  }

  private endHookRun(session: HostSession): void {
    if (session.hookRun === null) return;
    session.hookRun = null;
    this.opts.listener?.onHookRun?.(session.id, null);
  }

  /** A manual run's state ends only with its `end`: its end line and the agent's exit come before its reply. */
  private endCompaction(session: HostSession, end?: CompactOutcome): void {
    if (end === undefined && (session.compaction === null || session.op?.kind === 'compact')) return;
    session.compaction = null;
    this.opts.listener?.onCompaction?.(session.id, null, end);
  }

  private async runCompact(session: HostSession, op: OpRecord, load: LoadInput): Promise<CompactOutcome> {
    try {
      await this.opts.providers.settled();
      let running: Running | null = null;
      for (let attempt = 0; running === null; attempt++) {
        const loaded = await this.ensureLoaded(session, load, op, () => op.stopping);
        if (loaded === null) return 'stopped';
        if ('refused' in loaded) return 'failed';
        const endpoint = await this.opts.readEndpoint();
        if (op.stopping) return 'stopped';
        if (endpoint === null) return 'failed';
        if (sameEndpoint(endpointFingerprint(endpoint), loaded.running.fingerprint)) {
          running = loaded.running;
        } else {
          // The old process holds the old address or token: restart it once when nothing else runs on it.
          this.engineMoved();
          if (attempt > 0 || this.hasWork(op)) return 'failed';
        }
      }
      op.sent = true;
      await running.process.client.request('_x.ai/compact_conversation', { sessionId: session.id }, null);
      return 'compacted';
    } catch (error) {
      if (op.stopping) return 'stopped';
      if (error instanceof AgentHostError && error.code === 'load-slow') return 'load-slow';
      return error instanceof AcpRpcError && error.kind === 'compact_cancelled' ? 'stopped' : 'failed';
    }
  }

  private async runRewind(
    session: HostSession,
    op: OpRecord,
    load: LoadInput,
    input: RewindInput,
  ): Promise<RewindHostEnd> {
    try {
      await this.opts.providers.settled();
      const loaded = await this.ensureLoaded(session, load, op, () => false);
      if (loaded === null) return { outcome: 'stopped' };
      if ('refused' in loaded) return { outcome: 'failed', failure: loaded.refused };
      const running = loaded.running;
      const execute = async (force: boolean) =>
        readRewindResult(
          await running.process.client.request(
            '_x.ai/rewind/execute',
            { sessionId: session.id, targetPromptIndex: input.target, force, mode: REWIND_MODES[input.mode] },
            null,
          ),
        );
      const shown = new Map(input.files.map((file) => [file.path, file.state]));
      const checkFiles = async (): Promise<RewindHostEnd | string[]> => {
        const check = await execute(false);
        if (check === null || (check.error && check.files.every((file) => file.state === 'clean'))) {
          return { outcome: 'refused' };
        }
        const changed = check.files.some((file) => {
          const was = shown.get(file.path);
          return was === undefined || (was === 'clean' && file.state !== 'clean');
        });
        return changed ? { outcome: 'changed' } : check.files.map((file) => file.path);
      };
      let restores: string[] = [];
      if (input.mode !== 'conversation') {
        const check = await checkFiles();
        if (!Array.isArray(check)) return check;
        restores = check;
      }
      if (input.mode !== 'files') {
        const ends = await Promise.all([
          ...input.tasks.map((task) => this.stopTask(session.id, task, 'teardown')),
          ...input.schedules.map((schedule) => this.requestScheduleDelete(running, session.id, schedule)),
          ...input.runs.map((runId) => this.stopRunAs(session.id, runId, 'you')),
        ]);
        if (ends.includes('failed')) return { outcome: 'tasks' };
        // The work being stopped runs until its stop and can write a file the last check found clean.
        if (input.mode === 'all' && ends.length > 0) {
          const check = await checkFiles();
          if (!Array.isArray(check)) return check;
          restores = check;
        }
      }
      op.sent = true;
      const reply = await execute(true);
      if (reply === null || !reply.success) return { outcome: 'refused' };
      const failed = restores.filter((path) => !reply.reverted.includes(path));
      if (failed.length > 0) return { outcome: 'partial', restored: reply.reverted.length, failed };
      return { outcome: 'done', restored: reply.reverted.length };
    } catch (error) {
      if (error instanceof AgentHostError) return { outcome: 'failed', failure: error.code };
      if (error instanceof AcpRpcError) return { outcome: 'refused' };
      return { outcome: 'stopped' };
    }
  }

  private async runRecap(session: HostSession, wait: RecapWait, load: LoadInput): Promise<void> {
    const gone = (): boolean => session.recap !== wait;
    try {
      await this.opts.providers.settled();
      let running: Running | null = null;
      for (let attempt = 0; running === null; attempt++) {
        if (gone()) return;
        const loaded = await this.ensureLoaded(session, load, wait, gone);
        if (loaded === null) return;
        const endpoint = 'refused' in loaded ? null : await this.opts.readEndpoint();
        if (gone()) return;
        if ('refused' in loaded || endpoint === null) {
          this.endRecap(session, wait, 'failed');
          return;
        }
        if (sameEndpoint(endpointFingerprint(endpoint), loaded.running.fingerprint)) {
          running = loaded.running;
        } else {
          // The old process holds the old address or token: restart it once when nothing else runs on it.
          this.engineMoved();
          if (attempt > 0 || this.hasWork(wait)) {
            this.endRecap(session, wait, 'failed');
            return;
          }
        }
      }
      wait.timer = setTimeout(() => this.endRecap(session, wait, 'none'), this.timings.recapMs);
      let ack: { disabled: boolean } | null;
      try {
        ack = readRecapAck(
          await running.process.client.requestSoft(
            '_x.ai/recap',
            { sessionId: session.id, auto: false },
            this.timings.recapAckMs,
          ),
        );
      } catch (error) {
        this.endRecap(session, wait, error instanceof AcpSoftTimeoutError ? 'none' : 'failed');
        return;
      }
      if (ack === null) this.endRecap(session, wait, 'failed');
      else if (ack.disabled) this.endRecap(session, wait, 'off');
    } catch (error) {
      this.endRecap(
        session,
        wait,
        error instanceof AgentHostError && error.code === 'load-slow' ? 'load-slow' : 'failed',
      );
    }
  }

  /** A deleted session's recap wait ends `cancelled`, so its run stops and never loads the deleted session again. */
  private recapDeleted(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (session?.recap != null) this.endRecap(session, session.recap, 'cancelled');
  }

  /** Ends this recap wait once; a later end, or one for a wait that was replaced, does nothing. */
  private endRecap(session: HostSession, wait: RecapWait, end: RecapEnd): void {
    if (session.recap !== wait) return;
    session.recap = null;
    if (wait.timer !== null) clearTimeout(wait.timer);
    wait.settle(end);
    this.opts.listener?.onRecap?.(session.id, null, end);
    this.idleStep();
  }

  /** `_x.ai/scheduler/delete` (soft): `failed` when the agent gave no answer in time or an error answer. */
  private async requestScheduleDelete(
    running: Running,
    sessionId: string,
    scheduleId: string,
  ): Promise<ScheduleDelete> {
    try {
      const end = readScheduleDelete(
        await running.process.client.requestSoft(
          '_x.ai/scheduler/delete',
          { sessionId, taskId: scheduleId },
          this.timings.scheduleDeleteMs,
        ),
      );
      return end ?? 'failed';
    } catch {
      return 'failed';
    }
  }

  /** Stop of a manual compaction: `session/cancel` once the request went out (the agent's no-turn path), else none. */
  private stopCompact(session: HostSession): boolean {
    const op = session.op;
    if (op === null || op.kind !== 'compact') return false;
    if (op.stopping) return true;
    op.stopping = true;
    if (session.compaction !== null) {
      session.compaction = { ...session.compaction, phase: 'stopping' };
      this.opts.listener?.onCompaction?.(session.id, { ...session.compaction });
    }
    const running = this.running;
    if (op.sent && running !== null) {
      try {
        running.process.client.notify('session/cancel', { sessionId: session.id });
      } catch {
        // The process is gone: the run ends at its exit.
      }
    }
    return true;
  }

  private cancelTurn(session: HostSession, cause: TurnEndCause): void {
    const turn = session.turn;
    if (turn === null) return;
    if (turn.recorded === null) turn.recorded = cause;
    this.closeOwned(session, (o) => o.kind === 'turn' && o.turn === turn, 'cancelled');
    // No prompt is in flight: nothing goes to the agent, and the turn ends on its length reply.
    if (turn.between) {
      turn.gap?.();
      return;
    }
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
    this.closeOwned(session, () => true, reason);
  }

  /** Withdraws the open requests whose owner matches. */
  private closeOwned(session: HostSession, match: (owner: Owner) => boolean, reason: 'cancelled' | 'expired'): void {
    for (const [id, open] of session.interactions) {
      if (!match(open.owner)) continue;
      session.interactions.delete(id);
      open.settle(open.withdrawn);
      this.opts.listener?.onInteractionClosed?.(session.id, id, reason);
      this.mcpClosed(session, open.interaction, open.withdrawn, reason);
    }
  }

  /** A typed turn that owns an answered or withdrawn card ends as `denied`; a run or the wake records nothing. */
  private recordDenied(session: HostSession, owner: Owner, cause: TurnEndCause = 'denied'): void {
    if (owner.kind === 'turn' && session.turn === owner.turn && owner.turn.recorded === null) {
      owner.turn.recorded = cause;
    }
  }

  /** Auto's count of an owner: the typed turn's, or the session's for runs and the wake; null for a gone turn. */
  private countOf(session: HostSession, owner: Owner): ApprovalCount | null {
    if (owner.kind !== 'turn') return session.bg;
    return session.turn === owner.turn ? owner.turn : null;
  }

  // ── workflow runs and the wake turn ──────────────────────────────────────────────────────────────────────────

  /**
   * The highest revision wins; a run is live while its status is `active`, and leaving `active` closes its cards and
   * moves it to the ended list, where later revisions (a drain, a stop of a paused run) still reach it. `agentMs`: the
   * agent's stamp of the update's line.
   */
  private onWorkflow(session: HostSession, update: WorkflowUpdate, agentMs: number | null): void {
    const seen = session.revisions.get(update.runId);
    if (seen !== undefined && update.revision <= seen) return;
    session.revisions.set(update.runId, update.revision);
    session.runNames.set(update.runId, update.name);
    this.readRunStateSoon(session, update.runId);
    const due = session.runWake;
    if (due?.runId === update.runId) {
      if (REPORTED.has(update.status)) due.revision = update.revision;
      else session.runWake = null;
    }
    const live = session.runs.get(update.runId);
    const ended = session.endedRuns.get(update.runId);
    if (update.status === 'active') {
      if (live !== undefined) {
        if (live.name === update.name) return;
        live.name = update.name;
        this.emitRuns(session);
        return;
      }
      if (ended !== undefined) {
        clearStopTimer(ended);
        session.endedRuns.delete(update.runId);
      }
      session.runs.set(update.runId, {
        name: update.name,
        status: update.status,
        stopping: false,
        stopBy: null,
        stopTimer: null,
        by: null,
        toolVerb: null,
      });
      this.clearIdle();
      this.emitRuns(session);
      this.emitState();
      return;
    }
    const run = live ?? ended;
    if (run === undefined) return;
    const waited = runsWait(session);
    // A run that waited has no watcher left to send a drain update: its stop is over once no agent of it runs.
    const drained =
      update.lastEvent === 'workflow_outcome_ignored' ||
      (WAITING.has(run.status) && !session.tracker.runHelping(update.runId));
    run.name = update.name;
    if (update.status !== run.status) {
      const mover = this.runMover(session, update.runId, run, update.status);
      if (!REPORTED.has(run.status) && this.agentWakes(session, update.runId, update.status, run, mover)) {
        session.runWake = {
          runId: update.runId,
          revision: update.revision,
          since: agentMs,
          behind: session.wakeDue !== null,
        };
      }
      if (mover !== null) run.by = mover;
      run.status = update.status;
      run.toolVerb = null;
    }
    if (run.stopping && (update.status !== 'cancelled' || drained)) endStopping(run);
    if (live === undefined) {
      this.emitRuns(session);
      if (runsWait(session) !== waited || session.runWake !== due) this.idleStep();
      return;
    }
    session.runs.delete(update.runId);
    session.endedRuns.set(update.runId, run);
    for (const [runId, old] of session.endedRuns) {
      if (session.endedRuns.size <= ENDED_RUNS_KEPT) break;
      clearStopTimer(old);
      session.endedRuns.delete(runId);
    }
    // First, so the ops queued before it (this update's line) reach the page with it.
    this.emitRuns(session);
    this.closeOwned(session, (o) => o.kind === 'run' && o.runId === update.runId, 'cancelled');
    this.emitState();
    this.idleStep();
  }

  /**
   * The agent queues a wake turn for a run that reached a reportable status (grok `run_loop.rs`
   * `WorkflowCompletionTurn`), unless the run is marked reported (the main agent's own stop, or Stop run here), the
   * goal loop runs, or a wake of a run is queued or running already. Its cancel barrier needs a `cancelTrigger`,
   * which this host never sends. Only the run's watcher queues it, as it ends (grok `manager.rs`): a stop of a run
   * that waits finds it ended once the run's helpers did, while a stop during a pause's drain is what it reports.
   */
  private agentWakes(
    session: HostSession,
    runId: string,
    status: string,
    run: RunRecord,
    mover: RunBy | null,
  ): boolean {
    if (!REPORTED.has(status) || (status === 'cancelled' && run.stopping) || mover === 'main-agent') return false;
    if (status === 'cancelled' && WAITING.has(run.status) && !session.tracker.runHelping(runId)) return false;
    if (session.goal === 'active') return false;
    return session.runWake === null && session.wake?.runId == null;
  }

  /**
   * A `turn_completed` while a run's wake is queued: that wake ran (the host took it for another turn), or a turn
   * that started after the run's end reported the run at its start (grok `drain_between_turn_completions`), so the
   * agent drops its queued wake. A turn that started before the run's end, or that has no times, ends nothing.
   */
  private runWakeEnds(session: HostSession, line: UpdateLine): boolean {
    const due = session.runWake;
    if (due === null) return false;
    const promptId = typeof line.update.prompt_id === 'string' ? line.update.prompt_id : '';
    if (!promptId.startsWith(`${WORKFLOW_WAKE_PREFIX}${due.runId}-`)) {
      if (session.goal === 'active' || due.since === null) return false;
      const at = line.meta?.agentTimestampMs;
      const elapsed = line.update.elapsed_ms;
      if (typeof at !== 'number' || typeof elapsed !== 'number' || at - elapsed <= due.since) return false;
    }
    session.runWake = null;
    return true;
  }

  /** Reads a run's `state.json` {@link RUN_STATE_READ_MS} after its last new revision, through the reader's rules. */
  private readRunStateSoon(session: HostSession, runId: string): void {
    if (this.mode !== 'work' || this.opts.sessionDir === undefined || this.closed) return;
    const pending = session.runStateTimers.get(runId);
    if (pending !== undefined) clearTimeout(pending);
    const timer = setTimeout(() => {
      session.runStateTimers.delete(runId);
      if (this.sessions.get(session.id) !== session) return;
      const dir = this.opts.sessionDir?.(session.id) ?? null;
      if (dir === null) return;
      void readRunState(dir, runId).then((read) => {
        if (read === null || this.closed || this.sessions.get(session.id) !== session) return;
        this.opts.listener?.onRunState?.(session.id, read.state, read.fileTs);
      });
    }, this.timings.runStateMs);
    timer.unref?.();
    session.runStateTimers.set(runId, timer);
  }

  /**
   * Who moved a run to `status`: this host's stop (a cancel), the main agent's pause or stop call of it, or a typed
   * `/workflow` command whose name picks it. Null when none of them did.
   */
  private runMover(session: HostSession, runId: string, run: RunRecord, status: string): RunBy | null {
    if (run.stopping && run.stopBy !== null && status === 'cancelled') return run.stopBy;
    if (run.toolVerb !== null && verbMoves(run.toolVerb, status)) return 'main-agent';
    const typed = session.turn?.runCommand ?? null;
    if (typed !== null && verbMoves(typed.verb, status) && runsNamed(typed.name, session.runNames).includes(runId)) {
      return 'you';
    }
    return null;
  }

  /**
   * Work: a `workflow` call of the main agent that pauses or stops a run marks that run, so its next status names the
   * main agent and starts no wake. When the call's input came only after the run moved, its end does both.
   */
  private trackRunCall(session: HostSession, update: Record<string, unknown>): void {
    const line = readWorkflowCall(update);
    if (line === null) return;
    let call = session.runCalls.get(line.toolCallId);
    if (call === undefined) {
      if (line.tool !== WORKFLOW_TOOL) return;
      call = { verb: null, runId: null };
      session.runCalls.set(line.toolCallId, call);
    }
    if (call.verb === null && line.control !== null) {
      call.verb = line.control.verb;
      call.runId = findRun(line.control.key, session.runNames);
      const run = call.runId === null ? undefined : session.runs.get(call.runId);
      if (run !== undefined) run.toolVerb = call.verb;
    }
    if (line.ended === null) return;
    session.runCalls.delete(line.toolCallId);
    if (call.verb === null || call.runId === null) return;
    const live = session.runs.get(call.runId);
    if (live !== undefined) {
      if (line.ended === 'failed' && live.toolVerb === call.verb) live.toolVerb = null;
      return;
    }
    const ended = session.endedRuns.get(call.runId);
    if (line.ended === 'failed' || ended === undefined || ended.by !== null || !verbMoves(call.verb, ended.status)) {
      return;
    }
    ended.by = 'main-agent';
    if (session.runWake?.runId === call.runId) {
      session.runWake = null;
      this.idleStep();
    }
    this.emitRuns(session);
  }

  /**
   * A content update with no typed prompt open starts the wake turn. When it comes in the stdout chunk of the typed
   * turn's reply, the typed turn has not ended yet: the wake shows once it has ({@link finishTurn}).
   */
  private startWake(session: HostSession): void {
    const startedAt = this.now();
    const run = session.runWake !== null && !session.runWake.behind ? session.runWake : null;
    session.wake = { turnId: randomUUID(), startedAt, phase: 'running', runId: run?.runId ?? null };
    const due = session.wakeDue;
    session.wakeDue = null;
    if (run !== null) session.runWake = null;
    else if (session.runWake !== null) session.runWake.behind = false;
    this.clearIdle();
    if (session.turn === null) this.emitTurn(session);
    this.emitState();
    // The report names the row by its `ts`: it reaches the page before the row, so the row never shows another title.
    const reported = session.tracker.wakeStarted(startedAt);
    if (reported) this.activityNow(session);
    const promptId =
      run !== null
        ? `${WORKFLOW_WAKE_PREFIX}${run.runId}-${run.revision}`
        : due !== null
          ? wakePromptId(due)
          : reported
            ? NOTIFICATIONS_WAKE
            : null;
    // The agent writes a wake's prompt to disk only. Without this line the live transcript has no wake row, and the
    // answer joins the turn before it. A wake nothing made due gets its cause from its `turn_completed`; one after
    // monitor events is the agent's drain of them until then.
    this.opts.listener?.onUpdate?.(session.id, {
      xai: false,
      update: {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: '' },
        _meta: { hideFromScrollback: true, ...(promptId === null ? {} : { [WAKE_PROMPT_ID]: promptId }) },
      },
      meta: { agentTimestampMs: startedAt },
    });
    // After the row: the runs event carries the row's op, so the report's home never moves.
    if (run !== null) this.emitRuns(session);
  }

  private endWake(session: HostSession, result: TurnResult): void {
    if (session.wake === null) return;
    session.wake = null;
    this.endHookRun(session);
    this.closeOwned(session, (o) => o.kind === 'wake', 'cancelled');
    if (session.turn === null) this.opts.listener?.onTurn?.(session.id, null);
    this.opts.listener?.onTurnEnd?.(session.id, result);
    this.emitState();
    this.idleStep();
  }

  private emitRuns(session: HostSession): void {
    this.opts.listener?.onRuns?.(session.id, runViews(session));
  }

  /** The page gets the live facts once the time since this first change passed: ten ticks in it make one event. */
  private activityChanged(session: HostSession): void {
    if (session.activityTimer !== null || this.closed) return;
    session.activityTimer = setTimeout(() => {
      session.activityTimer = null;
      if (this.sessions.get(session.id) !== session) return;
      this.opts.listener?.onActivity?.(session.id, this.liveOf(session));
    }, this.timings.activityEveryMs);
    session.activityTimer.unref?.();
  }

  /**
   * A press of the user (Stop) changed the live facts: they go to the page now, not at the next window's end, so the
   * row reads Stopping… while the press is fresh. A pending window still fires.
   */
  private activityNow(session: HostSession): void {
    if (this.closed) return;
    this.opts.listener?.onActivity?.(session.id, this.liveOf(session));
  }

  /** The session's live facts for the page: the tracker's, with where each Delete of a schedule is. */
  private liveOf(s: HostSession): LiveActivity {
    const deleting = new Map<string, ScheduleDeleting>();
    for (const [id, record] of s.deleting) deleting.set(id, record.phase);
    return s.tracker.live(deleting, this.running?.messagesOff === true);
  }

  /** Work: a teed line may announce a wake, or start or end a helper, a command, a monitor or a schedule. */
  private trackWork(session: HostSession, line: UpdateLine): void {
    const due = wakeDueOf(line);
    if (due !== null) session.wakeDue = due;
    const ticked = session.tracker.tickedCount();
    const tasks = session.tracker.tasksVersion();
    const schedules = session.tracker.schedulesVersion();
    const agents = session.tracker.agentsVersion();
    const moved = (): boolean =>
      session.tracker.tasksVersion() !== tasks ||
      session.tracker.schedulesVersion() !== schedules ||
      session.tracker.agentsVersion() !== agents;
    const applied = session.tracker.apply(line);
    if (session.tracker.agentsVersion() !== agents) this.agentsSettled(session);
    if (!applied) {
      if (moved()) this.activityChanged(session);
      return;
    }
    if (session.tracker.tickedCount() !== ticked || moved()) this.activityChanged(session);
    this.workChanged(session);
  }

  /** Work: the agents of the session the tracker no longer runs have ended, for their kept lines and views. */
  private agentsSettled(session: HostSession): void {
    this.watches?.settle(session.id, (child) => session.tracker.agent(child)?.childId === child);
  }

  /** What runs in the session changed: the list dot, the mode state and the idle timer follow. */
  private workChanged(session: HostSession): void {
    this.opts.listener?.onWork?.(session.id);
    this.emitState();
    this.idleStep();
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
    if (method === '_x.ai/mcp/elicit_complete') {
      this.linkDone(params);
      return;
    }
    const message = readSessionUpdate(method, params);
    if (message === null) return;
    const asked = message.xai ? readAskSignal(message.update) : null;
    if (asked !== null) {
      if (asked.kind === 'resolved' && asked.toolCallId !== null) this.expireAsk(message.sessionId, asked.toolCallId);
      return;
    }
    const session = this.sessions.get(message.sessionId);
    if (session === undefined) {
      // A helper's own line: its call leaves the folder, its output is checked for keys, it says what the helper
      // does now, and the Agent view keeps it.
      const parent = this.mode === 'work' ? this.children.get(message.sessionId) : undefined;
      if (parent !== undefined) {
        this.trackCall(parent, message.update, message.sessionId);
        this.checkPrinted(parent, message.update);
        if (parent.tracker.childLine(message.sessionId, message.update)) this.activityChanged(parent);
        const line: UpdateLine = { xai: message.xai, update: message.update, meta: message.meta };
        if (liveFilter(line, { turnRunning: false, modelId: '' }) === 'pass') {
          const running = parent.tracker.agent(message.sessionId)?.childId === message.sessionId;
          this.watches?.line(parent.id, message.sessionId, line, running);
        }
        return;
      }
      // The agent runs `session_start` hooks while `session/new` is still open: their stored lines (the batch, then
      // a `hook_annotation` for each hook's own message) reach the conversation once the service holds it.
      const early: UpdateLine = { xai: message.xai, update: message.update, meta: message.meta };
      const hook = message.xai ? readHookRun(message.update) : null;
      const start = hook?.kind === 'ended' && hook.event === 'session_start';
      const note =
        message.xai && message.update.sessionUpdate === 'hook_annotation' && this.startedEarly.has(message.sessionId);
      if (
        this.sessionsNew > 0 &&
        (start || note) &&
        liveFilter(early, { turnRunning: false, modelId: '' }) === 'pass'
      ) {
        if (start) this.startedEarly.add(message.sessionId);
        this.opts.listener?.onUpdate?.(message.sessionId, early);
      }
      return;
    }
    if (!message.xai && message.update.sessionUpdate === 'available_commands_update') {
      // The list is the host's alone (never stored, never an entry). But after each answer the agent re-sends this
      // line to carry the answer's token count (`_meta.totalTokens`, `AdvertiseTrigger::UsageMeta`): taking it would
      // leave the context count on the stored estimate, so a line with a count goes on to the reducer.
      const rows = readCommandsUpdate(message.update);
      if (rows !== null) this.setCommands(session, readCommands(withoutAppWorkflow(rows, this.appWorkflow)));
      if (typeof message.meta?.totalTokens !== 'number') return;
    }
    if (!message.xai && message.update.sessionUpdate === 'config_option_update') {
      // Live only, never stored: the host's alone.
      const effort = readEffortConfig(message.update.configOptions);
      session.efforts = effort?.ids ?? null;
      session.effort = effort?.current ?? null;
      return;
    }
    if (message.xai && message.update.sessionUpdate === 'subagent_progress') {
      // Live only: the agent never stores a tick, so the reducer must not see one.
      const tick = readSubagentProgress(message.update);
      if (this.mode === 'work' && tick !== null && session.tracker.progress(tick, this.now())) {
        this.activityChanged(session);
      }
      return;
    }
    if (message.xai && message.update.sessionUpdate === 'monitor_event') {
      // Live only: the agent never stores an event, so the reducer must not see one.
      const event = readMonitorEvent(message.update);
      if (this.mode !== 'work' || event === null) return;
      const description = shownCut(event.description.trim(), WAKE_DESCRIPTION_CHARS, this.cardNeedles());
      if (session.tracker.event(event.taskId, event.text, this.now(), description)) this.activityChanged(session);
      return;
    }
    if (message.xai && message.update.sessionUpdate === 'model_auto_switched') {
      // Live only, and the turn it ends stored nothing: its reason rides the turn's end, never an entry.
      const turn = session.turn;
      const line = readModelSwitch(message.update);
      if (turn === null || line === null) return;
      turn.model = {
        previous: line.previous === '' ? null : line.previous,
        next: line.next === '' ? null : line.next,
        reason: cutReason(line.reason),
      };
      if (turn.model.next === null && turn.recorded === null) turn.recorded = 'no-model';
      return;
    }
    const hookLine = message.xai ? readHookRun(message.update) : null;
    if (hookLine?.kind === 'started') {
      // Live only: the agent never stores it, so the reducer must not see it.
      const agentMs = message.meta?.agentTimestampMs;
      session.hookRun = {
        state: {
          event: hookLine.event,
          ...(hookLine.tool === null ? {} : { tool: hookLine.tool }),
          count: hookLine.count,
          startedAt: this.now(),
        },
        promptId: hookLine.promptId,
        agentMs: typeof agentMs === 'number' ? agentMs : null,
      };
      this.opts.listener?.onHookRun?.(session.id, { ...session.hookRun.state });
      return;
    }
    if (this.mode === 'work' && session.tracker.isTaskTick(message.update, message.meta)) {
      // Live only, and the call's stored result already ended its row: the reducer must not see it, nor may it start
      // a wake (the agent sends one whenever the task prints).
      this.checkPrinted(session, message.update);
      return;
    }
    if (this.mode === 'work') this.trackCall(session, message.update, session.id);
    const title = readTitle(message.update);
    if (title !== null) {
      this.opts.listener?.onTitle?.(session.id, title);
      return;
    }
    const line: UpdateLine = { xai: message.xai, update: message.update, meta: message.meta };
    const turnRunning = (session.turn !== null && session.turn.phase !== 'starting') || session.wake !== null;
    if (liveFilter(line, { turnRunning, modelId: session.modelId }) === 'drop') return;
    if (
      !line.xai &&
      session.wake === null &&
      WAKE_KINDS.has(String(line.update.sessionUpdate)) &&
      !this.promptOpen(session.id) &&
      session.turn?.between !== true
    ) {
      this.startWake(session);
    }
    const turn = session.turn;
    if (
      turn !== null &&
      !line.xai &&
      line.update.sessionUpdate === 'tool_call' &&
      typeof line.update.toolCallId === 'string'
    ) {
      turn.toolSeen = true;
    }
    if (line.xai && line.update.sessionUpdate === 'model_changed' && typeof line.update.model_id === 'string') {
      session.modelId = line.update.model_id;
      this.opts.listener?.onModel?.(session.id, session.modelId);
    }
    if (this.mode === 'work') {
      this.trackWork(session, line);
      this.trackRunCall(session, line.update);
    }
    if (this.mode === 'work' && line.xai && line.update.sessionUpdate === 'goal_updated') {
      session.goal = typeof line.update.status === 'string' ? line.update.status : null;
    }
    if (session.hookRun !== null && hookRunEnds(session.hookRun, line, hookLine)) this.endHookRun(session);
    this.opts.listener?.onUpdate?.(session.id, line);
    if (line.xai) {
      const compaction = readCompaction(line.update);
      if (compaction !== null) this.onCompaction(session, compaction);
      else if (line.update.sessionUpdate === 'turn_completed') {
        if (turn !== null) this.resumeCap(session, turn);
        this.endCompaction(session);
      }
      const recapped = line.update.sessionUpdate;
      if (session.recap !== null && (recapped === 'session_recap' || recapped === 'session_recap_unavailable')) {
        this.endRecap(session, session.recap, recapped === 'session_recap' ? 'shown' : 'none');
      }
      const workflow = readWorkflowUpdate(line.update);
      if (workflow !== null) {
        const agentMs = line.meta?.agentTimestampMs;
        this.onWorkflow(session, workflow, typeof agentMs === 'number' ? agentMs : null);
      }
      if (line.update.sessionUpdate === 'turn_completed' && this.runWakeEnds(session, line)) {
        this.emitRuns(session);
        this.idleStep();
      }
      const child = this.mode === 'work' ? readSpawnedChild(line.update) : null;
      if (child !== null) this.children.set(child, session);
    }
    if (turn !== null && line.xai && line.update.sessionUpdate === 'turn_completed') {
      turn.endSeen = true;
      turn.gap?.();
    }
    const wake = session.wake;
    if (wake !== null && line.xai && line.update.sessionUpdate === 'turn_completed' && !this.promptOpen(session.id)) {
      const outcome: TurnOutcome = line.update.stop_reason === 'cancelled' ? 'cancelled' : 'completed';
      this.endWake(session, { turnId: wake.turnId, outcome });
    }
    if (this.mode === 'work') this.checkPrinted(session, line.update);
    if (this.mode === 'chat') {
      const result = readWebToolResult(line.update);
      if (result !== null) {
        const before = session.ledger.size;
        applyWebResult(result, {
          ledger: session.ledger,
          onKeyCheck: (provider, check) =>
            this.opts.providers.markKeyCheck(provider, check, keyOf(session.web, provider)),
        });
        if (session.ledger.size !== before || session.ledgerDirty) this.saveLedger(session);
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
      this.opts.listener?.onNotice?.(session.id, { code: 'key-printed', tone: 'danger', provider });
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
    if (method === '_x.ai/ask_user_question') {
      return { result: this.mode === 'work' ? await this.workQuestion(proc, params) : QUESTION_CANCELLED };
    }
    if (method === '_x.ai/mcp/elicit') {
      return { result: this.mode === 'work' ? await this.workElicit(proc, params) : MCP_CANCEL };
    }
    return null;
  }

  /** A permission request the reader could not parse: fail closed on every running turn. */
  private cancelUnknownTurns(): void {
    for (const s of this.sessions.values()) if (s.turn !== null) this.cancelTurnLater(s, 'tool-refused');
  }

  private cancelTurnLater(session: HostSession, cause: TurnEndCause): void {
    if (session.turn !== null && session.turn.recorded === null) session.turn.recorded = cause;
    // Two microtasks: the first lets `answerRequest`'s `.then(...)` write the
    // permission response, and the second writes `session/cancel` before the
    // I/O poll loop can read the agent's prompt reply and clear `session.turn`.
    queueMicrotask(() => queueMicrotask(() => this.cancelTurn(session, cause)));
  }

  private chatPermission(req: PermissionRequest): PermissionAnswer {
    const session = this.sessions.get(req.sessionId);
    const decision = decideChatPermission(req, {
      sessionGuarded: session?.guarded === true && session.turn !== null,
      approvals: session?.turn?.approvals ?? 0,
    });
    if (decision.kind === 'approve') {
      if (session?.turn != null) session.turn.approvals += 1;
      if (session?.ledgerDirty === true) this.saveLedger(session);
      return decision.answer;
    }
    if (session?.turn != null) {
      if (session.turn.recorded === null) session.turn.recorded = decision.cause;
      if (decision.cancelTurn) this.cancelTurnLater(session, decision.cause);
    } else if (decision.cancelTurn && this.running !== null) {
      const running = this.running;
      queueMicrotask(() =>
        queueMicrotask(() => {
          try {
            running.process.client.notify('session/cancel', { sessionId: req.sessionId });
          } catch {
            // The process is gone.
          }
        }),
      );
    }
    return decision.answer;
  }

  /**
   * Work: the request is decided by the conversation's permission mode ({@link openWorkCard}) for its owner: a helper
   * of a live run asks for the run, else the typed turn owns it, else the wake turn. A request with no live owner, or
   * from a process that is gone, is answered `cancelled`.
   */
  private workPermission(proc: AgentProcess, req: PermissionRequest): Promise<unknown> {
    const session = this.sessions.get(req.sessionId);
    const owner = session === undefined ? null : ownerOf(session, req);
    if (session === undefined || owner === null || !this.asks(proc, session, owner)) {
      return Promise.resolve(PERMISSION_CANCELLED);
    }
    const requester = req.requester;
    const elsewhere = requester !== null && folderKey(requester.cwd) !== folderKey(session.cwd);
    // A helper in another folder names its paths from there.
    const place = { cwd: elsewhere ? requester.cwd : session.cwd, home: this.opts.homeDir ?? homedir() };
    const ask: Ask = { proc, session, owner, req, place, callSession: requester?.sessionId ?? session.id, elsewhere };
    const seen: Seen = { work: this.folderCalls(session.cwd).gen, temps: new Map() };
    for (const top of tempRoots(this.gatePlace(place)).map((r) => r.top)) {
      seen.temps.set(top, this.folderCalls(top).gen);
    }
    const key = callKey(ask.callSession, req.toolCallId);
    const call = { gone: false };
    this.deciding.set(key, call);
    const { path } = requestTarget(req);
    const launch = readPdfLaunch(req);
    const pdfFacts = this.opts.pdfFacts;
    const decide = (): Promise<unknown> =>
      launch !== null && pdfFacts !== undefined
        ? pdfFacts(session.id, launch.pdf).then(
            (facts) => this.openWorkCard({ ...ask, pdf: facts }, null, seen),
            () => this.openWorkCard(ask, null, seen),
          )
        : path === null
          ? this.openWorkCard(ask, null, seen)
          : editLanding(agentPath(path, place)).then(
              (landing) => this.openWorkCard(ask, landing, seen),
              () => this.openWorkCard(ask, null, seen),
            );
    const never = this.opts.never;
    const shell = shellCommand(req);
    const decided =
      never === undefined || shell === null
        ? decide()
        : never
            .ready()
            .then(() => this.ruleFirst(ask, never, shell.text))
            .then((ruled) => (ruled === null ? decide() : ruled));
    return decided.finally(() => {
      if (this.deciding.get(key) === call) this.deciding.delete(key);
    });
  }

  /**
   * Work, before the mode is read: a shell command (a command or a monitor) a never-allow rule of the Work folder
   * matches gets no card in any mode. It is denied with the rule's note, so the agent goes on; past
   * {@link NEVER_PER_TURN_MAX} hits of its owner the deny is plain and a typed turn ends `never-loop`. A command the app
   * cannot read (`command` null: not text), in a folder with rules, is marked so that Bypass asks. Null: the mode decides.
   */
  private ruleFirst(ask: Ask, never: NeverRules, command: string | null): unknown {
    const { proc, session, owner, req } = ask;
    if (!this.asks(proc, session, owner)) return PERMISSION_CANCELLED;
    const rules = never.rulesOf(session.cwd);
    const found = command !== null ? neverMatch(command, rules) : rules.length === 0 ? null : 'unread';
    if (found === null) return null;
    if (found === 'unread') {
      ask.neverUnread = true;
      return null;
    }
    const record =
      owner.kind === 'turn' ? owner.turn : owner.kind === 'wake' ? owner.wake : session.runs.get(owner.runId);
    const hits = record === undefined ? 1 : (this.neverHits.get(record) ?? 0) + 1;
    if (record !== undefined) this.neverHits.set(record, hits);
    const words = ruleText(found.words);
    const deny = req.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID && o.kind === 'reject_once');
    const noted = deny && hits <= NEVER_PER_TURN_MAX ? noteAnswer(REJECT_ONCE_OPTION_ID, neverNote(words)) : null;
    if (noted === null) this.recordDenied(session, owner, deny ? 'never-loop' : 'denied');
    const shown = showableWhole(req) ? { ...this.card(ask, null), options: [] } : null;
    this.opts.listener?.onRuleDenied?.(session.id, {
      toolCallId: req.toolCallId,
      access: accessOf(req),
      card: shown,
      ...childOf(session, ask.callSession),
      ruleId: found.id,
      words,
      noted: noted !== null,
    });
    if (noted !== null) return noted;
    return deny ? { outcome: { outcome: 'selected', optionId: REJECT_ONCE_OPTION_ID } } : PERMISSION_CANCELLED;
  }

  /**
   * A card opens only for a live owner of the process that asked: the typed turn while it runs (not stopped), a run
   * while it is `active` and not stopping, the wake turn while it runs. Anything else is answered `cancelled`.
   */
  private asks(proc: AgentProcess, session: HostSession, owner: Owner): boolean {
    if (!session.guarded || this.sessions.get(session.id) !== session || this.running?.process !== proc) return false;
    switch (owner.kind) {
      case 'turn':
        return session.turn === owner.turn && owner.turn.phase !== 'cancelling' && owner.turn.recorded === null;
      case 'run': {
        const run = session.runs.get(owner.runId);
        return run !== undefined && !run.stopping;
      }
      case 'wake':
        return session.wake === owner.wake && owner.wake.phase === 'running';
    }
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
  private async openWorkCard(ask: Ask, landing: string | null, seen: Seen): Promise<unknown> {
    const { proc, session, owner, req } = ask;
    for (let round = 0; round < MODE_ROUNDS; round++) {
      if (!this.asks(proc, session, owner)) return PERMISSION_CANCELLED;
      const mode = this.modeOf(session.id);
      if (mode !== 'ask' && !offersOnce(req)) {
        this.recordDenied(session, owner);
        return PERMISSION_CANCELLED;
      }
      if (mode === 'bypass') {
        if (ask.neverUnread === true) {
          if (!showableWhole(req)) return this.tooLong(ask);
          return this.workCard(ask, { ...this.card(ask, landing), neverUnread: true });
        }
        const answer = await this.allowAll(ask, landing);
        if (answer !== MOVED) return answer;
        continue;
      }
      if (!showableWhole(req)) return this.tooLong(ask);
      const interaction = this.card(ask, landing);
      if (mode === 'ask') return this.workCard(ask, interaction);
      const answer = await this.autoAnswer(ask, landing, seen, interaction);
      if (answer !== MOVED) return answer;
    }
    if (!this.asks(proc, session, owner)) return PERMISSION_CANCELLED;
    if (!showableWhole(req)) return this.tooLong(ask);
    return this.workCard(ask, this.card(ask, landing));
  }

  /** The card a request would open: who asks (a helper, its run, its folder when not the Work folder) included. */
  private card(ask: Ask, landing: string | null): PermissionInteraction {
    const { req, session } = ask;
    const requester = req.requester;
    const from =
      requester === null
        ? undefined
        : {
            label: requester.label,
            run: requester.runId === null ? null : (session.runs.get(requester.runId)?.name ?? null),
            cwd: ask.elsewhere ? requester.cwd : null,
            child: requester.sessionId,
          };
    const card: PermissionInteraction = {
      ...workInteraction(randomUUID(), req, this.cardNeedles(), ask.place, landing, from),
      openedAt: this.now(),
    };
    const pdf = pdfLaunchRow(req, ask.pdf ?? null);
    if (pdf !== null) card.pdfRead = { ...pdf, name: shownCut(pdf.name, CARD_NAME_MAX, this.cardNeedles()) };
    const command = requestTarget(req).command;
    const deny = card.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID && o.kind === 'reject_once');
    const offered =
      this.opts.never === undefined || command === null || !deny ? null : neverOffer(command, this.cardNeedles());
    if (offered === null) return card;
    card.never = { short: ruleText(offered.short), exact: ruleText(offered.exact) };
    this.neverOffers.set(card, offered);
    return card;
  }

  /** A command or path too long to show whole is denied without asking: the card must show exactly what would run. */
  private tooLong(ask: Ask): unknown {
    this.recordDenied(ask.session, ask.owner);
    const deny = ask.req.options.some((o) => o.optionId === REJECT_ONCE_OPTION_ID && o.kind === 'reject_once');
    return deny ? { outcome: { outcome: 'selected', optionId: REJECT_ONCE_OPTION_ID } } : PERMISSION_CANCELLED;
  }

  /** The card, opened in this tick; the user's answer settles it. */
  private workCard(ask: Ask, interaction: PermissionInteraction): Promise<unknown> {
    if (this.deciding.get(callKey(ask.callSession, ask.req.toolCallId))?.gone === true) {
      return Promise.resolve(PERMISSION_CANCELLED);
    }
    return this.openInteraction(
      ask.session,
      interaction,
      PERMISSION_CANCELLED,
      ask.owner,
      { session: ask.callSession, id: ask.req.toolCallId },
      staysBehind(ask.req),
    );
  }

  /**
   * Bypass: `allow-once` for any request, after its history line. No gate, no count, no folder limit, no window
   * check. With no `allow-once` offered the card opens. The mode is read again after the line: a move away ends the
   * round `moved`, and a line that was written then is withdrawn.
   */
  private async allowAll(ask: Ask, landing: string | null): Promise<unknown> {
    const { proc, session, owner, req } = ask;
    if (!req.options.some((o) => o.optionId === ALLOW_ONCE_OPTION_ID && o.kind === 'allow_once')) {
      if (!showableWhole(req)) return this.tooLong(ask);
      return this.workCard(ask, this.card(ask, landing));
    }
    const shown = showableWhole(req) ? { ...this.card(ask, landing), options: [] } : null;
    const mark: AllowedMark = {
      kind: 'allowed',
      at: this.now(),
      mode: 'bypass',
      toolCallId: req.toolCallId,
      access: accessOf(req),
      card: shown,
      ...childOf(session, ask.callSession),
    };
    const written = await this.writeLine(session, mark);
    if (!this.asks(proc, session, owner)) {
      if (written) this.withdraw(session, mark);
      return PERMISSION_CANCELLED;
    }
    if (!written || this.modeOf(session.id) !== 'bypass') {
      if (written) this.withdraw(session, mark);
      return MOVED;
    }
    this.callStarted(session, ask.callSession, req.toolCallId, {
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

  /**
   * A call of the session's folder (and of each temp root it reaches) starts running; one that may change it raises
   * `gen`. `callSession` is the session that runs it: a helper's own id for a helper's call.
   */
  private callStarted(session: HostSession, callSession: string, toolCallId: string, call: FolderCall): void {
    const id = callKey(callSession, toolCallId);
    this.folderCalls(session.cwd).busy.set(id, call);
    for (const top of call.reach ?? []) this.folderCalls(top).busy.set(id, call);
    if (call.changes) this.raise(session, call.wide, call.reach);
  }

  /**
   * A call leaves the folder (it ended, or its allow was given back). `gen` goes up when it could change the folder,
   * and for an unknown call of a session the agent may run calls for without asking.
   */
  private callEnded(session: HostSession, callSession: string, toolCallId: string): void {
    const folder = this.folderCalls(session.cwd);
    const key = callKey(callSession, toolCallId);
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
  private folderBusy(
    session: HostSession,
    callSession: string,
    toolCallId: string,
    changes: boolean,
    reach: readonly string[],
  ): boolean {
    const own = callKey(callSession, toolCallId);
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
   * so a new call that is not a read joins the folder as one that may change it. `callSession` is the session whose
   * line this is: the session itself, or a helper of it.
   */
  private trackCall(session: HostSession, update: Record<string, unknown>, callSession: string): void {
    const ended = endedCall(update);
    if (ended !== null) {
      this.callEnded(session, callSession, ended);
      return;
    }
    if (!session.granted || update.sessionUpdate !== 'tool_call' || typeof update.toolCallId !== 'string') return;
    const kind = update.kind;
    if (kind === 'read' || kind === 'search' || kind === 'think' || kind === 'fetch') return;
    if (!this.folderCalls(session.cwd).busy.has(callKey(callSession, update.toolCallId))) {
      this.callStarted(session, callSession, update.toolCallId, { changes: true, wide: true });
    }
  }

  /**
   * Auto: the app's own `allow-once` when every rule of the gate passes and the history line is written; else the
   * card with the reason, opened in the tick of the last mode check. `MOVED` when the mode is no longer Auto at a
   * check. A call is counted in the same tick as its decision, so requests that arrive together cannot pass the cap;
   * a call that does not get its allow gives the count back. Any error asks.
   */
  private async autoAnswer(
    ask: Ask,
    landing: string | null,
    seen: Seen,
    interaction: PermissionInteraction,
  ): Promise<unknown> {
    const { proc, session, owner, req, place, callSession } = ask;
    const count = this.countOf(session, owner);
    if (count === null) return PERMISSION_CANCELLED;
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
      if (!this.asks(proc, session, owner)) return PERMISSION_CANCELLED;
      if (moved(facts.reach)) {
        // A call that may change the folder started or ended while the facts were read: read them once more.
        const again = gens(facts.reach);
        const path = requestTarget(req).path;
        if (path !== null && (await editLanding(agentPath(path, place)).catch(() => null)) !== landing) stale = true;
        const reach = facts.reach;
        facts = await autoFacts(req, gate, landing, { cdpath: this.cdpath() });
        if (!this.asks(proc, session, owner)) return PERMISSION_CANCELLED;
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
        elsewhere: ask.elsewhere,
        facts,
        allowed: count.approvals,
        pdf: ask.pdf ?? null,
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
      return this.workCard(ask, interaction);
    };
    if (this.modeOf(session.id) !== 'auto') return MOVED;
    if (!decision.allow) return asking(decision.why ?? 'unread', decision.detail);
    const launch = decision.pdf === true;
    if (!launch && interaction.access !== 'bash' && interaction.access !== 'edit') return asking('not-covered');
    const reach = launch ? [] : (facts?.reach ?? tempRoots(gate).map((r) => r.top));
    if (!launch) {
      if (stale || this.folderBusy(session, callSession, req.toolCallId, changes, reach)) {
        return asking('unknown-target');
      }
      this.callStarted(session, callSession, req.toolCallId, {
        changes,
        wide: false,
        ...(reach.length > 0 ? { reach } : {}),
      });
    }
    const mine = gens(reach);
    const epoch = count.approvalEpoch;
    count.approvals += 1;
    const giveBack = (): void => {
      if (count.approvalEpoch === epoch && count.approvals > 0) count.approvals -= 1;
      if (!launch) this.callEnded(session, callSession, req.toolCallId);
    };
    const mark: AllowedMark = {
      kind: 'allowed',
      at: this.now(),
      mode: 'auto',
      toolCallId: req.toolCallId,
      access: interaction.access,
      card: { ...interaction, options: [] },
      ...(facts?.touched === true ? { temp: true as const } : {}),
      ...childOf(session, callSession),
    };
    const written = await this.writeLine(session, mark);
    if (!this.asks(proc, session, owner)) {
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
    if (!launch && gens(reach).some((gen, k) => gen !== mine[k])) {
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
    if (req === null || session === undefined || turn === null) return Promise.resolve(PLAN_CANCELLED);
    const owner: Owner = { kind: 'turn', turn };
    if (!this.asks(proc, session, owner)) return Promise.resolve(PLAN_CANCELLED);
    const interaction = { ...planInteraction(randomUUID(), req, this.cardNeedles()), openedAt: this.now() };
    return this.openInteraction(session, interaction, PLAN_CANCELLED, owner, {
      session: session.id,
      id: req.toolCallId,
    });
  }

  /**
   * Work: `_x.ai/ask_user_question` becomes a question card for the typed turn, else the wake turn, in every permission
   * mode. "No answer" (`cancelled`) and no card: a request that cannot be read, one over the card's limits (with a
   * conversation notice), one past {@link ASKS_PER_TURN_MAX} calls of its turn, or one with no live owner. The agent
   * asks one question call at a time, so an older open card of the session is one it stopped waiting on.
   */
  private workQuestion(proc: AgentProcess, params: unknown): Promise<unknown> {
    const read = readQuestionRequest(params);
    const session =
      read === null ? undefined : this.sessions.get(read.kind === 'ok' ? read.req.sessionId : read.sessionId);
    if (read === null || session === undefined) return Promise.resolve(QUESTION_CANCELLED);
    const owner: Extract<Owner, { kind: 'turn' | 'wake' }> | null =
      session.turn !== null
        ? { kind: 'turn', turn: session.turn }
        : session.wake !== null
          ? { kind: 'wake', wake: session.wake }
          : null;
    if (owner === null || !this.asks(proc, session, owner)) return Promise.resolve(QUESTION_CANCELLED);
    if (read.kind === 'too-long') {
      this.opts.listener?.onNotice?.(session.id, { code: 'question-too-long', tone: 'warning' });
      return Promise.resolve(QUESTION_CANCELLED);
    }
    const record = owner.kind === 'turn' ? owner.turn : owner.wake;
    const calls = (this.questionCalls.get(record) ?? 0) + 1;
    this.questionCalls.set(record, calls);
    if (calls > ASKS_PER_TURN_MAX) return Promise.resolve(QUESTION_CANCELLED);
    for (const [id, open] of session.interactions) {
      if (open.interaction.kind === 'question') this.settleInteraction(session, id, open.withdrawn, 'expired');
    }
    const interaction = questionInteraction(randomUUID(), read.req, this.cardNeedles(), this.now());
    this.questionRequests.set(interaction, read.req);
    return this.openInteraction(session, interaction, QUESTION_CANCELLED, owner, {
      session: session.id,
      id: read.req.toolCallId,
    });
  }

  /**
   * Work: `_x.ai/mcp/elicit` becomes an MCP card in every permission mode, owned by the typed turn, else the wake
   * turn, else a live run (the request names no helper); with none it is answered `cancel`. A form the app cannot draw,
   * or a link that is not a safe web address, still opens a card, with Decline only.
   */
  private workElicit(proc: AgentProcess, params: unknown): Promise<unknown> {
    const req = readElicitRequest(params);
    const session = req === null ? undefined : this.sessions.get(req.sessionId);
    if (req === null || session === undefined) return Promise.resolve(MCP_CANCEL);
    const run = [...session.runs].find(([, r]) => !r.stopping)?.[0];
    const owner: Owner | null =
      session.turn !== null
        ? { kind: 'turn', turn: session.turn }
        : session.wake !== null
          ? { kind: 'wake', wake: session.wake }
          : run !== undefined
            ? { kind: 'run', runId: run }
            : null;
    if (owner === null || !this.asks(proc, session, owner)) return Promise.resolve(MCP_CANCEL);
    let interaction: McpInteraction;
    if (req.mode === 'url') {
      interaction = linkInteraction(randomUUID(), req, this.cardNeedles(), this.now());
      this.mcpForms.set(interaction, { server: req.server, specs: null, elicitationId: req.elicitationId });
    } else {
      const specs = req.schema === undefined ? [] : parseForm(req.schema);
      interaction = mcpInteraction(randomUUID(), req, specs, this.cardNeedles(), this.now());
      this.mcpForms.set(interaction, { server: req.server, specs, elicitationId: null });
    }
    if (owner.kind === 'run') interaction.run = session.runs.get(owner.runId)?.name ?? owner.runId;
    return this.openInteraction(session, interaction, MCP_CANCEL, owner, { session: session.id, id: req.toolCallId });
  }

  /** Every needle is known before a card text is cut: a key cut in two would no longer match. */
  private cardNeedles(): string[] {
    return needleList((this.opts.needles ?? (() => this.opts.providers.secretNeedles()))());
  }

  /** `call`: the session whose call asks (a helper's, or the session's own) and the agent's tool call id. */
  private openInteraction(
    session: HostSession,
    interaction: Interaction,
    withdrawn: unknown,
    owner: Owner,
    call: { session: string; id: string },
    stays = false,
  ): Promise<unknown> {
    return new Promise<unknown>((resolve) => {
      session.interactions.set(interaction.id, {
        interaction,
        settle: resolve,
        withdrawn,
        owner,
        callSession: call.session,
        callId: call.id,
        stays,
      });
      this.opts.listener?.onInteraction?.(session.id, interaction);
      this.emitState();
    });
  }

  /**
   * The agent stopped waiting on a request it sent from `callSession` (a helper's session for a helper's ask): every
   * open card of that call closes as `expired`, and the request gets the answer that withdraws it. A request still
   * being decided opens no card. A call answered here already changes nothing.
   */
  private expireAsk(callSession: string, toolCallId: string): void {
    const call = this.deciding.get(callKey(callSession, toolCallId));
    if (call !== undefined) call.gone = true;
    for (const session of this.sessions.values()) {
      const gone = [...session.interactions].filter(
        ([, open]) => open.callSession === callSession && open.callId === toolCallId,
      );
      for (const [id, open] of gone) this.settleInteraction(session, id, open.withdrawn, 'expired');
    }
  }

  // ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────

  private call(running: Running, method: string, params: unknown, timeoutMs: number): Promise<unknown> {
    return this.replyOf(running, running.process.client.request(method, params, timeoutMs));
  }

  /** A soft load's {@link call}: a miss of `ms` is `load-slow` and leaves the agent alive (the late reply is dropped). */
  private async softLoadCall(running: Running, method: string, params: unknown, ms: number): Promise<unknown> {
    try {
      return await running.process.client.requestSoft(method, params, ms);
    } catch (error) {
      if (!(error instanceof AcpSoftTimeoutError)) return this.replyOf(running, Promise.reject(error));
      error.late.catch(() => {});
      throw new AgentHostError('load-slow');
    }
  }

  /** A reply, or the error the host throws for it: an error reply as it came, else a protocol error or a crash. */
  private async replyOf(running: Running, reply: Promise<unknown>): Promise<unknown> {
    try {
      return await reply;
    } catch (error) {
      if (error instanceof AcpRpcError) throw error;
      const broke = running.process.brokeProtocol || running.process.running;
      throw new AgentHostError(broke ? 'agent-protocol-error' : 'agent-crashed');
    }
  }

  private emitTurn(session: HostSession): void {
    this.opts.listener?.onTurn?.(session.id, shownTurn(session));
  }

  private emitState(): void {
    this.opts.listener?.onState?.(this.modeState());
  }
}

function turnStateOf(t: TurnRecord): TurnState {
  return {
    turnId: t.turnId,
    phase: t.phase,
    startedAt: t.startedAt,
    wait: t.wait,
    ...(t.continues > 0 ? { continues: t.continues } : {}),
  };
}

/** The turn the page shows: the typed one, else the wake. */
function shownTurn(s: HostSession): TurnState | null {
  if (s.turn !== null) return turnStateOf(s.turn);
  if (s.wake === null) return null;
  return { turnId: s.wake.turnId, phase: s.wake.phase, startedAt: s.wake.startedAt, wait: null, origin: 'wake' };
}

/** A wake turn, a workflow run or a helper is live: work that uses the engine. */
function engineWork(s: HostSession): boolean {
  return s.wake !== null || s.runs.size > 0 || s.tracker.helperCount() > 0;
}

/** A command or monitor is live: work that needs only the process. */
function processWork(s: HostSession): boolean {
  return s.tracker.taskCount() > 0;
}

/** The agent's answer to a message, as the page reads it; null (an error envelope or any other shape) is `error`. */
function messageResult(kind: ReturnType<typeof readSubagentMessage>): SubagentMessage {
  switch (kind) {
    case 'accepted':
      return 'sent';
    case 'admission_uncertain':
      return 'unsure';
    case 'not_active':
      return 'not-ready';
    case 'saturated':
      return 'busy';
    case 'not_accepted_before_deadline':
      return 'late';
    case 'limit':
      return 'too-long';
    case 'rejected':
    case 'channel_closed':
      return 'gone';
    case 'unsupported_content':
    case null:
      return 'error';
  }
}

/**
 * A prompt of the session is open in `running`: the typed turn's, past its start, not between two prompts and with no
 * reply yet, or the wake's. A history op has none. A goal round counts from its start: the agent starts it by itself
 * at the round before's end, before its first line here makes it a wake.
 */
function steerable(s: HostSession, running: Running): boolean {
  if (s.op !== null) return false;
  const turn = s.turn;
  if (turn === null) return s.wake !== null ? s.wake.phase === 'running' : s.hot && s.goal === 'active';
  const prompt = turn.prompt;
  return (
    turn.phase === 'running' &&
    !turn.between &&
    prompt !== null &&
    prompt.client === running.process.client &&
    prompt.client.isPending(prompt.id)
  );
}

/** Engine work or a history op: what holds the conversation besides its typed turn. */
function hasBackground(s: HostSession): boolean {
  return engineWork(s) || s.op !== null;
}

function opOf(s: HostSession): HistoryOp | 'recap' | null {
  return s.op?.kind ?? (s.recap === null ? null : 'recap');
}

/** The prefix of the turn the agent runs for what reached it while idle (`notification_drain.rs`). */
const NOTIFICATIONS_WAKE = 'notifications-';
/** The prompt id of a run's wake is this, the run id, `-` and the revision (`run_loop.rs`). */
const WORKFLOW_WAKE_PREFIX = 'workflow-completed-';

/** The prompt id the agent gives the wake a due one starts (grok `notification_bridge.rs`, `spawn.rs`, `run_loop.rs`). */
function wakePromptId(due: WakeDue): string {
  switch (due.kind) {
    case 'task':
      return `task-completed-${due.taskId}`;
    case 'subagent':
      return `subagent-completed-${due.subagentId}`;
  }
}

/** The statuses the agent reports in a wake turn (`is_completion_reportable`). */
const REPORTED: ReadonlySet<string> = new Set(['interrupted', 'complete', 'failed', 'cancelled', 'budget_limited']);

/** The statuses of a run that waits on the person: only a resume goes on with it. */
const WAITING: ReadonlySet<string> = new Set([
  'user_paused',
  'back_off_paused',
  'no_progress_paused',
  'infra_paused',
  'blocked',
  'budget_limited',
]);

/** The statuses of a run that waits on the person that the agent's Stop takes: it refuses a run out of agents. */
const STOPPABLE_WAITS: ReadonlySet<string> = new Set([...WAITING].filter((status) => status !== 'budget_limited'));

/** How many runs that left `active` a session keeps for the page. */
const ENDED_RUNS_KEPT = 16;

/** The agent's tool that launches, checks, pauses, stops and resumes workflow runs. */
const WORKFLOW_TOOL = 'workflow';

/** The status a pause or stop leaves a run in. */
function verbMoves(verb: 'pause' | 'resume' | 'stop', status: string): boolean {
  return verb === 'stop' ? status === 'cancelled' : verb === 'pause' && status === 'user_paused';
}

function endStopping(run: RunRecord): void {
  run.stopping = false;
  run.stopBy = null;
  clearStopTimer(run);
}

function clearStopTimer(run: RunRecord): void {
  if (run.stopTimer !== null) clearTimeout(run.stopTimer);
  run.stopTimer = null;
}

function clearRunStateTimers(s: HostSession): void {
  for (const timer of s.runStateTimers.values()) clearTimeout(timer);
  s.runStateTimers.clear();
}

function clearStopTimers(s: HostSession): void {
  for (const run of s.runs.values()) clearStopTimer(run);
  for (const run of s.endedRuns.values()) clearStopTimer(run);
}

/** The runs whose agents the idle guard keeps: the live ones, and the ended ones that drain a stop. */
function runsHeld(s: HostSession): Set<string> {
  const held = new Set(s.runs.keys());
  for (const [runId, run] of s.endedRuns) if (run.stopping) held.add(runId);
  return held;
}

/** An ended run of the session waits on the person. */
function runsWait(s: HostSession): boolean {
  for (const run of s.endedRuns.values()) if (WAITING.has(run.status)) return true;
  return false;
}

/**
 * The hook batch the session waits on is over: its own `hook_execution`, the turn's end line, or the model's text or
 * thought stamped after the batch started. The agent sends the batch's start at once but buffers text, so a chunk made
 * before it can come after it (`P/acp/tracker.rs` `chunk_predates_hook_batch`); any other line ends nothing.
 */
function hookRunEnds(
  run: NonNullable<HostSession['hookRun']>,
  line: UpdateLine,
  hookLine: ReturnType<typeof readHookRun>,
): boolean {
  if (hookLine?.kind === 'ended')
    return (
      hookLine.event === run.state.event &&
      hookLine.tool === (run.state.tool ?? null) &&
      hookLine.promptId === run.promptId
    );
  const kind = line.update.sessionUpdate;
  if (line.xai) return kind === 'turn_completed';
  if (kind !== 'agent_message_chunk' && kind !== 'agent_thought_chunk') return false;
  const at = line.meta?.agentTimestampMs;
  return run.agentMs === null || typeof at !== 'number' || at > run.agentMs;
}

function runViews(s: HostSession): RunView[] {
  const view = (runId: string, run: RunRecord, live: boolean): RunView => ({
    runId,
    name: run.name,
    status: run.status,
    live,
    stopping: run.stopping,
    by: run.by,
    waiting: !live && WAITING.has(run.status),
    wakeDue: s.runWake?.runId === runId,
  });
  return [
    ...[...s.runs].map(([runId, run]) => view(runId, run, true)),
    ...[...s.endedRuns].map(([runId, run]) => view(runId, run, false)),
  ];
}

/**
 * The owner of a Work request: a helper of a live, not stopping run asks for that run (a helper of a gone run has
 * none); else the typed turn; else the wake turn; else none.
 */
function ownerOf(s: HostSession, req: PermissionRequest): Owner | null {
  const runId = req.requester?.runId ?? null;
  if (runId !== null) {
    const run = s.runs.get(runId);
    return run === undefined || run.stopping ? null : { kind: 'run', runId };
  }
  if (s.turn !== null) return { kind: 'turn', turn: s.turn };
  if (s.wake !== null) return { kind: 'wake', wake: s.wake };
  return null;
}

/** The `child` of an allowed mark: the asking helper's session, none for the session's own call. */
function childOf(s: HostSession, callSession: string): { child?: string } {
  return callSession === s.id ? {} : { child: callSession };
}

/** One Work request being decided, with its owner and where its call runs. */
interface Ask {
  proc: AgentProcess;
  session: HostSession;
  owner: Owner;
  req: PermissionRequest;
  /** The folder the request's paths are named from: the helper's own when it is not the Work folder. */
  place: WorkPlace;
  callSession: string;
  /** A helper whose folder is not the Work folder. */
  elsewhere: boolean;
  /** The app cannot read the command and the Work folder has never-allow rules: Bypass asks. */
  neverUnread?: true;
  /** A `read-pdf` launch: what the host read for its `pdf`. */
  pdf?: PdfFacts | null;
}

/** Update kinds that mean the agent is working. */
const WAKE_KINDS: ReadonlySet<string> = new Set([
  'user_message_chunk',
  'agent_message_chunk',
  'agent_thought_chunk',
  'tool_call',
  'tool_call_update',
  'plan',
]);

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
/** Characters of a PDF's name on a card. */
const CARD_NAME_MAX = 200;
/** Characters of a monitor's description a wake row may name it by. */
const WAKE_DESCRIPTION_CHARS = 200;

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
  if (isWinPath(cwd)) {
    return cwd
      .replaceAll('/', '\\')
      .split('\\')
      .map((p) => foldName(p))
      .join('\\');
  }
  return cwd
    .split('/')
    .map((p) => foldName(p))
    .join('/');
}

/** Two folder keys that are the same folder, or one holds the other. */
function nestedFolders(a: string, b: string): boolean {
  return below(a, b) || below(b, a);
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

/** False when the write failed; the older file (or none) stays: fewer addresses, never more. */
function saveLedgerFile(path: string | null, members: readonly string[]): boolean {
  if (path === null) return true;
  try {
    writeLedgerFile(path, members);
    return true;
  } catch {
    return false;
  }
}

function removeLedgerFile(path: string | null): void {
  if (path === null) return;
  try {
    unlinkSync(path);
  } catch {
    // Gone already, or the next start's sweep removes it.
  }
}
