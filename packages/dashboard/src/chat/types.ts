/** Wire shapes of Chat and Work. The SPA keeps a copy (`ui/src/lib/chat-types.ts`), pinned equal by type. */

import type { TranscriptEntry, TurnMetric } from '../agent-sessions/types.js';
import type { KeyCheck, ProviderId, SearchProviderId } from './providers/types.js';

export type ChatMode = 'chat' | 'work';

export type ChatFailureCode =
  | 'inference-stopped'
  | 'inference-failed'
  | 'no-models'
  | 'agent-missing'
  | 'agent-too-old'
  | 'agent-incompatible'
  | 'agent-protocol-error'
  | 'agent-crashed'
  | 'agent-failed'
  | 'agent-slow'
  | 'profile-mismatch'
  | 'model-unavailable'
  | 'model-load-failed'
  | 'cwd-invalid'
  | 'cwd-mismatch'
  | 'conversation-busy'
  | 'engine-busy'
  | 'slash-text'
  | 'keychain-unavailable'
  | 'skill-bundle-corrupt';
/** `message` is fixed text per code, never data from the agent. */
export interface ChatFailure {
  code: ChatFailureCode;
  message: string;
}

export type TurnOutcome =
  | 'completed'
  | 'cancelled'
  | 'output-limit'
  | 'tool-loop'
  | 'turn-timeout'
  | 'inference-lost'
  | 'agent-crashed'
  | 'model-error'
  | 'refused'
  /** Chat: the model asked for something Chat never allows. */
  | 'tool-refused'
  /** Chat: `web_fetch` of an address that is not in the ledger. */
  | 'address-refused'
  /** Chat: a web call's input text could become an object that the app cannot read the same way. */
  | 'arguments-unreadable'
  /** Work: the user pressed Deny; the agent ends the turn on a rejection. */
  | 'denied'
  /** The agent ended the turn as cancelled and the app recorded no cause, or the Control Panel closed. */
  | 'interrupted';

export interface ConversationInfo {
  /** The agent's sessionId. */
  id: string;
  mode: ChatMode;
  /** The agent's title, else the title of the first typed prompt; '' only for a session with no prompt. */
  title: string;
  /** Work only. */
  cwd: string | null;
  modelId: string;
  createdAt: number;
  updatedAt: number;
  /** Loaded in a running agent process. */
  hot: boolean;
  /** Work: why the stored folder cannot be used now (it went away, it is no longer a folder, …). */
  cwdProblem?: CwdProblem;
  /** Work: the permission mode. Memory only; Ask after an app restart. The worker's list rows say Ask. */
  permissionMode: PermissionMode;
  /** Work: a hook of the user can ask before a tool runs (a file check of the agent's hook sources). Text only. */
  userHooks?: true;
}
export type ConversationRow = ConversationInfo & { turns: number };

/** Why a Work folder is refused. Each has a fixed sentence (`CWD_PROBLEM_MESSAGES`). */
export type CwdProblem = 'not-absolute' | 'missing' | 'not-directory' | 'unreadable' | 'root' | 'home' | 'app-data';

/** A warning line on a Work permission card. It blocks nothing. */
export type CommandFlag =
  | 'installs'
  | 'credentials'
  | 'prints-secrets'
  | 'deletes'
  | 'elevated'
  | 'outside-folder'
  | 'provider-jobs'
  | 'unchecked';

export interface TurnState {
  turnId: string;
  phase: 'starting' | 'running' | 'cancelling';
  startedAt: number;
  /** `model`: the app waits for the engine to put this conversation's model in memory. */
  wait: 'model' | null;
}

export interface PermissionInteraction {
  id: string;
  kind: 'permission';
  toolCallId: string;
  /** From `_meta["x.ai/tool"].name`, else `toolCall.kind`. */
  toolName: string;
  /** Decides the label of the "always" button. */
  access: 'bash' | 'edit' | 'mcp' | 'other';
  /** At most 200 characters. */
  title: string;
  /** JSON text of `rawInput`, whole (a longer input is denied without a card), redacted. */
  input: string;
  /** `bash`: the whole command the agent will run, never cut, redacted, hidden characters shown as `\u{…}`. */
  command: string | null;
  /** `edit`: the whole path of the file, as `command`. */
  path: string | null;
  /** `edit`: where the write really lands when a link on disk leads elsewhere; null when it lands at `path`. */
  resolvedPath: string | null;
  /** `edit`: the app cannot tell which file the edit changes (a link it cannot follow, or a look-alike name). */
  landingUnknown?: true;
  /** `name` is the agent's own option text, at most 120 characters. `enable-always-approve` and `reject_always` are never listed. */
  options: Array<{ optionId: string; kind: 'allow_once' | 'allow_always' | 'reject_once'; name: string }>;
  flags: CommandFlag[];
  /** Set when the card opened in Auto: why this call still asks. The page draws it only while the mode is Auto. */
  autoAsk?: AutoAsk;
}
export interface PlanInteraction {
  id: string;
  kind: 'plan';
  /** The tool row the card belongs to. */
  toolCallId: string;
  /** Markdown from the model: redacted, hidden characters shown as `\u{…}`. */
  plan: string;
}
export type Interaction = PermissionInteraction | PlanInteraction;

/** The permission mode of a Work conversation: every request is a card, the Auto gate decides, or every request is allowed. */
export type PermissionMode = 'ask' | 'auto' | 'bypass';

/** Why a call still asks in Auto. */
export type AutoAsk =
  | 'risky'
  | 'key'
  | 'unread'
  | 'not-known'
  | 'protected'
  | 'background'
  | 'unknown-target'
  | 'unknown-landing'
  | 'not-covered'
  | 'hook'
  | 'helper-start'
  | 'helper'
  | 'many'
  | 'no-window';

/** The permission record of a conversation, oldest first. */
export type PermissionMark =
  | {
      kind: 'mode';
      at: number;
      mode: PermissionMode;
      from: PermissionMode;
      by: 'user' | 'restart' | 'record-failed';
      /** To Auto: "Allow all edits" was picked before, so the agent does not ask for edits. */
      editsUnasked?: true;
      /** Up: a hook of the user can ask. Auto: its requests for the calls Auto allows are not seen. Bypass: allowed. */
      userHooks?: true;
    }
  | {
      kind: 'allowed';
      at: number;
      mode: 'auto' | 'bypass';
      toolCallId: string;
      /** Auto: bash or edit only. */
      access: 'bash' | 'edit' | 'mcp' | 'other';
      /** The card as it would have shown, `options: []`. Null when read from disk, dropped from memory, or too long. */
      card: PermissionInteraction | null;
    }
  /** A settings file of the agent changed while the mode was not Ask. */
  | { kind: 'settings'; at: number };

/** Fixed text per code, in memory only. */
export interface ConversationNotice {
  code: 'key-printed';
  provider: ProviderId;
}

export type TranscriptOp =
  | { op: 'push'; entry: TranscriptEntry }
  /** The entry at `index` is `assistant` or `thinking`. */
  | { op: 'text'; index: number; append: string }
  /** Whole-entry replace: tool, plan, subagent. */
  | { op: 'set'; index: number; entry: TranscriptEntry }
  | { op: 'metric'; metric: TurnMetric };

/** How a turn ended: the `turn` event with `turn: null` carries it, and `sync` repeats the last one (memory only). */
export interface TurnEnd {
  turnId: string;
  outcome?: TurnOutcome;
  failure?: ChatFailure;
  /** The turn ended before `session/prompt` went to the agent. */
  unsent?: true;
}

export type ConversationEvent =
  | {
      type: 'sync';
      seq: number;
      conversation: ConversationInfo;
      entries: TranscriptEntry[];
      turns: TurnMetric[];
      turn: TurnState | null;
      interactions: Interaction[];
      notices: ConversationNotice[];
      transcriptError: string | null;
      lastEnd?: TurnEnd;
      /** Every permission mark of the conversation; card text for the newest 200 only. */
      permissionMarks: PermissionMark[];
    }
  | { type: 'ops'; seq: number; ops: TranscriptOp[] }
  | { type: 'notice'; seq: number; notice: ConversationNotice }
  | {
      type: 'turn';
      seq: number;
      turn: TurnState | null;
      /** With `turn: null`: the turn that ended. */
      turnId?: string;
      outcome?: TurnOutcome;
      failure?: ChatFailure;
      unsent?: true;
    }
  | { type: 'interaction'; seq: number; interaction: Interaction }
  | { type: 'interaction-closed'; seq: number; id: string; reason: 'answered' | 'cancelled' | 'expired' }
  | { type: 'conversation'; seq: number; conversation: ConversationInfo }
  | { type: 'permission-mark'; seq: number; mark: PermissionMark }
  | { type: 'gone'; seq: number; reason: 'deleted' | 'unknown' };

export interface AgentModeState {
  state: 'stopped' | 'starting' | 'ready' | 'crashed' | 'failed' | 'blocked';
  stale: boolean;
  failure: ChatFailure | null;
  agentVersion: string | null;
  turnsInFlight: number;
}
export interface ChatHostState {
  inference: 'ready' | 'starting' | 'loading' | 'stopped';
  /** Set while the engine loads or swaps a model. `startedAt` is the engine's own clock (epoch ms). */
  loading: { modelId: string; startedAt: number } | null;
  /** The chat models the engine holds in memory now; `sizeBytes` is the size on disk, null when unknown. */
  resident: Array<{ modelId: string; sizeBytes: number | null }>;
  /**
   * Per resident chat model, from the engine's live file: answers that run, the most that run at once, requests that
   * wait, and when the newest running answer started (epoch ms; null with none listed).
   */
  answers: Array<{
    modelId: string;
    inFlight: number;
    maxConcurrent: number;
    queued: number;
    lastStart: number | null;
  }>;
  models: Array<{ id: string; contextTokens: number | null }>;
  /** The resident chat model when one is in `models`, else the first of `models`. */
  defaultModelId: string | null;
  agent: {
    binary: 'ok' | 'missing' | 'too-old';
    source: string;
    tag: string | null;
    minTag: string;
    versionVerified: boolean;
    install: 'idle' | 'running' | 'failed';
  };
  modes: { chat: AgentModeState; work: AgentModeState };
  providers: {
    search: SearchProviderId | null;
    searchCheck: KeyCheck | null;
    firecrawl: boolean;
    fetch: ProviderId | null;
    applyPending: boolean;
  };
  /** `waiting`: an interaction is open. */
  active: Array<{ conversationId: string; mode: ChatMode; busy: boolean; waiting: boolean }>;
  /** The Work conversations not in Ask, with their mode (the list has no other source). */
  permissionRaised: Array<{ id: string; mode: 'auto' | 'bypass' }>;
}
export type ChatHostEvent = { type: 'state'; state: ChatHostState } | { type: 'conversations-changed' };
