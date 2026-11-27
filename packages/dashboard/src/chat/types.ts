/** Wire shapes of Chat and Work. The SPA keeps a copy (`ui/src/lib/chat-types.ts`), pinned equal by type. */

import type { TranscriptEntry, TurnMetric } from '../agent-sessions/types.js';
import type { UserAttachment } from './attachment-rule.js';
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
  /** A history op, a recap or a schedule's Delete: the agent did not open the cold conversation in time. */
  | 'load-slow'
  | 'profile-mismatch'
  | 'model-unavailable'
  | 'model-load-failed'
  | 'cwd-invalid'
  | 'cwd-mismatch'
  | 'conversation-busy'
  | 'engine-busy'
  /** The engine restarted under live commands: only the next message may restart the agent. */
  | 'engine-lost'
  | 'slash-text'
  | 'attachment-refused'
  | 'pictures-full'
  | 'keychain-unavailable'
  | 'skill-bundle-corrupt';
/** `message` is fixed text per code, never data from the agent. */
export interface ChatFailure {
  code: ChatFailureCode;
  message: string;
}

/** The message of the Continue button and of the automatic continue after a length stop in Work. */
export const CONTINUE_PROMPT = 'continue';

/** A thinking level as the engine names it, least thinking first: `none` is Off. */
export const THINKING_LEVELS = ['none', 'low', 'medium', 'high', 'max'] as const;
export type ThinkingLevel = (typeof THINKING_LEVELS)[number];

/** A model's thinking levels (least thinking first) and the one it runs at when the user picked none. */
export interface ModelThinking {
  levels: ThinkingLevel[];
  default: ThinkingLevel;
}

export type TurnOutcome =
  | 'completed'
  | 'cancelled'
  | 'output-limit'
  | 'tool-loop'
  /** Chat only: a Work turn has no time limit. */
  | 'turn-timeout'
  | 'inference-lost'
  | 'agent-crashed'
  | 'model-error'
  | 'refused'
  /** Chat: the model asked for something Chat never allows. */
  | 'tool-refused'
  /** Chat: a web call's input text could become an object that the app cannot read the same way. */
  | 'arguments-unreadable'
  /** Work: the user pressed Deny; the agent ends the turn on a rejection. */
  | 'denied'
  /** The agent ended the turn as cancelled and the app recorded no cause, or the Control Panel closed. */
  | 'interrupted'
  /** The agent had no model it could use: it ran nothing and stored nothing (`model_auto_switched` with no new id). */
  | 'no-model'
  /** A hook blocked the prompt: the agent ended the turn as cancelled with the category `HookDenied`. */
  | 'hook-stopped'
  /** Work: the model kept trying commands a never-allow rule denies; the app denied the next one plainly. */
  | 'never-loop';

export interface ConversationInfo {
  /** The agent's sessionId. */
  id: string;
  mode: ChatMode;
  /** The agent's title, else the title of the first typed prompt; '' only for a session with no prompt. */
  title: string;
  /** Work only. */
  cwd: string | null;
  /** Work: started without a folder; `cwd` is its own folder under the scratch root. */
  noFolder?: true;
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
  /** A fork: the conversation it was copied from, as the agent's summary names it. */
  forkOf?: ForkOrigin;
  /** Work: how many never-allow rules its Work folder has; absent with none. */
  neverRules?: number;
  /** The thinking level picked for the conversation (`composer/<id>.json`); absent: the model's default. */
  thinking?: ThinkingLevel;
}
export type ConversationRow = ConversationInfo & { turns: number };

/** The conversation a fork was copied from. `title: null`: it no longer exists (a list row: it is not listed). */
export interface ForkOrigin {
  id: string;
  title: string | null;
}

/** Why a Work folder is refused. Each has a fixed sentence (`CWD_PROBLEM_MESSAGES`). */
export type CwdProblem = 'not-absolute' | 'missing' | 'not-directory' | 'unreadable' | 'root' | 'home' | 'app-data';

/** A local branch of a Work folder's work tree, as the New screen's branch menu lists it. */
export interface WorkBranch {
  name: string;
  /** Committer time of its tip, unix seconds; 0 when git gave none. */
  at: number;
  /** HEAD of this work tree. */
  current: boolean;
  /** The other worktree that has it checked out, or null. */
  elsewhere: string | null;
}

/** HEAD of a work tree: a branch, a detached commit (short hash), or a branch with no commit yet. */
export type WorkHead = { branch: string } | { detached: string } | { unborn: string };

/**
 * `GET /api/chat/work/git`. `repo`: the work tree's top, HEAD, the local branches (current first, then newest, at
 * most 1,000 of `total`), how many working Work conversations share the folder, and the uncommitted change count (null:
 * unknown). `no-git`: no git 2.23 or newer; `slow`: git ran out of time.
 */
export type WorkGit =
  | { kind: 'not-git' | 'no-git' | 'slow' }
  | {
      kind: 'repo';
      top: string;
      head: WorkHead;
      branches: WorkBranch[];
      total: number;
      busy: number;
      changes: number | null;
    };

/** `POST /api/chat/work/git/switch`: the outcome and the read after it. `text`: git's refusal, cleaned. */
export type WorkSwitchReply =
  | { outcome: 'switched'; git: WorkGit }
  | { outcome: 'refused' | 'timeout'; text: string; git: WorkGit };

/**
 * The `reason` of a switch the app refuses before git runs. `dirty`: the folder has uncommitted changes;
 * `status-unknown`: git could not count them in time.
 */
export type WorkGitErrorCode = 'branch-unknown' | 'folder-busy' | 'git-unavailable' | 'dirty' | 'status-unknown';

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
  /** Work: automatic continues sent after a length stop in this turn; absent while none was sent. */
  continues?: number;
  /** Work: the agent started this turn itself (a wake after a workflow run); no prompt was typed. */
  origin?: 'wake';
}

/**
 * The agent summarizes older messages of this conversation now (from its `auto_compact_started` until the end update,
 * the turn's end or the agent's exit), or a manual run (Compact now) from its start to its reply. `startedAt` is the
 * app's clock; `tokensBefore` the context the agent counted. `stopping`: the user stopped a manual run.
 */
export interface CompactionState {
  phase: 'running' | 'stopping';
  startedAt: number;
  tokensBefore?: number;
}

/**
 * How a manual compaction (Compact now) ended. Live only: the agent stores only a done one. `load-slow`: it failed
 * because the agent did not open the cold conversation in time.
 */
export type CompactOutcome = 'compacted' | 'stopped' | 'failed' | 'load-slow';

/**
 * A batch of hooks the agent waits on now (`hook_run_started`, live only): its event and tool as the agent named them,
 * how many run, and since when (the app's clock). It ends with the batch's `hook_execution`, the model's next words,
 * the turn's end, a compaction or the agent's exit.
 */
export interface HookRunState {
  event: string;
  tool?: string;
  count: number;
  startedAt: number;
}

/**
 * A recap the user asked for is being written: from the request until its line, the watchdog, a sent message or the
 * agent's exit. `startedAt` is the app's clock.
 */
export interface RecapState {
  startedAt: number;
}

/**
 * How a recap the user asked for ended. Live only. `shown`: its row came. `none`: the agent wrote none, or none came
 * in time. `off`: recaps are turned off in the agent's settings. `cancelled`: a message was sent while it was written,
 * or the conversation was deleted. `failed`: the agent could not be asked, or it stopped. `load-slow`: it failed because
 * the agent did not open the cold conversation in time.
 */
export type RecapEnd = 'shown' | 'none' | 'off' | 'cancelled' | 'failed' | 'load-slow';

/** What a rewind puts back: the conversation and the files, the conversation only, or the files only. */
export type RewindMode = 'all' | 'conversation' | 'files';

/**
 * A file a restore puts back, as the agent's dry run names it (relative to the Work folder): untouched since the agent
 * wrote it (`clean`), or changed, deleted or made again outside the agent since then; a restore overwrites those too.
 */
export type RewindFileState = 'clean' | 'modified' | 'deleted' | 'created';

export interface RewindFile {
  /** Redacted, hidden characters shown as `\u{…}`. */
  path: string;
  state: RewindFileState;
}

/** Work: a press on the goal strip (`POST …/goal`). */
export type GoalAction = 'pause' | 'resume' | 'clear';

/**
 * What Rewind to here would do, from the page's entry: the later steps it removes (`messages`: user messages among
 * them, the clicked one included), the background commands and schedules it stops and deletes, whether helper rows go
 * with it and whether the message had attachments. Work: `files`, the files a restore puts back (the agent's dry run,
 * conflicts first, at most {@link REWIND_FILES_MAX}), and `filesMore` past those; Chat has none. `token` names the
 * conversation as it was read; a commit with another one is refused.
 */
export interface RewindPlan {
  token: string;
  removes: { messages: number; steps: number };
  background: { commands: number; schedules: number; runs: number };
  helpers: boolean;
  attachments: boolean;
  files: RewindFile[];
  filesMore: number;
}

/** The most files a rewind plan lists. */
export const REWIND_FILES_MAX = 200;

/**
 * How a rewind ended, on the conversation's stream. `done`: the thread was cut before `entry` (`restored` files put
 * back). `partial`: as `done`, but the agent could not put `failed` back (redacted, at most {@link REWIND_FILES_MAX},
 * and `failedMore` past those) and keeps no copy of them. `changed`: a file changed outside the agent since the plan
 * showed it clean: nothing was rewound, and the page plans again. `refused`: the agent said no. `tasks`: a command or
 * schedule of the removed part could not be stopped, and nothing was rewound. `stopped`: the agent exited before its
 * reply. `failed`: the agent could not be started or loaded, or the engine is not running. `load-slow`: the agent did
 * not open the cold conversation in time; nothing was rewound.
 */
export type RewindEnd =
  | { outcome: 'done'; mode: RewindMode; restored: number; entry: number }
  | { outcome: 'partial'; mode: RewindMode; restored: number; entry: number; failed: string[]; failedMore: number }
  | { outcome: 'changed' }
  | { outcome: 'refused' | 'tasks' | 'stopped' | 'failed' | 'load-slow' };

/**
 * The agent's split of the context, read from `_x.ai/session/info` while the conversation is loaded. Estimates: the
 * three parts need not add up to the context's use.
 */
export interface ContextParts {
  systemTokens: number;
  toolCount: number;
  toolTokens: number;
  messageTokens: number;
  /** Where the agent compacts by itself, 0–100 of the window. */
  autoCompactPercent: number;
  /** Compactions in the agent process so far. */
  compactions: number;
}

/** `GET …/:id/context`. `parts` is null while the conversation is not loaded in a running agent, or its read failed. */
export interface ConversationContext {
  /** Tokens in the context now: the agent's count when loaded, else the last one stored. */
  used: number | null;
  parts: ContextParts | null;
}

/** What every open request of the agent carries, whatever its kind. */
export interface AskBase {
  id: string;
  /** The agent's call the request is about; the key of its `interaction_resolved`. */
  toolCallId: string;
  /** Host clock (epoch ms) when the card opened. */
  openedAt: number;
  /**
   * Work: a helper agent asks, not the conversation's own turn. `run`: the workflow run's name, null for a helper the
   * turn started; `cwd`: the helper's folder when it is not the Work folder; `child`: the helper's session id (the
   * `childId` of its `subagent` entry).
   */
  from?: { label: string; run: string | null; cwd: string | null; child: string };
}

export interface PermissionInteraction extends AskBase {
  kind: 'permission';
  /** From `_meta["x.ai/tool"].name`, else `toolCall.kind`. */
  toolName: string;
  /** Decides the label of the "always" button. */
  access: 'bash' | 'edit' | 'mcp' | 'other';
  /** At most 200 characters. */
  title: string;
  /** The server cut `title` at its cap. */
  titleCut?: true;
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
  /**
   * With `autoAsk`: the part that asks. `part` is text already on the card (redacted, at most 60 characters), empty
   * when the cause names no part; `more`: other simple commands that ask too.
   */
  autoAskDetail?: { part: string; more: number; cause: AutoAskCause | 'redirect' | null };
  /**
   * `bash` with `reject-once` offered: the words a never-allow rule would keep, as the card shows them (redacted,
   * hidden characters shown). `exact` equals `short` when the whole command gives no other scope.
   */
  never?: { short: string; exact: string };
  /** Set when the card opened in Bypass: the app cannot read the command and the Work folder has never-allow rules. */
  neverUnread?: true;
  /**
   * A `workflow` call that starts the PDF tool on a PDF of this conversation: its name (redacted, at most 200
   * characters), the pages it reads and its reader agents.
   */
  pdfRead?: { name: string; pages: number; readers: number };
}
/** The agent asks to leave plan mode; `toolCallId` is the plan tool row. */
export interface PlanInteraction extends AskBase {
  kind: 'plan';
  /** Markdown from the model: redacted, hidden characters shown as `\u{…}`. */
  plan: string;
}
/**
 * Work: the conversation's own agent asks the user its questions (`_x.ai/ask_user_question`); helpers never can. Texts
 * are the agent's, redacted and with hidden characters shown; the page answers by index, never by text.
 */
export interface QuestionInteraction extends AskBase {
  kind: 'question';
  /** The session is in plan mode. */
  planMode: boolean;
  /** In the agent's order. `multi`: pick any; `preview`: what the option looks like, single select only. */
  questions: Array<{
    text: string;
    multi: boolean;
    options: Array<{ label: string; description: string; preview: string | null }>;
  }>;
}
/**
 * A field of an MCP server's form as the card draws it, in the schema's order. Texts are the server's, redacted and with
 * hidden characters shown; option values stay in the host, and the page answers by field and option index.
 */
export type McpField = { name: string; title: string; description: string | null; required: boolean } & (
  | {
      type: 'text';
      format: 'email' | 'uri' | 'date' | 'date-time' | null;
      minLength: number | null;
      maxLength: number | null;
      initial: string;
    }
  | { type: 'number' | 'integer'; min: number | null; max: number | null; initial: string }
  | { type: 'boolean'; initial: boolean }
  | { type: 'choice'; options: Array<{ label: string }>; initial: number | null }
  | {
      type: 'choices';
      options: Array<{ label: string }>;
      minItems: number | null;
      maxItems: number | null;
      initial: number[];
    }
);
/**
 * Work: an MCP server asks the user for input (`_x.ai/mcp/elicit`): a form (`mode: 'form'`) or a page to open
 * (`mode: 'url'`). The request names no helper, so `from` is absent; `toolCallId` is the agent's own id for the
 * request, no tool row has it. `unusable`: the app cannot draw the request (`schema`: the form is not readable;
 * `field`: a needed field of a kind it does not know; `address`: not a safe web address), so only Decline is offered.
 */
export interface McpInteraction extends AskBase {
  kind: 'mcp-input';
  mode: 'form' | 'url';
  server: string;
  message: string;
  /** A form; absent when `unusable`. `leftOut`: fields of a kind the app cannot draw, none of them needed. */
  form?: { fields: McpField[]; leftOut: number };
  /**
   * A link; absent when `unusable`. `url`: the whole address, exactly what opens; `host`: its site; `secure`: https;
   * `lookalike`: a part of the site name is in its `xn--` form (letters that can look like others).
   */
  link?: { url: string; host: string; secure: boolean; lookalike?: true };
  unusable?: 'schema' | 'field' | 'address';
  /**
   * The workflow run whose wait the request belongs to (no answer or wake ran when the server asked): its name. The
   * request names no helper, so this is all the app knows of who asks.
   */
  run?: string;
}
export type Interaction = PermissionInteraction | PlanInteraction | QuestionInteraction | McpInteraction;

/** Why the host refused one value of an MCP form; `n` is the bound a bound code names. */
export type McpFieldError =
  | { code: 'required' | 'integer' | 'number' | 'email' | 'uri' | 'date' | 'dateTime' }
  | { code: 'minLength' | 'maxLength' | 'min' | 'max' | 'minItems' | 'maxItems'; n: number };

/** The reply of `POST …/interactions/:iid`: `fields` when the host refused an MCP form (one per shown field, null for a good one); the card stays open. */
export interface InteractionAnswerResult {
  fields?: Array<McpFieldError | null>;
  /** A never-allow answer: the rule and its words, or `saved: false` when the rule was not kept (the call is denied). */
  never?: { saved: true; ruleId: string; words: string } | { saved: false };
}

/** Work: one never-allow rule of a folder, as the rules list shows it. */
export interface NeverRuleView {
  id: string;
  words: string;
  at: number;
}

/** `GET …/never`: the Work folder and its rules, oldest first. */
export interface NeverRulesResponse {
  folder: string;
  rules: NeverRuleView[];
}

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
  | 'no-window'
  | 'temp';

/** Why a place in a temp root still asks: the unit rule that failed, or a mode change there (`mode`). */
export type TempAsk =
  | 'top'
  | 'private'
  | 'owner'
  | 'shared'
  | 'special'
  | 'volume'
  | 'app'
  | 'tool'
  | 'mode'
  | 'unreadable';

/** The cause of a disk check that asks: a unit rule, the Git repository, the global Git config, or the place itself. */
export type AutoAskCause = TempAsk | 'git' | 'git-global' | 'place';

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
      /** To Auto: the confirm named the temp folder. */
      temp?: true;
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
      /** Auto: a real place of the call is in the temp root. */
      temp?: true;
      /**
       * The helper session that asked (a subagent, a workflow run's agent): the `childId` of its `subagent` entry.
       * Absent for the conversation's own calls.
       */
      child?: string;
    }
  /**
   * A call the user denied with a note (`note`), or one a never-allow rule denied (`rule`): the agent skipped it and
   * went on. `ruleId` and `words` with `rule` only: `words` is the rule as the rule file holds it now, null when the
   * rule is gone.
   */
  | {
      kind: 'denied';
      at: number;
      toolCallId: string;
      access: 'bash' | 'edit' | 'mcp' | 'other';
      by: 'note' | 'rule';
      ruleId?: string;
      words?: string | null;
      /** The card as it showed, `options: []`. Null when read from disk, dropped from memory, or too long. */
      card: PermissionInteraction | null;
      /** The helper session that asked: the `childId` of its `subagent` entry. Absent for the conversation's own calls. */
      child?: string;
    }
  /**
   * Settings files of the agent changed while the mode was not Ask. `changed`: their paths, the user's home as `~`,
   * in a fixed order, at least one.
   */
  | { kind: 'settings'; at: number; changed: string[] }
  /**
   * Work: the user rewound the conversation (`conversation`, `all`) or put files back (`files`); `files`: how many
   * were put back. `from`: the time of the first entry the rewind removed; the calls inside `[from, at)` are gone.
   */
  | { kind: 'rewind'; at: number; mode: RewindMode; files: number; from?: number }
  /**
   * Work: an MCP server asked for input; never what was typed. `accept`: sent (a link: opened); `decline`; `cancel`:
   * closed by Stop or by the host; `expired`: closed with no answer (the server stopped waiting, no window);
   * `completed` (a link only): the server said the page is done. `server`: its name as printable text, at most 64
   * characters.
   */
  | {
      kind: 'mcp';
      at: number;
      server: string;
      mode: 'form' | 'url';
      outcome: 'accept' | 'decline' | 'cancel' | 'expired' | 'completed';
    };

/**
 * A one-shot fact of a conversation: fixed text per code, in memory only. The page shows one at a time, danger first,
 * then newest; `key-printed` stays for the conversation. `no-plan-mode` (Work): a message asked for plan mode and the
 * agent never entered it; it goes when the next message is sent.
 */
export type ConversationNotice =
  | { code: 'key-printed'; tone: 'warning' | 'danger'; provider: ProviderId }
  | { code: 'no-plan-mode'; tone: 'warning'; provider?: undefined }
  /** Work: the agent's question was over the card's limits and was answered "no answer"; it goes when the next message is sent. */
  | { code: 'question-too-long'; tone: 'warning'; provider?: undefined }
  /**
   * The agent did not take the conversation's thinking level before an answer (it refused, missed its deadline or does
   * not list the level); the answer ran at the agent's level. It goes when a later set succeeds or the level or model
   * changes.
   */
  | { code: 'thinking-not-set'; tone: 'warning'; provider?: undefined };

export type TranscriptOp =
  | { op: 'push'; entry: TranscriptEntry }
  /** The entry at `index` is `assistant` or `thinking`. */
  | { op: 'text'; index: number; append: string; endTs?: number }
  /** Whole-entry replace: tool, plan, subagent. */
  | { op: 'set'; index: number; entry: TranscriptEntry }
  | { op: 'metric'; metric: TurnMetric }
  /** The conversation's current context in tokens: the agent's `_meta.totalTokens`. */
  | { op: 'context'; used: number };

/** Who stopped or paused a run: this window or a typed `/workflow` command, the main agent's own call, an engine change. */
export type RunBy = 'you' | 'main-agent' | 'engine';

/**
 * A workflow run of the running agent process: `live` while it is `active`, else one of the newest that left it.
 * `stopping`: the host sent Stop and the run's agents have not drained. `waiting`: it waits on the person.
 * `wakeDue`: its end made a wake due that has not started: the main agent writes up the result next.
 */
export interface RunView {
  runId: string;
  name: string;
  status: string;
  live: boolean;
  stopping: boolean;
  by: RunBy | null;
  waiting: boolean;
  wakeDue: boolean;
}

/** Work: a helper's last `subagent_progress` tick (live only, never stored). */
export interface HelperProgress {
  /** The helper's child session id: its subagent entry's `childId`, its run card row's `agentId`. */
  childId: string;
  turns: number;
  toolCalls: number;
  /** The helper's current context in tokens, not its total use. */
  tokens: number;
  /** 0..100. */
  contextPct: number;
  errors: number;
  /** The helper's run time at the tick, by the agent's clock. */
  durationMs: number;
  /** The model's window in tokens, as the tick names it; null when it names none. */
  contextWindow: number | null;
  /** Host clock of the tick (epoch ms). */
  at: number;
  /** What its own live lines say it does now; null before the first of them. */
  now: AgentDoing | null;
}

/**
 * What a helper does now, read from its own live lines: a thought or a message chunk, its newest open tool call (the
 * call's newest title, one line, cut at 120 characters with `…`), or a compaction.
 */
export type AgentDoing =
  | { kind: 'thinking' }
  | { kind: 'writing' }
  | { kind: 'tool'; title: string }
  | { kind: 'compacting' };

/** Work: one `monitor_event` as the host keeps it (live only, never stored). */
export interface TaskEvent {
  /** Host clock of the event (epoch ms). */
  at: number;
  /** The monitor's raw output, cut at 500 characters (a cut ends in `…`), redacted. Never Markdown. */
  text: string;
  /** The host cut the text: its `…` is the host's, not the monitor's. */
  cut?: true;
}

/** Work: a command or monitor the running agent process holds now. */
export interface TaskLive {
  /** The task id: its tool entry's `task.id`. */
  id: string;
  kind: 'command' | 'monitor';
  /** Stop was sent and no end came yet. */
  stopping: boolean;
  /** Monitor events seen in this agent process. */
  events: number;
  /** The last 5 events, oldest first. */
  recent: TaskEvent[];
}

/**
 * Work: a schedule the running agent process holds, or one the user's Delete is on. `deleting`: `opening` while the
 * Delete loads the conversation first, `removing` while the agent deletes it. `firing`: the child session id of the
 * run its last fire started, until that run ends.
 */
export interface ScheduleLive {
  id: string;
  deleting: 'opening' | 'removing' | null;
  firing: string | null;
  /**
   * The next fire time (epoch ms) its last fire named: the agent sends a fire live only, so the stored entry keeps the
   * time the schedule was made with. Absent before a fire of this process.
   */
  nextTs?: number;
}

/** Work: the monitors that reported between the last turn end and the start of one wake; at most the last 20 wakes. */
export interface WakeReport {
  /** That wake entry's `ts`. */
  wakeTs: number;
  /**
   * Each monitor by its task id, with the description its last event named (redacted, cut, hidden characters shown):
   * the wake's words when the page holds no tool row of that monitor.
   */
  monitors: WakeMonitor[];
}

export interface WakeMonitor {
  id: string;
  description: string;
}

/**
 * Work: one agent the running agent process holds (a subagent of the conversation or an agent of a workflow run), from
 * its start line to its end line, ticked or not. A contract retry takes its row: the attempts before it stay as aliases.
 */
export interface AgentLive {
  /** Its newest attempt's child session id: its subagent entry's `childId`, the id its Stop goes to. */
  childId: string;
  /** The workflow run it works for; null for the conversation's own subagent. */
  run: string | null;
  /** Child session ids of its attempts, oldest first; the last is {@link childId}. */
  attempts: string[];
  /** Stop was sent and no end came yet. */
  stopping: boolean;
  /**
   * Whether a message can reach it: `on` while the host holds the address its live start line named, `off` once
   * this agent process answered that it has no message method, `no-address` when its start line named none (an agent
   * of a workflow run never has one).
   */
  messages: 'on' | 'off' | 'no-address';
}

/**
 * Work: the Agent view's start for one agent (`childId`, the attempt it shows): the list its `updates.jsonl` and the
 * host's kept live lines give, redacted. `live`: the agent runs, and its later lines reach the window as
 * `agent-entries` events; else the list is all there is.
 */
export interface AgentWatchAnswer {
  childId: string;
  /** Child session ids of the attempts of its row, oldest first; `childId` is one of them. */
  attempts: string[];
  entries: TranscriptEntry[];
  live: boolean;
}

/** The longest message to a subagent, in UTF-8 bytes: the agent refuses a longer one. */
export const SUBAGENT_MESSAGE_MAX_BYTES = 32 * 1024;

/**
 * Work: how a message to a subagent ended. `sent`: it took it. `unsure`: no clear answer came (no reply in time, or the
 * agent could not tell), so it may have arrived. `not-ready`: it is not active now. `busy`: it holds messages it has
 * not read yet. `late`: it did not take it in time. `too-long`: past {@link SUBAGENT_MESSAGE_MAX_BYTES}. `off`: this
 * agent process takes no messages. `error`: any other refusal or failure.
 */
export type SubagentMessageResult = 'sent' | 'unsure' | 'not-ready' | 'busy' | 'late' | 'too-long' | 'off' | 'error';

/** Work: what the host knows live of the conversation's background work, beside the stored entries. */
export interface LiveActivity {
  /** The helpers of the running agent process that ticked at least once. */
  helpers: HelperProgress[];
  /** The agents of the running agent process (subagents and workflow agents), in start order. */
  agents: AgentLive[];
  /** The commands and monitors of the running agent process, in start order. */
  tasks: TaskLive[];
  /** The schedules of the running agent process and the ones a Delete is on, in the order they came. */
  schedules: ScheduleLive[];
  /** Oldest first. A wake row with no stored cause of its own reads its title from these. */
  reports: WakeReport[];
  /** Child session ids of the agents this host's Stop ended, oldest first: only these read "Stopped by you". */
  stoppedByYou: string[];
}

/** Messages one conversation's queue holds at most. */
export const QUEUE_MAX = 5;

/** A message typed while an answer ran, in the host's queue: its text, and its files as names and sizes. */
export interface QueuedMessage {
  id: string;
  text: string;
  files: UserAttachment[];
  /** Work: the message asks for plan mode (`true`) or leaves it (`false`), as the chip was when it was queued. */
  plan?: boolean;
  /** Host clock (epoch ms) when it was queued. */
  queuedAt: number;
  /** A page has the row open for edit: the queue sends nothing while one is. */
  editing: boolean;
}

/**
 * Why the queue waits for the user: their Stop, an answer that did not complete, an app restart, or a check that
 * failed when the next row was due (`reason`: its fixed sentence; `code`: its reason code, such as `engine-lost`).
 */
export type QueueHold =
  | { why: 'stopped' | 'failed' | 'restart' }
  | { why: 'blocked'; reason: string; code: string | null };

/** The queue of one conversation, oldest first; `held: null` sends the next row when the conversation is free. */
export interface QueueState {
  items: QueuedMessage[];
  held: QueueHold | null;
  /**
   * The rows the host counts against {@link QUEUE_MAX}: `items` and the row whose answer starts (out of the list until
   * its prompt went to the agent). The page offers Queue while this is under the cap.
   */
  count: number;
}

/**
 * Work: a command of the agent's list that the conversation may send as typed: an allowed built-in, or one of the
 * user's skills and workflows. Name, words and hint are agent text: one line, cut, redacted.
 */
export interface CommandView {
  name: string;
  kind: 'builtin' | 'skill' | 'workflow';
  description: string;
  /** The words it takes, as the agent hints them; null when none. */
  hint: string | null;
}

/**
 * The agent's `model_auto_switched` during a turn: the model it could not use, the one it took instead, its words
 * (cut at 300 characters). `''` from the agent is null. Live only: the agent never stores it.
 */
export interface ModelSwitch {
  previous: string | null;
  next: string | null;
  reason: string;
}

/** How a turn ended: the `turn` event with `turn: null` carries it, and `sync` repeats the last one (memory only). */
export interface TurnEnd {
  turnId: string;
  outcome?: TurnOutcome;
  failure?: ChatFailure;
  /** The turn ended before `session/prompt` went to the agent. */
  unsent?: true;
  /** The agent could not use the conversation's model during this turn. */
  model?: ModelSwitch;
  /** A hook blocked the prompt (the agent stores none of it): its text, for the message box. */
  blocked?: string;
}

/** A rewind's end with its number in this app run (from 1): the `rewind` event and `sync` carry the same end. */
export interface RewindEndMark {
  n: number;
  end: RewindEnd;
}

/**
 * The running answer's last retry episode as the reducer holds it (its row may not be placed yet): open, or ended with
 * how it ended.
 */
export type RetryLive = Extract<TranscriptEntry, { kind: 'notice'; code: 'retry' }>;

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
      /** The conversation's current context in tokens; absent while no agent update carried it. */
      contextUsed?: number;
      /** A compaction runs now; absent when none does. */
      compaction?: CompactionState;
      /** The agent waits on a batch of hooks now; absent when it does not. */
      hookRun?: HookRunState;
      /** A recap is being written now; absent when none is. */
      recap?: RecapState;
      /** The last rewind end of this app run; absent before the first. */
      rewindEnd?: RewindEndMark;
      /** The running answer's last retry episode; absent while it has none. */
      retry?: RetryLive;
      /** Work: the conversation's workflow runs of the running agent: live ones and the newest ended ones. */
      runs: RunView[];
      /** Work: the live facts of its background work. */
      live: LiveActivity;
      /** The queued messages; absent while none waits and nothing is held. */
      queue?: QueueState;
      /** The agent's commands the conversation may send; absent while the session is not loaded. */
      commands?: CommandView[];
    }
  | { type: 'ops'; seq: number; ops: TranscriptOp[] }
  | { type: 'notice'; seq: number; notice: ConversationNotice }
  /** Every notice with this code is gone. */
  | { type: 'notice-cleared'; seq: number; code: ConversationNotice['code'] }
  | {
      type: 'turn';
      seq: number;
      turn: TurnState | null;
      /** With `turn: null`: the turn that ended. */
      turnId?: string;
      outcome?: TurnOutcome;
      failure?: ChatFailure;
      unsent?: true;
      model?: ModelSwitch;
      blocked?: string;
    }
  /** A compaction started or changed (the state) or ended (null); `end`: how a manual run ended. */
  | { type: 'compaction'; seq: number; compaction: CompactionState | null; end?: CompactOutcome }
  /** A batch of hooks started (its state) or ended (null). */
  | { type: 'hook-run'; seq: number; hookRun: HookRunState | null }
  /** A recap the user asked for started (the state) or ended (null, with how). */
  | { type: 'recap'; seq: number; recap: RecapState | null; end?: RecapEnd }
  /**
   * A rewind of the conversation ended (`n`: its number, as `sync` repeats it); a `done` one comes after the cut's
   * `sync`.
   */
  | { type: 'rewind'; seq: number; n: number; end: RewindEnd }
  /** The running answer's last retry episode changed (the episode), or the answer ended (null). */
  | { type: 'retry'; seq: number; retry: RetryLive | null }
  /**
   * Work: the workflow runs changed; the whole list. `ops`: the transcript ops queued before it, applied first, so a
   * run's status and the host's word on its wake reach the page in one step.
   */
  | { type: 'runs'; seq: number; runs: RunView[]; ops?: TranscriptOp[] }
  /** Work: the live facts changed (at most one a second); the whole state. */
  | { type: 'activity'; seq: number; live: LiveActivity }
  /**
   * Work: the Agent view of `window` on `childId`: ops on the list its watch answered, in order. At the agent's end
   * `stored` holds the stored list in place of that list, and nothing more comes for the watch.
   */
  | {
      type: 'agent-entries';
      seq: number;
      window: string;
      childId: string;
      ops: TranscriptOp[];
      stored?: TranscriptEntry[];
    }
  /** The queued messages changed; the whole state. */
  | { type: 'queue'; seq: number; queue: QueueState }
  /** The loaded session's command list changed (the whole list), or the agent process exited (null). */
  | { type: 'commands'; seq: number; commands: CommandView[] | null }
  | { type: 'interaction'; seq: number; interaction: Interaction }
  | { type: 'interaction-closed'; seq: number; id: string; reason: 'answered' | 'cancelled' | 'expired' }
  | { type: 'conversation'; seq: number; conversation: ConversationInfo }
  | { type: 'permission-mark'; seq: number; mark: PermissionMark }
  /** Work: a never-allow rule was removed: its marks now read "rule removed". */
  | { type: 'rule-gone'; seq: number; ruleId: string }
  | { type: 'gone'; seq: number; reason: 'deleted' | 'unknown' };

export interface AgentModeState {
  state: 'stopped' | 'starting' | 'ready' | 'crashed' | 'failed' | 'blocked';
  stale: boolean;
  failure: ChatFailure | null;
  agentVersion: string | null;
  turnsInFlight: number;
  /** Sessions with engine work (a workflow run, a wake turn, a helper) or a history op. */
  background: number;
  /** Live commands and monitors of every session: they hold the process, not the engine. */
  tasks: number;
  /** The engine restarted under the running process; its commands run on until the next message restarts it. */
  engineLost: boolean;
}
export interface ChatHostState {
  inference: 'ready' | 'starting' | 'loading' | 'stopped';
  /**
   * The loads the engine runs now, oldest first; empty when none. `startedAt` is the engine's own clock (epoch ms).
   * `phase: 'kernels'` with `steps` known: the load prepares GPU kernels, `stepsDone` of `steps`.
   */
  loading: Array<{
    modelId: string;
    startedAt: number;
    phase: 'weights' | 'kernels';
    stepsDone: number | null;
    steps: number | null;
  }>;
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
  /**
   * The requests the engine holds now, oldest first: `modelId` is the name the request asked for, `session` the agent
   * session it named (null when it named none or the engine is older), `phase`, the prefill counts and the tokens
   * generated so far as the In flight panel shows them.
   */
  requests: Array<{
    session: string | null;
    modelId: string;
    phase: 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | null;
    startedAt: number;
    prefillDone: number | null;
    prefillTotal: number | null;
    generated: number | null;
  }>;
  /**
   * The chat models the engine serves. `base` groups the builds of one model (equal strings, letter for letter);
   * `variant` names the build, unique inside its base; `sizeBytes` is the size on disk (every part of a split GGUF), null
   * when unknown; `contextTokens` is the window an agent process last listed for the model (kept on disk, so known
   * while no agent runs), null before any agent listed one. `images`: the model takes pictures, by the loaded model's
   * own answer while it is in memory, else by its files on disk. `family`: the engine's model family (`qwen3_5`,
   * `gemma4`, …), null when the endpoint names none. `thinking`: the levels an agent process listed for the model in
   * this app run (null: it listed none); absent before a process listed the model, so the Thinking row is not drawn.
   */
  models: Array<{
    id: string;
    base: string;
    variant: string;
    format: 'gguf' | 'mlx';
    contextTokens: number | null;
    sizeBytes: number | null;
    images: boolean;
    family: string | null;
    thinking?: ModelThinking | null;
  }>;
  /** The resident chat model when one is in `models`, else the first of `models`. */
  defaultModelId: string | null;
  agent: {
    binary: 'ok' | 'missing' | 'too-old';
    source: string;
    tag: string | null;
    minTag: string;
    versionVerified: boolean;
    install: 'idle' | 'running' | 'failed';
    /** False when in-app agent installation/updates are unavailable in this build. */
    canInstall?: boolean;
  };
  modes: { chat: AgentModeState; work: AgentModeState };
  providers: {
    search: SearchProviderId | null;
    searchCheck: KeyCheck | null;
    firecrawl: boolean;
    fetch: ProviderId | null;
    applyPending: boolean;
  };
  /**
   * `waiting`: an interaction is open; `asks`: how many are. `background`: engine work (a workflow run, a wake turn, a
   * helper) or a history op is live; `helpers`: a subagent of the conversation's own runs; `tasks`: its live commands
   * and monitors.
   */
  active: Array<{
    conversationId: string;
    mode: ChatMode;
    busy: boolean;
    waiting: boolean;
    asks: number;
    background: boolean;
    helpers: boolean;
    tasks: number;
    /** A history op or a recap of the conversation runs now. */
    op: 'rewind' | 'compact' | 'fork' | 'recap' | null;
  }>;
  /** The Work conversations not in Ask, with their mode (the list has no other source). */
  permissionRaised: Array<{ id: string; mode: 'auto' | 'bypass' }>;
  /** The mode picked on the New Work screen for the draft `draft`, before its conversation exists. */
  permissionDraft: { draft: string; mode: 'auto' | 'bypass' } | null;
  /** The temp root Auto covers, as people type it (`/tmp`); null when Auto covers the Work folder only. */
  tempRoot: string | null;
  /** The real top of that temp root: a Work folder that holds or equals it gets no temp root. */
  tempTop: string | null;
  /** Where Work without a folder makes each conversation's own folder (`~/Portato`). */
  scratchRoot: string;
  /** Per mode, whether a PDF that does not fit one message can go to the PDF tool. */
  pdfTool: { chat: boolean; work: boolean };
  /** How earlier Control Panel lives ended, newest first, from MAIN's ledger: what stopped a turn that was running. */
  appEnds: AppEnd[];
}

/**
 * One end of a Control Panel life. `decidedAt` is when MAIN decided the stop (for an exit nobody asked for, the exit
 * less the kill grace); `silentMs` how long the panel had not answered (`no-ping` only); `convs` the turns MAIN's last
 * pong listed; `cancelled` the turns the dying panel said it cancelled, null when that note never arrived.
 */
export interface AppEnd {
  trigger: 'no-ping' | 'crash' | 'signal' | 'user';
  decidedAt: number;
  exitAt: number;
  silentMs: number | null;
  code: number | null;
  convs: string[];
  cancelled: string[] | null;
}
export type ChatHostEvent = { type: 'state'; state: ChatHostState } | { type: 'conversations-changed' };
