/** Wire shapes of the Sessions API, read from grok-build (`mlx agent`) session folders. */

import type { UserAttachment } from '../chat/attachment-rule.js';

export interface SessionTokens {
  input: number;
  output: number;
  cached: number;
}

/** Who started a session: a terminal `mlx agent`, or the app in Chat or Work mode. */
export type SessionOrigin = 'agent' | 'chat' | 'work';

export interface SessionRow {
  id: string;
  origin: SessionOrigin;
  /** The agent's title, else the title of the first typed prompt; '' only for a session with no prompt. */
  title: string;
  cwd: string;
  model: string | null;
  branch: string | null;
  createdAt: string;
  lastActiveAt: string;
  messageCount: number;
  live: boolean;
  kind: string | null;
  /** Fork source or subagent parent; may name a session that no longer exists. */
  parentId: string | null;
  subagentCount: number;
  /** `null` when no completed turn carried usage. */
  tokens: SessionTokens | null;
}

export interface SubagentRef {
  id: string;
  title: string;
  type: string;
  description: string;
  status: string;
  model: string | null;
  /** False when the child folder is gone or ambiguous; the UI must not link it. */
  exists: boolean;
}

export interface SessionDetail extends SessionRow {
  gitRoot: string | null;
  headCommit: string | null;
  remotes: string[];
  agentName: string | null;
  reasoningEffort: string | null;
  lastTurnSummary: string | null;
  /** `title: null` means the parent folder is missing. */
  parent: { id: string; title: string | null } | null;
  subagents: SubagentRef[];
}

/** What woke the agent for a turn the user did not type, read from the prompt id the agent recorded. */
export type WakeCause = 'workflow' | 'subagent' | 'task' | 'notifications' | 'goal' | 'message' | 'unknown';

/** Why the agent sent a model request again, read from its `retry_state` / `auto_recovery_*` lines. */
export type RetryCause = 'stalled' | 'connection' | 'server' | 'busy' | 'loop' | 'recovery' | 'other';

/** One picture the agent made smaller (`image_compressed`): bytes and pixel sides before and after. */
export interface PictureSize {
  fromBytes: number;
  toBytes: number;
  from: [number, number];
  to: [number, number];
}

export type TranscriptEntry = (
  | {
      kind: 'user';
      ts: number;
      text: string;
      promptIndex: number | null;
      interjection?: boolean;
      /** The `continue` prompt right after a length end: drawn as a divider, not a message. */
      continued?: true;
      /** The files and pictures sent with the message: names and sizes only, never their content. */
      attachments?: UserAttachment[];
      /** The user's note on a denied call: part of the running answer, never a new turn. */
      note?: true;
      /**
       * A host turn (`_meta.hostTurn`) typed as `/workflow pause|resume|stop <name>`: the verb and the run the name
       * picks by the agent's rule (a prefix of its run id or name, an exact one first). None when no run or two match.
       */
      runControl?: { verb: 'pause' | 'resume' | 'stop'; runId: string };
    }
  /** A hidden prompt the agent wrote to itself when a background event woke it. Never a message of the user. */
  | {
      kind: 'wake';
      ts: number;
      cause: WakeCause;
      /**
       * cause 'workflow': the run and the revision its prompt id names (`workflow-completed-<run>-<revision>`), and that
       * revision's stored update (null when the transcript holds none). A later update never changes it: a resume
       * empties the run's report, and a run that wakes twice has two rows. `report` is cut at 20 KB.
       */
      workflow?: {
        runId: string;
        revision: number;
        name: string;
        snapshot: {
          status: string;
          report?: string;
          truncated?: true;
          pauseMessage?: string;
          agentsUsed: number;
          agentBudget: number | null;
          /** Roster rows that say `failed`. */
          agentsFailed: number;
          elapsedMs: number;
        } | null;
      };
      /** cause 'subagent': the child whose finish woke the agent. */
      subagent?: { childId: string; description: string };
      /** cause 'task': the command or monitor whose end woke the agent. */
      task?: { id: string; kind: 'command' | 'monitor'; label: string };
      /** cause 'notifications': what ended with no wake of its own since the turn before, at most 8. */
      tasks?: Array<{ id: string; kind: 'command' | 'monitor'; label: string }>;
      /** Cut at 16 KB. */
      text: string;
      truncated?: true;
    }
  | { kind: 'assistant'; ts: number; text: string; endTs?: number }
  | {
      kind: 'thinking';
      ts: number;
      text: string;
      endTs?: number;
      /** Made from live lines: `ts` and `endTs` are its first and last chunk. A thought read from disk has no time. */
      timed?: true;
    }
  | {
      kind: 'tool';
      ts: number;
      id: string;
      name: string;
      title: string;
      status: string;
      input?: string;
      output?: string;
      truncated?: boolean;
      endTs?: number;
      /** A shell command's non-zero exit code. */
      exitCode?: number;
      /** A shell command stopped by the agent or the system (`timeout`, `oom`, …). */
      signal?: string;
      timedOut?: true;
      /**
       * The background task this call started: a command the agent moved to the background, or a monitor. `label` is
       * the call's description, else the first line of its command. `endTs`: the agent said it ended (it exited, was
       * killed or hit its time limit, or a new agent process no longer has it); unset while it runs. `killed`: its end
       * line says it was stopped (the user's Stop, the model's kill, a rewind), not that it ended by itself.
       */
      task?: { id: string; kind: 'command' | 'monitor'; label: string; endTs?: number; killed?: true };
      /**
       * The schedule this call made, with the agent's last words on it: its prompt, how it repeats (`every 5 minutes`)
       * and the next run it named (epoch ms; null when it named none). `endTs`: the agent removed it (deleted, expired,
       * or a one-time run done). The agent process stopping is no removal: the schedule comes back when it loads again.
       */
      schedule?: { id: string; prompt: string; repeat: string | null; nextTs: number | null; endTs?: number };
      /**
       * The change this call asks for (before its result) or made (status `completed`), per file, as the agent's diff
       * blocks say. The page gets hunks only, never the whole old or new file.
       */
      edits?: Array<{
        path: string;
        /** The result had no old text: a new file. */
        create?: true;
        added: number;
        removed: number;
        /**
         * `start`: old and new number of the hunk's first line (old 0 when the old side is empty); null when the agent
         * named no line. `body`: one line per row, first character ' ', '-' or '+'.
         */
        hunks: Array<{ start: [number, number] | null; body: string }>;
        /** Changed lines left out by the caps. */
        cut?: number;
        /** The texts differ only in line endings or the last newline. */
        endsOnly?: true;
      }>;
      /**
       * `ask_user_question`: the questions as the model wrote them (option labels only) and what the stored result
       * sentence says. `result` null: no result yet (or the call failed). `answered`: per question, the picked labels
       * and the user's own text, `raw` when that answer cannot be read exactly (shown as written, a pick is never
       * guessed), null when it was not answered. `none`: skipped, stopped or timed out (one sentence on disk).
       * `discuss` / `plan-now`: the plan-mode replies. `text`: a sentence the reader does not know.
       */
      ask?: {
        questions: Array<{ text: string; multi: boolean; options: string[] }>;
        result:
          | null
          | {
              kind: 'answered';
              answers: Array<{ picks: string[]; other: string | null } | { raw: string } | null>;
            }
          | { kind: 'none' }
          | { kind: 'discuss' }
          | { kind: 'plan-now' }
          | { kind: 'text'; text: string };
      };
      /**
       * A hook denied this call: the agent's words on it (`hook_annotation` of kind `tool_outcome`, paired with the
       * call's "Hook denied: " failure), cut at 600 characters.
       */
      blocked?: string;
      /**
       * Pictures the call read (`read_file` of a picture, or of a PDF as one picture per page), sizes only: the page
       * asks the session reader for one picture's bytes when the row opens. `n`: the picture's place among the call's
       * image blocks, from 0; `bytes`: its decoded size; `page`: the PDF page it shows, from 1.
       */
      pictures?: Array<{ n: number; mediaType: string; bytes: number; page?: number }>;
      /** A PDF read: how many pages the file has. */
      pdfPages?: number;
      /**
       * A `workflow` call, by its `rawInput.source.type`: `launch` (a name, script or script path), `check`
       * (`validate_only`), `pause`, `stop`, `resume`. `id`: the run it started or names; null when none came back and
       * the run it names is not known here (a refused launch, a check).
       */
      run?: { id: string | null; kind: 'launch' | 'check' | 'pause' | 'stop' | 'resume' };
    }
  | { kind: 'plan'; ts: number; entries: { content: string; status: string; priority: string }[] }
  | {
      kind: 'subagent';
      ts: number;
      childId: string;
      type: string;
      description: string;
      status?: string;
      durationMs?: number;
      /** `tokens_used`: the context size at the end of this attempt. */
      tokens?: number;
      /** `total_tokens_used`: input plus output tokens this attempt billed, nested subagents included; absent at 0. */
      tokensUsed?: number;
      /** `subagent_finished`: the helper's tool calls and turns. */
      toolCalls?: number;
      turns?: number;
      error?: string;
      /** Set by the detail route; only `true` may be linked. */
      exists?: boolean;
      /** Time of the `subagent_finished` line. */
      endTs?: number;
      /** Made by a finish line with no spawn line: `ts` is the finish time. */
      noSpawn?: true;
      /** `workflow_run_id`: the run that started it, whether or not the run's `workflow` entry is in the transcript. */
      run?: string;
      /** `resumed_from`: the child session it resumed, by a contract retry, a script `resume_from` or a Task `resume_from`. */
      resumedFrom?: string;
      /** `retry_of`: the attempt this contract retry follows. */
      retryOf?: string;
      /** `subagent_spawned.model`: the model it runs on. */
      model?: string;
    }
  | { kind: 'model'; ts: number; modelId: string }
  /**
   * The agent's session mode moved (`current_mode_update`): `plan` writes a plan and changes no files. A repeat of the
   * mode makes none. Like a notice, it never starts a turn and is placed right after the prompt it came with.
   */
  | { kind: 'plan-mode'; ts: number; mode: 'plan' | 'default' | 'ask' }
  /** The agent summarized older messages to free context (`auto_compact_*`): done, failed, or stopped by Stop. */
  | {
      kind: 'compaction';
      ts: number;
      outcome: 'done' | 'failed' | 'stopped';
      tokensBefore?: number;
      tokensAfter?: number;
      /** A failure the agent tries again at its next step. */
      retry?: true;
    }
  /**
   * The agent's "where things stand" (`session_recap`), written when the user asked for one. `text` is the model's,
   * cut at 2,000 characters. Like a notice, it never starts a turn and never counts as work.
   */
  | { kind: 'recap'; ts: number; text: string }
  /**
   * One goal of the goal loop (`/goal`), from its `goal_updated` lines: made at the first line of its id, then each
   * stored line sets it again (live-only ticks too, but a tick sends the page nothing unless a shown field moved).
   * `status` and `phase` are the agent's words (`active`, `user_paused`, `back_off_paused`, `no_progress_paused`,
   * `infra_paused`, `blocked`, `budget_limited`, `complete`; `cleared` once `/goal clear` ran). `asOf` is the time of
   * the line `elapsedMs` and `tokensUsed` were read from. `pauseMessage` is the agent's, cut at 300 characters.
   * `verifying` / `planning`: the agent checks the result or writes the plan now. Like a notice, it never starts a
   * turn and never counts as work.
   */
  | {
      kind: 'goal';
      ts: number;
      goalId: string;
      objective: string;
      status: string;
      phase: string;
      tokensUsed: number;
      tokenBudget?: number;
      elapsedMs: number;
      pauseMessage?: string;
      verifying?: true;
      planning?: true;
      asOf: number;
    }
  /**
   * One workflow run, from its `workflow_updated` lines: made at the first, replaced by each with a higher `revision`.
   * `status`, `phases[].state` and `agents[].state` are the agent's words. The Sessions reader also reads the run's
   * `state.json`: a newer revision there replaces the fields, and it adds the steps' `detail` and the `history`.
   */
  | {
      kind: 'workflow';
      /** The first update. */
      ts: number;
      /** The update applied last (for a newer `state.json`, the file's time). */
      updatedTs: number;
      runId: string;
      revision: number;
      name: string;
      status: string;
      objective?: string;
      /** `elapsed_ms`: run time with paused time left out, as of `updatedTs`. */
      elapsedMs: number;
      /** `agent_usage_incomplete`: `agentsUsed` may be low. */
      usageIncomplete?: true;
      /** `detail`: the script's words for the step, from `state.json` only. */
      phases: { title: string; state: string; detail?: string }[];
      currentPhase: string | null;
      agentBudget: number | null;
      agentsUsed: number;
      agents: {
        agentId: string;
        label: string;
        phase: string | null;
        /** Only when the script named one. */
        model?: string;
        state: string;
        /** The agent's total use over all its attempts, not its context. */
        tokens: number;
        durationMs: number;
      }[];
      /** Each `phase_entered` an update or `state.json` named: the step and when it began, oldest first. */
      phasesEntered: { title: string; ts: number }[];
      /** `state.json`'s history (steps, `log` lines, pauses, the end), oldest first; absent without the file. */
      history?: { event: string; detail: string | null; ts: number }[];
      pauseMessage?: string;
      /** `result_summary`, cut at 20 KB: the agent cuts it at 16 KB, so its own cut is the one shown. */
      report?: string;
      truncated?: true;
    }
  | {
      kind: 'turn-end';
      ts: number;
      stopReason: string;
      elapsedMs?: number;
      error?: string;
      /** The turn ended on the model's length limit (the agent stores it as an error). */
      lengthLimit?: true;
      /** A hook blocked the turn's prompt: the stored end is cancelled with the category `HookDenied`. */
      hookStopped?: true;
    }
  /**
   * A stored fact of the agent's that is no step of the answer. It never starts a turn and never counts as work; one
   * made while no turn is open is placed right after the next turn's prompt.
   */
  | ({ kind: 'notice'; ts: number } & (
      | {
          /**
           * One retry episode: the agent sent the model request again (`attempts`, `max`: its last try and budget).
           * `endTs` is set when output came back (`recovered`), the agent gave up (`gave-up`), or the turn ended (no
           * outcome); none while it retries. `detail`: the agent's words, cut at 300 characters.
           */
          code: 'retry';
          cause: RetryCause;
          attempts: number;
          max: number;
          endTs?: number;
          outcome?: 'recovered' | 'gave-up';
          detail?: string;
        }
      | {
          /**
           * Pictures the agent left out of a request (`image_dropped`) or made smaller (`image_compressed`). `count`: the
           * pictures made smaller, or a dropped line's notes (one note can speak for several pictures); `notes`: the
           * agent's words, each cut at 300 characters, at most 6; `sizes`: each shrunk picture.
           */
          code: 'picture';
          what: 'dropped' | 'shrunk';
          count: number;
          notes: string[];
          sizes?: PictureSize[];
        }
      | {
          /**
           * A hook run that failed (a hook's block is no failure), once per hook, event, tool and error in the session:
           * `times` counts them, `lastTs` is the newest. `hook`: the name as the agent sent it without its
           * `:{event}[i].hooks[j]` tail; `event` and `tool` as the agent named them; `detail`: the error, cut at 600
           * characters.
           */
          code: 'hook';
          event: string;
          tool?: string;
          hook: string;
          detail: string;
          times: number;
          lastTs: number;
        }
      | {
          /** A hook's own message, or the agent's words on a hook's verdict (`hook_annotation`), cut at 600 characters. */
          code: 'hook-note';
          text: string;
        }
    ))
) & {
  /**
   * Made from a line another session wrote (its `_meta.eventId` does not start with this session's id and `-`): a
   * retried or resumed child starts with a copy of the attempt it continues, which keeps that attempt's times. Set on a
   * child session's own transcript only.
   */
  copied?: true;
};

export interface TurnMetric {
  promptId: string;
  ts: number;
  /** The `turn_completed` line is a copy from another session, as {@link TranscriptEntry} `copied` says. */
  copied?: true;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
  elapsedMs: number | null;
  /** Time to first token of the turn's first model call. */
  ttftMs: number | null;
  /** Summed measured prefill windows; set with `prefillTokens`, never alone. */
  prefillMs: number | null;
  prefillTokens: number | null;
  /** Model calls in the prefill sums. */
  prefillCalls: number | null;
  /** Model calls of the turn. */
  modelCalls: number | null;
  decodeTps: number | null;
  model: string | null;
}

export interface SessionListQuery {
  q?: string;
  cwd?: string;
  limit?: number;
  offset?: number;
}

export interface SessionListResponse {
  sessions: SessionRow[];
  total: number;
  cwds: string[];
}

export interface SessionDetailResponse {
  session: SessionDetail;
  transcript: TranscriptEntry[];
  turns: TurnMetric[];
  transcriptError?: string;
  /** The last `_meta.totalTokens` in `updates.jsonl`: the conversation's current context in tokens. */
  contextUsed?: number;
  /** Null for a Chat conversation: a terminal cannot open the Chat agent's home. */
  resumeCommand: string | null;
}

/** `PATCH /api/sessions/:id` body `{title}`; `title` is the sanitized title as stored. */
export interface SessionRenameResponse {
  id: string;
  title: string;
}

/** `DELETE /api/sessions/:id`: the session first, then its subagent children. */
export interface SessionDeleteResponse {
  deleted: string[];
}

/** `GET /api/sessions/:id/pictures/:call/:n`: the n-th picture (from 0) a tool call read, as the agent stored it. */
export interface SessionPictureResponse {
  /** png, jpeg, webp or gif. */
  mediaType: string;
  /** base64. */
  data: string;
}

export type TimelineLaneStatus = 'completed' | 'failed' | 'cancelled' | 'running' | 'unknown';
export type TimelineNoSteps =
  | 'missing'
  | 'empty'
  | 'too-large'
  | 'unreadable'
  /** The file has events, none inside this run. */
  | 'out-of-range'
  /** A finish line with no spawn line. */
  | 'no-spawn'
  | 'folded';

export interface TimelineLane {
  /** 'main', the child session id, or 'folded'. */
  id: string;
  role: 'main' | 'subagent' | 'folded';
  name: string;
  /** Subagent type; null on the main lane of a top-level session and on the folded lane. */
  type: string | null;
  model: string | null;
  /** 0..7 for a subagent lane; null = neutral. */
  slot: number | null;
  /** The lane's own session when its folder exists; only then the UI links it. */
  sessionId: string | null;
  status: TimelineLaneStatus;
  /** The lane's first run was spawned before this window. */
  carriedIn: boolean;
  /** The window is finished and the lane's last run returns after it. */
  carriedOut: boolean;
  /** Set when the lane has one `run` span instead of steps. */
  noSteps: TimelineNoSteps | null;
  /** Size of the file that was refused, for 'too-large'. */
  fileBytes: number | null;
  /** A step crossed a handoff time and was cut there. Events are never moved. */
  clamped: boolean;
  /** Subagents this lane spawned itself (not drawn). On the folded lane: how many subagents it stands for. */
  subagents: number;
  /** Usage reports inside the window, ascending. */
  tokenReports: Array<{ atMs: number; tokens: number }>;
}

interface TimelineSpanBase {
  /** `${lane}:${index}`, stable while the files do not change. */
  id: string;
  lane: string;
  /** ms from range.startMs. */
  startMs: number;
  /** null only with status 'running'. */
  endMs: number | null;
  status: 'ok' | 'failed' | 'running' | 'unfinished';
  inferredStart: boolean;
  inferredEnd: boolean;
  /** The turn's window cut the span on this side: it began before the window or went on after it. */
  cutStart: boolean;
  cutEnd: boolean;
  /** A model request retry at or right before this span. Only on `wait` / `model` and `generate` spans. */
  retry: { attempt: number; max: number } | null;
}

export type TimelineSpan =
  | (TimelineSpanBase & { kind: 'generate'; phase: 'reasoning' | 'answer' | 'tool-call' })
  | (TimelineSpanBase & { kind: 'tool'; tools: string[]; calls: number; failedCalls: number })
  /** until: what ended the wait. 'open' = still waiting (running) or cut by the lane's end. */
  | (TimelineSpanBase & { kind: 'wait'; reason: 'model'; until: 'text' | 'tool' | 'open' })
  /** calls: spawn tool calls merged into it; 0 = built from spawn lines alone. */
  | (TimelineSpanBase & { kind: 'wait'; reason: 'subagent'; tools: string[]; calls: number; failedCalls: number })
  | (TimelineSpanBase & { kind: 'run'; noSteps: TimelineNoSteps })
  /** The main lane before a woken turn: no turn was open. At most one, the main lane's first span. */
  | (TimelineSpanBase & { kind: 'idle' });

export interface TimelineHandoff {
  /** `${lane}:${kind}:${run index}`. */
  id: string;
  /** The child lane. The other end is always 'main'. */
  lane: string;
  kind: 'spawn' | 'return';
  atMs: number;
  failed: boolean;
}

/** `GET /api/sessions/:id/timeline?turn=<n>`: one turn as lanes of spans. Metadata only, never text. */
export interface SessionTimelineResponse {
  turn: number;
  /** Finished turns in the transcript; an open turn is `turns + 1`. */
  turns: number;
  open: boolean;
  /** The session's live verdict when this body was built (the rule of SessionRow.live). */
  live: boolean;
  stopReason: string | null;
  /** Worker clock when this body was built, epoch ms. */
  asOfMs: number;
  /**
   * Epoch ms. null = the turn has no usable time; lanes, spans and handoffs are then empty.
   * endMs >= turnStartMs >= startMs always. leadMs: how far the prompt lies before the start the agent measured (0 = not
   * at all).
   */
  range: {
    /** Axis origin. Earlier than turnStartMs only when a background event started the turn and subagents ran before it. */
    startMs: number;
    /** Where the turn itself starts: the prompt or the start the agent measured, whichever is earlier. */
    turnStartMs: number;
    endMs: number | null;
    lastEventMs: number;
    /** Refers to turnStartMs. */
    startExact: boolean;
    leadMs: number;
  } | null;
  lanes: TimelineLane[];
  /** Ordered by lane order, then start. */
  spans: TimelineSpan[];
  handoffs: TimelineHandoff[];
  notes: { untimedEvents: number; foldedSubagents: number; droppedSpans: number; unplacedRetries: number };
  /** Same text as SessionDetailResponse.transcriptError; range is then null. */
  transcriptError?: string;
}
