/** Wire shapes of the Sessions API, read from grok-build (`mlx agent`) session folders. */

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
export type WakeCause = 'workflow' | 'subagent' | 'task' | 'notifications' | 'goal' | 'unknown';

export type TranscriptEntry =
  | {
      kind: 'user';
      ts: number;
      text: string;
      promptIndex: number | null;
      interjection?: boolean;
      /** The `continue` prompt right after a length end: drawn as a divider, not a message. */
      continued?: true;
    }
  /** A hidden prompt the agent wrote to itself when a background event woke it. Never a message of the user. */
  | {
      kind: 'wake';
      ts: number;
      cause: WakeCause;
      /** cause 'workflow': the run's name and status when it woke the agent. */
      workflow?: { name: string; status: string };
      /** cause 'subagent': the child whose finish woke the agent. */
      subagent?: { childId: string; description: string };
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
      tokens?: number;
      error?: string;
      /** Set by the detail route; only `true` may be linked. */
      exists?: boolean;
      /** Time of the `subagent_finished` line. */
      endTs?: number;
      /** Made by a finish line with no spawn line: `ts` is the finish time. */
      noSpawn?: true;
    }
  | { kind: 'model'; ts: number; modelId: string }
  | {
      kind: 'turn-end';
      ts: number;
      stopReason: string;
      elapsedMs?: number;
      error?: string;
      /** The turn ended on the model's length limit (the agent stores it as an error). */
      lengthLimit?: true;
    };

export interface TurnMetric {
  promptId: string;
  ts: number;
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
  | (TimelineSpanBase & { kind: 'generate'; phase: 'reasoning' | 'answer' })
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
