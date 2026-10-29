/** Wire shapes of the Sessions API, read from grok-build (`mlx agent`) session folders. */

export interface SessionTokens {
  input: number;
  output: number;
  cached: number;
}

export interface SessionRow {
  id: string;
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

export type TranscriptEntry =
  | { kind: 'user'; ts: number; text: string; promptIndex: number | null; interjection?: boolean }
  | { kind: 'assistant'; ts: number; text: string }
  | { kind: 'thinking'; ts: number; text: string }
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
    }
  | { kind: 'model'; ts: number; modelId: string }
  | { kind: 'turn-end'; ts: number; stopReason: string; elapsedMs?: number; error?: string };

export interface TurnMetric {
  promptId: string;
  ts: number;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
  elapsedMs: number | null;
  ttftMs: number | null;
  prefillTps: number | null;
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
  resumeCommand: string;
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
