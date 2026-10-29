/**
 * Defensive readers for grok's `summary.json` and `subagents/<child>/meta.json`.
 * Field semantics mirror grok-build `session/persistence.rs` (`Summary`).
 */

export interface GrokSummary {
  id: string;
  cwd: string;
  sessionSummary: string;
  generatedTitle: string | null;
  titleIsManual: boolean;
  createdAt: string;
  updatedAt: string;
  lastActiveAt: string | null;
  numMessages: number;
  currentModelId: string | null;
  reasoningEffort: string | null;
  gitRootDir: string | null;
  gitRemotes: string[];
  headCommit: string | null;
  headBranch: string | null;
  agentName: string | null;
  parentSessionId: string | null;
  forkedAt: string | null;
  sessionKind: string | null;
  hidden: boolean | null;
  lastTurnSummary: string | null;
}

export interface SubagentMeta {
  parentSessionId: string;
  childSessionId: string;
  subagentType: string;
  description: string;
  status: string;
  startedAt: string | null;
  effectiveModelId: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function nonEmpty(value: unknown): string | null {
  return typeof value === 'string' && value !== '' ? value : null;
}

/** `null` when the file is not a summary grok could identify (no `info.id` / `info.cwd`). */
export function parseSummary(text: string): GrokSummary | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(raw) || !isRecord(raw.info)) return null;
  const id = nonEmpty(raw.info.id);
  const cwd = str(raw.info.cwd);
  if (id === null || cwd === null) return null;
  const numMessages = raw.num_messages;
  return {
    id,
    cwd,
    sessionSummary: str(raw.session_summary) ?? '',
    generatedTitle: str(raw.generated_title),
    titleIsManual: raw.title_is_manual === true,
    createdAt: str(raw.created_at) ?? '',
    updatedAt: str(raw.updated_at) ?? '',
    lastActiveAt: str(raw.last_active_at),
    numMessages: typeof numMessages === 'number' && Number.isFinite(numMessages) ? Math.max(0, numMessages) : 0,
    currentModelId: nonEmpty(raw.current_model_id),
    reasoningEffort: nonEmpty(raw.reasoning_effort),
    gitRootDir: nonEmpty(raw.git_root_dir),
    gitRemotes: Array.isArray(raw.git_remotes) ? raw.git_remotes.filter((r): r is string => typeof r === 'string') : [],
    headCommit: nonEmpty(raw.head_commit),
    headBranch: nonEmpty(raw.head_branch),
    agentName: nonEmpty(raw.agent_name),
    parentSessionId: nonEmpty(raw.parent_session_id),
    forkedAt: str(raw.forked_at),
    sessionKind: str(raw.session_kind),
    hidden: typeof raw.hidden === 'boolean' ? raw.hidden : null,
    lastTurnSummary: nonEmpty(raw.last_turn_summary),
  };
}

/** `generated_title` (trimmed) when non-blank, else `session_summary` (grok `display_title`). */
export function displayTitle(s: GrokSummary): string {
  const generated = s.generatedTitle?.trim() ?? '';
  return generated !== '' ? generated : s.sessionSummary;
}

/** grok `is_hidden`: explicit flag, else any `subagent*` kind. */
export function isHidden(s: GrokSummary): boolean {
  return s.hidden ?? s.sessionKind?.startsWith('subagent') === true;
}

export function isHeadless(s: GrokSummary): boolean {
  return s.sessionKind === 'headless';
}

/** grok `is_unused_optimistic_husk`: untitled, no messages, no fork provenance. */
export function isUnusedHusk(s: GrokSummary): boolean {
  if (s.sessionKind === 'fork' || s.parentSessionId !== null || s.forkedAt !== null) return false;
  return s.numMessages === 0 && displayTitle(s).trim() === '';
}

/**
 * Sortable instant for an RFC 3339 string, keeping grok's microseconds below the
 * millisecond so ties inside one millisecond still order like grok's `DateTime`.
 */
export function instantKey(value: string): number {
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return 0;
  const frac = /\.(\d+)/.exec(value)?.[1] ?? '';
  return frac.length > 3 ? ms + Number(`0.${frac.slice(3)}`) : ms;
}

/** grok's list order key: `last_active_at` else `updated_at`. */
export function lastActive(s: GrokSummary): string {
  return s.lastActiveAt ?? s.updatedAt;
}

export function parseSubagentMeta(text: string): SubagentMeta | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(raw)) return null;
  const parentSessionId = nonEmpty(raw.parent_session_id);
  const childSessionId = nonEmpty(raw.child_session_id);
  if (parentSessionId === null || childSessionId === null) return null;
  return {
    parentSessionId,
    childSessionId,
    subagentType: str(raw.subagent_type) ?? '',
    description: str(raw.description) ?? '',
    status: nonEmpty(raw.status) ?? 'unknown',
    startedAt: str(raw.started_at),
    effectiveModelId: nonEmpty(raw.effective_model_id),
  };
}
