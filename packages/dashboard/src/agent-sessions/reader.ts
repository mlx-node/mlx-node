/**
 * Read-only view of grok-build (`mlx agent`) sessions under `<agentHome>/sessions`.
 *
 * Every call re-walks the folders (cheap: readdir + lstat) and reuses parsed
 * files whose (mtimeMs, size) did not change. Synchronous by design: it runs on
 * the dashboard's database worker, where blocking is expected.
 */

import { type Dirent, lstatSync, readdirSync, readFileSync, type Stats } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

import {
  type ActiveRegistry,
  isPidAlive,
  type LiveVerdict,
  NOT_LIVE,
  ownLiveness,
  readActiveRegistry,
} from './liveness.js';
import {
  displayTitle,
  type GrokSummary,
  instantKey,
  isHeadless,
  isHidden,
  isUnusedHusk,
  lastActive,
  parseSubagentMeta,
  parseSummary,
  type SubagentMeta,
} from './summary.js';
import { type ParsedUpdates, parseUpdates } from './transcript.js';
import type {
  SessionDetail,
  SessionDetailResponse,
  SessionListQuery,
  SessionListResponse,
  SessionRow,
  SessionTokens,
  SubagentRef,
} from './types.js';

/** Larger `updates.jsonl` files are not parsed (typical is ≤ 3 MB). */
export const MAX_UPDATES_BYTES = 64 * 1024 * 1024;
export const DEFAULT_LIST_LIMIT = 100;
export const MAX_LIST_LIMIT = 500;
const TRANSCRIPT_CACHE_ENTRIES = 4;

export interface SessionRef {
  id: string;
  /** `<sessionsRoot>/<encodedCwd>/<id>`, a real directory (not a symlink). */
  dir: string;
  summary: GrokSummary;
}

export interface ChildLink {
  parentId: string;
  childId: string;
  meta: SubagentMeta;
}

export interface SessionIndex {
  sessions: Map<string, SessionRef>;
  /** Ids present under more than one cwd folder; grok skips these and so do we. */
  ambiguous: Set<string>;
  /** child id → link, from `<parent>/subagents/<child>/meta.json`. */
  parentOf: Map<string, ChildLink>;
  childrenOf: Map<string, ChildLink[]>;
}

interface Stamp {
  mtimeMs: number;
  size: number;
}

interface Cached<T> extends Stamp {
  value: T;
}

interface UpdateStats {
  tokens: SessionTokens | null;
  openTurn: boolean;
}

type UpdatesResult = { ok: true; parsed: ParsedUpdates } | { ok: false; error: string };

const EMPTY_PARSED: ParsedUpdates = { transcript: [], turns: [], turnsWithUsage: 0, openTurn: false };

function fresh(entry: Stamp | undefined, st: Stats): boolean {
  return entry !== undefined && entry.mtimeMs === st.mtimeMs && entry.size === st.size;
}

function readDirents(dir: string): Dirent[] {
  try {
    return readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

function regularFile(path: string): Stats | null {
  try {
    const st = lstatSync(path);
    return st.isFile() ? st : null;
  } catch {
    return null;
  }
}

/** POSIX single-quote: no expansion of any kind; `'` becomes `'\''`. */
export function shQuote(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
}

/**
 * The command that reopens a session: `cd '<cwd>' && mlx agent --resume <id>`,
 * with `GROK_HOME='<home>'` when the home is not the default `~/.mlx-agent`.
 */
export function resumeCommand(
  cwd: string,
  id: string,
  agentHome: string,
  defaultAgentHome: string = join(homedir(), '.mlx-agent'),
): string {
  const env = resolve(agentHome) === resolve(defaultAgentHome) ? '' : `GROK_HOME=${shQuote(agentHome)} `;
  const arg = /^[A-Za-z0-9._-]+$/.test(id) ? id : shQuote(id);
  return `cd ${shQuote(cwd)} && ${env}mlx agent --resume ${arg}`;
}

export interface AgentSessionsReaderOptions {
  agentHome: string;
  /** Test seam; defaults to `~/.mlx-agent`. */
  defaultAgentHome?: string;
  /** Test seam for the liveness clock. */
  now?: () => number;
  /** Test seam for the registry pid probe. */
  isPidAlive?: (pid: number) => boolean;
}

export class AgentSessionsReader {
  readonly agentHome: string;
  readonly sessionsRoot: string;
  private readonly defaultAgentHome: string | undefined;
  private readonly now: () => number;
  private readonly alive: (pid: number) => boolean;
  private readonly summaries = new Map<string, Cached<GrokSummary | null>>();
  private readonly metas = new Map<string, Cached<SubagentMeta | null>>();
  private readonly stats = new Map<string, Cached<UpdateStats>>();
  private readonly transcripts = new Map<string, Cached<ParsedUpdates>>();

  constructor(options: AgentSessionsReaderOptions) {
    this.agentHome = options.agentHome;
    this.sessionsRoot = join(options.agentHome, 'sessions');
    this.defaultAgentHome = options.defaultAgentHome;
    this.now = options.now ?? Date.now;
    this.alive = options.isPidAlive ?? isPidAlive;
  }

  /** Walk `<sessionsRoot>/<encodedCwd>/<id>/summary.json` like grok's `RelocationView`. */
  scan(): SessionIndex {
    const candidates = new Map<string, { dir: string; summaryPath: string; st: Stats }[]>();
    let cwdEntries: Dirent[];
    try {
      cwdEntries = readdirSync(this.sessionsRoot, { withFileTypes: true });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
      cwdEntries = [];
    }
    for (const cwdEntry of cwdEntries) {
      if (!cwdEntry.isDirectory()) continue;
      const cwdDir = join(this.sessionsRoot, cwdEntry.name);
      for (const entry of readDirents(cwdDir)) {
        if (entry.name.startsWith('.') || !entry.isDirectory()) continue;
        const dir = join(cwdDir, entry.name);
        const summaryPath = join(dir, 'summary.json');
        const st = regularFile(summaryPath);
        if (st === null) continue;
        const list = candidates.get(entry.name);
        if (list === undefined) candidates.set(entry.name, [{ dir, summaryPath, st }]);
        else list.push({ dir, summaryPath, st });
      }
    }

    const sessions = new Map<string, SessionRef>();
    const ambiguous = new Set<string>();
    const seenSummaries = new Set<string>();
    for (const [id, list] of [...candidates].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
      if (list.length !== 1) {
        ambiguous.add(id);
        continue;
      }
      const { dir, summaryPath, st } = list[0]!;
      seenSummaries.add(summaryPath);
      const summary = this.cachedRead(this.summaries, summaryPath, st, parseSummary);
      if (summary === null || summary.id !== id) continue;
      sessions.set(id, { id, dir, summary });
    }

    const parentOf = new Map<string, ChildLink>();
    const childrenOf = new Map<string, ChildLink[]>();
    const seenMetas = new Set<string>();
    for (const ref of sessions.values()) {
      const subagentsDir = join(ref.dir, 'subagents');
      for (const entry of readDirents(subagentsDir)) {
        if (entry.name.startsWith('.') || !entry.isDirectory()) continue;
        const metaPath = join(subagentsDir, entry.name, 'meta.json');
        const st = regularFile(metaPath);
        if (st === null) continue;
        seenMetas.add(metaPath);
        const meta = this.cachedRead(this.metas, metaPath, st, parseSubagentMeta);
        // grok checks the meta's parent before following it; the dir name is the child id.
        if (meta === null || meta.parentSessionId !== ref.id || meta.childSessionId !== entry.name) continue;
        if (meta.childSessionId === ref.id || parentOf.has(meta.childSessionId)) continue;
        const link: ChildLink = { parentId: ref.id, childId: meta.childSessionId, meta };
        parentOf.set(link.childId, link);
        const siblings = childrenOf.get(ref.id);
        if (siblings === undefined) childrenOf.set(ref.id, [link]);
        else siblings.push(link);
      }
    }

    for (const key of this.summaries.keys()) if (!seenSummaries.has(key)) this.summaries.delete(key);
    for (const key of this.metas.keys()) if (!seenMetas.has(key)) this.metas.delete(key);
    const liveUpdates = new Set([...sessions.values()].map((r) => join(r.dir, 'updates.jsonl')));
    for (const key of this.stats.keys()) if (!liveUpdates.has(key)) this.stats.delete(key);
    for (const key of this.transcripts.keys()) if (!liveUpdates.has(key)) this.transcripts.delete(key);

    return { sessions, ambiguous, parentOf, childrenOf };
  }

  list(query: SessionListQuery = {}): SessionListResponse {
    const index = this.scan();
    const top = [...index.sessions.values()].filter(
      (r) => !isHidden(r.summary) && !isHeadless(r.summary) && !isUnusedHusk(r.summary) && !index.parentOf.has(r.id),
    );
    const cwds = [...new Set(top.map((r) => r.summary.cwd))].sort();
    const needle = query.q?.trim().toLowerCase() ?? '';
    const filtered = top.filter((r) => {
      if (query.cwd !== undefined && query.cwd !== '' && r.summary.cwd !== query.cwd) return false;
      if (needle === '') return true;
      return displayTitle(r.summary).toLowerCase().includes(needle) || r.summary.cwd.toLowerCase().includes(needle);
    });
    filtered.sort((a, b) => {
      const d = instantKey(lastActive(b.summary)) - instantKey(lastActive(a.summary));
      if (d !== 0) return d;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
    const limit = Math.min(
      MAX_LIST_LIMIT,
      Math.max(1, Math.trunc(Number.isFinite(query.limit) ? query.limit! : DEFAULT_LIST_LIMIT)),
    );
    const offset = Math.max(0, Math.trunc(Number.isFinite(query.offset) ? query.offset! : 0));
    const registry = readActiveRegistry(this.agentHome, this.alive);
    const now = this.now();
    return {
      sessions: filtered.slice(offset, offset + limit).map((r) => this.row(r, index, registry, now)),
      total: filtered.length,
      cwds,
    };
  }

  /** Any indexed session, hidden subagent children included. `null` = unknown or ambiguous id. */
  detail(id: string): SessionDetailResponse | null {
    const index = this.scan();
    const ref = index.sessions.get(id);
    if (ref === undefined) return null;
    const registry = readActiveRegistry(this.agentHome, this.alive);
    const row = this.row(ref, index, registry, this.now());
    const s = ref.summary;
    const parentRef = row.parentId === null ? undefined : index.sessions.get(row.parentId);
    const subagents: SubagentRef[] = [...(index.childrenOf.get(id) ?? [])]
      .sort((a, b) => {
        const d = instantKey(a.meta.startedAt ?? '') - instantKey(b.meta.startedAt ?? '');
        return d !== 0 ? d : a.childId < b.childId ? -1 : a.childId > b.childId ? 1 : 0;
      })
      .map((link) => {
        const child = index.sessions.get(link.childId);
        const childTitle = child === undefined ? '' : displayTitle(child.summary).trim();
        return {
          id: link.childId,
          title: childTitle !== '' ? childTitle : link.meta.description,
          type: link.meta.subagentType,
          description: link.meta.description,
          status: link.meta.status,
          model: link.meta.effectiveModelId,
          exists: child !== undefined,
        };
      });
    const session: SessionDetail = {
      ...row,
      gitRoot: s.gitRootDir,
      headCommit: s.headCommit,
      remotes: s.gitRemotes,
      agentName: s.agentName,
      reasoningEffort: s.reasoningEffort,
      lastTurnSummary: s.lastTurnSummary,
      parent:
        row.parentId === null
          ? null
          : { id: row.parentId, title: parentRef === undefined ? null : displayTitle(parentRef.summary).trim() },
      subagents,
    };
    const updates = this.updates(ref, true);
    return {
      session,
      transcript: updates.ok ? updates.parsed.transcript : [],
      turns: updates.ok ? updates.parsed.turns : [],
      ...(updates.ok ? {} : { transcriptError: updates.error }),
      resumeCommand: resumeCommand(s.cwd, id, this.agentHome, this.defaultAgentHome),
    };
  }

  /**
   * Live verdict for `id` and its subagent ancestors (children run inside the
   * parent's process and never register). `failClosed` turns an unreadable
   * registry or session dir into "live", for write guards.
   */
  liveness(id: string, failClosed: boolean, index: SessionIndex = this.scan()): LiveVerdict {
    return this.livenessIn(index, id, readActiveRegistry(this.agentHome, this.alive), this.now(), failClosed);
  }

  private livenessIn(
    index: SessionIndex,
    id: string,
    registry: ActiveRegistry,
    now: number,
    failClosed: boolean,
  ): LiveVerdict {
    const visited = new Set<string>();
    let cur: string | undefined = id;
    while (cur !== undefined && !visited.has(cur)) {
      visited.add(cur);
      const ref = index.sessions.get(cur);
      if (ref !== undefined) {
        const verdict = ownLiveness(
          { id: cur, dir: ref.dir, openTurn: this.updateStats(ref).openTurn },
          registry,
          now,
          failClosed,
        );
        if (verdict.live) return verdict;
      } else if (registry.ok && registry.ids.has(cur)) {
        return { live: true, reason: 'registered', sessionId: cur };
      }
      cur = index.parentOf.get(cur)?.parentId;
    }
    return NOT_LIVE;
  }

  private row(ref: SessionRef, index: SessionIndex, registry: ActiveRegistry, now: number): SessionRow {
    const s = ref.summary;
    return {
      id: ref.id,
      title: displayTitle(s).trim(),
      cwd: s.cwd,
      model: s.currentModelId,
      branch: s.headBranch,
      createdAt: s.createdAt,
      lastActiveAt: lastActive(s),
      messageCount: s.numMessages,
      live: this.livenessIn(index, ref.id, registry, now, false).live,
      kind: s.sessionKind,
      parentId: parentIdOf(ref, index),
      subagentCount: index.childrenOf.get(ref.id)?.length ?? 0,
      tokens: this.updateStats(ref).tokens,
    };
  }

  private cachedRead<T>(
    cache: Map<string, Cached<T | null>>,
    path: string,
    st: Stats,
    parse: (text: string) => T | null,
  ): T | null {
    const hit = cache.get(path);
    if (fresh(hit, st)) return hit!.value;
    let value: T | null;
    try {
      value = parse(readFileSync(path, 'utf8'));
    } catch {
      value = null;
    }
    cache.set(path, { mtimeMs: st.mtimeMs, size: st.size, value });
    return value;
  }

  private updateStats(ref: SessionRef): UpdateStats {
    const path = join(ref.dir, 'updates.jsonl');
    const st = statUpdates(path);
    if (st !== null && !(st instanceof Error)) {
      const hit = this.stats.get(path);
      if (fresh(hit, st)) return hit!.value;
    }
    const result = this.updates(ref, false);
    return result.ok ? statsOf(result.parsed) : { tokens: null, openTurn: false };
  }

  /** Parse `updates.jsonl`, refreshing the stats cache; `keep` also retains the transcript. */
  private updates(ref: SessionRef, keep: boolean): UpdatesResult {
    const path = join(ref.dir, 'updates.jsonl');
    const st = statUpdates(path);
    if (st === null) return { ok: true, parsed: EMPTY_PARSED };
    if (st instanceof Error) return { ok: false, error: st.message };
    const hit = this.transcripts.get(path);
    if (fresh(hit, st)) {
      this.transcripts.delete(path);
      this.transcripts.set(path, hit!);
      if (!fresh(this.stats.get(path), st)) {
        this.stats.set(path, { mtimeMs: st.mtimeMs, size: st.size, value: statsOf(hit!.value) });
      }
      return { ok: true, parsed: hit!.value };
    }
    let text: string;
    try {
      text = readFileSync(path, 'utf8');
    } catch (err) {
      return { ok: false, error: `cannot read updates.jsonl: ${(err as Error).message}` };
    }
    const parsed = parseUpdates(text);
    this.stats.set(path, { mtimeMs: st.mtimeMs, size: st.size, value: statsOf(parsed) });
    if (keep) {
      this.transcripts.set(path, { mtimeMs: st.mtimeMs, size: st.size, value: parsed });
      while (this.transcripts.size > TRANSCRIPT_CACHE_ENTRIES) {
        this.transcripts.delete(this.transcripts.keys().next().value!);
      }
    }
    return { ok: true, parsed };
  }
}

/** `null` = no file; an `Error` = present but unusable. */
function statUpdates(path: string): Stats | Error | null {
  let st: Stats;
  try {
    st = lstatSync(path);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null;
    return new Error(`cannot stat updates.jsonl: ${(err as Error).message}`);
  }
  if (!st.isFile()) return new Error('updates.jsonl is not a regular file');
  if (st.size > MAX_UPDATES_BYTES) {
    return new Error(`updates.jsonl is too large to display (${Math.round(st.size / 1_048_576)} MB)`);
  }
  return st;
}

function statsOf(parsed: ParsedUpdates): UpdateStats {
  if (parsed.turnsWithUsage === 0) return { tokens: null, openTurn: parsed.openTurn };
  const tokens: SessionTokens = { input: 0, output: 0, cached: 0 };
  for (const t of parsed.turns) {
    tokens.input += t.inputTokens;
    tokens.output += t.outputTokens;
    tokens.cached += t.cachedTokens;
  }
  return { tokens, openTurn: parsed.openTurn };
}

/**
 * Subagent parent from the parent's `meta.json`; else the summary's
 * `parent_session_id`, except for `subagent_resume`, where it names the resumed
 * SOURCE child rather than a parent.
 */
function parentIdOf(ref: SessionRef, index: SessionIndex): string | null {
  const link = index.parentOf.get(ref.id);
  if (link !== undefined) return link.parentId;
  if (ref.summary.sessionKind === 'subagent_resume') return null;
  return ref.summary.parentSessionId;
}
