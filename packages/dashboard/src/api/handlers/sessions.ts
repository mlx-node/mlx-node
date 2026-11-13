import { type AgentSessionsReader, DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT } from '../../agent-sessions/reader.js';
import { instantKey } from '../../agent-sessions/summary.js';
import type {
  SessionDetailResponse,
  SessionListQuery,
  SessionListResponse,
  SessionRow,
} from '../../agent-sessions/types.js';
import {
  assertSessionIdShape,
  deleteSession,
  renameSession,
  SessionWriteError,
  validateRenameTitle,
} from '../../agent-sessions/writes.js';
import { workIds } from '../../chat/index-file.js';
import type { ApiRequest, WorkerApiContext } from '../context.js';
import { queryInt, requireBody } from '../context.js';
import { ApiError } from '../errors.js';

function toApiError(err: unknown): unknown {
  if (!(err instanceof SessionWriteError)) return err;
  switch (err.kind) {
    case 'bad-request':
      return ApiError.badRequest(err.message);
    case 'not-found':
      return ApiError.notFound(err.message);
    case 'conflict':
      return ApiError.conflict(err.message);
  }
}

function writing<T>(run: () => T): T {
  try {
    return run();
  } catch (err) {
    throw toApiError(err);
  }
}

function newestFirst(a: SessionRow, b: SessionRow): number {
  const d = instantKey(b.lastActiveAt) - instantKey(a.lastActiveAt);
  if (d !== 0) return d;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/** Rows of the user's agent home that the Work index names are Work conversations. */
function tagWork(rows: SessionRow[], work: Set<string>): SessionRow[] {
  return rows.map((row) => (work.has(row.id) ? { ...row, origin: 'work' } : row));
}

/** The first `n` rows of a reader's list, read in pages of the reader's maximum. */
function prefix(reader: AgentSessionsReader, q: SessionListQuery, n: number): SessionListResponse {
  const sessions: SessionRow[] = [];
  let page = reader.list({ ...q, limit: MAX_LIST_LIMIT, offset: 0 });
  const first = page;
  while (true) {
    sessions.push(...page.sessions);
    if (sessions.length >= n || page.sessions.length < MAX_LIST_LIMIT) break;
    page = reader.list({ ...q, limit: MAX_LIST_LIMIT, offset: sessions.length });
  }
  return { sessions: sessions.slice(0, n), total: first.total, cwds: first.cwds };
}

/** Both agent homes, newest first: the user's (terminal and Work sessions) and the Chat agent's. */
export function handleSessionsList(ctx: WorkerApiContext, req: ApiRequest): SessionListResponse {
  const rawLimit = queryInt(req.query, 'limit');
  const rawOffset = queryInt(req.query, 'offset');
  const limit = Math.min(MAX_LIST_LIMIT, Math.max(1, Math.trunc(rawLimit ?? DEFAULT_LIST_LIMIT)));
  const offset = Math.max(0, Math.trunc(rawOffset ?? 0));
  const q: SessionListQuery = { q: req.query.get('q') ?? undefined, cwd: req.query.get('cwd') ?? undefined };
  const main = prefix(ctx.agentSessions, q, offset + limit);
  const chat = prefix(ctx.chatSessions, q, offset + limit);
  const rows = [...tagWork(main.sessions, workIds(ctx.chatIndex())), ...chat.sessions].sort(newestFirst);
  return {
    sessions: rows.slice(offset, offset + limit),
    total: main.total + chat.total,
    cwds: [...new Set([...main.cwds, ...chat.cwds])].sort(),
  };
}

/** The user's agent home first, then the Chat agent home. */
export function handleSessionDetail(ctx: WorkerApiContext, req: ApiRequest): SessionDetailResponse {
  const main = ctx.agentSessions.detail(req.params.id);
  if (main !== null) {
    if (!workIds(ctx.chatIndex()).has(main.session.id)) return main;
    return { ...main, session: { ...main.session, origin: 'work' } };
  }
  const chat = ctx.chatSessions.detail(req.params.id);
  if (chat === null) throw ApiError.notFound(`Session ${req.params.id} not found`);
  return chat;
}

/** `?turn=<n>`: one turn as an agent timeline. Metadata only. */
export function handleSessionTimeline(ctx: WorkerApiContext, req: ApiRequest): unknown {
  const turn = queryInt(req.query, 'turn');
  if (turn === null || !Number.isSafeInteger(turn) || turn < 1) {
    throw ApiError.badRequest('turn must be a positive integer');
  }
  const timeline = ctx.agentSessions.timeline(req.params.id, turn) ?? ctx.chatSessions.timeline(req.params.id, turn);
  if (timeline === null) throw ApiError.notFound(`Session ${req.params.id} not found`);
  if (timeline === 'no-turn') throw ApiError.notFound(`Turn ${String(turn)} not found in session ${req.params.id}`);
  return timeline;
}

export function handleSessionRename(ctx: WorkerApiContext, req: ApiRequest): unknown {
  const id = req.params.id;
  return writing(() => {
    assertSessionIdShape(id);
    // An unknown id is a 404 even when the body is also bad.
    if (!ctx.agentSessions.scan().sessions.has(id)) throw new SessionWriteError('not-found', `Session ${id} not found`);
    const body = requireBody(req);
    const raw = typeof body === 'object' && body !== null ? (body as { title?: unknown }).title : undefined;
    return renameSession(ctx.agentSessions, id, validateRenameTitle(raw));
  });
}

export function handleSessionDelete(ctx: WorkerApiContext, req: ApiRequest): unknown {
  return writing(() => deleteSession(ctx.agentSessions, req.params.id));
}
