import {
  assertSessionIdShape,
  deleteSession,
  renameSession,
  SessionWriteError,
  validateRenameTitle,
} from '../../agent-sessions/writes.js';
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

export function handleSessionsList(ctx: WorkerApiContext, req: ApiRequest): unknown {
  return ctx.agentSessions.list({
    q: req.query.get('q') ?? undefined,
    cwd: req.query.get('cwd') ?? undefined,
    limit: queryInt(req.query, 'limit') ?? undefined,
    offset: queryInt(req.query, 'offset') ?? undefined,
  });
}

export function handleSessionDetail(ctx: WorkerApiContext, req: ApiRequest): unknown {
  const detail = ctx.agentSessions.detail(req.params.id);
  if (detail === null) throw ApiError.notFound(`Session ${req.params.id} not found`);
  return detail;
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
