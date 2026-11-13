import { DownloadConflictError, DownloadsClosedError, type ReplaceBlocker } from '../../download.js';
import type { ApiPaths, ApiRequest, MainApiContext } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

export function handleDownloadsList(ctx: MainApiContext): unknown {
  return { jobs: ctx.downloads.jobs() };
}

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

/** `replaceBlocker` of the body: absent, or the four numbers of a Reinstall. Other fields are ignored. */
function parseReplaceBlocker(value: unknown): ReplaceBlocker | undefined {
  if (value === undefined) return undefined;
  const fields = typeof value === 'object' && value !== null && !Array.isArray(value) ? value : undefined;
  const { kind, fileCount, sizeBytes, truncated } = (fields ?? {}) as Record<string, unknown>;
  if (
    fields === undefined ||
    (kind !== 'directory' && kind !== 'symlink' && kind !== 'file') ||
    !isCount(fileCount) ||
    !isCount(sizeBytes) ||
    typeof truncated !== 'boolean'
  ) {
    throw ApiError.badRequest(
      '"replaceBlocker" must be { kind: "directory" | "symlink" | "file", fileCount, sizeBytes, truncated }',
    );
  }
  return { kind, fileCount, sizeBytes, truncated };
}

export function handleDownloadStart(ctx: MainApiContext, req: ApiRequest): unknown {
  const body = requireBody(req);
  const fields = (body ?? {}) as { repo?: unknown; replaceBlocker?: unknown };
  const repo = fields.repo;
  if (typeof repo !== 'string' || repo === '') {
    throw ApiError.badRequest('Body must include a "repo" string');
  }
  const replaceBlocker = parseReplaceBlocker(fields.replaceBlocker);
  try {
    const id = ctx.downloads.start(repo, replaceBlocker === undefined ? undefined : { replaceBlocker });
    // Accepted, not done: the route declares `successStatus: 202` so the HTTP
    // adapter still answers 202 for a job that is only queued.
    return { id, repo };
  } catch (err) {
    // The manager is draining for shutdown: the request is well-formed, the
    // server is simply going away, so answer 503 rather than blaming the caller.
    if (err instanceof DownloadsClosedError) throw ApiError.unavailable(err.message);
    if (err instanceof DownloadConflictError) throw ApiError.conflict(err.message);
    throw ApiError.badRequest(err instanceof Error ? err.message : 'Failed to start download');
  }
}

/**
 * Cancel/abort or dismiss a download. A `running` job is aborted (its
 * job-private staging dir cleans up); a terminal job (`done`/`error`/`cancelled`)
 * is dismissed from the registry. Both cases return 200. This deliberately never
 * touches the shared HF blob cache (the `mlx download` CLI resumes from it). A
 * 404 means no such id, or the job is in its brief non-cancellable `committing`
 * (publish) window.
 */
export function handleDownloadCancel(ctx: MainApiContext, req: ApiRequest): unknown {
  if (ctx.downloads.cancel(req.params.id)) {
    return { cancelled: true, id: req.params.id };
  }
  throw ApiError.notFound(`No cancellable download "${req.params.id}"`);
}

/**
 * Placeholder for `GET /api/downloads/:id/events`. The event stream is owned by
 * the transport (SSE framing, heartbeats, backpressure), which intercepts the
 * route before dispatch; a transport with no streaming channel reaches this and
 * is told so rather than getting a misleading 404.
 */
export function handleDownloadEventsUnsupported(_ctx: ApiPaths, req: ApiRequest): unknown {
  throw ApiError.unavailable(`Streaming route ${req.path} is not available over this transport`);
}
