import { DownloadsClosedError } from '../../download.js';
import type { ApiPaths, ApiRequest, MainApiContext } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

export function handleDownloadsList(ctx: MainApiContext): unknown {
  return { jobs: ctx.downloads.jobs() };
}

export function handleDownloadStart(ctx: MainApiContext, req: ApiRequest): unknown {
  const body = requireBody(req);
  const repo = (body as { repo?: unknown } | null)?.repo;
  if (typeof repo !== 'string' || repo === '') {
    throw ApiError.badRequest('Body must include a "repo" string');
  }
  try {
    const id = ctx.downloads.start(repo);
    // Accepted, not done: the route declares `successStatus: 202` so the HTTP
    // adapter still answers 202 for a job that is only queued.
    return { id, repo };
  } catch (err) {
    // The manager is draining for shutdown: the request is well-formed, the
    // server is simply going away, so answer 503 rather than blaming the caller.
    if (err instanceof DownloadsClosedError) throw ApiError.unavailable(err.message);
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
