import { catalogTargetForSlug } from '@mlx-node/agent/catalog';

import {
  DownloadConflictError,
  DownloadsClosedError,
  ModelsDirUnavailableError,
  type ReplaceBlocker,
} from '../../download.js';
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

/** `adopt` of the body: absent, or the token of an Update plan, 64 lower-case hex characters. */
function parseAdoptToken(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  let hex = typeof value === 'string' && value.length === 64;
  if (hex) {
    for (const ch of value as string) {
      if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
        hex = false;
        break;
      }
    }
  }
  if (!hex) throw ApiError.badRequest('"adopt" must be the 64-character token of an Update plan');
  return value as string;
}

/**
 * `POST /api/downloads` with `{ slug, replaceBlocker?, adopt? }`: `slug` is the catalog folder to install. Answers
 * `{ id, slug, repo }`.
 */
export function handleDownloadStart(ctx: MainApiContext, req: ApiRequest): unknown {
  const body = requireBody(req);
  const fields = (body ?? {}) as { slug?: unknown; replaceBlocker?: unknown; adopt?: unknown };
  const slug = fields.slug;
  if (typeof slug !== 'string' || slug === '') {
    throw ApiError.badRequest('Body must include a "slug" string');
  }
  const replaceBlocker = parseReplaceBlocker(fields.replaceBlocker);
  const adopt = parseAdoptToken(fields.adopt);
  if (adopt !== undefined && replaceBlocker !== undefined) {
    throw ApiError.badRequest('"adopt" and "replaceBlocker" cannot come together');
  }
  try {
    const id = ctx.downloads.start(
      slug,
      replaceBlocker !== undefined ? { replaceBlocker } : adopt !== undefined ? { adopt } : undefined,
    );
    // Accepted, not done: the route declares `successStatus: 202` so the HTTP
    // adapter still answers 202 for a job that is only queued. `start` accepted
    // the slug, so the catalog has its target.
    return { id, slug, repo: catalogTargetForSlug(slug)!.repo };
  } catch (err) {
    // The manager is draining for shutdown: the request is well-formed, the
    // server is simply going away, so answer 503 rather than blaming the caller.
    if (err instanceof DownloadsClosedError || err instanceof ModelsDirUnavailableError) {
      throw ApiError.unavailable(err.message);
    }
    if (err instanceof DownloadConflictError) throw ApiError.conflict(err.message);
    throw ApiError.badRequest(err instanceof Error ? err.message : 'Failed to start download');
  }
}

/** `POST /api/downloads/adopt-plan` with `{ slug }`: the plan of an Update of that folder, whose token a later start must carry. */
export async function handleDownloadAdoptPlan(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  const body = requireBody(req);
  const slug = ((body ?? {}) as { slug?: unknown }).slug;
  if (typeof slug !== 'string' || slug === '') {
    throw ApiError.badRequest('Body must include a "slug" string');
  }
  try {
    return await ctx.downloads.planAdoption(slug);
  } catch (err) {
    if (err instanceof DownloadsClosedError || err instanceof ModelsDirUnavailableError) {
      throw ApiError.unavailable(err.message);
    }
    if (err instanceof DownloadConflictError) throw ApiError.conflict(err.message);
    throw ApiError.badRequest(err instanceof Error ? err.message : 'Could not check the folder');
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
