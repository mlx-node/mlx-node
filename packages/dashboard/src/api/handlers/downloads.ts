import { isAbsolute } from 'node:path';

import {
  DownloadConflictError,
  DownloadsClosedError,
  HubUnreachableError,
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

/** A plan token of the body (`adopt`, `locate`): absent, or 64 lower-case hex characters. */
function parseToken(value: unknown, field: 'adopt' | 'locate'): string | undefined {
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
  if (!hex) {
    throw ApiError.badRequest(
      `"${field}" must be the 64-character token of ${field === 'adopt' ? 'an Update' : 'a Locate'} plan`,
    );
  }
  return value as string;
}

/**
 * `POST /api/downloads` with `{ slug, replaceBlocker?, adopt?, locate? }`: `slug` is the catalog folder to install; at
 * most one of the three options. Answers `{ id, slug, repo }`.
 */
export function handleDownloadStart(ctx: MainApiContext, req: ApiRequest): unknown {
  const body = requireBody(req);
  const fields = (body ?? {}) as { slug?: unknown; replaceBlocker?: unknown; adopt?: unknown; locate?: unknown };
  const slug = fields.slug;
  if (typeof slug !== 'string' || slug === '') {
    throw ApiError.badRequest('Body must include a "slug" string');
  }
  const replaceBlocker = parseReplaceBlocker(fields.replaceBlocker);
  const adopt = parseToken(fields.adopt, 'adopt');
  const locate = parseToken(fields.locate, 'locate');
  if ([replaceBlocker, adopt, locate].filter((option) => option !== undefined).length > 1) {
    throw ApiError.badRequest('"replaceBlocker", "adopt" and "locate" cannot come together');
  }
  try {
    const { id, repo } = ctx.downloads.start(
      slug,
      replaceBlocker !== undefined
        ? { replaceBlocker }
        : adopt !== undefined
          ? { adopt }
          : locate !== undefined
            ? { locate }
            : undefined,
    );
    // Accepted, not done: the route declares `successStatus: 202` so the HTTP
    // adapter still answers 202 for a job that is only queued.
    return { id, slug, repo };
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

/** The longest folder path a Locate plan takes. */
const LOCATE_PATH_MAX_CHARS = 4096;

/**
 * `POST /api/downloads/locate-plan` with `{ slug, path }`: the plan of a Locate of the catalog folder `slug` from the
 * absolute folder `path`, whose token a later start must carry. Writes nothing.
 */
export async function handleDownloadLocatePlan(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  const body = requireBody(req);
  const { slug, path } = (body ?? {}) as { slug?: unknown; path?: unknown };
  if (
    typeof slug !== 'string' ||
    slug === '' ||
    typeof path !== 'string' ||
    path.length > LOCATE_PATH_MAX_CHARS ||
    path.includes('\0') ||
    !isAbsolute(path)
  ) {
    throw ApiError.badRequest('Body must include a "slug" and an absolute "path".');
  }
  try {
    return await ctx.downloads.planLocate(slug, path);
  } catch (err) {
    if (
      err instanceof DownloadsClosedError ||
      err instanceof ModelsDirUnavailableError ||
      err instanceof HubUnreachableError
    ) {
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
