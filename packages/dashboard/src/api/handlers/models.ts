import { catalogWithState } from '../../catalog.js';
import { discoverLocalModels, deleteLocalModel } from '../../models.js';
import type { ApiPaths, ApiRequest, MainApiContext } from '../context.js';
import { ApiError } from '../errors.js';

export function handleModels(ctx: ApiPaths): unknown {
  const { models, warnings } = discoverLocalModels(ctx.modelsDir);
  // `dir` lets the UI show WHERE these checkpoints live — the directory is
  // configurable (`--models-dir`), so the count alone is ambiguous.
  return { models, warnings, dir: ctx.modelsDir };
}

export function handleDeleteModel(ctx: ApiPaths, req: ApiRequest): unknown {
  try {
    deleteLocalModel(ctx.modelsDir, req.params.name);
    return { deleted: true, name: req.params.name };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // The store surfaces "not found" only through the message text; keep sniffing
    // it verbatim so an unknown name stays a 404 and every other failure a 400.
    throw /not found/i.test(message) ? ApiError.notFound(message) : ApiError.badRequest(message);
  }
}

export function handleCatalog(ctx: ApiPaths): unknown {
  return { items: catalogWithState(ctx.modelsDir) };
}

/**
 * Upstream commit sha for every visible catalog repo — the remote half of the
 * staleness check. The comparison against the local marker is the caller's:
 * `/api/catalog` already carries `installed` and `localRevision`.
 *
 * Split from `/api/catalog` on purpose. That route is synchronous,
 * filesystem-only and offline-safe, and the Models page cannot render without
 * it; this one dials Hugging Face and is allowed to fail. Keeping them apart
 * means a flaky network can never delay or break the install state.
 *
 * Lives on the main thread because that is where the network is (the worker is
 * SQLite plus synchronous FS walks) and because `DownloadManager` already owns
 * the sha resolver and its fetch seam. Touches no filesystem for the same
 * reason: this thread emits download progress and receives cancels, and a
 * `catalogWithState` walk of the models directory blocks both.
 */
export async function handleCatalogUpdates(ctx: MainApiContext): Promise<unknown> {
  const remote = await ctx.downloads.checkCatalogUpdates();
  return { items: [...remote].map(([hfRepo, remoteRevision]) => ({ hfRepo, remoteRevision })) };
}
