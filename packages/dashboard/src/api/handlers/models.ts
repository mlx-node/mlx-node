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
 * Which installed catalog models have newer bytes upstream.
 *
 * Split from `/api/catalog` on purpose. That route is synchronous,
 * filesystem-only and offline-safe, and the Models page cannot render without
 * it; this one dials Hugging Face and is allowed to fail. Keeping them apart
 * means a flaky network can never delay or break the install state.
 *
 * Lives on the main thread because that is where the network is (the worker is
 * SQLite plus synchronous FS walks) and because `DownloadManager` already owns
 * the sha resolver and its fetch seam.
 */
export async function handleCatalogUpdates(ctx: MainApiContext): Promise<unknown> {
  const remote = await ctx.downloads.checkCatalogUpdates();
  const items = catalogWithState(ctx.modelsDir)
    .filter((item) => !item.hidden)
    .map((item) => {
      const remoteRevision = remote.get(item.hfRepo) ?? null;
      return {
        hfRepo: item.hfRepo,
        remoteRevision,
        // Gated on `installed`, never `present`: only a dashboard-owned install
        // at the canonical slug carries a revision to compare AND can actually
        // be re-installed. See `localRevision` on CatalogItem.
        updateAvailable:
          item.installed &&
          item.localRevision !== null &&
          remoteRevision !== null &&
          remoteRevision !== item.localRevision,
      };
    });
  return { items };
}
