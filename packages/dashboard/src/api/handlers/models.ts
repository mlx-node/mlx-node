import { catalogWithState } from '../../catalog.js';
import { discoverLocalModels, deleteLocalModel } from '../../models.js';
import type { ApiContext, ApiRequest } from '../context.js';
import { ApiError } from '../errors.js';

export function handleModels(ctx: ApiContext): unknown {
  const { models, warnings } = discoverLocalModels(ctx.modelsDir);
  // `dir` lets the UI show WHERE these checkpoints live — the directory is
  // configurable (`--models-dir`), so the count alone is ambiguous.
  return { models, warnings, dir: ctx.modelsDir };
}

export function handleDeleteModel(ctx: ApiContext, req: ApiRequest): unknown {
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

export function handleCatalog(ctx: ApiContext): unknown {
  return { items: catalogWithState(ctx.modelsDir) };
}
