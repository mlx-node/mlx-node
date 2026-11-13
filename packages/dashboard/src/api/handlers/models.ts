import { statfsSync } from 'node:fs';
import { totalmem } from 'node:os';
import { isAbsolute, relative, sep } from 'node:path';

import { discoverMlxModels } from '@mlx-node/agent/models';

import { catalogWithState } from '../../catalog.js';
import {
  discoverLocalModels,
  deleteLocalModel,
  isModelPresent,
  isRegularFile,
  type ModelsResponse,
} from '../../models.js';
import type { ApiPaths, ApiRequest, MainApiContext } from '../context.js';
import { ApiError } from '../errors.js';

/** The first path segment of `path` under `root`, or null when it is not inside it. */
function firstSegmentUnder(root: string, path: string): string | null {
  const rel = relative(root, path);
  if (rel === '' || isAbsolute(rel) || rel === '..' || rel.startsWith(`..${sep}`)) return null;
  const cut = rel.indexOf(sep);
  return cut === -1 ? rel : rel.slice(0, cut);
}

/** Served names per top-level entry of the models directory; empty when discovery fails. */
async function servedIdsByEntry(modelsDir: string): Promise<Map<string, string[]>> {
  const out = new Map<string, string[]>();
  let discovered;
  try {
    discovered = await discoverMlxModels(modelsDir);
  } catch {
    return out;
  }
  for (const { discovered: model } of discovered) {
    const entry = firstSegmentUnder(modelsDir, model.path);
    if (entry === null) continue;
    const list = out.get(entry);
    if (list === undefined) out.set(entry, [model.name]);
    else list.push(model.name);
  }
  for (const list of out.values()) list.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  return out;
}

function diskFreeBytes(dir: string): number | null {
  try {
    const fs = statfsSync(dir);
    return Number(fs.bavail) * Number(fs.bsize);
  } catch {
    return null;
  }
}

export async function handleModels(ctx: ApiPaths): Promise<ModelsResponse> {
  const { models, companions, warnings } = discoverLocalModels(ctx.modelsDir);
  const served = await servedIdsByEntry(ctx.modelsDir);
  for (const model of models) model.servedIds = served.get(model.name) ?? [];
  // `dir` lets the UI show WHERE these checkpoints live — the directory is
  // configurable (`--models-dir`), so the count alone is ambiguous.
  return {
    models,
    companions,
    warnings,
    dir: ctx.modelsDir,
    host: { totalMemBytes: totalmem(), diskFreeBytes: diskFreeBytes(ctx.modelsDir) },
  };
}

export async function handleCodingAgentModels(ctx: ApiPaths): Promise<unknown> {
  return {
    // Share the agent's IDs (including GGUF variants and collision suffixes),
    // while retaining the setup page's incomplete-checkpoint guard for directories.
    models: (await discoverMlxModels(ctx.modelsDir))
      .filter(({ discovered }) => isRegularFile(discovered.path) || isModelPresent(discovered.path))
      .map(({ discovered }) => ({ name: discovered.name })),
  };
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
