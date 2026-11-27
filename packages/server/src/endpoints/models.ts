/** GET /v1/models endpoint. */

import type { ServerResponse } from 'node:http';

import { sendError } from '../errors.js';
import type { PublicModelEntry } from '../handler.js';
import { reasoningEffortsFor } from '../host/thinking-levels.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import type { ModelWorkCoordinator } from '../model-work-coordinator.js';
import type { ModelRegistry } from '../registry.js';
import { stringifyStrictJson } from '../strict-json.js';

/** With `thinkingLevels`, a row that names a family with levels lists them as `reasoningEfforts`. */
export function handleListModels(
  res: ServerResponse,
  registry: ModelRegistry,
  listModels?: () => PublicModelEntry[],
  thinkingLevels = false,
): void {
  const listed = listedRows(registry, listModels);
  const models = thinkingLevels ? listed.map(withReasoningEfforts) : listed;
  const body = {
    object: 'list',
    data: models,
    models: models.map((model) => ({ name: model.id, description: 'Local MLX model', release_date: '' })),
  };
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(stringifyStrictJson(body));
}

function listedRows(
  registry: ModelRegistry,
  listModels?: () => PublicModelEntry[],
): Array<{ id: string; model_family?: string }> {
  return listModels ? listModels() : registry.list();
}

function withReasoningEfforts<T extends { model_family?: string }>(entry: T): T {
  const efforts = entry.model_family === undefined ? undefined : reasoningEffortsFor(entry.model_family);
  return efforts === undefined ? entry : { ...entry, reasoningEfforts: efforts };
}

/** The family of a name's `/v1/models` row: what a server with `thinkingLevels` maps its levels through. */
export function listedFamily(
  name: string,
  registry: ModelRegistry,
  listModels?: () => PublicModelEntry[],
): string | undefined {
  return listedRows(registry, listModels).find((entry) => entry.id === name)?.model_family;
}

/**
 * POST /v1/models/unload: drops the resident chat models and their aliases. The selection is kept, so the next
 * request loads it again. Never waits: an answer (from its dispatch lease on), a load or a queued load is a 409.
 */
export async function handleUnloadModels(
  res: ServerResponse,
  unloadModels: () => Promise<string[]>,
  registry: ModelRegistry,
  coordinator: ModelWorkCoordinator,
  idleSweeper: IdleSweeper | null | undefined,
): Promise<void> {
  if (registry.dispatchLeaseCount > 0 || coordinator.readerCount > 0 || coordinator.writerActive) {
    sendError(res, 409, 'conflict_error', 'The engine is busy: an answer or a model load is running.');
    return;
  }
  const run = (): Promise<string[]> => coordinator.withWeightRelease(unloadModels);
  // Leaving the bracket arms the idle sweeper, which clears the MLX cache after its usual delay.
  const unloaded = idleSweeper ? await idleSweeper.withSuspendedDrains(run) : await run();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(stringifyStrictJson({ unloaded }));
}
