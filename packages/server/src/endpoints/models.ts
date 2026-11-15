/** GET /v1/models endpoint. */

import type { ServerResponse } from 'node:http';

import { sendError } from '../errors.js';
import type { PublicModelEntry } from '../handler.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import type { ModelWorkCoordinator } from '../model-work-coordinator.js';
import type { ModelRegistry } from '../registry.js';
import { stringifyStrictJson } from '../strict-json.js';

export function handleListModels(
  res: ServerResponse,
  registry: ModelRegistry,
  listModels?: () => PublicModelEntry[],
): void {
  const models = listModels ? listModels() : registry.list();
  const body = {
    object: 'list',
    data: models,
    models: models.map((model) => ({ name: model.id, description: 'Local MLX model', release_date: '' })),
  };
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(stringifyStrictJson(body));
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
