/** GET /v1/models endpoint. */

import type { ServerResponse } from 'node:http';

import type { PublicModelEntry } from '../handler.js';
import type { ModelRegistry } from '../registry.js';

export function handleListModels(
  res: ServerResponse,
  registry: ModelRegistry,
  listModels?: () => PublicModelEntry[],
): void {
  const models = listModels ? listModels() : registry.list();
  const body = {
    object: 'list',
    data: models,
  };
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}
