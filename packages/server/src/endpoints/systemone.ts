import { randomUUID } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';

import { validateClefRequest } from '@mlx-node/lm/clef-types';

import { DecisionQueueFullError } from '../decision-registry.js';
import { sendBadRequest, sendInternalError, sendNotFound, sendRateLimit } from '../errors.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import { ModelLoadQueueFullError, type ModelWorkCoordinator } from '../model-work-coordinator.js';
import type { ModelRegistry } from '../registry.js';

export async function handleSystemOne(
  req: IncomingMessage,
  res: ServerResponse,
  raw: string,
  registry: ModelRegistry,
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
  coordinator?: ModelWorkCoordinator,
): Promise<void> {
  res.setHeader('x-typesafe-request-id', randomUUID());
  let body: { model: string };
  try {
    const parsed: unknown = JSON.parse(raw);
    validateClefRequest(parsed);
    if (!('model' in parsed) || typeof parsed.model !== 'string' || !parsed.model.trim())
      throw new Error('model is required');
    body = parsed as typeof parsed & { model: string };
  } catch (error) {
    sendBadRequest(res, (error as Error).message);
    return;
  }
  const abort = new AbortController();
  const onClose = () => abort.abort();
  res.once('close', onClose);
  req.once('aborted', onClose);
  if (req.aborted || res.destroyed) abort.abort();
  let loadAdmission: ReturnType<ModelWorkCoordinator['beginRequestLoadAdmission']> | undefined;
  let lease: ReturnType<ModelRegistry['decisions']['acquire']>;
  idleSweeper?.beginRequest();
  try {
    abort.signal.throwIfAborted();
    if (!registry.decisions.get(body.model) && resolveModel) {
      loadAdmission = coordinator?.beginRequestLoadAdmission(body.model);
      const resolve = () => resolveModel(body.model);
      if (coordinator) await coordinator.withModelLoad(resolve);
      else await resolve();
    }
    abort.signal.throwIfAborted();
    const lane = registry.decisions.lane(body.model);
    const permit = lane ? loadAdmission?.transferToResident(lane) : undefined;
    lease = registry.decisions.acquire(body.model, permit);
    if (!lease) {
      if (registry.get(body.model)) sendBadRequest(res, `Model "${body.model}" does not support decisions`);
      else sendNotFound(res, `Model "${body.model}" not found`);
      return;
    }
    const dispatch = lease;
    const run = () =>
      dispatch.run(async () => {
        if (!dispatch.current()) throw new Error('Model binding changed while waiting; retry the request');
        const result = await dispatch.model.decideRaw(raw, { signal: abort.signal });
        abort.signal.throwIfAborted();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ model: dispatch.canonicalName, ...result }));
      }, abort.signal);
    if (coordinator) await coordinator.withInference(run);
    else await run();
  } catch (error) {
    if (abort.signal.aborted || res.destroyed) return;
    if (error instanceof DecisionQueueFullError || error instanceof ModelLoadQueueFullError)
      sendRateLimit(res, error.message);
    else if (error instanceof Error && error.message.includes('Invalid CLEF request:'))
      sendBadRequest(res, error.message);
    else sendInternalError(res, error instanceof Error ? error.message : 'CLEF inference failed');
  } finally {
    lease?.release();
    loadAdmission?.release();
    idleSweeper?.endRequest();
    res.removeListener('close', onClose);
    req.removeListener('aborted', onClose);
  }
}
