/**
 * System One (Clairic + CLEF) Control Panel routes, owned by the transport (main) thread.
 */

import type { SystemOneRunInput, SystemOneService } from '../../systemone.js';
import type { ApiRequest, MainApiContext } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

function service(ctx: MainApiContext): SystemOneService {
  if (ctx.systemone === undefined) {
    throw ApiError.unavailable('System One service is not available.');
  }
  return ctx.systemone;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function handleSystemOneState(ctx: MainApiContext): Promise<unknown> {
  return service(ctx).state();
}

export function handleSystemOneRun(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  const body = requireBody(req);
  if (!isPlainObject(body)) {
    throw ApiError.badRequest('Request body must be a JSON object');
  }
  if (typeof body.model !== 'string' || body.model.trim() === '') {
    throw ApiError.badRequest('model is required');
  }
  if (!isPlainObject(body.payload)) {
    throw ApiError.badRequest('payload must be a JSON object');
  }
  if (body.mode !== undefined && body.mode !== 'extract' && body.mode !== 'decide') {
    throw ApiError.badRequest('mode must be "extract" or "decide"');
  }
  if (body.skipHistory !== undefined && typeof body.skipHistory !== 'boolean') {
    throw ApiError.badRequest('skipHistory must be a boolean');
  }
  const input: SystemOneRunInput = {
    model: body.model,
    payload: body.payload,
    ...(body.mode !== undefined ? { mode: body.mode } : {}),
    ...(body.skipHistory === true ? { skipHistory: true } : {}),
  };
  return service(ctx).run(input);
}

export function handleSystemOneJobGet(ctx: MainApiContext, req: ApiRequest): unknown {
  const id = req.params.id?.trim() ?? '';
  if (id === '') throw ApiError.badRequest('job id is required');
  return service(ctx).job(id);
}

export function handleSystemOneJobCancel(ctx: MainApiContext, req: ApiRequest): unknown {
  const id = req.params.id?.trim() ?? '';
  if (id === '') throw ApiError.badRequest('job id is required');
  return service(ctx).cancel(id);
}

export function handleSystemOneHistoryDelete(ctx: MainApiContext): Promise<unknown> {
  return service(ctx).clearHistory();
}

export function handleSystemOneUnload(ctx: MainApiContext): Promise<unknown> {
  return service(ctx).unload();
}
