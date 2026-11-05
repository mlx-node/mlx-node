import type { MainApiContext, ApiRequest } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

export function handleCodingAgents(ctx: MainApiContext): unknown {
  if (!ctx.codingAgents) throw new ApiError('E_UNAVAILABLE', 'Coding agent setup is unavailable.');
  return ctx.codingAgents.state();
}

export function handleCodingAgentAction(ctx: MainApiContext, req: ApiRequest): unknown {
  if (!ctx.codingAgents) throw new ApiError('E_UNAVAILABLE', 'Coding agent setup is unavailable.');
  const body = requireBody(req) as { action?: unknown; agent?: unknown } | null;
  if (body?.action === 'refresh') return ctx.codingAgents.refresh();
  if (body?.action === 'install' && typeof body.agent === 'string' && body.agent)
    return ctx.codingAgents.install(body.agent);
  throw ApiError.badRequest('Choose a coding agent and a valid setup action.');
}
