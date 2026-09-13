import type { MainApiContext, ApiRequest } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

export function handleCodingAgents(ctx: MainApiContext): unknown {
  if (!ctx.codingAgents) throw new ApiError('E_UNAVAILABLE', 'Coding agent setup is unavailable.');
  return ctx.codingAgents.state();
}

export function handleCodingAgentAction(ctx: MainApiContext, req: ApiRequest): unknown {
  if (!ctx.codingAgents) throw new ApiError('E_UNAVAILABLE', 'Coding agent setup is unavailable.');
  const body = requireBody(req) as { action?: unknown; agent?: unknown; force?: unknown } | null;
  if (
    !body ||
    (body.action !== 'detect' && body.action !== 'install' && body.action !== 'refresh') ||
    (body.agent !== undefined && typeof body.agent !== 'string') ||
    (body.force !== undefined && typeof body.force !== 'boolean') ||
    (body.action === 'install' && !body.agent)
  ) {
    throw ApiError.badRequest('Choose a coding agent and a valid setup action.');
  }
  if (body.action === 'refresh') return ctx.codingAgents.refresh();
  return ctx.codingAgents.start(body.action, body.agent as string | undefined, body.force === true);
}
