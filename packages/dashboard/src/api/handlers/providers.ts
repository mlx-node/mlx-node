/**
 * The web search provider routes. Bodies are checked against their exact field lists, and every handler runs under
 * the error text law: Keychain, `fetch` and `fs` errors can quote a key or a path, so none of their text reaches a
 * reply. The key appears in exactly one payload, the body of `PUT /api/providers/:id/key`, and in no reply.
 */

import type { ProvidersService } from '../../chat/providers/service.js';
import { guarded } from '../../chat/route-errors.js';
import { noBody, providerIdParam, providerKeyInput, providerSelectionInput } from '../../chat/route-input.js';
import type { ApiRequest, MainApiContext } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

function service(ctx: MainApiContext): ProvidersService {
  if (ctx.providers === undefined) throw new ApiError('E_UNAVAILABLE', 'The Keychain cannot be reached.');
  return ctx.providers;
}

export function handleProvidersList(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('provider-key', () => {
    noBody(req.body);
    return service(ctx).list();
  });
}

export function handleProviderKeyPut(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('provider-key', () => {
    const id = providerIdParam(req.params.id);
    return service(ctx).putKey(id, providerKeyInput(requireBody(req)).key);
  });
}

export function handleProviderKeyDelete(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('provider-key', () => {
    const id = providerIdParam(req.params.id);
    noBody(req.body);
    return service(ctx).deleteKey(id);
  });
}

export function handleProviderValidate(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('provider-validate', () => {
    const id = providerIdParam(req.params.id);
    noBody(req.body);
    return service(ctx).validate(id);
  });
}

export function handleProviderSelection(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('provider-selection', () => service(ctx).select(providerSelectionInput(requireBody(req))));
}
