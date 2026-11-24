import type { HostResponse } from './types';
import { type AsyncState, useJson } from './use-api';

/**
 * This Mac's memory and free disk. A transport-thread route, so it never waits behind the worker's walk of the models
 * folder; a page that filters the catalog by memory treats it as loading until it settles.
 */
export function useHost(): AsyncState<HostResponse> {
  return useJson<HostResponse>('/host');
}
