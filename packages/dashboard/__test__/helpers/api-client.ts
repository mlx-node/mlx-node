/**
 * A `fetch`-shaped client over anything that answers an {@link ApiCall}.
 *
 * The API suite drives every assertion through `call`, so its assertions test API
 * behaviour rather than transport plumbing. The response is deliberately JSON
 * round-tripped, so every value a test sees is byte-for-byte what a transport
 * hands it.
 *
 * The parameter is the CALL SURFACE, not the runtime: `DashboardRuntime` and the
 * RPC client expose the same `call(ApiCall) => Promise<ApiResponse>` on purpose,
 * so the identical suite can be pointed at either and a transport that quietly
 * mangles an envelope shows up as an API-behaviour failure.
 */

import { parseJsonBody } from '../../src/api/context.js';
import type { ApiResponse } from '../../src/api/errors.js';
import type { ApiCall } from '../../src/runtime.js';

/** Anything that answers an API call: the runtime, or an RPC client over a port. */
export interface ApiCaller {
  call(call: ApiCall): Promise<ApiResponse>;
}

export interface TestResponse {
  status: number;
  json(): Promise<unknown>;
}

export interface TestRequestInit {
  method?: string;
  /** Accepted and ignored: a call has no headers. Kept so call sites are unchanged. */
  headers?: Record<string, string>;
  body?: string;
  /** Send a body the transport could not parse, the way a real bad payload arrives. */
  rawBody?: string;
}

export interface TestClient {
  fetch(path: string, init?: TestRequestInit): Promise<TestResponse>;
}

export function createTestClient(caller: ApiCaller): TestClient {
  return {
    async fetch(path: string, init?: TestRequestInit): Promise<TestResponse> {
      // `rawBody` goes through the same `parseJsonBody` a transport uses, so an
      // unparseable payload reaches the handler as `bodyError` data rather than
      // throwing at read time — which is what keeps a handler's own 404 ahead of
      // a body-parse 400.
      const source = init?.rawBody ?? init?.body;
      const parsed = source === undefined ? { body: null } : parseJsonBody(source);
      if (init?.rawBody === undefined && !('body' in parsed)) {
        throw new Error(`test client sent an unparseable body: ${(parsed as { bodyError: string }).bodyError}`);
      }
      const response = await caller.call({ method: init?.method ?? 'GET', path, ...parsed });
      const payload = response.ok ? response.body : { error: response.message };
      return {
        status: response.status,
        json: async () => JSON.parse(JSON.stringify(payload)) as unknown,
      };
    },
  };
}
