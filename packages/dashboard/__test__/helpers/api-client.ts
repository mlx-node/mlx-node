/**
 * A `fetch`-shaped client over the transport-independent dashboard runtime.
 *
 * The API suite used to drive every assertion through a real socket; it now
 * drives the same routes through `runtime.call`, so the assertions test API
 * behaviour rather than HTTP plumbing. The response is deliberately JSON
 * round-tripped, so every value a test sees is byte-for-byte what the HTTP
 * transport used to hand it. Genuinely transport-level assertions (Host/Origin
 * guards, content-type headers, SSE, static SPA) stay on real `fetch`.
 */

import { parseJsonBody } from '../../src/api/context.js';
import type { DashboardRuntime } from '../../src/runtime.js';

export interface TestResponse {
  status: number;
  json(): Promise<unknown>;
}

export interface TestRequestInit {
  method?: string;
  /** Accepted and ignored: the runtime has no headers. Kept so call sites are unchanged. */
  headers?: Record<string, string>;
  body?: string;
}

export interface TestClient {
  fetch(path: string, init?: TestRequestInit): Promise<TestResponse>;
}

export function createTestClient(runtime: DashboardRuntime): TestClient {
  return {
    async fetch(path: string, init?: TestRequestInit): Promise<TestResponse> {
      const parsed = init?.body === undefined ? { body: null } : parseJsonBody(init.body);
      if (!('body' in parsed)) throw new Error(`test client sent an unparseable body: ${parsed.bodyError}`);
      const response = await runtime.call({ method: init?.method ?? 'GET', path, body: parsed.body });
      const payload = response.ok ? response.body : { error: response.message };
      return {
        status: response.status,
        json: async () => JSON.parse(JSON.stringify(payload)) as unknown,
      };
    },
  };
}
