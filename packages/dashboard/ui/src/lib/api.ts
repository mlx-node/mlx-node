/**
 * Typed browser client for the dashboard HTTP API.
 *
 * All requests target the same-origin `/api` surface (proxied to the server in
 * dev via `vite.config.ts`, same-origin in the built SPA). Paths are given
 * relative to `/api` — `getJson('/models')` hits `/api/models` — but a fully
 * qualified `/api/...` path is accepted too. Non-2xx responses reject with a
 * `DashboardApiError` carrying the HTTP status and the server's `error` message.
 */

/** Error thrown for any non-2xx API response; carries the HTTP status. */
export class DashboardApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'DashboardApiError';
    this.status = status;
  }
}

export type HttpMutation = 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/** Prefix a caller path with `/api`, tolerating an already-qualified path. */
function apiUrl(path: string): string {
  if (path === '/api' || path.startsWith('/api/')) return path;
  return `/api${path.startsWith('/') ? '' : '/'}${path}`;
}

/** Turn a failed response into a `DashboardApiError`, reading the JSON `error` field when present. */
async function toError(res: Response): Promise<DashboardApiError> {
  let message = res.statusText || `HTTP ${res.status}`;
  try {
    const body: unknown = await res.json();
    if (body !== null && typeof body === 'object' && typeof (body as { error?: unknown }).error === 'string') {
      message = (body as { error: string }).error;
    }
  } catch {
    // Non-JSON error body: keep the status text.
  }
  return new DashboardApiError(res.status, message);
}

/** GET a JSON resource, typed as `T`. */
export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path), { headers: { Accept: 'application/json' } });
  if (!res.ok) throw await toError(res);
  return (await res.json()) as T;
}

/** Send a mutating request (POST/PUT/PATCH/DELETE) with an optional JSON body, typed as `T`. */
export async function mutate<T>(method: HttpMutation, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  const init: RequestInit = { method, headers };
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const res = await fetch(apiUrl(path), init);
  if (!res.ok) throw await toError(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text === '' ? undefined : JSON.parse(text)) as T;
}

/** Handle to an open SSE stream; call `close()` to unsubscribe. */
export interface SseSubscription {
  close(): void;
}

/**
 * Subscribe to a Server-Sent Events endpoint. `onEvent` receives each frame's
 * `data` parsed as JSON (`T`); malformed frames are ignored. `onError` (if
 * given) receives the raw `EventSource` error event.
 *
 * `eventNames` lists the named SSE event types (`event:` lines) to also listen
 * for — `EventSource.onmessage` fires only for the default (unnamed) `message`
 * event, so a server that tags frames (e.g. the download stream's
 * `event: progress`) delivers nothing unless those names are registered here.
 */
export function subscribeSSE<T = unknown>(
  path: string,
  onEvent: (data: T) => void,
  onError?: (err: Event) => void,
  eventNames?: readonly string[],
): SseSubscription {
  const source = new EventSource(apiUrl(path));
  const handle = (ev: MessageEvent<string>): void => {
    try {
      onEvent(JSON.parse(ev.data) as T);
    } catch {
      // Ignore frames that are not valid JSON.
    }
  };
  source.onmessage = handle;
  for (const name of eventNames ?? []) {
    source.addEventListener(name, handle as EventListener);
  }
  if (onError !== undefined) source.onerror = onError;
  return {
    close(): void {
      source.close();
    },
  };
}
