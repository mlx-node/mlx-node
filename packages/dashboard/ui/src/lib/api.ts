/**
 * Typed client for the dashboard API, spoken over a MessagePort.
 *
 * There is no HTTP here and no origin to be same as: the page is served from
 * `app://` and reaches the runtime through a port the desktop app's preload hands
 * over. Paths keep the `/api` shape (`getJson('/models')` → `/api/models`) because
 * the route table is addressed that way on both sides — it is an address, not a
 * URL.
 *
 * The port is INJECTED (see {@link connectDashboardApi}) rather than picked off a
 * global: the preload hop delivers it as `window.postMessage(…, [port])`, whoever
 * receives that message decides what to do with it, and an injected port is the
 * only version of this that can be tested.
 */

import type { DownloadEvent } from '../../../src/download.js';
import { createRpcClient, type RpcClient, type RpcClientOptions } from '../../../src/rpc/client.js';
import type { RpcPort } from '../../../src/rpc/port.js';
import { bumpConnectionGeneration } from './connection';
import { clearCache } from './json-cache';

/**
 * Error thrown for any failed API call.
 *
 * `code` is the stable discriminator the runtime raised; `status` is derived from
 * it and kept because pages and hooks already branch on it. Both come off the
 * wire — nothing here re-derives a status from a guess.
 */
export class DashboardApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'DashboardApiError';
    this.status = status;
    this.code = code;
  }
}

export type HttpMutation = 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/** Prefix a caller path with `/api`, tolerating an already-qualified path. */
function apiPath(path: string): string {
  if (path === '/api' || path.startsWith('/api/')) return path;
  return `/api${path.startsWith('/') ? '' : '/'}${path}`;
}

let client: RpcClient | null = null;
/** Identity guard for recovery callbacks from clients retired by a reconnect. */
let rpcGeneration = 0;

/**
 * Attach the API to a port. Call once, before rendering anything that fetches.
 * `opts.onUnresponsive` must tear down the runtime that owns the peer port.
 * Replacing an existing connection closes the old one, so a reconnect cannot
 * leave the previous client's subscriptions registered on the runtime.
 */
export function connectDashboardApi(port: RpcPort, opts: RpcClientOptions): void {
  const generation = ++rpcGeneration;
  const previous = client;
  // Invalidate the old generation BEFORE asking it to drain. A synchronous
  // postMessage failure during `close()` must not let a retired client restart
  // the healthy runtime whose replacement port just arrived.
  client = null;
  previous?.close();
  // Cached bodies describe the runtime on the other end of the OLD port. A
  // reconnect is a different runtime — a respawned sidecar, or a fresh stub in
  // a test — so anything held from before it is not stale, it is about
  // something else entirely.
  clearCache();
  // `clearCache` only reaches future mounts. This is what tells the hooks
  // already on screen that they are bound to a runtime that no longer exists.
  bumpConnectionGeneration();
  client = createRpcClient(port, {
    ...opts,
    onUnresponsive(reason): void {
      // A delayed timer from a superseded connection is not evidence about the
      // runtime currently serving the page. A plain disconnect does not advance
      // this generation, so its close drain still retains recovery coverage.
      if (generation === rpcGeneration) opts.onUnresponsive(reason);
    },
  });
}

/**
 * Drop the connection. In-flight calls drain to an acknowledged cancellation
 * or their real result before the retired port closes.
 */
export function disconnectDashboardApi(): void {
  client?.close();
  clearCache();
  client = null;
}

function requireClient(): RpcClient {
  if (client === null) {
    throw new DashboardApiError(503, 'E_UNAVAILABLE', 'The dashboard is not connected to its runtime yet');
  }
  return client;
}

async function callApi<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await requireClient().call({
    method,
    path: apiPath(path),
    ...(body === undefined ? {} : { body }),
  });
  if (!response.ok) throw new DashboardApiError(response.status, response.code, response.message);
  return response.body as T;
}

/** GET a JSON resource, typed as `T`. */
export function getJson<T>(path: string): Promise<T> {
  return callApi<T>('GET', path);
}

/**
 * Send a mutating request (POST/PUT/PATCH/DELETE) with an optional JSON body,
 * typed as `T`.
 *
 * Drops every cached GET body, on failure as well as success: a write that
 * reports an error may still have acted (a delete that removed the files and
 * then failed to verify), and serving a body that a partly-applied write
 * invalidated is worse than one extra round-trip.
 */
export async function mutate<T>(method: HttpMutation, path: string, body?: unknown): Promise<T> {
  try {
    return await callApi<T>(method, path, body);
  } finally {
    clearCache();
  }
}

/** Handle to a live subscription; call `close()` to stop delivery. */
export interface Subscription {
  close(): void;
}

/**
 * Subscribe to one download job's progress events.
 *
 * Unlike the SSE stream this replaces, there are no frame names to register and
 * no reconnect: the port either carries events or is gone, and a gone port stops
 * delivering rather than silently retrying forever behind a stale progress bar.
 */
export function subscribeDownload(id: string, onEvent: (event: DownloadEvent) => void): Subscription {
  const unsubscribe = requireClient().subscribe(id, onEvent);
  return { close: unsubscribe };
}
