/**
 * Transport-independent dispatch: match one request against {@link ROUTES}, run
 * its handler, and return an {@link ApiResponse} envelope. No `node:http` here —
 * the HTTP adapter in `server.ts` and the Electron MessagePort bridge both feed
 * this the same normalized input.
 *
 * Matching is thread-agnostic (it needs no context at all), so the 405-vs-404
 * distinction is identical wherever a request lands. Only running the handler
 * needs a context, and each thread has just its own half — hence
 * {@link dispatchMain} / {@link dispatchWorker} beside the omniscient
 * {@link dispatch}.
 */

import type { ApiContext, ApiRequest, MainApiContext, WorkerApiContext } from './context.js';
import { ApiError, failure, toFailure, type ApiResponse } from './errors.js';
import { ROUTES, type Route, type RouteThread } from './routes.js';

/** One normalized request, already split into path + query by the transport. */
export interface ApiRequestInput {
  method: string;
  /** Path only, no query string. */
  pathname: string;
  query: URLSearchParams;
  /** Parsed payload; `null` when the caller sent none. */
  body?: unknown;
  /** Set when the transport could not parse the payload (see `requireBody`). */
  bodyError?: string;
}

/**
 * Whether a path belongs to the API surface at all. The HTTP adapter serves the
 * static SPA for everything else; a path under `/api` that matches no route is
 * still an API path and yields a JSON 404, not `index.html`.
 */
export function isApiPath(pathname: string): boolean {
  return pathname === '/health' || pathname === '/api' || pathname.startsWith('/api/');
}

export function splitSegments(pathname: string): string[] {
  return pathname.split('/').filter((s) => s.length > 0);
}

/** Match a `:param` route against concrete path segments. */
function matchRoute(route: Route, method: string, segments: string[]): Record<string, string> | null {
  if (route.method !== method) return null;
  if (route.segments.length !== segments.length) return null;
  const params: Record<string, string> = {};
  for (let i = 0; i < route.segments.length; i++) {
    const pat = route.segments[i];
    if (pat.startsWith(':')) {
      // `decodeURIComponent` throws a bare `URIError` on a malformed escape
      // (`%`, `%zz`, a truncated `%E0%A4%A`). Raised as an `ApiError` instead so
      // it lands as 400 rather than the 500 `toFailure` gives an unknown throw —
      // this is client-supplied input, not an internal fault.
      //
      // Deliberately NOT `return null`: `segmentsMatch` skips param positions,
      // so a null here would leave the path "matched" and answer a misleading
      // 405 for what is really a malformed request.
      try {
        params[pat.slice(1)] = decodeURIComponent(segments[i]);
      } catch {
        throw ApiError.badRequest(`Malformed percent-escape in path segment "${segments[i]}"`);
      }
    } else if (pat !== segments[i]) {
      return null;
    }
  }
  return params;
}

/** Whether a route's segment pattern shape matches concrete segments (ignoring method/values). */
function segmentsMatch(pattern: string[], segments: string[]): boolean {
  if (pattern.length !== segments.length) return false;
  for (let i = 0; i < pattern.length; i++) {
    if (!pattern[i].startsWith(':') && pattern[i] !== segments[i]) return false;
  }
  return true;
}

/**
 * The transport-owned stream route (SSE download progress) this request names,
 * if any. A transport that can stream calls this BEFORE {@link dispatch} and
 * handles the request itself; one that cannot simply dispatches and gets the
 * handler's "not available over this transport" failure.
 */
export function matchStreamRoute(method: string, pathname: string): { params: Record<string, string> } | null {
  const segments = splitSegments(pathname);
  for (const r of ROUTES) {
    if (r.stream !== true) continue;
    const params = matchRoute(r, method, segments);
    if (params !== null) return { params };
  }
  return null;
}

/**
 * The thread that owns the route this request names, or `null` when nothing
 * matches (a 404/405 needs no handler and is answered wherever it arrives).
 * Callers route by THIS, never by a hard-coded path list.
 */
export function routeThreadFor(method: string, pathname: string): RouteThread | null {
  const segments = splitSegments(pathname);
  for (const r of ROUTES) {
    if (matchRoute(r, method, segments) !== null) return r.thread;
  }
  return null;
}

/** Runs the matched route's handler against whatever context the caller holds. */
type RouteRunner = (route: Route, req: ApiRequest) => unknown;

async function dispatchWith(run: RouteRunner, input: ApiRequestInput): Promise<ApiResponse> {
  const { method, pathname, query } = input;
  const segments = splitSegments(pathname);
  let pathMatched = false;
  for (const r of ROUTES) {
    const params = matchRoute(r, method, segments);
    if (params === null) {
      if (segmentsMatch(r.segments, segments)) pathMatched = true;
      continue;
    }
    const req: ApiRequest = {
      method,
      path: pathname,
      query,
      params,
      body: input.body ?? null,
      ...(input.bodyError !== undefined ? { bodyError: input.bodyError } : {}),
    };
    try {
      return { ok: true, status: r.successStatus ?? 200, body: await run(r, req) };
    } catch (err) {
      return toFailure(err);
    }
  }

  // A known path with the wrong method → 405; otherwise 404.
  if (pathMatched) {
    return failure('E_METHOD_NOT_ALLOWED', `Method ${method} not allowed for ${pathname}`);
  }
  return failure('E_NOT_FOUND', `No route matches ${method} ${pathname}`);
}

/** Dispatch against a context carrying BOTH halves (a test driving the table). */
export function dispatch(ctx: ApiContext, input: ApiRequestInput): Promise<ApiResponse> {
  return dispatchWith((route, req) => route.handler(ctx, req), input);
}

function wrongThread(route: Route, thread: RouteThread): ApiError {
  return ApiError.internal(
    `Route ${route.method} /${route.segments.join('/')} is owned by the ${route.thread} thread, not ${thread}`,
  );
}

/** Dispatch a transport-thread route. A worker-owned route never runs here. */
export function dispatchMain(ctx: MainApiContext, input: ApiRequestInput): Promise<ApiResponse> {
  return dispatchWith((route, req) => {
    if (route.thread !== 'main') throw wrongThread(route, 'main');
    return route.handler(ctx, req);
  }, input);
}

/** Dispatch a database-worker route. A transport-owned route never runs here. */
export function dispatchWorker(ctx: WorkerApiContext, input: ApiRequestInput): Promise<ApiResponse> {
  return dispatchWith((route, req) => {
    if (route.thread !== 'worker') throw wrongThread(route, 'worker');
    return route.handler(ctx, req);
  }, input);
}
