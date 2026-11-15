/**
 * The caller's end of the dashboard RPC: `call` / `subscribe` / `watchLive` / `watchChat` over one
 * {@link RpcPort}, shaped so no caller can hang.
 *
 * `call` mirrors `runtime.call` exactly — same argument, same
 * never-rejects-for-a-failure contract, an {@link ApiResponse} envelope either
 * way. That is deliberate: code written against the in-process runtime works
 * unchanged over a port, and a transport fault arrives as a `code` the UI can
 * branch on instead of a bare rejection it can only stringify.
 *
 * Three independent things can stop an ordinary reply from arriving: the peer
 * closing (the port's `close` event), a slow peer (the per-request deadline
 * starts an acknowledged cancellation, and the grace after it fails that one
 * call with `E_TIMEOUT`), and the payload being unclonable (`postMessage` throws
 * synchronously before the message ever leaves).
 *
 * The page never judges whether the peer is alive. A call with no reply fails
 * alone; streams and other calls are untouched.
 *
 * No `node:` imports — this module runs in the renderer.
 */

import { failure, type ApiResponse } from '../api/errors.js';
import type { ChatHostEvent, ConversationEvent } from '../chat/types.js';
import type { DownloadEvent } from '../download.js';
import type { LiveEvent } from '../live.js';
import type { ApiCall } from '../runtime.js';
import type { RpcPort } from './port.js';
import { isRpcReply, type ChatScope, type RpcRequest } from './protocol.js';

/** Per-request deadline: at expiry the client asks the host to withdraw the call. */
const DEFAULT_TIMEOUT_MS = 30_000;
/** After the withdrawal request, how long the original reply may still take before the call fails with `E_TIMEOUT`. */
const DEFAULT_CANCELLATION_GRACE_MS = 5_000;

export interface RpcClientOptions {
  /** Delay before requesting acknowledged cancellation. Defaults to 30 s. */
  timeoutMs?: number;
  /** Bound after cancellation is requested; expiry fails that call with `E_TIMEOUT`. Defaults to 5 s. */
  cancellationGraceMs?: number;
  /** Called once when the peer closed the port. Never called after `close()`. */
  onDown?(reason: string): void;
}

/** The failure of a call that got no reply of any kind within the deadline and its grace. */
export function timeoutFailure(method: string, seconds: number): ApiResponse {
  const base = `The app did not answer in ${String(seconds)} s.`;
  return failure('E_TIMEOUT', method === 'GET' ? base : `${base} The change may or may not have been applied.`);
}

export interface RpcClient {
  /** Answer one API call. Never rejects for an API failure — see {@link ApiResponse}. */
  call(call: ApiCall): Promise<ApiResponse>;
  /** Subscribe to a download job's progress events; returns the unsubscribe. */
  subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void;
  /** Subscribe to live engine state and data-change events; returns the unsubscribe. */
  watchLive(listener: (event: LiveEvent) => void): () => void;
  /** Open a Chat stream (host state, or one conversation); returns the unsubscribe. */
  watchChat(scope: ChatScope, listener: (event: ChatHostEvent | ConversationEvent) => void): () => void;
  /**
   * Stop new work and subscriptions, then drain in-flight calls to an
   * acknowledged cancellation or real result before closing the port.
   * Idempotent.
   */
  close(): void;
}

export function createRpcClient(port: RpcPort, opts: RpcClientOptions = {}): RpcClient {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const cancellationGraceMs = opts.cancellationGraceMs ?? DEFAULT_CANCELLATION_GRACE_MS;
  const timeoutSeconds = Math.round((timeoutMs + cancellationGraceMs) / 1000);

  const pending = new Map<
    number,
    {
      settle: (response: ApiResponse) => void;
      method: string;
      timer: ReturnType<typeof setTimeout>;
      cancellationRequested: boolean;
    }
  >();
  const listeners = new Map<number, (event: DownloadEvent) => void>();
  const liveListeners = new Map<number, (event: LiveEvent) => void>();
  const chatListeners = new Map<number, (event: ChatHostEvent | ConversationEvent) => void>();
  let nextId = 1;
  /** Set once no further reply can arrive; every later call fails fast instead of waiting out its deadline. */
  let down: string | null = null;

  let closed = false;
  let portFinished = false;
  let downReported = false;
  let detach = (): void => {};

  const reportDown = (reason: string): void => {
    if (downReported) return;
    downReported = true;
    try {
      opts.onDown?.(reason);
    } catch {
      // Runs from the port's own event dispatch or a timer; nothing could catch it.
    }
  };

  const finishClose = (): void => {
    if (!closed || portFinished || pending.size > 0) return;
    portFinished = true;
    detach();
    port.close();
  };

  const goDown = (reason: string): void => {
    // Override a graceful local-close reason: losing the peer while calls are
    // draining means their outcome is unknown, which is materially different
    // from an acknowledged cancellation.
    down = reason;
    // Drain by swapping the map out first: settling a caller can re-enter
    // `call()`, and that call must see `down` and fail fast rather than land in a
    // collection we are mid-iteration over.
    const waiting = [...pending.values()];
    pending.clear();
    listeners.clear();
    liveListeners.clear();
    chatListeners.clear();
    for (const entry of waiting) {
      clearTimeout(entry.timer);
      entry.settle(failure('E_UNAVAILABLE', `Dashboard runtime is unavailable: ${down}`));
    }
    finishClose();
  };

  detach = port.listen({
    onMessage(data: unknown): void {
      // A port is only as trusted as whoever holds the other end; a malformed
      // payload is dropped rather than allowed to throw out of the port's own
      // event dispatch, which no caller could catch.
      if (!isRpcReply(data)) return;
      if (data.kind === 'event') {
        listeners.get(data.id)?.(data.event);
        return;
      }
      if (data.kind === 'live') {
        liveListeners.get(data.id)?.(data.event);
        return;
      }
      if (data.kind === 'chat') {
        chatListeners.get(data.id)?.(data.event);
        return;
      }
      const entry = pending.get(data.id);
      // No entry means the request already settled or the client went down.
      // Dropping the late reply is what keeps a caller from being settled twice.
      if (entry === undefined) return;
      pending.delete(data.id);
      clearTimeout(entry.timer);
      entry.settle(data.response);
      finishClose();
    },
    onClose(): void {
      closed = true;
      goDown('the message port was closed; call outcome is unknown');
      reportDown('the message port was closed');
    },
  });

  const post = (request: RpcRequest): void => {
    port.postMessage(request);
  };

  const expire = (id: number, entry: { settle: (response: ApiResponse) => void; method: string }): void => {
    // No reply of any kind for the whole deadline and grace. Only THIS call fails: whether the peer is alive is
    // MAIN's to judge, never one slow reply's. A late reply finds no entry and is dropped.
    if (pending.get(id) !== entry) return;
    pending.delete(id);
    entry.settle(timeoutFailure(entry.method, timeoutSeconds));
    finishClose();
  };

  const requestCancellation = (
    id: number,
    entry: {
      settle: (response: ApiResponse) => void;
      method: string;
      timer: ReturnType<typeof setTimeout>;
      cancellationRequested: boolean;
    },
  ): void => {
    if (entry.cancellationRequested) return;
    entry.cancellationRequested = true;
    clearTimeout(entry.timer);
    try {
      post({ kind: 'cancel', id });
    } catch {
      // The withdrawal could not cross; the grace below still bounds the call.
    }
    entry.timer = setTimeout(() => expire(id, entry), cancellationGraceMs);
  };

  /** Register `listener` under a fresh id and post the opening request; returns the unsubscribe. */
  const open = <E>(
    registry: Map<number, (event: E) => void>,
    listener: (event: E) => void,
    request: (id: number) => RpcRequest,
  ): (() => void) => {
    // A dead port can deliver nothing; hand back an inert unsubscribe rather
    // than register a listener that would leak until `close()`.
    if (down !== null) return () => {};
    const id = nextId++;
    registry.set(id, listener);
    try {
      post(request(id));
    } catch {
      registry.delete(id);
      return () => {};
    }
    let live = true;
    return () => {
      if (!live) return;
      live = false;
      // Local first: an event the host posted before the unsubscribe reached it
      // is already in flight, and dropping the listener here is the only thing
      // that stops it being delivered after the caller asked it to stop.
      registry.delete(id);
      if (down !== null) return;
      try {
        post({ kind: 'unsubscribe', id });
      } catch {
        // The peer is gone; its subscription died with it.
      }
    };
  };

  return {
    call(apiCall: ApiCall): Promise<ApiResponse> {
      if (down !== null) {
        return Promise.resolve(failure('E_UNAVAILABLE', `Dashboard runtime is unavailable: ${down}`));
      }
      const id = nextId++;
      return new Promise<ApiResponse>((resolve) => {
        const timer = setTimeout(() => {
          const entry = pending.get(id);
          if (entry !== undefined) requestCancellation(id, entry);
        }, timeoutMs);
        pending.set(id, { settle: resolve, method: apiCall.method, timer, cancellationRequested: false });
        try {
          post({ kind: 'call', id, call: apiCall });
        } catch (err) {
          // A payload the structured clone algorithm refuses throws HERE, before
          // the peer ever sees it. Settling now is what keeps the caller from
          // waiting out a deadline for a request that was never delivered.
          pending.delete(id);
          clearTimeout(timer);
          resolve(
            failure(
              'E_BAD_REQUEST',
              `Request cannot cross the message port: ${err instanceof Error ? err.message : String(err)}`,
            ),
          );
        }
      });
    },

    subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void {
      return open(listeners, listener, (id) => ({ kind: 'subscribe', id, jobId }));
    },

    watchLive(listener: (event: LiveEvent) => void): () => void {
      return open(liveListeners, listener, (id) => ({ kind: 'watch', id, topic: 'live' }));
    },

    watchChat(scope: ChatScope, listener: (event: ChatHostEvent | ConversationEvent) => void): () => void {
      return open(chatListeners, listener, (id) => ({ kind: 'chat', id, ...scope }));
    },

    close(): void {
      if (closed) return;
      closed = true;
      downReported = true;
      down = 'the client was closed';

      // Subscriptions have no terminal acknowledgement to wait for. Drop local
      // delivery immediately and ask the host to release each registration
      // while the old port remains alive for pending calls.
      for (const id of [...listeners.keys(), ...liveListeners.keys(), ...chatListeners.keys()]) {
        try {
          post({ kind: 'unsubscribe', id });
        } catch {
          // The close below/onClose path releases the host side.
        }
      }
      listeners.clear();
      liveListeners.clear();
      chatListeners.clear();

      // A graceful reconnect is not permission to guess the outcome of old
      // mutations. Request cancellation and retain the port until each call gets
      // an acknowledged skip, the real result of already-started work, or its
      // `E_TIMEOUT` after the grace.
      for (const [id, entry] of pending) requestCancellation(id, entry);
      finishClose();
    },
  };
}
