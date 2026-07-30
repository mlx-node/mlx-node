/**
 * The caller's end of the dashboard RPC: `call` / `subscribe` over one
 * {@link RpcPort}, shaped so no caller can hang.
 *
 * `call` mirrors `runtime.call` exactly — same argument, same
 * never-rejects-for-a-failure contract, an {@link ApiResponse} envelope either
 * way. That is deliberate: code written against the in-process runtime works
 * unchanged over a port, and a transport fault arrives as a `code` the UI can
 * branch on instead of a bare rejection it can only stringify.
 *
 * Three independent things can stop a reply from arriving, and all three settle:
 * the peer closing (the port's `close` event), the peer wedging (the per-request
 * deadline), and the payload being unclonable (`postMessage` throws synchronously
 * before the message ever leaves).
 *
 * No `node:` imports — this module runs in the renderer.
 */

import { failure, type ApiResponse } from '../api/errors.js';
import type { DownloadEvent } from '../download.js';
import type { ApiCall } from '../runtime.js';
import type { RpcPort } from './port.js';
import { isRpcReply, type RpcRequest } from './protocol.js';

/**
 * Per-request deadline. The old HTTP transport had none — a hung server left a
 * `fetch` pending until the user gave up — so this is a new guarantee rather than
 * a preserved one, and it is generous: it exists to bound a wedged peer, not to
 * police slow queries.
 */
const DEFAULT_TIMEOUT_MS = 30_000;

export interface RpcClientOptions {
  /** Per-request deadline in ms. Defaults to 30 s. */
  timeoutMs?: number;
}

export interface RpcClient {
  /** Answer one API call. Never rejects for an API failure — see {@link ApiResponse}. */
  call(call: ApiCall): Promise<ApiResponse>;
  /** Subscribe to a download job's progress events; returns the unsubscribe. */
  subscribe(jobId: string, listener: (event: DownloadEvent) => void): () => void;
  /** Detach, settle every in-flight call, drop every subscription, close the port. Idempotent. */
  close(): void;
}

export function createRpcClient(port: RpcPort, opts: RpcClientOptions = {}): RpcClient {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const pending = new Map<number, { settle: (response: ApiResponse) => void; timer: ReturnType<typeof setTimeout> }>();
  const listeners = new Map<number, (event: DownloadEvent) => void>();
  let nextId = 1;
  /** Set once no further reply can arrive; every later call fails fast instead of waiting out its deadline. */
  let down: string | null = null;

  const goDown = (reason: string, withdrawPending = false): void => {
    down ??= reason;
    // Drain by swapping the map out first: settling a caller can re-enter
    // `call()`, and that call must see `down` and fail fast rather than land in a
    // collection we are mid-iteration over.
    const waiting = [...pending.entries()];
    pending.clear();
    listeners.clear();
    for (const [id, entry] of waiting) {
      clearTimeout(entry.timer);
      if (withdrawPending) {
        try {
          // Post before settling: a renderer reconnect/close must not report
          // failure while leaving a destructive call merely queued on the old
          // runtime connection.
          port.postMessage({ kind: 'cancel', id } satisfies RpcRequest);
        } catch {
          // Closing the port below still makes the host abort every call owned
          // by this connection.
        }
      }
      entry.settle(failure('E_UNAVAILABLE', `Dashboard runtime is unavailable: ${down}`));
    }
  };

  const detach = port.listen({
    onMessage(data: unknown): void {
      // A port is only as trusted as whoever holds the other end; a malformed
      // payload is dropped rather than allowed to throw out of the port's own
      // event dispatch, which no caller could catch.
      if (!isRpcReply(data)) return;
      if (data.kind === 'event') {
        listeners.get(data.id)?.(data.event);
        return;
      }
      const entry = pending.get(data.id);
      // No entry means the request already timed out and settled. Dropping the
      // late reply is what keeps a caller from being settled twice.
      if (entry === undefined) return;
      pending.delete(data.id);
      clearTimeout(entry.timer);
      entry.settle(data.response);
    },
    onClose(): void {
      goDown('the message port was closed');
    },
  });

  let closed = false;

  const post = (request: RpcRequest): void => {
    port.postMessage(request);
  };

  return {
    call(apiCall: ApiCall): Promise<ApiResponse> {
      if (down !== null) {
        return Promise.resolve(failure('E_UNAVAILABLE', `Dashboard runtime is unavailable: ${down}`));
      }
      const id = nextId++;
      return new Promise<ApiResponse>((resolve) => {
        const timer = setTimeout(() => {
          // Enqueue cancellation across the RPC boundary BEFORE settling. On
          // this live MessagePort it stays ordered after the call; the host turns
          // the id into an AbortSignal, and the runtime forwards that through
          // the worker's dedicated withdrawal port. The host may process it
          // after this callback returns, so this promises ordering, not an ack.
          try {
            post({ kind: 'cancel', id });
          } catch {
            // The peer is gone. Its port-close path aborts every call belonging
            // to this connection; the local deadline must still settle.
          }
          // Drop the entry before resolving so a reply that was already in
          // flight cannot settle the promise twice.
          pending.delete(id);
          resolve(failure('E_UNAVAILABLE', `Dashboard runtime did not answer within ${timeoutMs}ms`));
        }, timeoutMs);
        pending.set(id, { settle: resolve, timer });
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
      // A dead port can deliver nothing; hand back an inert unsubscribe rather
      // than register a listener that would leak until `close()`.
      if (down !== null) return () => {};
      const id = nextId++;
      listeners.set(id, listener);
      try {
        post({ kind: 'subscribe', id, jobId });
      } catch {
        listeners.delete(id);
        return () => {};
      }
      let live = true;
      return () => {
        if (!live) return;
        live = false;
        // Local first: an event the host posted before the unsubscribe reached it
        // is already in flight, and dropping the listener here is the only thing
        // that stops it being delivered after the caller asked it to stop.
        listeners.delete(id);
        if (down !== null) return;
        try {
          post({ kind: 'unsubscribe', id });
        } catch {
          // The peer is gone; its subscription died with it.
        }
      };
    },

    close(): void {
      if (closed) return;
      closed = true;
      detach();
      goDown('the client was closed', true);
      port.close();
    },
  };
}
