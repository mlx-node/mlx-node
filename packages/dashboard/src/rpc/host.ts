/**
 * The runtime's end of the dashboard RPC: answer {@link RpcRequest}s arriving on
 * one {@link RpcPort}.
 *
 * Only `call` and `subscribe` are reachable over the port. `ingestNow`, `drain`
 * and `close` are deliberately NOT — they are the supervisor's, and the process
 * that owns the runtime already holds it directly. A renderer that could close
 * the runtime it is rendering is not a feature.
 */

import { toFailure, type ApiResponse } from '../api/errors.js';
import type { DownloadEvent } from '../download.js';
import type { DashboardRuntime } from '../runtime.js';
import type { RpcPort } from './port.js';
import { isRpcRequest, type RpcReply } from './protocol.js';

/** Everything a port peer may reach. A stub satisfying this is enough to test the host. */
export type RpcRuntime = Pick<DashboardRuntime, 'call' | 'subscribe'>;

/**
 * Serve `runtime` over `port` until the returned dispose is called (or the peer
 * closes). Disposing releases every subscription this port opened — the manager
 * outlives the window, so a listener left behind is a real leak.
 */
export function serveRuntimeOverPort(runtime: RpcRuntime, port: RpcPort): () => void {
  /** Live subscriptions, keyed by the subscribe request's id. */
  const subscriptions = new Map<number, () => void>();

  const post = (reply: RpcReply): void => {
    try {
      port.postMessage(reply);
    } catch {
      // The peer is gone (or refused the clone). Nothing to recover: the caller's
      // deadline settles it, and there is no channel left to report on.
    }
  };

  /**
   * Post a response, falling back to a plain failure envelope when the handler's
   * body is not structured-clonable. Without this the caller waits out its whole
   * deadline for a reply that was never sent, and the fault is invisible.
   */
  const postResponse = (id: number, response: ApiResponse): void => {
    try {
      port.postMessage({ kind: 'response', id, response } satisfies RpcReply);
    } catch (err) {
      post({
        kind: 'response',
        id,
        response: toFailure(
          new Error(`Response cannot cross the message port: ${err instanceof Error ? err.message : String(err)}`),
        ),
      });
    }
  };

  const release = (id: number): void => {
    const unsubscribe = subscriptions.get(id);
    if (unsubscribe === undefined) return;
    subscriptions.delete(id);
    unsubscribe();
  };

  const releaseAll = (): void => {
    const all = [...subscriptions.values()];
    subscriptions.clear();
    for (const unsubscribe of all) unsubscribe();
  };

  const detach = port.listen({
    onMessage(data: unknown): void {
      // A malformed payload is dropped, never thrown: this runs inside the port's
      // own event dispatch, where a throw would surface as an unhandled error in
      // the process hosting the runtime rather than at any caller.
      if (!isRpcRequest(data)) return;
      switch (data.kind) {
        case 'call': {
          const id = data.id;
          // `runtime.call` documents that it never rejects; the catch is here so
          // that a broken contract costs one failure envelope instead of a caller
          // hanging until its deadline.
          void Promise.resolve()
            .then(() => runtime.call(data.call))
            .then(
              (response) => postResponse(id, response),
              (err: unknown) => postResponse(id, toFailure(err)),
            );
          return;
        }
        case 'subscribe': {
          const id = data.id;
          // A duplicate id from a misbehaving peer would otherwise orphan the
          // first subscription with no way left to release it.
          release(id);
          subscriptions.set(
            id,
            runtime.subscribe(data.jobId, (event: DownloadEvent) => {
              post({ kind: 'event', id, event });
            }),
          );
          return;
        }
        case 'unsubscribe':
          release(data.id);
          return;
      }
    },
    onClose(): void {
      // The peer is gone but the subscriptions it opened are still registered on
      // the download manager, which outlives the window.
      releaseAll();
    },
  });

  let disposed = false;
  return () => {
    if (disposed) return;
    disposed = true;
    detach();
    releaseAll();
    port.close();
  };
}
