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
 * outlives the window, so a listener left behind is a real leak. In-flight calls
 * keep the retired response path alive until cancellation is acknowledged or
 * already-started work returns its real result.
 */
export function serveRuntimeOverPort(runtime: RpcRuntime, port: RpcPort): () => void {
  /** Live subscriptions, keyed by the subscribe request's id. */
  const subscriptions = new Map<number, () => void>();
  /**
   * Calls owned by this renderer connection. The controller is deliberately
   * host-local: AbortSignal is not structured-clonable, so `cancel` carries only
   * the RPC id and the host turns it back into a signal for the runtime.
   */
  const calls = new Map<number, AbortController>();
  let disposing = false;
  let portFinished = false;
  let detach = (): void => {};

  const finishDispose = (): void => {
    if (!disposing || portFinished || calls.size > 0) return;
    portFinished = true;
    detach();
    port.close();
  };

  const post = (reply: RpcReply): void => {
    try {
      port.postMessage(reply);
    } catch {
      // The peer is gone (or refused the clone). Nothing to recover: the caller's
      // port-close path reports an explicitly unknown outcome, and there is no
      // channel left to answer on.
    }
  };

  /**
   * Post a response, falling back to a plain failure envelope when the handler's
   * body is not structured-clonable. Without this the caller waits for a reply
   * that can never arrive, and the fault is invisible.
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

  const requestCancellation = (id: number): void => {
    const controller = calls.get(id);
    if (controller === undefined) return;
    // Abort dispatch is synchronous. If the database client is already live,
    // its listener starts the worker acknowledgement handshake before
    // `abort()` returns; if runtime.call has not begun, its pre-aborted check
    // prevents the post.
    //
    // Keep the map entry: runtime.call now resolves only with either the
    // worker's acknowledged cancellation or the real result of work that had
    // already started, and the renderer is waiting for that authoritative
    // response.
    controller.abort();
  };

  const discardCall = (id: number): void => {
    const controller = calls.get(id);
    if (controller === undefined) return;
    calls.delete(id);
    controller.abort();
  };

  const cancelAll = (): void => {
    for (const controller of calls.values()) controller.abort();
  };

  const discardAll = (): void => {
    const all = [...calls.values()];
    calls.clear();
    for (const controller of all) controller.abort();
  };

  detach = port.listen({
    onMessage(data: unknown): void {
      // A malformed payload is dropped, never thrown: this runs inside the port's
      // own event dispatch, where a throw would surface as an unhandled error in
      // the process hosting the runtime rather than at any caller.
      if (!isRpcRequest(data)) return;
      switch (data.kind) {
        case 'call': {
          const id = data.id;
          // A duplicate id from a malformed peer must not orphan the earlier
          // mutation. Cancel it before assigning the id to the replacement.
          discardCall(id);
          const controller = new AbortController();
          calls.set(id, controller);
          // `runtime.call` documents that it never rejects; the catch is here so
          // that a broken contract costs one failure envelope instead of a caller
          // hanging until its deadline.
          void Promise.resolve()
            .then(() => runtime.call(data.call, controller.signal))
            .then(
              (response) => {
                // Missing or different means this connection discarded the
                // call, or a duplicate id has since replaced it. Either way its
                // late reply belongs to no live renderer request.
                if (calls.get(id) !== controller) return;
                calls.delete(id);
                postResponse(id, response);
                finishDispose();
              },
              (err: unknown) => {
                if (calls.get(id) !== controller) return;
                calls.delete(id);
                postResponse(id, toFailure(err));
                finishDispose();
              },
            );
          return;
        }
        case 'cancel':
          requestCancellation(data.id);
          return;
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
      // The peer is gone. Its subscriptions otherwise leak on the long-lived
      // download manager, and its queued calls otherwise remain able to mutate
      // state after the renderer has disappeared.
      disposing = true;
      discardAll();
      releaseAll();
      finishDispose();
    },
  });

  return () => {
    if (disposing) return;
    disposing = true;
    // Stop accepting work on the retired connection, but preserve its output
    // path until every old call reaches acknowledged cancellation or its real
    // result. ControlPanelSession replaces ports while the runtime outlives
    // them.
    detach();
    cancelAll();
    releaseAll();
    finishDispose();
  };
}
