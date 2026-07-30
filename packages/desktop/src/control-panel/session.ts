/**
 * CONTROL PANEL's half of the connection: serve the dashboard runtime on whichever port
 * MAIN last brokered.
 *
 * The runtime outlives every port. A window reload mints a fresh
 * `MessageChannelMain` (see `src/main/broker.ts`), so this process can be handed
 * a second port while it is still serving the first. Nothing reliably closes the
 * old port for us: the RPC host only learns about a peer going away through the
 * port's own `close` event, and a renderer that navigated does not always produce
 * one.
 *
 * That matters because a subscription registered by the old port stays on the
 * download manager, which lives for the life of the process — one leaked
 * listener per reload, each still receiving progress for a job nobody is
 * watching. So `attach` replaces rather than adds: subscriptions are released
 * immediately, while old calls keep their response path only until cancellation
 * is acknowledged or already-started work reports its real result.
 *
 * Electron-free: an `RpcPort` is whatever `bindEventEmitterPort` /
 * `bindEventTargetPort` returns, so the tests drive this over a real
 * `node:worker_threads` MessageChannel with the real `serveRuntimeOverPort`.
 */

import { serveRuntimeOverPort, type RpcPort, type RpcRuntime } from '@mlx-node/dashboard';

/** What the session needs of `DashboardRuntime` — the two RPC members plus shutdown. */
export interface ControlPanelSessionRuntime extends RpcRuntime {
  close(): Promise<void>;
}

export interface ControlPanelSessionOptions {
  runtime: ControlPanelSessionRuntime;
  /**
   * Test seam. Defaults to `serveRuntimeOverPort`, which is what makes only
   * `call` and `subscribe` reachable over the port — `close`, `drain` and
   * `ingestNow` stay this process's, because a renderer that could close the
   * runtime it is rendering is not a feature.
   */
  serve?: (runtime: RpcRuntime, port: RpcPort) => () => void;
}

export interface ControlPanelSession {
  /** Serve on `port`, retiring and draining whatever was being served before. */
  attach(port: RpcPort): void;
  /** How many ports have been attached. */
  generation(): number;
  /**
   * Release the current port and close the runtime. Idempotent.
   *
   * The runtime's own `close()` drains in-flight downloads before it ends the
   * database worker — that is what keeps a SIGTERM from orphaning a partial,
   * potentially multi-GB `.staging` tree — so this is worth awaiting on the way
   * out rather than letting the process be killed under it.
   */
  close(): Promise<void>;
}

export function createControlPanelSession(options: ControlPanelSessionOptions): ControlPanelSession {
  const serve = options.serve ?? serveRuntimeOverPort;
  let release: (() => void) | null = null;
  let generation = 0;
  let closing: Promise<void> | null = null;

  function releaseCurrent(): void {
    const current = release;
    release = null;
    if (current === null) return;
    try {
      current();
    } catch {
      // Disposing a serve whose port the peer already closed must not stop the
      // next one from being installed.
    }
  }

  return {
    attach(port: RpcPort): void {
      if (closing !== null) {
        // Shutting down. Serving a port now would register subscriptions on a
        // runtime that is being drained, and answer calls against a database
        // that is about to close.
        port.close();
        return;
      }
      releaseCurrent();
      generation += 1;
      release = serve(options.runtime, port);
    },
    generation: () => generation,
    close(): Promise<void> {
      closing ??= (async () => {
        releaseCurrent();
        await options.runtime.close();
      })();
      return closing;
    },
  };
}
