/**
 * ADMIN: the dashboard runtime, in its own `utilityProcess`.
 *
 * ## Why this is a process
 *
 * Not for head-of-line blocking any more — `46524cfd` moved `DashboardDb` and
 * every synchronous route onto a `node:worker_threads` worker. Two reasons
 * survive it:
 *
 *  1. **Crash isolation.** An unhandled throw in dashboard code — a route
 *     handler, an ingest, the HF download client — must not take down the tray
 *     and the INFERENCE supervisor with it. In MAIN it would: the utilityProcess
 *     children die with their parent, so a dashboard bug would stop a model the
 *     user is mid-conversation with.
 *  2. **The download manager still blocks its own thread.** `listStagedFiles`
 *     runs one `statSync` per file over a recursive listing of a staging tree,
 *     and `realpathSync`/`lstatSync` bracket every publish. On a multi-GB
 *     checkpoint that is exactly the kind of block MAIN measured at 2960 ms of
 *     event-loop lag with the tray and the window chrome frozen through it.
 *
 * ## What this process must NOT do
 *
 * Link the native addon. `@mlx-node/dashboard` is addon-free — its only
 * `@mlx-node` dependency at runtime is `@mlx-node/agent/catalog`, a data leaf
 * with no imports at all — and `__test__/child-entries.test.ts` walks the import
 * graph from this file and probes a fresh process to keep it that way. The addon
 * belongs to INFERENCE alone; that split is the reason there are three processes.
 *
 * Nothing is decided here: `session.ts` holds the one rule (a new port replaces
 * the old), and the runtime and the RPC host come from `@mlx-node/dashboard`.
 */

import { writeSync } from 'node:fs';

import { bindEventEmitterPort, createDashboardRuntime } from '@mlx-node/dashboard';

import { createAdminSession } from './session.js';

/**
 * `process.parentPort` in an Electron `utilityProcess`, structurally.
 *
 * Declared rather than imported from `electron`: this file needs one global that
 * Electron installs on `process`, not the module — and an `import` of that
 * specifier would make an addon-free entry look like an Electron one to the
 * import-graph test, for a type that is erased anyway.
 */
interface ParentPort {
  on(event: 'message', listener: (event: { data: unknown; ports: unknown[] }) => void): void;
}

function logError(line: string): void {
  // stdout/stderr are async pipes; an exit right after a write truncates it, and
  // for a utilityProcess this line is the only diagnostic MAIN gets.
  writeSync(2, `${line}\n`);
}

const parentPort = (process as unknown as { parentPort?: ParentPort }).parentPort;
if (parentPort === undefined) {
  logError('[mlx] dist/admin/index.js must be forked as an Electron utilityProcess');
  process.exit(78);
}

const runtime = createDashboardRuntime({
  // Below MAIN's 10 s quit deadline, so a worker that will not answer is
  // terminated by the runtime rather than by the app running out of time.
  shutdownTimeoutMs: 4_000,
  onLifecycle: (event) => {
    // The database worker dying is invisible from the UI: every worker-thread
    // route just starts failing. Say so where the crash report can see it.
    logError(`[mlx] dashboard db worker: ${JSON.stringify(event)}`);
  },
});

const session = createAdminSession({ runtime });

parentPort.on('message', (event) => {
  const port = event.ports.at(0);
  // Every message MAIN sends carries a port; one that does not is not a
  // handshake, and dropping it is better than tearing down the live session.
  if (port === undefined) return;
  session.attach(bindEventEmitterPort(port as Parameters<typeof bindEventEmitterPort>[0]));
});

for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    void session
      .close()
      .catch((error: unknown) => {
        logError(
          `[mlx] dashboard runtime did not close cleanly: ${error instanceof Error ? error.message : String(error)}`,
        );
      })
      .then(() => {
        process.exit(0);
      });
  });
}
