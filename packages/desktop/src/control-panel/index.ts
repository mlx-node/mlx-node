/**
 * CONTROL PANEL: the dashboard runtime, in its own `utilityProcess`.
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

import { createControlPanelSession } from './session.js';
import { CONTROL_PANEL_PROCESS_EXIT_CAP_MS, CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS } from './shutdown-timings.js';

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
  logError('[mlx] dist/control-panel/index.js must be forked as an Electron utilityProcess');
  process.exit(78);
}

/**
 * Set before any deliberate teardown, so a worker death during it is read as
 * part of the teardown rather than as a crash.
 *
 * The runtime's own `closing` latch is not enough: it flips inside
 * `worker.close()`, which runs AFTER `drain()`. A worker that dies during the
 * drain phase therefore still reports `worker-down`, and reacting to that by
 * exiting would abort the in-flight download shutdown that drain exists to
 * perform — orphaning a partial multi-GB `.staging` tree.
 */
let shuttingDown = false;

/** Distinct from the clean `0` so a crash report says which path exited. */
const EXIT_WORKER_DOWN = 70;

/**
 * Close the runtime, then leave. Shared by the signal handlers and `worker-down`.
 *
 * Only reachable after module init — from a signal, or from the worker's async
 * `error`/`exit` events — so `session` below is always assigned by the time this
 * runs. Do not call it during construction.
 */
function teardown(code: number): void {
  shuttingDown = true;
  // `runtime.close()` spends up to one worker deadline draining ingest, then a
  // SECOND worker deadline closing SQLite/terminating the thread. The enclosing
  // cap includes both sequential phases plus cleanup margin; MAIN's still-larger
  // broker grace is the next backstop.
  const cap = setTimeout(() => process.exit(code), CONTROL_PANEL_PROCESS_EXIT_CAP_MS);
  cap.unref();
  void session
    .close()
    .catch((error: unknown) => {
      logError(
        `[mlx] dashboard runtime did not close cleanly: ${error instanceof Error ? error.message : String(error)}`,
      );
    })
    .then(() => {
      process.exit(code);
    });
}

const runtime = createDashboardRuntime({
  // One budget per sequential worker phase. The process/broker/app caps are
  // derived from this in `shutdown-timings.ts`.
  shutdownTimeoutMs: CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS,
  onLifecycle: (event) => {
    // The database worker dying is invisible from the UI: every worker-thread
    // route just starts failing. Say so where the crash report can see it.
    logError(`[mlx] dashboard db worker: ${JSON.stringify(event)}`);

    // `down` is a write-once latch with no respawn, so 12 of the 16 dashboard
    // routes now fail forever while this process stays perfectly healthy. The
    // broker sees no exit, so its restart budget never runs; reloading or
    // reopening the window re-attaches to this same dead runtime. Exiting is
    // what hands the problem to `broker.ts`, which restarts CONTROL PANEL and
    // re-brokers a fresh port to the window that is still open.
    //
    // `db-closed` is the normal witness of a clean close and must not trigger
    // this.
    if (event.type === 'worker-down' && !shuttingDown) teardown(EXIT_WORKER_DOWN);
  },
});

const session = createControlPanelSession({ runtime });

parentPort.on('message', (event) => {
  const port = event.ports.at(0);
  // Every message MAIN sends carries a port; one that does not is not a
  // handshake, and dropping it is better than tearing down the live session.
  if (port === undefined) return;
  session.attach(bindEventEmitterPort(port as Parameters<typeof bindEventEmitterPort>[0]));
});

for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    teardown(0);
  });
}
