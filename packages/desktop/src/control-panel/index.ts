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
 * the old), `lifecycle.ts` what is told to MAIN, and the runtime and the RPC host
 * come from `@mlx-node/dashboard`. No worker event ends this process: a stuck or
 * dead database worker is replaced inside the runtime.
 */

import { writeSync } from 'node:fs';

import {
  bindEventEmitterPort,
  createDashboardRuntime,
  createOpenCalls,
  serveRuntimeOverPort,
  workCommandPath,
  type DashboardRuntime,
} from '@mlx-node/dashboard';

import { createCliLauncher, runAgentUpdate, type DesktopCliConfig } from '../cli-launcher.js';
import { firstClosing, LEDGER_DIR_ENV, onRuntimeLifecycle, pongOf, readAppEnds, readPing } from './lifecycle.js';
import { createControlPanelSession, type ControlPanelSession } from './session.js';
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
  postMessage(message: unknown): void;
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
const toMain: ParentPort = parentPort;

function post(message: unknown): void {
  try {
    toMain.postMessage(message);
  } catch {
    // MAIN is gone; the process is about to be.
  }
}

const calls = createOpenCalls();
let runtime: DashboardRuntime | null = null;
let session: ControlPanelSession | null = null;
/** Ports MAIN handed over before the session existed. */
const earlyPorts: unknown[] = [];

function attachPort(port: unknown): void {
  if (session === null) {
    earlyPorts.push(port);
    return;
  }
  session.attach(bindEventEmitterPort(port as Parameters<typeof bindEventEmitterPort>[0]));
}

// First, before the runtime: MAIN's ping must be answered from this thread alone, from the first tick on.
toMain.on('message', (event) => {
  const ping = readPing(event.data);
  if (ping !== null) {
    post(pongOf(ping.n, runtime, calls, Date.now()));
    return;
  }
  const port = event.ports.at(0);
  // Every other message MAIN sends carries a port; one that does not is not a
  // handshake, and dropping it is better than tearing down the live session.
  if (port === undefined) return;
  attachPort(port);
});

const closing = firstClosing();

/** Close the runtime, then leave. Reached only from a signal. */
function teardown(code: number): void {
  // Before anything async: MAIN keeps the list even when the close below never finishes.
  const note = closing(runtime);
  if (note !== null) post(note);
  // `runtime.close()` spends up to one worker deadline draining ingest, then a
  // SECOND worker deadline closing SQLite/terminating the thread. The enclosing
  // cap includes both sequential phases plus cleanup margin; MAIN's still-larger
  // broker grace is the next backstop.
  const cap = setTimeout(() => process.exit(code), CONTROL_PANEL_PROCESS_EXIT_CAP_MS);
  cap.unref();
  void (session?.close() ?? Promise.resolve())
    .catch((error: unknown) => {
      logError(
        `[mlx] dashboard runtime did not close cleanly: ${error instanceof Error ? error.message : String(error)}`,
      );
    })
    .then(() => {
      process.exit(code);
    });
}

const cli = process.env.MLX_DESKTOP_CLI
  ? createCliLauncher(JSON.parse(process.env.MLX_DESKTOP_CLI) as DesktopCliConfig)
  : undefined;

/** The Chat page's Install action: the app's own `mlx agent update`. */
function agentInstaller(launcher: NonNullable<typeof cli>): () => Promise<void> {
  return async () => runAgentUpdate(await launcher.prepare());
}
runtime = createDashboardRuntime({
  prepareDelegation: cli ? () => cli.prepare() : undefined,
  chat: {
    ...(cli ? { installAgent: agentInstaller(cli) } : {}),
    appEnds: readAppEnds(process.env[LEDGER_DIR_ENV]),
    workPath: workCommandPath,
    // MAIN never holds a turn count older than the turn itself.
    onBusyChange: () => post(pongOf(0, runtime, calls, Date.now())),
  },
  // One budget per sequential worker phase. The process/broker/app caps are
  // derived from this in `shutdown-timings.ts`.
  shutdownTimeoutMs: CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS,
  // A worker restart is told to MAIN; it never ends this process or a turn.
  onLifecycle: (event) => onRuntimeLifecycle(event, post, () => runtime?.busyConversations().length ?? 0, logError),
});

session = createControlPanelSession({
  runtime,
  serve: (served, port) => serveRuntimeOverPort(served, port, { calls }),
});
for (const port of earlyPorts.splice(0)) attachPort(port);

for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    teardown(0);
  });
}
