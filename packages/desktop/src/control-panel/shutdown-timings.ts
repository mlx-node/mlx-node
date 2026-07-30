/**
 * CONTROL PANEL shutdown budgets, shared by the child, MAIN's broker, and the
 * whole-app quit deadline.
 *
 * `DashboardRuntime.close()` has two sequential database-worker phases:
 *
 *   1. `worker.drain()` waits for the ingest barrier;
 *   2. `worker.close()` asks the worker to close SQLite, then terminates it.
 *
 * Each phase gets {@link CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS}. The enclosing
 * deadlines therefore have to be nested, not independently chosen:
 *
 *   worker phases (8 s) < child cap (10 s) < broker SIGKILL (12 s)
 *     < whole-app quit (14 s)
 *
 * The gaps are cleanup/scheduling margin. They let the worker termination
 * promise settle, the utility process emit `exit`, and MAIN observe that event
 * before the next enclosing deadline fires.
 */

/** One database-worker shutdown phase: drain, then close. */
export const CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS = 4_000;

/** Time for both worker phases plus worker/process cleanup. */
export const CONTROL_PANEL_PROCESS_EXIT_CAP_MS = CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS * 2 + 2_000;

/** SIGTERM window before MAIN escalates the CONTROL PANEL child to SIGKILL. */
export const CONTROL_PANEL_BROKER_KILL_GRACE_MS = CONTROL_PANEL_PROCESS_EXIT_CAP_MS + 2_000;

/** Whole-app cap, including broker exit observation and the settings flush. */
export const DESKTOP_QUIT_DEADLINE_MS = CONTROL_PANEL_BROKER_KILL_GRACE_MS + 2_000;
