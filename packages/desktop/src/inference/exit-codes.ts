/**
 * Sidecar exit codes, and the start-failure line, shared between the child
 * that exits and MAIN that judges the exit.
 *
 * A leaf with no imports on purpose: `main/supervisor/state.ts` reads this to
 * decide restartability, and it must not pull in `sidecar.ts` — that module
 * is the child's side of the seam and value-imports the inference host, the
 * one thing MAIN's pure state machine can never evaluate (the three-process
 * split in `supervisor/index.ts`'s header).
 */

/**
 * Startup failed before the host existed (`createHost` rejected). Distinct
 * from 1 so a crash report can tell it from a throw — and from every other
 * code so the supervisor can tell it from a CRASH: the classic cause is
 * `NoModelsDiscoveredError`, and every cause is a deterministic environment
 * problem that retrying an unchanged child cannot fix.
 */
export const EXIT_STARTUP_FAILED = 78;

/**
 * The start of the one stderr line the child writes when `createHost` rejects;
 * the rest of the line is the error's message. MAIN finds it in the stderr tail
 * and shows the message as the failure's reason.
 */
export const STARTUP_FAILURE_PREFIX = '[mlx] inference host failed to start: ';
