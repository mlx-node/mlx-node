/**
 * Sidecar exit codes shared between the child that exits and MAIN that judges
 * the exit.
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
