/**
 * The stop budgets of `mlx serve` and of the `mlx agent` that runs it, in one
 * place (Splash `supervision`).
 *
 * Stopping, from the outside in:
 * 1. `mlx serve`, on SIGINT or SIGTERM, closes its HTTP server within
 *    {@link SERVER_CLOSE_MS}, then gives the waiting restore points of every
 *    persistent SSD cache {@link FLUSH_MS} in all to reach the disk, and exits.
 * 2. `mlx agent` stops its serve with a signal and sends SIGKILL to one still
 *    running after {@link STOP_MS}.
 */

/** HTTP server close (in-flight requests, then forced sockets). */
export const SERVER_CLOSE_MS = 5000;

/** Splash `kFlush`: one deadline shared by every model's `closeCache`. */
export const FLUSH_MS = 6000;

/** Splash `kStop`: from a stop signal to serve's exit, its flush included. */
export const STOP_MS = 15000;

if (!(SERVER_CLOSE_MS + FLUSH_MS < STOP_MS)) {
  throw new Error('SERVER_CLOSE_MS + FLUSH_MS must stay below STOP_MS');
}
