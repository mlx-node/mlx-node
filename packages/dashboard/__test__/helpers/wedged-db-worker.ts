/**
 * A database worker that never answers and never exits — the wedged thread a
 * bounded shutdown has to survive. The message listener keeps its event loop
 * alive, so only `worker.terminate()` ends it.
 */

import { parentPort } from 'node:worker_threads';

parentPort?.on('message', () => {
  // Deliberately silent: the runtime must fall back to its shutdown timeout.
});
