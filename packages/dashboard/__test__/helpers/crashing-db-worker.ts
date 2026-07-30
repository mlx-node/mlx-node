/**
 * A database worker that dies mid-request: it answers nothing and kills its
 * thread the moment the first message arrives.
 *
 * Stands in for the real worker so a crash can be provoked deterministically —
 * every in-flight `call` must settle with a failure envelope, never hang.
 * Imports nothing relative, so it needs no `.js`→`.ts` resolve hook.
 */

import { parentPort } from 'node:worker_threads';

parentPort?.on('message', () => {
  process.exit(3);
});
