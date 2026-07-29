/**
 * INFERENCE: the supervised sidecar, and the ONLY process in the app that maps
 * the native addon.
 *
 * `import { createInferenceHost } from '@mlx-node/server/host'` statically pulls
 * `@mlx-node/lm`, which dlopens `mlx-core.darwin-arm64.node` at module scope —
 * measured at ~29 MB RSS on the import alone, before a single model is read. That
 * is the whole reason this is a process rather than a module: MAIN supervises it,
 * CONTROL PANEL renders next to it, and neither pays for it.
 *
 * Nothing is decided here. `sidecar.ts` holds the handshake, the request table
 * and the shutdown, against an interface a stub can satisfy, because a file that
 * loads the addon cannot be imported by a test that has to run on any machine.
 * Same split as `protocol.ts` vs `www.ts`.
 *
 * `NAPI_RS_NATIVE_LIBRARY_PATH` must already be in this process's environment
 * when the import below runs — it is set at fork time by `sidecarEnvOverrides`
 * (src/main/child-env.ts). Writing it here would be too late by one module.
 */

import { writeSync } from 'node:fs';

import { createInferenceHost } from '@mlx-node/server/host';

import { NoParentChannelError, resolveParentChannel, runSidecar, sidecarHostOptions } from './sidecar.js';

/** stdout/stderr are async pipes under both transports; an exit right after a write truncates it. */
function logError(line: string): void {
  writeSync(2, `${line}\n`);
}

const channel = (() => {
  try {
    return resolveParentChannel(process);
  } catch (error) {
    if (error instanceof NoParentChannelError) {
      // Running this entry by hand is a legitimate way to reproduce a wedge in a
      // terminal, and `mlx serve` is the supported way to do it. Say so, rather
      // than failing on the first `postMessage` with a TypeError.
      logError('[mlx] dist/inference/index.js must be forked by the desktop supervisor. Try: mlx serve');
    }
    throw error;
  }
})();

// Hoisted so the token the host is gated with is the same one the handshake
// carries. Calling `sidecarHostOptions()` twice would mint two.
const hostOptions = sidecarHostOptions();

void runSidecar({
  channel,
  createHost: () => createInferenceHost(hostOptions),
  authToken: hostOptions.authToken,
  onStopSignal: (handler) => {
    // SIGTERM is what `utilityProcess.kill()` sends and the only stop signal
    // production ever produces. SIGINT is here for the hand-run reproduction
    // above, so Ctrl-C runs the same `close()` rather than orphaning the host's
    // temp root.
    process.on('SIGTERM', handler);
    process.on('SIGINT', handler);
  },
  logError,
  exit: (code) => {
    process.exit(code);
  },
});
