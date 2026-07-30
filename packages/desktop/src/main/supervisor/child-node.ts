/**
 * `ChildTransport` over `node:child_process.fork`.
 *
 * Two jobs, both load-bearing:
 *
 *   1. it is the escape hatch if the GPU watchdog (ml-explore/mlx#3267) turns
 *      out to fire inside an Electron `utilityProcess` after all — Phase 0 did
 *      not reproduce it, which is not the same as ruling it out;
 *   2. it is the transport the tests drive with real processes. Exit-code
 *      interpretation is the crux of the supervisor, and a fake transport can
 *      only prove that the supervisor believes its own fake.
 *
 * It is NOT the default: `createSupervisor` requires an explicit transport so
 * the fallback cannot be shipped by omission.
 */

import { fork } from 'node:child_process';

import type { ChildEvents, ChildHandle, ChildSpec, ChildTransport } from './types.js';

export const nodeChildTransport: ChildTransport = (spec: ChildSpec, events: ChildEvents): ChildHandle => {
  const child = fork(spec.modulePath, [...spec.args], {
    env: { ...spec.env },
    ...(spec.cwd !== undefined ? { cwd: spec.cwd } : {}),
    // stdin closed, stdout/stderr piped, IPC on fd 3. The pipes are not
    // optional: a class-(c) failure — a C++ throw unwinding across an
    // `extern "C"` shim, which aborts Rust — leaves nothing behind but the
    // stderr line, and the `[MLX] Exception in ...` swallows are stderr-only
    // by construction.
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
    // V8 structured-clone semantics instead of the default JSON round-trip, so
    // this transport preserves the same message shapes `utilityProcess`'s
    // `postMessage` does. With `json` a `Map` or an `undefined` property would
    // survive one path and not the other, and the tests would be proving the
    // wrong serialiser.
    serialization: 'advanced',
  });

  child.on('message', (message) => {
    events.onMessage(message);
  });
  child.on('error', (error) => {
    events.onError(error);
  });
  child.on('exit', (code, signal) => {
    events.onExit({ code, signal });
  });

  return {
    get pid(): number | undefined {
      return child.pid;
    },
    stdout: child.stdout,
    stderr: child.stderr,
    postMessage(message: unknown): void {
      // `connected` can go false between the check and the send, and `send`
      // throws ERR_IPC_CHANNEL_CLOSED when it does. A message lost to a dying
      // child is expected; an exception thrown into the main process's event
      // loop from a background write is not.
      if (!child.connected) return;
      try {
        child.send(message as never);
      } catch {
        /* channel closed underneath us */
      }
    },
    kill(): void {
      child.kill('SIGTERM');
    },
    forceKill(): void {
      child.kill('SIGKILL');
    },
  };
};
