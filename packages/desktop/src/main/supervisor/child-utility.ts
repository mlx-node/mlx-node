/**
 * `ChildTransport` over Electron's `utilityProcess` — the production path.
 *
 * `import { utilityProcess } from 'electron'` cannot resolve outside an
 * Electron process, so nothing in this file is reachable from a plain Node
 * test. It is therefore kept as close to a bare `fork` call as the interface
 * allows, and every decision that could be wrong lives in `state.ts`,
 * `trace.ts` and `index.ts` where it can be proven. Same split as
 * `protocol.ts` vs `www.ts`.
 *
 * Three differences from `child-node.ts` that the interface exists to absorb:
 *
 *   - `exit` carries a bare `code`, never a signal, so `ChildExit.signal` is
 *     always `null` here. Nothing may classify on it.
 *   - `kill()` takes no argument — SIGTERM on POSIX, always — so escalation has
 *     to go around the handle via the pid.
 *   - `error` means a non-continuable V8 fault and is documented to be followed
 *     by `exit`; the `child_process` transport can emit `error` with no `exit`
 *     at all. `index.ts` tolerates both.
 */

import { utilityProcess } from 'electron';

import type { ChildEvents, ChildHandle, ChildSpec, ChildTransport } from './types.js';

export const utilityChildTransport: ChildTransport = (spec: ChildSpec, events: ChildEvents): ChildHandle => {
  const child = utilityProcess.fork(spec.modulePath, [...spec.args], {
    env: { ...spec.env },
    ...(spec.cwd !== undefined ? { cwd: spec.cwd } : {}),
    stdio: 'pipe',
    // Shows up in Activity Monitor and in crash reports as its own entry,
    // instead of an anonymous "Helper (Plugin)" the user cannot attribute.
    serviceName: 'mlx-inference',
  });

  child.on('message', (message: unknown) => {
    events.onMessage(message);
  });
  child.on('error', (type: string, location: string, report: string) => {
    // The Node.js diagnostic report is large; the type and location are what
    // identify the fault, and the report body would swamp the tray log.
    events.onError(new Error(`${type} at ${location} (diagnostic report: ${report.length} bytes)`));
  });
  child.on('exit', (code: number) => {
    events.onExit({ code, signal: null });
  });

  return {
    get pid(): number | undefined {
      return child.pid;
    },
    stdout: child.stdout,
    stderr: child.stderr,
    postMessage(message: unknown): void {
      try {
        child.postMessage(message);
      } catch {
        /* channel closed underneath us */
      }
    },
    kill(): void {
      child.kill();
    },
    forceKill(): void {
      // `utilityProcess.kill()` cannot escalate, so the only way to SIGKILL a
      // sidecar wedged inside a Metal call is through its pid. `pid` is
      // `undefined` before spawn completes and after `exit`, and in both of
      // those cases there is nothing left to kill.
      const pid = child.pid;
      if (pid === undefined) return;
      try {
        process.kill(pid, 'SIGKILL');
      } catch {
        /* already reaped */
      }
    },
  };
};
