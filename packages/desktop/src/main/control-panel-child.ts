/**
 * `BrokerDeps` over Electron — fork the CONTROL PANEL utilityProcess and mint channels.
 *
 * `import { MessageChannelMain, utilityProcess } from 'electron'` cannot resolve
 * outside an Electron process, so nothing here is reachable from a plain Node
 * test. It is kept to plumbing for that reason; every rule lives in `broker.ts`.
 * Same split as `child-utility.ts` vs `supervisor/state.ts`.
 */

import { MessageChannelMain, utilityProcess, type WebContents } from 'electron';

import type { ControlPanelChild, BrokerDeps, BrokerPort } from './broker.js';
import { CONTROL_PANEL_PORT_CHANNEL } from './window-policy.js';

export interface ControlPanelChildOptions {
  /** Absolute path to `dist/control-panel/index.js`. */
  entry: string;
  /** The child's complete environment. See `supervisor/env.ts` for why complete. */
  env: Readonly<Record<string, string>>;
  /** Forwarded verbatim; CONTROL PANEL's stderr is the only place its faults surface. */
  onLog(stream: 'stdout' | 'stderr', line: string): void;
}

export function electronBrokerDeps(options: ControlPanelChildOptions): BrokerDeps<WebContents> {
  return {
    spawn(events): ControlPanelChild {
      const child = utilityProcess.fork(options.entry, [], {
        env: { ...options.env },
        stdio: 'pipe',
        // Its own entry in Activity Monitor and in crash reports, instead of an
        // anonymous "Helper (Plugin)" the user cannot attribute.
        serviceName: 'mlx-control-panel',
      });

      const log = (stream: 'stdout' | 'stderr', line: string): void => {
        options.onLog(stream, line);
      };
      attachLineReader(child.stdout, 'stdout', log);
      attachLineReader(child.stderr, 'stderr', log);

      child.on('exit', (code: number) => {
        events.onExit(code);
      });
      child.on('error', (type: string, location: string) => {
        // A non-continuable V8 fault. Electron documents that `exit` follows, so
        // this only needs to be visible, not acted on.
        options.onLog('stderr', `[mlx] control panel fault: ${type} at ${location}`);
      });

      return {
        sendPort(port: BrokerPort): void {
          // No `message` worth sending: the port IS the message, and the CONTROL PANEL
          // entry reads `event.ports[0]` and ignores the rest.
          child.postMessage(null, [port as unknown as Electron.MessagePortMain]);
        },
        kill(): void {
          child.kill();
        },
        forceKill(): void {
          // `utilityProcess.kill()` cannot escalate, so SIGKILL has to go through
          // the pid. `pid` is undefined before spawn completes and after exit,
          // and in both cases there is nothing left to kill.
          const pid = child.pid;
          if (pid === undefined) return;
          try {
            process.kill(pid, 'SIGKILL');
          } catch {
            /* already reaped */
          }
        },
      };
    },

    createChannel(): { port1: BrokerPort; port2: BrokerPort } {
      return new MessageChannelMain();
    },

    sendToRenderer(target: WebContents, port: BrokerPort, generation: number): void {
      // Throws on a destroyed WebContents, which is what `broker.ts` treats as
      // "this renderer is gone" — do NOT soften it into a no-op.
      target.postMessage(CONTROL_PANEL_PORT_CHANNEL, generation, [port as unknown as Electron.MessagePortMain]);
    },

    isRendererAlive(target: WebContents): boolean {
      return !target.isDestroyed();
    },

    report(): void {
      // Replaced by the caller; a default that swallows keeps this factory
      // usable without one.
    },

    now: () => Date.now(),

    delay(fn: () => void, ms: number): () => void {
      const timer = setTimeout(fn, ms);
      // Must never be the reason the process is still alive during quit.
      timer.unref();
      return () => {
        clearTimeout(timer);
      };
    },
  };
}

function attachLineReader(
  stream: NodeJS.ReadableStream | null,
  name: 'stdout' | 'stderr',
  onLog: (stream: 'stdout' | 'stderr', line: string) => void,
): void {
  if (stream === null) return;
  let buffered = '';
  stream.on('data', (chunk: Buffer | string) => {
    buffered += typeof chunk === 'string' ? chunk : chunk.toString('utf8');
    const lines = buffered.split('\n');
    buffered = lines.pop() ?? '';
    for (const line of lines) onLog(name, line);
  });
}
