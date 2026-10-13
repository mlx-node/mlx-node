/**
 * The MLX Agent binary's update channel, kept separate from the app updater:
 * `electron-updater` can only replace the whole signed bundle, while the
 * agent ships on its own release cadence from mlx-node/grok-build. The
 * download itself lives in the CLI (`mlx agent update`), run through a
 * utilityProcess — MAIN never blocks on network or disk — and `--check`
 * hands back the status as one JSON line.
 *
 * Like every other MAIN module, nothing Electron-shaped is imported here so
 * the state machine stays testable; `index.ts` injects the fork.
 */

import { UPDATE_INTERVAL_MS, type UpdatePresentation } from './updates.js';

export type AgentUpdateStatus = 'idle' | 'checking' | 'available' | 'installing' | 'current' | 'error';

/** Subset of `AgentUpdateStatus` (CLI) the menu needs; unknown fields pass through. */
export interface AgentUpdateCheck {
  updateAvailable?: boolean;
  latest?: string;
  current?: string | null;
}

/** Runs `mlx agent update <args…>` in a child process; resolves with the parsed status. */
export type AgentUpdateRunner = (args: string[]) => Promise<AgentUpdateCheck>;

export interface AgentUpdater {
  status(): AgentUpdateStatus;
  /** Newest tag the last check saw — shown in the "Install …" label. */
  latest(): string | null;
  start(): void;
  check(): void;
  install(): void;
  stop(): void;
}

export function presentAgentUpdate(status: AgentUpdateStatus, latest: string | null = null): UpdatePresentation {
  switch (status) {
    case 'idle':
      return { label: 'Check for MLX Agent Update…', enabled: true };
    case 'checking':
      return { label: 'Checking for MLX Agent Update…', enabled: false };
    case 'available':
      return {
        label: latest === null ? 'Install MLX Agent Update…' : `Install MLX Agent ${shortTag(latest)}…`,
        enabled: true,
      };
    case 'installing':
      return { label: 'Downloading MLX Agent Update…', enabled: false };
    case 'current':
      return { label: 'MLX Agent Up to Date — Check Again…', enabled: true };
    case 'error':
      return { label: 'MLX Agent Update Failed — Try Again…', enabled: true };
  }
}

/** `mlx-agent-v0.3.0` → `0.3.0` for menu copy. */
function shortTag(tag: string): string {
  return tag.startsWith('mlx-agent-v') ? tag.slice('mlx-agent-v'.length) : tag;
}

/** @internal A piped child stdio stream — the subset {@link collectAgentUpdateResult} reads. */
export interface AgentUpdateChildStream {
  on(event: 'data', listener: (chunk: Buffer | string) => void): unknown;
  once(event: 'close', listener: () => void): unknown;
}

/**
 * @internal The piped `utilityProcess` (or Node child) running `mlx agent
 * update`, narrowed to what the output collection needs so the Electron
 * import stays in `index.ts` and tests can fake it.
 */
export interface AgentUpdateChild {
  stdout?: AgentUpdateChildStream | null;
  stderr?: AgentUpdateChildStream | null;
  on(event: 'exit', listener: (code: number) => void): unknown;
  on(event: 'error', listener: (...detail: unknown[]) => void): unknown;
}

/**
 * Collect `mlx agent update` output from a piped child. The promise settles
 * only when the process has exited AND every piped stream has closed — Node
 * (and therefore utilityProcess) may fire `exit` while buffered `data` is
 * still in flight, so parsing at `exit` can see an empty or truncated status
 * line. `expectStatus` marks a `--check` run: it MUST end in the status JSON
 * — absent or malformed output rejects rather than reporting "current". The
 * install run also ends with the status line when the CLI emits it; older CLIs
 * print human text only, so no JSON still counts as success.
 */
export function collectAgentUpdateResult(child: AgentUpdateChild, expectStatus: boolean): Promise<AgentUpdateCheck> {
  return new Promise((resolvePromise, reject) => {
    let stdout = '';
    let stderr = '';
    let openStreams = 0;
    let exitCode: number | undefined;
    let settled = false;

    const finish = (code: number): void => {
      if (settled) return;
      settled = true;
      if (code !== 0) {
        reject(new Error(`mlx agent update exited ${code}: ${stderr.trim() || stdout.trim()}`));
        return;
      }
      // The install path prints human lines; the last JSON-looking line is
      // the status when one exists.
      const line = stdout
        .trim()
        .split('\n')
        .reverse()
        .find((l) => l.trimStart().startsWith('{'));
      if (line === undefined) {
        if (expectStatus) reject(new Error('mlx agent update exited 0 without a status line'));
        else resolvePromise({});
        return;
      }
      try {
        resolvePromise(JSON.parse(line) as AgentUpdateCheck);
      } catch (error) {
        reject(
          new Error(`mlx agent update returned malformed status: ${error instanceof Error ? error.message : String(error)}`),
        );
      }
    };

    const maybeFinish = (): void => {
      if (exitCode !== undefined && openStreams === 0) finish(exitCode);
    };

    const tap = (stream: AgentUpdateChildStream | null | undefined, sink: (chunk: string) => void): void => {
      if (stream === null || stream === undefined) return;
      openStreams++;
      stream.on('data', (chunk) => sink(chunk.toString()));
      stream.once('close', () => {
        openStreams--;
        maybeFinish();
      });
    };
    tap(child.stdout, (chunk) => {
      stdout += chunk;
    });
    tap(child.stderr, (chunk) => {
      stderr += chunk;
    });
    child.on('exit', (code) => {
      exitCode = code;
      maybeFinish();
    });
    child.on('error', (...detail: unknown[]) => {
      if (settled) return;
      settled = true;
      reject(new Error(`mlx agent update child crashed: ${detail.join(' at ')}`));
    });
  });
}

export function createAgentUpdater(options: {
  run: AgentUpdateRunner;
  onChange(): void;
  report(error: unknown): void;
}): AgentUpdater {
  let status: AgentUpdateStatus = 'idle';
  let latest: string | null = null;
  let started = false;
  let stopped = false;
  // One child at a time: a click during a check must not fork a second CLI
  // writing the same managed binary.
  let inFlight = false;
  let timer: NodeJS.Timeout | null = null;

  function change(next: AgentUpdateStatus): void {
    status = next;
    if (!stopped) options.onChange();
  }

  function drive(args: string[], apply: (result: AgentUpdateCheck) => AgentUpdateStatus): void {
    change(args.length === 0 ? 'installing' : 'checking');
    inFlight = true;
    void options
      .run(args)
      .then((result) => {
        if (typeof result.latest === 'string') latest = result.latest;
        change(apply(result));
      })
      .catch((error: unknown) => {
        options.report(error);
        change('error');
      })
      .finally(() => {
        inFlight = false;
      });
  }

  function check(): void {
    if (stopped || inFlight) return;
    drive(['--check'], (result) => (result.updateAvailable === true ? 'available' : 'current'));
  }

  return {
    status: () => status,
    latest: () => latest,
    start(): void {
      if (started || stopped) return;
      started = true;
      timer = setInterval(() => {
        check();
      }, UPDATE_INTERVAL_MS);
      timer.unref();
      check();
    },
    check,
    install(): void {
      if (stopped || inFlight || status !== 'available') return;
      // A failed install keeps its own 'error' state rather than reverting to
      // 'available'; the label already offers retry. The status line the CLI
      // ends with is authoritative: a fallback install landing below latest
      // reports updateAvailable — claiming 'current' would hide it until the
      // next periodic check. Absent JSON (older CLI) stays 'current'.
      drive([], (result) => (result.updateAvailable === true ? 'available' : 'current'));
    },
    stop(): void {
      stopped = true;
      if (timer !== null) clearInterval(timer);
      timer = null;
    },
  };
}
