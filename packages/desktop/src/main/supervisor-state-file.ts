/**
 * `~/.mlx-node/desktop/supervisor.json`: the supervisor state the Control
 * Panel cannot learn from the inference heartbeat alone (`stopped` vs
 * `restarting` vs `failed` vs `lying`). MAIN writes it; it never sends
 * anything over the port.
 *
 * Metadata only: exit reasons, stderr, native error detail and the sidecar
 * URL are never written.
 */

import { mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { resolveMlxNodeHome } from '@mlx-node/server/host/paths';

import { MAX_NATIVE_ERRORS } from './supervisor/index.js';
import type { SupervisorSnapshot, SupervisorState } from './supervisor/types.js';

export type SupervisorHealthStatus = 'ok' | 'loading' | 'degraded' | 'error';

export interface SupervisorStateFile {
  version: 1;
  writtenAt: number;
  /** MAIN's pid. A reader ignores the file when this pid is dead. */
  mainPid: number;
  state: SupervisorState;
  /** Sidecar pid. */
  pid: number | null;
  generation: number;
  /** `snapshot.streak`: restarts in a row since a child served a full probation. */
  streak: number;
  /** `snapshot.shortLivedFailures`: failures in a row within probation; the supervisor stops restarting at its cap. */
  shortLivedFailures: number;
  lastExit: { verdict: 'clean' | 'crash'; code: number | null; signal: string | null; atMs: number } | null;
  healthStatus: SupervisorHealthStatus | null;
  /** `snapshot.nativeErrors.length`. The supervisor keeps at most {@link MAX_NATIVE_ERRORS}. */
  nativeErrorCount: number;
  /** True when the count sits at that cap; the real number may be higher. */
  nativeErrorsCapped: boolean;
  /** Shim name only (`array_eval`, …): 1–32 chars of a-z, 0-9, `_`; else null. */
  lastNativeErrorContext: string | null;
  lastNativeErrorAt: number | null;
}

const STATES: readonly SupervisorState[] = ['stopped', 'starting', 'running', 'restarting', 'failed', 'lying'];
const HEALTH: readonly SupervisorHealthStatus[] = ['ok', 'loading', 'degraded', 'error'];

export function supervisorStateFilePath(): string {
  return join(resolveMlxNodeHome(), 'desktop', 'supervisor.json');
}

function member<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  for (const candidate of allowed) if (candidate === value) return candidate;
  return null;
}

/** 1–32 chars of `a-z`, `0-9`, `_`. */
function safeShimName(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0 || value.length > 32) return null;
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (!(c >= 97 && c <= 122) && !(c >= 48 && c <= 57) && c !== 95) return null;
  }
  return value;
}

/** A signal name such as `SIGKILL`: 1–16 chars of `A-Z`, `0-9`. */
function safeSignal(value: unknown): string | null {
  if (typeof value !== 'string' || value.length === 0 || value.length > 16) return null;
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (!(c >= 65 && c <= 90) && !(c >= 48 && c <= 57)) return null;
  }
  return value;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function projectSupervisorState(
  snapshot: SupervisorSnapshot,
  mainPid: number,
  now: number,
): SupervisorStateFile {
  const errors = snapshot.nativeErrors;
  const last = errors.length > 0 ? errors[errors.length - 1] : undefined;
  const exit = snapshot.lastExit;
  return {
    version: 1,
    writtenAt: now,
    mainPid,
    state: member(snapshot.state, STATES) ?? 'stopped',
    pid: finiteOrNull(snapshot.pid),
    generation: finiteOrNull(snapshot.generation) ?? 0,
    streak: finiteOrNull(snapshot.streak) ?? 0,
    shortLivedFailures: finiteOrNull(snapshot.shortLivedFailures) ?? 0,
    lastExit:
      exit === null
        ? null
        : {
            verdict: exit.verdict === 'clean' ? 'clean' : 'crash',
            code: finiteOrNull(exit.code),
            signal: safeSignal(exit.signal),
            atMs: finiteOrNull(exit.atMs) ?? 0,
          },
    healthStatus: member(snapshot.health?.status, HEALTH),
    nativeErrorCount: errors.length,
    nativeErrorsCapped: errors.length >= MAX_NATIVE_ERRORS,
    lastNativeErrorContext: last === undefined ? null : safeShimName(last.context),
    lastNativeErrorAt: last === undefined ? null : finiteOrNull(last.observedAtMs),
  };
}

export interface SupervisorStateWriterIo {
  mkdir: (path: string, opts: { recursive: true; mode: number }) => Promise<unknown>;
  writeFile: (path: string, data: string) => Promise<void>;
  rename: (from: string, to: string) => Promise<void>;
  unlink: (path: string) => Promise<void>;
}

const DEFAULT_IO: SupervisorStateWriterIo = { mkdir, writeFile, rename, unlink };

/** Latest-wins writer: one async tmp + rename in flight; a newer state replaces any queued one. */
export function createSupervisorStateWriter(
  path: string,
  io: SupervisorStateWriterIo = DEFAULT_IO,
): { write(state: SupervisorStateFile): void; dispose(): Promise<void> } {
  const tmp = `${path}.tmp`;
  let next: SupervisorStateFile | null = null;
  let inFlight: Promise<void> | null = null;
  let disposed = false;

  async function flush(): Promise<void> {
    try {
      while (next !== null && !disposed) {
        const state = next;
        next = null;
        try {
          await io.mkdir(dirname(path), { recursive: true, mode: 0o700 });
          await io.writeFile(tmp, JSON.stringify(state));
          await io.rename(tmp, path);
        } catch {
          // Best effort: the next state retries.
        }
      }
    } finally {
      inFlight = null;
    }
  }

  return {
    write(state) {
      if (disposed) return;
      next = state;
      inFlight ??= flush();
    },
    async dispose() {
      disposed = true;
      next = null;
      if (inFlight !== null) await inFlight;
      await io.unlink(path).catch(() => undefined);
      await io.unlink(tmp).catch(() => undefined);
    },
  };
}
