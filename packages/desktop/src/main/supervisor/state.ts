/**
 * The supervisor's decisions, with no process, no clock and no I/O in them.
 *
 * Everything here is a pure function of what was observed, so the parts that
 * are easy to get quietly wrong — above all "was this exit a crash?" — can be
 * driven directly from a test instead of inferred from behaviour.
 */

import { constants } from 'node:os';

import { EXIT_STARTUP_FAILED } from '../../inference/exit-codes.js';
import type { ChildExit, Lifecycle, SupervisorState } from './types.js';

const osSignals: Readonly<Record<string, number | undefined>> = constants.signals;

/**
 * Why the supervisor believes the current child should be running, or not.
 * Set BEFORE the kill that produces the exit, never derived from the exit.
 */
export type ChildIntent =
  /** It should be up. Any exit is unrequested. */
  | 'run'
  /** The user (or app shutdown) asked for it to stop. */
  | 'stop'
  /** We are killing it because it already failed — the failure is recorded. */
  | 'abort';

export interface ExitAssessment {
  verdict: 'clean' | 'crash';
  /** One line, for the tray. */
  reason: string;
}

/**
 * Decide whether a child that just exited crashed.
 *
 * **The verdict reads `intent` and nothing else, and that is the entire point
 * of this module.** From an Electron `utilityProcess`, measured in Phase 0:
 *
 *     process.exit(7)   ->  7
 *     uncaught throw    ->  1
 *     SIGSEGV           -> 11
 *     child.kill()      ->  0     <- indistinguishable from a clean voluntary exit
 *
 * A supervisor that reads `code === 0` as "clean" therefore treats a sidecar
 * that returned on its own — an entry module that fell off the end of `main`,
 * an `InferenceHost` that closed itself, a `process.exit(0)` in a shutdown path
 * that ran for the wrong reason — as a normal stop, and never restarts it. The
 * user gets a tray icon that says everything is fine and a server that is gone.
 *
 * `exit.signal` is used for the reason string only. It is always `null` under
 * `utilityProcess`, so a verdict that consulted it would be decided differently
 * in production than in any test that drives the `child_process` transport.
 */
export function assessExit(input: {
  intent: ChildIntent;
  /** Did this child ever reach `running`? Changes the reason, never the verdict. */
  becameReady: boolean;
  exit: ChildExit;
  /** Set when `intent` is `abort`: why we killed it. */
  abortReason?: string;
}): ExitAssessment {
  const { intent, becameReady, exit } = input;
  if (intent === 'stop') return { verdict: 'clean', reason: `stopped on request (${describeExit(exit)})` };
  if (intent === 'abort') {
    return { verdict: 'crash', reason: input.abortReason ?? `terminated after a failure (${describeExit(exit)})` };
  }

  const stage = becameReady ? '' : ' before it ever became ready';
  if (exit.signal !== null) return { verdict: 'crash', reason: `killed by ${exit.signal}${stage}` };
  if (exit.code === 0) {
    // Worth spelling out in the UI: this is the one crash whose exit status
    // looks like success, and the message is what stops someone "fixing" the
    // supervisor by trusting the code.
    return { verdict: 'crash', reason: `exited 0 without being asked to${stage}` };
  }
  return { verdict: 'crash', reason: `exited ${exit.code ?? 'with no status'}${stage}` };
}

function describeExit(exit: ChildExit): string {
  if (exit.signal !== null) return `signal ${exit.signal}`;
  return exit.code === null ? 'no status' : `code ${exit.code}`;
}

/**
 * Splash `RestartPolicy` (server/supervision.py), with its two inputs:
 * `BACKOFF_BASE_SECONDS` and `kProbation` (runtime/Supervision.hpp), which is
 * both `Thresholds.short_lived` and `Thresholds.backoff_cap`.
 */
export interface RestartPolicy {
  backoffBaseMs: number;
  probationMs: number;
}

export const DEFAULT_RESTART_POLICY: RestartPolicy = Object.freeze({
  backoffBaseMs: 5_000,
  probationMs: 60_000,
});

/** Splash `BACKOFF_DOUBLINGS`. */
export function backoffDoublings(policy: RestartPolicy): number {
  if (!(policy.backoffBaseMs > 0) || !(policy.probationMs >= policy.backoffBaseMs)) {
    throw new RangeError(
      `restart policy needs 0 < backoffBaseMs <= probationMs (got ${policy.backoffBaseMs}, ${policy.probationMs})`,
    );
  }
  return Math.ceil(Math.log2(policy.probationMs / policy.backoffBaseMs));
}

/** Splash `CRASH_LOOP_FAILURES`: the short-lived failure in a row that stops the relaunches. */
export function crashLoopFailures(policy: RestartPolicy): number {
  return 2 + backoffDoublings(policy);
}

/** Splash `RestartPolicy.streak` and `.short_lived_failures`. */
export interface RestartState {
  streak: number;
  shortLivedFailures: number;
}

export const FRESH_RESTART_STATE: RestartState = Object.freeze({ streak: 0, shortLivedFailures: 0 });

/** Splash `EngineFailure.served_seconds` and `permanent()`. */
export interface EngineFailure {
  /** Time served with nothing loading, 0 for a child that never became ready. */
  servedMs: number;
  permanent: boolean;
}

export type RestartDecision = { action: 'restart'; delayMs: number } | { action: 'give-up'; permanent: boolean };

/** Splash `RestartPolicy.after`. */
export function restartAfter(
  policy: RestartPolicy,
  state: RestartState,
  failure: EngineFailure,
): { decision: RestartDecision; state: RestartState } {
  const doublings = backoffDoublings(policy);
  if (failure.permanent) return { decision: { action: 'give-up', permanent: true }, state };
  let { streak, shortLivedFailures } = state;
  if (failure.servedMs >= policy.probationMs) {
    streak = 0;
    shortLivedFailures = 0;
  } else {
    shortLivedFailures += 1;
    if (shortLivedFailures === 2 + doublings) {
      return { decision: { action: 'give-up', permanent: false }, state: { streak, shortLivedFailures } };
    }
  }
  streak += 1;
  const delayMs =
    streak === 1 ? 0 : Math.min(policy.probationMs, policy.backoffBaseMs * 2 ** Math.min(streak - 2, doublings));
  return { decision: { action: 'restart', delayMs }, state: { streak, shortLivedFailures } };
}

/** The cause in Splash's `_stop_message`, for a failure `restartAfter` gave up on. */
export function stopCause(policy: RestartPolicy, permanent: boolean, reason: string): string {
  if (permanent) return reason;
  return `it failed ${crashLoopFailures(policy)} times in a row, each within ${policy.probationMs / 1000} s of starting (${reason})`;
}

/**
 * Whether a child exit is permanent (Splash `permanent()`): `createHost`
 * refused its environment (`EXIT_STARTUP_FAILED`), or the entry module is
 * missing, the one launch failure Splash's `EngineLaunchFailed` names.
 */
export function isPermanentExit(exit: ChildExit, entryMissing: boolean): boolean {
  return entryMissing || exit.code === EXIT_STARTUP_FAILED;
}

/** Splash `Quarantine`. Sets are replaced, never changed in place. */
export interface QuarantineState {
  suspects: ReadonlySet<string>;
  refused: ReadonlySet<string>;
}

export const EMPTY_QUARANTINE: QuarantineState = Object.freeze({
  suspects: new Set<string>(),
  refused: new Set<string>(),
});

/** Splash `Quarantine.after`: count a failure a request may have caused. */
export function quarantineAfter(
  quarantine: QuarantineState,
  started: ReadonlySet<string>,
): { state: QuarantineState; refused: ReadonlySet<string> } {
  const refused = new Set<string>();
  for (const fingerprint of quarantine.suspects) {
    if (started.has(fingerprint) && !quarantine.refused.has(fingerprint)) refused.add(fingerprint);
  }
  return {
    state: {
      suspects: new Set(started),
      refused: new Set([...quarantine.refused, ...refused]),
    },
    refused,
  };
}

const FAULT_SIGNAL_NAMES = ['SIGSEGV', 'SIGBUS', 'SIGILL', 'SIGABRT', 'SIGFPE', 'SIGTRAP'] as const;

/** Splash `FAULT_SIGNALS`, as this platform's signal numbers. */
const FAULT_SIGNAL_CODES: ReadonlySet<number> = new Set(
  FAULT_SIGNAL_NAMES.map((name) => osSignals[name]).filter((code): code is number => typeof code === 'number'),
);

/**
 * Splash `request_caused`: a fault-signal death of a child that became ready
 * and that we did not stop or kill. Both transports report a signal death as
 * its number in `code`.
 */
export function requestCaused(input: { intent: ChildIntent; becameReady: boolean; code: number | null }): boolean {
  return input.intent === 'run' && input.becameReady && input.code !== null && FAULT_SIGNAL_CODES.has(input.code);
}

/**
 * Fold the class-(b) trace signal into the reported state.
 *
 * Only a `running` child can lie: a stopped or restarting one is not producing
 * output for anyone to trust. The flag is sticky for the life of the child —
 * once `mlx_array_eval` has swallowed one C++ exception, every array that
 * depended on it is suspect and there is no evidence available that would clear
 * the suspicion. It resets when the child does, because a restart gets a fresh
 * trace file.
 */
export function projectState(lifecycle: Lifecycle, nativeErrorCount: number): SupervisorState {
  return lifecycle === 'running' && nativeErrorCount > 0 ? 'lying' : lifecycle;
}

/**
 * `/health` statuses that mean "this sidecar can serve traffic".
 *
 * From `deriveHealthStatus` in packages/server/src/health.ts: `loading` means
 * a load holds the writer slot and the supervisor should WAIT, not restart;
 * `error` means the last load failed with nothing resident. `degraded` is
 * saturated or contended, but it is answering.
 */
const READY_STATUSES: ReadonlySet<string> = new Set(['ok', 'degraded']);

/**
 * An unrecognised status is NOT ready, on purpose. If the server grows a rung
 * this list has not caught up with, the failure is a readiness timeout the user
 * can see, rather than the supervisor routing traffic into a state it does not
 * understand.
 */
export function isServingStatus(status: string): boolean {
  return READY_STATUSES.has(status);
}

/**
 * Refuse a handshake URL that does not point at this machine.
 *
 * The sidecar reports the port it bound, which makes it the authority on where
 * `/health` lives — and also the one input that could aim the readiness probe
 * somewhere that answers `200 {"status":"ok"}` while the sidecar itself is
 * dead. A remote URL is the cheapest possible way to make the supervisor lie,
 * so it is refused rather than polled.
 */
export function isLoopbackHttpUrl(value: string): boolean {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return false;
  }
  if (url.protocol !== 'http:') return false;
  const host = url.hostname;
  if (host === 'localhost' || host === '[::1]' || host === '::1') return true;
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host);
}
