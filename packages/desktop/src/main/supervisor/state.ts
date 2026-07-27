/**
 * The supervisor's decisions, with no process, no clock and no I/O in them.
 *
 * Everything here is a pure function of what was observed, so the parts that
 * are easy to get quietly wrong — above all "was this exit a crash?" — can be
 * driven directly from a test instead of inferred from behaviour.
 */

import type { ChildExit, Lifecycle, SupervisorState } from './types.js';

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

export interface RestartPolicy {
  /**
   * Consecutive crashes tolerated. The Nth crash with `N >= this` gives up, so
   * a value of 5 yields 4 restarts and then a resting `failed` state.
   */
  maxConsecutiveCrashes: number;
  baseDelayMs: number;
  maxDelayMs: number;
  /**
   * A child that stayed READY at least this long has proved it can run, and
   * earns a fresh crash budget. Measured from `ready` to `exit`, not from
   * spawn: a 9B model off a cold USB mmap can take a minute to load, and
   * counting that as uptime would hand a crash-looping sidecar an infinite
   * budget as long as each attempt loaded slowly enough.
   */
  healthyForMs: number;
}

export const DEFAULT_RESTART_POLICY: RestartPolicy = Object.freeze({
  maxConsecutiveCrashes: 5,
  baseDelayMs: 1_000,
  maxDelayMs: 30_000,
  healthyForMs: 60_000,
});

export type RestartDecision =
  | { action: 'restart'; delayMs: number; attempt: number }
  | { action: 'give-up'; attempt: number };

/**
 * What to do after the `consecutiveCrashes`-th consecutive crash.
 *
 * Exponential from `baseDelayMs`, clamped at `maxDelayMs`, with no jitter: a
 * single local child has nothing to stampede against, and jitter would make the
 * give-up boundary untestable for the sake of a property nothing here needs.
 */
export function planRestart(policy: RestartPolicy, consecutiveCrashes: number): RestartDecision {
  if (consecutiveCrashes >= policy.maxConsecutiveCrashes) return { action: 'give-up', attempt: consecutiveCrashes };
  const raw = policy.baseDelayMs * 2 ** (consecutiveCrashes - 1);
  return { action: 'restart', delayMs: Math.min(raw, policy.maxDelayMs), attempt: consecutiveCrashes };
}

/**
 * The crash count to carry forward. A child that was ready long enough resets
 * the run to 1 rather than to 0 — the crash that just happened still counts.
 */
export function nextCrashCount(policy: RestartPolicy, previous: number, readyForMs: number): number {
  return readyForMs >= policy.healthyForMs ? 1 : previous + 1;
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
