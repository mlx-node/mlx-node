import { constants } from 'node:os';

import { describe, expect, it } from 'vite-plus/test';

import { EXIT_STARTUP_FAILED } from '../src/inference/exit-codes.js';
import { buildChildEnv } from '../src/main/supervisor/env.js';
import {
  assessExit,
  backoffDoublings,
  crashLoopFailures,
  DEFAULT_RESTART_POLICY,
  EMPTY_QUARANTINE,
  FRESH_RESTART_STATE,
  isLoopbackHttpUrl,
  isPermanentExit,
  isServingStatus,
  projectState,
  quarantineAfter,
  requestCaused,
  restartAfter,
  stopCause,
  type EngineFailure,
  type QuarantineState,
  type RestartPolicy,
  type RestartState,
} from '../src/main/supervisor/state.js';
import { parseChildMessage } from '../src/main/supervisor/types.js';

describe('assessExit', () => {
  // The measurement this whole component is built around: from an Electron
  // utilityProcess, `child.kill()` reports code 0 — byte-identical to a
  // sidecar that returned on its own. Both directions must be proven, because
  // a supervisor that reads the code gets exactly one of them right.
  it('calls an UNREQUESTED exit 0 a crash', () => {
    expect(assessExit({ intent: 'run', becameReady: true, exit: { code: 0, signal: null } })).toEqual({
      verdict: 'crash',
      reason: 'exited 0 without being asked to',
    });
  });

  it('calls a REQUESTED exit 0 clean', () => {
    expect(assessExit({ intent: 'stop', becameReady: true, exit: { code: 0, signal: null } })).toMatchObject({
      verdict: 'clean',
    });
  });

  // A stop is a stop whatever status it produced: the process is gone and we
  // asked for that, so respawning it would fight the user.
  it('calls a REQUESTED exit clean even with a non-zero status', () => {
    expect(assessExit({ intent: 'stop', becameReady: true, exit: { code: 7, signal: null } })).toMatchObject({
      verdict: 'clean',
    });
    expect(assessExit({ intent: 'stop', becameReady: true, exit: { code: null, signal: 'SIGKILL' } })).toMatchObject({
      verdict: 'clean',
    });
  });

  it('calls an unrequested non-zero exit a crash and names the status', () => {
    expect(assessExit({ intent: 'run', becameReady: true, exit: { code: 7, signal: null } })).toEqual({
      verdict: 'crash',
      reason: 'exited 7',
    });
    expect(assessExit({ intent: 'run', becameReady: true, exit: { code: 11, signal: null } })).toMatchObject({
      reason: 'exited 11',
    });
  });

  // `signal` is always null under utilityProcess, so it may colour the message
  // and nothing else. If it reached the verdict, production would decide
  // differently from every test that drives the child_process transport.
  it('uses the signal for the reason only', () => {
    expect(assessExit({ intent: 'run', becameReady: true, exit: { code: null, signal: 'SIGSEGV' } })).toEqual({
      verdict: 'crash',
      reason: 'killed by SIGSEGV',
    });
  });

  it('carries the recorded reason when we killed a child that had already failed', () => {
    expect(
      assessExit({
        intent: 'abort',
        becameReady: false,
        exit: { code: 0, signal: null },
        abortReason: 'never announced itself within 50ms',
      }),
    ).toEqual({ verdict: 'crash', reason: 'never announced itself within 50ms' });
  });

  it('says so when the child died before it was ever ready', () => {
    expect(assessExit({ intent: 'run', becameReady: false, exit: { code: 1, signal: null } })).toMatchObject({
      reason: 'exited 1 before it ever became ready',
    });
  });
});

/** Feeds `failures` through `restartAfter`; the delays, then the decision that ended it. */
function run(
  policy: RestartPolicy,
  failures: readonly EngineFailure[],
  from: RestartState = FRESH_RESTART_STATE,
): { delays: number[]; last: ReturnType<typeof restartAfter> } {
  let state = from;
  const delays: number[] = [];
  let last = restartAfter(policy, state, failures[0]);
  for (const failure of failures) {
    last = restartAfter(policy, state, failure);
    state = last.state;
    if (last.decision.action === 'give-up') break;
    delays.push(last.decision.delayMs);
  }
  return { delays, last };
}

const short = (n: number, servedMs = 1_000): EngineFailure[] =>
  Array.from({ length: n }, () => ({ servedMs, permanent: false }));

// Splash dev/tests/server/test_supervision.py `RestartPolicyTests`.
describe('restartAfter (Splash RestartPolicy.after)', () => {
  it('the sixth short-lived failure in a row stops', () => {
    expect(backoffDoublings(DEFAULT_RESTART_POLICY)).toBe(4);
    expect(crashLoopFailures(DEFAULT_RESTART_POLICY)).toBe(6);
    const { delays, last } = run(DEFAULT_RESTART_POLICY, short(6));
    expect(delays).toEqual([0, 5_000, 10_000, 20_000, 40_000]);
    expect(last.decision).toEqual({ action: 'give-up', permanent: false });
    expect(last.state.shortLivedFailures).toBe(6);
    expect(stopCause(DEFAULT_RESTART_POLICY, false, 'exited 1')).toBe(
      'it failed 6 times in a row, each within 60 s of starting (exited 1)',
    );
  });

  it('a full probation of service starts the count anew', () => {
    let { delays, last } = run(DEFAULT_RESTART_POLICY, short(5));
    expect(delays).toEqual([0, 5_000, 10_000, 20_000, 40_000]);
    const served = restartAfter(DEFAULT_RESTART_POLICY, last.state, { servedMs: 60_000, permanent: false });
    expect(served.decision).toEqual({ action: 'restart', delayMs: 0 });
    expect(served.state).toEqual({ streak: 1, shortLivedFailures: 0 });
    ({ delays, last } = run(DEFAULT_RESTART_POLICY, short(6), served.state));
    expect(delays).toEqual([5_000, 10_000, 20_000, 40_000, 60_000]);
    expect(last.decision).toEqual({ action: 'give-up', permanent: false });
    // One millisecond short of the probation is short-lived.
    expect(run(DEFAULT_RESTART_POLICY, short(6, 59_999)).last.decision.action).toBe('give-up');
  });

  it('only a failure a relaunch would meet again stops at once', () => {
    const permanent = restartAfter(DEFAULT_RESTART_POLICY, FRESH_RESTART_STATE, { servedMs: 0, permanent: true });
    expect(permanent.decision).toEqual({ action: 'give-up', permanent: true });
    expect(permanent.state).toEqual(FRESH_RESTART_STATE);
    expect(stopCause(DEFAULT_RESTART_POLICY, true, 'the entry is missing')).toBe('the entry is missing');
    expect(isPermanentExit({ code: EXIT_STARTUP_FAILED, signal: null }, false)).toBe(true);
    expect(isPermanentExit({ code: null, signal: null }, true)).toBe(true);
    expect(isPermanentExit({ code: 1, signal: null }, false)).toBe(false);
    // Splash INTERNAL: a start that failed for another reason backs off as any.
    expect(run(DEFAULT_RESTART_POLICY, short(3, 0)).delays).toEqual([0, 5_000, 10_000]);
  });

  it('refuses a policy whose probation is shorter than its backoff base', () => {
    expect(() => backoffDoublings({ backoffBaseMs: 2, probationMs: 1 })).toThrow(RangeError);
    expect(() => backoffDoublings({ backoffBaseMs: 0, probationMs: 1 })).toThrow(RangeError);
    expect(crashLoopFailures({ backoffBaseMs: 20, probationMs: 40 })).toBe(3);
    expect(crashLoopFailures({ backoffBaseMs: 1, probationMs: 1 })).toBe(2);
  });
});

const fp = (c: string): string => c.repeat(64);

// Splash `QuarantineTests`.
describe('quarantineAfter (Splash Quarantine.after)', () => {
  it('a request started at two failures in a row is refused', () => {
    const [x, y, z] = [fp('a'), fp('b'), fp('c')];
    let q: QuarantineState = EMPTY_QUARANTINE;
    let counted = quarantineAfter(q, new Set([x, y]));
    expect([...counted.refused]).toEqual([]);
    q = counted.state;
    counted = quarantineAfter(q, new Set([x, z]));
    expect([...counted.refused]).toEqual([x]);
    q = counted.state;
    counted = quarantineAfter(q, new Set([y]));
    expect([...counted.refused]).toEqual([]);
    q = counted.state;
    expect(q.refused.has(x)).toBe(true);
    expect(q.refused.has(y)).toBe(false);
    // Never changed in place.
    expect(EMPTY_QUARANTINE.refused.size).toBe(0);
  });
});

describe('requestCaused (Splash request_caused)', () => {
  const signals: Readonly<Record<string, number | undefined>> = constants.signals;
  const code = (name: string): number => signals[name] ?? -1;

  it('only a fault-signal death of a ready child it did not stop counts', () => {
    for (const name of ['SIGSEGV', 'SIGBUS', 'SIGILL', 'SIGABRT', 'SIGFPE', 'SIGTRAP']) {
      expect(requestCaused({ intent: 'run', becameReady: true, code: code(name) })).toBe(true);
    }
    for (const other of [null, 0, 1, code('SIGKILL'), code('SIGTERM'), EXIT_STARTUP_FAILED]) {
      expect(requestCaused({ intent: 'run', becameReady: true, code: other })).toBe(false);
    }
    expect(requestCaused({ intent: 'stop', becameReady: true, code: code('SIGSEGV') })).toBe(false);
    expect(requestCaused({ intent: 'abort', becameReady: true, code: code('SIGSEGV') })).toBe(false);
    expect(requestCaused({ intent: 'run', becameReady: false, code: code('SIGSEGV') })).toBe(false);
  });
});

describe('projectState', () => {
  // Readiness is not enough to report health: a class-(b) swallow leaves
  // /health saying ok and the output wrong.
  it('turns running into lying once a native error has been swallowed', () => {
    expect(projectState('running', 0)).toBe('running');
    expect(projectState('running', 1)).toBe('lying');
  });

  it('does not label a child that is not running', () => {
    for (const lifecycle of ['stopped', 'starting', 'restarting', 'failed'] as const) {
      expect(projectState(lifecycle, 3)).toBe(lifecycle);
    }
  });
});

describe('isServingStatus', () => {
  it('serves on ok and degraded', () => {
    expect(isServingStatus('ok')).toBe(true);
    expect(isServingStatus('degraded')).toBe(true);
  });

  // `loading` means a load holds the writer slot — deriveHealthStatus's own
  // documentation says the supervisor should wait, not restart.
  it('waits on loading and error', () => {
    expect(isServingStatus('loading')).toBe(false);
    expect(isServingStatus('error')).toBe(false);
  });

  // A rung this list has not caught up with must fail loudly (readiness
  // timeout) rather than route traffic into a state we do not understand.
  it('treats an unrecognised status as not serving', () => {
    expect(isServingStatus('wedged')).toBe(false);
    expect(isServingStatus('')).toBe(false);
  });
});

describe('isLoopbackHttpUrl', () => {
  it('accepts the loopback forms a host can actually bind', () => {
    for (const url of ['http://127.0.0.1:51234', 'http://localhost:80', 'http://[::1]:8080', 'http://127.5.6.7:1']) {
      expect(isLoopbackHttpUrl(url)).toBe(true);
    }
  });

  // The handshake URL is the one input that could aim the readiness probe at
  // something that answers 200 {"status":"ok"} while the sidecar is dead —
  // the cheapest possible way to make the supervisor lie.
  it('refuses anything that is not this machine over plain http', () => {
    for (const url of [
      'http://example.com:8080',
      'http://127.0.0.1.evil.com/',
      'https://127.0.0.1:443',
      'file:///etc/passwd',
      'not a url',
      '',
    ]) {
      expect(isLoopbackHttpUrl(url)).toBe(false);
    }
  });
});

describe('parseChildMessage', () => {
  it('reads the three messages the contract defines', () => {
    expect(parseChildMessage({ kind: 'mlx:ready', url: 'http://127.0.0.1:1', authToken: 'tok' })).toEqual({
      kind: 'mlx:ready',
      url: 'http://127.0.0.1:1',
      authToken: 'tok',
    });
    // Fail closed. Every inference route is gated, so a handshake with no token
    // describes an endpoint MAIN could advertise but nothing could ever use;
    // ignoring it surfaces as a readiness timeout with a reason, rather than a
    // running server whose copied client commands are silently broken.
    expect(parseChildMessage({ kind: 'mlx:ready', url: 'http://127.0.0.1:1' })).toBeNull();
    expect(parseChildMessage({ kind: 'mlx:response', id: 4, ok: true, value: 9 })).toEqual({
      kind: 'mlx:response',
      id: 4,
      ok: true,
      value: 9,
    });
    expect(parseChildMessage({ kind: 'mlx:response', id: 4, ok: false, error: 'boom' })).toMatchObject({
      ok: false,
      error: 'boom',
    });
  });

  // Stringifying an object here would put `[object Object]` in front of the
  // user as the reason their request failed.
  it('does not stringify a non-string error', () => {
    expect(parseChildMessage({ kind: 'mlx:response', id: 4, ok: false, error: { code: 500 } })).toMatchObject({
      error: 'unknown error',
    });
  });

  // The sidecar is free to post its own telemetry; anything that is not the
  // contract must be dropped rather than half-read. A `ready` with no url that
  // slipped through would mark the supervisor ready with nothing to poll.
  it('drops malformed and foreign messages', () => {
    for (const message of [
      null,
      'ready',
      42,
      { kind: 'mlx:ready' },
      { kind: 'mlx:ready', url: 1234 },
      { kind: 'mlx:response', ok: true, value: 1 },
      { kind: 'mlx:response', id: '4', ok: true, value: 1 },
      { kind: 'mlx:response', id: 4 },
      { kind: 'telemetry', tokens: 12 },
    ]) {
      expect(parseChildMessage(message)).toBeNull();
    }
  });
});

describe('buildChildEnv', () => {
  const traceFile = '/tmp/mlx-trace.log';
  const journal = { turnJournal: '/tmp/mlx-1.turns', quarantined: new Set<string>() };

  it('never writes to the environment it was given', () => {
    const baseEnv: NodeJS.ProcessEnv = { PATH: '/usr/bin', HOME: '/Users/x' };
    const before = { ...baseEnv };
    buildChildEnv({ baseEnv, enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' }, traceFile, ...journal });
    expect(baseEnv).toEqual(before);
  });

  it('leaves the real process.env alone', () => {
    const before = JSON.stringify(process.env);
    buildChildEnv({
      baseEnv: process.env,
      enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' },
      traceFile,
      ...journal,
    });
    expect(JSON.stringify(process.env)).toBe(before);
  });

  // Engine policy is a DEFAULT — the same semantics `applyEnginePolicy` has
  // in-process, where a value already exported in the user's shell wins.
  it('lets the inherited environment override the engine policy', () => {
    const env = buildChildEnv({
      baseEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '512' },
      enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' },
      traceFile,
      ...journal,
    });
    expect(env.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('512');
  });

  it('applies the engine policy when the shell has no opinion', () => {
    const env = buildChildEnv({
      baseEnv: { PATH: '/usr/bin' },
      enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' },
      traceFile,
      ...journal,
    });
    expect(env).toMatchObject({ PATH: '/usr/bin', MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' });
  });

  it('lets supervisor overrides beat the shell', () => {
    const env = buildChildEnv({
      baseEnv: { MLX_NODE_MODELS_DIR: '/shell' },
      enginePolicyEnv: {},
      overrides: { MLX_NODE_MODELS_DIR: '/picked-in-the-ui' },
      traceFile,
      ...journal,
    });
    expect(env.MLX_NODE_MODELS_DIR).toBe('/picked-in-the-ui');
  });

  // The trace pair is not a preference. A user with MLX_INFERENCE_TRACE_FILE
  // already exported would otherwise point the child's trace at a file nobody
  // is watching, and the `lying` state would silently stop working — which is
  // the same invisible failure class (b) already is.
  it('forces the trace pair above everything, including the shell and overrides', () => {
    const env = buildChildEnv({
      baseEnv: { MLX_INFERENCE_TRACE: '0', MLX_INFERENCE_TRACE_FILE: '/somewhere/else.log' },
      enginePolicyEnv: {},
      overrides: { MLX_INFERENCE_TRACE: 'no', MLX_INFERENCE_TRACE_FILE: '/also/wrong.log' },
      traceFile,
      ...journal,
    });
    expect(env.MLX_INFERENCE_TRACE).toBe('1');
    expect(env.MLX_INFERENCE_TRACE_FILE).toBe(traceFile);
  });

  // A shell value would hide the journal from MAIN or lift a quarantine.
  it('forces the turn journal pair above the shell and overrides', () => {
    const env = buildChildEnv({
      baseEnv: { MLX_TURN_JOURNAL: '/elsewhere', MLX_QUARANTINED: '' },
      enginePolicyEnv: {},
      overrides: { MLX_TURN_JOURNAL: '/also/elsewhere' },
      traceFile,
      turnJournal: '/tmp/mlx-2.turns',
      quarantined: new Set([fp('a'), fp('b')]),
    });
    expect(env.MLX_TURN_JOURNAL).toBe('/tmp/mlx-2.turns');
    expect(env.MLX_QUARANTINED).toBe(`${fp('a')},${fp('b')}`);
  });

  // `fork({ env })` replaces the environment wholesale, so an undefined value
  // copied across as the string "undefined" would be handed to the child as a
  // real setting.
  it('drops unset inherited variables rather than stringifying them', () => {
    const env = buildChildEnv({
      baseEnv: { PATH: '/usr/bin', EMPTY: undefined },
      enginePolicyEnv: {},
      traceFile,
      ...journal,
    });
    expect('EMPTY' in env).toBe(false);
  });
});
