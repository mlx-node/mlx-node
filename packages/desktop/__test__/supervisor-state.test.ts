import { describe, expect, it } from 'vite-plus/test';

import { buildChildEnv } from '../src/main/supervisor/env.js';
import {
  assessExit,
  DEFAULT_RESTART_POLICY,
  isLoopbackHttpUrl,
  isServingStatus,
  nextCrashCount,
  planRestart,
  projectState,
  type RestartPolicy,
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

describe('planRestart', () => {
  const policy: RestartPolicy = { maxConsecutiveCrashes: 5, baseDelayMs: 100, maxDelayMs: 400, healthyForMs: 1_000 };

  it('backs off exponentially from the base delay', () => {
    expect(planRestart(policy, 1)).toEqual({ action: 'restart', delayMs: 100, attempt: 1 });
    expect(planRestart(policy, 2)).toEqual({ action: 'restart', delayMs: 200, attempt: 2 });
    expect(planRestart(policy, 3)).toEqual({ action: 'restart', delayMs: 400, attempt: 3 });
  });

  it('clamps at the ceiling instead of growing without bound', () => {
    expect(planRestart(policy, 4)).toEqual({ action: 'restart', delayMs: 400, attempt: 4 });
  });

  // The boundary, stated exactly: maxConsecutiveCrashes 5 means four restarts
  // and then a resting `failed` state. A sidecar that dies instantly must not
  // spin forever.
  it('gives up on the Nth crash, not the Nth+1', () => {
    expect(planRestart(policy, 5)).toEqual({ action: 'give-up', attempt: 5 });
    expect(planRestart(policy, 6)).toEqual({ action: 'give-up', attempt: 6 });
  });

  it('ships a give-up by default rather than an unbounded loop', () => {
    expect(Number.isFinite(DEFAULT_RESTART_POLICY.maxConsecutiveCrashes)).toBe(true);
    expect(planRestart(DEFAULT_RESTART_POLICY, DEFAULT_RESTART_POLICY.maxConsecutiveCrashes).action).toBe('give-up');
  });
});

describe('nextCrashCount', () => {
  const policy: RestartPolicy = { maxConsecutiveCrashes: 5, baseDelayMs: 100, maxDelayMs: 400, healthyForMs: 1_000 };

  it('accumulates while the child keeps dying quickly', () => {
    expect(nextCrashCount(policy, 0, 10)).toBe(1);
    expect(nextCrashCount(policy, 3, 999)).toBe(4);
  });

  // Reset to 1, not 0: the crash that just happened still counts.
  it('grants a fresh budget to a child that stayed ready long enough', () => {
    expect(nextCrashCount(policy, 4, 1_000)).toBe(1);
    expect(nextCrashCount(policy, 4, 5_000)).toBe(1);
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

  it('never writes to the environment it was given', () => {
    const baseEnv: NodeJS.ProcessEnv = { PATH: '/usr/bin', HOME: '/Users/x' };
    const before = { ...baseEnv };
    buildChildEnv({ baseEnv, enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' }, traceFile });
    expect(baseEnv).toEqual(before);
  });

  it('leaves the real process.env alone', () => {
    const before = JSON.stringify(process.env);
    buildChildEnv({ baseEnv: process.env, enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' }, traceFile });
    expect(JSON.stringify(process.env)).toBe(before);
  });

  // Engine policy is a DEFAULT — the same semantics `applyEnginePolicy` has
  // in-process, where a value already exported in the user's shell wins.
  it('lets the inherited environment override the engine policy', () => {
    const env = buildChildEnv({
      baseEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '512' },
      enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' },
      traceFile,
    });
    expect(env.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('512');
  });

  it('applies the engine policy when the shell has no opinion', () => {
    const env = buildChildEnv({
      baseEnv: { PATH: '/usr/bin' },
      enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' },
      traceFile,
    });
    expect(env).toMatchObject({ PATH: '/usr/bin', MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' });
  });

  it('lets supervisor overrides beat the shell', () => {
    const env = buildChildEnv({
      baseEnv: { MLX_NODE_MODELS_DIR: '/shell' },
      enginePolicyEnv: {},
      overrides: { MLX_NODE_MODELS_DIR: '/picked-in-the-ui' },
      traceFile,
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
    });
    expect(env.MLX_INFERENCE_TRACE).toBe('1');
    expect(env.MLX_INFERENCE_TRACE_FILE).toBe(traceFile);
  });

  // `fork({ env })` replaces the environment wholesale, so an undefined value
  // copied across as the string "undefined" would be handed to the child as a
  // real setting.
  it('drops unset inherited variables rather than stringifying them', () => {
    const env = buildChildEnv({
      baseEnv: { PATH: '/usr/bin', EMPTY: undefined },
      enginePolicyEnv: {},
      traceFile,
    });
    expect('EMPTY' in env).toBe(false);
  });
});
