/**
 * The supervisor, driven against REAL child processes through the
 * `node:child_process` transport.
 *
 * A fake transport can only prove the supervisor believes its own fake, and
 * exit-code interpretation is the whole crux of this component: the fixture
 * installs a SIGTERM handler that exits 0 — as a real inference host must, to
 * get `InferenceHost.close()` to run — so a requested stop and an unrequested
 * one both arrive as `(code 0, signal null)`, exactly as Electron's
 * `utilityProcess` delivers them.
 */

import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { constants, tmpdir } from 'node:os';
import { join } from 'node:path';
import { PassThrough } from 'node:stream';
import { fileURLToPath } from 'node:url';

import { FLUSH_MS } from '@mlx-node/server/supervision';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { DESKTOP_QUIT_DEADLINE_MS } from '../src/control-panel/shutdown-timings.js';
import { SIDECAR_HOST_CLOSE_TIMEOUT_MS } from '../src/inference/sidecar.js';
import { nodeChildTransport } from '../src/main/supervisor/child-node.js';
import {
  createSupervisor,
  DEFAULT_TIMINGS,
  type Supervisor,
  type SupervisorOptions,
} from '../src/main/supervisor/index.js';
import {
  SidecarRequestError,
  type ChildTransport,
  type SupervisorEvent,
  type SupervisorSnapshot,
} from '../src/main/supervisor/types.js';

const SIDECAR = fileURLToPath(new URL('./fixtures/sidecar.mjs', import.meta.url));

const live: { supervisor: Supervisor; dir: string }[] = [];

afterEach(async () => {
  // Never leave an orphan: every fixture is a real process holding a real port.
  for (const { supervisor, dir } of live.splice(0)) {
    await supervisor.dispose();
    rmSync(dir, { recursive: true, force: true });
  }
});

interface HarnessOptions extends Partial<Omit<SupervisorOptions, 'traceDir'>> {
  mode?: string;
  health?: string;
}

function makeSupervisor(opts: HarnessOptions = {}): { supervisor: Supervisor; events: SupervisorEvent[]; dir: string } {
  const dir = mkdtempSync(join(tmpdir(), 'mlx-supervisor-'));
  const { mode, health, ...rest } = opts;
  const supervisor = createSupervisor({
    entry: SIDECAR,
    traceDir: join(dir, 'trace'),
    transport: nodeChildTransport,
    enginePolicyEnv: { MLX_PAGED_PREFILL_CHUNK_SIZE: '2048' },
    ...rest,
    env: {
      ...(mode !== undefined ? { MLX_TEST_SIDECAR: mode } : {}),
      ...(health !== undefined ? { MLX_TEST_HEALTH: health } : {}),
      ...rest.env,
    },
    // Splash CRASH_LOOP_FAILURES = 2 + ceil(log2(40 / 20)) = 3: at once, 20 ms, stop.
    restart: { backoffBaseMs: 20, probationMs: 40, ...rest.restart },
    timings: {
      readyTimeoutMs: 5_000,
      healthPollMs: 20,
      healthIntervalMs: 100,
      healthRequestTimeoutMs: 1_000,
      requestTimeoutMs: 2_000,
      killGraceMs: 250,
      stdioFlushMs: 30,
      tracePollMs: 25,
      ...rest.timings,
    },
  });
  const events: SupervisorEvent[] = [];
  supervisor.on((event) => events.push(event));
  live.push({ supervisor, dir });
  return { supervisor, events, dir };
}

async function waitFor(
  supervisor: Supervisor,
  predicate: (snapshot: SupervisorSnapshot) => boolean,
  label: string,
  timeoutMs = 10_000,
): Promise<SupervisorSnapshot> {
  if (predicate(supervisor.snapshot())) return supervisor.snapshot();
  return new Promise<SupervisorSnapshot>((resolve, reject) => {
    const timer = setTimeout(() => {
      off();
      reject(new Error(`timed out waiting for ${label}; last snapshot ${JSON.stringify(supervisor.snapshot())}`));
    }, timeoutMs);
    const off = supervisor.on(() => {
      const snapshot = supervisor.snapshot();
      if (!predicate(snapshot)) return;
      clearTimeout(timer);
      off();
      resolve(snapshot);
    });
  });
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * `'settled'`, or `'hung'` after `ms`. A promise that never settles has to fail
 * as an assertion here, not as a two-minute suite timeout with no message.
 */
async function settleWithin(promise: Promise<unknown>, ms: number): Promise<'settled' | 'hung'> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<'hung'>((resolve) => {
    timer = setTimeout(() => resolve('hung'), ms);
  });
  try {
    return await Promise.race([promise.then(() => 'settled' as const), deadline]);
  } finally {
    clearTimeout(timer);
  }
}

describe('start-up', () => {
  it('becomes running on the port the child actually bound', async () => {
    const { supervisor, events } = makeSupervisor();
    await supervisor.start();

    const snapshot = supervisor.snapshot();
    expect(snapshot.state).toBe('running');
    expect(snapshot.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
    expect(snapshot.generation).toBe(1);
    expect(snapshot.pid).toBeTypeOf('number');
    expect(snapshot.health?.status).toBe('ok');
    expect(events.some((event) => event.type === 'ready')).toBe(true);
  });

  // A process that starts but never becomes ready is a failure, not a hang.
  it('fails a child that never announces itself', async () => {
    const { supervisor } = makeSupervisor({ mode: 'no-ready', timings: { readyTimeoutMs: 200 } });
    void supervisor.start().catch(() => {});

    const snapshot = await waitFor(supervisor, (s) => s.lastExit !== null, 'the first exit');
    expect(snapshot.lastExit).toMatchObject({ verdict: 'crash' });
    expect(snapshot.lastExit?.reason).toContain('never announced itself');
  });

  // `loading` means a load holds the writer slot. The supervisor waits — and
  // when it never clears, that is a readiness failure, not a restart trigger
  // halfway through a model load.
  it('does not call a child ready while /health still says loading', async () => {
    const { supervisor } = makeSupervisor({ health: 'loading', timings: { readyTimeoutMs: 300 } });
    void supervisor.start().catch(() => {});

    const snapshot = await waitFor(supervisor, (s) => s.lastExit !== null, 'the first exit');
    expect(snapshot.url).not.toBeNull();
    expect(snapshot.lastExit?.reason).toContain('never reported a serving /health');
  });

  // The handshake URL is the one input that could point the readiness probe at
  // something that answers 200 while the sidecar itself is dead.
  it('refuses a handshake url that is not loopback', async () => {
    const { supervisor } = makeSupervisor({ mode: 'remote-url' });
    void supervisor.start().catch(() => {});

    const snapshot = await waitFor(supervisor, (s) => s.lastExit !== null, 'the first exit');
    expect(snapshot.lastExit?.reason).toContain('non-loopback');
    expect(snapshot.url).toBeNull();
  });

  // A class-(c) failure — a C++ throw unwinding across an `extern "C"` shim,
  // which aborts Rust — leaves nothing but a stderr line. Proven here with the
  // cheapest reproducible stderr-only death there is.
  it('keeps the stderr that explains a crash', async () => {
    const { supervisor } = makeSupervisor({ mode: 'crash-now' });
    void supervisor.start().catch(() => {});

    const snapshot = await waitFor(supervisor, (s) => s.state === 'failed', 'give-up');
    expect(snapshot.lastExit?.stderrTail.join('\n')).toContain('fixture: dying on purpose');
  });
});

describe('shutdown timing', () => {
  it('reserves the SSD cache flush and cleanup after the child host exhausts its HTTP drain budget', () => {
    // host.close() drains HTTP, then flushes the persistent SSD cache for up to
    // FLUSH_MS; a SIGKILL inside that window loses the points a restart resumes
    // from. Keep a margin past both within MAIN's whole-app quit deadline.
    expect(DEFAULT_TIMINGS.killGraceMs - (SIDECAR_HOST_CLOSE_TIMEOUT_MS + FLUSH_MS)).toBeGreaterThanOrEqual(2_000);
    expect(DESKTOP_QUIT_DEADLINE_MS - DEFAULT_TIMINGS.killGraceMs).toBeGreaterThanOrEqual(2_000);
  });
});

describe('intent', () => {
  // ---------------------------------------------------------------------
  // The crux, both directions. Both children exit (0, null) — asserted below
  // so the test cannot silently start relying on a difference in the wire
  // signal — and the verdicts still differ, because only the intent flag
  // decides.
  // ---------------------------------------------------------------------

  it('treats an UNREQUESTED exit 0 as a crash and restarts', async () => {
    const { supervisor, events } = makeSupervisor();
    await supervisor.start();

    await expect(supervisor.request({ op: 'die', code: 0 })).rejects.toBeInstanceOf(SidecarRequestError);

    const crashed = await waitFor(supervisor, (s) => s.lastExit !== null, 'the crash');
    expect(crashed.lastExit).toMatchObject({ verdict: 'crash', code: 0, signal: null });
    expect(crashed.lastExit?.reason).toContain('exited 0 without being asked to');
    expect(events.some((event) => event.type === 'crashed')).toBe(true);

    const restarted = await waitFor(supervisor, (s) => s.state === 'running' && s.generation === 2, 'the respawn');
    expect(restarted.streak).toBe(1);
  });

  it('treats a REQUESTED stop as clean and does not restart', async () => {
    const { supervisor, events } = makeSupervisor();
    await supervisor.start();
    await supervisor.stop();

    const snapshot = supervisor.snapshot();
    // Identical to the unrequested case on the wire.
    expect(snapshot.lastExit).toMatchObject({ verdict: 'clean', code: 0, signal: null });
    expect(snapshot.state).toBe('stopped');
    expect(events.some((event) => event.type === 'crashed')).toBe(false);

    // Well past the 20ms restart backoff: nothing may come back.
    await sleep(200);
    expect(supervisor.snapshot()).toMatchObject({ state: 'stopped', generation: 1 });
  });

  it('escalates to SIGKILL when the child refuses SIGTERM', async () => {
    const { supervisor } = makeSupervisor({ mode: 'ignore-sigterm', timings: { killGraceMs: 100 } });
    await supervisor.start();
    await supervisor.stop();

    expect(supervisor.snapshot().lastExit).toMatchObject({ verdict: 'clean', signal: 'SIGKILL' });
  });

  // ---------------------------------------------------------------------
  // Readiness must never outlive the decision to kill. `markReady` clears the
  // WHOLE generation timer set, and that set is also where termination lives:
  // the SIGTERM→SIGKILL escalation and `stop`'s completion cap. A `/health`
  // that reports `ok` after `stop()` therefore used to disarm the only two
  // things that can end a child which ignores SIGTERM — leaving `stop()`, and
  // every app quit or restart behind it, pending forever.
  //
  // Two windows, one per guard: before the probe is issued, and across a probe
  // already in flight.
  // ---------------------------------------------------------------------

  // `mlx:ready` assigns `url`, emits, and only THEN starts polling — so a
  // `stop()` issued from that emit lands before the first probe exists.
  it('abandons the handshake when a stop lands before the first readiness probe', async () => {
    const { supervisor } = makeSupervisor({ mode: 'ignore-sigterm', timings: { killGraceMs: 100 } });

    const stops: Promise<void>[] = [];
    const off = supervisor.on(() => {
      if (stops.length > 0 || supervisor.snapshot().url === null) return;
      off();
      stops.push(supervisor.stop());
    });

    // Rejects once the stop wins. On the broken path it RESOLVES instead, which
    // is the other half of the same lie.
    void supervisor.start().catch(() => {});
    await waitFor(supervisor, (s) => s.url !== null, 'the handshake');
    expect(stops).toHaveLength(1);

    const pid = supervisor.snapshot().pid;
    try {
      // Well past the 200ms cap (2 × killGraceMs) that must still be armed.
      expect(await settleWithin(Promise.all(stops), 1_500)).toBe('settled');
      expect(supervisor.snapshot().lastExit).toMatchObject({ verdict: 'clean', signal: 'SIGKILL' });
      expect(supervisor.snapshot().state).toBe('stopped');
    } finally {
      // Leave no orphan holding a real port: when this bug is present nothing
      // else will ever kill this child, including `dispose()` in `afterEach`.
      if (pid !== undefined) {
        try {
          process.kill(pid, 'SIGKILL');
        } catch {
          // Already reaped — the expected case.
        }
      }
    }
  });

  // The reviewer's window, entered exactly: the fixture PARKS the first /health
  // and releases it — as `ok` — from its own SIGTERM handler, so the probe is
  // provably in flight when `stop()` runs and provably reports serving after.
  // Nothing here depends on a timing margin.
  it('does not let a readiness probe already in flight cancel a stop', async () => {
    const { supervisor } = makeSupervisor({
      mode: 'ignore-sigterm',
      env: { MLX_TEST_HOLD_HEALTH: '1' },
      // The parked probe must not be aborted by its own request timeout before
      // the stop lands.
      timings: { killGraceMs: 100, healthRequestTimeoutMs: 5_000 },
    });

    const parked = new Promise<void>((resolve) => {
      supervisor.on((event) => {
        if (event.type === 'log' && event.line.includes('holding /health')) resolve();
      });
    });

    void supervisor.start().catch(() => {});
    await parked;

    const pid = supervisor.snapshot().pid;
    // Synchronously: intent becomes `stop` and SIGTERM goes out, which is what
    // releases the parked response.
    const stopping = supervisor.stop();
    try {
      expect(await settleWithin(stopping, 1_500)).toBe('settled');
      expect(supervisor.snapshot().lastExit).toMatchObject({ verdict: 'clean', signal: 'SIGKILL' });
      expect(supervisor.snapshot().state).toBe('stopped');
    } finally {
      if (pid !== undefined) {
        try {
          process.kill(pid, 'SIGKILL');
        } catch {
          // Already reaped — the expected case.
        }
      }
    }
  });

  it('does not count a restart the user asked for as a crash', async () => {
    const { supervisor, events } = makeSupervisor();
    await supervisor.start();
    await supervisor.restart();

    expect(supervisor.snapshot()).toMatchObject({ state: 'running', generation: 2, streak: 0, shortLivedFailures: 0 });
    expect(events.some((event) => event.type === 'crashed')).toBe(false);
  });
});

describe('restart budget', () => {
  // A sidecar that dies instantly in a loop must not spin forever.
  it('gives up at the short failure whose backoff would reach the cap', async () => {
    const { supervisor, events } = makeSupervisor({ mode: 'crash-now' });
    await expect(supervisor.start()).rejects.toThrow(/gave up/);

    const snapshot = supervisor.snapshot();
    expect(snapshot.state).toBe('failed');
    expect(snapshot).toMatchObject({ streak: 2, shortLivedFailures: 3 });
    expect(events.filter((event) => event.type === 'gave-up')).toEqual([
      {
        type: 'gave-up',
        message:
          'stopped restarting inference: it failed 3 times in a row, each within 0.04 s of starting (exited 7 before it ever became ready). Start inference again after fixing the cause.',
      },
    ]);
    expect(snapshot.generation).toBe(3);
    expect(events.filter((event) => event.type === 'crashed')).toHaveLength(3);
    expect(events.some((event) => event.type === 'gave-up')).toBe(true);
    expect(snapshot.lastExit?.reason).toContain('exited 7');

    // `failed` is a resting state: nothing may come back on its own.
    await sleep(200);
    expect(supervisor.snapshot()).toMatchObject({ state: 'failed', generation: 3 });
  });

  // Quitting the app during the backoff must not leave a fork armed to fire
  // after everything else has been torn down.
  it('cancels a pending restart when asked to stop mid-backoff', async () => {
    const { supervisor } = makeSupervisor({ mode: 'crash-now', restart: { backoffBaseMs: 500, probationMs: 1_000 } });
    void supervisor.start().catch(() => {});

    await waitFor(supervisor, (s) => s.state === 'restarting' && s.generation === 2, 'the backoff');
    await supervisor.stop();

    expect(supervisor.snapshot().state).toBe('stopped');
    await sleep(700);
    expect(supervisor.snapshot()).toMatchObject({ state: 'stopped', generation: 2 });
  });

  /**
   * The test above is deliberately not enough. It voids the start promise, so it
   * passes while leaving that promise pending forever — `stop()` on the
   * `child === null` branch used to return without settling the waiters at all.
   *
   * `start()` documents that it "rejects if the supervisor gives up, or if it is
   * stopped before it gets there", and the branch that runs during restart
   * backoff was the one place that broke it.
   */
  it('rejects a start that was still pending when stop() landed mid-backoff', async () => {
    const { supervisor } = makeSupervisor({ mode: 'crash-now', restart: { backoffBaseMs: 500, probationMs: 1_000 } });
    const starting = supervisor.start();
    // Nothing may observe it before the assertion; an unhandled rejection would
    // fail the run on its own terms rather than on this test's.
    starting.catch(() => {});

    await waitFor(supervisor, (s) => s.state === 'restarting' && s.generation === 2, 'the backoff');
    await supervisor.stop();

    // `settleWithin` races on resolution, so a rejecting promise has to be
    // folded to a resolution first — otherwise the helper rethrows and the test
    // fails for the very outcome it is asserting.
    expect(
      await settleWithin(
        starting.catch(() => undefined),
        500,
      ),
    ).toBe('settled');
    await expect(starting).rejects.toThrow(/stopped before it became ready/);
  });

  /**
   * The sharper half of the same bug. A start waiter left pending is not just a
   * hang: if a LATER generation reaches ready, `markReady` resolves it too, so a
   * start the user explicitly stopped reports success — attributed to a
   * different child than the one it asked for.
   */
  it('does not resolve a stopped start when a later generation becomes ready', async () => {
    const { supervisor } = makeSupervisor({ mode: 'crash-now', restart: { backoffBaseMs: 500, probationMs: 1_000 } });
    const cancelled = supervisor.start();
    cancelled.catch(() => {});

    await waitFor(supervisor, (s) => s.state === 'restarting' && s.generation === 2, 'the backoff');
    await supervisor.stop();
    await expect(cancelled).rejects.toThrow();

    // A healthy child this time. The cancelled promise must stay rejected.
    const { supervisor: healthy } = makeSupervisor({});
    await healthy.start();
    expect(healthy.snapshot().state).toBe('running');
    await expect(cancelled).rejects.toThrow(/stopped before it became ready/);
  });
});

describe('the connection token', () => {
  /**
   * The token is what makes the advertised URL usable at all: every inference
   * route is gated, so a client handed only the URL gets 401 for everything. It
   * rides the ready handshake and is reachable only through this accessor.
   */
  it('is available once serving and never appears in the snapshot', async () => {
    const { supervisor } = makeSupervisor();
    expect(supervisor.connectionToken()).toBeNull();

    await supervisor.start();
    expect(supervisor.connectionToken()).toBe('fixture-token');

    // A live credential must not ride the snapshot: it is broadcast to every
    // listener, drawn into the tray, and stringified whole in diagnostics.
    expect(JSON.stringify(supervisor.snapshot())).not.toContain('fixture-token');
  });

  it('is cleared when the child goes away', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    expect(supervisor.connectionToken()).toBe('fixture-token');

    await supervisor.stop();
    // Stale would be worse than absent: a copied connect command that still
    // looks valid but authenticates against a process that no longer exists.
    expect(supervisor.connectionToken()).toBeNull();
  });
});

describe('a spawn that throws synchronously', () => {
  /**
   * `spawn()` sets `starting` and then does real work — `mkdirSync(traceDir)`,
   * then the transport call — none of which was guarded. A throw escaped past
   * `start()`'s own `.catch` (a synchronous throw is not a rejection), left the
   * lifecycle wedged in `starting` with no child and no readiness timer, and
   * from the restart timer became an uncaughtException on MAIN's event loop.
   *
   * `mkdirSync` is the reachable one: a plain file sitting where `traceDir`
   * should be is enough, no stub transport required.
   */
  it('reports a failed spawn instead of throwing out of start()', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-supervisor-trace-'));
    // A FILE where the trace directory has to be: mkdirSync throws EEXIST.
    writeFileSync(join(dir, 'trace'), 'not a directory');

    const { supervisor } = makeSupervisor({ traceDir: join(dir, 'trace') } as HarnessOptions);

    await expect(supervisor.start()).rejects.toThrow();
    expect(supervisor.snapshot().state).not.toBe('starting');
    // The reason has to survive: `assessExit` only reads `abortReason` when the
    // intent is `abort`, so without that the crash report loses the one line
    // that explains what happened.
    expect(supervisor.snapshot().lastExit?.reason ?? '').toContain('failed to spawn');

    rmSync(dir, { recursive: true, force: true });
  });
});

describe('requests', () => {
  it('round-trips a reply', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    await expect(supervisor.request({ op: 'echo', value: { n: 7 } })).resolves.toEqual({ n: 7 });
  });

  it('surfaces a sidecar-side failure as a rejection', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    await expect(supervisor.request({ op: 'fail', message: 'model not found' })).rejects.toThrow('model not found');
  });

  // Nothing in either transport rejects an unanswered message.
  it('rejects a request the sidecar never answers', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    await expect(supervisor.request({ op: 'silent' }, { timeoutMs: 100 })).rejects.toMatchObject({
      failure: 'timeout',
    });
  });

  // Requirement: every pending caller gets a rejection, never a hang. A
  // rejection rather than a failure envelope, because an envelope the caller
  // forgets to check renders a dead sidecar as an empty result.
  it('settles every in-flight request when the child dies', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();

    // Far longer than the child takes to die, so only the supervisor settling
    // it can end this promise — its own timeout cannot.
    const stranded = supervisor.request({ op: 'silent' }, { timeoutMs: 5_000 });
    const dying = supervisor.request({ op: 'die', code: 3 });

    await expect(stranded).rejects.toMatchObject({ failure: 'child-exited' });
    await expect(dying).rejects.toMatchObject({ failure: 'child-exited' });
  });

  it('refuses a request when nothing is running rather than queueing it', async () => {
    const { supervisor } = makeSupervisor();
    await expect(supervisor.request({ op: 'echo', value: 1 })).rejects.toMatchObject({ failure: 'not-running' });
  });
});

describe('spawn failure', () => {
  // The one place a stub is the honest tool rather than a shortcut: no real
  // transport emits `error` on demand, and the two disagree about what follows
  // it — `child_process` may report a process that never existed and then never
  // emit `exit` at all, while Electron documents that `exit` always comes. A
  // supervisor that waits for the `exit` it is owed parks in `starting` forever.
  it('does not park in starting when a transport reports an error with no exit', async () => {
    const transport: ChildTransport = (_spec, events) => {
      setTimeout(() => events.onError(new Error('spawn ENOENT')), 5);
      return {
        pid: undefined,
        stdout: null,
        stderr: null,
        postMessage: () => {},
        kill: () => {},
        forceKill: () => {},
      };
    };
    const { supervisor } = makeSupervisor({
      transport,
      timings: { readyTimeoutMs: 300 },
    });

    await expect(supervisor.start()).rejects.toThrow(/gave up/);
    expect(supervisor.snapshot().lastExit?.reason).toContain('failed to spawn');
  });

  it('ignores old transport and pipe events after an error timeout spawns a replacement', async () => {
    type Events = Parameters<ChildTransport>[1];
    const spawns: { events: Events; pid: number; stdout: PassThrough; stderr: PassThrough }[] = [];
    const transport: ChildTransport = (_spec, events) => {
      const pid = 10_000 + spawns.length;
      const stdout = new PassThrough();
      const stderr = new PassThrough();
      spawns.push({ events, pid, stdout, stderr });
      return {
        pid,
        stdout,
        stderr,
        postMessage: () => {},
        kill: () => events.onExit({ code: 0, signal: null }),
        forceKill: () => events.onExit({ code: null, signal: 'SIGKILL' }),
      };
    };
    const { supervisor } = makeSupervisor({
      transport,
      restart: { backoffBaseMs: 1, probationMs: 2 },
      timings: { readyTimeoutMs: 5_000, stdioFlushMs: 20 },
    });

    const starting = supervisor.start();
    starting.catch(() => {});
    expect(spawns).toHaveLength(1);
    // No newline yet: generation 1's reader must retain this as its own local
    // partial while the synthetic-exit path times out its stdio drain.
    spawns[0].stderr.write('native_error context=stale detail="old');
    expect(supervisor.snapshot().nativeErrors).toEqual([]);

    // child_process may never send exit after this; the supervisor therefore
    // synthesizes one and uses it to advance the restart state machine.
    spawns[0].events.onError(new Error('spawn ENOENT'));
    const replacement = await waitFor(
      supervisor,
      (s) => s.generation === 2 && s.pid !== undefined,
      'the replacement spawn',
    );
    expect(replacement).toMatchObject({
      state: 'starting',
      pid: spawns[1].pid,
      streak: 1,
    });

    // Electron documents that the real exit does follow error. Deliver that
    // old event only after generation 2 reset the global per-child fields.
    // Without a generation fence this immediately clears the replacement
    // handle and starts assessing the old exit with generation 2's state.
    spawns[0].events.onExit({ code: 7, signal: null });
    expect(supervisor.snapshot()).toMatchObject({
      state: 'starting',
      generation: 2,
      pid: spawns[1].pid,
      streak: 1,
    });
    expect(supervisor.snapshot().lastExit).toMatchObject({
      code: null,
      reason: expect.stringContaining('failed to spawn'),
    });

    // The old stderr listener also survives its bounded drain. Its data must
    // not complete the generation-1 partial into generation 2's crash tail or
    // native-error state.
    spawns[0].stderr.write(' child"\n');
    expect(supervisor.snapshot()).toMatchObject({ state: 'starting', generation: 2, nativeErrors: [] });

    // Nor may old end/error events mark generation 2's still-open pipes as
    // drained. Stop generation 2 while both of its streams remain open and
    // prove the stop stays pending until those exact streams end.
    spawns[0].stdout.emit('end');
    spawns[0].stderr.emit('error', new Error('old pipe failed'));
    let stopSettled = false;
    const stopping = supervisor.stop();
    void stopping.then(() => {
      stopSettled = true;
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(stopSettled).toBe(false);

    spawns[1].stdout.end();
    spawns[1].stderr.end();
    await stopping;
    expect(supervisor.snapshot().lastExit?.stderrTail.join('\n')).not.toContain('old child');
    await expect(starting).rejects.toThrow(/stopped/);

    for (const spawn of spawns) {
      spawn.stdout.destroy();
      spawn.stderr.destroy();
    }
  });
});

describe('robustness', () => {
  // This runs on MAIN's event loop, which must never block or die. A tray
  // renderer that throws while painting a state change cannot be allowed to
  // take the supervisor — and the app — with it.
  it('survives a subscriber that throws', async () => {
    const { supervisor } = makeSupervisor();
    supervisor.on(() => {
      throw new Error('listener blew up');
    });

    await supervisor.start();
    await expect(supervisor.request({ op: 'echo', value: 5 })).resolves.toBe(5);
  });
});

describe('lying', () => {
  // The third state. Readiness still says ok — asserted alongside — and the
  // output is wrong anyway.
  it('flips to lying on a swallowed error in the trace file', async () => {
    const { supervisor, events } = makeSupervisor();
    await supervisor.start();

    await supervisor.request({
      op: 'trace',
      lines: ['native_error context=array_eval detail="[metal::malloc] out of memory"'],
    });

    const snapshot = await waitFor(supervisor, (s) => s.state === 'lying', 'the lying state');
    expect(snapshot.health?.status).toBe('ok');
    expect(snapshot.nativeErrors).toEqual([
      expect.objectContaining({ context: 'array_eval', detail: '[metal::malloc] out of memory' }),
    ]);
    expect(events.some((event) => event.type === 'lying')).toBe(true);
  });

  // The second channel: mlx_synchronize / mlx_clear_cache write to stderr
  // unconditionally, so this one survives a fork that lost the trace vars.
  it('flips to lying on a swallowed error on stderr', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();

    await supervisor.request({ op: 'stderr', lines: ['[MLX] Exception in synchronize: command buffer failed'] });

    const snapshot = await waitFor(supervisor, (s) => s.state === 'lying', 'the lying state');
    expect(snapshot.nativeErrors[0]).toMatchObject({ context: 'synchronize' });
  });

  // A state that cries wolf is a state people learn to ignore. Routine
  // [MLX_TRACE] lines share the file and some of them end in `error=`.
  it('stays running through the ordinary trace firehose', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();

    await supervisor.request({
      op: 'trace',
      lines: [
        '[MLX_TRACE] weight_materialize_start arrays=284 total_gb=1.40',
        '[MLX_TRACE] gemma4 attention_paged_kv_write_fallback paged_idx=0 seq_len=8 error=shape mismatch',
      ],
    });
    await supervisor.request({ op: 'stderr', lines: ['[MLX] Exception in compile_clear_cache: nope'] });
    await sleep(120);

    expect(supervisor.snapshot()).toMatchObject({ state: 'running', nativeErrors: [] });
  });

  it('reassembles a swallowed error split across two writes', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();

    await supervisor.request({ op: 'trace-split', head: 'native_error context=async_ev' });
    await sleep(80);
    expect(supervisor.snapshot().state).toBe('running');

    await supervisor.request({ op: 'trace-tail', tail: 'al detail="out of memory"' });
    const snapshot = await waitFor(supervisor, (s) => s.state === 'lying', 'the lying state');
    expect(snapshot.nativeErrors[0]).toMatchObject({ context: 'async_eval' });
  });

  // A restart gets a fresh trace file, so the new child is not judged by the
  // old one's corruption.
  it('clears when the child is replaced', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    await supervisor.request({ op: 'trace', lines: ['native_error context=eval detail="x"'] });
    await waitFor(supervisor, (s) => s.state === 'lying', 'the lying state');

    await supervisor.restart();
    expect(supervisor.snapshot()).toMatchObject({ state: 'running', nativeErrors: [] });
  });
});

describe('fork-time environment', () => {
  it('starts built JavaScript without inheriting the parent loader flags', async () => {
    const before = [...process.execArgv];
    process.execArgv.push('--import', 'data:text/javascript,throw new Error("parent loader reached child")');
    try {
      const { supervisor } = makeSupervisor();
      await supervisor.start();
      expect(await supervisor.request({ op: 'echo', value: 'ready' })).toBe('ready');
    } finally {
      process.execArgv.splice(0, process.execArgv.length, ...before);
    }
  });

  // These latch through a Rust OnceLock on first read. A value absent at fork
  // time can never be supplied afterwards, and writing it later succeeds
  // silently while changing nothing.
  it('hands the child its trace pair and the engine policy', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();

    const seen = await supervisor.request<Record<string, string | null>>({
      op: 'env',
      names: ['MLX_INFERENCE_TRACE', 'MLX_INFERENCE_TRACE_FILE', 'MLX_PAGED_PREFILL_CHUNK_SIZE'],
    });
    expect(seen.MLX_INFERENCE_TRACE).toBe('1');
    expect(seen.MLX_INFERENCE_TRACE_FILE).toBe(supervisor.snapshot().traceFile);
    expect(seen.MLX_PAGED_PREFILL_CHUNK_SIZE).toBe('2048');
  });

  it('never mutates the supervising process environment', async () => {
    const before = JSON.stringify(process.env);
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    await supervisor.request({ op: 'echo', value: 1 });
    expect(JSON.stringify(process.env)).toBe(before);
  });
});

describe('a missing entry (Splash EngineLaunchFailed)', () => {
  // A missing entry stays missing; any other spawn error may not.
  it('stops at once with the reason instead of relaunching', async () => {
    const { supervisor, events } = makeSupervisor({ entry: '/nonexistent/mlx-inference/index.js' });
    await expect(supervisor.start()).rejects.toThrow(/gave up/);

    expect(supervisor.snapshot()).toMatchObject({ state: 'failed', generation: 1 });
    const gaveUp = events.filter((event) => event.type === 'gave-up');
    expect(gaveUp).toHaveLength(1);
    expect(gaveUp[0]).toMatchObject({
      message: expect.stringContaining(
        'stopped restarting inference: failed to spawn: the inference entry /nonexistent/mlx-inference/index.js is missing',
      ),
    });
  });
});

describe('quarantine (Splash Quarantine)', () => {
  const SIGSEGV: number = (constants.signals as Readonly<Record<string, number | undefined>>).SIGSEGV ?? -1;
  const [A, B, C, X] = ['a', 'b', 'c', 'd'].map((c) => c.repeat(64));
  const JOURNAL_ENV = ['MLX_QUARANTINED', 'MLX_TURN_JOURNAL'];

  async function crashWithTurns(supervisor: Supervisor, generation: number, fingerprints: readonly string[]) {
    await waitFor(supervisor, (s) => s.state === 'running' && s.generation === generation, `generation ${generation}`);
    for (const fp of fingerprints) await supervisor.request({ op: 'journal', fp });
    await expect(supervisor.request({ op: 'segv' })).rejects.toBeInstanceOf(SidecarRequestError);
  }

  it('the child_process transport reports a signal death as its number', async () => {
    const { supervisor } = makeSupervisor();
    await supervisor.start();
    await expect(supervisor.request({ op: 'segv' })).rejects.toBeInstanceOf(SidecarRequestError);
    const crashed = await waitFor(supervisor, (s) => s.lastExit !== null, 'the crash');
    expect(crashed.lastExit).toMatchObject({ verdict: 'crash', code: SIGSEGV, signal: 'SIGSEGV' });
    expect(crashed.lastExit?.reason).toBe('killed by SIGSEGV');
  });

  it('a turn in flight at two crashes in a row is refused by the next child', async () => {
    const { supervisor, events, dir } = makeSupervisor({ restart: { backoffBaseMs: 20, probationMs: 10_000 } });
    await supervisor.start();
    await crashWithTurns(supervisor, 1, [A, B]);
    await crashWithTurns(supervisor, 2, [A, C]);

    await waitFor(supervisor, (s) => s.state === 'running' && s.generation === 3, 'generation 3');
    expect(events.filter((event) => event.type === 'quarantined')).toEqual([
      { type: 'quarantined', refused: 1, total: 1 },
    ]);
    const third = await supervisor.request<Record<string, string | null>>({ op: 'env', names: JOURNAL_ENV });
    expect(third.MLX_QUARANTINED).toBe(A);
    expect(third.MLX_TURN_JOURNAL).toBe(join(dir, 'trace', 'inference-3.turns'));
    expect(readdirSync(third.MLX_TURN_JOURNAL ?? '')).toEqual([]);

    // Exit 1 is no request's doing: neither adds nor clears.
    await supervisor.request({ op: 'journal', fp: A });
    await expect(supervisor.request({ op: 'die', code: 1 })).rejects.toBeInstanceOf(SidecarRequestError);
    await waitFor(supervisor, (s) => s.state === 'running' && s.generation === 4, 'generation 4');
    const fourth = await supervisor.request<Record<string, string | null>>({ op: 'env', names: JOURNAL_ENV });
    expect(fourth.MLX_QUARANTINED).toBe(A);

    // A user restart is Splash's server restart.
    await supervisor.restart();
    const fifth = await supervisor.request<Record<string, string | null>>({ op: 'env', names: JOURNAL_ENV });
    expect(supervisor.snapshot().generation).toBe(5);
    expect(fifth.MLX_QUARANTINED).toBe('');
    for (const generation of [1, 2, 3, 4]) {
      expect(existsSync(join(dir, 'trace', `inference-${generation}.turns`))).toBe(false);
    }
  });

  it('a journal left by an earlier app run is never read', async () => {
    const { supervisor, events, dir } = makeSupervisor({ restart: { backoffBaseMs: 20, probationMs: 10_000 } });
    for (const generation of [1, 2]) {
      const stale = join(dir, 'trace', `inference-${generation}.turns`);
      mkdirSync(stale, { recursive: true });
      writeFileSync(join(stale, '1'), X);
    }
    await supervisor.start();
    await crashWithTurns(supervisor, 1, []);
    await crashWithTurns(supervisor, 2, []);
    await waitFor(supervisor, (s) => s.state === 'running' && s.generation === 3, 'generation 3');

    expect(events.some((event) => event.type === 'quarantined')).toBe(false);
    const third = await supervisor.request<Record<string, string | null>>({ op: 'env', names: JOURNAL_ENV });
    expect(third.MLX_QUARANTINED).toBe('');
  });
});
