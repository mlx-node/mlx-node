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

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { nodeChildTransport } from '../src/main/supervisor/child-node.js';
import { createSupervisor, type Supervisor, type SupervisorOptions } from '../src/main/supervisor/index.js';
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
    restart: { maxConsecutiveCrashes: 3, baseDelayMs: 20, maxDelayMs: 40, healthyForMs: 10_000, ...rest.restart },
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
    const { supervisor } = makeSupervisor({ entry: join(SIDECAR, '..', 'does-not-exist.mjs') });
    void supervisor.start().catch(() => {});

    const snapshot = await waitFor(supervisor, (s) => s.state === 'failed', 'give-up');
    expect(snapshot.lastExit?.stderrTail.join('\n')).toContain('MODULE_NOT_FOUND');
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
    expect(restarted.consecutiveCrashes).toBe(1);
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

  it('does not count a restart the user asked for as a crash', async () => {
    const { supervisor, events } = makeSupervisor();
    await supervisor.start();
    await supervisor.restart();

    expect(supervisor.snapshot()).toMatchObject({ state: 'running', generation: 2, consecutiveCrashes: 0 });
    expect(events.some((event) => event.type === 'crashed')).toBe(false);
  });
});

describe('restart budget', () => {
  // A sidecar that dies instantly in a loop must not spin forever.
  it('gives up after the configured number of consecutive crashes', async () => {
    const { supervisor, events } = makeSupervisor({ mode: 'crash-now' });
    await expect(supervisor.start()).rejects.toThrow(/gave up/);

    const snapshot = supervisor.snapshot();
    expect(snapshot.state).toBe('failed');
    expect(snapshot.consecutiveCrashes).toBe(3);
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
    const { supervisor } = makeSupervisor({ mode: 'crash-now', restart: { baseDelayMs: 500 } });
    void supervisor.start().catch(() => {});

    await waitFor(supervisor, (s) => s.state === 'restarting', 'the backoff');
    await supervisor.stop();

    expect(supervisor.snapshot().state).toBe('stopped');
    await sleep(700);
    expect(supervisor.snapshot()).toMatchObject({ state: 'stopped', generation: 1 });
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
      restart: { maxConsecutiveCrashes: 1 },
      timings: { readyTimeoutMs: 300 },
    });

    await expect(supervisor.start()).rejects.toThrow(/gave up/);
    expect(supervisor.snapshot().lastExit?.reason).toContain('failed to spawn');
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
