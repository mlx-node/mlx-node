/**
 * A real sidecar for the supervisor tests — real process, real HTTP, real exits.
 *
 * It implements the contract documented on `ChildToSupervisor` in
 * src/main/supervisor/types.ts: announce `mlx:ready` with the port actually
 * bound, answer `mlx:request` with `mlx:response`, and handle SIGTERM by
 * exiting 0.
 *
 * That SIGTERM handler is the whole point of driving these tests with real
 * processes. Without it, `child.kill()` under `node:child_process` reports
 * `(code=null, signal='SIGTERM')` and telling a requested stop from an
 * unrequested one would be trivial — but Electron's `utilityProcess.kill()`
 * reports plain `0`, exactly like a sidecar that returned on its own. A real
 * inference host handles SIGTERM (it has to, to run `InferenceHost.close()`),
 * so this fixture does too, and both directions then arrive as `(0, null)`:
 * indistinguishable, which is the case the supervisor must get right.
 *
 * Behaviour selected by MLX_TEST_SIDECAR.
 */

import { appendFileSync, writeSync } from 'node:fs';
import { createServer } from 'node:http';

const MODE = process.env.MLX_TEST_SIDECAR ?? 'ok';

function post(message) {
  process.send?.(message);
}

/** stdout/stderr are async pipes; an exit right after a `write` truncates it. */
function errSync(text) {
  writeSync(2, `${text}\n`);
}

if (MODE === 'crash-now') {
  errSync('fixture: dying on purpose');
  process.exit(Number(process.env.MLX_TEST_EXIT_CODE ?? 7));
}

if (MODE === 'no-ready') {
  // Alive, listening to nothing, announcing nothing.
  setInterval(() => {}, 60_000);
} else if (MODE === 'remote-url') {
  post({ kind: 'mlx:ready', url: 'http://example.com:8080', authToken: 'fixture-token' });
  setInterval(() => {}, 60_000);
} else {
  const status = process.env.MLX_TEST_HEALTH ?? 'ok';
  // Park the FIRST /health and release it only on SIGTERM. That is the only way
  // to put a readiness probe in flight ACROSS a `stop()` deterministically: the
  // supervisor's fetch cannot complete until the test has asked it to stop.
  const holdHealth = process.env.MLX_TEST_HOLD_HEALTH === '1';
  let heldHealth = null;

  function answerHealth(res) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status, uptimeMs: 1, pid: process.pid }));
  }

  const server = createServer((req, res) => {
    if (req.url === '/health') {
      if (holdHealth && heldHealth === null) {
        heldHealth = res;
        // The test's witness that the probe is parked, not merely scheduled.
        errSync('fixture: holding /health');
        return;
      }
      answerHealth(res);
      return;
    }
    res.writeHead(404).end();
  });
  server.listen(0, '127.0.0.1', () => {
    post({ kind: 'mlx:ready', url: `http://127.0.0.1:${server.address().port}`, authToken: 'fixture-token' });
  });

  if (MODE !== 'ignore-sigterm') {
    process.on('SIGTERM', () => {
      server.close();
      process.exit(0);
    });
  } else {
    process.on('SIGTERM', () => {
      errSync('fixture: refusing SIGTERM');
      // Answer the parked probe now, so the supervisor gets a serving /health
      // for a child it has ALREADY decided to kill.
      if (heldHealth !== null) {
        const res = heldHealth;
        heldHealth = null;
        answerHealth(res);
      }
    });
  }

  process.on('message', (message) => {
    if (message?.kind !== 'mlx:request') return;
    const { id, payload } = message;
    const reply = (value) => post({ kind: 'mlx:response', id, ok: true, value });

    switch (payload?.op) {
      case 'echo':
        reply(payload.value);
        break;
      case 'fail':
        post({ kind: 'mlx:response', id, ok: false, error: String(payload.message) });
        break;
      case 'silent':
        // Deliberately no reply: the supervisor's per-request timeout is the
        // only thing that can settle this one.
        break;
      case 'env':
        reply(Object.fromEntries(payload.names.map((name) => [name, process.env[name] ?? null])));
        break;
      case 'trace':
        // Impersonates `mlx_trace_native_error` in crates/mlx-sys/src/mlx_common.h.
        appendFileSync(process.env.MLX_INFERENCE_TRACE_FILE, `${payload.lines.join('\n')}\n`);
        reply(true);
        break;
      case 'trace-split':
        // Two writes, one line: proves the watcher buffers a partial tail.
        appendFileSync(process.env.MLX_INFERENCE_TRACE_FILE, payload.head);
        reply(true);
        break;
      case 'trace-tail':
        appendFileSync(process.env.MLX_INFERENCE_TRACE_FILE, `${payload.tail}\n`);
        reply(true);
        break;
      case 'stderr':
        for (const line of payload.lines) errSync(line);
        reply(true);
        break;
      case 'die':
        // No reply, on purpose: the pending request must be settled by the
        // supervisor when the process disappears, not by this process.
        errSync(`fixture: exiting ${payload.code}`);
        process.exit(payload.code);
        break;
      default:
        post({ kind: 'mlx:response', id, ok: false, error: `unknown op ${String(payload?.op)}` });
    }
  });
}
