/**
 * Phase 0 spike host — Electron main.
 *
 * Forks spike/workload.mjs as a utilityProcess, shows a VISIBLE window, and
 * repaints it on every streamed token. The point is to reproduce the conditions
 * of ml-explore/mlx#3267 (heavy Metal compute contending with WindowServer
 * compositing) inside Electron specifically, since an Electron app composites
 * continuously by construction.
 *
 * We record how the child dies. A GPU watchdog kill is uncatchable inside the
 * child, so the exit code observed HERE is the measurement.
 *
 *   run:  yarn workspace @mlx-node/desktop spike:electron
 */

import { app, BrowserWindow, utilityProcess } from 'electron';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { appendFileSync, mkdirSync } from 'node:fs';

const HERE = dirname(fileURLToPath(import.meta.url));
const LOG_DIR = process.env.MLX_SPIKE_LOG_DIR ?? join(HERE, '.spike-logs');
mkdirSync(LOG_DIR, { recursive: true });
const LOG_FILE = join(LOG_DIR, `electron-${process.pid}.jsonl`);

function log(entry) {
  const line = JSON.stringify({ ...entry, t: Date.now() });
  appendFileSync(LOG_FILE, line + '\n');
  if (entry.event !== 'token') console.log(line);
}

/**
 * Main-process event-loop lag. If the three-process split is doing its job this
 * stays low even while the child is saturating the GPU.
 */
function startLagProbe() {
  const INTERVAL = 100;
  let maxLag = 0;
  let last = Date.now();
  const timer = setInterval(() => {
    const now = Date.now();
    const lag = now - last - INTERVAL;
    if (lag > maxLag) maxLag = lag;
    last = now;
  }, INTERVAL);
  timer.unref?.();
  return { maxLag: () => maxLag };
}

let win;
let child;
const started = Date.now();

// DO NOT rewrite this as a top-level `await app.whenReady()`. It DEADLOCKS:
// Electron waits for the ESM main module to finish evaluating before it emits
// `ready`, so awaiting `ready` at module scope waits on an event that cannot
// fire until the await returns. Verified on Electron 43.2.0 — the process hangs
// with no output and has to be SIGTERMed. `void` here is deliberate, not lazy.
void app.whenReady().then(() => {
  const lag = startLagProbe();

  win = new BrowserWindow({
    width: 900,
    height: 640,
    show: true, // MUST be visible — the watchdog only fires with the display awake
    title: 'mlx-node — Phase 0 watchdog spike',
    webPreferences: {
      preload: join(HERE, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  // Never swallow this. A loadFile rejection is the classic packaged-app
  // failure — blank window, no error — and under app:// it is exactly what a
  // missing hashed asset or a broken protocol handler looks like.
  win.loadFile(join(HERE, 'index.html')).catch((err) => {
    log({ event: 'window-load-failed', message: String(err?.stack ?? err) });
  });

  child = utilityProcess.fork(join(HERE, 'workload.mjs'), [], {
    serviceName: 'mlx-inference-spike',
    stdio: 'pipe',
    env: {
      ...process.env,
      // Launcher policy from `mlx launch claude` — the Rust default is 0 (no chunking).
      MLX_PAGED_PREFILL_CHUNK_SIZE: process.env.MLX_PAGED_PREFILL_CHUNK_SIZE ?? '2048',
      // The ONLY way to observe class-(b) swallowed C++ errors, which otherwise
      // produce wrong output with no JS error and a healthy-looking server.
      MLX_INFERENCE_TRACE: '1',
      MLX_INFERENCE_TRACE_FILE: join(LOG_DIR, `mlx-trace-${process.pid}.log`),
    },
  });

  child.stdout?.on('data', (b) => log({ event: 'child-stdout', chunk: String(b).trim() }));
  child.stderr?.on('data', (b) => log({ event: 'child-stderr', chunk: String(b).trim() }));

  child.on('message', (msg) => {
    if (msg?.event === 'token') {
      win?.webContents.send('spike:token', msg.text ?? '');
      return;
    }
    log({ ...msg, mainMaxLagMs: lag.maxLag() });
    if (msg?.event === 'done') finish(0, null, lag);
  });

  child.on('exit', (code) => {
    log({
      event: 'child-exit',
      code,
      note: 'code 0 is ALSO what an explicit kill() reports — not proof of a clean exit',
    });
    finish(code, null, lag);
  });
});

function finish(code, signal, lag) {
  log({
    event: 'VERDICT',
    childExitCode: code,
    childSignal: signal,
    ranForMs: Date.now() - started,
    mainMaxLagMs: lag.maxLag(),
    watchdogSuspected: code !== 0 && code !== null,
    logFile: LOG_FILE,
  });
  setTimeout(() => app.quit(), 250);
}

app.on('window-all-closed', () => app.quit());
