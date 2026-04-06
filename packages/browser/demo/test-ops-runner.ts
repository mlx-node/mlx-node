/**
 * WebGPU Op Test Runner
 *
 * Spawns a test-worker that loads WASM + WebGPU, runs individual ops,
 * and reports pass/fail for each one.
 */

const statusEl = document.getElementById('status')!;
const resultsEl = document.getElementById('results')!;

function log(msg: string, cls: string = '') {
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = msg;
  resultsEl.appendChild(line);
  // Also log to console for Playwright capture
  console.log(`[TEST] ${msg}`);
}

const worker = new Worker(
  new URL('../src/test-worker.ts', import.meta.url),
  { type: 'module' }
);

worker.onmessage = (e: MessageEvent) => {
  const { type, name, passed, error, message, summary } = e.data;
  switch (type) {
    case 'status':
      statusEl.textContent = message;
      break;
    case 'result':
      if (passed) {
        log(`PASS  ${name}`, 'pass');
      } else {
        log(`FAIL  ${name}: ${error}`, 'fail');
      }
      break;
    case 'done':
      statusEl.textContent = summary;
      log(`\n${summary}`);
      break;
    case 'error':
      statusEl.textContent = `Error: ${message}`;
      log(`ERROR: ${message}`, 'fail');
      break;
  }
};

worker.onerror = (e) => {
  const msg = e.message || 'unknown';
  const file = (e as any).filename || '';
  const line = (e as any).lineno || 0;
  statusEl.textContent = `Worker error: ${msg}`;
  log(`WORKER ERROR: ${msg} at ${file}:${line}`, 'fail');
  console.error('[TEST RUNNER] worker error event:', e);
};

// Start tests
worker.postMessage({
  type: 'init',
  wasmUrl: new URL('/mlx-core.opt.wasm', location.href).href,
});
