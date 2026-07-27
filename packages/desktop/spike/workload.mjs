/**
 * Phase 0 spike workload — sustained MLX generation.
 *
 * Runs identically in two hosts so the two can be compared directly:
 *   - bare Node        `node spike/workload.mjs`
 *   - Electron child   forked via utilityProcess from spike/main.mjs
 *
 * What we are trying to provoke is ml-explore/mlx#3267: the macOS Metal GPU
 * watchdog killing the process because MLX command buffers block WindowServer
 * compositing. The reported trigger is single GPU ops in the ~1.2s range, and it
 * only reproduces with the display awake — so we deliberately use a long prompt
 * (big prefill ops) and stream every token out so the host can repaint.
 *
 * A watchdog kill is uncatchable. This process will simply die. That is the
 * signal — the supervising side records the exit, not us.
 */

import { homedir } from 'node:os';
import { join } from 'node:path';

const MODEL = process.env.MLX_SPIKE_MODEL ?? join(homedir(), '.mlx-node/models/gemma-4-12b-it');
const DURATION_MS = Number(process.env.MLX_SPIKE_DURATION_MS ?? 240_000);
const MAX_NEW_TOKENS = Number(process.env.MLX_SPIKE_MAX_NEW_TOKENS ?? 256);
const PROMPT_REPEATS = Number(process.env.MLX_SPIKE_PROMPT_REPEATS ?? 220);

/** utilityProcess gives us parentPort; bare node does not. */
const parentPort = process.parentPort;
const host = parentPort ? 'electron-utilityprocess' : 'bare-node';

function report(event) {
  const msg = { ...event, host, t: Date.now() };
  if (parentPort) parentPort.postMessage(msg);
  else console.log(JSON.stringify(msg));
}

/** Long prompt => large prefill ops, which is where the ~1.2s single-op trigger lives. */
function buildPrompt() {
  const para =
    'The Metal command buffer scheduler interleaves compute work with display ' +
    'compositing on Apple silicon, and the unified memory architecture means GPU ' +
    'and CPU contend for the same bandwidth. ';
  return `Summarize the following text in exactly one sentence.\n\n${para.repeat(PROMPT_REPEATS)}`;
}

async function main() {
  const { loadSession } = await import('@mlx-node/lm');

  const loadStart = Date.now();
  const session = await loadSession(MODEL);
  report({ event: 'loaded', model: MODEL, loadMs: Date.now() - loadStart });

  const prompt = buildPrompt();
  const deadline = Date.now() + DURATION_MS;
  let round = 0;
  let totalTokens = 0;

  while (Date.now() < deadline) {
    round += 1;
    const roundStart = Date.now();
    let chunks = 0;
    let tokens = 0;
    let firstTokenMs = null;

    for await (const ev of session.sendStream(prompt, {
      config: { temperature: 0, maxNewTokens: MAX_NEW_TOKENS },
    })) {
      if (ev.done) {
        // Only the terminal event carries the real token count. Counting stream
        // events instead is wrong by ~2 orders of magnitude on models that emit
        // one aggregated chunk (gemma4) rather than one event per token (qwen3.5).
        tokens = ev.numTokens ?? chunks;
        break;
      }
      if (firstTokenMs === null) firstTokenMs = Date.now() - roundStart;
      chunks += 1;
      // Stream every chunk out: in Electron this drives a renderer repaint, which
      // is the compositing pressure we are actually testing for.
      report({ event: 'token', round, chunks, text: ev.text });
    }

    totalTokens += tokens;
    const elapsed = Date.now() - roundStart;
    report({
      event: 'round',
      round,
      tokens,
      chunks,
      ttftMs: firstTokenMs,
      elapsedMs: elapsed,
      tokensPerSec: tokens > 0 ? +(tokens / (elapsed / 1000)).toFixed(2) : 0,
      totalTokens,
    });

    await session.reset();
  }

  report({ event: 'survived', rounds: round, totalTokens, durationMs: DURATION_MS });
  if (parentPort) parentPort.postMessage({ event: 'done', host });
  else process.exit(0);
}

main().catch((err) => {
  report({ event: 'error', message: String(err?.stack ?? err) });
  process.exit(1);
});
