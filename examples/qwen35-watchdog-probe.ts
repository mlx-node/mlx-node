#!/usr/bin/env node
/**
 * Metal GPU watchdog repro probe (diagnostic, throwaway).
 *
 * ONE process. Pays cold-start on a TINY prefill first, then sweeps increasing
 * prompt lengths. Each step is logged DURABLY via appendFileSync (flushed to
 * disk before the next GPU op) so an uncatchable C++ watchdog abort still leaves
 * the crash point on disk — the last "START len=L" with no matching "OK" is the
 * failing length.
 *
 *   tiny warmup OK + a later length crashes -> LENGTH (one command buffer too big)
 *   tiny warmup itself crashes              -> COLD-START
 *
 * Usage: PATH=/usr/bin:$PATH oxnode examples/qwen35-watchdog-probe.ts \
 *          --model /Volumes/P4510/.cache/models/qwen3.5-9B-bf16 --out /tmp/probe.tsv
 */
import { appendFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: '/Volumes/P4510/.cache/models/qwen3.5-9B-bf16' },
    lengths: { type: 'string', default: '16,64,256,512,768,1024,1300,1600,2048' },
    out: { type: 'string', default: '/tmp/fq_probe_progress.tsv' },
  },
});
const modelPath = values.model!;
const lengths = values.lengths!.split(',').map((s) => Number.parseInt(s.trim(), 10));
const out = values.out!;

const SENT =
  'The quick brown fox jumps over the lazy dog beside the quiet river as the evening sun slowly sets. ';
function buildPrompt(targetTok: number, nonce: string): string {
  const copies = Math.max(1, Math.ceil(targetTok / 16));
  return `${nonce}Read the following text.\n${SENT.repeat(copies)}\nContinue.`;
}
function durable(s: string) {
  appendFileSync(out, s + '\n');
  process.stderr.write(s + '\n');
}

writeFileSync(out, '');
const t0 = Date.now();
const loaded = await loadModel(modelPath);
durable(`LOADED\t${((Date.now() - t0) / 1000).toFixed(1)}s\t${modelPath}`);

for (let i = 0; i < lengths.length; i++) {
  const len = lengths[i];
  durable(`START\tlen=${len}\t(${i + 1}/${lengths.length})`);
  const session = new ChatSession(loaded as unknown as SessionCapableModel, {
    system: 'You are a helpful assistant.',
  });
  const ts = Date.now();
  const res = await session.send(buildPrompt(len, `probe-${i} `), {
    config: { maxNewTokens: 1, temperature: 0, reportPerformance: true, reuseCache: false },
  });
  const dt = ((Date.now() - ts) / 1000).toFixed(2);
  const p = res.performance;
  durable(
    `OK\tlen=${len}\tpromptTok=${res.promptTokens}\twall=${dt}s\tttftMs=${p?.ttftMs?.toFixed(0)}\tprefillTps=${p?.prefillTokensPerSecond?.toFixed(0)}`,
  );
}
durable('ALL_DONE');
