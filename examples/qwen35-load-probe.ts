#!/usr/bin/env node
// Minimal load-vs-prefill watchdog probe. Durable appendFileSync markers.
//   file has A only      -> abort during loadModel
//   file has A,B         -> load OK, abort during the 16-token prefill (cold-start)
//   file has A,B,C       -> both fine
import { appendFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const { values } = parseArgs({
  args: process.argv.slice(2),
  options: {
    model: { type: 'string', default: '/Volumes/P4510/.cache/models/qwen3.5-9B-bf16' },
    out: { type: 'string', default: '/tmp/fq_loadprobe.tsv' },
  },
});
const out = values.out!;
const mark = (s: string) => {
  appendFileSync(out, `${Date.now()}\t${s}\n`);
};

writeFileSync(out, '');
mark('A_before_load');
const loaded = await loadModel(values.model!);
mark('B_after_load');
const session = new ChatSession(loaded as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant.',
});
const res = await session.send('Read this short line.\nContinue.', {
  config: { maxNewTokens: 1, temperature: 0, reportPerformance: true, reuseCache: false },
});
mark(`C_after_tiny_prefill\tpromptTok=${res.promptTokens}\tttftMs=${res.performance?.ttftMs?.toFixed(0)}`);
mark('DONE');
