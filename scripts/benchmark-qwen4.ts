/** Matched public-API timing. Run each addon sequentially through guard-model-memory.py. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';

import { memoryStats, type ChatResult } from '@mlx-node/core';

import { ChatSession } from '../packages/lm/src/chat-session.js';
import { loadModel } from '../packages/lm/src/models/model-loader.js';
import { Qwen4ExpModel } from '../packages/lm/src/stream.js';

const [target, output] = process.argv.slice(2);
assert(target && output, 'Pass the first GGUF split and an output JSON path.');
const promptFile = process.env.QWEN4_BENCH_PROMPT_FILE;
let prompt = promptFile
  ? await readFile(promptFile, 'utf8')
  : 'Write the integers from 1 to 12, separated by commas. Output nothing else.';
const maxNewTokens = Number(process.env.QWEN4_BENCH_OUTPUT_TOKENS ?? 128);
const runs = Number(process.env.QWEN4_BENCH_RUNS ?? 5);
const warmupRuns = Number(process.env.QWEN4_BENCH_WARMUP_RUNS ?? 2);
assert(Number.isInteger(maxNewTokens) && maxNewTokens >= 1 && maxNewTokens <= 256);
assert(Number.isInteger(runs) && runs >= 1 && runs <= 20);
assert(Number.isInteger(warmupRuns) && warmupRuns >= 0 && warmupRuns < runs);
const addonPath = process.env.QWEN4_BENCH_ADDON ?? null;
let addonSha256: string | null = null;
if (addonPath) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(addonPath)) hash.update(chunk);
  addonSha256 = hash.digest('hex');
}
const memoryTrace =
  process.env.QWEN4_BENCH_MEMORY_TRACE === '1'
    ? setInterval(() => console.log(JSON.stringify({ event: 'allocation', memory: memoryStats() })), 500)
    : undefined;
memoryTrace?.unref();
const loadStart = performance.now();
console.log(JSON.stringify({ event: 'loading', target }));
const model = await loadModel(target, { autoLoadDraft: false });
assert(model instanceof Qwen4ExpModel, 'Benchmark requires the Qwen4 runtime.');
const loadSeconds = (performance.now() - loadStart) / 1000;
const inputTokens = process.env.QWEN4_BENCH_INPUT_TOKENS ? Number(process.env.QWEN4_BENCH_INPUT_TOKENS) : undefined;
if (inputTokens !== undefined) {
  assert(!promptFile, 'Choose a prompt file or a generated input length.');
  assert(Number.isInteger(inputTokens) && inputTokens >= 128 && inputTokens <= 2048);
  const instruction = 'Write the integers from 1 to 300, separated by commas. Output nothing else.';
  let count = inputTokens;
  for (let attempt = 0; attempt < 8; attempt++) {
    prompt = `Background notes: ${'note '.repeat(count)}\n\n${instruction}`;
    const ids = await model.applyChatTemplate([{ role: 'user', content: prompt }], true, null, false);
    if (ids.length === inputTokens) break;
    count += inputTokens - ids.length;
    assert(count >= 0);
  }
}
const promptIds = await model.applyChatTemplate([{ role: 'user', content: prompt }], true, null, false);
if (inputTokens !== undefined) assert.equal(promptIds.length, inputTokens);
const promptIdsSha256 = createHash('sha256')
  .update(JSON.stringify([...promptIds]))
  .digest('hex');
const residency = model.residencyInfo();
console.log(
  JSON.stringify({
    event: 'loaded',
    loadSeconds,
    residency,
    memory: memoryStats(),
  }),
);
const samples: {
  run: number;
  seconds: number;
  result: ChatResult;
  sha256: string;
  memory: ReturnType<typeof memoryStats>;
  warmup: boolean;
}[] = [];
for (let run = 0; run < runs; run++) {
  await model.resetCaches();
  console.log(JSON.stringify({ event: 'starting', run }));
  const session: ChatSession<Qwen4ExpModel> = new ChatSession(model, {
    defaultConfig: {
      cacheOwnerId: `qwen4-benchmark-${run}`,
      temperature: 0,
      maxNewTokens,
      reasoningEffort: 'none',
      enableMtp: false,
      reuseCache: true,
      reportPerformance: true,
    },
  });
  const start = performance.now();
  const result: ChatResult = await session.send(prompt);
  const seconds = (performance.now() - start) / 1000;
  assert.equal(result.cachedTokens, 0, 'Conversation prefix must be cold.');
  assert.equal(result.promptTokens, promptIds.length);
  if (!promptFile && !inputTokens && maxNewTokens === 32) assert.equal(result.numTokens, 32);
  assert((result.numTokens ?? 0) > 0);
  const sha256 = createHash('sha256').update(result.text).digest('hex');
  samples.push({
    run,
    seconds,
    result,
    sha256,
    memory: memoryStats(),
    warmup: run < warmupRuns,
  });
  await writeFile(
    output,
    JSON.stringify(
      {
        target,
        addonPath,
        addonSha256,
        prompt,
        promptIds: [...promptIds],
        promptIdsSha256,
        loadSeconds,
        residency,
        maxNewTokens,
        environment: Object.fromEntries(Object.entries(process.env).filter(([key]) => key.startsWith('MLX_QWEN4_'))),
        conditions:
          'Single stream; greedy; no MTP or prefix reuse. Subsequent runs retain the process weight cache. Exact input and output token counts are stored per sample. OS file cache, desktop load and thermals uncontrolled.',
        samples,
      },
      null,
      2,
    ),
  );
  console.log(
    JSON.stringify({
      event: 'sample',
      run,
      seconds,
      result,
      sha256,
      memory: memoryStats(),
    }),
  );
  if (process.env.QWEN4_BENCH_EXPECT_SHA256)
    assert.equal(sha256, process.env.QWEN4_BENCH_EXPECT_SHA256, 'Candidate output differs from the matched control.');
  if (run) assert.equal(sha256, samples[0].sha256, 'Repeated greedy output changed.');
}
if (memoryTrace) clearInterval(memoryTrace);
console.log(JSON.stringify({ event: 'passed' }));
