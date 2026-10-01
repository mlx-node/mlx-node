/// <reference types="node" />

import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

import type { ChatConfig, ChatMessage, ChatResult, ChatStreamChunk } from '../../../packages/core/index.cjs';

const [binding, output, tokenLimit = '96'] = process.argv.slice(2);
if (!binding || !output || !/^\d+$/.test(tokenLimit) || Number(tokenLimit) < 1 || process.argv.length > 5)
  throw new Error('Usage: oxnode docs/research/splash-qwen38/validate.ts <addon.node> <output.json> [tokens=96]');
const core: typeof import('../../../packages/core/index.cjs') = createRequire(import.meta.url)(resolve(binding));
const draft = resolve('.cache/models/qwen3.8-27b-dflash2');
const model = await core.Qwen35Model.load(resolve('.cache/models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf'), {
  draftModelPath: draft,
});
const config: ChatConfig = {
  maxNewTokens: Number(tokenLimit),
  temperature: 0,
  topK: 0,
  topP: 1,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
  enableMtp: true,
  mtpAdaptiveDepth: false,
  reasoningEffort: 'none',
  includeReasoning: true,
  reuseCache: true,
  reportPerformance: true,
};
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const records: { name: string; hash: string; result: ChatResult | ChatStreamChunk }[] = [];
function record(name: string, result: ChatResult | ChatStreamChunk) {
  assert.ok((result.numTokens ?? 0) > 0);
  assert.equal(typeof result.rawText, 'string');
  records.push({ name, hash: hash(result.rawText!), result });
  // Preserve the failing scenario too if a later lifecycle assertion fails.
  writeFileSync(output, JSON.stringify({ binding, draft, config, records }, null, 2));
}
const messages: ChatMessage[] = [
  {
    role: 'user',
    content: 'Implement a TypeScript queue with enqueue, dequeue, size and tests. Explain the implementation.',
  },
];
const first = await model.chatSessionStart(messages, config);
record('first', first);
assert.equal(first.finishReason, 'length');
assert.equal(first.numTokens, Number(tokenLimit));
assert.ok(
  !/\s$/u.test(first.rawText),
  'Positive cache fixture must end before trailing whitespace: the chat template trims it; choose a different token limit',
);
const history: ChatMessage[] = [
  ...messages,
  {
    role: 'assistant',
    content: first.text,
    reasoningContent: first.thinking,
    thinkingEnabled: first.thinkingEnabled,
  },
  { role: 'user', content: 'Now add peek and clear methods, and explain their edge cases.' },
];
const continued = await model.chatSessionContinue(history, config);
record('continued', continued);
assert.equal(
  continued.cachedTokens,
  first.promptTokens + first.numTokens,
  'continuation must reuse the complete verified target/draft prefix',
);
await model.resetCaches();
const coldContinuation = await model.chatSessionStart(history, config);
record('cold-continuation', coldContinuation);
// Warm vs cold target numerics can differ with prefill chunking. Retain both
// transcripts; compare each scenario between binaries instead of assuming
// the unrelated AR/parallel-prefill kernels are bit-identical.
await model.resetCaches();
const stream = await new Promise<ChatStreamChunk>((resolveResult, reject) => {
  model
    .chatStreamSessionStart(messages, config, (error, chunk) => {
      if (error) reject(error);
      else if (chunk.done) resolveResult(chunk);
    })
    .catch(reject);
});
record('stream', stream);
assert.equal(stream.rawText, first.rawText, 'streaming must preserve the same greedy transcript');
await model.resetCaches();
const sampled = await model.chatSessionStart(messages, { ...config, maxNewTokens: 32, temperature: 0.7, topP: 0.9 });
record('sampled', sampled);
assert.ok((sampled.performance?.mtpCycles ?? 0) > 0, 'sampled request must verify proposals');
// Exercise a real retained review prompt followed by a new turn, including
// the >2048-row draft sliding window and a target cache growth boundary.
const fixtures: { name: string; messages: ChatMessage[] }[] = JSON.parse(
  await readFile('.cache/benchmarks/fixtures-public-2026-09-11/qwen38-inputs.json', 'utf8'),
);
const long = fixtures.find((item) => item.name === '6k')!;
await model.resetCaches();
const longFirst = await model.chatSessionStart(long.messages, { ...config, maxNewTokens: 32, reasoningEffort: 'high' });
record('long-first', longFirst);
const longHistory: ChatMessage[] = [
  ...long.messages,
  {
    role: 'assistant',
    content: longFirst.text,
    reasoningContent: longFirst.thinking,
    thinkingEnabled: longFirst.thinkingEnabled,
  },
  { role: 'user', content: 'Which test would best expose the issue?' },
];
const longContinued = await model.chatSessionContinue(longHistory, {
  ...config,
  maxNewTokens: 32,
  reasoningEffort: 'high',
});
record('long-continued', longContinued);
assert.ok(longContinued.cachedTokens > 2048, 'long continuation must retain target prefix and sliding draft context');
await writeFile(output, JSON.stringify({ binding, draft, config, records }, null, 2));
console.log(
  JSON.stringify(
    records.map(({ name, hash, result }) => ({ name, hash, cached: result.cachedTokens, tokens: result.numTokens })),
  ),
);
await model.resetCaches();
process.exit(0);
