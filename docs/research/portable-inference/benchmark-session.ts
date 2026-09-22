/// <reference types="node" />
// Run from the repository root after yarn build:native. One model/process.
// Usage: oxnode docs/research/portable-inference/benchmark-session.ts MODEL.gguf OUTPUT.json [16k|32k]
import { createHash } from 'node:crypto';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { cpus, release, totalmem } from 'node:os';
import { resolve } from 'node:path';

import type { ChatMessage, ChatResult } from '../../../packages/core/index.cjs';

const [modelPath, outputPath, caseName = '16k'] = process.argv.slice(2);
if (!modelPath || !outputPath || !['16k', '32k'].includes(caseName)) {
  throw new Error('Expected MODEL.gguf OUTPUT.json [16k|32k]');
}
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const manifest = JSON.parse(await readFile('scripts/fixtures/qwen38-oxc-review-v1.json', 'utf8'));
const bytes = await readFile(manifest.payload.path);
if (bytes.length !== manifest.payload.bytes || sha(bytes) !== manifest.payload.sha256) {
  throw new Error('Benchmark fixture hash mismatch; use scripts/benchmark-fixture.ts fetch --fixture qwen38');
}
const modelBytes = (await stat(modelPath)).size;
if (modelBytes !== manifest.model.bytes || totalmem() < modelBytes + 12 * 1024 ** 3) {
  throw new Error('Use the pinned 27B GGUF on a machine with room for weights plus 12 GiB of runtime headroom');
}
const fixture = JSON.parse(bytes.toString()).find((item: { name: string }) => item.name === caseName);
if (!fixture) throw new Error('Fixture case missing');
const core: typeof import('../../../packages/core/index.cjs') = createRequire(import.meta.url)(
  resolve('packages/core/index.cjs'),
);
const started = performance.now();
const model = await core.Qwen35Model.load(resolve(modelPath));
if (!model.hasBlockPagedCache()) throw new Error('Benchmark requires the production paged model');
const loadMs = performance.now() - started;
const config = {
  cacheOwnerId: 'portable-prefill-bench',
  cacheRootOwnerId: 'portable-prefill-bench',
  maxNewTokens: 64,
  temperature: 0,
  enableMtp: false,
  reasoningEffort: 'none' as const,
  reportPerformance: true,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
};
const messages: ChatMessage[] = fixture.messages;
const summarize = (result: ChatResult, input: ChatMessage[], wallMs: number) => ({
  wallMs,
  inputSha256: sha(JSON.stringify(input)),
  outputSha256: sha(result.rawText),
  promptTokens: result.promptTokens,
  cachedTokens: result.cachedTokens,
  generatedTokens: result.numTokens,
  finishReason: result.finishReason,
  performance: result.performance,
});
const coldStart = performance.now();
const cold = await model.chatSessionStart(messages, config);
const coldSample = summarize(cold, messages, performance.now() - coldStart);
const continuation: ChatMessage[] = [
  ...messages,
  {
    role: 'assistant',
    content: cold.text,
    reasoningContent: cold.thinking ?? '',
    thinkingEnabled: cold.thinkingEnabled,
  },
  {
    role: 'user',
    content:
      'Continue the review, taking these additional constraints into account:\n' +
      Array.from(
        { length: 45 },
        (_, i) => `${i + 1}. Preserve public behavior, check error handling, and explain a concrete regression test.`,
      ).join('\n'),
  },
];
const warmStart = performance.now();
const warm = await model.chatSessionContinue(continuation, config);
const warmSample = summarize(warm, continuation, performance.now() - warmStart);
if (warm.cachedTokens <= 0 || cold.numTokens !== 64 || warm.numTokens !== 64) {
  throw new Error('Expected a live prefix continuation and exactly 64 generated tokens per turn');
}
await model.resetCaches();
const report = {
  device: cpus()[0]?.model,
  memoryBytes: totalmem(),
  osRelease: release(),
  modelBytes,
  fixture: manifest.id,
  fixtureSha256: manifest.payload.sha256,
  caseName,
  config,
  switches: Object.fromEntries(
    [
      'MLX_PORTABLE_D256_SDPA',
      'MLX_ENABLE_D256_FULL_SDPA',
      'MLX_PAGED_PREFILL_PAGED_ATTENTION',
      'MLX_PAGED_PREFILL_CHUNK_SIZE',
    ].map((key) => [key, process.env[key] ?? null]),
  ),
  loadMs,
  cold: coldSample,
  continuation: warmSample,
  memory: core.memoryStats(),
};
await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ cold: coldSample, continuation: warmSample }));
