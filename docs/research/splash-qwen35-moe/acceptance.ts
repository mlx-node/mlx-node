#!/usr/bin/env oxnode

/// <reference types="node" />

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';

import type {
  ChatConfig,
  ChatMessage,
  ChatResult,
  ChatStreamChunk,
  ChatStreamHandle,
} from '../../../packages/core/index.cjs';

// Bounded live acceptance against a frozen addon, separate from throughput runs.
const [binding, target, draft, output, adaptive = 'false', kvFormat = 'bf16'] = process.argv.slice(2);
if (
  !binding ||
  !target ||
  !draft ||
  !output ||
  !['true', 'false'].includes(adaptive) ||
  (kvFormat !== 'bf16' && kvFormat !== 'int8')
) {
  throw new Error(
    'Usage: oxnode acceptance.ts addon.node native-model-directory draft-directory result.json [adaptive=false] [kv-format=bf16]',
  );
}
const watchdog = setTimeout(() => {
  console.error('Live acceptance exceeded 180 seconds');
  process.exit(124);
}, 180_000);
watchdog.unref();
const core: typeof import('../../../packages/core/index.cjs') = createRequire(import.meta.url)(resolve(binding));
const model = await core.Qwen35MoeModel.load(resolve(target), { draftModelPath: resolve(draft), kvFormat });
assert.equal(model.hasMtpWeights(), true);
const messages: ChatMessage[] = [
  {
    role: 'user',
    content:
      'Implement a TypeScript LRU cache with get, set, delete, fixed capacity, and eviction tests. Explain the implementation and complexity in detail.',
  },
];
const config: ChatConfig = {
  maxNewTokens: 32,
  temperature: 0,
  topK: 0,
  topP: 1,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
  reasoningEffort: 'high',
  includeReasoning: true,
  enableMtp: true,
  mtpAdaptiveDepth: adaptive === 'true',
  reuseCache: true,
  cacheOwnerId: 'splash-port-acceptance',
  cacheRootOwnerId: 'splash-port-acceptance',
  cacheSalt: 'splash-port-acceptance',
  reportPerformance: true,
};
const records: unknown[] = [];
await mkdir(dirname(resolve(output)), { recursive: true });
const save = async (name: string, result: unknown) => {
  records.push({ name, result });
  await writeFile(
    output,
    JSON.stringify(
      {
        binding: resolve(binding),
        bindingSha256: createHash('sha256')
          .update(await readFile(binding))
          .digest('hex'),
        target: resolve(target),
        draft: resolve(draft),
        adaptive: adaptive === 'true',
        kvFormat,
        records,
        memory: core.memoryStats(),
      },
      null,
      2,
    ) + '\n',
  );
  console.log(name);
};
function completed(result: ChatResult | ChatStreamChunk, limit = 32, speculative = true) {
  assert.equal(result.numTokens, limit, 'fixture must reach its token budget');
  assert.ok(result.performance && Number.isFinite(result.performance.ttftMs));
  assert.ok(result.rawText?.length, 'completion must have raw token text');
  if (speculative) assert.ok((result.performance?.mtpCycles ?? 0) > 0, 'DFlash2 must actually execute');
  else assert.equal(result.performance?.mtpCycles ?? 0, 0, 'disabled speculation must use autoregressive decoding');
}
async function stream(
  cfg: ChatConfig,
  cancel = false,
): Promise<{ terminal: ChatStreamChunk; chunks: number; cancelled: boolean }> {
  let handle: ChatStreamHandle | undefined;
  let chunks = 0;
  let cancelled = false;
  let cancelPending = false;
  const terminal = await new Promise<ChatStreamChunk>((done, fail) => {
    void model
      .chatStreamSessionStart(messages, cfg, (error, chunk) => {
        if (error) {
          fail(error);
          return;
        }
        chunks++;
        if (chunk.done) {
          done(chunk);
          return;
        }
        if (cancel && !cancelled) {
          if (handle) {
            handle.cancel();
            cancelled = true;
          } else cancelPending = true;
        }
      })
      .then((value) => {
        handle = value;
        if (cancelPending) {
          handle.cancel();
          cancelled = true;
        }
      }, fail);
  });
  return { terminal, chunks, cancelled };
}

await model.resetCaches();
const sync = await model.chatSessionStart(messages, config);
await save('cold_sync', sync);
completed(sync);

const transcript: ChatMessage[] = [
  ...messages,
  {
    role: 'assistant',
    content: sync.text,
    reasoningContent: sync.thinking,
    thinkingEnabled: sync.thinkingEnabled,
    toolCalls: sync.toolCalls.map((call) => ({
      id: call.id,
      name: call.name,
      arguments: JSON.stringify(call.arguments),
    })),
  },
  { role: 'user', content: 'Continue with the eviction tests.' },
];
const continuation = await model.chatSessionContinue(transcript, config);
await save('warm_continuation', continuation);
completed(continuation);
assert.ok(continuation.cachedTokens > 0, 'continuation must reuse its verified prefix');

await model.resetCaches();
const streamed = await stream(config);
await save('cold_stream', streamed);
completed(streamed.terminal);
assert.equal(streamed.terminal.rawText, sync.rawText, 'cold sync/stream token text must match');

await model.resetCaches();
const cancelled = await stream({ ...config, maxNewTokens: 256 }, true);
await save('stream_cancel', cancelled);
assert.equal(cancelled.cancelled, true);
assert.equal(cancelled.terminal.finishReason, 'cancelled');
assert.ok((cancelled.terminal.numTokens ?? 0) < 256);

// Start on a different owner without resetting: cancellation and lane ownership
// must not expose partially advanced recurrent/KV state to a new request.
const recovered = await model.chatSessionStart(messages, { ...config, cacheOwnerId: 'acceptance-recovered' });
await save('cancel_recovery_new_owner', recovered);
completed(recovered);
assert.equal(recovered.rawText, sync.rawText);

const call = await model.beginChatSessionStart(messages, { ...config, maxNewTokens: 256 });
call.cancel();
await assert.rejects(call.result(), /chat session cancelled/);
await save('queued_sync_cancel', { rejected: true });

await model.resetCaches();
const sampled = await model.chatSessionStart(messages, {
  ...config,
  maxNewTokens: 16,
  temperature: 0.7,
  topK: 20,
  topP: 0.9,
  repetitionPenalty: 1.1,
});
await save('sampled_with_penalty', sampled);
completed(sampled, 16);

await model.resetCaches();
const ar = await model.chatSessionStart(messages, { ...config, enableMtp: false });
await save('autoregressive_lane', ar);
completed(ar, 32, false);
const reenabled = await model.chatSessionStart(messages, { ...config, cacheOwnerId: 'acceptance-reenabled' });
await save('reenabled_dflash_lane', reenabled);
completed(reenabled);
assert.equal(reenabled.rawText, sync.rawText);
await model.resetCaches();
clearTimeout(watchdog);
console.log('Live acceptance passed');
process.exit(0);
