#!/usr/bin/env node
/**
 * Test MLX Model with Multi-Round Chat
 *
 * Every generative model goes through the unified `ChatSession` API.
 * The session owns a KV cache on the native side and only prefills the
 * new user delta on each turn after turn 1. Image-change mid-session
 * (via `--image2`) resets the cache and re-prefills the full history
 * against the new image, demonstrating the expected TTFT spike.
 *
 * Usage:
 *   oxnode examples/lm.ts [model-name] [--image <path>] [--image2 <path>]
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import type { ChatResult, PerformanceMetrics, SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    image: { type: 'string' },
    image2: { type: 'string' },
  },
  allowPositionals: true,
});

const modelName = positionals[0] || 'qwen3.5-9B-unsloth';
const imagePath = values.image;
const imagePath2 = values.image2;

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

console.log(`Loading model from: ${MODEL_PATH}`);
if (imagePath) console.log(`Image: ${imagePath}`);
if (imagePath2) console.log(`Image2: ${imagePath2}`);

const loadedModel = await loadModel(MODEL_PATH);
if (loadedModel instanceof HarrierModel) {
  console.error('This example is for generative models, not embedding models.');
  process.exit(1);
}
console.log('Model loaded\n');

// Single code path: wrap any session-capable model in a ChatSession. The
// wrappers exported from `@mlx-node/lm` (Qwen3, Qwen3.5 dense, Qwen3.5
// MoE, Gemma4, LFM2) all structurally satisfy `SessionCapableModel`.
const session = new ChatSession(loadedModel as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant. Be concise.',
});

function printPerf(label: string, finishReason: string, numTokens: number, performance?: PerformanceMetrics) {
  if (!performance) return;
  console.log('-'.repeat(80));
  console.log(
    `${label} | Stop reason: ${finishReason} | ${numTokens} tokens | TTFT ${performance.ttftMs.toFixed(0)}ms | Prefill ${performance.prefillTokensPerSecond.toFixed(1)} tok/s | Decode ${performance.decodeTokensPerSecond.toFixed(1)} tok/s`,
  );
}

interface TurnStats {
  finishReason: string;
  numTokens: number;
  promptTokens: number;
  reasoningTokens: number;
  performance?: PerformanceMetrics;
}

/**
 * Run one streaming turn and collect stats. Prints the assistant reply
 * as it arrives. Works for both text and image turns — the session
 * decides internally whether the call takes the cheap continue path or
 * the expensive start/restart path.
 */
async function runTurn(
  label: string,
  userMessage: string,
  images: Uint8Array[] | undefined,
): Promise<TurnStats & Pick<ChatResult, 'rawText'>> {
  console.log(`\n── ${label} ──`);
  console.log(`User: ${userMessage}`);
  process.stdout.write('Assistant: ');

  let finishReason = 'unknown';
  let numTokens = 0;
  let promptTokens = 0;
  let reasoningTokens = 0;
  let performance: PerformanceMetrics | undefined;
  let rawText = '';

  for await (const event of session.sendStream(userMessage, {
    ...(images !== undefined && { images }),
    config: {
      maxNewTokens: 2048,
      temperature: 0.6,
      reportPerformance: true,
      reasoningEffort: 'low',
    },
  })) {
    if (event.done) {
      finishReason = event.finishReason;
      numTokens = event.numTokens;
      promptTokens = event.promptTokens;
      reasoningTokens = event.reasoningTokens;
      performance = event.performance;
      rawText = event.rawText;
    } else {
      process.stdout.write(event.text);
    }
  }
  process.stdout.write('\n');

  printPerf(label, finishReason, numTokens, performance);
  return { finishReason, numTokens, promptTokens, reasoningTokens, performance, rawText };
}

if (imagePath) {
  // ── VLM multi-round: same image discussed across turns, then an
  // optional image-change demo on turn 4. ──
  const imageBuffer = await readFile(resolve(process.cwd(), imagePath));
  const imageBytes = new Uint8Array(imageBuffer.buffer, imageBuffer.byteOffset, imageBuffer.byteLength);
  console.log(`Image: ${imageBytes.length} bytes\n`);

  // Turn 1: full VLM prefill — image attached.
  await runTurn('Turn 1 (full VLM prefill)', 'Describe this image briefly.', [imageBytes]);

  // Turn 2: text-only follow-up. Omitting `images` means "keep the
  // current image cache state" — the server-side cache still holds the
  // image context, so this takes the cheap delta path.
  await runTurn('Turn 2 (cache reuse, same image)', 'What colors do you see in the image?', undefined);

  // Turn 3: another text-only follow-up on the delta path.
  await runTurn(
    'Turn 3 (cache reuse)',
    'Summarize everything you told me about this image in one sentence.',
    undefined,
  );

  // Turn 4 (optional): pass a DIFFERENT image. ChatSession notices the
  // new image set differs from `lastImagesKey`, resets the native
  // caches, and re-prefills the full history against the new image —
  // the expected TTFT spike. Skipped if --image2 was not provided.
  if (imagePath2) {
    const imageBuffer2 = await readFile(resolve(process.cwd(), imagePath2));
    const imageBytes2 = new Uint8Array(imageBuffer2.buffer, imageBuffer2.byteOffset, imageBuffer2.byteLength);
    console.log(`\nImage2: ${imageBytes2.length} bytes (triggers mid-session restart)`);
    await runTurn('Turn 4 (image change — expect TTFT spike)', 'Now describe this new image in one sentence.', [
      imageBytes2,
    ]);
    console.log(
      '\nNote: turn 4 TTFT is expected to be much higher than turns 2/3 because the image-change path triggers a full chatSessionStart re-prefill against the preserved history.',
    );
  }
} else {
  // ── Text multi-round chat via ChatSession. Uniform code path for
  // all supported generative architectures. ──
  const userMessages = [
    'What is the capital of France?',
    'What about Germany?',
    'And Japan?',
    'Which of those three cities has the largest population?',
  ];

  const turnStats: TurnStats[] = [];
  for (let i = 0; i < userMessages.length; i++) {
    const stats = await runTurn(`Turn ${i + 1} (session sendStream)`, userMessages[i]!, undefined);
    turnStats.push(stats);
  }

  // ── TTFT assertions: turn 4 should be flat relative to turn 1 ──
  console.log('\n── TTFT summary ──');
  console.log('Turn | TTFT ms | Prompt tokens');
  console.log('-----+---------+--------------');
  for (let i = 0; i < turnStats.length; i++) {
    const p = turnStats[i]!.performance;
    const ttft = p ? p.ttftMs.toFixed(0) : 'n/a';
    const pt = turnStats[i]!.promptTokens;
    console.log(`  ${i + 1}  | ${ttft.padStart(7)} | ${String(pt).padStart(13)}`);
  }

  const ttft1Raw = turnStats[0]!.performance?.ttftMs;
  const ttft2Raw = turnStats[1]!.performance?.ttftMs;
  const ttft4Raw = turnStats[3]!.performance?.ttftMs;
  if (ttft1Raw === undefined || ttft2Raw === undefined || ttft4Raw === undefined) {
    console.error('FAIL: missing TTFT measurements; cannot validate cache reuse');
    process.exit(1);
    throw new Error('unreachable');
  }
  const ttft1: number = ttft1Raw;
  const ttft2: number = ttft2Raw;
  const ttft4: number = ttft4Raw;
  const ratio41 = ttft4 / ttft1;
  const ratio42 = ttft4 / ttft2;
  console.log(`\nTTFT turn4 / turn1 = ${ratio41.toFixed(2)}`);
  console.log(`TTFT turn4 / turn2 = ${ratio42.toFixed(2)}`);
  if (ratio41 >= 1.5) {
    console.error(`FAIL: TTFT regression detected (turn4/turn1 = ${ratio41.toFixed(2)} >= 1.5)`);
    process.exit(1);
  }
  if (ttft4 >= ttft2 * 2.0) {
    console.error(
      `FAIL: TTFT regression detected (turn4 = ${ttft4.toFixed(0)}ms, turn2 = ${ttft2.toFixed(0)}ms, ratio = ${ratio42.toFixed(2)} >= 2.0)`,
    );
    process.exit(1);
  }
  console.log('PASS: TTFT flat across 4 turns');
}
