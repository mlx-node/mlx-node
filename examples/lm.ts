#!/usr/bin/env node
/**
 * Test MLX Model with Multi-Round Chat
 *
 * For Qwen3.5 dense models this example uses the server-side
 * `Qwen35Session` API (streaming `sendStream`) — the session tracks its
 * own KV cache on the native side, so each turn only prefills the new
 * user delta on top of a cache that already ends on a clean ChatML
 * boundary. This replaces the legacy jinja prefix-matching cache-reuse
 * path that mis-handled `<think>` stripping and the
 * eos_token_id / eos_token mismatch, which caused a Metal GPU watchdog
 * hang on turn 4 of a 4-turn run.
 *
 * Gemma4, Qwen3 (legacy), Qianfan-OCR, and the VLM image branch still
 * go through `model.chat(messages, ...)` — those paths don't hit the
 * Qwen3.5-specific bug and the session API is currently text-only.
 *
 * Usage:
 *   oxnode examples/lm.ts [model-name] [--image <path>]
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { Gemma4Model, HarrierModel, QianfanOCRModel } from '@mlx-node/core';
import type { ChatResult, PerformanceMetrics } from '@mlx-node/lm';
import { loadModel, Qwen35Model, Qwen35Session, Qwen3Model } from '@mlx-node/lm';

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    image: { type: 'string' },
  },
  allowPositionals: true,
});

const modelName = positionals[0] || 'qwen3.5-9B-unsloth';
const imagePath = values.image;

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

console.log(`Loading model from: ${MODEL_PATH}`);
if (imagePath) console.log(`Image: ${imagePath}`);

const loadedModel = await loadModel(MODEL_PATH);
if (loadedModel instanceof HarrierModel) {
  console.error('This example is for generative models, not embedding models.');
  process.exit(1);
}
const isGemma4 = loadedModel instanceof Gemma4Model;
const isQwen3 = loadedModel instanceof Qwen3Model;
const isQianfan = loadedModel instanceof QianfanOCRModel;
const isQwen35 = loadedModel instanceof Qwen35Model;
const modelArch = isGemma4 ? 'Gemma4' : isQianfan ? 'Qianfan-OCR' : isQwen3 ? 'Qwen3' : isQwen35 ? 'Qwen3.5' : 'Other';
console.log(`Model loaded (${modelArch})\n`);

function printPerf(result: { finishReason: string; numTokens: number; performance?: ChatResult['performance'] }) {
  const p = result.performance;
  if (!p) return;
  console.log('-'.repeat(80));
  console.log(
    `Stop reason: ${result.finishReason} | ${result.numTokens} tokens | TTFT ${p.ttftMs.toFixed(0)}ms | Prefill ${p.prefillTokensPerSecond.toFixed(1)} tok/s | Decode ${p.decodeTokensPerSecond.toFixed(1)} tok/s`,
  );
}

async function chat(
  messages: { role: string; content: string; images?: Uint8Array[] }[],
  opts: { maxNewTokens: number; temperature: number },
): Promise<{
  text: string;
  rawText: string;
  finishReason: string;
  numTokens: number;
  performance?: ChatResult['performance'];
}> {
  if (isGemma4) {
    const model = loadedModel as Gemma4Model;
    const r = await model.chat(messages, {
      maxNewTokens: opts.maxNewTokens,
      temperature: opts.temperature,
    });
    return {
      text: r.text,
      rawText: r.text,
      finishReason: r.finishReason,
      numTokens: r.numTokens,
      performance: r.performance,
    };
  }
  const model = loadedModel as Exclude<typeof loadedModel, HarrierModel | Gemma4Model>;
  const r = await model.chat(messages, {
    maxNewTokens: opts.maxNewTokens,
    temperature: opts.temperature,
    reportPerformance: true,
  });
  return {
    text: r.text,
    rawText: r.rawText,
    finishReason: r.finishReason,
    numTokens: r.numTokens,
    performance: r.performance,
  };
}

if (imagePath) {
  // ── VLM multi-round: same image discussed across turns ──
  const imageBuffer = await readFile(resolve(process.cwd(), imagePath));
  const imageBytes = new Uint8Array(imageBuffer.buffer, imageBuffer.byteOffset, imageBuffer.byteLength);
  console.log(`Image: ${imageBytes.length} bytes\n`);

  const messages: { role: string; content: string; images?: Uint8Array[] }[] = [
    { role: 'user', content: 'Describe this image briefly.', images: [imageBytes] },
  ];

  // Turn 1: full VLM prefill
  console.log('── Turn 1 (full VLM prefill) ──');
  console.log(`User: ${messages[0].content}`);
  const r1 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r1.text}`);
  printPerf(r1);

  // Turn 2: cache reuse — only prefills the new user message
  messages.push({ role: 'assistant', content: r1.rawText });
  messages.push({ role: 'user', content: 'What colors do you see in the image?' });

  console.log('\n── Turn 2 (cache reuse, same image) ──');
  console.log(`User: ${messages[2].content}`);
  const r2 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r2.text}`);
  printPerf(r2);

  // Turn 3: another follow-up
  messages.push({ role: 'assistant', content: r2.rawText });
  messages.push({ role: 'user', content: 'Summarize everything you told me about this image in one sentence.' });

  console.log('\n── Turn 3 (cache reuse) ──');
  console.log(`User: ${messages[4].content}`);
  const r3 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r3.text}`);
  printPerf(r3);
} else if (isQwen35) {
  // ── Qwen3.5 dense text multi-round chat via server-side session ──
  //
  // The session owns the KV cache across turns. Turn 1 runs the jinja
  // chat template then streams a reply that ends on `<|im_end|>`;
  // turns 2..N build a raw ChatML delta on top of the live cache and
  // only prefill the new user message. This bypasses the legacy
  // prefix-matching cache-reuse path that caused the Metal watchdog
  // hang on turn 4.
  const qwen35Model = loadedModel as Qwen35Model;
  const session = new Qwen35Session(qwen35Model, {
    system: 'You are a helpful assistant. Be concise.',
  });

  const userMessages = [
    'What is the capital of France?',
    'What about Germany?',
    'And Japan?',
    'Which of those three cities has the largest population?',
  ];

  interface TurnStats {
    finishReason: string;
    numTokens: number;
    promptTokens: number;
    reasoningTokens: number;
    performance?: PerformanceMetrics;
  }
  const turnStats: TurnStats[] = [];

  for (let i = 0; i < userMessages.length; i++) {
    const turnNumber = i + 1;
    console.log(`${i === 0 ? '' : '\n'}── Turn ${turnNumber} (session sendStream) ──`);
    console.log(`User: ${userMessages[i]}`);
    process.stdout.write('Assistant: ');

    let finishReason = 'unknown';
    let numTokens = 0;
    let promptTokens = 0;
    let reasoningTokens = 0;
    let performance: PerformanceMetrics | undefined;

    for await (const event of session.sendStream(userMessages[i], {
      maxNewTokens: 2048,
      temperature: 0.6,
      reportPerformance: true,
    })) {
      if (event.done) {
        finishReason = event.finishReason;
        numTokens = event.numTokens;
        promptTokens = event.promptTokens;
        reasoningTokens = event.reasoningTokens;
        performance = event.performance;
      } else {
        process.stdout.write(event.text);
      }
    }
    process.stdout.write('\n');

    turnStats.push({ finishReason, numTokens, promptTokens, reasoningTokens, performance });
    printPerf({ finishReason, numTokens, performance });
  }

  // ── TTFT assertions: turn 4 should be flat relative to turn 1 ──
  console.log('\n── TTFT summary ──');
  console.log('Turn | TTFT ms | Prompt tokens');
  console.log('-----+---------+--------------');
  for (let i = 0; i < turnStats.length; i++) {
    const p = turnStats[i].performance;
    const ttft = p ? p.ttftMs.toFixed(0) : 'n/a';
    const pt = turnStats[i].promptTokens;
    console.log(`  ${i + 1}  | ${ttft.padStart(7)} | ${String(pt).padStart(13)}`);
  }

  const ttft1Raw = turnStats[0].performance?.ttftMs;
  const ttft2Raw = turnStats[1].performance?.ttftMs;
  const ttft4Raw = turnStats[3].performance?.ttftMs;
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
} else {
  // ── Legacy text multi-round chat for Gemma4 / Qwen3 / Qianfan-OCR ──
  //
  // These go through the jinja prefix-matching cache-reuse path — it
  // works fine for models whose tokenizer_config eos_token matches the
  // config.json eos_token_id and that don't emit `<think>` tags.
  const messages: { role: string; content: string }[] = [
    { role: 'system', content: 'You are a helpful assistant. Be concise.' },
    { role: 'user', content: 'What is the capital of France?' },
  ];

  // Turn 1: full prefill
  console.log('── Turn 1 (full prefill) ──');
  console.log(`User: ${messages[1].content}`);
  const r1 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r1.text}`);
  printPerf(r1);

  // Turn 2: cache reuse — only prefills assistant reply + new question
  messages.push({ role: 'assistant', content: r1.rawText });
  messages.push({ role: 'user', content: 'What about Germany?' });

  console.log('\n── Turn 2 (cache reuse) ──');
  console.log(`User: ${messages[3].content}`);
  const r2 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r2.text}`);
  printPerf(r2);

  // Turn 3: cache reuse again
  messages.push({ role: 'assistant', content: r2.rawText });
  messages.push({ role: 'user', content: 'And Japan?' });

  console.log('\n── Turn 3 (cache reuse) ──');
  console.log(`User: ${messages[5].content}`);
  const r3 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r3.text}`);
  printPerf(r3);

  // Turn 4: one more to show compounding savings
  messages.push({ role: 'assistant', content: r3.rawText });
  messages.push({ role: 'user', content: 'Which of those three cities has the largest population?' });

  console.log('\n── Turn 4 (cache reuse) ──');
  console.log(`User: ${messages[7].content}`);
  const r4 = await chat(messages, { maxNewTokens: 2048, temperature: 0.6 });
  console.log(`Assistant: ${r4.text}`);
  printPerf(r4);
}
