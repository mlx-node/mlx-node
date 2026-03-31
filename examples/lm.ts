#!/usr/bin/env node
/**
 * Test MLX Model with Multi-Round Chat & Cache Reuse
 *
 * Demonstrates KV cache reuse across conversation turns — each turn only
 * prefills the new tokens (assistant reply + user follow-up), not the
 * entire conversation history.
 *
 * Usage:
 *   oxnode examples/lm.ts [model-name] [--image <path>]
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { HarrierModel, QianfanOCRModel } from '@mlx-node/core';
import type { ChatResult } from '@mlx-node/lm';
import { loadModel, Qwen3Model } from '@mlx-node/lm';

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
const model = loadedModel;
const isQwen3 = model instanceof Qwen3Model;
const isQianfan = model instanceof QianfanOCRModel;
const modelArch = isQianfan ? 'Qianfan-OCR' : isQwen3 ? 'Qwen3' : 'Qwen3.5';
console.log(`Model loaded (${modelArch})\n`);

function printPerf(result: { finishReason: string; numTokens: number; performance?: ChatResult['performance'] }) {
  const p = result.performance;
  if (!p) return;
  console.log('-'.repeat(80));
  console.log(
    `Stop reason: ${result.finishReason} | ${result.numTokens} tokens | TTFT ${p.ttftMs.toFixed(0)}ms | Prefill ${p.prefillTokensPerSecond.toFixed(1)} tok/s | Decode ${p.decodeTokensPerSecond.toFixed(1)} tok/s`,
  );
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
  const r1 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r1.text}`);
  printPerf(r1);

  // Turn 2: cache reuse — only prefills the new user message
  messages.push({ role: 'assistant', content: r1.rawText });
  messages.push({ role: 'user', content: 'What colors do you see in the image?' });

  console.log('\n── Turn 2 (cache reuse, same image) ──');
  console.log(`User: ${messages[2].content}`);
  const r2 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r2.text}`);
  printPerf(r2);

  // Turn 3: another follow-up
  messages.push({ role: 'assistant', content: r2.rawText });
  messages.push({ role: 'user', content: 'Summarize everything you told me about this image in one sentence.' });

  console.log('\n── Turn 3 (cache reuse) ──');
  console.log(`User: ${messages[4].content}`);
  const r3 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r3.text}`);
  printPerf(r3);
} else {
  // ── Text multi-round chat with cache reuse ──
  const messages: { role: string; content: string }[] = [
    { role: 'system', content: 'You are a helpful assistant. Be concise.' },
    { role: 'user', content: 'What is the capital of France?' },
  ];

  // Turn 1: full prefill
  console.log('── Turn 1 (full prefill) ──');
  console.log(`User: ${messages[1].content}`);
  const r1 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r1.text}`);
  printPerf(r1);

  // Turn 2: cache reuse — only prefills assistant reply + new question
  messages.push({ role: 'assistant', content: r1.rawText });
  messages.push({ role: 'user', content: 'What about Germany?' });

  console.log('\n── Turn 2 (cache reuse) ──');
  console.log(`User: ${messages[3].content}`);
  const r2 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r2.text}`);
  printPerf(r2);

  // Turn 3: cache reuse again
  messages.push({ role: 'assistant', content: r2.rawText });
  messages.push({ role: 'user', content: 'And Japan?' });

  console.log('\n── Turn 3 (cache reuse) ──');
  console.log(`User: ${messages[5].content}`);
  const r3 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r3.text}`);
  printPerf(r3);

  // Turn 4: one more to show compounding savings
  messages.push({ role: 'assistant', content: r3.rawText });
  messages.push({ role: 'user', content: 'Which of those three cities has the largest population?' });

  console.log('\n── Turn 4 (cache reuse) ──');
  console.log(`User: ${messages[7].content}`);
  const r4 = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 0.6,
    reportPerformance: true,
  });
  console.log(`Assistant: ${r4.text}`);
  printPerf(r4);
}
