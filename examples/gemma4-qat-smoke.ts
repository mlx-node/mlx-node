#!/usr/bin/env node
/**
 * Gemma 4 E2B QAT (wNa8o8) end-to-end smoke test.
 *
 * Loads a converted Gemma-4 E2B QAT checkpoint (quantized lm_head + quantized
 * embeddings) through the unified ChatSession API and decodes a fixed set of
 * greedy (temperature 0) text prompts, plus one optional image-caption turn.
 * Each prompt runs on a reset session so the greedy decodes are independent.
 * Output is printed verbatim for downstream coherence judging.
 *
 * Run with:
 *   oxnode examples/gemma4-qat-smoke.ts [model-dir] [--image <path>]
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import type { ChatResult, SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, loadModel } from '@mlx-node/lm';

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    image: { type: 'string' },
  },
  allowPositionals: true,
});

const MODEL_PATH = resolve(process.cwd(), positionals[0] || '.cache/models/gemma-4-e2b-qat-mlx');
const imagePath = values.image;

console.log(`Loading model from: ${MODEL_PATH}`);
const loadedModel = await loadModel(MODEL_PATH);
console.log('Model loaded.\n');

const session = new ChatSession(loadedModel as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant. Be concise.',
});

async function runTurn(
  prompt: string,
  images: Uint8Array[] | undefined,
): Promise<Pick<ChatResult, 'rawText'> & { finishReason: string; numTokens: number }> {
  let fullText = '';
  let finishReason = 'unknown';
  let numTokens = 0;
  for await (const event of session.sendStream(prompt, {
    ...(images !== undefined && { images }),
    config: {
      maxNewTokens: 128,
      temperature: 0,
      reportPerformance: false,
    },
  })) {
    if (event.done) {
      fullText = event.rawText;
      finishReason = event.finishReason;
      numTokens = event.numTokens;
    } else {
      process.stdout.write(event.text);
    }
  }
  process.stdout.write('\n');
  // Reset session between turns so each greedy decode is independent.
  await session.reset();
  return { rawText: fullText, finishReason, numTokens };
}

const PROMPTS = [
  'What is the capital of France?',
  'Write one sentence about the ocean.',
  'Count from 1 to 10.',
  'Explain what a transformer neural network is in two sentences.',
];

const results: { prompt: string; text: string; finishReason: string; numTokens: number }[] = [];

for (let i = 0; i < PROMPTS.length; i++) {
  const prompt = PROMPTS[i]!;
  console.log(`\n── Prompt ${i + 1} ──`);
  console.log(`User: ${prompt}`);
  process.stdout.write('Assistant: ');
  const r = await runTurn(prompt, undefined);
  results.push({ prompt, text: r.rawText, finishReason: r.finishReason, numTokens: r.numTokens });
}

let imageCaption: { text: string; finishReason: string; numTokens: number } | undefined;
if (imagePath) {
  console.log(`\n── Image prompt ──`);
  const imageBuffer = await readFile(resolve(process.cwd(), imagePath));
  const imageBytes = new Uint8Array(imageBuffer.buffer, imageBuffer.byteOffset, imageBuffer.byteLength);
  console.log(`Image: ${imagePath} (${imageBytes.length} bytes)`);
  console.log(`User: Describe this image.`);
  process.stdout.write('Assistant: ');
  const r = await runTurn('Describe this image.', [imageBytes]);
  imageCaption = { text: r.rawText, finishReason: r.finishReason, numTokens: r.numTokens };
}

console.log('\n════════════════ VERBATIM RESULTS ════════════════');
for (let i = 0; i < results.length; i++) {
  console.log(`\n[PROMPT ${i + 1}] ${results[i]!.prompt}`);
  console.log(`[FINISH ${i + 1}] ${results[i]!.finishReason} | tokens=${results[i]!.numTokens}`);
  console.log(`[OUTPUT ${i + 1}] ${JSON.stringify(results[i]!.text)}`);
}
if (imageCaption) {
  console.log(`\n[IMAGE PROMPT] Describe this image.`);
  console.log(`[IMAGE FINISH] ${imageCaption.finishReason} | tokens=${imageCaption.numTokens}`);
  console.log(`[IMAGE CAPTION] ${JSON.stringify(imageCaption.text)}`);
}
console.log('\n════════════════ END RESULTS ════════════════');
