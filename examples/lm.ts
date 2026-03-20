#!/usr/bin/env node
/**
 * Test Converted MLX Model
 *
 * Simple script to test generation quality with the converted float32 model.
 * Auto-detects model type (Qwen3 vs Qwen3.5) from config.json.
 *
 * Usage:
 *   oxnode examples/lm.ts <model-name> [--image <path>]
 */

import { resolve } from 'node:path';
import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { loadModel, Qwen3Model } from '@mlx-node/lm';

const { values, positionals } = parseArgs({
  args: process.argv.slice(2),
  options: {
    ocr: { type: 'string' },
  },
  allowPositionals: true,
});

const modelName = positionals[0] || 'qwen3.5-9B-unsloth';
const imagePath = values.ocr;

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

console.log('╔════════════════════════════════════════════════════════╗');
console.log('║   Testing Converted MLX Model                          ║');
console.log('╚════════════════════════════════════════════════════════╝\n');

console.log(`Loading model from: ${MODEL_PATH}`);
if (imagePath) console.log(`Image: ${imagePath}`);
console.log('(Tokenizer will be loaded automatically)\n');

// Load model — auto-detects Qwen3 vs Qwen3.5 from config.json
const model = await loadModel(MODEL_PATH);
const isQwen3 = model instanceof Qwen3Model;

console.log(`✓ Model loaded (${isQwen3 ? 'Qwen3' : 'Qwen3.5'})`);

// If --image provided, do a single VLM chat with the image
if (imagePath) {
  const imageBuffer = await readFile(resolve(process.cwd(), imagePath));
  const imageBytes = new Uint8Array(imageBuffer.buffer, imageBuffer.byteOffset, imageBuffer.byteLength);

  console.log(`Image loaded: ${imageBytes.length} bytes`);
  console.log('─'.repeat(60));
  console.log('Prompt: "OCR this image. Extract all text."');
  console.log('─'.repeat(60));

  const messages = [
    { role: 'user' as const, content: 'OCR this image and extract all text you can see.', images: [imageBytes] },
  ];

  const result = await model.chat(messages, {
    maxNewTokens: 2048,
    temperature: 1.0,
    reportPerformance: true,
  });

  console.log(result.text);
  console.log('─'.repeat(60));
  console.log(
    `TTFT: ${result.performance?.ttftMs.toFixed(2)}ms, Prefill: ${result.performance?.prefillTokensPerSecond.toFixed(2)} tokens/s, Decode: ${result.performance?.decodeTokensPerSecond.toFixed(2)} tokens/s`,
  );
  console.log('');
} else {
  // Text-only test prompts
  const prompts = [
    'Hello! How are you today?',
    'What is the capital of France?',
    'Write a haiku about coding:',
    'Explain what machine learning is in one sentence:',
  ];

  for (const prompt of prompts) {
    console.log('─'.repeat(60));
    console.log(`Prompt: "${prompt}"`);
    console.log('─'.repeat(60));

    const messages = [{ role: 'user', content: prompt }];

    const result = await model.chat(messages, {
      maxNewTokens: 2048,
      temperature: 0.6,
      topP: 0.9,
      reportPerformance: true,
    });

    console.log(result.text);
    console.log('─'.repeat(60));
    console.log(
      `TTFT: ${result.performance?.ttftMs.toFixed(2)}ms, Prefill: ${result.performance?.prefillTokensPerSecond.toFixed(2)} tokens/s, Decode: ${result.performance?.decodeTokensPerSecond.toFixed(2)} tokens/s`,
    );
    console.log('');
  }
}
