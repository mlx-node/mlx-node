#!/usr/bin/env node
/**
 * VLM Inference — Chat with images using Qianfan-OCR or PaddleOCR-VL
 *
 * Supports multi-turn conversation with image input and streaming output.
 *
 * Usage:
 *   oxnode examples/vlm-inference.ts [model-path] [--image <path>] [--prompt <text>]
 *
 * Examples:
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --prompt "Extract all text"
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --stream
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr                  # text-only chat
 */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import type { ChatMessage, ChatConfig } from '@mlx-node/core';
import { QianfanOCRModel } from '@mlx-node/vlm';

// Manual arg parsing — oxnode doesn't preserve shell quoting for multi-word values
const rawArgs = process.argv.slice(2);

function getFlag(name: string): string | undefined {
  const idx = rawArgs.indexOf(`--${name}`);
  if (idx === -1) return undefined;
  return rawArgs[idx + 1];
}
function hasFlag(name: string): boolean {
  return rawArgs.includes(`--${name}`);
}

// Collect positionals (args not starting with --)
const positionals: string[] = [];
for (let i = 0; i < rawArgs.length; i++) {
  if (rawArgs[i].startsWith('--')) {
    // Skip flags that take a value
    if (['--image', '--prompt', '--max-tokens', '--temperature', '-i', '-p'].includes(rawArgs[i])) i++;
    continue;
  }
  positionals.push(rawArgs[i]);
}

// --prompt collects everything between --prompt and the next -- flag
function getPromptArg(): string | undefined {
  const idx = rawArgs.indexOf('--prompt');
  const idx2 = rawArgs.indexOf('-p');
  const start = Math.max(idx, idx2);
  if (start === -1) return undefined;
  const parts: string[] = [];
  for (let i = start + 1; i < rawArgs.length; i++) {
    if (rawArgs[i].startsWith('--')) break;
    parts.push(rawArgs[i]);
  }
  return parts.join(' ') || undefined;
}

const modelPath = positionals[0];
const imagePath = getFlag('image') || getFlag('i');
const stream = hasFlag('stream');
const maxTokens = getFlag('max-tokens') ? parseInt(getFlag('max-tokens')!, 10) : 2048;
const temperature = getFlag('temperature') ? parseFloat(getFlag('temperature')!) : undefined;
const enableThinking = hasFlag('thinking');
const promptArg = getPromptArg();

if (!modelPath) {
  console.log(`VLM Inference — Chat with images using Qianfan-OCR

Usage:
  oxnode examples/vlm-inference.ts <model-path> [options]

Options:
  --image <path>       Image file to process (PNG/JPEG)
  --prompt <text>      Custom prompt (default: auto-selected based on mode)
  --stream             Stream output token-by-token
  --max-tokens <n>     Max tokens to generate (default: 2048)
  --temperature <f>    Sampling temperature
  --thinking           Enable Layout-as-Thought mode

Examples:
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --prompt "Parse to markdown"
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --stream --thinking
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr  # text-only`);
  process.exit(1);
}

const resolvedModelPath = resolve(process.cwd(), modelPath);

// --- Load model ---
console.log(`Loading model from: ${resolvedModelPath}`);
console.time('Load');
const model = await QianfanOCRModel.load(resolvedModelPath);
console.timeEnd('Load');
console.log();

// --- Build messages ---
const config: ChatConfig = {
  maxNewTokens: maxTokens,
  ...(temperature != null && { temperature }),
  ...(enableThinking && { enableThinking }),
  reportPerformance: true,
};

if (imagePath) {
  const imageBuffer = await readFile(resolve(process.cwd(), imagePath));
  const imageBytes = new Uint8Array(imageBuffer.buffer, imageBuffer.byteOffset, imageBuffer.byteLength);
  console.log(`Image: ${imagePath} (${imageBytes.length} bytes)`);

  const defaultPrompt = 'Extract all text from this image.';
  const prompt = promptArg || defaultPrompt;

  const messages: ChatMessage[] = [{ role: 'user', content: prompt, images: [imageBytes] }];

  if (stream) {
    // --- Streaming mode ---
    console.log(`Prompt: ${prompt}\n`);
    const t0 = Date.now();
    let tokens = 0;
    for await (const event of model.chatStream(messages, config)) {
      if (!event.done) {
        process.stdout.write(event.text);
      } else {
        tokens = event.numTokens;
        console.log();
        console.log('-'.repeat(80));
        console.log(`${tokens} tokens | ${Date.now() - t0}ms | finish: ${event.finishReason}`);
        if (event.thinking) {
          console.log(`\nThinking:\n${event.thinking}`);
        }
      }
    }
  } else {
    // --- Non-streaming mode ---
    console.log(`Prompt: ${prompt}\n`);
    console.time('Generate');
    const result = await model.chat(messages, config);
    console.timeEnd('Generate');
    console.log();
    console.log(result.text);
    console.log('-'.repeat(80));
    console.log(
      `${result.performance?.ttftMs}ms | ${result.performance?.prefillTokensPerSecond} tok/s | ${result.performance?.decodeTokensPerSecond} tok/s | ${result.numTokens} tokens | finish: ${result.finishReason}`,
    );
    if (result.thinking) {
      console.log(`\nThinking:\n${result.thinking}`);
    }

    // --- Multi-turn follow-up ---
    const followUp = 'Now format the extracted text as a markdown table if there are any tables.';
    messages.push({ role: 'assistant', content: result.rawText });
    messages.push({ role: 'user', content: followUp });

    console.log(`\n── Turn 2 (cache reuse) ──`);
    console.log(`User: ${followUp}\n`);
    console.time('Generate (turn 2)');
    const r2 = await model.chat(messages, config);
    console.timeEnd('Generate (turn 2)');
    console.log();
    console.log(r2.text);
    console.log('-'.repeat(80));
    console.log(
      `${r2.performance?.ttftMs}ms | ${r2.performance?.prefillTokensPerSecond} tok/s | ${r2.performance?.decodeTokensPerSecond} tok/s | ${r2.numTokens} tokens | finish: ${r2.finishReason}`,
    );
  }
} else {
  // --- Text-only multi-turn chat ---
  const messages: ChatMessage[] = [{ role: 'user', content: promptArg || 'What can you do? Answer briefly.' }];

  console.log(`Prompt: ${messages[0].content}\n`);

  if (stream) {
    for await (const event of model.chatStream(messages, config)) {
      if (!event.done) {
        process.stdout.write(event.text);
      } else {
        console.log();
        console.log('-'.repeat(80));
        console.log(
          `${event.performance?.ttftMs}ms | ${event.performance?.prefillTokensPerSecond} tok/s | ${event.performance?.decodeTokensPerSecond} tok/s | ${event.numTokens} tokens | finish: ${event.finishReason}`,
        );
      }
    }
  } else {
    console.time('Generate');
    const result = await model.chat(messages, config);
    console.timeEnd('Generate');
    console.log();
    console.log(result.text);
    console.log('-'.repeat(80));
    console.log(
      `${result.performance?.ttftMs}ms | ${result.performance?.prefillTokensPerSecond} tok/s | ${result.performance?.decodeTokensPerSecond} tok/s | ${result.numTokens} tokens | finish: ${result.finishReason}`,
    );
  }
}
