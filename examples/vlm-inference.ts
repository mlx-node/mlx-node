#!/usr/bin/env node
/**
 * VLM Inference — Chat with images using Qianfan-OCR
 *
 * Supports multi-turn conversation with image input and streaming
 * output via the unified `ChatSession` API. Because `@mlx-node/vlm`
 * depends on `@mlx-node/lm`, the `QianfanOCRModel` wrapper exported
 * from `@mlx-node/vlm` structurally satisfies `SessionCapableModel`
 * and can be handed directly to `new ChatSession(model)`.
 *
 * Usage:
 *   oxnode examples/vlm-inference.ts [model-path] [--image <path>] [--prompt <text>]
 *
 * Examples:
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --prompt "Extract all text"
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --image2 doc2.png
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --stream
 *   oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr                  # text-only chat
 */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import type { ChatConfig } from '@mlx-node/core';
import { ChatSession } from '@mlx-node/lm';
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
  if (rawArgs[i]!.startsWith('--')) {
    // Skip flags that take a value
    if (['--image', '--image2', '--prompt', '--max-tokens', '--temperature', '-i', '-p'].includes(rawArgs[i]!)) i++;
    continue;
  }
  positionals.push(rawArgs[i]!);
}

// --prompt collects everything between --prompt and the next -- flag
function getPromptArg(): string | undefined {
  const idx = rawArgs.indexOf('--prompt');
  const idx2 = rawArgs.indexOf('-p');
  const start = Math.max(idx, idx2);
  if (start === -1) return undefined;
  const parts: string[] = [];
  for (let i = start + 1; i < rawArgs.length; i++) {
    if (rawArgs[i]!.startsWith('--')) break;
    parts.push(rawArgs[i]!);
  }
  return parts.join(' ') || undefined;
}

const modelPath = positionals[0];
const imagePath = getFlag('image') || getFlag('i');
const imagePath2 = getFlag('image2');
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
  --image2 <path>      Optional second image to demo mid-session image-change restart
  --prompt <text>      Custom prompt (default: auto-selected based on mode)
  --stream             Stream output token-by-token
  --max-tokens <n>     Max tokens to generate (default: 2048)
  --temperature <f>    Sampling temperature
  --thinking           Enable Layout-as-Thought mode

Examples:
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --prompt "Parse to markdown"
  oxnode examples/vlm-inference.ts .cache/models/qianfan-ocr --image doc.png --image2 doc2.png
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

// --- Build config + session ---
const config: ChatConfig = {
  maxNewTokens: maxTokens,
  ...(temperature != null && { temperature }),
  ...(enableThinking && { enableThinking }),
  reportPerformance: true,
};

const session = new ChatSession(model);

if (imagePath) {
  const imageBuffer = await readFile(resolve(process.cwd(), imagePath));
  const imageBytes = new Uint8Array(imageBuffer.buffer, imageBuffer.byteOffset, imageBuffer.byteLength);
  console.log(`Image: ${imagePath} (${imageBytes.length} bytes)`);

  const defaultPrompt = 'Extract all text from this image.';
  const prompt = promptArg || defaultPrompt;

  if (stream) {
    // --- Streaming mode — image attached on turn 1 only ---
    console.log(`Prompt: ${prompt}\n`);
    const t0 = Date.now();
    let tokens = 0;
    for await (const event of session.sendStream(prompt, { images: [imageBytes], config })) {
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
    const result = await session.send(prompt, { images: [imageBytes], config });
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

    // --- Multi-turn follow-up. Omitting `images` keeps the current
    // image cache state, so this takes the cheap delta path. ---
    const followUp = 'Now format the extracted text as a markdown table if there are any tables.';
    console.log(`\n── Turn 2 (cache reuse, no image re-prefill) ──`);
    console.log(`User: ${followUp}\n`);
    console.time('Generate (turn 2)');
    const r2 = await session.send(followUp, { config });
    console.timeEnd('Generate (turn 2)');
    console.log();
    console.log(r2.text);
    console.log('-'.repeat(80));
    console.log(
      `${r2.performance?.ttftMs}ms | ${r2.performance?.prefillTokensPerSecond} tok/s | ${r2.performance?.decodeTokensPerSecond} tok/s | ${r2.numTokens} tokens | finish: ${r2.finishReason}`,
    );

    // --- Optional turn 3: image-change demo. Passing a NEW image
    // triggers `chatSessionStart` + full re-prefill of the preserved
    // history against the new image set. TTFT will spike on this
    // turn — that's expected, not a bug. ---
    if (imagePath2) {
      const imageBuffer2 = await readFile(resolve(process.cwd(), imagePath2));
      const imageBytes2 = new Uint8Array(imageBuffer2.buffer, imageBuffer2.byteOffset, imageBuffer2.byteLength);
      console.log(`\nImage2: ${imagePath2} (${imageBytes2.length} bytes)`);
      console.log(`\n── Turn 3 (image change — expect TTFT spike) ──`);
      const imageChangePrompt = 'Now describe this new image.';
      console.log(`User: ${imageChangePrompt}\n`);
      console.time('Generate (turn 3, image change)');
      const r3 = await session.send(imageChangePrompt, { images: [imageBytes2], config });
      console.timeEnd('Generate (turn 3, image change)');
      console.log();
      console.log(r3.text);
      console.log('-'.repeat(80));
      console.log(
        `${r3.performance?.ttftMs}ms | ${r3.performance?.prefillTokensPerSecond} tok/s | ${r3.performance?.decodeTokensPerSecond} tok/s | ${r3.numTokens} tokens | finish: ${r3.finishReason}`,
      );
      console.log(
        '\nNote: turn 3 TTFT is expected to be much higher than turn 2 because the image change triggers a full chatSessionStart re-prefill against the preserved history.',
      );
    }
  }
} else {
  // --- Text-only multi-turn chat ---
  const firstPrompt = promptArg || 'What can you do? Answer briefly.';
  console.log(`Prompt: ${firstPrompt}\n`);

  if (stream) {
    for await (const event of session.sendStream(firstPrompt, { config })) {
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
    const result = await session.send(firstPrompt, { config });
    console.timeEnd('Generate');
    console.log();
    console.log(result.text);
    console.log('-'.repeat(80));
    console.log(
      `${result.performance?.ttftMs}ms | ${result.performance?.prefillTokensPerSecond} tok/s | ${result.performance?.decodeTokensPerSecond} tok/s | ${result.numTokens} tokens | finish: ${result.finishReason}`,
    );
  }
}
