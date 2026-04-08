#!/usr/bin/env node
/**
 * Test Gemma4 Model Performance
 *
 * Usage:
 *   oxnode examples/gemma4.ts [model-name]
 *   oxnode examples/gemma4.ts gemma-4-26b-a4b-mlx
 *   oxnode examples/gemma4.ts gemma-4-e2b
 */

import { resolve } from 'node:path';

import { Gemma4Model } from '@mlx-node/core';

const modelName = process.argv[2] || 'gemma-4-26b-a4b-mlx';
const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

console.log(`Loading model from: ${MODEL_PATH}`);
const t0 = Date.now();
const model = await Gemma4Model.load(MODEL_PATH);
console.log(`Model loaded in ${Date.now() - t0}ms\n`);

// Multi-turn conversation
const messages: { role: string; content: string }[] = [
  { role: 'user', content: 'What is the capital of France?' },
];

// Turn 1
console.log('── Turn 1 ──');
console.log(`User: ${messages[0].content}`);
const t1 = Date.now();
const r1 = await model.chat(messages, { maxNewTokens: 256 });
const e1 = Date.now() - t1;
console.log(`Model: ${r1.text}`);
console.log(`  ${r1.numTokens} tokens in ${e1}ms (${(r1.numTokens / (e1 / 1000)).toFixed(1)} tok/s)\n`);

// Turn 2
messages.push({ role: 'assistant', content: r1.text });
messages.push({ role: 'user', content: 'What about Germany?' });

console.log('── Turn 2 ──');
console.log(`User: ${messages[2].content}`);
const t2 = Date.now();
const r2 = await model.chat(messages, { maxNewTokens: 256 });
const e2 = Date.now() - t2;
console.log(`Model: ${r2.text}`);
console.log(`  ${r2.numTokens} tokens in ${e2}ms (${(r2.numTokens / (e2 / 1000)).toFixed(1)} tok/s)\n`);

// Turn 3
messages.push({ role: 'assistant', content: r2.text });
messages.push({ role: 'user', content: 'Which of those two cities has more museums?' });

console.log('── Turn 3 ──');
console.log(`User: ${messages[4].content}`);
const t3 = Date.now();
const r3 = await model.chat(messages, { maxNewTokens: 256 });
const e3 = Date.now() - t3;
console.log(`Model: ${r3.text}`);
console.log(`  ${r3.numTokens} tokens in ${e3}ms (${(r3.numTokens / (e3 / 1000)).toFixed(1)} tok/s)\n`);

console.log('── Summary ──');
const totalTokens = r1.numTokens + r2.numTokens + r3.numTokens;
const totalMs = e1 + e2 + e3;
console.log(`Total: ${totalTokens} tokens in ${(totalMs / 1000).toFixed(1)}s (${(totalTokens / (totalMs / 1000)).toFixed(1)} tok/s avg)`);

process.exit(0);
