#!/usr/bin/env node
/**
 * Session reset determinism test.
 *
 * One `ChatSession`, five iterations, `reset()` between each. Every
 * zero-temperature run against the same prompt must produce the exact
 * same output — if `reset()` fully clears session state (native KV
 * caches + TS-side history/image key/turn counter), byte-identical
 * output is the expected result.
 *
 * Usage:
 *   oxnode examples/lm_samep.ts [model-name]
 */
import { resolve } from 'node:path';

import type { SessionCapableModel } from '@mlx-node/lm';
import { ChatSession, HarrierModel, loadModel } from '@mlx-node/lm';

const modelName = process.argv[2] || 'qwen3.5-0.8b-mlx-bf16';
const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', modelName);

console.log(`Loading model from: ${MODEL_PATH}`);
const loadedModel = await loadModel(MODEL_PATH);
if (loadedModel instanceof HarrierModel) {
  console.error('Embedding model.');
  process.exit(1);
}
const model = loadedModel as unknown as SessionCapableModel;
console.log('Model loaded\n');

const session = new ChatSession(model, { system: 'You are a helpful assistant. Be concise.' });
const userMessage = 'What is the capital of France?';
const config = { maxNewTokens: 64, temperature: 0 };

const outputs: string[] = [];
for (let i = 1; i <= 5; i++) {
  const result = await session.send(userMessage, { config });
  console.log(`── Turn ${i} ──`);
  console.log(`Assistant (${result.numTokens} tok, ${result.finishReason}): ${result.text.slice(0, 100)}`);
  outputs.push(result.text);
  await session.reset();
}

const allEqual = outputs.every((t) => t === outputs[0]);
if (!allEqual) {
  console.error('FAIL: reset() leaked state — outputs diverged at temperature=0');
  for (let i = 0; i < outputs.length; i++) {
    console.error(`  [${i}] ${JSON.stringify(outputs[i]!.slice(0, 120))}`);
  }
  process.exit(1);
}
console.log('\nPASS: all 5 reset-then-send iterations produced byte-identical output');
