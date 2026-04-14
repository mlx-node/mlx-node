#!/usr/bin/env node
/**
 * Session isolation demo.
 *
 * Replaces the legacy `reuseCache: false` example: instead of disabling
 * the cache via a per-call flag, each iteration instantiates a FRESH
 * `ChatSession`, runs exactly one `send`, and drops the session. This
 * verifies that different sessions don't see each other's state — each
 * fresh session sees `turns === 1` after its single call, and two
 * zero-temperature runs on independent sessions produce byte-identical
 * output.
 *
 * Usage:
 *   oxnode examples/lm_session_isolation.ts [model-name]
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

const userMessage = 'What is the capital of France?';
const config = { maxNewTokens: 64, temperature: 0 };
const outputs: string[] = [];

for (let i = 1; i <= 3; i++) {
  // Fresh session per iteration — no state carries over.
  const session = new ChatSession(model, { system: 'You are a helpful assistant. Be concise.' });
  const result = await session.send(userMessage, { config });
  console.log(`── Iteration ${i} ──`);
  console.log(`Assistant (${result.numTokens} tok, ${result.finishReason}): ${result.text.slice(0, 100)}`);
  console.log(`session.turns = ${session.turns} (expected 1)`);
  if (session.turns !== 1) {
    console.error(`FAIL: fresh session turns counter should be 1, got ${session.turns}`);
    process.exit(1);
  }
  outputs.push(result.text);
}

// Determinism check: all zero-temperature outputs on fresh sessions
// should be byte-identical, proving no cross-session state leakage.
const allEqual = outputs.every((t) => t === outputs[0]);
if (!allEqual) {
  console.error('FAIL: fresh-session outputs diverged at temperature=0 — possible state leak across sessions');
  for (let i = 0; i < outputs.length; i++) {
    console.error(`  [${i}] ${JSON.stringify(outputs[i]!.slice(0, 120))}`);
  }
  process.exit(1);
}
console.log('\nPASS: all fresh sessions produced byte-identical output at temperature=0');
