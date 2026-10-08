#!/usr/bin/env oxnode

import { resolve } from 'node:path';

import { ChatSession, HarrierModel, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const modelArg = process.argv[2];
if (modelArg == null) {
  console.error('Usage: oxnode scripts/smoke-model.ts <model-directory>');
  process.exit(2);
}

const modelPath = resolve(modelArg);
const startedAt = performance.now();
console.log(JSON.stringify({ event: 'load_start', modelPath }));

const model = await loadModel(modelPath);
if (model instanceof HarrierModel) {
  throw new Error(`Expected a generative model, got HarrierModel: ${modelPath}`);
}

const loadedAt = performance.now();
const session = new ChatSession(model as unknown as SessionCapableModel, {
  system: 'Reply with exactly OK.',
});
const result = await session.send('Reply with exactly OK.', {
  config: {
    maxNewTokens: 1,
    temperature: 0,
    reasoningEffort: 'none',
    reportPerformance: true,
  },
});

console.log(
  JSON.stringify({
    event: 'smoke_ok',
    modelPath,
    loadMs: Math.round(loadedAt - startedAt),
    totalMs: Math.round(performance.now() - startedAt),
    finishReason: result.finishReason,
    numTokens: result.numTokens,
    text: result.text,
    rawText: result.rawText,
    performance: result.performance,
  }),
);
