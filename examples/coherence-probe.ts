#!/usr/bin/env node
// Reusable coherence probe for the garbage-output bisection.
// Loads a small WARM model and greedily decodes a trivial prompt; prints
// rawText + finishReason so each rebuild can be judged coherent vs garbage.
import { ChatSession, loadModel, type SessionCapableModel } from '@mlx-node/lm';

const model = process.argv[2] ?? '/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16';
const loaded = await loadModel(model);
const session = new ChatSession(loaded as unknown as SessionCapableModel, {
  system: 'You are a helpful assistant.',
});
const res = await session.send('Say hello in one short sentence.', {
  config: { maxNewTokens: 32, temperature: 0, reportPerformance: true, reuseCache: false },
});
const r = res as any;
console.log(`MODEL ${model}`);
console.log(`  numTokens=${r.numTokens} finishReason=${r.finishReason}`);
console.log(`  rawText=${JSON.stringify(r.rawText)}`);
console.log(`  VERDICT=${/[A-Za-z].*[A-Za-z].*[A-Za-z]/.test(String(r.rawText)) && r.finishReason !== 'repetition' ? 'LOOKS-COHERENT' : 'GARBAGE/REPETITION'}`);
