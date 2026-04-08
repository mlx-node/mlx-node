#!/usr/bin/env node
import { resolve } from 'node:path';
import { Gemma4Model } from '@mlx-node/core';

const MODEL_PATH = resolve(process.cwd(), '.cache', 'models', process.argv[2] || 'gemma-4-26b-a4b-mlx');
const model = await Gemma4Model.load(MODEL_PATH);

const prompts = [
  'What is the capital of France?',
  'What language do people speak in Brazil?',
  'Name three colors.',
  'What is 2+2?',
  'Hi there!',
];

for (let i = 0; i < prompts.length; i++) {
  const r = await model.chat(
    [{ role: 'user', content: prompts[i] }],
    { maxNewTokens: 32, temperature: 0.0 },
  );
  const p = r.performance;
  const ttft = p ? `TTFT=${p.ttftMs.toFixed(0)}ms` : '';
  const prefill = p ? `prefill=${p.prefillTokensPerSecond.toFixed(0)} tok/s` : '';
  const decode = p ? `decode=${p.decodeTokensPerSecond.toFixed(1)} tok/s` : '';
  console.log(`Run ${i+1}: ${r.numTokens}tok | ${ttft} ${prefill} ${decode} | "${r.text.slice(0,40).trim()}..."`);
}
process.exit(0);
