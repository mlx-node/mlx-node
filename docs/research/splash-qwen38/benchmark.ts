/// <reference types="node" />

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';

// Run from the repository root. See README.md for the frozen-binary protocol.
import type { ChatConfig, ChatMessage, ChatResult } from '../../../packages/core/index.cjs';

const [binding, output, mode = 'dflash', names = 'short', count = '1', tokenLimit = '128'] = process.argv.slice(2);
if (!binding || !output || !['ar', 'dflash'].includes(mode) || process.argv.length > 8) {
  throw new Error(
    'Usage: oxnode docs/research/splash-qwen38/benchmark.ts <addon.node> <output.json> [ar|dflash] [short,6k,16k,32k] [runs] [tokens]',
  );
}
for (const value of [count, tokenLimit]) {
  if (!/^\d+$/.test(value) || Number(value) < 1) throw new Error('Runs and tokens must be positive integers');
}
const core: typeof import('../../../packages/core/index.cjs') = createRequire(import.meta.url)(resolve(binding));
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
interface Case {
  name: string;
  messages: ChatMessage[];
  promptTokens?: number;
}
const fixtures: Case[] = JSON.parse(
  await readFile('.cache/benchmarks/fixtures-public-2026-09-11/qwen38-inputs.json', 'utf8'),
);
const cases: Case[] = [
  {
    name: 'short',
    messages: [
      {
        role: 'user',
        content:
          'Implement a TypeScript LRU cache with a fixed capacity. Include get, set, delete, and tests for eviction and updating existing keys. Explain the design and complexity in detail.',
      },
    ],
  },
  ...fixtures,
];
// Record only performance controls, never the complete process environment.
const environment = Object.fromEntries(
  [
    'MLX_QMM_SPLITK_MIN_M',
    'MLX_METAL_COMMAND_TRACE',
    'MLX_DFLASH2_PHASE_TIME',
    'MLX_MAX_OPS_PER_BUFFER',
    'MLX_MAX_MB_PER_BUFFER',
  ].map((name) => [name, process.env[name] ?? null]),
);
const target = resolve('.cache/models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf');
const draft = resolve('.cache/models/qwen3.8-27b-dflash2');
const started = performance.now();
const model = await core.Qwen35Model.load(target, mode === 'dflash' ? { draftModelPath: draft } : undefined);
const records: (ChatResult & {
  name: string;
  run: number;
  mode: string;
  wallMs: number;
  inputHash: string;
  outputHash: string;
})[] = [];
console.log(JSON.stringify({ event: 'loaded', mode, loadMs: performance.now() - started, memory: core.memoryStats() }));
const config: ChatConfig = {
  maxNewTokens: Number(tokenLimit),
  temperature: 0,
  topK: 0,
  topP: 1,
  minP: 0,
  repetitionPenalty: 1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  reasoningEffort: 'high',
  includeReasoning: true,
  enableMtp: mode === 'dflash',
  mtpAdaptiveDepth: false,
  maxConsecutiveTokens: 0,
  maxNgramRepeats: 0,
  reuseCache: true,
  reportPerformance: true,
};
await model.chatSessionStart(cases[0].messages, { ...config, maxNewTokens: 16 });
for (let run = 1; run <= Number(count); run++) {
  for (const name of names.split(',')) {
    const item = cases.find((c) => c.name === name);
    if (!item) throw new Error(`Unknown case ${name}`);
    await model.resetCaches();
    console.log(JSON.stringify({ event: 'benchmark-start', name, run }));
    const t = performance.now();
    const result = await model.chatSessionStart(item.messages, config);
    const row = {
      name,
      run,
      mode,
      wallMs: performance.now() - t,
      inputHash: hash(item.messages),
      outputHash: hash(result.rawText),
      ...result,
      memory: core.memoryStats(),
    };
    if (result.cachedTokens !== 0 || result.numTokens !== Number(tokenLimit))
      throw new Error(`Invalid cold sample: ${JSON.stringify(result.performance)}`);
    if (item.promptTokens && result.promptTokens !== item.promptTokens)
      throw new Error('Fixture prompt token count drift');
    if (mode === 'dflash' && !((result.performance?.mtpCycles ?? 0) > 0)) throw new Error('DFlash did not execute');
    records.push(row);
    await writeFile(output, JSON.stringify({ binding, target, draft, environment, config, records }, null, 2));
    const { rawText: _raw, text: _text, thinking: _thinking, publicRawText: _public, ...summary } = row;
    console.log(JSON.stringify(summary));
  }
}
await model.resetCaches();
process.exit(0);
