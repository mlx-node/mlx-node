#!/usr/bin/env oxnode
/// <reference types="node" />

import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';

import { ROOT, assertCleanEnv, assertLoadedAddon, loadFixture, writeJson } from '../common.js';
import { FIXTURE, MODEL_PATH, readManifest } from './suite.js';

interface StreamEvent {
  done: boolean;
  promptTokens?: number;
  cachedTokens?: number;
  [field: string]: unknown;
}

interface Model {
  modelAssetsPath(): string;
  contextLimits(): unknown;
  hasBlockPagedCache(): boolean;
  hasMtpWeights(): boolean;
  resetCaches(): Promise<void>;
  chatStreamSessionStart(messages: unknown[], config: Record<string, unknown>): AsyncGenerator<StreamEvent>;
}

interface Tokenizer {
  applyChatTemplate(
    messages: unknown[],
    addGenerationPrompt: boolean,
    tools: null,
    enableThinking: boolean,
  ): Promise<Uint32Array>;
}

const out = process.argv[2];
if (!out) throw new Error('Usage: oxnode scripts/perf-tracker/qwen38/mlx-worker.ts <out dir>');
const { protocol } = await readManifest();
assertCleanEnv([...Object.keys(protocol.mlxEnv), 'MLX_INFERENCE_TRACE', 'MLX_INFERENCE_TRACE_FILE']);
for (const [name, value] of Object.entries(protocol.mlxEnv)) {
  if (process.env[name] !== value) throw new Error(`${name} must be ${value} for the recorded protocol`);
}
const { cases } = await loadFixture(FIXTURE);
const { loadModel } = (await import(pathToFileURL(join(ROOT, 'packages/lm/src/models/model-loader.ts')).href)) as {
  loadModel: (path: string, options: { autoLoadDraft: boolean }) => Promise<Model>;
};
const { memoryStats, Qwen3Tokenizer } = (await import(pathToFileURL(join(ROOT, 'packages/core/index.cjs')).href)) as {
  memoryStats: () => unknown;
  Qwen3Tokenizer: { fromPretrained(path: string): Promise<Tokenizer> };
};
assertLoadedAddon();

const loadStart = performance.now();
// loadModel attaches a qwen3.8-27b-dflash2 directory found beside the GGUF;
// the recorded protocol loads no draft.
const model = await loadModel(MODEL_PATH, { autoLoadDraft: false });
const assets = model.modelAssetsPath();
console.log(
  JSON.stringify({
    event: 'loaded',
    loadMs: performance.now() - loadStart,
    pid: process.pid,
    assets,
    context: model.contextLimits(),
    paged: model.hasBlockPagedCache(),
    hasMtp: model.hasMtpWeights(),
  }),
);
// The recorded protocol is the block-paged BF16 K/V cache; the flat fallback defaults to int8 K/V.
if (!model.hasBlockPagedCache()) throw new Error('Block-paged cache inactive; not the recorded protocol');

// Native sessions render with preserve_thinking=true, the standalone tokenizer with
// false; a private copy of the loaded template renders what the session renders.
const tokenizerDir = join(out, 'session-tokenizer');
await mkdir(tokenizerDir, { recursive: true });
await copyFile(join(assets, 'tokenizer.json'), join(tokenizerDir, 'tokenizer.json'));
const config = JSON.parse(await readFile(join(assets, 'tokenizer_config.json'), 'utf8')) as { chat_template?: unknown };
const template =
  typeof config.chat_template === 'string'
    ? config.chat_template
    : await readFile(join(assets, 'chat_template.jinja'), 'utf8');
config.chat_template = `{% set preserve_thinking = true %}${template}`;
await writeFile(join(tokenizerDir, 'tokenizer_config.json'), JSON.stringify(config));
const tokenizer = await Qwen3Tokenizer.fromPretrained(join(tokenizerDir, 'tokenizer.json'));

for await (const line of createInterface({ input: process.stdin })) {
  const request = JSON.parse(line) as { action?: string; name: string; run: number | string; maxNewTokens: number };
  if (request.action === 'exit') break;
  const item = cases.find((candidate) => candidate.name === request.name);
  if (!item) throw new Error(`Unknown case ${request.name}`);
  await model.resetCaches();
  const tokens = Array.from(await tokenizer.applyChatTemplate(item.messages, true, null, true));
  if (JSON.stringify(tokens) !== JSON.stringify(item.tokenIds)) throw new Error('Prompt token mismatch');
  const started = performance.now();
  let result: StreamEvent | undefined;
  let firstEventMs: number | undefined;
  for await (const event of model.chatStreamSessionStart(item.messages, {
    maxNewTokens: request.maxNewTokens,
    temperature: 0,
    topK: 0,
    topP: 1,
    minP: 0,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    reasoningEffort: 'high',
    includeReasoning: true,
    enableMtp: false,
    reuseCache: true,
    reportPerformance: true,
    cacheSalt: `benchmark-${process.pid}-${request.run}-${request.name}`,
  })) {
    firstEventMs ??= performance.now() - started;
    if (event.done) result = event;
  }
  if (!result) throw new Error('No terminal result');
  const report: Record<string, unknown> = {
    event: 'result',
    runtime: 'mlx-node',
    name: request.name,
    run: request.run,
    pid: process.pid,
    wallMs: performance.now() - started,
    firstEventMs,
    ...result,
    memory: memoryStats(),
  };
  await writeJson(join(out, 'raw', `mlx-${request.run}-${request.name}.json`), report);
  if (result.promptTokens !== item.tokenIds.length || result.cachedTokens !== 0) {
    throw new Error(
      `Prompt/cache accounting mismatch: expected ${item.tokenIds.length}/0; received ${result.promptTokens}/${result.cachedTokens}`,
    );
  }
  const { text: _text, rawText: _rawText, thinking: _thinking, publicRawText: _publicRawText, ...summary } = report;
  console.log(JSON.stringify(summary));
}
process.exit(0);
