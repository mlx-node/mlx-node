#!/usr/bin/env oxnode
/// <reference types="node" />

import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { ROOT, assertCleanEnv, assertLoadedAddon, loadFixture } from '../common.js';
import { serveSamples, sessionTokenizer, type SessionModel, type Tokenizer } from './session.js';
import { FIXTURE, MODEL_PATH, readManifest } from './suite.js';

interface Model extends SessionModel {
  modelAssetsPath(): string;
  contextLimits(): unknown;
  hasBlockPagedCache(): boolean;
  hasMtpWeights(): boolean;
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

const tokenizer = await sessionTokenizer(assets, out, (path) => Qwen3Tokenizer.fromPretrained(path));
await serveSamples(model, tokenizer, cases, out, memoryStats);
process.exit(0);
