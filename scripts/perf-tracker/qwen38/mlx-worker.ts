#!/usr/bin/env oxnode
/// <reference types="node" />

import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { ROOT, assertCleanEnv, assertLoadedAddon, loadFixture } from '../common.js';
import { serveSamples, sessionTokenizer, type SessionModel, type Tokenizer } from './session.js';
import { DRAFT_DIRS, FIXTURE, MODEL_PATH, readManifest, type Protocol } from './suite.js';

interface Model extends SessionModel {
  modelAssetsPath(): string;
  contextLimits(): unknown;
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
  loadModel: (path: string, options: { draftModelPath: string } & Protocol['mlxLoadOptions']) => Promise<Model>;
};
const { memoryStats, Qwen3Tokenizer } = (await import(pathToFileURL(join(ROOT, 'packages/core/index.cjs')).href)) as {
  memoryStats: () => unknown;
  Qwen3Tokenizer: { fromPretrained(path: string): Promise<Tokenizer> };
};
assertLoadedAddon();

const loadStart = performance.now();
// An explicit draftModelPath fails the load unless the DFlash2 draft validates against the target.
const model = await loadModel(MODEL_PATH, { draftModelPath: DRAFT_DIRS['mlx-node'], ...protocol.mlxLoadOptions });
if (!model.hasMtpWeights()) throw new Error('No speculative draft loaded; not the recorded protocol');
const assets = model.modelAssetsPath();
console.log(
  JSON.stringify({
    event: 'loaded',
    loadMs: performance.now() - loadStart,
    pid: process.pid,
    assets,
    context: model.contextLimits(),
    draft: DRAFT_DIRS['mlx-node'],
    kvFormat: protocol.mlxLoadOptions.kvFormat,
    hasMtp: model.hasMtpWeights(),
  }),
);

const tokenizer = await sessionTokenizer(assets, out, (path) => Qwen3Tokenizer.fromPretrained(path));
await serveSamples(model, tokenizer, cases, out, memoryStats, protocol.mlxRequest);
process.exit(0);
