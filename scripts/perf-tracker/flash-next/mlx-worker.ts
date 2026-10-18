#!/usr/bin/env oxnode
/// <reference types="node" />

import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { ROOT, assertCleanEnv, assertLoadedAddon, loadFixture, readJson, sh } from '../common.js';
import { serveSamples, sessionTokenizer, type SessionModel, type Tokenizer } from '../qwen38/session.js';
import { FIXTURE } from '../qwen38/suite.js';
import { PROTOCOL, type Provenance, type Residency } from './suite.js';

interface Model extends SessionModel {
  modelAssetsPath(): string;
  contextLimits(): unknown;
  residencyInfo?(): Residency;
  hasMtpWeights(): boolean;
  mtpAutoEnabled(): boolean;
}

const out = process.argv[2];
if (!out) throw new Error('Usage: oxnode scripts/perf-tracker/flash-next/mlx-worker.ts <out dir>');
assertCleanEnv([...Object.keys(PROTOCOL.mlxEnv), 'MLX_INFERENCE_TRACE', 'MLX_INFERENCE_TRACE_FILE']);
for (const [name, value] of Object.entries(PROTOCOL.mlxEnv)) {
  if (process.env[name] !== value) throw new Error(`The mlx worker needs ${name}=${value}`);
}
const provenance = await readJson<Provenance>(join(out, 'provenance.json'));
const { cases } = await loadFixture(FIXTURE);
const { loadModel } = (await import(pathToFileURL(join(ROOT, 'packages/lm/src/models/model-loader.ts')).href)) as {
  loadModel: (path: string, options: { autoLoadDraft: boolean; auxiliaryModelPath: string }) => Promise<Model>;
};
const { memoryStats, Qwen3Tokenizer } = (await import(pathToFileURL(join(ROOT, 'packages/core/index.cjs')).href)) as {
  memoryStats: () => unknown;
  Qwen3Tokenizer: { fromPretrained(path: string): Promise<Tokenizer> };
};
assertLoadedAddon();

// The automatic weight budget is sized from free memory at this moment.
const beforeLoad = {
  at: new Date().toISOString(),
  pressureLevel: Number(sh('sysctl', ['-n', 'kern.memorystatus_vm_pressure_level'])),
  vmStat: sh('vm_stat', []),
};
const loadStart = performance.now();
const model = await loadModel(provenance.model.splits[0].path, {
  ...PROTOCOL.mlxLoadOptions,
  auxiliaryModelPath: provenance.mtp.auxDir,
});
const loadMs = performance.now() - loadStart;
if (typeof model.residencyInfo !== 'function') throw new Error('The GGUF did not load as a qwen4_exp model');
if (!model.hasMtpWeights()) throw new Error('mlx-node loaded no MTP weights; not the recorded protocol');
const assets = model.modelAssetsPath();
console.log(
  JSON.stringify({
    event: 'loaded',
    loadMs,
    pid: process.pid,
    assets,
    context: model.contextLimits(),
    residency: model.residencyInfo(),
    memory: memoryStats(),
    hasMtp: model.hasMtpWeights(),
    mtpAutoEnabled: model.mtpAutoEnabled(),
    beforeLoad,
  }),
);

const tokenizer = await sessionTokenizer(assets, out, (path) => Qwen3Tokenizer.fromPretrained(path));
await serveSamples(model, tokenizer, cases, out, memoryStats, PROTOCOL.mlxRequest);
process.exit(0);
