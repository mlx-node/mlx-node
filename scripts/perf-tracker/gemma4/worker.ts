#!/usr/bin/env oxnode
/// <reference types="node" />

import { spawn } from 'node:child_process';
import { closeSync, openSync } from 'node:fs';
import { readFile, rm } from 'node:fs/promises';
import { createServer } from 'node:net';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

import type { ChatConfig, ChatMessage } from '../../../packages/core/index.cjs';
import {
  ADDON_PATH,
  ROOT,
  assertCleanEnv,
  assertLoadedAddon,
  llamaServerPath,
  sha256,
  sha256File,
  sleep,
  writeJson,
} from '../common.js';
import { MODEL_FILE, PILOT_RUN, RUNTIMES, readFixture, sampleId, samplePath, type Runtime } from './suite.js';

type Core = typeof import('../../../packages/core/index.cjs');

const { values } = parseArgs({
  options: {
    out: { type: 'string' },
    runtime: { type: 'string' },
    case: { type: 'string' },
    run: { type: 'string' },
  },
});
const { out, case: name, run } = values;
const runtime = values.runtime as Runtime;
if (!out || !name || !run || !RUNTIMES.includes(runtime)) {
  throw new Error('Usage: worker.ts --out <dir> --runtime mlx|llama --case <name> --run <id>');
}
const { manifest, cases } = await readFixture();
const { protocol } = manifest;
const data = cases.find((item) => item.name === name);
const warmup = cases.find((item) => item.name === protocol.warmupCase);
if (!data || !warmup) throw new Error(`Unknown case: ${name}`);

// Pilots run without the protocol variables, as the historical pilots did.
const expectedEnv = run === PILOT_RUN ? {} : protocol.mlxEnv;
assertCleanEnv(Object.keys(expectedEnv));
for (const [key, value] of Object.entries(expectedEnv)) {
  if (process.env[key] !== value) throw new Error(`${key} must be ${value}; run this through run.ts`);
}

const id = sampleId(runtime, data.name, run);
const save = (sample: unknown) => writeJson(samplePath(out, id), sample);

async function mlx() {
  const planLog = join(out!, 'raw', `${id}.plans.jsonl`);
  // The addon appends; a retried sample must not inherit a failed attempt's plans.
  await rm(planLog, { force: true });
  process.env.MLX_NODE_LOG = 'mlx_core::decode_tuning=info';
  process.env.MLX_NODE_LOG_FILE = planLog;
  const n = data!.promptTokens;
  const { Gemma4Model, getMemorySnapshot, resetPeakMemory } = (await import(
    pathToFileURL(join(ROOT, 'packages/core/index.cjs')).href
  )) as Core;
  assertLoadedAddon();
  const started = performance.now();
  const model = await Gemma4Model.load(MODEL_FILE);
  const loadMs = performance.now() - started;
  const memoryAfterLoad = getMemorySnapshot();
  const config: ChatConfig = {
    temperature: 0,
    topK: 1,
    topP: 1,
    minP: 0,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    reasoningEffort: 'high',
    tools: data!.tools as ChatConfig['tools'],
    enableMtp: false,
    reuseCache: true,
    reportPerformance: true,
  };
  await model.chatSessionStart(warmup!.messages as ChatMessage[], { ...config, maxNewTokens: protocol.warmupTokens });
  await model.resetCaches();
  resetPeakMemory();
  const memoryBeforeRequest = getMemorySnapshot();
  const start = performance.now();
  const result = await model.chatSessionStart(data!.messages as ChatMessage[], {
    ...config,
    maxNewTokens: protocol.outputTokens,
  });
  const wallMs = performance.now() - start;
  const memoryAfterRequest = getMemorySnapshot();
  const p = result.performance!;
  if (result.promptTokens !== n || result.cachedTokens !== 0 || result.numTokens !== protocol.outputTokens) {
    throw new Error(`Workload mismatch: ${JSON.stringify(result)}`);
  }
  const decodeMs = ((result.numTokens - 1) / p.decodeTokensPerSecond) * 1000;
  const sample = {
    runtime,
    env: Object.fromEntries(Object.entries(process.env).filter(([key]) => key.startsWith('MLX_'))),
    addonSha256: await sha256File(ADDON_PATH),
    name: data!.name,
    n,
    run,
    loadMs,
    memoryAfterLoad,
    memoryBeforeRequest,
    memoryAfterRequest,
    decodePlans: await decodePlans(planLog),
    wallMs,
    prefillMs: p.ttftMs,
    prefillTps: p.prefillTokensPerSecond,
    decodeMs,
    decodeTps: p.decodeTokensPerSecond,
    accountedMs: p.ttftMs + decodeMs,
    promptTokens: result.promptTokens,
    generatedTokens: result.numTokens,
    cachedTokens: result.cachedTokens,
    textSha256: sha256(result.rawText),
    fixtureSha256: data!.sha256,
    completedAt: new Date().toISOString(),
    result,
  };
  await save(sample);
  console.log(JSON.stringify({ ...sample, result: undefined }));
  await model.resetCaches();
}

interface PlanEvent {
  event: string;
  context_bucket: number;
  stage: string;
  early_layers: number;
  grouped_stripes?: number;
  samples: string;
  candidate_early_layers: string;
  candidate_stripes?: string;
}

// Every fresh process calibrates at least one context bucket; an empty list
// means the event or its log target changed, not that no plan was chosen.
async function decodePlans(planLog: string) {
  const plans = (await readFile(planLog, 'utf8'))
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as PlanEvent)
    .filter((event) => event.event === 'decode_tuned')
    .map((event) => ({
      contextBucket: event.context_bucket,
      stage: event.stage,
      earlyLayers: event.early_layers,
      groupedStripes: event.grouped_stripes ?? null,
      samples: JSON.parse(event.samples) as number[][],
      candidateEarlyLayers: JSON.parse(event.candidate_early_layers) as number[],
      candidateStripes:
        event.candidate_stripes === undefined ? null : (JSON.parse(event.candidate_stripes) as number[]),
    }));
  if (!plans.length) throw new Error(`${planLog}: no decode_tuned event`);
  return plans;
}

async function freePort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((done) => server.listen(0, '127.0.0.1', done));
  const address = server.address();
  if (!address || typeof address !== 'object') throw new Error('No free port');
  await new Promise<void>((done, fail) => server.close((error) => (error ? fail(error) : done())));
  return address.port;
}

async function llama() {
  const n = data!.promptTokens;
  const port = await freePort();
  const args = [
    '-m',
    MODEL_FILE,
    ...protocol.llamaServerArgs,
    '--host',
    '127.0.0.1',
    '--port',
    String(port),
    '--no-webui',
  ];
  const log = openSync(join(out!, 'raw', `${id}.stderr.txt`), 'w');
  const started = performance.now();
  const server = spawn(llamaServerPath(), args, { stdio: ['ignore', log, log], env: process.env });
  closeSync(log);
  const exit = new Promise<void>((done) => server.once('exit', () => done()));
  let ready = false;
  try {
    for (let i = 0; i < 240; i++) {
      if (server.exitCode !== null) throw new Error(`llama-server exited: ${server.exitCode}`);
      try {
        if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) {
          ready = true;
          break;
        }
      } catch {
        // Not listening yet.
      }
      await sleep(250);
    }
    if (!ready) throw new Error('llama-server readiness timeout');
    const loadMs = performance.now() - started;
    const post = async (path: string, body: unknown) => {
      const response = await fetch(`http://127.0.0.1:${port}/${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`${path}: ${await response.text()}`);
      return response.json() as Promise<Record<string, any>>;
    };
    const tokenized = await post('tokenize', { content: data!.rendered, add_special: false, parse_special: true });
    if (JSON.stringify(tokenized.tokens) !== JSON.stringify(data!.tokenIds)) throw new Error('Tokenizer mismatch');
    const request = {
      prompt: data!.tokenIds,
      temperature: 0,
      top_k: 1,
      top_p: 1,
      min_p: 0,
      repeat_penalty: 1,
      presence_penalty: 0,
      frequency_penalty: 0,
      seed: 1234,
      cache_prompt: false,
      stream: false,
      return_tokens: true,
      ignore_eos: false,
    };
    await post('completion', { ...request, prompt: warmup!.tokenIds, n_predict: protocol.warmupTokens });
    const start = performance.now();
    const result = await post('completion', { ...request, n_predict: protocol.outputTokens });
    const wallMs = performance.now() - start;
    const t = result.timings;
    if (t.prompt_n !== n || t.predicted_n !== protocol.outputTokens || t.cache_n !== 0 || result.truncated) {
      throw new Error(`Workload mismatch: ${JSON.stringify(result)}`);
    }
    const sample = {
      runtime,
      name: data!.name,
      n,
      run,
      args,
      loadMs,
      wallMs,
      prefillMs: t.prompt_ms,
      prefillTps: t.prompt_per_second,
      decodeMs: t.predicted_ms,
      decodeTps: t.predicted_per_second,
      accountedMs: t.prompt_ms + t.predicted_ms,
      promptTokens: t.prompt_n,
      generatedTokens: t.predicted_n,
      cachedTokens: t.cache_n,
      textSha256: sha256(result.content),
      fixtureSha256: data!.sha256,
      completedAt: new Date().toISOString(),
      result,
    };
    await save(sample);
    console.log(JSON.stringify({ ...sample, args: undefined, result: undefined }));
  } finally {
    server.kill('SIGTERM');
    await Promise.race([exit, sleep(5000)]);
    if (server.exitCode === null) {
      server.kill('SIGKILL');
      await exit;
    }
  }
}

if (runtime === 'mlx') await mlx();
else await llama();
