#!/usr/bin/env oxnode

/// <reference types="node" />

import { fork } from 'node:child_process';
import { randomInt } from 'node:crypto';
import { mkdir, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { cpus, platform, release, totalmem, version as osVersion } from 'node:os';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import type { ChatMessage, SessionCapableModel } from '@mlx-node/lm';
import type { ServerInstance } from '@mlx-node/server';

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const MODEL_NAME = 'benchmark-qwen3';
const DEFAULT_RUNS = 3;
const DEFAULT_COOLDOWN_SECONDS = 180;
const DEFAULT_MAX_OUTPUT_TOKENS = 512;
const DEFAULT_PROMPT_TOKENS = 4_096;
const DEFAULT_MIXED_PREFILL_TOKENS = 16_000;
const CLIENT_TIMEOUT_MS = 180_000;
const MINIMUM_COMPLETION_RATIO = 0.9;
const UNIFORM_CONCURRENCIES = [1, 2, 4, 8] as const;

type Mode = 'serial' | 'batched';

interface CliOptions {
  modelPath: string;
  runs: number;
  cooldownSeconds: number;
  maxOutputTokens: number;
  promptTokens: number;
  mixedPrefillTokens: number;
  outputPath: string | null;
  skipGates: boolean;
  workerMode: Mode | null;
}

interface SchedulerStatsSnapshot {
  maxBatchOccupancy: number;
  decodeBatchOccupancyHist: Array<{ occupancy: number; steps: number }>;
}

interface BenchmarkModel extends SessionCapableModel {
  schedulerStats(): Promise<SchedulerStatsSnapshot>;
  maxConcurrentSequences(): number;
}

interface ClientConfig {
  baseUrl: string;
  index: number;
  input: string;
  maxOutputTokens: number;
  timeoutMs: number;
}

interface ClientSample {
  index: number;
  pid: number;
  status: number;
  startedAtMs: number;
  firstTokenAtMs: number;
  completedAtMs: number;
  clientTtftMs: number;
  wallMs: number;
  serverTtftMs: number;
  serverTotalTtftMs: number;
  serverQueueMs: number;
  serverDecodeTokensPerSecond: number;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  responseStatus: string;
}

interface UniformWaveSample {
  concurrency: number;
  wallMs: number;
  aggregateDecodeTokensPerSecond: number;
  clientTtftP50Ms: number;
  clientTtftP95Ms: number;
  serverTtftP50Ms: number;
  serverTtftP95Ms: number;
  serverQueueP95Ms: number;
  maxBatchOccupancy: number;
  occupancyHistogramDelta: Array<{ occupancy: number; steps: number }>;
  clients: ClientSample[];
}

interface MixedWaveSample {
  wallMs: number;
  longPrefillInputTokens: number;
  chatterClientTtftP50Ms: number;
  chatterClientTtftP95Ms: number;
  chatterServerTtftP50Ms: number;
  chatterServerTtftP95Ms: number;
  maxBatchOccupancy: number;
  occupancyHistogramDelta: Array<{ occupancy: number; steps: number }>;
  longPrefill: ClientSample;
  chatters: ClientSample[];
}

interface WorkerSample {
  type: 'benchmark-concurrent-sample';
  mode: Mode;
  pid: number;
  loadMs: number;
  nativeCapacity: number;
  uniform: UniformWaveSample[];
  mixed: MixedWaveSample;
}

interface ClientResultMessage {
  type: 'benchmark-concurrent-client-result';
  sample: ClientSample;
}

interface ClientRequestMessage {
  type: 'benchmark-concurrent-client-request';
  config: ClientConfig;
}

function usage(): string {
  return `Usage: oxnode scripts/benchmark-concurrent.ts <model-directory> [options]

Runs same-binary serial and continuous-batching waves in randomized A/B order.
Each measured A or B sample uses a fresh child process and one model load.

Options:
  --runs <count>                  A/B samples per mode (default: ${DEFAULT_RUNS})
  --cooldown <seconds>            Cooldown between fresh workers (default: ${DEFAULT_COOLDOWN_SECONDS})
  --max-output-tokens <count>     Output cap per uniform client (default: ${DEFAULT_MAX_OUTPUT_TOKENS})
  --prompt-tokens <count>         Minimum uniform prompt size (default: ${DEFAULT_PROMPT_TOKENS})
  --mixed-prefill-tokens <count>  Minimum long-prefill size (default: ${DEFAULT_MIXED_PREFILL_TOKENS})
  --output <json-path>            Atomically write the JSON report
  --skip-gates                    Report without enforcing N=1/ship gates
  -h, --help                      Show this help

Default gates (median of three): N=1 decode >=95% of forced serial, N=1 TTFT
<=110%, and aggregate decode speedup >=1.7x/3.0x/4.5x at N=2/4/8.
`;
}

function parseInteger(raw: string, name: string, minimum: number): number {
  if (!/^\d+$/u.test(raw)) throw new Error(`${name} must be an integer, got ${JSON.stringify(raw)}`);
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`${name} must be at least ${minimum}, got ${JSON.stringify(raw)}`);
  }
  return value;
}

function parseCooldown(raw: string): number {
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`--cooldown must be a non-negative number, got ${JSON.stringify(raw)}`);
  }
  return value;
}

function parseCli(): CliOptions | null {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    strict: true,
    options: {
      runs: { type: 'string', default: String(DEFAULT_RUNS) },
      cooldown: { type: 'string', default: String(DEFAULT_COOLDOWN_SECONDS) },
      'max-output-tokens': { type: 'string', default: String(DEFAULT_MAX_OUTPUT_TOKENS) },
      'prompt-tokens': { type: 'string', default: String(DEFAULT_PROMPT_TOKENS) },
      'mixed-prefill-tokens': { type: 'string', default: String(DEFAULT_MIXED_PREFILL_TOKENS) },
      output: { type: 'string' },
      'skip-gates': { type: 'boolean', default: false },
      'worker-mode': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (values.help) {
    console.log(usage());
    return null;
  }
  if (positionals.length !== 1) throw new Error(`Expected one model directory.\n\n${usage()}`);
  const workerMode = values['worker-mode'];
  if (workerMode !== undefined && workerMode !== 'serial' && workerMode !== 'batched') {
    throw new Error(`--worker-mode must be serial or batched, got ${JSON.stringify(workerMode)}`);
  }
  const output = values.output?.trim();
  if (output !== undefined && output.length === 0) throw new Error('--output must not be empty');

  return {
    modelPath: resolve(positionals[0]!),
    runs: parseInteger(values.runs, '--runs', 1),
    cooldownSeconds: parseCooldown(values.cooldown),
    maxOutputTokens: parseInteger(values['max-output-tokens'], '--max-output-tokens', 2),
    promptTokens: parseInteger(values['prompt-tokens'], '--prompt-tokens', 32),
    mixedPrefillTokens: parseInteger(values['mixed-prefill-tokens'], '--mixed-prefill-tokens', 128),
    outputPath: output === undefined ? null : resolve(output),
    skipGates: values['skip-gates'],
    workerMode: workerMode ?? null,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function finiteMetric(value: unknown, name: string, allowZero = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || (allowZero ? value < 0 : value <= 0)) {
    throw new Error(`Missing or invalid ${name}: ${String(value)}`);
  }
  return value;
}

function parseTerminalEvent(raw: string): Record<string, unknown> {
  let terminal: Record<string, unknown> | undefined;
  for (const block of raw.split(/\r?\n\r?\n/u)) {
    let event = '';
    const data: string[] = [];
    for (const line of block.split(/\r?\n/u)) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      if (line.startsWith('data:')) data.push(line.slice('data:'.length).trimStart());
    }
    if (event === 'response.failed') {
      throw new Error(`Benchmark request failed: ${data.join('\n')}`);
    }
    if (event === 'response.completed') {
      const parsed: unknown = JSON.parse(data.join('\n'));
      if (!isRecord(parsed) || !isRecord(parsed.response)) {
        throw new Error('response.completed carried no response object');
      }
      terminal = parsed.response;
    }
  }
  if (terminal === undefined) throw new Error('SSE stream ended without response.completed');
  return terminal;
}

async function runHttpClient(config: ClientConfig, outerSignal?: AbortSignal): Promise<ClientSample> {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(new Error(`benchmark client ${config.index} exceeded ${config.timeoutMs}ms`)),
    config.timeoutMs,
  );
  const signal = outerSignal === undefined ? controller.signal : AbortSignal.any([controller.signal, outerSignal]);
  const startedAtMs = performance.timeOrigin + performance.now();
  try {
    const response = await fetch(`${config.baseUrl}/v1/responses`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL_NAME,
        input: config.input,
        stream: true,
        temperature: 0,
        max_output_tokens: config.maxOutputTokens,
      }),
      signal,
    });
    if (response.status !== 200 || response.body === null) {
      throw new Error(`benchmark client ${config.index} got HTTP ${response.status}: ${await response.text()}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let raw = '';
    let firstTokenAtMs = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      raw += decoder.decode(value, { stream: true });
      if (
        firstTokenAtMs === 0 &&
        /event: response\.(?:output_text|reasoning_summary_text)\.delta(?:\r?\n)/u.test(raw)
      ) {
        firstTokenAtMs = performance.timeOrigin + performance.now();
      }
    }
    raw += decoder.decode();
    const completedAtMs = performance.timeOrigin + performance.now();
    if (firstTokenAtMs === 0) firstTokenAtMs = completedAtMs;
    const terminal = parseTerminalEvent(raw);
    const usage = terminal.usage;
    if (!isRecord(usage)) throw new Error('response.completed carried no usage object');
    const inputDetails = isRecord(usage.input_tokens_details) ? usage.input_tokens_details : {};
    const outputTokens = finiteMetric(usage.output_tokens, 'usage.output_tokens');
    const minimumTokens = Math.ceil(config.maxOutputTokens * MINIMUM_COMPLETION_RATIO);
    if (outputTokens < minimumTokens) {
      throw new Error(
        `client ${config.index} ended early: ${outputTokens}/${config.maxOutputTokens} output tokens ` +
          `(status=${String(terminal.status)}, minimum=${minimumTokens})`,
      );
    }
    return {
      index: config.index,
      pid: process.pid,
      status: response.status,
      startedAtMs,
      firstTokenAtMs,
      completedAtMs,
      clientTtftMs: firstTokenAtMs - startedAtMs,
      wallMs: completedAtMs - startedAtMs,
      serverTtftMs: finiteMetric(usage.server_time_to_first_token_ms, 'server_time_to_first_token_ms'),
      serverTotalTtftMs: finiteMetric(usage.server_total_time_to_first_token_ms, 'server_total_time_to_first_token_ms'),
      serverQueueMs: finiteMetric(usage.server_queue_ms, 'server_queue_ms', true),
      serverDecodeTokensPerSecond: finiteMetric(
        usage.server_decode_tokens_per_second,
        'server_decode_tokens_per_second',
      ),
      inputTokens: finiteMetric(usage.input_tokens, 'usage.input_tokens'),
      outputTokens,
      cachedTokens:
        typeof inputDetails.cached_tokens === 'number' && Number.isFinite(inputDetails.cached_tokens)
          ? inputDetails.cached_tokens
          : 0,
      responseStatus: String(terminal.status),
    };
  } finally {
    clearTimeout(timer);
  }
}

async function sendIpc(message: WorkerSample | ClientResultMessage): Promise<void> {
  if (process.send === undefined || process.disconnect === undefined) {
    throw new Error('benchmark child was started without IPC');
  }
  await new Promise<void>((resolvePromise, rejectPromise) => {
    process.send!(message, (error) => (error === null ? resolvePromise() : rejectPromise(error)));
  });
  process.disconnect!();
}

async function runClientChild(): Promise<void> {
  const message = await new Promise<ClientRequestMessage>((resolvePromise, rejectPromise) => {
    const timer = setTimeout(() => rejectPromise(new Error('client child received no configuration')), 10_000);
    process.once('message', (value: unknown) => {
      clearTimeout(timer);
      if (!isRecord(value) || value.type !== 'benchmark-concurrent-client-request' || !isRecord(value.config)) {
        rejectPromise(new Error('client child received an invalid configuration'));
        return;
      }
      resolvePromise(value as unknown as ClientRequestMessage);
    });
  });
  await sendIpc({ type: 'benchmark-concurrent-client-result', sample: await runHttpClient(message.config) });
}

function runClientProcess(config: ClientConfig, signal?: AbortSignal): Promise<ClientSample> {
  return new Promise((resolvePromise, rejectPromise) => {
    if (signal?.aborted === true) {
      rejectPromise(signal.reason);
      return;
    }
    const child = fork(SCRIPT_PATH, ['--client'], {
      execArgv: process.execArgv,
      stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
    });
    let sample: ClientSample | undefined;
    let settled = false;
    const finish = (error?: unknown): void => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener('abort', abort);
      if (error !== undefined) rejectPromise(error);
      else resolvePromise(sample!);
    };
    const abort = (): void => {
      child.kill();
      finish(signal?.reason ?? new Error('benchmark client process aborted'));
    };
    signal?.addEventListener('abort', abort, { once: true });
    child.once('error', finish);
    child.on('message', (message: unknown) => {
      if (isRecord(message) && message.type === 'benchmark-concurrent-client-result' && isRecord(message.sample)) {
        sample = message.sample as unknown as ClientSample;
      }
    });
    child.once('spawn', () => {
      child.send({ type: 'benchmark-concurrent-client-request', config } satisfies ClientRequestMessage);
    });
    child.once('exit', (code, signal) => {
      if (code !== 0) {
        finish(new Error(`client process failed (${signal === null ? `exit ${code}` : `signal ${signal}`})`));
      } else if (sample === undefined) {
        finish(new Error('client process returned no sample'));
      } else {
        finish();
      }
    });
  });
}

function percentile(values: number[], quantile: number): number {
  if (values.length === 0) throw new Error('cannot take percentile of an empty sample');
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(quantile * sorted.length) - 1);
  return sorted[index]!;
}

function histogramMap(stats: SchedulerStatsSnapshot): Map<number, number> {
  return new Map(stats.decodeBatchOccupancyHist.map((bucket) => [bucket.occupancy, bucket.steps]));
}

function histogramDelta(
  before: SchedulerStatsSnapshot,
  after: SchedulerStatsSnapshot,
): Array<{ occupancy: number; steps: number }> {
  const baseline = histogramMap(before);
  return after.decodeBatchOccupancyHist
    .map((bucket) => ({
      occupancy: bucket.occupancy,
      steps: bucket.steps - (baseline.get(bucket.occupancy) ?? 0),
    }))
    .filter((bucket) => bucket.steps > 0)
    .sort((left, right) => left.occupancy - right.occupancy);
}

function summarizeWave(
  concurrency: number,
  startedAt: number,
  completedAt: number,
  clients: ClientSample[],
  occupancyHistogramDelta: Array<{ occupancy: number; steps: number }>,
): UniformWaveSample {
  const decodeStart = Math.min(...clients.map((client) => client.firstTokenAtMs));
  const decodeEnd = Math.max(...clients.map((client) => client.completedAtMs));
  const decodeTokens = clients.reduce((sum, client) => sum + Math.max(0, client.outputTokens - 1), 0);
  return {
    concurrency,
    wallMs: completedAt - startedAt,
    aggregateDecodeTokensPerSecond: decodeTokens / Math.max(0.001, (decodeEnd - decodeStart) / 1_000),
    clientTtftP50Ms: percentile(
      clients.map((client) => client.clientTtftMs),
      0.5,
    ),
    clientTtftP95Ms: percentile(
      clients.map((client) => client.clientTtftMs),
      0.95,
    ),
    serverTtftP50Ms: percentile(
      clients.map((client) => client.serverTtftMs),
      0.5,
    ),
    serverTtftP95Ms: percentile(
      clients.map((client) => client.serverTtftMs),
      0.95,
    ),
    serverQueueP95Ms: percentile(
      clients.map((client) => client.serverQueueMs),
      0.95,
    ),
    maxBatchOccupancy: occupancyHistogramDelta.reduce((maximum, bucket) => Math.max(maximum, bucket.occupancy), 0),
    occupancyHistogramDelta,
    clients,
  };
}

async function buildPromptAtLeast(model: BenchmarkModel, targetTokens: number, label: string): Promise<string> {
  if (model.applyChatTemplate === undefined) throw new Error('benchmark model has no applyChatTemplate()');
  const count = async (repetitions: number): Promise<number> => {
    const content =
      `Benchmark request ${label}. Continue writing detailed implementation notes without concluding. ` +
      `${'scheduler cache tensor kernel '.repeat(repetitions)}`;
    const messages: ChatMessage[] = [{ role: 'user', content }];
    return (await model.applyChatTemplate!(messages, true, null, false)).length;
  };
  let high = Math.max(1, Math.ceil(targetTokens / 4));
  while ((await count(high)) < targetTokens) high *= 2;
  let low = 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if ((await count(middle)) >= targetTokens) high = middle;
    else low = middle + 1;
  }
  return (
    `Benchmark request ${label}. Continue writing detailed implementation notes without concluding. ` +
    `${'scheduler cache tensor kernel '.repeat(low)}`
  );
}

function isBenchmarkModel(model: SessionCapableModel): model is BenchmarkModel {
  return (
    typeof model.applyChatTemplate === 'function' &&
    typeof model.hasBlockPagedCache === 'function' &&
    typeof model.maxConcurrentSequences === 'function' &&
    typeof (model as Partial<BenchmarkModel>).schedulerStats === 'function'
  );
}

async function runUniformWave(
  instance: ServerInstance,
  model: BenchmarkModel,
  baseUrl: string,
  concurrency: number,
  options: CliOptions,
): Promise<UniformWaveSample> {
  await model.resetCaches();
  const before = await model.schedulerStats();
  const prompts = await Promise.all(
    Array.from({ length: concurrency }, (_, index) =>
      buildPromptAtLeast(model, options.promptTokens, `uniform-${concurrency}-${index}`),
    ),
  );
  const configs = prompts.map((input, index) => ({
    baseUrl,
    index,
    input,
    maxOutputTokens: options.maxOutputTokens,
    timeoutMs: CLIENT_TIMEOUT_MS,
  }));
  const startedAt = performance.now();
  const controller = new AbortController();
  let clients: ClientSample[];
  try {
    clients = await Promise.all(
      configs.map((config) =>
        concurrency >= 4 ? runClientProcess(config, controller.signal) : runHttpClient(config, controller.signal),
      ),
    );
  } finally {
    controller.abort(new Error(`uniform N=${concurrency} wave finished`));
  }
  const completedAt = performance.now();
  const after = await model.schedulerStats();
  const delta = histogramDelta(before, after);
  const summary = summarizeWave(concurrency, startedAt, completedAt, clients, delta);
  const registry = instance.registry.getSessionRegistry(MODEL_NAME);
  if (registry?.queueDepth !== 0 || registry.preDispatchAdmitCount !== 0) {
    throw new Error(
      `uniform N=${concurrency} leaked admission state: queue=${registry?.queueDepth} ` +
        `preDispatch=${registry?.preDispatchAdmitCount}`,
    );
  }
  return summary;
}

async function runMixedWave(
  instance: ServerInstance,
  model: BenchmarkModel,
  baseUrl: string,
  options: CliOptions,
): Promise<MixedWaveSample> {
  await model.resetCaches();
  const before = await model.schedulerStats();
  const longPrompt = await buildPromptAtLeast(model, options.mixedPrefillTokens, 'mixed-long-prefill');
  const chatterPrompts = await Promise.all(
    Array.from({ length: 4 }, (_, index) =>
      buildPromptAtLeast(model, Math.min(512, options.promptTokens), `mixed-chatter-${index}`),
    ),
  );
  const startedAt = performance.now();
  const controller = new AbortController();
  try {
    const longPrefillPromise = runHttpClient(
      {
        baseUrl,
        index: 0,
        input: longPrompt,
        maxOutputTokens: Math.min(64, options.maxOutputTokens),
        timeoutMs: CLIENT_TIMEOUT_MS,
      },
      controller.signal,
    );
    let longPrefillSettled = false;
    let longPrefillError: unknown;
    void longPrefillPromise.then(
      () => {
        longPrefillSettled = true;
      },
      (error: unknown) => {
        longPrefillSettled = true;
        longPrefillError = error;
      },
    );
    const admissionDeadline = performance.now() + 5_000;
    while (instance.health().work.inFlight < 1) {
      if (longPrefillSettled) {
        throw longPrefillError ?? new Error('mixed long-prefill request completed before admission was observed');
      }
      if (performance.now() >= admissionDeadline) {
        throw new Error('mixed long-prefill request did not enter server in-flight accounting');
      }
      await new Promise((resolvePromise) => setTimeout(resolvePromise, 10));
    }
    const chatterPromises = chatterPrompts.map((input, index) =>
      runClientProcess(
        {
          baseUrl,
          index: index + 1,
          input,
          maxOutputTokens: options.maxOutputTokens,
          timeoutMs: CLIENT_TIMEOUT_MS,
        },
        controller.signal,
      ),
    );
    const [longPrefill, ...chatters] = await Promise.all([longPrefillPromise, ...chatterPromises]);
    const completedAt = performance.now();
    const delta = histogramDelta(before, await model.schedulerStats());
    const registry = instance.registry.getSessionRegistry(MODEL_NAME);
    if (registry?.queueDepth !== 0 || registry.preDispatchAdmitCount !== 0) {
      throw new Error(
        `mixed wave leaked admission state: queue=${registry?.queueDepth} preDispatch=${registry?.preDispatchAdmitCount}`,
      );
    }
    return {
      wallMs: completedAt - startedAt,
      longPrefillInputTokens: longPrefill.inputTokens,
      chatterClientTtftP50Ms: percentile(
        chatters.map((client) => client.clientTtftMs),
        0.5,
      ),
      chatterClientTtftP95Ms: percentile(
        chatters.map((client) => client.clientTtftMs),
        0.95,
      ),
      chatterServerTtftP50Ms: percentile(
        chatters.map((client) => client.serverTtftMs),
        0.5,
      ),
      chatterServerTtftP95Ms: percentile(
        chatters.map((client) => client.serverTtftMs),
        0.95,
      ),
      maxBatchOccupancy: delta.reduce((maximum, bucket) => Math.max(maximum, bucket.occupancy), 0),
      occupancyHistogramDelta: delta,
      longPrefill,
      chatters,
    };
  } finally {
    controller.abort(new Error('mixed benchmark wave finished'));
  }
}

async function runWorker(options: CliOptions, mode: Mode): Promise<WorkerSample> {
  if (mode === 'serial') process.env.MLX_SERVE_FORCE_SERIAL = '1';
  else delete process.env.MLX_SERVE_FORCE_SERIAL;

  const [{ loadModel }, { createServer }] = await Promise.all([import('@mlx-node/lm'), import('@mlx-node/server')]);
  const loadStartedAt = performance.now();
  let model!: BenchmarkModel;
  const instance = await createServer({
    port: 0,
    host: '127.0.0.1',
    disableStore: true,
    idleClearCacheMs: 3_600_000,
    maxQueueDepthPerModel: 16,
  });
  try {
    await instance.loadModel({
      name: MODEL_NAME,
      samplingDefaults: {
        temperature: 0,
        repetitionPenalty: 1,
        presencePenalty: 0,
        frequencyPenalty: 0,
        maxConsecutiveTokens: 0,
        maxNgramRepeats: 0,
        ngramSize: 0,
        thinkingTokenBudget: 0,
        includeReasoning: false,
      },
      maxOutputTokens: options.maxOutputTokens,
      load: async () => {
        const loaded = await loadModel(options.modelPath);
        if (!isBenchmarkModel(loaded as unknown as SessionCapableModel)) {
          throw new Error('benchmark requires a scheduler-capable dense Qwen3 model');
        }
        model = loaded as unknown as BenchmarkModel;
        return model;
      },
    });
    const address = instance.server.address();
    if (address === null || typeof address === 'string') throw new Error('benchmark server has no TCP address');
    const baseUrl = `http://127.0.0.1:${address.port}`;
    const loadMs = performance.now() - loadStartedAt;
    const nativeCapacity = model.maxConcurrentSequences();
    if (mode === 'serial' && nativeCapacity !== 1) {
      throw new Error(`forced-serial worker reported native capacity ${nativeCapacity}, expected 1`);
    }
    if (mode === 'batched' && nativeCapacity < 2) {
      throw new Error(`batched worker reported native capacity ${nativeCapacity}, expected >=2`);
    }

    const uniform: UniformWaveSample[] = [];
    for (const concurrency of UNIFORM_CONCURRENCIES) {
      uniform.push(await runUniformWave(instance, model, baseUrl, concurrency, options));
    }
    const mixed = await runMixedWave(instance, model, baseUrl, options);
    return {
      type: 'benchmark-concurrent-sample',
      mode,
      pid: process.pid,
      loadMs,
      nativeCapacity,
      uniform,
      mixed,
    };
  } finally {
    await instance.close({ timeoutMs: 10_000 });
  }
}

function workerArgs(options: CliOptions, mode: Mode): string[] {
  return [
    options.modelPath,
    '--worker-mode',
    mode,
    '--runs',
    String(options.runs),
    '--cooldown',
    String(options.cooldownSeconds),
    '--max-output-tokens',
    String(options.maxOutputTokens),
    '--prompt-tokens',
    String(options.promptTokens),
    '--mixed-prefill-tokens',
    String(options.mixedPrefillTokens),
    ...(options.skipGates ? ['--skip-gates'] : []),
  ];
}

function runWorkerProcess(options: CliOptions, mode: Mode): Promise<WorkerSample> {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = fork(SCRIPT_PATH, workerArgs(options, mode), {
      execArgv: process.execArgv,
      stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
    });
    let sample: WorkerSample | undefined;
    child.once('error', rejectPromise);
    child.on('message', (message: unknown) => {
      if (isRecord(message) && message.type === 'benchmark-concurrent-sample') {
        sample = message as unknown as WorkerSample;
      }
    });
    child.once('exit', (code, signal) => {
      if (code !== 0) {
        rejectPromise(new Error(`benchmark worker failed (${signal === null ? `exit ${code}` : `signal ${signal}`})`));
      } else if (sample === undefined) {
        rejectPromise(new Error('benchmark worker returned no sample'));
      } else {
        resolvePromise(sample);
      }
    });
  });
}

function median(values: number[]): number {
  if (values.length === 0) throw new Error('cannot take median of an empty sample');
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function samplesForMode(samples: WorkerSample[], mode: Mode): WorkerSample[] {
  return samples.filter((sample) => sample.mode === mode);
}

function uniformMedian(samples: WorkerSample[], mode: Mode, concurrency: number) {
  const waves = samplesForMode(samples, mode).map((sample) => {
    const wave = sample.uniform.find((candidate) => candidate.concurrency === concurrency);
    if (wave === undefined) throw new Error(`missing ${mode} N=${concurrency} wave`);
    return wave;
  });
  return {
    aggregateDecodeTokensPerSecond: median(waves.map((wave) => wave.aggregateDecodeTokensPerSecond)),
    clientTtftP50Ms: median(waves.map((wave) => wave.clientTtftP50Ms)),
    clientTtftP95Ms: median(waves.map((wave) => wave.clientTtftP95Ms)),
    serverTtftP50Ms: median(waves.map((wave) => wave.serverTtftP50Ms)),
    serverTtftP95Ms: median(waves.map((wave) => wave.serverTtftP95Ms)),
    maxBatchOccupancy: Math.max(...waves.map((wave) => wave.maxBatchOccupancy)),
  };
}

function buildSummary(samples: WorkerSample[]) {
  const uniform = UNIFORM_CONCURRENCIES.map((concurrency) => {
    const serial = uniformMedian(samples, 'serial', concurrency);
    const batched = uniformMedian(samples, 'batched', concurrency);
    return {
      concurrency,
      serial,
      batched,
      aggregateDecodeSpeedup: batched.aggregateDecodeTokensPerSecond / serial.aggregateDecodeTokensPerSecond,
    };
  });
  const mixed = (['serial', 'batched'] as const).map((mode) => {
    const waves = samplesForMode(samples, mode).map((sample) => sample.mixed);
    return {
      mode,
      chatterClientTtftP50Ms: median(waves.map((wave) => wave.chatterClientTtftP50Ms)),
      chatterClientTtftP95Ms: median(waves.map((wave) => wave.chatterClientTtftP95Ms)),
      chatterServerTtftP50Ms: median(waves.map((wave) => wave.chatterServerTtftP50Ms)),
      chatterServerTtftP95Ms: median(waves.map((wave) => wave.chatterServerTtftP95Ms)),
      maxBatchOccupancy: Math.max(...waves.map((wave) => wave.maxBatchOccupancy)),
    };
  });
  const n1 = uniform[0]!;
  const gates = {
    n1DecodeRatio: n1.batched.aggregateDecodeTokensPerSecond / n1.serial.aggregateDecodeTokensPerSecond,
    n1ClientTtftRatio: n1.batched.clientTtftP50Ms / n1.serial.clientTtftP50Ms,
    n1ServerTtftRatio: n1.batched.serverTtftP50Ms / n1.serial.serverTtftP50Ms,
    n2Speedup: uniform[1]!.aggregateDecodeSpeedup,
    n4Speedup: uniform[2]!.aggregateDecodeSpeedup,
    n8Speedup: uniform[3]!.aggregateDecodeSpeedup,
  };
  return { uniform, mixed, gates };
}

function enforceGates(summary: ReturnType<typeof buildSummary>): void {
  const failures: string[] = [];
  if (summary.gates.n1DecodeRatio < 0.95) {
    failures.push(`N=1 decode ratio ${summary.gates.n1DecodeRatio.toFixed(3)} < 0.95`);
  }
  if (summary.gates.n1ClientTtftRatio > 1.1) {
    failures.push(`N=1 client TTFT ratio ${summary.gates.n1ClientTtftRatio.toFixed(3)} > 1.10`);
  }
  if (summary.gates.n1ServerTtftRatio > 1.1) {
    failures.push(`N=1 server TTFT ratio ${summary.gates.n1ServerTtftRatio.toFixed(3)} > 1.10`);
  }
  for (const [name, actual, minimum] of [
    ['N=2', summary.gates.n2Speedup, 1.7],
    ['N=4', summary.gates.n4Speedup, 3.0],
    ['N=8', summary.gates.n8Speedup, 4.5],
  ] as const) {
    if (actual < minimum) failures.push(`${name} aggregate speedup ${actual.toFixed(3)}x < ${minimum.toFixed(1)}x`);
  }
  const batchedOccupancies = summary.uniform
    .filter((wave) => wave.concurrency > 1)
    .map((wave) => wave.batched.maxBatchOccupancy);
  if (batchedOccupancies.some((occupancy, index) => occupancy < UNIFORM_CONCURRENCIES[index + 1]!)) {
    failures.push(`batched occupancy gate failed: ${JSON.stringify(batchedOccupancies)}`);
  }
  if (failures.length > 0) throw new Error(`Concurrent benchmark gates failed:\n- ${failures.join('\n- ')}`);
}

function randomModeOrder(): Mode[] {
  return randomInt(2) === 0 ? ['serial', 'batched'] : ['batched', 'serial'];
}

async function sleep(milliseconds: number): Promise<void> {
  await new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

async function writeJsonAtomically(outputPath: string, contents: string): Promise<void> {
  await mkdir(dirname(outputPath), { recursive: true });
  const temporaryPath = `${outputPath}.${process.pid}.${Date.now()}.tmp`;
  try {
    await writeFile(temporaryPath, `${contents}\n`, { encoding: 'utf8', flag: 'wx' });
    await rename(temporaryPath, outputPath);
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined);
    throw new Error(`failed to write ${outputPath}`, { cause: error });
  }
}

async function gitSha(): Promise<string> {
  const { execFile } = await import('node:child_process');
  return await new Promise<string>((resolvePromise, rejectPromise) => {
    execFile('git', ['rev-parse', 'HEAD'], { cwd: dirname(SCRIPT_PATH) }, (error, stdout) => {
      if (error !== null) rejectPromise(error);
      else resolvePromise(stdout.trim());
    });
  });
}

async function assertModelDirectory(modelPath: string): Promise<void> {
  const modelStat = await stat(modelPath);
  if (!modelStat.isDirectory()) throw new Error(`model path is not a directory: ${modelPath}`);
}

async function runParent(options: CliOptions): Promise<void> {
  await assertModelDirectory(options.modelPath);
  const startedAt = new Date();
  const order: Array<{ run: number; mode: Mode }> = [];
  for (let run = 1; run <= options.runs; run += 1) {
    for (const mode of randomModeOrder()) order.push({ run, mode });
  }
  const samples: WorkerSample[] = [];
  console.log(`Benchmarking ${options.modelPath}`);
  console.log(
    `Randomized worker order: ${order.map((entry) => `${entry.mode[0]!.toUpperCase()}${entry.run}`).join(' ')}`,
  );
  for (let index = 0; index < order.length; index += 1) {
    const entry = order[index]!;
    console.log(`\n${index + 1}/${order.length}: ${entry.mode} run ${entry.run}`);
    const sample = await runWorkerProcess(options, entry.mode);
    samples.push(sample);
    for (const wave of sample.uniform) {
      console.log(
        `  N=${wave.concurrency}: ${wave.aggregateDecodeTokensPerSecond.toFixed(1)} aggregate decode tok/s, ` +
          `client TTFT p50/p95 ${wave.clientTtftP50Ms.toFixed(0)}/${wave.clientTtftP95Ms.toFixed(0)} ms, ` +
          `occupancy ${wave.maxBatchOccupancy}`,
      );
    }
    console.log(
      `  mixed chatter TTFT p50/p95 ${sample.mixed.chatterClientTtftP50Ms.toFixed(0)}/` +
        `${sample.mixed.chatterClientTtftP95Ms.toFixed(0)} ms`,
    );
    if (index + 1 < order.length && options.cooldownSeconds > 0) {
      console.log(`Cooling down for ${options.cooldownSeconds}s...`);
      await sleep(options.cooldownSeconds * 1_000);
    }
  }

  const summary = buildSummary(samples);
  const report = {
    schemaVersion: 1,
    gitSha: await gitSha(),
    modelPath: options.modelPath,
    startedAt: startedAt.toISOString(),
    completedAt: new Date().toISOString(),
    environment: {
      platform: platform(),
      arch: process.arch,
      osRelease: release(),
      osVersion: osVersion(),
      cpu: cpus()[0]?.model ?? null,
      logicalCpus: cpus().length,
      totalMemoryBytes: totalmem(),
      nodeVersion: process.version,
    },
    configuration: {
      runs: options.runs,
      cooldownSeconds: options.cooldownSeconds,
      maxOutputTokens: options.maxOutputTokens,
      promptTokens: options.promptTokens,
      mixedPrefillTokens: options.mixedPrefillTokens,
      temperature: 0,
      minimumCompletionRatio: MINIMUM_COMPLETION_RATIO,
      uniformConcurrencies: UNIFORM_CONCURRENCIES,
      clientIsolation: 'child processes for N>=4 and the four mixed-wave chatter clients',
      workerIsolation: 'fresh process and model load per A/B sample',
      order,
    },
    samples,
    medians: summary,
  };
  const json = JSON.stringify(report, null, 2);
  if (options.outputPath !== null) await writeJsonAtomically(options.outputPath, json);
  console.log('\nJSON summary:');
  console.log(json);
  if (!options.skipGates) enforceGates(summary);
}

async function main(): Promise<void> {
  if (process.argv.includes('--client')) {
    await runClientChild();
    return;
  }
  const options = parseCli();
  if (options === null) return;
  if (options.workerMode !== null) {
    await sendIpc(await runWorker(options, options.workerMode));
    return;
  }
  await runParent(options);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
});
