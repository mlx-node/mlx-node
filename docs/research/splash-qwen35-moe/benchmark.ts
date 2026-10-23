#!/usr/bin/env oxnode

/// <reference types="node" />

// Frozen-addon, matched-token comparison. Run --help without loading Metal.
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { cpus, release, totalmem } from 'node:os';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import type { ChatConfig, ChatMessage } from '../../../packages/core/index.cjs';

type Core = typeof import('../../../packages/core/index.cjs');
interface Fixture {
  name: string;
  messages: ChatMessage[];
  tokenIds: number[];
  promptTokens: number;
  tokenIdsSha256: string;
}
interface Fixtures {
  version: 1;
  tokenizer: string;
  tokenizerSha256: string;
  reasoningEffort: 'high';
  cases: Fixture[];
}
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    binding: { type: 'string' },
    tokenizer: { type: 'string' },
    target: { type: 'string' },
    draft: { type: 'string' },
    fixtures: { type: 'string' },
    output: { type: 'string' },
    case: { type: 'string', default: 'short' },
    mode: { type: 'string', default: 'ar' },
    'kv-format': { type: 'string', default: 'bf16' },
    'verify-rows': { type: 'string', default: 'all' },
    tokens: { type: 'string', default: '128' },
    runs: { type: 'string', default: '1' },
    warmup: { type: 'string', default: '16' },
    'warmup-case': { type: 'string', default: 'short' },
    'max-peak-gib': { type: 'string', default: '0' },
    replay: { type: 'boolean', default: false },
    endpoint: { type: 'string', default: 'http://127.0.0.1:18282' },
    help: { type: 'boolean', default: false },
  },
});
if (values.help) {
  console.log(`Usage: oxnode docs/research/splash-qwen35-moe/benchmark.ts <fixture|node|splash> [options]
fixture --binding addon.node --tokenizer tokenizer.json --output fixtures.json
node --binding addon.node --target model.gguf --fixtures fixtures.json --output result.json
     [--mode ar|mtp|dflash] [--draft directory] [--kv-format bf16|int8]
     [--max-peak-gib 0] (optional live allocator limit; 0 disables)
splash --endpoint http://127.0.0.1:18282 --fixtures fixtures.json --output result.json
Both runners: [--case short|2k|32k] [--tokens 128] [--runs 1] [--warmup 16] [--replay]
             [--verify-rows all|auto]
             [--warmup-case short|selected]
Run one engine process at a time. Cold means no prefix reuse; warmup is recorded separately.
Use --warmup 0 in a fresh process to measure first-request compilation separately.
Splash must already serve the exact target/draft with the selected --verify-rows and matched KV/state formats.
Splash receives frozen token IDs; node verifies its current tokenizer renders those exact IDs.
Early stops, prefix-cache contamination and missing speculation fail the run.`);
  process.exit(0);
}
const mode = positionals[0];
if (positionals.length !== 1 || !['fixture', 'node', 'splash'].includes(mode)) throw new Error('Use --help');
const required = (name: keyof typeof values) => {
  const value = values[name];
  if (typeof value !== 'string' || !value) throw new Error(`Missing --${name}`);
  return value;
};
const positive = (value: string, name: string, minimum = 1) => {
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < minimum)
    throw new Error(`Invalid ${name}`);
  return Number(value);
};
const hash = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex');
const tokenHash = (ids: number[]) => hash(JSON.stringify(ids));
const output = resolve(required('output'));
await mkdir(dirname(output), { recursive: true });
const save = (value: unknown) => writeFile(output, JSON.stringify(value, null, 2) + '\n');
const loadCore = (): Core => createRequire(import.meta.url)(resolve(required('binding')));
const system = { cpu: cpus()[0]?.model, osRelease: release(), totalMemoryBytes: totalmem() };

if (mode === 'fixture') {
  const tokenizerPath = resolve(required('tokenizer'));
  const tokenizer = await loadCore().Qwen3Tokenizer.fromPretrained(tokenizerPath);
  const render = async (messages: ChatMessage[]) =>
    Array.from(await tokenizer.applyChatTemplate(messages, true, undefined, true, undefined, undefined, 'high'));
  const task =
    'Implement a TypeScript LRU cache with get, set, delete, fixed capacity, and eviction tests. Explain the implementation and complexity in detail.';
  const filler = 'export function cacheKey(namespace: string, key: string): string { return namespace + ":" + key; }\n';
  const make = async (name: string, content: string): Promise<Fixture> => {
    const messages: ChatMessage[] = [{ role: 'user', content }];
    const tokenIds = await render(messages);
    return { name, messages, tokenIds, promptTokens: tokenIds.length, tokenIdsSha256: tokenHash(tokenIds) };
  };
  const cases: Fixture[] = [await make('short', task)];
  for (const [name, budget] of [
    ['2k', 2048],
    ['32k', 32768],
  ] as const) {
    const full = filler.repeat(Math.ceil(budget / 10));
    let lo = 0,
      hi = full.length;
    let fixture = await make(name, task);
    while (lo <= hi) {
      const chars = Math.floor((lo + hi) / 2);
      const candidate = await make(
        name,
        `Review this existing code before implementing the cache.\n\n${full.slice(0, chars)}\n\n${task}`,
      );
      if (candidate.promptTokens <= budget) {
        fixture = candidate;
        lo = chars + 1;
      } else hi = chars - 1;
    }
    cases.push(fixture);
  }
  const fixtures: Fixtures = {
    version: 1,
    tokenizer: tokenizerPath,
    tokenizerSha256: hash(await readFile(tokenizerPath)),
    reasoningEffort: 'high',
    cases,
  };
  await save(fixtures);
  for (const item of cases) {
    const bytes = Buffer.alloc(item.tokenIds.length * 4);
    item.tokenIds.forEach((id, i) => bytes.writeUInt32LE(id, i * 4));
    await writeFile(`${output}.${item.name}.u32`, bytes);
  }
  console.log(
    JSON.stringify(cases.map(({ name, promptTokens, tokenIdsSha256 }) => ({ name, promptTokens, tokenIdsSha256 }))),
  );
  process.exit(0);
}

const fixtures: Fixtures = JSON.parse(await readFile(required('fixtures'), 'utf8'));
const item = fixtures.cases.find((entry) => entry.name === values.case);
if (!item || item.promptTokens !== item.tokenIds.length || tokenHash(item.tokenIds) !== item.tokenIdsSha256)
  throw new Error('Missing or invalid fixture');
const tokens = positive(values.tokens, 'tokens');
const runs = positive(values.runs, 'runs');
const warmup = positive(values.warmup, 'warmup', 0);
const maxPeakBytes = positive(values['max-peak-gib'], 'max-peak-gib', 0) * 1024 ** 3;
if (!['all', 'auto'].includes(values['verify-rows'])) throw new Error('Unknown verify rows policy');
if (!['short', 'selected'].includes(values['warmup-case'])) throw new Error('Unknown warmup case');
const warmupItem = values['warmup-case'] === 'selected' ? item : fixtures.cases[0];
const records: unknown[] = [];
let checkPeakMemory = () => {};
const report: Record<string, unknown> = {
  engine: mode,
  decodeMode: mode === 'splash' ? 'dflash' : values.mode,
  system,
  fixture: item,
  config: values,
  records,
};
const persist = async (row: unknown) => {
  checkPeakMemory();
  records.push(row);
  await save(report);
  console.log(JSON.stringify(row));
};

if (mode === 'node') {
  if (!['ar', 'mtp', 'dflash'].includes(values.mode)) throw new Error('Unknown decode mode');
  if (values.mode === 'dflash' && !values.draft) throw new Error('DFlash requires --draft');
  const core = loadCore();
  if (maxPeakBytes > 0) {
    checkPeakMemory = () => {
      const memory = core.memoryStats();
      if (memory.peak > maxPeakBytes) {
        console.error(JSON.stringify({ event: 'allocator_peak_limit', maxPeakBytes, memory }));
        process.exit(125);
      }
    };
    setInterval(checkPeakMemory, 100).unref();
  }
  const loadedAt = performance.now();
  const load = core.Qwen35MoeModel.load.bind(core.Qwen35MoeModel) as unknown as (
    path: string,
    options?: { draftModelPath?: string; kvFormat?: string },
  ) => ReturnType<Core['Qwen35MoeModel']['load']>;
  const model = await load(resolve(required('target')), {
    draftModelPath: values.draft,
    kvFormat: values['kv-format'],
  });
  report.loadMs = performance.now() - loadedAt;
  report.bindingSha256 = hash(await readFile(required('binding')));
  report.environment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) =>
      /^MLX_(QWEN|DFLASH|MTP|KQUANT|PORTABLE|PAGED|SPLASH|MOE|MIXED|GDN|ATTENTION|SDPA|SEGMENTED|PROFILE|METAL|MAX_OPS|MAX_MB|CACHE)/.test(
        key,
      ),
    ),
  );
  report.hasBlockPagedCache = model.hasBlockPagedCache();
  report.hasMtpWeights = model.hasMtpWeights();
  report.modelAssetsPath = model.modelAssetsPath();
  report.loadedMemory = { mlx: core.memoryStats(), hostRssBytes: process.memoryUsage().rss };
  checkPeakMemory();
  console.log(JSON.stringify({ event: 'loaded', loadMs: report.loadMs, memory: report.loadedMemory }));
  const tokenizer = await core.Qwen3Tokenizer.fromPretrained(`${model.modelAssetsPath()}/tokenizer.json`);
  const actual = Array.from(
    await tokenizer.applyChatTemplate(
      item.messages,
      true,
      undefined,
      true,
      undefined,
      undefined,
      fixtures.reasoningEffort,
    ),
  );
  if (tokenHash(actual) !== item.tokenIdsSha256) throw new Error('Loaded model prompt token IDs differ from fixture');
  const config: ChatConfig = {
    maxNewTokens: tokens,
    temperature: 0,
    topK: 0,
    topP: 1,
    minP: 0,
    repetitionPenalty: 1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    maxConsecutiveTokens: 0,
    maxNgramRepeats: 0,
    reasoningEffort: fixtures.reasoningEffort,
    includeReasoning: true,
    enableMtp: values.mode !== 'ar',
    mtpAdaptiveDepth: values['verify-rows'] === 'auto',
    reuseCache: true,
    reportPerformance: true,
  };
  if (warmup) {
    await model.resetCaches();
    const startedAtUtc = new Date().toISOString();
    const started = performance.now();
    const result = await model.chatSessionStart(warmupItem.messages, {
      ...config,
      maxNewTokens: warmup,
      cacheSalt: randomUUID(),
    });
    report.warmup = {
      startedAtUtc,
      finishedAtUtc: new Date().toISOString(),
      wallMs: performance.now() - started,
      result,
    };
    await save(report);
    if (result.numTokens !== warmup || result.promptTokens !== warmupItem.promptTokens || result.cachedTokens !== 0)
      throw new Error('Incomplete warmup or warmup prefix-cache contamination');
    console.log(JSON.stringify({ event: 'warmup_complete', outputTokens: warmup }));
  }
  for (let run = 1; run <= runs; run++) {
    await model.resetCaches();
    const salt = randomUUID();
    for (const phase of values.replay ? ['cold', 'replay'] : ['cold']) {
      const startedAtUtc = new Date().toISOString();
      const started = performance.now();
      const result = await model.chatSessionStart(item.messages, { ...config, cacheSalt: salt });
      const row = {
        phase,
        run,
        startedAtUtc,
        finishedAtUtc: new Date().toISOString(),
        wallMs: performance.now() - started,
        outputSha256: hash(result.rawText),
        memory: core.memoryStats(),
        hostRssBytes: process.memoryUsage().rss,
        hostPeakRssKiB: process.resourceUsage().maxRSS,
        result,
      };
      await persist(row);
      if (result.numTokens !== tokens || result.promptTokens !== item.promptTokens || !result.performance)
        throw new Error('Incomplete sample or prompt token count drift');
      if (phase === 'cold' && result.cachedTokens !== 0) throw new Error('Cold sample reused cache');
      if (phase === 'replay' && result.cachedTokens === 0) throw new Error('Replay did not reuse cache');
      if (values.mode !== 'ar' && !(result.performance.mtpCycles && result.performance.mtpCycles > 0))
        throw new Error('Requested speculation did not execute');
    }
  }
  await model.resetCaches();
} else {
  const endpoint = values.endpoint.replace(/\/$/, '');
  const request = async (path: string, body?: unknown) => {
    const response = await fetch(endpoint + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? undefined : { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(600_000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(`Splash HTTP ${response.status}: ${JSON.stringify(result)}`);
    return result as any;
  };
  report.initialStatus = await request('/status');
  if ((report.initialStatus as any).verify_rows !== values['verify-rows'])
    throw new Error('Splash server verify rows policy does not match the requested benchmark');
  const body = (input: Fixture, cap: number, salt: string) => ({
    prompt: input.tokenIds,
    max_tokens: cap,
    temperature: 0,
    top_k: 0,
    top_p: 1,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    ignore_eos: false,
    cache_salt: salt,
  });
  if (warmup) {
    const startedAtUtc = new Date().toISOString();
    const result = await request('/v1/completions', body(warmupItem, warmup, randomUUID()));
    report.warmup = result;
    report.warmupRange = { startedAtUtc, finishedAtUtc: new Date().toISOString() };
    await save(report);
    if (
      result.usage.completion_tokens !== warmup ||
      result.usage.prompt_tokens !== warmupItem.promptTokens ||
      result.timings.cache_n !== 0
    )
      throw new Error('Incomplete Splash warmup or warmup prefix-cache contamination');
    console.log(JSON.stringify({ event: 'warmup_complete', outputTokens: warmup }));
  }
  for (let run = 1; run <= runs; run++) {
    const salt = randomUUID();
    for (const phase of values.replay ? ['cold', 'replay'] : ['cold']) {
      const before = await request('/status');
      const startedAtUtc = new Date().toISOString();
      const started = performance.now();
      const result = await request('/v1/completions', body(item, tokens, salt));
      const wallMs = performance.now() - started;
      const finishedAtUtc = new Date().toISOString();
      const after = await request('/status');
      const keys = [
        'prefill_wall_ms',
        'decode_wall_ms',
        'decode_cycle_ms',
        'prefill_input_tokens',
        'decode_output_tokens',
        'drafted_tokens',
        'accepted_draft_tokens',
      ];
      const delta = Object.fromEntries(keys.map((key) => [key, after.metrics[key] - before.metrics[key]]));
      await persist({
        phase,
        run,
        startedAtUtc,
        finishedAtUtc,
        wallMs,
        outputSha256: hash(result.choices[0].text),
        nativeDelta: delta,
        result,
      });
      if (result.usage.prompt_tokens !== item.promptTokens || result.usage.completion_tokens !== tokens)
        throw new Error('Incomplete Splash sample or prompt token count drift');
      const cached = result.timings.cache_n;
      if (phase === 'cold' && cached !== 0) throw new Error('Cold Splash sample reused cache');
      if (phase === 'replay' && !(cached > 0)) throw new Error('Splash replay did not reuse cache');
    }
  }
}
await save(report);
process.exit(0);
