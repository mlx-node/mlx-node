/**
 * A seeded, disposable `~/.mlx-node`-shaped home for the dev harness and its screenshots: model folders, 30 days of
 * v2 traces, runtime series, a live heartbeat, a cold cache, agent sessions and coding-agent files. Deterministic for
 * a seed and a `now`. Never point it at the real home: the ingest prunes traces older than 30 days where it reads.
 */

import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  truncateSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

import type { SupervisorStateFile } from '../../desktop/src/main/supervisor-state-file.js';
import type { MetricsTraceRecord } from '../../server/src/metrics-trace.js';
import type { RuntimeLiveFile, RuntimeTraceRecord } from '../../server/src/runtime-trace.js';
import { managedBlock } from '../src/coding-agents.js';

export const DEFAULT_SEED = 0x4d4c58;
const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;
const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

/** The app command the harness hands the Coding Agents page; `.claude/CLAUDE.md` is installed for it. */
export const FIXTURE_COMMAND = '/Applications/mlx-node.app/Contents/Resources/bin/mlx';
const OLD_COMMAND = '/usr/local/bin/mlx-old';

/** The file that marks a directory as written by this generator; only such a directory is ever deleted. */
export const FIXTURE_MARKER = '.mlx-node-dev-fixture';

const GROK_FIXTURE = join(import.meta.dirname, '../__test__/fixtures/grok-sessions/home');
const RICH_SESSION = '01a20000-0000-7000-8000-00000000000a';
const RICH_SESSION_CWD = '/Users/dev/proj';

export interface FixtureOptions {
  seed?: number;
  now?: number;
  /** Load the extra Engine models of {@link LiveProcess.engineStates}. */
  engineStates?: boolean;
}

/** The home layout the runtime options point at. */
export function fixturePaths(home: string): {
  dbPath: string;
  modelsDir: string;
  tracesDir: string;
  runtimeDir: string;
  supervisorPath: string;
  cacheRoot: string;
  agentHome: string;
} {
  return {
    dbPath: join(home, 'dashboard.db'),
    modelsDir: join(home, 'models'),
    tracesDir: join(home, 'metrics', 'traces'),
    runtimeDir: join(home, 'metrics', 'runtime'),
    supervisorPath: join(home, 'desktop', 'supervisor.json'),
    cacheRoot: join(home, 'cold'),
    agentHome: join(home, '.mlx-agent'),
  };
}

type Random = () => number;

function mulberry32(seed: number): Random {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function between(random: Random, low: number, high: number): number {
  return low + random() * (high - low);
}

function int(random: Random, low: number, high: number): number {
  return Math.floor(between(random, low, high + 1));
}

/** Log-normal around `median`; `spread` is the sigma of the underlying normal. */
function logNormal(random: Random, median: number, spread: number): number {
  const u = Math.max(random(), 1e-9);
  const v = random();
  const normal = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return median * Math.exp(spread * normal);
}

function pick<T>(random: Random, items: readonly T[]): T {
  return items[Math.min(items.length - 1, Math.floor(random() * items.length))]!;
}

function round(value: number, digits = 1): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function hex(random: Random, length: number): string {
  let out = '';
  for (let i = 0; i < length; i += 1) out += '0123456789abcdef'[Math.floor(random() * 16)];
  return out;
}

function utcDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

function writeLines(path: string, records: readonly unknown[]): void {
  writeFileSync(path, records.map((record) => `${JSON.stringify(record)}\n`).join(''));
}

interface FixtureModel {
  name: string;
  modelType: string;
  installed: boolean;
  ctxWindow: number;
  decodeTps: number;
  prefillTps: number;
  mtp: boolean;
  weight: number;
  sizeGb: number;
  bits: number | null;
}

/** 6 installed and 4 gone, so the 9th and 10th model exercise the second color encoding. */
export const FIXTURE_MODELS: readonly FixtureModel[] = [
  {
    name: 'Qwen3.5-27B-4bit',
    modelType: 'qwen3_5',
    installed: true,
    ctxWindow: 131072,
    decodeTps: 38,
    prefillTps: 1350,
    mtp: true,
    weight: 6,
    sizeGb: 15,
    bits: 4,
  },
  {
    name: 'Qwen3-8B-4bit',
    modelType: 'qwen3',
    installed: true,
    ctxWindow: 40960,
    decodeTps: 92,
    prefillTps: 3900,
    mtp: false,
    weight: 4,
    sizeGb: 4.6,
    bits: 4,
  },
  {
    name: 'gemma-4-e4b-it-4bit',
    modelType: 'gemma4',
    installed: true,
    ctxWindow: 131072,
    decodeTps: 74,
    prefillTps: 2800,
    mtp: false,
    weight: 3,
    sizeGb: 5.2,
    bits: 4,
  },
  {
    name: 'LFM2.5-1.2B-Instruct-8bit',
    modelType: 'lfm2',
    installed: true,
    ctxWindow: 32768,
    decodeTps: 210,
    prefillTps: 9800,
    mtp: false,
    weight: 2,
    sizeGb: 2,
    bits: 8,
  },
  {
    name: 'Nemotron-3.5-Lightning-4bit',
    modelType: 'nemotron_h',
    installed: true,
    ctxWindow: 262144,
    decodeTps: 55,
    prefillTps: 2100,
    mtp: true,
    weight: 2,
    sizeGb: 18,
    bits: 4,
  },
  {
    name: 'Qwen3.5-4B-bf16',
    modelType: 'qwen3_5',
    installed: true,
    ctxWindow: 131072,
    decodeTps: 64,
    prefillTps: 4700,
    mtp: true,
    weight: 2,
    sizeGb: 8.4,
    bits: null,
  },
  {
    name: 'Qwen3-30B-A3B-4bit',
    modelType: 'qwen3_moe',
    installed: false,
    ctxWindow: 40960,
    decodeTps: 81,
    prefillTps: 2500,
    mtp: false,
    weight: 3,
    sizeGb: 17,
    bits: 4,
  },
  {
    name: 'gemma-3-12b-it-4bit',
    modelType: 'gemma3',
    installed: false,
    ctxWindow: 131072,
    decodeTps: 41,
    prefillTps: 1700,
    mtp: false,
    weight: 2,
    sizeGb: 7,
    bits: 4,
  },
  {
    name: 'Llama-3.2-3B-Instruct-4bit',
    modelType: 'llama',
    installed: false,
    ctxWindow: 131072,
    decodeTps: 120,
    prefillTps: 6100,
    mtp: false,
    weight: 1,
    sizeGb: 1.8,
    bits: 4,
  },
  {
    name: 'Mistral-Small-24B-4bit',
    modelType: 'mistral',
    installed: false,
    ctxWindow: 32768,
    decodeTps: 33,
    prefillTps: 1150,
    mtp: false,
    weight: 0,
    sizeGb: 13,
    bits: 4,
  },
];

/** Silent for whole weeks (gaps in every trend). */
const GAP_DAYS: ReadonlyMap<string, (day: number) => boolean> = new Map([
  ['Qwen3-30B-A3B-4bit', (day: number) => (day >= 7 && day <= 13) || (day >= 21 && day <= 27)],
  ['gemma-3-12b-it-4bit', (day: number) => day >= 14 && day <= 20],
]);
/** 1–3 turns on odd days only (low sample). */
const LOW_SAMPLE_MODEL = 'Mistral-Small-24B-4bit';
/** The day (days ago) with cold-tier corruption and queue drops. */
const TROUBLE_DAY = 3;

function writeModels(home: string, random: Random): void {
  const { modelsDir } = fixturePaths(home);
  mkdirSync(modelsDir, { recursive: true });
  for (const model of FIXTURE_MODELS.filter((entry) => entry.installed)) {
    const dir = join(modelsDir, model.name);
    mkdirSync(dir, { recursive: true });
    const config: Record<string, unknown> = {
      model_type: model.modelType,
      max_position_embeddings: model.ctxWindow,
    };
    if (model.bits !== null) config.quantization = { group_size: 64, bits: model.bits };
    writeJson(join(dir, 'config.json'), config);
    writeJson(join(dir, 'tokenizer_config.json'), { model_max_length: model.ctxWindow });
    const weights = join(dir, 'model.safetensors');
    writeFileSync(weights, '');
    truncateSync(weights, Math.round(model.sizeGb * GIB));
  }
  for (const name of ['Qwen3.5-27B-DFlash2-draft', 'Qwen3.5-4B-DFlash2-draft']) {
    const dir = join(modelsDir, name);
    mkdirSync(dir, { recursive: true });
    writeJson(join(dir, 'config.json'), { model_type: 'qwen3', architectures: ['DFlash2DraftModel'] });
    const weights = join(dir, 'model.safetensors');
    writeFileSync(weights, '');
    truncateSync(weights, int(random, 300, 900) * MIB);
  }
  const broken = join(modelsDir, 'partial-download');
  mkdirSync(broken, { recursive: true });
  writeFileSync(join(broken, 'model-00001-of-00004.safetensors'), '');
}

/** A line the server wrote before it measured prefill: it still carries the TTFT-derived `prefillTps`. */
type FixtureTraceRecord = MetricsTraceRecord & { prefillTps?: number };

/** Every turn of this model is non-streamed, so it never gets a prefill measurement. */
const UNMEASURED_MODEL = 'Llama-3.2-3B-Instruct-4bit';

function traceFor(
  random: Random,
  model: FixtureModel,
  ts: number,
  pid: number,
  index: number,
  coldRoot: string,
  trouble: boolean,
  coldTotals: { corruptions: number; drops: number },
  legacy: boolean,
): FixtureTraceRecord {
  const roll = random();
  const outcome: MetricsTraceRecord['outcome'] = roll < 0.94 ? 'ok' : roll < 0.98 ? 'error' : 'cancelled';
  const endpointRoll = random();
  const endpoint: MetricsTraceRecord['endpoint'] =
    endpointRoll < 0.7 ? 'messages' : endpointRoll < 0.95 ? 'responses' : 'other';
  const dispatchRoll = random();
  const dispatch: MetricsTraceRecord['dispatch'] =
    dispatchRoll < 0.4 ? 'start' : dispatchRoll < 0.75 ? 'continue' : 'continue_tool';
  const stream = model.name !== UNMEASURED_MODEL && random() < 0.85;
  const promptTokens = Math.round(Math.min(model.ctxWindow * 0.9, logNormal(random, 6000, 0.9)));
  const cachedTokens = dispatch === 'start' ? 0 : Math.round(promptTokens * between(random, 0.4, 0.95));
  const outputTokens = Math.max(1, Math.round(logNormal(random, 380, 0.8)));
  const reasoningTokens = random() < 0.5 ? Math.round(outputTokens * between(random, 0.1, 0.6)) : 0;
  const prefillTokens = Math.max(1, promptTokens - cachedTokens);
  const prefillRate = Math.max(1, logNormal(random, model.prefillTps, 0.15));
  const prefillMs = round((prefillTokens * 1000) / prefillRate + between(random, 8, 30));
  const setupMs = random() < 0.05 ? between(random, 500, 3000) : logNormal(random, 60, 0.5);
  const ttftMs = round(setupMs + prefillMs);
  const decodeTps = round(Math.max(1, logNormal(random, model.decodeTps, 0.12)), 2);
  const queueMs = random() < 0.15 ? round(between(random, 5, 900)) : round(between(random, 0, 3));
  const loadWaitMs = random() < 0.03 ? round(between(random, 800, 6000)) : undefined;
  const concurrency = int(random, 1, 3);
  const record: FixtureTraceRecord = {
    v: 2,
    traceId: `fx-${ts.toString(36)}-${index.toString(36)}`,
    ts,
    pid,
    sessionId: `fixture-session-${int(random, 1, 40)}`,
    model: random() < 0.1 ? 'claude-haiku-4-5' : model.name,
    modelId: model.name,
    modelType: model.modelType,
    outcome,
    endpoint,
    stream,
    dispatch,
    finishReason: outcome === 'ok' ? pick(random, ['stop', 'stop', 'stop', 'tool_calls', 'length']) : outcome,
    durationMs: round(queueMs + (loadWaitMs ?? 0) + ttftMs + (outputTokens / decodeTps) * 1000),
    queueMs,
    preDispatchMs: round(queueMs + (loadWaitMs ?? 0) + between(random, 0.2, 2)),
    promptTokens,
    cachedTokens,
    ctxWindow: model.ctxWindow,
    toolCount: int(random, 0, 14),
    toolCallCount: dispatch === 'continue_tool' ? int(random, 1, 4) : 0,
    imageCount: random() < 0.05 ? 1 : 0,
    concurrency,
    concurrencyPeak: concurrency + (random() < 0.2 ? 1 : 0),
    memActiveBytes: Math.round(model.sizeGb * GIB + between(random, 0.5, 3) * GIB),
    memPeakBytes: Math.round(model.sizeGb * GIB + between(random, 3, 6) * GIB),
  };
  if (loadWaitMs !== undefined) record.loadWaitMs = loadWaitMs;
  if (outcome === 'ok') {
    record.outputTokens = outputTokens;
    record.reasoningTokens = reasoningTokens;
    record.ttftMs = ttftMs;
    record.decodeTps = decodeTps;
    if (legacy) {
      record.prefillTps = round(((promptTokens - cachedTokens) * 1000) / ttftMs, 2);
    } else if (stream) {
      record.prefillMs = prefillMs;
      record.prefillTokens = prefillTokens;
    }
    if (model.mtp) {
      const accepted = round(between(random, 1.6, 2.7), 3);
      record.mtpCycles = Math.max(1, Math.round(outputTokens / accepted));
      record.mtpMeanAccepted = accepted;
      record.mtpMeanDrafts = round(accepted - 1, 3);
      record.mtpMeanDepth = 3;
      record.mtpAcceptByPos = [
        round(between(random, 0.7, 0.9), 4),
        round(between(random, 0.45, 0.7), 4),
        round(between(random, 0.2, 0.45), 4),
      ];
    }
    if (random() < 0.8) {
      record.coldRoot = coldRoot;
      record.coldEnabled = true;
      record.coldHits = int(random, 0, 24);
      record.coldMisses = int(random, 0, 6);
      record.coldBytesRestored = (record.coldHits ?? 0) * 2 * MIB;
      record.coldBytesWritten = int(random, 0, 8) * 2 * MIB;
      record.coldEnqueued = int(random, 0, 8);
      if (trouble && random() < 0.3) {
        coldTotals.corruptions += 1;
        coldTotals.drops += int(random, 2, 5);
        record.coldCorruptions = 1;
        record.coldQueueDrops = int(random, 2, 5);
      }
      record.coldCorruptionsTotal = coldTotals.corruptions;
      record.coldQueueDropsTotal = coldTotals.drops;
    } else {
      record.coldEnabled = false;
    }
  } else {
    if (outcome === 'error') record.errorKind = random() < 0.7 ? 'finish_error' : 'exception';
    if (stream) {
      record.failPhase = random() < 0.5 ? 'decode' : 'prefill';
      record.prefillDoneTokens = Math.round(promptTokens * between(random, 0.2, 1));
      if (record.failPhase === 'decode') record.outputTokens = Math.round(outputTokens * between(random, 0.1, 0.8));
    }
  }
  return record;
}

const TRACE_PIDS = [41001, 41002] as const;

function writeTraces(home: string, random: Random, now: number): number {
  const { tracesDir, cacheRoot } = fixturePaths(home);
  mkdirSync(tracesDir, { recursive: true });
  const weighted = FIXTURE_MODELS.filter((model) => model.weight > 0);
  const totalWeight = weighted.reduce((sum, model) => sum + model.weight, 0);
  const low = FIXTURE_MODELS.find((model) => model.name === LOW_SAMPLE_MODEL)!;
  const files = new Map<string, FixtureTraceRecord[]>();
  const coldTotals = { corruptions: 0, drops: 0 };
  let index = 0;
  for (let day = 29; day >= 0; day -= 1) {
    const pid = day >= 15 ? TRACE_PIDS[0] : TRACE_PIDS[1];
    const dayEnd = now - day * DAY_MS;
    const stamps: Array<{ ts: number; model: FixtureModel }> = [];
    const turns = int(random, 100, 140);
    for (let i = 0; i < turns; i += 1) {
      let target = random() * totalWeight;
      let model = weighted[0]!;
      for (const candidate of weighted) {
        target -= candidate.weight;
        if (target <= 0) {
          model = candidate;
          break;
        }
      }
      if (GAP_DAYS.get(model.name)?.(day) === true) continue;
      stamps.push({ ts: Math.round(dayEnd - random() * DAY_MS), model });
    }
    if (day % 2 === 1) {
      const count = int(random, 1, 3);
      for (let i = 0; i < count; i += 1) stamps.push({ ts: Math.round(dayEnd - random() * DAY_MS), model: low });
    }
    stamps.sort((a, b) => a.ts - b.ts);
    for (const stamp of stamps) {
      index += 1;
      const record = traceFor(
        random,
        stamp.model,
        stamp.ts,
        pid,
        index,
        cacheRoot,
        day === TROUBLE_DAY,
        coldTotals,
        day >= 15,
      );
      const name = `${utcDay(stamp.ts)}-${pid}.jsonl`;
      const list = files.get(name);
      if (list === undefined) files.set(name, [record]);
      else list.push(record);
    }
  }
  const oldest = now - 29 * DAY_MS;
  for (const [name, records] of files) {
    const path = join(tracesDir, name);
    writeLines(path, records);
    const last = records[records.length - 1]!.ts;
    const mtime = new Date(Math.min(now, Math.max(oldest, last)));
    utimesSync(path, mtime, mtime);
  }
  return index;
}

export interface LiveProcess {
  pid: number;
  startedAt: number;
  /** The load the harness engine runs now. */
  loading?: { model: string; startedAt: number } | null;
  /** Chat models the harness engine holds, after the two fixture models. */
  chatModels?: readonly string[];
  /**
   * Four more loaded models, one per state of the Engine pool section: turns that ran alone with 0 blocks held (the
   * owner's case), alone with blocks held, a flat-cache model, a paged model before its first snapshot, and a model
   * whose turns all ran before the last hour.
   */
  engineStates?: boolean;
  /** Fixed Now values instead of the moving decode rate. */
  ticker?: { decodeTps: number; memActive: number };
  /**
   * The turns in flight. 'two-models' (default): one turn on each fixture model, the big one asked through an alias.
   * 'one-model': two turns on the big model, one of them asked through the alias `mlx-local`.
   */
  flight?: 'two-models' | 'one-model';
  /**
   * The disk cache the heartbeat reports. 'whole-prompt' (default): on, the whole prompt per turn. 'native': on, the
   * native 128 blocks. 'off': off. 'older': neither field, as an engine from before them writes.
   */
  diskCache?: 'whole-prompt' | 'native' | 'off' | 'older';
}

/** A paged model whose turns never went through the batch scheduler. */
interface SerialModel {
  name: string;
  modelType: string;
  ctxWindow: number;
  held: number;
  /** ms before the end of the series: the load, then one snapshot per entry of `turns`. */
  loadedAgo: number;
  snapshots: ReadonlyArray<{ ago: number; turns: number }>;
  inFlight: number;
}

const SERIAL_MODELS: readonly SerialModel[] = [
  {
    name: 'Qwen3.8-27B-UD-Q4_K_XL',
    modelType: 'qwen3_5',
    ctxWindow: 266_000,
    held: 0,
    loadedAgo: 6 * 60_000,
    snapshots: [
      { ago: 6 * 60_000 - 1000, turns: 0 },
      { ago: 5 * 60_000 + 15_000, turns: 1 },
      { ago: 4 * 60_000, turns: 9 },
    ],
    inFlight: 1,
  },
  {
    name: 'Qwen3.5-4B-bf16',
    modelType: 'qwen3_5',
    ctxWindow: 262_144,
    held: 1200,
    loadedAgo: 30 * 60_000,
    snapshots: [
      { ago: 30 * 60_000 - 1000, turns: 0 },
      { ago: 12 * 60_000, turns: 12 },
    ],
    inFlight: 0,
  },
  {
    name: 'LFM2.5-1.2B-Instruct-8bit',
    modelType: 'lfm2',
    ctxWindow: 32_768,
    held: 0,
    loadedAgo: 4 * HOUR_MS,
    snapshots: [
      { ago: 4 * HOUR_MS - 1000, turns: 0 },
      { ago: 3 * HOUR_MS, turns: 5 },
    ],
    inFlight: 0,
  },
];
const FLAT_MODEL = { name: 'Mistral-Small-24B-4bit', modelType: 'mistral', ctxWindow: 32_768 };
const UNREAD_MODEL = { name: 'Nemotron-3.5-Lightning-4bit', modelType: 'nemotron_h', ctxWindow: 262_144 };
const SERIAL_CAPACITY = 16_384;

function serialSched(
  base: { v: 1; ts: number; pid: number; startedAt: number },
  model: SerialModel,
  turns: number,
): RuntimeTraceRecord {
  return {
    ...base,
    k: 'sched',
    model: model.name,
    turns,
    blockCapacity: SERIAL_CAPACITY,
    allocatedBlocks: model.held,
    freeBlocks: SERIAL_CAPACITY - model.held,
    reclaimableBlocks: 0,
    watermarkBlocks: 0,
    reservedBlocks: 0,
    memoryCapacityBytes: 16 * GIB,
    memoryWatermarkBytes: 0,
    reservedBlockBytes: 0,
    reservedStateBytes: 0,
    globalSteps: 0,
    admitted: 0,
    completed: 0,
    maxBatchOccupancy: 0,
    fusedGreedyEpilogueSteps: 0,
    admissionDeferredBlocks: 0,
    admissionDeferredState: 0,
    rowsAllocEvicted: 0,
    preemptions: 0,
    preemptionsRecompute: 0,
    preemptionsSsd: 0,
    ssdRestoreWaiting: 0,
    ssdRestoreBytes: 0,
    ssdRestoreWaitMs: 0,
    occHist: [],
  };
}

/** Load and snapshot records of the {@link SERIAL_MODELS}, ending at `to`. */
function serialRecords(base: (ts: number) => { v: 1; ts: number; pid: number; startedAt: number }, to: number) {
  const records: RuntimeTraceRecord[] = [];
  for (const model of SERIAL_MODELS) {
    records.push({
      ...base(to - model.loadedAgo),
      k: 'load',
      model: model.name,
      modelType: model.modelType,
      capability: 'chat',
      ms: 36_980,
      ok: true,
    });
    for (const snap of model.snapshots) records.push(serialSched(base(to - snap.ago), model, snap.turns));
  }
  return records;
}

const LIVE_MODELS = ['Qwen3.5-27B-4bit', 'Qwen3-8B-4bit'] as const;
const LIVE_PORT = 52114;
const FAILED_LOAD_MODEL = 'gemma-4-e4b-it-4bit';
/** A 60-character name: the Engine tables must hold it on one line at every card width. */
export const LONG_MODEL_NAME = 'qwen3.6-35b-a3b-agentworld-instruct-ud-q4-k-xl-mlx-dwq-8bits';
const DECODE_WAVE_MS = 9_000;
const PREFILL_TOTAL = 7_168;

function schedRecord(
  base: { v: 1; ts: number; pid: number; startedAt: number },
  model: string,
  step: number,
  random: Random,
): RuntimeTraceRecord {
  const capacity = 4096;
  const allocated = int(random, 200, 1400);
  const reclaimable = int(random, 100, 600);
  return {
    ...base,
    k: 'sched',
    model,
    turns: step * 16,
    blockCapacity: capacity,
    allocatedBlocks: allocated,
    freeBlocks: capacity - allocated,
    reclaimableBlocks: reclaimable,
    watermarkBlocks: 64,
    reservedBlocks: 32,
    memoryCapacityBytes: 6 * GIB,
    memoryWatermarkBytes: 96 * MIB,
    reservedBlockBytes: 48 * MIB,
    reservedStateBytes: 12 * MIB,
    globalSteps: step * 900,
    admitted: step * 14,
    completed: step * 14 - 1,
    maxBatchOccupancy: Math.min(4, 1 + Math.floor(step / 6)),
    fusedGreedyEpilogueSteps: step * 300,
    admissionDeferredBlocks: Math.floor(step / 5),
    admissionDeferredState: 0,
    rowsAllocEvicted: Math.floor(step / 8),
    preemptions: Math.floor(step / 7),
    preemptionsRecompute: Math.floor(step / 9),
    preemptionsSsd: Math.floor(step / 21),
    ssdRestoreWaiting: 0,
    ssdRestoreBytes: Math.floor(step / 21) * 64 * MIB,
    ssdRestoreWaitMs: Math.floor(step / 21) * 40,
    occHist: [
      [1, step * 500],
      [2, step * 280],
      [3, step * 90],
      [4, step * 30],
    ],
  };
}

function seriesFor(
  process: LiveProcess,
  from: number,
  to: number,
  random: Random,
  exit: boolean,
  failedLoadAt: number | null = null,
): RuntimeTraceRecord[] {
  const base = (ts: number) => ({ v: 1 as const, ts, pid: process.pid, startedAt: process.startedAt });
  const records: RuntimeTraceRecord[] = [
    {
      ...base(from),
      k: 'proc',
      host: 'desktop',
      version: '0.0.21',
      port: LIVE_PORT,
      wiredLimitBytes: 96 * GIB,
      totalMemBytes: 128 * GIB,
      chip: 'Apple M5 Max',
    },
  ];
  for (const [i, model] of LIVE_MODELS.entries()) {
    const info = FIXTURE_MODELS.find((entry) => entry.name === model)!;
    records.push({
      ...base(from + 1000 * (i + 1)),
      k: 'load',
      model,
      modelType: info.modelType,
      capability: 'chat',
      ms: int(random, 1800, 9000),
      ok: true,
    });
  }
  let previous = from;
  let step = 0;
  let failedLoad = failedLoadAt;
  for (let ts = from + 60_000; ts <= to;) {
    if (failedLoad !== null && ts >= failedLoad) {
      records.push({
        ...base(failedLoad),
        k: 'load',
        model: FAILED_LOAD_MODEL,
        modelType: 'gemma4',
        capability: 'chat',
        ms: 4200,
        ok: false,
      });
      records.push({
        ...base(failedLoad + 1000),
        k: 'load',
        model: LONG_MODEL_NAME,
        modelType: 'qwen3_5',
        capability: 'chat',
        ms: 204_000,
        ok: true,
      });
      failedLoad = null;
    }
    const busy = random() < 0.35;
    const dtMs = ts - previous;
    const busyMs = busy ? Math.round(dtMs * between(random, 0.3, 1)) : Math.round(dtMs * between(random, 0, 0.1));
    const ok = busy ? int(random, 1, 6) : 0;
    records.push({
      ...base(ts),
      k: 'sample',
      dtMs,
      busy,
      busyMs,
      health: 'ok',
      memActive: Math.round(between(random, 22, 30) * GIB),
      memPeak: Math.round(between(random, 31, 34) * GIB),
      memCache: Math.round(between(random, 1, 4) * GIB),
      rss: Math.round(between(random, 1.2, 1.8) * GIB),
      inFlight: busy ? int(random, 1, 3) : 0,
      queued: busy && random() < 0.1 ? 1 : 0,
      resident: LIVE_MODELS.length,
      dOk: ok,
      dError: busy && random() < 0.05 ? 1 : 0,
      dCancelled: 0,
      dOutTokens: ok * int(random, 120, 900),
      dRejected: busy && step % 13 === 5 ? 2 : 0,
    });
    if (!busy && step % 10 === 0) {
      for (const model of LIVE_MODELS) records.push(schedRecord(base(ts + 1), model, step / 10 + 1, random));
    }
    step += 1;
    previous = ts;
    ts += busy ? 5_000 : 60_000;
  }
  if (process.engineStates === true) {
    records.push(...serialRecords(base, to));
    records.sort((a, b) => a.ts - b.ts);
  }
  if (exit) records.push({ ...base(to + 1000), k: 'exit', uptimeMs: to + 1000 - process.startedAt });
  return records;
}

function writeSeries(dir: string, process: LiveProcess, records: readonly RuntimeTraceRecord[]): void {
  const byDay = new Map<string, RuntimeTraceRecord[]>();
  for (const record of records) {
    const name = `${utcDay(record.ts)}-${process.pid}.jsonl`;
    const list = byDay.get(name);
    if (list === undefined) byDay.set(name, [record]);
    else list.push(record);
  }
  for (const [name, list] of byDay) {
    const proc = records[0]!;
    const withHeader = list[0] === proc ? list : [{ ...proc, ts: list[0]!.ts }, ...list];
    const path = join(dir, name);
    writeLines(path, withHeader);
    const mtime = new Date(list[list.length - 1]!.ts);
    utimesSync(path, mtime, mtime);
  }
}

function writeRuntime(home: string, random: Random, now: number, live: LiveProcess): void {
  const { runtimeDir } = fixturePaths(home);
  mkdirSync(runtimeDir, { recursive: true });
  const old: LiveProcess = { pid: 41002, startedAt: now - 2 * DAY_MS };
  writeSeries(runtimeDir, old, seriesFor(old, old.startedAt, now - 7 * HOUR_MS, random, true));
  writeSeries(runtimeDir, live, seriesFor(live, live.startedAt, now - 30_000, random, false, now - 40 * 60_000));
}

/**
 * The heartbeat of one running process with 2 models and 2 turns in flight, one decoding and one in prefill. The
 * decode rate and the prefill progress follow `now`, so each rewrite moves them. Rewrite it to keep it fresh.
 */
export function writeLiveFile(home: string, live: LiveProcess, now: number = Date.now()): void {
  const { runtimeDir } = fixturePaths(home);
  const path = join(runtimeDir, `live-${live.pid}.json`);
  assertInsideHome(home, `${path}.tmp`);
  mkdirSync(runtimeDir, { recursive: true });
  const [big, small] = LIVE_MODELS;
  const oneModel = live.flight === 'one-model';
  const diskCache = live.diskCache ?? 'whole-prompt';
  const coldCapture: RuntimeLiveFile['coldCapture'] =
    diskCache === 'whole-prompt'
      ? { blocks: 63_125, ms: 10_000 }
      : diskCache === 'native'
        ? { blocks: 128, ms: 250 }
        : null;
  const file: RuntimeLiveFile = {
    v: 1,
    pid: live.pid,
    startedAt: live.startedAt,
    ts: now,
    intervalMs: 5000,
    host: 'desktop',
    version: '0.0.21',
    port: LIVE_PORT,
    persistPagedCache: diskCache !== 'off',
    coldCapture,
    health: 'ok',
    mem: {
      active: live.ticker?.memActive ?? 27 * GIB,
      peak: 33 * GIB,
      cache: 2 * GIB,
      wiredLimit: 96 * GIB,
      rss: Math.round(1.4 * GIB),
    },
    decodeTps: live.ticker?.decodeTps ?? round(37.5 + 6 * Math.sin(now / DECODE_WAVE_MS), 1),
    loading: live.loading ?? null,
    counters: { ok: 412, error: 6, cancelled: 3, outputTokens: 389_120, loads: 3, rejected: 5 },
    models: [
      {
        name: big,
        modelType: 'qwen3_5',
        ctxWindow: 131072,
        paged: true,
        maxConcurrent: 4,
        loadedAt: live.startedAt + 1000,
        loadMs: 6200,
        inFlight: oneModel ? 2 : 1,
        queued: 0,
        queueLimit: 32,
        pool: {
          at: now - 40_000,
          blockCapacity: 4096,
          allocatedBlocks: 1180,
          freeBlocks: 2916,
          reclaimableBlocks: 420,
        },
      },
      {
        name: small,
        modelType: 'qwen3',
        ctxWindow: 40960,
        paged: true,
        maxConcurrent: 4,
        loadedAt: live.startedAt + 2000,
        loadMs: 2100,
        inFlight: oneModel ? 0 : 1,
        queued: 0,
        queueLimit: 32,
        pool: { at: now - 40_000, blockCapacity: 2048, allocatedBlocks: 310, freeBlocks: 1738, reclaimableBlocks: 120 },
      },
      ...(live.chatModels ?? []).map((name) => ({
        name,
        modelType: 'qwen3_5',
        ctxWindow: 32768,
        paged: true,
        maxConcurrent: 4,
        loadedAt: now - 1000,
        loadMs: 4000,
        inFlight: 0,
        queued: 0,
        queueLimit: 32,
        pool: null,
      })),
      ...(live.engineStates === true ? engineStateModels(live, now) : []),
    ],
    turns: [
      {
        traceId: 'fx-live-turn',
        model: oneModel ? big : 'claude-sonnet-4-5',
        modelId: big,
        ctxWindow: 131072,
        startedAt: now - 6400,
        stream: true,
        dispatch: 'continue_tool',
        phase: 'decode',
        promptTokens: 18_432,
        cachedTokens: 16_896,
        prefillTotal: 1536,
        prefillDone: 1536,
        generatedTokens: 214,
      },
      {
        traceId: 'fx-live-prefill',
        model: oneModel ? 'mlx-local' : small,
        modelId: oneModel ? big : small,
        ctxWindow: oneModel ? 131072 : 40960,
        startedAt: now - 1800,
        stream: true,
        dispatch: 'start',
        phase: 'prefill',
        promptTokens: 9_216,
        cachedTokens: 2_048,
        prefillTotal: PREFILL_TOTAL,
        prefillDone: Math.floor((PREFILL_TOTAL * ((Math.floor(now / 1000) % 10) + 1)) / 12),
        generatedTokens: null,
      },
    ],
  };
  if (diskCache === 'older') {
    const { persistPagedCache: _switch, coldCapture: _capture, ...older } = file;
    writeFileSync(`${path}.tmp`, JSON.stringify(older));
  } else {
    writeFileSync(`${path}.tmp`, JSON.stringify(file));
  }
  renameSync(`${path}.tmp`, path);
}

/** The live entries of the {@link LiveProcess.engineStates} models. */
function engineStateModels(live: LiveProcess, now: number): RuntimeLiveFile['models'] {
  const end = now - 30_000;
  const serial = SERIAL_MODELS.map((model) => {
    const last = model.snapshots[model.snapshots.length - 1]!;
    return {
      name: model.name,
      modelType: model.modelType,
      ctxWindow: model.ctxWindow,
      paged: true,
      maxConcurrent: 1,
      loadedAt: end - model.loadedAgo,
      loadMs: 36_980,
      inFlight: model.inFlight,
      queued: 0,
      queueLimit: 16,
      pool: {
        at: end - last.ago,
        blockCapacity: SERIAL_CAPACITY,
        allocatedBlocks: model.held,
        freeBlocks: SERIAL_CAPACITY - model.held,
        reclaimableBlocks: 0,
      },
    };
  });
  const loadedAt = live.startedAt + 3000;
  return [
    ...serial,
    {
      ...FLAT_MODEL,
      paged: false,
      maxConcurrent: 1,
      loadedAt,
      loadMs: 5100,
      inFlight: 0,
      queued: 0,
      queueLimit: 16,
      pool: null,
    },
    {
      ...UNREAD_MODEL,
      paged: true,
      maxConcurrent: 8,
      loadedAt: now - 20_000,
      loadMs: 9800,
      inFlight: 0,
      queued: 0,
      queueLimit: 32,
      pool: null,
    },
  ];
}

/** The desktop supervisor's state file, with `mainPid` alive so the reader trusts it. */
export function writeSupervisorFile(home: string, mainPid: number, sidecarPid: number, now: number = Date.now()): void {
  const { supervisorPath } = fixturePaths(home);
  assertInsideHome(home, supervisorPath);
  mkdirSync(join(home, 'desktop'), { recursive: true });
  const state: SupervisorStateFile = {
    version: 1,
    writtenAt: now,
    mainPid,
    state: 'running',
    pid: sidecarPid,
    generation: 2,
    streak: 0,
    shortLivedFailures: 0,
    lastExit: { verdict: 'clean', code: 0, signal: null, atMs: now - 7 * HOUR_MS },
    healthStatus: 'ok',
    nativeErrorCount: 0,
    nativeErrorsCapped: false,
    lastNativeErrorContext: null,
    lastNativeErrorAt: null,
  };
  writeJson(supervisorPath, state);
}

function writeColdCache(home: string, random: Random, now: number): void {
  const { cacheRoot } = fixturePaths(home);
  mkdirSync(cacheRoot, { recursive: true });
  const ages = [0.1, 0.5, 2, 5, 10, 20, 28];
  for (let i = 0; i < 140; i += 1) {
    const key = hex(random, 64);
    const sidecar = i % 9 === 0;
    const name = sidecar ? `${key}.${i % 2 === 0 ? 'gdn_state' : 'sliding_window'}.safetensors` : `${key}.safetensors`;
    const path = join(cacheRoot, name);
    writeFileSync(path, '');
    truncateSync(path, int(random, 4, 64) * MIB);
    const age = pick(random, ages) * DAY_MS * between(random, 0.6, 1);
    const mtime = new Date(now - age);
    utimesSync(path, mtime, mtime);
  }
}

function cloneId(n: number): string {
  return `01a20001-0000-7000-8000-${(n + 1).toString(16).padStart(12, '0')}`;
}

/** The newest cloned session: the Session detail screenshot opens it. */
export const FIRST_SESSION_ID = cloneId(0);

/** The last two sit under the real home folder, as text only, so the window shows them as `~/…`. */
const SESSION_CWDS = [
  '/Users/dev/proj',
  '/Users/dev/other',
  '/Users/dev/api',
  '/tmp/scratch',
  '/Users/dev/site',
  join(homedir(), 'workspace', 'github', 'mlx-node'),
  join(homedir(), 'workspace', 'notes'),
];

function shiftUpdates(text: string, oldId: string, newId: string, offsetMs: number): string {
  const out: string[] = [];
  for (const line of text.split('\n')) {
    if (line.trim().length === 0) continue;
    let parsed: { timestamp?: unknown; params?: { _meta?: { agentTimestampMs?: unknown } } };
    try {
      parsed = JSON.parse(line.split(oldId).join(newId)) as typeof parsed;
    } catch {
      continue;
    }
    if (typeof parsed.timestamp === 'number') parsed.timestamp = Math.round(parsed.timestamp + offsetMs / 1000);
    const meta = parsed.params?._meta;
    if (meta !== undefined && typeof meta.agentTimestampMs === 'number') meta.agentTimestampMs += offsetMs;
    out.push(JSON.stringify(parsed));
  }
  return `${out.join('\n')}\n`;
}

function writeSessions(home: string, random: Random, now: number): number {
  const { agentHome } = fixturePaths(home);
  cpSync(GROK_FIXTURE, agentHome, { recursive: true });
  const sourceDir = join(agentHome, 'sessions', encodeURIComponent(RICH_SESSION_CWD), RICH_SESSION);
  const summary = JSON.parse(readFileSync(join(sourceDir, 'summary.json'), 'utf8')) as Record<string, unknown>;
  const updates = readFileSync(join(sourceDir, 'updates.jsonl'), 'utf8');
  const lastActive = Date.parse(String(summary.last_active_at));
  const installed = FIXTURE_MODELS.filter((model) => model.installed).map((model) => model.name);
  const titles = [
    'Research the project layout',
    'Fix the flaky cache test',
    'Add a CSV export',
    'Review the parser',
    'Tidy the release script',
    'Explain the scheduler',
  ];
  const count = 240;
  for (let n = 0; n < count; n += 1) {
    const id = cloneId(n);
    const cwd = SESSION_CWDS[n % SESSION_CWDS.length]!;
    const active = now - n * 3 * HOUR_MS - int(random, 0, 50) * 60_000;
    const offset = active - lastActive;
    const dir = join(agentHome, 'sessions', encodeURIComponent(cwd), id);
    mkdirSync(dir, { recursive: true });
    const clone = {
      ...summary,
      info: { id, cwd },
      created_at: new Date(Date.parse(String(summary.created_at)) + offset).toISOString(),
      updated_at: new Date(active).toISOString(),
      last_active_at: new Date(active).toISOString(),
      current_model_id: installed[n % installed.length],
      generated_title: `${titles[n % titles.length]} #${n + 1}`,
      git_root_dir: `${cwd}/`,
    };
    writeJson(join(dir, 'summary.json'), clone);
    writeFileSync(join(dir, 'updates.jsonl'), shiftUpdates(updates, RICH_SESSION, id, offset));
  }
  return count;
}

/** A long finished session (with subagents), and a live one whose last turn is still running. */
export const LONG_SESSION_ID = '01a20002-0000-7000-8000-000000000001';
export const LIVE_SESSION_ID = '01a20002-0000-7000-8000-000000000002';
const LONG_SESSION_CWD = '/Users/dev/proj';
const LONG_TURNS = 48;
const MAIN_MODEL = 'Qwen3.5-27B-4bit';

/** A subagent run the parent spawns inside a turn. */
interface ChildPlan {
  id: string;
  type: string;
  description: string;
  model: string;
  /** foreground: the parent waits for the return; background: the parent goes on. */
  mode: 'foreground' | 'background';
  status: 'completed' | 'failed';
  /** ms after the previous spawn (the first: after the parent's last step). */
  startAfterMs: number;
  /** missing: no session folder; empty: a folder whose event file is empty; open-tool: ends in a tool in progress. */
  steps: 'normal' | 'failed-tool' | 'open-tool' | 'empty' | 'missing';
  /** false: no finish line (still running). */
  finish: boolean;
  /** tool: a `task` call, then the spawn line; lines: the spawn and finish lines only. */
  spawn: 'tool' | 'lines';
  /** Tool calls of the child's own run. Default 3. */
  tools?: number;
}

function childId(n: number): string {
  return `01a20002-0000-7000-8000-0000000000${String(n).padStart(2, '0')}`;
}

function childPlan(n: number, description: string, type: string, overrides: Partial<ChildPlan> = {}): ChildPlan {
  return {
    id: childId(n),
    type,
    description,
    model: MAIN_MODEL,
    mode: 'foreground',
    status: 'completed',
    startAfterMs: 40,
    steps: 'normal',
    finish: true,
    spawn: 'tool',
    ...overrides,
  };
}

const MAP_QUEUES = childPlan(11, 'Map the scheduler queues', 'explore');

/** Subagents of the long session, by 0-based turn index. */
const TURN_CHILDREN: ReadonlyMap<number, readonly ChildPlan[]> = new Map([
  [
    5,
    [
      MAP_QUEUES,
      childPlan(12, 'Write the planner tests', 'general-purpose', { model: 'Qwen3-8B-4bit' }),
      childPlan(13, 'Bisect the cache regression', 'codex:codex-rescue', { status: 'failed', steps: 'failed-tool' }),
      childPlan(14, 'Check the docs', 'explore', { steps: 'missing' }),
    ],
  ],
  [7, [childPlan(15, 'Trace the loader', 'explore', { spawn: 'lines' })]],
  [17, [childPlan(16, 'Watch the nightly build', 'general-purpose', { mode: 'background', tools: 4 })]],
  [
    29,
    Array.from({ length: 10 }, (_, i) => childPlan(17 + i, `Survey crate ${String(i + 1)}`, 'explore', { tools: 1 })),
  ],
  [41, [MAP_QUEUES, childPlan(27, 'Collect the benchmark logs', 'general-purpose', { steps: 'empty' })]],
]);

/** Turn 26 starts a background workflow; its children run on after the turn; its stop wakes the agent for turn 27. */
const WORKFLOW_TURN = 25;
const WORKFLOW_RUN = 'wf_dev_0001';
const WORKFLOW_NAME = 'deep-research';
const WORKFLOW_CHILDREN = [
  childPlan(34, 'Survey the prior art', 'explore', { mode: 'background', spawn: 'lines', tools: 5 }),
  childPlan(35, 'Verify the sources', 'explore', { mode: 'background', spawn: 'lines', tools: 2 }),
  childPlan(36, 'Write the report', 'general-purpose', { mode: 'background', spawn: 'lines', tools: 1 }),
] as const;
/** The prompt the agent writes to itself when a background workflow stops (grok `run_loop.rs`). */
const WORKFLOW_WAKE_TEXT =
  'A background workflow stopped. Review the workflow completion reminder, report the result to the user, and take any appropriate next action.';

/** The live session's open turn: one child back, one failed, one still running. */
const LIVE_CHILDREN: readonly ChildPlan[] = [
  childPlan(31, 'Read the upload client', 'explore', { tools: 1 }),
  childPlan(32, 'Find the retry helper', 'explore', { status: 'failed', steps: 'failed-tool', tools: 1 }),
  childPlan(33, 'Run the upload tests', 'general-purpose', { steps: 'open-tool', finish: false, tools: 2 }),
];

const PROMPTS = [
  'Split the planner into three modules and keep the public API unchanged.',
  'Run the cache tests and fix whatever fails.',
  'Why does the scheduler stall when two sessions share a prefix?',
  'Add a CSV export to the metrics page.',
  'Review the diff and list anything risky.',
  'Summarise what changed for the release notes.',
];
const THOUGHTS = [
  'The planner mixes parsing, scheduling and IO. ',
  'I should read the module first, then move one piece at a time so the tests stay green. ',
  'The failing test reads a stale snapshot; the fix belongs in the loader, not in the test. ',
];
const ANSWERS = [
  'Done. The planner is now split into **three** modules:\n\n- `parse.rs` reads the request\n- `plan.rs` orders the work\n- `io.rs` talks to the disk\n\nThe public API is unchanged and `cargo test -p mlx-core` passes.',
  'Two tests failed because the loader cached a body across a reconnect. I now drop the cache on reconnect; all **42** tests pass.',
  'The stall comes from both sessions waiting on the same prefix block. The fix releases the block when the first prefill finishes.',
];
const TOOLS: ReadonlyArray<{ name: string; kind: string; input: (n: number) => unknown; output: string }> = [
  {
    name: 'read_file',
    kind: 'read',
    input: (n) => ({ target_file: `/Users/dev/proj/src/planner/mod${n % 4}.rs` }),
    output: 'pub fn plan(request: &Request) -> Plan {\n    let parsed = parse(request);\n    order(parsed)\n}\n',
  },
  {
    name: 'grep',
    kind: 'search',
    input: (n) => ({ pattern: 'prefix_block', path: 'src', head_limit: 20 + n }),
    output: 'src/sched.rs:212: let block = prefix_block(&key);\nsrc/sched.rs:340: release(prefix_block(&key));\n',
  },
  {
    name: 'run_terminal_command',
    kind: 'execute',
    input: () => ({ command: 'cargo test -p mlx-core planner' }),
    output: 'running 42 tests\n........................................ 42 passed\n',
  },
  {
    name: 'edit_file',
    kind: 'edit',
    input: (n) => ({ target_file: `/Users/dev/proj/src/planner/plan${n % 3}.rs` }),
    output: 'Applied 1 edit.',
  },
];

interface SessionWriter {
  lines: string[];
  seq: number;
  /** Lines due at a later time (a background child's return): written before the first line at or after it. */
  pending: Array<{ at: number; write: () => void }>;
  /** Lines carry no time: `timestamp` 0 and no `agentTimestampMs`. */
  untimed: boolean;
}

function newWriter(): SessionWriter {
  return { lines: [], seq: 0, pending: [], untimed: false };
}

function flushPending(writer: SessionWriter, upTo: number): void {
  for (;;) {
    let best = -1;
    for (let i = 0; i < writer.pending.length; i += 1) {
      const item = writer.pending[i]!;
      if (item.at <= upTo && (best === -1 || item.at < writer.pending[best]!.at)) best = i;
    }
    if (best === -1) return;
    const [item] = writer.pending.splice(best, 1);
    item!.write();
  }
}

function sessionEvent(
  writer: SessionWriter,
  id: string,
  ms: number,
  method: string,
  update: Record<string, unknown>,
  meta: Record<string, unknown> = {},
): void {
  flushPending(writer, ms);
  writer.seq += 1;
  const time = writer.untimed ? {} : { agentTimestampMs: ms };
  writer.lines.push(
    JSON.stringify({
      timestamp: writer.untimed ? 0 : Math.floor(ms / 1000),
      method,
      params: { sessionId: id, update, _meta: { eventId: `${id}-${writer.seq}`, ...time, ...meta } },
    }),
  );
}

interface TurnPlan {
  start: number;
  prompt: string;
  promptIndex: number;
  model: string;
  tools: number;
  failTool: boolean;
  plan: boolean;
  children: readonly ChildPlan[];
  /** Leave the turn running: no answer, no end, the last tool (or the last spawn call) in progress. */
  open: boolean;
  cancelled: boolean;
  /**
   * `first-call`: the `perf` block today's agent writes (the first model call only). `measured`: also the summed
   * prefill window of every call. `partial`: the sums cover one call fewer than the turn made.
   */
  perf: 'first-call' | 'measured' | 'partial';
  /** Overlapping read_file calls in place of the sequential tools. */
  parallel?: number;
  /** A model request retry this many ms after the first reasoning chunk. */
  retryAt?: number;
  /** The first tool call gets no completed update. */
  noEndTool?: boolean;
  /** Every line of the turn has no time. */
  untimed?: boolean;
  /** The answer does not start before this time (a background child returns first). */
  holdUntil?: number;
  /** Background work the turn starts after its tools; returns the time the turn goes on. */
  background?: (at: number) => number;
  /** A background event woke the agent: the prompt id it recorded; the prompt is hidden and nothing is typed. */
  wake?: string;
}

interface ChildRecord {
  plan: ChildPlan;
  writer: SessionWriter;
  /** Order of the first spawn among the session's children. */
  order: number;
  runs: Array<{ spawn: number; finish: number | null; end: number }>;
}

type ChildRegistry = Map<string, ChildRecord>;

/** One run of a child on the parent's clock: its own steps from the spawn; returns the finish time, null if still running. */
function runChild(kids: ChildRegistry, child: ChildPlan, spawn: number, random: Random): number | null {
  let record = kids.get(child.id);
  if (record === undefined) {
    record = { plan: child, writer: newWriter(), order: kids.size, runs: [] };
    kids.set(child.id, record);
  }
  const end =
    child.steps === 'missing' || child.steps === 'empty'
      ? spawn + int(random, 6_000, 20_000)
      : writeTurn(record.writer, child.id, random, {
          start: spawn + 5,
          prompt: child.description,
          promptIndex: record.runs.length,
          model: child.model,
          tools: child.tools ?? 3,
          failTool: child.steps === 'failed-tool',
          plan: false,
          children: [],
          open: child.steps === 'open-tool',
          cancelled: false,
          perf: record.order % 2 === 0 ? 'first-call' : 'partial',
        });
  const finish = child.finish ? end + 20 : null;
  record.runs.push({ spawn, finish, end });
  return finish;
}

const SPAWN_TOOL = {
  version: 1,
  name: 'task',
  kind: 'other',
  namespace: 'grok_build',
  label: 'task',
  read_only: false,
};

/** One agent turn as grok records it; returns the ms it ended at. */
function writeTurn(
  writer: SessionWriter,
  id: string,
  random: Random,
  turn: TurnPlan,
  kids: ChildRegistry = new Map(),
): number {
  writer.untimed = turn.untimed === true;
  const update = 'session/update';
  const custom = '_x.ai/session/update';
  const promptId = turn.wake ?? `${id.slice(-6)}-p${turn.promptIndex}`;
  let at = turn.start;
  if (turn.wake !== undefined) {
    sessionEvent(writer, id, at, custom, {
      sessionUpdate: 'hook_execution',
      event_name: 'user_prompt_submit',
      prompt_id: turn.wake,
      runs: [],
    });
  }
  sessionEvent(writer, id, at, update, {
    sessionUpdate: 'user_message_chunk',
    content: { type: 'text', text: turn.wake === undefined ? turn.prompt : WORKFLOW_WAKE_TEXT },
    _meta: {
      modelId: turn.model,
      promptIndex: turn.promptIndex,
      ...(turn.wake === undefined ? {} : { hideFromScrollback: true }),
    },
  });
  const ttft = int(random, 300, 2_400);
  at += ttft;
  const chunk = { promptId, turnStartMs: turn.start, streamStartMs: at, updateType: 'AgentMessageChunk' };
  if (turn.retryAt !== undefined) {
    const retryAt = at + turn.retryAt;
    writer.pending.push({
      at: retryAt,
      write: () =>
        sessionEvent(writer, id, retryAt, custom, {
          sessionUpdate: 'retry_state',
          type: 'retrying',
          attempt: 1,
          max_retries: 3,
          reason: 'Response stalled; retrying request',
          error_type: 'idle_timeout',
        }),
    });
  }
  for (const text of [pick(random, THOUGHTS), pick(random, THOUGHTS)]) {
    sessionEvent(
      writer,
      id,
      at,
      update,
      { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text } },
      chunk,
    );
    at += int(random, 400, 1_600);
  }
  if (turn.plan) {
    sessionEvent(writer, id, at, update, {
      sessionUpdate: 'plan',
      entries: [
        { content: 'Read the planner module', priority: 'high', status: 'completed' },
        { content: 'Move parsing into parse.rs', priority: 'medium', status: turn.open ? 'in_progress' : 'completed' },
        { content: 'Run the tests', priority: 'medium', status: turn.open ? 'pending' : 'completed' },
      ],
    });
    at += 200;
  }
  if (turn.parallel !== undefined) {
    const read = TOOLS[0]!;
    let latest = at;
    for (let n = 0; n < turn.parallel; n += 1) {
      const callAt = at + n * 40;
      const callId = `toolu_${turn.promptIndex}_p${n}`;
      sessionEvent(writer, id, callAt, update, {
        sessionUpdate: 'tool_call',
        toolCallId: callId,
        title: read.name,
        rawInput: read.input(turn.promptIndex + n),
        _meta: {
          'x.ai/tool': {
            version: 1,
            name: read.name,
            kind: read.kind,
            namespace: 'grok_build',
            label: read.name,
            read_only: true,
          },
        },
      });
      const doneAt = callAt + int(random, 200, 700);
      latest = Math.max(latest, doneAt);
      writer.pending.push({
        at: doneAt,
        write: () =>
          sessionEvent(writer, id, doneAt, update, {
            sessionUpdate: 'tool_call_update',
            toolCallId: callId,
            status: 'completed',
            content: [{ type: 'content', content: { type: 'text', text: read.output } }],
          }),
      });
    }
    flushPending(writer, latest);
    at = latest + int(random, 300, 1_800);
  }
  for (let n = 0; n < turn.tools; n += 1) {
    const tool = TOOLS[(turn.promptIndex + n) % TOOLS.length]!;
    const callId = `toolu_${turn.promptIndex}_${n}`;
    sessionEvent(writer, id, at, update, {
      sessionUpdate: 'tool_call',
      toolCallId: callId,
      title: tool.name,
      rawInput: tool.input(turn.promptIndex + n),
      _meta: {
        'x.ai/tool': {
          version: 1,
          name: tool.name,
          kind: tool.kind,
          namespace: 'grok_build',
          label: tool.name,
          read_only: tool.kind === 'read',
        },
      },
    });
    const last = n === turn.tools - 1;
    if (turn.open && last && turn.children.length === 0) {
      sessionEvent(writer, id, at + 50, update, {
        sessionUpdate: 'tool_call_update',
        toolCallId: callId,
        status: 'in_progress',
      });
      return at;
    }
    at += tool.kind === 'execute' ? int(random, 8_000, 40_000) : int(random, 80, 900);
    const failed = turn.failTool && last;
    if (turn.noEndTool === true && n === 0) {
      at += int(random, 300, 1_800);
      continue;
    }
    sessionEvent(writer, id, at, update, {
      sessionUpdate: 'tool_call_update',
      toolCallId: callId,
      status: failed ? 'failed' : 'completed',
      content: [
        {
          type: 'content',
          content: {
            type: 'text',
            text: failed
              ? 'error[E0425]: cannot find value `prefix` in this scope\n --> src/sched.rs:214:9'
              : tool.output,
          },
        },
      ],
    });
    at += int(random, 300, 1_800);
  }
  if (turn.background !== undefined) at = turn.background(at);
  if (turn.children.length > 0) {
    let spawnAt = at;
    let waitUntil = at;
    let running: string | null = null;
    for (const [index, child] of turn.children.entries()) {
      spawnAt += child.startAfterMs;
      const callId = `task_${turn.promptIndex}_${index}`;
      if (child.spawn === 'tool') {
        sessionEvent(writer, id, spawnAt, update, {
          sessionUpdate: 'tool_call',
          toolCallId: callId,
          title: 'task',
          rawInput: { subagent_type: child.type, run_in_background: child.mode === 'background' },
          _meta: { 'x.ai/tool': SPAWN_TOOL },
        });
      }
      const spawn = spawnAt + (child.spawn === 'tool' ? 10 : 0);
      sessionEvent(writer, id, spawn, custom, {
        sessionUpdate: 'subagent_spawned',
        subagent_id: child.id,
        attempt_id: 'at1.1',
        parent_session_id: id,
        parent_prompt_id: promptId,
        child_session_id: child.id,
        subagent_type: child.type,
        description: child.description,
        effective_context_source: 'new',
        model: child.model,
      });
      const finish = runChild(kids, child, spawn, random);
      if (finish !== null) {
        const finished = {
          sessionUpdate: 'subagent_finished',
          subagent_id: child.id,
          attempt_id: 'at1.1',
          child_session_id: child.id,
          status: child.status,
          tool_calls: int(random, 2, 9),
          turns: 1,
          duration_ms: finish - spawn,
          tokens_used: int(random, 4_000, 30_000),
          will_wake: child.mode === 'background',
          ...(child.status === 'failed' ? { error: 'The subagent stopped after its tool budget.' } : {}),
        };
        writer.pending.push({ at: finish, write: () => sessionEvent(writer, id, finish, custom, finished) });
      }
      if (child.spawn === 'tool') {
        const done = child.mode === 'background' ? spawn + 80 : finish === null ? null : finish + 10;
        if (done === null) {
          running = callId;
          sessionEvent(writer, id, spawn + 50, update, {
            sessionUpdate: 'tool_call_update',
            toolCallId: callId,
            status: 'in_progress',
          });
        } else {
          writer.pending.push({
            at: done,
            write: () =>
              sessionEvent(writer, id, done, update, {
                sessionUpdate: 'tool_call_update',
                toolCallId: callId,
                status: 'completed',
                content: [{ type: 'content', content: { type: 'text', text: 'Subagent finished.' } }],
              }),
          });
          waitUntil = Math.max(waitUntil, done);
        }
      } else if (child.mode === 'foreground' && finish !== null) {
        waitUntil = Math.max(waitUntil, finish);
      }
    }
    if (turn.open && running !== null) {
      flushPending(writer, waitUntil);
      return Math.max(spawnAt, waitUntil);
    }
    flushPending(writer, waitUntil);
    at = Math.max(spawnAt, waitUntil) + 400;
  }
  if (turn.holdUntil !== undefined) at = Math.max(at, turn.holdUntil);
  const answer = pick(random, ANSWERS);
  const half = Math.floor(answer.length / 2);
  for (const text of [answer.slice(0, half), answer.slice(half)]) {
    sessionEvent(
      writer,
      id,
      at,
      update,
      { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } },
      chunk,
    );
    at += int(random, 600, 3_000);
  }
  const input = int(random, 12_000, 90_000);
  const cached = Math.round(input * between(random, 0.55, 0.95));
  const output = int(random, 300, 4_000);
  const decode = between(random, 28, 64);
  const modelCalls = turn.tools + 1;
  const firstCallTokens = int(random, 200, 900);
  const prefillTokens = Math.max(1, input - cached);
  const prefillCalls = turn.perf === 'partial' ? Math.max(1, modelCalls - 1) : modelCalls;
  const prefillMs = round(
    Math.min(
      (prefillTokens / between(random, 500, 1_400)) * 1000 + prefillCalls * between(random, 8, 30),
      (at - turn.start) * 0.5,
    ),
  );
  const measured =
    turn.perf === 'first-call'
      ? {}
      : { prefill_ms: prefillMs, prefill_tokens: prefillTokens, prefill_calls: prefillCalls };
  sessionEvent(writer, id, at, custom, {
    sessionUpdate: 'turn_completed',
    prompt_id: promptId,
    stop_reason: turn.cancelled ? 'cancelled' : 'end_turn',
    usage: {
      inputTokens: input,
      outputTokens: output,
      totalTokens: input + output,
      cachedReadTokens: cached,
      cacheCreationTokens: 0,
      reasoningTokens: int(random, 40, 600),
      modelCalls,
      apiDurationMs: at - turn.start,
      numTurns: turn.tools + 1,
    },
    elapsed_ms: at - turn.start,
    perf: {
      ttft_ms: ttft,
      prefill_tokens_per_second: (firstCallTokens * 1000) / ttft,
      decode_tokens_per_second: decode,
      prefill_input_tokens: firstCallTokens,
      cached_prefix_tokens: cached,
      ...measured,
    },
  });
  return at;
}

function writeSessionSummary(
  dir: string,
  id: string,
  cwd: string,
  title: string,
  model: string,
  created: number,
  active: number,
  extra: Record<string, unknown> = {},
): void {
  writeJson(join(dir, 'summary.json'), {
    info: { id, cwd },
    agent_id: id,
    attempt_id: 'at1.0000',
    session_summary: title,
    created_at: new Date(created).toISOString(),
    updated_at: new Date(active).toISOString(),
    num_messages: 2,
    num_chat_messages: 2,
    current_model_id: model,
    next_trace_turn: 1,
    chat_format_version: 1,
    grok_home: '/Users/dev/.mlx-agent',
    generated_title: title,
    last_active_at: new Date(active).toISOString(),
    ...extra,
  });
}

/** The long session, its subagents, and the live session registered under `pid`. */
/** Session folder, summary and meta.json of every child a parent ran; a `missing` child gets meta.json only. */
function writeChildren(home: string, parentDir: string, parentId: string, cwd: string, kids: ChildRegistry): void {
  const { agentHome } = fixturePaths(home);
  for (const record of kids.values()) {
    const { plan, runs } = record;
    const first = runs[0]!;
    const last = runs[runs.length - 1]!;
    const metaDir = join(parentDir, 'subagents', plan.id);
    mkdirSync(metaDir, { recursive: true });
    writeJson(join(metaDir, 'meta.json'), {
      subagent_id: plan.id,
      attempt_id: 'at1.1',
      parent_session_id: parentId,
      child_session_id: plan.id,
      subagent_type: plan.type,
      description: plan.description,
      prompt: plan.description,
      status: last.finish === null ? 'running' : plan.status,
      started_at: new Date(first.spawn).toISOString(),
      effective_context_source: 'new',
      child_cwd: cwd,
      effective_model_id: plan.model,
      ...(last.finish === null
        ? {}
        : { completed_at: new Date(last.finish).toISOString(), duration_ms: last.finish - last.spawn }),
      tool_calls: plan.tools ?? 3,
      turns: runs.length,
      error: plan.status === 'failed' ? 'The subagent stopped after its tool budget.' : null,
    });
    // A missing child's folder is gone, so the page shows it unlinked.
    if (plan.steps === 'missing') continue;
    const childDir = join(agentHome, 'sessions', encodeURIComponent(cwd), plan.id);
    mkdirSync(childDir, { recursive: true });
    flushPending(record.writer, Number.POSITIVE_INFINITY);
    const lines = record.writer.lines;
    writeFileSync(join(childDir, 'updates.jsonl'), lines.length === 0 ? '' : `${lines.join('\n')}\n`);
    writeSessionSummary(childDir, plan.id, cwd, plan.description, plan.model, first.spawn, last.end, {
      session_kind: 'subagent',
      agent_name: plan.type,
    });
  }
}

/**
 * The background workflow of turn 26: `start` writes the workflow call and spawns its first child inside the turn;
 * `finish` writes, after the turn's end, that child's return, two more children and the stop. Returns the stop time.
 * Its own random stream keeps every other turn of the fixture as it was.
 */
function backgroundWorkflow(
  writer: SessionWriter,
  kids: ChildRegistry,
): { start: (at: number) => number; finish: (turnEnd: number) => number } {
  const random = mulberry32(0x77ac);
  const update = 'session/update';
  const custom = '_x.ai/session/update';
  const [first, second, third] = WORKFLOW_CHILDREN;
  let carried: { spawn: number; finish: number | null } | null = null;
  const status = (ms: number, value: string, revision: number): void =>
    sessionEvent(writer, LONG_SESSION_ID, ms, custom, {
      sessionUpdate: 'workflow_updated',
      run_id: WORKFLOW_RUN,
      name: WORKFLOW_NAME,
      status: value,
      revision,
      foreground: false,
      ...(value === 'complete' ? { result_summary: 'Three sources agree; the report is written.' } : {}),
    });
  const spawn = (child: ChildPlan, ms: number): number | null => {
    sessionEvent(writer, LONG_SESSION_ID, ms, custom, {
      sessionUpdate: 'subagent_spawned',
      subagent_id: child.id,
      attempt_id: 'at1.1',
      parent_session_id: LONG_SESSION_ID,
      child_session_id: child.id,
      subagent_type: child.type,
      description: child.description,
      effective_context_source: 'new',
      model: child.model,
      workflow_run_id: WORKFLOW_RUN,
    });
    return runChild(kids, child, ms, random);
  };
  const finished = (child: ChildPlan, spawned: number, ms: number): void => {
    const runs = kids.get(child.id)!.runs;
    runs[runs.length - 1]!.finish = ms;
    sessionEvent(writer, LONG_SESSION_ID, ms, custom, {
      sessionUpdate: 'subagent_finished',
      subagent_id: child.id,
      attempt_id: 'at1.1',
      child_session_id: child.id,
      status: child.status,
      tool_calls: child.tools ?? 3,
      turns: 1,
      duration_ms: ms - spawned,
      tokens_used: int(random, 4_000, 30_000),
      will_wake: false,
    });
  };
  return {
    start: (at) => {
      const callId = 'workflow_25';
      sessionEvent(writer, LONG_SESSION_ID, at, update, {
        sessionUpdate: 'tool_call',
        toolCallId: callId,
        title: 'workflow',
        rawInput: { name: WORKFLOW_NAME, background: true },
        _meta: { 'x.ai/tool': { ...SPAWN_TOOL, name: 'workflow', label: 'workflow' } },
      });
      sessionEvent(writer, LONG_SESSION_ID, at + 60, update, {
        sessionUpdate: 'tool_call_update',
        toolCallId: callId,
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: 'Workflow started in the background.' } }],
      });
      status(at + 65, 'active', 1);
      carried = { spawn: at + 70, finish: spawn(first, at + 70) };
      return at + 400;
    },
    finish: (turnEnd) => {
      const back = Math.max(carried!.finish ?? 0, turnEnd + 20_000);
      finished(first, carried!.spawn, back);
      status(back + 20, 'active', 5);
      const runs = [second, third].map((child, i) => {
        const at = back + 40 + i * 5;
        return { child, at, finish: spawn(child, at) ?? at };
      });
      for (const run of [...runs].sort((a, b) => a.finish - b.finish)) finished(run.child, run.at, run.finish);
      const stop = Math.max(...runs.map((run) => run.finish)) + 30;
      status(stop, 'complete', 9);
      return stop;
    },
  };
}

/** The long session, its subagents, and the live session registered under `pid`. */
function writeAgentWork(home: string, random: Random, now: number, pid: number): void {
  const { agentHome } = fixturePaths(home);
  const model = MAIN_MODEL;
  const longDir = join(agentHome, 'sessions', encodeURIComponent(LONG_SESSION_CWD), LONG_SESSION_ID);
  mkdirSync(longDir, { recursive: true });
  const writer = newWriter();
  const kids: ChildRegistry = new Map();
  const created = now - 26 * HOUR_MS;
  let at = created;
  const workflow = backgroundWorkflow(writer, kids);
  let wokenAt: number | null = null;
  for (let n = 0; n < LONG_TURNS; n += 1) {
    // A background child still out: the next turn starts before it returns and waits for it.
    let due = 0;
    for (const item of writer.pending) due = Math.max(due, item.at);
    const typedStart =
      due > at ? at + Math.min(1_000, Math.floor((due - at) / 2)) : at + int(random, 60_000, 25 * 60_000);
    // The woken turn starts right after the workflow stops; the draw above still runs, so later turns stay as they were.
    const start = wokenAt ?? typedStart;
    const tools = int(random, 0, 5);
    at = writeTurn(
      writer,
      LONG_SESSION_ID,
      random,
      {
        start,
        prompt: PROMPTS[n % PROMPTS.length]!,
        promptIndex: n,
        model: n >= 30 ? 'Qwen3-8B-4bit' : model,
        tools: n === 6 ? 0 : n === 11 ? Math.max(2, tools) : tools,
        failTool: n % 7 === 3,
        plan: n % 9 === 0,
        children: TURN_CHILDREN.get(n) ?? [],
        open: false,
        cancelled: n === 41,
        perf: 'measured',
        ...(n === 6 ? { parallel: 3 } : {}),
        ...(n === 8 ? { retryAt: 100 } : {}),
        ...(n === 11 ? { noEndTool: true } : {}),
        ...(n === 19 ? { untimed: true } : {}),
        ...(due > at ? { holdUntil: due + 1_500 } : {}),
        ...(n === WORKFLOW_TURN ? { background: workflow.start } : {}),
        ...(wokenAt !== null ? { wake: `workflow-completed-${WORKFLOW_RUN}-9` } : {}),
      },
      kids,
    );
    wokenAt = n === WORKFLOW_TURN ? workflow.finish(at) + 66 : null;
    if (n === 29) {
      sessionEvent(writer, LONG_SESSION_ID, at + 1_000, '_x.ai/session/update', {
        sessionUpdate: 'model_changed',
        model_id: 'Qwen3-8B-4bit',
      });
    }
  }
  writer.untimed = false;
  flushPending(writer, Number.POSITIVE_INFINITY);
  writeFileSync(join(longDir, 'updates.jsonl'), `${writer.lines.join('\n')}\n`);
  writeSessionSummary(
    longDir,
    LONG_SESSION_ID,
    LONG_SESSION_CWD,
    'Port the planner to three modules',
    model,
    created,
    at,
    {
      num_messages: 96,
      num_chat_messages: 96,
      git_root_dir: `${LONG_SESSION_CWD}/`,
      git_remotes: ['git@example.com:dev/proj.git', 'git@example.com:fork/proj.git'],
      head_commit: '3f2a91c0d6b54e1aa2c7d9e8f0b1c2d3e4f5a6b7',
      head_branch: 'planner-split',
      agent_name: 'grok-build-plan',
      reasoning_effort: 'high',
      last_turn_summary:
        'Split the planner into three modules, moved the tests next to them, and checked that the public API did not change. Two flaky cache tests were fixed by dropping the cached body on reconnect.',
    },
  );
  writeChildren(home, longDir, LONG_SESSION_ID, LONG_SESSION_CWD, kids);

  const liveCwd = '/Users/dev/api';
  const liveDir = join(agentHome, 'sessions', encodeURIComponent(liveCwd), LIVE_SESSION_ID);
  mkdirSync(liveDir, { recursive: true });
  const live = newWriter();
  const liveKids: ChildRegistry = new Map();
  const liveStart = now - 40 * 60_000;
  let liveAt = liveStart;
  for (let n = 0; n < 5; n += 1) {
    liveAt = writeTurn(live, LIVE_SESSION_ID, random, {
      start: liveAt + int(random, 60_000, 4 * 60_000),
      prompt: PROMPTS[(n + 2) % PROMPTS.length]!,
      promptIndex: n,
      model,
      tools: int(random, 1, 4),
      failTool: n === 2,
      plan: n === 0,
      children: [],
      open: false,
      cancelled: false,
      perf: 'first-call',
    });
  }
  writeTurn(
    live,
    LIVE_SESSION_ID,
    random,
    {
      start: now - 60_000,
      prompt: 'Add retries to the upload client and run its tests.',
      promptIndex: 5,
      model,
      tools: 1,
      failTool: false,
      plan: true,
      children: LIVE_CHILDREN,
      open: true,
      cancelled: false,
      perf: 'first-call',
    },
    liveKids,
  );
  flushPending(live, Number.POSITIVE_INFINITY);
  writeFileSync(join(liveDir, 'updates.jsonl'), `${live.lines.join('\n')}\n`);
  writeSessionSummary(
    liveDir,
    LIVE_SESSION_ID,
    liveCwd,
    'Add retries to the upload client',
    model,
    liveStart,
    now - 5_000,
    {
      num_messages: 11,
      num_chat_messages: 11,
      git_root_dir: `${liveCwd}/`,
      head_branch: 'upload-retries',
      head_commit: '9b8a7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b',
    },
  );
  writeChildren(home, liveDir, LIVE_SESSION_ID, liveCwd, liveKids);
  writeActiveSessions(home, pid);
}

/** grok's registry of running sessions: the live session, held by `pid` (this process). */
function writeActiveSessions(home: string, pid: number): void {
  const { agentHome } = fixturePaths(home);
  const path = join(agentHome, 'active_sessions.json');
  assertInsideHome(home, path);
  writeJson(path, [{ session_id: LIVE_SESSION_ID, pid }]);
}

/** One state per agent: Claude Code current, Codex on an old command, Grok with a broken block. */
function writeCodingAgents(home: string): void {
  mkdirSync(join(home, '.claude'), { recursive: true });
  writeFileSync(join(home, '.claude', 'CLAUDE.md'), `# Personal notes\n\n${managedBlock(FIXTURE_COMMAND)}\n`);
  mkdirSync(join(home, '.codex'), { recursive: true });
  writeFileSync(join(home, '.codex', 'AGENTS.md'), `${managedBlock(OLD_COMMAND)}\n`);
  mkdirSync(join(home, '.grok'), { recursive: true });
  const broken = managedBlock(FIXTURE_COMMAND).split('\n').slice(0, -1).join('\n');
  writeFileSync(join(home, '.grok', 'AGENTS.md'), `# Team rules\n\n${broken}\n`);
}

/**
 * A catalog entry installed by the app, from before the entry named its tokenizer source: it reads as installed with
 * an update, with no network.
 */
export const CATALOG_INSTALL = {
  repo: 'unsloth/gemma-4-26B-A4B-it-GGUF',
  slug: 'gemma-4-26b-a4b-it-gguf',
  file: 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf',
  sizeGb: 16.9,
} as const;

function writeCatalogInstall(home: string, now: number): void {
  const dir = join(fixturePaths(home).modelsDir, CATALOG_INSTALL.slug);
  mkdirSync(dir, { recursive: true });
  writeJson(join(dir, 'config.json'), { model_type: 'gemma4', max_position_embeddings: 262144 });
  const weights = join(dir, CATALOG_INSTALL.file);
  writeFileSync(weights, '');
  truncateSync(weights, Math.round(CATALOG_INSTALL.sizeGb * GIB));
  writeJson(join(dir, '.mlx-download-complete.json'), {
    repo: CATALOG_INSTALL.repo,
    revision: '5d1c0e2a9b7f4c3d2e1f0a9b8c7d6e5f4a3b2c1d',
    completedAt: new Date(now - 9 * DAY_MS).toISOString(),
    files: ['config.json', CATALOG_INSTALL.file],
    scope: 'full',
  });
}

/**
 * A catalog slug held by a leftover the app did not create (one companion file, no config.json): the card shows
 * "Needs cleanup" with Reinstall.
 */
export const CATALOG_BLOCKED = {
  slug: 'qwen-agentworld-35b-a3b-gguf',
  file: 'imatrix_unsloth.gguf_file',
  sizeBytes: 192_223_904,
} as const;

function writeCatalogBlocked(home: string): void {
  const dir = join(fixturePaths(home).modelsDir, CATALOG_BLOCKED.slug);
  mkdirSync(dir, { recursive: true });
  const file = join(dir, CATALOG_BLOCKED.file);
  writeFileSync(file, '');
  truncateSync(file, CATALOG_BLOCKED.sizeBytes);
}

export interface FixtureSummary {
  traces: number;
  sessions: number;
  live: LiveProcess;
}

/**
 * Write the whole fixture into `home` (created when missing; safe on an empty dir). The live file and the supervisor
 * file name this process, so they read as alive for as long as it runs.
 */
export function writeFixtureHome(home: string, options: FixtureOptions = {}): FixtureSummary {
  const now = options.now ?? Date.now();
  const random = mulberry32(options.seed ?? DEFAULT_SEED);
  if (!disposable(home)) throw new Error(`${home} holds files that are not a dev fixture; pick an empty --home`);
  assertFixtureTree(home);
  mkdirSync(home, { recursive: true });
  writeFileSync(join(home, FIXTURE_MARKER), 'mlx-node dev fixture\n');
  const live: LiveProcess = { pid: process.pid, startedAt: now - 6 * HOUR_MS, engineStates: options.engineStates };
  writeModels(home, random);
  const traces = writeTraces(home, random, now);
  writeRuntime(home, random, now, live);
  writeLiveFile(home, live, now);
  writeSupervisorFile(home, process.pid, process.pid, now);
  writeColdCache(home, random, now);
  const sessions = writeSessions(home, random, now);
  writeAgentWork(home, random, now, process.pid);
  writeCatalogInstall(home, now);
  writeCatalogBlocked(home);
  writeCodingAgents(home);
  return { traces, sessions, live };
}

/**
 * Rewrite the runtime folder of an existing fixture home for `live`: the series, the live file and the supervisor file
 * all name this pid and start, ending at `now`. The runtime ingest drops the rows of the files removed here.
 */
export function refreshRuntime(home: string, live: LiveProcess, now: number = Date.now(), seed = DEFAULT_SEED): void {
  assertFixtureTree(home);
  const { runtimeDir } = fixturePaths(home);
  let names: string[];
  try {
    names = readdirSync(runtimeDir);
  } catch {
    names = [];
  }
  for (const name of names) {
    if (!name.endsWith('.jsonl') && !name.startsWith('live-')) continue;
    const path = join(runtimeDir, name);
    assertInsideHome(home, path);
    rmSync(path, { force: true });
  }
  writeRuntime(home, mulberry32(seed), now, live);
  writeLiveFile(home, live, now);
  writeSupervisorFile(home, live.pid, live.pid, now);
  writeActiveSessions(home, live.pid);
}

/** `--home <dir>` and `--fresh`, parsed by hand. */
export function fixtureArgs(
  argv: readonly string[],
  fallbackHome: string,
): { home: string; fresh: boolean; port: number } {
  let home = fallbackHome;
  let fresh = false;
  let port = 5179;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--fresh') fresh = true;
    else if (arg === '--home' && argv[i + 1] !== undefined) home = resolve(argv[(i += 1)]!);
    else if (arg === '--port' && argv[i + 1] !== undefined) {
      const value = Number(argv[(i += 1)]);
      if (Number.isInteger(value) && value > 0 && value < 65536) port = value;
    }
  }
  assertFixtureHome(home);
  return { home, fresh, port };
}

function contains(parent: string, child: string): boolean {
  const path = relative(parent, child);
  return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith(`..${sep}`));
}

function isSymbolicLink(path: string): boolean {
  try {
    return lstatSync(path).isSymbolicLink();
  } catch {
    return false;
  }
}

/**
 * `path` with every symbolic link resolved: the real path of its nearest existing ancestor plus the missing tail.
 * Throws on a broken link, since a write through it would create the file wherever the link points.
 */
function canonical(path: string): string {
  let head = resolve(path);
  const tail: string[] = [];
  for (;;) {
    try {
      return join(realpathSync(head), ...tail);
    } catch {
      if (isSymbolicLink(head)) throw new Error(`refusing ${path}: ${head} is a broken symbolic link`);
    }
    const parent = dirname(head);
    if (parent === head) return resolve(path);
    tail.unshift(basename(head));
    head = parent;
  }
}

/** Throws for the user's home, the real `~/.mlx-node`, anything inside `~/.mlx-node`, and every ancestor of them. */
export function assertFixtureHome(home: string, userHome: string = homedir()): void {
  const target = resolve(home);
  const realHome = join(userHome, '.mlx-node');
  const canonicalTarget = canonical(target);
  const canonicalUser = canonical(userHome);
  const canonicalReal = canonical(realHome);
  if (
    contains(target, userHome) ||
    contains(realHome, target) ||
    contains(canonicalTarget, canonicalUser) ||
    contains(canonicalReal, canonicalTarget)
  ) {
    throw new Error(`refusing fixture home ${target}: it is or holds ${userHome} or ${realHome}`);
  }
}

function assertInsideHome(home: string, path: string): void {
  const root = canonical(home);
  const target = canonical(path);
  if (!contains(root, target)) throw new Error(`refusing to write ${path}: it resolves to ${target}, outside ${root}`);
}

/**
 * Throws when anything in the existing tree under `home` would carry a write outside its real path: a symbolic link
 * that leaves it or is broken, or a file with a second hard link (truncating it truncates the other name too).
 */
export function assertFixtureTree(home: string): void {
  if (!existsSync(home)) {
    assertInsideHome(dirname(resolve(home)), home);
    return;
  }
  const root = realpathSync(home);
  const pending = [root];
  while (pending.length > 0) {
    const dir = pending.pop()!;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        let target: string;
        try {
          target = realpathSync(path);
        } catch {
          throw new Error(`refusing fixture home ${home}: ${path} is a broken symbolic link`);
        }
        if (!contains(root, target)) {
          throw new Error(`refusing fixture home ${home}: ${path} links outside it to ${target}`);
        }
      } else if (entry.isDirectory()) {
        pending.push(path);
      } else if (lstatSync(path).nlink > 1) {
        throw new Error(`refusing fixture home ${home}: ${path} has another hard link`);
      }
    }
  }
}

function disposable(home: string): boolean {
  return !existsSync(home) || readdirSync(home).length === 0 || existsSync(join(home, FIXTURE_MARKER));
}

/** The default fixture home: `packages/dashboard/.dev-home` (gitignored). */
export const DEFAULT_HOME = join(import.meta.dirname, '../.dev-home');

/** A home is complete when the generator finished: it writes the coding-agent files last. */
export function fixtureExists(home: string): boolean {
  const traces = join(home, 'metrics', 'traces');
  return (
    existsSync(join(home, FIXTURE_MARKER)) &&
    existsSync(join(home, '.codex', 'AGENTS.md')) &&
    existsSync(traces) &&
    readdirSync(traces).length > 0
  );
}

/** Deletes `home` only when it is absent, empty, or carries the fixture marker. */
export function resetHome(home: string, userHome: string = homedir()): void {
  assertFixtureHome(home, userHome);
  if (!disposable(home)) throw new Error(`refusing to delete ${home}: it is not a dev fixture home`);
  rmSync(home, { recursive: true, force: true });
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { home, fresh } = fixtureArgs(process.argv.slice(2), DEFAULT_HOME);
  if (fresh) resetHome(home);
  const summary = writeFixtureHome(home);
  process.stdout.write(`fixture home ${home}: ${summary.traces} turns, ${summary.sessions} cloned sessions\n`);
}
