/**
 * The native load, warm-up and turn-phase reports the server reads. Every
 * reader returns null (or leaves a field out) when a report is missing or
 * malformed, never 0 and never an error.
 */

import * as core from '@mlx-node/core';
import type {
  ChatStreamProgress,
  LoadProgress,
  PipelineStats,
  Qwen4ExpResidencyInfo,
  WarmupReport,
} from '@mlx-node/core';

import {
  EXPERT_RESIDENCIES,
  RESIDENCY_DELTA_FIELDS,
  type ExpertResidency,
  type ResidencyDeltaField,
} from './metrics-trace.js';

export type { LoadProgress, PipelineStats, WarmupReport };

/** The residency facts of a load, fixed when the load chose where the experts live. */
export const LOAD_RESIDENCY_FIELDS = [
  'bankBudgetBytes',
  'liveNonReclaimableBytes',
  'anonymousNeedBytes',
  'hostWiredBytes',
  'hostCompressorBytes',
  'hostAnonymousBytes',
  'hostPurgeableBytes',
  'selfMlxBytes',
] as const satisfies readonly (keyof Qwen4ExpResidencyInfo)[];

export type LoadResidencyField = (typeof LOAD_RESIDENCY_FIELDS)[number];

/** A validated `residencyInfo()`: the closed enum, the slot counts, and every number the records read. */
export interface ResidencySnapshot {
  residency: ExpertResidency;
  /** Slots per layer, or the expert count on a direct store. */
  expertSlots: number;
  expertCount: number;
  load: Record<LoadResidencyField, number>;
  counters: Record<ResidencyDeltaField, number>;
}

/** What the heartbeat says about a model's experts. */
export interface ExpertsLive {
  inMemory: number;
  total: number;
  streaming: boolean;
}

/** Every phase a native turn reports. */
export type TurnPhase = ChatStreamProgress['phase'];

/**
 * A native progress snapshot. `waitMs` runs from creation to pickup;
 * `slotMs` and `prepareMs` are absent before pickup. An older addon sends
 * none of the three, so `waitMs` is optional here.
 */
export interface TurnProgressSnapshot extends Omit<ChatStreamProgress, 'waitMs'> {
  waitMs?: number;
}

type Obj = Record<string, unknown>;

function isObj(value: unknown): value is Obj {
  return typeof value === 'object' && value !== null;
}

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null;
}

function ms(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

/** A validated load progress, or null. */
export function loadProgressOf(value: unknown): LoadProgress | null {
  if (!isObj(value)) return null;
  const phase = value.phase === 'weights' || value.phase === 'kernels' ? value.phase : null;
  const steps = count(value.steps);
  const stepsDone = count(value.stepsDone);
  if (phase === null || steps === null || stepsDone === null) return null;
  return { phase, steps, stepsDone: Math.min(stepsDone, steps) };
}

/** A validated pipeline counter, or null. */
export function pipelineStatsOf(value: unknown): PipelineStats | null {
  if (!isObj(value)) return null;
  const createdLoading = count(value.createdLoading);
  const createdServing = count(value.createdServing);
  const loadingMs = ms(value.loadingMs);
  const servingMs = ms(value.servingMs);
  if (createdLoading === null || createdServing === null || loadingMs === null || servingMs === null) return null;
  return { createdLoading, createdServing, loadingMs, servingMs };
}

/** A validated warm-up report, or null. */
export function warmupReportOf(value: unknown): WarmupReport | null {
  if (!isObj(value)) return null;
  const report = {
    ms: ms(value.ms),
    steps: count(value.steps),
    skipped: count(value.skipped),
    reset: count(value.reset),
    pipelines: count(value.pipelines),
    compileMs: ms(value.compileMs),
    peakBytes: count(value.peakBytes),
  };
  for (const field of Object.values(report)) if (field === null) return null;
  return report as WarmupReport;
}

/** The native load progress of `path`; null when no load reports. */
export function readLoadProgress(path: string): LoadProgress | null {
  try {
    return loadProgressOf(core.loadProgress(path));
  } catch {
    return null;
  }
}

/** The process-wide pipeline counter; null while it is off or dead. */
export function readPipelineStats(): PipelineStats | null {
  try {
    return pipelineStatsOf(core.pipelineStats());
  } catch {
    return null;
  }
}

function nonNegative(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

/**
 * A validated residency report, or null. Every field must be present and valid: a model whose
 * `residencyInfo()` has another shape (DeepSeek V4.1's) gives null.
 */
export function residencySnapshotOf(value: unknown): ResidencySnapshot | null {
  if (!isObj(value)) return null;
  let residency: ExpertResidency | null = null;
  for (const kind of EXPERT_RESIDENCIES) if (kind === value.residency) residency = kind;
  const expertSlots = count(value.partialExpertSlots);
  const expertCount = count(value.expertCount);
  if (residency === null || expertSlots === null || expertCount === null || expertCount === 0) return null;
  const load = {} as Record<LoadResidencyField, number>;
  for (const key of LOAD_RESIDENCY_FIELDS) {
    const field = nonNegative(value[key]);
    if (field === null) return null;
    load[key] = field;
  }
  const counters = {} as Record<ResidencyDeltaField, number>;
  for (const key of RESIDENCY_DELTA_FIELDS) {
    const field = nonNegative(value[key]);
    if (field === null) return null;
    counters[key] = field;
  }
  return { residency, expertSlots, expertCount, load, counters };
}

/** The `residencyInfo()` of a loaded model; null for a model without one or with another shape. */
export function readResidency(model: unknown): ResidencySnapshot | null {
  try {
    if (!isObj(model)) return null;
    const read = model.residencyInfo;
    return typeof read === 'function' ? residencySnapshotOf(read.call(model)) : null;
  } catch {
    return null;
  }
}

/** The heartbeat's experts line of a residency report. */
export function expertsLiveOf(snapshot: ResidencySnapshot): ExpertsLive {
  return {
    inMemory: Math.min(snapshot.expertSlots, snapshot.expertCount),
    total: snapshot.expertCount,
    streaming: snapshot.residency !== 'direct',
  };
}

/** The `warmup` getter of a loaded model; null for a model without one or a load that ran none. */
export function readWarmupReport(model: unknown): WarmupReport | null {
  try {
    return isObj(model) ? warmupReportOf(model.warmup) : null;
  } catch {
    return null;
  }
}
