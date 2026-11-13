/**
 * The native load, warm-up and turn-phase reports the server reads, typed
 * here so an addon that predates them reads as "not known": every reader
 * returns null (or leaves a field out) when the export or the field is
 * missing, never 0 and never an error.
 */

import * as core from '@mlx-node/core';
import type { ChatStreamProgress } from '@mlx-node/core';

/** Every phase a native turn reports. */
export type TurnPhase = 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | 'done';

/**
 * A native progress snapshot. `waitMs` runs from creation to pickup;
 * `slotMs` and `prepareMs` are absent before pickup. An older addon sends
 * none of the three, so all three are optional here whatever the addon's
 * declaration says.
 */
export interface TurnProgressSnapshot extends Omit<ChatStreamProgress, 'phase' | 'waitMs' | 'slotMs' | 'prepareMs'> {
  phase: TurnPhase;
  waitMs?: number;
  slotMs?: number;
  prepareMs?: number;
}

/** One model load in flight. `weights`: `steps` and `stepsDone` are 0. */
export interface LoadProgress {
  phase: 'weights' | 'kernels';
  steps: number;
  stepsDone: number;
}

/** The load-time warm-up of a paged Qwen3.5 load. */
export interface WarmupReport {
  ms: number;
  steps: number;
  skipped: number;
  reset: number;
  pipelines: number;
  compileMs: number;
  peakBytes: number;
}

/** Process-wide, cumulative GPU pipeline creation counts. */
export interface PipelineStats {
  createdLoading: number;
  createdServing: number;
  loadingMs: number;
  servingMs: number;
}

interface NativeReports {
  loadProgress?: (path: string) => unknown;
  pipelineStats?: () => unknown;
}

const NATIVE = core as unknown as NativeReports;

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

/** The native load progress of `path`; null when no load reports or the addon predates the export. */
export function readLoadProgress(path: string): LoadProgress | null {
  try {
    const read = NATIVE.loadProgress;
    return typeof read === 'function' ? loadProgressOf(read(path)) : null;
  } catch {
    return null;
  }
}

/** The process-wide pipeline counter; null while it is off, dead, or the addon predates it. */
export function readPipelineStats(): PipelineStats | null {
  try {
    const read = NATIVE.pipelineStats;
    return typeof read === 'function' ? pipelineStatsOf(read()) : null;
  } catch {
    return null;
  }
}

/** The `warmup` getter of a loaded model; null for a model without one or a load that ran none. */
export function readWarmupReport(model: unknown): WarmupReport | null {
  try {
    return isObj(model) ? warmupReportOf(model.warmup) : null;
  } catch {
    return null;
  }
}
