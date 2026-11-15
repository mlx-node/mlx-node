/**
 * The native load, warm-up and turn-phase reports the server reads. Every
 * reader returns null (or leaves a field out) when a report is missing or
 * malformed, never 0 and never an error.
 */

import * as core from '@mlx-node/core';
import type { ChatStreamProgress, LoadProgress, PipelineStats, WarmupReport } from '@mlx-node/core';

export type { LoadProgress, PipelineStats, WarmupReport };

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

/** The `warmup` getter of a loaded model; null for a model without one or a load that ran none. */
export function readWarmupReport(model: unknown): WarmupReport | null {
  try {
    return isObj(model) ? warmupReportOf(model.warmup) : null;
  } catch {
    return null;
  }
}
