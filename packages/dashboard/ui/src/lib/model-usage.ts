/** The join between a checkpoint directory and the names it is served and traced under. Never by folder name alone. */

import type { LocalModel, MetricsSummary, MetricsTimeseries, Quantiles } from './types';

/** True when `key` (a model_key, a live models[].name, a turn's modelId) belongs to this checkpoint directory. */
export function matchesModel(model: Pick<LocalModel, 'name' | 'servedIds'>, key: string): boolean {
  const served = model.servedIds ?? [];
  return served.includes(key) || (served.length === 0 && model.name === key);
}

/** The local checkpoint that serves `key`, or undefined (still loading, or a model outside the models directory). */
export function findLocalModel(models: readonly LocalModel[] | undefined, key: string): LocalModel | undefined {
  return models?.find((model) => matchesModel(model, key));
}

/** The keys this folder is traced under: `servedIds`, or `[name]` when it serves nothing. Never empty. */
export function usageKeys(model: Pick<LocalModel, 'name' | 'servedIds'>): string[] {
  const served = model.servedIds ?? [];
  return served.length > 0 ? [...served] : [model.name];
}

export interface ModelUsage {
  /** The usage key with the most turns (ties: first in `usageKeys`). Quantiles, `ctxWindow` and the color come from it. */
  key: string;
  /** Usage keys with at least one turn. More than one = a folder with several GGUF variants in use. */
  keys: string[];
  turns: number;
  failed: number;
  cancelled: number;
  decodeTps: Quantiles;
  ttftMs: Quantiles;
  ctxFillPct: Quantiles;
  ctxWindow: number | null;
  mtpTurns: number;
  lastTs: number;
}

type ByModel = MetricsSummary['byModel'][number];

/** Keyed by LocalModel.name. A row of `summary.byModel` belongs to the folder that matchesModel() it. No fuzzy match. */
export function joinModelUsage(
  models: readonly LocalModel[],
  summary: MetricsSummary | undefined,
): Map<string, ModelUsage> {
  const joined = new Map<string, ModelUsage>();
  if (summary === undefined) return joined;
  const rows = new Map<string, ByModel>();
  for (const row of summary.byModel) rows.set(row.model, row);
  for (const model of models) {
    const used: ByModel[] = [];
    for (const key of usageKeys(model)) {
      const row = rows.get(key);
      if (row !== undefined && row.totals.turns > 0) used.push(row);
    }
    if (used.length === 0) continue;
    let main = used[0]!;
    for (const row of used) if (row.totals.turns > main.totals.turns) main = row;
    let turns = 0;
    let failed = 0;
    let cancelled = 0;
    let mtpTurns = 0;
    let lastTs = 0;
    for (const row of used) {
      turns += row.totals.turns;
      failed += row.totals.error;
      cancelled += row.totals.cancelled;
      mtpTurns += row.mtpTurns;
      lastTs = Math.max(lastTs, row.lastTs);
    }
    joined.set(model.name, {
      key: main.model,
      keys: used.map((row) => row.model),
      turns,
      failed,
      cancelled,
      decodeTps: main.latency.decodeTps,
      ttftMs: main.latency.ttftMs,
      ctxFillPct: main.latency.ctxFillPct,
      ctxWindow: main.ctxWindow,
      mtpTurns,
      lastTs,
    });
  }
  return joined;
}

/** One value per entry of `starts` (alignedRange('30d').starts): turns summed over `keys`; a day with no bucket is 0 (a real zero). */
export function dailyTurns(
  points: MetricsTimeseries['points'],
  keys: readonly string[],
  starts: readonly number[],
): number[] {
  const wanted = new Set(keys);
  const byStart = new Map<number, number>();
  for (const point of points) {
    if (!wanted.has(point.model)) continue;
    byStart.set(point.t, (byStart.get(point.t) ?? 0) + point.totals.turns);
  }
  return starts.map((start) => byStart.get(start) ?? 0);
}
