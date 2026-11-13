/**
 * Nearest-rank percentiles and fixed-edge histograms, computed in JS on the
 * database worker from one scan per window. The reference definition in SQL is
 * `rn = (n * p + 99) / 100` over `ROW_NUMBER() OVER (ORDER BY v)`;
 * `api.test.ts` checks the two agree.
 */

export interface Quantiles {
  n: number;
  p05: number | null;
  p50: number | null;
  p95: number | null;
}

export interface Histogram {
  /** Samples counted. */
  n: number;
  /** One count per bin; `counts.length === edges.length`. */
  counts: number[];
}

/** Nearest rank on an ascending array: `sorted[ceil(n * p / 100) - 1]`; null when empty. */
export function nearestRank(sorted: Float64Array, p: number): number | null {
  const n = sorted.length;
  if (n === 0) return null;
  const rank = Math.max(1, Math.ceil((n * p) / 100));
  return sorted[Math.min(rank, n) - 1];
}

/** Sorts a copy and returns `{ n, p05, p50, p95 }`. */
export function quantilesOf(values: readonly number[]): Quantiles {
  const sorted = Float64Array.from(values).sort();
  return {
    n: sorted.length,
    p05: nearestRank(sorted, 5),
    p50: nearestRank(sorted, 50),
    p95: nearestRank(sorted, 95),
  };
}

/** Counts per bin; `edges` are ascending lower bounds and the last bin is open-ended. A value below the first edge lands in bin 0. */
export function histogramOf(values: readonly number[], edges: readonly number[]): Histogram {
  const counts = Array.from({ length: edges.length }, () => 0);
  if (edges.length === 0) return { n: 0, counts };
  for (const value of values) {
    let lo = 0;
    let hi = edges.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (edges[mid] <= value) lo = mid;
      else hi = mid - 1;
    }
    counts[lo] += 1;
  }
  return { n: values.length, counts };
}

/** Collects named value columns per group key while a statement is iterated. NULL is skipped per column. */
export class ValueScan {
  private readonly byGroup = new Map<string, Map<string, number[]>>();
  private readonly union = new Map<string, number[]>();

  add(group: string, column: string, value: number | null): void {
    if (value === null || !Number.isFinite(value)) return;
    let columns = this.byGroup.get(group);
    if (columns === undefined) {
      columns = new Map();
      this.byGroup.set(group, columns);
    }
    push(columns, column, value);
    push(this.union, column, value);
  }

  /** Group keys in first-seen order. */
  groups(): string[] {
    return [...this.byGroup.keys()];
  }

  values(group: string, column: string): number[] {
    return this.byGroup.get(group)?.get(column) ?? [];
  }

  /** Union over all groups: the "all models" series. Never derived from per-group results. */
  all(column: string): number[] {
    return this.union.get(column) ?? [];
  }
}

function push(map: Map<string, number[]>, key: string, value: number): void {
  const list = map.get(key);
  if (list === undefined) map.set(key, [value]);
  else list.push(value);
}
