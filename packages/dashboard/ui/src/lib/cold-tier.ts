/**
 * Cold-tier counting rules for the Cache page, kept as pure functions with NO
 * imports so a node-environment test in `packages/dashboard/__test__` can reach
 * them (the SPA has no DOM test environment and the `@/…` alias exists only in
 * the UI's own vite config).
 *
 * The tier holds TWO kinds of object and the server reports them separately:
 * `entryCount` is KV blocks alone, while `ageHistogram`, `totalBytes`, and both
 * mtimes cover blocks AND sidecars — because clear/evict remove both, so an
 * eviction preview that skipped sidecars would understate what it is about to
 * delete. That asymmetry is correct on the server and was the bug in the UI: a
 * "Blocks" tile reading 136 sat beside a "Blocks by age" chart summing to 138,
 * the difference being exactly `sidecarCount`.
 *
 * The fix is not to change either number, it is to stop calling both "blocks".
 * These helpers name the unit at every call site.
 */

/** The two object kinds and their sum, so no caller has to re-derive it. */
export interface ColdObjectCounts {
  /** Canonical `<64-hex>.safetensors` KV blocks. */
  blocks: number;
  /** Canonical `<64-hex>.<group>.safetensors` state sidecars. */
  sidecars: number;
  /** Everything the quota, the age histogram, and clear/evict operate on. */
  objects: number;
}

/** Shape of the fields these helpers read (a structural subset of the API type). */
interface DiskCountsLike {
  entryCount: number;
  sidecarCount: number;
}

/** Shape of one age bucket (a structural subset of the API type). */
interface AgeBucketLike {
  count: number;
  bytes: number;
}

/**
 * Block / sidecar / total object counts for a cold-tier scan.
 *
 * `objects` is the number the age histogram, the quota bytes, and the
 * clear-all button all refer to. Anything that must agree with the histogram
 * MUST use `objects`, never `blocks`.
 */
export function coldObjectCounts(disk: DiskCountsLike | undefined): ColdObjectCounts {
  const blocks = disk?.entryCount ?? 0;
  const sidecars = disk?.sidecarCount ?? 0;
  return { blocks, sidecars, objects: blocks + sidecars };
}

/**
 * Predicted eviction for a day threshold, summed from the age histogram. The
 * evict thresholds (1/7/30) align to the histogram bucket boundaries, so this
 * is exact at scan time (the server removes by mtime at request time). Counts
 * OBJECTS, blocks and sidecars alike, matching what `evictOlderThan` unlinks.
 */
export function evictPreview(buckets: readonly AgeBucketLike[], days: number): { count: number; bytes: number } {
  const startIndex = days >= 30 ? 3 : days >= 7 ? 2 : 1;
  let count = 0;
  let bytes = 0;
  for (let i = startIndex; i < buckets.length; i++) {
    count += buckets[i].count;
    bytes += buckets[i].bytes;
  }
  return { count, bytes };
}

/** The sidecar counters of the health card, read once so the tiles and their alarms agree. */
export interface SidecarReading {
  captures: number;
  installed: number;
  persisted: number;
  written: number;
  noAnchor: number;
  unrooted: number;
  /** State was on disk and prefixes restored, yet nothing was installed. */
  replayed: boolean;
  /** Captures ran and never wrote nor found a sidecar. */
  stranded: boolean;
}

/** Shape of the health counters this reading uses (a structural subset of the API type). */
interface SidecarHealthLike {
  sidecarCaptureReached: number;
  sidecarInstalled: number;
  sidecarAlreadyPersisted: number;
  sidecarEnqueued: number;
  sidecarChainEmpty: number;
  sidecarBoundarySkips: number;
}

/**
 * `replayed`: the regression `installed` exists to expose — prefixes were restored (real hits) and the family's state
 * was on disk, yet nothing was installed, so every restored turn replayed its whole prefix and still produced correct
 * state. `stranded`: the capture ran and never once wrote a sidecar nor found one already there; `persisted` is what
 * separates it from the healthy steady state where the first turn already wrote the chain.
 */
export function sidecarReading(
  health: SidecarHealthLike | undefined,
  scope: { unrootedSidecarCaptures: number } | undefined,
  trendHits: number,
): SidecarReading {
  const captures = health?.sidecarCaptureReached ?? 0;
  const installed = health?.sidecarInstalled ?? 0;
  const persisted = health?.sidecarAlreadyPersisted ?? 0;
  const written = health?.sidecarEnqueued ?? 0;
  return {
    captures,
    installed,
    persisted,
    written,
    noAnchor: (health?.sidecarChainEmpty ?? 0) + (health?.sidecarBoundarySkips ?? 0),
    unrooted: scope?.unrootedSidecarCaptures ?? 0,
    replayed: installed === 0 && persisted > 0 && trendHits > 0,
    stranded: captures > 0 && written === 0 && persisted === 0,
  };
}

/** Shape of one trend day (a structural subset of the API type). */
interface TrendDayLike {
  t: number;
  hits: number;
  misses: number;
}

/** Lookups and bytes of the trend days that start in `starts`: the Cache page and the Overview count the same window. */
export function coldWindowTotals(
  trend: ReadonlyArray<TrendDayLike & { bytesWritten: number; bytesRestored: number }>,
  starts: readonly number[],
): { hits: number; misses: number; bytesWritten: number; bytesRestored: number } {
  const days = new Set(starts);
  const acc = { hits: 0, misses: 0, bytesWritten: 0, bytesRestored: 0 };
  for (const row of trend) {
    if (!days.has(row.t)) continue;
    acc.hits += row.hits;
    acc.misses += row.misses;
    acc.bytesWritten += row.bytesWritten;
    acc.bytesRestored += row.bytesRestored;
  }
  return acc;
}

/** Hit rate in percent per day start; null on a day with no lookup (never a 0 % that was not measured). */
export function hitRateSeries(
  trend: readonly TrendDayLike[],
  starts: readonly number[],
): Array<{ t: number; rate: number | null; lookups: number; hits: number }> {
  const byDay = new Map<number, { hits: number; misses: number }>();
  for (const day of trend) {
    const seen = byDay.get(day.t);
    byDay.set(day.t, { hits: (seen?.hits ?? 0) + day.hits, misses: (seen?.misses ?? 0) + day.misses });
  }
  return starts.map((t) => {
    const day = byDay.get(t);
    const hits = day?.hits ?? 0;
    const lookups = hits + (day?.misses ?? 0);
    return { t, rate: lookups > 0 ? (100 * hits) / lookups : null, lookups, hits };
  });
}
