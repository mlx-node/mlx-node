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
 * Total objects across every age bucket — the histogram's own count of what it
 * is charting. Equal to `coldObjectCounts(disk).objects` for any scan; the two
 * disagreeing is precisely the defect this module exists to prevent.
 */
export function histogramObjectTotal(buckets: readonly AgeBucketLike[]): number {
  let total = 0;
  for (const bucket of buckets) total += bucket.count;
  return total;
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
