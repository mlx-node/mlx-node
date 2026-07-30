import { canonicalCacheRoot, coldTierRestoreFamilyList } from '@mlx-node/agent/catalog';

import { scanColdCache, clearColdCache, coldCacheRoot, evictOlderThan } from '../../cache.js';
import { TRACE_RETENTION_DAYS } from '../../ingest/traces.js';
import type { ApiPaths, ApiRequest, WorkerApiContext } from '../context.js';
import { requireBody, toInt } from '../context.js';
import { ApiError } from '../errors.js';

/**
 * Cold-tier view for ONE cache root: the on-disk scan, the trace-derived trend
 * SCOPED TO THAT SAME ROOT, and an explicit account of everything the scope
 * excluded.
 *
 * The scoping is the whole point. Before it, `disk` was a filesystem scan of one
 * root while `trend` was an unfiltered `SUM()` over every trace row ever
 * ingested, so a dashboard pointed at an EMPTY cache dir reported
 * `exists: false, entryCount: 0` beside a hit rate summed from every cache
 * directory the machine had ever used. Two disjoint notions of "the cache" in
 * one payload.
 *
 * Rows the scope drops are REPORTED, never silently discarded. The buckets
 * below are disjoint AND EXHAUSTIVE over every trace row — that is the point of
 * them, and it is asserted directly (the sum of trend + legacy + otherRoots +
 * unattributed hits equals the table's own `SUM(cold_hits)`). An earlier
 * partition had a hole: `cold_enabled = 1` with a NULL `cold_root` matched no
 * arm at all, so a turn whose tier was open but whose root went unrecorded had
 * its hits vanish — the exact outcome the scope design exists to refuse.
 *  - `legacy` — written before the agent recorded a root (`cold_enabled` NULL).
 *    Counting these into the shown cache IS the original bug; dropping them
 *    without a word makes a real 16K-lookup history vanish next to a root path
 *    the user is staring at. The bucket is bounded and self-clearing: trace
 *    JSONL is pruned at {@link TRACE_RETENTION_DAYS} days and the rows go with
 *    it, so it empties on its own once the recording build has shipped that
 *    long.
 *  - `otherRoots` — genuinely a different cache directory. A different user
 *    story from `legacy`, so a different line.
 *  - `unattributed` — the tier was ON (`cold_enabled` truthy) but the turn
 *    carried no root. The writer no longer produces this (it gates the record
 *    on the CANONICAL root, the same emptiness test `canonicalCacheRoot`
 *    applies), so the bucket should stay empty; it exists so that if one ever
 *    appears the lookups are shown rather than dropped on the floor.
 *  - `disabledTurns` — the tier was off (`--no-persist-cache`, or a family off
 *    the restore allowlist). Their deltas are all zero, so they are counted but
 *    carry no lookups. Overlaps the buckets above by design: it is a TURN
 *    count, not a lookup bucket.
 *  - `unrootedSidecarCaptures` — the one delta a rootless turn can still carry,
 *    and the reason "their deltas are all zero" is true of the BLOCK counters
 *    only. `record_capture_reached()` is the first statement of every family's
 *    sidecar capture, above its `adapter.cold_tier()` guard
 *    (`crates/mlx-core/src/models/gemma4/model.rs:2839` vs `:2843`, and the
 *    same shape in qwen3_5 and qwen3_5_moe), so a hybrid turn with persistence
 *    off — or one whose tier failed to open — counts the capture and then
 *    returns with no tier to name a root from. Summed here rather than folded
 *    into `health`: `health` means "against the root above", and these rows
 *    name no root, so the root scoping cannot be widened to reach them without
 *    also letting `otherRoots` in. Dropping them is what let the page report
 *    "not reached — dense families" over a hybrid capture that ran every turn
 *    and failed before it acquired a tier.
 *
 * `health` carries the counters that are NOT lookups, and therefore cannot be
 * read off the hit rate at all:
 *  - `writeErrors` — writes the queue accepted that never reached disk. The
 *    native writer is fail-open and returns the error to nobody, so a root that
 *    is read-only, full or unmounted otherwise produces a spotless payload:
 *    zero drops, zero corruptions, and a hit rate computed over a cache that
 *    is storing nothing.
 *  - `restoreDeclines` / `restoreSuppressed` — restores that were refused. A
 *    refusal performs no block lookup, so it moves neither `hits` nor `misses`
 *    and lands in `trend` as an absent row, which the UI would otherwise render
 *    as "no lookups recorded" — "nothing ran", when the truth is the opposite.
 *  - `sidecar*` — the family state that lives OUTSIDE the paged pool.
 *    `sidecarInstalled` is the only read-side one and the only counter in this
 *    payload that nothing else can stand in for: every
 *    `install_*_cold_sidecar` early-return falls through to a full O(prefix)
 *    replay that produces CORRECT state, so "restored and INSTALLED" and
 *    "restored and silently re-derived" agree on hits, cached tokens,
 *    corruptions and the emitted text alike.
 *
 * The two families reduce differently and must not be swapped. `*Total` fields
 * are per-process cumulative counters reduced with MAX (see below); declines
 * have no total and are SUMMED, because unlike a corruption a decline has a
 * legitimate non-zero steady state — every prompt's first turn declines — so a
 * "did this ever happen" latch would be pinned on from the first minute.
 */
export function handleCacheGet(ctx: WorkerApiContext): unknown {
  const requestedRoot = coldCacheRoot(ctx.cacheRoot);
  const disk = scanColdCache(requestedRoot);
  // Both sides of the join canonicalize through the SAME helper the agent used
  // when it wrote the row. A raw string compare would be a silent-zero trap:
  // the writer and the reader are different processes resolving the root from
  // their own environments, and on macOS `/tmp` → `/private/tmp` alone makes
  // two identical-looking spellings unequal.
  const scopeRoot = canonicalCacheRoot(requestedRoot);
  const trend = ctx.dash.sqlite
    .prepare(
      `SELECT date(ts / 1000, 'unixepoch') AS day,
              COALESCE(SUM(cold_hits), 0) AS hits, COALESCE(SUM(cold_misses), 0) AS misses,
              COALESCE(SUM(cold_bytes_written), 0) AS bytesWritten,
              COALESCE(SUM(cold_bytes_restored), 0) AS bytesRestored
       FROM traces WHERE ts > 0 AND cold_root = ? GROUP BY day ORDER BY day`,
    )
    .all(scopeRoot)
    .map((row) => ({
      day: String(row.day),
      hits: toInt(row.hits),
      misses: toInt(row.misses),
      bytesWritten: toInt(row.bytesWritten),
      bytesRestored: toInt(row.bytesRestored),
    }));

  // `CASE WHEN` rather than `FILTER` — portable across whatever SQLite the
  // bundled `node:sqlite` links.
  const excluded = ctx.dash.sqlite
    .prepare(
      `SELECT
         COUNT(CASE WHEN cold_root IS NULL AND cold_enabled IS NULL THEN 1 END) AS legacyTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NULL THEN cold_hits END), 0) AS legacyHits,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NULL THEN cold_misses END), 0) AS legacyMisses,
         COUNT(CASE WHEN cold_root IS NOT NULL AND cold_root <> ? THEN 1 END) AS otherTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NOT NULL AND cold_root <> ? THEN cold_hits END), 0) AS otherHits,
         COALESCE(SUM(CASE WHEN cold_root IS NOT NULL AND cold_root <> ? THEN cold_misses END), 0) AS otherMisses,
         COUNT(CASE WHEN cold_root IS NULL AND cold_enabled IS NOT NULL AND cold_enabled <> 0 THEN 1 END)
           AS unattributedTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NOT NULL AND cold_enabled <> 0
                           THEN cold_hits END), 0) AS unattributedHits,
         COALESCE(SUM(CASE WHEN cold_root IS NULL AND cold_enabled IS NOT NULL AND cold_enabled <> 0
                           THEN cold_misses END), 0) AS unattributedMisses,
         COUNT(CASE WHEN cold_enabled = 0 THEN 1 END) AS disabledTurns,
         COALESCE(SUM(CASE WHEN cold_root IS NULL THEN cold_sidecar_capture_reached END), 0)
           AS unrootedSidecarCaptures
       FROM traces WHERE ts > 0`,
    )
    .get(scopeRoot, scopeRoot, scopeRoot);

  // Cumulative counters use MAX, not SUM: each process's total restarts at 0
  // when its tier opens, so summing across processes double-counts. We only
  // need 0-vs-non-zero ("did this EVER happen against this cache"), and
  // MAX > 0 answers exactly that — including for a turn that aborted before it
  // could record a delta.
  const health = ctx.dash.sqlite
    .prepare(
      `SELECT COALESCE(SUM(cold_enqueued), 0) AS enqueued,
              COALESCE(SUM(cold_queue_drops), 0) AS queueDrops,
              COALESCE(SUM(cold_evictions), 0) AS evictions,
              COALESCE(SUM(cold_corruptions), 0) AS corruptions,
              COALESCE(MAX(cold_corruptions_total), 0) AS corruptionsTotal,
              COALESCE(MAX(cold_queue_drops_total), 0) AS queueDropsTotal,
              COALESCE(SUM(cold_write_errors), 0) AS writeErrors,
              COALESCE(MAX(cold_write_errors_total), 0) AS writeErrorsTotal,
              COALESCE(SUM(cold_restore_declines), 0) AS restoreDeclines,
              COALESCE(SUM(cold_sidecar_restore_suppressed), 0) AS restoreSuppressed,
              COALESCE(SUM(cold_sidecar_capture_reached), 0) AS sidecarCaptureReached,
              COALESCE(SUM(cold_sidecar_chain_empty), 0) AS sidecarChainEmpty,
              COALESCE(SUM(cold_sidecar_boundary_skips), 0) AS sidecarBoundarySkips,
              COALESCE(SUM(cold_sidecar_already_persisted), 0) AS sidecarAlreadyPersisted,
              COALESCE(SUM(cold_sidecar_enqueued), 0) AS sidecarEnqueued,
              COALESCE(SUM(cold_sidecar_queue_drops), 0) AS sidecarQueueDrops,
              COALESCE(SUM(cold_sidecar_installed), 0) AS sidecarInstalled
       FROM traces WHERE ts > 0 AND cold_root = ?`,
    )
    .get(scopeRoot);

  return {
    disk,
    trend,
    scope: {
      root: scopeRoot,
      trendWindowDays: TRACE_RETENTION_DAYS,
      legacy: {
        turns: toInt(excluded?.legacyTurns),
        hits: toInt(excluded?.legacyHits),
        misses: toInt(excluded?.legacyMisses),
      },
      otherRoots: {
        turns: toInt(excluded?.otherTurns),
        hits: toInt(excluded?.otherHits),
        misses: toInt(excluded?.otherMisses),
      },
      unattributed: {
        turns: toInt(excluded?.unattributedTurns),
        hits: toInt(excluded?.unattributedHits),
        misses: toInt(excluded?.unattributedMisses),
      },
      disabledTurns: toInt(excluded?.disabledTurns),
      // The one counter above that keeps moving after the root is gone, so the
      // one the buckets above cannot stand in for: they carry hits and misses,
      // and a tier-less turn has neither. Reported here rather than merged into
      // `health` because `health` means "this cache root", and these rows name
      // no root at all — merging them would put numbers from a run that never
      // touched the shown cache under a heading that promises it did.
      unrootedSidecarCaptures: toInt(excluded?.unrootedSidecarCaptures),
    },
    health: {
      enqueued: toInt(health?.enqueued),
      queueDrops: toInt(health?.queueDrops),
      evictions: toInt(health?.evictions),
      corruptions: toInt(health?.corruptions),
      corruptionsTotal: toInt(health?.corruptionsTotal),
      queueDropsTotal: toInt(health?.queueDropsTotal),
      writeErrors: toInt(health?.writeErrors),
      writeErrorsTotal: toInt(health?.writeErrorsTotal),
      // SUM, not MAX, and deliberately so: unlike corruptions and write
      // errors these two have a legitimate non-zero steady state (any prompt's
      // first turn declines), so a "did it ever happen" latch would be pinned
      // on from the first minute and carry no information.
      restoreDeclines: toInt(health?.restoreDeclines),
      restoreSuppressed: toInt(health?.restoreSuppressed),
      // The other seven `ColdSidecarStats` counters. Every one is a per-turn
      // DELTA, so every one SUMS — none has a cumulative `*_total` twin, and
      // MAX would not stand in for one: over deltas it reports the busiest
      // single turn rather than "did this ever happen".
      //
      // `sidecarEnqueued` / `sidecarQueueDrops` are the sidecar SHARE of the
      // object-scoped `enqueued` / `queueDrops` above (a sidecar admission
      // bumps both counters), so a consumer reads them as a subset and never
      // adds the pairs. `restoreSuppressed` is the eighth of this family and
      // keeps its unprefixed name because it is read beside `restoreDeclines`,
      // the other way reuse is refused.
      sidecarCaptureReached: toInt(health?.sidecarCaptureReached),
      sidecarChainEmpty: toInt(health?.sidecarChainEmpty),
      sidecarBoundarySkips: toInt(health?.sidecarBoundarySkips),
      sidecarAlreadyPersisted: toInt(health?.sidecarAlreadyPersisted),
      sidecarEnqueued: toInt(health?.sidecarEnqueued),
      sidecarQueueDrops: toInt(health?.sidecarQueueDrops),
      sidecarInstalled: toInt(health?.sidecarInstalled),
    },
    // Sent over the wire rather than bundled into the SPA: the browser build
    // cannot import `@mlx-node/agent` (it transitively loads the native addon),
    // and a hardcoded copy in the UI would sit outside the drift guard in
    // `packages/agent/__test__/cold-tier-families.test.ts`.
    restoreFamilies: coldTierRestoreFamilyList(),
  };
}

export function handleCacheDelete(ctx: ApiPaths, req: ApiRequest): unknown {
  const body = requireBody(req);
  const parsed = body as { all?: unknown; olderThanDays?: unknown } | null;
  const all = parsed?.all;
  const olderThanDays = parsed?.olderThanDays;
  const root = ctx.cacheRoot;
  let result: { removed: number; freedBytes: number };
  // Clear-all needs an explicit `{"all": true}` discriminator; selective
  // eviction needs a positive finite `olderThanDays`. Anything else (absent,
  // string, zero, negative, misspelled) is a 400 — never a silent whole-cache
  // wipe from a typing slip.
  if (all === true) {
    result = root !== undefined ? clearColdCache(root) : clearColdCache();
  } else if (typeof olderThanDays === 'number' && Number.isFinite(olderThanDays) && olderThanDays > 0) {
    result = root !== undefined ? evictOlderThan(olderThanDays, root) : evictOlderThan(olderThanDays);
  } else {
    throw ApiError.badRequest(
      'Body must be {"all":true} to clear all, or {"olderThanDays":<positive number>} to evict',
    );
  }
  return result;
}
