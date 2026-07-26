/**
 * Drift guard for the GDN checkpoint store's owner capacity.
 *
 * Supply and demand for the same resource are declared in two languages.
 * `GDN_PREFIX_CHECKPOINT_LIMIT` (`crates/mlx-core/src/models/qwen3_5/
 * gdn_checkpoint_store.rs`, published as `gdnPrefixCheckpointLimit()`) is the
 * supply. `MAX_CONCURRENCY + 1` — the root session plus one native cache owner
 * per concurrent task loop — is the demand.
 *
 * Past the cap every owner holds a single entry and the store is still over it,
 * so each publish takes somebody's last checkpoint and the loser re-forwards its
 * whole cached prefix through the GDN layers on its next turn. `retention_sim`
 * measures that cliff: 0 blind turns in 40 at five owners, 28 of 40 at six, with
 * 84% of every cached prefix replayed.
 *
 * No Rust gate sees a TypeScript-only edit. `retention_sim`'s `SIZED_FLEET_OWNERS`
 * is a deliberate literal 5 (a bound that moved with the cap could never reach a
 * cliff that sits one owner past it), so nothing on that side reads
 * `MAX_CONCURRENCY`. This test is the only gate on that direction.
 *
 * Deliberately NOT `MAX_PARALLEL_TASKS` (8): a parallel batch mints up to nine
 * distinct owner ids over its life, which already exceeds the cap. Retired
 * children are the usual eviction victims because the last-resort search takes
 * the oldest non-root entry, but it does not distinguish live from retired.
 * Asserting the submission limit here would be red today and is a separate
 * design question, not drift.
 *
 * What this does not catch: raising both sides in lockstep. That satisfies this
 * assertion while leaving `SIZED_FLEET_OWNERS` behind, so `retention_sim` would
 * stop sweeping the real fleet. Closing that needs the sim's demand constant
 * published too, which moves a Rust test's bound and deserves its own run.
 */
import { gdnPrefixCheckpointLimit } from '@mlx-node/core';
import { describe, expect, it } from 'vite-plus/test';

import { MAX_CONCURRENCY } from '../src/extensions/subagent.js';

describe('GDN checkpoint store capacity', () => {
  it('holds one checkpoint for every owner the subagent fleet runs at once', () => {
    // One assertion on purpose. A stale addon or an uncommitted `index.cjs`
    // resolves the import to `undefined` and the call below throws
    // `TypeError: gdnPrefixCheckpointLimit is not a function` — verified, not
    // assumed — so guarding the return value would be an assertion that cannot
    // fail rather than a second gate.
    expect(MAX_CONCURRENCY + 1).toBeLessThanOrEqual(gdnPrefixCheckpointLimit());
  });
});
