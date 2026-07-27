/**
 * Drift guard for the cold-tier family allowlist.
 *
 * The same decision — "may this family's paged prefix blocks be restored from
 * the SSD cold tier?" — is expressed twice: `COLD_TIER_RESTORE_FAMILIES` in
 * `packages/agent/src/cold-tier.ts` decides whether the agent asks for
 * persistence at load time, and `COLD_RESTORE_FAMILIES` in
 * `crates/mlx-core/src/cold_tier.rs` (exposed as `coldRestoreFamilies()`)
 * is the native side of the same gate. A family present on one side but not
 * the other is either a silently dead opt-in or a restore of state the pool
 * never held, so the two lists must stay identical.
 *
 * The definition moved out of `provider/model-host.ts` (which value-imports
 * `@mlx-node/lm`, and therefore the native addon) so the dashboard viewer
 * process and `mlx agent --help` can read it without linking Metal. That move
 * creates a new drift risk of its own — a re-export could silently diverge from
 * the leaf, or a consumer could grow its own copy — so this file pins the LEAF
 * against native AND pins every re-export to be the very same object.
 */
import { coldRestoreFamilies } from '@mlx-node/core';
import { describe, expect, it } from 'vite-plus/test';

import { COLD_TIER_RESTORE_FAMILIES as viaCatalog, coldTierRestoreFamilyList } from '../src/catalog.js';
import { COLD_TIER_RESTORE_FAMILIES } from '../src/cold-tier.js';
import { COLD_TIER_RESTORE_FAMILIES as viaModelHost } from '../src/provider/model-host.js';

describe('cold-tier restore allowlist', () => {
  it('agrees exactly with the native allowlist', () => {
    expect([...COLD_TIER_RESTORE_FAMILIES].sort()).toEqual([...coldRestoreFamilies()].sort());
  });

  // Identity, not equality: a re-export that ever became a second `new Set([…])`
  // would satisfy a deep compare on the day it was written and drift the day
  // after. The host is what actually gates a load, and the catalog subpath is
  // what the dashboard and the CLI help text read — all three must BE the leaf.
  it('is the same object everywhere it is consumed', () => {
    expect(viaModelHost).toBe(COLD_TIER_RESTORE_FAMILIES);
    expect(viaCatalog).toBe(COLD_TIER_RESTORE_FAMILIES);
  });

  // The display list feeds the `--no-persist-cache` help text and the /api/cache
  // payload the SPA renders, so it must be derived from the set rather than
  // being a fifth hand-maintained copy.
  it('derives its display list from the set', () => {
    expect(coldTierRestoreFamilyList()).toEqual([...COLD_TIER_RESTORE_FAMILIES].sort());
    expect(coldTierRestoreFamilyList()).toEqual([...coldRestoreFamilies()].sort());
  });
});
