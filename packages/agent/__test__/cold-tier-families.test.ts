/**
 * Drift guard for the cold-tier family allowlist.
 *
 * The same decision — "may this family's paged prefix blocks be restored from
 * the SSD cold tier?" — is expressed twice: `COLD_TIER_RESTORE_FAMILIES` in
 * `packages/agent/src/provider/model-host.ts` decides whether the agent asks
 * for persistence at load time, and `COLD_RESTORE_FAMILIES` in
 * `crates/mlx-core/src/cold_tier.rs` (exposed as `coldRestoreFamilies()`)
 * is the native side of the same gate. A family present on one side but not
 * the other is either a silently dead opt-in or a restore of state the pool
 * never held, so the two lists must stay identical.
 */
import { coldRestoreFamilies } from '@mlx-node/core';
import { describe, expect, it } from 'vite-plus/test';

import { COLD_TIER_RESTORE_FAMILIES } from '../src/provider/model-host.js';

describe('cold-tier restore allowlist', () => {
  it('agrees exactly with the native allowlist', () => {
    expect([...COLD_TIER_RESTORE_FAMILIES].sort()).toEqual([...coldRestoreFamilies()].sort());
  });
});
