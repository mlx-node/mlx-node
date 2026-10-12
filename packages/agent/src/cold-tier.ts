/**
 * Cold-tier facts live in the native-free `@mlx-node/lm/cold-tier` leaf so
 * the serve process (`@mlx-node/server` → `@mlx-node/lm`) can reach them
 * without depending on the agent. This shim keeps every existing
 * `./cold-tier.js` import — including `catalog.ts`'s native-free re-export
 * surface — pointing at the single implementation.
 */

export { COLD_TIER_RESTORE_FAMILIES, canonicalCacheRoot, coldTierRestoreFamilyList } from '@mlx-node/lm/cold-tier';
