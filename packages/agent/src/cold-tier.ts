/**
 * Cold-tier facts shared by the agent (writer), the CLI (help text) and the
 * dashboard (reader), kept in a NATIVE-FREE leaf module.
 *
 * `provider/model-host.ts` value-imports `@mlx-node/lm`, which loads the native
 * addon; the dashboard is a separate viewer process that must never link it
 * ("no Metal init, instant start" — docs/dashboard.md), and `mlx agent --help`
 * must print without loading weights. This module therefore has NO runtime
 * imports beyond `node:` builtins, and reaches those consumers through the
 * existing native-free `@mlx-node/agent/catalog` subpath, which re-exports it.
 * There is deliberately no `./cold-tier` entry in `package.json`: a second
 * published subpath would be a second thing to keep native-free, and `catalog`
 * already carries that guarantee.
 *
 * Everything here is a single source of truth. Do not copy the family list or
 * re-implement the root canonicalization at a call site; import them.
 */

import { realpathSync } from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

import type { ModelType } from '@mlx-node/lm';

/**
 * Model families whose paged prefix blocks can be restored from the SSD cold
 * tier soundly. The single source of truth on the TypeScript side, mirroring
 * `COLD_RESTORE_FAMILIES` in `crates/mlx-core/src/cold_tier.rs`; the two gate
 * the same decision from opposite ends, and
 * `packages/agent/__test__/cold-tier-families.test.ts` asserts they never
 * drift.
 *
 * A family belongs here only when EVERY piece of per-token state its forward
 * pass carries between turns is reconstructible from the tier — either because
 * it all lives inside the paged pool, or because the part that does not is
 * persisted alongside as a cold-tier sidecar and the restore reconciles down to
 * a boundary that sidecar actually backs:
 *
 *  - `qwen3` (dense) sizes its pool over all layers, so the pool holds the
 *    complete KV for the prefix and needs no sidecar.
 *  - `gemma4` sizes its pool over the full-attention layers only, but persists
 *    its out-of-pool sliding-window `RotatingKVCache` state as a
 *    `ColdGroup::SlidingWindow` sidecar, and its `ColdSidecarPolicy` makes the
 *    native restore walk refuse any boundary a validated sidecar does not back.
 *  - `qwen3_5` (dense) sizes its pool over the full-attention layers only, but
 *    persists its out-of-pool GDN recurrent state (conv + recurrent) as a
 *    `ColdGroup::GdnState` sidecar, and its `ColdSidecarPolicy` reconciles the
 *    native restore down to the deepest block-aligned boundary a validated
 *    sidecar backs (a recurrent state is valid only at its exact prefix length).
 *  - `qwen3_5_moe` keeps the SAME GDN recurrent state outside the pool — same
 *    shapes, same dtype, same layer mapping — so it shares the dense family's
 *    sidecar codec and `ColdSidecarPolicy` verbatim. Its restart-parity gate has
 *    since been run on `Qwen3.6-35b-a3b-UD-Q2_K_XL-mlx`: the restore reconciled
 *    onto ladder rung 304 of `[16, 64, 304, 1248]` with `hits=42` and
 *    `corruptions=0`, and the text matched a no-persist baseline.
 *  - `lfm2` / `lfm2_moe` keep short-conv state outside the pool with no
 *    serialization path for it at all, AND drive the uniform native adapter
 *    API whose restore branch is already wired to the tier — asking for
 *    persistence on their behalf would restore an incomplete prefix silently.
 *
 * Widening this set is a correctness decision, never a perf one, and it is
 * authorized by exactly one thing: the family's native restart-parity gate
 * passing on real weights with `hits > 0` and `corruptions == 0`.
 *
 * `mlx agent --no-persist-cache` applies to EVERY family in this set, not to
 * one of them: it is a SINGLE process-wide boolean handed to every load whose
 * family is on the allowlist (`MlxModelHost.runWithResident` →
 * `resolveModelPathFn(entry, { persistPagedCache })`). Families off the
 * allowlist are handed no policy at all — not because the flag spares them, but
 * because `cold_tier::resolve_persist_cold` refuses to persist them under ANY
 * setting. The help text says so in exactly one place, built from
 * {@link coldTierRestoreFamilyList} in `packages/cli/src/commands/agent/index.ts`
 * and asserted by `packages/cli/__test__/agent-cmd.test.ts`.
 */
export const COLD_TIER_RESTORE_FAMILIES: ReadonlySet<string> = new Set<ModelType>([
  'gemma4',
  'qwen3',
  'qwen3_5',
  'qwen3_5_moe',
]);

/** Allowlisted families in a stable, human-facing order (help text, API payload). */
export function coldTierRestoreFamilyList(): string[] {
  return [...COLD_TIER_RESTORE_FAMILIES].sort();
}

/**
 * Expand a leading `~` / `~/…` against the current home directory. A bare `~`
 * prefix that is part of a longer name (`~cache`) is left alone — only the
 * shell's own two spellings are expanded.
 */
/**
 * `realpath(3)` rather than Node's JS walk.
 *
 * On a case-insensitive volume — APFS's default, i.e. essentially every Mac —
 * `MLX_COLD_CACHE_DIR=~/CacheDir` and `~/cachedir` name the SAME directory, but
 * `fs.realpathSync` hands back whichever spelling the caller typed. Two join
 * keys for one cache is the F1 symptom inverted: a populated disk scan sitting
 * beside a flat 0/0 trend. `realpathSync.native` folds each existing component
 * to its on-disk spelling, so both env spellings produce one key.
 */
function realpathNative(path: string): string {
  return realpathSync.native(path);
}

function expandTilde(path: string): string {
  if (path === '~') return homedir();
  if (path.startsWith('~/')) return join(homedir(), path.slice(2));
  return path;
}

/**
 * Canonical identity for a cold-cache root, used as the JOIN KEY between the
 * agent that wrote a trace and the dashboard that reads it back.
 *
 * The two live in DIFFERENT PROCESSES resolving the root from their own
 * environments: the native tier reports `manager.root().display()` (the path as
 * constructed, never realpath'd) while the dashboard builds its own from
 * `MLX_COLD_CACHE_DIR` or the `~/.mlx-node` default. On macOS `/tmp` is a
 * symlink to `/private/tmp`, so two identical-looking spellings canonicalize
 * differently — a raw string compare is a silent-zero trap, not a loud failure.
 * Both sides MUST route through this one function.
 *
 * Resolution order: expand `~`, make absolute, then `realpath` the DEEPEST
 * EXISTING ANCESTOR and re-attach every missing segment lexically. A tier that
 * has never been opened has no root directory — that is the normal state, not
 * an edge case — and when `MLX_COLD_CACHE_DIR` itself has never been created
 * the parent is missing too. Stopping after one level returned a wholly
 * UNcanonicalized path, so under a symlinked ancestor (`/tmp` → `/private/tmp`
 * on macOS) the reader's key diverged from the writer's and rows from the SAME
 * cache were reported as "a different cache directory". Only a path with no
 * resolvable ancestor at all falls back to the lexical absolute form; this is
 * an identity key, not a security boundary, so it fails OPEN to a stable string
 * rather than to `undefined`.
 *
 * An empty input stays empty: the native struct reports `root: ''` while the
 * tier is disabled, and that must never be recorded as a real cache identity.
 */
export function canonicalCacheRoot(root: string): string {
  const expanded = expandTilde(root.trim());
  if (expanded.length === 0) return '';
  const absolute = resolve(expanded);
  const missing: string[] = [];
  let cursor = absolute;
  for (;;) {
    try {
      const resolved = realpathNative(cursor);
      return missing.length === 0 ? resolved : join(resolved, ...missing);
    } catch {
      const parent = dirname(cursor);
      // `dirname` is a fixpoint at the filesystem root, which is the loop's
      // termination condition: nothing above it is left to canonicalize.
      if (parent === cursor) return absolute;
      missing.unshift(basename(cursor));
      cursor = parent;
    }
  }
}
