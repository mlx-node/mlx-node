/**
 * Auto-enable block-paged KV cache for Qwen3.5 dense + MoE under
 * `mlx launch claude`.
 *
 * Why this helper exists: Qwen3.5's per-model `use_block_paged_cache`
 * default is `false` because Qwen3.5 dense + MoE have a compiled C++
 * forward path (`mlx::core::compile`) that is bypassed when the paged
 * adapter is active. For standalone `mlx serve` the compiled path is
 * the right default — bs=1 single-conversation decode wants the
 * compiled fast path.
 *
 * For `mlx launch claude` the workload is different: Claude Code fires
 * parallel internal one-shots (session-init, title-summarizer,
 * count_tokens) sharing the main conversation's system prompt, plus
 * multi-turn conversations with long contexts (50K+ tokens observed).
 * The block-paged cache's content-addressed prefix-hash table
 * refcounts SYS blocks across all those parallel sessions, while the
 * single-warm-slot fallback evicts them. The prefill skip on a turn-2
 * cache hit (50K tokens) easily dwarfs any decode-throughput
 * regression from bypassing the compiled forward — turning paged ON
 * is the right call for this command.
 *
 * The Rust `Qwen3_5Model::load` and `Qwen3_5MoeModel::load` NAPI
 * methods take only a path — adding a config-override parameter is a
 * cross-cutting NAPI change. Instead this helper mirrors what
 * `crates/mlx-core/tests/qwen3_5_paged_vs_flat_parity.rs` does: clone
 * the source model directory under a stable temp path with weight
 * symlinks (cheap — no disk-OOM) and a freshly-written `config.json`
 * that pins `use_block_paged_cache: true`. The cloned path is what
 * `loadModel` sees.
 *
 * Pool-size bump: the Rust default for `paged_cache_memory_mb` is
 * 2048 (qwen3_5_moe/model.rs:500, qwen3_5/model.rs:569). On the
 * Qwen3.6-35b-a3b workload that gives ~4500 blocks ≈ 72k tokens of
 * total KV across all in-flight conversations — which a single
 * Claude-Code subagent fanning Reads across a test directory can
 * blow past in one turn (observed: 386 KB / ~95k-token tool_result
 * batch from an Explore agent). The launcher floors this to
 * `DEFAULT_PAGED_CACHE_MB` (16 GB, ~50k blocks ≈ 800k tokens for
 * Qwen3.6 MoE bf16) so concurrent conversations + long contexts
 * fit. Override via `MLX_PAGED_CACHE_MEMORY_MB`. A source config
 * already setting a value ≥ the floor is preserved as-is.
 *
 * Only Qwen3.5 dense + MoE are intercepted. Qwen3, LFM2, and Gemma4
 * are already default-on for `use_block_paged_cache`; pass-through
 * those untouched. Any other model type (VLMs, OCR, document
 * pipeline) is also pass-through — they have no paged adapter.
 */

import { mkdir, readdir, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PAGED_OVERRIDE_ROOT = join(tmpdir(), 'mlx-launch-claude-paged-overrides');

/** Model types whose default `use_block_paged_cache` is `false` and that we want to flip ON for `mlx launch claude`. */
const QWEN35_MODEL_TYPES = new Set(['qwen3_5', 'qwen3_5_moe']);

/** Floor for `paged_cache_memory_mb` (in MB). Sized for Qwen3.6 MoE on a 128 GB Mac with ~30 GB of weights resident. */
const DEFAULT_PAGED_CACHE_MB = 16384;

function resolvePagedCacheFloorMb(): number {
  const raw = process.env.MLX_PAGED_CACHE_MEMORY_MB;
  if (raw == null || raw === '') return DEFAULT_PAGED_CACHE_MB;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_PAGED_CACHE_MB;
  return parsed;
}

/** Tracks every override directory we created so the launcher can clean up on shutdown. */
const createdOverrides = new Set<string>();

/**
 * If `modelPath` is a Qwen3.5 dense / MoE checkpoint, returns a
 * cloned-and-patched directory whose `config.json` pins
 * `use_block_paged_cache: true`. Weight files in the clone are
 * symlinks to the originals so disk usage is negligible.
 *
 * For all other model types (or any error reading the config) returns
 * the original `modelPath` untouched so `loadModel` proceeds normally.
 *
 * The clone is keyed by the absolute source path so that re-loading
 * the same model returns the same override directory — no churn on
 * `/model` swap.
 */
export async function resolvePagedAwareModelPath(modelPath: string): Promise<string> {
  let modelType: string | null;
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw) as { model_type?: string };
    modelType = config.model_type ?? null;
  } catch {
    return modelPath;
  }

  if (modelType == null || !QWEN35_MODEL_TYPES.has(modelType)) {
    return modelPath;
  }

  const pagedCacheFloorMb = resolvePagedCacheFloorMb();

  // Bail out early only if the source already satisfies BOTH knobs we'd otherwise patch.
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw) as {
      use_block_paged_cache?: boolean;
      paged_cache_memory_mb?: number;
    };
    const flagOk = config.use_block_paged_cache === true;
    const memOk = typeof config.paged_cache_memory_mb === 'number' && config.paged_cache_memory_mb >= pagedCacheFloorMb;
    if (flagOk && memOk) return modelPath;
  } catch {
    return modelPath;
  }

  const overrideDir = join(PAGED_OVERRIDE_ROOT, hashPath(modelPath));
  await mkdir(overrideDir, { recursive: true });

  const sourceEntries = await readdir(modelPath);
  for (const name of sourceEntries) {
    const src = join(modelPath, name);
    const dst = join(overrideDir, name);
    if (name === 'config.json') {
      const raw = await readFile(src, 'utf-8');
      const config = JSON.parse(raw) as Record<string, unknown>;
      config.use_block_paged_cache = true;
      const sourceMem = config.paged_cache_memory_mb;
      const sourceMemNum = typeof sourceMem === 'number' && sourceMem > 0 ? sourceMem : 0;
      config.paged_cache_memory_mb = Math.max(sourceMemNum, pagedCacheFloorMb);
      await writeFile(dst, JSON.stringify(config, null, 2), 'utf-8');
      continue;
    }

    let isFile = true;
    try {
      const st = await stat(src);
      isFile = st.isFile();
    } catch {
      continue;
    }
    if (!isFile) continue;

    try {
      await symlink(src, dst);
    } catch (err) {
      // EEXIST — clone was reused; assume the existing symlink is correct.
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err;
    }
  }

  createdOverrides.add(overrideDir);
  return overrideDir;
}

/**
 * Best-effort cleanup of every override directory created during this
 * process. Safe to call from `process.on('exit')` / SIGINT handlers;
 * does not throw on missing dirs.
 */
export async function cleanupPagedOverrides(): Promise<void> {
  const tasks: Promise<void>[] = [];
  for (const dir of createdOverrides) {
    tasks.push(rm(dir, { recursive: true, force: true }).catch(() => undefined));
  }
  createdOverrides.clear();
  await Promise.all(tasks);
}

function hashPath(absPath: string): string {
  // FNV-1a is plenty for "is this path the same as a previous launch's path"
  // collision-resistance; we don't need cryptographic strength here. Avoiding
  // a `crypto` import keeps this file dependency-free.
  let h = 0x811c9dc5;
  for (let i = 0; i < absPath.length; i++) {
    h ^= absPath.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}
