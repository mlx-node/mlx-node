//! Launch plan of the in-place paged prefill / verify attention
//! (`paged_prefill_attention.metal`): Splash `ops/PagedAttention.cpp`
//! `sharedSplits` / `mostSplits` / `prefillSplits` / `prefillWorkspace`,
//! `kv::verifyAttentionSplits` and `KernelCalibration::of`. Pure functions.

/// Tokens of one page the tile reads per step (Splash `SplashKvPageTokens`).
pub const PAGE_TOKENS: u32 = 32;
/// Query rows of a tile (Splash `SPLASH_PREFILL_ATTENTION_TILE_ROWS`).
pub const TILE_ROWS: u32 = 8;
/// Rows of one prefill chunk (Splash `SPLASH_PREFILL_TOKEN_BUDGET`).
pub const PREFILL_MAX_ROWS: u32 = 2048;
/// Splits of a prefill launch (Splash `SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS`).
pub const PREFILL_MAX_SPLITS: u32 = 32;
/// Rows of a verify launch (Splash `SPLASH_TARGET_VERIFY_ROWS`).
pub const VERIFY_ROWS: u32 = 8;
/// Splash `kv::kVerifySplits`, `kVerifyMaximumSplits`, `kVerifyPagesPerSplit`.
pub const VERIFY_MIN_SPLITS: u32 = 32;
pub const VERIFY_MAX_SPLITS: u32 = 128;
pub const VERIFY_PAGES_PER_SPLIT: u32 = 16;

/// A prefill chunk or a verify block: the split rule and the workspace bound
/// differ (Splash `prefillSplits` vs `verifyAttentionSplits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Prefill,
    Verify,
}

impl Pass {
    /// The launch's `max_splits` (the kernel's contract and reduce bound).
    pub fn max_splits(self) -> u32 {
        match self {
            Pass::Prefill => PREFILL_MAX_SPLITS,
            Pass::Verify => VERIFY_MAX_SPLITS,
        }
    }
}

pub fn tiles(rows: u32) -> u32 {
    rows.div_ceil(TILE_ROWS)
}

/// Splash `sharedSplits`: splits of a launch whose tiles share the GPU.
pub fn shared_splits(tiles: u32) -> u32 {
    (PREFILL_MAX_SPLITS / tiles.max(1)).clamp(1, PREFILL_MAX_SPLITS)
}

/// Splash `mostSplits`: at most a 2048-row chunk's row-splits in flight.
pub fn most_splits(tiles: u32) -> u32 {
    (self::tiles(PREFILL_MAX_ROWS) / tiles.max(1)).clamp(shared_splits(tiles), PREFILL_MAX_SPLITS)
}

/// Splash `prefillSplits` (no window): one split per `pages_per_split`
/// visible pages, floor, between `shared_splits` and `most_splits`.
pub fn prefill_splits(rows: u32, history: u32, pages_per_split: u32) -> u32 {
    let t = tiles(rows);
    let pages = (u64::from(history) + u64::from(rows)).div_ceil(u64::from(PAGE_TOKENS));
    let scaled = pages / u64::from(pages_per_split.max(1));
    scaled.clamp(u64::from(shared_splits(t)), u64::from(most_splits(t))) as u32
}

/// Splash `kv::verifyAttentionSplits` (no window).
pub fn verify_splits(committed: u32) -> u32 {
    let pages = (u64::from(committed) + u64::from(VERIFY_ROWS)).div_ceil(u64::from(PAGE_TOKENS));
    let scaled = pages.div_ceil(u64::from(VERIFY_PAGES_PER_SPLIT));
    scaled.clamp(u64::from(VERIFY_MIN_SPLITS), u64::from(VERIFY_MAX_SPLITS)) as u32
}

/// The launch's split count.
pub fn splits(pass: Pass, rows: u32, history: u32, pages_per_split: u32) -> u32 {
    match pass {
        Pass::Prefill => prefill_splits(rows, history, pages_per_split),
        Pass::Verify => verify_splits(history),
    }
}

/// Splash `SPLASH_ATTENTION_TILE_PARTS`.
pub fn tile_parts(group: u32) -> u32 {
    match group {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    }
}

/// Splash `prefillWorkspace`: the most tile-splits any chunk of at most
/// `rows` rows launches, for every history and every device's pages per
/// split.
pub fn prefill_workspace_slots(rows: u32) -> u32 {
    let tiles = tiles(rows.min(PREFILL_MAX_ROWS));
    (1..=tiles).map(|t| t * most_splits(t)).max().unwrap_or(0)
}

/// f32 elements of the workspace one launch writes: partials (8 rows x
/// query heads x head dim per tile-split) then statistics (2 per row).
pub fn workspace_floats(rows: u32, splits: u32, query_heads: u32, head_dim: u32) -> u64 {
    u64::from(tiles(rows))
        * u64::from(splits)
        * u64::from(TILE_ROWS)
        * u64::from(query_heads)
        * (u64::from(head_dim) + 2)
}

/// f32 elements of a workspace that holds every launch of `pass`
/// (Splash `prefillWorkspace` of a 2048-row chunk; `verifyWorkspaceBound`
/// of one lane).
pub fn workspace_bound_floats(pass: Pass, query_heads: u32, head_dim: u32) -> u64 {
    let slots = match pass {
        Pass::Prefill => prefill_workspace_slots(PREFILL_MAX_ROWS),
        Pass::Verify => VERIFY_MAX_SPLITS,
    };
    u64::from(slots) * u64::from(TILE_ROWS) * u64::from(query_heads) * (u64::from(head_dim) + 2)
}

/// Bytes the attention of one launch of up to `rows` rows holds: the
/// workspace bound, the KV-head-major query staging (rows rounded up to a
/// page), the bf16 output and the chunk's bf16 K and V rows the store reads
/// (Splash `RuntimeArenas` `AttentionPartials`, `AttentionStatistics`,
/// `FullQueries`, `FullAttention`, `ChunkKeys`, `ChunkValues`). No history
/// term. Rows are one lane's: Splash pads `kRaggedAttentionRows` for four.
pub fn attention_bytes(
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    pass: Pass,
) -> u64 {
    let rows = match pass {
        Pass::Prefill => rows.min(PREFILL_MAX_ROWS),
        Pass::Verify => VERIFY_ROWS,
    };
    let row_bytes = u64::from(query_heads) * u64::from(head_dim) * 2;
    let kv_row_bytes = u64::from(kv_heads) * u64::from(head_dim) * 2;
    workspace_bound_floats(pass, query_heads, head_dim) * 4
        + u64::from(rows.next_multiple_of(PAGE_TOKENS)) * row_bytes
        + u64::from(rows) * row_bytes
        + 2 * u64::from(rows) * kv_row_bytes
}

/// Splash `DeviceCapabilities::kAssumedGpuCores`: the core count planned for
/// a device that does not report one.
pub const ASSUMED_GPU_CORES: u32 = 32;
/// Splash `kApple9SmallGpuCores`.
pub const APPLE9_SMALL_GPU_CORES: u32 = 10;
/// `mlx_gpu_architecture_gen` of the first Apple10 GPU (M5).
pub const APPLE10_GEN: i32 = 17;

/// Splash `KernelCalibration::of` `prefillPagesPerSplit`: 32 on Apple10 and
/// on Apple9 above 10 cores, 64 on Apple9 with at most 10. `reported_cores`
/// 0 means the device does not report its cores (Splash
/// `plannedGpuCores`). Apple7 and Apple8 (M1, M2) take the Apple9 row, as
/// Splash `gpuClass` maps every family below 10: unmeasured there.
pub fn pages_per_split(gpu_gen: i32, reported_cores: u32) -> u32 {
    let cores = if reported_cores == 0 {
        ASSUMED_GPU_CORES
    } else {
        reported_cores
    };
    if gpu_gen >= APPLE10_GEN || cores > APPLE9_SMALL_GPU_CORES {
        32
    } else {
        64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splash `paged_attention_plan_test.mm` `checkPlans` and the chunk points
    /// the kernel tests cover.
    #[test]
    fn split_rule_matches_splash() {
        for (rows, history, want) in [
            (2048, 0, 1),
            (2048, 30720, 1),
            (547, 70000, 3),
            (9, 70000, 32),
            (64, 65536, 32),
            (256, 65536, 8),
            (1, 0, 32),
            (9, 0, 16),
            (64, 8192, 8),
            (64, 0, 4),
            (256, 1500, 1),
            (64, 4096, 4),
            (256, 4096, 4),
            (256, 16384, 8),
            (256, 131072, 8),
            (512, 65536, 4),
            (300, 65536, 6),
            (2048, 131072, 1),
        ] {
            assert_eq!(
                prefill_splits(rows, history, 32),
                want,
                "({rows}, {history})"
            );
        }
        assert_eq!(prefill_splits(64, 8192, 64), 4);
        assert_eq!(verify_splits(0), 32);
        assert_eq!(verify_splits(16384), 33);
        assert_eq!(verify_splits(70000), 128);
        assert_eq!(splits(Pass::Verify, 5, 16384, 32), 33);
        for (group, parts) in [(4, 1), (6, 1), (8, 1), (16, 2), (32, 4)] {
            assert_eq!(tile_parts(group), parts, "group {group}");
        }
    }

    /// Every chunk of every history fits the workspace bound, and its splits
    /// lie between shared and most (Splash `checkPlans`).
    #[test]
    fn every_plan_fits_the_workspace_bound() {
        for rows in 1..=PREFILL_MAX_ROWS {
            let t = tiles(rows);
            let bound = prefill_workspace_slots(rows);
            for history in [0, 4000, 8192, 32768, 131072, 1_000_000] {
                for pps in [32, 64] {
                    let s = prefill_splits(rows, history, pps);
                    assert!(
                        (shared_splits(t)..=most_splits(t)).contains(&s),
                        "({rows}, {history}, {pps}) -> {s}"
                    );
                    assert!(
                        t * s <= bound,
                        "({rows}, {history}, {pps}): {t} x {s} > {bound}"
                    );
                }
            }
        }
        assert_eq!(prefill_workspace_slots(2048), 256);
        assert_eq!(prefill_workspace_slots(547), 256);
        assert_eq!(prefill_workspace_slots(8), 32);
    }

    /// 27B (24 query heads, 4 KV heads, head dim 256): the Splash workspace
    /// bound plus the query staging, the output and the K / V rows.
    #[test]
    fn attention_bytes_is_the_splash_workspace() {
        assert_eq!(
            attention_bytes(2048, 24, 4, 256, Pass::Prefill),
            109_445_120
        );
        assert_eq!(
            attention_bytes(1 << 20, 24, 4, 256, Pass::Prefill),
            109_445_120
        );
        assert_eq!(attention_bytes(8, 24, 4, 256, Pass::Verify), 25_886_720);
        assert_eq!(
            workspace_floats(2048, 1, 24, 256),
            workspace_bound_floats(Pass::Prefill, 24, 256)
        );
    }

    #[test]
    fn pages_per_split_follows_kernel_calibration() {
        assert_eq!(pages_per_split(17, 10), 32);
        assert_eq!(pages_per_split(17, 0), 32);
        assert_eq!(pages_per_split(16, 10), 64);
        assert_eq!(pages_per_split(15, 8), 64);
        assert_eq!(pages_per_split(16, 16), 32);
        assert_eq!(pages_per_split(15, 0), 32);
        assert_eq!(pages_per_split(13, 8), 64);
        assert_eq!(pages_per_split(14, 10), 64);
        assert_eq!(pages_per_split(14, 38), 32);
    }
}
