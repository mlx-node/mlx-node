use super::*;
use crate::models::qwen3_5::dflash2_decode::tests::qwen38_27b_cfg;
use crate::models::qwen3_5::dflash2_decode::{
    PrefillSdpaKernels, dflash2_lane_transient, pool_lane_transient,
};

/// Splash `dev/tests/engine/status_golden.json` `memory_plan` working set.
const GOLDEN_WORKING_SET: u64 = 25_769_803_776;
/// M3 Pro inputs: a 27 GiB working set with the 27B target and its DFlash2
/// draft resident.
const M3_PRO_WORKING_SET: u64 = 27 * GIB;
const M3_PRO_ACTIVE: u64 = 17_357 * MIB;
const STAGING: u64 = 64 * MIB;
/// Qwen3.8-27B GDN cell (`Qwen3_5Config::recurrent_state_bytes`).
const QWEN38_CELL: u64 = 153_944_064;
/// The 27B draft's KV window buffers: 5 layers x 2 x 8 heads x 128 x 2 B x
/// 2,311 rows.
const QWEN38_DRAFT_RING: u64 = 5 * 2 * 8 * 128 * 2 * 2_311;
/// The 27B draft taps 5 target layers (`target_layer_ids`).
const QWEN38_DRAFT_TAPS: usize = 5;
const QWEN38_DRAFT_BLOCK: usize = 7;
/// Splash `family_values_golden.txt` 27B `lane`.
const SPLASH_QWEN38_LANE: u64 = 356_122_624;
/// `qwen35_runway_blocks` on the 27B: 2,048 tokens of 16-token blocks and
/// one GDN cell in 1 MiB blocks.
const QWEN38_RUNWAY_BLOCKS: u32 = 128 + 147;
const KERNELS: PrefillSdpaKernels = PrefillSdpaKernels {
    d256_full: false,
    portable_d256: true,
};

fn footprint(working_set: u64, active: u64, max_memory: Option<u64>) -> MemoryFootprint {
    MemoryFootprint {
        working_set_bytes: working_set,
        max_memory_bytes: max_memory,
        active_bytes: active,
        other_pool_bytes: 0,
        staging_bytes: STAGING,
        draft_weights_bytes: 0,
        held_back_bytes: 0,
    }
}

fn bf16_layout() -> KvLayout {
    KvLayout::of(16, 4, 256, DType::BFloat16, KvFormat::Bf16)
}

fn int8_layout() -> KvLayout {
    KvLayout::of(16, 4, 256, DType::BFloat16, KvFormat::Int8)
}

/// The 27B plan: the bf16 pool lane and, with `draft`, the int8 DFlash2
/// lane.
fn qwen38_plan(
    footprint: MemoryFootprint,
    explicit_mib: Option<u32>,
    draft: bool,
) -> Result<MemoryPlan, PlanError> {
    let cfg = qwen38_27b_cfg();
    assert_eq!(cfg.recurrent_state_bytes(), QWEN38_CELL);
    let pool_transient = |window| pool_lane_transient(&cfg, DType::BFloat16, KERNELS, window);
    let flat_transient = |window| {
        dflash2_lane_transient(
            &cfg,
            QWEN38_DRAFT_BLOCK,
            KvFormat::Int8,
            QWEN38_DRAFT_TAPS,
            window,
        )
    };
    let pool = PoolInput {
        shape: LaneShape {
            trained: 262_144,
            lane_bytes: QWEN38_CELL,
            layout: bf16_layout(),
            scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
            transient: &pool_transient,
        },
        bytes_per_block: MIB,
        layers: 16,
        runway_blocks: QWEN38_RUNWAY_BLOCKS,
        explicit_mib,
        lane_in_pool: true,
    };
    let speculative = draft.then(|| FlatInput {
        shape: LaneShape {
            trained: 262_144,
            lane_bytes: speculative_lane_bytes(QWEN38_CELL, QWEN38_DRAFT_RING),
            layout: int8_layout(),
            scratch_rows: QWEN38_DRAFT_BLOCK as u64,
            transient: &flat_transient,
        },
        copy_layers: Some(16),
    });
    MemoryPlan::evaluate(footprint, Some(pool), speculative, None)
}

fn m3_pro_plan() -> MemoryPlan {
    qwen38_plan(
        footprint(M3_PRO_WORKING_SET, M3_PRO_ACTIVE, None),
        None,
        true,
    )
    .expect("plan")
}

/// The margin keeps 85% of the working set less 1 GiB (integer split as
/// Splash's); Splash's own `max(1 GiB, 2%)` gives 24,696,061,952 here.
/// Mutation: Splash's margin in `working_set_margin_bytes` → red (margin
/// 1,073,741,824).
#[test]
fn hard_budget_on_the_splash_golden_working_set() {
    assert_eq!(working_set_margin_bytes(GOLDEN_WORKING_SET), 4_939_212_391);
    assert_eq!(hard_budget_bytes(GOLDEN_WORKING_SET, None), 20_830_591_385);
}

/// Mutations: take the flag unconditionally in `hard_budget_bytes` → red
/// (64 GiB); `MemoryBound::of` on `max < hard` instead of `max == hard` →
/// red (a flag equal to the automatic budget reads as this Mac's memory).
#[test]
fn max_memory_only_lowers_the_budget() {
    let automatic = hard_budget_bytes(M3_PRO_WORKING_SET, None);
    assert_eq!(automatic, 23_568_633_036);
    let lowered = hard_budget_bytes(M3_PRO_WORKING_SET, Some(16 * GIB));
    assert_eq!(lowered, 16 * GIB);
    assert_eq!(
        MemoryBound::of(Some(16 * GIB), lowered),
        MemoryBound::MaxMemory
    );
    let ignored = hard_budget_bytes(M3_PRO_WORKING_SET, Some(64 * GIB));
    assert_eq!(ignored, automatic);
    assert_eq!(
        MemoryBound::of(Some(64 * GIB), ignored),
        MemoryBound::DeviceMemory
    );
    let equal = hard_budget_bytes(M3_PRO_WORKING_SET, Some(automatic));
    assert_eq!(equal, automatic);
    assert_eq!(
        MemoryBound::of(Some(automatic), equal),
        MemoryBound::MaxMemory
    );
    assert_eq!(MemoryBound::of(None, automatic), MemoryBound::DeviceMemory);
}

/// Splash's extent for the 27B int8 target KV: the int8 scales need
/// 128-page alignment, and one aligned unit (130 MiB) is both the smallest
/// and the largest extent; the bf16 layout's extents run 32..96 pages.
/// Mutation: align on the data region alone in `extent_alignment_pages` →
/// red (alignment 2 pages).
#[test]
fn splash_extent_pages_on_qwen38_27b() {
    let layout = int8_layout();
    assert_eq!(layout.page_bytes(), 1_064_960);
    assert_eq!(layout.row_bytes(), 33_280);
    assert_eq!(layout.extent_alignment_pages(), 128);
    for pages in [128, 4_368, 100_000] {
        assert_eq!(layout.extent_pages_for(pages), 128, "{pages} pages");
    }
    assert_eq!(layout.extent_pages_for(127), 0);
    assert_eq!(layout.extent_rows_for(4_368 * 1_064_960), 4_096);
    assert_eq!(layout.extent_rows_for(0), 4_096);
    assert_eq!(layout.runway_pages(), 128);
    let bf16 = bf16_layout();
    assert_eq!(bf16.page_bytes(), 2 * MIB);
    assert_eq!(bf16.minimum_extent_pages(), 32);
    assert_eq!(bf16.runway_pages(), 64);
}

/// Splash's golden plan through the lane math: its footprint (4 GiB of
/// weights, the two arenas 734,396,416 + 160,669,696 standing in as a
/// constant transient, lane 350,224,384) at Splash's hard budget gives its
/// fixed bytes, dynamic budget, extent, capacity, resident window and
/// minimum dynamic bytes. Mutations: drop `RUNTIME_OVERHEAD_RESERVE_BYTES`
/// from the fixed bytes → red (5,458,468,864); every page instead of whole
/// extents in `capacity_of` → red (17,231).
#[test]
fn lane_math_reproduces_the_splash_golden_plan() {
    let arenas = 734_396_416 + 160_669_696;
    let transient = |_| arenas;
    let lane = FlatInput {
        shape: LaneShape {
            trained: 262_144,
            lane_bytes: 350_224_384,
            layout: int8_layout(),
            scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
            transient: &transient,
        },
        copy_layers: None,
    };
    let splash_hard = 24_696_061_952;
    let mut golden = footprint(32 * GIB, 4 * GIB, Some(splash_hard));
    golden.staging_bytes = 0;
    let plan = MemoryPlan::evaluate(golden, None, Some(lane), None).expect("plan");
    assert_eq!(plan.hard_budget_bytes, splash_hard);
    assert_eq!(plan.bound, MemoryBound::MaxMemory);
    let lane = plan.speculative.expect("lane");
    assert_eq!(plan.fixed_bytes + lane.transient_bytes, 5_995_339_776);
    assert_eq!(
        plan.dynamic_budget_bytes - lane.transient_bytes,
        18_700_722_176
    );
    assert_eq!(lane.extent_pages, 128);
    assert_eq!(lane.capacity_pages, 17_152);
    assert_eq!(lane.capacity_pages * KV_PAGE_TOKENS, 548_864);
    assert_eq!(lane.window.maximum(), 262_144);
    assert_eq!(lane.window.bound(), "model");
    assert_eq!(lane.minimum_bytes - lane.transient_bytes, 486_539_264);
}

/// mlx-node's 27B lane against Splash's: no 4-bit draft context window
/// (−5,898,240), no 16 KiB alignment of the conv state (−2 x 196,608), a
/// ring of 2,311 rows instead of 2,048 slots (+5,386,240). Mutation: one
/// GDN cell in `speculative_lane_bytes` → red (201,273,344).
#[test]
fn qwen38_27b_lane_bytes_against_splash() {
    let lane = speculative_lane_bytes(QWEN38_CELL, QWEN38_DRAFT_RING);
    assert_eq!(lane, 355_217_408);
    assert_eq!(SPLASH_QWEN38_LANE - lane, 905_216);
}

/// The M3 Pro plan. hard 23,568,633,036, fixed 17,357 MiB with 64 MiB of
/// staging and the 768 MiB reserves, dynamic 4,496,084,172. Pool lane: the
/// lane state lives outside the pool, the pool keeps 1/17 for its last grow
/// and pays the last prefill chunk's scratch with the pool SDPA gather
/// (792,403,968 at the window), so 3,186 blocks and an AR window of 48,601.
/// DFlash2 lane: 2 cells + ring, the 275 runway blocks held, one layer's
/// copy, the verify cycle at the window with its decode free pool, in
/// 128-page extents: 32,761.
/// Mutations: no grow copy in `LaneKv::Pool` → red (3,363 blocks); drop the
/// attention term from `pool_lane_transient` → red (3,648 blocks);
/// scratch rows 0 in `plan_lane` → red (transient 792,461,312); no
/// transient on the flat lane → red (106,489); runway held twice → red
/// (held 576,716,800).
#[test]
fn m3_pro_plan_prices_every_term() {
    let plan = m3_pro_plan();
    assert_eq!(plan.hard_budget_bytes, 23_568_633_036);
    assert_eq!(plan.bound, MemoryBound::DeviceMemory);
    assert_eq!(plan.fixed_bytes, M3_PRO_ACTIVE + STAGING + 768 * MIB);
    assert_eq!(plan.dynamic_budget_bytes, 4_496_084_172);
    let pool = plan.pool.expect("pool");
    assert_eq!(pool.max_blocks, 3_186);
    assert_eq!(pool.max_mib(), 3_186);
    assert_eq!(pool.runway_blocks, QWEN38_RUNWAY_BLOCKS);
    assert_eq!(pool.lane.transient_bytes, 792_403_968);
    assert_eq!(pool.lane.window.maximum(), 48_601);
    assert_eq!(pool.lane.window.bound(), "memory");
    let speculative = plan.speculative.expect("speculative lane");
    assert_eq!(speculative.held_bytes, 275 * MIB);
    assert_eq!(speculative.extent_pages, 128);
    assert_eq!(speculative.window.maximum(), 32_761);
    assert_eq!(speculative.window.bound(), "memory");
    assert!(plan.prices_speculative_lane());
}

/// One request at the pool window fits the dynamic budget with the lane's
/// GDN cell outside the pool (Splash `availableForOneRequestKv = dynamic -
/// laneStateBytes`): the pool at its maximum, its last grow's copy, the cell
/// and the transient at the window. At the lane minimum: 231 blocks, window
/// 1,314, transient 287,776,768; on the M3 Pro: 3,186 / 48,601 /
/// 792,403,968. Mutation: leave `lane_bytes` out of the `LaneKv::Pool`
/// budget → red (271 blocks, 153,935,872 over at the minimum).
#[test]
fn pool_lane_peak_fits_dynamic() {
    let minimum = qwen38_plan(
        footprint(M3_PRO_WORKING_SET, active_for_dynamic(699_203_584), None),
        None,
        false,
    )
    .expect("at the lane minimum");
    for (plan, blocks, window, transient) in [
        (minimum, 231, 1_314, 287_776_768),
        (m3_pro_plan(), 3_186, 48_601, 792_403_968),
    ] {
        let pool = plan.pool.expect("pool");
        let pool_bytes = u64::from(pool.max_blocks) * MIB;
        let peak = pool_bytes + pool_bytes.div_ceil(16) + QWEN38_CELL + pool.lane.transient_bytes;
        assert!(
            peak <= plan.dynamic_budget_bytes,
            "peak {peak} over dynamic {}",
            plan.dynamic_budget_bytes
        );
        assert_eq!(pool.max_blocks, blocks);
        assert_eq!(pool.lane.window.maximum(), window);
        assert_eq!(pool.lane.transient_bytes, transient);
    }
}

/// A planned pool under the config floor still validates (Splash has no
/// pool floor): the 27B pool at the lane minimum is 231 MiB and carries
/// 256; the M3 Pro pool carries its own 3,186. Mutation: `config_mib`
/// returns `max_mib()` → red (231 vs 256).
#[test]
fn pool_below_the_config_floor_validates() {
    let minimum = qwen38_plan(
        footprint(M3_PRO_WORKING_SET, active_for_dynamic(699_203_584), None),
        None,
        false,
    )
    .expect("at the lane minimum");
    let cfg = qwen38_27b_cfg();
    for (plan, max_mib, config_mib) in [(minimum, 231, 256), (m3_pro_plan(), 3_186, 3_186)] {
        let pool = plan.pool.expect("pool");
        assert_eq!(pool.max_mib(), max_mib);
        assert_eq!(pool.config_mib(), config_mib);
        mlx_paged_attn::PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb: pool.config_mib(),
            head_size: cfg.head_dim as u32,
            num_kv_heads: cfg.num_kv_heads as u32,
            num_layers: cfg.full_attention_layer_count() as u32,
            use_fp8_cache: Some(false),
            max_seq_len: Some(cfg.max_position_embeddings as u32),
            max_batch_size: Some(32),
        }
        .validate()
        .expect("the planned pool's config validates");
    }
}

/// The DFlash2 lane's transient is the verify cycle at the window and the
/// free pool MLX keeps of it (twice the cycle, `decode_cache_limit`): MLX
/// collects that pool only at 0.95·WS, above the plan's hard budget. On
/// the M3 Pro: 848,494,592 x 3 at 32,768 keys, window 32,761, copy
/// 32,768 rows x 33,280 / 16. Mutation: drop the `decode_cache_limit`
/// term in `dflash2_cycle_transient_bytes` → red (1,553,137,664 at window
/// 61,433).
#[test]
fn transient_charges_the_decode_free_pool() {
    let speculative = m3_pro_plan().speculative.expect("speculative lane");
    assert_eq!(speculative.transient_bytes, 2_545_483_776);
    assert_eq!(speculative.copy_bytes, 68_157_440);
    assert_eq!(speculative.window.maximum(), 32_761);
}

/// One 2,048-row chunk: scratch 276,824,064; the draft taps 5 layers of
/// 5,120 bf16 hidden rows (104,857,600); int8 rows add the dequantized
/// prefix and its concatenation (25,165,824); the pool lane's gather holds
/// four K/V copies, two query copies and the fixed overhead (134,217,728).
/// At 4,000 rows the chunk still beats the verify cycle with its free pool
/// (177,405,952 below 4,096 keys). Mutations: drop the taps in
/// `dflash2_lane_transient` → red (276,824,064); taps of the whole window
/// instead of one chunk → red (481,624,064 at 4,000 rows).
#[test]
fn chunk_transient_counts_one_chunk_of_taps() {
    let cfg = qwen38_27b_cfg();
    let bf16 = |window| {
        dflash2_lane_transient(
            &cfg,
            QWEN38_DRAFT_BLOCK,
            KvFormat::Bf16,
            QWEN38_DRAFT_TAPS,
            window,
        )
    };
    assert_eq!(bf16(2_048), 276_824_064 + 104_857_600);
    assert_eq!(bf16(4_000), 276_824_064 + 104_857_600);
    assert_eq!(
        dflash2_lane_transient(
            &cfg,
            QWEN38_DRAFT_BLOCK,
            KvFormat::Int8,
            QWEN38_DRAFT_TAPS,
            2_048
        ),
        276_824_064 + 104_857_600 + 25_165_824
    );
    assert_eq!(
        pool_lane_transient(&cfg, DType::BFloat16, KERNELS, 2_048),
        276_824_064 + 134_217_728
    );
}

/// An explicit pool size is checked in blocks against the plan's, printed
/// as whole MiB of them, and resets the pool window to what it holds.
/// Mutations: skip the `PoolAbovePlan` check → red (`Ok`); keep the plan's
/// lane under an explicit ceiling → red (extent 51 instead of 50).
#[test]
fn explicit_pool_above_the_plan_is_refused() {
    let m3 = footprint(M3_PRO_WORKING_SET, M3_PRO_ACTIVE, None);
    let err = qwen38_plan(m3, Some(16_384), true).expect_err("above the plan");
    assert_eq!(
        err,
        PlanError::PoolAbovePlan {
            requested_mib: 16_384,
            plan_mib: 3_186,
            bound: MemoryBound::DeviceMemory,
        }
    );
    let message = err.message("Qwen3.5");
    assert!(message.contains("this Mac's memory"), "{message}");
    assert!(message.contains("at most 3186"), "{message}");

    let lowered = footprint(M3_PRO_WORKING_SET, M3_PRO_ACTIVE, Some(20 * GIB));
    let err = qwen38_plan(lowered, Some(16_384), true).expect_err("above the plan");
    assert!(err.message("Qwen3.5").contains("--max-memory"), "{err:?}");

    let ignored = footprint(M3_PRO_WORKING_SET, M3_PRO_ACTIVE, Some(22 * GIB));
    let err = qwen38_plan(ignored, Some(16_384), true).expect_err("above the plan");
    assert!(
        err.message("Qwen3.5").contains("this Mac's memory"),
        "{err:?}"
    );
    let plan = qwen38_plan(ignored, None, true).expect("plan");
    assert!(
        plan.max_memory_ignored()
            .is_some_and(|line| line.contains("22.0 GiB") && line.contains("can only lower")),
        "{:?}",
        plan.max_memory_ignored()
    );

    let at_plan = qwen38_plan(m3, Some(3_186), true).expect("at the plan");
    assert_eq!(at_plan.pool.expect("pool").max_blocks, 3_186);

    let plan = qwen38_plan(m3, Some(2_048), true).expect("below the plan");
    let pool = plan.pool.expect("pool");
    assert_eq!(pool.max_blocks, 2_048);
    assert_eq!(pool.runway_blocks, QWEN38_RUNWAY_BLOCKS);
    assert_eq!(pool.lane.extent_pages, 50);
    assert_eq!(pool.lane.window.maximum(), 30_393);

    assert_eq!(
        qwen38_plan(m3, Some(100), true).expect_err("below one lane"),
        PlanError::PoolBelowLane {
            requested_mib: 100,
            minimum_mib: 275,
        }
    );
}

/// Active bytes that leave a dynamic budget of `dynamic` on the M3 Pro.
fn active_for_dynamic(dynamic: u64) -> u64 {
    hard_budget_bytes(M3_PRO_WORKING_SET, None) - STAGING - 768 * MIB - dynamic
}

/// The DFlash2 lane's minimum: lane 355,217,408, the 275 MiB runway held,
/// Splash's int8 runway of 128 pages (136,314,880) and a 4,096-token turn's
/// transient (the verify cycle at 4,103 keys with its free pool,
/// 194,183,168 x 3). One byte under it the load fails by one byte.
/// Mutation: `dynamic <= minimum` in `evaluate` → red at the minimum.
#[test]
fn plan_that_cannot_hold_a_lane_and_the_runway_fails() {
    let minimum = m3_pro_plan().speculative.expect("lane").minimum_bytes;
    assert_eq!(minimum, 355_217_408 + 275 * MIB + 136_314_880 + 582_549_504);
    assert_eq!(minimum, 1_362_440_192);
    let under = footprint(M3_PRO_WORKING_SET, active_for_dynamic(minimum - 1), None);
    match qwen38_plan(under, None, true) {
        Err(PlanError::KvPoolDoesNotFit {
            required,
            budget,
            bound,
        }) => {
            assert_eq!(required - budget, 1);
            assert_eq!(budget, 23_568_633_036);
            assert_eq!(bound, MemoryBound::DeviceMemory);
        }
        other => panic!("{other:?}"),
    }
    let at = footprint(M3_PRO_WORKING_SET, active_for_dynamic(minimum), None);
    let plan = qwen38_plan(at, None, true).expect("at the minimum");
    assert_eq!(plan.dynamic_budget_bytes, minimum);
}

/// Every lane the load builds is checked: a draftless load's pool lane
/// needs one cell, Splash's bf16 runway of 64 pages (134,217,728) and a
/// 2,048-token turn's transient (411,041,792). Mutation: check only the
/// speculative lane in `evaluate` → red (`Ok` one byte under).
#[test]
fn draftless_plan_checks_the_pool_lane() {
    let pool = qwen38_plan(
        footprint(M3_PRO_WORKING_SET, M3_PRO_ACTIVE, None),
        None,
        false,
    )
    .expect("plan")
    .pool
    .expect("pool");
    let minimum = pool.lane.minimum_bytes;
    assert_eq!(minimum, QWEN38_CELL + 134_217_728 + 411_041_792);
    let under = footprint(M3_PRO_WORKING_SET, active_for_dynamic(minimum - 1), None);
    assert!(
        matches!(
            qwen38_plan(under, None, false),
            Err(PlanError::KvPoolDoesNotFit { .. })
        ),
        "one byte under"
    );
    let at = footprint(M3_PRO_WORKING_SET, active_for_dynamic(minimum), None);
    assert!(qwen38_plan(at, None, false).is_ok());
}

/// A lane that also holds pool bytes per token (an MTP draft's state)
/// gets the longest window whose KV and those bytes the pool holds.
/// Mutation: ignore `extra` in `PoolPlan::window_with` → red (the lane's
/// window).
#[test]
fn mtp_lane_window_pays_its_held_draft_state() {
    let pool = m3_pro_plan().pool.expect("pool");
    let held = |window: u32| u64::from(window) * 4_096;
    let window = pool.window_with(&held);
    let fits = |window: u32| {
        let kv = u64::from(pool.max_blocks) * MIB - QWEN38_CELL - held(window);
        resident_tokens(&bf16_layout(), kv, SPECULATIVE_SCRATCH_TOKENS) >= u64::from(window)
    };
    assert!(window < pool.lane.window.maximum());
    assert!(fits(window));
    assert!(!fits(window + 1));
    assert_eq!(pool.window_with(&|_| 0), pool.lane.window.maximum());
}

/// Live headroom is the ceiling (hard less the reserves) less what is live,
/// with no process reserve and no 9/10 share. Mutation: `/ 10 * 9` on the
/// allocator path of `live_headroom` → red (944,892,799).
#[test]
fn live_headroom_has_no_reserve_or_nine_tenths() {
    let plan = m3_pro_plan();
    let ceiling = 23_568_633_036 - 768 * MIB;
    assert_eq!(plan.ceiling_bytes(), ceiling);
    let mut snapshot = PagedPrefillMemorySnapshot {
        allocator_active_bytes: Some(18 * GIB),
        allocator_cached_bytes: Some(GIB),
        allocator_limit_bytes: Some(40 * GIB),
        metal_recommended_working_set_bytes: Some(M3_PRO_WORKING_SET),
        metal_current_allocated_bytes: None,
        paged_pool_allocated_bytes: Some(2 * GIB),
    };
    assert_eq!(
        plan.live_headroom(snapshot),
        Some(ceiling - 18 * GIB - 2 * GIB)
    );
    snapshot.metal_current_allocated_bytes = Some(21 * GIB + 512 * MIB);
    assert_eq!(
        plan.live_headroom(snapshot),
        Some(ceiling - (21 * GIB + 512 * MIB - GIB))
    );
    assert_eq!(
        plan.live_headroom(PagedPrefillMemorySnapshot::default()),
        None
    );
}

/// Mutations: skip the pool rule in `audit` → red (`Ok` at 3,310 blocks);
/// skip the hard-budget rule → red (`Ok` one byte over).
#[test]
fn audit_refuses_a_footprint_over_hard() {
    let plan = m3_pro_plan();
    let measured = MeasuredMemory {
        active_bytes: M3_PRO_ACTIVE,
        pool_bytes: 275 * MIB,
        other_pool_bytes: 0,
    };
    assert_eq!(audit(&plan, measured), Ok(()));
    let over_pool = MeasuredMemory {
        pool_bytes: 3_310 * MIB,
        ..measured
    };
    let err = audit(&plan, over_pool).expect_err("pool over its plan");
    assert!(err.starts_with("KV actual bytes"), "{err}");
    let at_hard = 23_568_633_036 - 275 * MIB - STAGING - 768 * MIB;
    let fits = MeasuredMemory {
        active_bytes: at_hard,
        ..measured
    };
    assert_eq!(audit(&plan, fits), Ok(()));
    let over = MeasuredMemory {
        active_bytes: at_hard + 1,
        ..measured
    };
    let err = audit(&plan, over).expect_err("over hard");
    assert!(err.contains("exceeds hard budget"), "{err}");
}

/// On a 96 GiB working set both windows reach the trained 262,144 and the
/// bound is the model; the speculative lane is still the plan's, so an
/// omitted max_tokens stays the rest of the window. Mutation:
/// `prices_speculative_lane` on `bound() == "memory"` → red.
#[test]
fn m5_max_plan_is_bound_by_the_model() {
    let plan = qwen38_plan(footprint(96 * GIB, M3_PRO_ACTIVE, None), None, true).expect("plan");
    let pool = plan.pool.expect("pool");
    assert_eq!(pool.lane.window.maximum(), 262_144);
    assert_eq!(pool.max_blocks, 58_140);
    let speculative = plan.speculative.expect("speculative lane");
    assert_eq!(speculative.window.maximum(), 262_144);
    assert_eq!(speculative.window.bound(), "model");
    assert!(plan.prices_speculative_lane());
}

#[test]
fn working_set_within_the_margin_is_refused() {
    let tiny = footprint(GIB, 0, None);
    assert_eq!(
        qwen38_plan(tiny, None, true),
        Err(PlanError::WorkingSetTooSmall {
            working_set: GIB,
            margin: GIB,
        })
    );
}

/// The breakdown names every term with its bytes, as Splash
/// `EngineMemoryBreakdown::describe` does. Mutation: drop the speculative
/// lane line from `describe` → red.
#[test]
fn describe_lists_every_term() {
    let text = m3_pro_plan().describe();
    for line in [
        "hard budget: 23568633036 bytes (22476.80 MiB) (this Mac's memory)",
        "fixed runtime: 19072548864 bytes",
        "elastic state/KV budget: 4496084172 bytes",
        "KV pool: 3186 blocks of 1048576 bytes (1.00 MiB) at most, 275 at the start",
        "pool lane: context 48601 tokens (memory; window 262144, resident 48601)",
        "speculative lane: context 32761 tokens (memory; window 262144, resident 32761)",
    ] {
        assert!(text.contains(line), "{line}\n{text}");
    }
}

/// Gate: on the M3 Pro plan, after an AR turn grew the runway pool to the
/// plan's 3,186 blocks (3,250 MiB registered with the restore staging), a
/// 61,440-row DFlash2 booking (15 extents of 4,096 rows, 2,044,723,200
/// bytes) is refused under the plan's ceiling (1,155,321,036 live); its
/// shortfall takes 7 pool extents of 128 MiB, and at 2,290 blocks
/// (2,094,845,132 live) it books. Mutations: `live / 10 * 9` in
/// `dflash2_growth_fits` → red (still refused after the cut); round the
/// extent count down in `pool_extents_for_bytes` → red (7 extents).
#[test]
fn dflash2_growth_follows_the_plan_headroom() {
    use crate::models::qwen3_5::dflash2_turn::{
        KvGrowthNeed, dflash2_growth_fits, dflash2_kv_growth_shortfall,
    };
    use crate::transformer::paged_kv_cache_adapter::pool_extents_for_bytes;
    let plan = m3_pro_plan();
    assert_eq!(plan.pool.expect("pool").max_blocks, 3_186);
    assert_eq!(plan.ceiling_bytes(), 22_763_326_668);
    let snapshot = |pool_blocks: u64| PagedPrefillMemorySnapshot {
        allocator_active_bytes: Some(M3_PRO_ACTIVE),
        allocator_cached_bytes: Some(0),
        allocator_limit_bytes: Some(M3_PRO_WORKING_SET),
        metal_recommended_working_set_bytes: Some(M3_PRO_WORKING_SET),
        metal_current_allocated_bytes: None,
        paged_pool_allocated_bytes: Some(pool_blocks * MIB + STAGING),
    };
    let need = KvGrowthNeed {
        from_rows: 0,
        rows: 61_440,
        kv_bytes: 61_440 * 33_280,
        transient_bytes: 0,
    };
    assert_eq!(need.kv_bytes, 2_044_723_200);
    let live = plan.live_headroom(snapshot(3_186)).expect("readable");
    assert_eq!(live, 1_155_321_036);
    assert!(!dflash2_growth_fits(live, &need), "refused");
    let extents = pool_extents_for_bytes(dflash2_kv_growth_shortfall(live, &need), 128 * MIB);
    assert_eq!(extents, 7);
    let blocks = 3_186 - u64::from(extents) * 128;
    assert_eq!(blocks, 2_290);
    let after = plan.live_headroom(snapshot(blocks)).expect("readable");
    assert_eq!(after, 2_094_845_132);
    assert!(dflash2_growth_fits(after, &need), "books");
}

/// Splash `engine_memory_plan_test` `kStudio` (M5 Max 36 GB) working set.
const STUDIO_WORKING_SET: u64 = 30_150_672_384;
/// Splash `kFlashNext`: the images that load and the routed experts.
const FLASH_NEXT_LOADED: u64 = 3_076_735_760;
const FLASH_NEXT_EXPERTS: u64 = 33_976_681_712;
/// The top 10 of 512 experts of all 48 routed layers and one layer more.
const FLASH_NEXT_EXPERT_MINIMUM: u64 = 10 * (FLASH_NEXT_EXPERTS + FLASH_NEXT_EXPERTS / 48) / 512;
/// One Flash Next page: the K/V of 12 layers and their block index.
const FLASH_NEXT_PAGE: u64 = 811_008;
/// The row cache's read staging (`MAX_READ_BYTES * 6`).
const ROW_READ_STAGING: u64 = 6 * 64 * MIB;
/// PLE: 16 ids a token (8 heads of each of 2 n-gram orders), rows of 160
/// bf16; token rows of 2,560 bf16 in a 248,320-token vocabulary.
const PLE_IDS_PER_TOKEN: u64 = 16;
const PLE_ROW_BYTES: u64 = 320;
const PLE_ROWS: u64 = 20_000_000;
const TOKEN_ROWS: u64 = 248_320;
const TOKEN_ROW_BYTES: u64 = 5_120;
const FLASH_NEXT_TRAINED: u32 = 262_144;

fn flash_next_layout() -> KvLayout {
    KvLayout::of(12, 2, 256, DType::BFloat16, KvFormat::Bf16).with_index(128)
}

/// The Flash Next pool lane: the cell, 1,024-row prefill chunks priced at
/// the window, 12 layers of priced pages from the 96-page runway.
fn flash_next_pool<'a>(
    transient: &'a dyn Fn(u32) -> u64,
    lane_bytes: u64,
    lane_in_pool: bool,
) -> PoolInput<'a> {
    let layout = flash_next_layout();
    PoolInput {
        shape: LaneShape {
            trained: FLASH_NEXT_TRAINED,
            lane_bytes,
            layout,
            scratch_rows: SPECULATIVE_SCRATCH_TOKENS,
            transient,
        },
        bytes_per_block: layout.page_bytes(),
        layers: 12,
        runway_blocks: layout.runway_pages() as u32,
        explicit_mib: None,
        lane_in_pool,
    }
}

fn flash_next_rows() -> StreamedPart {
    StreamedPart {
        all: table_share_bytes(PLE_IDS_PER_TOKEN, PLE_ROWS, PLE_ROW_BYTES)
            + table_share_bytes(1, TOKEN_ROWS, TOKEN_ROW_BYTES)
            + ROW_READ_STAGING,
        minimum: ROW_READ_STAGING,
    }
}

/// What a Flash Next load plans: the store before any read, then the engine
/// plan with the store at its cap; the AR window, and with MTP weights the
/// MTP lane's.
struct FlashNextLoad {
    allowance: u64,
    minimum_dynamic: u64,
    store: Result<StorePlan, PlanError>,
    plan: Option<MemoryPlan>,
    ar_window: u32,
    mtp_window: Option<u32>,
}

fn flash_next_load(
    working_set: u64,
    max_memory: Option<u64>,
    mtp: bool,
    lane_in_pool: bool,
) -> FlashNextLoad {
    use crate::models::qwen4_exp::{
        draft_state_bytes, flash_next_config, prefill_scratch, row_state_bytes,
    };
    let c = flash_next_config();
    let transient = |window: u32| prefill_scratch(&c, 1024, window as usize);
    let layout = flash_next_layout();
    let pool = flash_next_pool(&transient, row_state_bytes(&c), lane_in_pool);
    let draft = |window: u32| draft_state_bytes(&c, u64::from(window));
    let ar = |dynamic: u64| pool.plan_at(dynamic).lane.window.maximum();
    let mtp_lane = |dynamic: u64| pool.plan_at(dynamic).window_with(&draft);
    let lanes: Vec<&dyn Fn(u64) -> u32> = if mtp { vec![&ar, &mtp_lane] } else { vec![&ar] };
    let tokens = pool.shape.trained.min(RESIDENT_ALLOWANCE_TOKENS);
    let allowance = allowance_dynamic_bytes(&lanes, tokens);
    let runway_tokens = (layout.runway_pages() * KV_PAGE_TOKENS) as u32;
    let pool_minimum = pool.plan_at(0).lane.minimum_bytes;
    let minimum_dynamic = if mtp {
        pool_minimum + draft(runway_tokens)
    } else {
        pool_minimum
    };
    let hard = hard_budget_bytes(working_set, max_memory);
    let store = plan_streamed_store(&StreamedInputs {
        hard,
        bound: MemoryBound::of(max_memory, hard),
        fixed: FLASH_NEXT_LOADED,
        allowance,
        minimum_dynamic,
        experts: StreamedPart {
            all: FLASH_NEXT_EXPERTS,
            minimum: FLASH_NEXT_EXPERT_MINIMUM,
        },
        rows: flash_next_rows(),
    });
    let plan = store.as_ref().ok().and_then(|store| {
        let footprint = MemoryFootprint {
            working_set_bytes: working_set,
            max_memory_bytes: max_memory,
            active_bytes: FLASH_NEXT_LOADED,
            other_pool_bytes: 0,
            staging_bytes: 0,
            draft_weights_bytes: 0,
            held_back_bytes: store.expert_cap_bytes + store.row_cache_bytes,
        };
        MemoryPlan::evaluate(footprint, Some(pool), None, None).ok()
    });
    let ar_window = plan
        .as_ref()
        .and_then(|plan| plan.pool)
        .map_or(0, |pool| pool.lane.window.maximum());
    let mtp_window = mtp.then(|| {
        plan.as_ref()
            .and_then(|plan| plan.pool)
            .map_or(0, |pool| pool.window_with(&draft))
    });
    FlashNextLoad {
        allowance,
        minimum_dynamic,
        store,
        plan,
        ar_window,
        mtp_window,
    }
}

fn ar_load(working_set: u64, max_memory: Option<u64>) -> FlashNextLoad {
    flash_next_load(working_set, max_memory, false, false)
}

/// The 36 GB input: hard 24,554,329,702; the floors leave the allowance of
/// a 65,536-token window (4,468,093,952), the row share (1,744,830,464) and
/// a streamed store of 14,459,363,158; the engine plan with the store at
/// its cap gives that window in 2,080 pages, from a 96-page runway.
/// Mutation: trained window `.min(32_768)` in the lane → red (allowance
/// 3,031,544,832).
#[test]
fn flash_next_window_for_a_fixed_plan_input() {
    let load = ar_load(STUDIO_WORKING_SET, None);
    assert_eq!(hard_budget_bytes(STUDIO_WORKING_SET, None), 24_554_329_702);
    assert_eq!(load.allowance, 4_468_093_952);
    let store = load.store.expect("store plan");
    assert_eq!(
        store,
        StorePlan {
            whole: false,
            expert_cap_bytes: 14_459_363_158,
            row_cache_bytes: 1_744_830_464,
        }
    );
    let plan = load.plan.expect("engine plan");
    assert_eq!(plan.dynamic_budget_bytes, 4_468_093_952);
    let pool = plan.pool.expect("pool");
    assert_eq!(pool.lane.window.maximum(), 65_536);
    assert_eq!(pool.lane.window.bound(), "memory");
    assert_eq!(pool.max_blocks, 2_080);
    assert_eq!(pool.runway_blocks, 96);
    assert_eq!(pool.bytes_per_block, FLASH_NEXT_PAGE);
}

/// The 64K allowance holds on Splash's 36 GB Mac and on the M3 Pro inputs
/// (store 13,473,666,492). Mutation: the lane minimum instead of the
/// allowance in `plan_streamed_store` → red (store 17,081,392,982, window
/// 0).
#[test]
fn flash_next_36gb_plan_keeps_64k() {
    for (working_set, cap) in [
        (STUDIO_WORKING_SET, 14_459_363_158),
        (M3_PRO_WORKING_SET, 13_473_666_492),
    ] {
        let load = ar_load(working_set, None);
        assert_eq!(
            load.store.expect("store plan").expert_cap_bytes,
            cap,
            "{working_set}"
        );
        assert!(
            load.ar_window >= 65_536,
            "{working_set}: {}",
            load.ar_window
        );
    }
}

/// The streamed experts take all the allowance leaves: one page less of
/// dynamic budget and the window drops below 64K (65,529). Mutation: the
/// allowance plus one 160-page extent in `plan_streamed_store` → red
/// (68,259 a page under).
#[test]
fn streamed_experts_take_all_the_allowance_leaves() {
    use crate::models::qwen4_exp::{flash_next_config, prefill_scratch, row_state_bytes};
    let load = ar_load(STUDIO_WORKING_SET, None);
    let dynamic = load.plan.expect("engine plan").dynamic_budget_bytes;
    let c = flash_next_config();
    let transient = |window: u32| prefill_scratch(&c, 1024, window as usize);
    let pool = flash_next_pool(&transient, row_state_bytes(&c), false);
    assert!(pool.plan_at(dynamic).lane.window.maximum() >= 65_536);
    assert_eq!(
        pool.plan_at(dynamic - FLASH_NEXT_PAGE)
            .lane
            .window
            .maximum(),
        65_529
    );
}

/// The experts load whole at hard 44,071,648,256 (the loaded bytes, the
/// reserves, every expert, the row share and the allowance), and the
/// window never shrinks as `--max-memory` grows from 12 GiB to the 128 GiB
/// working set's hard budget in 16 MiB steps. Mutation: no allowance in
/// the whole-experts test of `plan_streamed_store` → red (whole at
/// 39,611,006,976, with no room left for the window).
#[test]
fn flash_next_window_never_shrinks_as_the_budget_grows() {
    const THRESHOLD: u64 = 44_071_648_256;
    let at = ar_load(128 * GIB, Some(THRESHOLD));
    assert!(at.store.expect("store plan").whole);
    assert_eq!(at.ar_window, 65_536);
    let hard = hard_budget_bytes(128 * GIB, None);
    let (mut previous, mut max_memory) = (0u32, 12 * GIB);
    while max_memory <= hard {
        let load = ar_load(128 * GIB, Some(max_memory));
        let whole = load.store.expect("store plan").whole;
        assert_eq!(whole, max_memory >= THRESHOLD, "{max_memory}");
        assert!(
            load.ar_window >= previous,
            "window {} under {previous} at {max_memory}",
            load.ar_window
        );
        previous = load.ar_window;
        max_memory += 16 * MIB;
    }
    assert_eq!(previous, 262_144);
}

/// The Flash Next page is Splash's: K/V 786,432 and the index plane 24,576
/// (12 layers of 8 blocks of 128 bf16); the 2,048-byte index region needs
/// 32-page alignment, so extents run 96..224 pages and the runway is 96.
/// Mutation: leave the index bytes out of `page_bytes` → red (786,432).
#[test]
fn flash_next_page_prices_the_index_plane() {
    let layout = flash_next_layout();
    assert_eq!(layout.page_bytes(), FLASH_NEXT_PAGE);
    assert_eq!(layout.row_bytes(), 25_344);
    assert_eq!(layout.extent_alignment_pages(), 32);
    assert_eq!(layout.minimum_extent_pages(), 96);
    assert_eq!(layout.maximum_extent_pages(), 224);
    assert_eq!(layout.runway_pages(), 96);
    assert_eq!(layout.extent_pages_for(2_080), 160);
    let bare = KvLayout::of(12, 2, 256, DType::BFloat16, KvFormat::Bf16);
    assert_eq!(bare.page_bytes(), 786_432);
    assert_eq!(bare.extent_alignment_pages(), 2);
    assert_eq!(ar_load(STUDIO_WORKING_SET, None).ar_window, 65_536);
}

/// A cell held beside the pool takes no pool blocks; inside it as well (the
/// Qwen3.5 shape) the same budget holds 61,433. Mutation: subtract the lane
/// from the pool KV whatever `lane_in_pool` says → red (the allowance grows
/// to hold the lane, and the in-pool plan then reads 65,536).
#[test]
fn off_pool_lane_is_not_inside_the_pool() {
    let off = ar_load(STUDIO_WORKING_SET, None);
    assert_eq!(off.ar_window, 65_536);
    let store = off.store.expect("store plan");
    let footprint = off.plan.expect("engine plan").footprint;
    use crate::models::qwen4_exp::{flash_next_config, prefill_scratch, row_state_bytes};
    let c = flash_next_config();
    let transient = |window: u32| prefill_scratch(&c, 1024, window as usize);
    let inside = flash_next_pool(&transient, row_state_bytes(&c), true);
    let plan = MemoryPlan::evaluate(footprint, Some(inside), None, None).expect("plan");
    assert_eq!(
        footprint.held_back_bytes,
        store.expert_cap_bytes + store.row_cache_bytes
    );
    assert_eq!(plan.pool.expect("pool").lane.window.maximum(), 61_433);
}

/// With MTP weights the MTP lane (the cell and the draft state of its
/// window) holds 64K too: allowance 5,746,994,176, store 13,180,462,934,
/// pool 2,982 pages, AR 95,225, MTP 65,655. Mutation: check only the first
/// lane in `allowance_dynamic_bytes` → red (allowance 4,468,093,952).
#[test]
fn mtp_lane_holds_the_allowance() {
    let load = flash_next_load(STUDIO_WORKING_SET, None, true, false);
    assert_eq!(load.allowance, 5_746_994_176);
    assert_eq!(
        load.store.expect("store plan").expert_cap_bytes,
        13_180_462_934
    );
    assert_eq!(
        load.plan.expect("plan").pool.expect("pool").max_blocks,
        2_982
    );
    assert_eq!(load.ar_window, 95_225);
    assert_eq!(load.mtp_window, Some(65_655));
}

/// The row share is the rows a 128K-token prompt reads (Splash
/// `kTableShareTokens`): 2,097,152 PLE rows of 320 B, every token row of
/// the 131,072 it can read, and the read staging. A table smaller than the
/// prompt's ids is held whole. Mutation: `TABLE_SHARE_TOKENS` of 256K → red
/// (1,342,177,280 per table).
#[test]
fn row_cache_share_is_a_128k_prompt() {
    assert_eq!(
        table_share_bytes(PLE_IDS_PER_TOKEN, PLE_ROWS, PLE_ROW_BYTES),
        671_088_640
    );
    assert_eq!(
        table_share_bytes(1, TOKEN_ROWS, TOKEN_ROW_BYTES),
        671_088_640
    );
    assert_eq!(table_share_bytes(1, 100_000, TOKEN_ROW_BYTES), 512_000_000);
    assert_eq!(ROW_READ_STAGING, 402_653_184);
    assert_eq!(flash_next_rows().all, 1_744_830_464);
}

/// On a 128 GiB working set the experts load whole and the window is the
/// trained one (dynamic 76,145,814,323); at the exact whole threshold the
/// experts load whole, a byte under they stream. Mutation: `>` for `>=` in
/// the whole-experts test of `plan_streamed_store` → red at the threshold.
#[test]
fn whole_experts_on_128g() {
    let load = ar_load(128 * GIB, None);
    assert!(load.store.expect("store plan").whole);
    let plan = load.plan.expect("plan");
    assert_eq!(plan.dynamic_budget_bytes, 76_145_814_323);
    let pool = plan.pool.expect("pool");
    assert_eq!(pool.lane.window.maximum(), 262_144);
    assert_eq!(pool.lane.window.bound(), "model");
    let threshold = 44_071_648_256;
    assert!(
        ar_load(128 * GIB, Some(threshold))
            .store
            .expect("at the threshold")
            .whole
    );
    let under = ar_load(128 * GIB, Some(threshold - 1))
        .store
        .expect("a byte under");
    assert!(!under.whole);
    assert_eq!(under.expert_cap_bytes, FLASH_NEXT_EXPERTS - 1);
}

/// One byte under the floors (the loaded bytes, the reserves, the expert
/// and row-cache minimums, the lane minimum) the plan refuses before any
/// read, by that byte, with the bound's advice. Mutation: skip the floors
/// check in `plan_streamed_store` → red (`Ok`).
#[test]
fn floors_over_hard_refuse_before_reading() {
    let probe = ar_load(STUDIO_WORKING_SET, None);
    let floors = FLASH_NEXT_LOADED
        + PIPELINE_RESERVE_BYTES
        + RUNTIME_OVERHEAD_RESERVE_BYTES
        + FLASH_NEXT_EXPERT_MINIMUM
        + ROW_READ_STAGING
        + probe.minimum_dynamic;
    let under = ar_load(STUDIO_WORKING_SET, Some(floors - 1));
    let err = under.store.expect_err("under the floors");
    match err {
        PlanError::FloorsDoNotFit {
            required,
            budget,
            bound,
        } => {
            assert_eq!(required - budget, 1);
            assert_eq!(bound, MemoryBound::MaxMemory);
        }
        ref other => panic!("{other:?}"),
    }
    let message = err.message("Qwen4Exp");
    assert!(
        message.contains("the floors of the memory plan require")
            && message.ends_with("raise --max-memory"),
        "{message}"
    );
    assert!(ar_load(STUDIO_WORKING_SET, Some(floors)).store.is_ok());
}

/// The engine plan counts the expert store at its cap: a GiB less live with
/// a GiB more held back plans the same window. Mutation: leave
/// `held_back_bytes` out of the fixed bytes in `evaluate` → red (dynamic
/// 21,746,029,398).
#[test]
fn held_back_counts_the_store_at_its_cap() {
    use crate::models::qwen4_exp::{flash_next_config, prefill_scratch, row_state_bytes};
    let load = ar_load(STUDIO_WORKING_SET, None);
    let mut footprint = load.plan.expect("plan").footprint;
    footprint.active_bytes -= GIB;
    footprint.held_back_bytes += GIB;
    let c = flash_next_config();
    let transient = |window: u32| prefill_scratch(&c, 1024, window as usize);
    let pool = flash_next_pool(&transient, row_state_bytes(&c), false);
    let plan = MemoryPlan::evaluate(footprint, Some(pool), None, None).expect("plan");
    assert_eq!(plan.dynamic_budget_bytes, 4_468_093_952);
    assert_eq!(plan.pool.expect("pool").lane.window.maximum(), 65_536);
    let text = plan.describe();
    assert!(
        text.contains("expert store and row cache above what is live: 17277935446 bytes"),
        "{text}"
    );
}

/// The int8 pool block is the Splash page in 16-token blocks, and 5,051 MiB
/// rounds to whole extents (`kvCapacityPagesFor`). Mutation M7: the int8
/// scale term of `profile::bytes_per_block` is 0.
#[test]
fn int8_block_bytes_match_the_memory_plan() {
    use mlx_paged_attn::metal::MetalDtype;
    let block = mlx_paged_attn::profile::bytes_per_block(16, 4, 256, 16, MetalDtype::Int8).unwrap();
    assert_eq!(block, int8_layout().page_bytes() * 16 / KV_PAGE_TOKENS);
    assert_eq!(block, 532_480);
    let page = mlx_paged_attn::profile::bytes_per_block(16, 4, 256, 32, MetalDtype::Int8).unwrap();
    assert_eq!(page, int8_layout().page_bytes());
    let (_, pages) = int8_layout().capacity_of(5_051 * MIB);
    assert_eq!(pages, 4_864);
}
