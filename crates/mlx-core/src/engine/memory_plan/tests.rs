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
    MemoryPlan::evaluate(footprint, Some(pool), speculative)
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
    let plan = MemoryPlan::evaluate(golden, None, Some(lane)).expect("plan");
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
/// copy, the verify cycle at the window, in 128-page extents: 61,433.
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
    assert_eq!(speculative.window.maximum(), 61_433);
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

/// The verify cycle at the window, with no allocator free pool beside it,
/// is the DFlash2 lane's transient at 61,433. Mutation: add
/// `decode_cache_limit` of the cycle in `dflash2_lane_transient` → red
/// (2,545,483,776 at window 32,761).
#[test]
fn transient_has_no_free_pool() {
    let speculative = m3_pro_plan().speculative.expect("speculative lane");
    assert_eq!(speculative.transient_bytes, 1_553_137_664);
    assert_eq!(speculative.copy_bytes, 127_795_200);
    assert_eq!(speculative.window.maximum(), 61_433);
}

/// One 2,048-row chunk: scratch 276,824,064; the draft taps 5 layers of
/// 5,120 bf16 hidden rows (104,857,600); int8 rows add the dequantized
/// prefix and its concatenation (25,165,824); the pool lane's gather holds
/// four K/V copies, two query copies and the fixed overhead (134,217,728).
/// Mutations: drop the taps in `dflash2_lane_transient` → red
/// (276,824,064); taps of the whole window instead of one chunk → red
/// (696,254,464 at 8,192 rows).
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
    assert_eq!(bf16(8_192), 276_824_064 + 104_857_600);
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
/// transient (423,624,704). One byte under it the load fails by one byte.
/// Mutation: `dynamic <= minimum` in `evaluate` → red at the minimum.
#[test]
fn plan_that_cannot_hold_a_lane_and_the_runway_fails() {
    let minimum = m3_pro_plan().speculative.expect("lane").minimum_bytes;
    assert_eq!(minimum, 355_217_408 + 275 * MIB + 136_314_880 + 423_624_704);
    assert_eq!(minimum, 1_203_515_392);
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
        "speculative lane: context 61433 tokens (memory; window 262144, resident 61433)",
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
