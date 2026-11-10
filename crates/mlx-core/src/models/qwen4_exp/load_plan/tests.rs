use super::*;
use crate::models::qwen4_exp::memory::{
    self, PoolCharge, draft_state_bytes, flash_next_config, row_state_bytes, scheduled_state_budget,
};

const MIB: u64 = 1 << 20;
/// Splash `engine_memory_plan_test` `kStudio` (M5 Max 36 GB) working set.
const STUDIO_WORKING_SET: u64 = 30_150_672_384;
/// Splash `kFlashNext`: the images that load and the routed experts.
const FLASH_NEXT_LOADED: u64 = 3_076_735_760;
const FLASH_NEXT_EXPERTS: u64 = 33_976_681_712;
const PHYSICAL_BLOCK: u64 = 786_432;

/// The 36 GB input as a load's descriptors give it.
fn inventory() -> LoadInventory {
    let c = flash_next_config();
    LoadInventory {
        store_fixed_bytes: FLASH_NEXT_LOADED,
        vision_bytes: 0,
        experts: StreamedPart {
            all: FLASH_NEXT_EXPERTS,
            minimum: 10 * (FLASH_NEXT_EXPERTS + FLASH_NEXT_EXPERTS / 48) / 512,
        },
        rows: row_cache_share(&c, 6 * 64 * MIB),
    }
}

fn live(active: u64) -> LiveMemory {
    LiveMemory {
        working_set_bytes: STUDIO_WORKING_SET,
        active_bytes: active,
        registered_pool_bytes: 0,
    }
}

/// The store plan, then the engine plan of the loaded weights with the
/// store at its cap.
fn load(has_mtp: bool) -> (StoreDecision, MemoryPlan) {
    let c = flash_next_config();
    let decision = plan_store("Qwen4Exp", &c, &inventory(), live(0), None, has_mtp).unwrap();
    let held_back = decision.store.expert_cap_bytes + decision.store.row_cache_bytes;
    let plan = plan_engine("Qwen4Exp", &c, live(FLASH_NEXT_LOADED), held_back, None).unwrap();
    (decision, plan)
}

/// The load's wiring gives the plan of the fixed input: page 811,008, the
/// cell beside the pool, a 1,744,830,464-byte row share, a streamed store of
/// 14,459,363,158 and a 65,536-token window in 2,080 pages from a 96-page
/// runway. Mutation: `lane_in_pool: true` in `pool_input` → red (store
/// 14,333,724,502).
#[test]
fn flash_next_load_plans_the_fixed_input() {
    let c = flash_next_config();
    assert_eq!(flash_next_layout(&c).page_bytes(), 811_008);
    assert_eq!(index_block_bytes(&c), 24_576);
    assert_eq!(inventory().rows.all, 1_744_830_464);
    let (decision, plan) = load(false);
    assert_eq!(
        decision.store,
        StorePlan {
            whole: false,
            expert_cap_bytes: 14_459_363_158,
            row_cache_bytes: 1_744_830_464,
        }
    );
    assert_eq!(decision.allowance, 4_468_093_952);
    let pool = plan.pool.unwrap();
    assert_eq!(pool.lane.window.maximum(), 65_536);
    assert_eq!((pool.runway_blocks, pool.max_blocks), (96, 2_080));
}

/// Blocks a pool grown from `runway` by `extent`-block extents holds once
/// `pages` are in use.
fn grown(runway: u64, extent: u64, pages: u64) -> u64 {
    runway + pages.saturating_sub(runway).div_ceil(extent) * extent
}

/// On the MTP plan (AR 95,225, MTP 65,655) one MTP owner at its window
/// (848,225,280 B with the 255-token lookahead) fits the state budget beside
/// its 2,060 pages in a pool grown to 2,148 blocks, and two do not; four AR
/// rows fit at the runway. Mutation: leave the transient out of
/// `scheduled_state_budget` → red (two owners, 1,696,450,560 B, fit a
/// 3,866,340,352 B budget).
#[test]
fn flash_next_scheduled_state_budget_holds_one_long_mtp_owner() {
    let c = flash_next_config();
    let (_, plan) = load(true);
    let pool = plan.pool.unwrap();
    let window = pool.lane.window.maximum();
    assert_eq!(window, 95_225);
    let mtp = mtp_window(&c, &plan).unwrap();
    assert_eq!(mtp, 65_655);
    let row = row_state_bytes(&c);
    let owner = row + draft_state_bytes(&c, u64::from(mtp) + 255);
    assert_eq!(owner, 848_225_280);
    let pages = (u64::from(mtp) + 255).div_ceil(32);
    assert_eq!(pages, 2_060);
    let extent = u64::from(
        crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(PHYSICAL_BLOCK),
    );
    assert_eq!(extent, 171);
    let total = grown(96, extent, pages);
    assert_eq!(total, 2_148);
    let charge = |total: u64, allocated: u64| PoolCharge {
        total_blocks: total,
        max_blocks: u64::from(pool.max_blocks),
        allocated_blocks: allocated,
        physical_block_bytes: PHYSICAL_BLOCK,
        index_block_bytes: index_block_bytes(&c),
        growth_copy_bytes: total * PHYSICAL_BLOCK / 12,
    };
    let transient = memory::prefill_scratch(&c, super::super::MAX_PREFILL_CHUNK, window as usize);
    let budget =
        scheduled_state_budget(plan.dynamic_budget_bytes, transient, &charge(total, pages));
    assert!(owner <= budget, "one owner {owner} B, budget {budget} B");
    assert!(
        2 * owner > budget,
        "two owners {} B, budget {budget} B",
        2 * owner
    );
    let runway = scheduled_state_budget(plan.dynamic_budget_bytes, transient, &charge(96, 0));
    assert!(
        4 * row <= runway,
        "four rows {} B, budget {runway} B",
        4 * row
    );
}

/// A load whose weights measure 64 MiB over their descriptors gives the
/// shortfall back from the store's cap before any slot exists, and keeps the
/// 64K window. Mutation: return the first engine plan in `settle` → red
/// (nothing is given back).
#[test]
fn replan_after_load_keeps_the_allowance() {
    let c = flash_next_config();
    let (decision, _) = load(false);
    let held_back = decision.store.expert_cap_bytes + decision.store.row_cache_bytes;
    let engine = |held: u64| {
        plan_engine(
            "Qwen4Exp",
            &c,
            live(FLASH_NEXT_LOADED + 64 * MIB),
            held,
            None,
        )
    };
    let unsettled = engine(held_back).unwrap();
    assert!(unsettled.pool.unwrap().lane.window.maximum() < 65_536);
    let mut lowered = 0;
    let plan = settle(&engine, held_back, decision.allowance, &mut |bytes| {
        lowered = bytes;
        Ok(true)
    })
    .unwrap();
    assert_eq!(lowered, 64 * MIB);
    assert!(plan.pool.unwrap().lane.window.maximum() >= 65_536);
    let refused = settle(&engine, held_back, decision.allowance, &mut |_| Ok(false)).unwrap();
    assert_eq!(refused, unsettled);
}

/// The pool starts at the plan's runway, grows by 171-block extents to its
/// pages and probes each growth against the plan's ceiling. Mutation: build
/// the pool at its maximum in `paged::create` → red (2,080 blocks).
#[test]
fn pool_starts_at_the_runway_and_grows_to_the_plan() {
    let c = flash_next_config();
    let (_, plan) = load(false);
    let pool = plan.pool.unwrap();
    let adapter = super::super::paged::create(&c, &pool, plan.ceiling_bytes()).unwrap();
    assert_eq!(adapter.block_capacity(), 96);
    let telemetry = adapter.block_telemetry().unwrap();
    assert_eq!(telemetry.max_total_blocks, 2_080);
    assert_eq!(adapter.extent_growth_blocks(), Some(171));
    assert_eq!(adapter.memory_ceiling(), Some(plan.ceiling_bytes()));
    assert_eq!(adapter.bytes_per_block().unwrap(), PHYSICAL_BLOCK);
}

/// The context limits publish the plan: the AR window and the pool's pages,
/// the trained window, the rest-of-window rules, and the MTP lane's window
/// only with MTP weights. Mutation: publish the trained window as the
/// effective one in `Qwen4ExpContextLimits::from_plan` → red (262,144).
#[test]
fn context_limits_publish_the_plan() {
    let c = flash_next_config();
    let (_, ar) = load(false);
    let limits = super::super::Qwen4ExpContextLimits::from_plan(&c, &ar, false).unwrap();
    assert_eq!(
        limits,
        super::super::Qwen4ExpContextLimits {
            effective_window_tokens: 65_536,
            paged_block_capacity: 2_080,
            paged_block_size: 32,
            trained_window_tokens: 262_144,
            speculative_window_tokens: None,
            rest_of_window_output_budget: true,
            omitted_budget_is_rest_of_window: true,
            speculative_omitted_budget_is_rest_of_window: false,
        }
    );
    let (_, mtp) = load(true);
    let limits = super::super::Qwen4ExpContextLimits::from_plan(&c, &mtp, true).unwrap();
    assert_eq!(limits.effective_window_tokens, 95_225);
    assert_eq!(limits.speculative_window_tokens, Some(65_655));
    assert!(limits.speculative_omitted_budget_is_rest_of_window);
}

/// The load-free estimate is the plan the load makes: 65,536 on the 36 GB
/// input, and AR 95,225 / MTP 65,655 with MTP weights; a lower
/// `--max-memory` never serves more. Mutation: leave the stores' cap out of
/// the held-back bytes in `estimate` → red (the AR window grows past the
/// load's).
#[test]
fn estimate_is_the_load_plan() {
    let c = flash_next_config();
    let ar = estimate("Qwen4Exp", &c, &inventory(), live(0), None, false).unwrap();
    assert_eq!(
        ar,
        Qwen4ExpContextEstimate {
            window_tokens: load(false).1.pool.unwrap().lane.window.maximum(),
            speculative_window_tokens: None,
            bound: "memory".into(),
        }
    );
    assert_eq!(ar.window_tokens, 65_536);
    let mtp = estimate("Qwen4Exp", &c, &inventory(), live(0), None, true).unwrap();
    assert_eq!(
        (mtp.window_tokens, mtp.speculative_window_tokens),
        (95_225, Some(65_655))
    );
    let lower = estimate("Qwen4Exp", &c, &inventory(), live(0), Some(16 << 30), false).unwrap();
    assert!(lower.window_tokens <= ar.window_tokens);
}
