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
/// The bf16 pool block: the priced page, index keys included.
const PHYSICAL_BLOCK: u64 = 811_008;

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
    load_with(&flash_next_config(), has_mtp, STUDIO_WORKING_SET)
}

/// [`load`] of `c` on a `working_set`.
pub(in crate::models::qwen4_exp) fn load_with(
    c: &Config,
    has_mtp: bool,
    working_set: u64,
) -> (StoreDecision, MemoryPlan) {
    let at = |active: u64| LiveMemory {
        working_set_bytes: working_set,
        ..live(active)
    };
    let decision = plan_store(
        "Qwen4Exp",
        c,
        &inventory(),
        at(0),
        None,
        None,
        has_mtp,
        false,
    )
    .unwrap();
    let held_back = decision.store.expert_cap_bytes + decision.store.row_cache_bytes;
    let plan = plan_engine(
        "Qwen4Exp",
        c,
        at(FLASH_NEXT_LOADED),
        held_back,
        None,
        has_mtp,
        false,
    )
    .unwrap();
    (decision, plan)
}

/// The Flash Next config with int8 pages.
pub(in crate::models::qwen4_exp) fn int8_config() -> Config {
    let mut c = flash_next_config();
    c.kv_format = KvFormat::Int8;
    c
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

/// On the MTP plan (one 65,536-token window) one MTP owner at the window
/// (847,737,856 B with the 255-token lookahead) fits the state budget beside
/// its 2,056 pages in a pool grown by 165-block extents to 2,076 blocks
/// (the plan's maximum is 2,080), and two do not; four AR rows fit at the
/// runway. Mutation: leave the transient out
/// of `scheduled_state_budget` → red (two owners fit).
#[test]
fn flash_next_scheduled_state_budget_holds_one_long_mtp_owner() {
    let c = flash_next_config();
    let (_, plan) = load(true);
    let pool = plan.pool.unwrap();
    let window = pool.lane.window.maximum();
    assert_eq!(window, 65_536);
    let row = row_state_bytes(&c);
    let owner = row + draft_state_bytes(&c, u64::from(window) + 255);
    assert_eq!(owner, 847_737_856);
    let pages = (u64::from(window) + 255).div_ceil(32);
    assert_eq!(pages, 2_056);
    let extent = u64::from(
        crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(PHYSICAL_BLOCK),
    );
    assert_eq!(extent, 165);
    let total = grown(96, extent, pages).min(u64::from(pool.max_blocks));
    assert_eq!(total, 2_076);
    let charge = |total: u64| PoolCharge {
        total_blocks: total,
        max_blocks: u64::from(pool.max_blocks),
        physical_block_bytes: PHYSICAL_BLOCK,
        growth_copy_bytes: total * PHYSICAL_BLOCK / 12,
    };
    let transient = memory::prefill_scratch(&c, super::super::MAX_PREFILL_CHUNK, window as usize);
    let budget = scheduled_state_budget(plan.dynamic_budget_bytes, transient, &charge(total));
    assert!(owner <= budget, "one owner {owner} B, budget {budget} B");
    assert!(
        2 * owner > budget,
        "two owners {} B, budget {budget} B",
        2 * owner
    );
    let runway = scheduled_state_budget(plan.dynamic_budget_bytes, transient, &charge(96));
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
            false,
            false,
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

/// The pool starts at the plan's runway, grows by 165-block extents to its
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
    assert_eq!(adapter.extent_growth_blocks(), Some(165));
    assert_eq!(adapter.memory_ceiling(), Some(plan.ceiling_bytes()));
    assert_eq!(adapter.bytes_per_block().unwrap(), PHYSICAL_BLOCK);
}

/// The context limits publish the plan: the one window and the pool's
/// pages, the trained window, the rest-of-window rules and the page format.
/// Mutation: publish the trained window as the effective one in
/// `Qwen4ExpContextLimits::from_plan` → red (262,144).
#[test]
fn context_limits_publish_the_plan() {
    let c = flash_next_config();
    let (_, ar) = load(false);
    let limits = super::super::Qwen4ExpContextLimits::from_plan(&c, &ar, 262_144).unwrap();
    assert_eq!(
        limits,
        super::super::Qwen4ExpContextLimits {
            effective_window_tokens: 65_536,
            paged_block_capacity: 2_080,
            paged_block_size: 32,
            trained_window_tokens: 262_144,
            rest_of_window_output_budget: true,
            omitted_budget_is_rest_of_window: true,
            kv_format: "bf16".into(),
        }
    );
    let (_, mtp) = load(true);
    let limits = super::super::Qwen4ExpContextLimits::from_plan(&c, &mtp, 262_144).unwrap();
    assert_eq!(limits.effective_window_tokens, 65_536);
    let int8 = int8_config();
    let (_, plan) = load_with(&int8, false, STUDIO_WORKING_SET);
    let limits = super::super::Qwen4ExpContextLimits::from_plan(&int8, &plan, 262_144).unwrap();
    assert_eq!(limits.kv_format, "int8");
}

/// The load-free estimate is the plan the load makes: 65,536 on the 36 GB
/// input with or without MTP weights and with int8 pages; a lower
/// `--max-memory` never serves more. Mutation: leave the stores' cap out of
/// the held-back bytes in `estimate` → red (the AR window grows past the
/// load's).
#[test]
fn estimate_is_the_load_plan() {
    let c = flash_next_config();
    let ar = estimate(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        None,
        None,
        false,
        false,
    )
    .unwrap();
    assert_eq!(
        ar,
        Qwen4ExpContextEstimate {
            window_tokens: load(false).1.pool.unwrap().lane.window.maximum(),
            bound: "memory".into(),
        }
    );
    assert_eq!(ar.window_tokens, 65_536);
    let mtp = estimate(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        None,
        None,
        true,
        false,
    )
    .unwrap();
    assert_eq!(mtp.window_tokens, 65_536);
    let int8 = estimate(
        "Qwen4Exp",
        &int8_config(),
        &inventory(),
        live(0),
        None,
        None,
        false,
        false,
    )
    .unwrap();
    assert_eq!(int8.window_tokens, 65_536);
    let lower = estimate(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        Some(16 << 30),
        None,
        false,
        false,
    )
    .unwrap();
    assert!(lower.window_tokens <= ar.window_tokens);
}

/// A transformers checkpoint's stacked experts are the store's experts, not
/// its fixed weights: 48 BF16 layers inventory 241,591,910,400 B of experts
/// with a 4,816,896,000 B minimum, and the 36 GB plan streams them under a
/// cap below the inventory. Mutation: drop the `.mlp.experts.` arm of
/// `routed_expert_layer` → red (no experts; the floors do not fit).
#[test]
fn hf_experts_stream_from_the_plan() {
    use super::super::weights::{Store, Tensor};
    let c = flash_next_config();
    let mut store = Store::open_descriptors(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-exp/bf16"),
    )
    .unwrap();
    assert!(!store.gguf);
    store.tensors.clear();
    let bf16 = |shape: Vec<usize>| {
        let bytes = shape.iter().product::<usize>() as u64 * 2;
        Tensor::safe_descriptor(shape, "BF16", bytes)
    };
    for layer in 0..48 {
        store.tensors.insert(
            format!("layers.{layer}.mlp.experts.gate_up_proj"),
            bf16(vec![512, 1280, 2560]),
        );
        store.tensors.insert(
            format!("layers.{layer}.mlp.experts.down_proj"),
            bf16(vec![512, 2560, 640]),
        );
    }
    let fixed_bytes = 4096 * 2560 * 2;
    store.tensors.insert(
        "layers.0.self_attn.q_proj.weight".into(),
        bf16(vec![4096, 2560]),
    );
    let inventory = store.load_inventory(&c, false).unwrap();
    assert_eq!(inventory.experts.all, 241_591_910_400);
    assert_eq!(inventory.experts.minimum, 4_816_896_000);
    assert_eq!(inventory.store_fixed_bytes, fixed_bytes);
    let decision = plan_store(
        "Qwen4Exp",
        &c,
        &inventory,
        live(0),
        None,
        None,
        false,
        false,
    )
    .unwrap_or_else(|error| panic!("the HF store must stream: {error}"));
    assert!(!decision.store.whole);
    assert!(decision.store.expert_cap_bytes < inventory.experts.all);
}

/// The KV allowance follows the requested context (Splash `allowance`'s
/// non-spill branch), so a 100K cap on the 36 GB input serves 102,400.
/// Mutation: keep `min(trained, 64K)` whatever the request → red
/// (refused, the plan holds 65,536).
#[test]
fn flash_next_allowance_is_the_requested_context() {
    let c = flash_next_config();
    let transient = |window: u32| transient(&c, window);
    let pool = pool_input(&c, &transient, &|_| 0);
    let ar = |dynamic: u64| pool.plan_at(dynamic).lane.window.maximum();
    let decision = plan_store(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        None,
        Some(102_400),
        false,
        false,
    )
    .unwrap();
    assert_eq!(decision.allowance, allowance_dynamic_bytes(&[&ar], 102_400));
    let estimate = estimate(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        None,
        Some(102_400),
        false,
        false,
    )
    .unwrap();
    assert_eq!(estimate.window_tokens, 102_400);
    let unset = plan_store(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        None,
        None,
        false,
        false,
    )
    .unwrap();
    assert_eq!(unset.allowance, allowance_dynamic_bytes(&[&ar], 65_536));
}

/// A context the plan cannot hold under `--max-memory` fails with Splash's
/// text. Mutation: skip `check_context` → red (Ok).
#[test]
fn flash_next_context_above_the_plan_is_refused() {
    let c = flash_next_config();
    let error = estimate(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        Some(8 << 30),
        Some(131_072),
        false,
        false,
    )
    .unwrap_err();
    assert!(
        error
            .reason
            .starts_with("Qwen4Exp: --max-context 131072 exceeds the "),
        "{}",
        error.reason
    );
    assert!(
        error
            .reason
            .contains("tokens the model and --max-memory allow; omit it or pass at most "),
        "{}",
        error.reason
    );
}

/// 400K serves the YaRN window: the pool lane prices 409,600 and a 128 GiB
/// working set holds it. Mutation: price the native window in
/// `pool_input` → red (refused).
#[test]
fn flash_next_400k_serves_the_yarn_window() {
    let mut c = flash_next_config();
    serve_context("Qwen4Exp", &mut c, Some(409_600)).unwrap();
    assert!(c.rope_scaling.yarn());
    let transient = |window: u32| transient(&c, window);
    assert_eq!(pool_input(&c, &transient, &|_| 0).shape.trained, 409_600);
    let large = LiveMemory {
        working_set_bytes: 128 << 30,
        active_bytes: 0,
        registered_pool_bytes: 0,
    };
    let estimate = estimate(
        "Qwen4Exp",
        &c,
        &inventory(),
        large,
        None,
        Some(409_600),
        false,
        false,
    )
    .unwrap();
    assert_eq!(estimate.window_tokens, 409_600);
}

/// The published window is at most the cap, the trained one stays native.
/// Mutation: no cap in `from_plan` → red (65,536).
#[test]
fn flash_next_limits_publish_the_cap() {
    let c = flash_next_config();
    let (_, mtp) = load(true);
    let limits = super::super::Qwen4ExpContextLimits::from_plan(&c, &mtp, 50_000).unwrap();
    assert_eq!(limits.effective_window_tokens, 50_000);
    assert_eq!(limits.trained_window_tokens, 262_144);
}

/// Splash `engine_memory_plan_test` M3 Pro working set.
const M3_PRO_WORKING_SET: u64 = 28_991_029_248;

/// An int8 load builds an int8 pool: 423,936-byte blocks (K/V, their f32
/// scale planes and the index keys), the 256-page runway, the plan's 2,304
/// pages.
/// Mutation D1: `pool_block` keeps BF16 for an int8 config → red.
#[test]
fn flash_next_pool_follows_kv_format() {
    let c = int8_config();
    let (_, plan) = load_with(&c, false, STUDIO_WORKING_SET);
    let pool = plan.pool.unwrap();
    let adapter = super::super::paged::create(&c, &pool, plan.ceiling_bytes()).unwrap();
    assert_eq!(
        adapter.cache_dtype(),
        mlx_paged_attn::metal::MetalDtype::Int8
    );
    assert_eq!(adapter.bytes_per_block().unwrap(), 423_936);
    assert_eq!(adapter.block_capacity(), 256);
    assert_eq!(adapter.block_telemetry().unwrap().max_total_blocks, 2_304);
}

/// The pool block is the priced page in both formats (Splash
/// `bytesPerModelPage`): K/V, scales and the indexer's pooled keys in the
/// block's index plane, 8 bf16 rows of 128 per layer; K and V token-major.
/// Mutation X1: `page_layout` without the index plane → red (399,360).
#[test]
fn flash_next_pool_block_is_the_priced_page() {
    for (c, page) in [(flash_next_config(), 811_008), (int8_config(), 423_936)] {
        assert_eq!(flash_next_layout(&c).page_bytes(), page);
        let adapter = super::super::paged::create_with_blocks(&c, 1, 1).unwrap();
        assert_eq!(adapter.bytes_per_block().unwrap(), page);
        assert_eq!(
            adapter.page_layout(),
            mlx_paged_attn::PageLayout {
                values_token_major: true,
                index_rows: 8,
                index_dimension: 128,
            }
        );
    }
}

/// Unset `kvFormat` is int8 where the int8 page kernels serve the head
/// (128, 256) and bf16 elsewhere; an explicit int8 they cannot serve and an
/// unknown format fail. Mutation D2: unset resolves to bf16 → red.
#[test]
fn flash_next_kv_format_defaults_to_int8_where_pages_serve() {
    let mut c = flash_next_config();
    assert_eq!(resolve_kv_format("Q", None, &c).unwrap(), KvFormat::Int8);
    c.head_dim = 128;
    assert_eq!(resolve_kv_format("Q", None, &c).unwrap(), KvFormat::Int8);
    c.head_dim = 32;
    assert_eq!(resolve_kv_format("Q", None, &c).unwrap(), KvFormat::Bf16);
    let error = resolve_kv_format("Q", Some("int8"), &c).unwrap_err();
    assert!(
        error
            .reason
            .contains("no int8 page kernels for head_dim 32"),
        "{error}"
    );
    c.head_dim = 256;
    assert_eq!(
        resolve_kv_format("Q", Some("bf16"), &c).unwrap(),
        KvFormat::Bf16
    );
    assert!(resolve_kv_format("Q", Some("fp8"), &c).is_err());
}

/// The int8 page is Splash's: K/V 16,384 + 256 B of scales per tensor and
/// the bf16 index, 423,936 B; 256-page extents and runway. The 36 GB plan
/// keeps the 64K window in 2,304 pages and gives the bytes int8 saves to
/// the experts (cap 15,228,690,262); the M3 Pro too. Mutation W1:
/// `flash_next_layout` keeps bf16 → red (page 811,008).
#[test]
fn int8_pool_window_from_the_plan() {
    let c = int8_config();
    let layout = flash_next_layout(&c);
    assert_eq!(layout.page_bytes(), 423_936);
    assert_eq!(layout.row_bytes(), 13_248);
    assert_eq!(layout.extent_alignment_pages(), 256);
    assert_eq!(layout.minimum_extent_pages(), 256);
    assert_eq!(layout.extent_pages_for(2_304), 256);
    assert_eq!(layout.runway_pages(), 256);
    let (decision, plan) = load_with(&c, false, STUDIO_WORKING_SET);
    assert_eq!(decision.allowance, 3_698_766_848);
    assert_eq!(
        decision.store,
        StorePlan {
            whole: false,
            expert_cap_bytes: 15_228_690_262,
            row_cache_bytes: 1_744_830_464,
        }
    );
    let pool = plan.pool.unwrap();
    assert_eq!(pool.lane.window.maximum(), 65_536);
    assert_eq!(pool.lane.window.bound(), "memory");
    assert_eq!((pool.runway_blocks, pool.max_blocks), (256, 2_304));
    let (decision, plan) = load_with(&c, false, M3_PRO_WORKING_SET);
    assert_eq!(decision.store.expert_cap_bytes, 14_242_993_596);
    assert_eq!(plan.pool.unwrap().lane.window.maximum(), 65_536);
}

/// The MTP draft state is the one lane's, charged once: at a fixed dynamic
/// budget the int8 window is 66,175 (90,964 without MTP), and a load with
/// MTP weights keeps 64K in both formats. Mutation O1: `LaneKv::Pool`
/// also takes the state from the pool's KV (the state charged twice) → red
/// (a smaller window).
#[test]
fn mtp_state_prices_the_one_pool_window() {
    const DYNAMIC: u64 = 4_468_093_952;
    let c = int8_config();
    let transient = |window: u32| transient(&c, window);
    let state = lane_state(&c, true, false);
    let mtp = pool_input(&c, &transient, &state).plan_at(DYNAMIC);
    assert_eq!(mtp.lane.window.maximum(), 66_175);
    assert_eq!(mtp.max_blocks, 2_304);
    assert_eq!(
        mtp.lane.lane_bytes,
        row_state_bytes(&c) + lane_state(&c, true, false)(66_175)
    );
    let ar = pool_input(&c, &transient, &|_| 0).plan_at(DYNAMIC);
    assert_eq!((ar.lane.window.maximum(), ar.max_blocks), (90_964, 3_072));
    for (config, allowance, cap, max_blocks) in [
        (int8_config(), 4_454_974_464, 14_472_482_646, 2_304),
        (flash_next_config(), 5_224_301_568, 13_703_155_542, 2_080),
    ] {
        let (decision, plan) = load_with(&config, true, STUDIO_WORKING_SET);
        assert_eq!(decision.allowance, allowance);
        assert_eq!(decision.store.expert_cap_bytes, cap);
        let pool = plan.pool.unwrap();
        assert_eq!(pool.lane.window.maximum(), 65_536);
        assert_eq!(pool.max_blocks, max_blocks);
    }
}

/// With the MTP head every prefill keeps its draft history: the lane's state
/// at a window is the MTP turn's draft state, the draft indexer keys of every
/// row (64 B at the Flash Next geometry) and a full prefill window of BF16
/// hidden rows waiting for the next draft pass (1,024 x 20,480 B). Without
/// the head the lane holds none.
/// Mutation: the waiting rows are left out of `lane_state` → red.
#[test]
fn the_load_plan_prices_the_lane_draft() {
    let c = flash_next_config();
    assert_eq!(memory::draft_index_bytes_per_token(&c), 64);
    assert_eq!(memory::draft_hidden_row_bytes(&c), 20_480);
    for window in [65_536u32, 262_144] {
        assert_eq!(
            lane_state(&c, true, false)(window),
            draft_state_bytes(&c, u64::from(window)) + u64::from(window) * 64 + 20_971_520
        );
        assert_eq!(lane_state(&c, false, false)(window), 0);
    }
}

/// With the disk tier on, the lane books its staging whatever the window:
/// an encoded state (cell and, with the head, a window of waiting rows and
/// the seed) for every parcel and one more for the writer's copy, the copy
/// queue and the writer's queue of pages or draft segments behind it; off,
/// nothing.
/// Mutation: two states are booked, as before the parcels went per lane →
/// red.
#[test]
fn the_load_plan_books_the_state_staging() {
    use crate::models::qwen4_exp::prefix_disk::{PARCELS, QUEUE_BYTES};
    let c = flash_next_config();
    let cell = memory::state_cell_bytes(&c);
    let page = super::flash_next_layout(&c).page_bytes();
    let writer = (mlx_paged_attn::DEFAULT_QUEUE_DEPTH as u64 + 3) * page;
    let states = (PARCELS as u64 + 1) * memory::encoded_state_bytes(&c, false);
    assert!(memory::encoded_state_bytes(&c, false) > cell);
    assert!(memory::encoded_state_bytes(&c, true) > cell + 1025 * 20_480);
    for window in [65_536u32, 262_144] {
        let off = lane_state(&c, false, true)(window);
        assert!(off > states + QUEUE_BYTES + writer, "{off}");
        assert_eq!(off, memory::disk_staging_bytes(&c, false));
        assert_eq!(
            lane_state(&c, true, true)(window),
            lane_state(&c, true, false)(window) + memory::disk_staging_bytes(&c, true)
        );
        assert!(memory::disk_staging_bytes(&c, true) > off);
        assert_eq!(lane_state(&c, false, false)(window), 0);
    }
    let (_, off) = load_with(&c, false, STUDIO_WORKING_SET);
    let decision = plan_store(
        "Qwen4Exp",
        &c,
        &inventory(),
        live(0),
        None,
        None,
        false,
        true,
    )
    .unwrap();
    let held_back = decision.store.expert_cap_bytes + decision.store.row_cache_bytes;
    let on = plan_engine(
        "Qwen4Exp",
        &c,
        LiveMemory {
            working_set_bytes: STUDIO_WORKING_SET,
            ..live(FLASH_NEXT_LOADED)
        },
        held_back,
        None,
        false,
        true,
    )
    .unwrap();
    assert!(
        on.pool.unwrap().lane.lane_bytes > off.pool.unwrap().lane.lane_bytes,
        "the lane holds the staging"
    );
}
