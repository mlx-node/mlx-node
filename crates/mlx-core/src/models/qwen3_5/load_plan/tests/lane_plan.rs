//! The plan's verifying-lane count where it reaches the plan's transient.
use super::*;
use crate::models::qwen3_5::dflash2_decode::pool_verify_lane_bytes;

fn lane_pool(
    config: &Qwen3_5Config,
    working_set: u64,
    lanes: u32,
    draft: DraftTerms,
    draft_lanes: bool,
) -> crate::engine::memory_plan::PoolPlan {
    shaped_draft_plan(config, 17_357 * MIB, working_set, lanes, draft, draft_lanes)
        .unwrap()
        .pool
        .unwrap()
}

/// The 27B with its draft at the served window (262,144): at every lane
/// count the plan's transient is the N-row autoregressive step or the
/// tapped prefill chunk, so the verifying lanes' rows are not the term that
/// sets it (a plan whose draft turns are not lanes has the same transient).
/// On a 128 GB machine the plan grants 4 lanes with the transient the
/// measured load printed (826,900,480 B).
#[test]
fn the_27b_plan_at_the_served_window_is_set_by_the_ar_step() {
    let config = shaped_config(5120, 64, 24, 4, 17408, 16, 48, false);
    let draft = draft_27b("chk-plan-served-window");
    let lane = pool_verify_lane_bytes(&config);
    assert_eq!(lane, 43_188_224);
    for lanes in [2u32, 3, 4, 6, 7, 8] {
        let on = lane_pool(&config, 384 << 30, lanes, draft, true);
        let off = lane_pool(&config, 384 << 30, lanes, draft, false);
        let window = on.lane.window.maximum();
        let verify = pool_lane_transient(&config, Some(draft.tap_layers), window, lanes);
        let one = pool_lane_transient(&config, Some(draft.tap_layers), window, 1);
        let step = pool_decode_step_bytes(&config, u64::from(lanes), window);
        println!(
            "chk plan 27B served window: lanes offered {lanes} granted {} window {window}: transient lanes-on {} lanes-off {}; verify side of {lanes} lanes {verify}, of one lane {one}, AR step of {lanes} rows {step}",
            on.lanes, on.lane.transient_bytes, off.lane.transient_bytes
        );
        assert_eq!(on.lanes, lanes);
        assert_eq!(window, 262_144);
        assert_eq!(on.lane.transient_bytes, verify.max(step));
        assert_eq!(off.lane.transient_bytes, one.max(step));
        assert_eq!(
            on.lane.transient_bytes, off.lane.transient_bytes,
            "{lanes} lanes: the verifying lanes' rows do not set the transient"
        );
        assert!(
            verify == one || step > verify,
            "{lanes} lanes: the prefill chunk or the AR step is the larger"
        );
    }
    let machine = lane_pool(&config, working_set_of(128), 4, draft, true);
    println!(
        "chk plan 27B on 128 GB: lanes {} window {} transient {} lane bytes {} kv bytes {} minimum {}",
        machine.lanes,
        machine.lane.window.maximum(),
        machine.lane.transient_bytes,
        machine.lane.lane_bytes,
        machine.lane.kv_bytes,
        machine.lane.minimum_bytes
    );
    assert_eq!(machine.lanes, 4);
    assert_eq!(machine.lane.transient_bytes, 826_900_480);
    assert_eq!(
        pool_lane_transient(&config, Some(draft.tap_layers), 262_144, 4),
        491_126_784,
        "4 verifying lanes (376,418,304) stay under the tapped prefill chunk"
    );
}

/// A 27B-shaped load with a short served window (no two-pass attention
/// partials in its decode step): the verify of 7 and 8 lanes is the plan's
/// peak, so a plan of draft lanes prices every verifying lane and a plan
/// whose draft turns are not lanes prices one.
#[test]
fn a_plan_of_draft_lanes_prices_its_lanes_where_the_verify_is_the_peak() {
    let mut config = shaped_config(5120, 64, 24, 4, 17408, 16, 48, false);
    config.max_position_embeddings = 2_048;
    config.rope_scaling = None;
    let draft = draft_27b("chk-plan-short-window");
    let lane: u64 = 43_188_224;
    let verify_one: u64 = 246_853_632;
    let chunk: u64 = 491_126_784;
    for lanes in [4u32, 7, 8] {
        let on = lane_pool(&config, 384 << 30, lanes, draft, true);
        let off = lane_pool(&config, 384 << 30, lanes, draft, false);
        let window = on.lane.window.maximum();
        let step = pool_decode_step_bytes(&config, u64::from(lanes), window);
        println!(
            "chk plan short window: lanes offered {lanes} granted {} window {window}: transient lanes-on {} lanes-off {}; AR step of {lanes} rows {step}",
            on.lanes, on.lane.transient_bytes, off.lane.transient_bytes
        );
        assert_eq!(on.lanes, lanes);
        assert_eq!(window, 2_048);
        assert!(step < chunk, "the AR step is under the prefill chunk");
        assert_eq!(
            on.lane.transient_bytes,
            chunk.max(verify_one + u64::from(lanes - 1) * lane),
            "{lanes} draft lanes price every verifying lane"
        );
        assert_eq!(
            off.lane.transient_bytes, chunk,
            "{lanes} lanes whose draft turns are not lanes price one verify"
        );
    }
    let eight = lane_pool(&config, 384 << 30, 8, draft, true);
    assert_eq!(eight.lane.transient_bytes, 549_171_200);
    let seven = lane_pool(&config, 384 << 30, 7, draft, true);
    assert_eq!(seven.lane.transient_bytes, 505_982_976);
}
