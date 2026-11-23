use super::*;

#[test]
fn workspace_reuse_scales_down_and_stays_within_working_reserve() {
    assert_eq!(super::workspace_cache_bytes(Some(16 << 30)), 256 << 20);
    assert_eq!(super::workspace_cache_bytes(Some(32 << 30)), 512 << 20);
    assert_eq!(super::workspace_cache_bytes(Some(128 << 30)), 2 << 30);
    assert_eq!(super::workspace_cache_bytes(Some(512 << 30)), 2 << 30);
    assert_eq!(super::workspace_cache_bytes(None), super::FREE_CACHE_BYTES);
}

#[test]
fn admission_keeps_headroom_and_rejects_overflow() {
    assert!(fits_with_headroom(16 << 30, 32 << 30, 16 << 30));
    assert!(!fits_with_headroom(16 << 30, (32 << 30) - 1, 16 << 30));
    assert!(!fits_with_headroom(32 << 30, 40 << 30, 16 << 30));
    assert!(!fits_with_headroom(u64::MAX, u64::MAX, 16 << 30));
}

#[cfg(target_os = "macos")]
#[test]
fn constrained_fixture_host_preserves_real_memory_gates() {
    let busy = coherent(HostMemory {
        physical: 8 << 30,
        free: 4 << 30,
        file: 0,
        anonymous: 1 << 30,
        purgeable: 0,
        wired: 2 << 30,
        compressor: 1 << 30,
    });
    let fixture = coherent(fixture_host());
    let tiny = 256 << 20;
    let _fixture = scoped_test_host(fixture);
    let mut planned = plan_without_snapshot();
    host_snapshot(&mut planned).unwrap();
    assert_eq!(planned.available_bytes, Some(12 << 30));
    assert_eq!(planned.physical_bytes, Some(16 << 30));
    let mut direct = planned.clone();
    direct_gate(&mut direct, 1 << 30, tiny).unwrap();
    assert_eq!(direct.bank_budget, 7 << 30);
    admit_direct(tiny).unwrap();

    {
        let _busy = scoped_test_host(busy);
        let mut refused = plan_without_snapshot();
        refused.physical_bytes = Some(busy.physical);
        let error = direct_gate(&mut refused, 1 << 30, tiny).unwrap_err();
        assert!(error.reason.contains("host leaves 0.0 GiB"));
        let error = admit_direct(tiny).unwrap_err();
        assert!(error.reason.contains("host leaves 0.0 GiB"));
    }
    admit_direct(tiny).unwrap();
}

fn gib(n: f64) -> u64 {
    (n * (1u64 << 30) as f64) as u64
}

fn coherent(host: HostMemory) -> HostMemory {
    let sum = host.free + host.file + host.anonymous + host.wired + host.compressor;
    assert!(sum <= host.physical + (1 << 30), "incoherent host fixture");
    assert!(host.purgeable <= host.anonymous);
    host
}

#[test]
fn bank_gate_reads_live_memory_not_free() {
    let banks = gib(73.15);
    let need = gib(13.0);
    let quiet = coherent(HostMemory {
        physical: 128 << 30,
        free: gib(77.3),
        file: gib(10.0),
        anonymous: gib(20.1),
        purgeable: gib(0.7),
        wired: gib(13.2),
        compressor: gib(7.5),
    });
    let gate = bank_gate(&quiet, need, 0);
    assert_eq!(
        gate.non_reclaimable,
        gib(13.2) + gib(7.5) + gib(20.1) - gib(0.7)
    );
    assert_eq!(
        gate.bank_budget,
        (128u64 << 30) - gate.non_reclaimable - need
    );
    assert!(gate.bank_budget >= banks);
    assert!(bank_gate(&quiet, need, DIRECT_MARGIN).bank_budget < banks);
    let hog = coherent(HostMemory {
        free: gib(37.3),
        anonymous: gib(20.1) + gib(40.0),
        ..quiet
    });
    assert!(bank_gate(&hog, need, 0).bank_budget < banks);
    assert!(bank_gate(&hog, need, 0).bank_budget < gate.bank_budget - gib(39.0));
    let hog20 = coherent(HostMemory {
        free: gib(57.3),
        anonymous: gib(20.1) + gib(20.0),
        ..quiet
    });
    assert!(bank_gate(&hog20, need, 0).bank_budget < banks);
    let purgeable = coherent(HostMemory {
        purgeable: gib(30.0),
        ..hog20
    });
    assert!(bank_gate(&purgeable, need, 0).bank_budget >= banks);
    let wired = coherent(HostMemory {
        free: gib(57.3),
        wired: gib(13.2) + gib(20.0),
        ..quiet
    });
    assert!(bank_gate(&wired, need, 0).bank_budget < banks);
    assert_eq!(bank_gate(&quiet, u64::MAX, 0).bank_budget, 0);
}

pub(super) fn plan_without_snapshot() -> Plan {
    Plan {
        budget: 73 << 30,
        hot_bytes: 83_631_869_440,
        available_bytes: None,
        physical_bytes: None,
        resident_bank_bytes: Arc::default(),
        partial_expert_slots: 442,
        direct: false,
        partial_direct_gate: false,
        direct_bytes: 0,
        direct_layers: 0,
        bank_budget: 0,
        live_non_reclaimable: 0,
        host: None,
        anonymous_need: 0,
        self_mlx_bytes: 0,
    }
}

#[test]
fn active_budget_reclaims_and_repolls_before_refusing() {
    use std::sync::{Arc, Mutex};
    let readings = Arc::new(Mutex::new(
        vec![40.0 * (1 << 30) as f64, 10.0 * (1 << 30) as f64].into_iter(),
    ));
    let mut reclaims = 0;
    let mut sleeps = 0;
    super::check_active_memory_budget(
        32 << 30,
        {
            let readings = Arc::clone(&readings);
            move || readings.lock().unwrap().next().unwrap_or(0.0)
        },
        || reclaims += 1,
        |_| sleeps += 1,
    )
    .unwrap();
    assert_eq!(reclaims, 1);
    assert_eq!(sleeps, 1);
}

#[test]
fn active_budget_refuses_after_bounded_repolls() {
    let mut reclaims = 0;
    let mut sleeps = 0;
    let err = super::check_active_memory_budget(
        8 << 30,
        || 40.0 * (1 << 30) as f64,
        || reclaims += 1,
        |_| sleeps += 1,
    )
    .unwrap_err();
    assert_eq!(reclaims, 1);
    assert_eq!(sleeps, 3);
    assert!(err.reason.contains("SSD safety limit"));
}

/// Without a refused gate every pass is the whole chunk. With one, a pass
/// keeps the chunk up to the context the gate priced, one byte less halves
/// it, and a pass at a longer context halves until its scratch fits the
/// reservation.
#[test]
fn prefill_window_halves_only_past_the_scratch_a_refused_gate_reserved() {
    let c = flash_next_config();
    let chunk = super::super::MAX_PREFILL_CHUNK;
    assert_eq!(prefill_window(&c, None, 262_144), chunk);
    for context in [65_536, 262_144] {
        let priced = prefill_scratch(&c, chunk, context);
        assert_eq!(prefill_window(&c, Some(priced), context), chunk);
        assert_eq!(prefill_window(&c, Some(priced - 1), context), chunk / 2);
    }
    let reserved = prefill_scratch(&c, chunk, 65_536);
    let long = prefill_window(&c, Some(reserved), 262_144);
    assert!((2..chunk).contains(&long), "{long} rows");
    assert!(prefill_scratch(&c, long, 262_144) <= reserved);
    assert!(prefill_scratch(&c, long * 2, 262_144) > reserved);
    assert_eq!(prefill_window(&c, Some(0), 262_144), 1);
}

/// The Flash Next cell on its storage: 36 linear layers of 3 bf16 conv
/// rows of 10,240 (2,211,840) and F32 recurrent matrices (113,246,208), 9
/// bf16 PLE history rows of 10,240 (184,320), and 3 pending bf16 index keys
/// of 128 in 12 attention layers (9,216). Splash `cellBytes` aligns each part
/// to 16 KiB: 115,802,112. Mutation: the F32 formula with 32,768 tokens of
/// pooled index keys in `row_state_bytes` → red (169,967,616).
#[test]
fn flash_next_cell_matches_the_storage() {
    let c = flash_next_config();
    assert_eq!(
        row_state_parts(&c),
        [2_211_840, 113_246_208, 184_320, 9_216]
    );
    assert_eq!(row_state_bytes(&c), 115_651_584);
    const SPLASH_FLASH_NEXT_CELL: u64 = 115_802_112;
    assert_eq!(SPLASH_FLASH_NEXT_CELL - row_state_bytes(&c), 150_528);
}

/// Two Flash Next owners parked right after a 1024-row prefill slice, with
/// 1023 trusted rows and the seed waiting and nothing drafted: each keeps
/// 1024 BF16 hidden rows of 20,480 B beside its recurrent row and no draft
/// K/V. A budget sized to the draft-K/V figure for those rows admits a third
/// owner only after reclaim.
/// MUTATION: charge waiting rows as draft K/V in
/// `draft_owner_bytes`; `held` falls to the K/V figure, the third owner fits
/// and the AfterReclaim assertion fails.
#[test]
fn flash_next_parked_owners_charge_their_waiting_hidden_rows() {
    use crate::engine::hybrid_scheduler::{
        ScheduledStateCapacity, ScheduledStateFit, scheduled_state_fit,
    };
    let c = flash_next_config();
    assert_eq!(draft_hidden_row_bytes(&c), 20_480);
    let row = row_state_bytes(&c);
    let parked = draft_owner_bytes(&c, 0, 1024, false);
    assert_eq!(parked, 1024 * 20_480);
    let kv_figure = 1024 * draft_kv_bytes_per_token(&c);
    assert!(parked > kv_figure);
    assert_eq!(
        draft_owner_bytes(&c, 7, 1, true),
        4 * row + 7 * (4096 + 64) + 20_480,
        "cached rows pay K/V, the seed pays its hidden row, a tracked owner its snapshots"
    );
    let candidate = row + draft_state_bytes(&c, 256);
    let budget = 2 * (row + kv_figure) + candidate;
    let held = 2 * (row + parked);
    assert_eq!(
        scheduled_state_fit(Some(ScheduledStateCapacity { budget, held }), 0, candidate),
        ScheduledStateFit::AfterReclaim,
        "the retained hidden rows overcommit a budget sized to their K/V figure"
    );
    assert_eq!(
        scheduled_state_fit(
            Some(ScheduledStateCapacity {
                budget,
                held: 2 * (row + kv_figure),
            }),
            0,
            candidate
        ),
        ScheduledStateFit::Fits
    );
}

/// A Flash Next state at a grid line is the cell without pending index
/// keys: 2,211,840 + 113,246,208 + 184,320 = 115,642,368 B, plus 4 B per
/// lineage token and 8 B per lineage block. On the plan numbers of a busy
/// host (dynamic 10,724,034,662 B, transient 5,746,196,480 B, a pool of
/// 8,192 int8 pages of 423,936 B, one row held) the room holds 12 states of
/// 4,096-token lineages and 11 of 61,440-token ones.
/// Mutation: `state_room` charges the pool only as it is now (64 pages) →
/// red (the room grows past 40 states).
#[test]
fn flash_next_states_fit_the_room_the_plan_leaves() {
    use crate::models::qwen3_5::dflash2_state::store::lineage_bytes;
    let c = flash_next_config();
    assert_eq!(state_cell_bytes(&c), 115_642_368);
    let state = |tokens: usize| state_cell_bytes(&c) + lineage_bytes(tokens, tokens / 32);
    let pool = |total_blocks: u64| PoolCharge {
        total_blocks,
        max_blocks: 8_192,
        physical_block_bytes: 423_936,
        growth_copy_bytes: total_blocks * 423_936 / 12,
    };
    let room =
        |pool: &PoolCharge| state_room(10_724_034_662, 5_746_196_480, pool, row_state_bytes(&c));
    assert_eq!(room(&pool(8_192)), 1_389_302_886);
    assert_eq!(room(&pool(64)), room(&pool(8_192)));
    assert_eq!(room(&pool(8_192)) / state(4_096), 12);
    assert_eq!(room(&pool(8_192)) / state(61_440), 11);
}

/// A state is refused when it would leave the host less than the pass
/// gate's floor, or MLX's active memory past the SSD read limit.
/// Mutation: `state_fits_live` drops the state's bytes from the host check
/// → red.
#[test]
fn a_state_keeps_the_live_headroom_a_pass_checks() {
    let host = HostMemory {
        physical: 128 << 30,
        free: 10 << 30,
        file: 6 << 30,
        ..HostMemory::default()
    };
    let floor = system_reserve(host.physical);
    assert_eq!(floor, 16 << 30);
    assert!(!state_fits_live(&host, 0.0, None, 1, false));
    assert!(state_fits_live(&host, 0.0, None, 0, false));
    assert!(state_fits_live(&host, 0.0, None, 14 << 30, true));
    assert!(!state_fits_live(&host, 0.0, None, (14 << 30) + 1, true));
    assert!(state_fits_live(
        &host,
        1e9,
        Some(2_000_000_000),
        1_000_000_000,
        true
    ));
    assert!(!state_fits_live(
        &host,
        1e9,
        Some(2_000_000_000),
        1_000_000_001,
        true
    ));
}
