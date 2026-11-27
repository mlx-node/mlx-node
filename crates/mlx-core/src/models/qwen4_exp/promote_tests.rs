//! A store the host gate refused at load moves up to whole banks at the
//! close of the last open turn, through the load's own gate. The deep
//! moe-shape fixture (12 layers, 12 experts, top 10) on a 16 GiB host: a
//! load beside too much other memory keeps 10 or 11 slots per layer (the
//! plan's cap leaves room for the largest layer's staging).
use super::prefix_reuse::{finish, fork, greedy, prompt, reuse_over, turn};
use super::route_step::{deep_moe_shape_fixture, moe_shape_decoder};
use super::*;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::models::qwen4_exp::live::{Fallback, Residency};

const PHYSICAL: u64 = 16 << 30;
/// What the plan holds beside the experts at load.
const NEED: u64 = 1 << 30;
const LAYERS: usize = 12;
const EXPERTS: usize = 12;
/// The context the fixture plans and the tokens its gate priced.
const WINDOW: usize = 256;
const ALLOWANCE: u32 = 128;

/// A host whose non-reclaimable memory is `non_reclaimable` bytes (2 GiB
/// wired, 1 GiB compressed, the rest other processes' anonymous pages),
/// with 8 GiB free or in the file cache.
pub(super) fn host_with(non_reclaimable: u64) -> memory::HostMemory {
    memory::HostMemory {
        physical: PHYSICAL,
        free: 6 << 30,
        file: 2 << 30,
        anonymous: non_reclaimable - (3 << 30),
        purgeable: 0,
        wired: 2 << 30,
        compressor: 1 << 30,
    }
}

/// Cached direct (whole-bank) store template on the deep fixture; cloned via
/// `clone_direct_for_test` so tests never reload 525 MiB of banks.
fn deep_direct_store() -> (weights::Store, config::Config) {
    static DIRECT: std::sync::OnceLock<(std::sync::Mutex<weights::Store>, config::Config)> =
        std::sync::OnceLock::new();
    let (store, config) = DIRECT.get_or_init(|| {
        let (mut decoder, _) = moe_shape_decoder(deep_moe_shape_fixture());
        decoder.weights.plan_fixture_store().unwrap();
        decoder.weights.plan.physical_bytes = Some(PHYSICAL);
        decoder.register_cache_limit();
        {
            let _host = memory::scoped_test_host(memory::fixture_host());
            decoder.weights.select_residency(NEED).unwrap();
        }
        decoder.weights.prepare_fixture_hot().unwrap();
        decoder.weights.finish_fixture_load().unwrap();
        let probe = MxArray::from_uint32(&[0], &[1]).unwrap();
        for layer in 0..decoder.config.num_hidden_layers {
            decoder
                .weights
                .device_expert_slots(layer, &probe)
                .unwrap()
                .expect("a filled direct layer");
        }
        decoder.weights.start_direct_fill().unwrap();
        decoder.weights.publish_live();
        assert!(
            decoder.weights.plan.direct,
            "the fixture host admits the banks"
        );
        let c = (*decoder.config).clone();
        (std::sync::Mutex::new(decoder.weights), c)
    });
    let cloned = store
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone_direct_for_test();
    (cloned, config.clone())
}

/// Cached host-gate partial store template (empty slots, resident fixed banks)
/// on the deep fixture; cloned via `clone_partial_for_test`.
fn deep_host_gate_store(bank: u64) -> (weights::Store, config::Config) {
    static GATE: std::sync::OnceLock<(std::sync::Mutex<weights::Store>, config::Config)> =
        std::sync::OnceLock::new();
    let (store, config) = GATE.get_or_init(|| {
        let host = host_with(edge(bank) + bank / 24);
        let (mut decoder, _) = moe_shape_decoder(deep_moe_shape_fixture());
        decoder.weights.plan_fixture_store().unwrap();
        decoder.weights.plan.physical_bytes = Some(PHYSICAL);
        decoder.register_cache_limit();
        {
            let _host = memory::scoped_test_host(host);
            decoder.weights.select_residency(NEED).unwrap();
        }
        decoder.weights.prepare_fixture_hot().unwrap();
        decoder.weights.finish_fixture_load().unwrap();
        decoder.weights.publish_live();
        let c = (*decoder.config).clone();
        (std::sync::Mutex::new(decoder.weights), c)
    });
    let cloned = store
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone_partial_for_test(None);
    (cloned, config.clone())
}

fn model_over_store(store: weights::Store, c: config::Config) -> Inner {
    let vision = media::Vision::metadata(&serde_json::Value::Null, &store, &c).unwrap();
    let mut decoder = decoder::Decoder::new(c.clone(), store).unwrap();
    decoder.paged = Some(paged::create_fixture(&c).unwrap());
    decoder.register_cache_limit();
    decoder.weights.publish_live();
    decoder.adopt_plan(WINDOW, ALLOWANCE);
    reuse_over(inner_over(decoder, vision, false))
}

pub(super) fn model_direct() -> Inner {
    let (store, c) = deep_direct_store();
    model_over_store(store, c)
}

/// The routed bank bytes of the whole store.
pub(super) fn bank_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| deep_direct_store().0.plan.direct_bytes)
}

/// The other processes' non-reclaimable bytes at which a load just admits
/// every bank.
fn edge(bank: u64) -> u64 {
    PHYSICAL - memory::DIRECT_MARGIN - NEED - bank
}

/// A load beside other memory that leaves 23/24 of the banks: 10 or 11
/// slots, one or two fewer than the experts.
pub(super) fn host_gate_model(bank: u64) -> Inner {
    let (store, c) = deep_host_gate_store(bank);
    let inner = model_over_store(store, c);
    let weights = &inner.decoder.weights;
    assert!(!weights.plan.direct && weights.plan.partial_direct_gate);
    assert_eq!(weights.live.residency(), Residency::HostGate);
    assert!((10..EXPERTS).contains(&weights.plan.partial_expert_slots));
    assert!(inner.decoder.host_prefill_scratch.is_some());
    inner
}

/// This model's bytes the host holds, read from the ledgers one by one.
pub(super) fn own(inner: &Inner) -> u64 {
    inner.decoder.weights.ledgered_bytes()
        + inner
            .decoder
            .paged
            .as_ref()
            .unwrap()
            .pool_allocated_bytes()
            .unwrap()
        + inner.prefix_states.bytes()
}

/// One turn of the exclusive lane on `seq`: the prompt, `steps` greedy
/// steps, the epilogue. The close runs on `close`, a host holding this
/// model's bytes beside `other` bytes of other processes.
fn whole_turn(
    inner: &mut Inner,
    seq: u32,
    tokens: &[u32],
    steps: usize,
    other: u64,
) -> (usize, Vec<Vec<f32>>) {
    let (cached, out) = open_turn(inner, seq, tokens, steps);
    let close = host_with(other + own(inner));
    let _host = memory::scoped_test_host(close);
    inner.set_turn_cancel_flag(None);
    (cached, out)
}

/// [`whole_turn`] up to the close: the turn's flag stays installed.
fn open_turn(inner: &mut Inner, seq: u32, tokens: &[u32], steps: usize) -> (usize, Vec<Vec<f32>>) {
    let _host = memory::scoped_test_host(host_with(3 << 30));
    inner.set_turn_cancel_flag(Some(Arc::new(AtomicBool::new(false))));
    let (cached, logits) = turn(inner, seq, tokens);
    let mut out = vec![logits.clone()];
    out.extend(greedy(inner, &logits, steps));
    finish(inner);
    (cached, out)
}

fn slot_counts(inner: &Inner) -> Vec<Option<usize>> {
    (0..LAYERS)
        .map(|layer| {
            inner
                .decoder
                .weights
                .slot_occupants(layer)
                .map(|slots| slots.len())
        })
        .collect()
}

fn promotions(inner: &Inner) -> u64 {
    inner
        .decoder
        .weights
        .live
        .promotions
        .load(Ordering::Acquire)
}

/// Turn 1 on a refused store publishes three states; its close finds a host
/// with room and moves the store up: 12 slots in every layer, `direct`,
/// no fallback, no halved prefill, the slot fills' readers gone, one
/// promotion counted, banks and ledgers of a whole load; the pool, the
/// state room and the wired lease are what they were. Turn 2 (another
/// conversation that shares 98 tokens) resumes at the state turn 1
/// published at 96, and its logits and greedy steps equal, bit for bit, a
/// whole load's and the refused store's that never moved.
/// MUTATION: the move keeps layer 0's partial bank and its slot map → red
/// (turn 2 fails: layer 0 never publishes identity).
/// MUTATION: the move re-plans the dynamic budget → red (state room).
/// MUTATION: the move is not counted → red (promotions 0).
#[test]
fn a_host_gate_store_moves_up_at_its_last_close_and_decodes_as_a_whole_load() {
    let bank = bank_bytes();
    let quiet = 3 << 30;
    let busy = edge(bank) + bank / 24;
    let mut moving = host_gate_model(bank);
    let mut stays = host_gate_model(bank);
    let mut whole = model_direct();
    assert_eq!(whole.decoder.weights.live.residency(), Residency::Direct);

    let a = prompt(68, 1);
    let (_, w1) = whole_turn(&mut whole, 1, &a, 2, quiet);
    let (_, s1) = whole_turn(&mut stays, 1, &a, 2, busy);
    let (cached, p1) = open_turn(&mut moving, 1, &a, 2);
    assert_eq!(cached, 0);
    assert_eq!(p1, s1, "turn 1: the refused stores");
    assert_eq!(p1, w1, "turn 1: refused against whole");
    assert_eq!(moving.prefix_states.counters.published, 2);
    let room = moving.state_room();
    let wired = moving.wired_bytes();
    let dynamic = moving.memory_plan.dynamic_budget_bytes;
    let pool = moving
        .decoder
        .paged
        .as_ref()
        .unwrap()
        .pool_allocated_bytes()
        .unwrap();
    let budget = moving.decoder.weights.cache_budget();
    assert!(moving.decoder.weights.slot_ledger().2, "turn 1 read slots");
    {
        let _host = memory::scoped_test_host(host_with(quiet + own(&moving)));
        moving.set_turn_cancel_flag(None);
    }

    let weights = &moving.decoder.weights;
    let reference = &whole.decoder.weights;
    assert_eq!(promotions(&moving), 1);
    assert_eq!(promotions(&stays), 0);
    assert!(weights.plan.direct && !weights.plan.partial_direct_gate);
    assert_eq!(weights.live.residency(), Residency::Direct);
    assert_eq!(weights.live.fallback(), None);
    assert_eq!(
        weights.live.partial_expert_slots.load(Ordering::Acquire),
        EXPERTS as u64
    );
    assert_eq!(slot_counts(&moving), vec![Some(EXPERTS); LAYERS]);
    assert_eq!(moving.decoder.host_prefill_scratch, None);
    assert_eq!(
        weights.slot_ledger(),
        (reference.slot_ledger().0, Some(EXPERTS), false)
    );
    assert_eq!(weights.plan.direct_bytes, bank);
    assert_eq!(weights.plan.direct_layers, LAYERS);
    assert_eq!(weights.cache_budget(), budget);
    assert_eq!(moving.state_room(), room, "state room");
    assert_eq!(moving.wired_bytes(), wired, "wired lease");
    assert_eq!(moving.memory_plan.dynamic_budget_bytes, dynamic);
    assert_eq!(
        moving
            .decoder
            .paged
            .as_ref()
            .unwrap()
            .pool_allocated_bytes()
            .unwrap(),
        pool
    );
    assert_eq!(stays.decoder.weights.live.residency(), Residency::HostGate);
    let slots = stays.decoder.weights.plan.partial_expert_slots;
    assert_eq!(slot_counts(&stays), vec![Some(slots); LAYERS]);

    let misses = moving.decoder.weights.slot_misses;
    let b = fork(&a, 66, 78, 2);
    let (cached, p2) = whole_turn(&mut moving, 2, &b, 3, quiet);
    assert_eq!(cached, 64, "the state published before the move");
    assert_eq!(moving.prefix_states.counters.restored, 1);
    let (_, w2) = whole_turn(&mut whole, 2, &b, 3, quiet);
    let (_, s2) = whole_turn(&mut stays, 2, &b, 3, busy);
    assert_eq!(p2, w2, "turn 2: moved against a whole load");
    assert_eq!(p2, s2, "turn 2: moved against the refused store");
    assert_eq!(promotions(&moving), 1, "a whole store moves no more");
    assert!(moving.decoder.weights.device_slots_ready(LAYERS));
    assert_eq!(
        moving.decoder.weights.slot_misses, misses,
        "a whole store misses no expert"
    );
}

/// For other processes' memory around the line where a load just admits
/// every bank, a refused store moves up exactly where a fresh load beside
/// the same memory is whole; the host at the close also holds this model's
/// own bytes.
/// MUTATION: the promotion keeps this model's bytes in the host's number →
/// red (no move at the line).
/// MUTATION: the promotion drops the anonymous need → red (a move 1 byte
/// past the line).
#[test]
fn the_promotion_gate_is_the_load_gate_with_this_models_bytes_taken_out() {
    let bank = bank_bytes();
    let line = edge(bank);
    let mut refused = host_gate_model(bank);
    let a = prompt(40, 3);
    open_turn(&mut refused, 1, &a, 2);
    eprintln!("own {} B, banks {bank} B, line {line} B", own(&refused));
    assert!(own(&refused) > 0);
    let cases = [line + (1 << 20), line + 1, line, line - (1 << 20)];
    for (i, other) in cases.into_iter().enumerate() {
        let whole = {
            let mut store = weights::Store::open_fixture(deep_moe_shape_fixture(), None).unwrap();
            store.plan_fixture_store().unwrap();
            store.plan.physical_bytes = Some(PHYSICAL);
            let _host = memory::scoped_test_host(host_with(other));
            store.select_residency(NEED).unwrap();
            store.plan.direct
        };
        assert_eq!(whole, other <= line, "a load beside {other} B");
        let close = host_with(other + own(&refused));
        {
            let _host = memory::scoped_test_host(close);
            refused.set_turn_cancel_flag(None);
        }
        let moved = refused.decoder.weights.plan.direct;
        assert_eq!(moved, whole, "a close beside {other} B");
        if moved && i + 1 < cases.len() {
            refused = host_gate_model(bank);
            open_turn(&mut refused, 1, &a, 2);
        }
    }
}

/// Two scheduled owners: the close of the first leaves the store refused
/// while the second's turn is open; the second's close moves it.
/// MUTATION: the close ignores the open turns → red.
#[test]
fn a_promotion_waits_for_the_last_open_turn() {
    let bank = bank_bytes();
    let mut inner = host_gate_model(bank);
    let close = host_with((3 << 30) + own(&inner));
    let _host = memory::scoped_test_host(close);
    inner.scheduled_turn_opened(1);
    inner.scheduled_turn_opened(2);
    inner.scheduled_turn_closed(1);
    inner.set_turn_cancel_flag(None);
    assert_eq!(
        inner.decoder.weights.live.residency(),
        Residency::HostGate,
        "a turn is open"
    );
    assert_eq!(promotions(&inner), 0);
    inner.scheduled_turn_closed(2);
    inner.set_turn_cancel_flag(None);
    assert_eq!(inner.decoder.weights.live.residency(), Residency::Direct);
    assert_eq!(promotions(&inner), 1);
}

/// A refused store whose cap the engine plan lowered, a store the plan
/// itself streams, and a load with the MTP head keep their slots on a
/// host with room.
/// MUTATION: the cap check is dropped → red (lowered cap moves).
/// MUTATION: a store-plan partial is taken as a refused gate → red.
/// MUTATION: the MTP check is dropped → red.
#[test]
fn a_lowered_cap_a_store_plan_partial_and_an_mtp_load_never_move() {
    let bank = bank_bytes();
    let quiet = 3 << 30;

    let mut lowered = host_gate_model(bank);
    assert!(lowered.decoder.weights.lower_store_cap(1).unwrap());

    let (mut store, c) = super::route_step::streamed_moe_store(true, 11);
    store.plan.physical_bytes = Some(PHYSICAL);
    let mut streamed = model_over_store(store, c);
    assert_eq!(
        streamed.decoder.weights.live.residency(),
        Residency::StorePlan
    );

    let mut mtp = host_gate_model(bank);
    mtp.has_mtp = true;

    for (name, inner, kept) in [
        ("lowered cap", &mut lowered, Residency::HostGate),
        ("store plan", &mut streamed, Residency::StorePlan),
        ("mtp", &mut mtp, Residency::HostGate),
    ] {
        {
            let _host = memory::scoped_test_host(host_with(quiet + own(inner)));
            inner.set_turn_cancel_flag(None);
        }
        assert_eq!(inner.decoder.weights.live.residency(), kept, "{name}");
        assert!(!inner.decoder.weights.plan.direct, "{name}");
        assert_eq!(promotions(inner), 0, "{name}");
        assert!(
            matches!(
                inner.decoder.weights.live.fallback(),
                Some(Fallback::LoadGate { .. } | Fallback::StorePlan { .. })
            ),
            "{name}"
        );
    }
}

/// The move's own check passes, then the host refuses the third layer's
/// whole bank (it changed during the move): the move stops, the
/// layers it reached lose their banks, the rest keep their slots, the plan
/// and the residency stay partial, and the next turn runs on the old slot
/// count with the logits and greedy steps of a store that never tried.
/// MUTATION: the layers already moved keep their whole banks → red.
#[test]
fn a_refused_allocation_half_way_leaves_a_partial_store_for_the_next_turn() {
    let bank = bank_bytes();
    let busy = edge(bank) + bank / 24;
    let mut tried = host_gate_model(bank);
    let mut never = host_gate_model(bank);
    let a = prompt(68, 1);
    whole_turn(&mut tried, 1, &a, 2, busy);
    whole_turn(&mut never, 1, &a, 2, busy);
    let (before, ..) = tried.decoder.weights.slot_ledger();
    let slots = tried.decoder.weights.plan.partial_expert_slots;
    assert_eq!(slot_counts(&tried), vec![Some(slots); LAYERS]);
    let mut calls = 0;
    let error = {
        let _host = memory::scoped_test_host(host_with((3 << 30) + own(&tried)));
        tried
            .promote_store(&mut |_| {
                calls += 1;
                if calls == 4 {
                    Err(Error::from_reason("injected refusal of layer 2's bank"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err()
    };
    assert!(error.reason.contains("layer 2's bank"), "{error}");
    let weights = &tried.decoder.weights;
    assert!(!weights.plan.direct && weights.plan.partial_direct_gate);
    assert_eq!(weights.live.residency(), Residency::HostGate);
    assert_eq!(
        weights.live.partial_expert_slots.load(Ordering::Acquire),
        slots as u64
    );
    assert_eq!(promotions(&tried), 0);
    let mut kept = vec![None; 3];
    kept.extend([Some(slots); LAYERS - 3]);
    assert_eq!(slot_counts(&tried), kept);
    let (after, capacity, _) = weights.slot_ledger();
    assert_eq!(capacity, Some(slots));
    assert!(after > 0 && after < before, "{after} of {before} B left");
    assert!(tried.decoder.host_prefill_scratch.is_some());

    let b = fork(&a, 66, 78, 2);
    let (cached, got) = whole_turn(&mut tried, 2, &b, 3, busy);
    let (_, want) = whole_turn(&mut never, 2, &b, 3, busy);
    assert_eq!(cached, 64);
    assert_eq!(got, want);
    assert_eq!(slot_counts(&tried), vec![Some(slots); LAYERS]);
    assert_eq!(
        tried.decoder.weights.slot_ledger().0,
        never.decoder.weights.slot_ledger().0
    );
}

/// Near the line the gate admits the move, but the host cannot hold the
/// last layer's whole bank beside the layers before it. A host model: a
/// layer's admission sees `room` plus the partial banks dropped up to that
/// layer, less the whole banks allocated before it (a partial bank is the
/// whole one at `slots` of the experts' rows). One byte under the last
/// layer's need, the move refuses before any bank drops: every slot keeps
/// its expert, the ledger and the residency are the refused store's. At
/// the need every layer's admission passes and the store moves.
/// MUTATION: the move skips the check of its largest step → red (the last
/// layer is refused after every other layer dropped its warm slots).
#[test]
fn a_move_the_host_cannot_finish_keeps_every_warm_slot() {
    let bank = bank_bytes();
    let busy = edge(bank) + bank / 24;
    let quiet = 3 << 30;
    let whole = model_direct();
    assert_eq!(
        whole.decoder.weights.slot_ledger().0,
        bank,
        "a whole load's slot banks are the gate's bank set"
    );
    drop(whole);
    let staging = 3 * weights::MAX_READ_BYTES;
    let a = prompt(68, 1);
    let mut tried = host_gate_model(bank);
    whole_turn(&mut tried, 1, &a, 2, busy);
    let slots = tried.decoder.weights.plan.partial_expert_slots as u64;
    let (partial_bytes, ..) = tried.decoder.weights.slot_ledger();
    assert_eq!(slot_counts(&tried), vec![Some(slots as usize); LAYERS]);
    let warm: Vec<_> = (0..LAYERS)
        .map(|layer| tried.decoder.weights.slot_occupants(layer))
        .collect();
    assert!(warm.iter().flatten().flatten().any(Option::is_some));
    let growth = bank - partial_bytes;
    for (short, moves) in [(1u64, false), (0, true)] {
        let room = growth + staging - short;
        let mut asked = Vec::new();
        let mut dropped = 0u64;
        let mut allocated = 0u64;
        let mut layer_calls = 0;
        let moved = {
            let _host = memory::scoped_test_host(host_with(quiet + own(&tried)));
            tried
                .promote_store(&mut |bytes| {
                    asked.push(bytes);
                    let free = if bytes == growth + staging {
                        room
                    } else {
                        let whole = bytes - staging;
                        assert_eq!(whole * slots % EXPERTS as u64, 0, "{whole} B");
                        dropped += whole * slots / EXPERTS as u64;
                        layer_calls += 1;
                        let free = room + dropped - allocated;
                        allocated += whole;
                        free
                    };
                    if bytes <= free {
                        Ok(())
                    } else {
                        Err(Error::from_reason("the host holds no more"))
                    }
                })
                .unwrap()
        };
        eprintln!(
            "room {room} B (short {short}): asked {asked:?}, moved {moved}, partial {partial_bytes} B, layer admissions {layer_calls}, modelled partial {dropped} B"
        );
        assert_eq!(moved, moves, "short {short}");
        let weights = &tried.decoder.weights;
        if moves {
            assert_eq!(asked.len(), 1 + LAYERS);
            assert_eq!(layer_calls, LAYERS);
            assert_eq!(dropped, partial_bytes, "the model's partial banks");
            assert_eq!(allocated, bank);
            assert_eq!(weights.live.residency(), Residency::Direct);
            assert_eq!(slot_counts(&tried), vec![Some(EXPERTS); LAYERS]);
            assert_eq!(promotions(&tried), 1);
        } else {
            assert_eq!(asked, vec![growth + staging]);
            assert_eq!(weights.live.residency(), Residency::HostGate);
            assert!(!weights.plan.direct && weights.plan.partial_direct_gate);
            assert_eq!(weights.slot_ledger().0, partial_bytes);
            let kept: Vec<_> = (0..LAYERS)
                .map(|layer| weights.slot_occupants(layer))
                .collect();
            assert_eq!(kept, warm, "every warm slot stays");
            assert_eq!(promotions(&tried), 0);
            assert!(tried.decoder.host_prefill_scratch.is_some());
        }
    }
}

fn promotion_refusals(inner: &Inner) -> u64 {
    inner
        .decoder
        .weights
        .live
        .promotion_refusals
        .load(Ordering::Acquire)
}

/// A close the gate refuses counts nothing. A close the gate admits but
/// whose growth the host cannot take counts one refusal and keeps the
/// partial store; a later close that moves counts a promotion and no
/// refusal.
/// MUTATION: the refusal before any bank drops is not counted → red.
#[test]
fn a_move_the_host_refuses_before_any_bank_drops_is_counted() {
    let bank = bank_bytes();
    let busy = edge(bank) + bank / 24;
    let quiet = 3 << 30;
    let mut inner = host_gate_model(bank);
    whole_turn(&mut inner, 1, &prompt(68, 1), 2, busy);
    let residency = |inner: &Inner| inner.decoder.weights.live.residency();
    assert_eq!(residency(&inner), Residency::HostGate);
    assert_eq!((promotions(&inner), promotion_refusals(&inner)), (0, 0));
    let moved = {
        let _host = memory::scoped_test_host(host_with(quiet + own(&inner)));
        inner
            .promote_store(&mut |_| Err(Error::from_reason("the host holds no more")))
            .unwrap()
    };
    assert!(!moved);
    assert_eq!(residency(&inner), Residency::HostGate);
    assert_eq!((promotions(&inner), promotion_refusals(&inner)), (0, 1));
    let moved = {
        let _host = memory::scoped_test_host(host_with(quiet + own(&inner)));
        inner.promote_store(&mut |_| Ok(())).unwrap()
    };
    assert!(moved);
    assert_eq!(residency(&inner), Residency::Direct);
    assert_eq!((promotions(&inner), promotion_refusals(&inner)), (1, 1));
}

/// Pass size alone moves bits, so a state the refused store wrote on halved
/// passes must not serve the moved store. Turn 1 on a refused store halving
/// past 64 rows publishes states at 64 and 96; after the move, turn 2 (a
/// fork sharing 98 rows) resumes at 64, not 96, and equals a cold whole load.
/// MUTATION pass-key: a block's key leaves out the pass ends inside it → red.
#[test]
fn a_state_written_on_halved_passes_does_not_serve_the_moved_store() {
    use crate::engine::backend::ResetScope;
    let bits = |rows: &[Vec<f32>]| -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| row.iter().map(|v| v.to_bits()).collect())
            .collect()
    };
    let bank = bank_bytes();
    let quiet = host_with(3 << 30);
    let open = |inner: &mut Inner| {
        inner.set_turn_cancel_flag(Some(Arc::new(AtomicBool::new(false))));
    };
    let halve = |inner: &mut Inner| {
        let config = Arc::clone(&inner.decoder.config);
        inner.decoder.host_prefill_scratch = Some(memory::prefill_scratch(&config, 32, 64));
    };
    let a = prompt(100, 1);
    let b = fork(&a, 98, 110, 2);
    let mut whole = model_direct();
    let mut cold_prompt = |halved: bool| -> (Vec<usize>, Vec<Vec<u32>>) {
        whole.reset_caches(ResetScope::Command).unwrap();
        whole.prefix_states.reset(true);
        whole.decoder.pass_samples.clear();
        whole.decoder.host_prefill_scratch = None;
        if halved {
            halve(&mut whole);
        }
        let _host = memory::scoped_test_host(quiet);
        open(&mut whole);
        let (cached, logits) = turn(&mut whole, 1, &a);
        assert_eq!(cached, 0);
        let rows = whole.decoder.pass_samples.iter().map(|&(r, _)| r).collect();
        (rows, bits(&[logits]))
    };
    let (whole_rows, whole_bits) = cold_prompt(false);
    let (halved_rows, halved_bits) = cold_prompt(true);
    assert_eq!(whole_rows, [32, 32, 32, 4]);
    assert_eq!(halved_rows, [32, 32, 16, 16, 4]);
    assert!(
        whole_bits != halved_bits,
        "halved passes gave the whole passes' bits"
    );

    let mut moving = host_gate_model(bank);
    assert!(moving.decoder.weights.plan.partial_direct_gate);
    halve(&mut moving);
    {
        let _host = memory::scoped_test_host(quiet);
        open(&mut moving);
        let (_, logits) = turn(&mut moving, 1, &a);
        greedy(&mut moving, &logits, 2);
        finish(&mut moving);
    }
    let mut depths: Vec<usize> = moving
        .prefix_states
        .store
        .entries()
        .map(|e| e.depth())
        .collect();
    depths.sort_unstable();
    assert_eq!(depths, [64, 96], "turn 1's states");
    {
        let _host = memory::scoped_test_host(host_with((3 << 30) + own(&moving)));
        moving.set_turn_cancel_flag(None);
    }
    assert_eq!(moving.decoder.weights.live.residency(), Residency::Direct);
    let (cached, moved) = {
        let _host = memory::scoped_test_host(quiet);
        open(&mut moving);
        let (cached, logits) = turn(&mut moving, 2, &b);
        let mut out = vec![logits.clone()];
        out.extend(greedy(&mut moving, &logits, 3));
        (cached, out)
    };
    let cold = {
        whole.reset_caches(ResetScope::Command).unwrap();
        whole.prefix_states.reset(true);
        whole.decoder.host_prefill_scratch = None;
        let _host = memory::scoped_test_host(quiet);
        open(&mut whole);
        let (cached, logits) = turn(&mut whole, 1, &b);
        assert_eq!(cached, 0);
        let mut out = vec![logits.clone()];
        out.extend(greedy(&mut whole, &logits, 3));
        out
    };
    assert_eq!(cached, 64, "the state at 96 was written on halved passes");
    assert!(
        bits(&moved) == bits(&cold),
        "after the move: restored text = no-cache text"
    );
}
