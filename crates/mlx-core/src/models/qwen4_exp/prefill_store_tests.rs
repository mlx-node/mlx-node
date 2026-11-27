use super::*;

/// `[fillCalls, fillMicros, fillReadBytes, slotMisses, checkpointBytesRead,
/// groupPasses, groupWindows]` as `residencyInfo` reads them.
fn records(live: &live::Live) -> [u64; 7] {
    [
        live.fill_calls.load(Ordering::Relaxed),
        live.fill_micros.load(Ordering::Relaxed),
        live.fill_read_bytes.load(Ordering::Relaxed),
        live.slot_misses.load(Ordering::Relaxed),
        live.bytes_read.load(Ordering::Relaxed),
        live.group_passes.load(Ordering::Relaxed),
        live.group_windows.load(Ordering::Relaxed),
    ]
}

fn delta(after: [u64; 7], before: [u64; 7]) -> [u64; 7] {
    std::array::from_fn(|i| after[i] - before[i])
}

/// The bytes of the read windows the store recorded from `from` on.
fn window_bytes(decoder: &decoder::Decoder, from: usize) -> u64 {
    decoder.weights.direct_read_windows[from..]
        .iter()
        .map(|&(_, len)| len as u64)
        .sum()
}

/// On a streamed store of 3 slots per layer (4 experts, top 2), a lease
/// with two misses is one slot fill that takes time, reads two experts and
/// counts the 16 KiB pages it read as fill bytes (more than the wanted
/// bytes `checkpointBytesRead` counts). A one-window pass adds no group; a
/// group of three windows adds one group pass and three group windows, and
/// the fill bytes of its leases stay the bytes of their reads.
/// MUTATION: a group adds its window count to `groupPasses` -> 3, not 1 -> red.
/// MUTATION: a one-window pass counts as a group -> red.
/// MUTATION: `fillReadBytes` adds the wanted bytes, not the read windows -> red.
/// MUTATION: a fill counts one call per expert -> 2, not 1 -> red.
#[test]
fn fill_counters_follow_a_lease() {
    assert!(crate::utils::gguf_gpu::gpu_repack_available());
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let store = weights::Store::open_fixture(&path, None).unwrap();
    let mut decoder = decoder::Decoder::new(config.clone(), store).unwrap();
    decoder.weights.stream_fixture_store(3).unwrap();
    decoder.register_cache_limit();
    decoder.weights.publish_live();
    decoder.residency_changed();
    let live = Arc::clone(&decoder.weights.live);
    let load = records(&live);
    assert_eq!(
        [load[0], load[1], load[2], load[3], load[5], load[6]],
        [0; 6],
        "the load fills no slot and runs no group"
    );

    decoder.weights.begin_turn().unwrap();
    let windows = decoder.weights.direct_read_windows.len();
    let before = records(&live);
    let (_, local, capacity) = decoder
        .weights
        .expert_slots(0, config.num_experts, config.num_experts_per_tok, &[0, 2])
        .unwrap()
        .expect("two experts fit three slots");
    assert_eq!(capacity, 3);
    assert_eq!(local.len(), 2);
    let lease = delta(records(&live), before);
    let read = window_bytes(&decoder, windows);
    assert_eq!(lease[0], 1, "two misses are one fill");
    assert!(lease[1] > 0, "the fill's wall time is counted");
    assert_eq!(lease[3], 2, "two slot misses");
    assert_eq!(lease[2], read, "fill bytes are the read windows' bytes");
    assert!(
        lease[2] > lease[4] && lease[4] > 0,
        "the windows hold more than the wanted bytes: {} vs {}",
        lease[2],
        lease[4]
    );
    assert_eq!([lease[5], lease[6]], [0, 0], "a lease is no group");

    let prompt: Vec<u32> = (0..64)
        .map(|i| ((i * 13 + i / 4) % 61 + 2) as u32)
        .collect();
    decoder.prefill_window_pin = Some(16);
    decoder.prefill_chunk_size = 16;
    let mut adapter = paged::create_fixture(&config).unwrap();
    adapter.begin_request(1).unwrap();
    adapter.allocate_suffix_blocks(64).unwrap();
    decoder.paged = Some(adapter);

    let windows = decoder.weights.direct_read_windows.len();
    let before = records(&live);
    decoder
        .prefill_group(&[&prompt[..16]], &[], true, false)
        .unwrap();
    let one = delta(records(&live), before);
    assert_eq!([one[5], one[6]], [0, 0], "a one-window pass adds no group");

    let before = records(&live);
    let group: Vec<&[u32]> = prompt[16..].chunks(16).collect();
    decoder.prefill_group(&group, &[], true, false).unwrap();
    let grouped = delta(records(&live), before);
    assert_eq!(
        [grouped[5], grouped[6]],
        [1, 3],
        "one group pass of three windows"
    );
    assert_eq!(decoder.history, prompt);
    assert!(one[3] + grouped[3] > 0, "the prompt's leases read experts");
    assert_eq!(
        one[2] + grouped[2],
        window_bytes(&decoder, windows),
        "the prompt's fill bytes are its read windows' bytes"
    );
    eprintln!(
        "lease [calls, micros, fill bytes, misses, wanted bytes] {:?}; prompt one-window {:?}, group {:?}",
        &lease[..5],
        &one[..5],
        &grouped[..5]
    );
}

/// `count` prompt tokens inside the fixtures' vocabulary.
fn prompt(count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| ((i * 13 + i / 4) % 61 + 2) as u32)
        .collect()
}

/// The machine runs the tentative prefill kernels: the gate's own host
/// conditions besides the store.
fn tentative_kernels_available() -> bool {
    crate::engine::persistence::compiled_forward_backend_available()
        && unsafe { mlx_sys::mlx_metal_is_nax_available() }
        && super::super::tf32_enabled()
}

/// A partial store holds what a tentative prefill window needs only when
/// every layer maps every expert: three slots of four experts, every slot
/// filled, never do; four slots holding all four do. The bank-mode half of
/// the gate stays its own check: this fixture's Q5_1 banks fail it, so the
/// whole gate is shut here (`a_partial_store_group_never_goes_tentative`
/// opens it on the Flash Next shape).
/// MUTATION: keep the old rule "every layer's first slot is occupied" ->
/// true at three slots -> red.
#[test]
fn a_store_that_can_miss_tries_no_tentative_prefill() {
    assert!(crate::utils::gguf_gpu::gpu_repack_available());
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let (layers, experts) = (config.num_hidden_layers, config.num_experts);
    assert_eq!(experts, 4);
    for capacity in [3, 4] {
        let store = weights::Store::open_fixture(&path, None).unwrap();
        let mut decoder = decoder::Decoder::new(config.clone(), store).unwrap();
        decoder.weights.stream_fixture_store(capacity).unwrap();
        decoder.register_cache_limit();
        decoder.weights.publish_live();
        decoder.residency_changed();
        decoder.weights.begin_turn().unwrap();
        let store = &mut decoder.weights;
        assert!(!store.plan.direct);
        let held: Vec<u32> = (0..capacity as u32).collect();
        for layer in 0..layers {
            assert!(
                !store.slots_hold_every_expert(layers),
                "capacity {capacity}: layer {layer} has no slot yet"
            );
            store
                .expert_slots(layer, experts, config.num_experts_per_tok, &held)
                .unwrap()
                .expect("the held experts fit the slots");
            let occupants = store.slot_occupants(layer).unwrap();
            assert!(occupants.iter().all(Option::is_some), "every slot filled");
        }
        assert_eq!(
            store.slots_hold_every_expert(layers),
            capacity == experts,
            "capacity {capacity} of {experts} experts"
        );
        assert!(
            !store.prefill_device_slots_ready(layers),
            "Q5_1 banks fail the fused bank modes"
        );
    }
}

/// On the Flash Next shape (banks that pass the fused modes) a partial
/// store of 11 slots for 12 experts, every slot filled, runs a group of two
/// 256-row windows and then a one-window pass on the plain path: no
/// tentative commit, no replay. The same shape at 12 slots holding all
/// twelve runs both tentatively, and both hit.
/// MUTATION: the gate drops the residency rule -> the 11-slot group runs
/// tentatively (and fails on a missed expert) -> red.
#[test]
fn a_partial_store_group_never_goes_tentative() {
    if !tentative_kernels_available() {
        eprintln!("skipped: no tentative prefill kernels on this machine");
        return;
    }
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let prompt = prompt(768);
    for capacity in [11, 12] {
        let mut inner = super::route_step::moe_shape_inner(capacity);
        let c = inner.decoder.config.clone();
        let full = capacity == c.num_experts;
        // The fixture's trained window is 256 rows; three windows need 768.
        inner.decoder.context_window = prompt.len();
        inner.activate_paged_seq(1).unwrap();
        let held: Vec<u32> = (0..capacity as u32).collect();
        super::route_step::fill_slots(&mut inner.decoder, &held);
        let gate = inner
            .decoder
            .weights
            .prefill_device_slots_ready(c.num_hidden_layers);
        let live = Arc::clone(&inner.decoder.weights.live);
        let replays = live.route_replays.load(Ordering::Relaxed);
        inner.decoder.tentative_commits.clear();
        let grouped =
            inner
                .decoder
                .prefill_group(&[&prompt[..256], &prompt[256..512]], &[], true, false);
        assert_eq!(
            inner.decoder.tentative_commits,
            if full { vec![2] } else { vec![] },
            "capacity {capacity}: the group's tentative commits"
        );
        grouped.unwrap_or_else(|error| panic!("capacity {capacity}: {}", error.reason));
        assert_eq!(gate, full, "capacity {capacity}: the gate");
        inner
            .decoder
            .prefill_chunk(&prompt[512..], None, true)
            .unwrap();
        assert_eq!(
            inner.decoder.tentative_commits,
            if full { vec![2, 1] } else { vec![] },
            "capacity {capacity}: the window's tentative commit"
        );
        assert_eq!(inner.decoder.history, prompt);
        assert_eq!(
            live.route_replays.load(Ordering::Relaxed),
            replays,
            "capacity {capacity}: no window replayed"
        );
    }
}

/// A 256-row window on the Flash Next shape has the same bits on the
/// tentative device path (12 slots holding all 12 experts) as on the plain
/// path (the same store with no slot filled, so the gate is shut and the
/// window fills its slots): logits, history, every state array, the last
/// hidden rows, and the int8 pages with their scales.
/// MUTATION: the tentative path combines its rows in sorted order -> red.
#[test]
fn a_tentative_window_equals_the_ordinary_window() {
    if !tentative_kernels_available() {
        eprintln!("skipped: no tentative prefill kernels on this machine");
        return;
    }
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let prompt = prompt(256);
    let mut tentative = super::route_step::moe_shape_inner(12);
    let mut ordinary = super::route_step::moe_shape_inner(12);
    let c = tentative.decoder.config.clone();
    let all: Vec<u32> = (0..c.num_experts as u32).collect();
    tentative.activate_paged_seq(1).unwrap();
    ordinary.activate_paged_seq(1).unwrap();
    super::route_step::fill_slots(&mut tentative.decoder, &all);
    let layers = c.num_hidden_layers;
    assert!(tentative.decoder.weights.prefill_device_slots_ready(layers));
    assert!(!ordinary.decoder.weights.prefill_device_slots_ready(layers));
    let got = tentative
        .decoder
        .prefill_chunk(&prompt, None, true)
        .unwrap();
    let want = ordinary.decoder.prefill_chunk(&prompt, None, true).unwrap();
    assert_eq!(tentative.decoder.tentative_commits, [1]);
    assert!(ordinary.decoder.tentative_commits.is_empty());
    for inner in [&tentative, &ordinary] {
        let live = &inner.decoder.weights.live;
        assert_eq!(live.route_replays.load(Ordering::Relaxed), 0);
    }
    assert_same_bits(&got, &want, "logits");
    super::route_step::assert_same_decoder_state(
        &mut tentative.decoder,
        &mut ordinary.decoder,
        "tentative vs plain window",
    );
}

/// The documented victim rule over one layer's book (occupants, ages,
/// tick) for a lease of `ids`: the missing ids in id order each take, among
/// the slots this lease has not taken whose occupant it does not want, an
/// empty slot first, else the least recent (ties to the lower slot); then
/// every id of `ids` in order ticks its slot. Returns the misses.
fn rule_lease(book: &mut (Vec<Option<u32>>, Vec<u64>, u64), ids: &[u32]) -> usize {
    let (occupants, ages, tick) = book;
    let wanted: std::collections::HashSet<u32> = ids.iter().copied().collect();
    let mut missing: Vec<u32> = wanted
        .iter()
        .copied()
        .filter(|&e| !occupants.contains(&Some(e)))
        .collect();
    missing.sort_unstable();
    let mut chosen: Vec<usize> = Vec::new();
    for _ in &missing {
        let slot = (0..occupants.len())
            .filter(|s| !chosen.contains(s) && occupants[*s].is_none_or(|e| !wanted.contains(&e)))
            .min_by_key(|&s| {
                if occupants[s].is_none() {
                    0
                } else {
                    ages[s] + 1
                }
            })
            .expect("the lease fits the slots");
        chosen.push(slot);
    }
    for (&expert, &slot) in missing.iter().zip(&chosen) {
        occupants[slot] = Some(expert);
    }
    for &id in ids {
        let slot = occupants.iter().position(|&e| e == Some(id)).unwrap();
        *tick += 1;
        ages[slot] = *tick;
    }
    missing.len()
}

/// The rows of every occupied slot of every layer's banks are the same bits
/// in `a` and `b` (the slots hold the same experts).
fn assert_same_slot_banks(a: &decoder::Decoder, b: &decoder::Decoder, what: &str) {
    for layer in 0..a.config.num_hidden_layers {
        let occupants = a.weights.slot_occupants(layer);
        assert_eq!(
            occupants,
            b.weights.slot_occupants(layer),
            "{what}: layer {layer} occupants"
        );
        let Some(occupants) = occupants else {
            continue;
        };
        let (x, y) = (
            a.weights.slot_banks(layer).unwrap(),
            b.weights.slot_banks(layer).unwrap(),
        );
        for part in 0..3 {
            let arrays = [
                (Some(&x[part].values), Some(&y[part].values)),
                (x[part].scales.as_ref(), y[part].scales.as_ref()),
                (x[part].biases.as_ref(), y[part].biases.as_ref()),
            ];
            for (plane, (x, y)) in arrays.into_iter().enumerate() {
                let (Some(x), Some(y)) = (x, y) else {
                    assert!(
                        x.is_none() && y.is_none(),
                        "{what}: layer {layer} part {part}"
                    );
                    continue;
                };
                let rows = x.shape().unwrap()[0] / occupants.len() as i64;
                for (slot, _) in occupants.iter().enumerate().filter(|(_, e)| e.is_some()) {
                    let (lo, hi) = (slot as i64 * rows, (slot as i64 + 1) * rows);
                    assert_same_bits(
                        &x.slice_axis(0, lo, hi).unwrap(),
                        &y.slice_axis(0, lo, hi).unwrap(),
                        &format!("{what}: layer {layer} part {part} plane {plane} slot {slot}"),
                    );
                }
            }
        }
    }
}

/// Prefill a 32-row window and a 12-row tail (two chunks so recovery is
/// checked across a subsequent chunk with live decoder state), then decode
/// `steps` greedy tokens, on both: the logits of every call and the decoder
/// state end equal bit for bit.
fn assert_same_turn(a: &mut Inner, b: &mut Inner, steps: usize, what: &str) {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let prompt = prompt(44);
    for inner in [&mut *a, &mut *b] {
        inner.decoder.context_window = 768;
        inner.activate_paged_seq(1).unwrap();
    }
    let mut logits = Vec::new();
    for inner in [&mut *a, &mut *b] {
        let mut out = vec![
            inner
                .decoder
                .prefill_chunk(&prompt[..32], None, true)
                .unwrap(),
            inner
                .decoder
                .prefill_chunk(&prompt[32..], None, true)
                .unwrap(),
        ];
        for _ in 0..steps {
            let token = argmax(out.last().unwrap());
            out.push(inner.decoder.step(token).unwrap());
        }
        logits.push(out);
    }
    for (call, (x, y)) in logits[0].iter().zip(&logits[1]).enumerate() {
        assert_same_bits(x, y, &format!("{what}: logits of call {call}"));
    }
    super::route_step::assert_same_decoder_state(&mut a.decoder, &mut b.decoder, what);
    assert_same_slot_banks(&a.decoder, &b.decoder, what);
}

/// On the Flash Next shape (12 experts, 11 slots) a scripted lease
/// sequence with 10, 2, 11 and 1 misses runs one fill per lease (the old
/// loop: 2, 1, 2, 1), and after every lease the occupants and ages of both
/// equal the documented rule. A turn after it (prefill and decode) leaves
/// the same logits, state and slot bytes as the old loop.
/// MUTATION: protect only the first 8 wanted ids -> a wanted expert is
/// taken as a victim -> the lease fails -> red.
/// MUTATION: a band repacks from its range's own rounded start, not from
/// where its merged read put it -> slot bytes red.
#[test]
fn one_lease_runs_one_fill_with_the_chunked_victims() {
    let _host = memory::scoped_fixture_host();
    const CAPACITY: usize = 11;
    let mut new = super::route_step::moe_shape_inner(CAPACITY);
    let mut old = super::route_step::moe_shape_inner(CAPACITY);
    old.decoder.weights.old_slot_fill = true;
    let c = new.decoder.config.clone();
    let (experts, top) = (c.num_experts, c.num_experts_per_tok);
    assert_eq!(experts, 12);
    let leases: [(usize, &[u32]); 4] = [
        (0, &[9, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (0, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        (1, &[10, 11, 0, 3, 8, 1, 5, 2, 7, 4, 6]),
        (1, &[9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
    ];
    let mut books = vec![(vec![None; CAPACITY], vec![0u64; CAPACITY], 0u64); c.num_hidden_layers];
    let before = [&new, &old].map(|inner| records(&inner.decoder.weights.live));
    let (mut leases_with_a_miss, mut chunks, mut misses) = (0u64, 0u64, 0u64);
    for (layer, ids) in leases {
        let missed = rule_lease(&mut books[layer], ids);
        leases_with_a_miss += u64::from(missed > 0);
        chunks += missed.div_ceil(8) as u64;
        misses += missed as u64;
        for (name, inner) in [("one fill", &mut new), ("chunks of 8", &mut old)] {
            let store = &mut inner.decoder.weights;
            store
                .expert_slots(layer, experts, top, ids)
                .unwrap()
                .expect("the lease fits the slots");
            assert_eq!(
                store.slot_occupants(layer).unwrap(),
                books[layer].0,
                "{name}: layer {layer} occupants after {ids:?}"
            );
            assert_eq!(
                store.slot_ages(layer).unwrap(),
                (books[layer].1.clone(), books[layer].2),
                "{name}: layer {layer} ages after {ids:?}"
            );
        }
    }
    assert_eq!((leases_with_a_miss, chunks, misses), (4, 6, 24));
    let [one, eight] = [&new, &old].map(|inner| records(&inner.decoder.weights.live));
    let (one, eight) = (delta(one, before[0]), delta(eight, before[1]));
    assert_eq!([one[0], one[3]], [4, 24], "one fill per lease with a miss");
    assert_eq!(
        [eight[0], eight[3]],
        [6, 24],
        "the old loop: a fill per 8 misses"
    );
    assert_same_slot_banks(&new.decoder, &old.decoder, "after the leases");
    assert_same_turn(&mut new, &mut old, 2, "one fill vs chunks of 8");
}

/// Five slot fills (four layers of 11 experts, then one more expert) on
/// 2 MiB batches, so each fill wraps the ring several times, start the
/// readers once (8 threads) and allocate the ring once (4 buffers); the
/// slots hold the old loop's bytes and a turn after it is the old loop's.
/// MUTATION: build the readers and ring per fill -> 40 threads -> red.
#[test]
fn slot_fills_reuse_the_readers_and_the_staging() {
    let _host = memory::scoped_fixture_host();
    let mut new = super::route_step::moe_shape_inner(11);
    let mut old = super::route_step::moe_shape_inner(11);
    new.decoder.weights.fill_batch_bytes = Some(2 << 20);
    old.decoder.weights.old_slot_fill = true;
    let c = new.decoder.config.clone();
    let held: Vec<u32> = (0..11).collect();
    let before = records(&new.decoder.weights.live);
    for inner in [&mut new, &mut old] {
        super::route_step::fill_slots(&mut inner.decoder, &held);
        inner
            .decoder
            .weights
            .expert_slots(0, c.num_experts, c.num_experts_per_tok, &[11])
            .unwrap()
            .expect("one expert fits");
    }
    let fills = delta(records(&new.decoder.weights.live), before);
    assert_eq!(fills[0], c.num_hidden_layers as u64 + 1, "fill calls");
    assert_eq!(
        new.decoder.weights.slot_fill_builds,
        [8, 4],
        "[reader threads, staging buffers]"
    );
    assert_eq!(old.decoder.weights.slot_fill_builds, [0, 0]);
    let reads = &new.decoder.weights.direct_read_windows;
    assert!(
        reads.iter().all(|&(_, len)| len <= 2 << 20),
        "a read past its batch"
    );
    assert!(
        fills[2] / fills[0] > 4 * (2 << 20),
        "a fill of {} bytes fits the ring once",
        fills[2] / fills[0]
    );
    assert_same_slot_banks(&new.decoder, &old.decoder, "after five fills");
    assert_same_turn(&mut new, &mut old, 2, "reused ring vs the old loop");
    assert_eq!(
        new.decoder.weights.slot_fill_builds[0], 8,
        "the turn's fills start no reader"
    );
    assert!(new.decoder.weights.slot_fill_builds[1] <= 4);
}

/// A slot fill whose second batch fails (as on a read error) returns the
/// error and leaves no slot of its layer mapped (its victims may hold
/// part of the new expert), the other layers as they were, and readers that
/// serve the next fill: that layer refills and a turn is a fresh store's.
/// MUTATION: the readers are not recovered after the failure -> the refill
/// fails -> red.
/// MUTATION: the failed layer keeps its mapping -> red.
#[test]
fn a_failed_read_leaves_committed_slots_valid() {
    let _host = memory::scoped_fixture_host();
    let mut failed = super::route_step::moe_shape_inner(11);
    let mut fresh = super::route_step::moe_shape_inner(11);
    failed.decoder.weights.fill_batch_bytes = Some(2 << 20);
    let c = failed.decoder.config.clone();
    let held: Vec<u32> = (0..11).collect();
    super::route_step::fill_slots(&mut failed.decoder, &held);
    super::route_step::fill_slots(&mut fresh.decoder, &held);
    let layers = c.num_hidden_layers;
    let kept: Vec<_> = (0..layers)
        .map(|layer| failed.decoder.weights.slot_occupants(layer))
        .collect();
    failed.decoder.weights.fail_fill_batch = Some(1);
    let error = failed
        .decoder
        .weights
        .expert_slots(1, c.num_experts, c.num_experts_per_tok, &[11])
        .err()
        .expect("the second batch fails");
    assert!(
        error.reason.contains("injected read failure at batch 1"),
        "{error}"
    );
    failed.decoder.weights.fail_fill_batch = None;
    for (layer, kept) in kept.iter().enumerate() {
        let occupants = failed.decoder.weights.slot_occupants(layer);
        if layer == 1 {
            assert!(
                occupants.unwrap().iter().all(Option::is_none),
                "the failed layer maps no slot"
            );
        } else {
            assert_eq!(&occupants, kept, "layer {layer} kept its slots");
        }
    }
    super::route_step::fill_slots(&mut failed.decoder, &held);
    assert_same_slot_banks(
        &failed.decoder,
        &fresh.decoder,
        "refilled after the failure",
    );
    assert_same_turn(&mut failed, &mut fresh, 2, "after a failed fill vs fresh");
}

/// The six counters a prefill pass moves, then their prefill shares.
fn shares(live: &live::Live) -> [[u64; 6]; 2] {
    let pairs = live.prefill_shares();
    [
        pairs.map(|(total, _)| total.load(Ordering::Relaxed)),
        pairs.map(|(_, share)| share.load(Ordering::Relaxed)),
    ]
}

/// On the Flash Next shape at 10 slots, a 256-row window and its 44-row
/// tail move the fill, miss, read and replay counters by exactly their
/// prefill shares; six decode tokens after it fill slots and move the
/// counters, and no share.
/// MUTATION: the tail pass is counted twice -> shares exceed the counters
/// -> red.
/// MUTATION: a decode step counts as prefill -> shares move in decode ->
/// red.
#[test]
fn prefill_shares_hold_only_the_prefill_passes() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let mut inner = super::route_step::moe_shape_inner(10);
    inner.decoder.context_window = 768;
    inner.activate_paged_seq(1).unwrap();
    let live = Arc::clone(&inner.decoder.weights.live);
    let prompt = prompt(300);
    let [total, share] = shares(&live);
    let (logits, _) = inner
        .decoder
        .prefill_group(&[&prompt[..256]], &prompt[256..], true, false)
        .unwrap();
    let [after_total, after_share] = shares(&live);
    let moved: [u64; 6] = std::array::from_fn(|i| after_total[i] - total[i]);
    let counted: [u64; 6] = std::array::from_fn(|i| after_share[i] - share[i]);
    assert_eq!(
        counted, moved,
        "prefill shares vs counters over the prefill"
    );
    assert!(
        moved[0] > 0 && moved[3] > 0,
        "the prefill filled slots: {moved:?}"
    );
    assert_eq!(inner.decoder.history, prompt);

    let mut token = argmax(&logits);
    for _ in 0..6 {
        token = argmax(&inner.decoder.step(token).unwrap());
    }
    let [decoded_total, decoded_share] = shares(&live);
    assert!(
        decoded_total[0] > after_total[0],
        "decode tokens filled slots: {after_total:?} -> {decoded_total:?}"
    );
    assert_eq!(decoded_share, after_share, "decode moves no prefill share");
    eprintln!("prefill moved {moved:?}; decode totals {after_total:?} -> {decoded_total:?}");
}

/// A slot fill of four experts with gaps between them asks the files for
/// their ranges rounded out to 16 KiB pages and no byte between two
/// ranges: `fillReadBytes` stays within two pages per tensor range of the
/// wanted bytes. The old loop's 8 MiB windows read the gaps and pass that
/// bound.
/// MUTATION: ranges within 8 MiB merge into one read -> the gaps count ->
/// red.
#[test]
fn fill_read_bytes_are_the_pages_of_the_wanted_ranges() {
    let _host = memory::scoped_fixture_host();
    let mut new = super::route_step::moe_shape_inner(11);
    let mut old = super::route_step::moe_shape_inner(11);
    old.decoder.weights.old_slot_fill = true;
    let c = new.decoder.config.clone();
    let ids = [0u32, 3, 6, 9];
    let ranges = 3 * ids.len() as u64;
    let mut asked = Vec::new();
    for inner in [&mut new, &mut old] {
        let before = records(&inner.decoder.weights.live);
        inner
            .decoder
            .weights
            .expert_slots(0, c.num_experts, c.num_experts_per_tok, &ids)
            .unwrap()
            .expect("four experts fit the slots");
        let lease = delta(records(&inner.decoder.weights.live), before);
        assert_eq!(lease[3], ids.len() as u64, "four misses");
        assert!(lease[2] >= lease[4], "pages hold the wanted bytes");
        asked.push((lease[2], lease[4]));
    }
    let bound = |(read, wanted): (u64, u64)| read - wanted < ranges * 2 * 16384;
    assert!(
        bound(asked[0]),
        "[fill bytes, wanted bytes] {:?}: more than the pages of the ranges",
        asked[0]
    );
    assert!(
        !bound(asked[1]),
        "the old loop's windows {:?} should read the gaps",
        asked[1]
    );
    assert_same_slot_banks(&new.decoder, &old.decoder, "pages vs windows");
}

/// A decoder over the direct GGUF fixture at `path` (4 experts, top 2) on
/// a streamed store of `capacity` slots per layer, windows of `width` rows
/// and context and pages for `tokens`.
fn streamed_direct_decoder(
    path: &Path,
    config: &config::Config,
    capacity: usize,
    width: usize,
    tokens: usize,
) -> decoder::Decoder {
    let store = weights::Store::open_fixture(path, None).unwrap();
    let mut decoder = decoder::Decoder::new(config.clone(), store).unwrap();
    decoder.weights.stream_fixture_store(capacity).unwrap();
    decoder.register_cache_limit();
    decoder.weights.publish_live();
    decoder.residency_changed();
    decoder.weights.begin_turn().unwrap();
    decoder.prefill_window_pin = Some(width);
    decoder.prefill_chunk_size = width;
    decoder.context_window = decoder.context_window.max(tokens);
    let mut adapter = paged::create_fixture(config).unwrap();
    adapter.begin_request(1).unwrap();
    adapter.allocate_suffix_blocks(tokens as u32).unwrap();
    decoder.paged = Some(adapter);
    decoder
}

/// The kernel class each union lease logged, window after window, for
/// every layer.
fn union_classes(decoder: &decoder::Decoder) -> Vec<String> {
    decoder
        .submit_log
        .borrow()
        .iter()
        .filter_map(|e| e.strip_prefix("qwen4::union::"))
        .map(str::to_owned)
        .collect()
}

/// On a streamed store (4 experts, top 2) a group of three windows and a
/// 5-row tail gives the bytes of four one-window passes, at 2 and 3 slots
/// (fewer than a window's experts) and at 4 slots with experts 0 and 1
/// held first (a window's experts then fit one lease but span two
/// sub-groups), for windows of 16 and 48 rows, whose kernel classes are
/// Rows and Gather: logits, hidden views, history, every cache array and
/// pool K/V position, then the logits of 4 greedy decode steps. Every layer
/// of the group ran one union lease, each window with its own class (the
/// log names it). Tiles windows of this fixture take MLX's sorted gather,
/// which leaves NaN rows now and then on its Q5_1 banks in both runs;
/// `an_int8_group_matches_windows_bit_for_bit` proves the Tiles class.
/// MUTATION: a sub-group's rows are not registered as readers of the layer
/// -> the next sub-group's fill overwrites slots they still read ->
/// "logits" red.
#[test]
fn a_slotted_group_matches_windows_bit_for_bit() {
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let layers = config.num_hidden_layers;
    for (width, class) in [(16usize, "Rows"), (48, "Gather")] {
        let prompt = prompt(3 * width + 5);
        let tokens = prompt.len() + 4;
        let mut windowed = streamed_direct_decoder(&path, &config, 2, width, tokens);
        let (ref_windows, ref_tail) = windowed.group_windows(&prompt);
        let mut window_hidden = Vec::new();
        let mut expected = None;
        for (k, chunk) in ref_windows.iter().copied().chain([ref_tail]).enumerate() {
            expected = Some(windowed.prefill_chunk(chunk, None, k == 3).unwrap());
            window_hidden.extend(windowed.last_chunk_hidden.iter().map(f32s));
        }
        let expected_logits = expected.unwrap();
        let want_history = windowed.history.clone();
        let want_frontier = frontier_bytes(&windowed);
        assert!(union_classes(&windowed).is_empty(), "width {width}");
        let mut decode_last = expected_logits.clone();
        let mut want_decode = Vec::with_capacity(4);
        for _ in 0..4 {
            let token = argmax(&decode_last);
            decode_last = windowed.step(token).unwrap();
            want_decode.push(decode_last.clone());
        }
        for (capacity, held) in [(2usize, &[][..]), (3, &[]), (4, &[0u32, 1])] {
            let what = format!("width {width}, {capacity} slots holding {held:?}");
            let mut grouped = streamed_direct_decoder(&path, &config, capacity, width, tokens);
            if !held.is_empty() {
                super::route_step::fill_slots(&mut grouped, held);
            }
            let (windows, tail) = grouped.group_windows(&prompt);
            assert_eq!(windows.len(), 3, "{what}");
            assert_eq!(tail.len(), 5, "{what}");
            let (logits, views) = grouped.prefill_group(&windows, tail, true, true).unwrap();
            let mut group_hidden: Vec<Vec<f32>> = views.iter().flatten().map(f32s).collect();
            group_hidden.extend(grouped.last_chunk_hidden.iter().map(f32s));
            assert_same_bits(&logits, &expected_logits, &format!("{what}: logits"));
            assert_eq!(group_hidden, window_hidden, "{what}: hidden views");
            assert_eq!(grouped.history, want_history, "{what}: history");
            assert_eq!(
                frontier_bytes(&grouped),
                want_frontier,
                "{what}: caches and pool K/V"
            );
            assert_eq!(
                union_classes(&grouped),
                (0..layers)
                    .flat_map(|i| [0, 1, 2].map(|_| format!("{i}::{class}")))
                    .collect::<Vec<_>>(),
                "{what}: one union lease per layer, each window's own class"
            );
            let mut logits = logits;
            for (step, want) in want_decode.iter().enumerate() {
                let token = argmax(&logits);
                logits = grouped.step(token).unwrap();
                assert_same_bits(&logits, want, &format!("{what}: decode step {step}"));
            }
        }
    }
}

/// On the Flash Next shape with int8 pages (12 experts, top 10, 256-row
/// windows: the Tiles class) at 10 and 11 slots, and at 10 slots holding
/// experts 0-9 first, a group of two windows and a 44-row tail gives the
/// bits of three one-window passes: every window's hidden views, the tail's
/// logits, history, positions, every state array, the last hidden rows,
/// index blocks, the int8 pages and their scales; then 6 greedy decode
/// steps keep equal logits and state.
/// MUTATION: window w's K/V reach the pool after window w + 1's attention
/// -> window 1 reads window 0's rows from an unwritten pool -> "hidden
/// views" red (the attention layer is the last, so only window 1's own
/// output sees it).
#[test]
fn an_int8_group_matches_windows_bit_for_bit() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let prompt = prompt(556);
    let windows = [&prompt[..256], &prompt[256..512]];
    let tail = &prompt[512..];

    let mut windowed = super::route_step::moe_shape_inner(10);
    windowed.decoder.context_window = 768;
    windowed.activate_paged_seq(1).unwrap();
    let mut expected = None;
    let mut window_views: Vec<Vec<Vec<u32>>> = Vec::new();
    for chunk in windows.into_iter().chain([tail]) {
        expected = Some(windowed.decoder.prefill_chunk(chunk, None, true).unwrap());
        window_views.push(
            windowed
                .decoder
                .last_chunk_hidden
                .iter()
                .map(f32_bits)
                .collect(),
        );
    }
    let want_logits = expected.unwrap();
    let want_before_decode = super::route_step::snapshot_decoder_state(&mut windowed.decoder);
    let mut decode_last = want_logits.clone();
    let mut want_decode = Vec::with_capacity(6);
    for _ in 0..6 {
        let token = argmax(&decode_last);
        decode_last = windowed.decoder.step(token).unwrap();
        want_decode.push(decode_last.clone());
    }
    let want_after_decode = super::route_step::snapshot_decoder_state(&mut windowed.decoder);
    drop(windowed);

    for (capacity, held) in [(10usize, 0u32), (11, 0), (10, 10)] {
        let what = format!("{capacity} slots holding {held} experts");
        let mut grouped = super::route_step::moe_shape_inner(capacity);
        grouped.decoder.context_window = 768;
        grouped.activate_paged_seq(1).unwrap();
        let held_ids: Vec<u32> = (0..held).collect();
        if !held_ids.is_empty() {
            super::route_step::fill_slots(&mut grouped.decoder, &held_ids);
        }
        let c = grouped.decoder.config.clone();
        assert_eq!(c.kv_format, crate::transformer::KvFormat::Int8);
        let (mut logits, views) = grouped
            .decoder
            .prefill_group(&windows, tail, true, true)
            .unwrap();
        assert_eq!(views.len(), 2, "{what}: one view list per window");
        for (w, (got, want)) in views.iter().zip(&window_views).enumerate() {
            assert_eq!(got.len(), want.len(), "{what}: window {w} views");
            for (row, (x, y)) in got.iter().zip(want).enumerate() {
                super::route_step::assert_same_f32_bits(
                    x,
                    y,
                    &format!("{what}: hidden views, window {w} row {row}"),
                );
            }
        }
        assert_same_bits(&logits, &want_logits, &format!("{what}: logits"));
        super::route_step::assert_decoder_matches_snapshot(
            &mut grouped.decoder,
            &want_before_decode,
            &what,
        );
        assert_eq!(
            union_classes(&grouped.decoder),
            (0..c.num_hidden_layers)
                .flat_map(|i| [0, 1].map(|_| format!("{i}::Tiles")))
                .collect::<Vec<_>>(),
            "{what}: one union lease per layer"
        );
        for (step, want) in want_decode.iter().enumerate() {
            let token = argmax(&logits);
            logits = grouped.decoder.step(token).unwrap();
            assert_same_bits(&logits, want, &format!("{what}: decode step {step}"));
        }
        super::route_step::assert_decoder_matches_snapshot(
            &mut grouped.decoder,
            &want_after_decode,
            &format!("{what}, after decode"),
        );
        let pool = grouped.decoder.paged.as_ref().unwrap().key_pool_array(0);
        assert_eq!(pool.unwrap().dtype().unwrap(), crate::array::DType::Int8);
    }
}

struct GroupAfterWindow {
    layers: usize,
    experts: usize,
    before: Vec<Vec<Option<u32>>>,
    moved: [u64; 7],
    union_routes: Vec<Vec<u32>>,
    expert_fills: std::collections::HashMap<(usize, u32), u32>,
    layer_leases: std::collections::HashMap<usize, (std::collections::HashSet<u32>, u64)>,
    occupants: Vec<Vec<Option<u32>>>,
    ages: Vec<(Vec<u64>, u64)>,
}

/// The Flash Next shape at 10 slots after a 256-row window, then a group of
/// two 256-row windows, cached once per process for the read-once and
/// last-use-order tests.
fn group_after_a_window() -> &'static GroupAfterWindow {
    static SNAP: std::sync::OnceLock<GroupAfterWindow> = std::sync::OnceLock::new();
    SNAP.get_or_init(|| {
        use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
        let prompt = prompt(768);
        let mut inner = super::route_step::moe_shape_inner(10);
        inner.decoder.context_window = 768;
        inner.activate_paged_seq(1).unwrap();
        inner
            .decoder
            .prefill_chunk(&prompt[..256], None, true)
            .unwrap();
        let layers = inner.decoder.config.num_hidden_layers;
        let experts = inner.decoder.config.num_experts;
        let before: Vec<_> = (0..layers)
            .map(|layer| inner.decoder.weights.slot_occupants(layer).unwrap())
            .collect();
        inner.decoder.weights.layer_leases.clear();
        inner.decoder.weights.expert_fills.clear();
        let records_before = records(&inner.decoder.weights.live);
        inner
            .decoder
            .prefill_group(&[&prompt[256..512], &prompt[512..]], &[], true, false)
            .unwrap();
        let moved = delta(records(&inner.decoder.weights.live), records_before);
        let union_routes = (0..layers)
            .map(|layer| union_routes(&inner.decoder, layer))
            .collect();
        let occupants = (0..layers)
            .map(|layer| inner.decoder.weights.slot_occupants(layer).unwrap())
            .collect();
        let ages = (0..layers)
            .map(|layer| inner.decoder.weights.slot_ages(layer).unwrap())
            .collect();
        GroupAfterWindow {
            layers,
            experts,
            before,
            moved,
            union_routes,
            expert_fills: inner.decoder.weights.expert_fills.clone(),
            layer_leases: inner.decoder.weights.layer_leases.clone(),
            occupants,
            ages,
        }
    })
}

/// The group's routes of `layer`, window after window.
fn union_routes(decoder: &decoder::Decoder, layer: usize) -> Vec<u32> {
    let routes: Vec<_> = decoder
        .union_routes
        .iter()
        .filter(|(l, _)| *l == layer)
        .collect();
    assert_eq!(routes.len(), 1, "layer {layer}: one union lease");
    routes[0].1.clone()
}

/// On the Flash Next shape at 10 slots, a group of two 256-row windows
/// after a first window reads each expert its union misses once per layer:
/// the experts filled are exactly the union minus the slots' experts before
/// the group, each once, and `slotMisses` grows by their count summed over
/// the layers.
/// MUTATION: sub-groups by last use over the union (the window-major split)
/// instead of the slots' experts first -> a held expert of a later
/// sub-group is evicted by an earlier one and read again -> red.
#[test]
fn a_group_reads_each_missing_expert_once_per_layer() {
    let _host = memory::scoped_fixture_host();
    let snap = group_after_a_window();
    let mut expected = 0u64;
    for (layer, held) in snap.before.iter().enumerate() {
        let union: std::collections::BTreeSet<u32> =
            snap.union_routes[layer].iter().copied().collect();
        let missing: std::collections::BTreeSet<u32> = union
            .iter()
            .copied()
            .filter(|e| !held.contains(&Some(*e)))
            .collect();
        let filled: std::collections::BTreeMap<u32, u32> = snap
            .expert_fills
            .iter()
            .filter(|((l, _), _)| *l == layer)
            .map(|(&(_, e), &n)| (e, n))
            .collect();
        assert_eq!(
            filled,
            missing.iter().map(|&e| (e, 1)).collect(),
            "layer {layer}: experts filled (held {held:?}, union {union:?})"
        );
        let (leased, misses) = &snap.layer_leases[&layer];
        assert_eq!(
            leased
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            union,
            "layer {layer}: the leases asked for the union"
        );
        assert_eq!(*misses, missing.len() as u64, "layer {layer}: misses");
        expected += missing.len() as u64;
    }
    assert!(expected > 0, "the group missed experts");
    assert_eq!(
        snap.moved[3], expected,
        "slotMisses = |union - held| over layers"
    );
    assert_eq!(
        [snap.moved[5], snap.moved[6]],
        [1, 2],
        "one group of two windows"
    );
}

/// After a group the slots' experts of its union are newer than every other
/// occupant, in the order of their last use over the group's routes (window
/// after window), as one window's lease leaves them.
/// MUTATION: no final touch -> the experts the group loaded stay newer than
/// held experts used later -> red.
#[test]
fn a_group_leaves_last_used_experts_newest() {
    let _host = memory::scoped_fixture_host();
    let snap = group_after_a_window();
    for layer in 0..snap.layers {
        let order = route_plan::expert_order(&snap.union_routes[layer], snap.experts).unwrap();
        let occupants = &snap.occupants[layer];
        let (ages, _) = &snap.ages[layer];
        let mut by_age: Vec<(u64, Option<u32>)> = ages
            .iter()
            .copied()
            .zip(occupants.iter().copied())
            .collect();
        by_age.sort_unstable();
        let newest: Vec<u32> = by_age
            .iter()
            .filter_map(|&(_, e)| e.filter(|e| order.contains(e)))
            .collect();
        let held_order: Vec<u32> = order
            .iter()
            .copied()
            .filter(|e| occupants.contains(&Some(*e)))
            .collect();
        assert_eq!(newest, held_order, "layer {layer}: ages follow last use");
        let first_union = by_age
            .iter()
            .position(|&(_, e)| e.is_some_and(|e| order.contains(&e)))
            .unwrap();
        assert!(
            by_age[first_union..]
                .iter()
                .all(|&(_, e)| e.is_some_and(|e| order.contains(&e))),
            "layer {layer}: the union's experts are the newest {by_age:?}"
        );
    }
}

struct PlaneSnapshot {
    slots: Vec<(usize, Vec<u32>)>,
}

struct LayerBankSnapshot {
    occupants: Vec<Option<u32>>,
    parts: [[Option<PlaneSnapshot>; 3]; 3],
}

#[derive(Debug, PartialEq, Eq)]
struct SlotBookSnapshot {
    occupants: Option<Vec<Option<u32>>>,
    ages: Option<(Vec<u64>, u64)>,
}

fn snapshot_slot_banks(decoder: &decoder::Decoder) -> Vec<Option<LayerBankSnapshot>> {
    (0..decoder.config.num_hidden_layers)
        .map(|layer| {
            let occupants = decoder.weights.slot_occupants(layer)?;
            let banks = decoder.weights.slot_banks(layer).unwrap();
            let parts = std::array::from_fn(|part| {
                let arrays = [
                    Some(&banks[part].values),
                    banks[part].scales.as_ref(),
                    banks[part].biases.as_ref(),
                ];
                std::array::from_fn(|plane| {
                    arrays[plane].map(|a| {
                        let rows = a.shape().unwrap()[0] / occupants.len() as i64;
                        let slots = occupants
                            .iter()
                            .enumerate()
                            .filter(|(_, e)| e.is_some())
                            .map(|(slot, _)| {
                                let (lo, hi) = (slot as i64 * rows, (slot as i64 + 1) * rows);
                                (slot, f32_bits(&a.slice_axis(0, lo, hi).unwrap()))
                            })
                            .collect();
                        PlaneSnapshot { slots }
                    })
                })
            });
            Some(LayerBankSnapshot { occupants, parts })
        })
        .collect()
}

fn assert_same_slot_bank_snapshots(
    a: &[Option<LayerBankSnapshot>],
    b: &[Option<LayerBankSnapshot>],
    what: &str,
) {
    assert_eq!(a.len(), b.len(), "{what}: layer count");
    for (layer, (x, y)) in a.iter().zip(b).enumerate() {
        match (x, y) {
            (None, None) => continue,
            (Some(xa), Some(xb)) => {
                assert_eq!(
                    xa.occupants, xb.occupants,
                    "{what}: layer {layer} occupants"
                );
                for part in 0..3 {
                    for plane in 0..3 {
                        match (&xa.parts[part][plane], &xb.parts[part][plane]) {
                            (None, None) => {}
                            (Some(sa), Some(sb)) => {
                                assert_eq!(
                                    sa.slots.len(),
                                    sb.slots.len(),
                                    "{what}: layer {layer} part {part}"
                                );
                                for ((slot_a, ba), (slot_b, bb)) in sa.slots.iter().zip(&sb.slots) {
                                    assert_eq!(slot_a, slot_b);
                                    super::route_step::assert_same_bit_slices(
                                        ba,
                                        bb,
                                        &format!(
                                            "{what}: layer {layer} part {part} plane {plane} slot {slot_a}"
                                        ),
                                    );
                                }
                            }
                            _ => panic!("{what}: layer {layer} part {part}"),
                        }
                    }
                }
            }
            _ => panic!("{what}: layer {layer} occupants presence"),
        }
    }
}

struct FillTurnSnapshot {
    layers: usize,
    logits: Vec<Vec<u32>>,
    hidden: Vec<Vec<f32>>,
    state: super::route_step::DecoderStateSnapshot,
    slot_books: Vec<SlotBookSnapshot>,
    slot_banks: Vec<Option<LayerBankSnapshot>>,
    records: [u64; 7],
    events: [Vec<String>; 2],
    fill_bank_checks: Option<Vec<(usize, bool, bool)>>,
}

fn snapshot_fill_turn(
    mut inner: Inner,
    logits: Vec<MxArray>,
    hidden: Vec<Vec<f32>>,
    events: [Vec<String>; 2],
) -> FillTurnSnapshot {
    let layers = inner.decoder.config.num_hidden_layers;
    let state = super::route_step::snapshot_decoder_state(&mut inner.decoder);
    let slot_books = (0..layers)
        .map(|layer| {
            let weights = &inner.decoder.weights;
            SlotBookSnapshot {
                occupants: weights.slot_occupants(layer),
                ages: weights.slot_ages(layer),
            }
        })
        .collect();
    let slot_banks = snapshot_slot_banks(&inner.decoder);
    let recs = records(&inner.decoder.weights.live);
    let fill_bank_checks = inner.decoder.fill_bank_checks.clone();
    FillTurnSnapshot {
        layers,
        logits: logits.iter().map(f32_bits).collect(),
        hidden,
        state,
        slot_books,
        slot_banks,
        records: recs,
        events,
        fill_bank_checks,
    }
}

/// A turn on the Flash Next shape at 10 slots holding experts 0-5 (a
/// union of all 12 then runs 6 residents first and reads 6), with fills
/// read ahead of their lease or not (`ahead`) and batches of 2 MiB (more
/// than the ring holds, so reads are queued during the finish) or 64 MiB:
/// a group of two 256-row windows (`group`) or one 256-row window, then a
/// 44-row tail, then 4 greedy decode steps. Cached per `(ahead, small, group)`
/// with `watch` enabled on `(true, true, _)` so both fill-ahead tests share
/// the same turns.
fn cached_fill_ahead_turn(ahead: bool, small: bool, group: bool) -> Arc<FillTurnSnapshot> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(bool, bool, bool), Arc<FillTurnSnapshot>>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (ahead, small, group);
    if let Some(snap) = cache.lock().unwrap().get(&key) {
        return Arc::clone(snap);
    }
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let prompt = prompt(556);
    let mut inner = super::route_step::moe_shape_inner(10);
    inner.decoder.weights.no_fill_ahead = !ahead;
    if small {
        inner.decoder.weights.fill_batch_bytes = Some(2 << 20);
    }
    inner.decoder.context_window = 768;
    inner.activate_paged_seq(1).unwrap();
    let held: Vec<u32> = (0..6).collect();
    super::route_step::fill_slots(&mut inner.decoder, &held);
    if ahead && small {
        inner.decoder.fill_bank_checks = Some(Vec::new());
    }
    inner.decoder.submit_log.borrow_mut().clear();
    let decoder = &mut inner.decoder;
    let mut logits = Vec::new();
    let mut hidden = Vec::new();
    if group {
        let (out, views) = decoder
            .prefill_group(
                &[&prompt[..256], &prompt[256..512]],
                &prompt[512..],
                true,
                true,
            )
            .unwrap();
        hidden.extend(views.iter().flatten().map(f32s));
        hidden.extend(decoder.last_chunk_hidden.iter().map(f32s));
        logits.push(out);
    } else {
        for chunk in [&prompt[..256], &prompt[256..300]] {
            logits.push(decoder.prefill_chunk(chunk, None, true).unwrap());
            hidden.extend(decoder.last_chunk_hidden.iter().map(f32s));
        }
    }
    let prefill = std::mem::take(&mut *decoder.submit_log.borrow_mut());
    for _ in 0..4 {
        let token = argmax(logits.last().unwrap());
        logits.push(decoder.step(token).unwrap());
    }
    let decode = std::mem::take(&mut *decoder.submit_log.borrow_mut());
    let snap = Arc::new(snapshot_fill_turn(inner, logits, hidden, [prefill, decode]));
    let mut map = cache.lock().unwrap();
    Arc::clone(map.entry(key).or_insert(snap))
}

/// The test events of the fills read ahead in `log`, per layer in order.
fn fill_events(log: &[String], layers: usize) -> Vec<Vec<&'static str>> {
    (0..layers)
        .map(|layer| {
            log.iter()
                .filter_map(|e| {
                    ["begin", "rows", "finish"]
                        .into_iter()
                        .find(|step| *e == format!("qwen4::fill::{step}::{layer}"))
                })
                .collect()
        })
        .collect()
}

/// The two turns gave the same bits: logits, hidden views, decoder state
/// (int8 pages and scales included), slot book (occupants, ages, tick),
/// slot bank rows and the records but fill time.
fn assert_same_fill_turn_snapshots(a: &FillTurnSnapshot, b: &FillTurnSnapshot, what: &str) {
    assert_eq!(a.logits.len(), b.logits.len(), "{what}: logits count");
    for (call, (x, y)) in a.logits.iter().zip(&b.logits).enumerate() {
        super::route_step::assert_same_bit_slices(x, y, &format!("{what}: logits of call {call}"));
    }
    assert_eq!(a.hidden, b.hidden, "{what}: hidden views");
    super::route_step::assert_same_decoder_state_snapshots(&a.state, &b.state, what);
    assert_eq!(a.slot_books, b.slot_books, "{what}: slot book");
    assert_same_slot_bank_snapshots(&a.slot_banks, &b.slot_banks, what);
    let (x, y) = (a.records, b.records);
    assert_eq!(
        [x[0], x[2], x[3], x[4], x[5], x[6]],
        [y[0], y[2], y[3], y[4], y[5], y[6]],
        "{what}: records but fill time"
    );
}

/// On the Flash Next shape a fill read ahead of its lease runs as begin
/// (victims chosen, reads queued), then the rows of the sub-group before it
/// are submitted, then the finish (readers completed, repack, mapping), in
/// that order at every layer, for a group of two 256-row windows and for a
/// one-window pass, at 64 MiB and at 2 MiB batches. Each turn's bits equal
/// the turn whose fills wait for their lease; decode steps read nothing
/// ahead.
/// MUTATION: the fill finishes before the rows of the sub-group before it
/// are dispatched -> finish before rows -> red.
#[test]
fn residents_are_submitted_before_the_fill_finishes() {
    let _host = memory::scoped_fixture_host();
    for group in [true, false] {
        for small in [false, true] {
            let what = format!("group {group}, 2 MiB batches {small}");
            let ahead = cached_fill_ahead_turn(true, small, group);
            let waited = cached_fill_ahead_turn(false, small, group);
            let layers = ahead.layers;
            let events = fill_events(&ahead.events[0], layers);
            for (layer, steps) in events.iter().enumerate() {
                assert!(!steps.is_empty(), "{what}: layer {layer} read ahead");
                for fill in steps.chunks(3) {
                    assert_eq!(
                        fill,
                        ["begin", "rows", "finish"],
                        "{what}: layer {layer} order {steps:?}"
                    );
                }
            }
            for (turn, phase) in [(&ahead, 1), (&waited, 0), (&waited, 1)] {
                assert!(
                    fill_events(&turn.events[phase], layers)
                        .iter()
                        .all(Vec::is_empty),
                    "{what}: no read ahead in phase {phase} of a turn"
                );
            }
            assert_same_fill_turn_snapshots(&ahead, &waited, &what);
        }
    }
}

/// A fill read ahead of its lease writes no bank row the sub-group before
/// it reads: at every layer a victim of the fill is a slot that sub-group
/// leased (the reads would replace a resident in use), and those slots'
/// bank rows are unchanged once the sub-group's rows are submitted; the
/// turn's bits equal the turn whose fills wait for their lease.
/// MUTATION: the fill repacks at its begin -> the residents' rows change
/// under their reader -> "unchanged" red, logits red.
#[test]
fn a_fill_never_writes_under_a_pending_reader() {
    let _host = memory::scoped_fixture_host();
    for group in [true, false] {
        let what = format!("group {group}");
        let ahead = cached_fill_ahead_turn(true, true, group);
        let waited = cached_fill_ahead_turn(false, true, group);
        let layers = ahead.layers;
        let checks = ahead.fill_bank_checks.as_ref().unwrap();
        for layer in 0..layers {
            assert!(
                checks.iter().any(|&(l, read, _)| l == layer && read),
                "{what}: layer {layer} evicts a slot the residents read {checks:?}"
            );
        }
        assert!(
            checks.iter().all(|&(_, _, unchanged)| unchanged),
            "{what}: bank rows unchanged under their reader {checks:?}"
        );
        assert_same_fill_turn_snapshots(&ahead, &waited, &what);
    }
}

/// Windows of at most 8 rows keep their path: an 8-row prompt window that
/// misses (more experts than slots) and the decode steps after it read
/// nothing ahead, and give the bits of the turn whose fills wait.
/// MUTATION: windows of any row count read ahead -> a fill begins in the
/// 8-row window -> red.
#[test]
fn windows_of_eight_rows_read_nothing_ahead() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let prompt = prompt(8);
    let [ahead, waited] = [true, false].map(|ahead| {
        let mut inner = super::route_step::moe_shape_inner(10);
        inner.decoder.weights.no_fill_ahead = !ahead;
        inner.decoder.context_window = 768;
        inner.activate_paged_seq(1).unwrap();
        let held: Vec<u32> = (0..6).collect();
        super::route_step::fill_slots(&mut inner.decoder, &held);
        inner.decoder.submit_log.borrow_mut().clear();
        let before = records(&inner.decoder.weights.live);
        let mut logits = vec![inner.decoder.prefill_chunk(&prompt, None, true).unwrap()];
        let moved = delta(records(&inner.decoder.weights.live), before);
        assert!(moved[3] > 0, "the 8-row window misses");
        for _ in 0..4 {
            let token = argmax(logits.last().unwrap());
            logits.push(inner.decoder.step(token).unwrap());
        }
        let log = std::mem::take(&mut *inner.decoder.submit_log.borrow_mut());
        snapshot_fill_turn(inner, logits, Vec::new(), [log, Vec::new()])
    });
    let layers = ahead.layers;
    assert!(
        fill_events(&ahead.events[0], layers)
            .iter()
            .all(Vec::is_empty),
        "no read ahead for 8 rows: {:?}",
        fill_events(&ahead.events[0], layers)
    );
    assert_same_fill_turn_snapshots(&ahead, &waited, "8 rows");
}

/// A fill read ahead and left (its lease never came) is abandoned by the
/// next slot fill, which reads its own experts: after a begin of [6, 7, 8]
/// on layer 0 and a lease of [9, 10, 11] instead, then a lease of [6, 7, 8],
/// the slot book and bank rows equal a store that never read ahead; a begin
/// on layer 1 finished by its own lease equals a plain lease.
/// MUTATION: a slot fill takes the fill read ahead whatever experts it was
/// for -> the wrong experts' rows land in the slots -> red.
#[test]
fn a_fill_read_ahead_and_left_is_abandoned() {
    let _host = memory::scoped_fixture_host();
    let [mut ahead, mut plain] = [0, 1].map(|_| {
        let mut inner = super::route_step::moe_shape_inner(10);
        let held: Vec<u32> = (0..6).collect();
        super::route_step::fill_slots(&mut inner.decoder, &held);
        inner
    });
    let c = ahead.decoder.config.clone();
    let (experts, top) = (c.num_experts, c.num_experts_per_tok);
    assert!(
        ahead
            .decoder
            .weights
            .begin_expert_fill(0, experts, top, &[6, 7, 8])
            .unwrap()
    );
    assert!(ahead.decoder.weights.pending_victims().is_some());
    for inner in [&mut ahead, &mut plain] {
        for ids in [&[9u32, 10, 11][..], &[6, 7, 8]] {
            inner
                .decoder
                .weights
                .expert_slots(0, experts, top, ids)
                .unwrap()
                .unwrap();
        }
    }
    assert!(ahead.decoder.weights.pending_victims().is_none());
    assert!(
        ahead
            .decoder
            .weights
            .begin_expert_fill(1, experts, top, &[6, 7, 8])
            .unwrap()
    );
    for inner in [&mut ahead, &mut plain] {
        inner
            .decoder
            .weights
            .expert_slots(1, experts, top, &[6, 7, 8])
            .unwrap()
            .unwrap();
    }
    for layer in 0..c.num_hidden_layers {
        let book = |inner: &Inner| {
            let weights = &inner.decoder.weights;
            (weights.slot_occupants(layer), weights.slot_ages(layer))
        };
        assert_eq!(book(&ahead), book(&plain), "layer {layer} slot book");
    }
    assert_same_slot_banks(&ahead.decoder, &plain.decoder, "abandoned fill");
    let (x, y) = (
        records(&ahead.decoder.weights.live),
        records(&plain.decoder.weights.live),
    );
    assert_eq!(
        [x[0], x[2], x[3]],
        [y[0], y[2], y[3]],
        "fills, bytes, misses"
    );
}

/// A fill read ahead has queued, when its begin returns, every batch the
/// ring's four buffers hold: the whole of a fill of up to four batches, the
/// first four of a longer one; its lease reads the rest, each batch once.
/// MUTATION: the begin queues one batch less than the ring holds -> a
/// one-batch fill reads nothing ahead -> red.
/// MUTATION: the lease queues again the last batch the begin queued -> its
/// bytes are read twice -> red.
#[test]
fn a_fill_queues_the_whole_ring_at_its_begin() {
    let _host = memory::scoped_fixture_host();
    let mut counts = Vec::new();
    for batch_mib in [64usize, 8, 4, 2] {
        let mut inner = super::route_step::moe_shape_inner(10);
        inner.decoder.weights.fill_batch_bytes = Some(batch_mib << 20);
        let held: Vec<u32> = (0..6).collect();
        super::route_step::fill_slots(&mut inner.decoder, &held);
        let c = inner.decoder.config.clone();
        let (experts, top) = (c.num_experts, c.num_experts_per_tok);
        let ids = [6u32, 7, 8, 9];
        let weights = &mut inner.decoder.weights;
        let before = weights.slot_fill_submitted_bytes();
        assert!(weights.begin_expert_fill(0, experts, top, &ids).unwrap());
        let batches = weights.pending_batch_bytes();
        let at_begin = weights.slot_fill_submitted_bytes() - before;
        weights
            .expert_slots(0, experts, top, &ids)
            .unwrap()
            .unwrap();
        let read = weights.slot_fill_submitted_bytes() - before;
        let what = format!("{batch_mib} MiB batches of {batches:?} bytes");
        assert_eq!(
            at_begin,
            batches.iter().take(4).sum::<u64>(),
            "{what}: queued at the begin"
        );
        assert_eq!(
            read,
            batches.iter().sum::<u64>(),
            "{what}: read by the fill"
        );
        counts.push(batches.len());
    }
    assert!(
        counts.contains(&1) && counts.iter().any(|&n| n > 4),
        "batch counts {counts:?}"
    );
}

/// A store whose slots can miss groups from the memory plan alone: on a host
/// that reports nothing available, a prompt of nine 16-row windows groups
/// the most windows (up to four) whose group scratch fits beside one pass's
/// scratch at the prompt's last row inside the planned transient; none
/// without a plan, below two fitting windows, for one window and a tail, or
/// on a store without slots. A direct fill on the same host reads the live
/// host and does not group.
/// MUTATION: the partial rule also asks the live host for the group scratch
/// -> no group at 0 available -> red.
#[test]
fn a_partial_store_groups_from_the_plan() {
    use decoder::GroupLimit::Windows;
    let _host = memory::scoped_test_host(memory::HostMemory {
        free: 0,
        file: 0,
        ..memory::fixture_host()
    });
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let (slice, rows) = (16usize, 9 * 16usize);
    let mut d = streamed_direct_decoder(&path, &config, 2, slice, rows + 16);
    assert!(d.weights.slot_store());
    assert_eq!(d.group_limit(0, rows), None, "no plan");
    let group = |k| memory::group_scratch(&config, k, slice);
    let fitting = |pass: u64, planned: u64| {
        (2..=4)
            .rev()
            .find(|&k| pass + group(k) <= planned)
            .map(Windows)
    };
    let pass = memory::prefill_scratch(&config, slice, rows);
    for (planned, want) in [
        (pass + group(4), Some(Windows(4))),
        (pass + group(4) - 1, Some(Windows(3))),
        (pass + group(3), Some(Windows(3))),
        (pass + group(3) - 1, Some(Windows(2))),
        (pass + group(2), Some(Windows(2))),
        (pass + group(2) - 1, None),
    ] {
        d.planned_scratch = Some(planned);
        assert_eq!(d.group_limit(0, rows), want, "planned {planned}");
    }
    d.planned_scratch = Some(pass + group(4));
    let deep = 40_000;
    let deeper = memory::prefill_scratch(&config, slice, deep + rows);
    assert!(deeper > pass, "a deeper prompt prices a larger pass");
    assert_eq!(
        d.group_limit(deep, rows),
        fitting(deeper, pass + group(4)),
        "the pass is priced at the prompt's last row"
    );
    d.planned_scratch = Some(u64::MAX);
    assert_eq!(d.group_limit(0, 2 * slice), Some(Windows(2)));
    assert_eq!(d.group_limit(0, slice + 15), None, "a window and a tail");
    assert_eq!(d.group_limit(0, 3 * slice + 15), Some(Windows(3)));
    let mut plain = group_fixture_decoder(slice, 1);
    plain.planned_scratch = Some(u64::MAX);
    assert!(!plain.weights.slot_store());
    assert_eq!(plain.group_limit(0, rows), None, "no slots");
    d.weights.plan.direct = true;
    d.weights.set_direct_fill_pass_for_test(true);
    d.prefill_chunk_size = 256;
    assert_eq!(
        d.group_limit(0, 9 * 256),
        None,
        "the direct fill reads the live host"
    );
}

/// The arrays a layer-major union holds at its widest (two 256-row windows
/// of the Flash Next shape: streams, MLP inputs, routes, scores, routed rows
/// and sums) fit the group scratch the plan prices, and the price is not
/// far above them.
/// MUTATION: the price leaves out the routed rows -> below the arrays -> red.
#[test]
fn the_group_scratch_prices_the_arrays_a_group_holds() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let mut inner = super::route_step::moe_shape_inner(10);
    inner.decoder.context_window = 768;
    inner.activate_paged_seq(1).unwrap();
    let c = inner.decoder.config.clone();
    let prompt = prompt(512);
    inner
        .decoder
        .prefill_group(&[&prompt[..256], &prompt[256..]], &[], true, false)
        .unwrap();
    let held = inner.decoder.group_held.clone();
    assert_eq!(held.len(), c.num_hidden_layers, "one union per layer");
    let widest = *held.iter().max().unwrap();
    let priced = memory::group_scratch(&c, 2, 256);
    eprintln!("group held {held:?}, priced {priced}");
    assert!(widest <= priced, "held {widest} > priced {priced}");
    assert!(
        priced <= widest + widest / 2,
        "priced {priced} far above held {widest}"
    );
}

/// A decoder like [`streamed_direct_decoder`] served as a model: no MTP
/// head, windows of `width` rows, 1,024 tokens of context, a plan with room
/// for any group.
fn streamed_direct_inner(
    path: &Path,
    config: &config::Config,
    capacity: usize,
    width: usize,
) -> Inner {
    let store = weights::Store::open_fixture(path, None).unwrap();
    let vision = media::Vision::metadata(&serde_json::Value::Null, &store, config).unwrap();
    let mut decoder = decoder::Decoder::new(config.clone(), store).unwrap();
    decoder.weights.stream_fixture_store(capacity).unwrap();
    decoder.register_cache_limit();
    decoder.weights.publish_live();
    decoder.prefill_window_pin = Some(width);
    decoder.residency_changed();
    decoder.planned_scratch = Some(u64::MAX);
    decoder.context_window = 1024;
    decoder.paged = Some(paged::create_fixture(config).unwrap());
    inner_over(decoder, vision, false)
}

/// The union routes of every layer, one entry per group.
fn union_sizes(decoder: &decoder::Decoder) -> Vec<Vec<usize>> {
    (0..decoder.config.num_hidden_layers)
        .map(|layer| {
            decoder
                .union_routes
                .iter()
                .filter(|(l, _)| *l == layer)
                .map(|(_, routes)| routes.len())
                .collect()
        })
        .collect()
}

/// Nine 16-row windows and a 5-row tail on a store whose slots can miss (4
/// experts, top 2, 2 slots) run as groups of 4, 4 and 1 windows, the last
/// with the tail, through `prefill_prompt` and through a paged prompt with
/// no text turn; each gives the logits, history, caches and pool K/V of a
/// window-at-a-time run, and the logits of 4 greedy decode steps after it.
/// MUTATION: one group over every window -> one union of nine windows per
/// layer -> red.
#[test]
fn a_partial_prompt_runs_groups_of_at_most_four_windows() {
    use crate::engine::backend::PagedBackend;
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let prompt = prompt(9 * 16 + 5);
    let groups = vec![vec![4 * 16 * 2; 2]; config.num_hidden_layers];
    for paged in [false, true] {
        let what = if paged {
            "paged prompt"
        } else {
            "prefill_prompt"
        };
        let mut runs = Vec::new();
        for grouped in [true, false] {
            let mut inner = streamed_direct_inner(&path, &config, 2, 16);
            if !grouped {
                inner.decoder.group_pin = Some(false);
            }
            let mut logits = if paged {
                inner.activate_paged_seq(1).unwrap();
                inner
                    .decoder
                    .paged
                    .as_mut()
                    .unwrap()
                    .allocate_suffix_blocks(prompt.len() as u32 + 4)
                    .unwrap();
                let prefix = paged::Prefix {
                    cached: 0,
                    suffix: prompt.len(),
                };
                inner
                    .paged_prefill(&prompt, &prefix, super::prefix_reuse::gpu())
                    .unwrap()
            } else {
                let d = &mut inner.decoder;
                d.weights.begin_turn().unwrap();
                let mut adapter = paged::create_fixture(&config).unwrap();
                adapter.begin_request(1).unwrap();
                adapter
                    .allocate_suffix_blocks(prompt.len() as u32 + 4)
                    .unwrap();
                d.paged = Some(adapter);
                d.prefill_prompt(&prompt).unwrap()
            };
            let mut out = vec![logits.clone()];
            for _ in 0..4 {
                let token = argmax(&logits);
                logits = inner.decoder.step(token).unwrap();
                out.push(logits.clone());
            }
            let sizes = union_sizes(&inner.decoder);
            runs.push((out, inner, sizes));
        }
        let [(grouped, a, sizes), (windowed, b, none)] = <[_; 2]>::try_from(runs).ok().unwrap();
        assert_eq!(sizes, groups, "{what}: groups of four windows");
        assert!(none.iter().all(Vec::is_empty), "{what}");
        for (call, (x, y)) in grouped.iter().zip(&windowed).enumerate() {
            assert_same_bits(x, y, &format!("{what}: logits of call {call}"));
        }
        assert_eq!(a.decoder.history, b.decoder.history, "{what}: history");
        assert_eq!(
            frontier_bytes(&a.decoder),
            frontier_bytes(&b.decoder),
            "{what}: caches and pool K/V"
        );
    }
}

/// The prompt of the scheduled tests: 480 tokens, assistant openings at
/// tokens 100 and 450 (block rows 96 and 448), so on 64-row windows the
/// passes end at 64, 96, then every 64 rows up to 416, then 448 and 480;
/// with a checkpoint every 128 rows the points are 160, 288 (checkpoints)
/// and 448 (replay).
fn scheduled_prompt() -> Vec<u32> {
    let open = [62u32, 63];
    use super::prefix_reuse::prompt;
    [
        &prompt(100, 1)[..],
        &open,
        &prompt(348, 2),
        &open,
        &prompt(28, 3),
    ]
    .concat()
}

/// The direct GGUF fixture (4 experts, top 2, 2 slots) served on 64-row
/// windows with a checkpoint every 128 rows.
fn scheduled_model(path: &Path, config: &config::Config) -> Inner {
    let mut inner = streamed_direct_inner(path, config, 2, 64);
    inner.prefix_states.interval = 128;
    inner.prefix_states.open = vec![62, 63];
    inner
}

/// [`scheduled_model`] with its turn on `seq` admitted for
/// [`scheduled_prompt`].
fn scheduled_inner(path: &Path, config: &config::Config, seq: u32) -> (Inner, paged::Prefix) {
    use crate::engine::hybrid_scheduler::{HybridSchedulerBackend, ScheduledPrefixAdmission};
    let mut inner = scheduled_model(path, config);
    let prompt = scheduled_prompt();
    let prefix = match inner.prepare_scheduled_prefix(seq, &prompt, &[], true, 0, 32) {
        Ok(ScheduledPrefixAdmission::Ready(prefix)) => prefix,
        Ok(_) => panic!("a text prompt is admitted at once"),
        Err(error) => panic!("admission: {error}"),
    };
    inner.scheduled_turn_opened(seq);
    (inner, prefix)
}

/// The stored states: depth, checkpoint flag and every array.
fn stored_cells(inner: &Inner) -> Vec<(usize, bool, Vec<Vec<f32>>)> {
    let mut cells: Vec<_> = inner
        .prefix_states
        .store
        .entries()
        .map(|entry| {
            (
                entry.depth(),
                entry.checkpoint(),
                entry.ram().unwrap().arrays().map(f32s).collect(),
            )
        })
        .collect();
    cells.sort_by_key(|(depth, ..)| *depth);
    cells
}

/// A scheduled turn on a store whose slots can miss: the one whole slice
/// the scheduler pins for a lone grouped turn, and the slices at the
/// backend's group ends, give the window-at-a-time turn's logits, history,
/// caches and pool K/V, stored states and their counters (every point is
/// reached: three published, one retired), and its greedy decode steps.
/// Two groups of two windows run, over rows 160..288 and 288..416.
/// MUTATION: `set_cursor` keeps the group base for every window -> "logits"
/// red.
/// MUTATION: capped runs cut only at the deepest checkpoint -> one group
/// over rows 96..288 holds the point at 160 -> red.
#[test]
fn a_scheduled_partial_group_matches_a_window_major_turn() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let prompt = scheduled_prompt();
    let len = prompt.len();
    let ends = vec![64u32, 96, 160, 288, 416, 448, 480];
    let mut arms = Vec::new();
    for arm in ["windows", "one slice", "group ends"] {
        let (mut inner, prefix) = scheduled_inner(&path, &config, 1);
        if arm == "windows" {
            inner.decoder.group_pin = Some(false);
        }
        let slices = match arm {
            "windows" => inner.scheduler_prefill_ends(1, len as u32, 0).unwrap(),
            "one slice" => vec![len as u32],
            _ => inner.scheduler_prefill_ends(1, len as u32, 0).unwrap(),
        };
        if arm == "group ends" {
            assert_eq!(slices, ends, "the backend's group ends");
        }
        assert_eq!(
            inner.scheduler_whole_prompt_prefill(len as u32, 0),
            arm != "windows",
            "{arm}: whole flag"
        );
        let mut start = 0;
        let mut logits = None;
        for end in slices {
            inner.activate_paged_seq(1).unwrap();
            logits = inner
                .run_scheduled_prefill_slice(
                    1,
                    &prompt,
                    &prefix,
                    start,
                    end as usize,
                    super::prefix_reuse::gpu(),
                    start == 0,
                )
                .unwrap();
            start = end as usize;
        }
        let logits = logits.unwrap();
        let mut out = vec![logits.clone()];
        let mut last = logits;
        for _ in 0..4 {
            last = inner.decoder.step(argmax(&last)).unwrap();
            out.push(last.clone());
        }
        let sizes = union_sizes(&inner.decoder);
        arms.push((arm, out, inner, sizes));
    }
    let (_, want, windows, none) = &arms[0];
    assert!(none.iter().all(Vec::is_empty));
    let counters = windows.prefix_states.counters;
    assert_eq!(
        [counters.published, counters.retired],
        [3, 1],
        "the window-at-a-time turn reaches every point"
    );
    let groups = vec![vec![2 * 64 * 2; 2]; config.num_hidden_layers];
    for (arm, got, inner, sizes) in &arms[1..] {
        assert_eq!(sizes, &groups, "{arm}: two groups of two windows");
        for (call, (x, y)) in got.iter().zip(want).enumerate() {
            assert_same_bits(x, y, &format!("{arm}: logits of call {call}"));
        }
        assert_eq!(inner.decoder.history, windows.decoder.history, "{arm}");
        assert_eq!(
            frontier_bytes(&inner.decoder),
            frontier_bytes(&windows.decoder),
            "{arm}: caches and pool K/V"
        );
        assert_eq!(inner.prefix_states.counters, counters, "{arm}: counters");
        assert_eq!(
            stored_cells(inner),
            stored_cells(windows),
            "{arm}: stored states"
        );
    }
}

/// The scheduler pins group ends and the whole flag only while the turn is
/// the only open turn: a second open turn gets the pass ends and no flag,
/// and its close gives the lone turn its groups back. A plan with no room
/// for two windows pins pass ends.
/// MUTATION: no open-turn check -> the flag with two open turns -> red.
#[test]
fn group_ends_are_pinned_only_for_a_lone_turn() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let len = scheduled_prompt().len() as u32;
    let (mut inner, _) = scheduled_inner(&path, &config, 1);
    let passes: Vec<u32> = inner.prefix_states.turns[&1]
        .ends
        .iter()
        .map(|&end| end as u32)
        .collect();
    assert_eq!(passes, [64, 96, 160, 224, 288, 352, 416, 448, 480]);
    let pinned = |inner: &Inner| {
        (
            inner.scheduler_whole_prompt_prefill(len, 0),
            inner.scheduler_prefill_ends(1, len, 0).unwrap(),
        )
    };
    let groups = vec![64u32, 96, 160, 288, 416, 448, 480];
    assert_eq!(pinned(&inner), (true, groups.clone()), "one open turn");
    inner.scheduled_turn_opened(2);
    assert_eq!(pinned(&inner), (false, passes.clone()), "two open turns");
    inner.scheduled_turn_closed(2);
    assert_eq!(pinned(&inner), (true, groups), "alone again");
    inner.decoder.planned_scratch = Some(0);
    assert_eq!(pinned(&inner), (false, passes), "no room for a group");
}

/// A Stop inside a capped group (the one over rows 160..288, which ends at
/// the checkpoint 288) fails the prefill and the turn aborts; only the state
/// at 160, published before the group, stays. The same prompt on another
/// row resumes there and gives a cold window-at-a-time turn's logits and
/// greedy decode steps.
/// MUTATION: a failed group still reaches its end point -> a state at 288
/// from the reset decoder -> red.
#[test]
fn a_cancelled_group_leaves_no_reusable_block() {
    use crate::engine::backend::PagedBackend;
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let prompt = scheduled_prompt();
    let (mut inner, prefix) = scheduled_inner(&path, &config, 1);
    let flag = Arc::new(AtomicBool::new(false));
    inner.decoder.cancelled = Some(Arc::clone(&flag));
    let stop = Arc::clone(&flag);
    inner.decoder.after_sweep = Some(Box::new(move |decoder, layer| {
        if decoder.history.len() == 160 && layer == 0 {
            stop.store(true, Ordering::Release);
        }
    }));
    inner.activate_paged_seq(1).unwrap();
    let error = inner
        .run_scheduled_prefill_slice(
            1,
            &prompt,
            &prefix,
            0,
            prompt.len(),
            super::prefix_reuse::gpu(),
            true,
        )
        .err()
        .expect("the Stop fails the prefill");
    assert!(error.reason.contains("cancelled"), "{}", error.reason);
    inner.abort_paged_turn();
    inner.scheduled_turn_closed(1);
    inner.decoder.cancelled = None;
    inner.decoder.after_sweep = None;
    let depths: Vec<_> = stored_cells(&inner)
        .into_iter()
        .map(|(depth, checkpoint, _)| (depth, checkpoint))
        .collect();
    assert_eq!(depths, [(160, true)], "only the state before the group");

    let run = |inner: &mut Inner, seq: u32| {
        inner.activate_paged_seq(seq).unwrap();
        let prefix = inner.prime_prefix_state(&prompt, true, 32, &[], 0).unwrap();
        let cached = prefix.cached;
        let mut last = inner
            .paged_prefill(&prompt[cached..], &prefix, super::prefix_reuse::gpu())
            .unwrap();
        let mut out = vec![f32s(&last)];
        for _ in 0..4 {
            last = inner.decoder.step(argmax(&last)).unwrap();
            out.push(f32s(&last));
        }
        (cached, out)
    };
    let (cached, resumed) = run(&mut inner, 2);
    assert_eq!(cached, 160, "the resume row");
    let mut cold = scheduled_model(&path, &config);
    cold.decoder.group_pin = Some(false);
    let (cold_cached, want) = run(&mut cold, 1);
    assert_eq!(cold_cached, 0);
    assert_eq!(resumed, want, "logits and greedy steps of a cold turn");
}

/// A lone turn on a store whose slots can miss is pinned to the backend's
/// group ends with the whole flag, and the scheduler steps one group per
/// step, each its own row past a budget below every group.
/// MUTATION: ignore the backend's ends when whole -> the one end 480 -> red.
#[test]
fn a_whole_slice_keeps_the_backends_ends() {
    use crate::engine::hybrid_scheduler::prefill_breaks;
    use crate::engine::scheduler::{Prepared, Scheduler, StepKind, TurnState};
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let len = scheduled_prompt().len() as u32;
    let (inner, _) = scheduled_inner(&path, &config, 1);
    let (breaks, whole) = prefill_breaks(&inner, 1, 0, len, Vec::new());
    let ends = vec![64u32, 96, 160, 288, 416, 448, 480];
    assert_eq!((&breaks, whole), (&ends, true), "the backend's group ends");
    let mut scheduler = Scheduler::<(), (), ()>::new(4, 16).unwrap();
    let mut turn = TurnState::new(1, vec![1; len as usize], 0, breaks, None, ()).unwrap();
    turn.whole_prompt_prefill = whole;
    scheduler.enqueue_turn(turn).unwrap();
    let mut steps = Vec::new();
    while let Prepared::Plan(plan) = scheduler.prepare_step(None).unwrap() {
        let rows: Vec<_> = plan
            .rows
            .iter()
            .map(|row| (row.kind, row.token_start + row.num_tokens))
            .collect();
        scheduler.turn_mut(1).unwrap().num_computed_tokens += plan.rows[0].num_tokens;
        steps.push(rows);
        assert!(steps.len() <= ends.len(), "{steps:?}");
    }
    let want: Vec<_> = ends
        .iter()
        .map(|&end| vec![(StepKind::Prefill, end)])
        .collect();
    assert_eq!(steps, want, "one group per step, past the budget");
}

/// A tokenizer over the fixture's vocabulary (words `w0`..) whose chat
/// template renders `b` when the first message says "b", else `a`.
fn two_prompt_tokenizer(vocab: usize, a: &[u32], b: &[u32]) -> Arc<Qwen3Tokenizer> {
    let words = |tokens: &[u32]| {
        tokens
            .iter()
            .map(|token| format!("w{token}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let template = format!(
        "{{% if messages[0]['content'] == 'b' %}}{}{{% else %}}{}{{% endif %}}",
        words(b),
        words(a)
    );
    let vocab: serde_json::Map<String, serde_json::Value> = (0..vocab)
        .map(|id| (format!("w{id}"), serde_json::Value::from(id)))
        .collect();
    let dir = Temp::new();
    std::fs::write(
        dir.0.join("tokenizer_config.json"),
        serde_json::json!({ "chat_template": template }).to_string(),
    )
    .unwrap();
    let path = dir.0.join("tokenizer.json");
    std::fs::write(
        &path,
        serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": { "type": "WhitespaceSplit" },
            "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "w0" }
        })
        .to_string(),
    )
    .unwrap();
    Arc::new(Qwen3Tokenizer::load_from_file_sync(path.to_str().unwrap()).unwrap())
}

/// A user turn saying `content` for the cache owner `content`, greedy, 4
/// new tokens.
fn start_command(
    content: &str,
) -> (
    crate::engine::model_command::ModelCommand<std::convert::Infallible>,
    TurnReply,
) {
    use crate::engine::cmd::ChatCmd;
    use crate::engine::model_command::ModelCommand;
    let messages = vec![
        serde_json::from_value(serde_json::json!({"role": "user", "content": content})).unwrap(),
    ];
    let config = crate::engine::types::ChatConfig {
        max_new_tokens: Some(4),
        temperature: Some(0.0),
        report_performance: Some(true),
        cache_owner_id: Some(content.to_owned()),
        ..Default::default()
    };
    let (reply, rx) = tokio::sync::oneshot::channel();
    let (_, stream_rx) = crate::model_thread::stream_channel(256);
    let command = ChatCmd::SessionStart {
        messages,
        config,
        reply,
        cancelled: Arc::new(AtomicBool::new(false)),
    };
    (
        ModelCommand::from_chat(command),
        TurnReply {
            kind: TurnKind::Start,
            rx,
            stream_rx,
            streamed: String::new(),
        },
    )
}

/// A second request admitted while a lone turn on a store whose slots can
/// miss is in its group over rows 160..288 (pinned to the group ends with
/// the whole flag) runs its first prefill row in the very next step, beside
/// one pass of the first turn: the first turn is pinned again to its pass
/// ends. Each turn's text is the text it gives alone.
/// MUTATION: no re-pin on admission -> the first turn keeps its whole
/// groups and the newcomer's first row waits for the first turn's whole
/// prompt (at 480, not 352) -> red.
#[test]
fn a_newcomer_waits_at_most_one_group() {
    use crate::engine::hybrid_scheduler::HybridSchedulerState;
    use crate::model_thread::LoopControl;
    let _host = memory::scoped_fixture_host();
    let temp = Temp::new();
    let path = temp.0.join("direct.gguf");
    let config = write_direct_gguf_fixture(&path);
    let a = scheduled_prompt();
    let b = super::prefix_reuse::prompt(200, 5);
    let model = || {
        let mut inner = scheduled_model(&path, &config);
        inner.decoder.paged = Some(paged::create_with_blocks(&config, 48, 48).unwrap());
        inner.max_context = Some(1024);
        inner.tokenizer = two_prompt_tokenizer(config.vocab_size, &a, &b);
        let mut state = HybridSchedulerState::new(inner).unwrap();
        state.enable_continuous_batching();
        state
    };
    let alone = |content: &str| {
        let mut state = model();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        let (command, mut reply) = start_command(content);
        assert!(sender.send(command).is_ok());
        finish_turn(&mut state, &mut receiver, &mut reply)
            .unwrap()
            .0
    };
    let (want_a, want_b) = (alone("a"), alone("b"));

    let mut state = model();
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let (command, mut reply_a) = start_command("a");
    assert!(sender.send(command).is_ok());
    let pinned = |state: &mut HybridSchedulerState<Inner>, seq| {
        state.turn_mut(seq).map(|turn| {
            (
                turn.prompt_tokens,
                turn.whole_prompt_prefill,
                turn.pinned_prefill_breaks.clone(),
            )
        })
    };
    let computed = |state: &mut HybridSchedulerState<Inner>, seq| {
        state.turn_mut(seq).map(|turn| turn.num_computed_tokens)
    };
    let mut alone_steps = Vec::new();
    while computed(&mut state, 1) != Some(288) {
        assert!(alone_steps.len() < 16, "{alone_steps:?}");
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
        alone_steps.push(computed(&mut state, 1));
        if alone_steps.len() == 1 {
            assert_eq!(
                pinned(&mut state, 1),
                Some((480, true, vec![64, 96, 160, 288, 416, 448, 480])),
                "a lone turn: the group ends and the whole flag"
            );
        }
    }
    assert_eq!(
        alone_steps,
        [Some(64), Some(96), Some(160), Some(288)],
        "one group per step"
    );

    let (command, mut reply_b) = start_command("b");
    assert!(sender.send(command).is_ok());
    assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
    let next = (computed(&mut state, 1), computed(&mut state, 2));
    assert_eq!(
        next,
        (Some(352), Some(64)),
        "the newcomer's first row runs in the step after the group, beside one pass"
    );
    assert_eq!(
        pinned(&mut state, 1),
        Some((480, false, vec![352, 416, 448, 480])),
        "the first turn pinned again to its pass ends from 288"
    );
    assert_eq!(
        pinned(&mut state, 2).map(|(prompt, whole, _)| (prompt, whole)),
        Some((200, false)),
        "the newcomer"
    );
    let (mut got_a, mut got_b) = (None, None);
    for _ in 0..256 {
        if got_a.is_none() {
            got_a = reply_a.poll();
        }
        if got_b.is_none() {
            got_b = reply_b.poll();
        }
        if got_a.is_some() && got_b.is_some() {
            break;
        }
        assert!(matches!(state.drive(&mut receiver), LoopControl::Continue));
    }
    let (got_a, got_b) = (got_a.unwrap().unwrap().0, got_b.unwrap().unwrap().0);
    eprintln!("alone {want_a:?} {want_b:?}; together {got_a:?} {got_b:?}");
    assert!(!want_a.is_empty() && !want_b.is_empty(), "texts to compare");
    assert_eq!(got_a, want_a, "the first turn's text");
    assert_eq!(got_b, want_b, "the newcomer's text");
}
