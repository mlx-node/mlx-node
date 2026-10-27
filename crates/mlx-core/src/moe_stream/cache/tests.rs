use super::*;

/// xorshift64*: deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn wanted(r: &mut Rng, n: u32, k: usize) -> Vec<u32> {
    let mut w = Vec::new();
    while w.len() < k {
        let e = r.below(u64::from(n)) as u32;
        if !w.contains(&e) {
            w.push(e);
        }
    }
    w.sort_unstable();
    w
}

/// One random layer step: plan, commit (or release, 1 in 16), complete
/// (most of the time). Checks every plan line it can.
fn step(c: &mut Cache, r: &mut Rng, layers: usize, n: u32, k: usize) {
    let layer = r.below(layers as u64) as usize;
    let w = wanted(r, n, k);
    let decode = r.below(8) != 0;
    let before: Vec<Option<u32>> = w.iter().map(|&e| c.lookup(layer, e)).collect();
    let (epoch, src) = c.plan(layer, &w, decode);
    assert_eq!(src.len(), w.len());
    let mut used = BTreeSet::new();
    for (s, b) in src.iter().zip(&before) {
        match *s {
            Source::Hit(slot) => assert_eq!(Some(slot), *b, "hit of an unmapped expert"),
            Source::Victim(slot) => {
                assert!(decode && b.is_none(), "victim for a hit or in prefill");
                assert!(c.occ(slot).is_none(), "victim still mapped");
            }
            Source::Staging => assert!(b.is_none()),
        }
        if let Source::Hit(slot) | Source::Victim(slot) = *s {
            assert!(
                used.insert(slot),
                "slot {slot} serves two experts in one step"
            );
        }
    }
    for (&e, s) in w.iter().zip(&src) {
        if let Source::Victim(slot) = *s {
            if r.below(16) == 0 {
                c.release(slot);
            } else {
                c.commit(slot, layer, e);
            }
        }
    }
    if r.below(4) != 0 {
        c.complete(epoch);
    }
}

#[test]
fn bijection_after_1e5_random_ops() {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    for &(slots, layers, n, k) in &[
        (1, 3, 12, 6),
        (16, 8, 12, 6),
        (64, 40, 384, 6),
        (240, 40, 384, 6),
    ] {
        let mut c = Cache::lru(slots, layers, n as usize);
        for i in 0..100_000 {
            step(&mut c, &mut r, layers, n, k);
            if i % 997 == 0 {
                c.check().unwrap();
            }
            if r.below(50_000) == 0 {
                c.clear();
                assert_eq!(c.occupied(), 0);
            }
        }
        c.check().unwrap();
        assert!(c.occupied() <= slots);
        println!("slots {slots}: {} occupied after 1e5 ops", c.occupied());
    }
}

/// A victim never holds an expert of the step's wanted set, even when
/// that expert is the least recently used and the only other slot is busy.
#[test]
fn wanted_set_never_evicted() {
    let mut c = Cache::lru(2, 2, 8);
    for (e, slot) in [(1u32, 0u32), (2, 1)] {
        let (ep, src) = c.plan(0, &[e], true);
        assert_eq!(src, vec![Source::Victim(slot)]);
        c.commit(slot, 0, e);
        c.complete(ep);
    }
    // LRU order: slot 0 (layer 0 expert 1) then slot 1 (expert 2).
    let (ep, src) = c.plan(0, &[1, 5], true);
    assert_eq!(src, vec![Source::Hit(0), Source::Victim(1)]);
    c.commit(1, 0, 5);
    c.complete(ep);
    // Expert 1 is now the LRU slot but wanted: the miss takes slot 1.
    let (_, src) = c.plan(0, &[1, 6], true);
    assert_eq!(src, vec![Source::Hit(0), Source::Victim(1)]);
    // Both slots busy in this epoch: a third expert goes to staging.
    let mut c = Cache::lru(1, 1, 8);
    let (ep, src) = c.plan(0, &[3], true);
    c.commit(0, 0, 3);
    c.complete(ep);
    assert_eq!(src, vec![Source::Victim(0)]);
    let (_, src) = c.plan(0, &[3, 4], true);
    assert_eq!(src, vec![Source::Hit(0), Source::Staging]);
    // Random: no victim is ever a wanted expert's slot.
    let mut r = Rng(7);
    let mut c = Cache::lru(8, 4, 16);
    for _ in 0..20_000 {
        let layer = r.below(4) as usize;
        let w = wanted(&mut r, 16, 6);
        let mapped: Vec<u32> = w.iter().filter_map(|&e| c.lookup(layer, e)).collect();
        let (ep, src) = c.plan(layer, &w, true);
        for (&e, s) in w.iter().zip(&src) {
            if let Source::Victim(v) = *s {
                assert!(!mapped.contains(&v), "victim {v} held a wanted expert");
                c.commit(v, layer, e);
            }
        }
        c.complete(ep);
    }
    c.check().unwrap();
}

/// A slot whose last use is not complete is never a victim; once the epoch
/// completes it is.
#[test]
fn epoch_protected_slot_never_chosen() {
    let mut c = Cache::lru(1, 2, 8);
    let (ep1, src) = c.plan(0, &[2], true);
    assert_eq!(src, vec![Source::Victim(0)]);
    c.commit(0, 0, 2);
    // epoch ep1 still running on the GPU
    let (_, src) = c.plan(1, &[3], true);
    assert_eq!(src, vec![Source::Staging], "in-flight slot evicted");
    assert_eq!(c.lookup(0, 2), Some(0));
    c.complete(ep1);
    let (_, src) = c.plan(1, &[3], true);
    assert_eq!(src, vec![Source::Victim(0)]);
    assert_eq!(c.lookup(0, 2), None);
    // Random: every victim's previous use is complete.
    let mut r = Rng(11);
    let mut c = Cache::lru(6, 3, 12);
    let mut pending: Vec<u64> = Vec::new();
    for _ in 0..20_000 {
        let layer = r.below(3) as usize;
        let w = wanted(&mut r, 12, 3);
        let last: Vec<u64> = (0..6).map(|s| c.last_use[s]).collect();
        let done = c.completed;
        let (ep, src) = c.plan(layer, &w, true);
        for (&e, s) in w.iter().zip(&src) {
            if let Source::Victim(v) = *s {
                assert!(
                    last[v as usize] <= done,
                    "victim {v} last used in an open epoch"
                );
                c.commit(v, layer, e);
            }
        }
        pending.push(ep);
        if r.below(3) == 0 {
            let upto = pending[r.below(pending.len() as u64) as usize];
            c.complete(upto);
            pending.retain(|&p| p > upto);
        }
    }
    c.check().unwrap();
}

/// `clear` (bank `Empty`) leaves no mapping: every later lookup misses and
/// the next plan reads every expert again.
#[test]
fn empty_clears_directory() {
    let mut c = Cache::lru(4, 2, 8);
    for (l, e) in [(0, 1u32), (0, 2), (1, 1), (1, 7)] {
        let (ep, src) = c.plan(l, &[e], true);
        let Source::Victim(v) = src[0] else {
            panic!("{src:?}")
        };
        c.commit(v, l, e);
        c.complete(ep);
    }
    assert_eq!(c.occupied(), 4);
    c.clear();
    c.check().unwrap();
    assert_eq!(c.occupied(), 0);
    for (l, e) in [(0, 1u32), (0, 2), (1, 1), (1, 7)] {
        assert_eq!(c.lookup(l, e), None);
    }
    let (_, src) = c.plan(1, &[1, 7], true);
    assert!(
        src.iter().all(|s| matches!(s, Source::Victim(_))),
        "{src:?}"
    );
    // prefill never takes victims
    let (_, src) = c.plan(0, &[1, 2], false);
    assert_eq!(src, vec![Source::Staging; 2]);
}

/// LFU-decay picks the slot with the lowest decayed use count, checked
/// against a direct f64 reference (every score x 2^(-1/h) per use, +1 on
/// the touched slot) across several rescales of the inflated scores.
#[test]
fn lfu_decay_orders_by_decayed_count() {
    let h = 37.0;
    let mut p = LfuDecay::new(h);
    let mut want = [None::<f64>; 8];
    let mut r = Rng(5);
    let decay = (-1.0f64 / h).exp2();
    for i in 0..60_000u32 {
        // skewed: low slots used more often
        let s = (r.below(8) * r.below(8) / 7) as u32;
        for v in want.iter_mut().flatten() {
            *v *= decay;
        }
        *want[s as usize].get_or_insert(0.0) += 1.0;
        p.touch(s);
        if r.below(500) == 0 {
            let f = r.below(8) as u32;
            p.forget(f);
            want[f as usize] = None;
        }
        let got = p.victim(&|_| true);
        let min = want.iter().flatten().copied().fold(f64::INFINITY, f64::min);
        match got {
            Some(g) => {
                let v = want[g as usize].expect("victim is a forgotten slot");
                assert!(
                    v <= min * (1.0 + 1e-9) + 1e-300,
                    "tick {i}: victim {g} score {v} > min {min}"
                );
            }
            None => assert!(want.iter().all(Option::is_none)),
        }
        // `ok` filters: the pick is the lowest accepted score
        if let Some(g) = got {
            let other = p.victim(&|s| s != g);
            if let Some(o) = other {
                assert_ne!(o, g);
                let rest = want
                    .iter()
                    .enumerate()
                    .filter(|&(s, _)| s as u32 != g)
                    .filter_map(|(_, v)| *v)
                    .fold(f64::INFINITY, f64::min);
                assert!(want[o as usize].unwrap() <= rest * (1.0 + 1e-9) + 1e-300);
            }
        }
    }
    assert!(p.base > 0, "no rescale happened");
}

/// A long-unused slot loses to a recently used one once its count has
/// decayed below; LRU-like at a short half-life, LFU-like at a long one.
#[test]
fn lfu_decay_half_life_trades_frequency_for_recency() {
    for (h, want) in [(1.0, 0u32), (1e6, 1u32)] {
        let mut p = LfuDecay::new(h);
        for _ in 0..10 {
            p.touch(0);
        }
        p.touch(1);
        for _ in 0..8 {
            p.touch(2);
        }
        // short half-life: slot 0's 10 uses are old, slot 1's single use is
        // newer -> the oldest (0) goes first; long half-life: plain counts
        // 10 / 1 / 8 -> slot 1 goes first
        assert_eq!(p.victim(&|s| s != 2), Some(want), "half-life {h}");
    }
}

#[test]
fn bijection_after_1e5_random_ops_lfu_decay() {
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    for &(slots, layers, n, k, h) in &[(16, 8, 12, 6, 50.0), (240, 40, 384, 6, 240.0 * 16.0)] {
        let mut c = Cache::lfu_decay(slots, layers, n as usize, h);
        for i in 0..100_000 {
            step(&mut c, &mut r, layers, n, k);
            if i % 997 == 0 {
                c.check().unwrap();
            }
        }
        c.check().unwrap();
        assert!(c.occupied() <= slots);
    }
}

/// `Seeded` prefill: at most `slots / layers` victims per layer per turn,
/// most-used experts first (ties to the lowest id); `Lru` prefill takes
/// none; `new_turn` refills the quota.
#[test]
fn seeded_prefill_quota_by_use_count() {
    // 8 slots, 2 layers: quota 4 per layer per turn
    let mut c = Cache::lru(8, 2, 12).with_mode(CacheMode::Seeded);
    let w = [1, 2, 3, 4, 5, 6];
    let uses = [3, 1, 5, 1, 2, 1];
    let (e1, src) = c.plan_step(0, &w, &uses, false, &|_| false);
    let picked: Vec<u32> = w
        .iter()
        .zip(&src)
        .filter(|(_, s)| matches!(s, Source::Victim(_)))
        .map(|(&e, _)| e)
        .collect();
    // counts 5 (3), 3 (1), 2 (5), then the lowest id among the 1s (2)
    assert_eq!(picked, vec![1, 2, 3, 5]);
    for (&e, s) in w.iter().zip(&src) {
        if let Source::Victim(v) = *s {
            c.commit(v, 0, e);
        }
    }
    c.complete(e1);
    // same layer, same turn: the quota is spent
    let (e2, src) = c.plan_step(0, &[7, 8], &[4, 4], false, &|_| false);
    assert!(src.iter().all(|s| *s == Source::Staging), "{src:?}");
    c.complete(e2);
    // the other layer has its own quota
    let (e3, src) = c.plan_step(1, &[7, 8], &[1, 1], false, &|_| false);
    assert!(
        src.iter().all(|s| matches!(s, Source::Victim(_))),
        "{src:?}"
    );
    for (&e, s) in [7, 8].iter().zip(&src) {
        if let Source::Victim(v) = *s {
            c.commit(v, 1, e);
        }
    }
    c.complete(e3);
    c.new_turn();
    let (_, src) = c.plan_step(0, &[7, 8], &[1, 1], false, &|_| false);
    assert!(
        src.iter().all(|s| matches!(s, Source::Victim(_))),
        "{src:?}"
    );
    c.check().unwrap();
    // Lru: prefill never takes a victim
    let mut l = Cache::lru(8, 2, 12);
    let (_, src) = l.plan_step(0, &w, &uses, false, &|_| false);
    assert!(src.iter().all(|s| *s == Source::Staging));
    // fewer slots than layers: quota 0
    let mut z = Cache::lru(1, 2, 12).with_mode(CacheMode::Seeded);
    let (_, src) = z.plan_step(0, &w, &uses, false, &|_| false);
    assert!(src.iter().all(|s| *s == Source::Staging));
}

/// `Seeded` decode: a miss whose staging slot holds it stays `Staging` (no
/// victim, no read); `Lru` reads it into a victim as before.
#[test]
fn seeded_decode_uses_valid_staging_copy() {
    let staged = |e: u32| e == 2;
    let mut s = Cache::lru(8, 2, 12).with_mode(CacheMode::Seeded);
    let (_, src) = s.plan_step(0, &[1, 2, 3], &[], true, &staged);
    assert!(matches!(src[0], Source::Victim(_)));
    assert_eq!(src[1], Source::Staging);
    assert!(matches!(src[2], Source::Victim(_)));
    let mut l = Cache::lru(8, 2, 12);
    let (_, src) = l.plan_step(0, &[1, 2, 3], &[], true, &staged);
    assert!(
        src.iter().all(|x| matches!(x, Source::Victim(_))),
        "{src:?}"
    );
}

/// The 1e5-op bijection run under `Seeded` (prefill victims, staged
/// decode misses), checking the per-layer per-turn quota.
#[test]
fn bijection_after_1e5_random_ops_seeded() {
    let mut r = Rng(0x2545_f491_4f6c_dd1d);
    for &(slots, layers, n, k) in &[(1, 3, 12, 6), (16, 8, 12, 6), (240, 40, 384, 6)] {
        let mut c = Cache::lru(slots, layers, n as usize).with_mode(CacheMode::Seeded);
        let quota = slots / layers;
        let mut seeded = vec![0usize; layers];
        for i in 0..100_000 {
            if r.below(64) == 0 {
                c.new_turn();
                seeded.fill(0);
            }
            let layer = r.below(layers as u64) as usize;
            let decode = r.below(4) != 0;
            let rows = if decode { 1 } else { 1 + r.below(8) as usize };
            let mut ids = Vec::new();
            for _ in 0..rows {
                ids.extend(wanted(&mut r, n, k));
            }
            let mut w = ids.clone();
            w.sort_unstable();
            w.dedup();
            let uses: Vec<u32> = w
                .iter()
                .map(|e| ids.iter().filter(|x| *x == e).count() as u32)
                .collect();
            let st: Vec<u32> = w.iter().copied().filter(|_| r.below(4) == 0).collect();
            let before: Vec<Option<u32>> = w.iter().map(|&e| c.lookup(layer, e)).collect();
            let (epoch, src) = c.plan_step(layer, &w, &uses, decode, &|e| st.contains(&e));
            let mut used = BTreeSet::new();
            for ((s, b), e) in src.iter().zip(&before).zip(&w) {
                match *s {
                    Source::Hit(slot) => assert_eq!(Some(slot), *b),
                    Source::Victim(slot) => {
                        assert!(b.is_none() && c.occ(slot).is_none());
                        assert!(
                            !(decode && st.contains(e)),
                            "staged decode miss got a victim"
                        );
                        if !decode {
                            seeded[layer] += 1;
                        }
                    }
                    Source::Staging => assert!(b.is_none()),
                }
                if let Source::Hit(slot) | Source::Victim(slot) = *s {
                    assert!(used.insert(slot));
                }
            }
            assert!(
                seeded[layer] <= quota,
                "layer {layer}: {} > quota {quota}",
                seeded[layer]
            );
            for (&e, s) in w.iter().zip(&src) {
                if let Source::Victim(slot) = *s {
                    if r.below(16) == 0 {
                        c.release(slot);
                    } else {
                        c.commit(slot, layer, e);
                    }
                }
            }
            if r.below(4) != 0 {
                c.complete(epoch);
            }
            if i % 997 == 0 {
                c.check().unwrap();
            }
        }
        c.check().unwrap();
        println!(
            "seeded slots {slots}: {} occupied after 1e5 ops",
            c.occupied()
        );
    }
}

/// Turn end under `Seeded`: a decode use served from staging moves into a
/// victim when its staging copy is still valid; invalid copies, pairs the
/// cache already holds, and a second call (the list is per turn) give none.
#[test]
fn seeded_turn_end_promotes_valid_staging_copies() {
    let mut c = Cache::lru(4, 2, 8).with_mode(CacheMode::Seeded);
    let (e1, src) = c.plan_step(0, &[1, 2, 3], &[], true, &|e| e != 1);
    assert!(matches!(src[0], Source::Victim(_)));
    assert_eq!(&src[1..], &[Source::Staging, Source::Staging]);
    if let Source::Victim(v) = src[0] {
        c.commit(v, 0, 1);
    }
    c.complete(e1);
    let (e2, _) = c.plan_step(1, &[2], &[], true, &|_| true);
    c.complete(e2);
    // (0, 3) no longer valid: only (0, 2) then (1, 2), in use order
    let moved = c.promote_staged(&|l, e| !(l == 0 && e == 3));
    assert_eq!(
        moved.iter().map(|m| (m.1, m.2)).collect::<Vec<_>>(),
        vec![(0, 2), (1, 2)]
    );
    for &(slot, l, e) in &moved {
        assert_eq!(c.lookup(l, e), Some(slot));
    }
    c.check().unwrap();
    assert!(c.promote_staged(&|_, _| true).is_empty());
    // the promoted slots are victims again at once (copies are host work)
    let (_, src) = c.plan_step(0, &[5, 6, 7], &[], true, &|_| false);
    assert!(
        src.iter().all(|s| matches!(s, Source::Victim(_))),
        "{src:?}"
    );
    // Lru never records staged uses
    let mut l = Cache::lru(4, 2, 8);
    let (e3, _) = l.plan_step(0, &[2], &[], true, &|_| true);
    l.complete(e3);
    assert!(l.promote_staged(&|_, _| true).is_empty());
}

/// A `Seeded` prefill seed takes a free slot or one of its own layer's
/// residents (LRU among them), never another layer's: the same prefill pass
/// may still hit those.
#[test]
fn seeded_prefill_never_evicts_other_layers() {
    // 4 slots, 2 layers: quota 2
    let mut c = Cache::lru(4, 2, 8).with_mode(CacheMode::Seeded);
    for (layer, w) in [(1, [1, 2, 3]), (0, [4, 5, 6])] {
        let (ep, src) = c.plan_step(layer, &w[..2], &[], true, &|_| false);
        for (&e, s) in w.iter().zip(&src) {
            if let Source::Victim(v) = *s {
                c.commit(v, layer, e);
            }
        }
        c.complete(ep);
    }
    c.new_turn();
    // layer 1 holds the two oldest slots; a layer-0 seed must take layer 0's
    let (ep, src) = c.plan_step(0, &[3, 4], &[1, 1], false, &|_| false);
    assert_eq!(src[1], Source::Hit(c.lookup(0, 4).unwrap()));
    let Source::Victim(v) = src[0] else {
        panic!("no seed: {src:?}");
    };
    assert_eq!(c.occ(v), None);
    c.commit(v, 0, 3);
    c.complete(ep);
    assert!(
        c.lookup(1, 1).is_some() && c.lookup(1, 2).is_some(),
        "layer 1 lost a resident"
    );
    assert!(
        c.lookup(0, 5).is_none(),
        "layer 0's LRU resident should have made room"
    );
    // only layer-1 residents (and the busy step's own hits) left for layer 0: staged
    let mut d = Cache::lru(2, 2, 8).with_mode(CacheMode::Seeded);
    let (ep, src) = d.plan_step(1, &[1, 2], &[], true, &|_| false);
    for (&e, s) in [1, 2].iter().zip(&src) {
        if let Source::Victim(v) = *s {
            d.commit(v, 1, e);
        }
    }
    d.complete(ep);
    let (_, src) = d.plan_step(0, &[3], &[5], false, &|_| false);
    assert_eq!(src, vec![Source::Staging]);
    d.check().unwrap();
}
