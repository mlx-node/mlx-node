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
