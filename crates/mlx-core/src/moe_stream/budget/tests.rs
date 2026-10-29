use super::*;

const G: u64 = 1 << 30;
/// Gate+up+down MXFP4 bytes of one expert (real file).
const E: u64 = 18_800_640;
/// Measured dense arrays (real smoke: 8.536 GiB).
const DENSE: u64 = 9_165_803_968;
/// Measured Metal `max_recommended_working_set_size` on the 128 GiB host.
const WS: u64 = 115_448_725_504;
const PHYS: u64 = 128 * G;

fn model() -> ModelSize {
    ModelSize {
        dense: DENSE,
        slot: E,
        n_experts: 384,
        ctx: 16_384,
    }
}

/// A host whose `host_used` is exactly `used` bytes.
fn host(used: u64) -> HostMemory {
    HostMemory {
        physical: PHYS,
        wired: used,
        ..Default::default()
    }
}

/// `others` + `own` on the host, `own` all MLX active.
fn at(others: u64, own: u64) -> (HostMemory, Own) {
    (
        host(others + own),
        Own {
            active: own,
            ..Default::default()
        },
    )
}

/// The plan's fixed sum written out by hand (no `Terms`).
fn fixed_by_hand(layers: u64, ws: u64) -> u64 {
    let ctx = 16_384u64;
    let kv = 10_485_760 + 6_400 * ctx;
    let compute = 1_134_024_803 + 772_038 * ctx;
    DENSE
        + layers * 384 * E
        + kv
        + kv / 5
        + compute
        + compute / 5
        + (G / 4 + 40 * 1024 * ctx)
        + (4 * G).max(ws / 20)
        + 2 * G
}

fn admit_at(others: u64, own: u64, limits: &Limits) -> Result<Admission> {
    let (h, o) = at(others, own);
    admit(&model(), limits, &h, &o, WS)
}

/// The staging rule written out by hand: a second layer when the cache does
/// not need its bytes, or when at least `SECOND_STAGING_MIN_CACHE` is left.
fn layers_by_hand(budget: u64, cap: Option<u64>, ws: u64) -> u64 {
    let (one, two) = (fixed_by_hand(1, ws), fixed_by_hand(2, ws));
    let cache1 = (budget - one).min(cap.unwrap_or(u64::MAX)) / E * E;
    if two + cache1 <= budget || (two <= budget && budget - two >= SECOND_STAGING_MIN_CACHE) {
        2
    } else {
        1
    }
}

/// Others' bytes that leave exactly `cache` after two staging layers.
fn others_for_room2(cache: u64) -> u64 {
    PHYS - RESERVE - fixed_by_hand(2, WS) - cache
}

#[test]
fn terms_match_the_s14c_input() {
    let t = Terms::new(&model(), WS);
    println!("{t:?}");
    assert!(
        (gib(t.compute) - 15.4038).abs() < 1e-3,
        "{}",
        gib(t.compute)
    );
    assert_eq!(t.staging_layer, 7_219_445_760);
    assert_eq!(t.driver, WS / 20);
    assert_eq!(t.fixed(1), fixed_by_hand(1, WS));
    assert_eq!(t.fixed(2), fixed_by_hand(2, WS));
}

#[test]
fn today_others_50() {
    let a = admit_at(50 * G, 0, &Limits::default()).unwrap();
    println!("today: {a}");
    let live = PHYS - 50 * G - RESERVE;
    assert_eq!(a.live, live);
    assert_eq!(a.others, 50 * G);
    let layers = layers_by_hand(live, None, WS);
    assert_eq!(a.staging_layers, layers);
    assert_eq!(a.fixed, fixed_by_hand(layers, WS));
    assert_eq!(a.cache_slots, (live - fixed_by_hand(layers, WS)) / E);
    assert_eq!(a.bound, Bound::Live);
    assert_eq!(a.wired_limit(), a.fixed + a.cache_slots * E);
    assert!(a.live - a.wired_limit() < E);
}

#[test]
fn quiet_others_25_and_ws_bound() {
    // live (91 GiB) < ws (107.52 GiB) at others 25: live binds
    let q = admit_at(25 * G, 0, &Limits::default()).unwrap();
    println!("quiet: {q}");
    let live = PHYS - 25 * G - RESERVE;
    assert_eq!(q.staging_layers, 2);
    assert_eq!(q.cache_slots, (live - fixed_by_hand(2, WS)) / E);
    assert_eq!(q.bound, Bound::Live);
    assert!(gib(q.cache_bytes()) > 40.0, "no 64 GiB cap needed here");
    // others below physical - reserve - ws: cache = ws - fixed
    let w = admit_at(5 * G, 0, &Limits::default()).unwrap();
    println!("ws-bound: {w}");
    assert_eq!(w.bound, Bound::WorkingSet);
    assert_eq!(w.staging_layers, 2);
    assert_eq!(w.cache_slots, (WS - fixed_by_hand(2, WS)) / E);
    assert!(gib(w.cache_bytes()) > 60.0, "no 64 GiB cap");
    assert!(w.wired_limit() <= WS);
}

#[test]
fn busy_others_95_refuses() {
    let e = admit_at(95 * G, 0, &Limits::default()).unwrap_err();
    let msg = e.to_string();
    println!("busy: {msg}");
    assert!(msg.contains("need") && msg.contains("vs budget"), "{msg}");
    assert!(msg.contains("others 95.00 GiB"), "{msg}");
}

/// A small host (a 16 GiB machine, nothing else running) refuses the real
/// model at load: the reserve and the fixed set never fit, so nothing is
/// allocated and nothing swaps.
#[test]
fn small_host_refuses() {
    let h = HostMemory {
        physical: 16 * G,
        wired: 3 * G,
        ..Default::default()
    };
    let e = admit(&model(), &Limits::default(), &h, &Own::default(), 12 * G).unwrap_err();
    assert!(e.to_string().contains("vs budget"), "{e}");
}

#[test]
fn never_over_ws_or_live() {
    let mut n = 0;
    for ws in [WS, 96 * G, 48 * G] {
        for cap in [None, Some(0), Some(4 * G), Some(1000 * G)] {
            let limits = Limits {
                cache_cap: cap,
                ..Limits::default()
            };
            for q in 0..=480u64 {
                let others = q * G / 4;
                let (h, o) = at(others, 3 * G);
                let live = PHYS.saturating_sub(others + RESERVE);
                let one = fixed_by_hand(1, ws);
                match admit(&model(), &limits, &h, &o, ws) {
                    Ok(a) => {
                        n += 1;
                        assert!(a.wired_limit() <= ws && a.wired_limit() <= live, "{a}");
                        assert_eq!(a.fixed, fixed_by_hand(a.staging_layers, ws));
                        assert_eq!(a.staging_layers, layers_by_hand(a.budget(), cap, ws), "{a}");
                        assert!(cap.is_none_or(|c| a.cache_bytes() <= c), "{a}");
                    }
                    Err(_) => assert!(one > live.min(ws), "refused with room"),
                }
            }
        }
    }
    assert!(n > 1000, "{n}");
}

#[test]
fn env_only_lowers() {
    let base = admit_at(50 * G, 0, &Limits::default()).unwrap();
    let big = Limits::from_values(Some("1000"), None).unwrap();
    assert_eq!(admit_at(50 * G, 0, &big).unwrap(), base);
    let four = Limits::from_values(Some("4"), None).unwrap();
    let a = admit_at(50 * G, 0, &four).unwrap();
    assert_eq!((a.cache_slots, a.bound), ((4 * G) / E, Bound::Env));
    let zero = Limits::from_values(Some("0"), None).unwrap();
    assert_eq!(admit_at(50 * G, 0, &zero).unwrap().cache_slots, 0);
    let half = Limits::from_values(Some("0.5"), None).unwrap();
    assert_eq!(half.cache_cap, Some(G / 2));
    // the reserve only grows
    let r8 = Limits::from_values(None, Some("8")).unwrap();
    assert_eq!(r8.reserve, RESERVE);
    let r22 = Limits::from_values(None, Some("22")).unwrap();
    assert_eq!(r22.reserve, 22 * G);
    let a22 = admit_at(50 * G, 0, &r22).unwrap();
    let budget = PHYS - 50 * G - 22 * G;
    let layers = layers_by_hand(budget, None, WS);
    assert_eq!(a22.staging_layers, layers);
    assert_eq!(a22.cache_slots, (budget - fixed_by_hand(layers, WS)) / E);
    assert_eq!(
        Limits::from_values(Some(" "), None).unwrap(),
        Limits::default()
    );
    for v in ["-1", "abc", "nan", "inf", "1e9"] {
        assert!(Limits::from_values(Some(v), None).is_err(), "{v}");
        assert!(Limits::from_values(None, Some(v)).is_err(), "{v}");
    }
}

#[test]
fn own_growth_keeps_the_bank() {
    let l = Limits::default();
    let load = admit_at(50 * G, 2 * G, &l).unwrap();
    let (h, o) = at(50 * G, 12 * G);
    assert_eq!(
        readmit(&load, &model(), &l, &h, &o, WS).unwrap(),
        Turn::Keep
    );
    // volatile bank bytes leave host_used and own together
    let h2 = host(50 * G + 2 * G);
    let o2 = Own {
        active: 2 * G + 30 * G,
        pool: 0,
        volatile: 30 * G,
    };
    assert_eq!(
        readmit(&load, &model(), &l, &h2, &o2, WS).unwrap(),
        Turn::Keep
    );
}

#[test]
fn others_growth_drops_then_recovers() {
    let (l, m) = (Limits::default(), model());
    let load = admit_at(50 * G, 2 * G, &l).unwrap();
    // a little growth inside the keep slack
    let (h, o) = at(51 * G, 2 * G);
    assert_eq!(readmit(&load, &m, &l, &h, &o, WS).unwrap(), Turn::Keep);
    // others +10 GiB: staging only, cache 0
    let (h, o) = at(60 * G, 2 * G);
    let Turn::StagingOnly(s) = readmit(&load, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected staging-only");
    };
    println!("staging-only: {s}");
    assert_eq!((s.cache_slots, s.staging_layers), (0, 1));
    assert_eq!(s.wired_limit(), fixed_by_hand(1, WS));
    // next turn, others back: the original bank comes back
    let (h, o) = at(50 * G, 2 * G);
    let Turn::Rebuild(r) = readmit(&s, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected rebuild");
    };
    assert_eq!(r, load);
    // next turn, others still +10: a smaller bank that fits
    let (h, o) = at(60 * G, 2 * G);
    let Turn::Rebuild(r) = readmit(&s, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected rebuild");
    };
    assert!(r.cache_slots > 0 && r.cache_slots < load.cache_slots);
    assert!(r.wired_limit() <= r.live);
    // staging-only with no room for any slot stays staging-only
    let tight = PHYS - RESERVE - fixed_by_hand(1, WS);
    let (h, o) = at(tight, 2 * G);
    assert_eq!(readmit(&s, &m, &l, &h, &o, WS).unwrap(), Turn::Keep);
}

#[test]
fn readmit_refuses_only_when_fixed_does_not_fit() {
    let (l, m) = (Limits::default(), model());
    let load = admit_at(50 * G, 0, &l).unwrap();
    let (h, o) = at(95 * G, 0);
    let msg = readmit(&load, &m, &l, &h, &o, WS).unwrap_err().to_string();
    assert!(msg.contains("vs budget"), "{msg}");
    let edge = PHYS - RESERVE - fixed_by_hand(1, WS);
    let (h, o) = at(edge, 0);
    assert!(matches!(
        readmit(&load, &m, &l, &h, &o, WS).unwrap(),
        Turn::StagingOnly(_)
    ));
}

#[test]
fn grows_only_by_grow_min() {
    let (l, m) = (Limits::default(), model());
    let load = admit_at(50 * G, 0, &l).unwrap();
    let (h, o) = at(45 * G, 0);
    assert_eq!(readmit(&load, &m, &l, &h, &o, WS).unwrap(), Turn::Keep);
    let (h, o) = at(25 * G, 0);
    let Turn::Rebuild(r) = readmit(&load, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected rebuild");
    };
    assert!(r.cache_bytes() >= load.cache_bytes() + GROW_MIN);
}

/// The second staging layer: free when a capped cache leaves its bytes;
/// paid with cache only while `SECOND_STAGING_MIN_CACHE` is left (one slot
/// either side of the edge); never when it does not fit.
#[test]
fn second_staging_layer_rule() {
    let l = Limits::default();
    let x = SECOND_STAGING_MIN_CACHE;
    // at the edge: exactly X left after two layers
    let at_x = admit_at(others_for_room2(x), 0, &l).unwrap();
    println!("edge: {at_x}");
    assert_eq!((at_x.staging_layers, at_x.cache_slots), (2, x / E));
    let above = admit_at(others_for_room2(x + E), 0, &l).unwrap();
    assert_eq!((above.staging_layers, above.cache_slots), (2, (x + E) / E));
    // one slot short of X: one layer, and its bytes go to the cache
    let below = admit_at(others_for_room2(x - E), 0, &l).unwrap();
    println!("below: {below}");
    assert_eq!(below.staging_layers, 1);
    assert_eq!(
        below.cache_slots,
        (x - E + fixed_by_hand(2, WS) - fixed_by_hand(1, WS)) / E
    );
    // a cap that leaves room for the second layer gets it for free
    let small = others_for_room2(x - E);
    for cap in [0, 4 * G, x - E - 384 * E] {
        let capped = Limits {
            cache_cap: Some(cap),
            ..l
        };
        let c = admit_at(small, 0, &capped).unwrap();
        assert_eq!((c.staging_layers, c.cache_slots), (2, cap / E), "cap {cap}");
        assert_eq!(c.bound, Bound::Env);
    }
    // a cap above what two layers leave (X - 1 slot): one layer, cache = cap
    let capped = Limits {
        cache_cap: Some(x + E),
        ..l
    };
    let c = admit_at(small, 0, &capped).unwrap();
    assert_eq!((c.staging_layers, c.cache_slots), (1, (x + E) / E));
    // two layers do not fit at all: one layer, the cache takes what is left
    let edge = PHYS - RESERVE - fixed_by_hand(1, WS) - G;
    let f = admit_at(edge, 0, &l).unwrap();
    assert_eq!((f.staging_layers, f.cache_slots), (1, G / E));
}

/// Re-admission across the rule's edge: a one-layer bank moves to two
/// layers only through a rebuild (others shrank by the second layer + the
/// grow step); a two-layer bank that no longer fits drops to one layer,
/// cache 0, and comes back as the rule says.
#[test]
fn readmit_across_the_staging_edge() {
    let (l, m) = (Limits::default(), model());
    let x = SECOND_STAGING_MIN_CACHE;
    let one = admit_at(others_for_room2(x - E), 2 * G, &l).unwrap();
    assert_eq!(one.staging_layers, 1);
    // others shrink a little: two layers would now be granted, but the
    // cache would not grow by GROW_MIN, so the bank is kept
    let (h, o) = at(others_for_room2(x + 4 * G), 2 * G);
    assert_eq!(readmit(&one, &m, &l, &h, &o, WS).unwrap(), Turn::Keep);
    // others shrink by the second layer + GROW_MIN: two layers, larger cache
    let shrink = fixed_by_hand(2, WS) - fixed_by_hand(1, WS) + GROW_MIN + E;
    let (h, o) = at(others_for_room2(x - E) - shrink, 2 * G);
    let Turn::Rebuild(two) = readmit(&one, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected rebuild");
    };
    println!("1 -> 2: {two}");
    assert_eq!(two.staging_layers, 2);
    assert!(two.cache_bytes() >= one.cache_bytes() + GROW_MIN);
    // others grow past the keep slack: one layer, cache 0
    let (h, o) = at(others_for_room2(x - E) - shrink + 3 * G + KEEP_SLACK, 2 * G);
    let Turn::StagingOnly(s) = readmit(&two, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected staging-only");
    };
    assert_eq!((s.staging_layers, s.cache_slots), (1, 0));
    assert_eq!(s.wired_limit(), fixed_by_hand(1, WS));
    // and back below X: a one-layer bank
    let (h, o) = at(others_for_room2(x - E), 2 * G);
    let Turn::Rebuild(r) = readmit(&s, &m, &l, &h, &o, WS).unwrap() else {
        panic!("expected rebuild");
    };
    assert_eq!(r, one);
}

/// Host sizes the product meets: a quiet 96 GB Mac (Metal working set ~72
/// GiB) and a 64 GB one (refused: the fixed set does not fit).
#[test]
fn smaller_hosts() {
    let quiet = |physical: u64| HostMemory {
        physical,
        wired: 20 * G,
        ..Default::default()
    };
    let ws96 = 72 * G;
    let a = admit(
        &model(),
        &Limits::default(),
        &quiet(96 * G),
        &Own::default(),
        ws96,
    )
    .unwrap();
    println!("96 GB: {a}");
    let budget = (96 * G - 20 * G - RESERVE).min(ws96);
    assert_eq!(a.staging_layers, layers_by_hand(budget, None, ws96));
    assert!(a.wired_limit() <= budget);
    let e = admit(
        &model(),
        &Limits::default(),
        &quiet(64 * G),
        &Own::default(),
        48 * G,
    );
    assert!(e.unwrap_err().to_string().contains("vs budget"));
}

#[test]
fn this_host_prints() {
    let ws = working_set();
    let own = Own::measure();
    let Some(h) = crate::utils::host::host_memory() else {
        println!("no host reading on this target");
        return;
    };
    println!("this host: ws {:.2} GiB, own {own:?}", gib(ws));
    match admit(&model(), &Limits::default(), &h, &own, ws) {
        Ok(a) => {
            println!("this host admits: {a}");
            assert!(a.wired_limit() <= a.budget());
        }
        Err(e) => println!("this host refuses: {e}"),
    }
}
