//! `Profile` forms against numbers pinned from the code before `Profile`
//! existed (a line-by-line port of that `Terms::new` / `admit` / `readmit`,
//! integer arithmetic): DS V4.1 bytes must not move.

use super::*;

const G: u64 = 1 << 30;
const OWN: u64 = 2 * G;
const PHYS: u64 = 128 * G;

const MODELS: [ModelSize; 2] = [
    ModelSize {
        dense: 9_165_803_968,
        slot: 18_800_640,
        n_experts: 384,
        ctx: 16_384,
    },
    ModelSize {
        dense: 9_165_803_968,
        slot: 18_800_640,
        n_experts: 384,
        ctx: 4_096,
    },
];
const WSS: [u64; 2] = [115_448_725_504, 96 * G];

fn limits(i: usize) -> Limits {
    [
        Limits::default(),
        Limits {
            reserve: RESERVE,
            cache_cap: Some(4 * G),
        },
        Limits {
            reserve: 22 * G,
            cache_cap: None,
        },
    ][i]
}

/// `others` GiB of other processes plus this process's `OWN` bytes.
fn at(others: u64) -> (HostMemory, Own) {
    (
        HostMemory {
            physical: PHYS,
            wired: others * G + OWN,
            ..Default::default()
        },
        Own {
            active: OWN,
            ..Default::default()
        },
    )
}

/// (model, ws): staging_layer, kv, compute, io, driver, fixed(1).
const TERMS: [(usize, usize, [u64; 6]); 4] = [
    (
        0,
        0,
        [
            7_219_445_760,
            138_412_032,
            16_539_714_474,
            939_524_096,
            5_772_436_275,
            41_922_820_253,
        ],
    ),
    (
        0,
        1,
        [
            7_219_445_760,
            138_412_032,
            16_539_714_474,
            939_524_096,
            5_153_960_755,
            41_304_344_733,
        ],
    ),
    (
        1,
        0,
        [
            7_219_445_760,
            44_040_192,
            5_155_550_941,
            436_207_616,
            5_772_436_275,
            29_940_968_400,
        ],
    ),
    (
        1,
        1,
        [
            7_219_445_760,
            44_040_192,
            5_155_550_941,
            436_207_616,
            5_153_960_755,
            29_322_492_880,
        ],
    ),
];

type Pinned = Option<(u64, u64, u64, Bound)>;

/// (model, ws, limits, others GiB): staging layers, fixed, cache slots,
/// bound; `None` = refused.
const ADMITS: [(usize, usize, usize, u64, Pinned); 60] = [
    (
        0,
        0,
        0,
        5,
        Some((2, 49_142_266_013, 3_526, Bound::WorkingSet)),
    ),
    (0, 0, 0, 25, Some((2, 49_142_266_013, 2_583, Bound::Live))),
    (0, 0, 0, 50, Some((2, 49_142_266_013, 1_155, Bound::Live))),
    (0, 0, 0, 80, None),
    (0, 0, 0, 95, None),
    (0, 0, 1, 5, Some((2, 49_142_266_013, 228, Bound::Env))),
    (0, 0, 1, 25, Some((2, 49_142_266_013, 228, Bound::Env))),
    (0, 0, 1, 50, Some((2, 49_142_266_013, 228, Bound::Env))),
    (0, 0, 1, 80, None),
    (0, 0, 1, 95, None),
    (0, 0, 2, 5, Some((2, 49_142_266_013, 3_154, Bound::Live))),
    (0, 0, 2, 25, Some((2, 49_142_266_013, 2_012, Bound::Live))),
    (0, 0, 2, 50, Some((1, 41_922_820_253, 968, Bound::Live))),
    (0, 0, 2, 80, None),
    (0, 0, 2, 95, None),
    (
        0,
        1,
        0,
        5,
        Some((2, 48_523_790_493, 2_901, Bound::WorkingSet)),
    ),
    (0, 1, 0, 25, Some((2, 48_523_790_493, 2_616, Bound::Live))),
    (0, 1, 0, 50, Some((2, 48_523_790_493, 1_188, Bound::Live))),
    (0, 1, 0, 80, None),
    (0, 1, 0, 95, None),
    (0, 1, 1, 5, Some((2, 48_523_790_493, 228, Bound::Env))),
    (0, 1, 1, 25, Some((2, 48_523_790_493, 228, Bound::Env))),
    (0, 1, 1, 50, Some((2, 48_523_790_493, 228, Bound::Env))),
    (0, 1, 1, 80, None),
    (0, 1, 1, 95, None),
    (
        0,
        1,
        2,
        5,
        Some((2, 48_523_790_493, 2_901, Bound::WorkingSet)),
    ),
    (0, 1, 2, 25, Some((2, 48_523_790_493, 2_045, Bound::Live))),
    (0, 1, 2, 50, Some((1, 41_304_344_733, 1_001, Bound::Live))),
    (0, 1, 2, 80, None),
    (0, 1, 2, 95, None),
    (
        1,
        0,
        0,
        5,
        Some((2, 37_160_414_160, 4_164, Bound::WorkingSet)),
    ),
    (1, 0, 0, 25, Some((2, 37_160_414_160, 3_220, Bound::Live))),
    (1, 0, 0, 50, Some((2, 37_160_414_160, 1_792, Bound::Live))),
    (1, 0, 0, 80, Some((1, 29_940_968_400, 463, Bound::Live))),
    (1, 0, 0, 95, None),
    (1, 0, 1, 5, Some((2, 37_160_414_160, 228, Bound::Env))),
    (1, 0, 1, 25, Some((2, 37_160_414_160, 228, Bound::Env))),
    (1, 0, 1, 50, Some((2, 37_160_414_160, 228, Bound::Env))),
    (1, 0, 1, 80, Some((1, 29_940_968_400, 228, Bound::Env))),
    (1, 0, 1, 95, None),
    (1, 0, 2, 5, Some((2, 37_160_414_160, 3_791, Bound::Live))),
    (1, 0, 2, 25, Some((2, 37_160_414_160, 2_649, Bound::Live))),
    (1, 0, 2, 50, Some((2, 37_160_414_160, 1_221, Bound::Live))),
    (1, 0, 2, 80, None),
    (1, 0, 2, 95, None),
    (
        1,
        1,
        0,
        5,
        Some((2, 36_541_938_640, 3_539, Bound::WorkingSet)),
    ),
    (1, 1, 0, 25, Some((2, 36_541_938_640, 3_253, Bound::Live))),
    (1, 1, 0, 50, Some((2, 36_541_938_640, 1_825, Bound::Live))),
    (1, 1, 0, 80, Some((1, 29_322_492_880, 496, Bound::Live))),
    (1, 1, 0, 95, None),
    (1, 1, 1, 5, Some((2, 36_541_938_640, 228, Bound::Env))),
    (1, 1, 1, 25, Some((2, 36_541_938_640, 228, Bound::Env))),
    (1, 1, 1, 50, Some((2, 36_541_938_640, 228, Bound::Env))),
    (1, 1, 1, 80, Some((1, 29_322_492_880, 228, Bound::Env))),
    (1, 1, 1, 95, None),
    (
        1,
        1,
        2,
        5,
        Some((2, 36_541_938_640, 3_539, Bound::WorkingSet)),
    ),
    (1, 1, 2, 25, Some((2, 36_541_938_640, 2_682, Bound::Live))),
    (1, 1, 2, 50, Some((2, 36_541_938_640, 1_254, Bound::Live))),
    (1, 1, 2, 80, None),
    (1, 1, 2, 95, None),
];

#[test]
fn dsv41_profile_is_the_pinned_constants() {
    assert_eq!(
        DSV41,
        Profile {
            cache_env: "MLX_DSV41_EXPERT_CACHE_GB",
            reserve_env: "MLX_DSV41_ADMIT_RESERVE_GB",
            compute_base: 1_134_024_803,
            compute_per_row: 772_038,
            compute_fixed: 0,
            compute_per_ctx: 0,
            kv_base: 10_485_760,
            kv_per_ctx: 6_400,
            io_base: 268_435_456,
            io_per_row: 40_960,
            io_fixed: 0,
            second_staging_min_cache: 17 * G,
        }
    );
}

#[test]
fn dsv41_terms_are_the_pinned_bytes() {
    for (mi, wi, [staging, kv, compute, io, driver, one]) in TERMS {
        let (m, ws) = (&MODELS[mi], WSS[wi]);
        let t = Terms::new(m, ws);
        assert_eq!(t, Terms::with(&DSV41, m, ws, m.ctx), "model {mi} ws {wi}");
        assert_eq!(t.dense, m.dense);
        assert_eq!(
            [t.staging_layer, t.kv, t.compute, t.io, t.driver, t.pool],
            [staging, kv, compute, io, driver, POOL],
            "model {mi} ws {wi}"
        );
        assert_eq!(t.fixed(1), one, "model {mi} ws {wi}");
    }
}

#[test]
fn dsv41_admission_is_the_pinned_grid() {
    for (mi, wi, li, others, want) in ADMITS {
        let (m, ws, l) = (&MODELS[mi], WSS[wi], limits(li));
        let (h, o) = at(others);
        let old = admit(m, &l, &h, &o, ws);
        let new = admit_with(&DSV41, m, m.ctx, &l, &h, &o, ws);
        let case = format!("model {mi} ws {wi} limits {li} others {others}");
        match (old, new, want) {
            (Ok(a), Ok(b), Some((layers, fixed, slots, bound))) => {
                assert_eq!(a, b, "{case}");
                assert_eq!(
                    (a.staging_layers, a.fixed, a.cache_slots, a.bound),
                    (layers, fixed, slots, bound),
                    "{case}"
                );
                assert_eq!(a.others, others * G, "{case}");
                assert_eq!(a.own, OWN, "{case}");
            }
            (Err(a), Err(b), None) => assert_eq!(a.reason, b.reason, "{case}"),
            (a, b, w) => panic!("{case}: old {a:?}, new {b:?}, pinned {w:?}"),
        }
    }
}

#[test]
fn dsv41_readmission_is_pinned() {
    let (m, ws, l) = (&MODELS[0], WSS[0], Limits::default());
    let (h, o) = at(50);
    let load = admit(m, &l, &h, &o, ws).unwrap();
    assert_eq!((load.staging_layers, load.cache_slots), (2, 1_155));
    for others in [45, 51, 60, 25, 95] {
        let (h, o) = at(others);
        let old = readmit(&load, m, &l, &h, &o, ws);
        let new = readmit_with(&DSV41, &load, m, m.ctx, &l, &h, &o, ws);
        match (others, old, new) {
            (45 | 51, Ok(Turn::Keep), Ok(Turn::Keep)) => {}
            (60, Ok(Turn::StagingOnly(a)), Ok(Turn::StagingOnly(b))) => {
                assert_eq!(a, b);
                assert_eq!(
                    (a.staging_layers, a.cache_slots, a.fixed),
                    (1, 0, 41_922_820_253)
                );
            }
            (25, Ok(Turn::Rebuild(a)), Ok(Turn::Rebuild(b))) => {
                assert_eq!(a, b);
                assert_eq!((a.staging_layers, a.cache_slots), (2, 2_583));
            }
            (95, Err(a), Err(b)) => assert_eq!(a.reason, b.reason),
            (n, a, b) => panic!("others {n}: old {a:?}, new {b:?}"),
        }
    }
}

#[test]
fn limits_name_the_profile_env() {
    for (c, r) in [
        (None, None),
        (Some("4"), None),
        (Some(" "), Some("22")),
        (Some("0.5"), Some("8")),
    ] {
        assert_eq!(
            Limits::from_values(c, r).unwrap(),
            Limits::from_values_with(&DSV41, c, r).unwrap()
        );
    }
    let ds = Limits::from_values(Some("abc"), None).unwrap_err().reason;
    assert!(ds.contains("MLX_DSV41_EXPERT_CACHE_GB"), "{ds}");
    let other = Profile {
        cache_env: "MLX_TEST_CACHE_GB",
        reserve_env: "MLX_TEST_RESERVE_GB",
        ..DSV41
    };
    let e = Limits::from_values_with(&other, Some("-1"), None)
        .unwrap_err()
        .reason;
    assert!(e.contains("MLX_TEST_CACHE_GB"), "{e}");
    let e = Limits::from_values_with(&other, None, Some("nan"))
        .unwrap_err()
        .reason;
    assert!(e.contains("MLX_TEST_RESERVE_GB"), "{e}");
}

/// Prefill rows move compute and io only; the context moves kv and the
/// per-context compute term.
#[test]
fn prefill_rows_and_context_terms() {
    let p = Profile {
        compute_fixed: 1_000,
        compute_per_ctx: 3,
        io_fixed: 70,
        ..DSV41
    };
    let m = ModelSize {
        ctx: 262_144,
        ..MODELS[0]
    };
    let x12 = |b: u64| b + b / 5;
    for rows in [256u64, 2_048, 8_192] {
        let t = Terms::with(&p, &m, WSS[0], rows);
        assert_eq!(t.kv, x12(p.kv_base + p.kv_per_ctx * m.ctx));
        assert_eq!(
            t.compute,
            x12(p.compute_base + p.compute_per_row * rows + 1_000 + 3 * m.ctx)
        );
        assert_eq!(t.io, p.io_base + p.io_per_row * rows + 70);
        assert_eq!(t.staging_layer, m.n_experts * m.slot);
    }
    // DS at 256K charges compute and io at every context row: refused on
    // the 128 GiB host; the same context at 2048-row chunks fits
    let (h, o) = at(20);
    assert!(admit(&m, &Limits::default(), &h, &o, WSS[1]).is_err());
    let a = admit_with(&DSV41, &m, 2_048, &Limits::default(), &h, &o, WSS[1]).unwrap();
    assert!(a.cache_slots > 0, "{a}");
}
