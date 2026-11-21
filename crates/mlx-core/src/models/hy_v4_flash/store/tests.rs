use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::models::hy_v4_flash::bank::{class_types, geometry};
use crate::models::hy_v4_flash::config::MoeConfig;
use crate::models::hy_v4_flash::scan::{ExpertClass, ExpertLayer, Located};
use crate::moe_stream::budget::{Bound, Terms};
use crate::utils::gguf::GgufTensorInfo;

const G: u64 = 1 << 30;

fn fixture(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn u(v: &Value) -> u64 {
    v.as_u64().unwrap_or_else(|| panic!("not a u64: {v}"))
}

/// `admission-addons.json` host: `others` bytes in use, nothing of this model.
fn host(inputs: &Value, others: u64) -> (HostMemory, u64) {
    (
        HostMemory {
            physical: u(&inputs["physical"]),
            wired: others,
            ..Default::default()
        },
        u(&inputs["ws"]),
    )
}

fn base(inputs: &Value, variant: &Value) -> ModelSize {
    ModelSize {
        dense: u(&variant["dense"]),
        slot: u(&inputs["slot"]),
        n_experts: u(&inputs["n_experts"]),
        ctx: 0,
    }
}

#[test]
fn profile_is_the_fixture_profile() {
    let f = fixture("admission-addons.json");
    let pf = &f["profile"];
    let p = profile(u(&pf["kv_per_ctx_mtp_on"]));
    assert_eq!(p.kv_base, u(&pf["KV_BASE"]) + u(&pf["kv_fixed_ladder"]));
    assert_eq!(p.compute_base, u(&pf["COMPUTE_BASE"]));
    assert_eq!(p.compute_per_row, u(&pf["COMPUTE_PER_ROW"]));
    assert_eq!(p.compute_fixed, u(&pf["compute_fixed"]));
    assert_eq!(p.compute_per_ctx, u(&pf["compute_per_ctx"]));
    assert_eq!(p.io_base, u(&pf["IO_BASE"]));
    assert_eq!(p.io_per_row, u(&pf["IO_PER_ROW"]));
    assert_eq!(p.io_fixed, u(&pf["io_fixed"]));
    assert_eq!(
        p.second_staging_min_cache,
        u(&pf["SECOND_STAGING_MIN_CACHE"])
    );
    assert_eq!((p.cache_env, p.reserve_env), (CACHE_ENV, RESERVE_ENV));
}

/// Every (ctx, chunk, others) row of both MTP variants: terms, refusal,
/// staging layers, fixed, cache slots, wired limit and bound.
#[test]
fn admission_matches_the_fixture_rows() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let mut n = 0;
    for name in ["mtp_on", "mtp_off"] {
        let v = &f["variants"][name];
        let p = profile(u(&v["kv_per_ctx"]));
        for r in v["rows"].as_array().unwrap() {
            let m = ModelSize {
                ctx: u(&r["ctx"]),
                ..base(inputs, v)
            };
            let rows = u(&r["chunk_rows"]);
            let others = u(&r["others_gib"]) * G;
            let (h, ws) = host(inputs, others);
            let case = format!("{name} ctx {} chunk {rows} others {others}", m.ctx);
            let t = Terms::with(&p, &m, ws, rows);
            let want = &r["terms"];
            assert_eq!(
                [
                    t.dense,
                    t.staging_layer,
                    t.kv,
                    t.compute,
                    t.io,
                    t.driver,
                    t.pool
                ],
                [
                    "dense",
                    "staging_layer",
                    "kv",
                    "compute",
                    "io",
                    "driver",
                    "pool"
                ]
                .map(|k| u(&want[k])),
                "{case}"
            );
            let got = admit_with(&p, &m, rows, &Limits::default(), &h, &Own::default(), ws);
            match (got, r["refused"].as_bool().unwrap()) {
                (Err(e), true) => assert!(e.reason.contains("vs budget"), "{case}: {e}"),
                (Ok(a), false) => {
                    assert_eq!(a.live, u(&r["live"]), "{case}");
                    assert_eq!(a.budget(), u(&r["budget"]), "{case}");
                    assert_eq!(a.staging_layers, u(&r["staging_layers"]), "{case}");
                    assert_eq!(a.fixed, u(&r["fixed"]), "{case}");
                    assert_eq!(a.cache_slots, u(&r["cache_slots"]), "{case}");
                    assert_eq!(a.cache_bytes(), u(&r["cache_bytes"]), "{case}");
                    assert_eq!(a.wired_limit(), u(&r["wired_limit"]), "{case}");
                    let bound = match r["bound"].as_str().unwrap() {
                        "Live" => Bound::Live,
                        "WorkingSet" => Bound::WorkingSet,
                        b => panic!("{case}: bound {b}"),
                    };
                    assert_eq!(a.bound, bound, "{case}");
                }
                (a, refused) => panic!("{case}: got {a:?}, fixture refused {refused}"),
            }
            n += 1;
        }
    }
    assert_eq!(n, 2 * 5 * 3 * 3);
}

/// The context ladder and chunk rule for every `ladder_others_gib` host.
#[test]
fn pick_matches_the_fixture_ladder() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let max_ctx = u(&inputs["max_context"]) as usize;
    assert_eq!(max_ctx, crate::models::hy_v4_flash::DEFAULT_CONTEXT_CAP);
    assert_eq!(u(&inputs["min_context"]) as usize, MIN_CONTEXT);
    assert_eq!(
        inputs["chunk_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| u(c) as usize)
            .collect::<Vec<_>>(),
        CHUNKS
    );
    for name in ["mtp_on", "mtp_off"] {
        let v = &f["variants"][name];
        let p = profile(u(&v["kv_per_ctx"]));
        let b = base(inputs, v);
        for (others, want) in v["pick"].as_object().unwrap() {
            let others = others.parse::<u64>().unwrap() * G;
            let (h, ws) = host(inputs, others);
            let case = format!("{name} others {others}");
            let got = pick(&p, &b, max_ctx, &Limits::default(), &h, &Own::default(), ws);
            if !want["admitted"].as_bool().unwrap() {
                let e = got.unwrap_err().reason;
                assert!(e.contains("refused at context 16384"), "{case}: {e}");
                continue;
            }
            let got = got.unwrap();
            assert_eq!(got.ctx, u(&want["ctx"]) as usize, "{case}");
            assert_eq!(got.chunk, u(&want["chunk_rows"]) as usize, "{case}");
            let a = got.admission;
            assert_eq!(a.cache_bytes(), u(&want["cache_bytes"]), "{case}");
            assert_eq!(a.staging_layers, u(&want["staging_layers"]), "{case}");
            assert_eq!(a.wired_limit(), u(&want["wired_limit"]), "{case}");
            assert_eq!(
                a.terms,
                Terms::with(
                    &p,
                    &ModelSize {
                        ctx: got.ctx as u64,
                        ..b
                    },
                    ws,
                    got.chunk as u64
                )
            );
        }
    }
}

/// A context below every chunk charges its own length; a refused context
/// below the floor errs at once.
#[test]
fn pick_below_the_floor() {
    let p = profile(15_812);
    let b = ModelSize {
        dense: 11_352_192_436,
        slot: 4_663_296,
        n_experts: 1024,
        ctx: 0,
    };
    let h = HostMemory {
        physical: 128 * G,
        wired: 20 * G,
        ..Default::default()
    };
    let ws = 96_000_000_000;
    let got = pick(&p, &b, 4096, &Limits::default(), &h, &Own::default(), ws).unwrap();
    assert_eq!((got.ctx, got.chunk), (4096, 4096));
    let m = ModelSize { ctx: 4096, ..b };
    assert_eq!(got.admission.terms, Terms::with(&p, &m, ws, 4096));
    let busy = HostMemory {
        wired: 100 * G,
        ..h
    };
    let e = pick(&p, &b, 8192, &Limits::default(), &busy, &Own::default(), ws)
        .unwrap_err()
        .reason;
    assert!(e.contains("refused at context 8192"), "{e}");
}

/// The second turn of a 256K load bound by live memory: the sessions' KV,
/// allocated after the load's reading, is this model's and keeps the bank;
/// read as others' it would drop the bank to staging only.
#[test]
fn readmission_counts_the_session_kv_as_own() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let others = 50 * G;
    let (h0, ws) = host(inputs, others);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let load = pick(
        &p,
        &b,
        max_ctx,
        &Limits::default(),
        &h0,
        &Own::default(),
        ws,
    )
    .unwrap();
    assert_eq!(load.ctx, max_ctx);
    assert_eq!(load.admission.bound, Bound::Live);
    let m = ModelSize {
        ctx: load.ctx as u64,
        ..b
    };
    let kv = p.kv_per_ctx * load.ctx as u64;
    assert!(kv > crate::moe_stream::budget::KEEP_SLACK);
    let h1 = HostMemory {
        wired: others + b.dense + kv,
        ..h0
    };
    let turn = |own: Own| {
        readmit_with(
            &p,
            &load.admission,
            &m,
            load.chunk as u64,
            &Limits::default(),
            &h1,
            &own,
            ws,
        )
        .unwrap()
    };
    assert_eq!(turn(own_of(b.dense, kv, 0, true, 0, 0)), Turn::Keep);
    assert!(matches!(
        turn(own_of(b.dense, 0, 0, true, 0, 0)),
        Turn::StagingOnly(_)
    ));
}

/// A load bound by live memory at a turn: the MLX free pool counts as own
/// up to the admission's pool term, so it keeps the bank; read as others'
/// with more than the slack in use elsewhere it drops it. Pool past the
/// term and any other footprint of the process (another loaded model,
/// scratch) are others'.
#[test]
fn readmission_counts_the_free_pool_up_to_its_term() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let others = 50 * G;
    let (h0, ws) = host(inputs, others);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let load = pick(
        &p,
        &b,
        max_ctx,
        &Limits::default(),
        &h0,
        &Own::default(),
        ws,
    )
    .unwrap();
    assert_eq!(load.admission.bound, Bound::Live);
    let m = ModelSize {
        ctx: load.ctx as u64,
        ..b
    };
    let a = load.admission;
    let term = a.terms.pool;
    assert_eq!(term, crate::moe_stream::budget::KEEP_SLACK);
    let resident = a.cache_bytes();
    let counted = b.dense + resident;
    let turn = |in_use: u64, own: Own| {
        let h = HostMemory {
            wired: others + counted + in_use,
            ..h0
        };
        readmit_with(
            &p,
            &a,
            &m,
            load.chunk as u64,
            &Limits::default(),
            &h,
            &own,
            ws,
        )
        .unwrap()
    };
    let elsewhere = G;
    assert_eq!(
        turn(
            term + elsewhere,
            own_of(b.dense, 0, resident, true, term, term)
        ),
        Turn::Keep
    );
    assert!(matches!(
        turn(
            term + elsewhere,
            own_of(b.dense, 0, resident, true, term, 0)
        ),
        Turn::StagingOnly(_)
    ));
    let over = 3 * G;
    assert!(matches!(
        turn(
            term + over,
            own_of(b.dense, 0, resident, true, term + over, term)
        ),
        Turn::StagingOnly(_)
    ));
    let model = 3 * G;
    assert!(matches!(
        turn(model, own_of(b.dense, 0, resident, true, 0, term)),
        Turn::StagingOnly(_)
    ));
    assert_eq!(
        own_of(b.dense, 0, resident, true, term + over, term).bytes(),
        counted + term
    );
    assert_eq!(
        own_of(b.dense, 0, resident, false, 0, term).volatile,
        resident
    );
}

/// A turn start whose host still counts the bank purgeable after Metal made
/// it non-volatile (measured readings, GB): `host_used` leaves purgeable
/// pages out (from the anonymous ones), so others' pages vanished (bank
/// wired) or the bank did (bank anonymous) and the bank grew past the host.
/// Settled for the bank pages still counted, others read as at a normal
/// turn start; other processes' volatile pages stay reclaimable.
#[test]
fn readmission_settles_the_bank_pages_still_counted_purgeable() {
    let gb = |x: f64| (x * 1e9) as u64;
    let hm = |wired, anonymous, purgeable| HostMemory {
        physical: gb(137.44),
        free: 0,
        file: 0,
        anonymous: gb(anonymous),
        purgeable: gb(purgeable),
        wired: gb(wired),
        compressor: gb(5.17),
    };
    let resident = gb(54.94);
    let held = Residency {
        resident,
        volatile: 0,
    };
    let listed = Residency {
        resident,
        volatile: resident,
    };
    let o = own_of(gb(11.35), gb(3.83), resident, true, 0, 0);
    let others = |h: HostMemory, bank: Residency| {
        settled(h, bank, true)
            .non_reclaimable()
            .saturating_sub(o.bytes()) as f64
            / 1e9
    };
    let normal = others(hm(79.75, 20.22, 0.47), held);
    for h in [hm(79.66, 20.15, 55.47), hm(8.94, 90.61, 55.47)] {
        assert!(others(h, held) < normal - 15.0, "{}", others(h, held));
        assert!(
            (others(h, listed) - normal).abs() < 1.0,
            "{}",
            others(h, listed)
        );
    }
    let foreign = hm(79.75, 70.22, 50.47);
    assert!((others(foreign, held) - normal).abs() < 0.01);
    let h = hm(79.66, 20.15, 55.47);
    assert_eq!(settled(h, listed, false), h);

    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let others = 50 * G;
    let (h0, ws) = host(inputs, others);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let load = pick(
        &p,
        &b,
        max_ctx,
        &Limits::default(),
        &h0,
        &Own::default(),
        ws,
    )
    .unwrap();
    let a = load.admission;
    let m = ModelSize {
        ctx: load.ctx as u64,
        ..b
    };
    let resident = a.cache_bytes();
    assert!(resident > crate::moe_stream::budget::GROW_MIN);
    let lagging = HostMemory {
        wired: others - 20 * G + b.dense,
        anonymous: 20 * G + resident,
        purgeable: resident,
        ..h0
    };
    let o = own_of(b.dense, 0, resident, true, 0, 0);
    let turn = |volatile: u64| {
        let h = settled(lagging, Residency { resident, volatile }, true);
        readmit_with(
            &p,
            &a,
            &m,
            load.chunk as u64,
            &Limits::default(),
            &h,
            &o,
            ws,
        )
        .unwrap()
    };
    assert!(matches!(turn(0), Turn::Rebuild(_)));
    assert_eq!(turn(resident), Turn::Keep);
    assert_eq!(wary_growth(turn(0), || Ok(turn(resident))), Turn::Keep);
    let roomy = HostMemory {
        wired: others - 30 * G + b.dense + resident,
        anonymous: 20 * G,
        purgeable: G,
        ..h0
    };
    let at =
        |h: &HostMemory| readmit_with(&p, &a, &m, load.chunk as u64, &Limits::default(), h, &o, ws);
    let all = Residency {
        resident,
        volatile: resident,
    };
    let Turn::Rebuild(wary) = at(&settled(roomy, all, true)).unwrap() else {
        panic!("no room on the wary reading");
    };
    assert!(matches!(at(&roomy).unwrap(), Turn::Rebuild(_)));
    assert_eq!(
        wary_growth(at(&roomy).unwrap(), || at(&settled(roomy, all, true))),
        Turn::Rebuild(wary)
    );
    assert_eq!(
        wary_growth(Turn::Keep, || panic!("no wary reading needed")),
        Turn::Keep
    );
}

/// A bank the host has no room for shrinks to what the host admits now
/// (staging only when that admits no cache), at a turn start and inside a
/// turn; growth that would empty a bank holding cache slots waits for a
/// turn start.
#[test]
fn next_bank_shrinks_to_what_the_host_admits() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let (h0, ws) = host(inputs, 50 * G);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let a = pick(
        &p,
        &b,
        max_ctx,
        &Limits::default(),
        &h0,
        &Own::default(),
        ws,
    )
    .unwrap()
    .admission;
    assert!(a.cache_slots > 0);
    let smaller = Admission {
        cache_slots: a.cache_slots / 2,
        ..a
    };
    let staging = Admission {
        staging_layers: 1,
        fixed: a.terms.fixed(1),
        cache_slots: 0,
        ..a
    };
    let never = || -> Result<Admission> { panic!("no reading needed") };
    for in_turn in [false, true] {
        let next = |turn, cur: &Admission, lost| next_bank(turn, cur, lost, in_turn, never);
        assert_eq!(next(Turn::Keep, &a, false), None);
        assert_eq!(next(Turn::Keep, &a, true), Some(a));
        assert_eq!(next(Turn::Rebuild(a), &staging, false), Some(a));
        assert_eq!(next(Turn::Rebuild(a), &smaller, true), Some(a));
        assert_eq!(
            next_bank(Turn::Rebuild(a), &smaller, false, in_turn, never),
            (!in_turn).then_some(a)
        );
        for lost in [false, true] {
            assert_eq!(
                next_bank(Turn::StagingOnly(staging), &a, lost, in_turn, || Ok(
                    smaller
                )),
                Some(smaller)
            );
            assert_eq!(
                next_bank(Turn::StagingOnly(staging), &a, lost, in_turn, || Ok(
                    staging
                )),
                Some(staging)
            );
        }
    }
}

/// The wary reading counts others' volatile pages as used, so it can refuse
/// even the staging layer the host reading admitted: the turn (and every
/// lane's step inside one) then runs staging only instead of failing.
#[test]
fn next_bank_runs_staging_only_when_the_wary_reading_refuses() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let (h0, ws) = host(inputs, 50 * G);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let load = pick(
        &p,
        &b,
        max_ctx,
        &Limits::default(),
        &h0,
        &Own::default(),
        ws,
    )
    .unwrap();
    let a = load.admission;
    let m = ModelSize {
        ctx: load.ctx as u64,
        ..b
    };
    let resident = a.cache_bytes();
    let one = a.terms.fixed(1);
    // others' volatile pages the size of the bank; the host admits one
    // staging layer, the wary reading (the bank resident, those pages used)
    // not even that
    let o = own_of(b.dense, 0, resident, true, 0, 0);
    let room = one + G;
    let h = HostMemory {
        wired: h0.physical - Limits::default().reserve - room + b.dense,
        anonymous: 2 * resident,
        purgeable: resident,
        ..h0
    };
    let all = Residency {
        resident,
        volatile: resident,
    };
    let at = |h: &HostMemory| admit_with(&p, &m, load.chunk as u64, &Limits::default(), h, &o, ws);
    let host_turn = readmit_with(
        &p,
        &a,
        &m,
        load.chunk as u64,
        &Limits::default(),
        &h,
        &o,
        ws,
    )
    .unwrap();
    let Turn::StagingOnly(s) = host_turn else {
        panic!("{host_turn:?}");
    };
    assert_eq!((s.cache_slots, s.staging_layers), (0, 1));
    let e = at(&settled(h, all, true)).unwrap_err().reason;
    assert!(e.contains("vs budget"), "{e}");
    for in_turn in [false, true] {
        assert_eq!(
            next_bank(host_turn, &a, false, in_turn, || at(&settled(h, all, true))),
            Some(s)
        );
    }
}

/// A turn start's flip to non-volatile can leave the bank in the host's
/// purgeable count (measured, GB: 55.47 right after the flip, the bank's
/// 54.94 plus the usual 0.5; 0.3-0.5 at every in-turn reading). The count
/// before the flip tells the bank's share from others' volatile pages.
#[test]
fn flip_lag_reads_the_bank_left_in_the_count() {
    let gb = |x: f64| (x * 1e9) as u64;
    let bank = gb(54.94);
    let before = gb(0.47) + bank;
    assert_eq!(flip_lag(before, gb(55.47), bank), bank);
    assert_eq!(flip_lag(before, gb(0.47), bank), 0);
    assert_eq!(flip_lag(before, gb(0.40), bank), 0);
    assert_eq!(flip_lag(before, gb(0.60), bank), gb(0.60) - gb(0.47));
    assert_eq!(flip_lag(before, gb(30.47), bank), gb(30.47) - gb(0.47));
    assert_eq!(flip_lag(gb(10.0), gb(55.47), bank), 0);
    assert_eq!(flip_lag(gb(0.47), gb(0.47), 0), 0);
}

/// The shrink at a turn start reads the host with the bank the flip left in
/// the purgeable count taken out: a stale count no longer hides others that
/// grew past the slack, and others' own volatile pages stay reclaimable, so
/// a host with room keeps a warm bank (the wary reading would empty it).
#[test]
fn turn_start_shrink_reads_the_flip_lag() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let others = 50 * G;
    let (h0, ws) = host(inputs, others);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let load = pick(
        &p,
        &b,
        max_ctx,
        &Limits::default(),
        &h0,
        &Own::default(),
        ws,
    )
    .unwrap();
    let a = load.admission;
    assert_eq!(a.bound, Bound::Live);
    let m = ModelSize {
        ctx: load.ctx as u64,
        ..b
    };
    let resident = a.cache_bytes();
    let slack = crate::moe_stream::budget::KEEP_SLACK;
    assert!(resident > 2 * slack);
    let o = own_of(b.dense, 0, resident, true, 0, 0);
    let turn = |h: &HostMemory, volatile: u64| {
        let h = settled(*h, Residency { resident, volatile }, true);
        readmit_with(
            &p,
            &a,
            &m,
            load.chunk as u64,
            &Limits::default(),
            &h,
            &o,
            ws,
        )
        .unwrap()
    };
    // bank anonymous, 20 GiB of others' anonymous pages, the count still
    // holding the bank after the flip
    let stale = |grown: u64| HostMemory {
        wired: others + grown - 20 * G + b.dense,
        anonymous: 20 * G + resident,
        purgeable: resident,
        ..h0
    };
    let before = resident;
    for grown in [0, slack + 2 * G] {
        let h = stale(grown);
        let lag = flip_lag(before, h.purgeable, resident);
        assert_eq!(lag, resident);
        assert!(!matches!(turn(&h, 0), Turn::StagingOnly(_)));
        let read = turn(&h, lag);
        assert_eq!(matches!(read, Turn::StagingOnly(_)), grown > 0, "{read:?}");
    }
    // a current count holding others' volatile pages the size of the slack
    // and more: no lag, the bank stays
    let foreign = 3 * slack;
    let h = HostMemory {
        wired: others - 20 * G + b.dense,
        anonymous: 20 * G + foreign + resident,
        purgeable: foreign,
        ..h0
    };
    let lag = flip_lag(foreign + resident, h.purgeable, resident);
    assert_eq!(lag, 0);
    assert_eq!(turn(&h, lag), Turn::Keep);
    assert!(matches!(turn(&h, resident), Turn::StagingOnly(_)));
}

#[test]
fn staging_charge_is_two_full_layers() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let m = ModelSize {
        ctx: 262_144,
        ..base(inputs, v)
    };
    let t = Terms::with(&profile(u(&v["kv_per_ctx"])), &m, u(&inputs["ws"]), 8192);
    assert_eq!(t.staging_layer * 2, 9_550_430_208);
    let adm = fixture("admission.json");
    let bank = 2 * 512 * (u(&adm["inputs"]["slot_a"]) + u(&adm["inputs"]["slot_b"]));
    assert_eq!(bank, 8_965_324_800);
    assert!(t.staging_layer * 2 >= bank);
}

/// MLX's qmv batch limit per generation, size letter and shape class.
#[test]
fn qmv_batch_limit_follows_mlx() {
    let cases = [
        ("applegpu_g17s", [33, 25, 13]),
        ("applegpu_g17g", [33, 25, 13]),
        ("applegpu_g17d", [32, 18, 12]),
        ("applegpu_g16s", [13, 15, 13]),
        ("applegpu_g15s", [13, 15, 13]),
        ("applegpu_g15d", [32, 18, 12]),
        ("applegpu_g14s", [14, 10, 6]),
        ("applegpu_g13s", [14, 10, 7]),
        ("applegpu_g13g", [14, 10, 6]),
        ("applegpu_g13d", [32, 18, 12]),
        ("applegpu_g12s", [18, 12, 10]),
        ("", [18, 12, 10]),
    ];
    for (arch, want) in cases {
        let got =
            [(2048, 2048), (4096, 1024), (3072, 8192)].map(|(k, n)| qmv_batch_limit(arch, k, n));
        assert_eq!(got, want, "{arch}");
    }
    assert_eq!(qmv_batch_limit("applegpu_g17s", 2049, 16), 25);
    assert_eq!(qmv_batch_limit("applegpu_g17s", 16, 4097), 13);
}

/// Hy4's shape classes (small k_b / v_b, medium kv_a, large gate): 12 rows
/// on M3-M5 Max, 11 on an Ultra, 5-6 on M1/M2; the env sets every limit.
#[test]
fn row_cap_is_one_below_the_smallest_limit() {
    let shapes = [
        (192, 512),
        (512, 256),
        (3072, 576),
        (3072, 8192),
        (1536, 15_360),
    ];
    for (arch, want) in [
        ("applegpu_g17s", 12),
        ("applegpu_g16s", 12),
        ("applegpu_g15s", 12),
        ("applegpu_g17d", 11),
        ("applegpu_g15d", 11),
        ("applegpu_g14s", 5),
        ("applegpu_g13s", 6),
    ] {
        assert_eq!(row_cap(arch, None, &shapes).unwrap(), want, "{arch}");
    }
    assert_eq!(row_cap("applegpu_g17s", Some(4), &shapes).unwrap(), 3);
    assert_eq!(row_cap("applegpu_g17s", Some(1), &shapes).unwrap(), 0);
    assert!(row_cap("applegpu_g17s", None, &[]).is_err());
}

#[test]
fn splitk_env_reads_like_atoi() {
    for (raw, want) in [
        (None, None),
        (Some("8"), Some(8)),
        (Some("  12x"), Some(12)),
        (Some("+3"), Some(3)),
        (Some("-3"), None),
        (Some("0"), None),
        (Some("abc"), None),
        (Some(""), None),
        (Some("99999999999"), None),
    ] {
        assert_eq!(splitk_min_m(raw), want, "{raw:?}");
    }
}

/// Each lane past the first adds its floor to the KV base, nothing else.
#[test]
fn lanes_charge_their_floor() {
    let cfg = crate::models::hy_v4_flash::tests_model::tiny_cfg();
    let floor = lane_floor(&cfg, false);
    assert_eq!(floor, 2 * 11 * 24 * 2 + 2 * 6 * 16 * 4);
    assert_eq!(lane_floor(&cfg, true) - floor, SEEDS as u64 * 32 * 4);
    let p = profile(15_812);
    assert_eq!(lane_profile(&p, 1, floor), p);
    assert_eq!(lane_profile(&p, 0, floor), p);
    let four = lane_profile(&p, 4, floor);
    assert_eq!(four.kv_base, p.kv_base + 3 * floor);
    assert_eq!(
        Profile {
            kv_base: p.kv_base,
            ..four
        },
        p
    );
}

/// A lane past the first with the MTP layer also holds its draft's pooled
/// keys for the whole context (f16, one row per group).
/// Mutation: `lane_bytes` returns `lane_floor` → red.
#[test]
fn lane_bytes_price_the_draft_pooled_keys() {
    let cfg = crate::models::hy_v4_flash::tests_model::tiny_cfg();
    for ctx in [0, 64, 262_144] {
        assert_eq!(lane_bytes(&cfg, false, ctx), lane_floor(&cfg, false));
        assert_eq!(
            lane_bytes(&cfg, true, ctx),
            lane_floor(&cfg, true) + (ctx / 4) as u64 * 16 * 2,
            "ctx {ctx}"
        );
    }
}

/// The most lanes that admit win; one lane's refusal is the error.
#[test]
fn plan_lanes_steps_down_to_what_fits() {
    let fits = |max: usize| {
        move |n: usize| {
            if n <= max {
                Ok(n * 10)
            } else {
                Err(Error::from_reason(format!("{n} lanes")))
            }
        }
    };
    assert_eq!(plan_lanes(4, fits(4)).unwrap(), (4, 40));
    assert_eq!(plan_lanes(4, fits(2)).unwrap(), (2, 20));
    assert_eq!(plan_lanes(0, fits(4)).unwrap(), (1, 10));
    assert_eq!(plan_lanes(3, fits(0)).unwrap_err().reason, "1 lanes");
}

/// Lanes are fitted at the one-lane context and chunk: for every lane
/// floor the load keeps the one-lane pick's window and chunk and takes the
/// most lanes that admit there; with the ladder run per lane count some
/// floors would shrink the window or the chunk.
#[test]
fn lanes_keep_the_one_lane_window_and_chunk() {
    let f = fixture("admission-addons.json");
    let inputs = &f["inputs"];
    let v = &f["variants"]["mtp_on"];
    let p = profile(u(&v["kv_per_ctx"]));
    let b = base(inputs, v);
    let max_ctx = u(&inputs["max_context"]) as usize;
    let (h, ws) = host(inputs, 50 * G);
    let (l, own) = (Limits::default(), Own::default());
    let one = pick(&p, &b, max_ctx, &l, &h, &own, ws).unwrap();
    let at = ModelSize {
        ctx: one.ctx as u64,
        ..b
    };
    let mut shrunk = 0;
    for quarter in 1..=160u64 {
        let floor = quarter * G / 4;
        let admit = |n: usize| {
            admit_with(
                &lane_profile(&p, n, floor),
                &at,
                one.chunk as u64,
                &l,
                &h,
                &own,
                ws,
            )
        };
        let (lanes, got) = lanes_at(one, LANES_MAX, admit);
        assert_eq!((got.ctx, got.chunk), (one.ctx, one.chunk), "floor {floor}");
        let most = (1..=LANES_MAX)
            .rev()
            .find(|&n| n == 1 || admit(n).is_ok())
            .unwrap();
        assert_eq!(lanes, most, "floor {floor}");
        if lanes > 1 {
            assert_eq!(got.admission, admit(lanes).unwrap(), "floor {floor}");
        } else {
            assert_eq!(got.admission, one.admission, "floor {floor}");
        }
        if let Ok(ladder) = pick(
            &lane_profile(&p, LANES_MAX, floor),
            &b,
            max_ctx,
            &l,
            &h,
            &own,
            ws,
        ) && (ladder.ctx, ladder.chunk) != (one.ctx, one.chunk)
        {
            shrunk += 1;
        }
    }
    assert!(shrunk > 0, "no floor shrinks the window or the chunk");
}

#[test]
fn cache_split_follows_the_layer_counts() {
    let c = Classes {
        layers: [19, 22],
        slot: [4_663_296, 4_091_904],
    };
    for bytes in [0, 4_663_296, 17 * G, 48 * G + 12_345, 56_299_972_608] {
        let [a, b] = c.split(bytes);
        let used = a as u64 * c.slot[0] + b as u64 * c.slot[1];
        assert!(used <= bytes, "{bytes}");
        assert!(bytes - used < c.slot[1], "{bytes}: {a} {b}");
        if bytes >= 17 * G {
            // equal slots per layer within one round
            let (pa, pb) = (a as f64 / 19.0, b as f64 / 22.0);
            assert!((pa - pb).abs() < 1.0, "{bytes}: {pa} {pb}");
        }
    }
    let only_b = Classes {
        layers: [0, 22],
        ..c
    };
    assert_eq!(only_b.split(10 * c.slot[1] + 1), [0, 10]);
    let only_a = Classes {
        layers: [19, 0],
        ..c
    };
    assert_eq!(only_a.split(10 * c.slot[0] + 1), [10, 0]);
    let none = Classes {
        layers: [0, 0],
        ..c
    };
    assert_eq!(none.split(G), [0, 0]);
}

#[test]
fn chunk_env_values() {
    assert_eq!(chunk_cap(None).unwrap(), None);
    assert_eq!(chunk_cap(Some(" ")).unwrap(), None);
    assert_eq!(chunk_cap(Some("4096")).unwrap(), Some(4096));
    assert_eq!(chunk_cap(Some(" 256 ")).unwrap(), Some(256));
    for v in ["0", "128", "1000", "-256", "abc", "4096.0"] {
        let e = chunk_cap(Some(v)).unwrap_err().reason;
        assert!(e.contains(CHUNK_ENV), "{v}: {e}");
    }
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    Scan::open(&paths).unwrap()
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_sizes_match_the_oracle() {
    let s = real_scan();
    let sizes = Sizes::of(&s).unwrap();
    let addons = fixture("admission-addons.json");
    let inputs = &addons["inputs"];
    assert_eq!(sizes.dense_trunk, u(&inputs["dense_trunk"]));
    assert_eq!(sizes.dense_mtp, u(&inputs["dense_mtp_plan"]));
    let mtp = fixture("mtp-tensors.json");
    assert_eq!(sizes.dense_mtp, u(&mtp["totals"]["plan_bytes"]));
    let adm = fixture("admission.json");
    assert_eq!(
        sizes.classes.slot,
        [u(&adm["inputs"]["slot_a"]), u(&adm["inputs"]["slot_b"])]
    );
    assert_eq!(sizes.classes.layers, [19, 22]);
    let m = sizes.model(true, 262_144);
    assert_eq!(
        (m.dense, m.slot, m.n_experts),
        (
            u(&addons["variants"]["mtp_on"]["dense"]),
            u(&inputs["slot"]),
            u(&inputs["n_experts"])
        )
    );
    assert_eq!(sizes.model(false, 1).dense, sizes.dense_trunk);
    let pf = &addons["profile"];
    assert_eq!(kv_per_token(&s.cfg, true), u(&pf["kv_per_ctx_mtp_on"]));
    assert_eq!(kv_per_token(&s.cfg, false), u(&pf["kv_per_ctx_mtp_off"]));
}

/// Real headers (CPU): the lane floor with the MTP layer and the Q8
/// shapes the row cap is taken over.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_lane_floor_and_wave_shapes() {
    let s = real_scan();
    assert_eq!(
        lane_floor(&s.cfg, true),
        30 * 515 * 576 * 2 + 13 * 6 * 128 * 4 + 8 * 3072 * 4
    );
    assert_eq!(
        lane_floor(&s.cfg, false),
        30 * 515 * 576 * 2 + 12 * 6 * 128 * 4
    );
    let shapes = q8_shapes(&s, true).unwrap();
    for want in [
        (192, 512),
        (512, 256),
        (3072, 576),
        (3072, 16_384),
        (1536, 15_360),
        (13_824, 3072),
        (6144, 3072),
    ] {
        assert!(shapes.contains(&want), "{want:?} in {shapes:?}");
    }
    assert!(!q8_shapes(&s, false).unwrap().contains(&(6144, 3072)));
    assert_eq!(row_cap("applegpu_g17s", None, &shapes).unwrap(), 12);
    assert_eq!(row_cap("applegpu_g15d", None, &shapes).unwrap(), 11);
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_store_admits_at_load() {
    let s = real_scan();
    let inputs = fixture("admission-addons.json")["inputs"].clone();
    TEST_HOST.with(|c| c.set(Some(host(&inputs, 20 * G))));
    let opts = Options {
        max_context: None,
        mtp: true,
        chunk_cap: None,
        lanes: LANES_MAX,
        plan: CachePlan::AdmitWith(Limits::default()),
        max_memory: None,
        direct: DirectMode::Off,
    };
    let st = Store::open(&s.paths, &opts).unwrap();
    TEST_HOST.with(|c| c.set(None));
    assert_eq!((st.ctx, st.chunk, st.lanes), (262_144, 8192, LANES_MAX));
    assert!(st.row_cap >= 2, "row cap {}", st.row_cap);
    let a = st.admission().unwrap();
    assert_eq!(st.cache_slots(), st.sizes.classes.split(a.cache_bytes()));
    assert_eq!(st.bank.staging_layers() as u64, a.staging_layers);
    assert_eq!(st.dense_bytes, st.sizes.dense_trunk);
    assert!(st.model.new_session(st.ctx).is_ok());
}

struct Temp(PathBuf);

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn moe() -> MoeConfig {
    MoeConfig {
        n_experts: 4,
        n_used: 2,
        n_shared: 1,
        ff: 256,
        latent: 512,
        route_scale: 1.0,
        norm_topk: true,
    }
}

fn layer(m: &MoeConfig, il: usize, class: ExpertClass) -> ExpertLayer {
    let types = class_types(class).unwrap();
    let dims = [
        [m.latent, m.ff, m.n_experts],
        [m.latent, m.ff, m.n_experts],
        [m.ff, m.latent, m.n_experts],
    ];
    let loc = |p: usize, name: &str| Located {
        shard: 0,
        data_offset: 0,
        info: GgufTensorInfo {
            name: format!("blk.{il}.{name}.weight"),
            n_dims: 3,
            dims: dims[p].iter().map(|&d| d as u64).collect(),
            tensor_type: types[p],
            offset: 0,
        },
    };
    ExpertLayer {
        class,
        gate: loc(0, "ffn_gate_exps"),
        up: loc(1, "ffn_up_exps"),
        down: loc(2, "ffn_down_exps"),
    }
}

/// A two-class bank of 4 experts whose slots are never read.
fn tiny_bank(cache: [usize; 2], staging: usize) -> (KqBank, Classes, Temp) {
    let m = moe();
    let experts = vec![
        Some(layer(&m, 0, ExpertClass::Q6kDown)),
        Some(layer(&m, 1, ExpertClass::Q4kDown)),
        Some(layer(&m, 2, ExpertClass::Q4kDown)),
    ];
    let dir = std::env::temp_dir().join(format!("hy4-store-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("experts.bin");
    std::fs::write(&path, [0u8; 64]).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 1).unwrap();
    let bank = KqBank::new(pool, &m, &experts, cache, staging).unwrap();
    let slot = CLASSES.map(|c| {
        geometry(&m, c)
            .unwrap()
            .iter()
            .map(|g| g.slot_bytes())
            .sum::<u64>()
    });
    assert_eq!(slot, bank.slot_bytes());
    (
        bank,
        Classes {
            layers: [1, 2],
            slot,
        },
        Temp(dir),
    )
}

/// A staging-only bank is rebuilt at the first turn the host has room in,
/// kept at the next, and a refused turn leaves the bank volatile.
#[test]
fn lease_rebuilds_keeps_and_refuses() {
    let (bank, classes, _dir) = tiny_bank([0, 0], 1);
    let p = profile(100);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: 1024,
    };
    let rows = 1024;
    let cap = 64 * size.slot;
    let limits = Limits {
        cache_cap: Some(cap),
        ..Limits::default()
    };
    let roomy = (
        HostMemory {
            physical: 64 * G,
            wired: 10 * G,
            ..Default::default()
        },
        48 * G,
    );
    let loaded = admit_with(&p, &size, rows, &limits, &roomy.0, &Own::default(), roomy.1).unwrap();
    assert_eq!((loaded.cache_slots, loaded.bound), (64, Bound::Env));
    let staging_only = Admission {
        staging_layers: 1,
        fixed: loaded.terms.fixed(1),
        cache_slots: 0,
        ..loaded
    };
    let lease = Lease::new(
        p,
        size,
        rows,
        classes,
        Some(limits),
        Some(staging_only),
        &bank,
    );
    bank.end_turn().unwrap();
    TEST_HOST.with(|c| c.set(Some(roomy)));
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert!(lease.in_turn() && bank.in_turn());
    let a = lease.admission().unwrap();
    assert_eq!(a.cache_slots, 64);
    assert_eq!(bank.cache_slots(), classes.split(a.cache_bytes()));
    assert_eq!(bank.staging_layers() as u64, a.staging_layers);
    assert_eq!(lease.switches().len(), 1);
    lease.end_turn(&bank).unwrap();
    assert!(!lease.in_turn() && !bank.in_turn());
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!(lease.admission().unwrap(), a);
    assert_eq!(lease.switches().len(), 1);
    lease.end_turn(&bank).unwrap();
    let full = (
        HostMemory {
            wired: 60 * G,
            ..roomy.0
        },
        roomy.1,
    );
    TEST_HOST.with(|c| c.set(Some(full)));
    let e = lease.begin_turn(&bank, 0, 0).unwrap_err().reason;
    TEST_HOST.with(|c| c.set(None));
    assert!(e.contains("vs budget"), "{e}");
    assert!(!lease.in_turn() && !bank.in_turn());
    assert_eq!(lease.admission().unwrap(), a);
}

/// Inside a turn of a load bound by live memory the lease keeps the bank
/// for the free pool within its term and for room that came back (growth
/// waits for a turn start), and shrinks it to what the host admits when
/// others grow past the slack. The kernel holds the banks non-volatile
/// inside a turn and volatile after it. The admission prices 64 MiB slots
/// over the tiny bank's real ones.
#[test]
fn lease_recheck_keeps_for_the_pool_and_shrinks_for_others() {
    let s = 64 << 20;
    let classes = Classes {
        layers: [1, 2],
        slot: [s, s],
    };
    let p = profile(100);
    let size = ModelSize {
        dense: 0,
        slot: s,
        n_experts: 8,
        ctx: 1024,
    };
    let rows = 1024;
    let limits = Limits::default();
    let ws = 96 * G;
    let one = Terms::with(&p, &size, ws, rows).fixed(1);
    let others = 10 * G;
    let host = |wired| HostMemory {
        physical: limits.reserve + others + one + 64 * s,
        wired,
        ..Default::default()
    };
    let loaded = admit_with(&p, &size, rows, &limits, &host(others), &Own::default(), ws).unwrap();
    assert_eq!(loaded.bound, Bound::Live);
    assert!(loaded.cache_slots > 48, "{loaded}");
    let (bank, _, _dir) = tiny_bank(
        classes.split(loaded.cache_bytes()),
        loaded.staging_layers as usize,
    );
    let lease = Lease::new(p, size, rows, classes, Some(limits), Some(loaded), &bank);
    bank.end_turn().unwrap();
    let r = bank.residency().unwrap();
    assert_eq!(r.volatile, r.resident);
    TEST_HOST.with(|c| c.set(Some((host(others), ws))));
    TEST_POOL.with(|c| c.set(Some(0)));
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!(lease.admission().unwrap(), loaded);
    assert!(lease.switches().is_empty());
    let r = bank.residency().unwrap();
    assert_eq!(r.volatile, 0);
    let pool = loaded.terms.pool;
    TEST_HOST.with(|c| c.set(Some((host(others + r.resident + pool + G), ws))));
    TEST_POOL.with(|c| c.set(Some(pool)));
    assert!(!lease.recheck(&bank, 0, 0).unwrap());
    TEST_POOL.with(|c| c.set(Some(0)));
    TEST_HOST.with(|c| c.set(Some((host(r.resident), ws))));
    assert!(!lease.recheck(&bank, 0, 0).unwrap());
    assert_eq!(lease.admission().unwrap(), loaded);
    assert!(lease.switches().is_empty());
    TEST_HOST.with(|c| c.set(Some((host(others + r.resident + pool + G), ws))));
    assert!(lease.recheck(&bank, 0, 0).unwrap());
    TEST_HOST.with(|c| c.set(None));
    TEST_POOL.with(|c| c.set(None));
    let a = lease.admission().unwrap();
    assert!(
        a.cache_slots > 0 && a.cache_slots < loaded.cache_slots,
        "{a}"
    );
    assert_eq!(bank.cache_slots(), classes.split(a.cache_bytes()));
    assert_eq!(lease.switches().len(), 1);
    assert!(lease.in_turn() && bank.in_turn());
    lease.end_turn(&bank).unwrap();
}

/// A turn start reads the purgeable count before and after its flip: with
/// the count current (others' volatile pages the bank's size) the warm bank
/// stays; with the bank left in it, the same reading after the flip finds
/// others past the slack and shrinks the bank. The admission prices 64 MiB
/// slots over the tiny bank's real ones.
#[test]
fn lease_turn_start_takes_the_flip_lag_out() {
    let s = 64 << 20;
    let classes = Classes {
        layers: [1, 2],
        slot: [s, s],
    };
    let p = profile(100);
    let size = ModelSize {
        dense: 0,
        slot: s,
        n_experts: 8,
        ctx: 1024,
    };
    let rows = 1024;
    let limits = Limits::default();
    let ws = 96 * G;
    let one = Terms::with(&p, &size, ws, rows).fixed(1);
    let others = 10 * G;
    let physical = limits.reserve + others + one + 64 * s;
    let loaded = admit_with(
        &p,
        &size,
        rows,
        &limits,
        &HostMemory {
            physical,
            wired: others,
            ..Default::default()
        },
        &Own::default(),
        ws,
    )
    .unwrap();
    assert_eq!(loaded.bound, Bound::Live);
    let (bank, _, _dir) = tiny_bank(
        classes.split(loaded.cache_bytes()),
        loaded.staging_layers as usize,
    );
    let lease = Lease::new(p, size, rows, classes, Some(limits), Some(loaded), &bank);
    bank.touch();
    bank.end_turn().unwrap();
    let r = bank.residency().unwrap();
    assert!(r.resident > 0 && r.volatile == r.resident, "{r:?}");
    // others read exactly at the edge the held bank shrinks past once the
    // bank's pages count as used
    let held = loaded.fixed + loaded.cache_bytes();
    let edge = physical - limits.reserve + crate::moe_stream::budget::KEEP_SLACK - held;
    let h = HostMemory {
        physical,
        wired: edge + 1,
        anonymous: r.resident,
        purgeable: r.resident,
        ..Default::default()
    };
    TEST_HOST.with(|c| c.set(Some((h, ws))));
    TEST_POOL.with(|c| c.set(Some(0)));
    TEST_BEFORE_FLIP.with(|c| c.set(Some(2 * r.resident)));
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert_eq!(lease.admission().unwrap(), loaded);
    assert!(lease.switches().is_empty());
    lease.end_turn(&bank).unwrap();
    assert_eq!(bank.residency().unwrap(), r);
    TEST_BEFORE_FLIP.with(|c| c.set(Some(r.resident)));
    lease.begin_turn(&bank, 0, 0).unwrap();
    TEST_HOST.with(|c| c.set(None));
    TEST_POOL.with(|c| c.set(None));
    TEST_BEFORE_FLIP.with(|c| c.set(None));
    let a = lease.admission().unwrap();
    assert!(a.cache_slots < loaded.cache_slots, "{a}");
    assert_eq!(lease.switches().len(), 1);
    lease.end_turn(&bank).unwrap();
}

/// Without limits the lease never re-admits or switches.
#[test]
fn lease_without_admission_only_flips_the_bank() {
    let (bank, classes, _dir) = tiny_bank([2, 2], 1);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0],
        n_experts: 8,
        ctx: 64,
    };
    let lease = Lease::new(profile(100), size, 64, classes, None, None, &bank);
    bank.end_turn().unwrap();
    lease.begin_turn(&bank, 0, 0).unwrap();
    assert!(bank.in_turn() && !lease.in_turn());
    assert_eq!(bank.cache_slots(), [2, 2]);
    lease.end_turn(&bank).unwrap();
    assert!(!bank.in_turn());
    assert!(lease.switches().is_empty());
}

#[test]
fn max_memory_caps_the_working_set_only_below_it() {
    assert_eq!(capped_ws(100 * G, None), 100 * G);
    assert_eq!(capped_ws(100 * G, Some(64 * G)), 64 * G);
    assert_eq!(capped_ws(100 * G, Some(200 * G)), 100 * G);
}

#[test]
fn direct_env_values() {
    assert_eq!(direct_mode(None).unwrap(), DirectMode::Auto);
    assert_eq!(direct_mode(Some("")).unwrap(), DirectMode::Auto);
    assert_eq!(direct_mode(Some("  ")).unwrap(), DirectMode::Auto);
    assert_eq!(direct_mode(Some("auto")).unwrap(), DirectMode::Auto);
    assert_eq!(direct_mode(Some("AUTO")).unwrap(), DirectMode::Auto);
    assert_eq!(direct_mode(Some("1")).unwrap(), DirectMode::On);
    assert_eq!(direct_mode(Some("true")).unwrap(), DirectMode::On);
    assert_eq!(direct_mode(Some("TRUE")).unwrap(), DirectMode::On);
    assert_eq!(direct_mode(Some("0")).unwrap(), DirectMode::Off);
    assert_eq!(direct_mode(Some("false")).unwrap(), DirectMode::Off);
    assert_eq!(direct_mode(Some("FALSE")).unwrap(), DirectMode::Off);
    for bad in ["2", "yes", "no", "maybe"] {
        let e = direct_mode(Some(bad)).unwrap_err().reason;
        assert!(e.contains(DIRECT_ENV), "{bad}: {e}");
    }
}

#[test]
fn direct_gate_admits_quiet_and_falls_back_when_busy_or_disabled() {
    let sizes = Sizes {
        dense_trunk: 8_003_184_948,
        dense_mtp: 2_189_929_600,
        classes: Classes {
            layers: [19, 22],
            slot: [4_663_296, 4_091_904],
        },
        n_experts: 512,
    };
    let direct_bytes = sizes.direct_expert_bytes();
    assert_eq!(direct_bytes, 91_455_750_144);
    assert_eq!(sizes.total_experts(), [9_728, 11_264]);

    let dense_mtp = sizes.dense_trunk + sizes.dense_mtp;
    let p = profile(15_812);
    let ctx = 262_144usize;
    let kv_initial = p.kv_base + p.kv_per_ctx * ctx as u64;
    let scratch_8k = direct_scratch_bytes(&p, CHUNKS[0], ctx);
    let quiet = HostMemory {
        physical: 128 * G,
        wired: 20 * G,
        ..Default::default()
    };
    let d_auto = direct_gate(
        DirectMode::Auto,
        &quiet,
        &Own::default(),
        dense_mtp,
        direct_bytes,
        kv_initial,
        scratch_8k,
        DIRECT_MARGIN,
        None,
        None,
    );
    assert!(d_auto.admitted, "quiet 128 GiB host must admit direct mode");
    assert_eq!(d_auto.available, 128 * G - 20 * G - DIRECT_MARGIN);
    assert_eq!(d_auto.margin, DIRECT_MARGIN);
    assert_eq!(d_auto.scratch_bytes, scratch_8k);
    assert_eq!(d_auto.kv_and_scratch, kv_initial + scratch_8k);

    let d_on = direct_gate(
        DirectMode::On,
        &quiet,
        &Own::default(),
        dense_mtp,
        direct_bytes,
        kv_initial,
        scratch_8k,
        DIRECT_MARGIN,
        None,
        None,
    );
    assert!(d_on.admitted);

    // MLX_HY4_DIRECT=0 disables direct residency even on a quiet host.
    let d_off = direct_gate(
        DirectMode::Off,
        &quiet,
        &Own::default(),
        dense_mtp,
        direct_bytes,
        kv_initial,
        scratch_8k,
        DIRECT_MARGIN,
        None,
        None,
    );
    assert!(!d_off.admitted);

    // Busy host falls back cleanly to partial streaming.
    let busy = HostMemory {
        physical: 128 * G,
        wired: 45 * G,
        ..Default::default()
    };
    assert!(
        !direct_gate(
            DirectMode::Auto,
            &busy,
            &Own::default(),
            dense_mtp,
            direct_bytes,
            kv_initial,
            scratch_8k,
            DIRECT_MARGIN,
            None,
            None,
        )
        .admitted
    );
    assert!(
        !direct_gate(
            DirectMode::On,
            &busy,
            &Own::default(),
            dense_mtp,
            direct_bytes,
            kv_initial,
            scratch_8k,
            DIRECT_MARGIN,
            None,
            None,
        )
        .admitted
    );

    // max_memory_bytes or cache_cap below requirement disallows direct mode.
    assert!(
        !direct_gate(
            DirectMode::Auto,
            &quiet,
            &Own::default(),
            dense_mtp,
            direct_bytes,
            kv_initial,
            scratch_8k,
            DIRECT_MARGIN,
            Some(96 * G),
            None,
        )
        .admitted
    );
    assert!(
        !direct_gate(
            DirectMode::Auto,
            &quiet,
            &Own::default(),
            dense_mtp,
            direct_bytes,
            kv_initial,
            scratch_8k,
            DIRECT_MARGIN,
            None,
            Some(direct_bytes - 1),
        )
        .admitted
    );
    assert!(
        direct_gate(
            DirectMode::Auto,
            &quiet,
            &Own::default(),
            dense_mtp,
            direct_bytes,
            kv_initial,
            scratch_8k,
            DIRECT_MARGIN,
            None,
            Some(direct_bytes),
        )
        .admitted
    );

    // Wired limit request is always capped at working_set().
    let ws = 96_000_000_000;
    assert_eq!(direct_wired_limit(d_auto.need, ws), ws);
    assert_eq!(direct_wired_limit(50 * G, ws), 50 * G);
}

#[test]
fn direct_gate_respects_explicit_reserve_override_and_steps_chunk_down() {
    assert_eq!(parse_reserve_override(None), None);
    assert_eq!(parse_reserve_override(Some("  ")), None);
    assert_eq!(parse_reserve_override(Some("16")), Some(16 * G));
    assert_eq!(direct_margin(None), DIRECT_MARGIN);
    assert_eq!(direct_margin(Some(2 * G)), DIRECT_MARGIN);
    assert_eq!(direct_margin(Some(16 * G)), 16 * G);

    let sizes = Sizes {
        dense_trunk: 8_003_184_948,
        dense_mtp: 2_189_929_600,
        classes: Classes {
            layers: [19, 22],
            slot: [4_663_296, 4_091_904],
        },
        n_experts: 512,
    };
    let dense_mtp = sizes.dense_trunk + sizes.dense_mtp;
    let direct_bytes = sizes.direct_expert_bytes();
    let p_mtp = profile(15_812);
    let p_ar = profile(14_596);
    let ctx = 262_144usize;

    // Direct KV allowance caps at 64K tokens while respecting smaller explicit max_context.
    let kv_64k = direct_kv_allowance(&p_mtp, ctx);
    assert_eq!(
        kv_64k,
        p_mtp.kv_base + p_mtp.kv_per_ctx * DIRECT_KV_ALLOWANCE_TOKENS as u64
    );
    assert_eq!(
        direct_kv_allowance(&p_mtp, 32_768),
        p_mtp.kv_base + p_mtp.kv_per_ctx * 32_768
    );

    let s8 = direct_scratch_bytes(&p_mtp, 8192, ctx);
    let s4 = direct_scratch_bytes(&p_mtp, 4096, ctx);
    let s2 = direct_scratch_bytes(&p_mtp, 2048, ctx);
    assert!(s8 > s4 && s4 > s2);

    let fixed_base = dense_mtp + direct_bytes + kv_64k;
    let need_8k = fixed_base + s8;
    let need_4k = fixed_base + s4;
    let need_2k = fixed_base + s2;

    // With 64K direct KV allowance + chunk step-down, MTP-1 fits at 26 GiB others
    // and AR fits at 28 GiB others on a 128 GiB host.
    let (mtp_chunk, mtp_gate) = pick_direct(
        DirectMode::Auto,
        &p_mtp,
        ctx,
        &HostMemory {
            physical: 128 * G,
            wired: 26 * G,
            ..Default::default()
        },
        &Own::default(),
        dense_mtp,
        direct_bytes,
        None,
        None,
        None,
    );
    assert!(
        mtp_gate.admitted,
        "MTP-1 must admit direct at 26 GiB others"
    );
    assert_eq!(mtp_chunk, 2048);
    assert_eq!(mtp_gate.kv_and_scratch, kv_64k + s2);

    let (ar_chunk, ar_gate) = pick_direct(
        DirectMode::Auto,
        &p_ar,
        ctx,
        &HostMemory {
            physical: 128 * G,
            wired: 28 * G,
            ..Default::default()
        },
        &Own::default(),
        sizes.dense_trunk,
        direct_bytes,
        None,
        None,
        None,
    );
    assert!(ar_gate.admitted, "AR must admit direct at 28 GiB others");
    assert_eq!(ar_chunk, 2048);

    // Quiet host admits direct at 8192 when no reserve override is set.
    let quiet = HostMemory {
        physical: 128 * G,
        wired: 20 * G,
        ..Default::default()
    };
    let (chunk, gate) = pick_direct(
        DirectMode::Auto,
        &p_mtp,
        ctx,
        &quiet,
        &Own::default(),
        dense_mtp,
        direct_bytes,
        None,
        None,
        None,
    );
    assert!(gate.admitted);
    assert_eq!(chunk, 8192);
    assert_eq!(gate.margin, DIRECT_MARGIN);
    assert_eq!(gate.scratch_bytes, s8);
    assert_eq!(gate.kv_and_scratch, kv_64k + s8);

    // Raising the explicit reserve override pushes available just below 2048's need
    // and forces streamed fallback at the exact boundary.
    let exact_reserve = quiet.physical - quiet.wired - need_2k;
    let (chunk_ok, gate_ok) = pick_direct(
        DirectMode::Auto,
        &p_mtp,
        ctx,
        &quiet,
        &Own::default(),
        dense_mtp,
        direct_bytes,
        Some(exact_reserve),
        None,
        None,
    );
    assert!(gate_ok.admitted);
    assert_eq!(chunk_ok, 2048);
    assert_eq!(gate_ok.margin, exact_reserve.max(DIRECT_MARGIN));

    let (_, gate_refused) = pick_direct(
        DirectMode::Auto,
        &p_mtp,
        ctx,
        &quiet,
        &Own::default(),
        dense_mtp,
        direct_bytes,
        Some(exact_reserve + 1),
        None,
        None,
    );
    assert!(!gate_refused.admitted);
    assert_eq!(gate_refused.margin, (exact_reserve + 1).max(DIRECT_MARGIN));

    // Tight max_memory steps direct chunk down 8192 -> 4096 -> 2048 -> streamed.
    for (cap, want_admitted, want_chunk, want_scratch) in [
        (need_8k, true, 8192usize, s8),
        (need_8k - 1, true, 4096, s4),
        (need_4k, true, 4096, s4),
        (need_4k - 1, true, 2048, s2),
        (need_2k, true, 2048, s2),
        (need_2k - 1, false, 2048, s2),
    ] {
        let (got_chunk, got_gate) = pick_direct(
            DirectMode::Auto,
            &p_mtp,
            ctx,
            &quiet,
            &Own::default(),
            dense_mtp,
            direct_bytes,
            None,
            Some(cap),
            None,
        );
        assert_eq!(got_gate.admitted, want_admitted, "cap={cap}");
        assert_eq!(got_chunk, want_chunk, "cap={cap}");
        assert_eq!(got_gate.scratch_bytes, want_scratch, "cap={cap}");
    }
}

fn filled_fixture_bank(
    cache: [usize; 2],
    staging: usize,
) -> (KqBank, KqBank, Classes, MoeConfig, Temp) {
    let m = moe();
    let mut experts = Vec::new();
    let mut off = 0u64;
    for (il, class) in [
        (0, ExpertClass::Q6kDown),
        (1, ExpertClass::Q4kDown),
        (2, ExpertClass::Q4kDown),
    ] {
        let mut l = layer(&m, il, class);
        let geo = geometry(&m, class).unwrap();
        for (p, loc) in [&mut l.gate, &mut l.up, &mut l.down]
            .into_iter()
            .enumerate()
        {
            loc.data_offset = off;
            off += geo[p].file_bytes() * m.n_experts as u64;
        }
        experts.push(Some(l));
    }
    let dir = std::env::temp_dir().join(format!("hy4-store-promo-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("experts.bin");
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let payload: Vec<u8> = (0..off as usize)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect();
    std::fs::write(&path, &payload).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 2).unwrap();
    let ref_pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 2).unwrap();
    let bank = KqBank::new(pool, &m, &experts, cache, staging).unwrap();
    let reference = KqBank::new(ref_pool, &m, &experts, [0, 0], 1).unwrap();
    let classes = Classes {
        layers: [1, 2],
        slot: bank.slot_bytes(),
    };
    (bank, reference, classes, m, Temp(dir))
}

#[test]
fn lease_end_turn_idle_promotes_partial_to_direct_when_others_drop() {
    let (bank, reference, classes, m, _tmp) = filled_fixture_bank([2, 2], 1);
    let p = profile(15_812);
    let ctx = 262_144usize;
    let dense = 10 * G;
    let kv_live = 2 * G;
    let ws = 96 * G;
    let promo = DirectPromotion {
        mode: DirectMode::Auto,
        reserve_override: None,
        n_experts: m.n_experts,
    };
    // Size the virtual admission using real-scale 41-layer direct bytes? Here the
    // fixture has 3 tiny layers, so set physical so high-others refuses and
    // low-others admits cleanly.
    let direct_bytes = promo.direct_expert_bytes(&classes);
    let kv_64k = direct_kv_allowance(&p, ctx);
    let scratch_2k = direct_scratch_bytes(&p, 2048, ctx);
    let physical = 40 * G;
    let need_2k = dense + direct_bytes + kv_64k + scratch_2k;
    let size = ModelSize {
        dense,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: ctx as u64,
    };
    let limits = Limits::default();
    let partial_adm = Admission {
        physical,
        host_used: physical - need_2k - DIRECT_MARGIN + 2 * G,
        own: 0,
        others: physical - need_2k - DIRECT_MARGIN + 2 * G,
        reserve: limits.reserve,
        live: 20 * G,
        ws,
        terms: Terms::with(&p, &size, ws, 2048),
        staging_layers: 1,
        fixed: dense + kv_64k + scratch_2k,
        slot: size.slot,
        cache_slots: 4,
        bound: Bound::Live,
    };
    let lease = Lease::new(
        p,
        size,
        2048,
        classes,
        Some(limits),
        Some(partial_adm),
        &bank,
    )
    .with_direct_promotion(promo, None);

    // Turn 1: others stay above the direct threshold -> end_turn_idle does NOT promote.
    let r = bank.residency().unwrap();
    let high_others = physical - need_2k - DIRECT_MARGIN + G;
    TEST_HOST.with(|c| {
        c.set(Some((
            HostMemory {
                physical,
                wired: high_others + dense + r.resident + kv_live,
                ..Default::default()
            },
            ws,
        )))
    });
    assert_eq!(lease.end_turn_idle(&bank, dense, kv_live).unwrap(), None);
    assert!(!bank.is_direct());
    assert!(lease.switches().is_empty());

    // Turn 2: others drop below threshold -> end_turn_idle promotes to Direct!
    let low_others = 2 * G;
    TEST_HOST.with(|c| {
        c.set(Some((
            HostMemory {
                physical,
                wired: low_others + dense + r.resident + kv_live,
                ..Default::default()
            },
            ws,
        )))
    });
    let promoted = lease.end_turn_idle(&bank, dense, kv_live).unwrap();
    assert!(promoted.is_some());
    assert!(bank.is_direct());
    assert!(!bank.in_turn());
    assert_eq!(lease.switches().len(), 1);
    assert_eq!(lease.admission().unwrap().staging_layers, 0);
    assert!(lease.direct_budget().is_some());
    bank.wait_direct_fill().unwrap();
    assert!(bank.direct_fill_complete());

    // Promoted direct bank serves identical expert bytes across every layer.
    let all_experts: Vec<u32> = (0..m.n_experts as u32).collect();
    for il in 0..3 {
        let dir_p = bank.prepare(il, &all_experts, &[], true).unwrap();
        let ref_p = reference.prepare(il, &all_experts, &[], false).unwrap();
        assert_eq!(dir_p.slots, all_experts);
        for pi in 0..3 {
            let dw = dir_p.arrays[pi].w.to_uint32().unwrap().to_vec();
            let rw = ref_p.arrays[pi].w.to_uint32().unwrap().to_vec();
            let cols = dw.len() / m.n_experts;
            for (&e, &s) in ref_p.experts.iter().zip(&ref_p.slots) {
                let de = e as usize * cols..(e as usize + 1) * cols;
                let rs = s as usize * cols..(s as usize + 1) * cols;
                assert_eq!(dw[de], rw[rs], "layer {il} proj {pi} expert {e}");
            }
        }
        reference.settle_all();
    }

    // Subsequent turns stay direct and NonVolatile even if host memory spikes again.
    TEST_HOST.with(|c| {
        c.set(Some((
            HostMemory {
                physical,
                wired: physical - G,
                ..Default::default()
            },
            ws,
        )))
    });
    lease.begin_turn(&bank, dense, kv_live).unwrap();
    assert!(lease.in_turn() && bank.in_turn());
    assert_eq!(lease.end_turn_idle(&bank, dense, kv_live).unwrap(), None);
    assert_eq!(lease.switches().len(), 1);
    assert_eq!(bank.residency().unwrap().volatile, 0);
    TEST_HOST.with(|c| c.set(None));
    reference.end_turn().unwrap();
}

#[test]
fn lease_end_turn_idle_does_not_promote_when_disabled_capped_or_headroom_tight() {
    let p = profile(15_812);
    let ctx = 262_144usize;
    let dense = 10 * G;
    let physical = 40 * G;
    let ws = 96 * G;
    let quiet = (
        HostMemory {
            physical,
            wired: 12 * G,
            ..Default::default()
        },
        ws,
    );

    // 1. MLX_HY4_DIRECT=0 (DirectMode::Off) never promotes even on a quiet host.
    {
        let (bank, _r, classes, m, _tmp) = filled_fixture_bank([2, 2], 1);
        let size = ModelSize {
            dense,
            slot: classes.slot[0].max(classes.slot[1]),
            n_experts: 8,
            ctx: ctx as u64,
        };
        let lease = Lease::new(p, size, 2048, classes, Some(Limits::default()), None, &bank)
            .with_direct_promotion(
                DirectPromotion {
                    mode: DirectMode::Off,
                    reserve_override: None,
                    n_experts: m.n_experts,
                },
                None,
            );
        TEST_HOST.with(|c| c.set(Some(quiet)));
        assert_eq!(lease.end_turn_idle(&bank, dense, 0).unwrap(), None);
        assert!(!bank.is_direct());
        assert!(lease.switches().is_empty());
    }

    // 2. cache_cap below direct_expert_bytes never promotes.
    {
        let (bank, _r, classes, m, _tmp) = filled_fixture_bank([2, 2], 1);
        let promo = DirectPromotion {
            mode: DirectMode::Auto,
            reserve_override: None,
            n_experts: m.n_experts,
        };
        let direct_bytes = promo.direct_expert_bytes(&classes);
        let size = ModelSize {
            dense,
            slot: classes.slot[0].max(classes.slot[1]),
            n_experts: 8,
            ctx: ctx as u64,
        };
        let limits = Limits {
            cache_cap: Some(direct_bytes - 1),
            ..Limits::default()
        };
        let lease = Lease::new(p, size, 2048, classes, Some(limits), None, &bank)
            .with_direct_promotion(promo, None);
        TEST_HOST.with(|c| c.set(Some(quiet)));
        assert_eq!(lease.end_turn_idle(&bank, dense, 0).unwrap(), None);
        assert!(!bank.is_direct());
    }

    // 3. Large live KV exceeding physical headroom blocks promotion before dropping partial bank.
    {
        let (bank, _r, classes, m, _tmp) = filled_fixture_bank([2, 2], 1);
        let old_nbytes = bank.nbytes();
        let promo = DirectPromotion {
            mode: DirectMode::Auto,
            reserve_override: None,
            n_experts: m.n_experts,
        };
        let direct_bytes = promo.direct_expert_bytes(&classes);
        let size = ModelSize {
            dense,
            slot: classes.slot[0].max(classes.slot[1]),
            n_experts: 8,
            ctx: ctx as u64,
        };
        let lease = Lease::new(p, size, 2048, classes, Some(Limits::default()), None, &bank)
            .with_direct_promotion(promo, None);
        // Live KV + dense + others leaves less than DIRECT_MARGIN + growth, even
        // though subtracting own (dense + kv) makes pick_direct's 64K check pass.
        let huge_kv = 26 * G;
        TEST_HOST.with(|c| {
            c.set(Some((
                HostMemory {
                    physical,
                    wired: dense + huge_kv + G,
                    ..Default::default()
                },
                ws,
            )))
        });
        assert_eq!(lease.end_turn_idle(&bank, dense, huge_kv).unwrap(), None);
        assert!(!bank.is_direct());
        assert_eq!(bank.nbytes(), old_nbytes);
        assert!(direct_bytes > 0);
    }

    TEST_HOST.with(|c| c.set(None));
}

#[test]
fn lease_direct_bank_never_shrinks_or_flips_volatile() {
    let m = moe();
    let mut experts = Vec::new();
    let mut off = 0u64;
    for (il, class) in [
        (0, ExpertClass::Q6kDown),
        (1, ExpertClass::Q4kDown),
        (2, ExpertClass::Q4kDown),
    ] {
        let mut l = layer(&m, il, class);
        let geo = geometry(&m, class).unwrap();
        for (p, loc) in [&mut l.gate, &mut l.up, &mut l.down]
            .into_iter()
            .enumerate()
        {
            loc.data_offset = off;
            off += geo[p].file_bytes() * m.n_experts as u64;
        }
        experts.push(Some(l));
    }
    let dir = std::env::temp_dir().join(format!("hy4-store-direct-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let _tmp = Temp(dir.clone());
    let path = dir.join("experts.bin");
    std::fs::write(&path, vec![0u8; off as usize]).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 2).unwrap();
    let bank = KqBank::new_direct(pool, &m, &experts).unwrap();
    bank.wait_direct_fill().unwrap();

    let classes = Classes {
        layers: [1, 2],
        slot: bank.slot_bytes(),
    };
    let p = profile(100);
    let size = ModelSize {
        dense: 0,
        slot: classes.slot[0].max(classes.slot[1]),
        n_experts: 8,
        ctx: 1024,
    };
    let ws = 64 * G;
    let direct_adm = Admission {
        physical: 128 * G,
        host_used: 20 * G,
        own: 0,
        others: 20 * G,
        reserve: DIRECT_MARGIN,
        live: 104 * G,
        ws,
        terms: Terms {
            dense: 10 * G,
            staging_layer: 0,
            kv: 4 * G,
            compute: direct_scratch_bytes(&p, 1024, 1024),
            io: 0,
            driver: 0,
            pool: 0,
        },
        staging_layers: 0,
        fixed: 80 * G,
        slot: size.slot,
        cache_slots: 12,
        bound: Bound::Live,
    };
    let lease = Lease::new(
        p,
        size,
        1024,
        classes,
        Some(Limits::default()),
        Some(direct_adm),
        &bank,
    );
    // Wired limit must be capped at ws even though fixed + bank.nbytes() > ws.
    assert_eq!(lease.wired_limit(&bank), Some(ws));

    // Spiked host memory must NOT shrink, rebuild, or fail a direct bank.
    let busy = (
        HostMemory {
            physical: 128 * G,
            wired: 125 * G,
            ..Default::default()
        },
        ws,
    );
    TEST_HOST.with(|c| c.set(Some(busy)));
    lease.begin_turn(&bank, 10 * G, 4 * G).unwrap();
    assert!(lease.in_turn() && bank.in_turn());
    assert!(!lease.recheck(&bank, 10 * G, 4 * G).unwrap());
    assert!(lease.switches().is_empty());

    // end_turn releases wired lease but NEVER flips direct banks to Volatile.
    lease.end_turn(&bank).unwrap();
    assert!(!lease.in_turn() && !bank.in_turn());
    let r = bank.residency().unwrap();
    assert!(r.resident > 0);
    assert_eq!(
        r.volatile, 0,
        "direct banks must stay NonVolatile after end_turn"
    );

    // Second turn on busy host still succeeds with zero switches.
    lease.begin_turn(&bank, 10 * G, 4 * G).unwrap();
    assert!(lease.in_turn() && bank.in_turn());
    assert!(lease.switches().is_empty());
    lease.end_turn(&bank).unwrap();
    TEST_HOST.with(|c| c.set(None));
}

#[test]
fn shard1_nocache_and_page_cache_flags() {
    assert!(!shard1_nocache(None));
    assert!(!shard1_nocache(Some("")));
    assert!(!shard1_nocache(Some("0")));
    assert!(!shard1_nocache(Some("false")));
    assert!(shard1_nocache(Some("1")));
    assert!(shard1_nocache(Some("true")));
    assert!(shard1_nocache(Some(" TRUE ")));

    let m = moe();
    let experts = vec![None, Some(layer(&m, 1, ExpertClass::Q6kDown))];
    assert_eq!(
        shard_page_cache_for(2, &experts, 1, false),
        vec![true, false]
    );
    assert_eq!(
        shard_page_cache_for(2, &experts, 1, true),
        vec![false, false]
    );
    // Single-shard layout: Engram shard stays F_NOCACHE even when experts share it.
    assert_eq!(shard_page_cache_for(1, &experts, 0, false), vec![false]);
}
