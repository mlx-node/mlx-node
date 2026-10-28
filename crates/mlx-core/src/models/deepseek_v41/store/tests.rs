// Store v1 gates. Run with `MLX_ENABLE_TF32=0` (DECISIONS S5-1).

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::*;
use crate::array::{get_active_memory, get_cache_memory, set_cache_limit};
use crate::models::deepseek_v41::engram::{EngramRows, ResidentTable};
use crate::models::deepseek_v41::engram_hash::EngramHasher;
use crate::models::deepseek_v41::model::{Session, Trace};
use crate::models::deepseek_v41::test_fixture::{Npz, TinyGguf, fixture, read};
use crate::models::deepseek_v41::tests::{case_ids, mode_of, run, split, tiny};

fn tiny_store(mode: &str) -> Store {
    tiny_dense(mode, DenseMode::Exact)
}

fn tiny_dense(mode: &str, dense: DenseMode) -> Store {
    let (act, fq) = mode_of(mode);
    Store::open_dense(&[fixture("tiny-q8_0.gguf")], act, fq, CachePlan::Off, dense).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Store (pread dense, streamed experts, pread engram rows) == the S9
/// resident model on the same affine dense values: bitwise logits for a
/// prefill (c1), a decode (c3) and a chunked split (c2).
#[test]
fn tiny_store_matches_resident() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    for mode in ["fp32_on", "bf16_on"] {
        let (act, fq) = mode_of(mode);
        let resident = tiny(&g, act, fq, true);
        let store = tiny_store(mode);
        for case in ["c1", "c2", "c3"] {
            let ids = case_ids(&npz, case);
            let (want, _) = run(&resident, &ids, &split(case));
            let before = store.bank.stats();
            let (got, traces) = run(&store.model, &ids, &split(case));
            let after = store.bank.stats();
            assert_eq!(bits(&got), bits(&want), "{case} {mode}: store != resident");
            let used: usize = traces
                .iter()
                .flat_map(|t| t.routes.iter())
                .map(|r| r.ids.iter().collect::<BTreeSet<_>>().len())
                .sum();
            assert!(after.slots > before.slots && after.slots - before.slots <= used);
            println!(
                "{case} {mode}: bitwise; {} slot reads, {} B",
                after.slots - before.slots,
                after.bytes - before.bytes
            );
        }
    }
}

/// A gather whose slot holds another layer's expert panics (`assert!`).
#[test]
fn stale_tag_panics() {
    let npz = Npz::open("oracle.npz");
    let store = tiny_store("fp32_on");
    run(&store.model, &case_ids(&npz, "c1"), &[24]);
    let last = store.model.cfg.n_layers - 1;
    assert!(store.bank.tags().iter().flatten().all(|&(l, _)| l == last));
    store.bank.skip_fill.store(true, Ordering::Relaxed);
    let mut sess = store.model.new_session();
    let out = catch_unwind(AssertUnwindSafe(|| {
        store.model.forward(&mut sess, &[5], None, None)
    }));
    let Err(msg) = out else {
        panic!("stale gather must panic");
    };
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("stale slot"), "{msg}");
}

/// Outputs evaluated before a refill keep their values; the refilled slots
/// give the new layer's experts, both bitwise to whole-tensor banks.
#[test]
fn refill_after_eval_is_safe() {
    let g = TinyGguf::open();
    let store = tiny_store("fp32_on");
    let (w, s) = store.bank.arrays(0).unwrap();
    let streamed = ExpertBank {
        weight: w,
        scales: s,
    };
    let k = g.raw("blk.0.ffn_gate_exps.weight").0.mlx_shape()[2];
    let x: Vec<f32> = (0..3 * k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    let x = MxArray::from_float32(&x, &[3, 1, 1, k]).unwrap();
    let ids = MxArray::from_uint32(&[0, 5, 11], &[3, 1]).unwrap();
    let resident = |l: usize| {
        let (info, raw) = g.raw(&format!("blk.{l}.ffn_gate_exps.weight"));
        read(
            &ExpertBank::from_mxfp4(info, &raw)
                .unwrap()
                .gather(&x, &ids)
                .unwrap(),
        )
    };
    store.bank.fill(0, &[0, 5, 11]).unwrap();
    let out0 = streamed.gather(&x, &ids).unwrap();
    out0.eval();
    let first = read(&out0);
    store.bank.fill(1, &[0, 5, 11]).unwrap();
    store.bank.assert_tags(1, &[0, 5, 11]);
    let out1 = read(&streamed.gather(&x, &ids).unwrap());
    assert_eq!(bits(&first), bits(&resident(0)));
    assert_eq!(bits(&read(&out0)), bits(&first), "evaluated output changed");
    assert_eq!(bits(&out1), bits(&resident(1)));
    assert_ne!(bits(&out1), bits(&first));
}

/// Engram rows by pread == the resident table; a row outside its column's
/// `[offset, offset + prime)` is refused.
#[test]
fn row_range_checked() {
    let (g, npz) = (TinyGguf::open(), Npz::open("oracle.npz"));
    let cfg = DeepseekV41Config::from_gguf(&g.gguf).unwrap();
    let ec = cfg.engram.as_ref().unwrap();
    let (info, raw) = g.raw(ENGRAM_TABLE);
    let info = info.clone();
    let path = fixture("tiny-q8_0.gguf");
    let t = PreadTable::open(&path, g.gguf.data_offset, &info, ec).unwrap();
    let resident = ResidentTable::new(raw, ec.head_dim).unwrap();
    let h = EngramHasher::new(ec).unwrap();
    let (n, cols) = (h.n_layers(), h.n_cols());
    let rows = h.rows(&case_ids(&npz, "c1"), 0);
    for e in 0..n {
        let ids: Vec<u32> = rows
            .chunks(n * cols)
            .flat_map(|p| p[e * cols..(e + 1) * cols].iter().copied())
            .collect();
        let got = t.rows(e, &ids).unwrap();
        assert_eq!(bits(&got), bits(&resident.rows(e, &ids).unwrap()));
        let mut bad = ids[..cols].to_vec();
        bad[0] = ec.head_offsets[e * cols + 1] as u32;
        assert!(t.rows(e, &bad).is_err(), "row of column 1 in column 0");
        bad[0] = ec.head_offsets[e * cols] as u32 - u32::from(e > 0);
        if e > 0 {
            assert!(t.rows(e, &bad).is_err(), "row below column 0");
        }
        assert!(t.rows(e, &ids[..cols - 1]).is_err(), "partial position");
    }
}

fn cached_store(mode: &str, slots: usize) -> Store {
    cached_store_in(mode, slots, CacheMode::default())
}

fn cached_store_in(mode: &str, slots: usize, cache_mode: CacheMode) -> Store {
    let (act, fq) = mode_of(mode);
    Store::open_full(
        &[fixture("tiny-q8_0.gguf")],
        act,
        fq,
        CachePlan::Slots(slots),
        DenseMode::Exact,
        cache_mode,
    )
    .unwrap()
}

/// Logits of `ids` fed in calls of `sizes` on `sess`.
fn calls(store: &Store, sess: &mut Session, ids: &[u32], sizes: &[usize]) -> Vec<f32> {
    let (mut out, mut at) = (Vec::new(), 0);
    for &n in sizes {
        out.extend(read(
            &store
                .model
                .forward(sess, &ids[at..at + n], None, None)
                .unwrap(),
        ));
        at += n;
    }
    out
}

fn pool_ok(what: &str) {
    let c = get_cache_memory();
    assert!(
        c <= 2.0 * (1u64 << 30) as f64,
        "{what}: MLX cache {c} B > 2 GiB"
    );
}

/// Cache sizes 1 / 6 / 16 / 64 slots (12 experts x 8 layers = 96 pairs;
/// 1-16 exercise victims and the staging fallback, 64 hits):
/// logits bitwise == the staging-only store (S11) for c1, c2, c3 run one
/// after another on one store (later cases start from a warm cache).
#[test]
fn tiny_cache_bitwise_any_size() {
    let npz = Npz::open("oracle.npz");
    for mode in ["fp32_on", "bf16_on"] {
        let reference = tiny_store(mode);
        let want: Vec<Vec<f32>> = ["c1", "c2", "c3"]
            .iter()
            .map(|c| run(&reference.model, &case_ids(&npz, c), &split(c)).0)
            .collect();
        for (slots, cm) in [1, 6, 16, 64]
            .into_iter()
            .flat_map(|s| [(s, CacheMode::Seeded), (s, CacheMode::Lru)])
        {
            let store = cached_store_in(mode, slots, cm);
            assert_eq!(store.cache_slots(), slots);
            for (case, want) in ["c1", "c2", "c3"].iter().zip(&want) {
                let before = store.bank.stats();
                let (got, _) = run(&store.model, &case_ids(&npz, case), &split(case));
                let st = store.bank.stats();
                assert_eq!(
                    bits(&got),
                    bits(want),
                    "{mode} S={slots} {cm:?} {case}: cache != staging-only"
                );
                store.bank.with_cache_dir(|c| c.check()).unwrap();
                pool_ok(case);
                println!(
                    "{mode} S={slots} {cm:?} {case}: bitwise; uses {} hits {} reads {} occupied {}",
                    st.uses - before.uses,
                    st.hits - before.hits,
                    st.slots - before.slots,
                    store.bank.with_cache_dir(|c| c.occupied())
                );
            }
            let st = store.bank.stats();
            // one decode token wants up to 6 experts x 8 layers = 48 slots:
            // smaller LRU caches thrash (0 hits), 64 must hit
            if slots == 64 {
                assert!(st.hits > 0, "{mode} S={slots}: decode never hit the cache");
            }
            assert_eq!(store.bank.with_cache_dir(|c| c.occupied()), slots.min(96));
        }
    }
}

/// Stage P3: decode steps that overlap their reads with pass A (shared
/// expert + ready experts) give logits bitwise equal to the unsplit step,
/// for staging only and cache sizes 6 / 64 under both cache rules; the
/// split path ran with ready experts in pass A (S = 64) and without (S = 0).
#[test]
fn tiny_cache_split_bitwise() {
    let npz = Npz::open("oracle.npz");
    for mode in ["fp32_on", "bf16_on"] {
        for (slots, cm) in [0, 6, 64]
            .into_iter()
            .flat_map(|s| [(s, CacheMode::Seeded), (s, CacheMode::Lru)])
        {
            let off = cached_store_in(mode, slots, cm);
            let on = cached_store_in(mode, slots, cm);
            off.bank.set_split(false);
            on.bank.set_split(true);
            for case in ["c1", "c2", "c3"] {
                let ids = case_ids(&npz, case);
                let (a, _) = run(&off.model, &ids, &split(case));
                let (b, _) = run(&on.model, &ids, &split(case));
                assert_eq!(
                    bits(&b),
                    bits(&a),
                    "{mode} S={slots} {cm:?} {case}: split != unsplit"
                );
            }
            let (so, sn) = (off.bank.stats(), on.bank.stats());
            assert_eq!(so.split_steps, 0, "{mode} S={slots}: split off still split");
            assert_eq!(
                (so.uses, so.hits, so.slots),
                (sn.uses, sn.hits, sn.slots),
                "{mode} S={slots} {cm:?}: split changed the cache decisions"
            );
            assert!(sn.split_steps > 0, "{mode} S={slots} {cm:?}: no split step");
            if slots == 64 {
                assert!(
                    sn.split_mixed > 0,
                    "{mode} S=64 {cm:?}: pass A never had ready experts"
                );
            }
            on.bank.with_cache_dir(|c| c.check()).unwrap();
            pool_ok(mode);
            println!(
                "{mode} S={slots} {cm:?}: bitwise; split steps {} (mixed {}), reads {}, wait {} us of io {} us",
                sn.split_steps,
                sn.split_mixed,
                sn.slots,
                sn.wait_ns / 1000,
                sn.ns / 1000
            );
        }
    }
}

/// `MLX_DSV41_TEST_FORCE_EMPTY` hook: the bank is `Empty` between turn 1
/// and turn 2; turn 2 forgets every slot, re-reads, and its logits are
/// bitwise those of a store whose bank stayed (volatile, not purged).
#[test]
fn tiny_cache_forced_empty_between_turns() {
    let npz = Npz::open("oracle.npz");
    let ids = case_ids(&npz, "c2");
    let (t1, t2): (Vec<usize>, Vec<usize>) = (
        std::iter::once(12)
            .chain(std::iter::repeat_n(1, 6))
            .collect(),
        vec![1; 6],
    );
    let turns = |store: &Store, empty: bool| {
        store.bank.test_force_empty(empty);
        let mut sess = store.model.new_session();
        store.begin_turn().unwrap();
        let a = calls(store, &mut sess, &ids, &t1);
        store.end_turn().unwrap();
        let occupied = store.bank.with_cache_dir(|c| c.occupied());
        store.begin_turn().unwrap();
        let after_begin = (
            store.bank.with_cache_dir(|c| c.occupied()),
            store.bank.tags().iter().flatten().count(),
        );
        let before = store.bank.stats();
        let b = calls(store, &mut sess, &ids[18..], &t2);
        let reads = store.bank.stats().slots - before.slots;
        store.end_turn().unwrap();
        (a, b, occupied, after_begin, reads)
    };
    let reference = tiny_store("fp32_on");
    let want = calls(
        &reference,
        &mut reference.model.new_session(),
        &ids,
        &split("c2"),
    );
    let (a0, b0, occ0, begin0, reads0) = turns(&cached_store("fp32_on", 64), false);
    let (a1, b1, occ1, begin1, reads1) = turns(&cached_store("fp32_on", 64), true);
    assert_eq!(bits(&[a0.clone(), b0.clone()].concat()), bits(&want));
    assert_eq!(bits(&a1), bits(&a0));
    assert_eq!(
        bits(&b1),
        bits(&b0),
        "turn 2 after Empty != turn 2 of a kept bank"
    );
    assert!(occ0 > 0 && occ1 > 0);
    assert_eq!(begin1, (0, 0), "Empty bank kept directory entries or tags");
    assert!(
        begin0.0 > 0,
        "a volatile bank lost its directory without being purged"
    );
    assert!(
        reads1 > reads0,
        "turn 2 after Empty did not re-read ({reads1} vs {reads0})"
    );
    println!(
        "Empty: turn-2 reads {reads1} vs {reads0} kept; directory {occ1} -> {}",
        begin1.0
    );
}

/// Drop to staging-only mid-session, then back to a cache: logits bitwise
/// == staging-only throughout; the MLX pool stays <= 2 GiB after every
/// switch; a rebuild is refused while an array holds the old bank.
#[test]
fn tiny_cache_bank_switch_mid_session() {
    let npz = Npz::open("oracle.npz");
    let ids = case_ids(&npz, "c2");
    let turns: [Vec<usize>; 3] = [
        std::iter::once(12)
            .chain(std::iter::repeat_n(1, 4))
            .collect(),
        vec![1; 4],
        vec![1; 4],
    ];
    let reference = tiny_store("fp32_on");
    let want = calls(
        &reference,
        &mut reference.model.new_session(),
        &ids,
        &split("c2"),
    );
    let store = cached_store("fp32_on", 64);
    let full = store.bank.nbytes();
    let mut sess = store.model.new_session();
    let (mut got, mut at) = (Vec::new(), 0);
    for (i, sizes) in turns.iter().enumerate() {
        let n: usize = sizes.iter().sum();
        store.begin_turn().unwrap();
        got.extend(calls(&store, &mut sess, &ids[at..at + n], sizes));
        store.end_turn().unwrap();
        at += n;
        match i {
            0 => {
                store.set_cache_slots(0).unwrap();
                assert_eq!(store.cache_slots(), 0);
                assert!(store.bank.nbytes() < full);
            }
            1 => {
                store.set_cache_slots(64).unwrap();
                assert_eq!(store.bank.nbytes(), full);
            }
            _ => {}
        }
        pool_ok(&format!("switch after turn {i}"));
    }
    assert_eq!(bits(&got), bits(&want), "bank switches changed the logits");
    assert_eq!(store.switches().len(), 2);

    let held = store.bank.arrays(0).unwrap();
    let err = store.set_cache_slots(8).unwrap_err().to_string();
    assert!(err.contains("still holds the old bank"), "{err}");
    // a retry while the old buffers are still held is refused too (S16a codex)
    let err = store.set_cache_slots(8).unwrap_err().to_string();
    assert!(err.contains("still holds the old bank"), "retry: {err}");
    let mut s2 = store.model.new_session();
    assert!(
        store.model.forward(&mut s2, &ids[..1], None, None).is_err(),
        "no bank, yet a forward ran"
    );
    drop(held);
    store.set_cache_slots(8).unwrap();
    pool_ok("rebuild after the refused one");
    let mut s3 = store.model.new_session();
    let again = calls(&store, &mut s3, &ids, &split("c2"));
    assert_eq!(bits(&again), bits(&want));
}

/// A call past the admitted context (session + ids) is refused before the
/// session changes (S16a codex: the admission budgets KV / compute for the
/// context limit only).
#[test]
fn tiny_store_refuses_past_context_limit() {
    let npz = Npz::open("oracle.npz");
    let ids = case_ids(&npz, "c1");
    let mut store = cached_store("fp32_on", 0);
    let cfg_ctx = store.model.cfg.context_length;
    assert_eq!(store.model.ctx_limit, Some(cfg_ctx.min(super::CTX_LIMIT)));
    store.model.ctx_limit = Some(4);
    let mut sess = store.model.new_session();
    store
        .model
        .forward(&mut sess, &ids[..3], None, None)
        .unwrap();
    let Err(err) = store.model.forward(&mut sess, &ids[3..5], None, None) else {
        panic!("a call past the context limit ran");
    };
    let err = err.to_string();
    assert!(err.contains("context limit"), "{err}");
    assert!(
        store
            .model
            .forward_last(&mut sess, &ids[3..5], None, None)
            .is_err()
    );
    assert_eq!(sess.tokens.len(), 3, "refused call changed the session");
    store
        .model
        .forward(&mut sess, &ids[3..4], None, None)
        .unwrap();
    assert!(
        store
            .model
            .forward(&mut sess, &ids[4..5], None, None)
            .is_err()
    );
}

/// The admission's `own` is this store's dense + bank bytes only (bank
/// volatile between turns), not the process-wide MLX count: another
/// model's arrays stay `others` (S15 codex).
#[test]
fn tiny_cache_own_is_this_model_only() {
    let store = cached_store("fp32_on", 16);
    let (dense, bank) = (store.dense_bytes, store.bank.nbytes());
    let npz = Npz::open("oracle.npz");
    let ids = case_ids(&npz, "c1");
    store.begin_turn().unwrap();
    let mut sess = store.model.new_session();
    store.model.forward(&mut sess, &ids, None, None).unwrap();
    let in_turn = store.own().unwrap();
    let other = MxArray::zeros(&[16 << 20], Some(DType::Float32)).unwrap();
    other.eval();
    assert_eq!(
        store.own().unwrap(),
        in_turn,
        "another model's arrays counted as own"
    );
    // own = dense + the bank's resident pages (S16b-1: not its size)
    let resident = store.bank.resident_bytes().unwrap();
    assert!(resident > 0 && resident <= bank, "{resident} of {bank}");
    assert_eq!(in_turn.bytes(), dense + resident);
    store.end_turn().unwrap();
    assert_eq!(
        store.own().unwrap().bytes(),
        dense,
        "a volatile bank counted as own"
    );
    drop(other);
}

/// `replay` on a run's own routes gives the live bank's hits and reads.
#[test]
fn tiny_cache_replay_matches_live() {
    use crate::moe_stream::replay::{Step, replay};
    let npz = Npz::open("oracle.npz");
    for (slots, cm) in [0, 1, 6, 16, 64]
        .into_iter()
        .flat_map(|s| [(s, CacheMode::Seeded), (s, CacheMode::Lru)])
    {
        let store = cached_store_in("fp32_on", slots, cm);
        let mut steps = Vec::new();
        for case in ["c1", "c2", "c3"] {
            let (_, traces) = run(&store.model, &case_ids(&npz, case), &split(case));
            for t in &traces {
                for (layer, r) in t.routes.iter().enumerate() {
                    steps.push(Step {
                        layer,
                        rows: r.rows(),
                        ids: r.ids.clone(),
                    });
                }
            }
        }
        let cfg = &store.model.cfg;
        let mut c = crate::moe_stream::cache::Cache::lru(slots, cfg.n_layers, cfg.n_routed_experts)
            .with_mode(cm);
        let r = replay(&mut c, cfg.n_routed_experts, &steps);
        let st = store.bank.stats();
        assert_eq!(
            (r.uses, r.hits, r.reads),
            (st.uses as u64, st.hits as u64, st.slots as u64),
            "S={slots} {cm:?}: replay != live"
        );
        println!(
            "S={slots} {cm:?}: uses {} hits {} reads {} (h {:.3}, decode h {:.3})",
            r.uses,
            r.hits,
            r.reads,
            r.h(),
            r.decode_h()
        );
    }
}

fn dsv41_shards() -> Vec<PathBuf> {
    let dir = PathBuf::from(std::env::var("DSV41_DIR").expect("DSV41_DIR"));
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with("-of-00009.gguf"))
        .collect();
    v.sort();
    assert_eq!(v.len(), 9, "{v:?}");
    v
}

/// Real file: open 9 shards, load dense, run layers 0-2 for 4 tokens.
#[test]
#[ignore = "needs DSV41_DIR with all 9 real DeepSeek-V4.1 shards (~18 GiB, ~10.5 GB read)"]
fn real_load_smoke() {
    const GIB: f64 = (1u64 << 30) as f64;
    set_cache_limit(POOL_LIMIT as f64).unwrap();
    let paths = dsv41_shards();
    let t0 = Instant::now();
    let store = Store::open(&paths, DType::BFloat16, FakeQuant::ON).unwrap();
    let load_s = t0.elapsed().as_secs_f64();
    let held = (store.dense_bytes + store.bank.nbytes()) as f64;
    let active = get_active_memory();
    println!(
        "load {load_s:.1} s: dense read {} B, dense arrays {:.3} GiB, bank {:.3} GiB, active {:.3} GiB, cache {:.3} GiB",
        store.dense_read,
        store.dense_bytes as f64 / GIB,
        store.bank.nbytes() as f64 / GIB,
        active / GIB,
        get_cache_memory() / GIB
    );
    // shard 9 data minus the 40 vision-only `exp_probs_b_vl` biases (384 f32)
    assert_eq!(store.dense_read, 9_165_865_408 - 40 * 384 * 4);
    assert!(
        (active - held).abs() <= 0.02 * held,
        "active {active} vs held {held}"
    );

    let ids = [0u32, 6_000, 20_000, 100_000];
    let mut trace = Trace {
        keep_out: true,
        ..Default::default()
    };
    let max_cache = std::cell::Cell::new(0f64);
    let cancel = |l: usize| {
        max_cache.set(max_cache.get().max(get_cache_memory()));
        l == 3
    };
    let t1 = Instant::now();
    let mut sess = store.model.new_session();
    let Err(err) = store
        .model
        .forward(&mut sess, &ids, Some(&cancel), Some(&mut trace))
    else {
        panic!("not cancelled before layer 3");
    };
    println!("layers 0-2 in {:.2} s ({err})", t1.elapsed().as_secs_f64());
    assert!(sess.tokens.is_empty());
    assert_eq!((trace.out.len(), trace.routes.len()), (3, 3));
    for (l, o) in trace.out.iter().enumerate() {
        let v = read(&o.astype(DType::Float32).unwrap());
        assert!(v.iter().all(|x| x.is_finite()), "layer {l}: NaN/Inf");
    }
    let unions: Vec<BTreeSet<u32>> = trace
        .routes
        .iter()
        .map(|r| r.ids.iter().copied().collect())
        .collect();
    let st = store.bank.stats();
    let tagged: BTreeSet<u32> = store
        .bank
        .tags()
        .iter()
        .flatten()
        .filter(|t| t.0 == 2)
        .map(|t| t.1)
        .collect();
    println!(
        "unions {:?}; slots read {} ({} B in {:.2} s); max cache {:.3} GiB; peak active {:.3} GiB",
        unions.iter().map(BTreeSet::len).collect::<Vec<_>>(),
        st.slots,
        st.bytes,
        st.ns as f64 / 1e9,
        max_cache.get() / GIB,
        crate::array::get_peak_memory() / GIB
    );
    assert_eq!(tagged, unions[2]);
    assert_eq!(st.slots, unions.iter().map(BTreeSet::len).sum::<usize>());
    assert!(max_cache.get() <= 2.0 * GIB && get_cache_memory() <= 2.0 * GIB);

    let vm = std::process::Command::new("/usr/bin/vmmap")
        .arg(std::process::id().to_string())
        .output()
        .unwrap();
    let vm = String::from_utf8_lossy(&vm.stdout);
    assert!(vm.len() > 1000, "vmmap gave no output");
    assert!(!vm.contains("00008-of-00009"), "shard 8 is mapped");
    println!("vmmap: {} bytes, no mapping of shard 8", vm.len());

    store.bank.skip_fill.store(true, Ordering::Relaxed);
    let mut s2 = store.model.new_session();
    let stale = catch_unwind(AssertUnwindSafe(|| {
        store.model.forward(&mut s2, &ids[..1], None, None)
    }));
    assert!(stale.is_err(), "stale gather did not panic");
    println!("stale gather panicked as expected");
}

/// `CachePlan::Admit`: the re-admission at a turn start runs with the bank
/// non-volatile and counts its resident pages as own (S16b-1: a bank the GPU
/// used stays wired while volatile, so `host_used` keeps it); a forced
/// `Empty` between turns keeps the admission (no switch) and clears the
/// directory.
#[test]
fn tiny_cache_admit_turns() {
    let npz = Npz::open("oracle.npz");
    let ids = case_ids(&npz, "c1");
    let (act, fq) = mode_of("fp32_on");
    let limits = crate::moe_stream::budget::Limits {
        cache_cap: Some(1 << 30),
        ..Default::default()
    };
    let store = Store::open_dense(
        &[fixture("tiny-q8_0.gguf")],
        act,
        fq,
        CachePlan::Admit(limits),
        DenseMode::Exact,
    )
    .unwrap();
    assert!(store.cache_slots() > 0, "{:?}", store.admission());
    let mut sess = store.model.new_session();
    store.bank.test_force_empty(true);
    store.begin_turn().unwrap();
    store
        .model
        .forward(&mut sess, &ids[..4], None, None)
        .unwrap();
    store.end_turn().unwrap();
    store.bank.test_force_empty(false);
    store.begin_turn().unwrap();
    assert!(store.bank.in_turn());
    assert_eq!(store.bank.with_cache_dir(|c| c.occupied()), 0);
    assert!(store.switches().is_empty(), "{:?}", store.switches());
    store
        .model
        .forward(&mut sess, &ids[4..5], None, None)
        .unwrap();
    let own = store.own().unwrap();
    assert_eq!(
        own.bytes(),
        store.dense_bytes + store.bank.resident_bytes().unwrap()
    );
    store.end_turn().unwrap();
}

/// D-41 (stage P0) on the tiny file: `F32` == `Exact` bitwise (fp32 fq
/// on/off, bf16, c1-c3); `Q8` and `Mixed` within G9.1's fq-off bound of
/// the oracle (1e-4) and not bitwise `Exact` (the quantized matmul runs);
/// fq on is printed only (a 1e-6 change flips near-edge fp8 codes, S7-5:
/// Mixed c2 2.7e-2); resident dense bytes == the scan's prediction.
#[test]
fn tiny_dense_modes() {
    let npz = Npz::open("oracle.npz");
    for mode in ["fp32_on", "fp32_off", "bf16_on"] {
        let (exact, f32s) = (
            tiny_dense(mode, DenseMode::Exact),
            tiny_dense(mode, DenseMode::F32),
        );
        for case in ["c1", "c2", "c3"] {
            let ids = case_ids(&npz, case);
            let (a, _) = run(&exact.model, &ids, &split(case));
            let (b, _) = run(&f32s.model, &ids, &split(case));
            assert_eq!(bits(&a), bits(&b), "{case} {mode}: F32 != Exact");
        }
    }
    for dense in [DenseMode::Q8, DenseMode::Mixed] {
        for (mode, tol) in [("fp32_off", 1e-4f32), ("fp32_on", f32::INFINITY)] {
            let (st, exact) = (tiny_dense(mode, dense), tiny_dense(mode, DenseMode::Exact));
            let mut differs = false;
            for case in ["c1", "c2", "c3"] {
                let ids = case_ids(&npz, case);
                let (got, _) = run(&st.model, &ids, &split(case));
                let (ex, _) = run(&exact.model, &ids, &split(case));
                differs |= bits(&got) != bits(&ex);
                let want = &npz.get(&format!("{case}.{mode}.logits")).1;
                let d = got
                    .iter()
                    .zip(want)
                    .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                println!("{dense:?} {case} {mode}: max|dlogit| vs oracle {d:e} (gate {tol:e})");
                assert!(d < tol, "{dense:?} {case} {mode}: max|dlogit| {d:e}");
            }
            assert!(
                differs,
                "{dense:?} {mode}: bitwise Exact, quantized matmul not used"
            );
        }
    }
    for dense in [
        DenseMode::Exact,
        DenseMode::F32,
        DenseMode::Q8,
        DenseMode::Mixed,
    ] {
        let st = tiny_dense("fp32_on", dense);
        let size = model_size(&[fixture("tiny-q8_0.gguf")], dense).unwrap();
        println!("{dense:?}: dense resident {} B", st.dense_bytes);
        assert_eq!(st.dense_bytes, size.dense, "{dense:?}");
    }
}

/// `Dense::Qmm`: `mm_t` == x @ f32()^T within f32 accumulation noise, the
/// f32 form == the exact dequantization bitwise, grouped == per group.
#[test]
fn tiny_qmm_dense_ops() {
    let g = TinyGguf::open();
    let name = "blk.0.attn_output_a.weight";
    let (info, raw) = g.raw(name);
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor).unwrap();
    decode_affine_band_into(info.tensor_type, &raw, 0, ptrs.rows, ptrs.k, &ptrs).unwrap();
    let (w, scales) = (dst.weight, dst.scales.unwrap());
    let s = scales.astype(DType::Float32).unwrap();
    let b = s.mul_scalar(-128.0).unwrap();
    let q8 = Dense::Q8 {
        w: w.clone(),
        scales: scales.clone(),
    };
    let qmm = Dense::Qmm {
        w,
        scales: s,
        biases: b,
    };
    let (f_ex, f_q) = (q8.f32().unwrap(), qmm.f32().unwrap());
    assert_eq!(
        bits(&read(&f_ex)),
        bits(&read(&f_q)),
        "Qmm f32 form != exact dequant"
    );
    let (rows, k) = (qmm.shape_at(0).unwrap(), qmm.shape_at(1).unwrap());
    let xs: Vec<f32> = (0..3 * k)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let x = MxArray::from_float32(&xs, &[3, k]).unwrap();
    let want = read(&q8.mm_t(&x).unwrap());
    let got = read(&qmm.mm_t(&x).unwrap());
    let (num, den) = got.iter().zip(&want).fold((0f64, 0f64), |(n, d), (a, b)| {
        (n + f64::from(a - b).powi(2), d + f64::from(*b).powi(2))
    });
    let rel = (num / den).sqrt();
    println!("Qmm mm_t relL2 vs f32 {rel:e} ({rows} x {k})");
    assert!(rel < 1e-5, "{rel}");
    let ng = 2;
    let xg = MxArray::from_float32(
        &xs.iter().chain(&xs).copied().collect::<Vec<_>>(),
        &[ng, 3, k],
    )
    .unwrap();
    let grouped = read(&qmm.grouped_qmm_t(&xg, ng).unwrap().unwrap());
    let per = rows / ng;
    for i in 0..ng {
        let part = read(
            &qmm.row_range(i * per, (i + 1) * per)
                .unwrap()
                .mm_t(&x)
                .unwrap(),
        );
        let n = part.len();
        assert_eq!(
            bits(&grouped[i as usize * n..(i as usize + 1) * n]),
            bits(&part),
            "group {i}"
        );
    }
    assert!(q8.grouped_qmm_t(&xg, ng).unwrap().is_none());
    // P0 codex: bf16 activations round wo_a to bf16 (reference) on every form;
    // f32 activations take the Qmm product
    use crate::models::deepseek_v41::attention::grouped_wo_a;
    let width = k;
    let lora = per;
    let bf = grouped_wo_a(&qmm, &xg, ng, lora, width, DType::BFloat16).unwrap();
    let bf_exact = grouped_wo_a(&q8, &xg, ng, lora, width, DType::BFloat16).unwrap();
    assert_eq!(
        bits(&read(&bf)),
        bits(&read(&bf_exact)),
        "bf16 wo_a rounding"
    );
    let f = grouped_wo_a(&qmm, &xg, ng, lora, width, DType::Float32).unwrap();
    assert_eq!(bits(&read(&f)), bits(&grouped), "f32 Qmm wo_a");
    // P0 codex: header-derived resident sizes are checked
    let mut huge = info.clone();
    huge.dims = vec![32, 1u64 << 59];
    for mode in [
        DenseMode::Exact,
        DenseMode::F32,
        DenseMode::Q8,
        DenseMode::Mixed,
    ] {
        assert!(mode.resident(&huge).is_err(), "{mode:?} overflow");
    }
    huge.dims = vec![32, 1u64 << 57];
    assert!(DenseMode::F32.resident(&huge).is_err(), "f32 4n overflow");
}

/// Stage P4: a bank with two staging layers and the next-layer prefetch
/// (off / pred / all) gives the one-staging-layer store's logits bitwise
/// (c1-c3, S in {0, 6, 64} x {Seeded, Lru}); a cold one-call prefill reads
/// every used (layer, expert) once plus the prefetched ones no route used,
/// never twice (a prefill seed of a staged expert is copied, not re-read).
#[test]
fn tiny_prefetch_bitwise() {
    use crate::moe_stream::bank::PrefetchMode;
    let npz = Npz::open("oracle.npz");
    let c1 = case_ids(&npz, "c1");
    for mode in ["fp32_on", "bf16_on"] {
        for (slots, cm) in [0, 6, 64]
            .into_iter()
            .flat_map(|s| [(s, CacheMode::Seeded), (s, CacheMode::Lru)])
        {
            let reference = cached_store_in(mode, slots, cm);
            let want: Vec<Vec<f32>> = ["c1", "c2", "c3"]
                .iter()
                .map(|case| run(&reference.model, &case_ids(&npz, case), &split(case)).0)
                .collect();
            for pm in [PrefetchMode::Off, PrefetchMode::Pred, PrefetchMode::All] {
                let s = cached_store_in(mode, slots, cm);
                s.bank.rebuild_staging(slots, 2).unwrap();
                s.bank.set_prefetch_mode(pm);
                assert_eq!(s.bank.staging_layers(), 2);
                let mut sess = s.model.new_session();
                let mut tr = Trace::default();
                s.model
                    .forward(&mut sess, &c1, None, Some(&mut tr))
                    .unwrap();
                let st = s.bank.stats();
                let used: usize = tr
                    .routes
                    .iter()
                    .map(|r| r.ids.iter().collect::<BTreeSet<_>>().len())
                    .sum();
                assert_eq!(
                    st.slots,
                    used + st.prefetch_unused,
                    "{mode} S={slots} {cm:?} {pm:?}: reads {} != used {used} + unused prefetch {}",
                    st.slots,
                    st.prefetch_unused
                );
                assert!(st.prefetch_unused <= st.prefetch_slots);
                assert_eq!(st.prefetch_slots > 0, pm != PrefetchMode::Off, "{pm:?}");
                for (i, case) in ["c1", "c2", "c3"].iter().enumerate() {
                    let (got, _) = run(&s.model, &case_ids(&npz, case), &split(case));
                    assert_eq!(
                        bits(&got),
                        bits(&want[i]),
                        "{mode} S={slots} {cm:?} {pm:?} {case}: 2 staging layers != 1"
                    );
                }
                s.bank.with_cache_dir(|c| c.check()).unwrap();
                assert_eq!(s.bank.reads_in_flight(), 0);
                pool_ok(mode);
                println!(
                    "{mode} S={slots} {cm:?} {pm:?}: bitwise; cold c1 reads {} = used {used} + unused {} (prefetched {}, copied {})",
                    st.slots, st.prefetch_unused, st.prefetch_slots, st.promoted
                );
            }
        }
    }
}

/// Stage P4: `replay_turns_staged` with two staging layers gives the live
/// bank's (uses, hits, reads) without prefetch. With prefetch only the uses
/// are equal: a read ahead overwrites a same-half staging copy of layer
/// `l - 2` that a later decode step could have reused, so that step takes
/// a cache victim and the cache evolves differently (not modelled).
#[test]
fn tiny_replay_two_staging_layers() {
    use crate::moe_stream::bank::PrefetchMode;
    use crate::moe_stream::replay::{Replayed, Step, replay_turns_staged};
    let npz = Npz::open("oracle.npz");
    for (slots, cm) in [0, 6, 64]
        .into_iter()
        .flat_map(|s| [(s, CacheMode::Seeded), (s, CacheMode::Lru)])
    {
        for pm in [PrefetchMode::Off, PrefetchMode::Pred, PrefetchMode::All] {
            let store = cached_store_in("fp32_on", slots, cm);
            store.bank.rebuild_staging(slots, 2).unwrap();
            store.bank.set_prefetch_mode(pm);
            let mut steps = Vec::new();
            for case in ["c1", "c2", "c3"] {
                let (_, traces) = run(&store.model, &case_ids(&npz, case), &split(case));
                for t in &traces {
                    for (layer, r) in t.routes.iter().enumerate() {
                        steps.push(Step {
                            layer,
                            rows: r.rows(),
                            ids: r.ids.clone(),
                        });
                    }
                }
            }
            let cfg = &store.model.cfg;
            let mut c =
                crate::moe_stream::cache::Cache::lru(slots, cfg.n_layers, cfg.n_routed_experts)
                    .with_mode(cm);
            let r = replay_turns_staged(&mut c, cfg.n_routed_experts, 2, &[&steps])
                .into_iter()
                .fold(Replayed::default(), Replayed::plus);
            let st = store.bank.stats();
            let net = (st.slots - st.prefetch_unused) as u64;
            assert_eq!(r.uses, st.uses as u64, "S={slots} {cm:?} {pm:?}: uses");
            if pm == PrefetchMode::Off {
                assert_eq!(
                    (r.hits, r.reads),
                    (st.hits as u64, net),
                    "S={slots} {cm:?}: replay != live"
                );
            }
            println!(
                "S={slots} {cm:?} {pm:?}: uses {} hits replay {} live {} reads replay {} live net {net} (+{} unused prefetch)",
                r.uses, r.hits, st.hits, r.reads, st.prefetch_unused
            );
        }
    }
}

/// P3 codex: a `Pending` kept past its step (its `Prepared` dropped) must
/// not let a rebuild free the slots its reads still write: the rebuild
/// waits for every read in flight.
#[test]
fn tiny_rebuild_waits_for_reads_in_flight() {
    let s = cached_store_in("fp32_on", 0, CacheMode::Seeded);
    s.bank.test_delay_reads(300);
    let (p, pending) = s.bank.prepare_split(0, &[0, 1, 2], &[], false).unwrap();
    assert!(pending.is_some() && s.bank.reads_in_flight() == 1);
    drop(p);
    let t0 = Instant::now();
    s.bank.rebuild(0).unwrap();
    let waited = t0.elapsed();
    assert_eq!(s.bank.reads_in_flight(), 0, "rebuild left a read in flight");
    assert!(
        waited.as_millis() >= 250,
        "rebuild returned after {waited:?}, before the delayed reads"
    );
    drop(pending);
    s.bank.test_delay_reads(0);
    let npz = Npz::open("oracle.npz");
    let reference = cached_store_in("fp32_on", 0, CacheMode::Seeded);
    let case = "c2";
    let (want, _) = run(&reference.model, &case_ids(&npz, case), &split(case));
    let (got, _) = run(&s.model, &case_ids(&npz, case), &split(case));
    assert_eq!(bits(&got), bits(&want));
}
