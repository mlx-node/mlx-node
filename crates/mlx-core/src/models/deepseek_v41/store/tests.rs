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
    let (act, fq) = mode_of(mode);
    Store::open(&[fixture("tiny-q8_0.gguf")], act, fq).unwrap()
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
    let (act, fq) = mode_of(mode);
    Store::open_with(
        &[fixture("tiny-q8_0.gguf")],
        act,
        fq,
        CachePlan::Slots(slots),
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
        for slots in [1, 6, 16, 64] {
            let store = cached_store(mode, slots);
            assert_eq!(store.cache_slots(), slots);
            for (case, want) in ["c1", "c2", "c3"].iter().zip(&want) {
                let before = store.bank.stats();
                let (got, _) = run(&store.model, &case_ids(&npz, case), &split(case));
                let st = store.bank.stats();
                assert_eq!(
                    bits(&got),
                    bits(want),
                    "{mode} S={slots} {case}: cache != staging-only"
                );
                store.bank.with_cache_dir(|c| c.check()).unwrap();
                pool_ok(case);
                println!(
                    "{mode} S={slots} {case}: bitwise; uses {} hits {} reads {} occupied {}",
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

/// The admission's `own` is this store's dense + bank bytes only (bank
/// volatile between turns), not the process-wide MLX count: another
/// model's arrays stay `others` (S15 codex).
#[test]
fn tiny_cache_own_is_this_model_only() {
    let store = cached_store("fp32_on", 16);
    let (dense, bank) = (store.dense_bytes, store.bank.nbytes());
    store.begin_turn().unwrap();
    let in_turn = store.own();
    let other = MxArray::zeros(&[16 << 20], Some(DType::Float32)).unwrap();
    other.eval();
    assert_eq!(
        store.own(),
        in_turn,
        "another model's arrays counted as own"
    );
    assert_eq!(in_turn.bytes(), dense + bank);
    store.end_turn().unwrap();
    assert_eq!(store.own().bytes(), dense, "a volatile bank counted as own");
    drop(other);
}

/// `replay` on a run's own routes gives the live bank's hits and reads.
#[test]
fn tiny_cache_replay_matches_live() {
    use crate::moe_stream::replay::{Step, replay};
    let npz = Npz::open("oracle.npz");
    for slots in [0, 1, 6, 16, 64] {
        let store = cached_store("fp32_on", slots);
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
        let mut c = crate::moe_stream::cache::Cache::lru(slots, cfg.n_layers, cfg.n_routed_experts);
        let r = replay(&mut c, cfg.n_routed_experts, &steps);
        let st = store.bank.stats();
        assert_eq!(
            (r.uses, r.hits, r.reads),
            (st.uses as u64, st.hits as u64, st.slots as u64),
            "S={slots}: replay != live"
        );
        println!(
            "S={slots}: uses {} hits {} reads {} (h {:.3}, decode h {:.3})",
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
