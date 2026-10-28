// Real cache invariance on the 9 shards: the p32k8-en prompt greedy in
// three turns (prompt + 7 steps, 8 steps, 32 steps = 48 calls) at expert
// cache 0 (baseline hashes), at a fixed cache size, and at the auto
// admission (with the bank forced `Empty` between turn 1 and 2). Every call's
// last-row logits hash must equal the cache-0 run's (bitwise); the live hit /
// read counts must equal a replay of the run's own route trace. Run with
// `MLX_DSV41_EXPERT_CACHE_GB`, `DSV41_ROUTE_TRACE`, `DSV41_DENSE`,
// `DSV41_GOLDEN`, `DSV41_DIR`; `DSV41_CACHE_OUT` gets `<name>.json`.

use super::generate::generate_probed;
use super::store::{CachePlan, DenseMode, model_size};
use crate::moe_stream::budget::{CACHE_ENV, Limits, Own, admit, working_set};

/// Steps per turn after the first call of the turn (turn 1's first call is
/// the prompt, later turns start with the previous turn's last pick).
const CACHE_TURNS: [usize; 3] = [7, 7, 31];
const GIB_F64: f64 = (1u64 << 30) as f64;

fn test_env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

fn real_store(plan: CachePlan) -> Store {
    set_cache_limit(POOL_LIMIT as f64).unwrap();
    let t0 = std::time::Instant::now();
    let s = Store::open_dense(&real_shards(), DType::Float32, FakeQuant::ON, plan, real_dense())
        .unwrap();
    println!(
        "load {:.1} s; dense mode {:?}; dense read {} B, resident {} B; bank {:.3} GiB ({} cache slots) allocated in {:.3} s; admission {}",
        t0.elapsed().as_secs_f64(),
        s.dense_mode,
        s.dense_read,
        s.dense_bytes,
        s.bank.nbytes() as f64 / GIB_F64,
        s.cache_slots(),
        s.bank_alloc_s,
        s.admission()
            .map_or_else(|| "none".to_owned(), |a| a.to_string())
    );
    s
}

/// First GPU touch of every bank page: a full reduction over each
/// projection's codes and scales. Returns seconds.
fn touch_bank(store: &Store) -> f64 {
    let t0 = std::time::Instant::now();
    for p in 0..3 {
        let (w, s) = store.bank.arrays(p).unwrap();
        for a in [w, s] {
            let r = a.sum(None, None).unwrap();
            r.eval();
        }
    }
    crate::array::synchronize();
    t0.elapsed().as_secs_f64()
}

/// The decode terms of one or more all-decode turns.
fn decode_terms_json(c: &Counters, wall: f64, n: usize) -> Value {
    let n = n.max(1) as f64;
    let (io, eng, router) = (
        c.fill_ns as f64 * 1e-9,
        c.engram_ns as f64 * 1e-9,
        c.router_ns as f64 * 1e-9,
    );
    serde_json::json!({
        "tokens": n,
        "t_tok_s": wall / n,
        "t_c_s": (wall - io - eng) / n,
        "t_io_s": io / n,
        "t_io_wait_s": c.fill_wait_ns as f64 * 1e-9 / n,
        "t_engram_s": eng / n,
        "t_router_s": router / n,
        "bw_bytes_s": c.fill_bytes as f64 / io.max(1e-9),
        "miss_bytes_per_token": c.fill_bytes as f64 / n,
        "hits": c.hits,
        "uses": c.uses,
        "hit_rate": c.hits as f64 / c.uses.max(1) as f64,
        "tok_s": n / wall,
    })
}

fn sum_f64(a: &Counters, b: &Counters) -> Counters {
    Counters {
        router_ns: a.router_ns + b.router_ns,
        engram_ns: a.engram_ns + b.engram_ns,
        engram_bytes: a.engram_bytes + b.engram_bytes,
        fill_ns: a.fill_ns + b.fill_ns,
        fill_bytes: a.fill_bytes + b.fill_bytes,
        fill_slots: a.fill_slots + b.fill_slots,
        fill_wait_ns: a.fill_wait_ns + b.fill_wait_ns,
        uses: a.uses + b.uses,
        hits: a.hits + b.hits,
        prefetch_unused: a.prefetch_unused + b.prefetch_unused,
        staging_layers: b.staging_layers,
    }
}

fn median_f64(v: &[f64]) -> f64 {
    let mut d = v.to_vec();
    d.sort_by(f64::total_cmp);
    d.get(d.len() / 2).copied().unwrap_or(0.0)
}

/// One run: three turns on one session; `empty_after_first`: the bank is
/// `Empty` (not volatile) between turn 1 and 2.
fn real_cache_inv(name: &str, plan: CachePlan, turns: &[usize], empty_after_first: bool) {
    let gd = gold("p32k8-en");
    let trace_path =
        PathBuf::from(test_env("DSV41_ROUTE_TRACE").expect("DSV41_ROUTE_TRACE"));
    let (_, swap0) = super::stats::host_pressure();
    let sampler = FootprintMax::start();
    let store = real_store(plan);
    let mut fails: Vec<String> = Vec::new();
    let touch_s = touch_bank(&store);
    println!(
        "first GPU touch of the bank ({:.3} GiB): {touch_s:.3} s; bank alloc {:.3} s{}",
        store.bank.nbytes() as f64 / GIB_F64,
        store.bank_alloc_s,
        if touch_s > 5.0 || store.bank_alloc_s > 5.0 {
            " -> > 5 s: split the bank into layer-group slabs"
        } else {
            ""
        }
    );
    let vocab = store.model.cfg.vocab_size;
    let n_layers = store.model.cfg.n_layers;
    let n_experts = store.model.cfg.n_routed_experts;
    let stats0 = store.bank.stats();
    let max_pool = std::cell::Cell::new(0f64);
    let probe = |_l: usize| {
        max_pool.set(max_pool.get().max(get_cache_memory()));
        false
    };
    let pool_check = |what: &str, fails: &mut Vec<String>| {
        let c = get_cache_memory();
        if c > 2.0 * GIB_F64 {
            fails.push(format!("MLX cache {:.2} GiB {what}", c / GIB_F64));
        }
    };
    let mut sess = store.model.new_session();
    let mut feed = gd.prompt.clone();
    let (mut ids, mut hashes, mut bad_calls) = (Vec::new(), Vec::new(), Vec::new());
    let mut turns_doc = Vec::new();
    let mut turn_steps: Vec<Vec<crate::moe_stream::replay::Step>> = Vec::new();
    let mut header: Option<String> = None;
    let mut route_lines: Vec<String> = Vec::new();
    let (mut dec_c, mut dec_wall, mut dec_n) = (Counters::default(), 0.0, 0usize);
    let mut turn3: Option<Value> = None;
    // bank (uses, hits, slots read) after every call
    let mut per_call: Vec<(u64, u64, u64)> = Vec::new();
    for (ti, &steps) in turns.iter().enumerate() {
        store.bank.test_force_empty(empty_after_first && ti == 0);
        let switches_before = store.switches().len();
        store.begin_turn().unwrap();
        pool_check(&format!("after turn {} begin", ti + 1), &mut fails);
        let after_begin = (
            store.bank.with_cache_dir(|c| c.occupied()),
            store.bank.tags().iter().flatten().count(),
        );
        if empty_after_first && ti == 1 && after_begin != (0, 0) {
            fails.push(format!(
                "turn 2 after Empty starts with {} directory entries and {} tags",
                after_begin.0, after_begin.1
            ));
        }
        let t0 = std::time::Instant::now();
        let first_call = hashes.len();
        let g: Generated = generate_probed(
            &store,
            &mut sess,
            &feed,
            None,
            steps,
            None,
            false,
            Some(&probe),
            &mut |c| {
                let v = read(c.logits);
                let row = &v[v.len() - vocab..];
                hashes.push(hex_sha(row));
                let b = store.bank.stats();
                per_call.push((b.uses as u64, b.hits as u64, b.slots as u64));
                if !v.iter().all(|x| x.is_finite()) {
                    bad_calls.push(first_call + c.index);
                }
                Ok(())
            },
        )
        .unwrap();
        let wall = t0.elapsed().as_secs_f64();
        pool_check(&format!("after turn {}", ti + 1), &mut fails);
        let n_calls = g.argmax.len();
        ids.extend_from_slice(&g.argmax);
        feed = vec![*g.argmax.last().unwrap()];
        let text = std::fs::read_to_string(&trace_path).unwrap();
        let t = crate::moe_stream::replay::read_trace(text.as_bytes()).unwrap();
        assert_eq!(t.steps.len(), n_calls * n_layers, "turn {} trace", ti + 1);
        turn_steps.push(t.steps);
        let mut lines = text.lines();
        let h = lines.next().unwrap().to_owned();
        header.get_or_insert(h);
        route_lines.extend(lines.map(str::to_owned));
        let mut all_s = vec![g.ttft_s];
        all_s.extend_from_slice(&g.decode_s);
        let terms = decode_terms_json(&g.counters, wall, n_calls);
        if ti > 0 {
            dec_c = sum_f64(&dec_c, &g.counters);
            dec_wall += wall;
            dec_n += n_calls;
        }
        if ti == 2 {
            turn3 = Some(terms.clone());
        }
        let switches: Vec<(usize, f64)> = store.switches()[switches_before..].to_vec();
        for (slots, s) in &switches {
            println!(
                "turn {}: bank switch to {slots} cache slots in {s:.2} s",
                ti + 1
            );
        }
        println!(
            "turn {}: {n_calls} calls in {wall:.1} s (first {:.2} s, step median {:.2} s = {:.2} tok/s); misses {} hits {} of {} uses (h {:.3}); fill {:.1} s {:.1} GB; engram {:.2} s; router wait {:.2} s; begin: {} cached, {} tags; cache {} slots",
            ti + 1,
            g.ttft_s,
            median_f64(&g.decode_s),
            1.0 / median_f64(&g.decode_s).max(1e-9),
            g.counters.fill_slots,
            g.counters.hits,
            g.counters.uses,
            g.counters.hits as f64 / g.counters.uses.max(1) as f64,
            g.counters.fill_ns as f64 * 1e-9,
            g.counters.fill_bytes as f64 / 1e9,
            g.counters.engram_ns as f64 * 1e-9,
            g.counters.router_ns as f64 * 1e-9,
            after_begin.0,
            after_begin.1,
            store.cache_slots(),
        );
        turns_doc.push(serde_json::json!({
            "turn": ti + 1,
            "calls": n_calls,
            "wall_s": wall,
            "call_s": all_s,
            "step_median_s": median_f64(&g.decode_s),
            "counters": {
                "fill_slots": g.counters.fill_slots, "fill_bytes": g.counters.fill_bytes,
                "fill_ns": g.counters.fill_ns, "engram_ns": g.counters.engram_ns,
                "engram_bytes": g.counters.engram_bytes, "router_ns": g.counters.router_ns,
                "uses": g.counters.uses, "hits": g.counters.hits,
            },
            "terms": terms,
            "begin_cached": after_begin.0,
            "begin_tags": after_begin.1,
            "cache_slots": store.cache_slots(),
            "switches": switches,
            "admission": store.admission().map(|a| a.to_string()),
            "stats": g.stats,
        }));
    }
    store.bank.test_force_empty(false);
    // the whole session's route trace (header of turn 1, routes of every turn)
    let mut all = header.unwrap();
    for l in &route_lines {
        all.push('\n');
        all.push_str(l);
    }
    all.push('\n');
    std::fs::write(&trace_path, all).unwrap();
    let n_calls = hashes.len();
    if route_lines.len() != n_calls * n_layers {
        fails.push(format!(
            "route trace {} lines for {n_calls} calls x {n_layers} layers",
            route_lines.len()
        ));
    }
    if !bad_calls.is_empty() {
        fails.push(format!("non-finite logits in calls {bad_calls:?}"));
    }
    if max_pool.get() > 2.0 * GIB_F64 {
        fails.push(format!(
            "MLX cache {:.2} GiB at a layer",
            max_pool.get() / GIB_F64
        ));
    }

    // live == replay of the run's own routes (when the bank never switched)
    let st = store.bank.stats();
    // prefetched experts no route used are reads the replay cannot see
    let live = (
        (st.uses - stats0.uses) as u64,
        (st.hits - stats0.hits) as u64,
        (st.slots - stats0.slots - (st.prefetch_unused - stats0.prefetch_unused)) as u64,
    );
    let staging = store.bank.staging_layers();
    println!(
        "staging layers {staging}, prefetch {}: read ahead {}, unused {}",
        store.bank.prefetch(),
        st.prefetch_slots - stats0.prefetch_slots,
        st.prefetch_unused - stats0.prefetch_unused
    );
    let slots = store.cache_slots();
    let replayed = if store.switches().is_empty() {
        use crate::moe_stream::replay::{Replayed, replay_turns_staged};
        let replay_turns = |c: &mut crate::moe_stream::cache::Cache,
                            n: usize,
                            t: &[&[crate::moe_stream::replay::Step]]| {
            replay_turns_staged(c, n, staging, t)
        };
        let mk = || crate::moe_stream::cache::Cache::lru(slots, n_layers, n_experts);
        let sum = |v: Vec<Replayed>| v.into_iter().fold(Replayed::default(), Replayed::plus);
        let turns_s: Vec<&[crate::moe_stream::replay::Step]> =
            turn_steps.iter().map(Vec::as_slice).collect();
        let (r1, r2) = if empty_after_first {
            // the bank forgot every slot and tag between turn 1 and 2
            (
                sum(replay_turns(&mut mk(), n_experts, &turns_s[..1])),
                sum(replay_turns(&mut mk(), n_experts, &turns_s[1..])),
            )
        } else {
            // one cache and one set of staging tags across every turn
            let mut v = replay_turns(&mut mk(), n_experts, &turns_s);
            let rest = v.split_off(1);
            (sum(v), sum(rest))
        };
        let rep = (r1.uses + r2.uses, r1.hits + r2.hits, r1.reads + r2.reads);
        // with a read ahead only the uses must match (an overwritten
        // same-half staging copy changes later decode victims; not modelled)
        let ahead = staging == 2 && store.bank.prefetch();
        if ahead && rep.0 == live.0 {
            println!("live ~ replay (read ahead on, uses equal): live {live:?}, replay {rep:?}");
        } else if rep != live {
            fails.push(format!(
                "live (uses, hits, reads) {live:?} != replay of the run's trace {rep:?}"
            ));
        } else {
            println!("live == replay: (uses, hits, reads) {live:?}");
        }
        Some(rep)
    } else {
        println!("bank switched during the run: live/replay check skipped");
        None
    };

    // hashes vs the cache-0 baseline (written by real_cache_inv_c0 only)
    let doc = serde_json::json!({
        "prompt_ids": gd.prompt,
        "turns": turns,
        "ids": ids,
        "hashes": hashes,
    });
    // non-exact dense modes keep their own baselines; F32 is gated
    // on the exact ones
    let tag = store.dense_mode.tag();
    let base_name = format!("s16-cache-inv-c0{tag}");
    let base_path = golden_root().join(format!("hashes/{base_name}.json"));
    if name == "c0" || (!tag.is_empty() && !base_path.exists()) {
        // the first 32 calls are the greedy_en calls: same bits (a
        // non-exact mode's first run becomes its cache-invariance baseline)
        let p = golden_root().join(format!("hashes/greedy-p32k8-en{tag}.json"));
        let g: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let old: Vec<String> = g["hashes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_owned())
            .collect();
        let n = old.len().min(hashes.len());
        if old[..n] != hashes[..n] {
            let first = (0..n).find(|&i| old[i] != hashes[i]).unwrap();
            fails.push(format!(
                "{name}{tag} differs from greedy_en{tag} at call {first} of {n} (logits hash)"
            ));
        } else {
            println!("{name}{tag} == greedy_en{tag} hashes for the first {n} calls (bitwise)");
        }
        store_hashes(&base_name, doc.clone(), &mut fails);
    } else {
        let base: Value = serde_json::from_slice(
            &std::fs::read(&base_path).expect("run real_cache_inv_c0 first"),
        )
        .unwrap();
        assert_eq!(base["prompt_ids"], doc["prompt_ids"]);
        // a shorter run is a prefix of c0's turns (same turn boundaries)
        let bt = base["turns"].as_array().unwrap();
        assert!(
            turns.len() <= bt.len()
                && turns
                    .iter()
                    .zip(bt)
                    .all(|(a, b)| b.as_u64() == Some(*a as u64))
        );
        let bids = base["ids"].as_array().unwrap();
        if bids.len() < ids.len() || bids[..ids.len()] != doc["ids"].as_array().unwrap()[..] {
            fails.push(format!(
                "greedy ids differ from c0: {} vs {}",
                doc["ids"], base["ids"]
            ));
        }
        let bh = base["hashes"].as_array().unwrap();
        let diff: Vec<usize> = (0..n_calls)
            .filter(|&i| bh.get(i).and_then(Value::as_str) != hashes.get(i).map(String::as_str))
            .collect();
        if diff.is_empty() {
            println!("{name}: {n_calls} logits hashes == {base_name} (bitwise)");
        } else {
            fails.push(format!(
                "{name}: logits hashes differ from c0 at calls {diff:?}"
            ));
        }
    }

    // the first 16 decode tokens (calls 1..=16; call 0 is the prompt)
    let first16 = {
        let at = |i: usize| per_call.get(i).copied().unwrap_or_default();
        let (a, b) = (at(0), at(16.min(per_call.len().saturating_sub(1))));
        let (u, h, r) = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
        serde_json::json!({
            "tokens": 16.min(per_call.len().saturating_sub(1)),
            "uses": u, "hits": h, "reads": r,
            "h": h as f64 / u.max(1) as f64,
            "byte_h": 1.0 - r as f64 / u.max(1) as f64,
        })
    };
    println!("{name}: first 16 decode tokens {first16}");
    let (_, swap1) = super::stats::host_pressure();
    let swap_d = swap1.zip(swap0).map_or(0, |(a, b)| a as i64 - b as i64);
    let fp_peak = sampler.finish();
    let decode = decode_terms_json(&dec_c, dec_wall, dec_n);
    let out = serde_json::json!({
        "name": name,
        "dense_mode": format!("{:?}", store.dense_mode),
        "plan": format!("{:?}", store.plan()),
        "cache_slots": store.cache_slots(),
        "bank_bytes": store.bank.nbytes(),
        "dense_bytes": store.dense_bytes,
        "bank_alloc_s": store.bank_alloc_s,
        "bank_touch_s": touch_s,
        "admission": store.admission().map(|a| a.to_string()),
        "admission_cache_gib": store.admission().map(|a| a.cache_bytes() as f64 / GIB_F64),
        "switches": store.switches(),
        "turns": turns_doc,
        "decode_terms": decode,
        "first16": first16,
        "turn3_terms": turn3,
        "live": {"uses": live.0, "hits": live.1, "reads": live.2},
        "replay": replayed.map(|r| serde_json::json!({"uses": r.0, "hits": r.1, "reads": r.2})),
        "max_pool_gib": max_pool.get() / GIB_F64,
        "footprint_peak_gib": fp_peak as f64 / GIB_F64,
        "swap_delta": swap_d,
        "ids": ids,
        "fails": fails,
    });
    println!(
        "{name}: decode (turns 2-3) {:.3} tok/s, t_c {:.1} ms, io {:.1} ms, engram {:.1} ms, h {:.3}; footprint peak {:.2} GiB; max MLX cache {:.3} GiB; swap delta {swap_d}",
        decode["tok_s"].as_f64().unwrap(),
        decode["t_c_s"].as_f64().unwrap() * 1e3,
        decode["t_io_s"].as_f64().unwrap() * 1e3,
        decode["t_engram_s"].as_f64().unwrap() * 1e3,
        decode["hit_rate"].as_f64().unwrap(),
        fp_peak as f64 / GIB_F64,
        max_pool.get() / GIB_F64,
    );
    println!("s16 json: {out}");
    if let Some(dir) = test_env("DSV41_CACHE_OUT") {
        let p = PathBuf::from(dir).join(format!("s16-{name}.json"));
        std::fs::write(&p, serde_json::to_vec_pretty(&out).unwrap()).unwrap();
    }
    finish(&format!("real_cache_inv_{name}"), &fails);
}

/// `MLX_DSV41_EXPERT_CACHE_GB` must be `want` (the cache size is
/// explicit per run).
fn cache_gb_env(want: &str) {
    assert_eq!(test_env(CACHE_ENV).as_deref(), Some(want), "{CACHE_ENV}");
}

#[test]
#[ignore = "real 9 shards + oracle goldens; MLX_DSV41_EXPERT_CACHE_GB=0; ~18 GiB"]
fn real_cache_inv_c0() {
    cache_gb_env("0");
    real_cache_inv("c0", CachePlan::Off, &CACHE_TURNS, false);
}

#[test]
#[ignore = "real 9 shards + c0 hashes; MLX_DSV41_EXPERT_CACHE_GB=4; ~22 GiB"]
fn real_cache_inv_c4() {
    cache_gb_env("4");
    let slots = (4.0 * GIB_F64 / EXPERT_BYTES as f64) as usize;
    // 16 calls (plan): at 228 slots < 240 experts per token LRU thrashes, so
    // every use is a read
    real_cache_inv(
        "c4",
        CachePlan::Slots {
            cache: slots,
            staging: 1,
        },
        &CACHE_TURNS[..2],
        false,
    );
}

/// Admission at load and every turn (`MLX_DSV41_EXPERT_CACHE_GB` = the
/// dry run's N_auto as a cap, so the guard cap holds), bank `Empty` between
/// turn 1 and 2.
#[test]
#[ignore = "real 9 shards + c0 hashes; MLX_DSV41_EXPERT_CACHE_GB=<N_auto>; ~18 GiB + N_auto"]
fn real_cache_inv_auto() {
    assert!(test_env(CACHE_ENV).is_some(), "{CACHE_ENV} = N_auto (cap)");
    assert!(test_env("DSV41_ADMIT_ONLY").is_none());
    let limits = Limits::from_env().unwrap();
    real_cache_inv("auto", CachePlan::Admit(limits), &CACHE_TURNS, true);
}

/// `real_cache_inv_auto` with the bank kept (volatile, not emptied)
/// between turns, so the first 16 decode tokens see what turn 1's prefill
/// seeded; hashes == c0 (bitwise) as in every cache run.
#[test]
#[ignore = "real 9 shards + c0 hashes; MLX_DSV41_EXPERT_CACHE_GB=<N_auto>; ~18 GiB + N_auto"]
fn real_cache_keep_auto() {
    assert!(test_env(CACHE_ENV).is_some(), "{CACHE_ENV} = N_auto (cap)");
    assert!(test_env("DSV41_ADMIT_ONLY").is_none());
    let limits = Limits::from_env().unwrap();
    real_cache_inv("keep", CachePlan::Admit(limits), &CACHE_TURNS, false);
}

/// The admission this host gives right now, from the shard headers alone:
/// every term and the cache size N_auto (GiB). Allocates nothing.
#[test]
#[ignore = "real shard headers; DSV41_ADMIT_ONLY=1; < 1 GiB"]
fn real_admit_dry_run() {
    assert_eq!(test_env("DSV41_ADMIT_ONLY").as_deref(), Some("1"));
    let mode = real_dense();
    let size = model_size(&real_shards(), mode).unwrap();
    println!("dense mode {mode:?}");
    let limits = Limits::from_env().unwrap();
    let h = crate::utils::host::host_memory().expect("host memory");
    let ws = working_set();
    let a = admit(&size, &limits, &h, &Own::default(), ws).unwrap();
    let n_auto = a.cache_bytes() as f64 / GIB_F64;
    println!("model size {size:?}; limits {limits:?}");
    println!("admission: {a}");
    println!(
        "N_auto_gib={n_auto:.3} slots={} fixed_gib={:.3} wired_gib={:.3}",
        a.cache_slots,
        a.fixed as f64 / GIB_F64,
        a.wired_limit() as f64 / GIB_F64
    );
    let t = &a.terms;
    let doc = serde_json::json!({
        "dense_mode": format!("{mode:?}"),
        "n_auto_gib": n_auto,
        "cache_slots": a.cache_slots,
        "slot_bytes": a.slot,
        "physical_gib": a.physical as f64 / GIB_F64,
        "host_used_gib": a.host_used as f64 / GIB_F64,
        "others_gib": a.others as f64 / GIB_F64,
        "reserve_gib": a.reserve as f64 / GIB_F64,
        "live_gib": a.live as f64 / GIB_F64,
        "ws_gib": a.ws as f64 / GIB_F64,
        "fixed_gib": a.fixed as f64 / GIB_F64,
        "staging_layers": a.staging_layers,
        "terms_gib": {
            "dense": t.dense as f64 / GIB_F64, "staging_layer": t.staging_layer as f64 / GIB_F64,
            "kv": t.kv as f64 / GIB_F64, "compute": t.compute as f64 / GIB_F64,
            "io": t.io as f64 / GIB_F64, "driver": t.driver as f64 / GIB_F64, "pool": t.pool as f64 / GIB_F64,
        },
        "bound": format!("{:?}", a.bound),
        "wired_gib": a.wired_limit() as f64 / GIB_F64,
        "display": a.to_string(),
    });
    if let Some(dir) = test_env("DSV41_CACHE_OUT") {
        std::fs::write(
            PathBuf::from(dir).join("s16-admit.json"),
            serde_json::to_vec_pretty(&doc).unwrap(),
        )
        .unwrap();
    }
    let active = crate::array::get_active_memory();
    assert!(active < GIB_F64, "dry run allocated {active} B");
}

/// Timing: split (reads overlap pass A) vs unsplit decode of one
/// token at a fixed state, all-hit (its experts cached), mixed (~30 % of
/// the (layer, expert) pairs forgotten first, about the auto budget's decode
/// h) and all-miss (every tag and cache entry forgotten first), interleaved
/// per rep (ABBA). After
/// the p32k8-en prompt at the auto admission, the session is cloned and
/// the same greedy token decoded again each time: every logits row must be
/// bitwise equal (split on/off, hit/miss). `DSV41_SPLIT_REPS` (default 6).
#[test]
#[ignore = "real 9 shards + oracle goldens; MLX_DSV41_EXPERT_CACHE_GB=<N_auto>; ~18 GiB + N_auto, ~4.5 GB read per all-miss token"]
fn real_split_timing() {
    assert!(test_env(CACHE_ENV).is_some(), "{CACHE_ENV} = N_auto (cap)");
    let reps: usize = test_env("DSV41_SPLIT_REPS").map_or(6, |v| v.parse().unwrap());
    let gd = gold("p32k8-en");
    let limits = Limits::from_env().unwrap();
    let store = real_store(CachePlan::Admit(limits));
    let vocab = store.model.cfg.vocab_size;
    store.begin_turn().unwrap();
    let mut sess = store.model.new_session();
    let logits = store
        .model
        .forward(&mut sess, &gd.prompt, None, None)
        .unwrap();
    let v = read(&logits);
    let tok = super::generate::argmax(&v[v.len() - vocab..]);
    let base = sess.clone();
    let step = |on: bool| -> (f64, String, crate::moe_stream::bank::BankStats) {
        store.bank.set_split(on);
        let mut s = base.clone();
        let b0 = store.bank.stats();
        let t0 = std::time::Instant::now();
        let l = store.model.forward(&mut s, &[tok], None, None).unwrap();
        let row = read(&l);
        let dt = t0.elapsed().as_secs_f64();
        let b = store.bank.stats();
        let d = crate::moe_stream::bank::BankStats {
            uses: b.uses - b0.uses,
            hits: b.hits - b0.hits,
            slots: b.slots - b0.slots,
            bytes: b.bytes - b0.bytes,
            ns: b.ns - b0.ns,
            wait_ns: b.wait_ns - b0.wait_ns,
            split_steps: b.split_steps - b0.split_steps,
            split_mixed: b.split_mixed - b0.split_mixed,
            ..Default::default()
        };
        (dt, hex_sha(&row[row.len() - vocab..]), d)
    };
    // warm: every expert of this token read into the cache (none left to a
    // staging copy, so all-hit means cache hits only)
    store.bank.test_forget_all();
    let (_, want, _) = step(false);
    let mut fails = Vec::new();
    let mut rows: Vec<Value> = Vec::new();
    let mut t: std::collections::BTreeMap<(&str, bool), Vec<f64>> = Default::default();
    for r in 0..reps {
        let order = if r % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for kind in ["hit", "mixed", "miss"] {
            for on in order {
                match kind {
                    "miss" => store.bank.test_forget_all(),
                    "mixed" => store
                        .bank
                        .test_forget_where(|l, e| (l * 131 + e as usize * 7) % 10 < 3),
                    _ => {}
                }
                let (dt, h, d) = step(on);
                if h != want {
                    fails.push(format!("rep {r} {kind} split={on}: logits {h} != {want}"));
                }
                let ok = match kind {
                    "hit" => d.slots == 0 && d.hits == d.uses,
                    "mixed" => d.slots > 0 && d.hits > 0 && d.hits + d.slots <= d.uses,
                    _ => d.slots == d.uses && d.hits == 0,
                };
                if !ok {
                    fails.push(format!(
                        "rep {r} {kind} split={on}: uses {} hits {} reads {} (not all-{kind})",
                        d.uses, d.hits, d.slots
                    ));
                }
                if on != (d.split_steps > 0) && kind != "hit" {
                    fails.push(format!(
                        "rep {r} miss split={on}: {} split steps",
                        d.split_steps
                    ));
                }
                t.entry((kind, on)).or_default().push(dt);
                rows.push(serde_json::json!({
                    "rep": r, "kind": kind, "split": on, "s": dt, "uses": d.uses, "hits": d.hits,
                    "reads": d.slots, "bytes": d.bytes, "io_ms": d.ns as f64 * 1e-6,
                    "wait_ms": d.wait_ns as f64 * 1e-6, "split_steps": d.split_steps,
                    "split_mixed": d.split_mixed,
                }));
                println!(
                    "p3 rep {r} {kind:4} split={:<5} {:.1} ms ({:.2} tok/s) reads {} io {:.1} ms wait {:.1} ms split steps {}",
                    on,
                    dt * 1e3,
                    1.0 / dt,
                    d.slots,
                    d.ns as f64 * 1e-6,
                    d.wait_ns as f64 * 1e-6,
                    d.split_steps
                );
            }
        }
    }
    store.end_turn().unwrap();
    let med = |k: &str, on: bool| median_f64(&t[&(k, on)]);
    let mut summary = serde_json::Map::new();
    for kind in ["hit", "mixed", "miss"] {
        let (off, on) = (med(kind, false), med(kind, true));
        println!(
            "p3 {kind}: median off {:.1} ms ({:.3} tok/s) on {:.1} ms ({:.3} tok/s) -> {:+.2} %",
            off * 1e3,
            1.0 / off,
            on * 1e3,
            1.0 / on,
            (off / on - 1.0) * 100.0
        );
        summary.insert(
            kind.into(),
            serde_json::json!({"off_s": off, "on_s": on, "off_tok_s": 1.0 / off, "on_tok_s": 1.0 / on,
                "gain_pct": (off / on - 1.0) * 100.0}),
        );
    }
    let out = serde_json::json!({
        "cache_slots": store.cache_slots(),
        "token": tok,
        "hash": want,
        "reps": reps,
        "summary": summary,
        "rows": rows,
    });
    if let Some(dir) = test_env("DSV41_CACHE_OUT") {
        std::fs::write(
            PathBuf::from(dir).join("p3-split.json"),
            serde_json::to_vec_pretty(&out).unwrap(),
        )
        .unwrap();
    }
    finish("real_split_timing", &fails);
}
