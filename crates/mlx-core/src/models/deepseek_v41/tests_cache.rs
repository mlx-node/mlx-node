// S16b real cache invariance on the 9 shards: the p32k8-en prompt greedy in
// three turns (prompt + 7 steps, 8 steps, 32 steps = 48 calls) at expert
// cache 0 (baseline hashes), at a fixed cache size, and at the S15 auto
// admission (with the bank forced `Empty` between turn 1 and 2). Every call's
// last-row logits hash must equal the cache-0 run's (bitwise); the live hit /
// read counts must equal a replay of the run's own route trace. Run with
// `MLX_ENABLE_TF32=0`, `MLX_DSV41_EXPERT_CACHE_GB`, `MLX_DSV41_ROUTE_TRACE`,
// `DSV41_GOLDEN`, `DSV41_DIR`; `DSV41_CACHE_OUT` gets `<name>.json`.

use super::generate::generate_probed;
use super::store::{CachePlan, model_size};
use crate::moe_stream::budget::{CACHE_ENV, Limits, Own, admit, working_set};

/// Steps per turn after the first call of the turn (turn 1's first call is
/// the prompt, later turns start with the previous turn's last pick).
const S16_TURNS: [usize; 3] = [7, 7, 31];
const S16_GIB: f64 = (1u64 << 30) as f64;

fn s16_env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.trim().is_empty())
}

fn s16_store(plan: CachePlan) -> Store {
    assert_eq!(
        s16_env("MLX_ENABLE_TF32").as_deref(),
        Some("0"),
        "DECISIONS S5-1"
    );
    set_cache_limit(POOL_LIMIT as f64).unwrap();
    let t0 = std::time::Instant::now();
    let s = Store::open_with(&real_shards(), DType::Float32, FakeQuant::ON, plan).unwrap();
    println!(
        "load {:.1} s; dense read {} B, resident {} B; bank {:.3} GiB ({} cache slots) allocated in {:.3} s; admission {}",
        t0.elapsed().as_secs_f64(),
        s.dense_read,
        s.dense_bytes,
        s.bank.nbytes() as f64 / S16_GIB,
        s.cache_slots(),
        s.bank_alloc_s,
        s.admission()
            .map_or_else(|| "none".to_owned(), |a| a.to_string())
    );
    s
}

/// First GPU touch of every bank page: a full reduction over each
/// projection's codes and scales. Returns seconds.
fn s16_touch_bank(store: &Store) -> f64 {
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
fn s16_decode_terms(c: &Counters, wall: f64, n: usize) -> Value {
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

fn s16_sum(a: &Counters, b: &Counters) -> Counters {
    Counters {
        router_ns: a.router_ns + b.router_ns,
        engram_ns: a.engram_ns + b.engram_ns,
        engram_bytes: a.engram_bytes + b.engram_bytes,
        fill_ns: a.fill_ns + b.fill_ns,
        fill_bytes: a.fill_bytes + b.fill_bytes,
        fill_slots: a.fill_slots + b.fill_slots,
        uses: a.uses + b.uses,
        hits: a.hits + b.hits,
    }
}

fn s16_median(v: &[f64]) -> f64 {
    let mut d = v.to_vec();
    d.sort_by(f64::total_cmp);
    d.get(d.len() / 2).copied().unwrap_or(0.0)
}

/// One run: three turns on one session; `empty_after_first`: the bank is
/// `Empty` (not volatile) between turn 1 and 2.
fn real_cache_inv(name: &str, plan: CachePlan, turns: &[usize], empty_after_first: bool) {
    let gd = gold("p32k8-en");
    let trace_path =
        PathBuf::from(s16_env("MLX_DSV41_ROUTE_TRACE").expect("MLX_DSV41_ROUTE_TRACE"));
    let (_, swap0) = super::stats::host_pressure();
    let sampler = FootprintMax::start();
    let store = s16_store(plan);
    let mut fails: Vec<String> = Vec::new();
    let touch_s = s16_touch_bank(&store);
    println!(
        "first GPU touch of the bank ({:.3} GiB): {touch_s:.3} s; bank alloc {:.3} s{}",
        store.bank.nbytes() as f64 / S16_GIB,
        store.bank_alloc_s,
        if touch_s > 5.0 || store.bank_alloc_s > 5.0 {
            " -> > 5 s: plan S16b says split the bank into layer-group slabs"
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
        if c > 2.0 * S16_GIB {
            fails.push(format!("MLX cache {:.2} GiB {what} (rule 11)", c / S16_GIB));
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
        let terms = s16_decode_terms(&g.counters, wall, n_calls);
        if ti > 0 {
            dec_c = s16_sum(&dec_c, &g.counters);
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
            s16_median(&g.decode_s),
            1.0 / s16_median(&g.decode_s).max(1e-9),
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
            "step_median_s": s16_median(&g.decode_s),
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
    if max_pool.get() > 2.0 * S16_GIB {
        fails.push(format!(
            "MLX cache {:.2} GiB at a layer (rule 11)",
            max_pool.get() / S16_GIB
        ));
    }

    // live == replay of the run's own routes (when the bank never switched)
    let st = store.bank.stats();
    let live = (
        (st.uses - stats0.uses) as u64,
        (st.hits - stats0.hits) as u64,
        (st.slots - stats0.slots) as u64,
    );
    let slots = store.cache_slots();
    let replayed = if store.switches().is_empty() {
        let mut c = crate::moe_stream::cache::Cache::lru(slots, n_layers, n_experts);
        let r1 = crate::moe_stream::replay::replay(&mut c, n_experts, &turn_steps[0]);
        let rest: Vec<_> = turn_steps[1..].concat();
        let r2 = if empty_after_first {
            let mut c2 = crate::moe_stream::cache::Cache::lru(slots, n_layers, n_experts);
            crate::moe_stream::replay::replay(&mut c2, n_experts, &rest)
        } else {
            // one cache and one set of staging tags across every turn
            let mut c2 = crate::moe_stream::cache::Cache::lru(slots, n_layers, n_experts);
            let all: Vec<_> = turn_steps.concat();
            let r = crate::moe_stream::replay::replay(&mut c2, n_experts, &all);
            crate::moe_stream::replay::Replayed {
                uses: r.uses - r1.uses,
                hits: r.hits - r1.hits,
                decode_uses: r.decode_uses - r1.decode_uses,
                decode_hits: r.decode_hits - r1.decode_hits,
                reads: r.reads - r1.reads,
                decode_reads: r.decode_reads - r1.decode_reads,
            }
        };
        let rep = (r1.uses + r2.uses, r1.hits + r2.hits, r1.reads + r2.reads);
        if rep != live {
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
    let base_name = "s16-cache-inv-c0";
    if name == "c0" {
        // the first 32 calls are S14a's greedy_en calls: same bits
        let p = golden_root().join("hashes/greedy-p32k8-en.json");
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
                "c0 differs from S14a greedy_en at call {first} of {n} (logits hash)"
            ));
        } else {
            println!("c0 == S14a greedy_en hashes for the first {n} calls (bitwise)");
        }
        store_hashes(base_name, doc.clone(), &mut fails);
    } else {
        let p = golden_root().join(format!("hashes/{base_name}.json"));
        let base: Value =
            serde_json::from_slice(&std::fs::read(&p).expect("run real_cache_inv_c0 first"))
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
            println!("{name}: {n_calls} logits hashes == c0 (bitwise)");
        } else {
            fails.push(format!(
                "{name}: logits hashes differ from c0 at calls {diff:?}"
            ));
        }
    }

    let (_, swap1) = super::stats::host_pressure();
    let swap_d = swap1.zip(swap0).map_or(0, |(a, b)| a as i64 - b as i64);
    let fp_peak = sampler.finish();
    let decode = s16_decode_terms(&dec_c, dec_wall, dec_n);
    let out = serde_json::json!({
        "name": name,
        "plan": format!("{:?}", store.plan()),
        "cache_slots": store.cache_slots(),
        "bank_bytes": store.bank.nbytes(),
        "dense_bytes": store.dense_bytes,
        "bank_alloc_s": store.bank_alloc_s,
        "bank_touch_s": touch_s,
        "admission": store.admission().map(|a| a.to_string()),
        "admission_cache_gib": store.admission().map(|a| a.cache_bytes() as f64 / S16_GIB),
        "switches": store.switches(),
        "turns": turns_doc,
        "decode_terms": decode,
        "turn3_terms": turn3,
        "live": {"uses": live.0, "hits": live.1, "reads": live.2},
        "replay": replayed.map(|r| serde_json::json!({"uses": r.0, "hits": r.1, "reads": r.2})),
        "max_pool_gib": max_pool.get() / S16_GIB,
        "footprint_peak_gib": fp_peak as f64 / S16_GIB,
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
        fp_peak as f64 / S16_GIB,
        max_pool.get() / S16_GIB,
    );
    println!("s16 json: {out}");
    if let Some(dir) = s16_env("DSV41_CACHE_OUT") {
        let p = PathBuf::from(dir).join(format!("s16-{name}.json"));
        std::fs::write(&p, serde_json::to_vec_pretty(&out).unwrap()).unwrap();
    }
    finish(&format!("real_cache_inv_{name}"), &fails);
}

/// `MLX_DSV41_EXPERT_CACHE_GB` must be `want` (rule 12: the cache size is
/// explicit per run).
fn s16_cache_gb(want: &str) {
    assert_eq!(s16_env(CACHE_ENV).as_deref(), Some(want), "{CACHE_ENV}");
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~18 GiB"]
fn real_cache_inv_c0() {
    s16_cache_gb("0");
    real_cache_inv("c0", CachePlan::Off, &S16_TURNS, false);
}

#[test]
#[ignore = "real 9 shards + c0 hashes; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=4; ~22 GiB"]
fn real_cache_inv_c4() {
    s16_cache_gb("4");
    let slots = (4.0 * S16_GIB / EXPERT_BYTES as f64) as usize;
    // 16 calls (plan): at 228 slots < 240 experts per token LRU thrashes, so
    // every use is a read
    real_cache_inv("c4", CachePlan::Slots(slots), &S16_TURNS[..2], false);
}

/// S15 admission at load and every turn (`MLX_DSV41_EXPERT_CACHE_GB` = the
/// dry run's N_auto as a cap, so the guard cap holds), bank `Empty` between
/// turn 1 and 2.
#[test]
#[ignore = "real 9 shards + c0 hashes; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=<N_auto>; ~18 GiB + N_auto"]
fn real_cache_inv_auto() {
    assert!(s16_env(CACHE_ENV).is_some(), "{CACHE_ENV} = N_auto (cap)");
    assert!(s16_env("MLX_DSV41_ADMIT_ONLY").is_none());
    let limits = Limits::from_env().unwrap();
    real_cache_inv("auto", CachePlan::Admit(limits), &S16_TURNS, true);
}

/// The admission this host gives right now, from the shard headers alone:
/// every term and the cache size N_auto (GiB). Allocates nothing.
#[test]
#[ignore = "real shard headers; MLX_DSV41_ADMIT_ONLY=1; < 1 GiB"]
fn real_admit_dry_run() {
    assert_eq!(s16_env("MLX_DSV41_ADMIT_ONLY").as_deref(), Some("1"));
    let size = model_size(&real_shards()).unwrap();
    let limits = Limits::from_env().unwrap();
    let h = crate::utils::host::host_memory().expect("host memory");
    let ws = working_set();
    let a = admit(&size, &limits, &h, &Own::default(), ws).unwrap();
    let n_auto = a.cache_bytes() as f64 / S16_GIB;
    println!("model size {size:?}; limits {limits:?}");
    println!("admission: {a}");
    println!(
        "N_auto_gib={n_auto:.3} slots={} fixed_gib={:.3} wired_gib={:.3}",
        a.cache_slots,
        a.fixed as f64 / S16_GIB,
        a.wired_limit() as f64 / S16_GIB
    );
    let t = &a.terms;
    let doc = serde_json::json!({
        "n_auto_gib": n_auto,
        "cache_slots": a.cache_slots,
        "slot_bytes": a.slot,
        "physical_gib": a.physical as f64 / S16_GIB,
        "host_used_gib": a.host_used as f64 / S16_GIB,
        "others_gib": a.others as f64 / S16_GIB,
        "reserve_gib": a.reserve as f64 / S16_GIB,
        "live_gib": a.live as f64 / S16_GIB,
        "ws_gib": a.ws as f64 / S16_GIB,
        "fixed_gib": a.fixed as f64 / S16_GIB,
        "staging_layers": a.staging_layers,
        "terms_gib": {
            "dense": t.dense as f64 / S16_GIB, "staging_layer": t.staging_layer as f64 / S16_GIB,
            "kv": t.kv as f64 / S16_GIB, "compute": t.compute as f64 / S16_GIB,
            "io": t.io as f64 / S16_GIB, "driver": t.driver as f64 / S16_GIB, "pool": t.pool as f64 / S16_GIB,
        },
        "bound": format!("{:?}", a.bound),
        "wired_gib": a.wired_limit() as f64 / S16_GIB,
        "display": a.to_string(),
    });
    if let Some(dir) = s16_env("DSV41_CACHE_OUT") {
        std::fs::write(
            PathBuf::from(dir).join("s16-admit.json"),
            serde_json::to_vec_pretty(&doc).unwrap(),
        )
        .unwrap();
    }
    let active = crate::array::get_active_memory();
    assert!(active < S16_GIB, "dry run allocated {active} B");
}
