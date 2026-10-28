// Stage P4 prefill timing on the 9 shards: cold prompts of T ids
// (`$DSV41_PROMPTS/compute-<T>.json`, `DSV41_PREFILL_TS`, default
// 256,1024,8192) at the S15 auto admission, every tag and cache entry
// forgotten before each prompt (cold: every used expert is read once).
// Per prompt: TTFT (`forward_last` + last row to the host), the expert reads
// and, with the phase profile on (`stats::set_profiling`, evals at phase
// edges), the time split. Arms (`DSV41_PREFILL_ARMS`, interleaved ABBA per
// rep) must give bitwise equal last-row logits per T. `DSV41_PREFILL_SLOTS`
// replaces the admission by a fixed cache size. Run with
// `MLX_ENABLE_TF32=0`, `MLX_DSV41_EXPERT_CACHE_GB=<N_auto>`, `DSV41_PROMPTS`.

fn p4_prompt(t: usize) -> Vec<u32> {
    let dir = PathBuf::from(std::env::var("DSV41_PROMPTS").expect("DSV41_PROMPTS"));
    let pj: Value =
        serde_json::from_slice(&std::fs::read(dir.join(format!("compute-{t}.json"))).unwrap())
            .unwrap();
    let ids: Vec<u32> = pj["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids.len(), t);
    ids
}

/// Sets arm `name` on the store (P4: `base` = the store as opened; `off`,
/// `pred`, `all` = the next-layer prefetch mode).
fn p4_arm(store: &Store, name: &str) {
    use crate::moe_stream::bank::PrefetchMode;
    match name {
        "base" => {}
        "off" => store.bank.set_prefetch_mode(PrefetchMode::Off),
        "pred" => store.bank.set_prefetch_mode(PrefetchMode::Pred),
        "all" => store.bank.set_prefetch_mode(PrefetchMode::All),
        a => panic!("unknown arm {a}"),
    }
}

#[test]
#[ignore = "real 9 shards + $DSV41_PROMPTS; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=<N_auto>; ~19 GiB + N_auto + compute(T); ~0.3 TB read per prompt"]
fn real_prefill_timing() {
    assert!(s16_env(CACHE_ENV).is_some(), "{CACHE_ENV} = N_auto (cap)");
    let ts: Vec<usize> = s16_env("DSV41_PREFILL_TS")
        .unwrap_or_else(|| "256,1024,8192".into())
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let arms: Vec<String> = s16_env("DSV41_PREFILL_ARMS")
        .unwrap_or_else(|| "base".into())
        .split(',')
        .map(|v| v.trim().to_owned())
        .collect();
    let reps: usize = s16_env("DSV41_PREFILL_REPS").map_or(1, |v| v.parse().unwrap());
    // "both": a plain run (TTFT) and a profiled run (split) per (T, arm)
    let prof = s16_env("DSV41_PREFILL_PROFILE").unwrap_or_else(|| "both".into());
    let profs: Vec<bool> = match prof.as_str() {
        "both" => vec![false, true],
        "on" => vec![true],
        "off" => vec![false],
        p => panic!("DSV41_PREFILL_PROFILE={p}"),
    };
    let prompts: Vec<(usize, Vec<u32>)> = ts.iter().map(|&t| (t, p4_prompt(t))).collect();
    // `DSV41_PREFILL_SLOTS`: exactly this many cache slots, staging layers
    // from `MLX_DSV41_STAGING_LAYERS` (no admission: the A/B of a second
    // staging layer when the host's admission would not give one)
    let plan = match s16_env("DSV41_PREFILL_SLOTS") {
        Some(n) => CachePlan::Slots(n.parse().unwrap()),
        None => CachePlan::Admit(Limits::from_env().unwrap()),
    };
    let store = s16_store(plan);
    let m = &store.model;
    let (n_layers, n_exp) = (m.cfg.n_layers, m.cfg.n_routed_experts);
    let vocab = m.cfg.vocab_size;
    store.begin_turn().unwrap();
    println!("touch bank {:.2} s", s16_touch_bank(&store));
    let mut fails: Vec<String> = Vec::new();
    let mut rows: Vec<Value> = Vec::new();
    let mut want: std::collections::BTreeMap<usize, String> = Default::default();
    let mut ttft: std::collections::BTreeMap<(usize, String, bool), Vec<f64>> = Default::default();
    for r in 0..reps {
        for (t, ids) in &prompts {
            let mut order = arms.clone();
            if r % 2 == 1 {
                order.reverse();
            }
            for arm in &order {
                for &pf in &profs {
                    p4_arm(&store, arm);
                    super::stats::set_profiling(pf);
                    store.bank.test_forget_all();
                    crate::array::synchronize();
                    let c0 = Counters::now(&store.bank);
                    let ph0 = super::stats::phase_seconds();
                    let b0 = store.bank.stats();
                    let mut sess = store.model.new_session();
                    let mut trace = super::model::Trace::default();
                    let t0 = std::time::Instant::now();
                    let l = store
                        .model
                        .forward_last(&mut sess, ids, None, Some(&mut trace))
                        .unwrap();
                    let row = read(&l);
                    let dt = t0.elapsed().as_secs_f64();
                    super::stats::set_profiling(false);
                    let c = Counters::now(&store.bank).since(&c0);
                    let b = store.bank.stats();
                    let ph1 = super::stats::phase_seconds();
                    let h = hex_sha(&row[row.len() - vocab..]);
                    match want.get(t) {
                        Some(w) if *w != h => fails.push(format!(
                            "T={t} rep {r} arm {arm} prof {pf}: logits {h} != {w}"
                        )),
                        Some(_) => {}
                        None => {
                            want.insert(*t, h.clone());
                        }
                    }
                    // every (layer, expert) the prompt uses is read exactly once
                    // (cold), never twice; prefetched experts the routes do
                    // not use are counted apart
                    let mut used = 0usize;
                    for routes in &trace.routes {
                        let mut seen = vec![false; n_exp];
                        routes.ids.iter().for_each(|&e| seen[e as usize] = true);
                        used += seen.iter().filter(|&&x| x).count();
                    }
                    assert_eq!(trace.routes.len(), n_layers);
                    let wasted = b.prefetch_unused - b0.prefetch_unused;
                    if c.fill_slots as usize != used + wasted {
                        fails.push(format!(
                            "T={t} rep {r} arm {arm}: read {} experts, used {used} + unused prefetch {wasted}",
                            c.fill_slots
                        ));
                    }
                    let split: serde_json::Map<String, Value> = super::stats::PHASES
                        .iter()
                        .map(|p| {
                            (
                                p.name().to_owned(),
                                json!(ph1[*p as usize] - ph0[*p as usize]),
                            )
                        })
                        .collect();
                    let phase_sum: f64 = (0..ph0.len()).map(|i| ph1[i] - ph0[i]).sum();
                    ttft.entry((*t, arm.clone(), pf)).or_default().push(dt);
                    let io_s = c.fill_ns as f64 * 1e-9;
                    let v = json!({
                        "rep": r, "T": t, "arm": arm, "profile": pf, "ttft_s": dt,
                        "prefill_tok_s": *t as f64 / dt, "hash": h,
                        "used_experts": used, "read_experts": c.fill_slots,
                        "prefetch_unused": wasted, "prefetch_slots": b.prefetch_slots - b0.prefetch_slots,
                        "read_gb": c.fill_bytes as f64 / 1e9, "io_s": io_s,
                        "io_wait_s": c.fill_wait_ns as f64 * 1e-9,
                        "read_gbps": c.fill_bytes as f64 / 1e9 / io_s.max(1e-9),
                        "engram_gb": c.engram_bytes as f64 / 1e9, "engram_s": c.engram_ns as f64 * 1e-9,
                        "router_wait_s": c.router_ns as f64 * 1e-9,
                        "hits": c.hits, "uses": c.uses,
                        "split": split, "phase_sum_s": phase_sum,
                        "other_s": if pf { dt - phase_sum } else { 0.0 },
                        "mlx_cache_gib": crate::array::get_cache_memory() / S16_GIB,
                        "footprint_gib": super::stats::rusage().0 as f64 / S16_GIB,
                    });
                    println!("p4 prefill {v}");
                    rows.push(v);
                    if crate::array::get_cache_memory() > 2.0 * S16_GIB {
                        fails.push(format!("T={t}: MLX cache > 2 GiB (rule 11)"));
                    }
                }
            }
        }
    }
    store.end_turn().unwrap();
    let mut summary = serde_json::Map::new();
    for ((t, arm, pf), v) in &ttft {
        let med = s16_median(v);
        println!(
            "p4 T={t} arm={arm} profile={pf}: median TTFT {med:.2} s ({:.1} tok/s) over {}",
            *t as f64 / med,
            v.len()
        );
        summary.insert(
            format!("{t}/{arm}/{}", if *pf { "prof" } else { "plain" }),
            json!({"median_ttft_s": med, "prefill_tok_s": *t as f64 / med, "runs": v}),
        );
    }
    let out = json!({
        "cache_slots": store.cache_slots(),
        "staging_layers": store.bank.staging_layers(),
        "admission": store.admission().map(|a| a.to_string()),
        "summary": summary,
        "rows": rows,
    });
    if let Some(dir) = s16_env("DSV41_CACHE_OUT") {
        let name = s16_env("DSV41_PREFILL_NAME").unwrap_or_else(|| "p4-prefill".into());
        std::fs::write(
            PathBuf::from(dir).join(format!("{name}.json")),
            serde_json::to_vec_pretty(&out).unwrap(),
        )
        .unwrap();
    }
    finish("real_prefill_timing", &fails);
}
