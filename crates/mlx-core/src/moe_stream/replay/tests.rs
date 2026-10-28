use super::*;

const TRACE: &str = r#"{"kind":"qwen4_exp_expert_route_trace","schema_version":1,"mode":"ar","prompt_tokens":"2","layer_count":2,"expert_count":8,"top_k":2}
{"kind":"routes","sequence":0,"pass":0,"entry":0,"history_len":2,"source":"prefill_device_commit","layer":0,"rows":2,"top_k":2,"expert_ids":[1,2,2,3]}
{"kind":"routes","sequence":1,"pass":0,"entry":1,"history_len":2,"source":"prefill_device_commit","layer":1,"rows":2,"top_k":2,"expert_ids":[4,5,5,6]}
{"kind":"routes","sequence":2,"pass":1,"entry":0,"history_len":3,"source":"decode_device_commit","layer":0,"rows":1,"top_k":2,"expert_ids":[1,7]}
{"kind":"routes","sequence":3,"pass":1,"entry":1,"history_len":3,"source":"decode_device_commit","layer":1,"rows":1,"top_k":2,"expert_ids":[4,5]}
{"kind":"routes","sequence":4,"pass":2,"entry":0,"history_len":4,"source":"decode_device_commit","layer":0,"rows":1,"top_k":2,"expert_ids":[7,1]}
{"kind":"routes","sequence":5,"pass":2,"entry":1,"history_len":4,"source":"decode_device_commit","layer":1,"rows":1,"top_k":2,"expert_ids":[5,6]}
"#;

#[test]
fn reads_schema_1() {
    let t = read_trace(TRACE.as_bytes()).unwrap();
    assert_eq!(
        (t.layers, t.experts, t.top_k, t.prompt_tokens),
        (2, 8, 2, 2)
    );
    assert_eq!(t.steps.len(), 6);
    assert_eq!(
        t.steps[0],
        Step {
            layer: 0,
            rows: 2,
            ids: vec![1, 2, 2, 3]
        }
    );
    assert!(read_trace(TRACE.replace("\"top_k\":2}", "\"top_k\":3}").as_bytes()).is_err());
    assert!(read_trace(TRACE.replace("[4,5]", "[4,9]").as_bytes()).is_err());
    assert!(read_trace(&b"{\"kind\":\"other\"}\n"[..]).is_err());
}

/// Hand-counted: prefill reads 3 + 3 experts; with 16 slots (quota 8 per
/// layer) all six are seeded, so decode step 1 hits 1, 4, 5 and misses 7
/// (a victim), step 2 hits all 4.
#[test]
fn replay_counts_by_hand() {
    let t = read_trace(TRACE.as_bytes()).unwrap();
    let none = replay_lru(&t, 0);
    assert_eq!((none.uses, none.hits, none.decode_uses), (14, 0, 8));
    // 0 slots: decode stages too; only expert 7 is not already staged for its layer
    assert_eq!(none.reads, 6 + 1);
    let big = replay_lru(&t, 16);
    assert_eq!((big.hits, big.decode_hits, big.reads), (7, 7, 6 + 1));
    assert!((big.decode_h() - 7.0 / 8.0).abs() < 1e-12);
    // 1 slot: each decode step keeps one victim; the next step's wanted set
    // never evicts its own hit
    let one = replay_lru(&t, 1);
    assert!(one.hits <= big.hits && one.reads >= big.reads, "{one:?}");
}

/// Hand-counted: layer 1 pinned (every use a hit, nothing read), layer 0
/// staged as with cache 0; the hybrid with exactly one layer's slots is the
/// same, one slot less does not fit.
#[test]
fn pinned_layer_counts_by_hand() {
    let t = read_trace(TRACE.as_bytes()).unwrap();
    assert_eq!(reads_per_layer(&t), vec![(4, 1), (3, 0)]);
    let mut c = Cache::lru(0, t.layers, t.experts);
    let r = replay_pinned(&mut c, t.experts, &t.steps, &[false, true]);
    assert_eq!(
        (
            r.uses,
            r.hits,
            r.reads,
            r.decode_uses,
            r.decode_hits,
            r.decode_reads
        ),
        (14, 7, 4, 8, 4, 1)
    );
    assert_eq!(replay_hybrid(&t, 8, &[1], None), Some(r));
    assert_eq!(replay_hybrid(&t, 7, &[1], None), None);
    // nothing pinned == plain replay
    assert_eq!(replay_hybrid(&t, 16, &[], None), Some(replay_lru(&t, 16)));
    // a long half-life LFU on this tiny trace behaves like the hand count of LRU
    assert_eq!(replay_lfu(&t, 16, 1e9).hits, replay_lru(&t, 16).hits);
}

const GIB: f64 = (1u64 << 30) as f64;
/// Bytes of one routed expert (gate + up + down MXFP4) of the real file.
const EXPERT_BYTES: f64 = 18_800_640.0;

/// Decode time terms measured on a cache-0 run (`DSV41_DECODE_TERMS`).
struct Terms {
    t_c: f64,
    bw: f64,
    t_engram: f64,
    t_router: f64,
}

fn terms() -> Option<Terms> {
    let p = std::env::var_os("DSV41_DECODE_TERMS")?;
    let v: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    let d = &v["decode_terms"];
    let f = |k: &str| d[k].as_f64().unwrap_or_else(|| panic!("decode_terms.{k}"));
    Some(Terms {
        t_c: f("t_c_s"),
        bw: f("bw_bytes_s"),
        t_engram: f("t_engram_s"),
        t_router: f("t_router_s"),
    })
}

/// Decode tokens of a trace (decode steps of layer 0).
fn decode_tokens(t: &RouteTrace) -> u64 {
    t.steps
        .iter()
        .filter(|s| s.layer == 0 && decode_step(s.rows))
        .count() as u64
}

/// Predicted decode tok/s `1 / (t_c - saved + miss_bytes / BW + t_engram)`;
/// `saved` = router waits of pinned layers (upper bound).
fn tok_s(tm: &Terms, r: &Replayed, n_dec: u64, pinned: usize, layers: usize) -> (f64, f64) {
    let miss = r.decode_reads as f64 * EXPERT_BYTES / n_dec.max(1) as f64;
    let base = tm.t_c + miss / tm.bw + tm.t_engram;
    let saved = tm.t_router * pinned as f64 / layers as f64;
    (1.0 / base, 1.0 / (base - saved).max(1e-9))
}

/// h vs cache size on the real route traces in `DSV41_TRACES`
/// for the arms global LRU, global LFU-decay (half-life 16 / 64 / 256
/// tokens), layer-pinned (whole layers with the most staging-only decode
/// reads in the OTHER traces, rest streamed with cache 0) and hybrid (k
/// pinned layers + LRU or LFU-decay on the rest, best k and policy), plus the
/// predicted decode tok/s from the terms of a cache-0 run. Writes
/// `DSV41_HTABLE_OUT` (markdown) and a `.json` next to it. Skips without
/// `DSV41_TRACES`.
#[test]
fn real_traces() {
    let Some(dir) = std::env::var_os("DSV41_TRACES") else {
        println!("DSV41_TRACES unset: skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let names: Vec<String> = std::env::var("DSV41_REPLAY_SET")
        .unwrap_or_else(|_| "s16-c0,s14-greedy-zh,s14-greedy-code".into())
        .split(',')
        .map(str::to_owned)
        .collect();
    let mut traces: Vec<(String, RouteTrace)> = names
        .iter()
        .map(|n| {
            let f = std::fs::File::open(dir.join(format!("{n}.jsonl")))
                .unwrap_or_else(|e| panic!("{n}: {e}"));
            (n.clone(), read_trace(std::io::BufReader::new(f)).unwrap())
        })
        .collect();
    let per_layer: Vec<Vec<(u64, u64)>> = traces.iter().map(|(_, t)| reads_per_layer(t)).collect();
    // one session through all traces in order (the cache carries over)
    let mut mix = traces[0].1.clone();
    for (_, t) in &traces[1..] {
        assert_eq!(
            (t.layers, t.experts, t.top_k),
            (mix.layers, mix.experts, mix.top_k)
        );
        mix.steps.extend(t.steps.iter().cloned());
    }
    traces.push((format!("mix({})", names.join("+")), mix));
    let (layers, experts, top_k) = (traces[0].1.layers, traces[0].1.experts, traces[0].1.top_k);
    let tm = terms();
    let auto: Option<f64> = std::env::var("DSV41_AUTO_GIB")
        .ok()
        .map(|v| v.parse().unwrap());
    let mut sizes: Vec<(String, f64)> = [8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 80.0]
        .iter()
        .map(|&g| (format!("{g}"), g))
        .collect();
    if let Some(a) = auto {
        sizes.push((format!("auto {a:.2}"), a));
        sizes.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    let uses_per_token = (layers * top_k) as f64;
    let half_lives = [16.0, 64.0, 256.0];

    let mut md = String::new();
    let mut js = Vec::new();
    md.push_str(&format!(
        "# replay h table\n\nTraces: {}. Expert = {EXPERT_BYTES} B; one pinned layer = {experts} slots = {:.2} GiB. h = byte hit rate over every expert use; dec-h = over decode steps only (the decode speed term). Prefill misses are staged, not cached (S16a-1), so every run starts decode with an empty cache. LFU half-life in tokens ({uses_per_token} uses each); HYB hl 0 = LRU on the unpinned slots, k 0 = the better global arm. Pinned layers ranked by staging-only decode reads in the OTHER traces (all traces for the mix). Predicted tok/s = 1/(t_c + decode misses x E / BW / tokens + t_engram){}; `sync` = also minus the router wait of the pinned layers (upper bound). Saved host syncs per token = PIN N / HYB k (a pinned layer needs no router -> read wait).\n\n",
        names.join(", "),
        experts as f64 * EXPERT_BYTES / GIB,
        match &tm {
            Some(t) => format!(
                " with t_c {:.1} ms, BW {:.2} GB/s, t_engram {:.1} ms, router wait {:.1} ms/token (cache-0 run)",
                t.t_c * 1e3,
                t.bw / 1e9,
                t.t_engram * 1e3,
                t.t_router * 1e3
            ),
            None => " (no DSV41_DECODE_TERMS: not computed)".into(),
        }
    ));
    let mut checkpoint = None;
    for (ti, (name, t)) in traces.iter().enumerate() {
        let n_dec = decode_tokens(t);
        // rank layers by decode reads at cache 0 in the other traces
        let mut score = vec![0u64; layers];
        for (j, pl) in per_layer.iter().enumerate() {
            if j != ti || traces.len() - 1 == 1 {
                for (l, &(_, d)) in pl.iter().enumerate() {
                    score[l] += d;
                }
            }
        }
        let mut rank: Vec<usize> = (0..layers).collect();
        rank.sort_by_key(|&l| (std::cmp::Reverse(score[l]), l));
        let c0 = replay_lru(t, 0);
        md.push_str(&format!(
            "## {name}\n\n{} layer steps, {} decode tokens; cache 0: dec reads {} ({:.2} GB/token){}; pin order {:?}\n\n",
            t.steps.len(),
            n_dec,
            c0.decode_reads,
            c0.decode_reads as f64 * EXPERT_BYTES / 1e9 / n_dec.max(1) as f64,
            tm.as_ref()
                .map(|tm| format!(", predicted {:.2} tok/s", tok_s(tm, &c0, n_dec, 0, layers).0))
                .unwrap_or_default(),
            &rank[..10.min(layers)]
        ));
        md.push_str("| GiB | slots | LRU h / dec-h | LFU best dec-h (hl) | PIN N: dec-h | HYB best (k, hl): dec-h | tok/s LRU | LFU | PIN (sync) | HYB (sync) |\n|---|---|---|---|---|---|---|---|---|---|\n");
        for (label, g) in &sizes {
            let slots = (g * GIB / EXPERT_BYTES) as usize;
            let lru = replay_lru(t, slots);
            let lfu: Vec<(f64, Replayed)> = half_lives
                .iter()
                .map(|&h| (h, replay_lfu(t, slots, h * uses_per_token)))
                .collect();
            let (lfu_h, lfu_best) = lfu
                .iter()
                .min_by_key(|(h, r)| (r.decode_reads, *h as u64))
                .copied()
                .unwrap();
            let n_pin = (slots / experts).min(layers);
            let pin = replay_hybrid(t, n_pin * experts, &rank[..n_pin], None).unwrap();
            // hybrid: k pinned layers + the rest under LRU (hl 0) or LFU-decay;
            // k = 0 is the better of the two global arms
            let mut hyb = if lru.decode_reads <= lfu_best.decode_reads {
                (0usize, 0.0, lru)
            } else {
                (0usize, lfu_h, lfu_best)
            };
            for k in 1..=n_pin {
                for h in std::iter::once(0.0).chain(half_lives) {
                    let hl = (h > 0.0).then_some(h * uses_per_token);
                    let r = replay_hybrid(t, slots, &rank[..k], hl).unwrap();
                    if r.decode_reads < hyb.2.decode_reads {
                        hyb = (k, h, r);
                    }
                }
            }
            let pred = tm.as_ref().map(|tm| {
                (
                    tok_s(tm, &lru, n_dec, 0, layers).0,
                    tok_s(tm, &lfu_best, n_dec, 0, layers).0,
                    tok_s(tm, &pin, n_dec, n_pin, layers),
                    tok_s(tm, &hyb.2, n_dec, hyb.0, layers),
                )
            });
            let fmt_pred = |p: Option<f64>| p.map_or("-".into(), |v| format!("{v:.2}"));
            md.push_str(&format!(
                "| {label} | {slots} | {:.3} / {:.3} | {:.3} ({lfu_h}) | {n_pin}: {:.3} | ({}, {}): {:.3} | {} | {} | {} ({}) | {} ({}) |\n",
                lru.h(),
                lru.decode_h(),
                lfu_best.decode_h(),
                pin.decode_h(),
                hyb.0,
                hyb.1,
                hyb.2.decode_h(),
                fmt_pred(pred.map(|p| p.0)),
                fmt_pred(pred.map(|p| p.1)),
                fmt_pred(pred.map(|p| p.2.0)),
                fmt_pred(pred.map(|p| p.2.1)),
                fmt_pred(pred.map(|p| p.3.0)),
                fmt_pred(pred.map(|p| p.3.1)),
            ));
            let lfu_all: Vec<Value> = lfu
                .iter()
                .map(|(h, r)| serde_json::json!({"half_life_tokens": h, "h": r.h(), "decode_h": r.decode_h(), "decode_reads": r.decode_reads}))
                .collect();
            js.push(serde_json::json!({
                "trace": name, "size": label, "gib": g, "slots": slots, "decode_tokens": n_dec,
                "lru": {"h": lru.h(), "decode_h": lru.decode_h(), "decode_reads": lru.decode_reads, "reads": lru.reads},
                "lfu": lfu_all,
                "pinned": {"layers": &rank[..n_pin], "saved_syncs_per_token": n_pin, "saved_router_ms": tm.as_ref().map(|t| t.t_router * 1e3 * n_pin as f64 / layers as f64), "h": pin.h(), "decode_h": pin.decode_h(), "decode_reads": pin.decode_reads},
                "hybrid": {"k": hyb.0, "saved_syncs_per_token": hyb.0, "saved_router_ms": tm.as_ref().map(|t| t.t_router * 1e3 * hyb.0 as f64 / layers as f64), "half_life_tokens": hyb.1, "h": hyb.2.h(), "decode_h": hyb.2.decode_h(), "decode_reads": hyb.2.decode_reads},
                "pred_tok_s": pred.map(|p| serde_json::json!({"lru": p.0, "lfu": p.1, "pinned": p.2.0, "pinned_sync": p.2.1, "hybrid": p.3.0, "hybrid_sync": p.3.1})),
            }));
            if ti == 0 && (*g - 24.0).abs() < 1e-9 {
                checkpoint = Some((lru.decode_h(), lfu_best.decode_h(), hyb.2.decode_h()));
            }
            // a larger cache never reads more under LRU on the same trace
            assert!(
                lru.reads <= c0.reads,
                "{name} {label}: cache reads more than none"
            );
        }
        md.push('\n');
    }
    if let Some((l, f, h)) = checkpoint {
        let near = l < 0.3;
        md.push_str(&format!(
            "Checkpoint: {} at 24 GiB decode h LRU {l:.3}, LFU {f:.3}, hybrid {h:.3}{}\n",
            traces[0].0,
            if near {
                " -> BELOW 0.3 (near uniform)"
            } else {
                " (>= 0.3)"
            }
        ));
        println!(
            "checkpoint: 24 GiB decode h LRU {l:.3} LFU {f:.3} hybrid {h:.3} near_uniform={near}"
        );
    }
    print!("{md}");
    if let Some(out) = std::env::var_os("DSV41_HTABLE_OUT") {
        let out = std::path::PathBuf::from(out);
        std::fs::write(&out, &md).unwrap();
        std::fs::write(
            out.with_extension("json"),
            serde_json::to_vec_pretty(&Value::Array(js)).unwrap(),
        )
        .unwrap();
        println!("h table -> {}", out.display());
    }
}

/// Hand-counted seeding on the 2-layer trace with 4 slots (quota 2 per
/// layer): prefill seeds the two most-used experts of each layer (layer 0:
/// 2 (2 rows), then 1; layer 1: 5, then 4), so decode step 1 hits 1 and 4,
/// 5 and misses 7 (victim); step 2 hits 7, 1, 5 and serves 6 from its
/// still-valid staging copy (no read). The turn split and the per-token
/// view agree with the plain replay.
#[test]
fn seeded_counts_by_hand() {
    let t = read_trace(TRACE.as_bytes()).unwrap();
    let mut c = cache_for(&t, 4);
    let r = replay(&mut c, t.experts, &t.steps);
    // reads: prefill 3 + 3, decode 7 only
    assert_eq!(
        (r.uses, r.hits, r.reads, r.decode_hits, r.decode_reads),
        (14, 6, 7, 6, 1),
        "{r:?}"
    );
    let v = replay_turns(
        &mut cache_for(&t, 4),
        t.experts,
        &[&t.steps[..2], &t.steps[2..]],
    );
    assert_eq!(v[0].plus(v[1]), r);
    let tok = per_token(&mut cache_for(&t, 4), t.experts, t.layers, &[&t.steps]);
    assert_eq!(tok, vec![(4, 3, 1), (4, 3, 0)]);
}
