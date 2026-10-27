// S14a real end to end on the 9 shards (expert cache 0) vs the S12 goldens
// (`$DSV41_GOLDEN/<case>/{fp32,bf16}`). mlx-node runs in f32 (DECISIONS
// S13a-1, S14a-1). Per-step logits hashes go to `$DSV41_GOLDEN/hashes/`.
// Run with `MLX_ENABLE_TF32=0`, `MLX_DSV41_EXPERT_CACHE_GB=0`,
// `MLX_DSV41_STATS`, `MLX_DSV41_ROUTE_TRACE`.

use super::generate::{Call, Generated, generate};

/// Bytes of one routed expert (gate + up + down MXFP4).
const EXPERT_BYTES: u64 = 18_800_640;

/// The oracle run of one case: prompt, fp32 greedy picks (one per call),
/// and each call's (start, ids, ora32 logits, ora16 logits) `[s * vocab]`.
struct Gold {
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    calls: Vec<(usize, Vec<u32>, Vec<f32>, Vec<f32>)>,
    /// Template special-token ids of the case (S12a-4).
    special: BTreeSet<u32>,
}

fn golden_root() -> PathBuf {
    PathBuf::from(std::env::var("DSV41_GOLDEN").expect("DSV41_GOLDEN"))
}

fn gold(case: &str) -> Gold {
    let root = golden_root();
    let (d32, d16) = (root.join(case).join("fp32"), root.join(case).join("bf16"));
    let res: Value =
        serde_json::from_slice(&std::fs::read(d32.join("result.json")).unwrap()).unwrap();
    let ids_of = |v: &Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let prompt = ids_of(&res["prompt_ids"]);
    let greedy = ids_of(&res["generated_argmax"]);
    let n = res["calls"].as_array().unwrap().len();
    let calls = (0..n)
        .map(|c| {
            let mut f = GFile::open(d32.join(format!("c{c:03}/call.npz")));
            let mut b = GFile::open(d16.join(format!("c{c:03}/call.npz")));
            let ids: Vec<u32> = f.get("ids").i.iter().map(|&x| x as u32).collect();
            assert_eq!(b.get("ids").i.len(), ids.len());
            let start = f.get("start_pos").i[0] as usize;
            (start, ids, f.get("logits").f, b.get("logits").f)
        })
        .collect();
    let case_json = root.join("../oracle/cases").join(format!("{case}.json"));
    let cj: Value = serde_json::from_slice(&std::fs::read(case_json).unwrap()).unwrap();
    let special = ids_of(&cj["ids"])
        .into_iter()
        .zip(cj["tokens"].as_array().unwrap())
        .filter(|(_, t)| {
            let t = t.as_str().unwrap();
            t.starts_with("<｜") || t == "<think>" || t == "</think>"
        })
        .map(|(i, _)| i)
        .collect();
    Gold {
        prompt,
        greedy,
        calls,
        special,
    }
}

fn e2e_store() -> Store {
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "DECISIONS S5-1"
    );
    assert_eq!(
        std::env::var("MLX_DSV41_EXPERT_CACHE_GB").as_deref(),
        Ok("0"),
        "S14a runs at expert cache 0 (rule 12)"
    );
    set_cache_limit(POOL_LIMIT as f64).unwrap();
    let t0 = std::time::Instant::now();
    let s = Store::open(&real_shards(), DType::Float32, FakeQuant::ON).unwrap();
    println!(
        "load {:.1} s; dense read {} B, resident {} B",
        t0.elapsed().as_secs_f64(),
        s.dense_read,
        s.dense_bytes
    );
    s
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = row.iter().map(|&x| (f64::from(x) - m).exp()).sum();
    let lz = m + z.ln();
    row.iter().map(|&x| f64::from(x) - lz).collect()
}

/// KL(p || q) of two logit rows; NaN when either row is not all finite.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    if !p.iter().chain(q).all(|v| v.is_finite()) {
        return f64::NAN;
    }
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    let s: f64 = lp.iter().zip(&lq).map(|(&a, &b)| a.exp() * (a - b)).sum();
    if s.is_finite() { s.max(0.0) } else { f64::NAN }
}

fn hex_sha(v: &[f32]) -> String {
    use sha2::Digest;
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    sha2::Sha256::digest(&b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// §3 greedy site: `got` vs ora32's pick at one logits row; `Ok(None)` equal,
/// `Ok(Some(..))` an excused near tie (S13a rules), `Err` otherwise.
fn pick_ok(
    got: u32,
    l32: &[f32],
    l16: &[f32],
) -> std::result::Result<Option<(f32, f32, usize)>, String> {
    let want: BTreeSet<i64> = [i64::from(super::generate::argmax(l32))].into();
    sel_row(&[i64::from(got)], &want, l32, l16, 1)
}

/// G14.4: route trace + stats written, and expert bytes == misses x E
/// (+ <= 0.5 % page-window widening).
fn check_io(g: &Generated, n_calls: usize, n_layers: usize, fails: &mut Vec<String>) {
    let c = &g.counters;
    let want = c.fill_slots * EXPERT_BYTES;
    let ratio = c.fill_bytes as f64 / want.max(1) as f64;
    println!(
        "io: misses {} experts, expert bytes {} (x{ratio:.5} of misses x {EXPERT_BYTES}), engram {} B, total {} B; router wait {:.2} s, fill wait {:.2} s, engram {:.2} s",
        c.fill_slots,
        c.fill_bytes,
        c.engram_bytes,
        g.bytes_read_total,
        c.router_ns as f64 * 1e-9,
        c.fill_ns as f64 * 1e-9,
        c.engram_ns as f64 * 1e-9
    );
    if !(1.0..=1.005).contains(&ratio) {
        fails.push(format!("G14.4 expert bytes ratio {ratio}"));
    }
    let lines = |var: &str| {
        std::env::var_os(var).map(|p| {
            std::fs::read_to_string(p)
                .unwrap_or_default()
                .lines()
                .count()
        })
    };
    let trace = lines("MLX_DSV41_ROUTE_TRACE");
    if trace != Some(1 + n_calls * n_layers) {
        fails.push(format!(
            "G14.4 route trace lines {trace:?}, want {}",
            1 + n_calls * n_layers
        ));
    }
    let st = lines("MLX_DSV41_STATS");
    if st.is_none_or(|n| n < g.argmax.len()) {
        fails.push(format!("G14.4 stats lines {st:?} < {}", g.argmax.len()));
    }
    println!("stats: {}", g.stats);
}

fn finish(name: &str, fails: &[String]) {
    println!(
        "{name}: MLX peak {:.2} GiB, footprint peak {:.2} GiB",
        crate::array::get_peak_memory() / (1u64 << 30) as f64,
        super::stats::rusage().1 as f64 / (1u64 << 30) as f64
    );
    for f in fails.iter().take(60) {
        println!("FAIL {f}");
    }
    assert!(fails.is_empty(), "{} failures (first above)", fails.len());
}

/// G14.1: the oracle's calls teacher-forced (prompt + its fp32 greedy ids):
/// per position `KL(ora32 || mlx) <= max(2 KL(ora32 || ora16), 1e-4)`, top-1
/// mismatches only near ties (§3 / S13a), error-curve rule on every layer's
/// stream.
fn real_e2e_tf(case: &str) {
    let gd = gold(case);
    let store = e2e_store();
    let cfg = store.model.cfg.clone();
    let root = golden_root().join(case);
    let (d32, d16) = (root.join("fp32"), root.join("bf16"));
    let vocab = cfg.vocab_size;
    let steps = gd.calls.len() - 1;
    let force: Vec<u32> = gd.calls[1..].iter().map(|c| c.1[0]).collect();
    let mut sess = store.model.new_session();
    let mut fails = Vec::new();
    let mut acc = vec![[0f64; 3]; cfg.n_layers];
    let (mut worst_kl, mut excused) = (0f64, Vec::new());
    let mut hook = |c: &Call| -> napi::Result<()> {
        let (start, ids, l32, l16) = &gd.calls[c.index];
        assert_eq!((c.start, c.ids), (*start, &ids[..]), "call {}", c.index);
        let got = read(c.logits);
        if !got.iter().all(|v| v.is_finite()) {
            fails.push(format!("call {}: non-finite logits", c.index));
        }
        for p in 0..ids.len() {
            let w = p * vocab..(p + 1) * vocab;
            let (a, b, z) = (&got[w.clone()], &l32[w.clone()], &l16[w]);
            let (k_m, k_f) = (kl(b, a), kl(b, z));
            let bound = (2.0 * k_f).max(1e-4);
            worst_kl = worst_kl.max(k_m / bound);
            if k_m.is_nan() || k_m > bound {
                fails.push(format!(
                    "G14.1 call {} pos {}: KL {k_m:e} > {bound:e}",
                    c.index,
                    start + p
                ));
            }
            match pick_ok(super::generate::argmax(a), b, z) {
                Ok(None) => {}
                Ok(Some(t)) => excused.push(format!("call {} pos {}: {t:?}", c.index, start + p)),
                Err(e) => fails.push(format!(
                    "G14.1 top-1 call {} pos {}: {e}",
                    c.index,
                    start + p
                )),
            }
        }
        for (l, a) in acc.iter_mut().enumerate() {
            let mine = read(&c.trace.out[l]);
            let (o32, o16) = (
                gfile(&d32, c.index, l).get("out").f,
                gfile(&d16, c.index, l).get("out").f,
            );
            assert_eq!(mine.len(), o32.len(), "L{l} stream");
            for ((&x, &b), &z) in mine.iter().zip(&o32).zip(&o16) {
                let (x, b, z) = (f64::from(x), f64::from(b), f64::from(z));
                a[0] += (x - b) * (x - b);
                a[1] += (z - b) * (z - b);
                a[2] += b * b;
            }
        }
        let cache = get_cache_memory();
        assert!(cache <= 2.0 * (1u64 << 30) as f64, "MLX cache {cache} B");
        Ok(())
    };
    let g = generate(
        &store,
        &mut sess,
        &gd.prompt,
        None,
        steps,
        Some(&force),
        true,
        &mut hook,
    )
    .unwrap();
    let mut prev: Option<f64> = None;
    println!("layer  relL2(mlx,ora32)  floor(ora16,ora32)  r(L)");
    for (l, a) in acc.iter().enumerate() {
        let (e, f) = ((a[0] / a[2]).sqrt(), (a[1] / a[2]).sqrt());
        let r = e / f.max(1e-6);
        println!("L{l:02}  {e:.3e}  {f:.3e}  {r:.3}");
        if r.is_nan() || r > 2.0 {
            fails.push(format!("error curve L{l}: r {r}"));
        }
        if let Some(p) = prev.filter(|&p| r - p > 0.25) {
            fails.push(format!("error curve step L{l}: {p} -> {r}"));
        }
        prev = Some(r);
    }
    println!(
        "{case}: {} calls, worst KL/bound {worst_kl:.3}, top-1 excused {} ({excused:?}); mlx greedy {:?} vs oracle {:?}",
        gd.calls.len(),
        excused.len(),
        g.argmax,
        gd.greedy
    );
    // §3: more than 1 % of a site's rows excused also fails
    let n_rows: usize = gd.calls.iter().map(|c| c.1.len()).sum();
    if excused.len() as f64 > 0.01 * n_rows as f64 {
        fails.push(format!(
            "G14.1 top-1 excused {} of {n_rows} rows > 1 % (§3 site cap)",
            excused.len()
        ));
    }
    check_io(&g, gd.calls.len(), cfg.n_layers, &mut fails);
    finish(&format!("real_e2e_tf {case}"), &fails);
}

/// G14.5 (§3 quality rule) on the EN prompt, teacher-forced over the prompt:
/// `|NLL_mlx - NLL_ora32| <= 2 |NLL_ora16 - NLL_ora32| + 0.01` on all targets
/// and on user-text targets; oracle fp32 NLL <= 3.0 and top-1 >= 40 % on
/// user-text targets (S12a-4: template role tokens are never targets).
#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_nll_en() {
    let gd = gold("p32k8-en");
    let store = e2e_store();
    let vocab = store.model.cfg.vocab_size;
    let mut sess = store.model.new_session();
    let mut got = Vec::new();
    let g = generate(
        &store,
        &mut sess,
        &gd.prompt,
        None,
        0,
        None,
        false,
        &mut |c: &Call| {
            got = read(c.logits);
            Ok(())
        },
    )
    .unwrap();
    let (_, _, l32, l16) = &gd.calls[0];
    let p = gd.prompt.len();
    let mut fails = Vec::new();
    // (nll mlx, nll 32, nll 16, hit32) per target
    let rows: Vec<(f64, f64, f64, bool, bool)> = (0..p - 1)
        .map(|i| {
            let t = gd.prompt[i + 1] as usize;
            let w = i * vocab..(i + 1) * vocab;
            let nll = |v: &[f32]| -log_softmax(&v[w.clone()])[t];
            let hit = super::generate::argmax(&l32[w.clone()]) as usize == t;
            (
                nll(&got),
                nll(l32),
                nll(l16),
                hit,
                !gd.special.contains(&(t as u32)),
            )
        })
        .collect();
    for (name, user_only) in [("all targets", false), ("user-text targets", true)] {
        let sel: Vec<_> = rows.iter().filter(|r| !user_only || r.4).collect();
        let n = sel.len() as f64;
        let mean =
            |f: &dyn Fn(&&(f64, f64, f64, bool, bool)) -> f64| sel.iter().map(f).sum::<f64>() / n;
        let (m, a, b) = (mean(&|r| r.0), mean(&|r| r.1), mean(&|r| r.2));
        let top1 = mean(&|r| f64::from(u8::from(r.3)));
        let bound = 2.0 * (b - a).abs() + 0.01;
        println!(
            "{name} (n {}): NLL mlx {m:.6} ora32 {a:.6} ora16 {b:.6}; |mlx-ora32| {:.3e} <= {bound:.3e}; ora32 top-1 {top1:.3}",
            sel.len(),
            (m - a).abs()
        );
        if (m - a).is_nan() || (m - a).abs() > bound {
            fails.push(format!("G14.5 {name}: |{m} - {a}| > {bound}"));
        }
        if user_only && !(a <= 3.0 && top1 >= 0.4) {
            fails.push(format!("G14.5 oracle quality {name}: NLL {a} top-1 {top1}"));
        }
    }
    check_io(&g, 1, store.model.cfg.n_layers, &mut fails);
    finish("real_e2e_nll_en", &fails);
}

/// Ids from `DSV41_IDS` (a JSON array) when set, else the case's prompt.
fn prompt_ids(gd: &Gold) -> Vec<u32> {
    match std::env::var_os("DSV41_IDS") {
        Some(p) => serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap(),
        None => gd.prompt.clone(),
    }
}

/// Stores `doc` at `hashes/<name>.json` as the baseline, only for a run with
/// no failure and no stored baseline; a stored baseline with the same prompt
/// is never replaced and its `hashes` must match bitwise. Returns whether
/// `doc` became the baseline.
fn store_hashes(name: &str, doc: Value, fails: &mut Vec<String>) -> bool {
    let dir = golden_root().join("hashes");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.json"));
    if let Ok(old) = std::fs::read(&p) {
        let old: Value = serde_json::from_slice(&old).unwrap();
        if old["prompt_ids"] == doc["prompt_ids"] {
            if old["hashes"] != doc["hashes"] {
                fails.push(format!("{name}: logits hashes differ from the stored run"));
            } else {
                println!("{name}: logits hashes == stored run (bitwise)");
            }
            return false;
        }
    }
    if !fails.is_empty() {
        println!("{name}: failed run, no baseline written");
        return false;
    }
    std::fs::write(&p, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
    println!("{name}: hashes written to {}", p.display());
    true
}

/// G14.2: 32 greedy tokens from the prompt; the first ones the oracle made
/// (prefill + 8 steps) must match it except a near tie (after which the
/// texts part and nothing more is compared). Ids + per-step logits hashes
/// go to `hashes/greedy-<case>.json`; EN also keeps its logits (all prompt
/// rows + every step's row) for the chunk-7 gate.
fn real_e2e_greedy(case: &str) {
    let gd = gold(case);
    let prompt = prompt_ids(&gd);
    let store = e2e_store();
    let vocab = store.model.cfg.vocab_size;
    let mut sess = store.model.new_session();
    let (mut hashes, mut rows, mut bad_calls) = (Vec::new(), Vec::<f32>::new(), Vec::new());
    let g = generate(
        &store,
        &mut sess,
        &prompt,
        None,
        31,
        None,
        false,
        &mut |c: &Call| {
            let v = read(c.logits);
            hashes.push(hex_sha(&v[v.len() - vocab..]));
            rows.extend_from_slice(&v);
            if !v.iter().all(|x| x.is_finite()) {
                bad_calls.push(c.index);
            }
            Ok(())
        },
    )
    .unwrap();
    let mut fails = Vec::new();
    if !bad_calls.is_empty() {
        fails.push(format!("non-finite logits in calls {bad_calls:?}"));
    }
    let same_prompt = prompt == gd.prompt;
    let mut parted = None;
    if same_prompt {
        for (j, (&got, &want)) in g.argmax.iter().zip(&gd.greedy).enumerate() {
            if got == want {
                continue;
            }
            let (_, ids, l32, l16) = &gd.calls[j];
            let r = (ids.len() - 1) * vocab..ids.len() * vocab;
            match pick_ok(got, &l32[r.clone()], &l16[r]) {
                Ok(t) => parted = Some(format!("step {j}: {got} vs {want} excused {t:?}")),
                Err(e) => fails.push(format!("G14.2 step {j}: {got} vs oracle {want}: {e}")),
            }
            break;
        }
    }
    println!(
        "real_e2e_greedy {case}: ids {:?}\n  oracle fp32 greedy {:?}; parted: {parted:?}; ttft {:.1} s, decode median {:.2} s/token",
        g.argmax,
        gd.greedy,
        g.ttft_s,
        {
            let mut d = g.decode_s.clone();
            d.sort_by(f64::total_cmp);
            d[d.len() / 2]
        }
    );
    check_io(&g, g.argmax.len(), store.model.cfg.n_layers, &mut fails);
    // EN keeps its logits for the chunk-7 gate, bound to the baseline by sha256
    let keep = case == "p32k8-en" && same_prompt;
    let b: Vec<u8> = if keep {
        rows.iter().flat_map(|x| x.to_le_bytes()).collect()
    } else {
        Vec::new()
    };
    let logits_sha = keep.then(|| hex_sha(&rows));
    let fresh = store_hashes(
        &format!("greedy-{case}"),
        serde_json::json!({
            "prompt_ids": prompt,
            "ids": g.argmax,
            "hashes": hashes,
            "logits_sha256": logits_sha,
            "oracle_prefix_match": same_prompt && parted.is_none(),
            "parted": parted,
            "stats": g.stats,
        }),
        &mut fails,
    );
    if fresh && keep {
        std::fs::write(golden_root().join("hashes/greedy-p32k8-en.logits.f32"), b).unwrap();
    }
    finish(&format!("real_e2e_greedy {case}"), &fails);
}

/// G14.3: the EN prompt in calls of 7 + 8 greedy steps vs the whole-prompt
/// greedy run (`hashes/greedy-p32k8-en.logits.f32`): every row within 1e-5
/// relL2 while the inputs agree, same greedy ids except near ties; the
/// per-call logits hashes are the chunk-7 golden (later builds: bitwise).
#[test]
#[ignore = "real 9 shards + S12 goldens + greedy-p32k8-en logits; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_chunk7_en() {
    let gd = gold("p32k8-en");
    let whole: Vec<f32> = std::fs::read(golden_root().join("hashes/greedy-p32k8-en.logits.f32"))
        .expect("run real_e2e_greedy_en first")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(golden_root().join("hashes/greedy-p32k8-en.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        manifest["logits_sha256"].as_str(),
        Some(hex_sha(&whole).as_str()),
        "whole-prompt logits file does not match its baseline"
    );
    let store = e2e_store();
    let vocab = store.model.cfg.vocab_size;
    let p = gd.prompt.len();
    let mut sess = store.model.new_session();
    let (mut hashes, mut rows, mut bad_calls) = (Vec::new(), Vec::<f32>::new(), Vec::new());
    let g = generate(
        &store,
        &mut sess,
        &gd.prompt,
        Some(7),
        8,
        None,
        false,
        &mut |c: &Call| {
            let v = read(c.logits);
            hashes.push(hex_sha(&v));
            rows.extend_from_slice(&v);
            if !v.iter().all(|x| x.is_finite()) {
                bad_calls.push(c.index);
            }
            Ok(())
        },
    )
    .unwrap();
    let mut fails = Vec::new();
    if !bad_calls.is_empty() {
        fails.push(format!("non-finite logits in calls {bad_calls:?}"));
    }
    // whole-run picks: argmax of row p-1, then of each step row
    let whole_pick =
        |j: usize| super::generate::argmax(&whole[(p - 1 + j) * vocab..(p + j) * vocab]);
    let mut worst = 0f64;
    let n_rows = rows.len() / vocab;
    for r in 0..n_rows {
        let (a, b) = (
            &rows[r * vocab..(r + 1) * vocab],
            &whole[r * vocab..(r + 1) * vocab],
        );
        let e = rel_l2(a, b);
        worst = worst.max(e);
        if e.is_nan() || e > 1e-5 {
            fails.push(format!("G14.3a row {r}: relL2 {e:e} vs whole prompt"));
        }
        if r + 1 >= p {
            let j = r + 1 - p;
            let got = g.argmax[j];
            if got != whole_pick(j) {
                // greedy site: the whole run's own rows are the reference;
                // the oracle's bf16/fp32 rows give the noise floor where they exist
                let msg = format!("G14.3a step {j}: {got} vs whole-prompt {}", whole_pick(j));
                match gd.calls.get(j) {
                    Some((_, ids, l32, l16)) => {
                        let w = (ids.len() - 1) * vocab..ids.len() * vocab;
                        match pick_ok(got, &l32[w.clone()], &l16[w]) {
                            Ok(_) => println!("{msg}: near tie, comparison stops"),
                            Err(e) => fails.push(format!("{msg}: {e}")),
                        }
                    }
                    None => fails.push(msg),
                }
                break;
            }
        }
    }
    println!(
        "real_e2e_chunk7_en: {} calls, {n_rows} rows, worst relL2 vs whole prompt {worst:e}; ids {:?}",
        hashes.len(),
        g.argmax
    );
    check_io(&g, hashes.len(), store.model.cfg.n_layers, &mut fails);
    store_hashes(
        "chunk7-p32k8-en",
        serde_json::json!({"prompt_ids": gd.prompt, "chunk": 7, "ids": g.argmax, "hashes": hashes}),
        &mut fails,
    );
    finish("real_e2e_chunk7_en", &fails);
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_tf_en() {
    real_e2e_tf("p32k8-en");
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_tf_zh() {
    real_e2e_tf("p32k8-zh");
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_tf_code() {
    real_e2e_tf("p32k8-code");
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_greedy_en() {
    real_e2e_greedy("p32k8-en");
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_greedy_zh() {
    real_e2e_greedy("p32k8-zh");
}

#[test]
#[ignore = "real 9 shards + S12 goldens; MLX_ENABLE_TF32=0 MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB"]
fn real_e2e_greedy_code() {
    real_e2e_greedy("p32k8-code");
}

#[test]
fn argmax_takes_the_lowest_index_on_ties() {
    assert_eq!(super::generate::argmax(&[1.0, 3.0, 3.0, -1.0]), 1);
    assert_eq!(super::generate::argmax(&[f32::NEG_INFINITY]), 0);
    // a non-finite logit anywhere never reads as KL 0
    assert!(kl(&[0.0, 1.0, f32::NAN], &[0.0, 1.0, 2.0]).is_nan());
    assert!(kl(&[0.0, 1.0, 2.0], &[0.0, f32::INFINITY, 2.0]).is_nan());
    assert!(kl(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]) < 1e-12);
}
