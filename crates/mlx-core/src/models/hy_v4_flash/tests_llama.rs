//! Real weights against stored llama.cpp runs of the same ids: rendered
//! cases and their ids, teacher-forced logits at every position.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::model::Want;
use super::moe::RowKind;
use super::store::{Options, Store};
use crate::array::DType;
use crate::models::deepseek_v41::chat::{gguf_tokenizer, shard_paths};
use crate::tokenizer::{ChatMessage, ToolDefinition};

/// Rows per call, as llama's `-b 512 -ub 512`.
const CHUNK: usize = 512;

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| panic!("{key} is not set")))
}

fn read_i32(p: &Path) -> Vec<u32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| i32::from_le_bytes(b) as u32)
        .collect()
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

fn write_f32(p: &Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, bytes).unwrap();
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let s: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
    let l = m + s.ln();
    row.iter().map(|&x| x as f64 - l).collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |m, (i, &v)| {
            if v > m.1 { (i, v) } else { m }
        })
        .0
}

/// `HY4_REF_CASES` (a JSON list of `{name, messages, tools?,
/// add_generation_prompt?, reasoning_effort?}`; with an effort the request
/// render the engine uses) rendered with the GGUF template and encoded
/// as the engine encodes a prompt: `<name>.txt` and `<name>.i32` (int32 LE)
/// in `HY4_REF_PROMPTS`.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_render_cases() {
    let paths = shard_paths(&env_path("HY4_DIR")).unwrap();
    let (tok, _) = gguf_tokenizer(&paths[0]).unwrap();
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(env_path("HY4_REF_CASES")).unwrap()).unwrap();
    let out = env_path("HY4_REF_PROMPTS");
    std::fs::create_dir_all(&out).unwrap();
    assert!(!cases.is_empty());
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let messages: Vec<ChatMessage> = serde_json::from_value(c["messages"].clone()).unwrap();
        let tools: Option<Vec<ToolDefinition>> = c
            .get("tools")
            .map(|t| serde_json::from_value(t.clone()).unwrap());
        let agp = c
            .get("add_generation_prompt")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let text = match c.get("reasoning_effort").and_then(Value::as_str) {
            Some(effort) => {
                let cfg = crate::engine::types::ChatConfig {
                    tools,
                    reasoning_effort: Some(effort.to_string()),
                    ..Default::default()
                };
                tok.render_chat_template_with_config(&messages, agp, &cfg, false)
            }
            None => {
                tok.render_chat_template_sync(&messages, Some(agp), tools.as_deref(), None, false)
            }
        }
        .unwrap();
        let ids = tok.encode_sync(&text, Some(false)).unwrap();
        std::fs::write(out.join(format!("{name}.txt")), &text).unwrap();
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.i32")), bytes).unwrap();
        println!("{name}: {} tokens", ids.len());
    }
}

/// Top-1 agreement of two `[rows, vocab]` logit sets.
fn top1_agree(a: &[f32], b: &[f32], vocab: usize) -> f64 {
    let rows = a.len() / vocab;
    let same = a
        .chunks(vocab)
        .zip(b.chunks(vocab))
        .filter(|(x, y)| argmax(x) == argmax(y))
        .count();
    same as f64 / rows as f64
}

/// The top-1 bound of a case: 98 %, or 1 % under llama's agreement with its
/// own second run when that is lower, never under 96 %. Near ties flip on
/// rounding alone; llama against itself (an eval-callback split, nothing else
/// changed) reached 97.8 % on r2-en, its lowest case. The 96 % limit keeps a
/// broken or mismatched second run from lowering the gate.
fn top1_bound(floor: Option<f64>) -> f64 {
    floor.map_or(0.98, |f| (f - 0.01).clamp(0.96, 0.98))
}

/// For each `HY4_REF_NAMES` case: the ids `<HY4_REF_PROMPTS>/<name>.i32`
/// through the flat trunk in calls of 512 rows, every row's logits against
/// llama's `<HY4_LLAMA_LOGITS>/<name>/logits.f32`. Writes the logits and
/// per-position top-1 and KL(llama || ours) to `HY4_REF_OUT`. Passes at
/// mean KL <= 0.02 and top-1 >= `top1_bound` per case, the floor read from
/// `<HY4_LLAMA_FLOOR_LOGITS>/<name>/logits.f32` (a second llama run of the
/// same ids) when that variable is set.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards, stored llama logits and the GPU"]
fn real_teacher_forced_logits_match_llama() {
    let paths = shard_paths(&env_path("HY4_DIR")).unwrap();
    let prompts = env_path("HY4_REF_PROMPTS");
    let llama = env_path("HY4_LLAMA_LOGITS");
    let out = env_path("HY4_REF_OUT");
    std::fs::create_dir_all(&out).unwrap();
    let names: Vec<String> = std::env::var("HY4_REF_NAMES")
        .expect("HY4_REF_NAMES is not set")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert!(!names.is_empty());
    let floors = std::env::var("HY4_LLAMA_FLOOR_LOGITS")
        .ok()
        .map(PathBuf::from);

    let t0 = std::time::Instant::now();
    let store = Store::open(&paths, &Options::from_env(Some(8192), true).unwrap()).unwrap();
    println!("store open {:.1}s", t0.elapsed().as_secs_f64());
    let vocab = store.model.cfg().vocab_size;
    let mut failed = Vec::new();
    for name in &names {
        let ids = read_i32(&prompts.join(format!("{name}.i32")));
        let want = read_f32(&llama.join(name).join("logits.f32"));
        assert_eq!(want.len(), ids.len() * vocab, "{name}: llama rows");

        let t = std::time::Instant::now();
        let mut sess = store.model.new_session(store.ctx).unwrap();
        store.begin_turn(0).unwrap();
        let mut got: Vec<f32> = Vec::with_capacity(want.len());
        for part in ids.chunks(CHUNK) {
            let o = store
                .model
                .forward(&mut sess, part, RowKind::Prefill, Want::ALL, None, None)
                .unwrap();
            let l = o.logits.unwrap().astype(DType::Float32).unwrap();
            got.extend(l.to_float32().unwrap().to_vec());
        }
        store.end_turn().unwrap();
        let secs = t.elapsed().as_secs_f64();
        assert_eq!(got.len(), want.len(), "{name}: our rows");
        write_f32(&out.join(format!("{name}.mlx-logits.f32")), &got);

        let mut agree = 0usize;
        let mut kls = Vec::with_capacity(ids.len());
        let mut rows = Vec::with_capacity(ids.len());
        for (i, (g, w)) in got.chunks(vocab).zip(want.chunks(vocab)).enumerate() {
            assert!(
                g.iter().all(|v| v.is_finite()),
                "{name}: row {i} not finite"
            );
            let (lg, lw) = (log_softmax(g), log_softmax(w));
            let kl: f64 = lw.iter().zip(&lg).map(|(&a, &b)| a.exp() * (a - b)).sum();
            let (tg, tw) = (argmax(g), argmax(w));
            agree += usize::from(tg == tw);
            kls.push(kl);
            rows.push(json!({"pos": i, "kl": kl, "llama_top1": tw, "mlx_top1": tg,
                             "llama_top1_p": lw[tw].exp(), "mlx_p_of_llama_top1": lg[tw].exp()}));
        }
        let floor = floors.as_ref().map(|d| {
            let f = read_f32(&d.join(name).join("logits.f32"));
            assert_eq!(f.len(), want.len(), "{name}: floor rows");
            top1_agree(&want, &f, vocab)
        });
        let bound = top1_bound(floor);
        let n = ids.len() as f64;
        let top1 = agree as f64 / n;
        let mean = kls.iter().sum::<f64>() / n;
        let (max_pos, max) = kls
            .iter()
            .enumerate()
            .fold((0, 0.0f64), |m, (i, &k)| if k > m.1 { (i, k) } else { m });
        let summary = json!({"name": name, "tokens": ids.len(), "seconds": secs,
            "top1_agree": top1, "llama_self_top1": floor, "top1_bound": bound,
            "mean_kl": mean, "max_kl": max, "max_kl_pos": max_pos});
        println!("{summary}");
        std::fs::write(
            out.join(format!("{name}.stats.json")),
            serde_json::to_string_pretty(&json!({"summary": summary, "rows": rows})).unwrap(),
        )
        .unwrap();
        if top1 < bound || mean > 0.02 {
            failed.push(summary);
        }
    }
    assert!(failed.is_empty(), "cases past the bound: {failed:?}");
}

/// Every layer's output streams `[n, hc, d]` f32 for the ids
/// `<HY4_REF_PROMPTS>/<HY4_REF_NAMES>.i32` in one prefill call, written as
/// `<name>.mlx-l_out-<layer>.f32` to `HY4_REF_OUT` (llama's `l_out-<layer>`
/// has the same element order), and each MoE layer's routed expert ids
/// `[n, k]` as `<name>.mlx-routes-<layer>.u32`.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_layer_outputs() {
    let paths = shard_paths(&env_path("HY4_DIR")).unwrap();
    let name = std::env::var("HY4_REF_NAMES").expect("HY4_REF_NAMES is not set");
    let ids = read_i32(&env_path("HY4_REF_PROMPTS").join(format!("{name}.i32")));
    assert!(!ids.is_empty() && ids.len() <= CHUNK);
    let out = env_path("HY4_REF_OUT");
    std::fs::create_dir_all(&out).unwrap();
    let store = Store::open(&paths, &Options::from_env(Some(8192), true).unwrap()).unwrap();
    let mut sess = store.model.new_session(store.ctx).unwrap();
    let mut trace = super::model::Trace {
        keep_out: true,
        ..Default::default()
    };
    store.begin_turn(0).unwrap();
    let o = store
        .model
        .forward(
            &mut sess,
            &ids,
            RowKind::Prefill,
            Want::ALL,
            None,
            Some(&mut trace),
        )
        .unwrap();
    store.end_turn().unwrap();
    assert_eq!(trace.out.len(), store.model.cfg().n_layers);
    for (l, x) in trace.out.iter().enumerate() {
        let v = x
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        write_f32(&out.join(format!("{name}.mlx-l_out-{l}.f32")), &v);
    }
    for (l, r) in trace.routes.iter().enumerate() {
        if let Some(r) = r {
            let bytes: Vec<u8> = r.ids.iter().flat_map(|i| i.to_le_bytes()).collect();
            std::fs::write(out.join(format!("{name}.mlx-routes-{l}.u32")), bytes).unwrap();
        }
    }
    let logits = o.logits.unwrap().astype(DType::Float32).unwrap();
    write_f32(
        &out.join(format!("{name}.mlx-logits.f32")),
        &logits.to_float32().unwrap(),
    );
    println!("{name}: {} tokens, {} layers", ids.len(), trace.out.len());
}

/// llama's `index.tsv` (`call name op type ne0.. file`) in `dir`: tensor
/// name -> its file.
fn llama_index(dir: &Path) -> std::collections::HashMap<String, PathBuf> {
    let text = std::fs::read_to_string(dir.join("index.tsv")).unwrap();
    text.lines()
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            (f[1].to_string(), dir.join(f[8]))
        })
        .collect()
}

/// Mean over rows of `|a - b| / |b|`, `d` values per row (one position's
/// streams).
fn mean_row_rel(a: &[f32], b: &[f32], d: usize) -> f64 {
    let rows = b.len() / d;
    let sum: f64 = a
        .chunks(d)
        .zip(b.chunks(d))
        .map(|(x, y)| {
            let (mut e, mut n) = (0.0f64, 0.0f64);
            for (&p, &q) in x.iter().zip(y) {
                e += (p as f64 - q as f64).powi(2);
                n += (q as f64).powi(2);
            }
            (e / n).sqrt()
        })
        .sum();
    sum / rows as f64
}

/// `real_layer_outputs` with llama's tensors put in: `HY4_LLAMA_TENSORS`
/// holds llama's `l_out-<l>` and `HY4_LLAMA_ROUTES` its `ffn_moe_topk-<l>`
/// for the same ids. Three runs, each layer's output written as
/// `<name>.<run>-l_out-<l>.f32`: `inject` (every layer reads llama's
/// previous layer output), `inject_forced` (and takes llama's experts),
/// `forced` (our own chain on llama's experts, also its logits). Passes when
/// every `inject` layer is within a mean row error of 0.5 % of llama's: a
/// near-tie expert flip moves a few rows, a wrong op moves them all. The
/// routes file comes from another llama run, whose own near ties can land
/// on other experts than the `l_out` run's.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards, stored llama tensors and the GPU"]
fn real_layer_outputs_on_llama_inputs() {
    let paths = shard_paths(&env_path("HY4_DIR")).unwrap();
    let name = std::env::var("HY4_REF_NAMES").expect("HY4_REF_NAMES is not set");
    let ids = read_i32(&env_path("HY4_REF_PROMPTS").join(format!("{name}.i32")));
    assert!(!ids.is_empty() && ids.len() <= CHUNK);
    let out = env_path("HY4_REF_OUT");
    std::fs::create_dir_all(&out).unwrap();
    let tensors = llama_index(&env_path("HY4_LLAMA_TENSORS"));
    let topk = llama_index(&env_path("HY4_LLAMA_ROUTES"));
    let store = Store::open(&paths, &Options::from_env(Some(8192), true).unwrap()).unwrap();
    let cfg = store.model.cfg();
    let (n, hc, d) = (ids.len() as i64, cfg.hc.count as i64, cfg.dim as i64);
    let inputs: Vec<Option<crate::array::MxArray>> = (0..cfg.n_layers)
        .map(|l| {
            (l > 0).then(|| {
                let v = read_f32(&tensors[&format!("l_out-{}", l - 1)]);
                assert_eq!(v.len() as i64, n * hc * d, "llama l_out-{}", l - 1);
                crate::array::MxArray::from_float32(&v, &[n, hc, d]).unwrap()
            })
        })
        .collect();
    let routes: std::collections::HashMap<usize, Vec<u32>> = topk
        .iter()
        .filter_map(|(k, p)| {
            let l: usize = k.strip_prefix("ffn_moe_topk-")?.parse().ok()?;
            Some((l, read_f32(p).iter().map(|&e| e as u32).collect()))
        })
        .collect();
    assert!(!routes.is_empty(), "no ffn_moe_topk tensors");

    let mut worst = (0usize, 0.0f64);
    for run in ["inject", "inject_forced", "forced"] {
        let mut trace = super::model::Trace {
            keep_out: true,
            inputs: if run == "forced" {
                Vec::new()
            } else {
                inputs.clone()
            },
            ..Default::default()
        };
        let forced = if run == "inject" {
            Default::default()
        } else {
            routes.clone()
        };
        let o = super::moe::forced::with(forced, || {
            let mut sess = store.model.new_session(store.ctx).unwrap();
            store.begin_turn(0).unwrap();
            let o = store
                .model
                .forward(
                    &mut sess,
                    &ids,
                    RowKind::Prefill,
                    Want::ALL,
                    None,
                    Some(&mut trace),
                )
                .unwrap();
            store.end_turn().unwrap();
            o
        });
        for (l, x) in trace.out.iter().enumerate() {
            let v = x
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .to_vec();
            if run == "inject" {
                let rel = mean_row_rel(
                    &v,
                    &read_f32(&tensors[&format!("l_out-{l}")]),
                    (hc * d) as usize,
                );
                println!("{name} inject layer {l}: mean row rel {rel:.5}");
                if rel > worst.1 {
                    worst = (l, rel);
                }
            }
            write_f32(&out.join(format!("{name}.{run}-l_out-{l}.f32")), &v);
        }
        let logits = o.logits.unwrap().astype(DType::Float32).unwrap();
        write_f32(
            &out.join(format!("{name}.{run}-logits.f32")),
            &logits.to_float32().unwrap(),
        );
        println!("{name} {run}: {} layers", trace.out.len());
    }
    assert!(
        worst.1 <= 0.005,
        "layer {} on llama's input: mean row rel {:.5} > 0.005",
        worst.0,
        worst.1
    );
}

#[test]
fn top1_bound_follows_llamas_own_floor() {
    assert_eq!(top1_bound(None), 0.98);
    assert_eq!(top1_bound(Some(1.0)), 0.98);
    assert!((top1_bound(Some(0.9784)) - 0.9684).abs() < 1e-12);
    assert_eq!(top1_bound(Some(0.5)), 0.96);
    let (a, b) = ([0.0f32, 1.0, 1.0, 0.0], [0.0f32, 1.0, 0.0, 1.0]);
    assert_eq!(top1_agree(&a, &b, 2), 0.5);
}
