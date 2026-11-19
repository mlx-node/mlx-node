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

/// For each `HY4_REF_NAMES` case: the ids `<HY4_REF_PROMPTS>/<name>.i32`
/// through the flat trunk in calls of 512 rows, every row's logits against
/// llama's `<HY4_LLAMA_LOGITS>/<name>/logits.f32`. Writes the logits and
/// per-position top-1 and KL(llama || ours) to `HY4_REF_OUT`. Passes at
/// top-1 >= 98 % and mean KL <= 0.02 per case.
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
        let n = ids.len() as f64;
        let top1 = agree as f64 / n;
        let mean = kls.iter().sum::<f64>() / n;
        let (max_pos, max) = kls
            .iter()
            .enumerate()
            .fold((0, 0.0f64), |m, (i, &k)| if k > m.1 { (i, k) } else { m });
        let summary = json!({"name": name, "tokens": ids.len(), "seconds": secs,
            "top1_agree": top1, "mean_kl": mean, "max_kl": max, "max_kl_pos": max_pos});
        println!("{summary}");
        std::fs::write(
            out.join(format!("{name}.stats.json")),
            serde_json::to_string_pretty(&json!({"summary": summary, "rows": rows})).unwrap(),
        )
        .unwrap();
        if top1 < 0.98 || mean > 0.02 {
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
