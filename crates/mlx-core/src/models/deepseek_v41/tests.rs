use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;

use super::config::DeepseekV41Config;
use crate::utils::gguf::{GgufMetaValue, parse_gguf, parse_gguf_reader};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/deepseek-v41")
        .join(name)
}

fn tiny_config() -> DeepseekV41Config {
    let gguf = parse_gguf(fixture("tiny-q8_0.gguf")).unwrap();
    DeepseekV41Config::from_gguf(&gguf).unwrap()
}

fn uv(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

/// The tiny GGUF's config equals the oracle's release `config.json` (S2b
/// `oracle.json` header.config), field by field.
#[test]
fn config_tiny_matches_oracle() {
    let c = tiny_config();
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(fixture("oracle.json")).unwrap()).unwrap();
    let o = &oracle["header"]["config"];
    let n = |k: &str| o[k].as_u64().unwrap_or_else(|| panic!("{k}")) as usize;
    let f = |k: &str| o[k].as_f64().unwrap_or_else(|| panic!("{k}")) as f32;
    assert_eq!(o["score_func"], "sqrtsoftplus");
    for (got, key) in [
        (c.vocab_size, "vocab_size"),
        (c.dim, "dim"),
        (c.moe_inter_dim, "moe_inter_dim"),
        (c.n_layers, "n_layers"),
        (c.n_heads, "n_heads"),
        (c.n_routed_experts, "n_routed_experts"),
        (c.n_shared_experts, "n_shared_experts"),
        (c.n_activated_experts, "n_activated_experts"),
        (c.q_lora_rank, "q_lora_rank"),
        (c.head_dim, "head_dim"),
        (c.rope_head_dim, "rope_head_dim"),
        (c.o_groups, "o_groups"),
        (c.o_lora_rank, "o_lora_rank"),
        (c.window_size, "window_size"),
        (c.index_n_heads, "index_n_heads"),
        (c.index_head_dim, "index_head_dim"),
        (c.index_topk, "index_topk"),
        (c.hc_mult, "hc_mult"),
        (c.hc_sinkhorn_iters, "hc_sinkhorn_iters"),
    ] {
        assert_eq!(got, n(key), "{key}");
    }
    for (got, key) in [
        (c.route_scale, "route_scale"),
        (c.norm_eps, "norm_eps"),
        (c.rope_theta, "rope_theta"),
        (c.compress_rope_theta, "compress_rope_theta"),
        (c.hc_eps, "hc_eps"),
    ] {
        assert_eq!(got.to_bits(), f(key).to_bits(), "{key}");
    }
    assert!(c.norm_topk_prob);
    let lim = f("swiglu_limit");
    assert!(
        c.swiglu_clamp_exp
            .iter()
            .chain(&c.swiglu_clamp_shexp)
            .all(|&x| x == lim)
    );
    let y = c.yarn.as_ref().unwrap();
    assert_eq!(y.factor, f("rope_factor"));
    assert_eq!(y.original_seq_len, n("original_seq_len"));
    assert_eq!(y.beta_fast, f("beta_fast"));
    assert_eq!(y.beta_slow, f("beta_slow"));
    let ratios: Vec<u32> = uv(&o["compress_ratios"])[..c.n_layers]
        .iter()
        .map(|&r| r as u32)
        .collect();
    assert_eq!(c.compress_ratios, ratios);
    assert_eq!(c.kv_source_layers, uv(&o["kv_source_layers"]));
    assert_eq!(c.index_source_layers, uv(&o["index_source_layers"]));
    // ratios [0,0,2,2,2,1,1,1], kv [2,5], index [2,5,7]
    let s = |v: &[i64]| -> Vec<Option<usize>> {
        v.iter().map(|&x| (x >= 0).then_some(x as usize)).collect()
    };
    assert_eq!(c.kv_src, s(&[-1, -1, 2, 2, 2, 5, 5, 5]));
    assert_eq!(c.topk_src, s(&[-1, -1, 2, 2, 2, 5, 5, 7]));
    let cand = c.candidate.as_ref().unwrap();
    assert_eq!(cand.source_layer, n("candidate_source_layer"));
    assert_eq!(cand.topk_blocks, n("candidate_topk_blocks"));
    assert_eq!(cand.block_size, n("candidate_block_size"));
    let e = c.engram.as_ref().unwrap();
    assert_eq!(e.layers, uv(&o["engram_layer_ids"]));
    assert_eq!(e.max_ngram, n("engram_max_ngram_size"));
    assert_eq!(e.n_heads, n("engram_n_heads"));
    assert_eq!(e.head_dim, n("engram_head_dim"));
    assert_eq!(e.pad_token_id, n("engram_pad_id"));
    assert_eq!(e.compressed_vocab_size, n("engram_compressed_vocab_size"));
    let rows: Vec<usize> = (0..e.layers.len())
        .map(|i| e.layer_rows(i) as usize)
        .collect();
    assert_eq!(rows, uv(&o["engram_num_embeddings"]));
    assert_eq!(e.token_map.len(), c.vocab_size);
    assert!(c.dspark.is_none(), "the PR converter writes no dspark keys");
}

/// Each bad metadata value is refused; the unedited file still parses.
#[test]
fn config_refuses_bad_metadata() {
    use GgufMetaValue::{ArrayF32, ArrayI32, ArrayU32, ArrayU64, Int32, String as Str, Uint32};
    let mut primes = vec![
        101u64, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    ];
    primes[3] = 1;
    let cases = [
        ("general.architecture", Str("deepseek4".into())),
        ("deepseek41.hash_layer_count", Uint32(1)),
        (
            "deepseek41.attention.compress_ratios",
            ArrayI32(vec![0, 0, 2, 2, 2, 3, 1, 1, 0, 0, 0]),
        ),
        (
            "deepseek41.attention.compress_ratios",
            ArrayI32(vec![0, 0, 2, 2, 2, 1, 1]),
        ),
        (
            "deepseek41.attention.kv_source_layers",
            ArrayU32(vec![1, 2, 5]),
        ),
        (
            "deepseek41.attention.indexer.source_layers",
            ArrayU32(vec![2, 5, 8]),
        ),
        ("deepseek41.attention.kv_source_layers", ArrayU32(vec![2])),
        (
            "deepseek41.attention.indexer.candidate_source_layer",
            Int32(6),
        ),
        ("deepseek41.expert_gating_func", Uint32(1)),
        ("deepseek41.attention.head_count_kv", Uint32(2)),
        ("deepseek41.ple.head_vocab_sizes", ArrayU64(primes)),
        ("deepseek41.engram.pad_token_id", Uint32(512)),
        ("deepseek41.engram.compressed_vocab_size", Uint32(401)),
        ("deepseek41.swiglu_clamp_exp", ArrayF32(vec![10.0; 7])),
    ];
    let mut base = parse_gguf(fixture("tiny-q8_0.gguf")).unwrap();
    DeepseekV41Config::from_gguf(&base).expect("the fixture itself parses");
    let key = "deepseek41.engram.compressed_vocab_size";
    base.metadata.insert(key.into(), Uint32(402));
    DeepseekV41Config::from_gguf(&base).expect("a matching compressed vocab key is accepted");
    for (key, value) in cases {
        let mut g = parse_gguf(fixture("tiny-q8_0.gguf")).unwrap();
        let what = format!("{key} = {value:?}");
        g.metadata.insert(key.into(), value);
        let err = DeepseekV41Config::from_gguf(&g).expect_err(&what);
        println!("refused {what}: {err}");
    }
}

pub(super) fn open_nocache(path: &std::path::Path) -> std::fs::File {
    let f = std::fs::File::open(path).unwrap();
    // SAFETY: valid descriptor; both flags only change caching / readahead.
    #[cfg(target_os = "macos")]
    unsafe {
        use std::os::fd::AsRawFd;
        assert_eq!(libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1), 0);
        assert_eq!(libc::fcntl(f.as_raw_fd(), libc::F_RDAHEAD, 0), 0);
    }
    f
}

/// `want` is one `hdr/headers.json` KV entry; arrays carry `len`, `head` and
/// `full` (null for the 129,280-entry token map: then `len` + `head` prefix).
fn json_matches(meta: &GgufMetaValue, want: &Value) -> bool {
    let nums: Option<Vec<f64>> = match meta {
        GgufMetaValue::ArrayF32(v) => Some(v.iter().map(|&x| f64::from(x)).collect()),
        GgufMetaValue::ArrayU32(v) => Some(v.iter().map(|&x| x as f64).collect()),
        GgufMetaValue::ArrayI32(v) => Some(v.iter().map(|&x| x as f64).collect()),
        GgufMetaValue::ArrayU64(v) => Some(v.iter().map(|&x| x as f64).collect()),
        GgufMetaValue::ArrayI64(v) => Some(v.iter().map(|&x| x as f64).collect()),
        _ => None,
    };
    let as_f32 = matches!(meta, GgufMetaValue::ArrayF32(_) | GgufMetaValue::Float32(_));
    let same = |a: &Value, b: f64| match a.as_u64() {
        Some(x) if !as_f32 => x as f64 == b && b < 9.0e15,
        _ => {
            a.as_f64()
                .map(|a| if as_f32 { f64::from(a as f32) } else { a })
                == Some(b)
        }
    };
    if let Some(got) = nums {
        if want["len"].as_u64() != Some(got.len() as u64) {
            return false;
        }
        let w = want["full"]
            .as_array()
            .or_else(|| want["head"].as_array())
            .unwrap();
        return w.len() <= got.len() && w.iter().zip(&got).all(|(a, &b)| same(a, b));
    }
    match meta {
        GgufMetaValue::String(s) => want.as_str() == Some(s),
        GgufMetaValue::Bool(b) => want.as_bool() == Some(*b),
        GgufMetaValue::Float32(x) => same(want, f64::from(*x)),
        GgufMetaValue::Uint32(x) => want.as_u64() == Some(u64::from(*x)),
        GgufMetaValue::Int32(x) => want.as_i64() == Some(i64::from(*x)),
        GgufMetaValue::Uint64(x) => want.as_u64() == Some(*x),
        _ => false,
    }
}

/// Real file, header bytes only (F_NOCACHE, no readahead, never a tensor
/// byte): every tensor of every present shard equals the 01 inventory, the
/// shard-1 KV equals the raw header parse, and the config has the 01 values.
#[test]
#[ignore = "real DeepSeek-V4.1 GGUF headers; DSV41_DIR, DSV41_HEADERS, DSV41_INVENTORY"]
fn real_header_inventory() {
    let dir = PathBuf::from(std::env::var("DSV41_DIR").unwrap());
    let read_json = |var: &str| -> Value {
        serde_json::from_slice(&std::fs::read(std::env::var(var).unwrap()).unwrap()).unwrap()
    };
    let inv = read_json("DSV41_INVENTORY");
    let headers = read_json("DSV41_HEADERS");
    let shards = inv["shards"].as_array().unwrap();
    let mut by_shard: HashMap<u64, Vec<&Value>> = HashMap::new();
    for t in inv["tensors"].as_array().unwrap() {
        by_shard
            .entry(t["shard"].as_u64().unwrap())
            .or_default()
            .push(t);
    }
    let (mut checked, mut present) = (0usize, 0usize);
    let mut first = None;
    let mut engram_rows = None;
    for (i, s) in shards.iter().enumerate() {
        let path = dir.join(s["file"].as_str().unwrap());
        if !path.exists() {
            println!("shard {} absent: {}", i + 1, path.display());
            continue;
        }
        present += 1;
        let file = open_nocache(&path);
        assert_eq!(
            file.metadata().unwrap().len(),
            s["file_bytes"].as_u64().unwrap(),
            "shard {} size",
            i + 1
        );
        let g = parse_gguf_reader(std::io::BufReader::new(file)).unwrap();
        assert_eq!(
            g.data_offset,
            s["data_start"].as_u64().unwrap(),
            "shard {} data start",
            i + 1
        );
        let want = by_shard.remove(&(i as u64 + 1)).unwrap_or_default();
        assert_eq!(g.tensors.len(), want.len(), "shard {} tensor count", i + 1);
        assert_eq!(want.len() as u64, s["n_tensors"].as_u64().unwrap());
        for (t, w) in g.tensors.iter().zip(&want) {
            assert_eq!(t.name, w["name"].as_str().unwrap());
            assert_eq!(
                t.tensor_type.name(),
                w["type"].as_str().unwrap(),
                "{}",
                t.name
            );
            let shape: Vec<u64> = w["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            assert_eq!(t.dims, shape, "{}", t.name);
            assert_eq!(
                g.data_offset + t.offset,
                w["offset"].as_u64().unwrap(),
                "{}",
                t.name
            );
            assert_eq!(
                t.data_size().unwrap(),
                w["nbytes"].as_u64().unwrap(),
                "{}",
                t.name
            );
            if t.name == "per_layer_token_embd.weight" {
                engram_rows = Some((t.dims[0], t.dims[1]));
            }
            checked += 1;
        }
        if i == 0 {
            first = Some(g);
        }
    }
    let g = first.expect("shard 1 (all KV metadata) must be present");
    let kv = headers[0]["kv"].as_object().unwrap();
    let mut keys = 0;
    for (key, want) in kv.iter().filter(|(k, _)| k.starts_with("deepseek41.")) {
        let got = g
            .metadata
            .get(key)
            .unwrap_or_else(|| panic!("missing {key}"));
        assert!(json_matches(got, want), "{key}: {got:?}");
        keys += 1;
    }
    let c = DeepseekV41Config::from_gguf(&g).unwrap();
    let base: [(usize, usize); 21] = [
        (c.n_layers, 40),
        (c.vocab_size, 129_280),
        (c.dim, 5120),
        (c.n_heads, 64),
        (c.head_dim, 512),
        (c.rope_head_dim, 64),
        (c.q_lora_rank, 1280),
        (c.o_lora_rank, 1024),
        (c.o_groups, 8),
        (c.window_size, 128),
        (c.n_routed_experts, 384),
        (c.n_activated_experts, 6),
        (c.n_shared_experts, 1),
        (c.moe_inter_dim, 2304),
        (c.index_n_heads, 32),
        (c.index_head_dim, 128),
        (c.index_topk, 512),
        (c.hc_mult, 4),
        (c.hc_sinkhorn_iters, 20),
        (c.context_length, 1_048_576),
        (c.yarn.as_ref().unwrap().original_seq_len, 65_536),
    ];
    for (i, (got, want)) in base.iter().enumerate() {
        assert_eq!(got, want, "config value #{i}");
    }
    assert_eq!(c.route_scale, 1.5);
    assert_eq!(c.yarn.as_ref().unwrap().factor, 16.0);
    assert_eq!(c.compress_rope_theta, 160_000.0);
    let ratio = |il: usize| match il {
        0 | 1 => 0,
        2..=19 => 2,
        _ => 1,
    };
    assert_eq!(c.compress_ratios, (0..40).map(ratio).collect::<Vec<u32>>());
    let kv_src: Vec<Option<usize>> = (0..40)
        .map(|il| match il {
            0 | 1 => None,
            _ => Some(
                [2, 8, 14, 20]
                    .into_iter()
                    .filter(|&s| s <= il)
                    .max()
                    .unwrap(),
            ),
        })
        .collect();
    let topk_src: Vec<Option<usize>> = (0..40)
        .map(|il| match il {
            0 | 1 => None,
            _ => Some(
                [2, 8, 14, 20, 24, 28, 32, 36]
                    .into_iter()
                    .filter(|&s| s <= il)
                    .max()
                    .unwrap(),
            ),
        })
        .collect();
    assert_eq!(c.kv_src, kv_src);
    assert_eq!(c.topk_src, topk_src);
    let cand = c.candidate.as_ref().unwrap();
    assert_eq!(
        (cand.source_layer, cand.topk_blocks, cand.block_size),
        (20, 2048, 8)
    );
    let e = c.engram.as_ref().unwrap();
    assert_eq!(e.layers, vec![1, 14]);
    assert_eq!(
        (e.max_ngram, e.n_heads, e.head_dim, e.n_cols()),
        (4, 8, 256, 24)
    );
    assert_eq!(e.compressed_vocab_size, 99_092);
    assert_eq!(e.pad_token_id, 2);
    assert_eq!(e.vocab_size, Some(16_000_000));
    // ple offset chain: contiguous from 0, layer 14 (engram index 1) starts at
    // 384,006,168, and the table holds exactly the summed bucket rows.
    assert_eq!(e.head_offsets[0], 0);
    for c2 in 1..e.head_offsets.len() {
        assert_eq!(
            e.head_offsets[c2],
            e.head_offsets[c2 - 1] + e.head_vocab_sizes[c2 - 1],
            "col {c2}"
        );
    }
    assert_eq!(e.head_offsets[e.n_cols()], 384_006_168);
    let total = e.layer_rows(0) + e.layer_rows(1);
    assert_eq!(total, 768_022_850);
    if let Some((ne0, ne1)) = engram_rows {
        assert_eq!((ne0, ne1), (e.head_dim as u64, total));
    }
    let d = c.dspark.as_ref().unwrap();
    assert_eq!(
        (
            d.block_size,
            d.markov_rank,
            d.expert_count,
            d.expert_used_count
        ),
        (5, 256, 128, 3)
    );
    assert_eq!(d.target_layers, vec![37, 38, 39]);
    println!(
        "real_header_inventory: shards present {present}/9, tensors checked {checked} (1043 when all 9 exist), deepseek41 keys {keys}"
    );
    if present == 9 {
        assert_eq!(checked, 1043);
    }
}

// S9 tiny end-to-end gates
include!("tests_model.rs");
