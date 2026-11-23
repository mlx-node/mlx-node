use std::io::Cursor;
use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::{Attn, DEFAULT_CONTEXT_CAP};
use super::scan::{ENGRAM_TABLE, ExpertClass, MtpLayer, Role, Scan};
use super::test_support::{fixture, j_u64, j_u64s, j_usize, j_usizes, real_scan};
use crate::utils::gguf::{
    GgufFile, GgufMetaValue, GgufTensorInfo, GgufTensorType, parse_gguf_reader,
};

#[derive(Clone)]
enum Kv {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    I32s(Vec<i32>),
    U64s(Vec<u64>),
    F32s(Vec<f32>),
    Bools(Vec<bool>),
    Strs(Vec<String>),
}

const ALIGN: u64 = 32;
const N_TRUNK: usize = 4;
const DIM: u64 = 64;
const VOCAB: u64 = 32;
const LATENT: u64 = 256;
const FF: u64 = 512;
const N_EXP: u64 = 4;
const HC: u64 = 4;
const REAL_MTP_BYTES: u64 = 7_486_176_384;

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

fn put_kv(b: &mut Vec<u8>, key: &str, v: &Kv) {
    put_str(b, key);
    macro_rules! put {
        ($t:expr, $e:expr) => {{
            b.extend(($t as u32).to_le_bytes());
            $e;
        }};
        (arr $t:expr, $v:expr, $f:expr) => {{
            b.extend(9u32.to_le_bytes());
            b.extend(($t as u32).to_le_bytes());
            b.extend(($v.len() as u64).to_le_bytes());
            $v.iter().for_each($f);
        }};
    }
    match v {
        Kv::U32(x) => put!(4, b.extend(x.to_le_bytes())),
        Kv::F32(x) => put!(6, b.extend(x.to_le_bytes())),
        Kv::Bool(x) => put!(7, b.push(u8::from(*x))),
        Kv::Str(s) => put!(8, put_str(b, s)),
        Kv::I32s(v) => put!(arr 5, v, |x| b.extend(x.to_le_bytes())),
        Kv::U64s(v) => put!(arr 10, v, |x| b.extend(x.to_le_bytes())),
        Kv::F32s(v) => put!(arr 6, v, |x| b.extend(x.to_le_bytes())),
        Kv::Bools(v) => put!(arr 7, v, |x| b.push(u8::from(*x))),
        Kv::Strs(v) => put!(arr 8, v, |s| put_str(b, s)),
    }
}

fn tensor_size(name: &str, dims: &[u64], ty: GgufTensorType) -> u64 {
    let info = GgufTensorInfo {
        name: name.into(),
        n_dims: dims.len() as u32,
        dims: dims.to_vec(),
        tensor_type: ty,
        offset: 0,
    };
    info.data_size().unwrap()
}

/// A GGUF header (keys + tensor infos, no data) and the file length its tensors need.
struct Fixture {
    kv: Vec<(String, Kv)>,
    tensors: Vec<(String, Vec<u64>, GgufTensorType)>,
}

impl Fixture {
    fn set(&mut self, key: &str, v: Kv) {
        self.kv.retain(|(k, _)| k != key);
        self.kv.push((key.to_string(), v));
    }
    fn set_arch(&mut self, key: &str, v: Kv) {
        self.set(&format!("hy_v4_flash.{key}"), v);
    }
    fn remove_arch(&mut self, key: &str) {
        self.kv.retain(|(k, _)| k != &format!("hy_v4_flash.{key}"));
    }
    fn tensor(&mut self, name: &str) -> &mut (String, Vec<u64>, GgufTensorType) {
        self.tensors
            .iter_mut()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
    }
    fn bytes_and_len(&self) -> (Vec<u8>, u64) {
        let mut b = Vec::new();
        for w in [0x4655_4747u32, 3] {
            b.extend(w.to_le_bytes());
        }
        for n in [self.tensors.len(), self.kv.len()] {
            b.extend((n as u64).to_le_bytes());
        }
        self.kv.iter().for_each(|(k, v)| put_kv(&mut b, k, v));
        let mut off = 0u64;
        for (name, dims, ty) in &self.tensors {
            off = off.div_ceil(ALIGN) * ALIGN;
            put_str(&mut b, name);
            b.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
            b.extend((*ty as u32).to_le_bytes());
            b.extend(off.to_le_bytes());
            off += tensor_size(name, dims, *ty);
        }
        let data_start = (b.len() as u64).div_ceil(ALIGN) * ALIGN;
        (b, data_start + off)
    }
    fn header_with_len(&self, shrink: u64) -> (GgufFile, u64) {
        let (bytes, len) = self.bytes_and_len();
        let gguf = parse_gguf_reader(Cursor::new(bytes)).unwrap();
        (gguf, len - shrink)
    }
    fn scan(&self) -> napi::Result<Scan> {
        Scan::from_headers(&[PathBuf::from("tiny.gguf")], &[self.header_with_len(0)])
    }
}

fn indexed(il: usize) -> bool {
    matches!(il, 0 | 3 | 4)
}
fn heads(il: usize) -> u64 {
    if indexed(il) { 2 } else { 4 }
}

/// Four trunk layers (I S S I) plus one MTP layer; latent != ff so a swapped
/// expert orientation has other dims. Engram on layer 1, dense FFN on layer 0;
/// down experts Q6_K on layer 1, Q4_K on 2-3, BF16 on the MTP.
fn tiny() -> Fixture {
    use Kv::*;
    let u = |x: u64| U32(x as u32);
    let toks: Vec<String> = (0..VOCAB).map(|i| format!("t{i}")).collect();
    let hc: Vec<i32> = (0..=N_TRUNK).map(|il| heads(il) as i32).collect();
    let swa: Vec<bool> = (0..=N_TRUNK).map(|il| !indexed(il)).collect();
    let ix: Vec<bool> = (0..N_TRUNK).map(indexed).collect();
    let tmap: Vec<i32> = (0..VOCAB as i32).map(|i| i % 20).collect();
    let mut kv = vec![
        ("general.architecture".into(), Str("hy_v4_flash".into())),
        ("tokenizer.ggml.tokens".into(), Strs(toks)),
    ];
    let arch = [
        ("block_count", u(N_TRUNK as u64 + 1)),
        ("nextn_predict_layers", u(1)),
        ("vocab_size", u(VOCAB)),
        ("context_length", u(1 << 20)),
        ("embedding_length", u(DIM)),
        ("feed_forward_length", u(96)),
        ("leading_dense_block_count", u(1)),
        ("attention.head_count_kv", u(1)),
        ("attention.q_lora_rank", u(32)),
        ("attention.kv_lora_rank", u(16)),
        ("attention.key_length", u(24)),
        ("attention.value_length", u(16)),
        ("attention.key_length_mla", u(16)),
        ("attention.value_length_mla", u(12)),
        ("attention.sliding_window", u(8)),
        ("attention.layer_norm_rms_epsilon", F32(1e-5)),
        ("rope.dimension_count", u(8)),
        ("rope.freq_base", F32(1e7)),
        ("rope.freq_base_swa", F32(1e4)),
        ("expert_count", u(N_EXP)),
        ("expert_used_count", u(2)),
        ("expert_shared_count", u(1)),
        ("expert_feed_forward_length", u(FF)),
        ("expert_latent_length", u(LATENT)),
        ("expert_weights_scale", F32(2.5)),
        ("expert_weights_norm", Bool(true)),
        ("expert_gating_func", u(2)),
        ("hyper_connection.count", u(HC)),
        ("hyper_connection.epsilon", F32(1e-6)),
        ("hyper_connection.magnitude", F32(2.0)),
        ("attention.indexer.head_count", u(2)),
        ("attention.indexer.key_length", u(16)),
        ("attention.indexer.top_k", u(8)),
        ("attention.indexer.kpool", u(4)),
        ("attention.indexer.kpool_select_tail", Bool(true)),
        ("ple.ngram_size", u(3)),
        ("ple.heads_per_ngram", u(2)),
        ("embedding_length_per_layer_input", u(32)),
        ("engram.pad_token_id", u(3)),
        ("attention.head_count", I32s(hc)),
        ("attention.sliding_window_pattern", Bools(swa)),
        ("attention.indexer.types", Bools(ix)),
        ("swiglu_clamp_exp", F32s(vec![10.0; N_TRUNK + 1])),
        ("ple.layers", I32s(vec![1])),
        ("ple.layer_multipliers", U64s(vec![7, 11, 13])),
        ("ple.head_offsets", U64s(vec![0, 5, 12, 23])),
        ("ple.head_vocab_sizes", U64s(vec![5, 7, 11, 13])),
        ("engram.token_map", I32s(tmap)),
    ];
    for (k, v) in arch {
        kv.push((format!("hy_v4_flash.{k}"), v));
    }
    Fixture {
        kv,
        tensors: tiny_tensors(),
    }
}

fn tiny_tensors() -> Vec<(String, Vec<u64>, GgufTensorType)> {
    use GgufTensorType::{BF16, F32, MXFP8, Q4K, Q6K};
    let mut t: Vec<(String, Vec<u64>, GgufTensorType)> = vec![
        ("token_embd.weight".into(), vec![DIM, VOCAB], F32),
        ("output_norm.weight".into(), vec![DIM], F32),
        ("output.weight".into(), vec![DIM, VOCAB], F32),
        ("output_hc_fn.weight".into(), vec![HC * DIM, HC], F32),
        ("output_hc_base.weight".into(), vec![HC], F32),
        ("output_hc_scale.weight".into(), vec![1], F32),
        (ENGRAM_TABLE.into(), vec![32, 36], MXFP8),
    ];
    for il in 0..=N_TRUNK {
        let (mtp, h) = (il == N_TRUNK, heads(il));
        let mut add = |list: &[(&str, &[u64])]| {
            for &(n, d) in list {
                t.push((format!("blk.{il}.{n}.weight"), d.to_vec(), F32));
            }
        };
        add(&[
            ("attn_norm", &[DIM]),
            ("attn_sinks", &[h]),
            ("attn_q_a", &[DIM, 32]),
            ("attn_q_a_norm", &[32]),
            ("attn_q_b", &[32, h * 16]),
            ("attn_kv_a_mqa", &[DIM, 24]),
            ("attn_kv_a_norm", &[16]),
            ("attn_k_b", &[8, 16, h]),
            ("attn_v_b", &[16, 12, h]),
            ("attn_output", &[h * 12, DIM]),
            ("attn_gate", &[DIM, h * 12]),
            ("post_attention_norm", &[DIM]),
            ("ffn_norm", &[DIM]),
            ("post_ffw_norm", &[DIM]),
        ]);
        if mtp {
            add(&[
                ("nextn.enorm", &[DIM]),
                ("nextn.hnorm", &[DIM]),
                ("nextn.eh_proj", &[2 * DIM, DIM]),
                ("nextn.shared_head_norm", &[DIM]),
            ]);
        } else {
            add(&[
                ("hc_attn_fn", &[HC * DIM, 2 * HC]),
                ("hc_attn_base", &[2 * HC]),
                ("hc_attn_scale", &[2]),
                ("hc_ffn_fn", &[HC * DIM, 2 * HC]),
                ("hc_ffn_base", &[2 * HC]),
                ("hc_ffn_scale", &[2]),
            ]);
        }
        if indexed(il) {
            add(&[
                ("indexer.attn_k", &[DIM, 16]),
                ("indexer.k_norm", &[16]),
                ("indexer_compressor_value", &[16, 16]),
                ("indexer_compressor_gate", &[16, 16]),
                ("indexer_compressor_ape", &[16, 4]),
                ("indexer_compressor_norm", &[16]),
                ("indexer.attn_q_b", &[32, 32]),
                ("indexer.proj", &[DIM, 2]),
            ]);
        }
        if il == 1 {
            add(&[
                ("engram_kv", &[4 * 32, DIM * (HC + 1)]),
                ("engram_kv_b", &[DIM, HC + 1]),
                ("engram_gate", &[DIM, HC]),
                ("engram_lookup_norm", &[32, 4]),
                ("engram_pre_gamma", &[DIM]),
                ("engram_hc_scale", &[DIM, HC]),
            ]);
        }
        if il == 0 {
            add(&[
                ("ffn_gate", &[DIM, 96]),
                ("ffn_down", &[96, DIM]),
                ("ffn_up", &[DIM, 96]),
            ]);
        } else {
            add(&[
                ("ffn_gate_inp", &[DIM, N_EXP]),
                ("ffn_routed_down", &[DIM, LATENT]),
                ("ffn_routed_up", &[LATENT, DIM]),
                ("ffn_gate_shexp", &[DIM, FF]),
                ("ffn_down_shexp", &[FF, DIM]),
                ("ffn_up_shexp", &[DIM, FF]),
            ]);
            t.push((format!("blk.{il}.exp_probs_b.bias"), vec![N_EXP], F32));
            let (gu, dn) = match (il, mtp) {
                (1, _) => (Q4K, Q6K),
                (_, true) => (BF16, BF16),
                _ => (Q4K, Q4K),
            };
            for (n, d, ty) in [
                ("ffn_gate_exps", [LATENT, FF, N_EXP], gu),
                ("ffn_up_exps", [LATENT, FF, N_EXP], gu),
                ("ffn_down_exps", [FF, LATENT, N_EXP], dn),
            ] {
                t.push((format!("blk.{il}.{n}.weight"), d.to_vec(), ty));
            }
        }
    }
    t
}

fn err(f: &Fixture) -> String {
    match f.scan() {
        Ok(_) => panic!("scan accepted a fixture it must refuse"),
        Err(e) => e.reason,
    }
}

#[test]
fn tiny_fixture_scans() {
    use Attn::{Indexed as I, Sliding as S};
    use ExpertClass::{Bf16, Q4kDown, Q6kDown};
    let s = tiny().scan().unwrap();
    let c = &s.cfg;
    macro_rules! col {
        ($f:expr) => {
            c.layers.iter().map($f).collect::<Vec<_>>()
        };
    }
    assert_eq!((c.n_layers, c.layers.len(), c.mtp().len()), (4, 5, 1));
    assert_eq!(col!(|l| l.attn), [I, S, S, I, I]);
    assert_eq!(col!(|l| l.rope_theta), [1e7, 1e4, 1e4, 1e7, 1e7]);
    assert_eq!(col!(|l| l.n_heads), [2, 4, 4, 2, 2]);
    assert_eq!(col!(|l| l.dense_ffn), [true, false, false, false, false]);
    assert_eq!(col!(|l| l.engram), [None, Some(0), None, None, None]);
    assert!(c.layers[4].mtp && !c.layers[3].mtp);
    assert_eq!(
        (c.mla.qk_nope_dim, c.mla.kv_row(), c.mla.scale),
        (8, 24, 0.25)
    );
    assert_eq!((c.indexer.n_top_pool(), c.engram_rows()), (1, 36));
    assert_eq!(c.engram.compressed_vocab_size, 20);
    let cls: Vec<_> = s
        .experts
        .iter()
        .map(|e| e.as_ref().map(|e| e.class))
        .collect();
    assert_eq!(
        cls,
        [
            None,
            Some(Q6kDown),
            Some(Q4kDown),
            Some(Q4kDown),
            Some(Bf16)
        ]
    );
    assert_eq!(s.with_role(Role::Expert).count(), 12);
    assert_eq!(s.with_role(Role::EngramTable).count(), 1);
    assert!(s.with_role(Role::Mtp).all(|(n, _)| n.starts_with("blk.4.")));
    assert!(
        s.with_role(Role::Dense)
            .all(|(n, _)| !n.starts_with("blk.4."))
    );
    assert_eq!(s.engram_table.info.dims, [32, 36]);
    assert_eq!(s.tensors.len(), tiny_tensors().len());

    let pfx = format!("blk.{N_TRUNK}.");
    let mtp_bytes: u64 = tiny_tensors()
        .iter()
        .filter(|t| t.0.starts_with(&pfx))
        .map(|(n, d, ty)| tensor_size(n, d, *ty))
        .sum();
    assert!(mtp_bytes > 0);
    let want_mtp = MtpLayer {
        layer: N_TRUNK,
        bytes: mtp_bytes,
    };
    assert_eq!(s.mtp, Some(want_mtp));
    assert_eq!(s.mtp_experts().map(|e| e.class), Some(Bf16));
    assert_eq!(s.trunk_experts().len(), N_TRUNK);
}

/// The same file written as by a converter run with `--no-mtp`.
fn without_mtp() -> Fixture {
    let mut f = tiny();
    f.set_arch("block_count", Kv::U32(N_TRUNK as u32));
    f.remove_arch("nextn_predict_layers");
    let hc: Vec<i32> = (0..N_TRUNK).map(|i| heads(i) as i32).collect();
    let swa: Vec<bool> = (0..N_TRUNK).map(|i| !indexed(i)).collect();
    f.set_arch("attention.head_count", Kv::I32s(hc));
    f.set_arch("attention.sliding_window_pattern", Kv::Bools(swa));
    f.set_arch("swiglu_clamp_exp", Kv::F32s(vec![10.0; N_TRUNK]));
    let pfx = format!("blk.{N_TRUNK}.");
    f.tensors.retain(|t| !t.0.starts_with(&pfx));
    f
}

#[test]
fn a_file_without_the_mtp_layer_scans_without_it() {
    let s = without_mtp().scan().unwrap();
    assert_eq!((s.cfg.n_layers, s.cfg.layers.len()), (N_TRUNK, N_TRUNK));
    assert!(s.cfg.mtp().is_empty() && s.mtp.is_none() && s.mtp_experts().is_none());
    assert_eq!(s.trunk_experts().len(), N_TRUNK);
    assert_eq!(s.with_role(Role::Mtp).count(), 0);

    let mut f = without_mtp();
    let pfx = format!("blk.{N_TRUNK}.");
    let mtp = tiny_tensors().into_iter().filter(|t| t.0.starts_with(&pfx));
    f.tensors.extend(mtp);
    assert!(err(&f).contains("is not part of the hy_v4_flash layout"));
}

#[test]
fn max_context_is_the_least_of_file_cap_and_option() {
    let s = tiny().scan().unwrap();
    assert_eq!(
        (s.cfg.context_length, DEFAULT_CONTEXT_CAP),
        (1 << 20, 262_144)
    );
    for (req, want) in [
        (None, 262_144),
        (Some(1 << 21), 262_144),
        (Some(262_145), 262_144),
        (Some(100_000), 100_000),
        (Some(4096), 4096),
    ] {
        assert_eq!(s.cfg.max_context(req).unwrap(), want);
    }
    assert!(s.cfg.max_context(Some(0)).is_err());

    let mut f = tiny();
    f.set_arch("context_length", Kv::U32(32_768));
    let c = f.scan().unwrap().cfg;
    for (req, want) in [
        (None, 32_768),
        (Some(1 << 20), 32_768),
        (Some(16_384), 16_384),
    ] {
        assert_eq!(c.max_context(req).unwrap(), want);
    }
    f.set_arch("context_length", Kv::U32(0));
    assert!(err(&f).contains("context_length is 0"));
}

#[test]
fn swapped_expert_orientation_is_refused() {
    for (name, dims) in [
        ("blk.2.ffn_gate_exps.weight", vec![FF, LATENT, N_EXP]),
        ("blk.2.ffn_up_exps.weight", vec![FF, LATENT, N_EXP]),
        ("blk.1.ffn_down_exps.weight", vec![LATENT, FF, N_EXP]),
        ("blk.3.ffn_routed_down.weight", vec![LATENT, DIM]),
        ("blk.3.ffn_routed_up.weight", vec![DIM, LATENT]),
        ("blk.0.attn_k_b.weight", vec![16, 8, 2]),
    ] {
        let mut f = tiny();
        f.tensor(name).1 = dims;
        let e = err(&f);
        assert!(e.contains(name) && e.contains("ggml dims"), "{name}: {e}");
    }
}

#[test]
fn unsupported_metadata_is_refused() {
    use Kv::*;
    let hc_no: Vec<i32> = (0..N_TRUNK).map(|i| heads(i) as i32).collect();
    let hc_z: Vec<i32> = (0..=N_TRUNK)
        .map(|i| if i == N_TRUNK { 0 } else { heads(i) as i32 })
        .collect();
    let (hc, ix, swa) = (
        "attention.head_count",
        "attention.indexer.types",
        "attention.sliding_window_pattern",
    );
    let cases: &[(&str, Option<Kv>, &str)] = &[
        ("attention.head_count_kv", Some(U32(2)), "head_count_kv"),
        ("expert_gating_func", Some(U32(1)), "sigmoid"),
        ("attention.indexer.top_k", Some(U32(4)), "tail group"),
        ("attention.indexer.kpool", Some(U32(1)), "pool size"),
        (
            "rope.scaling.type",
            Some(Str("yarn".into())),
            "rope scaling",
        ),
        (
            "general.architecture",
            Some(Str("deepseek41".into())),
            "general.architecture",
        ),
        ("nextn_predict_layers", Some(U32(2)), "MTP layers"),
        (
            hc,
            Some(I32s(hc_no)),
            "attention.head_count has 4 entries, need 5",
        ),
        (hc, Some(I32s(hc_z)), "layer 4: head_count is 0"),
        (swa, None, "sliding_window_pattern"),
        (
            ix,
            Some(Bools(vec![true, true, false, true])),
            "cannot own an indexer",
        ),
        (
            ix,
            Some(Bools(vec![false, false, false, true])),
            "neither sliding nor indexed",
        ),
        (ix, Some(Bools(vec![true; 5])), "indexer.types has 5"),
        ("attention.key_length", Some(U32(16)), "key_length"),
        (
            "ple.head_offsets",
            Some(U64s(vec![0, 5, 13, 24])),
            "not contiguous",
        ),
        ("engram.token_map", Some(I32s(vec![0; 31])), "token_map"),
        ("ple.layers", Some(I32s(vec![4])), "engram layers"),
    ];
    for (key, val, want) in cases {
        let mut f = tiny();
        match val {
            Some(v) if key.starts_with("general.") => f.set(key, v.clone()),
            Some(v) => f.set_arch(key, v.clone()),
            None => f.remove_arch(key),
        }
        let e = err(&f);
        assert!(e.contains(want), "{key}: {e}");
    }
}

#[test]
fn tensor_set_must_match_the_layout() {
    use GgufTensorType::{BF16, F32, Q4K};
    let mut f = tiny();
    f.tensors
        .push(("blk.2.unknown.weight".into(), vec![4], F32));
    assert!(err(&f).contains("blk.2.unknown.weight"));

    let mut f = tiny();
    f.tensors.retain(|t| t.0 != "blk.3.attn_sinks.weight");
    assert!(err(&f).contains("blk.3.attn_sinks.weight' missing"));

    let mut f = tiny();
    f.tensors.retain(|t| t.0 != "blk.2.exp_probs_b.bias");
    assert!(f.scan().is_ok(), "exp_probs_b is optional");

    for (name, ty, want) in [
        (
            "blk.4.ffn_down_exps.weight",
            Q4K,
            "layer 4 (MTP): expert types",
        ),
        ("blk.2.ffn_down_exps.weight", BF16, "layer 2: expert types"),
        (ENGRAM_TABLE, BF16, "type 43"),
    ] {
        let mut f = tiny();
        f.tensor(name).2 = ty;
        assert!(err(&f).contains(want), "{name}");
    }
}

#[test]
fn truncated_file_is_refused() {
    let short = Scan::from_headers(&[PathBuf::from("tiny.gguf")], &[tiny().header_with_len(1)]);
    let e = short.err().map(|e| e.reason).unwrap_or_default();
    assert!(e.contains("truncated") || e.contains("Engram table"), "{e}");
}

#[test]
fn bool_arrays_survive_the_header_walk() {
    let (bytes, _) = tiny().bytes_and_len();
    let gguf = parse_gguf_reader(Cursor::new(bytes)).unwrap();
    let swa = gguf
        .metadata
        .get("hy_v4_flash.attention.sliding_window_pattern")
        .and_then(GgufMetaValue::as_bool_array)
        .unwrap();
    let ix = gguf
        .metadata
        .get("hy_v4_flash.attention.indexer.types")
        .and_then(GgufMetaValue::as_bool_array)
        .unwrap();
    assert_eq!(swa, &[false, true, true, false, false]);
    assert_eq!(ix, &[true, false, false, true]);
    let bool_arrays = gguf
        .metadata
        .values()
        .filter(|v| v.as_bool_array().is_some())
        .count();
    assert_eq!((bool_arrays, gguf.tensors.len()), (2, tiny_tensors().len()));
}

fn hex_bits(v: &Value) -> u32 {
    u32::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

fn class_name(c: ExpertClass) -> &'static str {
    match c {
        ExpertClass::Q6kDown => "A",
        ExpertClass::Q4kDown => "B",
        ExpertClass::Bf16 => "MTP_BF16",
    }
}

fn type_name(t: GgufTensorType) -> &'static str {
    match t {
        GgufTensorType::Q4K => "Q4_K",
        GgufTensorType::Q6K => "Q6_K",
        GgufTensorType::BF16 => "BF16",
        _ => "other",
    }
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_headers_match_the_oracle_config() {
    let (s, want) = (real_scan(), fixture("config.json"));
    let (c, sc, fb) = (&s.cfg, &want["scalars"], &want["f32_bits"]);
    let (d, e) = (&want["derived"], &want["engram"]);
    let jb = |v: &Value, k: &str| v[k].as_bool().unwrap();
    let jf = |v: &Value, k: &str| v[k].as_f64().unwrap();
    for (got, k) in [
        (c.layers.len(), "block_count"),
        (c.mtp().len(), "nextn_predict_layers"),
        (c.vocab_size, "vocab_size"),
        (c.context_length, "context_length"),
        (c.dim, "embedding_length"),
        (c.dense_ff, "feed_forward_length"),
        (c.n_dense_lead, "leading_dense_block_count"),
        (c.window, "attention.sliding_window"),
        (c.mla.q_lora_rank, "attention.q_lora_rank"),
        (c.mla.kv_lora_rank, "attention.kv_lora_rank"),
        (c.mla.kv_row(), "attention.key_length"),
        (c.mla.kv_lora_rank, "attention.value_length"),
        (c.mla.head_dim_qk, "attention.key_length_mla"),
        (c.mla.head_dim_v, "attention.value_length_mla"),
        (c.mla.rope_dim, "rope.dimension_count"),
        (c.moe.n_experts, "expert_count"),
        (c.moe.n_used, "expert_used_count"),
        (c.moe.n_shared, "expert_shared_count"),
        (c.moe.ff, "expert_feed_forward_length"),
        (c.moe.latent, "expert_latent_length"),
        (c.hc.count, "hyper_connection.count"),
        (c.indexer.n_heads, "attention.indexer.head_count"),
        (c.indexer.head_dim, "attention.indexer.key_length"),
        (c.indexer.top_k, "attention.indexer.top_k"),
        (c.indexer.kpool, "attention.indexer.kpool"),
    ] {
        assert_eq!(got, j_usize(&sc[k]), "{k}");
    }
    assert_eq!(c.moe.norm_topk, jb(sc, "expert_weights_norm"));
    assert_eq!(
        c.indexer.select_tail,
        jb(sc, "attention.indexer.kpool_select_tail")
    );
    for (got, k) in [
        (c.norm_eps, "attention.layer_norm_rms_epsilon"),
        (c.moe.route_scale, "expert_weights_scale"),
        (c.hc.eps, "hyper_connection.epsilon"),
        (c.hc.magnitude, "hyper_connection.magnitude"),
        (c.rope_theta, "rope.freq_base"),
        (c.rope_theta_swa, "rope.freq_base_swa"),
    ] {
        assert_eq!(got.to_bits(), hex_bits(&fb[k]), "{k}");
    }

    let lof = |a: Attn| {
        (0..c.n_layers)
            .filter(|&i| c.layers[i].attn == a)
            .collect::<Vec<_>>()
    };
    assert_eq!(lof(Attn::Indexed), j_usizes(&d["indexed_layers"]));
    assert_eq!(lof(Attn::Sliding), j_usizes(&d["sliding_layers"]));
    let mtp_ls: Vec<usize> = (c.n_layers..c.layers.len()).collect();
    assert_eq!(mtp_ls, j_usizes(&d["mtp_layers"]));
    assert_eq!(f64::from(c.mla.scale), jf(d, "attn_scale"));
    assert_eq!(c.indexer.n_top_pool(), j_usize(&d["indexer_n_top_pool"]));
    let ws = jf(d, "indexer_weight_scale");
    assert!((f64::from(c.indexer.weight_scale()) - ws).abs() <= ws * 1e-6);
    assert_eq!(
        c.max_context(None).unwrap(),
        j_usize(&d["context_target_v1"])
    );
    assert_eq!(c.context_length, j_usize(&d["context_length_gguf"]));

    let mtp = s.mtp.as_ref().unwrap();
    assert_eq!((mtp.layer, mtp.bytes), (c.n_layers, REAL_MTP_BYTES));
    let pfx = format!("blk.{}.", mtp.layer);
    let mut span = (u64::MAX, 0u64);
    for (name, ent) in s.tensors.iter().filter(|(n, _)| n.starts_with(&pfx)) {
        assert_eq!(ent.loc.shard, 0, "{name}");
        let at = ent.loc.abs_offset().unwrap();
        let end = at + ent.loc.info.data_size().unwrap();
        span = (span.0.min(at), span.1.max(end));
    }
    assert_eq!(span.1 - span.0, REAL_MTP_BYTES);
    assert_eq!(s.mtp_experts().unwrap().class, ExpertClass::Bf16);
    assert_eq!((c.mtp()[0].n_heads, c.mtp()[0].attn), (32, Attn::Indexed));

    for (il, w) in want["layers"].as_array().unwrap().iter().enumerate() {
        let l = &c.layers[il];
        let attn = if w["attn"] == "I" {
            Attn::Indexed
        } else {
            Attn::Sliding
        };
        assert_eq!(
            (j_usize(&w["il"]), l.attn, l.n_heads),
            (il, attn, j_usize(&w["heads"]))
        );
        assert_eq!(f64::from(l.rope_theta), jf(w, "rope_theta"));
        assert_eq!(f64::from(l.swiglu_limit), jf(w, "swiglu_clamp_exp"));
        assert_eq!(l.mtp, jb(w, "mtp"));
        assert_eq!(l.dense_ffn, w["ffn"] == "dense");
        assert_eq!(l.engram.is_some(), jb(w, "engram"));
        let has_ix = s.opt(&format!("blk.{il}.indexer.attn_k.weight")).is_some();
        assert_eq!(has_ix, jb(w, "has_indexer_tensors"));
    }

    let g = &c.engram;
    assert_eq!(g.layers, j_usizes(&e["layers"]));
    for (got, k) in [
        (g.max_ngram, "ngram_size"),
        (g.n_heads, "heads_per_ngram"),
        (g.n_cols(), "n_cols"),
        (g.head_dim, "head_dim"),
        (g.pad_token_id, "pad_token_id"),
        (g.token_map.len(), "token_map_len"),
    ] {
        assert_eq!(got, j_usize(&e[k]), "engram.{k}");
    }
    assert_eq!(g.multipliers, j_u64s(&e["multipliers"]));
    assert_eq!(g.head_offsets, j_u64s(&e["head_offsets"]));
    assert_eq!(g.head_vocab_sizes, j_u64s(&e["head_vocab_sizes"]));
    let rows: Vec<u64> = (0..g.layers.len()).map(|i| g.layer_rows(i)).collect();
    assert_eq!(rows, j_u64s(&e["rows_per_layer"]));
    assert_eq!(c.engram_rows(), j_u64(&e["total_rows"]));
    assert_eq!(
        u64::from(g.token_map[g.pad_token_id]),
        j_u64(&e["token_map_pad"])
    );
    assert_eq!(
        g.compressed_vocab_size as u64,
        j_u64(&e["token_map_max_all"]) + 1
    );
    let mut h = Sha256::new();
    g.token_map.iter().for_each(|&x| h.update(x.to_le_bytes()));
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, e["token_map_sha256_i32le"].as_str().unwrap());

    let t = &s.engram_table;
    assert_eq!(t.shard, 1);
    assert_eq!(t.info.dims, j_u64s(&e["table_dims_ggml"]));
    assert_eq!(t.info.tensor_type as u32, j_u64(&e["table_type_id"]) as u32);
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_expert_classes_and_extents_match_the_oracle() {
    let s = real_scan();
    let (classes, extents) = (
        fixture("expert-classes.json"),
        fixture("expert-extents.json"),
    );
    let lof = |c: ExpertClass| -> Vec<usize> {
        let ok = |i: usize| s.experts[i].as_ref().is_some_and(|e| e.class == c);
        (0..s.cfg.n_layers).filter(|&i| ok(i)).collect()
    };
    assert_eq!(
        lof(ExpertClass::Q6kDown),
        j_usizes(&classes["q6k_down_layers"])
    );
    assert_eq!(
        lof(ExpertClass::Q4kDown),
        j_usizes(&classes["q4k_down_layers"])
    );
    for w in classes["per_layer"].as_array().unwrap() {
        let il = j_usize(&w["il"]);
        let got = class_name(s.experts[il].as_ref().unwrap().class);
        assert_eq!(got, w["class"].as_str().unwrap(), "layer {il}");
    }
    assert!(s.experts[0].is_none());

    let n_exp = s.cfg.moe.n_experts as u64;
    for w in extents["layers"].as_array().unwrap() {
        let il = j_usize(&w["il"]);
        let e = s.experts[il].as_ref().unwrap();
        assert_eq!(class_name(e.class), w["class"].as_str().unwrap());
        for (key, loc) in [("gate", &e.gate), ("up", &e.up), ("down", &e.down)] {
            let (x, what) = (&w[key], format!("layer {il} {key}"));
            let bytes = loc.info.data_size().unwrap();
            assert_eq!(
                type_name(loc.info.tensor_type),
                x["type"].as_str().unwrap(),
                "{what}"
            );
            assert_eq!(loc.info.dims, j_u64s(&x["ggml_dims"]), "{what}");
            assert_eq!(loc.shard as u64, j_u64(&x["shard"]), "{what}");
            assert_eq!(loc.info.offset, j_u64(&x["rel_offset"]), "{what}");
            assert_eq!(loc.abs_offset().unwrap(), j_u64(&x["abs_offset"]), "{what}");
            assert_eq!(bytes, j_u64(&x["bytes"]), "{what}");
            assert_eq!(bytes / n_exp, j_u64(&x["per_expert_bytes"]), "{what}");
        }
    }
}
