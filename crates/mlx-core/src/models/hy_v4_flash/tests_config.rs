use std::io::Cursor;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::{Attn, DEFAULT_CONTEXT_CAP};
use super::header::{self, Header};
use super::scan::{ENGRAM_TABLE, ExpertClass, MtpLayer, Role, Scan};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

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

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

fn put_array(b: &mut Vec<u8>, elem: u32, len: usize) {
    b.extend(9u32.to_le_bytes());
    b.extend(elem.to_le_bytes());
    b.extend((len as u64).to_le_bytes());
}

fn put_kv(b: &mut Vec<u8>, key: &str, v: &Kv) {
    put_str(b, key);
    match v {
        Kv::U32(x) => {
            b.extend(4u32.to_le_bytes());
            b.extend(x.to_le_bytes());
        }
        Kv::F32(x) => {
            b.extend(6u32.to_le_bytes());
            b.extend(x.to_le_bytes());
        }
        Kv::Bool(x) => {
            b.extend(7u32.to_le_bytes());
            b.push(u8::from(*x));
        }
        Kv::Str(s) => {
            b.extend(8u32.to_le_bytes());
            put_str(b, s);
        }
        Kv::I32s(v) => {
            put_array(b, 5, v.len());
            v.iter().for_each(|x| b.extend(x.to_le_bytes()));
        }
        Kv::U64s(v) => {
            put_array(b, 10, v.len());
            v.iter().for_each(|x| b.extend(x.to_le_bytes()));
        }
        Kv::F32s(v) => {
            put_array(b, 6, v.len());
            v.iter().for_each(|x| b.extend(x.to_le_bytes()));
        }
        Kv::Bools(v) => {
            put_array(b, 7, v.len());
            v.iter().for_each(|x| b.push(u8::from(*x)));
        }
        Kv::Strs(v) => {
            put_array(b, 8, v.len());
            v.iter().for_each(|s| put_str(b, s));
        }
    }
}

/// A GGUF header (keys + tensor infos, no data) and the file length its
/// tensors need.
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
        let key = format!("hy_v4_flash.{key}");
        self.kv.retain(|(k, _)| *k != key);
    }

    fn tensor(&mut self, name: &str) -> &mut (String, Vec<u64>, GgufTensorType) {
        self.tensors
            .iter_mut()
            .find(|t| t.0 == name)
            .unwrap_or_else(|| panic!("fixture has no tensor {name}"))
    }

    fn bytes_and_len(&self) -> (Vec<u8>, u64) {
        let mut b = Vec::new();
        b.extend(0x4655_4747u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend((self.tensors.len() as u64).to_le_bytes());
        b.extend((self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            put_kv(&mut b, k, v);
        }
        let mut offset = 0u64;
        for (name, dims, ty) in &self.tensors {
            offset = offset.div_ceil(ALIGN) * ALIGN;
            put_str(&mut b, name);
            b.extend((dims.len() as u32).to_le_bytes());
            dims.iter().for_each(|d| b.extend(d.to_le_bytes()));
            b.extend((*ty as u32).to_le_bytes());
            b.extend(offset.to_le_bytes());
            let info = GgufTensorInfo {
                name: name.clone(),
                n_dims: dims.len() as u32,
                dims: dims.clone(),
                tensor_type: *ty,
                offset,
            };
            offset += info.data_size().unwrap();
        }
        let data_start = (b.len() as u64).div_ceil(ALIGN) * ALIGN;
        (b, data_start + offset)
    }

    fn header_with_len(&self, shrink: u64) -> Header {
        let (bytes, len) = self.bytes_and_len();
        let (gguf, bools) = header::parse(&mut Cursor::new(bytes)).unwrap();
        Header {
            gguf,
            bools,
            file_len: len - shrink,
        }
    }

    fn scan(&self) -> napi::Result<Scan> {
        Scan::from_headers(&[PathBuf::from("tiny.gguf")], &[self.header_with_len(0)])
    }
}

const N_TRUNK: usize = 4;
const DIM: u64 = 64;
const VOCAB: u64 = 32;
const LATENT: u64 = 256;
const FF: u64 = 512;
const N_EXP: u64 = 4;
const HC: u64 = 4;

fn heads(il: usize) -> u64 {
    if indexed(il) { 2 } else { 4 }
}

fn indexed(il: usize) -> bool {
    matches!(il, 0 | 3 | 4)
}

/// Four trunk layers (I S S I) plus one MTP layer; latent != ff so a
/// swapped expert orientation has other dims. Engram on layer 1, dense FFN
/// on layer 0; down experts Q6_K on layer 1, Q4_K on 2-3, BF16 on the MTP.
fn tiny() -> Fixture {
    let a = |k: &str| format!("hy_v4_flash.{k}");
    let mut kv: Vec<(String, Kv)> = vec![
        ("general.architecture".into(), Kv::Str("hy_v4_flash".into())),
        (
            "tokenizer.ggml.tokens".into(),
            Kv::Strs((0..VOCAB).map(|i| format!("t{i}")).collect()),
        ),
    ];
    let u = |x: u64| Kv::U32(x as u32);
    let scalars = [
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
        ("attention.layer_norm_rms_epsilon", Kv::F32(1e-5)),
        ("rope.dimension_count", u(8)),
        ("rope.freq_base", Kv::F32(1e7)),
        ("rope.freq_base_swa", Kv::F32(1e4)),
        ("expert_count", u(N_EXP)),
        ("expert_used_count", u(2)),
        ("expert_shared_count", u(1)),
        ("expert_feed_forward_length", u(FF)),
        ("expert_latent_length", u(LATENT)),
        ("expert_weights_scale", Kv::F32(2.5)),
        ("expert_weights_norm", Kv::Bool(true)),
        ("expert_gating_func", u(2)),
        ("hyper_connection.count", u(HC)),
        ("hyper_connection.epsilon", Kv::F32(1e-6)),
        ("hyper_connection.magnitude", Kv::F32(2.0)),
        ("attention.indexer.head_count", u(2)),
        ("attention.indexer.key_length", u(16)),
        ("attention.indexer.top_k", u(8)),
        ("attention.indexer.kpool", u(4)),
        ("attention.indexer.kpool_select_tail", Kv::Bool(true)),
        ("ple.ngram_size", u(3)),
        ("ple.heads_per_ngram", u(2)),
        ("embedding_length_per_layer_input", u(32)),
        ("engram.pad_token_id", u(3)),
    ];
    kv.extend(scalars.into_iter().map(|(k, v)| (a(k), v)));
    let arrays = [
        (
            "attention.head_count",
            Kv::I32s((0..=N_TRUNK).map(|il| heads(il) as i32).collect()),
        ),
        (
            "attention.sliding_window_pattern",
            Kv::Bools((0..=N_TRUNK).map(|il| !indexed(il)).collect()),
        ),
        (
            "attention.indexer.types",
            Kv::Bools((0..N_TRUNK).map(indexed).collect()),
        ),
        ("swiglu_clamp_exp", Kv::F32s(vec![10.0; N_TRUNK + 1])),
        ("ple.layers", Kv::I32s(vec![1])),
        ("ple.layer_multipliers", Kv::U64s(vec![7, 11, 13])),
        ("ple.head_offsets", Kv::U64s(vec![0, 5, 12, 23])),
        ("ple.head_vocab_sizes", Kv::U64s(vec![5, 7, 11, 13])),
        (
            "engram.token_map",
            Kv::I32s((0..VOCAB as i32).map(|i| i % 20).collect()),
        ),
    ];
    kv.extend(arrays.into_iter().map(|(k, v)| (a(k), v)));
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
        let mtp = il == N_TRUNK;
        let h = heads(il);
        let mut w: Vec<(&str, Vec<u64>)> = vec![
            ("attn_norm", vec![DIM]),
            ("attn_sinks", vec![h]),
            ("attn_q_a", vec![DIM, 32]),
            ("attn_q_a_norm", vec![32]),
            ("attn_q_b", vec![32, h * 16]),
            ("attn_kv_a_mqa", vec![DIM, 24]),
            ("attn_kv_a_norm", vec![16]),
            ("attn_k_b", vec![8, 16, h]),
            ("attn_v_b", vec![16, 12, h]),
            ("attn_output", vec![h * 12, DIM]),
            ("attn_gate", vec![DIM, h * 12]),
            ("post_attention_norm", vec![DIM]),
            ("ffn_norm", vec![DIM]),
            ("post_ffw_norm", vec![DIM]),
        ];
        if mtp {
            w.extend([
                ("nextn.enorm", vec![DIM]),
                ("nextn.hnorm", vec![DIM]),
                ("nextn.eh_proj", vec![2 * DIM, DIM]),
                ("nextn.shared_head_norm", vec![DIM]),
            ]);
        } else {
            w.extend([
                ("hc_attn_fn", vec![HC * DIM, 2 * HC]),
                ("hc_attn_base", vec![2 * HC]),
                ("hc_attn_scale", vec![2]),
                ("hc_ffn_fn", vec![HC * DIM, 2 * HC]),
                ("hc_ffn_base", vec![2 * HC]),
                ("hc_ffn_scale", vec![2]),
            ]);
        }
        if indexed(il) {
            w.extend([
                ("indexer.attn_k", vec![DIM, 16]),
                ("indexer.k_norm", vec![16]),
                ("indexer_compressor_value", vec![16, 16]),
                ("indexer_compressor_gate", vec![16, 16]),
                ("indexer_compressor_ape", vec![16, 4]),
                ("indexer_compressor_norm", vec![16]),
                ("indexer.attn_q_b", vec![32, 32]),
                ("indexer.proj", vec![DIM, 2]),
            ]);
        }
        if il == 1 {
            w.extend([
                ("engram_kv", vec![4 * 32, DIM * (HC + 1)]),
                ("engram_kv_b", vec![DIM, HC + 1]),
                ("engram_gate", vec![DIM, HC]),
                ("engram_lookup_norm", vec![32, 4]),
                ("engram_pre_gamma", vec![DIM]),
                ("engram_hc_scale", vec![DIM, HC]),
            ]);
        }
        if il == 0 {
            w.extend([
                ("ffn_gate", vec![DIM, 96]),
                ("ffn_down", vec![96, DIM]),
                ("ffn_up", vec![DIM, 96]),
            ]);
        } else {
            w.extend([
                ("ffn_gate_inp", vec![DIM, N_EXP]),
                ("ffn_routed_down", vec![DIM, LATENT]),
                ("ffn_routed_up", vec![LATENT, DIM]),
                ("ffn_gate_shexp", vec![DIM, FF]),
                ("ffn_down_shexp", vec![FF, DIM]),
                ("ffn_up_shexp", vec![DIM, FF]),
            ]);
        }
        for (n, dims) in w {
            t.push((format!("blk.{il}.{n}.weight"), dims, F32));
        }
        if il > 0 {
            t.push((format!("blk.{il}.exp_probs_b.bias"), vec![N_EXP], F32));
            let (gu, down) = match il {
                1 => (Q4K, Q6K),
                _ if mtp => (BF16, BF16),
                _ => (Q4K, Q4K),
            };
            t.push((
                format!("blk.{il}.ffn_gate_exps.weight"),
                vec![LATENT, FF, N_EXP],
                gu,
            ));
            t.push((
                format!("blk.{il}.ffn_up_exps.weight"),
                vec![LATENT, FF, N_EXP],
                gu,
            ));
            t.push((
                format!("blk.{il}.ffn_down_exps.weight"),
                vec![FF, LATENT, N_EXP],
                down,
            ));
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
    let s = tiny().scan().unwrap();
    let c = &s.cfg;
    assert_eq!((c.n_layers, c.layers.len(), c.mtp().len()), (4, 5, 1));
    let kinds: Vec<Attn> = c.layers.iter().map(|l| l.attn).collect();
    use Attn::{Indexed as I, Sliding as S};
    assert_eq!(kinds, [I, S, S, I, I]);
    let thetas: Vec<f32> = c.layers.iter().map(|l| l.rope_theta).collect();
    assert_eq!(thetas, [1e7, 1e4, 1e4, 1e7, 1e7]);
    let heads: Vec<usize> = c.layers.iter().map(|l| l.n_heads).collect();
    assert_eq!(heads, [2, 4, 4, 2, 2]);
    let dense: Vec<bool> = c.layers.iter().map(|l| l.dense_ffn).collect();
    assert_eq!(dense, [true, false, false, false, false]);
    let engram: Vec<Option<usize>> = c.layers.iter().map(|l| l.engram).collect();
    assert_eq!(engram, [None, Some(0), None, None, None]);
    assert!(c.layers[4].mtp && !c.layers[3].mtp);
    assert_eq!(
        (c.mla.qk_nope_dim, c.mla.kv_row(), c.mla.scale),
        (8, 24, 0.25)
    );
    assert_eq!((c.indexer.n_top_pool(), c.engram_rows()), (1, 36));
    assert_eq!(c.engram.compressed_vocab_size, 20);
    let classes: Vec<Option<ExpertClass>> = s
        .experts
        .iter()
        .map(|e| e.as_ref().map(|e| e.class))
        .collect();
    assert_eq!(
        classes,
        [
            None,
            Some(ExpertClass::Q6kDown),
            Some(ExpertClass::Q4kDown),
            Some(ExpertClass::Q4kDown),
            Some(ExpertClass::Bf16),
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

    let mtp_bytes: u64 = tiny_tensors()
        .into_iter()
        .filter(|t| t.0.starts_with(&format!("blk.{N_TRUNK}.")))
        .map(|(name, dims, tensor_type)| {
            GgufTensorInfo {
                name,
                n_dims: dims.len() as u32,
                dims,
                tensor_type,
                offset: 0,
            }
            .data_size()
            .unwrap()
        })
        .sum();
    assert!(mtp_bytes > 0);
    assert_eq!(
        s.mtp,
        Some(MtpLayer {
            layer: N_TRUNK,
            bytes: mtp_bytes,
        })
    );
    assert_eq!(s.mtp_experts().map(|e| e.class), Some(ExpertClass::Bf16));
    assert_eq!(s.trunk_experts().len(), N_TRUNK);
}

/// The same file written as by a converter run with `--no-mtp`.
fn without_mtp() -> Fixture {
    let mut f = tiny();
    f.set_arch("block_count", Kv::U32(N_TRUNK as u32));
    f.remove_arch("nextn_predict_layers");
    f.set_arch(
        "attention.head_count",
        Kv::I32s((0..N_TRUNK).map(|il| heads(il) as i32).collect()),
    );
    f.set_arch(
        "attention.sliding_window_pattern",
        Kv::Bools((0..N_TRUNK).map(|il| !indexed(il)).collect()),
    );
    f.set_arch("swiglu_clamp_exp", Kv::F32s(vec![10.0; N_TRUNK]));
    let prefix = format!("blk.{N_TRUNK}.");
    f.tensors.retain(|t| !t.0.starts_with(&prefix));
    f
}

#[test]
fn a_file_without_the_mtp_layer_scans_without_it() {
    let s = without_mtp().scan().unwrap();
    assert_eq!((s.cfg.n_layers, s.cfg.layers.len()), (N_TRUNK, N_TRUNK));
    assert!(s.cfg.mtp().is_empty());
    assert_eq!(s.mtp, None);
    assert!(s.mtp_experts().is_none());
    assert_eq!(s.trunk_experts().len(), N_TRUNK);
    assert_eq!(s.with_role(Role::Mtp).count(), 0);

    let mut f = without_mtp();
    f.tensors.extend(
        tiny_tensors()
            .into_iter()
            .filter(|t| t.0.starts_with(&format!("blk.{N_TRUNK}."))),
    );
    assert!(err(&f).contains("is not part of the hy_v4_flash layout"));
}

#[test]
fn max_context_is_the_least_of_file_cap_and_option() {
    let s = tiny().scan().unwrap();
    assert_eq!(s.cfg.context_length, 1 << 20);
    assert_eq!(DEFAULT_CONTEXT_CAP, 262_144);
    assert_eq!(s.cfg.max_context(None).unwrap(), 262_144);
    assert_eq!(s.cfg.max_context(Some(1 << 21)).unwrap(), 262_144);
    assert_eq!(s.cfg.max_context(Some(262_145)).unwrap(), 262_144);
    assert_eq!(s.cfg.max_context(Some(100_000)).unwrap(), 100_000);
    assert_eq!(s.cfg.max_context(Some(4096)).unwrap(), 4096);
    assert!(s.cfg.max_context(Some(0)).is_err());

    let mut f = tiny();
    f.set_arch("context_length", Kv::U32(32_768));
    let c = f.scan().unwrap().cfg;
    assert_eq!(c.max_context(None).unwrap(), 32_768);
    assert_eq!(c.max_context(Some(1 << 20)).unwrap(), 32_768);
    assert_eq!(c.max_context(Some(16_384)).unwrap(), 16_384);

    let mut f = tiny();
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
    let cases: Vec<(&str, Box<dyn Fn(&mut Fixture)>, &str)> = vec![
        (
            "kv heads",
            Box::new(|f| f.set_arch("attention.head_count_kv", Kv::U32(2))),
            "head_count_kv",
        ),
        (
            "softmax gating",
            Box::new(|f| f.set_arch("expert_gating_func", Kv::U32(1))),
            "sigmoid",
        ),
        (
            "top_k / kpool < 2",
            Box::new(|f| f.set_arch("attention.indexer.top_k", Kv::U32(4))),
            "tail group",
        ),
        (
            "kpool 1",
            Box::new(|f| f.set_arch("attention.indexer.kpool", Kv::U32(1))),
            "pool size",
        ),
        (
            "yarn",
            Box::new(|f| f.set_arch("rope.scaling.type", Kv::Str("yarn".into()))),
            "rope scaling",
        ),
        (
            "wrong arch",
            Box::new(|f| f.set("general.architecture", Kv::Str("deepseek41".into()))),
            "general.architecture",
        ),
        (
            "two MTP layers",
            Box::new(|f| f.set_arch("nextn_predict_layers", Kv::U32(2))),
            "MTP layers",
        ),
        (
            "no head count for the MTP layer",
            Box::new(|f| {
                f.set_arch(
                    "attention.head_count",
                    Kv::I32s((0..N_TRUNK).map(|il| heads(il) as i32).collect()),
                )
            }),
            "attention.head_count has 4 entries, need 5",
        ),
        (
            "MTP layer with no heads",
            Box::new(|f| {
                f.set_arch(
                    "attention.head_count",
                    Kv::I32s(
                        (0..=N_TRUNK)
                            .map(|il| if il == N_TRUNK { 0 } else { heads(il) as i32 })
                            .collect(),
                    ),
                )
            }),
            "layer 4: head_count is 0",
        ),
        (
            "no pattern",
            Box::new(|f| f.remove_arch("attention.sliding_window_pattern")),
            "sliding_window_pattern",
        ),
        (
            "sliding layer with an indexer",
            Box::new(|f| {
                f.set_arch(
                    "attention.indexer.types",
                    Kv::Bools(vec![true, true, false, true]),
                )
            }),
            "cannot own an indexer",
        ),
        (
            "full attention layer",
            Box::new(|f| {
                f.set_arch(
                    "attention.indexer.types",
                    Kv::Bools(vec![false, false, false, true]),
                )
            }),
            "neither sliding nor indexed",
        ),
        (
            "indexer.types length",
            Box::new(|f| f.set_arch("attention.indexer.types", Kv::Bools(vec![true; 5]))),
            "indexer.types has 5",
        ),
        (
            "cache row width",
            Box::new(|f| f.set_arch("attention.key_length", Kv::U32(16))),
            "key_length",
        ),
        (
            "engram offsets not contiguous",
            Box::new(|f| f.set_arch("ple.head_offsets", Kv::U64s(vec![0, 5, 13, 24]))),
            "not contiguous",
        ),
        (
            "token map length",
            Box::new(|f| f.set_arch("engram.token_map", Kv::I32s(vec![0; 31]))),
            "token_map",
        ),
        (
            "engram layer past the trunk",
            Box::new(|f| f.set_arch("ple.layers", Kv::I32s(vec![4]))),
            "engram layers",
        ),
    ];
    for (what, edit, want) in cases {
        let mut f = tiny();
        edit(&mut f);
        let e = err(&f);
        assert!(e.contains(want), "{what}: {e}");
    }
}

#[test]
fn tensor_set_must_match_the_layout() {
    let mut f = tiny();
    f.tensors
        .push(("blk.2.unknown.weight".into(), vec![4], GgufTensorType::F32));
    assert!(err(&f).contains("blk.2.unknown.weight"));

    let mut f = tiny();
    f.tensors.retain(|t| t.0 != "blk.3.attn_sinks.weight");
    assert!(err(&f).contains("blk.3.attn_sinks.weight' missing"));

    let mut f = tiny();
    f.tensors.retain(|t| t.0 != "blk.2.exp_probs_b.bias");
    assert!(f.scan().is_ok(), "exp_probs_b is optional");

    let mut f = tiny();
    f.tensor("blk.4.ffn_down_exps.weight").2 = GgufTensorType::Q4K;
    f.tensor("blk.4.ffn_down_exps.weight").1 = vec![FF, LATENT, N_EXP];
    assert!(err(&f).contains("layer 4 (MTP): expert types"));

    let mut f = tiny();
    f.tensor("blk.2.ffn_down_exps.weight").2 = GgufTensorType::BF16;
    assert!(err(&f).contains("layer 2: expert types"));

    let mut f = tiny();
    f.tensor(ENGRAM_TABLE).2 = GgufTensorType::BF16;
    assert!(err(&f).contains("type 43"));
}

#[test]
fn truncated_file_is_refused() {
    let f = tiny();
    let short = Scan::from_headers(&[PathBuf::from("tiny.gguf")], &[f.header_with_len(1)]);
    let e = short.err().map(|e| e.reason).unwrap_or_default();
    assert!(e.contains("truncated") || e.contains("Engram table"), "{e}");
}

#[test]
fn bool_arrays_survive_the_header_walk() {
    let (bytes, _) = tiny().bytes_and_len();
    let (gguf, bools) = header::parse(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(
        bools["hy_v4_flash.attention.sliding_window_pattern"],
        [false, true, true, false, false]
    );
    assert_eq!(
        bools["hy_v4_flash.attention.indexer.types"],
        [true, false, false, true]
    );
    assert_eq!(bools.len(), 2);
    assert_eq!(gguf.tensors.len(), tiny_tensors().len());
}

fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    assert_eq!(paths.len(), 2);
    Scan::open(&paths).unwrap()
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    u64s(v).into_iter().map(|x| x as usize).collect()
}

fn bits(v: &Value) -> u32 {
    let s = v.as_str().unwrap();
    u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards"]
fn real_headers_match_the_oracle_config() {
    let s = real_scan();
    let c = &s.cfg;
    let want = fixture("config.json");
    let sc = &want["scalars"];
    let n = |k: &str| sc[k].as_u64().unwrap() as usize;
    assert_eq!(c.layers.len(), n("block_count"));
    assert_eq!(c.mtp().len(), n("nextn_predict_layers"));
    assert_eq!(c.vocab_size, n("vocab_size"));
    assert_eq!(c.context_length, n("context_length"));
    assert_eq!(c.dim, n("embedding_length"));
    assert_eq!(c.dense_ff, n("feed_forward_length"));
    assert_eq!(c.n_dense_lead, n("leading_dense_block_count"));
    assert_eq!(c.window, n("attention.sliding_window"));
    assert_eq!(c.mla.q_lora_rank, n("attention.q_lora_rank"));
    assert_eq!(c.mla.kv_lora_rank, n("attention.kv_lora_rank"));
    assert_eq!(c.mla.kv_row(), n("attention.key_length"));
    assert_eq!(c.mla.kv_lora_rank, n("attention.value_length"));
    assert_eq!(c.mla.head_dim_qk, n("attention.key_length_mla"));
    assert_eq!(c.mla.head_dim_v, n("attention.value_length_mla"));
    assert_eq!(c.mla.rope_dim, n("rope.dimension_count"));
    assert_eq!(c.moe.n_experts, n("expert_count"));
    assert_eq!(c.moe.n_used, n("expert_used_count"));
    assert_eq!(c.moe.n_shared, n("expert_shared_count"));
    assert_eq!(c.moe.ff, n("expert_feed_forward_length"));
    assert_eq!(c.moe.latent, n("expert_latent_length"));
    assert_eq!(
        c.moe.norm_topk,
        sc["expert_weights_norm"].as_bool().unwrap()
    );
    assert_eq!(c.hc.count, n("hyper_connection.count"));
    assert_eq!(c.indexer.n_heads, n("attention.indexer.head_count"));
    assert_eq!(c.indexer.head_dim, n("attention.indexer.key_length"));
    assert_eq!(c.indexer.top_k, n("attention.indexer.top_k"));
    assert_eq!(c.indexer.kpool, n("attention.indexer.kpool"));
    assert_eq!(
        c.indexer.select_tail,
        sc["attention.indexer.kpool_select_tail"].as_bool().unwrap()
    );
    let fb = &want["f32_bits"];
    for (got, key) in [
        (c.norm_eps, "attention.layer_norm_rms_epsilon"),
        (c.moe.route_scale, "expert_weights_scale"),
        (c.hc.eps, "hyper_connection.epsilon"),
        (c.hc.magnitude, "hyper_connection.magnitude"),
        (c.rope_theta, "rope.freq_base"),
        (c.rope_theta_swa, "rope.freq_base_swa"),
    ] {
        assert_eq!(got.to_bits(), bits(&fb[key]), "{key}");
    }

    let d = &want["derived"];
    let layers_of = |a: Attn| -> Vec<usize> {
        (0..c.n_layers)
            .filter(|&il| c.layers[il].attn == a)
            .collect()
    };
    assert_eq!(layers_of(Attn::Indexed), usizes(&d["indexed_layers"]));
    assert_eq!(layers_of(Attn::Sliding), usizes(&d["sliding_layers"]));
    assert_eq!(
        (c.n_layers..c.layers.len()).collect::<Vec<_>>(),
        usizes(&d["mtp_layers"])
    );
    assert_eq!(f64::from(c.mla.scale), d["attn_scale"].as_f64().unwrap());
    assert_eq!(
        c.indexer.n_top_pool(),
        d["indexer_n_top_pool"].as_u64().unwrap() as usize
    );
    let ws = d["indexer_weight_scale"].as_f64().unwrap();
    assert!((f64::from(c.indexer.weight_scale()) - ws).abs() <= ws * 1e-6);
    assert_eq!(c.max_context(None).unwrap(), n_ctx_target(d));
    assert_eq!(
        c.context_length,
        d["context_length_gguf"].as_u64().unwrap() as usize
    );

    let mtp = s.mtp.as_ref().unwrap();
    assert_eq!(mtp.layer, c.n_layers);
    assert_eq!(mtp.bytes, REAL_MTP_BYTES);
    let prefix = format!("blk.{}.", mtp.layer);
    let mut span = (u64::MAX, 0u64);
    for (name, e) in &s.tensors {
        if name.starts_with(&prefix) {
            assert_eq!(e.loc.shard, 0, "{name}");
            let at = e.loc.abs_offset().unwrap();
            span = (
                span.0.min(at),
                span.1.max(at + e.loc.info.data_size().unwrap()),
            );
        }
    }
    assert_eq!(span.1 - span.0, REAL_MTP_BYTES);
    assert_eq!(s.mtp_experts().unwrap().class, ExpertClass::Bf16);
    assert_eq!(c.mtp()[0].n_heads, 32);
    assert_eq!(c.mtp()[0].attn, Attn::Indexed);

    for (il, w) in want["layers"].as_array().unwrap().iter().enumerate() {
        let l = &c.layers[il];
        assert_eq!(w["il"].as_u64().unwrap() as usize, il);
        let attn = match w["attn"].as_str().unwrap() {
            "I" => Attn::Indexed,
            _ => Attn::Sliding,
        };
        assert_eq!(l.attn, attn, "layer {il}");
        assert_eq!(
            l.n_heads,
            w["heads"].as_u64().unwrap() as usize,
            "layer {il}"
        );
        assert_eq!(f64::from(l.rope_theta), w["rope_theta"].as_f64().unwrap());
        assert_eq!(l.mtp, w["mtp"].as_bool().unwrap(), "layer {il}");
        assert_eq!(l.dense_ffn, w["ffn"] == "dense", "layer {il}");
        assert_eq!(
            l.engram.is_some(),
            w["engram"].as_bool().unwrap(),
            "layer {il}"
        );
        assert_eq!(
            f64::from(l.swiglu_limit),
            w["swiglu_clamp_exp"].as_f64().unwrap()
        );
        let has_indexer = s.opt(&format!("blk.{il}.indexer.attn_k.weight")).is_some();
        assert_eq!(has_indexer, w["has_indexer_tensors"].as_bool().unwrap());
    }

    let e = &want["engram"];
    let g = &c.engram;
    assert_eq!(g.layers, usizes(&e["layers"]));
    assert_eq!(g.max_ngram, e["ngram_size"].as_u64().unwrap() as usize);
    assert_eq!(g.n_heads, e["heads_per_ngram"].as_u64().unwrap() as usize);
    assert_eq!(g.n_cols(), e["n_cols"].as_u64().unwrap() as usize);
    assert_eq!(g.head_dim, e["head_dim"].as_u64().unwrap() as usize);
    assert_eq!(g.multipliers, u64s(&e["multipliers"]));
    assert_eq!(g.head_offsets, u64s(&e["head_offsets"]));
    assert_eq!(g.head_vocab_sizes, u64s(&e["head_vocab_sizes"]));
    let rows: Vec<u64> = (0..g.layers.len()).map(|i| g.layer_rows(i)).collect();
    assert_eq!(rows, u64s(&e["rows_per_layer"]));
    assert_eq!(c.engram_rows(), e["total_rows"].as_u64().unwrap());
    assert_eq!(g.pad_token_id, e["pad_token_id"].as_u64().unwrap() as usize);
    assert_eq!(
        u64::from(g.token_map[g.pad_token_id]),
        e["token_map_pad"].as_u64().unwrap()
    );
    assert_eq!(
        g.token_map.len(),
        e["token_map_len"].as_u64().unwrap() as usize
    );
    assert_eq!(
        g.compressed_vocab_size as u64,
        e["token_map_max_all"].as_u64().unwrap() + 1
    );
    let mut h = Sha256::new();
    g.token_map.iter().for_each(|&x| h.update(x.to_le_bytes()));
    let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, e["token_map_sha256_i32le"].as_str().unwrap());

    let t = &s.engram_table;
    assert_eq!(t.shard, 1);
    assert_eq!(t.info.dims, u64s(&e["table_dims_ggml"]));
    assert_eq!(
        t.info.tensor_type as u32,
        e["table_type_id"].as_u64().unwrap() as u32
    );
}

const REAL_MTP_BYTES: u64 = 7_486_176_384;

fn n_ctx_target(d: &Value) -> usize {
    d["context_target_v1"].as_u64().unwrap() as usize
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
fn real_expert_classes_and_extents_match_the_oracle() {
    let s = real_scan();
    let classes = fixture("expert-classes.json");
    let layers_of = |c: ExpertClass| -> Vec<usize> {
        (0..s.cfg.n_layers)
            .filter(|&il| s.experts[il].as_ref().is_some_and(|e| e.class == c))
            .collect()
    };
    assert_eq!(
        layers_of(ExpertClass::Q6kDown),
        usizes(&classes["q6k_down_layers"])
    );
    assert_eq!(
        layers_of(ExpertClass::Q4kDown),
        usizes(&classes["q4k_down_layers"])
    );
    for w in classes["per_layer"].as_array().unwrap() {
        let il = w["il"].as_u64().unwrap() as usize;
        let e = s.experts[il].as_ref().unwrap();
        assert_eq!(
            class_name(e.class),
            w["class"].as_str().unwrap(),
            "layer {il}"
        );
    }
    assert!(s.experts[0].is_none());

    let extents = fixture("expert-extents.json");
    let n_exp = s.cfg.moe.n_experts as u64;
    for w in extents["layers"].as_array().unwrap() {
        let il = w["il"].as_u64().unwrap() as usize;
        let e = s.experts[il].as_ref().unwrap();
        assert_eq!(class_name(e.class), w["class"].as_str().unwrap());
        for (key, loc) in [("gate", &e.gate), ("up", &e.up), ("down", &e.down)] {
            let x = &w[key];
            let what = format!("layer {il} {key}");
            assert_eq!(
                type_name(loc.info.tensor_type),
                x["type"].as_str().unwrap(),
                "{what}"
            );
            assert_eq!(loc.info.dims, u64s(&x["ggml_dims"]), "{what}");
            assert_eq!(loc.shard as u64, x["shard"].as_u64().unwrap(), "{what}");
            assert_eq!(loc.info.offset, x["rel_offset"].as_u64().unwrap(), "{what}");
            assert_eq!(
                loc.abs_offset().unwrap(),
                x["abs_offset"].as_u64().unwrap(),
                "{what}"
            );
            let bytes = loc.info.data_size().unwrap();
            assert_eq!(bytes, x["bytes"].as_u64().unwrap(), "{what}");
            assert_eq!(
                bytes / n_exp,
                x["per_expert_bytes"].as_u64().unwrap(),
                "{what}"
            );
        }
    }
}
