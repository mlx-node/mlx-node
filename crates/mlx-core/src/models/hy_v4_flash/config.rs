//! `hy_v4_flash` GGUF metadata -> model config. Key names and refusals follow
//! llama.cpp PR #32517 `llama_model_hy_v4_flash::load_arch_hparams`.

use std::collections::HashMap;

use napi::{Error, Result};

use crate::models::deepseek_v41::EngramConfig;
use crate::utils::gguf::{GgufFile, GgufMetaValue};

pub const ARCH: &str = "hy_v4_flash";
/// llama.cpp `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`.
const GATING_SIGMOID: usize = 2;
/// Most tokens one sequence may hold; the file and the load option can only
/// lower it.
pub const DEFAULT_CONTEXT_CAP: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attn {
    /// Last `window` rows incl. self, `rope.freq_base_swa`.
    Sliding,
    /// All rows through the k-pool indexer, `rope.freq_base`.
    Indexed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerConfig {
    pub attn: Attn,
    pub n_heads: usize,
    pub rope_theta: f32,
    /// Routed experts only; shared and dense FFNs are not clamped.
    pub swiglu_limit: f32,
    pub dense_ffn: bool,
    /// Index into `EngramConfig::layers`.
    pub engram: Option<usize>,
    pub mtp: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaConfig {
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_dim: usize,
    pub rope_dim: usize,
    /// `key_length_mla`: per-head query/key width (nope + rope).
    pub head_dim_qk: usize,
    /// `value_length_mla`: per-head output width after `v_b`.
    pub head_dim_v: usize,
    pub scale: f32,
}

impl MlaConfig {
    /// One cached latent row per token: `[kv | k_pe]`.
    pub fn kv_row(&self) -> usize {
        self.kv_lora_rank + self.rope_dim
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub n_used: usize,
    pub n_shared: usize,
    pub ff: usize,
    pub latent: usize,
    pub route_scale: f32,
    pub norm_topk: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HcConfig {
    pub count: usize,
    pub eps: f32,
    pub magnitude: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub kpool: usize,
    pub select_tail: bool,
}

impl IndexerConfig {
    /// History groups picked by score; the query's own group is added on top.
    pub fn n_top_pool(&self) -> usize {
        self.top_k / self.kpool - usize::from(self.select_tail)
    }

    pub fn weight_scale(&self) -> f32 {
        ((self.head_dim * self.n_heads) as f32).sqrt().recip()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyV4FlashConfig {
    /// Trunk layers (`block_count - nextn_predict_layers`).
    pub n_layers: usize,
    /// Trunk layers, then the MTP layers.
    pub layers: Vec<LayerConfig>,
    pub vocab_size: usize,
    /// The trained maximum; the runtime limit is `max_context`.
    pub context_length: usize,
    pub dim: usize,
    pub dense_ff: usize,
    pub n_dense_lead: usize,
    pub norm_eps: f32,
    pub window: usize,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub mla: MlaConfig,
    pub moe: MoeConfig,
    pub hc: HcConfig,
    pub indexer: IndexerConfig,
    pub engram: EngramConfig,
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 GGUF: {msg}"))
}

struct Meta<'a> {
    md: &'a HashMap<String, GgufMetaValue>,
    bools: &'a HashMap<String, Vec<bool>>,
}

impl Meta<'_> {
    fn opt(&self, key: &str) -> Option<&GgufMetaValue> {
        self.md.get(&format!("{ARCH}.{key}"))
    }
    fn get(&self, key: &str) -> Result<&GgufMetaValue> {
        self.opt(key)
            .ok_or_else(|| bad(format!("missing {ARCH}.{key}")))
    }
    fn int(v: &GgufMetaValue) -> Option<i64> {
        match v {
            GgufMetaValue::Uint8(x) => Some(i64::from(*x)),
            GgufMetaValue::Uint16(x) => Some(i64::from(*x)),
            GgufMetaValue::Uint32(x) => Some(i64::from(*x)),
            GgufMetaValue::Int32(x) => Some(i64::from(*x)),
            GgufMetaValue::Int64(x) => Some(*x),
            GgufMetaValue::Uint64(x) => i64::try_from(*x).ok(),
            _ => None,
        }
    }
    fn usize(&self, key: &str) -> Result<usize> {
        let v = Self::int(self.get(key)?)
            .ok_or_else(|| bad(format!("{ARCH}.{key} is not an integer")))?;
        usize::try_from(v).map_err(|_| bad(format!("{ARCH}.{key} must be non-negative")))
    }
    fn opt_usize(&self, key: &str) -> Result<Option<usize>> {
        self.opt(key).map(|_| self.usize(key)).transpose()
    }
    fn f32(&self, key: &str) -> Result<f32> {
        self.get(key)?
            .as_f32()
            .ok_or_else(|| bad(format!("{ARCH}.{key} is not a float")))
    }
    fn bool(&self, key: &str) -> Result<bool> {
        match self.get(key)? {
            GgufMetaValue::Bool(b) => Ok(*b),
            _ => Err(bad(format!("{ARCH}.{key} is not a bool"))),
        }
    }
    fn bools(&self, key: &str) -> Result<&[bool]> {
        self.bools
            .get(&format!("{ARCH}.{key}"))
            .map(Vec::as_slice)
            .ok_or_else(|| bad(format!("missing bool array {ARCH}.{key}")))
    }
    fn u64s(&self, key: &str) -> Result<Vec<u64>> {
        let ints: Vec<i64> = match self.get(key)? {
            GgufMetaValue::ArrayU32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            GgufMetaValue::ArrayI32(v) => v.iter().map(|&x| i64::from(x)).collect(),
            GgufMetaValue::ArrayI64(v) => v.clone(),
            GgufMetaValue::ArrayU64(v) => return Ok(v.clone()),
            _ => return Err(bad(format!("{ARCH}.{key} is not an integer array"))),
        };
        ints.into_iter()
            .map(|x| {
                u64::try_from(x).map_err(|_| bad(format!("{ARCH}.{key} has a negative entry")))
            })
            .collect()
    }
    fn usizes(&self, key: &str) -> Result<Vec<usize>> {
        self.u64s(key)?
            .into_iter()
            .map(|x| usize::try_from(x).map_err(|_| bad(format!("{ARCH}.{key} entry overflows"))))
            .collect()
    }
    /// A scalar for every layer, or one entry per layer.
    fn per_layer_usize(&self, key: &str, n: usize) -> Result<Vec<usize>> {
        let v = match self.get(key)? {
            v if Self::int(v).is_some() => vec![self.usize(key)?; n],
            _ => self.usizes(key)?,
        };
        Self::check_len(key, v, n)
    }
    fn per_layer_f32(&self, key: &str, n: usize) -> Result<Vec<f32>> {
        let v = match self.get(key)? {
            GgufMetaValue::ArrayF32(v) => v.clone(),
            v => vec![
                v.as_f32()
                    .ok_or_else(|| bad(format!("{ARCH}.{key} is not a float")))?;
                n
            ],
        };
        Self::check_len(key, v, n)
    }
    fn check_len<T>(key: &str, v: Vec<T>, n: usize) -> Result<Vec<T>> {
        if v.len() != n {
            return Err(bad(format!(
                "{ARCH}.{key} has {} entries, need {n}",
                v.len()
            )));
        }
        Ok(v)
    }
}

impl HyV4FlashConfig {
    /// `bools` = the bool arrays of the same header (`header::Header::bools`).
    pub fn from_gguf(gguf: &GgufFile, bools: &HashMap<String, Vec<bool>>) -> Result<Self> {
        let md = &gguf.metadata;
        let arch = md
            .get("general.architecture")
            .and_then(GgufMetaValue::as_str);
        if arch != Some(ARCH) {
            return Err(bad(format!(
                "general.architecture is {arch:?}, not {ARCH:?}"
            )));
        }
        let m = Meta { md, bools };
        let n_all = m.usize("block_count")?;
        let n_mtp = m.opt_usize("nextn_predict_layers")?.unwrap_or(0);
        if n_mtp > 1 {
            return Err(bad(format!("{n_mtp} MTP layers; at most 1 is supported")));
        }
        let n_layers = n_all
            .checked_sub(n_mtp)
            .filter(|&n| n > 0)
            .ok_or_else(|| bad("block_count leaves no trunk layer"))?;
        let vocab_size = vocab_size(&m)?;
        check_rope_scaling(&m)?;
        if m.usize("attention.head_count_kv")? != 1 {
            return Err(bad(
                "attention.head_count_kv must be 1 (one shared latent KV head)",
            ));
        }
        if m.usize("expert_gating_func")? != GATING_SIGMOID {
            return Err(bad("expert_gating_func must be 2 (sigmoid)"));
        }

        let dim = m.usize("embedding_length")?;
        let mla = mla(&m)?;
        let moe = moe(&m)?;
        let hc = HcConfig {
            count: m.usize("hyper_connection.count")?,
            eps: m.f32("hyper_connection.epsilon")?,
            magnitude: m.f32("hyper_connection.magnitude")?,
        };
        if hc.count == 0 {
            return Err(bad("hyper_connection.count is 0"));
        }
        let indexer = indexer(&m, mla.rope_dim)?;
        let engram = engram(&m, n_layers, vocab_size)?;
        let n_dense_lead = m.opt_usize("leading_dense_block_count")?.unwrap_or(0);
        if n_dense_lead > n_layers {
            return Err(bad("leading_dense_block_count exceeds the trunk"));
        }
        let rope_theta = m.f32("rope.freq_base")?;
        let rope_theta_swa = m.f32("rope.freq_base_swa")?;
        let window = m.usize("attention.sliding_window")?;
        if window == 0 || rope_theta <= 0.0 || rope_theta_swa <= 0.0 {
            return Err(bad("sliding_window and rope bases must be > 0"));
        }
        let context_length = m.usize("context_length")?;
        if context_length == 0 {
            return Err(bad("context_length is 0"));
        }
        let layers = layers(
            &m,
            n_layers,
            n_all,
            n_dense_lead,
            &engram,
            rope_theta,
            rope_theta_swa,
        )?;
        Ok(Self {
            n_layers,
            layers,
            vocab_size,
            context_length,
            dim,
            dense_ff: m.usize("feed_forward_length")?,
            n_dense_lead,
            norm_eps: m.f32("attention.layer_norm_rms_epsilon")?,
            window,
            rope_theta,
            rope_theta_swa,
            mla,
            moe,
            hc,
            indexer,
            engram,
        })
    }

    /// Rows of the one Engram table (`per_layer_token_embd`).
    pub fn engram_rows(&self) -> u64 {
        self.engram.head_vocab_sizes.iter().sum()
    }

    pub fn trunk(&self) -> &[LayerConfig] {
        &self.layers[..self.n_layers]
    }

    pub fn mtp(&self) -> &[LayerConfig] {
        &self.layers[self.n_layers..]
    }

    /// Tokens one sequence may hold: the least of `context_length`,
    /// `DEFAULT_CONTEXT_CAP` and the load option `maxContextTokens`.
    pub fn max_context(&self, requested: Option<usize>) -> Result<usize> {
        if requested == Some(0) {
            return Err(bad("maxContextTokens is 0"));
        }
        Ok(self
            .context_length
            .min(DEFAULT_CONTEXT_CAP)
            .min(requested.unwrap_or(usize::MAX)))
    }
}

fn vocab_size(m: &Meta) -> Result<usize> {
    let n = match m.md.get("tokenizer.ggml.tokens") {
        Some(GgufMetaValue::ArrayString(t)) if !t.is_empty() => t.len(),
        _ => return Err(bad("tokenizer.ggml.tokens is missing")),
    };
    if let Some(v) = m.opt_usize("vocab_size")?
        && v != n
    {
        return Err(bad(format!("vocab_size {v} != {n} tokens")));
    }
    Ok(n)
}

fn check_rope_scaling(m: &Meta) -> Result<()> {
    match m.opt("rope.scaling.type") {
        None => Ok(()),
        Some(v) => match v.as_str() {
            Some("none") => Ok(()),
            other => Err(bad(format!("rope scaling {other:?} is not supported"))),
        },
    }
}

fn mla(m: &Meta) -> Result<MlaConfig> {
    let kv_lora_rank = m.usize("attention.kv_lora_rank")?;
    let rope_dim = m.usize("rope.dimension_count")?;
    let head_dim_qk = m.usize("attention.key_length_mla")?;
    let head_dim_v = m.usize("attention.value_length_mla")?;
    for (key, want) in [
        ("attention.key_length_mla_swa", head_dim_qk),
        ("attention.value_length_mla_swa", head_dim_v),
    ] {
        if let Some(v) = m.opt_usize(key)?
            && v != want
        {
            return Err(bad(format!(
                "{key} {v} != {want}: one MLA geometry is required"
            )));
        }
    }
    if m.usize("attention.key_length")? != kv_lora_rank + rope_dim
        || m.usize("attention.value_length")? != kv_lora_rank
    {
        return Err(bad(
            "attention.key_length / value_length must be kv_lora_rank + rope / kv_lora_rank",
        ));
    }
    if rope_dim == 0 || !rope_dim.is_multiple_of(2) || head_dim_qk <= rope_dim || head_dim_v == 0 {
        return Err(bad(format!(
            "MLA head dims {head_dim_qk}/{head_dim_v} with rope {rope_dim} are inconsistent"
        )));
    }
    Ok(MlaConfig {
        q_lora_rank: m.usize("attention.q_lora_rank")?,
        kv_lora_rank,
        qk_nope_dim: head_dim_qk - rope_dim,
        rope_dim,
        head_dim_qk,
        head_dim_v,
        scale: (head_dim_qk as f32).sqrt().recip(),
    })
}

fn moe(m: &Meta) -> Result<MoeConfig> {
    let c = MoeConfig {
        n_experts: m.usize("expert_count")?,
        n_used: m.usize("expert_used_count")?,
        n_shared: m.usize("expert_shared_count")?,
        ff: m.usize("expert_feed_forward_length")?,
        latent: m.usize("expert_latent_length")?,
        route_scale: m.f32("expert_weights_scale")?,
        norm_topk: m.bool("expert_weights_norm")?,
    };
    if c.n_used == 0 || c.n_used > c.n_experts {
        return Err(bad("expert_used_count must be in 1..=expert_count"));
    }
    if c.n_shared == 0 || c.ff == 0 || c.latent == 0 {
        return Err(bad(
            "expert_shared_count, expert_feed_forward_length and expert_latent_length must be > 0",
        ));
    }
    Ok(c)
}

fn indexer(m: &Meta, rope_dim: usize) -> Result<IndexerConfig> {
    let c = IndexerConfig {
        n_heads: m.usize("attention.indexer.head_count")?,
        head_dim: m.usize("attention.indexer.key_length")?,
        top_k: m.usize("attention.indexer.top_k")?,
        kpool: m.usize("attention.indexer.kpool")?,
        select_tail: match m.opt("attention.indexer.kpool_select_tail") {
            None => false,
            Some(_) => m.bool("attention.indexer.kpool_select_tail")?,
        },
    };
    if c.kpool < 2 || c.top_k == 0 || !c.top_k.is_multiple_of(c.kpool) {
        return Err(bad(format!(
            "indexer top_k {} must be a multiple of the pool size {} (>= 2)",
            c.top_k, c.kpool
        )));
    }
    if c.top_k / c.kpool < 2 {
        return Err(bad(
            "indexer top_k / kpool must be >= 2 (one history group and one tail group)",
        ));
    }
    if c.n_heads == 0 || c.head_dim <= rope_dim {
        return Err(bad(format!(
            "indexer key length {} must exceed the rope dimension {rope_dim}",
            c.head_dim
        )));
    }
    Ok(c)
}

fn layers(
    m: &Meta,
    n_layers: usize,
    n_all: usize,
    n_dense_lead: usize,
    engram: &EngramConfig,
    rope_theta: f32,
    rope_theta_swa: f32,
) -> Result<Vec<LayerConfig>> {
    let heads = m.per_layer_usize("attention.head_count", n_all)?;
    let limits = m.per_layer_f32("swiglu_clamp_exp", n_all)?;
    let swa = m.bools("attention.sliding_window_pattern")?;
    if swa.len() != n_all {
        return Err(bad(format!(
            "attention.sliding_window_pattern has {} entries, need {n_all}",
            swa.len()
        )));
    }
    let types = m.bools("attention.indexer.types")?;
    if types.len() != n_layers {
        return Err(bad(format!(
            "attention.indexer.types has {} entries, need {n_layers}",
            types.len()
        )));
    }
    if !swa[..n_layers].iter().any(|&s| s) {
        return Err(bad("no sliding-window layer"));
    }
    (0..n_all)
        .map(|il| {
            let mtp = il >= n_layers;
            // every MTP layer owns an indexer (PR #32517 load_arch_tensors)
            let indexed = mtp || types[il];
            let attn = match (swa[il], indexed) {
                (true, false) => Attn::Sliding,
                (false, true) => Attn::Indexed,
                (true, true) => {
                    return Err(bad(format!(
                        "layer {il} is a sliding-window layer and cannot own an indexer"
                    )));
                }
                (false, false) => {
                    return Err(bad(format!(
                        "layer {il} is neither sliding nor indexed (dense full attention is not supported)"
                    )));
                }
            };
            if heads[il] == 0 {
                return Err(bad(format!("layer {il}: head_count is 0")));
            }
            Ok(LayerConfig {
                attn,
                n_heads: heads[il],
                rope_theta: match attn {
                    Attn::Sliding => rope_theta_swa,
                    Attn::Indexed => rope_theta,
                },
                swiglu_limit: limits[il],
                dense_ffn: !mtp && il < n_dense_lead,
                engram: engram.layers.iter().position(|&l| l == il),
                mtp,
            })
        })
        .collect()
}

fn engram(m: &Meta, n_layers: usize, vocab_size: usize) -> Result<EngramConfig> {
    let token_map = m
        .u64s("engram.token_map")?
        .into_iter()
        .map(|x| u32::try_from(x).map_err(|_| bad("engram.token_map entry overflows u32")))
        .collect::<Result<Vec<_>>>()?;
    let e = EngramConfig {
        layers: m.usizes("ple.layers")?,
        max_ngram: m.usize("ple.ngram_size")?,
        n_heads: m.usize("ple.heads_per_ngram")?,
        head_dim: m.usize("embedding_length_per_layer_input")?,
        multipliers: m.u64s("ple.layer_multipliers")?,
        head_offsets: m.u64s("ple.head_offsets")?,
        head_vocab_sizes: m.u64s("ple.head_vocab_sizes")?,
        pad_token_id: m.usize("engram.pad_token_id")?,
        compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
        token_map,
        vocab_size: None,
    };
    check_engram(&e, n_layers, vocab_size)?;
    Ok(e)
}

fn check_engram(e: &EngramConfig, n_layers: usize, vocab_size: usize) -> Result<()> {
    let n_e = e.layers.len();
    if n_e == 0
        || e.max_ngram < 2
        || e.n_heads == 0
        || e.head_dim == 0
        || e.multipliers.len() != n_e * e.max_ngram
        || e.head_vocab_sizes.len() != n_e * e.n_cols()
        || e.head_offsets.len() != n_e * e.n_cols()
        || e.pad_token_id >= e.token_map.len()
    {
        return Err(bad("engram metadata is inconsistent"));
    }
    if e.token_map.len() != vocab_size {
        return Err(bad(format!(
            "engram.token_map has {} entries, vocab is {vocab_size}",
            e.token_map.len()
        )));
    }
    if e.layers.iter().any(|&l| l >= n_layers) || !e.layers.is_sorted_by(|a, b| a < b) {
        return Err(bad("engram layers are out of range or out of order"));
    }
    let mut end = 0u64;
    for (c, (&p, &o)) in e.head_vocab_sizes.iter().zip(&e.head_offsets).enumerate() {
        // a bucket count below 2 traps in the hash; row ids are int32 (PR #32517)
        if p < 2 || o != end || o.checked_add(p).is_none_or(|x| x > i32::MAX as u64) {
            return Err(bad(format!(
                "engram head {c}: bucket range {o}+{p} is out of range or not contiguous"
            )));
        }
        end = o + p;
    }
    let max_id = e.token_map.iter().copied().max().map_or(0, u64::from);
    if e.multipliers
        .iter()
        .any(|&k| k.checked_mul(max_id).is_none())
    {
        return Err(bad("engram multiplier x max token id overflows u64"));
    }
    Ok(())
}
