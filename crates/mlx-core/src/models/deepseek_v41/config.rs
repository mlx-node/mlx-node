//! `deepseek41` GGUF metadata -> model config. Key names and the source-layer
//! derivation follow llama.cpp PR #30817 (`deepseek41.cpp` `load_arch_hparams`);
//! field meanings follow the DeepSeek 1017 reference `model.py` `ModelArgs`.

use std::collections::HashMap;

use napi::{Error, Result};

use crate::utils::gguf::{GgufFile, GgufMetaValue};

const ARCH: &str = "deepseek41";
/// llama.cpp `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`.
const GATING_SQRT_SOFTPLUS: u64 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct YarnConfig {
    pub factor: f32,
    pub original_seq_len: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateConfig {
    pub source_layer: usize,
    pub topk_blocks: usize,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngramConfig {
    pub layers: Vec<usize>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub multipliers: Vec<u64>,
    /// Global row offset of each (layer, n-gram order, head) column in the one
    /// GGUF table; `(max_ngram - 1) * n_heads` columns per engram layer.
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    /// Raw token id; the hash uses `token_map[pad_token_id]`.
    pub pad_token_id: usize,
    pub token_map: Vec<u32>,
    pub compressed_vocab_size: usize,
    pub vocab_size: Option<u64>,
}

impl EngramConfig {
    pub fn n_cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    /// Table rows of engram layer `e` (the reference's `engram_num_embeddings`).
    pub fn layer_rows(&self, e: usize) -> u64 {
        let n = self.n_cols();
        self.head_vocab_sizes[e * n..(e + 1) * n].iter().sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub noise_token_id: usize,
    pub target_layers: Vec<usize>,
    pub markov_rank: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepseekV41Config {
    pub n_layers: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window_size: usize,
    pub norm_eps: f32,
    /// `None` when the file has no YaRN scaling (plain RoPE).
    pub yarn: Option<YarnConfig>,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub n_shared_experts: usize,
    pub moe_inter_dim: usize,
    pub route_scale: f32,
    pub norm_topk_prob: bool,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// First `n_layers` entries; trailing MTP slots are dropped.
    pub compress_ratios: Vec<u32>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    /// Layer whose compressed KV this layer reads (`None` on ratio-0 layers).
    pub kv_src: Vec<Option<usize>>,
    /// Layer whose indexer top-k this layer uses (`None` on ratio-0 layers).
    pub topk_src: Vec<Option<usize>>,
    pub candidate: Option<CandidateConfig>,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram: Option<EngramConfig>,
    pub dspark: Option<DsparkConfig>,
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("DeepSeek-V4.1 GGUF: {msg}"))
}

struct Meta<'a>(&'a HashMap<String, GgufMetaValue>);

impl Meta<'_> {
    fn opt(&self, key: &str) -> Option<&GgufMetaValue> {
        self.0.get(&format!("{ARCH}.{key}"))
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
    fn signed(&self, key: &str) -> Result<i64> {
        Self::int(self.get(key)?).ok_or_else(|| bad(format!("{ARCH}.{key} is not an integer")))
    }
    fn usize(&self, key: &str) -> Result<usize> {
        usize::try_from(self.signed(key)?)
            .map_err(|_| bad(format!("{ARCH}.{key} must be non-negative")))
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
    fn f32s(&self, key: &str) -> Result<Vec<f32>> {
        match self.get(key)? {
            GgufMetaValue::ArrayF32(v) => Ok(v.clone()),
            _ => Err(bad(format!("{ARCH}.{key} is not a float array"))),
        }
    }
}

impl DeepseekV41Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let md = &gguf.metadata;
        let arch = md
            .get("general.architecture")
            .and_then(GgufMetaValue::as_str);
        if arch != Some(ARCH) {
            return Err(bad(format!(
                "general.architecture is {arch:?}, not {ARCH:?}"
            )));
        }
        let m = Meta(md);
        let n_layers = m.usize("block_count")?;
        if n_layers == 0 {
            return Err(bad("block_count is 0"));
        }
        let vocab_size = match md.get("tokenizer.ggml.tokens") {
            Some(GgufMetaValue::ArrayString(t)) if !t.is_empty() => t.len(),
            _ => return Err(bad("tokenizer.ggml.tokens is missing")),
        };
        if m.opt_usize("hash_layer_count")?.unwrap_or(0) != 0 {
            return Err(bad(
                "hash-routed layers (hash_layer_count > 0) are not supported",
            ));
        }
        if m.usize("attention.head_count_kv")? != 1 {
            return Err(bad(
                "attention.head_count_kv must be 1 (one shared latent KV head)",
            ));
        }
        let head_dim = m.usize("attention.key_length")?;
        if m.usize("attention.value_length")? != head_dim {
            return Err(bad("attention.value_length must equal key_length"));
        }
        if m.usize("expert_gating_func")? as u64 != GATING_SQRT_SOFTPLUS {
            return Err(bad("expert_gating_func must be 4 (sqrt softplus)"));
        }
        let dim = m.usize("embedding_length")?;
        let hc_mult = m.usize("hyper_connection.count")?;
        if let Some(out) = m.opt_usize("embedding_length_out")?
            && out != hc_mult * dim
        {
            return Err(bad(format!(
                "embedding_length_out {out} != hc {hc_mult} x dim {dim}"
            )));
        }

        let per_layer = |key: &str| -> Result<Vec<f32>> {
            let v = m.f32s(key)?;
            if v.len() != n_layers {
                return Err(bad(format!(
                    "{ARCH}.{key} has {} entries, need {n_layers}",
                    v.len()
                )));
            }
            Ok(v)
        };
        let swiglu_clamp_exp = per_layer("swiglu_clamp_exp")?;
        let swiglu_clamp_shexp = per_layer("swiglu_clamp_shexp")?;

        let ratios_all = m.u64s("attention.compress_ratios")?;
        if ratios_all.len() < n_layers {
            return Err(bad(format!(
                "attention.compress_ratios has {} entries, need {n_layers}",
                ratios_all.len()
            )));
        }
        let mut compress_ratios = Vec::with_capacity(n_layers);
        for (il, &r) in ratios_all[..n_layers].iter().enumerate() {
            if r > 2 {
                return Err(bad(format!(
                    "layer {il}: compress ratio {r} is not 0, 1 or 2"
                )));
            }
            compress_ratios.push(r as u32);
        }

        // PR #30817 deepseek41.cpp load_arch_hparams: each compressed layer reads
        // the KV / top-k of the latest source layer at or before it.
        let kv_source_layers = m.usizes("attention.kv_source_layers")?;
        let index_source_layers = m.usizes("attention.indexer.source_layers")?;
        if kv_source_layers
            .iter()
            .chain(&index_source_layers)
            .any(|&l| l >= n_layers)
        {
            return Err(bad("kv/index source layer out of range"));
        }
        let mut kv_src = vec![None; n_layers];
        let mut topk_src = vec![None; n_layers];
        let (mut last_kv, mut last_idx) = (None::<usize>, None::<usize>);
        for il in 0..n_layers {
            let ratio = compress_ratios[il];
            let is_kv = kv_source_layers.contains(&il);
            let is_idx = index_source_layers.contains(&il);
            if (is_kv || is_idx) && ratio == 0 {
                return Err(bad(format!("source layer {il} has compress ratio 0")));
            }
            if is_kv {
                last_kv = Some(il);
            }
            if is_idx {
                last_idx = Some(il);
            }
            if ratio == 0 {
                continue;
            }
            match (last_kv, last_idx) {
                (Some(k), Some(i))
                    if i >= k && compress_ratios[k] == ratio && compress_ratios[i] == ratio =>
                {
                    kv_src[il] = Some(k);
                    topk_src[il] = Some(i);
                }
                _ => {
                    return Err(bad(format!(
                        "layer {il}: ratio {ratio} without a matching kv/index source"
                    )));
                }
            }
        }

        let candidate = match m.opt("attention.indexer.candidate_source_layer") {
            None => None,
            Some(v) => match Meta::int(v) {
                Some(cs) if cs < 0 => None,
                Some(cs) => {
                    let cs = cs as usize;
                    let c = CandidateConfig {
                        source_layer: cs,
                        topk_blocks: m.usize("attention.indexer.candidate_top_k_blocks")?,
                        block_size: m.usize("attention.indexer.candidate_block_size")?,
                    };
                    if cs >= n_layers
                        || topk_src[cs] != Some(cs)
                        || c.block_size == 0
                        || c.topk_blocks == 0
                    {
                        return Err(bad(
                            "invalid candidate selection (source must be an index source, block size and top-k blocks > 0)",
                        ));
                    }
                    if (cs + 1..n_layers).any(|il| {
                        topk_src[il] == Some(il) && compress_ratios[il] != compress_ratios[cs]
                    }) {
                        return Err(bad(
                            "candidate consumers must share the source's compress ratio",
                        ));
                    }
                    Some(c)
                }
                None => return Err(bad("candidate_source_layer is not an integer")),
            },
        };

        let yarn = match md
            .get(&format!("{ARCH}.rope.scaling.type"))
            .and_then(GgufMetaValue::as_str)
        {
            None => None,
            Some("yarn") => Some(YarnConfig {
                factor: m.f32("rope.scaling.factor")?,
                original_seq_len: m.usize("rope.scaling.original_context_length")?,
                beta_fast: m.f32("rope.scaling.yarn_beta_fast")?,
                beta_slow: m.f32("rope.scaling.yarn_beta_slow")?,
            }),
            Some(other) => return Err(bad(format!("rope scaling {other:?} is not yarn"))),
        };

        let engram = Self::engram(&m, n_layers, vocab_size)?;
        let dspark = match m.opt("dspark.block_size") {
            None => None,
            Some(_) => Some(DsparkConfig {
                block_size: m.usize("dspark.block_size")?,
                noise_token_id: m.usize("dspark.noise_token_id")?,
                target_layers: m.usizes("dspark.target_layers")?,
                markov_rank: m.usize("dspark.markov_rank")?,
                expert_count: m.usize("dspark.expert_count")?,
                expert_used_count: m.usize("dspark.expert_used_count")?,
            }),
        };

        let cfg = Self {
            n_layers,
            vocab_size,
            context_length: m.usize("context_length")?,
            dim,
            n_heads: m.usize("attention.head_count")?,
            head_dim,
            rope_head_dim: m.usize("rope.dimension_count")?,
            q_lora_rank: m.usize("attention.q_lora_rank")?,
            o_lora_rank: m.usize("attention.output_lora_rank")?,
            o_groups: m.usize("attention.output_group_count")?,
            window_size: m.usize("attention.sliding_window")?,
            norm_eps: m.f32("attention.layer_norm_rms_epsilon")?,
            yarn,
            rope_theta: m.f32("rope.freq_base")?,
            compress_rope_theta: m.f32("attention.compress_rope_freq_base")?,
            n_routed_experts: m.usize("expert_count")?,
            n_activated_experts: m.usize("expert_used_count")?,
            n_shared_experts: m.usize("expert_shared_count")?,
            moe_inter_dim: m.usize("expert_feed_forward_length")?,
            route_scale: m.f32("expert_weights_scale")?,
            norm_topk_prob: m.bool("expert_weights_norm")?,
            swiglu_clamp_exp,
            swiglu_clamp_shexp,
            index_n_heads: m.usize("attention.indexer.head_count")?,
            index_head_dim: m.usize("attention.indexer.key_length")?,
            index_topk: m.usize("attention.indexer.top_k")?,
            compress_ratios,
            kv_source_layers,
            index_source_layers,
            kv_src,
            topk_src,
            candidate,
            hc_mult,
            hc_sinkhorn_iters: m.usize("hyper_connection.sinkhorn_iterations")?,
            hc_eps: m.f32("hyper_connection.epsilon")?,
            engram,
            dspark,
        };
        if cfg.index_topk == 0 {
            return Err(bad("indexer top_k must be > 0"));
        }
        if cfg.n_activated_experts == 0 || cfg.n_activated_experts > cfg.n_routed_experts {
            return Err(bad("expert_used_count must be in 1..=expert_count"));
        }
        if cfg.o_groups == 0
            || !cfg.n_heads.is_multiple_of(cfg.o_groups)
            || cfg.rope_head_dim > cfg.head_dim
        {
            return Err(bad("attention shape (o groups / rope dim) is inconsistent"));
        }
        Ok(cfg)
    }

    fn engram(m: &Meta, n_layers: usize, vocab_size: usize) -> Result<Option<EngramConfig>> {
        if m.opt("ple.layers").is_none() {
            return Ok(None);
        }
        let layers = m.usizes("ple.layers")?;
        if layers.is_empty() {
            return Ok(None);
        }
        let token_map = m
            .u64s("engram.token_map")?
            .into_iter()
            .map(|x| u32::try_from(x).map_err(|_| bad("engram.token_map entry overflows u32")))
            .collect::<Result<Vec<_>>>()?;
        let e = EngramConfig {
            max_ngram: m.usize("ple.ngram_size")?,
            n_heads: m.usize("ple.heads_per_ngram")?,
            head_dim: m.usize("embedding_length_per_layer_input")?,
            multipliers: m.u64s("ple.layer_multipliers")?,
            head_offsets: m.u64s("ple.head_offsets")?,
            head_vocab_sizes: m.u64s("ple.head_vocab_sizes")?,
            pad_token_id: m.usize("engram.pad_token_id")?,
            compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
            token_map,
            vocab_size: m
                .opt("engram.vocab_size")
                .map(|_| m.usize("engram.vocab_size"))
                .transpose()?
                .map(|v| v as u64),
            layers,
        };
        let n_e = e.layers.len();
        if e.max_ngram < 2
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
        if let Some(c) = m.opt_usize("engram.compressed_vocab_size")?
            && c != e.compressed_vocab_size
        {
            return Err(bad(format!(
                "engram.compressed_vocab_size {c} != max(token_map) + 1 = {}",
                e.compressed_vocab_size
            )));
        }
        if e.layers.iter().any(|&l| l >= n_layers) {
            return Err(bad("engram layer out of range"));
        }
        for (c, (&p, &o)) in e.head_vocab_sizes.iter().zip(&e.head_offsets).enumerate() {
            // a bucket count of 0 traps in the hash; row ids are int32 (PR #30817)
            if p < 2 || o.checked_add(p).is_none_or(|end| end > i32::MAX as u64) {
                return Err(bad(format!("engram head {c}: bucket range out of range")));
            }
        }
        Ok(Some(e))
    }
}
