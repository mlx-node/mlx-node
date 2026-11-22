//! Inkling-2-Mini DSpark parallel block drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`).
//!
//! Architecture (verified against `vllm/models/inkling2/nvidia/dspark.py` & `dspark-Inkling-2-Mini-Q8_0.gguf`):
//! - 3 dense Q8_0 transformer layers (`hidden=2304`, `ffn=9216`, `24Q / 8KV`, `head_dim=128`,
//!   NeoX RoPE `theta=1e6`, `sliding_window=1024`, `sliding_window_pattern=[true, false, true]`).
//! - Target hidden state taps `[5, 23, 41, 48]` mixed via `fuse_alpha = softmax(f32(dspark_fuse), -1)`
//!   and per-layer `dflare_norm` into committed BF16 context K/V caches (sliding 1024 on layers 0 & 2,
//!   global on layer 1) plus Markov context `f_rnn` (`fuse_alpha[3]` + `dspark_norm`) from the last
//!   committed token.
//! - Bidirectional 8-token query block `[embed_norm(embed(anchor)), mask_embd x 7]` -> `output_norm`
//!   -> positions `1..8` (`7` draft positions).
//! - Sequential 7-step recurrent Markov head (`rank = 256`) with `dspark_hidden_out` pre-folded once
//!   at load time in **FP64 -> BF16** into the target's shared `lm_head`:
//!   * `fold_weight = bf16(lm_head.W_f64 @ dspark_hidden_out.W_f64)` (`[201024, 256]`)
//!   * `fold_bias   = bf16(lm_head.W_f64 @ dspark_hidden_out.b_f64)` (`[201024]`)

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use napi::bindgen_prelude::{Error, Result};
use rand::RngExt;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::engine::backend::{DeviceSparseRows, DsparkProposal};
use crate::engine::dspark_turn::sampled_rows_take;
use crate::engine::params::ChatParams;
use crate::models::quantized_linear::{DEFAULT_QUANT_MODE, LinearProj, QuantizedLinear};
use crate::nn::{Activations, Embedding, Linear, RoPE};
use crate::sampling::{self, is_greedy_temperature};
use crate::utils::gguf::{
    GgufLoadOptions, GgufMetaValue, is_inkling2_dspark_gguf, load_gguf_tensors, parse_gguf,
};

use super::attn_res::apply_sublayer_rms_norm;
use super::config::Inkling2Config;

/// Total rows in one DSpark block (`1` anchor + `7` draft tokens).
pub(crate) const DSPARK_BLOCK_SIZE: usize = 8;

/// Maximum sparse top-K width supported by `dflash2_rows.metal` (`DFLASH2_ROWS_MAX_CANDIDATES`).
const DEVICE_SAMPLED_MAX_TOP_K: i32 = 64;

/// Row chunk size when folding `lm_head` (`[201024, 2304]`) into `fold_weight` (`[201024, 256]`)
/// in FP64 so peak host memory stays bounded (~75 MiB per chunk).
const FOLD_CHUNK_ROWS: usize = 4096;

/// Hyperparameters parsed from a DSpark GGUF (`dspark-Inkling-2-Mini-Q8_0.gguf`).
#[derive(Debug, Clone)]
pub struct Inkling2DsparkDraftConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: Vec<bool>,
    pub final_logit_softcapping: f64,
    pub max_position_embeddings: usize,
}

impl Inkling2DsparkDraftConfig {
    /// Draft tokens proposed per cycle (`block_size - 1`, i.e. `7` for Inkling-2-Mini).
    pub fn num_draft_tokens(&self) -> usize {
        self.block_size.saturating_sub(1)
    }

    /// Parse and validate DSpark hyperparameters from GGUF metadata.
    pub fn from_gguf_metadata(
        meta: &HashMap<String, GgufMetaValue>,
        target_config: &Inkling2Config,
    ) -> Result<Self> {
        if !is_inkling2_dspark_gguf(meta) {
            return Err(Error::from_reason(
                "GGUF file is not an Inkling2 DSpark draft checkpoint (expected general.architecture=\"dflash\" with dflash.dspark.markov_rank)",
            ));
        }
        let get_u = |k: &str| {
            meta.get(k)
                .and_then(GgufMetaValue::as_u64)
                .map(|v| v as usize)
        };
        let get_f = |k: &str| meta.get(k).and_then(GgufMetaValue::as_f32).map(f64::from);

        let hidden_size = get_u("dflash.embedding_length").unwrap_or(target_config.hidden_size);
        let intermediate_size = get_u("dflash.feed_forward_length").unwrap_or(hidden_size * 4);
        let num_hidden_layers = get_u("dflash.block_count").unwrap_or(3);
        let num_attention_heads =
            get_u("dflash.attention.head_count").unwrap_or(target_config.num_attention_heads);
        let num_key_value_heads =
            get_u("dflash.attention.head_count_kv").unwrap_or(target_config.num_key_value_heads);
        let head_dim = get_u("dflash.attention.key_length").unwrap_or(target_config.head_dim);
        let rope_theta = get_f("dflash.rope.freq_base").unwrap_or(1_000_000.0);
        let rms_norm_eps = get_f("dflash.attention.layer_norm_rms_epsilon")
            .unwrap_or(target_config.rms_norm_eps) as f32;
        let vocab_size = get_u("dflash.vocab_size")
            .unwrap_or_else(|| target_config.effective_padded_vocab_size());
        let unpadded_vocab_size = get_u("dflash.vocab_size_unpadded")
            .unwrap_or_else(|| target_config.effective_unpadded_vocab_size());
        let block_size = get_u("dflash.block_size").unwrap_or(DSPARK_BLOCK_SIZE);
        let markov_rank = get_u("dflash.dspark.markov_rank").unwrap_or(256);
        let sliding_window =
            get_u("dflash.attention.sliding_window").unwrap_or(target_config.sliding_window);
        let final_logit_softcapping = get_f("dflash.final_logit_softcapping")
            .unwrap_or(target_config.final_logit_softcapping);
        let max_position_embeddings =
            get_u("dflash.context_length").unwrap_or(target_config.max_position_embeddings);

        let target_layer_ids: Vec<usize> = match meta.get("dflash.target_layers") {
            Some(GgufMetaValue::ArrayU32(arr)) => arr.iter().map(|&v| v as usize).collect(),
            Some(GgufMetaValue::ArrayI32(arr)) => arr
                .iter()
                .map(|&v| {
                    usize::try_from(v)
                        .map_err(|_| Error::from_reason("negative dflash.target_layers entry"))
                })
                .collect::<Result<Vec<_>>>()?,
            Some(GgufMetaValue::ArrayU64(arr)) => arr.iter().map(|&v| v as usize).collect(),
            _ => {
                return Err(Error::from_reason(
                    "Inkling2 DSpark GGUF missing dflash.target_layers array",
                ));
            }
        };

        let sliding_window_pattern: Vec<bool> = match meta
            .get("dflash.attention.sliding_window_pattern")
        {
            Some(GgufMetaValue::ArrayBool(arr)) if arr.len() == num_hidden_layers => arr.clone(),
            _ => (0..num_hidden_layers)
                .map(|i| i != num_hidden_layers / 2)
                .collect(),
        };

        if hidden_size != target_config.hidden_size {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark hidden_size ({hidden_size}) != target ({})",
                target_config.hidden_size
            )));
        }
        if vocab_size != target_config.effective_padded_vocab_size() {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark vocab_size ({vocab_size}) != target padded vocab ({})",
                target_config.effective_padded_vocab_size()
            )));
        }
        if block_size < 2 || num_hidden_layers == 0 || markov_rank == 0 {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark invalid geometry: block_size={block_size}, layers={num_hidden_layers}, rank={markov_rank}"
            )));
        }
        let max_visit = target_config.num_layer_visits();
        if target_layer_ids.is_empty()
            || target_layer_ids.iter().any(|&v| v > max_visit)
            || !target_layer_ids.windows(2).all(|w| w[0] < w[1])
        {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark invalid target_layer_ids={target_layer_ids:?} for target visits={max_visit}"
            )));
        }

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_theta,
            rms_norm_eps,
            vocab_size,
            unpadded_vocab_size,
            block_size,
            target_layer_ids,
            markov_rank,
            sliding_window,
            sliding_window_pattern,
            final_logit_softcapping,
            max_position_embeddings,
        })
    }

    /// Conservative per-sequence BF16 draft KV cache footprint in bytes for `total_tokens`.
    pub fn context_bytes_for_tokens(&self, total_tokens: u32) -> u64 {
        let kv_per_tok = (self.num_key_value_heads as u64)
            .saturating_mul(self.head_dim as u64)
            .saturating_mul(4); // K + V in BF16 (2 bytes each)
        let mut total = 0u64;
        for &is_local in &self.sliding_window_pattern {
            let len = if is_local {
                u64::from(total_tokens).min(self.sliding_window as u64)
            } else {
                u64::from(total_tokens)
            };
            total = total.saturating_add(len.saturating_mul(kv_per_tok));
        }
        // Plus `f_rnn` (`[1, hidden_size]` BF16) + 8-token query scratch cushion.
        total.saturating_add((self.hidden_size as u64).saturating_mul(32))
    }
}

/// Auto-discover a colocated Inkling2 DSpark GGUF (`dspark-*.gguf` or any `*.gguf` matching
/// `is_inkling2_dspark_gguf`) inside `model_dir`.
pub(crate) fn discover_inkling2_dspark_gguf(model_dir: &Path) -> Option<PathBuf> {
    let dir = if model_dir.is_file() {
        model_dir.parent()?
    } else {
        model_dir
    };
    let mut dirs = vec![dir.to_path_buf()];
    if let Ok(canon) = fs::canonicalize(dir)
        && canon != dir
    {
        dirs.push(canon);
    }

    for candidate_dir in dirs {
        let Ok(entries) = fs::read_dir(&candidate_dir) else {
            continue;
        };
        let mut dspark_named = Vec::new();
        let mut other_ggufs = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let is_file = fs::metadata(&path).is_ok_and(|m| m.is_file());
            if !is_file {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".gguf") {
                continue;
            }
            if lower.starts_with("dspark-") || lower.contains("dspark") {
                dspark_named.push(path);
            } else if !lower.starts_with("mmproj") {
                other_ggufs.push(path);
            }
        }
        dspark_named.sort();
        other_ggufs.sort();
        for path in dspark_named.into_iter().chain(other_ggufs) {
            if let Ok(gguf) = parse_gguf(&path)
                && is_inkling2_dspark_gguf(&gguf.metadata)
            {
                return Some(path);
            }
        }
    }
    None
}

/// Resolve an explicit `draft_model_path` (file or directory) or auto-discover beside `asset_dir`.
///
/// - `Some("")` or `Some("none")` / `Some("off")` explicitly disables DSpark loading.
/// - `Some(path)` resolves a GGUF file or directory containing a DSpark GGUF (errors if missing).
/// - `None` auto-discovers a matching DSpark GGUF in `asset_dir` (returns `Ok(None)` if absent).
pub(crate) fn resolve_inkling2_dspark_path(
    asset_dir: &Path,
    explicit: Option<&str>,
) -> Result<Option<PathBuf>> {
    match explicit {
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty()
                || trimmed.eq_ignore_ascii_case("none")
                || trimmed.eq_ignore_ascii_case("off")
            {
                return Ok(None);
            }
            let p = PathBuf::from(trimmed);
            if p.is_file() {
                return Ok(Some(p));
            }
            if p.is_dir() {
                return discover_inkling2_dspark_gguf(&p)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "No Inkling2 DSpark GGUF found in draftModelPath directory {}",
                            p.display()
                        ))
                    })
                    .map(Some);
            }
            Err(Error::from_reason(format!(
                "Inkling2 draftModelPath does not exist: {}",
                p.display()
            )))
        }
        None => Ok(discover_inkling2_dspark_gguf(asset_dir)),
    }
}

/// One layer of the 3-layer Inkling2 DSpark parallel block drafter.
pub(crate) struct Inkling2DsparkLayer {
    pub is_local: bool,
    pub input_layernorm: MxArray,          // [H] F32
    pub post_attention_layernorm: MxArray, // [H] F32
    pub dflare_norm: MxArray,              // [H] F32
    pub q_norm: MxArray,                   // [D] F32
    pub k_norm: MxArray,                   // [D] F32
    pub k_ctx_norm: MxArray,               // [D] F32
    /// Fused `[Q, K, V]` (`[3072 + 1024 + 1024, 2304]`) when projections share format, else separate.
    pub fused_qkv: Option<LinearProj>,
    pub q_proj: LinearProj,
    pub k_proj: LinearProj,
    pub v_proj: LinearProj,
    /// Fused `[K_ctx, V_ctx]` (`[1024 + 1024, 2304]`).
    pub fused_kv_ctx: Option<LinearProj>,
    pub k_ctx_proj: LinearProj,
    pub v_ctx_proj: LinearProj,
    pub o_proj: LinearProj,
    /// Fused `[Gate, Up]` (`[9216 + 9216, 2304]`).
    pub fused_gate_up: Option<LinearProj>,
    pub gate_proj: LinearProj,
    pub up_proj: LinearProj,
    pub down_proj: LinearProj,
}

impl Inkling2DsparkLayer {
    fn weight_arrays(&self) -> Vec<MxArray> {
        let mut out = vec![
            self.input_layernorm.clone(),
            self.post_attention_layernorm.clone(),
            self.dflare_norm.clone(),
            self.q_norm.clone(),
            self.k_norm.clone(),
            self.k_ctx_norm.clone(),
        ];
        if let Some(ref f) = self.fused_qkv {
            super::attention::push_linear_proj_arrays(f, &mut out);
        } else {
            super::attention::push_linear_proj_arrays(&self.q_proj, &mut out);
            super::attention::push_linear_proj_arrays(&self.k_proj, &mut out);
            super::attention::push_linear_proj_arrays(&self.v_proj, &mut out);
        }
        if let Some(ref f) = self.fused_kv_ctx {
            super::attention::push_linear_proj_arrays(f, &mut out);
        } else {
            super::attention::push_linear_proj_arrays(&self.k_ctx_proj, &mut out);
            super::attention::push_linear_proj_arrays(&self.v_ctx_proj, &mut out);
        }
        super::attention::push_linear_proj_arrays(&self.o_proj, &mut out);
        if let Some(ref f) = self.fused_gate_up {
            super::attention::push_linear_proj_arrays(f, &mut out);
        } else {
            super::attention::push_linear_proj_arrays(&self.gate_proj, &mut out);
            super::attention::push_linear_proj_arrays(&self.up_proj, &mut out);
        }
        super::attention::push_linear_proj_arrays(&self.down_proj, &mut out);
        out
    }
}

/// Recurrent Markov head with `dspark_hidden_out` folded into `lm_head` once in FP64 -> BF16.
pub(crate) struct Inkling2MarkovHead {
    pub markov_rank: usize,
    pub dflare_norm: MxArray,      // [H] F32
    pub markov_w1: MxArray,        // [padded_vocab, R] BF16
    pub rnn_bias_f32: MxArray,     // [1, 3*R] F32
    pub rnn_proj_rec_w_t: MxArray, // [2*R, 3*R] BF16 (`joint_proj.W[:, :2*R].T`)
    pub rnn_proj_ctx_w_t: MxArray, // [2*H, 3*R] BF16 (`joint_proj.W[:, 2*R:].T`)
    pub fold_weight_t: MxArray,    // [R, padded_vocab] BF16 (`(lm_head @ hidden_out.W).T`)
    pub fold_bias: MxArray,        // [1, padded_vocab] BF16 (`lm_head @ hidden_out.b`)
    pub one_scalar: MxArray,       // Pre-allocated FP32 `1.0` scalar for `markov_gate`
}

impl Inkling2MarkovHead {
    fn weight_arrays(&self) -> Vec<MxArray> {
        vec![
            self.dflare_norm.clone(),
            self.markov_w1.clone(),
            self.rnn_bias_f32.clone(),
            self.rnn_proj_rec_w_t.clone(),
            self.rnn_proj_ctx_w_t.clone(),
            self.fold_weight_t.clone(),
            self.fold_bias.clone(),
            self.one_scalar.clone(),
        ]
    }
}

/// Loaded Inkling-2-Mini DSpark draft model.
pub struct Inkling2DsparkModel {
    pub config: Inkling2DsparkDraftConfig,
    pub(crate) layers: Vec<Inkling2DsparkLayer>,
    pub(crate) output_norm: MxArray, // [H] F32
    pub(crate) mask_embd: MxArray,   // [1, H] BF16
    /// Softmax-normalized mixing weights `[num_layers + 1, num_taps]` (rows `0..L` = draft layers,
    /// row `L` = Markov head `f_rnn`).
    pub(crate) fuse_alpha: Vec<Vec<f32>>,
    pub(crate) markov_head: Inkling2MarkovHead,
    pub(crate) rope: RoPE,
}

impl Inkling2DsparkModel {
    /// Load `dspark-Inkling-2-Mini-Q8_0.gguf` and fold its Markov head into `target_lm_head`
    /// in FP64 -> BF16 (chunked so peak RAM stays small).
    pub fn load_gguf(
        draft_path: &Path,
        target_config: &Inkling2Config,
        target_lm_head: &LinearProj,
    ) -> Result<Self> {
        let gguf = parse_gguf(draft_path)?;
        let config = Inkling2DsparkDraftConfig::from_gguf_metadata(&gguf.metadata, target_config)?;
        let tensors = load_gguf_tensors(draft_path, &gguf, GgufLoadOptions::default())?;

        let d = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let ffn = config.intermediate_size;
        let rank = config.markov_rank;

        // 1. Softmax fuse weights `[num_layers + 1, num_taps]`
        let fuse_raw = require_draft_tensor(&tensors, "dspark_fuse.weight")?.to_float32()?;
        let n_rows = config.num_hidden_layers + 1;
        let n_taps = config.target_layer_ids.len();
        if fuse_raw.len() != n_rows * n_taps {
            return Err(Error::from_reason(format!(
                "dspark_fuse.weight has {} elems, expected {n_rows}x{n_taps}",
                fuse_raw.len()
            )));
        }
        let mut fuse_alpha = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row = &fuse_raw[r * n_taps..(r + 1) * n_taps];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            fuse_alpha.push(exps.into_iter().map(|e| e / sum).collect());
        }

        // 2. Load 3 Q8_0 transformer layers (hoisting Q8_0 Float16 sidecars to F32 BEFORE row-concat fusion!)
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (i, &is_local) in config.sliding_window_pattern.iter().enumerate() {
            let p = format!("blk.{i}");
            let q_proj = load_draft_linear(&tensors, &format!("{p}.attn_q"), d, q_dim)?;
            let k_proj = load_draft_linear(&tensors, &format!("{p}.attn_k"), d, kv_dim)?;
            let v_proj = load_draft_linear(&tensors, &format!("{p}.attn_v"), d, kv_dim)?;
            let k_ctx_proj = load_draft_linear(&tensors, &format!("{p}.attn_k_ctx"), d, kv_dim)?;
            let v_ctx_proj = load_draft_linear(&tensors, &format!("{p}.attn_v_ctx"), d, kv_dim)?;
            let o_proj = load_draft_linear(&tensors, &format!("{p}.attn_output"), q_dim, d)?;
            let gate_proj = load_draft_linear(&tensors, &format!("{p}.ffn_gate"), d, ffn)?;
            let up_proj = load_draft_linear(&tensors, &format!("{p}.ffn_up"), d, ffn)?;
            let down_proj = load_draft_linear(&tensors, &format!("{p}.ffn_down"), ffn, d)?;

            let fused_qkv = match q_proj.concat_rows(&k_proj)? {
                Some(qk) => qk.concat_rows(&v_proj)?,
                None => None,
            };
            let fused_kv_ctx = k_ctx_proj.concat_rows(&v_ctx_proj)?;
            let fused_gate_up = gate_proj.concat_rows(&up_proj)?;

            layers.push(Inkling2DsparkLayer {
                is_local,
                input_layernorm: require_draft_tensor(&tensors, &format!("{p}.attn_norm.weight"))?
                    .astype(DType::Float32)?,
                post_attention_layernorm: require_draft_tensor(
                    &tensors,
                    &format!("{p}.ffn_norm.weight"),
                )?
                .astype(DType::Float32)?,
                dflare_norm: require_draft_tensor(&tensors, &format!("{p}.dflare_norm.weight"))?
                    .astype(DType::Float32)?,
                q_norm: require_draft_tensor(&tensors, &format!("{p}.attn_q_norm.weight"))?
                    .astype(DType::Float32)?,
                k_norm: require_draft_tensor(&tensors, &format!("{p}.attn_k_norm.weight"))?
                    .astype(DType::Float32)?,
                k_ctx_norm: require_draft_tensor(&tensors, &format!("{p}.attn_k_ctx_norm.weight"))?
                    .astype(DType::Float32)?,
                fused_qkv,
                q_proj,
                k_proj,
                v_proj,
                fused_kv_ctx,
                k_ctx_proj,
                v_ctx_proj,
                o_proj,
                fused_gate_up,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        // 3. Global output norm + mask embedding
        let output_norm =
            require_draft_tensor(&tensors, "output_norm.weight")?.astype(DType::Float32)?;
        let mask_embd = require_draft_tensor(&tensors, "mask_embd.weight")?
            .astype(DType::BFloat16)?
            .reshape(&[1, d as i64])?;

        // 4. Markov RNN head + FP64 fold into target `lm_head`
        let dspark_norm =
            require_draft_tensor(&tensors, "dspark_norm.weight")?.astype(DType::Float32)?;
        let markov_w1 =
            require_draft_tensor(&tensors, "markov_w1.weight")?.astype(DType::BFloat16)?;
        let joint_w =
            require_draft_tensor(&tensors, "joint_proj.weight")?.astype(DType::BFloat16)?;
        let joint_b = require_draft_tensor(&tensors, "joint_proj.bias")?
            .astype(DType::Float32)?
            .reshape(&[1, (3 * rank) as i64])?;

        let jw_shape = joint_w.shape()?;
        let expected_in = (2 * rank + 2 * d) as i64;
        let expected_out = (3 * rank) as i64;
        if jw_shape.as_ref() != [expected_out, expected_in] {
            return Err(Error::from_reason(format!(
                "Inkling2 DSpark joint_proj.weight shape {:?}, expected [{expected_out}, {expected_in}]",
                jw_shape.as_ref()
            )));
        }
        let split_col = (2 * rank) as i64;
        let rnn_proj_rec_w_t = joint_w
            .slice(&[0, 0], &[expected_out, split_col])?
            .transpose(None)?;
        let rnn_proj_ctx_w_t = joint_w
            .slice(&[0, split_col], &[expected_out, expected_in])?
            .transpose(None)?;

        let hidden_out_w = require_draft_tensor(&tensors, "dspark_hidden_out.weight")?;
        let hidden_out_b = require_draft_tensor(&tensors, "dspark_hidden_out.bias")?;
        let (fold_weight_t, fold_bias) = fold_markov_head_f64(
            target_lm_head,
            hidden_out_w,
            hidden_out_b,
            config.vocab_size,
            d,
            rank,
        )?;

        let markov_head = Inkling2MarkovHead {
            markov_rank: rank,
            dflare_norm: dspark_norm,
            markov_w1,
            rnn_bias_f32: joint_b,
            rnn_proj_rec_w_t,
            rnn_proj_ctx_w_t,
            fold_weight_t,
            fold_bias,
            one_scalar: MxArray::scalar_float(1.0)?,
        };

        let rope = RoPE::new(
            config.head_dim as i32,
            Some(false), // NeoX half-split RoPE
            Some(config.rope_theta),
            Some(1.0),
        );

        let model = Self {
            config,
            layers,
            output_norm,
            mask_embd,
            fuse_alpha,
            markov_head,
            rope,
        };
        Ok(model)
    }

    /// All resident weight arrays of the DSpark drafter (for `materialize_weights` & accounting).
    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::new();
        for l in &self.layers {
            arrays.extend(l.weight_arrays());
        }
        arrays.push(self.output_norm.clone());
        arrays.push(self.mask_embd.clone());
        arrays.extend(self.markov_head.weight_arrays());
        arrays
    }

    /// Approximate resident weight bytes.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_arrays()
            .iter()
            .map(|a| a.nbytes() as u64)
            .sum::<u64>()
    }
}

/// Per-layer BF16 context KV cache for one Inkling2 DSpark sequence.
#[derive(Clone, Default)]
pub(crate) struct DraftLayerKV {
    pub(crate) k: Option<MxArray>, // [1, H_kv, L, D] BF16
    pub(crate) v: Option<MxArray>, // [1, H_kv, L, D] BF16
}

/// Per-sequence DSpark draft context (3 BF16 KV layers + Markov head `f_rnn` + cursor).
#[derive(Clone)]
pub struct Inkling2DsparkContext {
    pub(crate) layers: Vec<DraftLayerKV>,
    /// Markov head context `[1, H]` BF16 mixed from the last committed token's taps.
    pub(crate) last_f_rnn: Option<MxArray>,
    /// Total committed target tokens represented in this draft context (RoPE offset for next block).
    pub committed_len: i32,
}

impl Inkling2DsparkContext {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| DraftLayerKV::default()).collect(),
            last_f_rnn: None,
            committed_len: 0,
        }
    }

    pub fn collect_arrays<'a>(&'a self, out: &mut Vec<&'a MxArray>) {
        for layer in &self.layers {
            if let Some(ref k) = layer.k {
                out.push(k);
            }
            if let Some(ref v) = layer.v {
                out.push(v);
            }
        }
        if let Some(ref f_rnn) = self.last_f_rnn {
            out.push(f_rnn);
        }
    }

    pub fn eval_state(&self) -> Result<()> {
        let mut arrays = Vec::with_capacity(self.layers.len() * 2 + 1);
        self.collect_arrays(&mut arrays);
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays)?;
        }
        Ok(())
    }

    /// Append `keep` committed tokens' tapped target features (`taps`: `4` arrays of shape `[>= keep, H]` BF16)
    /// into all 3 draft layers' BF16 KV caches (applying DFlare mix + `k_ctx_norm` + NeoX RoPE +
    /// SWA(1024) trimming on layers 0 & 2) and update `last_f_rnn` from row `keep - 1`.
    pub fn append_taps(
        &mut self,
        draft: &Inkling2DsparkModel,
        taps: &[MxArray],
        keep: usize,
    ) -> Result<()> {
        if keep == 0 {
            return Ok(());
        }
        let n_taps = draft.config.target_layer_ids.len();
        if taps.len() != n_taps {
            return Err(Error::from_reason(format!(
                "Inkling2DsparkContext::append_taps expected {n_taps} taps, got {}",
                taps.len()
            )));
        }
        let d = draft.config.hidden_size as i64;
        let kv_dim = (draft.config.num_key_value_heads * draft.config.head_dim) as i64;
        let n_kv = draft.config.num_key_value_heads as i64;
        let hd = draft.config.head_dim as i64;
        let keep_i64 = keep as i64;
        let start_pos = self.committed_len;

        // Slice each tap to `[keep, H]` if caller passed a wider verify buffer (`[8, H]`).
        let mut sliced_taps = Vec::with_capacity(n_taps);
        for tap in taps {
            let rows = tap.shape()?[0];
            if rows < keep_i64 {
                return Err(Error::from_reason(format!(
                    "Inkling2DsparkContext::append_taps: tap has {rows} rows < keep={keep}"
                )));
            }
            if rows == keep_i64 {
                sliced_taps.push(tap.clone());
            } else {
                sliced_taps.push(tap.slice(&[0, 0], &[keep_i64, d])?);
            }
        }
        let tap_refs: Vec<&MxArray> = sliced_taps.iter().collect();

        // 1. Update each draft layer's context KV cache
        let swa = draft.config.sliding_window as i64;
        for (i, layer) in draft.layers.iter().enumerate() {
            let ctx = mix_rmsnorm(
                &tap_refs,
                &draft.fuse_alpha[i],
                &layer.dflare_norm,
                draft.config.rms_norm_eps,
            )?;
            let (k_raw, v_raw) = if let Some(ref fused) = layer.fused_kv_ctx {
                let kv = fused.forward(&ctx)?;
                let mut parts = kv.split_sections(&[kv_dim], -1)?;
                let v = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split kv_ctx: v"))?;
                let k = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split kv_ctx: k"))?;
                (k, v)
            } else {
                (
                    layer.k_ctx_proj.forward(&ctx)?,
                    layer.v_ctx_proj.forward(&ctx)?,
                )
            };

            let k_normed = apply_sublayer_rms_norm(
                &k_raw.reshape(&[keep_i64 * n_kv, hd])?,
                &layer.k_ctx_norm,
                draft.config.rms_norm_eps,
                DType::BFloat16,
            )?
            .reshape(&[1, keep_i64, n_kv, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let k_rot = draft.rope.forward(&k_normed, Some(start_pos))?;
            let v_4d = v_raw
                .reshape(&[1, keep_i64, n_kv, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?;

            let slot = &mut self.layers[i];
            let (mut k_cat, mut v_cat) = match (&slot.k, &slot.v) {
                (Some(prev_k), Some(prev_v)) => (
                    MxArray::concatenate(prev_k, &k_rot, 2)?,
                    MxArray::concatenate(prev_v, &v_4d, 2)?,
                ),
                _ => (k_rot, v_4d),
            };
            if layer.is_local {
                let len = k_cat.shape()?[2];
                if len > swa {
                    let drop_n = len - swa;
                    k_cat = k_cat.slice(&[0, 0, drop_n, 0], &[1, n_kv, len, hd])?;
                    v_cat = v_cat.slice(&[0, 0, drop_n, 0], &[1, n_kv, len, hd])?;
                }
            }
            slot.k = Some(k_cat);
            slot.v = Some(v_cat);
        }

        // 2. Compute Markov head `f_rnn` from the LAST committed token (`row keep - 1`)
        let last_row = keep_i64 - 1;
        let mut last_taps = Vec::with_capacity(n_taps);
        for tap in &sliced_taps {
            last_taps.push(tap.slice(&[last_row, 0], &[keep_i64, d])?);
        }
        let last_refs: Vec<&MxArray> = last_taps.iter().collect();
        let f_rnn = mix_rmsnorm(
            &last_refs,
            &draft.fuse_alpha[draft.config.num_hidden_layers],
            &draft.markov_head.dflare_norm,
            draft.config.rms_norm_eps,
        )?;
        self.last_f_rnn = Some(f_rnn);
        self.committed_len = start_pos
            .checked_add(keep as i32)
            .ok_or_else(|| Error::from_reason("Inkling2DsparkContext committed_len overflow"))?;
        Ok(())
    }

    /// Batched `append_taps` across `N` lanes: stacks all kept tap rows `[sum(keep_i), H]` so
    /// each draft layer runs **1** `mix_rmsnorm`, **1** Q8_0 `fused_kv_ctx`, and **1** `k_ctx_norm`
    /// instead of `N` separate launches, then applies per-lane RoPE (`start_pos_i`) & KV concat!
    pub fn append_taps_batched(
        draft: &Inkling2DsparkModel,
        items: &mut [(&mut Inkling2DsparkContext, &[MxArray], usize)],
    ) -> Result<()> {
        let active: Vec<usize> = items
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, keep))| (*keep > 0).then_some(i))
            .collect();
        if active.is_empty() {
            return Ok(());
        }
        if active.len() == 1 {
            let idx = active[0];
            let (ctx, taps, keep) = &mut items[idx];
            return ctx.append_taps(draft, taps, *keep);
        }

        let n_taps = draft.config.target_layer_ids.len();
        let d = draft.config.hidden_size as i64;
        let kv_dim = (draft.config.num_key_value_heads * draft.config.head_dim) as i64;
        let n_kv = draft.config.num_key_value_heads as i64;
        let hd = draft.config.head_dim as i64;
        let swa = draft.config.sliding_window as i64;

        // Slice each lane's taps to `[keep_i, H]` and collect last rows for `f_rnn`.
        let mut per_tap_slices: Vec<Vec<MxArray>> = (0..n_taps)
            .map(|_| Vec::with_capacity(active.len()))
            .collect();
        let mut per_tap_last: Vec<Vec<MxArray>> = (0..n_taps)
            .map(|_| Vec::with_capacity(active.len()))
            .collect();
        let mut keeps = Vec::with_capacity(active.len());
        let mut total_keep = 0i64;

        for &idx in &active {
            let (_, taps, keep) = &items[idx];
            if taps.len() != n_taps {
                return Err(Error::from_reason(format!(
                    "Inkling2DsparkContext::append_taps_batched expected {n_taps} taps, got {}",
                    taps.len()
                )));
            }
            let keep_i64 = *keep as i64;
            let last_row = keep_i64 - 1;
            for (t_idx, tap) in taps.iter().enumerate() {
                let rows = tap.shape()?[0];
                if rows < keep_i64 {
                    return Err(Error::from_reason(format!(
                        "Inkling2DsparkContext::append_taps_batched: tap has {rows} rows < keep={keep}"
                    )));
                }
                let sliced = if rows == keep_i64 {
                    tap.clone()
                } else {
                    tap.slice(&[0, 0], &[keep_i64, d])?
                };
                per_tap_last[t_idx].push(sliced.slice(&[last_row, 0], &[keep_i64, d])?);
                per_tap_slices[t_idx].push(sliced);
            }
            keeps.push(keep_i64);
            total_keep += keep_i64;
        }

        let mut stacked_taps = Vec::with_capacity(n_taps);
        let mut stacked_last = Vec::with_capacity(n_taps);
        for t_idx in 0..n_taps {
            let refs: Vec<&MxArray> = per_tap_slices[t_idx].iter().collect();
            stacked_taps.push(MxArray::concatenate_many(refs, Some(0))?); // [total_keep, H]
            let lrefs: Vec<&MxArray> = per_tap_last[t_idx].iter().collect();
            stacked_last.push(MxArray::concatenate_many(lrefs, Some(0))?); // [N_active, H]
        }
        let tap_refs: Vec<&MxArray> = stacked_taps.iter().collect();

        // 1. Batched per-layer `mix_rmsnorm` -> 1 `fused_kv_ctx` -> 1 `k_ctx_norm` -> per-lane RoPE/KV!
        for (layer_idx, layer) in draft.layers.iter().enumerate() {
            let ctx_all = mix_rmsnorm(
                &tap_refs,
                &draft.fuse_alpha[layer_idx],
                &layer.dflare_norm,
                draft.config.rms_norm_eps,
            )?;
            let (k_raw, v_raw) = if let Some(ref fused) = layer.fused_kv_ctx {
                let kv = fused.forward(&ctx_all)?;
                let mut parts = kv.split_sections(&[kv_dim], -1)?;
                let v = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split kv_ctx: v"))?;
                let k = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split kv_ctx: k"))?;
                (k, v)
            } else {
                (
                    layer.k_ctx_proj.forward(&ctx_all)?,
                    layer.v_ctx_proj.forward(&ctx_all)?,
                )
            };

            let k_normed_all = apply_sublayer_rms_norm(
                &k_raw.reshape(&[total_keep * n_kv, hd])?,
                &layer.k_ctx_norm,
                draft.config.rms_norm_eps,
                DType::BFloat16,
            )?
            .reshape(&[total_keep, n_kv, hd])?;
            let v_all = v_raw.reshape(&[total_keep, n_kv, hd])?;

            let mut off = 0i64;
            for (&idx, &keep_i64) in active.iter().zip(&keeps) {
                let end = off + keep_i64;
                let k_lane = k_normed_all
                    .slice(&[off, 0, 0], &[end, n_kv, hd])?
                    .reshape(&[1, keep_i64, n_kv, hd])?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let v_lane = v_all
                    .slice(&[off, 0, 0], &[end, n_kv, hd])?
                    .reshape(&[1, keep_i64, n_kv, hd])?
                    .transpose(Some(&[0, 2, 1, 3]))?;
                let ctx = &mut *items[idx].0;
                let start_pos = ctx.committed_len;
                let k_rot = draft.rope.forward(&k_lane, Some(start_pos))?;

                let slot = &mut ctx.layers[layer_idx];
                let (mut k_cat, mut v_cat) = match (&slot.k, &slot.v) {
                    (Some(prev_k), Some(prev_v)) => (
                        MxArray::concatenate(prev_k, &k_rot, 2)?,
                        MxArray::concatenate(prev_v, &v_lane, 2)?,
                    ),
                    _ => (k_rot, v_lane),
                };
                if layer.is_local {
                    let len = k_cat.shape()?[2];
                    if len > swa {
                        let drop_n = len - swa;
                        k_cat = k_cat.slice(&[0, 0, drop_n, 0], &[1, n_kv, len, hd])?;
                        v_cat = v_cat.slice(&[0, 0, drop_n, 0], &[1, n_kv, len, hd])?;
                    }
                }
                slot.k = Some(k_cat);
                slot.v = Some(v_cat);
                off = end;
            }
        }

        // 2. Batched `f_rnn` across all active lanes in ONE `mix_rmsnorm`!
        let last_refs: Vec<&MxArray> = stacked_last.iter().collect();
        let f_rnn_all = mix_rmsnorm(
            &last_refs,
            &draft.fuse_alpha[draft.config.num_hidden_layers],
            &draft.markov_head.dflare_norm,
            draft.config.rms_norm_eps,
        )?; // [N_active, H]
        for (slot, (&idx, &keep_i64)) in active.iter().zip(&keeps).enumerate() {
            let s_i64 = slot as i64;
            let ctx = &mut *items[idx].0;
            ctx.last_f_rnn = Some(f_rnn_all.slice(&[s_i64, 0], &[s_i64 + 1, d])?);
            ctx.committed_len =
                ctx.committed_len
                    .checked_add(keep_i64 as i32)
                    .ok_or_else(|| {
                        Error::from_reason("Inkling2DsparkContext committed_len overflow")
                    })?;
        }
        Ok(())
    }

    /// Run the 3-layer bidirectional block drafter + 7-step recurrent Markov head for `anchor_id`,
    /// returning a [`DsparkProposal`] of length `max_drafts.min(7)`.
    ///
    /// - Greedy turns (`is_greedy_temperature`): keeps all 7 RNN steps on-device in ONE Metal graph
    ///   with ZERO intermediate GPU->CPU readbacks and returns `device_draft_ids`.
    /// - Sampled turns with `top_k in 1..=64` + `sampled_rows_take(params)`: emits `device_draft_ids`
    ///   + `device_draft_sparse` so `dflash2_rows.metal` (`submit_sampled_rows`) runs fully on-device.
    /// - General sampled turns (including Inkling2 default `top_k = 0, top_p = 1.0`): computes exact
    ///   full-vocab `draft_dists` (`q_i`) and samples lossless proposals for `accept_dspark_proposal`.
    pub(crate) fn propose_block(
        &self,
        draft: &Inkling2DsparkModel,
        target_embed: &Embedding,
        target_embed_norm: &MxArray,
        target_lm_head: &LinearProj,
        anchor_id: u32,
        max_drafts: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let block = draft.config.block_size;
        let max_spec = draft.config.num_draft_tokens();
        let n_draft = max_drafts.min(max_spec);
        if n_draft == 0 {
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }
        let f_rnn = self.last_f_rnn.as_ref().ok_or_else(|| {
            Error::from_reason("Inkling2DsparkContext::propose_block called before seeding context")
        })?;

        let d = draft.config.hidden_size as i64;
        let n_q = draft.config.num_attention_heads as i64;
        let n_kv = draft.config.num_key_value_heads as i64;
        let hd = draft.config.head_dim as i64;
        let q_dim = n_q * hd;
        let kv_dim = n_kv * hd;
        let ffn = draft.config.intermediate_size as i64;
        let block_i64 = block as i64;
        let eps = draft.config.rms_norm_eps;
        let offset = self.committed_len;

        // 1. Build 8-token query block: `[embed_norm(embed(anchor)), mask_embd x 7]`
        let anchor_arr = MxArray::from_int32(&[anchor_id as i32], &[1])?;
        let anchor_raw = target_embed.forward(&anchor_arr)?;
        let anchor_emb =
            apply_sublayer_rms_norm(&anchor_raw, target_embed_norm, eps, DType::BFloat16)?;
        let masks = draft.mask_embd.broadcast_to(&[block_i64 - 1, d])?;
        let mut x = MxArray::concatenate(&anchor_emb, &masks, 0)?; // [8, H] BF16

        // 2. Run 3 bidirectional block-attention draft layers
        let attn_scale = 1.0_f64 / (draft.config.head_dim as f64).sqrt();
        for (i, layer) in draft.layers.iter().enumerate() {
            let h = apply_sublayer_rms_norm(&x, &layer.input_layernorm, eps, DType::BFloat16)?;

            let (q_raw, k_raw, v_raw) = if let Some(ref fused) = layer.fused_qkv {
                let qkv = fused.forward(&h)?;
                let mut parts = qkv.split_sections(&[q_dim, q_dim + kv_dim], -1)?;
                let v = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qkv: v"))?;
                let k = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qkv: k"))?;
                let q = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qkv: q"))?;
                (q, k, v)
            } else {
                (
                    layer.q_proj.forward(&h)?,
                    layer.k_proj.forward(&h)?,
                    layer.v_proj.forward(&h)?,
                )
            };

            let q_normed = apply_sublayer_rms_norm(
                &q_raw.reshape(&[block_i64 * n_q, hd])?,
                &layer.q_norm,
                eps,
                DType::BFloat16,
            )?
            .reshape(&[1, block_i64, n_q, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
            let k_normed = apply_sublayer_rms_norm(
                &k_raw.reshape(&[block_i64 * n_kv, hd])?,
                &layer.k_norm,
                eps,
                DType::BFloat16,
            )?
            .reshape(&[1, block_i64, n_kv, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;

            let q_rot = draft.rope.forward(&q_normed, Some(offset))?;
            let k_rot = draft.rope.forward(&k_normed, Some(offset))?;
            let v_4d = v_raw
                .reshape(&[1, block_i64, n_kv, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?;

            // Concatenate committed context KV (already trimmed to <= 1024 on local layers)
            // + the 8 block tokens. Because local window is `1024 + 8`, every block query attends
            // to ALL keys in `[ctx, block]` with NO mask needed!
            let (k_all, v_all) = match (&self.layers[i].k, &self.layers[i].v) {
                (Some(ck), Some(cv)) => (
                    MxArray::concatenate(ck, &k_rot, 2)?,
                    MxArray::concatenate(cv, &v_4d, 2)?,
                ),
                _ => (k_rot, v_4d),
            };

            let attn_out = scaled_dot_product_attention(&q_rot, &k_all, &v_all, attn_scale, None)?
                .transpose(Some(&[0, 2, 1, 3]))?
                .reshape(&[block_i64, q_dim])?;
            let o = layer.o_proj.forward(&attn_out)?;

            // Residual add in FP32 -> round to BF16
            x = x
                .astype(DType::Float32)?
                .add(&o.astype(DType::Float32)?)?
                .astype(DType::BFloat16)?;

            // MLP: SwiGLU (`silu(gate) * up` in FP32 -> BF16)
            let h_mlp =
                apply_sublayer_rms_norm(&x, &layer.post_attention_layernorm, eps, DType::BFloat16)?;
            let (gate, up) = if let Some(ref fused) = layer.fused_gate_up {
                let gu = fused.forward(&h_mlp)?;
                let mut parts = gu.split_sections(&[ffn], -1)?;
                let u = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split gu: up"))?;
                let g = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split gu: gate"))?;
                (g, u)
            } else {
                (
                    layer.gate_proj.forward(&h_mlp)?,
                    layer.up_proj.forward(&h_mlp)?,
                )
            };
            let act = Activations::silu(&gate.astype(DType::Float32)?)?
                .mul(&up.astype(DType::Float32)?)?
                .astype(DType::BFloat16)?;
            let mlp_out = layer.down_proj.forward(&act)?;
            x = x
                .astype(DType::Float32)?
                .add(&mlp_out.astype(DType::Float32)?)?
                .astype(DType::BFloat16)?;
        }

        // 3. Final output_norm -> slice positions `1..1+n_draft` (`[n_draft, H]` BF16)
        let out_normed = apply_sublayer_rms_norm(&x, &draft.output_norm, eps, DType::BFloat16)?;
        let n_draft_i64 = n_draft as i64;
        let hidden = out_normed.slice(&[1, 0], &[1 + n_draft_i64, d])?;

        // 4. Recurrent Markov head over `n_draft` steps
        let head = &draft.markov_head;
        let base_raw = target_lm_head.forward_logits(&hidden)?;
        let base_logits = if base_raw.dtype()? == DType::BFloat16 {
            base_raw
        } else {
            base_raw.astype(DType::BFloat16)?
        }; // [n_draft, padded_vocab] BF16
        let padded_vocab = base_logits.shape()?[1];
        let unpadded_vocab = (draft.config.unpadded_vocab_size as i64).min(padded_vocab);

        let f_rnn_bcast = f_rnn.broadcast_to(&[n_draft_i64, d])?;
        let ctx_in = MxArray::concatenate(&hidden, &f_rnn_bcast, -1)?; // [n_draft, 2*H] BF16
        let ctx_all = ctx_in
            .matmul(&head.rnn_proj_ctx_w_t)?
            .astype(DType::Float32)?
            .add(&head.rnn_bias_f32)?; // [n_draft, 3*R] F32

        let rank_i64 = head.markov_rank as i64;
        let mut state = MxArray::zeros(&[1, rank_i64], Some(DType::BFloat16))?;
        let mut prev_tok = anchor_arr; // [1] Int32

        let temperature = params
            .sampling_config
            .and_then(|c| c.temperature)
            .unwrap_or(1.0);
        let greedy = is_greedy_temperature(temperature);
        let top_k = params.sampling_config.and_then(|c| c.top_k).unwrap_or(0);
        let use_device_sparse =
            !greedy && sampled_rows_take(params) && (1..=DEVICE_SAMPLED_MAX_TOP_K).contains(&top_k);
        let softcap = draft.config.final_logit_softcapping;

        if greedy {
            // Fully on-device 7-step Markov loop: ZERO intermediate CPU syncs!
            let use_metal = {
                #[cfg(target_os = "macos")]
                {
                    unsafe { mlx_sys::mlx_metal_is_available() }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    false
                }
            };
            let mut token_parts = Vec::with_capacity(n_draft);
            for step in 0..n_draft {
                let s_i64 = step as i64;
                let ctx_i = ctx_all.slice(&[s_i64, 0], &[s_i64 + 1, 3 * rank_i64])?;
                let (next_state, bias) = markov_rnn_step(head, &state, &prev_tok, &ctx_i)?;
                state = next_state;

                let base_i = base_logits.slice(&[s_i64, 0], &[s_i64 + 1, padded_vocab])?;
                let chosen = if use_metal {
                    super::kernels::inkling2_lm_head_softcap_argmax(
                        &base_i,
                        Some(&bias),
                        unpadded_vocab as usize,
                        softcap as f32,
                        true,
                    )?
                } else {
                    let logits = markov_add_softcap_slice(&base_i, &bias, unpadded_vocab, softcap)?;
                    logits.argmax(-1, Some(false))?.astype(DType::Int32)?
                };
                prev_tok = chosen.clone();
                token_parts.push(chosen);
            }
            let refs: Vec<&MxArray> = token_parts.iter().collect();
            let device_draft_ids = MxArray::concatenate_many(refs, Some(0))?; // [n_draft] Int32
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: Some(device_draft_ids),
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }

        if use_device_sparse {
            // Device-side top-K sparse draft sampling feeding `dflash2_rows.metal` (`submit_sampled_rows`).
            let k_width = top_k as i64;
            let mut token_parts = Vec::with_capacity(n_draft);
            let mut cand_ids_parts = Vec::with_capacity(n_draft);
            let mut cand_probs_parts = Vec::with_capacity(n_draft);
            let mut uniforms = Vec::with_capacity(n_draft);
            for _ in 0..n_draft {
                uniforms.push(rng.random::<f32>());
            }

            for (step, &u) in uniforms.iter().enumerate() {
                let s_i64 = step as i64;
                let ctx_i = ctx_all.slice(&[s_i64, 0], &[s_i64 + 1, 3 * rank_i64])?;
                let (next_state, bias) = markov_rnn_step(head, &state, &prev_tok, &ctx_i)?;
                state = next_state;

                let base_i = base_logits.slice(&[s_i64, 0], &[s_i64 + 1, padded_vocab])?;
                let logits_1d = markov_add_softcap_slice(&base_i, &bias, unpadded_vocab, softcap)?
                    .squeeze(Some(&[0]))?;
                let dist = sampling::sampling_distribution(&logits_1d, params.sampling_config)?
                    .astype(DType::Float32)?;

                // Top-K sparse support (all nonzero mass lies inside top-K when top_k <= 64).
                let neg_dist = dist.negative()?;
                let sorted_idx = neg_dist.argpartition(top_k - 1, Some(-1))?;
                let top_idx = sorted_idx.slice(&[0], &[k_width])?.astype(DType::Int32)?; // [K]
                let top_p = dist.take(&top_idx, 0)?; // [K]
                let p_sum = top_p
                    .sum(None, Some(true))?
                    .maximum(&MxArray::scalar_float(1e-30)?)?;
                let norm_p = top_p.div(&p_sum)?; // [K]

                // Inverse-CDF draw on-device using uniform `u`.
                // Guard against FP32 cumsum rounding slightly below `u`: fall back to the
                // argmax entry (guaranteed nonzero mass) instead of unsorted `top_idx[0]`.
                let cdf = norm_p.cumsum(-1)?;
                let u_arr = MxArray::scalar_float(f64::from(u))?;
                let gt_i32 = cdf.greater(&u_arr)?.astype(DType::Int32)?;
                let any_gt = gt_i32
                    .sum(None, Some(false))?
                    .greater(&MxArray::scalar_int(0)?)?;
                let raw_slot = gt_i32.argmax(0, Some(false))?;
                let fallback_slot = norm_p.argmax(0, Some(false))?;
                let slot = any_gt.where_(&raw_slot, &fallback_slot)?.reshape(&[1])?;
                let chosen = top_idx.take(&slot, 0)?.astype(DType::Int32)?; // [1]
                prev_tok = chosen.clone();

                token_parts.push(chosen);
                cand_ids_parts.push(top_idx.reshape(&[1, k_width])?);
                cand_probs_parts.push(norm_p.reshape(&[1, k_width])?);
            }

            let tok_refs: Vec<&MxArray> = token_parts.iter().collect();
            let id_refs: Vec<&MxArray> = cand_ids_parts.iter().collect();
            let prob_refs: Vec<&MxArray> = cand_probs_parts.iter().collect();
            let device_draft_ids = MxArray::concatenate_many(tok_refs, Some(0))?;
            let sparse_ids = MxArray::concatenate_many(id_refs, Some(0))?;
            let sparse_probs = MxArray::concatenate_many(prob_refs, Some(0))?;
            return Ok(DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: Some(device_draft_ids),
                device_draft_sparse: Some(DeviceSparseRows {
                    ids: sparse_ids,
                    probs: sparse_probs,
                    vocab_size: unpadded_vocab as usize,
                }),
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            });
        }

        // Exact general sampled path (supports Inkling2 default `top_k = 0, top_p = 1.0`, penalties, etc.)
        let mut draft_ids = Vec::with_capacity(n_draft);
        let mut draft_dists = Vec::with_capacity(n_draft);
        for step in 0..n_draft {
            let s_i64 = step as i64;
            let ctx_i = ctx_all.slice(&[s_i64, 0], &[s_i64 + 1, 3 * rank_i64])?;
            let (next_state, bias) = markov_rnn_step(head, &state, &prev_tok, &ctx_i)?;
            state = next_state;

            let base_i = base_logits.slice(&[s_i64, 0], &[s_i64 + 1, padded_vocab])?;
            let logits_1d = markov_add_softcap_slice(&base_i, &bias, unpadded_vocab, softcap)?
                .squeeze(Some(&[0]))?;
            let dist = sampling::sampling_distribution(&logits_1d, params.sampling_config)?
                .astype(DType::Float32)?;
            let sampled = sampling::sample_dense_distribution(&dist, rng)?;
            draft_ids.push(sampled);
            draft_dists.push(dist);
            prev_tok = MxArray::from_int32(&[sampled], &[1])?;
        }
        Ok(DsparkProposal {
            draft_ids,
            device_draft_ids: None,
            device_draft_sparse: None,
            draft_dists,
            draft_sparse_dists: Vec::new(),
            keep_probabilities: None,
        })
    }
}

/// DFlare weighted sum + RMSNorm matching `vllm/models/inkling2/nvidia/ops/dspark.py::mix_rmsnorm`.
pub(crate) fn mix_rmsnorm(
    taps: &[&MxArray],
    alpha: &[f32],
    weight: &MxArray,
    eps: f32,
) -> Result<MxArray> {
    if taps.is_empty() || taps.len() != alpha.len() {
        return Err(Error::from_reason(format!(
            "mix_rmsnorm: mismatched taps ({}) vs alpha ({})",
            taps.len(),
            alpha.len()
        )));
    }
    let mut mixed = taps[0]
        .astype(DType::Float32)?
        .mul_scalar(f64::from(alpha[0]))?;
    for (tap, &a) in taps.iter().skip(1).zip(alpha.iter().skip(1)) {
        let term = tap.astype(DType::Float32)?.mul_scalar(f64::from(a))?;
        mixed = mixed.add(&term)?;
    }
    apply_sublayer_rms_norm(&mixed, weight, eps, DType::BFloat16)
}

/// Per-lane request for batched multi-lane [`Inkling2DsparkModel::propose_blocks_batched`].
pub(crate) struct BatchedDsparkLane<'a> {
    pub ctx: &'a Inkling2DsparkContext,
    pub anchor_id: u32,
    pub max_drafts: usize,
    pub params: &'a ChatParams,
}

impl Inkling2DsparkModel {
    /// Batched multi-lane DSpark block proposal: runs all `N` greedy lanes stacked as
    /// `[N * 8, H]` through the 3 Q8_0 draft layers (keeping per-lane KV attention exact!),
    /// `[N * n_spec, H]` through `target_lm_head`, and `[N, R] @ fold_weight_t` across
    /// `n_spec = max(cap_i).min(7)` Markov steps using fused `inkling2_markov_gate` +
    /// `inkling2_lm_head_softcap_argmax`.
    ///
    /// Cuts 4-lane draft DRAM traffic ~4x and allows the caller to materialize all lanes'
    /// `device_draft_ids` in **1 single GPU sync** instead of `N` serial syncs!
    pub(crate) fn propose_blocks_batched(
        &self,
        target_embed: &Embedding,
        target_embed_norm: &MxArray,
        target_lm_head: &LinearProj,
        lanes: &[BatchedDsparkLane<'_>],
        rngs: &mut [&mut dyn rand::Rng],
    ) -> Result<Vec<DsparkProposal>> {
        if lanes.len() != rngs.len() {
            return Err(Error::from_reason(format!(
                "propose_blocks_batched: lanes ({}) != rngs ({})",
                lanes.len(),
                rngs.len()
            )));
        }
        let max_spec = self.config.num_draft_tokens();
        let mut active_indices = Vec::with_capacity(lanes.len());
        let mut active_caps = Vec::with_capacity(lanes.len());
        let mut all_greedy = true;
        for (idx, lane) in lanes.iter().enumerate() {
            let cap = lane.max_drafts.min(max_spec);
            if cap > 0 {
                active_indices.push(idx);
                active_caps.push(cap);
                let temp = lane
                    .params
                    .sampling_config
                    .and_then(|c| c.temperature)
                    .unwrap_or(1.0);
                if !is_greedy_temperature(temp) {
                    all_greedy = false;
                }
            }
        }

        // Single active lane or mixed/sampled batch: delegate directly to per-lane `propose_block`
        // (which still benefits from fused `inkling2_markov_gate` and chains only `cap` steps).
        if active_indices.len() <= 1 || !all_greedy {
            let mut out = Vec::with_capacity(lanes.len());
            for (lane, rng) in lanes.iter().zip(rngs.iter_mut()) {
                out.push(lane.ctx.propose_block(
                    self,
                    target_embed,
                    target_embed_norm,
                    target_lm_head,
                    lane.anchor_id,
                    lane.max_drafts,
                    lane.params,
                    *rng,
                )?);
            }
            return Ok(out);
        }

        let n = active_indices.len();
        let n_i64 = n as i64;
        let max_draft = active_caps.iter().copied().max().unwrap_or(0);
        let max_draft_i64 = max_draft as i64;

        let mut f_rnn_refs: Vec<&MxArray> = Vec::with_capacity(n);
        let mut anchor_ids: Vec<i32> = Vec::with_capacity(n);
        for &idx in &active_indices {
            let lane = &lanes[idx];
            let f_rnn = lane.ctx.last_f_rnn.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "Inkling2DsparkModel::propose_blocks_batched called before seeding context",
                )
            })?;
            f_rnn_refs.push(f_rnn);
            anchor_ids.push(lane.anchor_id as i32);
        }

        let d = self.config.hidden_size as i64;
        let n_q = self.config.num_attention_heads as i64;
        let n_kv = self.config.num_key_value_heads as i64;
        let hd = self.config.head_dim as i64;
        let q_dim = n_q * hd;
        let kv_dim = n_kv * hd;
        let ffn = self.config.intermediate_size as i64;
        let block_i64 = self.config.block_size as i64;
        let eps = self.config.rms_norm_eps;
        let total_rows = n_i64 * block_i64;

        // 1. Build stacked `[N * 8, H]` BF16 input block in 1 embed + 1 norm + 1 concat!
        let anchor_arr = MxArray::from_int32(&anchor_ids, &[n_i64])?;
        let anchors_raw = target_embed.forward(&anchor_arr)?; // [N, H]
        let anchors_emb =
            apply_sublayer_rms_norm(&anchors_raw, target_embed_norm, eps, DType::BFloat16)?
                .reshape(&[n_i64, 1, d])?;
        let masks = self
            .mask_embd
            .reshape(&[1, 1, d])?
            .broadcast_to(&[n_i64, block_i64 - 1, d])?;
        let mut x = MxArray::concatenate(&anchors_emb, &masks, 1)?.reshape(&[total_rows, d])?;

        // 2. Run 3 bidirectional draft layers: stacked Q8_0 projections + per-lane exact KV SDPA!
        let attn_scale = 1.0_f64 / (self.config.head_dim as f64).sqrt();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let h = apply_sublayer_rms_norm(&x, &layer.input_layernorm, eps, DType::BFloat16)?;
            let (q_raw, k_raw, v_raw) = if let Some(ref fused) = layer.fused_qkv {
                let qkv = fused.forward(&h)?;
                let mut parts = qkv.split_sections(&[q_dim, q_dim + kv_dim], -1)?;
                let v = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qkv: v"))?;
                let k = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qkv: k"))?;
                let q = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qkv: q"))?;
                (q, k, v)
            } else {
                (
                    layer.q_proj.forward(&h)?,
                    layer.k_proj.forward(&h)?,
                    layer.v_proj.forward(&h)?,
                )
            };

            let q_normed = apply_sublayer_rms_norm(
                &q_raw.reshape(&[total_rows * n_q, hd])?,
                &layer.q_norm,
                eps,
                DType::BFloat16,
            )?
            .reshape(&[n_i64, block_i64, n_q, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?; // [N, n_q, 8, hd]
            let k_normed = apply_sublayer_rms_norm(
                &k_raw.reshape(&[total_rows * n_kv, hd])?,
                &layer.k_norm,
                eps,
                DType::BFloat16,
            )?
            .reshape(&[n_i64, block_i64, n_kv, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?; // [N, n_kv, 8, hd]
            let v_4d = v_raw
                .reshape(&[n_i64, block_i64, n_kv, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?; // [N, n_kv, 8, hd]

            let mut lane_attn_outs = Vec::with_capacity(n);
            for (slot, &orig_idx) in active_indices.iter().enumerate() {
                let s_i64 = slot as i64;
                let ctx = lanes[orig_idx].ctx;
                let offset = ctx.committed_len;
                let q_j = q_normed.slice(&[s_i64, 0, 0, 0], &[s_i64 + 1, n_q, block_i64, hd])?;
                let k_j = k_normed.slice(&[s_i64, 0, 0, 0], &[s_i64 + 1, n_kv, block_i64, hd])?;
                let v_j = v_4d.slice(&[s_i64, 0, 0, 0], &[s_i64 + 1, n_kv, block_i64, hd])?;
                let q_rot = self.rope.forward(&q_j, Some(offset))?;
                let k_rot = self.rope.forward(&k_j, Some(offset))?;
                let (k_all, v_all) = match (&ctx.layers[layer_idx].k, &ctx.layers[layer_idx].v) {
                    (Some(ck), Some(cv)) => (
                        MxArray::concatenate(ck, &k_rot, 2)?,
                        MxArray::concatenate(cv, &v_j, 2)?,
                    ),
                    _ => (k_rot, v_j),
                };
                let ao = scaled_dot_product_attention(&q_rot, &k_all, &v_all, attn_scale, None)?
                    .transpose(Some(&[0, 2, 1, 3]))?
                    .reshape(&[block_i64, q_dim])?;
                lane_attn_outs.push(ao);
            }
            let attn_refs: Vec<&MxArray> = lane_attn_outs.iter().collect();
            let attn_out = MxArray::concatenate_many(attn_refs, Some(0))?; // [N * 8, q_dim]
            let o = layer.o_proj.forward(&attn_out)?;
            x = x
                .astype(DType::Float32)?
                .add(&o.astype(DType::Float32)?)?
                .astype(DType::BFloat16)?;

            // Stacked MLP: 1 norm + 1 gate_up + 1 SwiGLU + 1 down_proj across all N lanes!
            let h_mlp =
                apply_sublayer_rms_norm(&x, &layer.post_attention_layernorm, eps, DType::BFloat16)?;
            let (gate, up) = if let Some(ref fused) = layer.fused_gate_up {
                let gu = fused.forward(&h_mlp)?;
                let mut parts = gu.split_sections(&[ffn], -1)?;
                let u = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split gu: up"))?;
                let g = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split gu: gate"))?;
                (g, u)
            } else {
                (
                    layer.gate_proj.forward(&h_mlp)?,
                    layer.up_proj.forward(&h_mlp)?,
                )
            };
            let act = Activations::silu(&gate.astype(DType::Float32)?)?
                .mul(&up.astype(DType::Float32)?)?
                .astype(DType::BFloat16)?;
            let mlp_out = layer.down_proj.forward(&act)?;
            x = x
                .astype(DType::Float32)?
                .add(&mlp_out.astype(DType::Float32)?)?
                .astype(DType::BFloat16)?;
        }

        // 3. Final output_norm -> slice positions `1..1+max_draft` -> `[N, max_draft, H]`
        let out_normed = apply_sublayer_rms_norm(&x, &self.output_norm, eps, DType::BFloat16)?
            .reshape(&[n_i64, block_i64, d])?;
        let hidden_3d = out_normed.slice(&[0, 1, 0], &[n_i64, 1 + max_draft_i64, d])?;
        let hidden_2d = hidden_3d.reshape(&[n_i64 * max_draft_i64, d])?;

        // 4. Shared target `lm_head` ONCE for all `N * max_draft` draft rows!
        let head = &self.markov_head;
        let base_raw = target_lm_head.forward_logits(&hidden_2d)?;
        let base_2d = if base_raw.dtype()? == DType::BFloat16 {
            base_raw
        } else {
            base_raw.astype(DType::BFloat16)?
        };
        let padded_vocab = base_2d.shape()?[1];
        let unpadded_vocab = (self.config.unpadded_vocab_size as i64).min(padded_vocab);
        let base_logits = base_2d.reshape(&[n_i64, max_draft_i64, padded_vocab])?;

        // 5. Precompute Markov RNN contexts ONCE for all `N * max_draft` rows!
        let f_rnn_stacked = MxArray::concatenate_many(f_rnn_refs, Some(0))? // [N, H]
            .reshape(&[n_i64, 1, d])?
            .broadcast_to(&[n_i64, max_draft_i64, d])?;
        let ctx_in = MxArray::concatenate(&hidden_3d, &f_rnn_stacked, -1)?
            .reshape(&[n_i64 * max_draft_i64, 2 * d])?;
        let rank_i64 = head.markov_rank as i64;
        let ctx_all = ctx_in
            .matmul(&head.rnn_proj_ctx_w_t)?
            .astype(DType::Float32)?
            .add(&head.rnn_bias_f32)?
            .reshape(&[n_i64, max_draft_i64, 3 * rank_i64])?;

        // 6. Batched `max_draft`-step recurrent Markov loop (`[N, 256] @ fold_weight_t` reads
        //    the 103 MB folded lm_head ONCE per step instead of N times!).
        let use_metal = {
            #[cfg(target_os = "macos")]
            {
                unsafe { mlx_sys::mlx_metal_is_available() }
            }
            #[cfg(not(target_os = "macos"))]
            {
                false
            }
        };
        let softcap = self.config.final_logit_softcapping;
        let mut state = MxArray::zeros(&[n_i64, rank_i64], Some(DType::BFloat16))?;
        let mut prev_tok = anchor_arr; // [N] Int32
        let mut step_tokens = Vec::with_capacity(max_draft);

        for step in 0..max_draft {
            let s_i64 = step as i64;
            let ctx_i = ctx_all
                .slice(&[0, s_i64, 0], &[n_i64, s_i64 + 1, 3 * rank_i64])?
                .squeeze(Some(&[1]))?; // [N, 3*R] F32
            let (next_state, bias) = markov_rnn_step(head, &state, &prev_tok, &ctx_i)?;
            state = next_state;

            let base_i = base_logits
                .slice(&[0, s_i64, 0], &[n_i64, s_i64 + 1, padded_vocab])?
                .squeeze(Some(&[1]))?; // [N, V] BF16
            let chosen = if use_metal {
                super::kernels::inkling2_lm_head_softcap_argmax(
                    &base_i,
                    Some(&bias),
                    unpadded_vocab as usize,
                    softcap as f32,
                    true,
                )?
            } else {
                let logits = markov_add_softcap_slice(&base_i, &bias, unpadded_vocab, softcap)?;
                logits.argmax(-1, Some(false))?.astype(DType::Int32)?
            };
            prev_tok = chosen.clone();
            step_tokens.push(chosen.reshape(&[n_i64, 1])?);
        }

        let step_refs: Vec<&MxArray> = step_tokens.iter().collect();
        let all_drafts = MxArray::concatenate_many(step_refs, Some(1))?; // [N, max_draft] Int32

        let mut results: Vec<DsparkProposal> = (0..lanes.len())
            .map(|_| DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            })
            .collect();
        for (slot, (&orig_idx, &cap)) in active_indices.iter().zip(&active_caps).enumerate() {
            let s_i64 = slot as i64;
            let lane_ids = all_drafts
                .slice(&[s_i64, 0], &[s_i64 + 1, cap as i64])?
                .squeeze(Some(&[0]))?; // [cap] Int32
            results[orig_idx].device_draft_ids = Some(lane_ids);
        }
        Ok(results)
    }
}

/// One recurrent Markov head step matching `Inkling2MarkovHead::step` and `_markov_gate_kernel`:
/// - `inputs = [state, markov_w1[prev_tok]]` (`[N, 512]` BF16)
/// - `raw = context + f32(inputs @ rnn_proj_rec_w_t)` (`[N, 768]` FP32)
/// - Round `raw` to BF16, split `(gate, cand, out)` -> BF16-rounded gate/cand/update/tanh(out)
/// - `bias = bf16(fold_bias + bf16(latent @ fold_weight_t))` (`[N, padded_vocab]` BF16)
pub(crate) fn markov_rnn_step(
    head: &Inkling2MarkovHead,
    state: &MxArray,
    prev_tok: &MxArray,
    context: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let rank = head.markov_rank as i64;
    let prev_emb = head.markov_w1.take(prev_tok, 0)?; // [N, R] BF16
    let rec_in = MxArray::concatenate(state, &prev_emb, -1)?; // [N, 2*R] BF16
    let rec_mm = rec_in.matmul(&head.rnn_proj_rec_w_t)?; // [N, 3*R] BF16

    let (next_state, latent) = {
        #[cfg(target_os = "macos")]
        {
            if unsafe { mlx_sys::mlx_metal_is_available() } {
                super::kernels::inkling2_markov_gate(&rec_mm, Some(context), state)?
            } else {
                let raw = rec_mm.astype(DType::Float32)?.add(context)?;
                markov_gate_with_one(&raw, state, rank, &head.one_scalar)?
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let raw = rec_mm.astype(DType::Float32)?.add(context)?;
            markov_gate_with_one(&raw, state, rank, &head.one_scalar)?
        }
    };
    let bias = latent
        .matmul(&head.fold_weight_t)?
        .astype(DType::BFloat16)?
        .add(&head.fold_bias)?
        .astype(DType::BFloat16)?;
    Ok((next_state, bias))
}

/// BF16-rounded Markov RNN gate matching `vllm/models/inkling2/nvidia/ops/dspark.py::markov_gate`.
#[cfg(test)]
pub(crate) fn markov_gate(raw: &MxArray, state: &MxArray, rank: i64) -> Result<(MxArray, MxArray)> {
    let one = MxArray::scalar_float(1.0)?;
    markov_gate_with_one(raw, state, rank, &one)
}

pub(crate) fn markov_gate_with_one(
    raw: &MxArray,
    state: &MxArray,
    rank: i64,
    one: &MxArray,
) -> Result<(MxArray, MxArray)> {
    let raw_bf16 = raw.astype(DType::BFloat16)?;
    let n = raw_bf16.shape()?[0];
    let gate_raw = raw_bf16
        .slice(&[0, 0], &[n, rank])?
        .astype(DType::Float32)?;
    let cand_raw = raw_bf16
        .slice(&[0, rank], &[n, 2 * rank])?
        .astype(DType::Float32)?;
    let out_raw = raw_bf16
        .slice(&[0, 2 * rank], &[n, 3 * rank])?
        .astype(DType::Float32)?;

    let g = Activations::sigmoid(&gate_raw)?.astype(DType::BFloat16)?;
    let c = cand_raw.tanh()?.astype(DType::BFloat16)?;
    let one_minus_g = one
        .sub(&g.astype(DType::Float32)?)?
        .astype(DType::BFloat16)?;

    let keep = g
        .astype(DType::Float32)?
        .mul(&state.astype(DType::Float32)?)?
        .astype(DType::BFloat16)?;
    let update = one_minus_g
        .astype(DType::Float32)?
        .mul(&c.astype(DType::Float32)?)?
        .astype(DType::BFloat16)?;
    let next_state = keep
        .astype(DType::Float32)?
        .add(&update.astype(DType::Float32)?)?
        .astype(DType::BFloat16)?;
    let latent = out_raw.tanh()?.astype(DType::BFloat16)?;
    Ok((next_state, latent))
}

/// `softcap(bf16(base_logits + bias), cap)[:, :unpadded_vocab]` -> FP32 `[N, unpadded_vocab]`.
pub(crate) fn markov_add_softcap_slice(
    base_logits: &MxArray,
    bias: &MxArray,
    unpadded_vocab: i64,
    softcap: f64,
) -> Result<MxArray> {
    #[cfg(target_os = "macos")]
    if unsafe { mlx_sys::mlx_metal_is_available() } {
        return super::kernels::inkling2_lm_head_softcap(
            base_logits,
            Some(bias),
            unpadded_vocab as usize,
            softcap as f32,
            true,
        );
    }
    let summed = base_logits
        .astype(DType::Float32)?
        .add(&bias.astype(DType::Float32)?)?
        .astype(DType::BFloat16)?;
    let n = summed.shape()?[0];
    let sliced = if unpadded_vocab < summed.shape()?[1] {
        summed.slice(&[0, 0], &[n, unpadded_vocab])?
    } else {
        summed
    };
    let mut logits = sliced.astype(DType::Float32)?;
    if softcap > 0.0 {
        logits = logits
            .mul_scalar(1.0 / softcap)?
            .tanh()?
            .mul_scalar(softcap)?;
    }
    Ok(logits)
}

/// Fold `dspark_hidden_out` (`[H, R]` weight + `[H]` bias) into `target_lm_head` (`[V, H]`) once at
/// load time in **FP64**, returning `(fold_weight_t [R, V] BF16, fold_bias [1, V] BF16)`.
///
/// Dequantizes `target_lm_head` in chunks of [`FOLD_CHUNK_ROWS`] (4096 rows) so peak RAM stays
/// under ~80 MiB even for `V = 201024`.
pub(crate) fn fold_markov_head_f64(
    target_lm_head: &LinearProj,
    hidden_out_w: &MxArray,
    hidden_out_b: &MxArray,
    vocab_size: usize,
    hidden_size: usize,
    rank: usize,
) -> Result<(MxArray, MxArray)> {
    let w_shape = hidden_out_w.shape()?;
    if w_shape.as_ref() != [hidden_size as i64, rank as i64] {
        return Err(Error::from_reason(format!(
            "dspark_hidden_out.weight shape {:?}, expected [{hidden_size}, {rank}]",
            w_shape.as_ref()
        )));
    }
    let b_shape = hidden_out_b.shape()?;
    if b_shape.as_ref() != [hidden_size as i64] {
        return Err(Error::from_reason(format!(
            "dspark_hidden_out.bias shape {:?}, expected [{hidden_size}]",
            b_shape.as_ref()
        )));
    }

    let proj_f32 = hidden_out_w.astype(DType::Float32)?.to_float32()?;
    let bias_f32 = hidden_out_b.astype(DType::Float32)?.to_float32()?;
    let proj_f64: Vec<f64> = proj_f32.iter().map(|&v| f64::from(v)).collect();
    let bias_f64: Vec<f64> = bias_f32.iter().map(|&v| f64::from(v)).collect();

    let mut fold_weight_chunks: Vec<MxArray> =
        Vec::with_capacity(vocab_size.div_ceil(FOLD_CHUNK_ROWS));
    let mut fold_bias_f32: Vec<f32> = Vec::with_capacity(vocab_size);
    let mut chunk_head_f64: Vec<f64> = Vec::new();
    let mut chunk_out_f64: Vec<f64> = Vec::new();
    let mut chunk_out_f32: Vec<f32> = Vec::new();

    let mut start = 0usize;
    while start < vocab_size {
        let end = (start + FOLD_CHUNK_ROWS).min(vocab_size);
        let rows = end - start;

        let head_slice = match target_lm_head {
            LinearProj::Quantized(_) => target_lm_head
                .slice_rows(start as i64, end as i64)?
                .dequantized_weight()?
                .astype(DType::Float32)?,
            LinearProj::Standard(lin) => lin
                .get_weight()
                .slice(&[start as i64, 0], &[end as i64, hidden_size as i64])?
                .astype(DType::Float32)?,
        };
        let head_f32 = head_slice.to_float32()?;
        if head_f32.len() != rows * hidden_size {
            return Err(Error::from_reason(format!(
                "fold_markov_head_f64: lm_head chunk [{start}..{end}) has {} elems, expected {}",
                head_f32.len(),
                rows * hidden_size
            )));
        }

        chunk_head_f64.clear();
        chunk_head_f64.extend(head_f32.iter().map(|&v| f64::from(v)));
        drop(head_f32);
        drop(head_slice);

        // 1. `chunk_out_f64 = head_f64 [rows, H] @ proj_f64 [H, R]` -> `[rows, R]`
        chunk_out_f64.resize(rows * rank, 0.0);
        f64_gemm_nn(
            rows,
            rank,
            hidden_size,
            &chunk_head_f64,
            &proj_f64,
            &mut chunk_out_f64,
        );

        chunk_out_f32.clear();
        chunk_out_f32.extend(chunk_out_f64.iter().map(|&v| v as f32));
        let chunk_bf16 = MxArray::from_float32(&chunk_out_f32, &[rows as i64, rank as i64])?
            .astype(DType::BFloat16)?;
        chunk_bf16.eval();
        fold_weight_chunks.push(chunk_bf16);

        // 2. `fold_bias[start..end] = head_f64 [rows, H] @ bias_f64 [H]` -> `[rows]`
        let mut bias_chunk_f64 = vec![0.0_f64; rows];
        f64_gemv_n(
            rows,
            hidden_size,
            &chunk_head_f64,
            &bias_f64,
            &mut bias_chunk_f64,
        );
        fold_bias_f32.extend(bias_chunk_f64.iter().map(|&v| v as f32));

        start = end;
    }

    let refs: Vec<&MxArray> = fold_weight_chunks.iter().collect();
    let fold_weight = MxArray::concatenate_many(refs, Some(0))?; // [V, R] BF16
    let fold_weight_t = fold_weight.transpose(None)?; // [R, V] BF16
    let fold_bias =
        MxArray::from_float32(&fold_bias_f32, &[1, vocab_size as i64])?.astype(DType::BFloat16)?;
    MxArray::eval_arrays(&[&fold_weight_t, &fold_bias])?;
    Ok((fold_weight_t, fold_bias))
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
    fn cblas_dgemv(
        order: i32,
        transa: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
}

#[cfg(target_os = "macos")]
fn f64_gemm_nn(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    // SAFETY: `a` has `m*k`, `b` has `k*n`, `c` has `m*n` elements in row-major order.
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn f64_gemm_nn(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    c.fill(0.0);
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for p in 0..k {
            let av = a_row[p];
            let b_row = &b[p * n..(p + 1) * n];
            for j in 0..n {
                c_row[j] += av * b_row[j];
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn f64_gemv_n(m: usize, k: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    // SAFETY: `a` has `m*k`, `x` has `k`, `y` has `m` elements.
    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn f64_gemv_n(m: usize, k: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let mut acc = 0.0_f64;
        for p in 0..k {
            acc += a_row[p] * x[p];
        }
        y[i] = acc;
    }
}

fn require_draft_tensor<'a>(
    params: &'a HashMap<String, MxArray>,
    key: &str,
) -> Result<&'a MxArray> {
    params
        .get(key)
        .ok_or_else(|| Error::from_reason(format!("Inkling2 DSpark GGUF missing tensor '{key}'")))
}

/// Build a Q8_0 `QuantizedLinear` (reconstructing symmetric `-128 * scales` biases emitted by
/// `load_gguf_tensors`) or dense BF16 `LinearProj` for one draft projection.
fn load_draft_linear(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
) -> Result<LinearProj> {
    let w_key = format!("{prefix}.weight");
    let s_key = format!("{prefix}.scales");
    let b_key = format!("{prefix}.biases");
    let weight = require_draft_tensor(params, &w_key)?;
    if let Some(scales) = params.get(&s_key) {
        let biases = match params.get(&b_key) {
            Some(b) => b.clone(),
            // Q8_0 symmetric zero-point = 128 (`w = scale * (q - 128)`)
            None => scales.mul_scalar(-128.0)?.astype(scales.dtype()?)?,
        };
        let ql = QuantizedLinear::new(
            weight.clone(),
            scales.clone(),
            Some(biases),
            None,
            32,
            8,
            DEFAULT_QUANT_MODE.to_string(),
        );
        let mut proj = LinearProj::Quantized(ql);
        proj.promote_affine_sidecars_to_f32(DType::BFloat16)?;
        Ok(proj)
    } else {
        let mut lin = Linear::new(in_features as u32, out_features as u32, Some(false))?;
        lin.set_weight(&weight.astype(DType::BFloat16)?)?;
        Ok(LinearProj::Standard(lin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_markov_head_f64_matches_exact_double_precision() {
        let vocab = 12usize;
        let hidden = 8usize;
        let rank = 4usize;

        let lm_vals: Vec<f32> = (0..(vocab * hidden))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.125)
            .collect();
        let proj_vals: Vec<f32> = (0..(hidden * rank))
            .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
            .collect();
        let bias_vals: Vec<f32> = (0..hidden).map(|i| (i as f32 - 3.5) * 0.1).collect();

        let mut lm_lin = Linear::new(hidden as u32, vocab as u32, Some(false)).unwrap();
        lm_lin
            .set_weight(
                &MxArray::from_float32(&lm_vals, &[vocab as i64, hidden as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap(),
            )
            .unwrap();
        let lm_proj = LinearProj::Standard(lm_lin);
        let hw = MxArray::from_float32(&proj_vals, &[hidden as i64, rank as i64]).unwrap();
        let hb = MxArray::from_float32(&bias_vals, &[hidden as i64]).unwrap();

        let (fw_t, fb) = fold_markov_head_f64(&lm_proj, &hw, &hb, vocab, hidden, rank).unwrap();
        assert_eq!(fw_t.shape().unwrap().as_ref(), &[rank as i64, vocab as i64]);
        assert_eq!(fb.shape().unwrap().as_ref(), &[1, vocab as i64]);

        let got_w = fw_t
            .transpose(None)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let got_b = fb.astype(DType::Float32).unwrap().to_float32().unwrap();
        let lm_bf16_vals = lm_proj
            .dequantized_weight()
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();

        for v in 0..vocab {
            let mut exp_b = 0.0_f64;
            for h in 0..hidden {
                exp_b += f64::from(lm_bf16_vals[v * hidden + h]) * f64::from(bias_vals[h]);
            }
            assert!(
                (got_b[v] - exp_b as f32).abs() < 2e-2,
                "bias[{v}]: got {} expected {exp_b}",
                got_b[v]
            );
            for r in 0..rank {
                let mut exp_w = 0.0_f64;
                for h in 0..hidden {
                    exp_w += f64::from(lm_bf16_vals[v * hidden + h])
                        * f64::from(proj_vals[h * rank + r]);
                }
                let got = got_w[v * rank + r];
                assert!(
                    (got - exp_w as f32).abs() < 2e-2,
                    "weight[{v},{r}]: got {got} expected {exp_w}"
                );
            }
        }
    }

    #[test]
    fn markov_gate_matches_formula() {
        let rank = 4i64;
        let raw_vals: Vec<f32> = vec![
            0.5, -0.5, 1.0, -1.0, // gate
            0.25, -0.25, 0.75, -0.75, // cand
            0.1, -0.2, 0.3, -0.4, // out
        ];
        let state_vals: Vec<f32> = vec![0.2, -0.3, 0.4, -0.5];
        let raw = MxArray::from_float32(&raw_vals, &[1, 3 * rank]).unwrap();
        let state = MxArray::from_float32(&state_vals, &[1, rank])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (next_state, latent) = markov_gate(&raw, &state, rank).unwrap();
        let ns = next_state
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let lat = latent.astype(DType::Float32).unwrap().to_float32().unwrap();
        for c in 0..4 {
            let g = 1.0_f32 / (1.0 + (-raw_vals[c]).exp());
            let cand = raw_vals[4 + c].tanh();
            let exp_ns = g * state_vals[c] + (1.0 - g) * cand;
            let exp_lat = raw_vals[8 + c].tanh();
            assert!(
                (ns[c] - exp_ns).abs() < 2e-2,
                "state[{c}]: {} vs {exp_ns}",
                ns[c]
            );
            assert!(
                (lat[c] - exp_lat).abs() < 2e-2,
                "latent[{c}]: {} vs {exp_lat}",
                lat[c]
            );
        }
    }

    #[test]
    fn mix_rmsnorm_weights_and_normalizes() {
        let t0 = MxArray::from_float32(&[1.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
        let t1 = MxArray::from_float32(&[0.0, 3.0, 0.0, 0.0], &[1, 4]).unwrap();
        let w = MxArray::from_float32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let out = mix_rmsnorm(&[&t0, &t1], &[0.5, 0.5], &w, 1e-6)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        // mixed = [0.5, 1.5, 0, 0], mean_sq = (0.25 + 2.25)/4 = 0.625, inv_rms = 1/sqrt(0.625)
        let inv_rms = 1.0_f32 / (0.625_f32 + 1e-6).sqrt();
        assert!((out[0] - 0.5 * inv_rms).abs() < 2e-2);
        assert!((out[1] - 1.5 * inv_rms).abs() < 2e-2);
        assert!(out[2].abs() < 1e-4);
        assert!(out[3].abs() < 1e-4);
    }

    #[test]
    fn real_dspark_gguf_auto_discovery_and_metadata() {
        let dir = Path::new("/Users/brooklyn/.mlx-node/models/inkling-2-mini-gguf");
        if !dir.exists() {
            return;
        }
        let found = discover_inkling2_dspark_gguf(dir).expect("should discover dspark GGUF");
        assert!(
            found
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("dspark")
        );
        let gguf = parse_gguf(&found).expect("parse dspark GGUF");
        let target_cfg = Inkling2Config::from_json_str(
            &serde_json::json!({
                "model_type": "inkling2",
                "hidden_size": 2304,
                "num_hidden_layers": 48,
                "num_physical_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "sliding_window": 1024,
                "n_layers_per_attn_res_block": 6,
                "loop_config": {
                    "core_start": 6,
                    "core_end": 18,
                    "num_passes": 3,
                    "gate_rank": 64,
                    "is_varloop": false
                }
            })
            .to_string(),
            true,
        )
        .unwrap();
        let dcfg =
            Inkling2DsparkDraftConfig::from_gguf_metadata(&gguf.metadata, &target_cfg).unwrap();
        assert_eq!(dcfg.num_hidden_layers, 3);
        assert_eq!(dcfg.block_size, 8);
        assert_eq!(dcfg.num_draft_tokens(), 7);
        assert_eq!(dcfg.markov_rank, 256);
        assert_eq!(dcfg.target_layer_ids, vec![5, 23, 41, 48]);
        assert_eq!(dcfg.sliding_window, 1024);
        assert_eq!(dcfg.sliding_window_pattern, vec![true, false, true]);
    }
}
