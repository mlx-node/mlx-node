use super::{config::Config, math, weights::Store};
use crate::array::{DType, MxArray, scaled_dot_product_attention};
use crate::models::qwen4_exp::runtime_flags;
use crate::nn::Activations;
use napi::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
#[path = "attention_window.rs"]
mod attention_window;
#[path = "batch.rs"]
mod batch;
#[path = "device_routes.rs"]
mod device_routes;
#[path = "draft_window.rs"]
mod draft_window;
#[path = "scheduled_batch.rs"]
mod scheduled_batch;

struct AttentionProjections {
    qg: MxArray,
    k: MxArray,
    v: MxArray,
    iq: MxArray,
    ik: MxArray,
}

struct GdnProjections {
    qkv: MxArray,
    z: MxArray,
    a: MxArray,
    b: MxArray,
}
impl AttentionProjections {
    fn row(&self, row: usize) -> Result<Self> {
        let slice = |a: &MxArray| a.slice_axis(1, row as i64, row as i64 + 1);
        Ok(Self {
            qg: slice(&self.qg)?,
            k: slice(&self.k)?,
            v: slice(&self.v)?,
            iq: slice(&self.iq)?,
            ik: slice(&self.ik)?,
        })
    }
}

#[derive(Clone, Default)]
pub struct LayerCache {
    conv: Option<MxArray>,
    recurrent: Option<MxArray>,
    keys: Option<MxArray>,
    values: Option<MxArray>,
    index_tail: Vec<MxArray>,
    index_blocks: Vec<MxArray>,
    ple_conv: Option<MxArray>,
}
#[derive(Clone, Default)]
pub struct DecoderState {
    caches: Vec<LayerCache>,
    history: Vec<u32>,
    positions: Vec<[i64; 3]>,
    rope_delta: i64,
    media_digests: Vec<crate::engine::cache::ImageCacheDigest>,
    last_chunk_hidden: Vec<MxArray>,
    cancelled: Option<Arc<AtomicBool>>,
}
pub struct Decoder {
    pub batch_prefill: bool,
    pub window_carry: bool,
    pub async_prefill: bool,
    pub batch_rotary: bool,
    pub prefill_chunk_size: usize,
    // Like TrackFastModel's retained configuration, share immutable metadata
    // across forward helpers instead of copying its strings/lists per layer.
    pub config: Arc<Config>,
    pub weights: Store,
    pub caches: Vec<LayerCache>,
    pub history: Vec<u32>,
    pub positions: Vec<[i64; 3]>,
    pub rope_delta: i64,
    pub media_digests: Vec<crate::engine::cache::ImageCacheDigest>,
    pub cancelled: Option<Arc<AtomicBool>>,
    pub paged: Option<crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter>,
    paged_chunk_start: usize,
    scope: String,
    position_override: Option<usize>,
    rotary_tables: std::cell::RefCell<math::RotaryWindowCache>,
    verification: Option<Vec<DecoderState>>,
    device_routes: Option<Vec<(usize, MxArray)>>,
    device_route_cooldown: usize,
    pub last_chunk_hidden: Vec<MxArray>,
}
impl Decoder {
    pub fn prefill_slice_size(&self) -> usize {
        let requested = std::env::var("MLX_QWEN4_PREFILL_SLICE").ok();
        super::memory::prefill_slice(self.prefill_chunk_size, requested.as_deref())
    }

    pub fn new(config: Config, weights: Store) -> Result<Self> {
        // Validate the checkpoint topology before the first tensor payload read.
        let prefill_chunk_size = super::memory::prefill_window(&config, &weights.plan)?;
        let d = Self {
            prefill_chunk_size,
            window_carry: true,
            async_prefill: true,
            batch_rotary: true,
            batch_prefill: !runtime_flags::is_zero(c"MLX_QWEN4_BATCH_PREFILL"),
            caches: (0..config.num_hidden_layers)
                .map(|_| LayerCache::default())
                .collect(),
            config: Arc::new(config),
            weights,
            history: Vec::new(),
            positions: Vec::new(),
            rope_delta: 0,
            media_digests: Vec::new(),
            cancelled: None,
            paged: None,
            paged_chunk_start: 0,
            scope: String::new(),
            position_override: None,
            rotary_tables: Default::default(),
            verification: None,
            device_routes: None,
            device_route_cooldown: 0,
            last_chunk_hidden: Vec::new(),
        };
        d.preflight()?;
        Ok(d)
    }
    pub fn snapshot(&self) -> DecoderState {
        DecoderState {
            caches: self.caches.clone(),
            history: self.history.clone(),
            positions: self.positions.clone(),
            rope_delta: self.rope_delta,
            media_digests: self.media_digests.clone(),
            last_chunk_hidden: self.last_chunk_hidden.clone(),
            cancelled: self.cancelled.clone(),
        }
    }
    pub fn take_state(&mut self) -> DecoderState {
        let state = DecoderState {
            caches: std::mem::take(&mut self.caches),
            history: std::mem::take(&mut self.history),
            positions: std::mem::take(&mut self.positions),
            rope_delta: self.rope_delta,
            media_digests: self.media_digests.clone(),
            last_chunk_hidden: std::mem::take(&mut self.last_chunk_hidden),
            cancelled: self.cancelled.take(),
        };
        self.reset();
        state
    }
    pub fn restore_state(&mut self, state: DecoderState) {
        self.rotary_tables.get_mut().clear();
        self.caches = state.caches;
        self.history = state.history;
        self.positions = state.positions;
        self.rope_delta = state.rope_delta;
        self.media_digests = state.media_digests;
        self.last_chunk_hidden = state.last_chunk_hidden;
        self.cancelled = state.cancelled;
    }
    pub fn reset(&mut self) {
        self.rotary_tables.get_mut().clear();
        self.verification = None;
        self.device_routes = None;
        self.device_route_cooldown = 0;
        self.history.clear();
        self.positions.clear();
        self.rope_delta = 0;
        self.media_digests.clear();
        self.last_chunk_hidden.clear();
        self.caches = (0..self.config.num_hidden_layers)
            .map(|_| LayerCache::default())
            .collect();
    }
    fn rope(&self, x: &MxArray, pos: usize) -> Result<MxArray> {
        let cached = runtime_flags::is_one(c"MLX_QWEN4_ROTARY_TABLES");
        if !cached && (!self.scope.is_empty() || self.positions.is_empty()) {
            return math::rope(x, pos, self.config.rope_dims(), self.config.rope_theta());
        }
        let v = &self.config.rope_parameters;
        let (p, sections, interleaved) = if !self.scope.is_empty() || self.positions.is_empty() {
            ([pos as i64; 3], [0; 3], true)
        } else {
            (
                self.positions
                    .get(pos)
                    .copied()
                    .unwrap_or([pos as i64 + self.rope_delta; 3]),
                std::array::from_fn(|i| v["mrope_section"][i].as_u64().unwrap_or(0) as usize),
                v["mrope_interleaved"].as_bool().unwrap_or(true),
            )
        };
        if cached {
            return self.rotary_tables.borrow_mut().apply_singleton(
                x,
                p,
                self.config.rope_dims(),
                self.config.rope_theta(),
                sections,
                interleaved,
            );
        }
        math::mrope(
            x,
            p,
            self.config.rope_dims(),
            self.config.rope_theta(),
            sections,
            interleaved,
        )
    }
    fn gguf(&self) -> bool {
        self.weights.gguf && self.scope.is_empty()
    }
    fn key(&self, hf: &str, gg: &str) -> String {
        if self.gguf() {
            gg.into()
        } else {
            format!("{}{hf}", self.scope)
        }
    }
    fn dense(&mut self, hf: &str, gg: &str) -> Result<MxArray> {
        self.weights.dense(&self.key(hf, gg))
    }
    fn linear(&mut self, x: &MxArray, hf: &str, gg: &str) -> Result<MxArray> {
        self.weights.linear(&self.key(hf, gg), x)
    }
    fn linear_pair(
        &mut self,
        x: &MxArray,
        a: (&str, &str),
        b: (&str, &str),
    ) -> Result<(MxArray, MxArray)> {
        self.weights
            .linear_pair(x, &self.key(a.0, a.1), &self.key(b.0, b.1))
    }
    fn norm(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        group: usize,
        centered: bool,
    ) -> Result<MxArray> {
        let w = self.dense(hf, gg)?;
        math::norm(
            x,
            &w,
            group,
            self.config.rms_norm_eps,
            centered && !self.gguf(),
        )
    }
    pub fn check_cancelled(&self) -> Result<()> {
        if self
            .cancelled
            .as_ref()
            .is_some_and(|c| c.load(Ordering::Relaxed))
        {
            return Err(Error::from_reason("prefill cancelled"));
        }
        Ok(())
    }
    fn shape(&self, hf: &str, gg: &str, want: &[usize]) -> Result<()> {
        let key = self.key(hf, gg);
        let shape = &self.weights.descriptor(&key)?.shape;
        if shape != want {
            return Err(Error::from_reason(format!(
                "qwen4_exp tensor {key}: expected {want:?}, got {shape:?}"
            )));
        }
        Ok(())
    }
    fn preflight(&self) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        let hc = h * c.hc_count;
        self.shape(
            "embed_tokens.weight",
            "token_embd.weight",
            &[c.vocab_size, h],
        )?;
        self.shape("lm_head.weight", "output.weight", &[c.vocab_size, h])?;
        for (hf, gg, shape) in [
            (
                "hyper_connection_mixer.hc_norm.weight",
                "output_hc_norm.weight",
                vec![hc],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_down.weight",
                "output_hc_down.weight",
                vec![c.hc_lowrank, hc],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_up.weight",
                "output_hc_up.weight",
                vec![hc, c.hc_lowrank],
            ),
        ] {
            self.shape(hf, gg, &shape)?;
        }
        for i in 0..c.num_hidden_layers {
            for (hf, gg) in [
                ("attn_hyper_connection", "hc_attn"),
                ("mlp_hyper_connection", "hc_ffn"),
            ] {
                self.shape(
                    &format!("layers.{i}.{hf}.input_mix_weight_down.weight"),
                    &format!("blk.{i}.{gg}_down.weight"),
                    &[c.hc_lowrank, hc],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.input_mix_weight_up.weight"),
                    &format!("blk.{i}.{gg}_up.weight"),
                    &[hc, c.hc_lowrank],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.hc_norm.weight"),
                    &format!("blk.{i}.{gg}_norm.weight"),
                    &[hc],
                )?;
                self.shape(
                    &format!("layers.{i}.{hf}.block_inject_weight.weight"),
                    &format!("blk.{i}.{gg}_inject.weight"),
                    &[c.hc_count, hc],
                )?;
            }
            let kd = c.linear_num_key_heads * c.linear_key_head_dim;
            let vd = c.linear_num_value_heads * c.linear_value_head_dim;
            let p = format!("layers.{i}");
            let g = format!("blk.{i}");
            if c.linear(i) {
                for (hf, gg, shape) in [
                    (
                        "linear_attn.in_proj_qkv.weight",
                        "attn_qkv.weight",
                        vec![2 * kd + vd, h],
                    ),
                    (
                        "linear_attn.in_proj_z.weight",
                        "attn_gate.weight",
                        vec![vd, h],
                    ),
                    (
                        "linear_attn.in_proj_a.weight",
                        "ssm_alpha.weight",
                        vec![c.linear_num_value_heads, h],
                    ),
                    (
                        "linear_attn.in_proj_b.weight",
                        "ssm_beta.weight",
                        vec![c.linear_num_value_heads, h],
                    ),
                    ("linear_attn.out_proj.weight", "ssm_out.weight", vec![h, vd]),
                    ("linear_attn.A_log", "ssm_a", vec![c.linear_num_value_heads]),
                    (
                        "linear_attn.dt_bias",
                        "ssm_dt.bias",
                        vec![c.linear_num_value_heads],
                    ),
                    (
                        "linear_attn.norm.weight",
                        "ssm_norm.weight",
                        vec![c.linear_value_head_dim],
                    ),
                ] {
                    self.shape(&format!("{p}.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                self.shape(
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &format!("{g}.ssm_conv1d.weight"),
                    &if self.gguf() {
                        vec![2 * kd + vd, c.linear_conv_kernel_dim]
                    } else {
                        vec![2 * kd + vd, 1, c.linear_conv_kernel_dim]
                    },
                )?;
            } else {
                for (hf, gg, shape) in [
                    (
                        "self_attn.q_proj.weight",
                        "attn_q.weight",
                        vec![2 * c.num_attention_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.k_proj.weight",
                        "attn_k.weight",
                        vec![c.num_key_value_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.v_proj.weight",
                        "attn_v.weight",
                        vec![c.num_key_value_heads * c.head_dim, h],
                    ),
                    (
                        "self_attn.o_proj.weight",
                        "attn_output.weight",
                        vec![h, c.num_attention_heads * c.head_dim],
                    ),
                    (
                        "self_attn.q_norm.weight",
                        "attn_q_norm.weight",
                        vec![c.head_dim],
                    ),
                    (
                        "self_attn.k_norm.weight",
                        "attn_k_norm.weight",
                        vec![c.head_dim],
                    ),
                    (
                        "self_attn.indexer.q_layernorm.weight",
                        "indexer.q_norm.weight",
                        vec![c.indexer_head_dim],
                    ),
                    (
                        "self_attn.indexer.k_layernorm.weight",
                        "indexer.k_norm.weight",
                        vec![c.indexer_head_dim],
                    ),
                ] {
                    self.shape(&format!("{p}.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                if self.gguf() {
                    self.shape(
                        "",
                        &format!("{g}.indexer.q_proj.weight"),
                        &[c.indexer_n_heads * c.indexer_head_dim, h],
                    )?;
                    self.shape(
                        "",
                        &format!("{g}.indexer.k_proj.weight"),
                        &[c.indexer_head_dim, h],
                    )?;
                } else {
                    self.shape(
                        &format!("{p}.self_attn.indexer.index_qk_proj.weight"),
                        "",
                        &[(c.indexer_n_heads + 1) * c.indexer_head_dim, h],
                    )?;
                }
            }
            for (hf, gg, shape) in [
                (
                    "shared_expert.gate_proj.weight",
                    "ffn_gate_shexp.weight",
                    vec![c.shared_expert_intermediate_size, h],
                ),
                (
                    "shared_expert.up_proj.weight",
                    "ffn_up_shexp.weight",
                    vec![c.shared_expert_intermediate_size, h],
                ),
                (
                    "shared_expert.down_proj.weight",
                    "ffn_down_shexp.weight",
                    vec![h, c.shared_expert_intermediate_size],
                ),
            ] {
                self.shape(&format!("{p}.mlp.{hf}"), &format!("{g}.{gg}"), &shape)?;
            }
            self.shape(
                &format!("{p}.mlp.shared_expert_gate.weight"),
                &format!("{g}.ffn_gate_inp_shexp.weight"),
                &if self.gguf() { vec![h] } else { vec![1, h] },
            )?;
            if c.ple_layer_ids.contains(&(i + 1)) {
                for (hf, gg, shape) in [
                    (
                        "key_proj.weight",
                        "ple_key.weight",
                        vec![hc, c.ple_embed_dim],
                    ),
                    (
                        "value_proj.weight",
                        "ple_value.weight",
                        vec![h, c.ple_embed_dim],
                    ),
                    ("norm_key.weight", "ple_norm_key.weight", vec![hc]),
                    ("norm_query.weight", "ple_norm_query.weight", vec![hc]),
                    ("norm_conv.weight", "ple_norm_conv.weight", vec![hc]),
                ] {
                    self.shape(&format!("{p}.ple.{hf}"), &format!("{g}.{gg}"), &shape)?;
                }
                self.shape(
                    &format!("{p}.ple.conv1d.weight"),
                    &format!("{g}.ple_conv1d.weight"),
                    &if self.gguf() {
                        vec![hc, c.ple_conv_kernel_size]
                    } else {
                        vec![hc, 1, c.ple_conv_kernel_size]
                    },
                )?;
                let heads = (c.ngram_size - 1) * c.heads_per_ngram;
                let mut rows = 0usize;
                if self.gguf() {
                    let table = self.weights.descriptor("per_layer_token_embd.weight")?;
                    if table.shape.len() != 2 || table.width()? != c.ple_embed_dim / heads {
                        return Err(Error::from_reason("Invalid GGUF PLE shape"));
                    }
                    rows = table.rows()?;
                } else {
                    for shard in 0..c.split_ngram_parts {
                        let table = self.weights.descriptor(&format!(
                            "{p}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                        ))?;
                        if table.shape.len() != 2 || table.width()? != c.ple_embed_dim / heads {
                            return Err(Error::from_reason("Invalid PLE shard shape"));
                        }
                        rows = rows
                            .checked_add(table.rows()?)
                            .ok_or_else(|| Error::from_reason("PLE row count overflow"))?;
                    }
                }
                let constants = |hf: &str, gg: &str| {
                    if self.gguf() {
                        super::gguf::integers(&self.weights.metadata, &format!("qwen4exp.ple.{gg}"))
                    } else {
                        self.weights
                            .integers(&format!("{p}.ple.ple_embedding.{hf}"))
                    }
                };
                let sizes = constants("ngram_heads_vocab_sizes", "head_vocab_sizes")?;
                let offsets = constants("ngram_heads_offsets", "head_offsets")?;
                let multipliers = constants("layer_multipliers", "layer_multipliers")?;
                math::hash_ids(
                    0,
                    &[],
                    c.eos_token_id,
                    c.ngram_size,
                    c.heads_per_ngram,
                    &multipliers,
                    &sizes,
                    &offsets,
                )?;
                let mut end = 0u64;
                for (&size, &offset) in sizes.iter().zip(&offsets) {
                    if offset != end {
                        return Err(Error::from_reason(
                            "PLE hash vocabularies overlap or contain gaps",
                        ));
                    }
                    end = end
                        .checked_add(size)
                        .ok_or_else(|| Error::from_reason("PLE vocabulary size overflow"))?;
                }
                if end > rows as u64 {
                    return Err(Error::from_reason(
                        "PLE table does not cover its hashed vocabulary",
                    ));
                }
            }
            self.shape(
                &format!("layers.{i}.mlp.gate.weight"),
                &format!("blk.{i}.ffn_gate_inp.weight"),
                &[c.num_experts, h],
            )?;
            if self.gguf() {
                self.shape(
                    "",
                    &format!("blk.{i}.ffn_gate_exps.weight"),
                    &[c.num_experts, c.moe_intermediate_size, h],
                )?;
                self.shape(
                    "",
                    &format!("blk.{i}.ffn_up_exps.weight"),
                    &[c.num_experts, c.moe_intermediate_size, h],
                )?;
            } else {
                self.shape(
                    &format!("layers.{i}.mlp.experts.gate_up_proj"),
                    "",
                    &[c.num_experts, 2 * c.moe_intermediate_size, h],
                )?;
            }
            self.shape(
                &format!("layers.{i}.mlp.experts.down_proj"),
                &format!("blk.{i}.ffn_down_exps.weight"),
                &[c.num_experts, h, c.moe_intermediate_size],
            )?;
        }
        Ok(())
    }
    fn hyper(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.hyper_with_norm(x, hf, gg, inject, None)
    }

    fn hyper_with_norm(
        &mut self,
        x: &MxArray,
        hf: &str,
        gg: &str,
        inject: bool,
        normed: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let c = self.config.clone();
        let n = if let Some(normed) = normed {
            normed.clone()
        } else {
            self.norm(
                x,
                &format!("{hf}.hc_norm.weight"),
                &format!("{gg}_norm.weight"),
                c.hidden_size,
                true,
            )?
        };
        let paired = if (runtime_flags::is_one(c"MLX_QWEN4_MIXER_DOWN_INJECT")
            || runtime_flags::is_one(c"MLX_QWEN4_MIXER_SPLIT_K"))
            && inject
            && self.gguf()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && x.shape()?[1] == 1
        {
            let down_key = self.key(
                &format!("{hf}.input_mix_weight_down.weight"),
                &format!("{gg}_down.weight"),
            );
            let inject_key = self.key(
                &format!("{hf}.block_inject_weight.weight"),
                &format!("{gg}_inject.weight"),
            );
            match (
                self.weights.resident_bank(&down_key),
                self.weights.resident_bank(&inject_key),
            ) {
                (Some(down), Some(injection)) => down.mixer_down_inject(&n, &injection)?,
                _ => None,
            }
        } else {
            None
        };
        let activated = if paired.is_none()
            && self.gguf()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && (x.shape()?[1] >= 1024 || x.shape()?[1] == 1)
        {
            let down_key = self.key(
                &format!("{hf}.input_mix_weight_down.weight"),
                &format!("{gg}_down.weight"),
            );
            self.weights
                .resident_bank(&down_key)
                .map(|w| w.mixer_act(&n))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let is_activated = paired.is_some() || activated.is_some();
        let (down, inject_projection) = if let Some((down, gate)) = paired {
            (down, Some(gate))
        } else if let Some(activated) = activated {
            let gate = if inject {
                Some(self.linear(
                    &n,
                    &format!("{hf}.block_inject_weight.weight"),
                    &format!("{gg}_inject.weight"),
                )?)
            } else {
                None
            };
            (activated, gate)
        } else if inject {
            let (down, gate) = self.linear_pair(
                &n,
                (
                    &format!("{hf}.input_mix_weight_down.weight"),
                    &format!("{gg}_down.weight"),
                ),
                (
                    &format!("{hf}.block_inject_weight.weight"),
                    &format!("{gg}_inject.weight"),
                ),
            )?;
            (down, Some(gate))
        } else {
            (
                self.linear(
                    &n,
                    &format!("{hf}.input_mix_weight_down.weight"),
                    &format!("{gg}_down.weight"),
                )?,
                None,
            )
        };
        let down = if is_activated {
            down
        } else {
            Activations::silu(&down.div_scalar(c.hc_count as f64)?)?
        };
        let key = self.key(
            &format!("{hf}.input_mix_weight_up.weight"),
            &format!("{gg}_up.weight"),
        );
        if self.gguf()
            && c.hc_count == 4
            && c.hidden_size == 2560
            && let Some(injection) = inject_projection.as_ref()
            && let Some(weight) = self.weights.resident_bank(&key)
            && let Some((mixed, gate)) = weight.hyper_up_inject(&down, &n, injection)?
        {
            return Ok((mixed, Some(gate)));
        }
        let fused = if self.gguf() && c.hc_count == 4 && c.hidden_size == 2560 {
            self.weights
                .resident_bank(&key)
                .map(|w| w.hyper_up(&down, &n))
                .transpose()?
                .flatten()
        } else {
            None
        };
        let mixed = if let Some(mixed) = fused {
            mixed
        } else {
            let up = self.weights.linear(&key, &down)?;
            let combined = if self.gguf()
                && !runtime_flags::is_zero(c"MLX_QWEN4_PREFILL_HC_MIX")
                && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
                && crate::engine::persistence::compiled_forward_backend_available()
            {
                unsafe { mlx_sys::mlx_qwen4_prefill_hc_mix(up.as_raw_ptr(), n.as_raw_ptr()) }
            } else {
                std::ptr::null_mut()
            };
            if combined.is_null() {
                math::sigmoid_mul(&up, &n)?
                    .reshape(&[1, x.shape()?[1], c.hc_count as i64, c.hidden_size as i64])?
                    .mean(Some(&[-2]), Some(false))?
            } else {
                MxArray::from_handle(combined, "Qwen4 prefill mixer combine")?
            }
        };
        let gate = if inject {
            let w = inject_projection.ok_or_else(|| {
                Error::from_reason("Qwen4 hyper-connection is missing its injection projection")
            })?;
            Some(Activations::sigmoid(&w.div_scalar(c.hc_count as f64)?)?.mul_scalar(2.0)?)
        } else {
            None
        };
        Ok((mixed, gate))
    }

    // Normalized results stay on the forward's call stack, with no
    // cache/frontier state to restore after a device-route replay.
    fn inject_for_mlp(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        layer: usize,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        self.inject_for_hyper(
            x,
            branch,
            gate,
            &format!("layers.{layer}.mlp_hyper_connection.hc_norm.weight"),
            &format!("blk.{layer}.hc_ffn_norm.weight"),
            normalize,
        )
    }

    fn inject_for_next_attention(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        layer: usize,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let next = layer + 1;
        // PLE adds to the stream before its attention norm. Never carry a
        // normalization across that addition or across a forward boundary.
        if normalize
            && next < self.config.num_hidden_layers
            && !self.config.ple_layer_ids.contains(&(next + 1))
            && runtime_flags::is_one(c"MLX_QWEN4_INJECT_NEXT_NORM")
        {
            self.inject_for_hyper(
                x,
                branch,
                gate,
                &format!("layers.{next}.attn_hyper_connection.hc_norm.weight"),
                &format!("blk.{next}.hc_attn_norm.weight"),
                true,
            )
        } else {
            Ok((Self::inject(x, branch, gate)?, None))
        }
    }

    fn inject_for_hyper(
        &mut self,
        x: &MxArray,
        branch: &MxArray,
        gate: &MxArray,
        hf: &str,
        gg: &str,
        normalize: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        if normalize
            && self.gguf()
            && self.config.hidden_size == 2560
            && self.config.hc_count == 4
            && !runtime_flags::is_zero(c"MLX_QWEN4_INJECT_NORM")
            && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let w = self.dense(hf, gg)?;
            let (mut stream, mut normed) = (std::ptr::null_mut(), std::ptr::null_mut());
            if unsafe {
                mlx_sys::mlx_qwen4_inject_norm(
                    x.as_raw_ptr(),
                    branch.as_raw_ptr(),
                    gate.as_raw_ptr(),
                    w.as_raw_ptr(),
                    self.config.rms_norm_eps,
                    &mut stream,
                    &mut normed,
                )
            } {
                return Ok((
                    MxArray::from_handle(stream, "Qwen4 fused injection stream")?,
                    Some(MxArray::from_handle(
                        normed,
                        "Qwen4 fused injection normalization",
                    )?),
                ));
            }
        }
        Ok((Self::inject(x, branch, gate)?, None))
    }
    fn submit(&self, arrays: &[&MxArray], context: &str) -> Result<()> {
        if !runtime_flags::is_zero(c"MLX_QWEN4_ASYNC_SUBMISSION")
            && self.verification.is_none()
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            MxArray::async_eval_arrays(arrays);
            Ok(())
        } else {
            MxArray::eval_arrays_with_context(arrays, context)
        }
    }
    fn submit_state(&self, arrays: &[&MxArray], context: &str) -> Result<()> {
        // The reference submits whole layer groups. Inside a checked device
        // transaction, its final join also completes every auxiliary state
        // before bank readers are released or a failed window is replayed.
        if self.deferred_state_submission() {
            Ok(())
        } else {
            self.submit(arrays, context)
        }
    }
    fn inject(x: &MxArray, y: &MxArray, g: &MxArray) -> Result<MxArray> {
        if !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            return MxArray::from_handle(
                unsafe {
                    mlx_sys::mlx_qwen4_inject(x.as_raw_ptr(), y.as_raw_ptr(), g.as_raw_ptr())
                },
                "Qwen4 compiled residual injection",
            );
        }
        x.add(
            &y.expand_dims(-2)?
                .mul(&g.expand_dims(-1)?)?
                .reshape(&x.shape()?)?,
        )
    }
    fn gdn_projections(&mut self, x: &MxArray, p: &str, g: &str) -> Result<GdnProjections> {
        let (qkv, z) = self.linear_pair(
            x,
            (
                &format!("{p}.in_proj_qkv.weight"),
                &format!("{g}.attn_qkv.weight"),
            ),
            (
                &format!("{p}.in_proj_z.weight"),
                &format!("{g}.attn_gate.weight"),
            ),
        )?;
        let z = z.reshape(&[
            1,
            x.shape_at(1)?,
            self.config.linear_num_value_heads as i64,
            self.config.linear_value_head_dim as i64,
        ])?;
        // TrackFastModel.bindGDN groups compatible projections. The GGUF
        // gate matrices are dense F32, so pair them separately from Q8 QKV/Z.
        let (a, b) = if self.gguf() && runtime_flags::is_one(c"MLX_QWEN4_GDN_GATE_PAIR") {
            self.linear_pair(
                x,
                (
                    &format!("{p}.in_proj_a.weight"),
                    &format!("{g}.ssm_alpha.weight"),
                ),
                (
                    &format!("{p}.in_proj_b.weight"),
                    &format!("{g}.ssm_beta.weight"),
                ),
            )?
        } else {
            (
                self.linear(
                    x,
                    &format!("{p}.in_proj_a.weight"),
                    &format!("{g}.ssm_alpha.weight"),
                )?,
                self.linear(
                    x,
                    &format!("{p}.in_proj_b.weight"),
                    &format!("{g}.ssm_beta.weight"),
                )?,
            )
        };
        // TrackFastGDNDecode reads projected BF16 gates directly. Keep local
        // F32 gate arithmetic inside the consumer, without two cast buffers.
        let (a, b) = if self.gguf() && runtime_flags::is_one(c"MLX_QWEN4_GDN_GATE_INPUTS") {
            (a, b)
        } else {
            (a.astype(DType::Float32)?, b.astype(DType::Float32)?)
        };
        Ok(GdnProjections { qkv, z, a, b })
    }

    fn gdn(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.linear_attn");
        let g = format!("blk.{i}");
        let nh = c.linear_num_value_heads as i64;
        let kh = c.linear_num_key_heads as i64;
        let kd = c.linear_key_head_dim as i64;
        let vd = c.linear_value_head_dim as i64;
        let keydim = kh * kd;
        let GdnProjections { qkv, z, a, b } = self.gdn_projections(x, &p, &g)?;
        let conv = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ssm_conv1d.weight"),
        )?;
        if self.gguf()
            && self.verification.is_none()
            && (kh, nh, kd, vd) == (16, 48, 128, 128)
            && (2..=8).contains(&c.linear_conv_kernel_dim)
            && c.rms_norm_eps.is_finite()
            && c.rms_norm_eps > 0.0
            && c.output_gate_type == "sigmoid"
            && qkv.dtype()? == DType::BFloat16
            && !runtime_flags::is_zero(c"MLX_QWEN4_COMPLETE_GDN")
            && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_GDN")
            && !runtime_flags::is_zero(c"MLX_QWEN4_FUSED_POINTWISE")
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let scale = self.dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?;
            let dt = self.dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?;
            let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
            let history = match &cache.conv {
                Some(s) => s.clone(),
                None => MxArray::zeros(
                    &[(c.linear_conv_kernel_dim - 1) as i64, 2 * keydim + nh * vd],
                    Some(qkv.dtype()?),
                )?,
            };
            let state = match &cache.recurrent {
                Some(s) => s.clone(),
                None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
            };
            let (out, state, history) = math::complete_gdn(
                &qkv,
                &z,
                &a,
                &b,
                &conv,
                &history,
                &scale,
                &dt,
                &state,
                &w,
                c.rms_norm_eps,
            )?;
            self.submit_state(&[&out, &state, &history], "qwen4::decoder::complete_gdn")?;
            cache.recurrent = Some(state);
            cache.conv = Some(history);
            return self.linear(
                &out,
                &format!("{p}.out_proj.weight"),
                &format!("{g}.ssm_out.weight"),
            );
        }
        let qkv = math::conv(&qkv, &conv, &mut cache.conv, c.linear_conv_kernel_dim, 1)?;
        let q = math::l2(&qkv.slice_axis(2, 0, keydim)?.reshape(&[1, kh, kd])?)?
            .mul_scalar((kd as f64).powf(-0.5))?;
        let k = math::l2(
            &qkv.slice_axis(2, keydim, 2 * keydim)?
                .reshape(&[1, kh, kd])?,
        )?;
        let v = qkv
            .slice_axis(2, 2 * keydim, 2 * keydim + nh * vd)?
            .reshape(&[1, nh, vd])?
            .astype(DType::Float32)?;
        let (q, k) = if self.gguf() {
            (q, k)
        } else {
            (
                q.repeat((nh / kh) as i32, 1)?,
                k.repeat((nh / kh) as i32, 1)?,
            )
        };
        let q = q.astype(DType::Float32)?.reshape(&[1, -1, 1, kd])?;
        let k = k.astype(DType::Float32)?.reshape(&[1, -1, 1, kd])?;
        let a_log = self
            .dense(&format!("{p}.A_log"), &format!("{g}.ssm_a"))?
            .astype(DType::Float32)?;
        let a_scale = if self.gguf() {
            a_log
        } else {
            a_log.exp()?.negative()?
        };
        let dt = self
            .dense(&format!("{p}.dt_bias"), &format!("{g}.ssm_dt.bias"))?
            .astype(DType::Float32)?;
        let (decay, beta) = math::gdn_gates(&a, &b, &a_scale, &dt)?;
        let decay = decay.reshape(&[1, nh, 1, 1])?;
        let state = match &cache.recurrent {
            Some(s) => s.clone(),
            None => MxArray::zeros(&[1, nh, vd, kd], Some(DType::Float32))?,
        };
        let (out, state) =
            math::recurrent_step(&q, &k, &v, &decay, &beta.reshape(&[1, nh, 1])?, &state)?;
        let out = out.astype(x.dtype()?)?;
        self.submit_state(&[&state], "qwen4::decoder::state")?;
        cache.recurrent = Some(state);
        let w = self.dense(&format!("{p}.norm.weight"), &format!("{g}.ssm_norm.weight"))?;
        let out = math::norm(&out, &w, c.linear_value_head_dim, c.rms_norm_eps, false)?
            .astype(DType::Float32)?;
        let z = z.astype(DType::Float32)?;
        let out = if c.output_gate_type == "sigmoid" {
            math::sigmoid_mul(&z, &out)?
        } else {
            math::swiglu(&z, &out)?
        };
        let out = out.astype(x.dtype()?)?.reshape(&[1, 1, nh * vd])?;
        self.linear(
            &out,
            &format!("{p}.out_proj.weight"),
            &format!("{g}.ssm_out.weight"),
        )
    }
    fn attention(&mut self, x: &MxArray, i: usize, cache: &mut LayerCache) -> Result<MxArray> {
        let projections = self.attention_projections(x, i)?;
        self.attention_projected(x, i, cache, &projections, true)
    }

    fn attention_projections(&mut self, x: &MxArray, i: usize) -> Result<AttentionProjections> {
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let qg = self.linear(
            x,
            &format!("{p}.q_proj.weight"),
            &format!("{g}.attn_q.weight"),
        )?;
        let (k, v) = self.linear_pair(
            x,
            (&format!("{p}.k_proj.weight"), &format!("{g}.attn_k.weight")),
            (&format!("{p}.v_proj.weight"), &format!("{g}.attn_v.weight")),
        )?;
        let ih = self.config.indexer_n_heads as i64;
        let id = self.config.indexer_head_dim as i64;
        let (iq, ik) = if self.gguf() {
            (
                self.linear(x, "", &format!("{g}.indexer.q_proj.weight"))?,
                self.linear(x, "", &format!("{g}.indexer.k_proj.weight"))?,
            )
        } else {
            let qk = self.linear(x, &format!("{p}.indexer.index_qk_proj.weight"), "")?;
            (
                qk.slice_axis(2, 0, ih * id)?,
                qk.slice_axis(2, ih * id, (ih + 1) * id)?,
            )
        };
        Ok(AttentionProjections { qg, k, v, iq, ik })
    }

    /// Projection matrices can span a prompt window while rotary positions,
    /// sparse selection and recurrent/indexer state still advance causally.
    fn attention_projected(
        &mut self,
        x: &MxArray,
        i: usize,
        cache: &mut LayerCache,
        projections: &AttentionProjections,
        project_output: bool,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.self_attn");
        let g = format!("blk.{i}");
        let pos = self.position_override.unwrap_or(self.history.len());
        let hd = c.head_dim as i64;
        let nh = c.num_attention_heads as i64;
        let kh = c.num_key_value_heads as i64;
        let qg = projections.qg.reshape(&[1, 1, nh, 2 * hd])?;
        let q = qg.slice_axis(3, 0, hd)?;
        let q = self.attention_norm_rotary(
            &q.transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.q_norm.weight"),
            &format!("{g}.attn_q_norm.weight"),
            pos,
            1,
            false,
        )?;
        let k = self.attention_norm_rotary(
            &projections
                .k
                .reshape(&[1, 1, kh, hd])?
                .transpose(Some(&[0, 2, 1, 3]))?,
            &format!("{p}.k_norm.weight"),
            &format!("{g}.attn_k_norm.weight"),
            pos,
            1,
            false,
        )?;
        let v = projections
            .v
            .reshape(&[1, 1, kh, hd])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys = match &cache.keys {
            Some(old) => MxArray::concatenate(old, &k, 2)?,
            None => k,
        };
        let values = match &cache.values {
            Some(old) => MxArray::concatenate(old, &v, 2)?,
            None => v,
        };
        self.submit_state(&[&keys, &values], "qwen4::decoder::kv")?;
        cache.keys = Some(keys.clone());
        cache.values = Some(values.clone());
        let ih = c.indexer_n_heads as i64;
        let id = c.indexer_head_dim as i64;
        let (iq, ik) = (&projections.iq, &projections.ik);
        cache.index_tail.push(ik.reshape(&[1, id])?);
        if cache.index_tail.len() == c.indexer_compress_ratio {
            let raw = MxArray::concatenate_many(cache.index_tail.iter().collect(), Some(0))?
                .astype(DType::Float32)?
                .mean(Some(&[0]), Some(true))?
                .astype(x.dtype()?)?;
            let raw = self.norm(
                &raw,
                &format!("{p}.indexer.k_layernorm.weight"),
                &format!("{g}.indexer.k_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let block = self.rope(&raw, pos + 1 - c.indexer_compress_ratio)?;
            self.submit_state(&[&block], "qwen4::decoder::block")?;
            cache.index_blocks.push(block);
            cache.index_tail.clear();
        }
        let topk = c.indexer_budget / c.indexer_compress_ratio;
        let selected = if cache.index_blocks.len() > topk {
            let iq = iq.reshape(&[ih, id])?;
            let iq = self.norm(
                &iq,
                &format!("{p}.indexer.q_layernorm.weight"),
                &format!("{g}.indexer.q_norm.weight"),
                c.indexer_head_dim,
                true,
            )?;
            let iq = self.rope(&iq, pos)?.astype(DType::Float32)?;
            let blocks = MxArray::concatenate_many(cache.index_blocks.iter().collect(), Some(0))?
                .astype(DType::Float32)?;
            let scores = iq
                .matmul(&blocks.transpose(None)?)?
                .clip(Some(0.0), None)?
                .sum(Some(&[0]), Some(false))?;
            // ReLU produces exact ties. Use the reference MLX partition, not
            // a CPU sort with a different tie rule, to preserve selected blocks.
            let scores = scores
                .div_scalar((c.indexer_head_dim as f64).sqrt())?
                .reshape(&[1, 1, -1])?;
            let n = cache.index_blocks.len() as i64;
            let indices = scores
                .argpartition(-(topk as i32), Some(-1))?
                .slice_axis(2, n - topk as i64, n)?
                .reshape(&[-1])?
                .astype(DType::Int32)?
                .sort(Some(0))?;
            let offsets = MxArray::arange(
                0.,
                c.indexer_compress_ratio as f64,
                None,
                Some(DType::Int32),
            )?;
            let ids = indices
                .expand_dims(-1)?
                .mul_scalar(c.indexer_compress_ratio as f64)?
                .astype(DType::Int32)?
                .add(&offsets)?
                .reshape(&[-1])?;
            let tail = MxArray::arange(
                (cache.index_blocks.len() * c.indexer_compress_ratio) as f64,
                (pos + 1) as f64,
                None,
                Some(DType::Int32),
            )?;
            MxArray::concatenate(&ids, &tail, 0)?
        } else {
            MxArray::arange(0., (pos + 1) as f64, None, Some(DType::Int32))?
        };
        let (keys, values) = if let Some(adapter) = &self.paged {
            let layer = (0..i).filter(|&l| !self.config.linear(l)).count() as u32;
            super::paged::gather_window(
                adapter,
                layer,
                &selected,
                &keys,
                &values,
                self.paged_chunk_start,
            )?
        } else {
            (keys.take(&selected, 2)?, values.take(&selected, 2)?)
        };
        let out =
            scaled_dot_product_attention(&q, &keys, &values, (c.head_dim as f64).powf(-0.5), None)?;
        let out = math::attention_output(&out, &qg)?;
        if project_output {
            self.linear(
                &out,
                &format!("{p}.o_proj.weight"),
                &format!("{g}.attn_output.weight"),
            )
        } else {
            Ok(out)
        }
    }
    fn moe(&mut self, x: &MxArray, i: usize) -> Result<MxArray> {
        let c = self.config.clone();
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let logits = self.linear(
            x,
            &format!("{p}.gate.weight"),
            &format!("{g}.ffn_gate_inp.weight"),
        )?;
        let combined = if self.gguf()
            && runtime_flags::is_one(c"MLX_QWEN4_ROUTE_SHARED_GATE")
            && !runtime_flags::is_zero(c"MLX_QWEN4_DENSE_BF16_CACHE")
            && let Some(weight) = self
                .weights
                .resident_bank(&format!("{g}.ffn_gate_inp_shexp.weight"))
            && let Some(dense) = &weight.dense_bf16
        {
            math::routes_shared_gate(&logits, x, dense, c.num_experts_per_tok)?
        } else {
            None
        };
        let mut shared_gate = None;
        let (selected, scores) = if let Some((selected, scores, gate)) = combined {
            shared_gate = Some(gate);
            (selected, scores)
        } else if let Some(routes) = math::singleton_routes(&logits, c.num_experts_per_tok)? {
            routes
        } else {
            let probs = Activations::softmax_precise(&logits, Some(-1))?;
            let ne = c.num_experts as i64;
            let top = c.num_experts_per_tok as i64;
            let selected =
                probs
                    .argpartition(-(top as i32), Some(-1))?
                    .slice_axis(2, ne - top, ne)?;
            let scores = probs.take_along_axis(&selected, -1)?;
            let scores = scores.div(&scores.sum(Some(&[-1]), Some(true))?)?;
            (selected, scores)
        };
        if let Some(out) =
            self.tentative_shared_experts(x, &selected, &scores, shared_gate.as_ref(), i)?
        {
            return Ok(out);
        }
        let device = self.tentative_experts(x, &selected, &scores, i)?;
        let resident = if device.is_some() {
            device
        } else {
            self.resident_experts(x, &selected, &scores, i)?
        };
        let ids = if resident.is_none() {
            selected.to_uint32()?.to_vec()
        } else {
            Vec::new()
        };
        let sum = if let Some(sum) = resident {
            sum
        } else if let Some(sum) = self.selected_experts(x, &ids, &scores, i)? {
            sum
        } else {
            let mut weighted = Vec::new();
            for (j, &e) in ids.iter().enumerate() {
                let e = e as usize;
                self.check_cancelled()?;
                let m = c.moe_intermediate_size;
                let (gate, up) = if self.gguf() {
                    (
                        self.weights
                            .read(&format!("{g}.ffn_gate_exps.weight"), e * m, m)?
                            .linear(x)?,
                        self.weights
                            .read(&format!("{g}.ffn_up_exps.weight"), e * m, m)?
                            .linear(x)?,
                    )
                } else {
                    let key = self.key(&format!("{p}.experts.gate_up_proj"), "");
                    let both = self.weights.read(&key, e * 2 * m, 2 * m)?.linear(x)?;
                    (
                        both.slice_axis(2, 0, m as i64)?,
                        both.slice_axis(2, m as i64, 2 * m as i64)?,
                    )
                };
                let h = math::swiglu(&gate, &up)?;
                let key = self.key(
                    &format!("{p}.experts.down_proj"),
                    &format!("{g}.ffn_down_exps.weight"),
                );
                let out = self
                    .weights
                    .read(&key, e * c.hidden_size, c.hidden_size)?
                    .linear(&h)?;
                let value = out.mul(&scores.slice_axis(2, j as i64, j as i64 + 1)?)?;
                MxArray::eval_arrays_with_context(&[&value], "qwen4::decoder::value")?;
                weighted.push(value);
            }
            MxArray::concatenate_many(weighted.iter().collect(), Some(1))?
                .sum(Some(&[1]), Some(true))?
        };
        let (shared, gate) = self.shared_expert_parts(x, i, shared_gate)?;
        sum.astype(x.dtype()?)?
            .add(&math::sigmoid_mul(&gate, &shared)?)
    }
    fn shared_expert_parts(
        &mut self,
        x: &MxArray,
        i: usize,
        precomputed_gate: Option<MxArray>,
    ) -> Result<(MxArray, MxArray)> {
        let p = format!("layers.{i}.mlp");
        let g = format!("blk.{i}");
        let (gate, up) = self.linear_pair(
            x,
            (
                &format!("{p}.shared_expert.gate_proj.weight"),
                &format!("{g}.ffn_gate_shexp.weight"),
            ),
            (
                &format!("{p}.shared_expert.up_proj.weight"),
                &format!("{g}.ffn_up_shexp.weight"),
            ),
        )?;
        let activation = math::swiglu(&gate, &up)?;
        let shared = self.linear(
            &activation,
            &format!("{p}.shared_expert.down_proj.weight"),
            &format!("{g}.ffn_down_shexp.weight"),
        )?;
        drop(activation);
        let gate = match precomputed_gate {
            Some(gate) => gate,
            None => self.linear(
                x,
                &format!("{p}.shared_expert_gate.weight"),
                &format!("{g}.ffn_gate_inp_shexp.weight"),
            )?,
        };
        Ok((shared, gate))
    }

    fn ple(
        &mut self,
        x: &MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        self.ple_window(x, &[token], i, cache)
    }

    fn ple_window(
        &mut self,
        x: &MxArray,
        tokens: &[u32],
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        let c = self.config.clone();
        let t = tokens.len() as i64;
        let p = format!("layers.{i}.ple");
        let g = format!("blk.{i}");
        let constants = |name: &str| -> Result<Vec<u64>> {
            if self.gguf() {
                super::gguf::integers(&self.weights.metadata, &format!("qwen4exp.ple.{name}"))
            } else {
                self.weights.integers(&format!(
                    "{p}.ple_embedding.{}",
                    match name {
                        "head_offsets" => "ngram_heads_offsets",
                        "head_vocab_sizes" => "ngram_heads_vocab_sizes",
                        _ => name,
                    }
                ))
            }
        };
        let multipliers = constants("layer_multipliers")?;
        let sizes = constants("head_vocab_sizes")?;
        let offsets = constants("head_offsets")?;
        let mut previous =
            self.history[self.history.len().saturating_sub(c.ngram_size - 1)..].to_vec();
        let mut ids = Vec::new();
        for &token in tokens {
            self.check_cancelled()?;
            ids.extend(math::hash_ids(
                token,
                &previous,
                c.eos_token_id,
                c.ngram_size,
                c.heads_per_ngram,
                &multipliers,
                &sizes,
                &offsets,
            )?);
            previous.push(token);
        }

        let emb = if self.gguf() {
            self.weights.lookup_rows(
                "per_layer_token_embd.weight",
                &ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
            )?
        } else {
            let mut groups: std::collections::BTreeMap<String, (Vec<usize>, Vec<usize>)> =
                Default::default();
            for (position, id) in ids.iter().enumerate() {
                let mut local = *id as usize;
                let mut found = false;
                for shard in 0..c.split_ngram_parts {
                    let key = format!("{p}.ple_embedding.ngram_embedding.shard_{shard}.weight");
                    let count = self.weights.descriptor(&key)?.rows()?;
                    if local < count {
                        let (rows, positions) = groups.entry(key).or_default();
                        rows.push(local);
                        positions.push(position);
                        found = true;
                        break;
                    }
                    local -= count;
                }
                if !found {
                    return Err(Error::from_reason("PLE row outside checkpoint vocabulary"));
                }
            }
            let mut rows = vec![None; ids.len()];
            for (name, (indices, positions)) in groups {
                let bank = self.weights.lookup_rows(&name, &indices)?;
                for (row, position) in positions.into_iter().enumerate() {
                    rows[position] = Some(bank.slice_axis(0, row as i64, row as i64 + 1)?);
                }
            }
            let rows = rows
                .iter()
                .map(|row| {
                    row.as_ref().ok_or_else(|| {
                        Error::from_reason("Qwen4 PLE lookup did not populate every embedding row")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(rows, Some(0))?
        };
        let emb = emb
            .reshape(&[1, t, c.ple_embed_dim as i64])?
            .astype(x.dtype()?)?;

        let key = self.weights.linear_ple_window(
            &self.key(
                &format!("{p}.key_proj.weight"),
                &format!("{g}.ple_key.weight"),
            ),
            &emb,
        )?;

        let key = self
            .norm(
                &key,
                &format!("{p}.norm_key.weight"),
                &format!("{g}.ple_norm_key.weight"),
                c.hidden_size,
                true,
            )?
            .reshape(&[1, t, c.hc_count as i64, c.hidden_size as i64])?;
        let query = self
            .norm(
                x,
                &format!("{p}.norm_query.weight"),
                &format!("{g}.ple_norm_query.weight"),
                c.hidden_size,
                true,
            )?
            .reshape(&[1, t, c.hc_count as i64, c.hidden_size as i64])?;
        let value = self.weights.linear_ple_window(
            &self.key(
                &format!("{p}.value_proj.weight"),
                &format!("{g}.ple_value.weight"),
            ),
            &emb,
        )?;

        let score = key
            .mul(&query)?
            .sum(Some(&[-1]), Some(true))?
            .div_scalar((c.hidden_size as f64).sqrt())?;
        let score = score
            .sign()?
            .mul(&score.abs()?.clip(Some(1e-6), None)?.sqrt()?)?;
        let value = Activations::sigmoid(&score)?
            .mul(&value.expand_dims(-2)?)?
            .reshape(&x.shape()?)?;
        let normalized = self.norm(
            &value,
            &format!("{p}.norm_conv.weight"),
            &format!("{g}.ple_norm_conv.weight"),
            c.hidden_size,
            true,
        )?;
        let weight = self.dense(
            &format!("{p}.conv1d.weight"),
            &format!("{g}.ple_conv1d.weight"),
        )?;
        let convolved = if (tokens.len() > 8 && self.async_device_prefill())
            || (self.deferred_state_submission()
                && runtime_flags::is_one(c"MLX_QWEN4_DECODE_ASYNC_PLE"))
        {
            math::conv_window_with_completion(
                &normalized,
                &weight,
                &mut cache.ple_conv,
                c.ple_conv_kernel_size,
                c.ngram_size,
                true,
            )?
        } else {
            math::conv_window(
                &normalized,
                &weight,
                &mut cache.ple_conv,
                c.ple_conv_kernel_size,
                c.ngram_size,
            )?
        };
        value.add(&convolved)
    }
    /// Run the released one-layer MTP head against its own QSA state. Scope,
    /// position and the target page adapter are restored on every error path.
    pub fn draft_step(
        &mut self,
        hidden: &MxArray,
        token: u32,
        position: usize,
        cache: &mut LayerCache,
        project: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let _flags = runtime_flags::scope();
        let embedding = self.embed_token(token)?;
        let target_pages = self.paged.take();
        self.scope = "mtp.".into();
        self.position_override = Some(position);
        let result: Result<(MxArray, MxArray)> = (|| {
            let c = self.config.clone();
            let e = self.norm(
                &embedding,
                "pre_fc_norm_embedding.weight",
                "",
                c.hidden_size,
                true,
            )?;
            let e = self.linear(&e, "fc_embedding.weight", "")?;
            let h = self
                .norm(
                    hidden,
                    "pre_fc_norm_hidden.weight",
                    "",
                    c.hidden_size * c.hc_count,
                    true,
                )?
                .reshape(&[1, 1, c.hc_count as i64, c.hidden_size as i64])?;
            let h = self.linear(&h, "fc_hidden.weight", "")?;
            let mut x =
                h.add(&e.expand_dims(2)?)?
                    .reshape(&[1, 1, (c.hc_count * c.hidden_size) as i64])?;
            let (input, inject) = self.hyper(&x, "layers.0.attn_hyper_connection", "", true)?;
            let attn = self.attention(&input, 0, cache)?;
            let inject = inject.ok_or_else(|| {
                Error::from_reason("Qwen4 MTP attention is missing its injection gate")
            })?;
            x = Self::inject(&x, &attn, &inject)?;
            let (input, inject) = self.hyper(&x, "layers.0.mlp_hyper_connection", "", true)?;
            let mlp = self.moe(&input, 0)?;
            let inject = inject
                .ok_or_else(|| Error::from_reason("Qwen4 MTP MLP is missing its injection gate"))?;
            x = Self::inject(&x, &mlp, &inject)?;
            MxArray::eval_arrays_with_context(&[&x], "qwen4::decoder::x")?;
            let (mixed, _) = self.hyper(&x, "hyper_connection_mixer", "", false)?;
            Ok((x, mixed))
        })();
        self.scope.clear();
        self.position_override = None;
        self.paged = target_pages;
        let (hidden, mixed) = result?;
        let logits = if project {
            Some(self.linear(&mixed, "lm_head.weight", "output.weight")?)
        } else {
            None
        };
        if let Some(logits) = &logits {
            MxArray::eval_arrays_with_context(&[logits], "qwen4::decoder::logits")?;
        }
        Ok((hidden, logits))
    }
    pub fn step(&mut self, token: u32) -> Result<MxArray> {
        let _flags = runtime_flags::scope();
        self.rotary_tables.get_mut().clear();
        let result = self.forward_with_device_routes(token, true);
        self.rotary_tables.get_mut().clear();
        // A read failure or cancellation can happen after earlier layers have
        // advanced. Discard the entire prefix instead of reusing partial state.
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn layer(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        Ok(self.layer_with_norm(x, token, i, cache, None, false)?.0)
    }

    fn layer_with_norm(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
        incoming_norm: Option<&MxArray>,
        normalize_next: bool,
    ) -> Result<(MxArray, Option<MxArray>)> {
        let (x, normed) = self.attention_block_for_mlp(x, token, i, cache, true, incoming_norm)?;
        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.mlp_hyper_connection"),
            &format!("blk.{i}.hc_ffn"),
            true,
            normed.as_ref(),
        )?;
        let branch = self.moe(&mixed, i)?;
        let gate =
            gate.ok_or_else(|| Error::from_reason("Qwen4 MLP is missing its injection gate"))?;
        self.inject_for_next_attention(&x, &branch, &gate, i, normalize_next)
    }

    fn attention_block(
        &mut self,
        x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
    ) -> Result<MxArray> {
        Ok(self
            .attention_block_for_mlp(x, token, i, cache, false, None)?
            .0)
    }

    fn attention_block_for_mlp(
        &mut self,
        mut x: MxArray,
        token: u32,
        i: usize,
        cache: &mut LayerCache,
        normalize: bool,
        incoming_norm: Option<&MxArray>,
    ) -> Result<(MxArray, Option<MxArray>)> {
        if self.config.ple_layer_ids.contains(&(i + 1)) {
            x = x.add(&self.ple(&x, token, i, cache)?)?;
        }
        let (mixed, gate) = self.hyper_with_norm(
            &x,
            &format!("layers.{i}.attn_hyper_connection"),
            &format!("blk.{i}.hc_attn"),
            true,
            incoming_norm,
        )?;
        let branch = if self.config.linear(i) {
            self.gdn(&mixed, i, cache)?
        } else {
            self.attention(&mixed, i, cache)?
        };
        let gate = gate
            .ok_or_else(|| Error::from_reason("Qwen4 attention is missing its injection gate"))?;
        self.inject_for_mlp(&x, &branch, &gate, i, normalize)
    }

    pub fn embed_token(&mut self, token: u32) -> Result<MxArray> {
        let key = self.key("embed_tokens.weight", "token_embd.weight");
        self.weights
            .read(&key, token as usize, 1)?
            .dense()?
            .reshape(&[1, 1, self.config.hidden_size as i64])
    }

    /// Traverse a bounded prompt chunk layer by layer, sharing weight reads
    /// and projections across its tokens. Verification retains singleton
    /// arithmetic and captures state at every accepted frontier.
    pub fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        let _flags = runtime_flags::scope();
        if cfg!(target_os = "macos") {
            self.weights.check_forward_headroom()?;
        }
        self.rotary_tables.get_mut().clear();
        let result = self.prefill_with_device_routes(tokens, embeddings, project_logits);
        self.rotary_tables.get_mut().clear();
        if result.is_err() {
            self.reset();
        }
        result
    }
    pub fn verify_chunk(&mut self, tokens: &[u32]) -> Result<(MxArray, Vec<DecoderState>)> {
        let _flags = runtime_flags::scope();
        if tokens.is_empty() || tokens.len() > 4 {
            return Err(Error::from_reason("Qwen4 verification width must be 1-4"));
        }
        self.verification = Some(Vec::new());
        self.prefill_chunk(tokens, None, false)?;
        let states = self.verification.take().ok_or_else(|| {
            Error::from_reason("Qwen4 verification lost its accepted-frontier snapshots")
        })?;
        let mut logits = Vec::with_capacity(tokens.len());
        for h in self.last_chunk_hidden.clone() {
            let (mixed, _) = self.hyper(&h, "hyper_connection_mixer", "output_hc", false)?;
            logits.push(
                self.linear(&mixed, "lm_head.weight", "output.weight")?
                    .reshape(&[1, -1])?,
            );
        }
        Ok((
            MxArray::concatenate_many(logits.iter().collect(), Some(0))?,
            states,
        ))
    }

    fn prefill_chunk_inner(
        &mut self,
        tokens: &[u32],
        embeddings: Option<&MxArray>,
        project_logits: bool,
    ) -> Result<MxArray> {
        let base = self.history.len();
        let submit_span = self.decode_submit_span();
        let async_window = tokens.len() > 8 && self.async_device_prefill();
        let defer_final = self.deferred_forward_completion();
        if tokens.is_empty()
            || tokens.len() > super::MAX_PREFILL_CHUNK
            || base + tokens.len() > self.config.effective_context_limit()
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err(Error::from_reason(
                "Qwen4 prefill chunk exceeds token/context budget",
            ));
        }
        if let Some(e) = embeddings
            && *e.shape()? != [1, tokens.len() as i64, self.config.hidden_size as i64]
        {
            return Err(Error::from_reason("Qwen4 media embedding shape mismatch"));
        }
        self.paged_chunk_start = base;
        if let Some(adapter) = &mut self.paged {
            if adapter.request_tokens() != self.history {
                return Err(Error::from_reason(
                    "Qwen4 paged/recurrent frontier mismatch",
                ));
            }
            adapter.record_tokens(tokens).map_err(Error::from_reason)?;
        }
        let batched = self.batch_prefill && tokens.len() > 1 && self.verification.is_none();
        let carry =
            batched && self.window_carry && !runtime_flags::is_zero(c"MLX_QWEN4_WINDOW_CARRY");
        let mut hidden = Vec::with_capacity(if carry { 0 } else { tokens.len() });
        let window_embedding = if embeddings.is_none() && tokens.len() > 1 {
            Some(
                self.weights
                    .lookup_rows(
                        &self.key("embed_tokens.weight", "token_embd.weight"),
                        &tokens.iter().map(|&t| t as usize).collect::<Vec<_>>(),
                    )?
                    .reshape(&[1, tokens.len() as i64, self.config.hidden_size as i64])?,
            )
        } else {
            None
        };
        let mut window = if carry {
            let embeddings = embeddings.or(window_embedding.as_ref()).ok_or_else(|| {
                Error::from_reason("Qwen4 prefill window is missing its token embeddings")
            })?;
            Some(MxArray::tile(
                embeddings,
                &[1, 1, self.config.hc_count as i32],
            )?)
        } else {
            for (j, &token) in tokens.iter().enumerate() {
                self.check_cancelled()?;
                let row = match embeddings.or(window_embedding.as_ref()) {
                    Some(e) => e.slice_axis(1, j as i64, j as i64 + 1)?,
                    None => self.embed_token(token)?,
                };
                hidden.push(MxArray::tile(&row, &[1, 1, self.config.hc_count as i32])?);
            }
            None
        };
        if self.verification.is_some() {
            self.verification = Some(
                (0..tokens.len())
                    .map(|j| DecoderState {
                        caches: (0..self.config.num_hidden_layers)
                            .map(|_| LayerCache::default())
                            .collect(),
                        history: self
                            .history
                            .iter()
                            .chain(tokens[..=j].iter())
                            .copied()
                            .collect(),
                        positions: self.positions.clone(),
                        rope_delta: self.rope_delta,
                        media_digests: self.media_digests.clone(),
                        last_chunk_hidden: Vec::new(),
                        cancelled: self.cancelled.clone(),
                    })
                    .collect(),
            );
        }
        let mut attention_normed = None;
        for i in 0..self.config.num_hidden_layers {
            self.check_cancelled()?;
            let mut cache = std::mem::take(&mut self.caches[i]);
            let mut mlp_normed = None;
            // Verification retains the singleton arithmetic and a state at
            // every accepted frontier. Only ordinary prompt MLPs are batched.
            if carry {
                let input = window
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("Qwen4 attention lost its prefill window"))?;
                let (stream, normed) = self.attention_matrix_for_mlp(
                    input,
                    tokens,
                    i,
                    &mut cache,
                    true,
                    attention_normed.as_ref(),
                )?;
                window = Some(stream);
                mlp_normed = normed;
            } else if batched {
                hidden = self.attention_batch(&hidden, tokens, i, &mut cache)?;
            } else {
                for (j, &token) in tokens.iter().enumerate() {
                    self.check_cancelled()?;
                    let next = if tokens.len() == 1 && self.verification.is_none() {
                        let (stream, normed) = self.layer_with_norm(
                            hidden[j].clone(),
                            token,
                            i,
                            &mut cache,
                            attention_normed.as_ref(),
                            true,
                        )?;
                        attention_normed = normed;
                        stream
                    } else {
                        self.layer(hidden[j].clone(), token, i, &mut cache)?
                    };
                    if defer_final && i + 1 == self.config.num_hidden_layers {
                        // TrackFastModel returns a lazy final mixer/head. The
                        // checked route commit joins it with all state/readers.
                    } else if tokens.len() == 1
                        && (i + 1) % submit_span != 0
                        && i + 1 < self.config.num_hidden_layers
                    {
                        if self.decode_should_submit(i + 1) {
                            self.submit(&[&next], "qwen4::decoder::next")?;
                        }
                    } else {
                        MxArray::eval_arrays_with_context(&[&next], "qwen4::decoder::next")?;
                    }
                    hidden[j] = next;
                    self.history.push(token);
                    if let Some(states) = &mut self.verification {
                        let mut saved = cache.clone();
                        if self.paged.is_some() {
                            saved.keys = None;
                            saved.values = None;
                        }
                        states[j].caches[i] = saved;
                    }
                }
            }
            self.history.truncate(base);
            if carry {
                let input = window
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("Qwen4 MLP lost its prefill window"))?;
                let (stream, normed) =
                    self.mlp_matrix_for_attention(input, i, mlp_normed.as_ref(), true)?;
                window = Some(stream);
                attention_normed = normed;
            } else if batched {
                hidden = self.mlp_batch(&hidden, i)?;
            }
            if !self.config.linear(i)
                && let Some(adapter) = &mut self.paged
            {
                let layer = (0..i).filter(|&l| !self.config.linear(l)).count() as u32;
                let k = cache
                    .keys
                    .take()
                    .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no attention keys"))?
                    .transpose(Some(&[0, 2, 1, 3]))?
                    .reshape(&[
                        tokens.len() as i64,
                        self.config.num_key_value_heads as i64,
                        self.config.head_dim as i64,
                    ])?;
                let v = cache
                    .values
                    .take()
                    .ok_or_else(|| {
                        Error::from_reason("Qwen4 prefill produced no attention values")
                    })?
                    .transpose(Some(&[0, 2, 1, 3]))?
                    .reshape(&[
                        tokens.len() as i64,
                        self.config.num_key_value_heads as i64,
                        self.config.head_dim as i64,
                    ])?;
                super::paged::write_rows(adapter, layer, &k, &v, base as u32)?;
                if ((tokens.len() > 1 && !async_window)
                    || (i + 1) % submit_span == 0
                    || i + 1 == self.config.num_hidden_layers
                    || self.verification.is_some())
                    && !(defer_final && i + 1 == self.config.num_hidden_layers)
                {
                    adapter
                        .eval_pending_pool_writes()
                        .map_err(Error::from_reason)?;
                }
            }
            self.caches[i] = cache;
            if tokens.len() == 1
                && ((i + 1) % submit_span == 0 || i + 1 == self.config.num_hidden_layers)
                && !(defer_final && i + 1 == self.config.num_hidden_layers)
                && let Some(adapter) = &mut self.paged
            {
                adapter
                    .eval_pending_pool_writes()
                    .map_err(Error::from_reason)?;
            }
            super::memory::maintain_freelist(self.weights.plan.physical_bytes);
        }
        if let Some(window) = window {
            // Only public/MTP frontier capture needs token views. Keep one
            // matrix across layer boundaries instead of rebuilding it twice
            // per layer from thousands of token-sized lazy array objects.
            hidden = (0..tokens.len())
                .map(|j| window.slice_axis(1, j as i64, j as i64 + 1))
                .collect::<Result<Vec<_>>>()?;
        }
        if let Some(states) = &mut self.verification {
            for (state, h) in states.iter_mut().zip(&hidden) {
                state.last_chunk_hidden = vec![h.clone()];
            }
        }
        let last_hidden = hidden
            .last()
            .ok_or_else(|| Error::from_reason("Qwen4 prefill produced no hidden rows"))?
            .clone();
        self.last_chunk_hidden = hidden;
        let (mixed, _) = self.hyper(&last_hidden, "hyper_connection_mixer", "output_hc", false)?;
        let logits = if project_logits {
            self.linear(&mixed, "lm_head.weight", "output.weight")?
        } else {
            mixed
        };
        if !defer_final {
            MxArray::eval_arrays_with_context(&[&logits], "qwen4::decoder::logits")?;
        }

        self.history.extend_from_slice(tokens);
        Ok(logits)
    }

    fn forward_token_inner(&mut self, token: u32, project_logits: bool) -> Result<MxArray> {
        self.check_cancelled()?;
        let submit_span = self.decode_submit_span();
        let defer_final = self.deferred_forward_completion();
        if token as usize >= self.config.vocab_size
            || self.history.len() >= self.config.effective_context_limit()
        {
            return Err(Error::from_reason(
                "qwen4_exp token/context outside configured limits",
            ));
        }
        let key = self.key("embed_tokens.weight", "token_embd.weight");
        let row = self
            .weights
            .read(&key, token as usize, 1)?
            .dense()?
            .reshape(&[1, 1, self.config.hidden_size as i64])?;
        let mut x = MxArray::tile(&row, &[1, 1, self.config.hc_count as i32])?;
        let mut normed = None;
        for i in 0..self.config.num_hidden_layers {
            self.check_cancelled()?;
            let mut cache = std::mem::take(&mut self.caches[i]);
            (x, normed) = self.layer_with_norm(x, token, i, &mut cache, normed.as_ref(), true)?;
            if defer_final && i + 1 == self.config.num_hidden_layers {
                // The validated route commit completes the final head too.
            } else if (i + 1) % submit_span != 0 && i + 1 < self.config.num_hidden_layers {
                if self.decode_should_submit(i + 1) {
                    self.submit(&[&x], "qwen4::decoder::x")?;
                }
            } else {
                MxArray::eval_arrays_with_context(&[&x], "qwen4::decoder::x")?;
            }
            self.caches[i] = cache;
            // Keep reusable buffers bounded after all layer consumers complete.
            super::memory::maintain_freelist(self.weights.plan.physical_bytes);
        }
        self.last_chunk_hidden = vec![x.clone()];
        let (hidden, _) = self.hyper(&x, "hyper_connection_mixer", "output_hc", false)?;
        let logits = if project_logits {
            self.linear(&hidden, "lm_head.weight", "output.weight")?
        } else {
            hidden
        };
        if !defer_final {
            MxArray::eval_arrays_with_context(&[&logits], "qwen4::decoder::logits")?;
        }
        self.history.push(token);
        Ok(logits)
    }
}
