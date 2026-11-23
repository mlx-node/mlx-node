use std::collections::HashMap;

use napi::bindgen_prelude::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::utils::gguf::GgufMetaValue;

/// Loop configuration for Inkling-2-Mini (`Inkling2LoopForConditionalGeneration`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inkling2LoopConfig {
    /// First physical layer index in the looped core (e.g. `6`).
    pub core_start: usize,
    /// Exclusive end physical layer index of the looped core (e.g. `18`).
    pub core_end: usize,
    /// Total number of passes through the core (e.g. `3`).
    pub num_passes: usize,
    /// Low-rank bottleneck dimension for the 4 loop controllers (e.g. `64`).
    #[serde(default = "default_gate_rank")]
    pub gate_rank: usize,
    /// Whether the checkpoint uses variable loop passes.
    #[serde(default)]
    pub is_varloop: bool,
}

fn default_gate_rank() -> usize {
    64
}

/// One logical layer visit in the Inkling2 forward schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    /// Logical visit index (`0..num_layer_visits()`).
    pub visit_idx: usize,
    /// Physical layer index (`0..num_physical_layers()`).
    pub phys_layer: usize,
    /// Loop pass index (`0` for prelude, core pass 0, and coda; `1..num_passes` for core passes > 0).
    pub pass_idx: usize,
    /// Whether this visit uses local sliding-window attention (SWA) rather than global attention.
    pub is_local: bool,
}

/// Vision encoder hyperparameters (`mmproj-BF16.gguf` or HF `vision_config`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Inkling2VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_image_size")]
    pub image_size: usize,
    #[serde(default = "default_vision_patch_size", alias = "native_patch_size")]
    pub patch_size: usize,
    #[serde(
        default = "default_vision_spatial_merge_size",
        alias = "adapter_merge_size"
    )]
    pub spatial_merge_size: usize,
    #[serde(default = "default_vision_projection_dim", alias = "text_hidden_size")]
    pub projection_dim: usize,
    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_vision_image_mean")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_vision_image_std")]
    pub image_std: [f32; 3],
}

fn default_vision_hidden_size() -> usize {
    1024
}
fn default_vision_intermediate_size() -> usize {
    4096
}
fn default_vision_num_layers() -> usize {
    23
}
fn default_vision_num_heads() -> usize {
    16
}
fn default_vision_image_size() -> usize {
    640
}
fn default_vision_patch_size() -> usize {
    20
}
fn default_vision_spatial_merge_size() -> usize {
    2
}
fn default_vision_projection_dim() -> usize {
    2304
}
fn default_vision_layer_norm_eps() -> f64 {
    1e-5
}
fn default_vision_rope_theta() -> f64 {
    10_000.0
}
fn default_vision_image_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}
fn default_vision_image_std() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

impl Default for Inkling2VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_vision_hidden_size(),
            intermediate_size: default_vision_intermediate_size(),
            num_hidden_layers: default_vision_num_layers(),
            num_attention_heads: default_vision_num_heads(),
            image_size: default_vision_image_size(),
            patch_size: default_vision_patch_size(),
            spatial_merge_size: default_vision_spatial_merge_size(),
            projection_dim: default_vision_projection_dim(),
            layer_norm_eps: default_vision_layer_norm_eps(),
            rope_theta: default_vision_rope_theta(),
            image_mean: default_vision_image_mean(),
            image_std: default_vision_image_std(),
        }
    }
}

impl Inkling2VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads.max(1)
    }

    pub fn posemb_grid_size(&self) -> usize {
        self.image_size / self.patch_size.max(1)
    }

    pub fn alignment(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// Build vision config from `mmproj` GGUF metadata (`clip.vision.*`).
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, GgufMetaValue>,
        default_projection_dim: usize,
    ) -> Self {
        let mut cfg = Self {
            projection_dim: default_projection_dim,
            ..Self::default()
        };
        let get_u = |k: &str| {
            metadata
                .get(k)
                .and_then(GgufMetaValue::as_u64)
                .map(|v| v as usize)
        };
        if let Some(v) = get_u("clip.vision.embedding_length") {
            cfg.hidden_size = v;
        }
        if let Some(v) = get_u("clip.vision.feed_forward_length") {
            cfg.intermediate_size = v;
        }
        if let Some(v) = get_u("clip.vision.block_count") {
            cfg.num_hidden_layers = v;
        }
        if let Some(v) = get_u("clip.vision.attention.head_count") {
            cfg.num_attention_heads = v;
        }
        if let Some(v) = get_u("clip.vision.image_size") {
            cfg.image_size = v;
        }
        if let Some(v) = get_u("clip.vision.patch_size") {
            cfg.patch_size = v;
        }
        if let Some(v) = get_u("clip.vision.spatial_merge_size") {
            cfg.spatial_merge_size = v;
        }
        if let Some(v) = get_u("clip.vision.projection_dim") {
            cfg.projection_dim = v;
        }
        if let Some(v) = metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(GgufMetaValue::as_f32)
        {
            cfg.layer_norm_eps = f64::from(v);
        }
        if let Some(GgufMetaValue::ArrayF32(arr)) = metadata.get("clip.vision.image_mean")
            && arr.len() == 3
        {
            cfg.image_mean = [arr[0], arr[1], arr[2]];
        }
        if let Some(GgufMetaValue::ArrayF32(arr)) = metadata.get("clip.vision.image_std")
            && arr.len() == 3
        {
            cfg.image_std = [arr[0], arr[1], arr[2]];
        }
        cfg
    }
}

/// Optional DSpark drafter configuration placeholder for downstream Phase 5.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inkling2DsparkConfig {
    pub num_hidden_layers: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    #[serde(default)]
    pub target_layer_ids: Vec<usize>,
}

/// Unified Inkling2 / Inkling2-Mini model configuration.
///
/// Accepts:
/// - Normalized GGUF `config.json` from `extract_config` (both Unsloth `inkling2`
///   and mainline `llama.cpp#32694` `inkling2_loop` schemas).
/// - HuggingFace `config.json` (`inkling2` / `inkling2_loop`, flat or nested under `text_config`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inkling2Config {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Logical layer visit count (48 for Inkling-2-Mini).
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    /// Physical layer block count (24 for Inkling-2-Mini; equals `num_hidden_layers` when non-looped).
    #[serde(default)]
    pub num_physical_layers: Option<usize>,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub value_head_dim: Option<usize>,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default)]
    pub padded_vocab_size: Option<usize>,
    #[serde(default)]
    pub unpadded_vocab_size: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(
        default = "default_max_position_embeddings",
        alias = "model_max_length"
    )]
    pub max_position_embeddings: usize,
    #[serde(default = "default_sliding_window", alias = "sliding_window_size")]
    pub sliding_window: usize,
    #[serde(default)]
    pub sliding_window_pattern: Vec<bool>,
    #[serde(default)]
    pub local_layer_ids: Vec<usize>,
    #[serde(default = "default_sconv_kernel_size")]
    pub sconv_kernel_size: usize,
    #[serde(default = "default_n_layers_per_attn_res_block")]
    pub n_layers_per_attn_res_block: usize,
    #[serde(default = "default_d_rel")]
    pub d_rel: usize,
    #[serde(default = "default_rel_extent")]
    pub rel_extent: usize,
    #[serde(default = "default_qk_l2_clip_threshold")]
    pub qk_l2_clip_threshold: f64,
    #[serde(default = "default_rel_l2_clip_threshold")]
    pub rel_l2_clip_threshold: f64,
    #[serde(default = "default_log_scaling_n_floor")]
    pub log_scaling_n_floor: usize,
    #[serde(default = "default_log_scaling_alpha")]
    pub log_scaling_alpha: f64,
    #[serde(default = "default_num_experts", alias = "n_routed_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_route_scale")]
    pub route_scale: f64,
    #[serde(default = "default_mlp_gate_softcap")]
    pub mlp_gate_softcap: f64,
    #[serde(default = "default_mlp_up_softcap")]
    pub mlp_up_softcap: f64,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: f64,
    #[serde(default)]
    pub loop_config: Option<Inkling2LoopConfig>,
    #[serde(default)]
    pub vision_config: Option<Inkling2VisionConfig>,
    #[serde(default)]
    pub dspark_config: Option<Inkling2DsparkConfig>,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    /// True when `gate_up_proj` is stored interleaved `(g0, u0, g1, u1, ...)` (HF/vLLM);
    /// false when split `[gate (0..I), up (I..2I)]` (GGUF: both Unsloth and mainline llama.cpp).
    #[serde(default)]
    pub gate_up_interleaved: bool,
}

fn default_model_type() -> String {
    "inkling2".to_string()
}
fn default_hidden_size() -> usize {
    2304
}
fn default_num_hidden_layers() -> usize {
    48
}
fn default_num_attention_heads() -> usize {
    24
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_vocab_size() -> usize {
    201_024
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_max_position_embeddings() -> usize {
    1_048_576
}
fn default_sliding_window() -> usize {
    1024
}
fn default_sconv_kernel_size() -> usize {
    4
}
fn default_n_layers_per_attn_res_block() -> usize {
    6
}
fn default_d_rel() -> usize {
    16
}
fn default_rel_extent() -> usize {
    1024
}
fn default_qk_l2_clip_threshold() -> f64 {
    22.627_416_997_969_522
}
fn default_rel_l2_clip_threshold() -> f64 {
    8.0
}
fn default_log_scaling_n_floor() -> usize {
    8192
}
fn default_log_scaling_alpha() -> f64 {
    0.1
}
fn default_num_experts() -> usize {
    128
}
fn default_num_experts_per_tok() -> usize {
    8
}
fn default_moe_intermediate_size() -> usize {
    1152
}
fn default_route_scale() -> f64 {
    8.0
}
fn default_mlp_gate_softcap() -> f64 {
    4.0
}
fn default_mlp_up_softcap() -> f64 {
    25.0
}
fn default_final_logit_softcapping() -> f64 {
    30.0
}
fn default_eos_token_id() -> u32 {
    200_006
}
fn default_pad_token_id() -> u32 {
    199_999
}

impl Inkling2Config {
    /// Parse either a flat GGUF-extracted `config.json` or a HuggingFace composite
    /// `config.json` (which nests text fields inside `text_config` and vision fields
    /// inside `vision_config`).
    pub fn from_json_str(raw: &str, is_gguf_checkpoint: bool) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
            Error::from_reason(format!("Failed to parse Inkling2 config.json: {e}"))
        })?;
        let text_val = root.get("text_config").unwrap_or(&root);
        let mut cfg: Inkling2Config = serde_json::from_value(text_val.clone()).map_err(|e| {
            Error::from_reason(format!("Failed to deserialize Inkling2Config: {e}"))
        })?;

        if cfg.vision_config.is_none()
            && let Some(vc) = root.get("vision_config")
            && !vc.is_null()
            && let Ok(mut parsed) = serde_json::from_value::<Inkling2VisionConfig>(vc.clone())
        {
            if vc.get("image_size").is_none()
                && let Some(grid) = vc.get("posemb_grid_size").and_then(|v| v.as_u64())
                && grid > 0
            {
                parsed.image_size = (grid as usize).saturating_mul(parsed.patch_size);
            }
            if vc.get("intermediate_size").is_none()
                && let Some(ratio) = vc.get("mlp_ratio").and_then(|v| v.as_f64())
                && ratio > 0.0
            {
                parsed.intermediate_size = (parsed.hidden_size as f64 * ratio).round() as usize;
            }
            cfg.vision_config = Some(parsed);
        }

        if text_val.get("eos_token_id").is_none()
            && let Some(eos) = root.get("eos_token_id").and_then(|v| v.as_u64())
            && let Ok(eos_u32) = u32::try_from(eos)
        {
            cfg.eos_token_id = eos_u32;
        }

        if cfg.dspark_config.is_none()
            && let Some(mtp) = root.get("mtp_config")
            && !mtp.is_null()
            && let Ok(parsed) = serde_json::from_value::<Inkling2DsparkConfig>(mtp.clone())
        {
            cfg.dspark_config = Some(parsed);
        }

        // HF checkpoints have `num_hidden_layers = 24` (physical) with `loop_config`
        // and no explicit `num_physical_layers`.
        if cfg.num_physical_layers.is_none() {
            if let Some(ref lc) = cfg.loop_config {
                let extra = lc
                    .num_passes
                    .saturating_sub(1)
                    .saturating_mul(lc.core_end.saturating_sub(lc.core_start));
                if cfg.num_hidden_layers == lc.core_end
                    || cfg.num_hidden_layers < lc.core_end.saturating_add(extra)
                {
                    cfg.num_physical_layers = Some(cfg.num_hidden_layers);
                    cfg.num_hidden_layers = cfg.num_hidden_layers.saturating_add(extra);
                } else {
                    cfg.num_physical_layers = Some(cfg.num_hidden_layers.saturating_sub(extra));
                }
            } else {
                cfg.num_physical_layers = Some(cfg.num_hidden_layers);
            }
        }

        // GGUF checkpoints store split `[gate, up]`; HF checkpoints store interleaved.
        if !is_gguf_checkpoint && root.get("text_config").is_some() {
            cfg.gate_up_interleaved = true;
        }

        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let n_phys = self.num_physical_layers();
        if n_phys == 0 {
            return Err(Error::from_reason(
                "Inkling2Config: num_physical_layers must be > 0",
            ));
        }
        if self.hidden_size == 0 || self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err(Error::from_reason(
                "Inkling2Config: hidden_size and attention head counts must be > 0",
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::from_reason(format!(
                "Inkling2Config: num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.n_layers_per_attn_res_block == 0 {
            return Err(Error::from_reason(
                "Inkling2Config: n_layers_per_attn_res_block must be > 0",
            ));
        }
        if let Some(ref lc) = self.loop_config {
            let block = self.n_layers_per_attn_res_block;
            let min_passes = if lc.is_varloop { 1 } else { 2 };
            if lc.core_start == 0
                || lc.core_start >= lc.core_end
                || lc.core_end > n_phys
                || !lc.core_start.is_multiple_of(block)
                || !lc.core_end.is_multiple_of(block)
                || lc.num_passes < min_passes
                || lc.gate_rank == 0
            {
                return Err(Error::from_reason(format!(
                    "Inkling2Config: invalid loop_config {lc:?} for n_phys={n_phys}, block={block}"
                )));
            }
        }
        Ok(())
    }

    /// Number of physical transformer layers stored in checkpoint weights (24 for Inkling-2-Mini).
    pub fn num_physical_layers(&self) -> usize {
        if let Some(n) = self.num_physical_layers {
            return n;
        }
        if let Some(ref lc) = self.loop_config {
            let extra = lc
                .num_passes
                .saturating_sub(1)
                .saturating_mul(lc.core_end.saturating_sub(lc.core_start));
            self.num_hidden_layers.saturating_sub(extra)
        } else {
            self.num_hidden_layers
        }
    }

    /// Total number of logical layer visits executed per forward pass (48 for Inkling-2-Mini).
    pub fn num_layer_visits(&self) -> usize {
        let n_phys = self.num_physical_layers();
        if let Some(ref lc) = self.loop_config {
            n_phys + lc.num_passes.saturating_sub(1) * (lc.core_end - lc.core_start)
        } else {
            n_phys
        }
    }

    /// Unpadded vocabulary size sliced before final logit softcap (`200058` for Inkling-2-Mini).
    pub fn effective_unpadded_vocab_size(&self) -> usize {
        self.unpadded_vocab_size.unwrap_or(self.vocab_size)
    }

    /// Padded vocabulary size of `embed_tokens` / `lm_head` (`201024` for Inkling-2-Mini).
    pub fn effective_padded_vocab_size(&self) -> usize {
        self.padded_vocab_size.unwrap_or(self.vocab_size)
    }

    /// Effective MoE expert intermediate size (`1152` for Inkling-2-Mini).
    pub fn effective_moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
            .or(self.intermediate_size)
            .unwrap_or_else(default_moe_intermediate_size)
    }

    /// Effective value head dimension (equals `head_dim` = 128 on Inkling-2-Mini).
    pub fn effective_value_head_dim(&self) -> usize {
        self.value_head_dim.unwrap_or(self.head_dim)
    }

    /// Whether physical layer `phys` is a local (SWA) layer.
    pub fn is_local_physical(&self, phys: usize) -> bool {
        let n_phys = self.num_physical_layers();
        if self.sliding_window_pattern.len() == n_phys {
            return self.sliding_window_pattern[phys];
        }
        if let Some(ref lc) = self.loop_config
            && self.sliding_window_pattern.len() == self.num_layer_visits()
        {
            // Map physical index to its first visit index:
            // prelude (0..core_start) -> visit = phys
            // core pass 0 (core_start..core_end) -> visit = phys
            // coda (core_end..n_phys) -> visit = phys + (num_passes - 1) * (core_end - core_start)
            let visit = if phys < lc.core_end {
                phys
            } else {
                phys + (lc.num_passes - 1) * (lc.core_end - lc.core_start)
            };
            if let Some(&b) = self.sliding_window_pattern.get(visit) {
                return b;
            }
        }
        if !self.local_layer_ids.is_empty() {
            return self.local_layer_ids.contains(&phys);
        }
        // Default Inkling2 schedule: every `n_layers_per_attn_res_block`-th layer (5, 11, 17, 23) is global.
        let b = self.n_layers_per_attn_res_block.max(1);
        !(phys + 1).is_multiple_of(b)
    }

    /// Whether logical visit `visit` is a local (SWA) visit.
    pub fn is_local_visit(&self, visit: usize) -> bool {
        if let Some(&b) = self.sliding_window_pattern.get(visit)
            && self.sliding_window_pattern.len() == self.num_layer_visits()
        {
            return b;
        }
        let visits = self.visits();
        visits
            .get(visit)
            .map(|v| v.is_local)
            .unwrap_or_else(|| self.is_local_physical(visit))
    }

    /// Ordered execution schedule of logical layer visits:
    /// Prelude (`0..core_start`, pass 0) -> Core (`core_start..core_end`, passes `0..num_passes`)
    /// -> Coda (`core_end..num_physical_layers`, pass 0).
    pub fn visits(&self) -> Vec<LayerVisit> {
        let n_phys = self.num_physical_layers();
        let mut out = Vec::with_capacity(self.num_layer_visits());
        let mut visit_idx = 0usize;

        if let Some(ref lc) = self.loop_config {
            for phys in 0..lc.core_start {
                out.push(LayerVisit {
                    visit_idx,
                    phys_layer: phys,
                    pass_idx: 0,
                    is_local: self.is_local_physical(phys),
                });
                visit_idx += 1;
            }
            for pass in 0..lc.num_passes {
                for phys in lc.core_start..lc.core_end {
                    out.push(LayerVisit {
                        visit_idx,
                        phys_layer: phys,
                        pass_idx: pass,
                        is_local: self.is_local_physical(phys),
                    });
                    visit_idx += 1;
                }
            }
            for phys in lc.core_end..n_phys {
                out.push(LayerVisit {
                    visit_idx,
                    phys_layer: phys,
                    pass_idx: 0,
                    is_local: self.is_local_physical(phys),
                });
                visit_idx += 1;
            }
        } else {
            for phys in 0..n_phys {
                out.push(LayerVisit {
                    visit_idx,
                    phys_layer: phys,
                    pass_idx: 0,
                    is_local: self.is_local_physical(phys),
                });
                visit_idx += 1;
            }
        }
        out
    }

    /// Whether physical layer `phys` is inside the looped core `[core_start, core_end)`.
    pub fn is_core_physical(&self, phys: usize) -> bool {
        self.loop_config
            .as_ref()
            .is_some_and(|lc| phys >= lc.core_start && phys < lc.core_end)
    }

    /// Core offset `phys - core_start` if `phys` is inside the looped core.
    pub fn core_offset(&self, phys: usize) -> Option<usize> {
        let lc = self.loop_config.as_ref()?;
        (phys >= lc.core_start && phys < lc.core_end).then(|| phys - lc.core_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_visit_schedule_and_swa_pattern() {
        let raw = serde_json::json!({
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
        .to_string();

        let cfg = Inkling2Config::from_json_str(&raw, true).unwrap();
        assert_eq!(cfg.num_physical_layers(), 24);
        assert_eq!(cfg.num_layer_visits(), 48);

        let visits = cfg.visits();
        assert_eq!(visits.len(), 48);
        // Prelude
        assert_eq!(visits[0].phys_layer, 0);
        assert_eq!(visits[5].phys_layer, 5);
        assert!(!visits[5].is_local);
        // Core pass 0, 1, 2
        assert_eq!((visits[6].phys_layer, visits[6].pass_idx), (6, 0));
        assert_eq!((visits[18].phys_layer, visits[18].pass_idx), (6, 1));
        assert_eq!((visits[30].phys_layer, visits[30].pass_idx), (6, 2));
        assert_eq!((visits[41].phys_layer, visits[41].pass_idx), (17, 2));
        // Coda
        assert_eq!((visits[42].phys_layer, visits[42].pass_idx), (18, 0));
        assert_eq!((visits[47].phys_layer, visits[47].pass_idx), (23, 0));

        let global_visits: Vec<usize> = visits
            .iter()
            .filter(|v| !v.is_local)
            .map(|v| v.visit_idx)
            .collect();
        assert_eq!(global_visits, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(visits.iter().filter(|v| v.is_local).count(), 40);
    }

    #[test]
    fn hf_composite_config_normalizes_loop_and_interleaved() {
        let raw = serde_json::json!({
            "text_config": {
                "model_type": "inkling2_loop",
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "intermediate_size": 1152,
                "loop_config": {
                    "core_start": 6,
                    "core_end": 18,
                    "num_passes": 3,
                    "gate_rank": 64,
                    "is_varloop": false
                }
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 23,
                "num_attention_heads": 16,
                "patch_size": 20,
                "spatial_merge_size": 2,
                "projection_dim": 2304
            }
        })
        .to_string();

        let cfg = Inkling2Config::from_json_str(&raw, false).unwrap();
        assert_eq!(cfg.num_physical_layers(), 24);
        assert_eq!(cfg.num_layer_visits(), 48);
        assert!(cfg.gate_up_interleaved);
        assert_eq!(cfg.vision_config.as_ref().unwrap().num_hidden_layers, 23);
    }

    #[test]
    fn hf_original_safetensors_config_parses_vision_mtp_and_max_length() {
        let raw = serde_json::json!({
            "model_type": "inkling2_loop",
            "eos_token_id": 200006,
            "text_config": {
                "model_type": "inkling2_loop_text",
                "model_max_length": 1048576,
                "hidden_size": 2304,
                "num_hidden_layers": 24,
                "vocab_size": 201024,
                "unpadded_vocab_size": 200058,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "sliding_window_size": 1024,
                "n_routed_experts": 128,
                "num_experts_per_tok": 8,
                "intermediate_size": 1152,
                "loop_config": {
                    "core_start": 6,
                    "core_end": 18,
                    "num_passes": 3,
                    "gate_rank": 64,
                    "is_varloop": false
                }
            },
            "vision_config": {
                "model_type": "inkling2_vision",
                "text_hidden_size": 2304,
                "native_patch_size": 20,
                "adapter_merge_size": 2,
                "hidden_size": 1024,
                "num_hidden_layers": 23,
                "num_attention_heads": 16,
                "mlp_ratio": 4.0,
                "posemb_grid_size": 32,
                "layer_norm_eps": 1e-5
            },
            "mtp_config": {
                "num_hidden_layers": 3,
                "block_size": 8,
                "markov_rank": 256,
                "target_layer_ids": [5, 23, 41, 48]
            }
        })
        .to_string();

        let cfg = Inkling2Config::from_json_str(&raw, false).unwrap();
        assert_eq!(cfg.num_physical_layers(), 24);
        assert_eq!(cfg.num_layer_visits(), 48);
        assert_eq!(cfg.max_position_embeddings, 1_048_576);
        assert_eq!(cfg.eos_token_id, 200_006);
        assert!(cfg.gate_up_interleaved);

        let vc = cfg.vision_config.as_ref().unwrap();
        assert_eq!(vc.projection_dim, 2304);
        assert_eq!(vc.patch_size, 20);
        assert_eq!(vc.spatial_merge_size, 2);
        assert_eq!(vc.image_size, 640);
        assert_eq!(vc.intermediate_size, 4096);
        assert_eq!(vc.posemb_grid_size(), 32);

        let dc = cfg.dspark_config.as_ref().unwrap();
        assert_eq!(dc.num_hidden_layers, 3);
        assert_eq!(dc.block_size, 8);
        assert_eq!(dc.markov_rank, 256);
        assert_eq!(dc.target_layer_ids, vec![5, 23, 41, 48]);
    }
}
