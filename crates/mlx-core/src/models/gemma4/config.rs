use napi_derive::napi;

/// Gemma 4 model configuration (dense variant).
///
/// Supports E2B (2.3B), E4B (4.5B), and 31B dense models.
/// For MoE models (26B-A4B), use `Gemma4MoeConfig` from `gemma4_moe`.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gemma4Config {
    // Standard transformer fields
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: i32,

    // Hybrid attention (sliding window + full attention)
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Explicit per-layer attention type: "sliding_attention" or "full_attention".
    /// Parsed from `text_config.layer_types` in the HuggingFace config.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// RoPE theta for global (full) attention layers.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// RoPE theta for sliding (local) attention layers.
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f64,
    /// Fraction of head_dim to rotate for global attention (0.25 = 25%).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    // Global attention dimensions (differ from sliding in 31B/26B-A4B)
    /// KV heads for global layers. If None, uses num_key_value_heads.
    #[serde(default)]
    pub global_num_key_value_heads: Option<i32>,
    /// Head dimension for global layers. If None, uses head_dim.
    #[serde(default)]
    pub global_head_dim: Option<i32>,

    // K=V sharing (keys and values share projection weights)
    // HF config field: `attention_k_eq_v`
    #[serde(default)]
    pub attention_k_eq_v: bool,

    // Logit softcapping: tanh(logits / cap) * cap
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,

    // Per-layer embeddings (E2B/E4B only)
    // Detected from presence of `vocab_size_per_layer_input` in config.
    #[serde(default)]
    pub per_layer_input_embeds: bool,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<i32>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<i32>,

    // Token IDs
    #[serde(default)]
    pub pad_token_id: i32,
    #[serde(default = "default_eos_token_ids")]
    pub eos_token_ids: Vec<i32>,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: i32,

    #[serde(default)]
    pub attention_bias: bool,

    // Double-wide MLP for shared KV layers (E2B: last 20 layers get 2x intermediate_size)
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub num_kv_shared_layers: Option<i32>,

    // Sampling defaults (from generation_config.json)
    #[serde(default)]
    pub default_temperature: Option<f64>,
    #[serde(default)]
    pub default_top_k: Option<i32>,
    #[serde(default)]
    pub default_top_p: Option<f64>,

    // MoE (Mixture of Experts) — used by 26B-A4B model.
    // When enable_moe_block=true, ALL layers have MoE parallel to the dense MLP.
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,

    // Vision fields (None when no vision_config in config.json — text-only model)
    pub vision_config: Option<super::vision_config::Gemma4VisionConfig>,
    pub image_token_id: Option<i32>,               // 258880
    pub boi_token_id: Option<i32>,                 // 255999
    pub eoi_token_id: Option<i32>,                 // 258882
    pub vision_soft_tokens_per_image: Option<i32>, // 280
}

fn default_sliding_window() -> i32 {
    512
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_rope_local_base_freq() -> f64 {
    10_000.0
}
fn default_partial_rotary_factor() -> f64 {
    0.25
}
fn default_eos_token_ids() -> Vec<i32> {
    vec![1]
}
fn default_bos_token_id() -> i32 {
    2
}

impl Gemma4Config {
    /// Effective intermediate size for a given layer.
    /// Last `num_kv_shared_layers` layers get 2x if `use_double_wide_mlp`.
    pub fn effective_intermediate_size(&self, layer_idx: usize) -> i32 {
        if self.use_double_wide_mlp
            && let Some(shared) = self.num_kv_shared_layers
        {
            let threshold = self.num_hidden_layers - shared;
            if layer_idx as i32 >= threshold {
                return self.intermediate_size * 2;
            }
        }
        self.intermediate_size
    }

    /// Whether a given layer index uses global (full) attention.
    /// Checks the `layer_types` array; falls back to all-global if empty.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        if let Some(lt) = self.layer_types.get(layer_idx) {
            lt == "full_attention"
        } else {
            // Fallback: if layer_types is empty or too short, treat all as global
            true
        }
    }

    /// Whether a given layer index uses sliding (local) attention.
    pub fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        !self.is_global_layer(layer_idx)
    }

    /// Effective head dimension for a given layer type.
    pub fn effective_head_dim(&self, is_global: bool) -> i32 {
        if is_global {
            self.global_head_dim.unwrap_or(self.head_dim)
        } else {
            self.head_dim
        }
    }

    /// Effective number of KV heads for a given layer type.
    /// `global_num_key_value_heads` only applies to k_eq_v global layers,
    /// matching vLLM which only uses it for k_eq_v layers.
    pub fn effective_kv_heads(&self, is_global: bool) -> i32 {
        if is_global && self.attention_k_eq_v {
            self.global_num_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    /// RoPE dimensions for global attention (partial rotation).
    /// Uses `global_head_dim * partial_rotary_factor` because vLLM computes
    /// `rotary_dim = int(head_size * partial_rotary_factor)` where `head_size`
    /// is the per-layer head_dim (which is `global_head_dim` for global layers).
    /// For E2B: 512 * 0.25 = 128 rotary dims.
    pub fn rope_dims_global(&self) -> i32 {
        let global_hd = self.effective_head_dim(true);
        (global_hd as f64 * self.partial_rotary_factor) as i32
    }

    /// RoPE dimensions for sliding attention (full rotation).
    pub fn rope_dims_sliding(&self) -> i32 {
        self.head_dim
    }

    /// Per-layer embedding dimension (ple_dim).
    /// Returns `hidden_size_per_layer_input` if present, otherwise 0.
    pub fn ple_dim(&self) -> i32 {
        self.hidden_size_per_layer_input.unwrap_or(0)
    }

    /// First shared layer index (= num_hidden_layers - num_kv_shared_layers).
    /// Returns num_hidden_layers if KV sharing is not enabled.
    pub fn first_kv_shared_layer(&self) -> usize {
        let shared = self.num_kv_shared_layers.unwrap_or(0);
        if shared <= 0 {
            return self.num_hidden_layers as usize;
        }
        (self.num_hidden_layers - shared) as usize
    }

    /// Whether a layer is a KV-shared layer (reuses K/V from an anchor layer).
    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_kv_shared_layer()
    }

    /// For a shared layer, find the index of its anchor (last non-shared layer
    /// of the same attention type). Returns None if the layer is not shared.
    pub fn kv_shared_anchor(&self, layer_idx: usize) -> Option<usize> {
        if !self.is_kv_shared_layer(layer_idx) {
            return None;
        }
        let target_type = self.layer_types.get(layer_idx)?;
        let first_shared = self.first_kv_shared_layer();
        // Search backwards from first_shared-1 to 0 for the same layer type
        (0..first_shared)
            .rev()
            .find(|&i| self.layer_types.get(i).is_some_and(|t| t == target_type))
    }

    /// Whether a non-shared layer should store its full KV for sharing.
    /// True for the LAST non-shared layer of each attention type.
    pub fn should_store_shared_kv(&self, layer_idx: usize) -> bool {
        if self.is_kv_shared_layer(layer_idx) {
            return false;
        }
        let first_shared = self.first_kv_shared_layer();
        if first_shared >= self.num_hidden_layers as usize {
            // No sharing enabled
            return false;
        }
        let Some(my_type) = self.layer_types.get(layer_idx) else {
            return false;
        };
        // Check if this is the last non-shared layer of this type
        !(layer_idx + 1..first_shared)
            .any(|i| self.layer_types.get(i).is_some_and(|t| t == my_type))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_default_layer_types_synthesis() {
        // mlx-lm default: sliding_window_pattern=5 → 4 sliding + 1 full per cycle
        // 35 layers = 7 complete cycles: indices 4,9,14,19,24,29,34 are full
        let swp = 5usize;
        let n = 35usize;
        let mut pattern = Vec::with_capacity(swp);
        for _ in 0..swp - 1 {
            pattern.push("sliding_attention".to_string());
        }
        pattern.push("full_attention".to_string());
        let types: Vec<String> = (0..n).map(|i| pattern[i % pattern.len()].clone()).collect();

        assert_eq!(types[4], "full_attention");
        assert_eq!(types[9], "full_attention");
        assert_eq!(types[14], "full_attention");
        assert_eq!(types[34], "full_attention");
        assert_eq!(types[0], "sliding_attention");
        assert_eq!(types[3], "sliding_attention");
        assert_eq!(types[5], "sliding_attention");

        let full_count = types.iter().filter(|t| *t == "full_attention").count();
        assert_eq!(full_count, 7); // 35 / 5 = 7 full layers
    }
}
