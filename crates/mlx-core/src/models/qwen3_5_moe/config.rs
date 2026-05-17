use napi_derive::napi;

/// Qwen3.5 MoE model configuration.
///
/// Contains all fields including MoE-specific ones (num_experts, etc.).
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen3_5MoeConfig {
    // Standard transformer fields
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f64,
    pub head_dim: i32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub max_position_embeddings: i32,
    pub pad_token_id: i32,
    pub eos_token_id: i32,
    pub bos_token_id: i32,

    // Linear attention (GatedDeltaNet) fields
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: i32,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: i32,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: i32,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: i32,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: i32,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    // MoE fields (required for MoE variant)
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub shared_expert_intermediate_size: Option<i32>,
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    #[napi(ts_type = "number[] | undefined")]
    pub mlp_only_layers: Option<Vec<i32>>,

    // Paged attention options (opt-in, mirror Qwen3/Gemma4/LFM2 knobs).
    /// GPU memory budget for paged KV cache in megabytes.
    /// Only used when `use_block_paged_cache` is true.
    /// Default: 2048 (2GB).
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Block size for paged attention (tokens per block).
    /// Only used when `use_block_paged_cache` is true.
    /// Default: 16.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_block_size: Option<u32>,

    /// Use the block-paged KV cache adapter for full-attention layers.
    ///
    /// **OPT-IN — experimental.** Same semantics as the dense
    /// `Qwen3_5Config::use_block_paged_cache` field. Routes full-
    /// attention layers through `PagedKVCacheAdapter`; GDN linear-
    /// attention layers stay on `Qwen3_5LayerCache::Linear`. When
    /// enabled, the compiled MoE C++ forward path
    /// (`mlx_qwen35_moe_compiled_*`) is skipped — the paged adapter is
    /// incompatible with the in-graph compile cache.
    ///
    /// VLM (vision encoder present) is rejected with an error in
    /// `Qwen35MoeInner::new`.
    ///
    /// Default: `None` / `false`.
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub use_block_paged_cache: Option<bool>,

    /// Number of MTP (Multi-Token Prediction) head layers shipped with
    /// the checkpoint. Populated from `mtp_num_hidden_layers` /
    /// `num_nextn_predict_layers` in `config.json`. `0` means the
    /// checkpoint has no MTP heads and the speculative-decode path is
    /// unavailable.
    #[serde(default)]
    pub n_mtp_layers: i32,
}

fn default_linear_num_value_heads() -> i32 {
    64
}
fn default_linear_num_key_heads() -> i32 {
    16
}
fn default_linear_key_head_dim() -> i32 {
    192
}
fn default_linear_value_head_dim() -> i32 {
    128
}
fn default_linear_conv_kernel_dim() -> i32 {
    4
}
fn default_full_attention_interval() -> i32 {
    4
}
fn default_partial_rotary_factor() -> f64 {
    0.25
}
fn default_rope_theta() -> f64 {
    100_000.0
}
fn default_decoder_sparse_step() -> i32 {
    1
}
fn default_norm_topk_prob() -> bool {
    true
}

impl Qwen3_5MoeConfig {
    /// Convert to a dense Qwen3_5Config for shared types (attention, GatedDeltaNet)
    /// that expect the dense config struct.
    pub fn to_dense_config(&self) -> crate::models::qwen3_5::Qwen3_5Config {
        crate::models::qwen3_5::Qwen3_5Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            intermediate_size: self.intermediate_size,
            rms_norm_eps: self.rms_norm_eps,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            attention_bias: self.attention_bias,
            max_position_embeddings: self.max_position_embeddings,
            pad_token_id: self.pad_token_id,
            eos_token_id: self.eos_token_id,
            bos_token_id: self.bos_token_id,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            full_attention_interval: self.full_attention_interval,
            partial_rotary_factor: self.partial_rotary_factor,
            rope_theta: self.rope_theta,
            paged_cache_memory_mb: self.paged_cache_memory_mb,
            paged_block_size: self.paged_block_size,
            use_block_paged_cache: self.use_block_paged_cache,
            n_mtp_layers: self.n_mtp_layers,
        }
    }

    /// Number of full-attention layers, used to size the paged adapter
    /// pool.
    pub fn full_attention_layer_count(&self) -> usize {
        (0..self.num_layers as usize)
            .filter(|&i| !self.is_linear_layer(i))
            .count()
    }

    /// Returns whether a given layer index uses linear attention (GatedDeltaNet)
    /// vs full attention (Qwen3NextAttention).
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        if self.full_attention_interval <= 0 {
            return true;
        }
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval as usize)
    }

    /// Returns whether a given layer should use MoE MLP.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if self.num_experts <= 0 {
            return false;
        }
        let step = self.decoder_sparse_step;
        if step <= 0 {
            return false;
        }
        // Check if this layer is in the mlp_only_layers list (dense override)
        if let Some(ref mlp_only) = self.mlp_only_layers
            && mlp_only.contains(&(layer_idx as i32))
        {
            return false;
        }
        (layer_idx + 1).is_multiple_of(step as usize)
    }

    /// Compute the RoPE dimensions for partial rotary embedding.
    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f64 * self.partial_rotary_factor) as i32
    }

    /// Total key dimension for linear attention.
    pub fn linear_key_dim(&self) -> i32 {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Total value dimension for linear attention.
    pub fn linear_value_dim(&self) -> i32 {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Conv dimension = key_dim*2 + value_dim.
    pub fn linear_conv_dim(&self) -> i32 {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }

    /// Estimate total model memory in bytes (for WiredLimitContext).
    pub fn estimate_memory_bytes(&self) -> u64 {
        let h = self.hidden_size as u64;
        let v = self.vocab_size as u64;
        let n = self.num_layers as u64;

        let ne = self.num_experts as u64;
        let moe_i = self.moe_intermediate_size.unwrap_or(self.intermediate_size) as u64;
        let shared_i = self
            .shared_expert_intermediate_size
            .unwrap_or(self.intermediate_size) as u64;

        let embed = v * h;
        // Expert weights + shared expert + router
        let mlp_params = ne * 3 * h * moe_i + 3 * h * shared_i + ne * h;
        let per_layer = mlp_params
            + h * h * 2  // attention projections (rough)
            + h * 4; // norms, biases, etc.
        let total_params = embed * 2 + n * per_layer + h;

        // 2 bytes per param (bf16)
        total_params * 2
    }
}
