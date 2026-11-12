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

    // Block-paged pool options.
    /// Ceiling of the paged KV pool in MiB, at most the memory plan's.
    /// Default: the memory plan's.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Block size for paged attention (tokens per block).
    /// Default: 16.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_block_size: Option<u32>,

    /// Number of MTP (Multi-Token Prediction) head layers shipped with
    /// the checkpoint. Populated from `mtp_num_hidden_layers` /
    /// `num_nextn_predict_layers` in `config.json`. `0` means the
    /// checkpoint has no MTP heads and the speculative-decode path is
    /// unavailable.
    #[serde(default)]
    pub n_mtp_layers: i32,

    /// Internal layout marker written by native Qwen3.5/3.8 GGUF conversion.
    /// `Some("tiled")` keeps llama.cpp's value-head order and lets the shared
    /// GDN runtime map value head h to key head h % Hk without permuting
    /// packed weights.
    #[serde(default)]
    #[napi(ts_type = "string | undefined")]
    pub qwen35_gguf_gdn_layout: Option<String>,

    /// Full-attention K/V format of the block-paged pool, `int8` or `bf16`.
    /// Unset: the default of `Qwen3_5Config::kv_format_for` (int8 on a pool
    /// at head 256 when compute is BF16 and a block is 16 or 32 tokens,
    /// else BF16).
    /// Never saved (a runtime option, like Splash
    /// `--kv-format`).
    #[serde(default, skip_serializing)]
    #[napi(ts_type = "string | undefined")]
    pub kv_format: Option<String>,

    /// YaRN of a load past the native window (`maxContextTokens`); never
    /// saved, so `max_position_embeddings` stays the native window.
    #[napi(skip_typescript)]
    #[serde(skip)]
    pub rope_scaling: Option<crate::nn::rope_scaling::RopeScaling>,
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
    /// Splash `servedContextTokens`: the native window, or the YaRN window.
    pub(crate) fn served_context_tokens(&self) -> u32 {
        self.scaling()
            .served_context_tokens(self.max_position_embeddings.max(0) as u32)
    }

    /// The YaRN this load serves; none without `maxContextTokens` past the
    /// native window.
    pub(crate) fn scaling(&self) -> crate::nn::rope_scaling::RopeScaling {
        self.rope_scaling.unwrap_or_default()
    }

    /// Tokens per paged KV block; also the granularity of DFlash2 prefix
    /// accounting.
    pub(crate) fn paged_block_size(&self) -> u32 {
        self.paged_block_size.unwrap_or(16)
    }

    /// BF16 bytes for one request's complete GDN conv + recurrent state.
    /// The MoE feed-forward layers are stateless; full-attention K/V is
    /// accounted independently by the paged allocator.
    pub(crate) fn recurrent_state_bytes(&self) -> u64 {
        self.to_dense_config().recurrent_state_bytes()
    }

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
            n_mtp_layers: self.n_mtp_layers,
            qwen35_gguf_gdn_layout: self.qwen35_gguf_gdn_layout.clone(),
            kv_format: self.kv_format.clone(),
            rope_scaling: self.rope_scaling,
        }
    }

    /// See `Qwen3_5Config::kv_format_for`.
    pub fn kv_format_for(&self, compute: crate::array::DType) -> crate::transformer::KvFormat {
        self.to_dense_config().kv_format_for(compute)
    }

    /// See `Qwen3_5Config::pin_kv_format`.
    pub(crate) fn pin_kv_format(
        &mut self,
        requested: Option<crate::transformer::KvFormat>,
        compute: crate::array::DType,
    ) -> crate::transformer::KvFormat {
        let format = self.to_dense_config().pin_kv_format(requested, compute);
        self.kv_format = Some(format.as_str().to_string());
        format
    }

    /// See `Qwen3_5Config::kv_format`.
    pub fn kv_format(&self) -> crate::transformer::KvFormat {
        self.to_dense_config().kv_format()
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
}
