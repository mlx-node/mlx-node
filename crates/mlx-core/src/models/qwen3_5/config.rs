use napi_derive::napi;

/// Qwen3.5 model configuration.
///
/// Supports both dense and MoE variants. MoE fields are optional -
/// when `num_experts` is 0 or None, the model uses dense MLP layers.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Qwen3_5Config {
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

    // MoE fields (Optional — zero/None = dense)
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub num_experts: Option<i32>,
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default = "default_decoder_sparse_step")]
    #[napi(ts_type = "number | undefined")]
    pub decoder_sparse_step: Option<i32>,
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub shared_expert_intermediate_size: Option<i32>,
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub norm_topk_prob: Option<bool>,
    #[serde(default)]
    #[napi(ts_type = "number[] | undefined")]
    pub mlp_only_layers: Option<Vec<i32>>,
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
fn default_decoder_sparse_step() -> Option<i32> {
    Some(1)
}

impl Qwen3_5Config {
    /// Returns true if this is a MoE model.
    pub fn is_moe(&self) -> bool {
        self.num_experts.unwrap_or(0) > 0
    }

    /// Returns whether a given layer index uses linear attention (GatedDeltaNet)
    /// vs full attention (Qwen3NextAttention).
    ///
    /// Rule: `(layer_idx + 1) % full_attention_interval != 0` → linear attention
    /// When `full_attention_interval <= 0`, all layers use linear attention.
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        if self.full_attention_interval <= 0 {
            return true;
        }
        !(layer_idx + 1).is_multiple_of(self.full_attention_interval as usize)
    }

    /// Returns whether a given layer should use MoE MLP.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if !self.is_moe() {
            return false;
        }
        let step = self.decoder_sparse_step.unwrap_or(1);
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

    /// Conv dimension = key_dim*2 + value_dim (q + k + v channels through conv1d).
    pub fn linear_conv_dim(&self) -> i32 {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}
