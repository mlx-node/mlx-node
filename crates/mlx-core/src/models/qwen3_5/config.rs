use napi_derive::napi;

const BYTES_PER_MIB: u64 = 1024 * 1024;
const QWEN35_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB: u32 = 256;

/// Derive the default paged-KV budget from the model's advertised context.
///
/// Qwen3.5 checkpoints commonly advertise 262,144 tokens. The former fixed
/// 2 GiB default only held 104,848 tokens for the 35B-A3B layout (10 full
/// attention layers, 2 KV heads, head size 256), so a valid long conversation
/// could exhaust the allocator even though it was still well inside the model
/// context window. Keep explicit `paged_cache_memory_mb` overrides intact, but
/// size the implicit default for exactly one full-context sequence.
pub(crate) fn qwen35_default_paged_cache_memory_mb(
    max_seq_len: u32,
    block_size: u32,
    head_size: u32,
    num_kv_heads: u32,
    num_layers: u32,
) -> u32 {
    if max_seq_len == 0 || block_size == 0 || head_size == 0 || num_kv_heads == 0 || num_layers == 0
    {
        return QWEN35_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB;
    }

    let max_blocks = u64::from(max_seq_len.div_ceil(block_size));
    let bytes_per_block = 2u64
        .saturating_mul(u64::from(num_kv_heads))
        .saturating_mul(u64::from(head_size))
        .saturating_mul(u64::from(block_size))
        .saturating_mul(2)
        .saturating_mul(u64::from(num_layers));
    let required_mb = bytes_per_block
        .saturating_mul(max_blocks)
        .div_ceil(BYTES_PER_MIB)
        .max(u64::from(QWEN35_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB));
    u32::try_from(required_mb).unwrap_or(u32::MAX)
}

pub(crate) fn qwen35_resolve_paged_cache_memory_mb(
    configured_memory_mb: Option<u32>,
    default_memory_mb: u32,
) -> (u32, &'static str) {
    match configured_memory_mb {
        Some(memory_mb) => (memory_mb, "config"),
        None => (default_memory_mb, "auto_full_context"),
    }
}

/// Qwen3.5 model configuration (dense variant).
///
/// For MoE models, use `Qwen3_5MoeConfig` from `qwen3_5_moe`.
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

    // Paged attention options (opt-in, mirror Qwen3/Gemma4/LFM2 knobs).
    /// GPU memory budget for paged KV cache in megabytes.
    /// Only used when `use_block_paged_cache` is true.
    /// Default: automatically sized for one full-context sequence.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Block size for paged attention (tokens per block).
    /// Only used when `use_block_paged_cache` is true.
    /// Default: 16.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_block_size: Option<u32>,

    /// Use the block-paged KV cache adapter (`PagedKVCacheAdapter`) for
    /// full-attention layers.
    ///
    /// **OPT-IN — experimental.** When `Some(true)`, `Qwen35Inner`
    /// allocates a `BlockAllocator` + `LayerKVPool` pair sized for the
    /// model's full-attention layer count and constructs a
    /// `PagedKVCacheAdapter`. The chat-session forward dispatch routes
    /// full-attention layers through this adapter while linear-attention
    /// (GatedDeltaNet / GDN) layers continue to use the existing
    /// `Qwen3_5LayerCache::Linear(ArraysCache)` path with no
    /// cross-request prefix reuse — vLLM's `MambaManager`-style "no
    /// prefix reuse for recurrent layers" stance.
    ///
    /// **Paged vs flat eager**: this flag selects the eager paged decode
    /// over the eager flat decode. When `Some(true)`, full-attention
    /// layers run through the paged adapter (cross-request prefix reuse);
    /// when unset, they run the eager flat decode. Either way the forward
    /// is pure-Rust eager.
    ///
    /// **VLM under paged**: a VLM checkpoint defaults this flag ON at load, so
    /// dense image turns ONLY run on the paged-vision core. A fresh single-turn
    /// image-bearing prompt prefills through the paged adapter (M-RoPE positions
    /// feed the rotary; the merged vision embeddings feed the forward) and
    /// decodes plain AR — MTP weights are ignored on image turns. Warm
    /// image-bearing session continues / cache-hit reuse are still rejected at
    /// runtime (the GDN two-pass warm prefix is not byte-exact). A vision turn
    /// that reaches a None adapter (explicit `Some(false)`, non-Metal build, or
    /// a sym8 checkpoint) errors at dispatch.
    ///
    /// Default: `None` for text-only checkpoints (eager flat decode);
    /// `Some(true)` for VLM checkpoints (block-paged, set in `parse_config`).
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub use_block_paged_cache: Option<bool>,

    /// Persist the out-of-pool GDN recurrent state (and the paged KV blocks it
    /// gates) to the SSD cold tier so warm prefixes survive process restarts.
    /// Off unless explicitly enabled. See `crate::models::qwen3_5::gdn_sidecar`
    /// and `crate::cold_tier::resolve_persist_cold`.
    #[serde(default)]
    #[napi(ts_type = "boolean | undefined")]
    pub persist_paged_cache: Option<bool>,

    /// Number of MTP (Multi-Token Prediction) head layers shipped with the
    /// checkpoint. Populated from `mtp_num_hidden_layers` /
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

impl Qwen3_5Config {
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

    /// Number of full-attention layers (i.e. layers that use
    /// `Qwen3_5Attention` rather than `GatedDeltaNet`). Used to size the
    /// paged adapter's `LayerKVPool`.
    pub fn full_attention_layer_count(&self) -> usize {
        (0..self.num_layers as usize)
            .filter(|&i| !self.is_linear_layer(i))
            .count()
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

    /// Estimate total model memory in bytes (for WiredLimitContext).
    /// Assumes bf16 (2 bytes per param) for the main model weights.
    pub fn estimate_memory_bytes(&self) -> u64 {
        let h = self.hidden_size as u64;
        let v = self.vocab_size as u64;
        let n = self.num_layers as u64;
        let i = self.intermediate_size as u64;

        let embed = v * h;
        let mlp_params = 3 * h * i; // MLP gate/up/down
        let per_layer = mlp_params
            + h * h * 2  // attention projections (rough)
            + h * 4; // norms, biases, etc.
        let total_params = embed * 2 + n * per_layer + h;

        // 2 bytes per param (bf16)
        total_params * 2
    }
}

#[cfg(test)]
mod tests {
    use super::{qwen35_default_paged_cache_memory_mb, qwen35_resolve_paged_cache_memory_mb};
    use mlx_paged_attn::PagedAttentionConfig;

    fn paged_config(
        gpu_memory_mb: u32,
        head_size: u32,
        num_kv_heads: u32,
        num_layers: u32,
    ) -> PagedAttentionConfig {
        PagedAttentionConfig {
            block_size: 16,
            gpu_memory_mb,
            head_size,
            num_kv_heads,
            num_layers,
            use_fp8_cache: Some(false),
            max_seq_len: Some(262_144),
            max_batch_size: Some(32),
        }
    }

    #[test]
    fn paged_cache_default_covers_agents_a1_full_context() {
        // agents-a1 / Qwen3.5-35B-A3B: 40 layers at interval 4 gives
        // 10 physical full-attention layers, with 2 KV heads of width 256.
        let memory_mb = qwen35_default_paged_cache_memory_mb(262_144, 16, 256, 2, 10);
        assert_eq!(memory_mb, 5_120);

        let config = paged_config(memory_mb, 256, 2, 10);
        assert_eq!(config.calculate_num_blocks(), 16_384);
        assert_eq!(config.max_cached_tokens(), 262_144);
    }

    #[test]
    fn paged_cache_default_covers_qwen36_27b_full_context() {
        // Qwen3.6-27B: 64 layers at interval 4 gives 16 physical
        // full-attention layers, with 4 KV heads of width 256.
        let memory_mb = qwen35_default_paged_cache_memory_mb(262_144, 16, 256, 4, 16);
        assert_eq!(memory_mb, 16_384);

        let config = paged_config(memory_mb, 256, 4, 16);
        assert_eq!(config.calculate_num_blocks(), 16_384);
        assert_eq!(config.max_cached_tokens(), 262_144);
    }

    #[test]
    fn explicit_2048_mb_override_preserves_the_observed_small_pool() {
        let automatic = qwen35_default_paged_cache_memory_mb(262_144, 16, 256, 2, 10);
        let (memory_mb, source) = qwen35_resolve_paged_cache_memory_mb(Some(2_048), automatic);
        assert_eq!(memory_mb, 2_048);
        assert_eq!(source, "config");

        let config = paged_config(memory_mb, 256, 2, 10);
        assert_eq!(config.calculate_num_blocks(), 6_553);
        assert_eq!(config.max_cached_tokens(), 104_848);
        // The failing Image-agent turn was already beyond this physical
        // capacity while pi still correctly reported 40.4% of 262k.
        assert!(106_240 > config.max_cached_tokens());
    }
}
