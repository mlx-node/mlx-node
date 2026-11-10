use napi_derive::napi;

use crate::models::paged_config::PagedCacheConfig;

/// Resolve the shared dense/MoE Qwen3.5 paged-cache policy.
///
/// The environment override is intentionally an input rather than read here:
/// both loaders use this pure function, and unit tests can pin precedence
/// without mutating process-global environment state.
pub(crate) fn resolve_qwen35_paged_default(
    explicit: Option<bool>,
    env_override: Option<&str>,
) -> Option<bool> {
    match env_override {
        Some("1") | Some("true") | Some("TRUE") => Some(true),
        Some("0") | Some("false") | Some("FALSE") => Some(false),
        _ => PagedCacheConfig::resolve_use_paged_default(explicit, true),
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

    // Paged attention options (default-on, mirror Qwen3/Gemma4/LFM2 knobs).
    /// Ceiling of the paged KV pool in MiB, at most the memory plan's.
    /// Only used when `use_block_paged_cache` is true.
    /// Default: the memory plan's.
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
    /// When enabled (the default), `Qwen35Inner`
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
    /// over the eager flat decode. When enabled, full-attention
    /// layers run through the paged adapter (cross-request prefix reuse);
    /// an explicit false runs eager flat decode. Either way the forward is
    /// pure-Rust eager.
    ///
    /// **VLM under paged**: VLM checkpoints also default this flag ON, so
    /// dense image turns ONLY run on the paged-vision core. A fresh single-turn
    /// image-bearing prompt prefills through the paged adapter (M-RoPE positions
    /// feed the rotary; the merged vision embeddings feed the forward) and
    /// decodes plain AR — MTP weights are ignored on image turns. Warm
    /// image-bearing session continues / cache-hit reuse are still rejected at
    /// runtime (the GDN two-pass warm prefix is not byte-exact). A vision turn
    /// that reaches a None adapter (explicit `Some(false)`, non-Metal build, or
    /// a sym8 checkpoint) errors at dispatch.
    ///
    /// Load default: `Some(true)` for compatible text and VLM checkpoints.
    /// Explicit false remains available for flat-path diagnostics; sym8 is
    /// forced flat by persistence after its storage mode is known.
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

    /// Internal layout marker written by native Qwen3.5/3.8 GGUF conversion.
    /// `Some("tiled")` keeps llama.cpp's value-head order and lets GDN map
    /// value head h to key head h % Hk without permuting packed weights.
    #[serde(default)]
    #[napi(ts_type = "string | undefined")]
    pub qwen35_gguf_gdn_layout: Option<String>,

    /// Element format of the flat full-attention K/V cache: `"int8"`
    /// (per-(token, head) symmetric int8 rows with one fp32 scale each —
    /// half the K/V memory and read bandwidth; see `crate::array::kv_int8`)
    /// or `"bf16"`. Unset: int8 where the int8 kernels serve the model
    /// (Metal, head 256, GQA up to 8, BF16 compute), else BF16 — see
    /// `kv_format_for()`. Set from the `kvFormat` load option; the paged
    /// cache ignores it.
    #[serde(default)]
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

impl Qwen3_5Config {
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

    /// Bytes for one request's complete GDN state: bf16 conv history plus the
    /// f32-carried recurrent state. Full-attention K/V is accounted separately
    /// by the paged allocator.
    pub(crate) fn recurrent_state_bytes(&self) -> u64 {
        let linear_layers = (0..self.num_layers.max(0) as usize)
            .filter(|&layer| self.is_linear_layer(layer))
            .count() as u64;
        let conv_elements = u64::try_from((self.linear_conv_kernel_dim - 1).max(0))
            .unwrap_or(0)
            .saturating_mul(u64::try_from(self.linear_conv_dim().max(0)).unwrap_or(0));
        let recurrent_elements = u64::try_from(self.linear_num_value_heads.max(0))
            .unwrap_or(0)
            .saturating_mul(u64::try_from(self.linear_value_head_dim.max(0)).unwrap_or(0))
            .saturating_mul(u64::try_from(self.linear_key_head_dim.max(0)).unwrap_or(0));
        linear_layers.saturating_mul(
            conv_elements
                .saturating_mul(2)
                .saturating_add(recurrent_elements.saturating_mul(4)),
        )
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

    /// Element format of the flat full-attention K/V cache of a model that
    /// computes in `compute`. An explicit `kv_format` wins (an unparseable
    /// value is BF16 — persistence rejects it at load); otherwise int8,
    /// Splash's default, wherever the int8 segmented kernels serve the model
    /// (`KvFormat::default_for_geometry`: BF16 compute, head 256, GQA up to
    /// 8), and BF16 elsewhere so no model falls onto the dequantizing
    /// fallback.
    pub fn kv_format_for(&self, compute: crate::array::DType) -> crate::transformer::KvFormat {
        use crate::transformer::KvFormat;
        match self.kv_format.as_deref() {
            Some(value) if !value.trim().is_empty() => {
                KvFormat::parse(Some(value)).unwrap_or_default()
            }
            _ => KvFormat::default_for_geometry(
                i64::from(self.head_dim),
                i64::from(self.num_heads),
                i64::from(self.num_kv_heads),
                compute,
            ),
        }
    }

    /// Resolves the load's format (`requested` wins over the config) for the
    /// checkpoint's `compute` dtype and pins it into `kv_format`, so every
    /// cache and every window price after the load reads the same format.
    pub(crate) fn pin_kv_format(
        &mut self,
        requested: Option<crate::transformer::KvFormat>,
        compute: crate::array::DType,
    ) -> crate::transformer::KvFormat {
        let format = requested.unwrap_or_else(|| self.kv_format_for(compute));
        self.kv_format = Some(format.as_str().to_string());
        format
    }

    /// The format of the flat caches a model built from this config uses. A
    /// load pins it (`pin_kv_format`); an unpinned config builds a random-init
    /// model, whose weights are f32.
    pub fn kv_format(&self) -> crate::transformer::KvFormat {
        self.kv_format_for(crate::array::DType::Float32)
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
    use super::{Qwen3_5Config, resolve_qwen35_paged_default};

    #[test]
    fn gdn_state_bytes_follow_the_real_conv_and_recurrent_shapes() {
        let config = Qwen3_5Config {
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
            vocab_size: 32,
            hidden_size: 16,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 128,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 4,
            linear_value_head_dim: 3,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 2,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(true),
            persist_paged_cache: None,
            n_mtp_layers: 0,
        };
        // Two linear layers. conv=[3, (1*4)*2 + (2*3)=14] bf16, recurrent=[2,3,4] f32.
        assert_eq!(
            config.recurrent_state_bytes(),
            2 * (3 * 14 * 2 + 2 * 3 * 4 * 4)
        );
    }

    #[test]
    fn paged_default_and_override_precedence_match_dense_and_moe() {
        assert_eq!(resolve_qwen35_paged_default(None, None), Some(true));
        assert_eq!(resolve_qwen35_paged_default(Some(false), None), Some(false));
        assert_eq!(
            resolve_qwen35_paged_default(Some(false), Some("1")),
            Some(true)
        );
        assert_eq!(
            resolve_qwen35_paged_default(Some(true), Some("0")),
            Some(false)
        );
        assert_eq!(
            resolve_qwen35_paged_default(None, Some("unexpected")),
            Some(true)
        );
    }
}
