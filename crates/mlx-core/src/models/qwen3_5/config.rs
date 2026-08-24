use napi_derive::napi;

/// Name of the env var that overrides `paged_cache_initial_memory_mb` at
/// load time (u32 MiB). Env wins over config; unset both keeps the
/// historical fixed pool (initial == max).
pub(crate) const MLX_PAGED_CACHE_INITIAL_MB_ENV: &str = "MLX_PAGED_CACHE_INITIAL_MB";

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
    mlx_paged_attn::PagedAttentionConfig::memory_mb_for_one_full_sequence(
        max_seq_len,
        block_size,
        head_size,
        num_kv_heads,
        num_layers,
    )
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

/// Resolve the initial (grow-on-demand) paged-pool size in MiB.
///
/// `None` means "initial == max" — the historical fixed-pool behavior, byte
/// identical to today. `MLX_PAGED_CACHE_INITIAL_MB` wins over the config
/// field; a set-but-unparseable env value is ignored (same precedent as
/// `resolve_qwen35_paged_default`). The environment override is an input
/// rather than read here so unit tests can pin precedence without mutating
/// process-global environment state.
pub(crate) fn qwen35_resolve_paged_cache_initial_memory_mb(
    configured_initial: Option<u32>,
    env_override: Option<&str>,
) -> Option<u32> {
    match env_override.and_then(|raw| raw.trim().parse::<u32>().ok()) {
        Some(mb) => Some(mb),
        None => configured_initial,
    }
}

/// Convert a resolved initial MiB budget into an initial block count.
///
/// Same MiB→blocks math as the max path (`PagedAttentionConfig::
/// calculate_num_blocks`). `initial_mb == None` → the full max pool. A set
/// budget is clamped to `min(initial, max)` — first in the MiB domain
/// against `max_memory_mb`, then in the block domain against `max_blocks`
/// (which adaptive load-time sizing may have reduced below what
/// `max_memory_mb` itself holds). A budget that cannot hold one block is
/// `Err(())`; the caller raises the family-specific load error.
pub(crate) fn qwen35_initial_pool_blocks(
    initial_mb: Option<u32>,
    max_memory_mb: u32,
    max_blocks: u32,
    pa_config: &mlx_paged_attn::PagedAttentionConfig,
) -> Result<(u32, Option<u32>), ()> {
    let Some(mb) = initial_mb else {
        return Ok((max_blocks, None));
    };
    let clamped_mb = mb.min(max_memory_mb);
    let initial_config = mlx_paged_attn::PagedAttentionConfig {
        gpu_memory_mb: clamped_mb,
        ..pa_config.clone()
    };
    let blocks = initial_config.calculate_num_blocks();
    if blocks == 0 {
        return Err(());
    }
    Ok((blocks.min(max_blocks), Some(clamped_mb)))
}

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
        _ => Some(explicit.unwrap_or(true)),
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
    /// GPU memory budget for paged KV cache in megabytes.
    /// Only used when `use_block_paged_cache` is true.
    /// Default: automatically sized for one full-context sequence.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_memory_mb: Option<u32>,

    /// Initial paged KV pool size in MiB for the grow-on-demand pool: the
    /// pool starts at this size and grows toward the max budget
    /// (`paged_cache_memory_mb` or the auto one-full-context default) on
    /// exhaustion. Unset (default) makes the initial pool equal the max —
    /// the historical fixed-pool behavior. `MLX_PAGED_CACHE_INITIAL_MB` wins
    /// over this field at load time.
    #[serde(default)]
    #[napi(ts_type = "number | undefined")]
    pub paged_cache_initial_memory_mb: Option<u32>,

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
    /// BF16 bytes for one request's complete GDN conv + recurrent state.
    /// Full-attention K/V is accounted separately by the paged allocator.
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
        linear_layers
            .saturating_mul(conv_elements.saturating_add(recurrent_elements))
            .saturating_mul(2)
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
    use super::{
        Qwen3_5Config, qwen35_default_paged_cache_memory_mb, qwen35_initial_pool_blocks,
        qwen35_resolve_paged_cache_initial_memory_mb, qwen35_resolve_paged_cache_memory_mb,
        resolve_qwen35_paged_default,
    };
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
    fn gdn_state_bytes_follow_the_real_conv_and_recurrent_shapes() {
        let config = Qwen3_5Config {
            qwen35_gguf_gdn_layout: None,
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
            paged_cache_initial_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(true),
            persist_paged_cache: None,
            n_mtp_layers: 0,
        };
        // Two linear layers. conv=[3, (1*4)*2 + (2*3)=14], recurrent=[2,3,4].
        assert_eq!(config.recurrent_state_bytes(), 2 * (3 * 14 + 2 * 3 * 4) * 2);
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

    #[test]
    fn initial_memory_resolver_precedence_env_over_config() {
        assert_eq!(
            qwen35_resolve_paged_cache_initial_memory_mb(None, None),
            None
        );
        assert_eq!(
            qwen35_resolve_paged_cache_initial_memory_mb(Some(64), None),
            Some(64)
        );
        assert_eq!(
            qwen35_resolve_paged_cache_initial_memory_mb(Some(64), Some("128")),
            Some(128),
            "env must win over config"
        );
        assert_eq!(
            qwen35_resolve_paged_cache_initial_memory_mb(None, Some(" 128 ")),
            Some(128),
            "whitespace-padded env values parse"
        );
        assert_eq!(
            qwen35_resolve_paged_cache_initial_memory_mb(Some(64), Some("not-a-number")),
            Some(64),
            "an unparseable env value falls back to config"
        );
        assert_eq!(
            qwen35_resolve_paged_cache_initial_memory_mb(None, Some("0")),
            Some(0),
            "env=0 is an explicit (rejected-at-load) zero budget, not absence"
        );
    }

    #[test]
    fn initial_pool_blocks_unset_means_initial_equals_max() {
        // agents-a1 geometry: 320 KiB per block.
        let (max_memory_mb, _) = (5_120, "auto_full_context");
        let pa = paged_config(max_memory_mb, 256, 2, 10);
        let max_blocks = pa.calculate_num_blocks();
        assert_eq!(max_blocks, 16_384);
        let (blocks, mb) = qwen35_initial_pool_blocks(None, max_memory_mb, max_blocks, &pa)
            .expect("unset initial must succeed");
        assert_eq!(blocks, max_blocks, "initial == max when unset");
        assert_eq!(mb, None);
    }

    #[test]
    fn initial_pool_blocks_smaller_than_max_uses_the_same_block_math() {
        let max_memory_mb = 5_120;
        let pa = paged_config(max_memory_mb, 256, 2, 10);
        let max_blocks = pa.calculate_num_blocks();
        // 1 MiB / (320 KiB per block) = 3 blocks.
        let (blocks, mb) = qwen35_initial_pool_blocks(Some(1), max_memory_mb, max_blocks, &pa)
            .expect("a 1 MiB initial budget must succeed");
        assert_eq!(blocks, 3);
        assert_eq!(mb, Some(1));
        assert!(blocks < max_blocks, "initial must sit below the max pool");
    }

    #[test]
    fn initial_pool_blocks_clamps_to_the_max_budget() {
        let max_memory_mb = 5_120;
        let pa = paged_config(max_memory_mb, 256, 2, 10);
        let max_blocks = pa.calculate_num_blocks();
        let (blocks, mb) = qwen35_initial_pool_blocks(Some(99_999), max_memory_mb, max_blocks, &pa)
            .expect("an oversized initial budget clamps instead of failing");
        assert_eq!(blocks, max_blocks);
        assert_eq!(mb, Some(max_memory_mb));
    }

    #[test]
    fn initial_pool_blocks_clamps_below_an_adaptively_sized_max() {
        // `load_time_pool_sizing` may reduce the max below what the MiB
        // budget holds; the block-domain clamp must respect the final max.
        let max_memory_mb = 5_120;
        let pa = paged_config(max_memory_mb, 256, 2, 10);
        let adaptive_max_blocks = 100;
        let (blocks, mb) =
            qwen35_initial_pool_blocks(Some(64), max_memory_mb, adaptive_max_blocks, &pa)
                .expect("initial within budget must succeed");
        assert_eq!(blocks, adaptive_max_blocks);
        assert_eq!(mb, Some(64));
    }

    #[test]
    fn initial_pool_blocks_rejects_a_budget_that_cannot_hold_one_block() {
        let pa = paged_config(5_120, 256, 2, 10);
        assert!(
            qwen35_initial_pool_blocks(Some(0), 5_120, 16_384, &pa).is_err(),
            "0 MiB must fail with the one-block error"
        );
    }
}
