use napi_derive::napi;

use crate::models::paged_config::PagedCacheConfig;

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

/// The initial pool size of a load in MiB (`None` = see
/// [`qwen35_initial_pool_blocks`]) and whether the pool is a runway pool.
/// A DFlash2 load always starts at the runway and grows on demand: the env
/// and config initial sizes are ignored, as Splash builds its pool at
/// `warmupKvPages` with no size knob (RuntimeResources.mm:1141,
/// Model.hpp:607-610).
pub(crate) fn qwen35_initial_pool_request(
    configured_initial: Option<u32>,
    env_override: Option<&str>,
    has_dflash2_draft: bool,
) -> (Option<u32>, bool) {
    if has_dflash2_draft {
        return (None, true);
    }
    (
        qwen35_resolve_paged_cache_initial_memory_mb(configured_initial, env_override),
        false,
    )
}

/// Tokens the paged pool starts with when a DFlash2 draft is loaded:
/// DFlash2 turns never write the pool, so it holds only a runway for paged
/// turns and grows on demand (Splash `warmupKvPages`).
pub(crate) const QWEN35_DFLASH2_POOL_RUNWAY_TOKENS: u32 = 2048;

/// Blocks a runway pool asks the load-time sizer for: the requested pool
/// plus the one layer a grow holds beside it.
pub(crate) fn qwen35_runway_pool_sizing_request(requested_blocks: u32, layers: u32) -> u32 {
    requested_blocks.saturating_add(requested_blocks.div_ceil(layers.max(1)))
}

/// The maximum a runway pool publishes and grows to: the most blocks whose
/// last grow (the new pool plus one old layer) fits the `sized_blocks` the
/// sizer selected, never past `requested_blocks`.
pub(crate) fn qwen35_runway_pool_max_blocks(
    sized_blocks: u32,
    requested_blocks: u32,
    layers: u32,
) -> u32 {
    let layers = u64::from(layers.max(1));
    let fits = u64::from(sized_blocks) * layers / (layers + 1);
    (fits.min(u64::from(requested_blocks)) as u32).max(1)
}

/// Convert a resolved initial MiB budget into an initial block count.
///
/// Same MiB→blocks math as the max path (`PagedAttentionConfig::
/// calculate_num_blocks`). `initial_mb == None` → the full max pool, or with
/// a DFlash2 draft the runway plus one request's recurrent-state charge. A
/// set budget is clamped to `min(initial, max)` — first in the MiB domain
/// against `max_memory_mb`, then in the block domain against `max_blocks`
/// (which adaptive load-time sizing may have reduced below what
/// `max_memory_mb` itself holds). A budget that cannot hold one block is
/// `Err(())`; the caller raises the family-specific load error.
pub(crate) fn qwen35_initial_pool_blocks(
    initial_mb: Option<u32>,
    max_memory_mb: u32,
    max_blocks: u32,
    pa_config: &mlx_paged_attn::PagedAttentionConfig,
    has_dflash2_draft: bool,
    recurrent_state_bytes: u64,
) -> Result<(u32, Option<u32>), ()> {
    let Some(mb) = initial_mb else {
        if !has_dflash2_draft {
            return Ok((max_blocks, None));
        }
        let element_bytes: u64 = if pa_config.use_fp8_cache.unwrap_or(false) {
            1
        } else {
            2
        };
        let bytes_per_block = (2 * element_bytes)
            .saturating_mul(u64::from(pa_config.num_kv_heads))
            .saturating_mul(u64::from(pa_config.head_size))
            .saturating_mul(u64::from(pa_config.block_size))
            .saturating_mul(u64::from(pa_config.num_layers))
            .max(1);
        let blocks = QWEN35_DFLASH2_POOL_RUNWAY_TOKENS
            .div_ceil(pa_config.block_size.max(1))
            .saturating_add(
                recurrent_state_bytes
                    .div_ceil(bytes_per_block)
                    .min(u64::from(u32::MAX)) as u32,
            )
            .clamp(1, max_blocks.max(1));
        let mib = (u64::from(blocks) * bytes_per_block).div_ceil(1024 * 1024);
        return Ok((blocks, Some(mib.min(u64::from(u32::MAX)) as u32)));
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
    /// over this field at load time. Both are ignored with a DFlash2 draft:
    /// that pool starts at its runway.
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

    /// Element format of the flat full-attention K/V cache: `"int8"`
    /// (per-(token, head) symmetric int8 rows with one fp32 scale each —
    /// half the K/V memory and read bandwidth; see `crate::array::kv_int8`)
    /// or `"bf16"`. Unset: int8 where the int8 kernels serve the geometry
    /// (Metal, head 256, GQA up to 8), else BF16 — see `kv_format()`. Set
    /// from the `kvFormat` load option; the paged cache ignores it.
    #[serde(default)]
    #[napi(ts_type = "string | undefined")]
    pub kv_format: Option<String>,
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

    /// Element format of the flat full-attention K/V cache. An explicit
    /// `kv_format` wins (an unparseable value is BF16 — persistence rejects
    /// it at load); otherwise int8, Splash's default, wherever the int8
    /// segmented kernels serve the geometry
    /// (`KvFormat::default_for_geometry`), and BF16 elsewhere so no geometry
    /// falls onto the dequantizing fallback.
    pub fn kv_format(&self) -> crate::transformer::KvFormat {
        use crate::transformer::KvFormat;
        match self.kv_format.as_deref() {
            Some(value) if !value.trim().is_empty() => {
                KvFormat::parse(Some(value)).unwrap_or_default()
            }
            _ => KvFormat::default_for_geometry(
                i64::from(self.head_dim),
                i64::from(self.num_heads),
                i64::from(self.num_kv_heads),
            ),
        }
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
        qwen35_initial_pool_request, qwen35_resolve_paged_cache_initial_memory_mb,
        qwen35_resolve_paged_cache_memory_mb, qwen35_runway_pool_max_blocks,
        qwen35_runway_pool_sizing_request, resolve_qwen35_paged_default,
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
            paged_cache_initial_memory_mb: None,
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
        let (blocks, mb) =
            qwen35_initial_pool_blocks(None, max_memory_mb, max_blocks, &pa, false, 0)
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
        let (blocks, mb) =
            qwen35_initial_pool_blocks(Some(1), max_memory_mb, max_blocks, &pa, false, 0)
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
        let (blocks, mb) =
            qwen35_initial_pool_blocks(Some(99_999), max_memory_mb, max_blocks, &pa, false, 0)
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
            qwen35_initial_pool_blocks(Some(64), max_memory_mb, adaptive_max_blocks, &pa, false, 0)
                .expect("initial within budget must succeed");
        assert_eq!(blocks, adaptive_max_blocks);
        assert_eq!(mb, Some(64));
    }

    /// Gate: with a DFlash2 draft and no initial size the pool starts at the
    /// 2048-token runway plus the recurrent charge; without one, at the max.
    /// Mutation: ignore `has_dflash2_draft` → red (5051 ≠ 275).
    #[test]
    fn initial_pool_blocks_with_a_dflash2_draft_start_at_the_runway() {
        // Qwen3.8-27B: 16 full-attention layers × 4 KV heads × 256, 1 MiB
        // per 16-token BF16 block; GDN state 153,944,064 B = 147 blocks.
        let max_memory_mb = 5_051;
        let pa = paged_config(max_memory_mb, 256, 4, 16);
        let max_blocks = pa.calculate_num_blocks();
        assert_eq!(max_blocks, 5_051);
        let recurrent = 48 * (3 * 10_240 * 2 + 48 * 128 * 128 * 4);
        assert_eq!(recurrent, 153_944_064);
        let (blocks, mb) =
            qwen35_initial_pool_blocks(None, max_memory_mb, max_blocks, &pa, true, recurrent)
                .expect("runway");
        assert_eq!(blocks, 128 + 147);
        assert_eq!(mb, Some(275));
        let (blocks, mb) =
            qwen35_initial_pool_blocks(None, max_memory_mb, max_blocks, &pa, false, recurrent)
                .expect("no draft");
        assert_eq!(blocks, max_blocks);
        assert_eq!(mb, None);
        // The runway never exceeds the grow ceiling.
        assert_eq!(
            qwen35_initial_pool_blocks(None, max_memory_mb, 200, &pa, true, recurrent),
            Ok((200, Some(200)))
        );
    }

    /// Gate: on the 27B inputs a DFlash2 load starts at the 275-block runway
    /// whatever `MLX_PAGED_CACHE_INITIAL_MB` or the config field says, so its
    /// pool, and the window frozen from it, equal the unset load's. Without a
    /// draft the env still wins over the config field.
    /// Mutation: resolve the env/config initial first and set
    /// `runway = initial_mb.is_none() && has_dflash2_draft` → red
    /// (Some(4096) ≠ None).
    #[test]
    fn a_dflash2_load_ignores_the_initial_size_and_starts_at_the_runway() {
        let max_memory_mb = 5_051;
        let pa = paged_config(max_memory_mb, 256, 4, 16);
        let max_blocks = pa.calculate_num_blocks();
        let recurrent = 153_944_064;
        let unset = qwen35_initial_pool_request(None, None, true);
        assert_eq!(unset, (None, true));
        for (configured, env) in [
            (None, Some("4096")),
            (Some(300), None),
            (Some(300), Some("4096")),
        ] {
            let (initial, runway) = qwen35_initial_pool_request(configured, env, true);
            assert_eq!((initial, runway), unset);
            assert_eq!(
                qwen35_initial_pool_blocks(
                    initial,
                    max_memory_mb,
                    max_blocks,
                    &pa,
                    true,
                    recurrent
                ),
                Ok((275, Some(275)))
            );
        }
        assert_eq!(
            qwen35_initial_pool_request(Some(300), Some("4096"), false),
            (Some(4_096), false)
        );
        assert_eq!(
            qwen35_initial_pool_request(Some(300), None, false),
            (Some(300), false)
        );
        assert_eq!(
            qwen35_initial_pool_request(None, None, false),
            (None, false)
        );
    }

    /// Gate: on the M3 Pro inputs (a 5,051-block budget of one-MiB blocks,
    /// 16 pool layers, the 275-block runway, 128-block extents) the runway
    /// pool grows by extents all the way to its published max, 4,753 blocks,
    /// and the published pool window is 73,696. Each grow is admitted as
    /// `try_grow_pool` admits it: the sizer's selection for the target, with
    /// `pool_grow_reserved_bytes` reserved, must cover the target or at least
    /// one more block. Mutations: reserve the whole old pool in
    /// `pool_grow_reserved_bytes` → red (stalls at 2,579); publish the
    /// budget as the max in `qwen35_runway_pool_max_blocks` → red (stalls at
    /// 4,755 of 5,051).
    #[test]
    fn runway_pool_reaches_its_published_max() {
        const MIB: u64 = 1 << 20;
        const LAYERS: u32 = 16;
        const BUDGET: u32 = 5_051;
        const REQUESTED: u32 = 16_384;
        let probe = |request: u32, reserved: u64| {
            mlx_paged_attn::profile::compute_load_time_pool_sizing(
                request,
                u64::from(BUDGET) * MIB,
                None,
                0,
                1.0,
                0,
                MIB,
                reserved,
            )
            .expect("sizing")
            .selected_blocks
        };
        let sized = probe(qwen35_runway_pool_sizing_request(REQUESTED, LAYERS), 0);
        assert_eq!(sized, BUDGET);
        let max = qwen35_runway_pool_max_blocks(sized, REQUESTED, LAYERS);
        let extent = crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(MIB);
        assert_eq!(extent, 128);
        let mut blocks = 275u32;
        while blocks < max {
            let target = max.min(blocks + extent);
            let reserved = crate::transformer::paged_kv_cache_adapter::pool_grow_reserved_bytes(
                &vec![u64::from(blocks) * MIB / u64::from(LAYERS); LAYERS as usize],
                u64::from(target) * MIB / u64::from(LAYERS),
            );
            let selected = probe(target, reserved);
            if selected <= blocks {
                break;
            }
            blocks = selected.min(target);
        }
        assert_eq!(blocks, max, "the pool stalls below its published max");
        assert_eq!(max, 4_753);
        assert_eq!(
            crate::engine::hybrid_scheduler::pool_tokens_after_recurrent(
                max * 16,
                16,
                MIB,
                153_944_064,
            ),
            73_696
        );
        // Where the request binds, not the budget, the max is the request.
        assert_eq!(
            qwen35_runway_pool_max_blocks(17_408, REQUESTED, LAYERS),
            REQUESTED
        );
    }

    #[test]
    fn initial_pool_blocks_rejects_a_budget_that_cannot_hold_one_block() {
        let pa = paged_config(5_120, 256, 2, 10);
        assert!(
            qwen35_initial_pool_blocks(Some(0), 5_120, 16_384, &pa, true, 0).is_err(),
            "0 MiB must fail with the one-block error"
        );
    }
}
