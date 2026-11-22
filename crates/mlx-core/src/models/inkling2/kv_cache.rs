//! Grouped block-paged KV cache coordinator for Inkling-2-Mini.
//!
//! Inkling-2-Mini executes 24 physical layers across 48 logical visits:
//! - Group 0: **8 Global visits** (`5, 11, 17, 23, 29, 35, 41, 47`), `AttentionKind::Full`,
//!   `num_kv_heads = 8`, `head_size = 128`, `block_size = 32`.
//! - Group 1: **40 SWA visits**, `AttentionKind::SlidingWindow { sliding_window: 1024 }`,
//!   `num_kv_heads = 8`, `head_size = 128`, `block_size = 32`.
//!
//! Defaults to **INT8 32-token KTM** (`PageLayout { keys_token_major: true, .. }`,
//! `MetalDtype::Int8`) on Metal with `KvFormat::Bf16` opt-in fallback.
//! Expired SWA blocks beyond `sliding_window = 1024` are retired to a reserved null-block
//! sentinel while absolute logical positions stay intact for `tau(p)` and `rel_bias`.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{BlockAllocator, CacheRecency, LayerKVPool, PageLayout, PagedAttentionConfig};
use napi::bindgen_prelude::{Error, Result};

use crate::engine::backend::SpecFrontier;
use crate::engine::governor::governor::{Governor, ModelGuard, ModelRegistration};
use crate::engine::spec_paged::SpecPagedCache;
use crate::transformer::Activations;
use crate::transformer::kv_cache::KvFormat;
use crate::transformer::kv_cache_spec::{
    AttentionKind, KVCacheCoordinator, KVCacheDType, KVCacheGroup, KVCachePhysicalLayout,
    LayerKVCacheRoute, LayerKVCacheSpec, group_layer_kv_cache_specs,
};
use crate::transformer::paged_kv_cache_adapter::{ColdTierContext, PagedKVCacheAdapter};

use super::config::Inkling2Config;
use super::kernels::Inkling2KvDtype;
use super::memory::Inkling2PoolPlan;

/// Default page size (tokens per block) for Inkling-2-Mini.
/// INT8 KTM requires `block_size = 32`; BF16 supports `16` or `32` (default `32`).
pub const INKLING2_PAGED_BLOCK_SIZE: u32 = 32;

/// Maximum prefill chunk size (rows per in-place MPP chunk).
pub const INKLING2_PAGED_PREFILL_MAX_CHUNK: u32 = 2048;

/// Default logical context window budgeted for paged KV pools (grows/configures up to
/// `config.max_position_embeddings`). 32K tokens is ~786 MiB in INT8 KTM across both groups.
pub const INKLING2_DEFAULT_MAX_MODEL_LEN: u32 = 32_768;

/// Default concurrent sequence lanes sized into the grouped pools.
pub const INKLING2_DEFAULT_MAX_CONCURRENT_SEQS: u32 = 4;

const BYTES_PER_MIB: u64 = 1024 * 1024;
const MIN_POOL_MEMORY_MB: u64 = mlx_paged_attn::MIN_GPU_MEMORY_MB as u64;

/// Default KV format for Inkling-2-Mini: **INT8 32-token KTM** on Metal, BF16 elsewhere.
pub fn default_inkling2_kv_format() -> KvFormat {
    let metal_gpu = unsafe { mlx_sys::mlx_metal_is_available() }
        && unsafe { mlx_sys::mlx_default_device() } == 1;
    if metal_gpu {
        KvFormat::Int8
    } else {
        KvFormat::Bf16
    }
}

/// Configuration for [`Inkling2KVCacheCoordinator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2PagedConfig {
    pub kv_format: KvFormat,
    pub block_size: u32,
    pub max_model_len: u32,
    pub max_chunk: u32,
    pub max_concurrent_sequences: u32,
    pub gpu_memory_mb: Option<u32>,
}

impl Inkling2PagedConfig {
    pub fn default_for_config(config: &Inkling2Config) -> Self {
        let model_cap = u32::try_from(config.max_position_embeddings).unwrap_or(u32::MAX);
        Self {
            kv_format: default_inkling2_kv_format(),
            block_size: INKLING2_PAGED_BLOCK_SIZE,
            max_model_len: model_cap.clamp(1, INKLING2_DEFAULT_MAX_MODEL_LEN),
            max_chunk: INKLING2_PAGED_PREFILL_MAX_CHUNK,
            max_concurrent_sequences: INKLING2_DEFAULT_MAX_CONCURRENT_SEQS,
            gpu_memory_mb: None,
        }
    }

    pub fn with_kv_format(mut self, kv_format: KvFormat) -> Self {
        self.kv_format = kv_format;
        self
    }

    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn with_max_model_len(mut self, max_model_len: u32) -> Self {
        self.max_model_len = max_model_len;
        self
    }

    pub fn with_max_chunk(mut self, max_chunk: u32) -> Self {
        self.max_chunk = max_chunk;
        self
    }

    pub fn with_max_concurrent_sequences(mut self, max_concurrent_sequences: u32) -> Self {
        self.max_concurrent_sequences = max_concurrent_sequences.max(1);
        self
    }

    pub fn with_gpu_memory_mb(mut self, gpu_memory_mb: u32) -> Self {
        self.gpu_memory_mb = Some(gpu_memory_mb);
        self
    }
}

/// Convert [`KvFormat`] to [`KVCacheDType`] for model-independent KV grouping specs.
pub fn kv_format_to_spec_dtype(kv_format: KvFormat) -> KVCacheDType {
    match kv_format {
        KvFormat::Int8 => KVCacheDType::Int8,
        KvFormat::Bf16 => KVCacheDType::BFloat16,
    }
}

/// Convert [`KvFormat`] to [`MetalDtype`], [`PageLayout`], and [`Inkling2KvDtype`].
pub fn kv_format_metal_layout(kv_format: KvFormat) -> (MetalDtype, PageLayout, Inkling2KvDtype) {
    match kv_format {
        KvFormat::Int8 => (
            MetalDtype::Int8,
            PageLayout {
                keys_token_major: true,
                ..PageLayout::default()
            },
            Inkling2KvDtype::Int8,
        ),
        KvFormat::Bf16 => (
            MetalDtype::BFloat16,
            PageLayout::default(),
            Inkling2KvDtype::Bf16,
        ),
    }
}

/// Build per-visit [`LayerKVCacheSpec`]s for all logical visits (`0..config.num_layer_visits()`).
pub fn compute_visit_kv_cache_specs(
    config: &Inkling2Config,
    block_size: u32,
    kv_format: KvFormat,
) -> std::result::Result<Vec<LayerKVCacheSpec>, String> {
    if block_size == 0 {
        return Err("Inkling2 KV cache block_size must be > 0".to_string());
    }
    if kv_format == KvFormat::Int8 && block_size != 32 {
        return Err(format!(
            "Inkling2 INT8 KTM KV cache requires block_size=32, got {block_size}"
        ));
    }
    if kv_format == KvFormat::Bf16 && !matches!(block_size, 16 | 32) {
        return Err(format!(
            "Inkling2 BF16 KV cache requires block_size in {{16, 32}}, got {block_size}"
        ));
    }
    let num_kv_heads = u32::try_from(config.num_key_value_heads)
        .map_err(|_| format!("invalid num_key_value_heads={}", config.num_key_value_heads))?;
    let head_size = u32::try_from(config.head_dim)
        .map_err(|_| format!("invalid head_dim={}", config.head_dim))?;
    let sliding_window = u32::try_from(config.sliding_window)
        .map_err(|_| format!("invalid sliding_window={}", config.sliding_window))?;
    let layout = KVCachePhysicalLayout::new(
        block_size,
        num_kv_heads,
        head_size,
        kv_format_to_spec_dtype(kv_format),
    );

    let visits = config.visits();
    let mut specs = Vec::with_capacity(visits.len());
    for v in visits {
        let spec = if v.is_local {
            LayerKVCacheSpec::sliding_window(v.visit_idx, sliding_window, layout)
        } else {
            LayerKVCacheSpec::full(v.visit_idx, layout)
        };
        specs.push(spec);
    }
    Ok(specs)
}

/// Group all visit specs into deterministic physical KV cache groups:
/// - Group 0: `AttentionKind::Full` (8 global visits: `5, 11, 17, 23, 29, 35, 41, 47`)
/// - Group 1: `AttentionKind::SlidingWindow { sliding_window: 1024 }` (40 SWA visits)
pub fn compute_visit_kv_cache_groups(
    config: &Inkling2Config,
    block_size: u32,
    kv_format: KvFormat,
    max_model_len: u32,
    max_chunk: u32,
) -> std::result::Result<(Vec<LayerKVCacheSpec>, Vec<KVCacheGroup>), String> {
    let specs = compute_visit_kv_cache_specs(config, block_size, kv_format)?;
    let groups = group_layer_kv_cache_specs(&specs, max_model_len, max_chunk)
        .map_err(|e| format!("Inkling2 KV cache grouping failed: {e}"))?;
    Ok((specs, groups))
}

/// Per-group runtime manager owned by [`Inkling2KVCacheCoordinator`].
enum Inkling2KVCacheGroupManager {
    /// Marker aligned with [`Inkling2KVCacheCoordinator::full_adapter`].
    Full,
    SlidingWindow {
        sliding_window: u32,
        max_admission_blocks: u32,
        adapter: Box<PagedKVCacheAdapter>,
    },
}

/// Hybrid grouped block-paged KV cache coordinator for Inkling-2-Mini (8 Global + 40 SWA visits).
///
/// Implements [`SpecPagedCache`] directly so Phase 5 DSpark verify/commit/rollback works
/// atomically across both the Global and SWA cache groups.
pub struct Inkling2KVCacheCoordinator {
    inner: KVCacheCoordinator<Inkling2KVCacheGroupManager>,
    full_group_id: usize,
    full_adapter: Box<PagedKVCacheAdapter>,
    sliding_group_id: Option<usize>,
    kv_format: KvFormat,
    kv_dtype: Inkling2KvDtype,
    block_size: u32,
    max_concurrent_sequences: u32,
}

impl Inkling2KVCacheCoordinator {
    /// Construct a new coordinator with physical Metal pools sized for `config` and `paged_cfg`.
    pub fn new_for_config(config: &Inkling2Config, paged_cfg: Inkling2PagedConfig) -> Result<Self> {
        Self::new_for_config_with_recency(config, paged_cfg, Arc::new(CacheRecency::new()))
    }

    /// Construct a new coordinator sharing `cache_recency` with Group 0's `BlockAllocator`.
    pub(crate) fn new_for_config_with_recency(
        config: &Inkling2Config,
        paged_cfg: Inkling2PagedConfig,
        cache_recency: Arc<CacheRecency>,
    ) -> Result<Self> {
        let block_size = paged_cfg.block_size;
        let max_model_len = paged_cfg.max_model_len.max(1);
        let max_chunk = paged_cfg.max_chunk.max(1);
        let (specs, groups) = compute_visit_kv_cache_groups(
            config,
            block_size,
            paged_cfg.kv_format,
            max_model_len,
            max_chunk,
        )
        .map_err(Error::from_reason)?;

        let (metal_dtype, page_layout, kv_dtype) = kv_format_metal_layout(paged_cfg.kv_format);

        // Compute exact per-block byte cost for each group (including INT8 scale planes).
        let mut group_bytes_per_block = Vec::with_capacity(groups.len());
        for group in &groups {
            let num_layers = u32::try_from(group.physical_layer_indices.len())
                .map_err(|_| Error::from_reason("Inkling2 KV group layer count overflows u32"))?;
            let bpb = mlx_paged_attn::profile::bytes_per_block(
                num_layers,
                group.physical_layout.num_kv_heads,
                group.physical_layout.head_size,
                block_size,
                metal_dtype,
            )
            .map_err(|e| Error::from_reason(format!("Inkling2 bytes_per_block failed: {e}")))?;
            group_bytes_per_block.push(bpb);
        }

        // Minimum blocks needed for one full-context sequence + reserved null block on SWA.
        let one_seq_bytes =
            groups
                .iter()
                .zip(&group_bytes_per_block)
                .fold(0u64, |acc, (group, &bpb)| {
                    let extra_null = u64::from(matches!(
                        group.attention_kind,
                        AttentionKind::SlidingWindow { .. }
                    ));
                    let blocks = u64::from(group.max_admission_blocks).saturating_add(extra_null);
                    acc.saturating_add(blocks.saturating_mul(bpb))
                });

        let required_bytes_for_width = |width: u32| -> u64 {
            groups
                .iter()
                .zip(&group_bytes_per_block)
                .fold(0u64, |acc, (group, &bpb)| {
                    let blocks = group_reserved_blocks(
                        group.attention_kind,
                        group.max_admission_blocks,
                        width,
                        block_size,
                    );
                    acc.saturating_add(u64::from(blocks).saturating_mul(bpb))
                })
        };

        let mut max_concurrent_sequences = paged_cfg.max_concurrent_sequences.clamp(1, 32);
        let memory_bytes = if let Some(mb) = paged_cfg.gpu_memory_mb {
            let bytes = u64::from(mb).saturating_mul(BYTES_PER_MIB);
            if bytes < one_seq_bytes {
                return Err(Error::from_reason(format!(
                    "Inkling2 KV cache: gpu_memory_mb={mb} is smaller than one sequence's \
                     coordinated Global+SWA requirement ({} MiB)",
                    one_seq_bytes.div_ceil(BYTES_PER_MIB)
                )));
            }
            bytes
        } else {
            required_bytes_for_width(max_concurrent_sequences).max(one_seq_bytes)
        };

        while max_concurrent_sequences > 1
            && required_bytes_for_width(max_concurrent_sequences) > memory_bytes
        {
            max_concurrent_sequences -= 1;
        }
        let reserved_bytes = required_bytes_for_width(max_concurrent_sequences);
        let mut unassigned_bytes = memory_bytes.saturating_sub(reserved_bytes);

        let mut adapters = Vec::with_capacity(groups.len());
        for (group, &bpb) in groups.iter().zip(&group_bytes_per_block) {
            let mut desired_blocks = group_reserved_blocks(
                group.attention_kind,
                group.max_admission_blocks,
                max_concurrent_sequences,
                block_size,
            );
            if matches!(group.attention_kind, AttentionKind::Full) && bpb > 0 {
                let extra = u32::try_from(unassigned_bytes / bpb).unwrap_or(u32::MAX);
                desired_blocks = desired_blocks.saturating_add(extra);
                unassigned_bytes =
                    unassigned_bytes.saturating_sub(u64::from(extra).saturating_mul(bpb));
            }
            let group_memory_mb = bpb
                .saturating_mul(u64::from(desired_blocks))
                .div_ceil(BYTES_PER_MIB)
                .max(MIN_POOL_MEMORY_MB);
            let group_memory_mb = u32::try_from(group_memory_mb).unwrap_or(u32::MAX);
            let num_layers = u32::try_from(group.physical_layer_indices.len())
                .map_err(|_| Error::from_reason("Inkling2 KV group layer count overflows u32"))?;
            let pa_config = PagedAttentionConfig {
                block_size,
                gpu_memory_mb: group_memory_mb,
                head_size: group.physical_layout.head_size,
                num_kv_heads: group.physical_layout.num_kv_heads,
                num_layers,
                max_seq_len: Some(max_model_len),
                max_batch_size: Some(max_concurrent_sequences),
            };
            let allocator =
                match group.attention_kind {
                    AttentionKind::Full => Arc::new(Mutex::new(BlockAllocator::with_recency(
                        desired_blocks,
                        desired_blocks,
                        block_size,
                        Arc::clone(&cache_recency),
                    ))),
                    AttentionKind::SlidingWindow { .. } => Arc::new(Mutex::new(
                        BlockAllocator::new(desired_blocks, desired_blocks, block_size),
                    )),
                };
            let pool = LayerKVPool::new_with_layout(
                pa_config,
                desired_blocks,
                desired_blocks,
                metal_dtype,
                page_layout,
            )
            .map_err(|e| {
                Error::from_reason(format!(
                    "Failed to construct Inkling2 KV group {} pool: {e}",
                    group.group_id
                ))
            })?;
            let adapter = match group.attention_kind {
                AttentionKind::Full => {
                    PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                }
                AttentionKind::SlidingWindow { sliding_window } => {
                    PagedKVCacheAdapter::new_sliding(
                        allocator,
                        Arc::new(pool),
                        block_size,
                        sliding_window,
                        max_model_len,
                    )
                }
            }
            .map_err(|e| {
                Error::from_reason(format!(
                    "Failed to construct Inkling2 KV group {} adapter: {e}",
                    group.group_id
                ))
            })?;
            adapters.push(adapter);
        }

        Self::from_adapters(
            &specs,
            groups,
            adapters,
            paged_cfg.kv_format,
            kv_dtype,
            block_size,
            max_concurrent_sequences,
        )
        .map_err(Error::from_reason)
    }

    /// Construct a new coordinator sized from a unified [`Inkling2PoolPlan`], registering
    /// the Global extent-grown pool with [`Governor`] (`load_pool`), sharing `cache_recency`
    /// with Group 0's `BlockAllocator`, configuring extent growth + memory ceiling + activations,
    /// and holding both Group 0 and Group 1 Metal pools resident (`PoolResidency`).
    pub(crate) fn new_with_memory_plan(
        config: &Inkling2Config,
        plan: &Inkling2PoolPlan,
        model_id: u64,
        governor: &'static Governor,
        cache_recency: Arc<CacheRecency>,
        activations: Activations,
    ) -> Result<(ModelGuard<'static>, Self)> {
        let block_size = plan.block_size;
        let max_model_len = plan.effective_window_tokens.max(block_size);
        let max_chunk = plan.max_chunk.max(1);
        let (specs, groups) = compute_visit_kv_cache_groups(
            config,
            block_size,
            plan.kv_format,
            max_model_len,
            max_chunk,
        )
        .map_err(Error::from_reason)?;

        let (metal_dtype, page_layout, kv_dtype) = kv_format_metal_layout(plan.kv_format);
        let runway_bytes = u64::from(plan.global_runway_blocks)
            .saturating_mul(plan.global_bytes_per_block)
            .saturating_add(
                u64::from(plan.swa_runway_blocks).saturating_mul(plan.swa_bytes_per_block),
            );
        let registration = ModelRegistration {
            model_id,
            hard_budget: plan.memory_plan.hard_budget_bytes,
            untracked: plan.memory_plan.untracked_reserve_bytes(),
            page_bytes: plan.global_bytes_per_block,
        };
        let ceiling_bytes = plan.memory_plan.ceiling_bytes();
        let recency_clone = Arc::clone(&cache_recency);

        let (guard, mut coordinator) = governor
            .load_pool(registration, runway_bytes, || {
                let mut adapters = Vec::with_capacity(groups.len());
                for group in &groups {
                    let num_layers = u32::try_from(group.physical_layer_indices.len())
                        .map_err(|_| "Inkling2 KV group layer count overflows u32".to_string())?;
                    let (initial_blocks, max_blocks, bpb, extent_blocks) =
                        match group.attention_kind {
                            AttentionKind::Full => (
                                plan.global_runway_blocks,
                                plan.global_max_blocks,
                                plan.global_bytes_per_block,
                                plan.global_extent_blocks,
                            ),
                            AttentionKind::SlidingWindow { .. } => (
                                plan.swa_runway_blocks,
                                plan.swa_max_blocks,
                                plan.swa_bytes_per_block,
                                plan.swa_extent_blocks,
                            ),
                        };
                    let group_memory_mb = bpb
                        .saturating_mul(u64::from(max_blocks))
                        .div_ceil(BYTES_PER_MIB)
                        .max(MIN_POOL_MEMORY_MB);
                    let group_memory_mb = u32::try_from(group_memory_mb).unwrap_or(u32::MAX);
                    let pa_config = PagedAttentionConfig {
                        block_size,
                        gpu_memory_mb: group_memory_mb,
                        head_size: group.physical_layout.head_size,
                        num_kv_heads: group.physical_layout.num_kv_heads,
                        num_layers,
                        max_seq_len: Some(max_model_len),
                        max_batch_size: Some(plan.lanes.max(1)),
                    };
                    let allocator = match group.attention_kind {
                        AttentionKind::Full => Arc::new(Mutex::new(BlockAllocator::with_recency(
                            initial_blocks,
                            max_blocks,
                            block_size,
                            Arc::clone(&recency_clone),
                        ))),
                        AttentionKind::SlidingWindow { .. } => Arc::new(Mutex::new(
                            BlockAllocator::new(initial_blocks, max_blocks, block_size),
                        )),
                    };
                    let pool = LayerKVPool::new_with_layout(
                        pa_config,
                        initial_blocks,
                        max_blocks,
                        metal_dtype,
                        page_layout,
                    )
                    .map_err(|e| {
                        format!(
                            "Failed to construct Inkling2 KV group {} pool: {e}",
                            group.group_id
                        )
                    })?;
                    let mut adapter = match group.attention_kind {
                        AttentionKind::Full => {
                            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                        }
                        AttentionKind::SlidingWindow { sliding_window } => {
                            PagedKVCacheAdapter::new_sliding(
                                allocator,
                                Arc::new(pool),
                                block_size,
                                sliding_window,
                                max_model_len,
                            )
                        }
                    }
                    .map_err(|e| {
                        format!(
                            "Failed to construct Inkling2 KV group {} adapter: {e}",
                            group.group_id
                        )
                    })?;
                    adapter.set_admission(Arc::new(governor));
                    if initial_blocks < max_blocks {
                        adapter.set_extent_growth(extent_blocks.max(1));
                    }
                    adapter.set_memory_ceiling(ceiling_bytes);
                    adapter.set_activations(activations);
                    adapters.push(adapter);
                }

                Self::from_adapters(
                    &specs,
                    groups,
                    adapters,
                    plan.kv_format,
                    kv_dtype,
                    block_size,
                    plan.lanes.max(1),
                )
            })
            .map_err(Error::from_reason)?;

        coordinator.hold_resident_all()?;
        Ok((guard, coordinator))
    }

    /// Hold every KV group's Metal pool resident under the global residency clock (`--idle-release`).
    pub fn hold_resident_all(&mut self) -> Result<()> {
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter(group_id).map_err(Error::from_reason)?;
            adapter
                .layer_kv_pool()
                .hold_resident()
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }

    /// Host-transfer `(key_blob, value_blob)` byte lengths of one SWA block in one layer.
    pub fn sliding_block_blob_bytes(&self) -> std::result::Result<Option<(u64, u64)>, String> {
        let Some(adapter) = self.sliding_adapter() else {
            return Ok(None);
        };
        adapter.layer_kv_pool().block_blob_bytes().map(Some)
    }

    /// Read raw physical SWA blocks covering the live window ending at block-aligned `boundary`.
    pub fn read_sliding_raw_blocks_for(
        &self,
        seq_id: u32,
        boundary: u32,
    ) -> std::result::Result<Vec<Vec<(Vec<u8>, Vec<u8>)>>, String> {
        let adapter = self
            .sliding_adapter()
            .ok_or_else(|| "Inkling2 KV coordinator has no sliding group".to_string())?;
        adapter.read_sliding_raw_blocks_for(seq_id, boundary)
    }

    /// Restore raw physical SWA blocks into Group 1 at block-aligned `boundary`.
    pub fn restore_sliding_raw_blocks_for(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        boundary: u32,
        raw_blocks: &[Vec<(Vec<u8>, Vec<u8>)>],
    ) -> std::result::Result<(), String> {
        let adapter = self
            .sliding_adapter_mut()
            .ok_or_else(|| "Inkling2 KV coordinator has no sliding group".to_string())?;
        adapter.restore_sliding_raw_blocks_for(seq_id, prompt_tokens, boundary, raw_blocks)
    }

    /// Register Group 0 (8 Global visits) committed blocks up to `boundary` in the prefix cache.
    pub fn register_full_committed_blocks(
        &mut self,
        seq_id: u32,
        boundary: u32,
        cache_salt: u64,
    ) -> std::result::Result<(), String> {
        self.full_adapter.activate_request(seq_id)?;
        let keys = crate::engine::build_paged_extra_keys(
            boundary as usize,
            self.full_adapter.block_size(),
            &[],
        );
        self.full_adapter
            .register_committed_blocks(boundary as usize, &keys, cache_salt)
    }

    /// Register Group 0 full blocks for reuse during preemption/finalization.
    pub fn register_full_blocks_for_reuse_for(
        &mut self,
        seq_id: u32,
        cache_salt: u64,
        capture_cold: bool,
    ) -> std::result::Result<u32, String> {
        self.full_adapter
            .register_full_blocks_for_reuse_for(seq_id, &[], cache_salt, capture_cold)
    }

    /// Update Group 0's pinned state-mark block hashes (blocks referenced by StateStore entries).
    pub fn set_full_state_marks(&self, hashes: impl IntoIterator<Item = u64>) {
        self.full_adapter.set_state_marks(hashes);
    }

    /// Install a `ColdTierContext` (`ColdContextBacking::StateOnly`) on Group 0.
    pub fn set_full_cold_tier(&mut self, context: ColdTierContext) {
        self.full_adapter.set_cold_tier(context);
    }

    /// Remove and return Group 0's `ColdTierContext` (if any).
    pub fn take_full_cold_tier(&mut self) -> Option<ColdTierContext> {
        self.full_adapter.take_cold_tier()
    }

    fn from_adapters(
        specs: &[LayerKVCacheSpec],
        groups: Vec<KVCacheGroup>,
        adapters: Vec<PagedKVCacheAdapter>,
        kv_format: KvFormat,
        kv_dtype: Inkling2KvDtype,
        block_size: u32,
        max_concurrent_sequences: u32,
    ) -> std::result::Result<Self, String> {
        if groups.len() != adapters.len() {
            return Err(format!(
                "Inkling2 KV group count {} does not match adapter count {}",
                groups.len(),
                adapters.len()
            ));
        }
        let mut full_group = None;
        let mut sliding_group_id = None;
        let mut managers = Vec::with_capacity(groups.len());
        for (group, adapter) in groups.iter().zip(adapters) {
            let manager = match group.attention_kind {
                AttentionKind::Full => {
                    if full_group
                        .replace((group.group_id, Box::new(adapter)))
                        .is_some()
                    {
                        return Err(
                            "Inkling2 KV coordinator supports one full-attention KV group"
                                .to_string(),
                        );
                    }
                    Inkling2KVCacheGroupManager::Full
                }
                AttentionKind::SlidingWindow { sliding_window } => {
                    sliding_group_id.get_or_insert(group.group_id);
                    Inkling2KVCacheGroupManager::SlidingWindow {
                        sliding_window,
                        max_admission_blocks: group.max_admission_blocks,
                        adapter: Box::new(adapter),
                    }
                }
            };
            managers.push(manager);
        }
        let (full_group_id, full_adapter) = full_group
            .ok_or_else(|| "Inkling2 KV coordinator has no full-attention KV group".to_string())?;
        let inner =
            KVCacheCoordinator::from_groups(specs, groups, managers).map_err(|e| e.to_string())?;
        Ok(Self {
            inner,
            full_group_id,
            full_adapter,
            sliding_group_id,
            kv_format,
            kv_dtype,
            block_size,
            max_concurrent_sequences,
        })
    }

    pub fn kv_format(&self) -> KvFormat {
        self.kv_format
    }

    pub fn kv_dtype(&self) -> Inkling2KvDtype {
        self.kv_dtype
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn max_concurrent_sequences(&self) -> u32 {
        self.max_concurrent_sequences
    }

    pub fn groups(&self) -> &[KVCacheGroup] {
        self.inner.groups()
    }

    pub fn routes(&self) -> &[LayerKVCacheRoute] {
        self.inner.routes()
    }

    pub fn route_for_visit(&self, visit_idx: usize) -> Option<&LayerKVCacheRoute> {
        self.inner.route_for_layer(visit_idx)
    }

    pub fn full_group_id(&self) -> usize {
        self.full_group_id
    }

    pub fn sliding_group_id(&self) -> Option<usize> {
        self.sliding_group_id
    }

    pub fn full_adapter(&self) -> &PagedKVCacheAdapter {
        &self.full_adapter
    }

    pub fn full_adapter_mut(&mut self) -> &mut PagedKVCacheAdapter {
        &mut self.full_adapter
    }

    pub fn sliding_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        let gid = self.sliding_group_id?;
        self.adapter(gid).ok()
    }

    pub fn sliding_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        let gid = self.sliding_group_id?;
        self.adapter_mut(gid).ok()
    }

    pub fn adapter(&self, group_id: usize) -> std::result::Result<&PagedKVCacheAdapter, String> {
        if group_id == self.full_group_id {
            return Ok(&self.full_adapter);
        }
        match self.inner.manager(group_id) {
            Some(Inkling2KVCacheGroupManager::SlidingWindow { adapter, .. }) => Ok(adapter),
            Some(Inkling2KVCacheGroupManager::Full) => Err(format!(
                "Inkling2 KV group {group_id} is marked full but owned full group is {}",
                self.full_group_id
            )),
            None => Err(format!("Inkling2 KV group {group_id} is missing")),
        }
    }

    pub fn adapter_mut(
        &mut self,
        group_id: usize,
    ) -> std::result::Result<&mut PagedKVCacheAdapter, String> {
        if group_id == self.full_group_id {
            return Ok(&mut self.full_adapter);
        }
        match self.inner.manager_mut(group_id) {
            Some(Inkling2KVCacheGroupManager::SlidingWindow { adapter, .. }) => Ok(adapter),
            Some(Inkling2KVCacheGroupManager::Full) => Err(format!(
                "Inkling2 KV group {group_id} is marked full but owned full group is {}",
                self.full_group_id
            )),
            None => Err(format!("Inkling2 KV group {group_id} is missing")),
        }
    }

    pub fn sliding_capacity_summary(&self) -> (usize, u32, u32) {
        self.inner
            .groups()
            .iter()
            .filter_map(|group| match self.inner.manager(group.group_id) {
                Some(Inkling2KVCacheGroupManager::SlidingWindow {
                    sliding_window,
                    max_admission_blocks,
                    ..
                }) => Some((1usize, *sliding_window, *max_admission_blocks)),
                _ => None,
            })
            .fold((0, 0, 0), |acc, item| {
                (acc.0 + item.0, acc.1.max(item.1), acc.2.max(item.2))
            })
    }

    pub fn pool_allocated_bytes(&self) -> std::result::Result<u64, String> {
        (0..self.inner.groups().len()).try_fold(0u64, |total, group_id| {
            self.adapter(group_id)
                .and_then(PagedKVCacheAdapter::pool_allocated_bytes)
                .map(|bytes| total.saturating_add(bytes))
        })
    }

    /// Reset `seq_id` cleanly across every KV group and begin a fresh empty request.
    pub fn reset_scheduled_request(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        let _ = self.release_request_all(seq_id);
        for group_id in 0..self.inner.groups().len() {
            if let Err(error) = self.adapter_mut(group_id)?.begin_request(seq_id) {
                let _ = self.release_request_all(seq_id);
                return Err(format!(
                    "Inkling2 KV group {group_id} request reset failed: {error}"
                ));
            }
        }
        Ok(())
    }

    pub fn activate_request_all(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter_mut(group_id)?.activate_request(seq_id)?;
        }
        Ok(())
    }

    pub fn can_continue_all(&self, seq_id: u32, prompt_tokens: &[u32]) -> bool {
        (0..self.inner.groups().len()).all(|group_id| {
            let Ok(adapter) = self.adapter(group_id) else {
                return false;
            };
            adapter.is_live_for_continue_for(seq_id)
                && adapter.request_tokens_for(seq_id).is_some_and(|live| {
                    prompt_tokens.len() > live.len() && prompt_tokens.starts_with(live)
                })
        })
    }

    pub fn is_live_all(&self, seq_id: u32) -> bool {
        (0..self.inner.groups().len()).all(|group_id| {
            self.adapter(group_id)
                .is_ok_and(|adapter| adapter.is_live_for_continue_for(seq_id))
        })
    }

    pub fn request_token_count_all(&self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut count = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter(group_id)?;
            let group_count = adapter
                .request_tokens_for(seq_id)
                .map(|tokens| tokens.len() as u32)
                .ok_or_else(|| format!("Inkling2 KV group {group_id} has no sequence {seq_id}"))?;
            if count
                .replace(group_count)
                .is_some_and(|prior| prior != group_count)
            {
                return Err(format!(
                    "Inkling2 hybrid KV groups disagree on sequence {seq_id} token count"
                ));
            }
        }
        count.ok_or_else(|| "Inkling2 hybrid KV coordinator has no groups".to_string())
    }

    pub fn continue_turn_all(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
    ) -> std::result::Result<u32, String> {
        let mut boundary = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            let (group_boundary, _) = adapter.continue_turn(prompt_tokens, total_budget)?;
            if boundary
                .replace(group_boundary)
                .is_some_and(|prior| prior != group_boundary)
            {
                return Err(
                    "Inkling2 hybrid KV groups disagree on live continuation boundary".to_string(),
                );
            }
        }
        Ok(boundary.unwrap_or(0))
    }

    pub fn check_wave_fits_all(&self, rows: &[(u32, u32)]) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter(group_id)?.check_wave_fits(rows)?;
        }
        Ok(())
    }

    /// Atomically record `tokens` across every KV group, rolling back earlier groups on error.
    pub fn record_tokens_all(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
    ) -> std::result::Result<(), String> {
        let mut recorded = Vec::with_capacity(self.inner.groups().len());
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            if let Err(error) = adapter.record_tokens_for(seq_id, tokens) {
                let rollback_n = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
                for recorded_group in recorded.into_iter().rev() {
                    if let Ok(prev) = self.adapter_mut(recorded_group)
                        && prev.activate_request(seq_id).is_ok()
                    {
                        let _ = prev.rollback_last_tokens(rollback_n);
                    }
                }
                return Err(error);
            }
            recorded.push(group_id);
        }
        Ok(())
    }

    /// Patch the trailing `tokens.len()` placeholder (`u32::MAX`) entries across every KV group
    /// with their real verified token IDs (used when `verify_device` books KV slots before
    /// device-resident draft IDs are materialized on the host).
    pub fn patch_last_recorded_tokens_all(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            adapter.patch_last_recorded_tokens(tokens)?;
        }
        Ok(())
    }

    /// Reserve speculative lookahead rows atomically across every KV group (all or none).
    pub fn reserve_rows_all(&mut self, seq_id: u32, rows: u32) -> std::result::Result<u32, String> {
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            let current_tokens = adapter
                .request_tokens_for(seq_id)
                .map(|tokens| tokens.len() as u32)
                .ok_or_else(|| {
                    format!("Inkling2 KV group {group_id} has no sequence {seq_id} to reserve for")
                })?;
            let new_total = current_tokens.checked_add(rows).ok_or_else(|| {
                format!(
                    "Inkling2 KV group {group_id} reservation overflows cursor \
                     (current={current_tokens}, rows={rows})"
                )
            })?;
            let capacity = adapter.max_capacity_tokens();
            if new_total > capacity {
                return Err(format!(
                    "context_length_exceeded: Inkling2 KV group {group_id} cannot reserve \
                     {rows} lookahead row(s) at cursor {current_tokens} (capacity {capacity})"
                ));
            }
            let block_size = adapter.block_size();
            if block_size == 0 {
                return Err(format!("Inkling2 KV group {group_id} has block_size 0"));
            }
            let current_blocks = adapter
                .block_table_for(seq_id)
                .map(|table| table.num_blocks() as u32)
                .ok_or_else(|| {
                    format!("Inkling2 KV group {group_id} has no block table for {seq_id}")
                })?;
            let needed_blocks = new_total
                .div_ceil(block_size)
                .saturating_sub(current_blocks);
            let telemetry = adapter.block_telemetry()?;
            let available = telemetry
                .free_blocks
                .saturating_add(telemetry.evictable_blocks);
            if needed_blocks > available {
                return Err(format!(
                    "context_length_exceeded: Inkling2 KV group {group_id} needs \
                     {needed_blocks} block(s) for {rows} lookahead row(s) but only \
                     {available} are free or reclaimable"
                ));
            }
        }
        let mut new_blocks = 0u32;
        for group_id in 0..self.inner.groups().len() {
            let reserved = self
                .adapter_mut(group_id)?
                .reserve_rows_for(seq_id, rows)
                .map_err(|error| {
                    format!(
                        "Inkling2 KV group {group_id} reservation failed after pre-flight: {error}"
                    )
                })?;
            new_blocks = new_blocks.saturating_add(reserved);
        }
        Ok(new_blocks)
    }

    pub fn eval_pending_pool_writes_all(&mut self) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter_mut(group_id)?.eval_pending_pool_writes()?;
        }
        Ok(())
    }

    /// Prune expired SWA blocks beyond `sliding_window = 1024` (replaces with null sentinel while
    /// keeping absolute logical positions intact for `tau(p)` and `rel_bias`).
    pub fn prune_sliding_all(&mut self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        for group_id in 0..self.inner.groups().len() {
            released = released.saturating_add(
                self.adapter_mut(group_id)?
                    .prune_sliding_window_for(seq_id)?,
            );
        }
        Ok(released)
    }

    /// Committed-frontier SWA prune: never retires any block that a speculative rollback may
    /// rewind back into.
    pub fn prune_sliding_all_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u32,
    ) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        for group_id in 0..self.inner.groups().len() {
            released = released.saturating_add(
                self.adapter_mut(group_id)?
                    .prune_sliding_window_for_committed(seq_id, committed_tokens)?,
            );
        }
        Ok(released)
    }

    pub fn rollback_last_tokens_all(
        &mut self,
        seq_id: u32,
        token_count: u32,
    ) -> std::result::Result<(), String> {
        let mut first_error = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            let res = adapter
                .activate_request(seq_id)
                .and_then(|_| adapter.rollback_last_tokens(token_count));
            if let Err(error) = res {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub fn finalize_keep_live_all(
        &mut self,
        seq_id: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            let is_full = group_id == self.full_group_id;
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            if is_full {
                if extra_keys_per_block.is_empty() {
                    adapter.finalize_turn_keep_live(&[], cache_salt)?;
                } else {
                    adapter.finalize_turn_keep_live_per_block(extra_keys_per_block, cache_salt)?;
                }
                // A short turn (< block_size committed tokens) has no full block to publish
                // into the prefix cache, so `register_full_blocks_for_reuse*` returns `Ok(0)`
                // without marking `already_registered = true`. Mark the partial-block request
                // live independently of prefix publication so `chatSessionContinue` succeeds.
                if !adapter.is_live_for_continue() {
                    adapter.finalize_turn_keep_live_no_prefix()?;
                }
            } else {
                adapter.finalize_turn_keep_live_no_prefix()?;
            }
        }
        Ok(())
    }

    pub fn release_request_all(&mut self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        let mut first_error = None;
        for group_id in 0..self.inner.groups().len() {
            match self.adapter_mut(group_id)?.release_request_for(seq_id) {
                Ok(count) => released = released.saturating_add(count),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(released), Err)
    }

    pub fn release_all_and_purge(&mut self) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        let mut first_error = None;
        for group_id in 0..self.inner.groups().len() {
            match self
                .adapter_mut(group_id)?
                .release_request_and_purge_prefix_cache()
            {
                Ok(count) => released = released.saturating_add(count),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(released), Err)
    }

    /// Test helper: simulate hot-RAM eviction under memory pressure (releasing live requests and
    /// purging hot prefix blocks WITHOUT arming `suppress_cold_restore_once`, which is reserved
    /// for user-triggered `reset_caches()`).
    #[cfg(test)]
    pub fn evict_all_hot_blocks_without_suppress(&mut self) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            released = released.saturating_add(adapter.release_all_requests()?);
            let mut guard = adapter
                .allocator()
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.purge_prefix_cache();
        }
        Ok(released)
    }

    /// Common attention frontier across both Global and SWA groups for `seq_id`.
    pub(crate) fn spec_frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.request_token_count_all(seq_id)
            .ok()
            .map(|attn_tokens| SpecFrontier {
                attn_tokens: u64::from(attn_tokens),
                recurrent_tokens: None,
            })
    }
}

impl SpecPagedCache for Inkling2KVCacheCoordinator {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        match self.reserve_rows_all(seq_id, rows) {
            Ok(_) => Ok(true),
            Err(error) if error.starts_with("context_length_exceeded:") => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.record_tokens_all(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        let rows = u32::try_from(rows)
            .map_err(|_| format!("Inkling2 commit rollback of {rows} rows does not fit u32"))?;
        self.rollback_last_tokens_all(seq_id, rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        let committed = u32::try_from(committed_tokens).map_err(|_| {
            format!("Inkling2 committed frontier {committed_tokens} does not fit u32")
        })?;
        self.eval_pending_pool_writes_all()?;
        self.prune_sliding_all_committed(seq_id, committed)
            .map(|_| ())
    }

    fn settle_captures_durable_state(&self) -> bool {
        // Coordinator-scope settle evaluates pending pool writes and prunes expired
        // SWA blocks strictly below the committed cutoff; no durable sidecar capture.
        false
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.spec_frontier(seq_id)
    }
}

fn group_reserved_blocks(
    attention_kind: AttentionKind,
    max_admission_blocks: u32,
    concurrent_sequences: u32,
    block_size: u32,
) -> u32 {
    let extra_seqs = concurrent_sequences.saturating_sub(1);
    match attention_kind {
        AttentionKind::Full => {
            // One full-context sequence + starter blocks for concurrent lanes.
            max_admission_blocks.saturating_add(extra_seqs.saturating_mul(4))
        }
        AttentionKind::SlidingWindow { sliding_window } => {
            // One full chunk+window admission + decode window per concurrent lane + 1 reserved null block.
            let decode_window_blocks = sliding_window.div_ceil(block_size.max(1)).saturating_add(1);
            max_admission_blocks
                .saturating_add(extra_seqs.saturating_mul(decode_window_blocks))
                .saturating_add(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_config() -> Inkling2Config {
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
        Inkling2Config::from_json_str(&raw, true).unwrap()
    }

    #[test]
    fn inkling2_kv_groups_split_8_global_and_40_swa_visits() {
        let cfg = mini_config();
        let (specs, groups) =
            compute_visit_kv_cache_groups(&cfg, 32, KvFormat::Int8, 32_768, 2048).unwrap();
        assert_eq!(specs.len(), 48);
        assert_eq!(groups.len(), 2);

        // Group 0: 8 Global visits
        assert_eq!(groups[0].group_id, 0);
        assert_eq!(groups[0].attention_kind, AttentionKind::Full);
        assert_eq!(groups[0].physical_layout.block_size, 32);
        assert_eq!(groups[0].physical_layout.num_kv_heads, 8);
        assert_eq!(groups[0].physical_layout.head_size, 128);
        assert_eq!(groups[0].physical_layout.cache_dtype, KVCacheDType::Int8);
        assert_eq!(groups[0].layer_indices, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(groups[0].physical_layer_indices.len(), 8);

        // Group 1: 40 SWA visits
        assert_eq!(groups[1].group_id, 1);
        assert_eq!(
            groups[1].attention_kind,
            AttentionKind::SlidingWindow {
                sliding_window: 1024
            }
        );
        assert_eq!(groups[1].layer_indices.len(), 40);
        assert_eq!(groups[1].physical_layer_indices.len(), 40);
    }

    #[test]
    fn inkling2_coordinator_swa_pruning_and_spec_paged_rollback() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let cfg = mini_config();
        for kv_format in [KvFormat::Int8, KvFormat::Bf16] {
            let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
                .with_kv_format(kv_format)
                .with_max_model_len(4096)
                .with_max_chunk(512)
                .with_max_concurrent_sequences(2);
            let mut coord = Inkling2KVCacheCoordinator::new_for_config(&cfg, paged_cfg).unwrap();
            assert_eq!(coord.kv_format(), kv_format);
            assert_eq!(coord.block_size(), 32);
            assert_eq!(
                coord.full_adapter().page_layout().keys_token_major,
                kv_format == KvFormat::Int8
            );
            assert_eq!(
                coord
                    .sliding_adapter()
                    .unwrap()
                    .page_layout()
                    .keys_token_major,
                kv_format == KvFormat::Int8
            );

            // Start sequence 7 and record 1500 tokens (crossing sliding_window = 1024)
            let seq_id = 7u32;
            coord.reset_scheduled_request(seq_id).unwrap();
            let prompt: Vec<u32> = (1..=1500).collect();
            coord.record_tokens_all(seq_id, &prompt).unwrap();
            assert_eq!(coord.request_token_count_all(seq_id).unwrap(), 1500);

            // Prune SWA blocks strictly beyond 1024:
            // oldest live token = 1500 - 1024 = 476 -> blocks 0..14 (tokens 0..448) are expired!
            let freed = coord.prune_sliding_all(seq_id).unwrap();
            assert!(
                freed >= 14,
                "{kv_format:?}: expected >= 14 SWA blocks pruned at 1500 tokens, got {freed}"
            );
            // Absolute token count stays 1500 in BOTH groups!
            assert_eq!(coord.request_token_count_all(seq_id).unwrap(), 1500);

            // Exercise SpecPagedCache verify/commit/rollback + committed-basis prune
            assert!(coord.reserve_lookahead(seq_id, 8).unwrap());
            let verify_tokens: Vec<u32> = (1501..=1508).collect();
            let ticket = coord.record_verify(seq_id, &verify_tokens).unwrap();
            assert_eq!(coord.frontier(seq_id).unwrap().attn_tokens, 1508);

            // Committed-basis prune anchored at 1500 must NOT corrupt rollback to 1503
            coord.settle_committed(seq_id, 1500).unwrap();
            coord.commit_cycle(seq_id, ticket, 3).unwrap();
            assert_eq!(coord.frontier(seq_id).unwrap().attn_tokens, 1503);
            assert_eq!(coord.request_token_count_all(seq_id).unwrap(), 1503);
        }
    }

    #[test]
    fn inkling2_short_session_partial_block_can_continue() {
        if !unsafe { mlx_sys::mlx_metal_is_available() } {
            return;
        }
        let cfg = mini_config();
        let paged_cfg = Inkling2PagedConfig::default_for_config(&cfg)
            .with_kv_format(KvFormat::Int8)
            .with_max_model_len(2048)
            .with_max_chunk(512)
            .with_max_concurrent_sequences(2);
        let mut coord = Inkling2KVCacheCoordinator::new_for_config(&cfg, paged_cfg).unwrap();

        let seq_id = 1u32;
        coord.reset_scheduled_request(seq_id).unwrap();

        // Turn 1: 15 tokens (< 32 block_size -> zero full blocks to publish).
        let turn1: Vec<u32> = (1..=15).collect();
        coord.record_tokens_all(seq_id, &turn1).unwrap();
        coord.finalize_keep_live_all(seq_id, &[], 0).unwrap();
        assert!(
            coord.is_live_all(seq_id),
            "short <32-token session must still be live after finalize_keep_live_all"
        );

        // Turn 2: extends turn 1 to 24 tokens (still < 32).
        let turn2: Vec<u32> = (1..=24).collect();
        assert!(coord.can_continue_all(seq_id, &turn2));
        let boundary = coord.continue_turn_all(seq_id, &turn2, 24).unwrap();
        assert_eq!(boundary, 15);
        coord.record_tokens_all(seq_id, &turn2[15..]).unwrap();
        coord.finalize_keep_live_all(seq_id, &[], 0).unwrap();
        assert!(coord.is_live_all(seq_id));

        // Turn 3: extends past 32 tokens (crosses into a full block + trailing partial block).
        let turn3: Vec<u32> = (1..=40).collect();
        assert!(coord.can_continue_all(seq_id, &turn3));
        let boundary3 = coord.continue_turn_all(seq_id, &turn3, 40).unwrap();
        assert_eq!(boundary3, 24);
        coord.record_tokens_all(seq_id, &turn3[24..]).unwrap();
        coord.finalize_keep_live_all(seq_id, &[], 0).unwrap();
        assert!(coord.is_live_all(seq_id));
        assert_eq!(coord.request_token_count_all(seq_id).unwrap(), 40);

        // Releasing the failed/finished sequence returns every block in both groups.
        let free_before_full = coord.full_adapter().block_telemetry().unwrap().free_blocks;
        let free_before_swa = coord
            .sliding_adapter()
            .unwrap()
            .block_telemetry()
            .unwrap()
            .free_blocks;
        let released = coord.release_request_all(seq_id).unwrap();
        assert!(
            released >= 4,
            "expected blocks released from both groups, got {released}"
        );
        assert!(!coord.is_live_all(seq_id));
        assert!(
            coord.full_adapter().block_telemetry().unwrap().free_blocks
                + coord
                    .full_adapter()
                    .block_telemetry()
                    .unwrap()
                    .evictable_blocks
                > free_before_full
        );
        assert!(
            coord
                .sliding_adapter()
                .unwrap()
                .block_telemetry()
                .unwrap()
                .free_blocks
                > free_before_swa
        );
    }
}
