//! Model-neutral KV-cache grouping and admission policy.
//!
//! This is the vLLM-inspired boundary for hybrid models: model code declares
//! per-layer KV-cache specs, while common transformer infrastructure owns group
//! layout, layer-to-group mapping, storage ordinals, and sliding-window
//! admission arithmetic.

use std::collections::HashMap;

#[cfg(all(target_os = "macos", not(test)))]
use std::sync::{Arc, Mutex};

use crate::transformer::kv_cache_spec::{
    AttentionKind, KVCacheDType, KVCacheGroup, LayerKVCacheRoute, LayerKVCacheSpec,
    derive_layer_kv_cache_routes_from_groups, group_layer_kv_cache_specs,
};
#[cfg(all(target_os = "macos", not(test)))]
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

/// Stable identifier for a KV-cache group.
pub type KVCacheGroupId = usize;

/// Cache behavior for a physical KV-cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKVCacheKind {
    Full,
    SlidingWindow { window_size: u32 },
}

impl LayerKVCacheKind {
    pub fn sliding_window(self) -> Option<u32> {
        match self {
            Self::SlidingWindow { window_size } => Some(window_size),
            Self::Full => None,
        }
    }
}

impl From<AttentionKind> for LayerKVCacheKind {
    fn from(kind: AttentionKind) -> Self {
        match kind {
            AttentionKind::Full => Self::Full,
            AttentionKind::SlidingWindow { sliding_window } => Self::SlidingWindow {
                window_size: sliding_window,
            },
        }
    }
}

/// Immutable per-group configuration derived from layer specs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KVCacheGroupConfig {
    pub group_id: KVCacheGroupId,
    pub kind: LayerKVCacheKind,
    /// Logical layer indices that use this group. Shared-KV aliases are kept
    /// here so model runners can resolve their group metadata.
    pub layers: Vec<usize>,
    /// Logical layer indices that own physical KV storage. Shared-KV aliases
    /// are omitted because they reuse their anchor's K/V.
    pub physical_layers: Vec<usize>,
    pub block_size: u32,
    pub num_blocks: u32,
    pub head_size: u32,
    pub num_kv_heads: u32,
    pub cache_dtype: KVCacheDType,
    pub max_admission_blocks: u32,
    pub max_model_len: u32,
    pub max_chunk: u32,
    pub gpu_memory_mb: u32,
    pub max_batch_size: Option<u32>,
}

impl KVCacheGroupConfig {
    /// Current adapter storage supports full-attention groups. Sliding-window
    /// groups are represented here first; true paged sliding eviction is the
    /// next integration step.
    pub fn is_paged_adapter_group(&self) -> bool {
        !self.physical_layers.is_empty() && matches!(self.kind, LayerKVCacheKind::Full)
    }

    pub fn max_cached_tokens(&self) -> u64 {
        self.num_blocks as u64 * self.block_size as u64
    }

    pub fn sliding_window(&self) -> Option<u32> {
        self.kind.sliding_window()
    }
}

/// Runtime state for one group.
pub struct KVCacheGroupState {
    pub config: KVCacheGroupConfig,
    #[cfg(all(target_os = "macos", not(test)))]
    pub adapter: Option<PagedKVCacheAdapter>,
}

impl KVCacheGroupState {
    #[cfg(any(not(target_os = "macos"), test))]
    fn new(config: KVCacheGroupConfig) -> Self {
        Self { config }
    }

    #[cfg(all(target_os = "macos", not(test)))]
    fn new(config: KVCacheGroupConfig, adapter: Option<PagedKVCacheAdapter>) -> Self {
        Self { config, adapter }
    }
}

/// Common grouped manager for hybrid KV-cache layouts.
pub struct HybridKVCacheManager {
    groups: Vec<KVCacheGroupState>,
    layer_to_group: HashMap<usize, KVCacheGroupId>,
    group_id_to_index: HashMap<KVCacheGroupId, usize>,
    group_layer_ordinals: HashMap<usize, usize>,
    layer_routes: HashMap<usize, LayerKVCacheRoute>,
}

impl HybridKVCacheManager {
    pub fn from_kv_cache_specs(
        specs: &[LayerKVCacheSpec],
        max_model_len: u32,
        max_chunk: u32,
        gpu_memory_mb: u32,
        max_batch_size: Option<u32>,
        num_blocks_override: Option<u32>,
    ) -> Result<Self, String> {
        let shared_groups = group_layer_kv_cache_specs(specs, max_model_len, max_chunk)
            .map_err(|e| format!("failed to group KV cache specs: {e}"))?;
        Self::from_groups(
            specs,
            shared_groups,
            max_model_len,
            max_chunk,
            gpu_memory_mb,
            max_batch_size,
            num_blocks_override,
        )
    }

    fn from_groups(
        specs: &[LayerKVCacheSpec],
        shared_groups: Vec<KVCacheGroup>,
        max_model_len: u32,
        max_chunk: u32,
        gpu_memory_mb: u32,
        max_batch_size: Option<u32>,
        num_blocks_override: Option<u32>,
    ) -> Result<Self, String> {
        if shared_groups.is_empty() {
            return Err("HybridKVCacheManager requires at least one KV group".to_string());
        }
        let routes = derive_layer_kv_cache_routes_from_groups(specs, &shared_groups)
            .map_err(|e| format!("failed to derive KV cache routes: {e}"))?;

        let by_layer: HashMap<usize, &LayerKVCacheSpec> =
            specs.iter().map(|spec| (spec.layer_index, spec)).collect();
        let mut groups = Vec::with_capacity(shared_groups.len());
        let mut layer_to_group = HashMap::with_capacity(specs.len());
        let mut group_id_to_index = HashMap::with_capacity(shared_groups.len());
        let mut group_layer_ordinals = HashMap::with_capacity(specs.len());
        let layer_routes: HashMap<usize, LayerKVCacheRoute> = routes
            .into_iter()
            .map(|route| (route.layer_index, route))
            .collect();

        for shared_group in shared_groups {
            let physical_layers = shared_group.physical_layer_indices.clone();
            let layout = shared_group.physical_layout;
            let num_blocks = if let Some(num_blocks) = num_blocks_override {
                num_blocks
            } else {
                mlx_paged_attn::PagedAttentionConfig {
                    block_size: layout.block_size,
                    gpu_memory_mb,
                    head_size: layout.head_size,
                    num_kv_heads: layout.num_kv_heads,
                    num_layers: physical_layers.len().max(1) as u32,
                    use_fp8_cache: Some(matches!(layout.cache_dtype, KVCacheDType::Fp8)),
                    max_seq_len: Some(max_model_len),
                    max_batch_size,
                }
                .calculate_num_blocks()
            };
            if num_blocks == 0 {
                return Err(format!(
                    "KV group {}: computed num_blocks is 0",
                    shared_group.group_id
                ));
            }

            let mut physical_ordinals = HashMap::with_capacity(physical_layers.len());
            for (ordinal, layer_idx) in physical_layers.iter().copied().enumerate() {
                physical_ordinals.insert(layer_idx, ordinal);
            }
            for layer_idx in &shared_group.layer_indices {
                layer_to_group.insert(*layer_idx, shared_group.group_id);
                let spec = by_layer.get(layer_idx).ok_or_else(|| {
                    format!(
                        "KV group {} references missing layer {}",
                        shared_group.group_id, layer_idx
                    )
                })?;
                let physical_layer = spec.shared_kv_anchor.unwrap_or(*layer_idx);
                if let Some(ordinal) = physical_ordinals.get(&physical_layer) {
                    group_layer_ordinals.insert(*layer_idx, *ordinal);
                }
            }

            let config = KVCacheGroupConfig {
                group_id: shared_group.group_id,
                kind: shared_group.attention_kind.into(),
                layers: shared_group.layer_indices,
                physical_layers,
                block_size: layout.block_size,
                num_blocks,
                head_size: layout.head_size,
                num_kv_heads: layout.num_kv_heads,
                cache_dtype: layout.cache_dtype,
                max_admission_blocks: shared_group.max_admission_blocks,
                max_model_len,
                max_chunk,
                gpu_memory_mb,
                max_batch_size,
            };

            #[cfg(any(not(target_os = "macos"), test))]
            let state = KVCacheGroupState::new(config);

            #[cfg(all(target_os = "macos", not(test)))]
            let state = {
                let adapter = if config.is_paged_adapter_group() {
                    Some(build_paged_adapter(&config)?)
                } else {
                    None
                };
                KVCacheGroupState::new(config, adapter)
            };

            group_id_to_index.insert(state.config.group_id, groups.len());
            groups.push(state);
        }

        Ok(Self {
            groups,
            layer_to_group,
            group_id_to_index,
            group_layer_ordinals,
            layer_routes,
        })
    }

    pub fn groups(&self) -> &[KVCacheGroupState] {
        &self.groups
    }

    pub fn group_config(&self, group_id: KVCacheGroupId) -> Option<&KVCacheGroupConfig> {
        self.group_id_to_index
            .get(&group_id)
            .and_then(|idx| self.groups.get(*idx))
            .map(|state| &state.config)
    }

    pub fn group_for_layer(&self, layer_idx: usize) -> Option<KVCacheGroupId> {
        self.layer_to_group.get(&layer_idx).copied()
    }

    pub fn route_for_layer(&self, layer_idx: usize) -> Option<&LayerKVCacheRoute> {
        self.layer_routes.get(&layer_idx)
    }

    /// Physical cache ordinal within the layer's group. Shared-KV aliases map
    /// to their anchor's ordinal.
    pub fn group_layer_ordinal(&self, layer_idx: usize) -> Option<usize> {
        self.group_layer_ordinals.get(&layer_idx).copied()
    }

    pub fn max_admission_blocks_for_group(&self, group_id: KVCacheGroupId) -> Option<u32> {
        self.group_config(group_id)
            .map(|config| config.max_admission_blocks)
    }

    /// Number of currently needed blocks for a group at `logical_tokens`.
    ///
    /// Full-attention groups grow with the full context. Sliding-window groups
    /// need only the active local tail; `max_admission_blocks` remains the
    /// startup/runtime upper admission bound for chunked scheduling.
    pub fn live_block_count_for_group(
        &self,
        group_id: KVCacheGroupId,
        logical_tokens: u32,
    ) -> Option<u32> {
        let config = self.group_config(group_id)?;
        Some(Self::live_block_count(config, logical_tokens))
    }

    pub fn live_block_count(config: &KVCacheGroupConfig, logical_tokens: u32) -> u32 {
        let admitted_tokens = config
            .sliding_window()
            .map_or(logical_tokens, |window| logical_tokens.min(window));
        ceil_div(admitted_tokens, config.block_size)
            .min(config.num_blocks)
            .min(config.max_admission_blocks)
    }

    /// Number of oldest computed tokens that a sliding-window group can drop
    /// before the next allocation. Mirrors vLLM's
    /// `max(0, num_computed_tokens - sliding_window + 1)`.
    pub fn sliding_window_skipped_tokens_for_group(
        &self,
        group_id: KVCacheGroupId,
        num_computed_tokens: u32,
    ) -> Option<u32> {
        let config = self.group_config(group_id)?;
        Some(Self::sliding_window_skipped_tokens(
            config,
            num_computed_tokens,
        ))
    }

    pub fn sliding_window_skipped_tokens(
        config: &KVCacheGroupConfig,
        num_computed_tokens: u32,
    ) -> u32 {
        config.sliding_window().map_or(0, |window| {
            num_computed_tokens
                .saturating_sub(window)
                .saturating_add(if num_computed_tokens >= window { 1 } else { 0 })
        })
    }
}

fn ceil_div(n: u32, d: u32) -> u32 {
    debug_assert!(d > 0);
    if n == 0 { 0 } else { ((n - 1) / d) + 1 }
}

#[cfg(all(target_os = "macos", not(test)))]
fn build_paged_adapter(config: &KVCacheGroupConfig) -> Result<PagedKVCacheAdapter, String> {
    let pa_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: config.block_size,
        gpu_memory_mb: config.gpu_memory_mb,
        head_size: config.head_size,
        num_kv_heads: config.num_kv_heads,
        num_layers: config.physical_layers.len() as u32,
        use_fp8_cache: Some(matches!(config.cache_dtype, KVCacheDType::Fp8)),
        max_seq_len: Some(config.max_model_len),
        max_batch_size: config.max_batch_size,
    };

    let cache_dtype = match config.cache_dtype {
        KVCacheDType::Float16 => mlx_paged_attn::metal::MetalDtype::Float16,
        KVCacheDType::BFloat16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
        KVCacheDType::Fp8 => mlx_paged_attn::metal::MetalDtype::UChar,
    };
    let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
        config.num_blocks,
        config.block_size,
    )));
    let pool = mlx_paged_attn::LayerKVPool::new(pa_config, config.num_blocks, cache_dtype)?;

    PagedKVCacheAdapter::new(allocator, Arc::new(pool), config.block_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::kv_cache_spec::{
        KVCacheDType, KVCachePhysicalLayout, LayerKVCacheSpec,
    };

    fn layout(block_size: u32) -> KVCachePhysicalLayout {
        KVCachePhysicalLayout::new(block_size, 4, 128, KVCacheDType::BFloat16)
    }

    #[test]
    fn maps_layers_to_groups_and_physical_ordinals() {
        let specs = vec![
            LayerKVCacheSpec::sliding_window(0, 16, layout(8)),
            LayerKVCacheSpec::sliding_window(1, 16, layout(8)),
            LayerKVCacheSpec::full(2, layout(8)),
            LayerKVCacheSpec::full(4, layout(8)),
            LayerKVCacheSpec::full(5, layout(8)).shared_with_anchor(2),
        ];

        let manager =
            HybridKVCacheManager::from_kv_cache_specs(&specs, 128, 32, 2048, Some(32), Some(8))
                .unwrap();

        assert_eq!(manager.groups().len(), 2);
        let sliding_group = manager.group_for_layer(0).unwrap();
        let full_group = manager.group_for_layer(2).unwrap();
        assert_eq!(manager.group_for_layer(1), Some(sliding_group));
        assert_eq!(manager.group_for_layer(4), Some(full_group));
        assert_eq!(manager.group_for_layer(5), Some(full_group));
        assert_eq!(manager.group_for_layer(99), None);
        assert_eq!(manager.group_layer_ordinal(2), Some(0));
        assert_eq!(manager.group_layer_ordinal(4), Some(1));
        assert_eq!(
            manager.group_layer_ordinal(5),
            Some(0),
            "shared layer maps to anchor physical ordinal"
        );
    }

    #[test]
    fn full_attention_admits_full_context_up_to_capacity() {
        let specs = vec![LayerKVCacheSpec::full(0, layout(8))];
        let manager =
            HybridKVCacheManager::from_kv_cache_specs(&specs, 128, 32, 2048, Some(32), Some(3))
                .unwrap();
        let group_id = manager.group_for_layer(0).unwrap();

        assert_eq!(manager.live_block_count_for_group(group_id, 0), Some(0));
        assert_eq!(manager.live_block_count_for_group(group_id, 1), Some(1));
        assert_eq!(manager.live_block_count_for_group(group_id, 16), Some(2));
        assert_eq!(manager.live_block_count_for_group(group_id, 25), Some(3));
        assert_eq!(manager.live_block_count_for_group(group_id, 128), Some(3));
        assert_eq!(
            manager.sliding_window_skipped_tokens_for_group(group_id, 128),
            Some(0)
        );
    }

    #[test]
    fn sliding_window_uses_vllm_admission_cap_and_skip_math() {
        let specs = vec![LayerKVCacheSpec::sliding_window(0, 17, layout(8))];
        let manager =
            HybridKVCacheManager::from_kv_cache_specs(&specs, 128, 32, 2048, Some(32), Some(8))
                .unwrap();
        let group_id = manager.group_for_layer(0).unwrap();

        assert_eq!(
            manager.max_admission_blocks_for_group(group_id),
            Some(7),
            "ceil((17 - 1 + 32) / 8) + 1"
        );
        assert_eq!(manager.live_block_count_for_group(group_id, 16), Some(2));
        assert_eq!(manager.live_block_count_for_group(group_id, 17), Some(3));
        assert_eq!(manager.live_block_count_for_group(group_id, 64), Some(3));
        assert_eq!(
            manager.sliding_window_skipped_tokens_for_group(group_id, 16),
            Some(0)
        );
        assert_eq!(
            manager.sliding_window_skipped_tokens_for_group(group_id, 17),
            Some(1)
        );
        assert_eq!(
            manager.sliding_window_skipped_tokens_for_group(group_id, 64),
            Some(48)
        );
    }

    #[test]
    fn grouping_splits_incompatible_layouts() {
        let specs = vec![
            LayerKVCacheSpec::full(0, layout(8)),
            LayerKVCacheSpec::full(1, layout(16)),
        ];

        let manager =
            HybridKVCacheManager::from_kv_cache_specs(&specs, 128, 32, 2048, Some(32), Some(8))
                .unwrap();

        assert_eq!(manager.groups().len(), 2);
        assert_ne!(manager.group_for_layer(0), manager.group_for_layer(1));
    }
}
