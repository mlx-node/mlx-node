use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Cache element type used for physical KV storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KVCacheDType {
    Float16,
    BFloat16,
    Fp8,
}

/// Physical cache layout. Layers with equal layouts can share a cache group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KVCachePhysicalLayout {
    pub block_size: u32,
    pub num_kv_heads: u32,
    pub head_size: u32,
    pub cache_dtype: KVCacheDType,
}

impl KVCachePhysicalLayout {
    pub fn new(
        block_size: u32,
        num_kv_heads: u32,
        head_size: u32,
        cache_dtype: KVCacheDType,
    ) -> Self {
        Self {
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.block_size > 0 && self.num_kv_heads > 0 && self.head_size > 0
    }

    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }
}

/// Attention behavior that determines KV admission pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttentionKind {
    Full,
    SlidingWindow { sliding_window: u32 },
}

impl AttentionKind {
    pub fn full_attention_max_blocks(
        max_model_len: u32,
        block_size: u32,
    ) -> Result<u32, KVCacheSpecError> {
        if block_size == 0 {
            return Err(KVCacheSpecError::InvalidBlockSize { block_size });
        }
        Ok(div_ceil(max_model_len, block_size))
    }

    /// vLLM-style admission bound for sliding-window layers.
    ///
    /// A chunk can attend to the previous `sliding_window - 1` tokens plus the
    /// new chunk. The token bound is capped by `max_model_len`, rounded up to
    /// blocks, then one extra block is admitted for the trailing partial block.
    pub fn sliding_window_max_admission_blocks(
        sliding_window: u32,
        max_model_len: u32,
        max_chunk: u32,
        block_size: u32,
    ) -> Result<u32, KVCacheSpecError> {
        if block_size == 0 {
            return Err(KVCacheSpecError::InvalidBlockSize { block_size });
        }
        if sliding_window == 0 {
            return Err(KVCacheSpecError::InvalidSlidingWindow { sliding_window });
        }
        let uncapped = sliding_window.saturating_sub(1).saturating_add(max_chunk);
        let admitted_tokens = uncapped.min(max_model_len);
        Ok(div_ceil(admitted_tokens, block_size).saturating_add(1))
    }

    pub fn max_admission_blocks(
        self,
        max_model_len: u32,
        max_chunk: u32,
        block_size: u32,
    ) -> Result<u32, KVCacheSpecError> {
        match self {
            Self::Full => Self::full_attention_max_blocks(max_model_len, block_size),
            Self::SlidingWindow { sliding_window } => Self::sliding_window_max_admission_blocks(
                sliding_window,
                max_model_len,
                max_chunk,
                block_size,
            ),
        }
    }
}

/// Model-independent per-layer KV cache requirements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerKVCacheSpec {
    pub layer_index: usize,
    pub attention_kind: AttentionKind,
    pub physical_layout: KVCachePhysicalLayout,
    /// `Some(anchor)` means this logical layer reuses the anchor layer's
    /// physical KV cache and should not allocate a second cache of its own.
    pub shared_kv_anchor: Option<usize>,
}

impl LayerKVCacheSpec {
    pub fn new(
        layer_index: usize,
        attention_kind: AttentionKind,
        physical_layout: KVCachePhysicalLayout,
    ) -> Self {
        Self {
            layer_index,
            attention_kind,
            physical_layout,
            shared_kv_anchor: None,
        }
    }

    pub fn full(layer_index: usize, physical_layout: KVCachePhysicalLayout) -> Self {
        Self::new(layer_index, AttentionKind::Full, physical_layout)
    }

    pub fn sliding_window(
        layer_index: usize,
        sliding_window: u32,
        physical_layout: KVCachePhysicalLayout,
    ) -> Self {
        Self::new(
            layer_index,
            AttentionKind::SlidingWindow { sliding_window },
            physical_layout,
        )
    }

    pub fn shared_with_anchor(mut self, anchor_layer_index: usize) -> Self {
        self.shared_kv_anchor = Some(anchor_layer_index);
        self
    }

    pub fn physical_layer_index(&self) -> usize {
        self.shared_kv_anchor.unwrap_or(self.layer_index)
    }

    pub fn is_physical_layout_compatible_with(&self, other: &Self) -> bool {
        self.physical_layout
            .is_compatible_with(&other.physical_layout)
    }

    pub fn max_admission_blocks(
        &self,
        max_model_len: u32,
        max_chunk: u32,
    ) -> Result<u32, KVCacheSpecError> {
        self.attention_kind.max_admission_blocks(
            max_model_len,
            max_chunk,
            self.physical_layout.block_size,
        )
    }
}

/// A group of layers that can be served by a compatible physical cache pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KVCacheGroup {
    pub group_id: usize,
    pub attention_kind: AttentionKind,
    pub physical_layout: KVCachePhysicalLayout,
    pub layer_indices: Vec<usize>,
    /// Logical layers with `shared_kv_anchor` are omitted here because their
    /// anchor layer owns the physical KV storage.
    pub physical_layer_indices: Vec<usize>,
    pub max_admission_blocks: u32,
}

/// Model-neutral per-layer route into a grouped KV cache.
///
/// This is the stable boundary model code should consume after declaring
/// `LayerKVCacheSpec`s: it resolves logical layer index, group id, physical
/// storage owner, and the owner's ordinal inside the group. Shared-KV aliases
/// map to their anchor's physical layer and ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerKVCacheRoute {
    pub layer_index: usize,
    pub group_id: usize,
    pub attention_kind: AttentionKind,
    pub physical_layout: KVCachePhysicalLayout,
    pub physical_layer_index: usize,
    pub physical_layer_ordinal: usize,
    pub shared_kv_anchor: Option<usize>,
}

impl LayerKVCacheRoute {
    pub fn is_shared(&self) -> bool {
        self.shared_kv_anchor.is_some()
    }

    pub fn is_full_attention(&self) -> bool {
        matches!(self.attention_kind, AttentionKind::Full)
    }

    pub fn sliding_window(&self) -> Option<u32> {
        match self.attention_kind {
            AttentionKind::Full => None,
            AttentionKind::SlidingWindow { sliding_window } => Some(sliding_window),
        }
    }
}

/// Per-group prefix cache hit used by hybrid cache coordinators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KVCacheGroupPrefixHit {
    pub group_id: usize,
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KVCacheSpecError {
    DuplicateLayerIndex {
        layer_index: usize,
    },
    InvalidPhysicalLayout {
        layer_index: usize,
    },
    InvalidBlockSize {
        block_size: u32,
    },
    InvalidSlidingWindow {
        sliding_window: u32,
    },
    MissingSharedKVAnchor {
        layer_index: usize,
        anchor_layer_index: usize,
    },
    SharedKVAnchorIsAlias {
        layer_index: usize,
        anchor_layer_index: usize,
        root_anchor_layer_index: usize,
    },
    SharedKVIncompatible {
        layer_index: usize,
        anchor_layer_index: usize,
    },
    MissingPhysicalLayerOrdinal {
        layer_index: usize,
        physical_layer_index: usize,
    },
}

impl fmt::Display for KVCacheSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLayerIndex { layer_index } => {
                write!(f, "duplicate KV cache spec for layer {layer_index}")
            }
            Self::InvalidPhysicalLayout { layer_index } => {
                write!(
                    f,
                    "invalid KV cache physical layout for layer {layer_index}"
                )
            }
            Self::InvalidBlockSize { block_size } => {
                write!(f, "invalid KV cache block_size {block_size}")
            }
            Self::InvalidSlidingWindow { sliding_window } => {
                write!(f, "invalid sliding_window {sliding_window}")
            }
            Self::MissingSharedKVAnchor {
                layer_index,
                anchor_layer_index,
            } => write!(
                f,
                "layer {layer_index} shares KV with missing anchor layer {anchor_layer_index}"
            ),
            Self::SharedKVAnchorIsAlias {
                layer_index,
                anchor_layer_index,
                root_anchor_layer_index,
            } => write!(
                f,
                "layer {layer_index} shares KV with alias layer {anchor_layer_index}; \
                 use root anchor layer {root_anchor_layer_index}"
            ),
            Self::SharedKVIncompatible {
                layer_index,
                anchor_layer_index,
            } => write!(
                f,
                "layer {layer_index} is not KV-cache compatible with shared anchor \
                 layer {anchor_layer_index}"
            ),
            Self::MissingPhysicalLayerOrdinal {
                layer_index,
                physical_layer_index,
            } => write!(
                f,
                "layer {layer_index} maps to physical KV layer {physical_layer_index}, \
                 but that physical layer is not present in its KV group"
            ),
        }
    }
}

impl std::error::Error for KVCacheSpecError {}

/// Group layer specs by attention behavior and physical layout.
///
/// The returned order is deterministic: groups are sorted by
/// `(attention_kind, physical_layout)`, and each group's layer lists are sorted
/// by layer index.
pub fn group_layer_kv_cache_specs(
    specs: &[LayerKVCacheSpec],
    max_model_len: u32,
    max_chunk: u32,
) -> Result<Vec<KVCacheGroup>, KVCacheSpecError> {
    validate_layer_kv_cache_specs(specs)?;

    let mut groups: BTreeMap<(AttentionKind, KVCachePhysicalLayout), Vec<&LayerKVCacheSpec>> =
        BTreeMap::new();
    for spec in specs {
        groups
            .entry((spec.attention_kind, spec.physical_layout))
            .or_default()
            .push(spec);
    }

    groups
        .into_iter()
        .enumerate()
        .map(
            |(group_id, ((attention_kind, physical_layout), mut layers))| {
                layers.sort_by_key(|spec| spec.layer_index);
                let layer_indices = layers.iter().map(|spec| spec.layer_index).collect();
                let physical_layer_indices = layers
                    .iter()
                    .filter(|spec| spec.shared_kv_anchor.is_none())
                    .map(|spec| spec.layer_index)
                    .collect();
                let max_admission_blocks = attention_kind.max_admission_blocks(
                    max_model_len,
                    max_chunk,
                    physical_layout.block_size,
                )?;

                Ok(KVCacheGroup {
                    group_id,
                    attention_kind,
                    physical_layout,
                    layer_indices,
                    physical_layer_indices,
                    max_admission_blocks,
                })
            },
        )
        .collect()
}

pub fn derive_layer_kv_cache_routes(
    specs: &[LayerKVCacheSpec],
    max_model_len: u32,
    max_chunk: u32,
) -> Result<Vec<LayerKVCacheRoute>, KVCacheSpecError> {
    let groups = group_layer_kv_cache_specs(specs, max_model_len, max_chunk)?;
    derive_layer_kv_cache_routes_from_groups(specs, &groups)
}

pub fn derive_layer_kv_cache_routes_from_groups(
    specs: &[LayerKVCacheSpec],
    groups: &[KVCacheGroup],
) -> Result<Vec<LayerKVCacheRoute>, KVCacheSpecError> {
    validate_layer_kv_cache_specs(specs)?;

    let by_layer: BTreeMap<usize, &LayerKVCacheSpec> =
        specs.iter().map(|spec| (spec.layer_index, spec)).collect();
    let mut routes: BTreeMap<usize, LayerKVCacheRoute> = BTreeMap::new();

    for group in groups {
        let physical_ordinals: BTreeMap<usize, usize> = group
            .physical_layer_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(ordinal, layer_index)| (layer_index, ordinal))
            .collect();

        for layer_index in &group.layer_indices {
            let spec = by_layer
                .get(layer_index)
                .expect("group_layer_kv_cache_specs returned an unknown layer");
            let physical_layer_index = spec.physical_layer_index();
            let physical_layer_ordinal = physical_ordinals
                .get(&physical_layer_index)
                .copied()
                .ok_or(KVCacheSpecError::MissingPhysicalLayerOrdinal {
                    layer_index: *layer_index,
                    physical_layer_index,
                })?;

            routes.insert(
                *layer_index,
                LayerKVCacheRoute {
                    layer_index: *layer_index,
                    group_id: group.group_id,
                    attention_kind: group.attention_kind,
                    physical_layout: group.physical_layout,
                    physical_layer_index,
                    physical_layer_ordinal,
                    shared_kv_anchor: spec.shared_kv_anchor,
                },
            );
        }
    }

    Ok(routes.into_values().collect())
}

pub fn common_kv_cache_block_alignment(groups: &[KVCacheGroup]) -> u32 {
    groups
        .iter()
        .map(|group| group.physical_layout.block_size)
        .filter(|block_size| *block_size > 0)
        .fold(1, lcm_u32)
}

pub fn align_prefix_len_to_kv_cache_groups(prefix_len: u32, groups: &[KVCacheGroup]) -> u32 {
    let alignment = common_kv_cache_block_alignment(groups);
    if alignment == 0 {
        return prefix_len;
    }
    prefix_len / alignment * alignment
}

/// Intersect per-group prefix hits using vLLM's hybrid-cache rule: every
/// active group must accept the prefix, then the result is aligned to a block
/// boundary that is valid for every group.
pub fn intersect_kv_cache_group_prefix_hits(
    groups: &[KVCacheGroup],
    hits: &[KVCacheGroupPrefixHit],
) -> u32 {
    if groups.is_empty() || hits.is_empty() {
        return 0;
    }

    let by_group: BTreeMap<usize, u32> = hits
        .iter()
        .map(|hit| (hit.group_id, hit.cached_tokens))
        .collect();
    let mut candidate = u32::MAX;
    for group in groups {
        let Some(hit) = by_group.get(&group.group_id) else {
            return 0;
        };
        candidate = candidate.min(*hit);
    }

    align_prefix_len_to_kv_cache_groups(candidate, groups)
}

pub fn validate_layer_kv_cache_specs(specs: &[LayerKVCacheSpec]) -> Result<(), KVCacheSpecError> {
    let mut seen = BTreeSet::new();
    let by_layer: BTreeMap<usize, &LayerKVCacheSpec> =
        specs.iter().map(|spec| (spec.layer_index, spec)).collect();

    for spec in specs {
        if !seen.insert(spec.layer_index) {
            return Err(KVCacheSpecError::DuplicateLayerIndex {
                layer_index: spec.layer_index,
            });
        }
        if !spec.physical_layout.is_valid() {
            return Err(KVCacheSpecError::InvalidPhysicalLayout {
                layer_index: spec.layer_index,
            });
        }
        let Some(anchor_layer_index) = spec.shared_kv_anchor else {
            continue;
        };
        let Some(anchor) = by_layer.get(&anchor_layer_index) else {
            return Err(KVCacheSpecError::MissingSharedKVAnchor {
                layer_index: spec.layer_index,
                anchor_layer_index,
            });
        };
        if let Some(root_anchor_layer_index) = anchor.shared_kv_anchor {
            return Err(KVCacheSpecError::SharedKVAnchorIsAlias {
                layer_index: spec.layer_index,
                anchor_layer_index,
                root_anchor_layer_index,
            });
        }
        if spec.attention_kind != anchor.attention_kind
            || !spec.is_physical_layout_compatible_with(anchor)
        {
            return Err(KVCacheSpecError::SharedKVIncompatible {
                layer_index: spec.layer_index,
                anchor_layer_index,
            });
        }
    }

    Ok(())
}

fn div_ceil(n: u32, d: u32) -> u32 {
    if n == 0 { 0 } else { 1 + (n - 1) / d }
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

fn lcm_u32(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd_u32(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(block_size: u32) -> KVCachePhysicalLayout {
        KVCachePhysicalLayout::new(block_size, 4, 128, KVCacheDType::BFloat16)
    }

    #[test]
    fn physical_layout_compatibility_is_exact() {
        let a = layout(16);
        let b = layout(16);
        let c = KVCachePhysicalLayout::new(32, 4, 128, KVCacheDType::BFloat16);

        assert!(a.is_compatible_with(&b));
        assert!(!a.is_compatible_with(&c));
    }

    #[test]
    fn full_attention_max_blocks_rounds_up_to_model_len() {
        assert_eq!(
            AttentionKind::full_attention_max_blocks(4096, 16).unwrap(),
            256
        );
        assert_eq!(
            AttentionKind::full_attention_max_blocks(4097, 16).unwrap(),
            257
        );
    }

    #[test]
    fn sliding_window_max_blocks_uses_window_chunk_cap_and_partial_block() {
        // sliding_window - 1 + max_chunk = 127 + 32 = 159.
        // ceil(159 / 16) + 1 partial block = 11.
        assert_eq!(
            AttentionKind::sliding_window_max_admission_blocks(128, 4096, 32, 16).unwrap(),
            11
        );

        // Cap by max_model_len before block rounding.
        assert_eq!(
            AttentionKind::sliding_window_max_admission_blocks(4096, 512, 256, 16).unwrap(),
            33
        );
    }

    #[test]
    fn shared_kv_anchor_omits_alias_from_physical_layers() {
        let specs = vec![
            LayerKVCacheSpec::full(0, layout(16)),
            LayerKVCacheSpec::full(1, layout(16)).shared_with_anchor(0),
            LayerKVCacheSpec::full(2, layout(16)),
        ];

        let groups = group_layer_kv_cache_specs(&specs, 128, 32).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].layer_indices, vec![0, 1, 2]);
        assert_eq!(groups[0].physical_layer_indices, vec![0, 2]);
        assert_eq!(groups[0].max_admission_blocks, 8);
    }

    #[test]
    fn grouping_splits_attention_kind_and_layout() {
        let specs = vec![
            LayerKVCacheSpec::full(0, layout(16)),
            LayerKVCacheSpec::sliding_window(1, 128, layout(16)),
            LayerKVCacheSpec::full(2, layout(32)),
            LayerKVCacheSpec::sliding_window(3, 128, layout(16)),
        ];

        let groups = group_layer_kv_cache_specs(&specs, 256, 32).unwrap();

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].attention_kind, AttentionKind::Full);
        assert_eq!(groups[0].physical_layout.block_size, 16);
        assert_eq!(groups[0].layer_indices, vec![0]);
        assert_eq!(groups[1].attention_kind, AttentionKind::Full);
        assert_eq!(groups[1].physical_layout.block_size, 32);
        assert_eq!(groups[1].layer_indices, vec![2]);
        assert_eq!(
            groups[2].attention_kind,
            AttentionKind::SlidingWindow {
                sliding_window: 128
            }
        );
        assert_eq!(groups[2].layer_indices, vec![1, 3]);
    }

    #[test]
    fn routes_map_shared_layers_to_anchor_physical_ordinals() {
        let specs = vec![
            LayerKVCacheSpec::sliding_window(0, 128, layout(16)),
            LayerKVCacheSpec::full(1, layout(16)),
            LayerKVCacheSpec::sliding_window(2, 128, layout(16)),
            LayerKVCacheSpec::full(3, layout(16)),
            LayerKVCacheSpec::sliding_window(4, 128, layout(16)).shared_with_anchor(2),
            LayerKVCacheSpec::full(5, layout(16)).shared_with_anchor(3),
        ];

        let routes = derive_layer_kv_cache_routes(&specs, 4096, 512).unwrap();

        assert_eq!(routes.len(), 6);
        assert_eq!(routes[0].layer_index, 0);
        assert_eq!(routes[0].physical_layer_ordinal, 0);
        assert_eq!(routes[2].physical_layer_ordinal, 1);
        assert_eq!(routes[4].shared_kv_anchor, Some(2));
        assert_eq!(routes[4].physical_layer_index, 2);
        assert_eq!(routes[4].physical_layer_ordinal, 1);
        assert_eq!(routes[5].shared_kv_anchor, Some(3));
        assert_eq!(routes[5].physical_layer_index, 3);
        assert_eq!(routes[5].physical_layer_ordinal, 1);
    }

    #[test]
    fn hybrid_prefix_hits_intersect_and_align_to_common_block() {
        let specs = vec![
            LayerKVCacheSpec::full(0, layout(16)),
            LayerKVCacheSpec::sliding_window(
                1,
                128,
                KVCachePhysicalLayout::new(32, 4, 128, KVCacheDType::BFloat16),
            ),
        ];
        let groups = group_layer_kv_cache_specs(&specs, 4096, 512).unwrap();

        assert_eq!(common_kv_cache_block_alignment(&groups), 32);
        assert_eq!(
            intersect_kv_cache_group_prefix_hits(
                &groups,
                &[
                    KVCacheGroupPrefixHit {
                        group_id: groups[0].group_id,
                        cached_tokens: 160,
                    },
                    KVCacheGroupPrefixHit {
                        group_id: groups[1].group_id,
                        cached_tokens: 96,
                    },
                ],
            ),
            96
        );
        assert_eq!(
            intersect_kv_cache_group_prefix_hits(
                &groups,
                &[KVCacheGroupPrefixHit {
                    group_id: groups[0].group_id,
                    cached_tokens: 160,
                }],
            ),
            0,
            "missing a group hit means the hybrid prefix cannot be reused"
        );
    }

    #[test]
    fn shared_kv_anchor_must_exist_and_match() {
        let missing = vec![LayerKVCacheSpec::full(1, layout(16)).shared_with_anchor(0)];
        assert_eq!(
            validate_layer_kv_cache_specs(&missing),
            Err(KVCacheSpecError::MissingSharedKVAnchor {
                layer_index: 1,
                anchor_layer_index: 0
            })
        );

        let incompatible = vec![
            LayerKVCacheSpec::full(0, layout(16)),
            LayerKVCacheSpec::sliding_window(1, 128, layout(16)).shared_with_anchor(0),
        ];
        assert_eq!(
            validate_layer_kv_cache_specs(&incompatible),
            Err(KVCacheSpecError::SharedKVIncompatible {
                layer_index: 1,
                anchor_layer_index: 0
            })
        );
    }

    #[test]
    fn shared_kv_anchor_cannot_chain_through_alias() {
        let specs = vec![
            LayerKVCacheSpec::full(0, layout(16)),
            LayerKVCacheSpec::full(1, layout(16)).shared_with_anchor(0),
            LayerKVCacheSpec::full(2, layout(16)).shared_with_anchor(1),
        ];

        assert_eq!(
            validate_layer_kv_cache_specs(&specs),
            Err(KVCacheSpecError::SharedKVAnchorIsAlias {
                layer_index: 2,
                anchor_layer_index: 1,
                root_anchor_layer_index: 0
            })
        );
    }
}
