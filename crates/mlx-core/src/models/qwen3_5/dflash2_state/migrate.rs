//! The one-time move of the stored chats of a block-16 store into a load
//! whose key pages are 32 tokens: the source store's identity, computed
//! from the load (never found by a directory search), and the converter
//! that re-packs the source's objects into the new page (Splash
//! `paged_store_row.h`: K token-major, V dimension-major).

use std::collections::HashSet;

use mlx_paged_attn::{
    ColdCacheBlock, ColdCacheFingerprint, ColdCacheKey, ColdCacheLayout, ColdGroup, ColdLayerBlock,
    ColdStateLabel, ColdStateLayout, ColdStateObject,
};

use crate::disk_tier::PersistentIdentity;

use super::disk::{TierLoad, lineage_keys, tier_identity};
use super::store::Lineage;

/// The page of the source store.
pub(crate) const SOURCE_BLOCK: u32 = 16;

/// The cache identity text of the source store's pool.
pub(crate) const SOURCE_CACHE_IDENTITY: &str = "Int8";

/// The page of the new store: two source blocks.
pub(crate) const PAGE_BLOCK: u32 = 32;

/// The `cache_dtype` of an int8 block's layout.
const INT8_LAYOUT: &str = "Int8";

/// Int8 elements per vLLM K pack (`x` of `[KV][D/x][block][x]`).
const KEY_PACK: usize = 16;

/// Bytes of one f32 scale.
const SCALE_BYTES: usize = 4;

/// The persistent identity of the store the same load wrote with a
/// block-16 pool of identity `Int8`: the load's own files and config bytes
/// as they are hashed, only the page and the identity text changed.
pub(crate) fn source_identity(load: &TierLoad<'_>) -> Option<PersistentIdentity> {
    tier_identity(load, SOURCE_BLOCK, SOURCE_CACHE_IDENTITY)
}

/// The fingerprints a conversion reads and writes under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fingerprints {
    /// The source store's KV blocks.
    pub(crate) source_kv: ColdCacheFingerprint,
    /// The new store's pages and fragments.
    pub(crate) kv: ColdCacheFingerprint,
    /// The new store's state objects.
    pub(crate) state: ColdCacheFingerprint,
}

/// One page of the new store and the two source blocks it is made of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PagePlan {
    pub(crate) key: ColdCacheKey,
    pub(crate) sources: [ColdCacheKey; 2],
    /// The page's first row in the state.
    pub(crate) row: usize,
}

/// The 16 rows of a state past its last whole page, from the state's last
/// source block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FragmentPlan {
    pub(crate) key: ColdCacheKey,
    pub(crate) source: ColdCacheKey,
    pub(crate) row: usize,
}

/// What one source state becomes in the new store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StatePlan {
    pub(crate) rows: usize,
    /// The label's image keys per 32-token page (the part page included).
    pub(crate) extra_keys: Vec<Vec<u64>>,
    pub(crate) pages: Vec<PagePlan>,
    pub(crate) fragment: Option<FragmentPlan>,
    pub(crate) state_key: ColdCacheKey,
}

pub(crate) use crate::disk_tier::state_disk::part_page_keys;

/// The image keys of a 16-token-block lineage of `rows` rows per 32-token
/// page (`engine::build_paged_extra_keys` at block 32): per page the pairs
/// of its first half, then those of its second half with their positions
/// moved by 16, behind the one prefix both halves carry. `None` for keys
/// not in that form.
fn page_extra_keys(source: &[Vec<u64>], rows: usize) -> Option<Vec<Vec<u64>>> {
    let source_block = SOURCE_BLOCK as usize;
    let halves = rows / source_block;
    if source.len() > halves {
        return None;
    }
    let mut pages: Vec<Vec<u64>> = (0..halves.div_ceil(2))
        .map(|page| {
            let mut prefix: Option<[u64; 2]> = None;
            let mut pairs = Vec::new();
            for half in 0..2 {
                let Some(keys) = source.get(page * 2 + half).filter(|keys| !keys.is_empty()) else {
                    continue;
                };
                let (head, rest) = keys.split_first_chunk::<2>()?;
                if rest.is_empty() || prefix.is_some_and(|prefix| prefix != *head) {
                    return None;
                }
                prefix = Some(*head);
                let (chunks, odd) = rest.as_chunks::<2>();
                if !odd.is_empty() {
                    return None;
                }
                let shift = (half * source_block) as u64;
                for &[hash, position] in chunks {
                    if position >= source_block as u64 {
                        return None;
                    }
                    pairs.extend([hash, position + shift]);
                }
            }
            Some(prefix.map_or_else(Vec::new, |prefix| prefix.into_iter().chain(pairs).collect()))
        })
        .collect::<Option<_>>()?;
    while pages.last().is_some_and(Vec::is_empty) {
        pages.pop();
    }
    Some(pages)
}

/// The pages, fragment and state key a source state of `label` becomes.
/// `None` when its rows are not whole source blocks or its image keys are
/// not in the source's form.
pub(crate) fn plan_state(label: &ColdStateLabel, fingerprints: &Fingerprints) -> Option<StatePlan> {
    let rows = label.tokens.len();
    let source_block = SOURCE_BLOCK as usize;
    if rows == 0 || !rows.is_multiple_of(source_block) {
        return None;
    }
    let source = lineage_keys(
        ColdGroup::Kv,
        fingerprints.source_kv,
        Lineage {
            tokens: &label.tokens,
            extra_keys: &label.extra_keys,
            salt: label.salt,
        },
        SOURCE_BLOCK,
    );
    let extra_keys = page_extra_keys(&label.extra_keys, rows)?;
    let lineage = Lineage {
        tokens: &label.tokens,
        extra_keys: &extra_keys,
        salt: label.salt,
    };
    let (page_keys, fragment_key) =
        part_page_keys(ColdGroup::Kv, fingerprints.kv, lineage, PAGE_BLOCK);
    let (state_pages, state_part) = part_page_keys(
        ColdGroup::DFlash2State,
        fingerprints.state,
        lineage,
        PAGE_BLOCK,
    );
    let state_key = state_part.or_else(|| state_pages.last().copied())?;
    let page = PAGE_BLOCK as usize;
    let pages = page_keys
        .iter()
        .enumerate()
        .map(|(index, &key)| {
            Some(PagePlan {
                key,
                sources: [*source.get(2 * index)?, *source.get(2 * index + 1)?],
                row: index * page,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let fragment = match fragment_key {
        Some(key) => Some(FragmentPlan {
            key,
            source: *source.last()?,
            row: pages.len() * page,
        }),
        None => None,
    };
    Some(StatePlan {
        rows,
        extra_keys,
        pages,
        fragment,
        state_key,
    })
}

/// `(num_kv_heads, head_size)` of an int8 source block's layout.
fn source_geometry(layout: &ColdCacheLayout) -> Option<(usize, usize)> {
    if layout.block_size != SOURCE_BLOCK || layout.cache_dtype != INT8_LAYOUT {
        return None;
    }
    let heads = layout.num_kv_heads as usize;
    let dim = layout.head_size as usize;
    if heads == 0 || dim == 0 || !dim.is_multiple_of(KEY_PACK) {
        return None;
    }
    let rows = SOURCE_BLOCK as usize;
    let side = heads
        .checked_mul(dim)?
        .checked_mul(rows)?
        .checked_add(heads.checked_mul(rows)?.checked_mul(SCALE_BYTES)?)?;
    (layout.key_bytes_per_layer == side && layout.value_bytes_per_layer == side)
        .then_some((heads, dim))
}

/// The layout of an int8 block of the source store of a pool with
/// `num_layers` layers and `num_kv_heads` heads of `head_size`.
pub(crate) fn source_layout(
    num_layers: u32,
    num_kv_heads: u32,
    head_size: u32,
) -> Option<ColdCacheLayout> {
    let rows = SOURCE_BLOCK as usize;
    let heads = num_kv_heads as usize;
    let side = heads
        .checked_mul(head_size as usize)?
        .checked_mul(rows)?
        .checked_add(heads.checked_mul(rows)?.checked_mul(SCALE_BYTES)?)?;
    let layout = ColdCacheLayout {
        block_size: SOURCE_BLOCK,
        num_layers,
        num_kv_heads,
        head_size,
        cache_dtype: INT8_LAYOUT.to_string(),
        key_bytes_per_layer: side,
        value_bytes_per_layer: side,
    };
    source_geometry(&layout).map(|_| layout)
}

/// The layout of the page two blocks of `source` make.
pub(crate) fn page_layout(source: &ColdCacheLayout) -> ColdCacheLayout {
    ColdCacheLayout {
        block_size: PAGE_BLOCK,
        key_bytes_per_layer: source.key_bytes_per_layer * 2,
        value_bytes_per_layer: source.value_bytes_per_layer * 2,
        ..source.clone()
    }
}

/// One block of `16 x halves.len()` rows from consecutive int8 source
/// blocks: K `[KV][rows][D]` token-major from the vLLM pack
/// `[KV][D/16][16][16]`, V `[KV][D][rows]` from `[KV][D][16]`, each side's
/// f32 scales `[KV][rows]` from `[KV][16]` (blob order K, K scales, V,
/// V scales). `None` unless every half has the same int8 block-16 layout
/// and the halves' tokens are `tokens`.
fn repack(
    halves: &[&ColdCacheBlock],
    key: ColdCacheKey,
    fingerprint: ColdCacheFingerprint,
    tokens: &[u32],
) -> Option<ColdCacheBlock> {
    let source = &halves.first()?.layout;
    let (heads, dim) = source_geometry(source)?;
    let half_rows = SOURCE_BLOCK as usize;
    let rows = half_rows.checked_mul(halves.len())?;
    let layers = source.num_layers as usize;
    let joined: Vec<u32> = halves
        .iter()
        .flat_map(|half| half.tokens.iter().copied())
        .collect();
    if joined != tokens
        || halves.iter().any(|half| {
            half.layout != *source
                || half.layers.len() != layers
                || half.layers.iter().any(|layer| {
                    layer.keys.len() != source.key_bytes_per_layer
                        || layer.values.len() != source.value_bytes_per_layer
                })
        })
    {
        return None;
    }
    let source_plane = heads * dim * half_rows;
    let plane = heads * dim * rows;
    let side = plane + heads * rows * SCALE_BYTES;
    let packs = dim / KEY_PACK;
    let out = (0..layers)
        .map(|index| {
            let mut keys = vec![0u8; side];
            let mut values = vec![0u8; side];
            for (half, block) in halves.iter().enumerate() {
                let layer = &block.layers[index];
                let (key_plane, key_scales) = layer.keys.split_at(source_plane);
                let (value_plane, value_scales) = layer.values.split_at(source_plane);
                for head in 0..heads {
                    for row in 0..half_rows {
                        let token = half * half_rows + row;
                        let at = (head * rows + token) * dim;
                        for (pack, chunk) in keys[at..at + dim]
                            .as_chunks_mut::<KEY_PACK>()
                            .0
                            .iter_mut()
                            .enumerate()
                        {
                            let from = ((head * packs + pack) * half_rows + row) * KEY_PACK;
                            chunk.copy_from_slice(&key_plane[from..from + KEY_PACK]);
                        }
                        let from = (head * half_rows + row) * SCALE_BYTES;
                        let to = plane + (head * rows + token) * SCALE_BYTES;
                        keys[to..to + SCALE_BYTES]
                            .copy_from_slice(&key_scales[from..from + SCALE_BYTES]);
                        values[to..to + SCALE_BYTES]
                            .copy_from_slice(&value_scales[from..from + SCALE_BYTES]);
                    }
                    for d in 0..dim {
                        let from = (head * dim + d) * half_rows;
                        let to = (head * dim + d) * rows + half * half_rows;
                        values[to..to + half_rows]
                            .copy_from_slice(&value_plane[from..from + half_rows]);
                    }
                }
            }
            ColdLayerBlock { keys, values }
        })
        .collect();
    Some(ColdCacheBlock {
        key,
        fingerprint,
        tokens: tokens.to_vec(),
        layout: ColdCacheLayout {
            block_size: u32::try_from(rows).ok()?,
            key_bytes_per_layer: side,
            value_bytes_per_layer: side,
            ..source.clone()
        },
        layers: out,
    })
}

/// The page `plan` names, from its two source blocks as read; `tokens` are
/// the state's.
pub(crate) fn convert_page(
    plan: &PagePlan,
    tokens: &[u32],
    first: &ColdCacheBlock,
    second: &ColdCacheBlock,
    fingerprints: &Fingerprints,
) -> Option<ColdCacheBlock> {
    if [first.key, second.key] != plan.sources
        || first.fingerprint != fingerprints.source_kv
        || second.fingerprint != fingerprints.source_kv
    {
        return None;
    }
    let rows = tokens.get(plan.row..plan.row.checked_add(PAGE_BLOCK as usize)?)?;
    repack(&[first, second], plan.key, fingerprints.kv, rows)
}

/// The fragment `plan` names: the 16 rows of `tail` with K token-major
/// (`[KV][16][D]`), V and scales as stored; `tokens` are the state's.
pub(crate) fn convert_fragment(
    plan: &FragmentPlan,
    tokens: &[u32],
    tail: &ColdCacheBlock,
    fingerprints: &Fingerprints,
) -> Option<ColdCacheBlock> {
    if tail.key != plan.source || tail.fingerprint != fingerprints.source_kv {
        return None;
    }
    let rows = tokens.get(plan.row..plan.row.checked_add(SOURCE_BLOCK as usize)?)?;
    repack(&[tail], plan.key, fingerprints.kv, rows)
}

/// The source state object of `label` under the new store's key and
/// fingerprint: the same layout, tokens, salt, stamp, class and payload,
/// its image keys per 32-token page. `plan` is [`plan_state`] of `label`.
pub(crate) fn convert_state(
    plan: &StatePlan,
    layout: ColdStateLayout,
    label: ColdStateLabel,
    payload: Vec<u8>,
    fingerprints: &Fingerprints,
) -> Option<ColdStateObject> {
    if label.tokens.len() != plan.rows
        || layout.boundary_tokens as usize != plan.rows
        || layout.payload_bytes() != Some(payload.len() as u64)
    {
        return None;
    }
    Some(ColdStateObject {
        key: plan.state_key,
        fingerprint: fingerprints.state,
        layout,
        label: ColdStateLabel {
            extra_keys: plan.extra_keys.clone(),
            ..label
        },
        payload,
    })
}

/// The source block each state of `labels` ends on past its last whole
/// page. A newer state may hold that block as the first half of one of its
/// pages; the older state's fragment still needs it, so it stays in the
/// source until the end.
pub(crate) fn tail_keys<'a>(
    labels: impl IntoIterator<Item = &'a ColdStateLabel>,
    source_kv: ColdCacheFingerprint,
) -> HashSet<ColdCacheKey> {
    let source_block = SOURCE_BLOCK as usize;
    labels
        .into_iter()
        .filter(|label| {
            let rows = label.tokens.len();
            rows.is_multiple_of(source_block) && (rows / source_block) % 2 == 1
        })
        .filter_map(|label| {
            lineage_keys(
                ColdGroup::Kv,
                source_kv,
                Lineage {
                    tokens: &label.tokens,
                    extra_keys: &label.extra_keys,
                    salt: label.salt,
                },
                SOURCE_BLOCK,
            )
            .last()
            .copied()
        })
        .collect()
}

pub(crate) mod move_source;

#[cfg(all(test, target_os = "macos"))]
mod move_tests;

#[cfg(test)]
mod source_tests;

#[cfg(all(test, target_os = "macos"))]
mod tests;
