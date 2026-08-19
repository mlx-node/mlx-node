//! SSD sidecar identity for Muse-Glimmer's paged sliding-attention group.
//!
//! The full-attention group owns the content-addressed K/V chain. A restore is
//! usable only when the sliding group has state at the same boundary, so its
//! live window is persisted as a co-keyed `SlidingWindow` sidecar.

use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdSidecarPolicy};

use crate::models::gemma4::sliding_sidecar::{SlidingSidecarGeometry, policy_for_geometry};

use super::config::{LayerKind, MuseGlimmerConfig};

const ANCHOR_RATIO: u32 = 4;
const MAX_ANCHORS: usize = 5;

pub(crate) fn geometry(config: &MuseGlimmerConfig) -> Option<SlidingSidecarGeometry> {
    let physical_layers = u32::try_from(
        config
            .text_config
            .layer_kinds
            .iter()
            .filter(|kind| **kind == LayerKind::Sliding)
            .count(),
    )
    .ok()?;
    let window = u32::try_from(config.text_config.sliding_window).ok()?;
    let kv_heads = u32::try_from(config.text_config.num_key_value_heads).ok()?;
    let head_dim = u32::try_from(config.text_config.head_dim).ok()?;
    if physical_layers == 0 || window == 0 || kv_heads == 0 || head_dim == 0 {
        return None;
    }
    Some(SlidingSidecarGeometry {
        window,
        physical_layers,
        kv_heads,
        head_dim,
    })
}

pub(crate) fn policy(config: &MuseGlimmerConfig) -> Option<ColdSidecarPolicy> {
    policy_for_geometry(&geometry(config)?)
}

/// Stable block-aligned checkpoints retained while a turn is live.
///
/// Anchoring the ladder at zero makes the same sidecar reusable by later
/// prompts that share the prefix. Including the first block lets short chat
/// prompts benefit; deeper rungs cover the 128-block default SSD capture walk.
pub(crate) fn anchor_rungs(block_size: u32) -> Vec<u32> {
    if block_size == 0 {
        return Vec::new();
    }
    let mut rungs = Vec::with_capacity(MAX_ANCHORS);
    let mut rung = block_size;
    for _ in 0..MAX_ANCHORS {
        rungs.push(rung);
        let Some(next) = rung.checked_mul(ANCHOR_RATIO) else {
            break;
        };
        rung = next;
    }
    rungs
}

/// Derive the sidecar key using the same parent-linked block chain as the K/V
/// restore walk, but under the `SlidingWindow` domain tag.
pub(crate) fn chain_key(
    fingerprint: ColdCacheFingerprint,
    request_tokens: &[u32],
    extra_keys_per_block: &[Vec<u64>],
    block_size: u32,
    boundary: u32,
    cache_salt: u64,
) -> Option<ColdCacheKey> {
    if block_size == 0 || boundary == 0 || !boundary.is_multiple_of(block_size) {
        return None;
    }
    let blocks = boundary as usize / block_size as usize;
    let mut parent = None;
    for index in 0..blocks {
        let extra_keys = extra_keys_per_block.get(index)?;
        let start = index.checked_mul(block_size as usize)?;
        let end = start.checked_add(block_size as usize)?;
        let tokens = request_tokens.get(start..end)?;
        parent = Some(ColdCacheKey::chain(
            ColdGroup::SlidingWindow,
            fingerprint,
            parent,
            tokens,
            extra_keys,
            cache_salt,
            index,
        ));
    }
    parent
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::muse_glimmer::config::MuseGlimmerConfig;
    use crate::models::muse_glimmer::config::fixtures::{config_json, text_config_json};

    #[test]
    fn checkpoint_geometry_covers_every_sliding_layer() {
        let config = MuseGlimmerConfig::from_json_str(&config_json(&text_config_json(52)))
            .expect("checkpoint config");
        let geometry = geometry(&config).expect("hybrid sidecar geometry");
        assert_eq!(geometry.window, 2048);
        assert_eq!(geometry.physical_layers, 39);
        assert_eq!(geometry.kv_heads, 2);
        assert_eq!(geometry.head_dim, 128);
        assert!(policy(&config).is_some());
    }

    #[test]
    fn anchor_ladder_is_stable_and_block_aligned() {
        assert_eq!(anchor_rungs(16), vec![16, 64, 256, 1024, 4096]);
        assert!(anchor_rungs(0).is_empty());
    }
}
