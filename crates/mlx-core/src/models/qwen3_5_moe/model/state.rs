//! Free helpers and value types the Qwen3.5 MoE seams share: cache
//! construction, GDN checkpoint records, prefix-mismatch tracing, and the
//! media/decoder plan helpers.

use super::*;

pub(super) fn fresh_moe_layer_caches(config: &Qwen3_5MoeConfig) -> Vec<Qwen3_5LayerCache> {
    (0..config.num_layers as usize)
        .map(|i| {
            if config.is_linear_layer(i) {
                Qwen3_5LayerCache::new_linear()
            } else {
                Qwen3_5LayerCache::new_full_attention()
            }
        })
        .collect()
}

pub(super) struct MoeGdnPrefixCheckpoint {
    pub(super) owner_id: String,
    pub(super) prefix_len: u32,
    pub(super) block_size: u32,
    pub(super) final_block_hash: u64,
    pub(super) block_hashes: Vec<u64>,
    pub(super) tokens: Vec<u32>,
    pub(super) caches: Vec<Qwen3_5LayerCache>,
}

impl GdnCheckpointLineage for MoeGdnPrefixCheckpoint {
    fn owner_id(&self) -> &str {
        &self.owner_id
    }

    fn prefix_len(&self) -> u32 {
        self.prefix_len
    }

    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn final_block_hash(&self) -> u64 {
        self.final_block_hash
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn block_hashes(&self) -> &[u64] {
        &self.block_hashes
    }
}

pub(super) struct MoeGdnHistoryCheckpoint {
    pub(super) owner_id: String,
    pub(super) image_key: Option<u64>,
    pub(super) tokens: Vec<u32>,
    pub(super) caches: Vec<Qwen3_5LayerCache>,
}

pub(crate) struct MoeGdnPrefixPreparation {
    pub(crate) state: &'static str,
    pub(crate) already_primed: bool,
    pub(crate) restored_prefix_tokens: u32,
    pub(crate) replayed_prefix_tokens: u32,
}

#[derive(Default)]
pub(super) struct MoeGdnCheckpointStoreTrace {
    pub(super) stored: bool,
    pub(super) eval_ms: f64,
    pub(super) clone_ms: f64,
    pub(super) token_clone_ms: f64,
    pub(super) update_ms: f64,
    pub(super) total_ms: f64,
}

impl MoeGdnCheckpointStoreTrace {
    pub(super) fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

#[derive(Clone, Copy)]
pub(super) struct TokenPrefixMismatchTrace {
    pub(super) index: i64,
    pub(super) prompt_token: i64,
    pub(super) cached_token: i64,
}

impl Default for TokenPrefixMismatchTrace {
    fn default() -> Self {
        Self {
            index: -1,
            prompt_token: -1,
            cached_token: -1,
        }
    }
}

pub(super) fn token_prefix_mismatch_trace(
    prompt: &[u32],
    cached: &[u32],
) -> TokenPrefixMismatchTrace {
    let common_len = prompt.len().min(cached.len());
    for i in 0..common_len {
        if prompt[i] != cached[i] {
            return TokenPrefixMismatchTrace {
                index: i as i64,
                prompt_token: prompt[i] as i64,
                cached_token: cached[i] as i64,
            };
        }
    }

    TokenPrefixMismatchTrace {
        index: common_len as i64,
        prompt_token: prompt.get(common_len).map_or(-1, |token| *token as i64),
        cached_token: cached.get(common_len).map_or(-1, |token| *token as i64),
    }
}

pub(super) fn moe_paged_linear_caches_ready(
    config: &Qwen3_5MoeConfig,
    caches: Option<&[Qwen3_5LayerCache]>,
) -> bool {
    let Some(caches) = caches else {
        return false;
    };
    if caches.len() != config.num_layers as usize {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            return false;
        };
        if arrays.get(0).is_none() || arrays.get(1).is_none() {
            return false;
        }
    }
    true
}

pub(super) fn clone_moe_linear_layer_caches(
    config: &Qwen3_5MoeConfig,
    caches: &[Qwen3_5LayerCache],
) -> Option<Vec<Qwen3_5LayerCache>> {
    if !moe_paged_linear_caches_ready(config, Some(caches)) {
        return None;
    }

    let mut cloned = fresh_moe_layer_caches(config);
    for i in 0..config.num_layers as usize {
        if !config.is_linear_layer(i) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = &caches[i] else {
            return None;
        };
        cloned[i] = Qwen3_5LayerCache::Linear(arrays.clone());
    }
    Some(cloned)
}

/// Build the MoE media admission contract from this loaded family's own
/// components. Image execution needs the encoder, processor, and paged KV
/// adapter; incomplete stacks still enter the backend for its precise error.
pub(crate) const fn qwen35_moe_vision_active(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> bool {
    has_vision_encoder && has_image_processor && has_paged_adapter
}

pub(super) const fn qwen35_moe_media_plan(
    has_vision_encoder: bool,
    has_image_processor: bool,
    has_paged_adapter: bool,
) -> MediaPlan {
    let images_available =
        qwen35_moe_vision_active(has_vision_encoder, has_image_processor, has_paged_adapter);
    MediaPlan::with_backend_validation(
        MediaCapabilities {
            images: images_available,
            audio: false,
        },
        MediaCapabilities::IMAGES,
    )
}

/// Media represented by the live MoE session state. Paged text continuations
/// clear the image content key after prefix preparation, while the retained
/// M-RoPE delta continues to prove that the live KV prefix is image-derived.
pub(super) const fn qwen35_moe_session_media(
    has_cached_image_key: bool,
    has_cached_rope_delta: bool,
) -> MediaCapabilities {
    if has_cached_image_key || has_cached_rope_delta {
        MediaCapabilities::IMAGES
    } else {
        MediaCapabilities::NONE
    }
}

pub(super) fn qwen35_moe_session_media_matches_payloads(
    cached_image_key: Option<u64>,
    images: &[Vec<u8>],
    audio: &[Vec<u8>],
) -> bool {
    audio.is_empty()
        && !images.is_empty()
        && cached_image_key == Some(engine::compute_image_cache_key(images))
}

/// Project the engine-selected decoder into the legacy MoE whole-turn config
/// before that core re-extracts `ChatParams`.
pub(super) fn apply_qwen35_moe_planned_decoder(
    config: &mut ChatConfig,
    decoder: DecoderPlan,
) -> bool {
    let planned_mtp = matches!(
        decoder,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    );
    config.enable_mtp = Some(planned_mtp);
    planned_mtp
}
