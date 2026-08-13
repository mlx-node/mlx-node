//! Muse-Glimmer KV-cache specs: pure config in, model-independent specs out.
//!
//! This is the batching seam. It declares what each of the 52 decoder layers
//! needs from a paged KV cache and nothing else — no weights, no MLX arrays, no
//! device. The generic side (`crate::transformer::kv_cache_spec`) groups the
//! specs, sizes admission, and owns block tables.
//!
//! Two facts about this decoder shape the file:
//!
//!   * The attention pattern is `[Sliding, Sliding, Sliding, Full]` x 13, so 39
//!     sliding layers and 13 full ones.
//!   * Head geometry is UNIFORM. One `head_dim` (128) and one
//!     `num_key_value_heads` (2) serve every layer; the checkpoint has no
//!     `global_head_dim` / `global_num_key_value_heads` / `attention_k_eq_v`
//!     overrides. So unlike gemma4 there is no per-kind geometry to resolve, the
//!     physical layout is loop-invariant, and grouping yields exactly TWO groups.
//!
//! What is deliberately NOT here:
//!
//!   * NoPE. The 13 full layers are NoPE (`layer_rope_theta[i] == 0`), and that
//!     is invisible to caching and scheduling. Confirmed in vLLM: nothing under
//!     `vllm/v1/core/` branches on positional encoding, and block hashes are
//!     CHAINED over the prefix, so a cached block is bound to the offset it was
//!     computed at whatever the layer kind. "NoPE" also does not mean
//!     "position-independent" — Llama 4 applies `attn_temperature_tuning` only on
//!     its NoPE layers, as a function of positions. Do not try to exploit
//!     position-independence here or anywhere downstream.
//!   * KV sharing. This checkpoint has no `num_kv_shared_layers` or any
//!     aliasing key, so `shared_kv_anchor` stays `None` on every layer and each
//!     logical layer owns its own physical storage.

use crate::models::muse_glimmer::config::{LayerKind, MuseGlimmerConfig};
use crate::transformer::{
    KVCacheDType, KVCachePhysicalLayout, LayerKVCacheSpec, validate_layer_kv_cache_specs,
};

/// Build Muse-Glimmer's model-independent per-layer KV-cache specs.
///
/// Pure function of the validated config plus the two physical knobs the caller
/// owns (`block_size`, `cache_dtype`). Sliding layers carry the window; full
/// layers carry none. Fails closed on every input that cannot produce a valid
/// physical layout rather than emitting a spec the seam would have to interpret.
pub fn compute_layer_kv_cache_specs(
    config: &MuseGlimmerConfig,
    block_size: u32,
    cache_dtype: KVCacheDType,
) -> std::result::Result<Vec<LayerKVCacheSpec>, String> {
    let text = &config.text_config;

    if block_size == 0 {
        return Err("muse_glimmer KV cache specs require block_size > 0".to_string());
    }

    // Config load refuses both of these already; repeated here because this
    // function is `pub` and must not trust that its caller came through
    // `MuseGlimmerConfig::from_json_str`.
    let sliding_window = u32::try_from(text.sliding_window).map_err(|_| {
        format!(
            "muse_glimmer KV cache specs: sliding_window {} does not fit in a u32",
            text.sliding_window
        )
    })?;
    if sliding_window == 0 {
        return Err(
            "muse_glimmer KV cache specs require sliding_window > 0; a 0 window would \
             reach the seam as SlidingWindow { sliding_window: 0 } and widen 39 layers \
             to full attention"
                .to_string(),
        );
    }

    // ONE layout for all 52 layers. This is loop-invariant precisely because the
    // geometry is uniform — see the module docs. Do not move it inside the loop
    // "for symmetry with gemma4": that would invite a per-kind override this
    // checkpoint does not have.
    let head_size = u32::try_from(text.head_dim).map_err(|_| {
        format!(
            "muse_glimmer KV cache specs: head_dim {} does not fit in a u32",
            text.head_dim
        )
    })?;
    let num_kv_heads = u32::try_from(text.num_key_value_heads).map_err(|_| {
        format!(
            "muse_glimmer KV cache specs: num_key_value_heads {} does not fit in a u32",
            text.num_key_value_heads
        )
    })?;
    let layout = KVCachePhysicalLayout::new(block_size, num_kv_heads, head_size, cache_dtype);
    if !layout.is_valid() {
        return Err(format!(
            "muse_glimmer KV cache specs: invalid physical layout \
             block_size={block_size}, num_kv_heads={num_kv_heads}, head_size={head_size}"
        ));
    }

    // `from_json_str` guarantees this, but the loop below indexes `layer_kinds`
    // for every layer, so the arity is checked rather than assumed: a `pub` entry
    // point must fail closed on a config assembled elsewhere, never panic.
    if text.layer_kinds.len() != text.num_hidden_layers {
        return Err(format!(
            "muse_glimmer KV cache specs: layer_kinds has {} entries but \
             num_hidden_layers is {}",
            text.layer_kinds.len(),
            text.num_hidden_layers
        ));
    }

    let mut specs = Vec::with_capacity(text.num_hidden_layers);
    for layer_index in 0..text.num_hidden_layers {
        let Some(kind) = text.layer_kinds.get(layer_index) else {
            return Err(format!(
                "muse_glimmer KV cache specs: layer_kinds has no entry for layer \
                 {layer_index}"
            ));
        };
        // No `shared_with_anchor` branch: this decoder has no KV-shared layers.
        specs.push(match kind {
            LayerKind::Full => LayerKVCacheSpec::full(layer_index, layout),
            LayerKind::Sliding => {
                LayerKVCacheSpec::sliding_window(layer_index, sliding_window, layout)
            }
        });
    }

    validate_layer_kv_cache_specs(&specs)
        .map_err(|e| format!("muse_glimmer KV cache specs failed validation: {e}"))?;
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::muse_glimmer::config::fixtures::{config_json, text_config_json};
    use crate::transformer::AttentionKind;

    /// The checkpoint-shaped config, taken through the real parser rather than
    /// hand-built: these tests must run on a config that actually validated, so
    /// they cannot silently describe a model `from_json_str` would reject.
    fn checkpoint_config() -> MuseGlimmerConfig {
        MuseGlimmerConfig::from_json_str(&config_json(&text_config_json(52)))
            .expect("the checkpoint-shaped fixture must parse and validate")
    }

    fn specs(config: &MuseGlimmerConfig) -> Vec<LayerKVCacheSpec> {
        compute_layer_kv_cache_specs(config, 16, KVCacheDType::BFloat16)
            .expect("checkpoint-shaped specs must derive")
    }

    /// The full-layer index set, pinned independently in
    /// `config::tests::full_attention_layers_are_exactly_every_fourth_counted_from_the_last`.
    const FULL_LAYERS: [usize; 13] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51];

    /// Head geometry is uniform across kinds, so there is exactly ONE physical
    /// layout for all 52 layers — the property that makes grouping produce two
    /// groups instead of four. gemma4 needs `effective_head_dim(is_global)` and
    /// `effective_kv_heads(is_global)` because its full layers use a larger head
    /// dim and (with `attention_k_eq_v`) more KV heads; this checkpoint has no
    /// such override keys, so a per-kind geometry here would be invented.
    #[test]
    fn every_layer_shares_one_physical_layout_because_head_geometry_is_uniform() {
        let cfg = checkpoint_config();
        let specs = specs(&cfg);
        assert_eq!(specs.len(), 52);

        let expected = KVCachePhysicalLayout::new(16, 2, 128, KVCacheDType::BFloat16);
        for spec in &specs {
            assert_eq!(
                spec.physical_layout, expected,
                "layer {} must use the single uniform layout (block_size 16, \
                 num_kv_heads 2, head_size 128, bf16)",
                spec.layer_index
            );
        }
    }

    /// The KV geometry comes from `num_key_value_heads` (2) and `head_dim` (128),
    /// never from `num_attention_heads` (32) or `hidden_size` (6656). This
    /// decoder is 16x GQA and its `head_dim * num_attention_heads` (4096) is not
    /// `hidden_size`, so both wrong sources are locally plausible and would
    /// oversize the cache 16x and 1.6x respectively.
    #[test]
    fn layout_uses_kv_heads_and_head_dim_not_query_heads_or_hidden_size() {
        let cfg = checkpoint_config();
        let layout = specs(&cfg)[0].physical_layout;
        let text = &cfg.text_config;

        assert_eq!(layout.num_kv_heads as usize, text.num_key_value_heads);
        assert_eq!(layout.head_size as usize, text.head_dim);
        assert_ne!(layout.num_kv_heads as usize, text.num_attention_heads);
        assert_ne!(layout.head_size as usize, text.hidden_size);
    }

    /// Kind assignment follows `layer_kinds`, and the window payload is the
    /// config's `sliding_window`. Swapping the two arms, or dropping the window
    /// onto the full layers, is the whole correctness content of this function.
    #[test]
    fn sliding_layers_carry_the_window_and_full_layers_carry_none() {
        let cfg = checkpoint_config();
        let specs = specs(&cfg);
        let window = cfg.text_config.sliding_window as u32;
        assert_eq!(window, 2048);

        for spec in &specs {
            let expected = match cfg.text_config.layer_kinds[spec.layer_index] {
                LayerKind::Full => AttentionKind::Full,
                LayerKind::Sliding => AttentionKind::SlidingWindow {
                    sliding_window: window,
                },
            };
            assert_eq!(
                spec.attention_kind, expected,
                "layer {} kind mismatch",
                spec.layer_index
            );
        }

        let full: Vec<usize> = specs
            .iter()
            .filter(|spec| spec.attention_kind == AttentionKind::Full)
            .map(|spec| spec.layer_index)
            .collect();
        assert_eq!(full, FULL_LAYERS.to_vec());
        assert_eq!(specs.len() - full.len(), 39);
    }

    /// Specs are emitted in layer order, one per layer, with no gaps. Downstream
    /// route derivation is keyed on `layer_index`, so a permuted or short vector
    /// would misroute rather than error.
    #[test]
    fn specs_cover_every_layer_exactly_once_in_order() {
        let cfg = checkpoint_config();
        let indices: Vec<usize> = specs(&cfg).iter().map(|spec| spec.layer_index).collect();
        assert_eq!(indices, (0..52).collect::<Vec<usize>>());
    }

    /// No layer aliases another's KV storage: this checkpoint carries no
    /// `num_kv_shared_layers` and no aliasing key of any kind. An accidental
    /// anchor would silently drop a layer's cache and make it read a sibling's.
    #[test]
    fn no_layer_shares_kv_storage_with_an_anchor() {
        let cfg = checkpoint_config();
        for spec in &specs(&cfg) {
            assert_eq!(
                spec.shared_kv_anchor, None,
                "layer {} must own its own physical KV storage",
                spec.layer_index
            );
            assert_eq!(spec.physical_layer_index(), spec.layer_index);
        }
    }

    #[test]
    fn refuses_a_zero_block_size() {
        let cfg = checkpoint_config();
        let err = compute_layer_kv_cache_specs(&cfg, 0, KVCacheDType::BFloat16)
            .expect_err("a zero block_size cannot produce a valid layout");
        assert!(
            err.contains("block_size"),
            "the error must name block_size, got: {err}"
        );
    }

    /// `from_json_str` guarantees `layer_kinds.len() == num_hidden_layers`, but
    /// this function must not depend on that to stay in bounds — it is `pub`, and
    /// a caller can hand it a config assembled some other way. A raw
    /// `layer_kinds[i]` over `0..num_hidden_layers` would panic here instead.
    #[test]
    fn refuses_a_layer_kinds_table_shorter_than_num_hidden_layers() {
        let mut cfg = checkpoint_config();
        cfg.text_config.layer_kinds.pop();
        let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect_err("a short layer_kinds table must fail closed, not panic");
        assert!(
            err.contains("layer_kinds") && err.contains("51"),
            "the error must name the table and its observed length, got: {err}"
        );
    }

    /// Geometry crosses into the seam as `u32`. An `as u32` cast would turn
    /// `usize::MAX` into 4_294_967_295 — still "valid" to `is_valid()`, and a
    /// cache sized from garbage.
    #[test]
    fn refuses_geometry_that_does_not_fit_a_u32() {
        let mut cfg = checkpoint_config();
        cfg.text_config.head_dim = usize::MAX;
        let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect_err("an out-of-u32 head_dim must fail closed");
        assert!(
            err.contains("head_dim"),
            "the error must name head_dim, got: {err}"
        );

        let mut cfg = checkpoint_config();
        cfg.text_config.num_key_value_heads = usize::MAX;
        let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect_err("an out-of-u32 num_key_value_heads must fail closed");
        assert!(
            err.contains("num_key_value_heads"),
            "the error must name num_key_value_heads, got: {err}"
        );
    }

    /// `KVCachePhysicalLayout::is_valid()` also rejects a zero head/KV count.
    /// `from_json_str` refuses both already, so this is belt-and-braces for a
    /// config that did not come through it.
    #[test]
    fn refuses_a_degenerate_layout_from_a_config_that_skipped_validation() {
        let mut cfg = checkpoint_config();
        cfg.text_config.num_key_value_heads = 0;
        let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect_err("a zero num_key_value_heads cannot produce a valid layout");
        assert!(
            err.contains("num_kv_heads=0") || err.contains("num_key_value_heads"),
            "the error must name the offending count, got: {err}"
        );
    }

    /// The window reaching a spec must be the config's, so a config-level 0
    /// cannot arrive as a `SlidingWindow { sliding_window: 0 }` spec. Config load
    /// refuses it first (`config::tests::rejects_a_zero_sliding_window`); this
    /// pins the second line of defence for a config built without that parser.
    #[test]
    fn refuses_a_zero_sliding_window_even_though_config_load_already_did() {
        let mut cfg = checkpoint_config();
        cfg.text_config.sliding_window = 0;
        let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect_err("a zero window must never reach a SlidingWindow spec");
        assert!(
            err.contains("sliding_window"),
            "the error must name sliding_window, got: {err}"
        );
    }
}
