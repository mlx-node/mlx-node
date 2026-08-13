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
    KVCacheDType, KVCacheGroup, KVCachePhysicalLayout, LayerKVCacheSpec,
    group_layer_kv_cache_specs, validate_layer_kv_cache_specs,
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

/// Group Muse-Glimmer's layers into compatible KV cache pools.
///
/// `max_chunk` is the largest number of tokens that can be in flight against the
/// cache at once, and it is what sizes the sliding pool:
/// `min(window - 1 + max_chunk, max_model_len)` rounded up to blocks, plus one
/// for the block the live window straddles. It is NOT `window x max_num_seqs`.
/// Raising the prefill chunk raises this requirement linearly, so the caller must
/// pass the SAME value its prefill loop actually uses — vLLM keeps the pool sizer
/// and the runtime admission gate on one source of truth
/// (`single_type_kv_cache_manager.py`) precisely because drift between them is
/// either a deadlock or a mid-prefill OOM.
pub fn compute_layer_kv_cache_groups(
    config: &MuseGlimmerConfig,
    block_size: u32,
    cache_dtype: KVCacheDType,
    max_chunk: u32,
) -> std::result::Result<Vec<KVCacheGroup>, String> {
    // 0 is not "no tokens in flight". In this repo it is the chunk-size sentinel
    // for "do not chunk; run legacy single-shot prefill"
    // (`crate::array::paged_prefill_chunk_size`), so the effective chunk is the
    // WHOLE prompt. Sizing the sliding pool from a literal 0 would provision the
    // window alone and under-provision by the entire prompt length, which is the
    // failure mode this argument exists to prevent. Make the caller resolve the
    // sentinel instead of guessing which meaning it had.
    if max_chunk == 0 {
        return Err(
            "muse_glimmer KV cache groups require max_chunk > 0; 0 is this repo's \
             \"do not chunk, single-shot prefill\" sentinel, where the in-flight token \
             budget is the whole prompt — pass that budget (max_position_embeddings for \
             an unchunked prefill), not the sentinel"
                .to_string(),
        );
    }

    let specs = compute_layer_kv_cache_specs(config, block_size, cache_dtype)?;
    let max_model_len =
        u32::try_from(config.text_config.max_position_embeddings).map_err(|_| {
            format!(
                "muse_glimmer KV cache groups: max_position_embeddings {} does not fit in a u32",
                config.text_config.max_position_embeddings
            )
        })?;
    group_layer_kv_cache_specs(&specs, max_model_len, max_chunk)
        .map_err(|e| format!("muse_glimmer KV cache grouping failed: {e}"))
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

    // ── Grouping ──────────────────────────────────────────────────────────

    /// gemma4's prefill chunk, used here as a representative in-flight budget.
    /// This family has no chunk constant of its own yet — see the module docs on
    /// `compute_layer_kv_cache_groups`.
    const CHUNK: u32 = 512;

    fn groups(config: &MuseGlimmerConfig, max_chunk: u32) -> Vec<KVCacheGroup> {
        compute_layer_kv_cache_groups(config, 16, KVCacheDType::BFloat16, max_chunk)
            .expect("checkpoint-shaped groups must derive")
    }

    /// TWO groups, not four. The `[S,S,S,F]` x 13 pattern has two attention kinds
    /// and — because head geometry is uniform — exactly one physical layout, and
    /// the grouping key is the PAIR. If this checkpoint ever gained gemma4-style
    /// per-kind geometry the count would rise; if it were keyed on layout alone it
    /// would collapse to one. Both would be wrong here.
    ///
    /// This also structurally satisfies gemma4's ">1 full-attention group"
    /// refusal (`gemma4/model.rs:2100-2107`) with nothing to special-case: one
    /// layout cannot produce two distinct full groups.
    #[test]
    fn the_repeating_pattern_produces_exactly_two_groups_because_one_layout_serves_both_kinds() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);

        assert_eq!(groups.len(), 2);
        let full_groups = groups
            .iter()
            .filter(|group| group.attention_kind == AttentionKind::Full)
            .count();
        assert_eq!(
            full_groups, 1,
            "one uniform layout can only ever yield one full-attention group"
        );
        let layouts: Vec<KVCachePhysicalLayout> =
            groups.iter().map(|group| group.physical_layout).collect();
        assert_eq!(
            layouts[0], layouts[1],
            "both groups must share the one physical layout"
        );
    }

    /// `AttentionKind` declares `Full` before `SlidingWindow` and derives `Ord`,
    /// and grouping enumerates the sorted key order — so group 0 is the 13 full
    /// layers and group 1 the 39 sliding ones. Pinned because downstream code
    /// indexes managers and pools by `group_id`, and the reverse assumption is
    /// just as easy to write.
    #[test]
    fn group_zero_is_full_and_group_one_is_sliding_with_the_thirty_nine_thirteen_split() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);

        assert_eq!(groups[0].group_id, 0);
        assert_eq!(groups[0].attention_kind, AttentionKind::Full);
        assert_eq!(groups[0].layer_indices, FULL_LAYERS.to_vec());

        assert_eq!(groups[1].group_id, 1);
        assert_eq!(
            groups[1].attention_kind,
            AttentionKind::SlidingWindow {
                sliding_window: 2048
            }
        );
        let expected_sliding: Vec<usize> = (0..52).filter(|i| !FULL_LAYERS.contains(i)).collect();
        assert_eq!(groups[1].layer_indices, expected_sliding);
        assert_eq!(groups[1].layer_indices.len(), 39);

        // No aliases, so every logical layer is also a physical owner in both
        // groups. A shared-KV model would show a shorter physical list.
        for group in &groups {
            assert_eq!(group.physical_layer_indices, group.layer_indices);
        }
    }

    /// The grouping key is `(attention_kind, physical_layout)`, not the kind
    /// alone. Two layers of the SAME kind with different layouts must NOT merge:
    /// one group means one pool with one per-block byte size, so merging them
    /// would hand half the members blocks of the wrong stride.
    ///
    /// Muse-Glimmer cannot produce this shape today (uniform geometry), so the
    /// case is built by rewriting derived specs. It is the invariant the two-group
    /// claim above rests on: `uniform layout => 2 groups` is only true while the
    /// layout is half the key.
    #[test]
    fn grouping_keys_on_kind_and_layout_together_not_on_kind_alone() {
        let cfg = checkpoint_config();
        let mut specs = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect("checkpoint-shaped specs must derive");

        // Ten of the 39 sliding layers move to a coarser block size. Same kind,
        // same window, different layout.
        let coarse = KVCachePhysicalLayout::new(32, 2, 128, KVCacheDType::BFloat16);
        for spec in specs
            .iter_mut()
            .filter(|spec| matches!(spec.attention_kind, AttentionKind::SlidingWindow { .. }))
            .take(10)
        {
            spec.physical_layout = coarse;
        }

        let groups = group_layer_kv_cache_specs(&specs, 131_072, CHUNK)
            .expect("regrouping the rewritten specs must succeed");

        assert_eq!(
            groups.len(),
            3,
            "one full group plus TWO sliding groups; a kind-only key would give 2"
        );
        let sliding: Vec<&KVCacheGroup> = groups
            .iter()
            .filter(|group| matches!(group.attention_kind, AttentionKind::SlidingWindow { .. }))
            .collect();
        assert_eq!(sliding.len(), 2);
        assert_eq!(sliding[0].attention_kind, sliding[1].attention_kind);
        assert_ne!(sliding[0].physical_layout, sliding[1].physical_layout);
        assert_eq!(
            sliding[0].layer_indices.len() + sliding[1].layer_indices.len(),
            39,
            "the split must partition the sliding layers, not duplicate them"
        );
    }

    /// The whole prize, in absolute numbers rather than a ratio:
    ///   full    = div_ceil(131072, 16)                    = 8192 blocks
    ///   sliding = div_ceil(min(2047 + 512, 131072), 16) + 1 = 161 blocks
    /// A 51x smaller pool on 39 of 52 layers. Both numbers are hand-computed from
    /// vLLM's formula (`kv_cache_interface.py`, `SlidingWindowSpec`), so a change
    /// to either the cap order or the `+1` shows up here.
    #[test]
    fn admission_blocks_match_the_hand_computed_vllm_formula() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);

        assert_eq!(groups[0].max_admission_blocks, 8192);
        assert_eq!(groups[1].max_admission_blocks, 161);
    }

    /// Sliding admission is a function of the in-flight token budget, NOT of
    /// window x max_num_seqs. Raising `max_chunk` raises it linearly and leaves
    /// full attention untouched — which is exactly why a caller that raises its
    /// prefill chunk without re-deriving the pool under-provisions it.
    #[test]
    fn raising_max_chunk_raises_the_sliding_admission_and_never_the_full_one() {
        let cfg = checkpoint_config();

        let small = groups(&cfg, 512);
        let medium = groups(&cfg, 2048);
        let large = groups(&cfg, 8192);

        assert_eq!(
            (
                small[1].max_admission_blocks,
                medium[1].max_admission_blocks,
                large[1].max_admission_blocks
            ),
            (161, 257, 641),
            "sliding admission must track min(window - 1 + max_chunk, max_model_len)"
        );
        assert!(
            small[1].max_admission_blocks < medium[1].max_admission_blocks
                && medium[1].max_admission_blocks < large[1].max_admission_blocks
        );
        assert_eq!(
            (
                small[0].max_admission_blocks,
                medium[0].max_admission_blocks,
                large[0].max_admission_blocks
            ),
            (8192, 8192, 8192),
            "full attention is bounded by max_model_len alone"
        );
    }

    /// A 0 `max_chunk` must not be treated as "no tokens in flight". In this repo
    /// 0 is the chunk-size sentinel for "do not chunk; run legacy single-shot
    /// prefill" (`crate::array::paged_prefill_chunk_size`), so the effective chunk
    /// is the WHOLE prompt. Passing the sentinel through would size the sliding
    /// pool for the window alone — 129 blocks instead of up to 8193 — and
    /// under-provision it by the entire prompt length.
    #[test]
    fn refuses_a_zero_max_chunk_because_zero_means_single_shot_prefill() {
        let cfg = checkpoint_config();
        let err = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, 0)
            .expect_err("the chunk sentinel must not be sized as a chunk of zero");
        assert!(
            err.contains("max_chunk"),
            "the error must name max_chunk, got: {err}"
        );

        // What the sentinel means instead: one chunk covering the whole context.
        let single_shot = groups(&cfg, 131_072);
        assert_eq!(single_shot[1].max_admission_blocks, 8193);
    }

    /// `max_model_len` comes from the config's `max_position_embeddings`, and it
    /// crosses into the seam as a `u32`. Config load refuses an out-of-range
    /// value; this pins the wrapper's own conversion for a config built elsewhere.
    #[test]
    fn refuses_a_max_position_embeddings_that_does_not_fit_a_u32() {
        let mut cfg = checkpoint_config();
        cfg.text_config.max_position_embeddings = usize::MAX;
        let err = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, CHUNK)
            .expect_err("an out-of-u32 context length must fail closed");
        assert!(
            err.contains("max_position_embeddings"),
            "the error must name max_position_embeddings, got: {err}"
        );
    }

    /// `max_model_len` must be the context length, not some other config number
    /// that happens to be in range. 131072 is the only value that yields 8192
    /// full blocks; `sliding_window` (2048) would yield 128, `vocab_size`
    /// (202048) 12628, `hidden_size` (6656) 416.
    #[test]
    fn max_model_len_is_the_context_length_and_not_another_config_field() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);
        let text = &cfg.text_config;

        assert_eq!(
            groups[0].max_admission_blocks as usize,
            text.max_position_embeddings.div_ceil(16)
        );
        for other in [text.sliding_window, text.vocab_size, text.hidden_size] {
            assert_ne!(groups[0].max_admission_blocks as usize, other.div_ceil(16));
        }
    }

    /// The specs compose with the seam's route derivation, which is what a
    /// forward pass will actually consume. With no aliases the ordinal is simply
    /// the layer's position within its group — layer 3 is the first full layer,
    /// layer 51 the thirteenth; layer 4 is the fourth sliding layer because 3 is
    /// not in that group.
    #[test]
    fn routes_give_each_layer_its_position_within_its_own_group() {
        let cfg = checkpoint_config();
        let specs = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect("checkpoint-shaped specs must derive");
        let routes = crate::transformer::derive_layer_kv_cache_routes(&specs, 131_072, CHUNK)
            .expect("routes must derive from the checkpoint specs");

        assert_eq!(routes.len(), 52);
        for route in &routes {
            assert_eq!(route.shared_kv_anchor, None);
            assert_eq!(route.physical_layer_index, route.layer_index);
        }

        assert_eq!(
            (routes[3].group_id, routes[3].physical_layer_ordinal),
            (0, 0)
        );
        assert_eq!(
            (routes[51].group_id, routes[51].physical_layer_ordinal),
            (0, 12)
        );
        assert_eq!(
            (routes[0].group_id, routes[0].physical_layer_ordinal),
            (1, 0)
        );
        assert_eq!(
            (routes[4].group_id, routes[4].physical_layer_ordinal),
            (1, 3)
        );
    }

    /// The grouping wrapper must not paper over a bad spec input: its guards are
    /// the specs function's guards, reached through it.
    #[test]
    fn grouping_propagates_a_spec_level_refusal() {
        let cfg = checkpoint_config();
        let err = compute_layer_kv_cache_groups(&cfg, 0, KVCacheDType::BFloat16, CHUNK)
            .expect_err("a zero block_size must not survive grouping");
        assert!(
            err.contains("block_size"),
            "the error must name block_size, got: {err}"
        );
    }
}
