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
    AttentionKind, KVCacheDType, KVCacheGroup, KVCachePhysicalLayout, LayerKVCacheSpec,
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
/// (`single_type_kv_cache_manager.py:178-186`, whose comment names both failure
/// modes: "re-introduce the deadlock from issue #39734 or, worse, mid-prefill
/// OOM"; the lookup that wires that single source is `:1860-1875`) precisely
/// because drift between them is either a deadlock or a mid-prefill OOM.
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

    // Both halves of the `max_position_embeddings` contract, re-checked here for
    // the same reason as the geometry above: this is `pub` and must not trust a
    // config that skipped `from_json_str`. The zero half is the one that fails
    // OPEN — `u32::try_from(0)` succeeds, and a 0 `max_model_len` makes the
    // full-attention bound `div_ceil(0, block_size) == 0`. That is an `Ok`
    // describing a pool that can hold nothing, and it stays silent all the way
    // through a gemma4-shaped sizer: `group_reserved_blocks(Full, 0, width)` is 0
    // at every width, so the 13 full layers contribute no bytes and no memory
    // check ever trips.
    if config.text_config.max_position_embeddings == 0 {
        return Err(format!(
            "muse_glimmer KV cache groups: max_position_embeddings must be non-zero, got {}; \
             it is the seam's max_model_len and a 0 admits no blocks at all",
            config.text_config.max_position_embeddings
        ));
    }
    let max_model_len =
        u32::try_from(config.text_config.max_position_embeddings).map_err(|_| {
            format!(
                "muse_glimmer KV cache groups: max_position_embeddings {} does not fit in a u32",
                config.text_config.max_position_embeddings
            )
        })?;
    let groups = group_layer_kv_cache_specs(&specs, max_model_len, max_chunk)
        .map_err(|e| format!("muse_glimmer KV cache grouping failed: {e}"))?;

    // The two-group contract, enforced where it is produced.
    //
    // Downstream is told to HARD-CODE `group_id` 0 as the full group rather than
    // discover it, and group 0's adapter is the one returned from
    // `paged_adapter()` and the one that publishes into the content-addressed
    // prefix cache. A single-kind `layer_types` table parses cleanly — the
    // config's NoPE<->Full biconditional only fires on a disagreement BETWEEN
    // the two tables, and a uniform table agrees with itself — and would collapse
    // grouping to one group. All-sliding is the silent direction: group 0 becomes
    // the SLIDING group, and a sliding block's contents depend on where the
    // window was when it was written, so a later turn resumes from a prefix hit
    // describing a different window offset. All-full is the loud one: anything
    // indexing `groups[1]` panics.
    //
    // Deliberately "both kinds present" and NOT the `[S,S,S,F]` x 13 pattern —
    // a variant with a different ratio is still a hybrid, and this guard refuses
    // no legal Muse-Glimmer checkpoint. gemma4 carries the same half of this at
    // `gemma4/model.rs:2108-2113`. Note ">1 full group" is unreachable here: one
    // uniform physical layout cannot key two distinct full groups.
    let full_groups = groups
        .iter()
        .filter(|group| matches!(group.attention_kind, AttentionKind::Full))
        .count();
    let sliding_groups = groups
        .iter()
        .filter(|group| matches!(group.attention_kind, AttentionKind::SlidingWindow { .. }))
        .count();
    if full_groups == 0 || sliding_groups == 0 {
        return Err(format!(
            "muse_glimmer KV cache groups: this decoder is hybrid by construction, so \
             grouping must yield both attention kinds, but {} layers produced {} \
             full-attention group(s) and {} sliding-window group(s). Downstream hard-codes \
             group_id 0 as the full group — its adapter is the one published into the \
             content-addressed prefix cache — so a single-kind layer_types table would \
             hand that role to the wrong group instead of failing",
            specs.len(),
            full_groups,
            sliding_groups
        ));
    }

    Ok(groups)
}

/// Physical blocks to reserve for one KV group in a SHARED pool, given how many
/// sequences the scheduler may keep live at once.
///
/// A sliding group WIDENS: `max(max_admission_blocks, scheduler_width) + 1`.
///
///   * `max(…, scheduler_width)` — every live row needs at least one starter
///     block, so a window smaller than the scheduler width must not be the
///     binding constraint.
///   * `+ 1` — the group's NULL BLOCK, the sentinel `remove_skipped_blocks` /
///     `replace_block` write into retired slots. This is gemma4's
///     `null_block_bytes` term, one block per sliding group
///     (`gemma4/model.rs:2148-2157`), and it is what makes
///     `required_bytes_for_width(1)` equal `minimum_pool_bytes` exactly
///     (`gemma4/model.rs:2182-2196`). Drop it and the pool deadlocks at full
///     occupancy with no block to retire slots into.
///
///     It is NOT the straddled-window block. That one is a different block and is
///     already inside `max_admission_blocks`:
///     `AttentionKind::sliding_window_max_admission_blocks`
///     (`transformer/kv_cache_spec.rs:64-79`) ends in its own `+ 1` for the
///     unaligned window head, and vLLM spends it the same way —
///     `SlidingWindowSpec.max_memory_usage_bytes` is `max_blocks *
///     page_size_bytes` with nothing added on top, and
///     `ChunkedLocalAttentionSpec` omits that `+1` because chunk boundaries ARE
///     block-aligned. Two `+1`s, two blocks; reading them as one invites deleting
///     either, and both deletions are load-bearing.
///     `the_reservations_plus_one_is_the_null_block_not_the_straddled_window_block`
///     pins the distinction.
///
/// A full group is returned unchanged: it is already bounded by `max_model_len`,
/// and gemma4's `null_block_bytes` term excludes full groups.
///
/// This number is a POOL RESERVATION and never a block-table row width. Rows are
/// append-only while blocks are recycled — a recycled slot is overwritten with the
/// null block at its existing index, nothing shifts down — so the row width stays
/// `div_ceil(max_model_len, block_size)` for sliding layers too. vLLM's
/// `SlidingWindowSpec` deliberately does not override `max_num_blocks_per_req`
/// for this reason. Keep the two numbers in separate code paths so they cannot be
/// confused.
pub fn group_reserved_blocks(
    attention_kind: AttentionKind,
    max_admission_blocks: u32,
    scheduler_width: u32,
) -> u32 {
    match attention_kind {
        AttentionKind::Full => max_admission_blocks,
        AttentionKind::SlidingWindow { .. } => {
            max_admission_blocks.max(scheduler_width).saturating_add(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::muse_glimmer::config::fixtures::{
        config_json, layer_tables, text_config_json,
    };

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

    /// Pinned to the early guard's own wording, not merely to the string
    /// "block_size". The layout validity check downstream also names block_size,
    /// so a looser assertion would stay green with the early guard deleted and
    /// report an "invalid physical layout" for what is really a caller error.
    #[test]
    fn refuses_a_zero_block_size() {
        let cfg = checkpoint_config();
        let err = compute_layer_kv_cache_specs(&cfg, 0, KVCacheDType::BFloat16)
            .expect_err("a zero block_size cannot produce a valid layout");
        assert!(
            err.contains("require block_size > 0"),
            "the error must name block_size as a caller-supplied precondition, got: {err}"
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
            err.contains("layer_kinds has 51 entries but num_hidden_layers is 52"),
            "the error must report the arity mismatch itself; the per-layer fallback \
             message also contains \"layer_kinds\" and \"51\", so a looser assertion \
             would survive deleting the arity guard, got: {err}"
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

    /// The sibling of the test above, and the one that actually fails OPEN
    /// without a guard. `u32::try_from(0)` SUCCEEDS, so a 0 context length used
    /// to reach the seam intact and produce a full-attention group with
    /// `max_admission_blocks == div_ceil(0, 16) == 0` — an `Ok` result
    /// describing a pool that can hold nothing.
    ///
    /// That zero then propagates silently through a gemma4-shaped pool sizer:
    /// `group_reserved_blocks(Full, 0, width) == 0` at EVERY scheduler width, so
    /// all 13 full layers contribute zero bytes to `one_sequence_bytes`, the
    /// `memory_bytes < minimum_pool_bytes` check passes, and the
    /// `while … required_bytes_for_width(…) > memory_bytes` loop never trips
    /// (`gemma4/model.rs:2135-2199`). A KV pool with literally no full-attention
    /// capacity gets allocated without a single error.
    ///
    /// Asserted on the "must be non-zero" wording, not merely on the field name:
    /// the u32 guard right next to it also names `max_position_embeddings`, so a
    /// looser assertion would stay green with this guard deleted.
    #[test]
    fn refuses_a_zero_max_position_embeddings_because_a_full_group_would_admit_no_blocks() {
        let mut cfg = checkpoint_config();
        cfg.text_config.max_position_embeddings = 0;
        let err = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, CHUNK)
            .expect_err("a zero context length must fail closed, not admit zero blocks");
        assert!(
            err.contains("max_position_embeddings must be non-zero"),
            "the error must name the field AND the zero, mirroring \
             `config.rs`'s own guard; the u32-range guard beside it also contains \
             \"max_position_embeddings\", got: {err}"
        );

        // Positive control on the boundary: the guard rejects 0 only, not "small".
        // A mutation to `<= 1` or `< block_size` would trip here.
        cfg.text_config.max_position_embeddings = 1;
        let groups = groups(&cfg, CHUNK);
        assert_eq!(groups[0].max_admission_blocks, 1);
    }

    /// A config whose 52 layers are all ONE kind, built by rewriting the
    /// fixture's two parallel tables so it still goes through the real parser.
    ///
    /// Both directions PARSE. `from_json_str`'s NoPE<->Full biconditional only
    /// fires on a disagreement BETWEEN the tables (`theta == 0 && kind != Full`,
    /// or `kind == Full && theta != 0`), and a uniform table agrees with itself.
    /// Nothing in the parser counts or positions the full layers.
    fn uniform_kind_config(kind: &str, theta: &str) -> MuseGlimmerConfig {
        let (kinds, thetas) = layer_tables(52);
        let text = text_config_json(52)
            .replace(
                &format!("[{kinds}]"),
                &format!("[{}]", vec![kind; 52].join(",")),
            )
            .replace(
                &format!("[{thetas}]"),
                &format!("[{}]", vec![theta; 52].join(",")),
            );
        MuseGlimmerConfig::from_json_str(&config_json(&text)).expect(
            "a uniform layer table must PARSE — that is exactly why grouping has to catch it",
        )
    }

    /// An all-sliding layer table must be refused HERE, because the parser
    /// cannot see it and downstream is told not to look.
    ///
    /// "Exactly two groups, `group_id` 0 = Full" is a CONTRACT, not a discovery:
    /// the design doc instructs M1 not to derive it at runtime, and the tests
    /// above pin `groups[0].attention_kind == Full`. With 52 sliding layers
    /// grouping returns ONE group, so `groups[0]` — the group whose adapter is
    /// returned from `paged_adapter()` and the one used for prefix-cache
    /// finalization and cold capture (`gemma4/model.rs:490-510`) — is the
    /// SLIDING group. Sliding blocks then get published into the
    /// content-addressed prefix cache, and a sliding block's contents depend on
    /// where the window was when it was written: turn 2 resumes from a hit
    /// describing a different window offset. Fluent, wrong, no error.
    #[test]
    fn refuses_an_all_sliding_layer_table_because_group_zero_would_not_be_the_full_group() {
        let cfg = uniform_kind_config("\"sliding_attention\"", "500000.0");
        assert_eq!(cfg.text_config.layer_kinds.len(), 52);

        let err = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, CHUNK)
            .expect_err("a decoder with no full-attention layer must not group");
        assert!(
            err.contains("0 full-attention group"),
            "the error must report the observed kind counts, and specifically the \
             MISSING full group — the mirror case reports 0 sliding groups with the \
             same prose, got: {err}"
        );
    }

    /// The mirror direction: 52 full/NoPE layers also parse, also yield ONE
    /// group, and there any consumer indexing `groups[1]` for the sliding pool
    /// panics on an out-of-bounds index instead of misrouting. Still refused,
    /// and with the other count named, so neither test can pass on a generic
    /// message.
    #[test]
    fn refuses_an_all_full_layer_table_because_group_one_would_not_exist() {
        let cfg = uniform_kind_config("\"full_attention\"", "0");

        let err = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, CHUNK)
            .expect_err("a decoder with no sliding layer must not group");
        assert!(
            err.contains("0 sliding-window group"),
            "the error must report the MISSING sliding group specifically, got: {err}"
        );
    }

    /// The both-kinds guard must refuse no legal checkpoint. It is deliberately
    /// NOT the `[S,S,S,F]` x 13 pattern: a future variant with a different ratio
    /// or a different layer count is still a hybrid, and over-fitting the
    /// pattern here would reject it. A 4-layer `[S,S,S,F]` config groups fine.
    #[test]
    fn the_both_kinds_guard_accepts_any_hybrid_ratio_not_just_the_reference_pattern() {
        let cfg = MuseGlimmerConfig::from_json_str(&config_json(&text_config_json(4)))
            .expect("a 4-layer [S,S,S,F] config must validate");
        let groups = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, CHUNK)
            .expect("a 4-layer hybrid must still group into two groups");

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].attention_kind, AttentionKind::Full);
        assert_eq!(groups[0].layer_indices, vec![3]);
        assert_eq!(groups[1].layer_indices, vec![0, 1, 2]);
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

    /// Nothing on the path from config to group may be hard-coded at the
    /// reference value. Every assertion above compares a derived number against
    /// the SAME config field or against a constant that happens to equal the
    /// reference (2048, 131072, block_size 16, bf16), so a baked-in literal would
    /// satisfy all of them. Feeding non-reference values through is the only thing
    /// that pins the plumbing — the same argument as
    /// `config::tests::defaulted_fields_are_read_from_the_file_when_present`.
    #[test]
    fn no_argument_or_config_field_on_the_path_is_hard_coded_at_its_reference_value() {
        let text = text_config_json(52)
            .replace("\"sliding_window\": 2048", "\"sliding_window\": 1024")
            .replace(
                "\"max_position_embeddings\": 131072",
                "\"max_position_embeddings\": 65536",
            );
        let cfg = MuseGlimmerConfig::from_json_str(&config_json(&text))
            .expect("a non-reference window and context length must still validate");

        let specs = compute_layer_kv_cache_specs(&cfg, 32, KVCacheDType::Fp8)
            .expect("specs must derive for the non-reference config");
        assert_eq!(
            specs[0].attention_kind,
            AttentionKind::SlidingWindow {
                sliding_window: 1024
            },
            "the window must come from the config, not from a literal 2048"
        );
        assert_eq!(specs[0].physical_layout.block_size, 32);
        assert_eq!(specs[0].physical_layout.cache_dtype, KVCacheDType::Fp8);

        let groups = compute_layer_kv_cache_groups(&cfg, 32, KVCacheDType::Fp8, CHUNK)
            .expect("groups must derive for the non-reference config");
        // full    = div_ceil(65536, 32)                       = 2048
        // sliding = div_ceil(min(1023 + 512, 65536), 32) + 1   = 48 + 1 = 49
        assert_eq!(groups[0].max_admission_blocks, 2048);
        assert_eq!(groups[1].max_admission_blocks, 49);
    }

    // ── Pool reservation ──────────────────────────────────────────────────

    /// A sliding group's reservation only ever grows: it WIDENS to the scheduler
    /// width and then adds one.
    ///
    /// Narrowing is the trap this pins. Block-table ROWS are append-only while
    /// BLOCKS are recycled — a retired slot is overwritten with the null block at
    /// its existing index and nothing shifts down — so a row keeps growing with
    /// computed tokens even though the live window does not. Narrow a sliding
    /// group's reservation to the window and it overflows the moment computed
    /// tokens pass the window: the very first token past it needs a block the
    /// budget does not contain. The `+1` is this group's NULL BLOCK — the
    /// straddled window block is already inside the 161, see
    /// `the_reservations_plus_one_is_the_null_block_not_the_straddled_window_block`.
    #[test]
    fn a_sliding_group_only_ever_widens_because_rows_are_append_only_while_blocks_are_recycled() {
        let sliding = AttentionKind::SlidingWindow {
            sliding_window: 2048,
        };

        // The real checkpoint's sliding admission (161 blocks) with room for 8
        // live rows: admission binds, plus this group's null block.
        assert_eq!(group_reserved_blocks(sliding, 161, 8), 162);

        // A window smaller than the scheduler width must not be the binding
        // constraint: every live row still needs a starter block, plus null.
        assert_eq!(
            group_reserved_blocks(AttentionKind::SlidingWindow { sliding_window: 8 }, 2, 8),
            9
        );

        // Never below the admission bound, at any width — the narrowing check.
        for width in [0u32, 1, 2, 8, 32, 160, 161, 162, 4096] {
            let reserved = group_reserved_blocks(sliding, 161, width);
            assert!(
                reserved > 161,
                "width {width} reserved {reserved}, which does not exceed the 161-block \
                 admission bound; a sliding reservation must never narrow"
            );
            assert!(reserved >= width, "width {width} reserved only {reserved}");
        }
    }

    /// TWO different blocks, one `+1` each, and this pins which is which.
    ///
    /// The straddled-window block is spent INSIDE `max_admission_blocks`:
    /// `AttentionKind::sliding_window_max_admission_blocks`
    /// (`transformer/kv_cache_spec.rs:64-79`) already ends in `+ 1`, and vLLM's
    /// `SlidingWindowSpec.max_memory_usage_bytes` is `max_blocks *
    /// page_size_bytes` with nothing added on top — so 161 is the whole
    /// straddle-aware bound.
    ///
    /// `group_reserved_blocks`'s own `+1` is therefore a DIFFERENT block: the
    /// group's null-block sentinel, the one `remove_skipped_blocks` /
    /// `replace_block` write into retired slots. gemma4's pool sizer proves it —
    /// `minimum_pool_bytes = one_sequence_bytes + null_block_bytes` where
    /// `null_block_bytes` is one block per SLIDING group
    /// (`gemma4/model.rs:2148-2157`), and `required_bytes_for_width(1)` equals
    /// that exactly (`:2182-2196`), which is only true when the full group
    /// contributes admission-and-no-more and each sliding group contributes
    /// admission-plus-one.
    ///
    /// Without this test, a maintainer reading the two `+1`s as one block can
    /// delete either — dropping the seam's under-provisions every sliding group
    /// by the straddled block, dropping this one deadlocks the pool at full
    /// occupancy with no null block to retire slots into.
    #[test]
    fn the_reservations_plus_one_is_the_null_block_not_the_straddled_window_block() {
        let sliding = AttentionKind::SlidingWindow {
            sliding_window: 2048,
        };

        // The straddle +1 is already inside the admission bound.
        let admitted_tokens = (2048u32 - 1 + CHUNK).min(131_072);
        assert_eq!(
            admitted_tokens.div_ceil(16) + 1,
            161,
            "the hand formula's own trailing +1 is the straddled block"
        );
        assert_eq!(
            AttentionKind::sliding_window_max_admission_blocks(2048, 131_072, CHUNK, 16)
                .expect("the reference geometry must admit"),
            161,
            "so the seam's 161 already carries it"
        );

        // gemma4's pool identity at width 1: full contributes admission exactly,
        // each sliding group contributes admission + ONE null block.
        assert_eq!(
            group_reserved_blocks(AttentionKind::Full, 8192, 1) as i64 - 8192,
            0,
            "a full group has no null block in gemma4's null_block_bytes term"
        );
        assert_eq!(
            group_reserved_blocks(sliding, 161, 1) as i64 - 161,
            1,
            "exactly ONE block on top of the straddle-aware admission bound, and it \
             is the null block — two +1s here would over-reserve, zero would leave \
             no sentinel to retire recycled slots into"
        );
    }

    /// A full group is returned unchanged. It is already bounded by
    /// `max_model_len`, so widening it by the scheduler width would over-reserve
    /// the group that dominates the footprint (8192 of the 8354 blocks one
    /// sequence needs here).
    #[test]
    fn a_full_group_is_never_widened_or_padded() {
        assert_eq!(group_reserved_blocks(AttentionKind::Full, 8192, 8), 8192);
        assert_eq!(group_reserved_blocks(AttentionKind::Full, 8192, 4096), 8192);
        assert_eq!(group_reserved_blocks(AttentionKind::Full, 1, 64), 1);
    }

    /// The reservation is not a row width, and the two numbers must stay far
    /// apart. A sliding row is still `div_ceil(max_model_len, block_size)` wide
    /// (8192 here) because the row index IS `absolute_position / block_size`;
    /// only the block budget shrinks to 162. Feeding this helper's output in as a
    /// row width would cap addressable positions at ~2592 tokens.
    #[test]
    fn the_sliding_reservation_is_much_smaller_than_the_block_table_row_it_must_not_size() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);
        let row_width = cfg.text_config.max_position_embeddings.div_ceil(16) as u32;

        let sliding_reservation =
            group_reserved_blocks(groups[1].attention_kind, groups[1].max_admission_blocks, 8);
        assert_eq!(sliding_reservation, 162);
        assert_eq!(row_width, 8192);
        assert!(
            sliding_reservation * 8 < row_width,
            "the reservation ({sliding_reservation}) and the row width ({row_width}) are \
             different numbers and must live in different code paths"
        );

        // The full group's reservation and its row width DO coincide, which is
        // exactly why the sliding case is the one that gets confused.
        assert_eq!(
            group_reserved_blocks(groups[0].attention_kind, groups[0].max_admission_blocks, 8),
            row_width
        );
    }

    /// Saturating arithmetic: a pathological admission bound must clamp, not wrap
    /// to a tiny reservation.
    #[test]
    fn reservation_saturates_instead_of_wrapping() {
        let sliding = AttentionKind::SlidingWindow {
            sliding_window: 2048,
        };
        assert_eq!(group_reserved_blocks(sliding, u32::MAX, 1), u32::MAX);
        assert_eq!(group_reserved_blocks(sliding, 1, u32::MAX), u32::MAX);
    }

    // ── Real checkpoint (gated) ────────────────────────────────────────────

    /// Every number above is derived from `config::fixtures`, a hand
    /// transcription of the reference `config.json`, and the layout/admission
    /// literals (`(16, 2, 128, bf16)`, 8192, 161) were transcribed from the same
    /// reading. They therefore cross-check each other and nothing else: if the
    /// real `head_dim` were 64, or `num_key_value_heads` 4, or `sliding_window`
    /// 4096, the fixture and the literals would agree on the wrong model and
    /// every test in this file would stay green while production
    /// `compute_layer_kv_cache_specs` read the real value off disk and sized a
    /// different cache.
    ///
    /// This test is the one that makes the two-group claim and the 161-block
    /// claim facts about the CHECKPOINT. It derives everything from the real
    /// file, so a fixture that drifts from disk fails here.
    #[test]
    #[ignore = "requires the local Muse-Glimmer checkpoint; set MLX_TEST_MUSE_GLIMMER_MODEL_PATH and run with --ignored"]
    fn real_checkpoint_kv_geometry_and_admission_bounds() {
        let Ok(dir) = std::env::var("MLX_TEST_MUSE_GLIMMER_MODEL_PATH") else {
            eprintln!("skipping: MLX_TEST_MUSE_GLIMMER_MODEL_PATH not set");
            return;
        };
        let cfg = MuseGlimmerConfig::from_path(std::path::Path::new(&dir))
            .expect("the real checkpoint's config.json must parse and validate");

        // The three values that determine every byte of the cache, read from
        // disk rather than from the fixture.
        let text = &cfg.text_config;
        assert_eq!(text.head_dim, 128);
        assert_eq!(text.num_key_value_heads, 2);
        assert_eq!(text.sliding_window, 2048);
        assert_eq!(text.max_position_embeddings, 131_072);

        let specs = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect("the real checkpoint's specs must derive");
        assert_eq!(specs.len(), 52);
        let expected_layout = KVCachePhysicalLayout::new(16, 2, 128, KVCacheDType::BFloat16);
        for spec in &specs {
            assert_eq!(
                spec.physical_layout, expected_layout,
                "layer {} layout drifted from the documented (16, 2, 128, bf16)",
                spec.layer_index
            );
        }

        let groups = compute_layer_kv_cache_groups(&cfg, 16, KVCacheDType::BFloat16, CHUNK)
            .expect("the real checkpoint's groups must derive");
        assert_eq!(groups.len(), 2, "the two-group contract, off the real file");
        assert_eq!(groups[0].attention_kind, AttentionKind::Full);
        assert_eq!(groups[0].layer_indices, FULL_LAYERS.to_vec());
        assert_eq!(groups[0].layer_indices.len(), 13);
        assert_eq!(
            groups[1].attention_kind,
            AttentionKind::SlidingWindow {
                sliding_window: 2048
            }
        );
        assert_eq!(groups[1].layer_indices.len(), 39);
        assert_eq!(groups[0].max_admission_blocks, 8192);
        assert_eq!(groups[1].max_admission_blocks, 161);

        eprintln!(
            "real checkpoint KV OK: {} groups, {}/{} full/sliding layers, layout \
             (block_size {}, kv_heads {}, head_size {}), admission {}/{} blocks",
            groups.len(),
            groups[0].layer_indices.len(),
            groups[1].layer_indices.len(),
            expected_layout.block_size,
            expected_layout.num_kv_heads,
            expected_layout.head_size,
            groups[0].max_admission_blocks,
            groups[1].max_admission_blocks,
        );
    }
}
