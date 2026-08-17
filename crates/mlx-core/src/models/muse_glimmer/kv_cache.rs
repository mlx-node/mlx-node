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
    //
    // The window is bounded at `i32::MAX`, not at `u32::MAX`, even though
    // `AttentionKind` stores it as a `u32`. The reason is downstream: every
    // consumer of the window takes an `i32` — the kernel's `sliding_window` FFI
    // argument and `create_causal_mask`'s `window_size` — so a value in
    // `[2^31, 2^32-1]` fits the spec type and still arrives at a dispatch
    // negative (3_000_000_000 -> -1_294_967_296), where the mask route silently
    // masks every key. `paged_window_for_kind` refuses it a third time; see its
    // docs for the measurement.
    let sliding_window = i32::try_from(text.sliding_window)
        .map(|window| {
            // `text.sliding_window` is a `usize`, so the checked value cannot be
            // negative and this is the identity. Spelled `unsigned_abs` rather
            // than `as u32` so no cast survives in this file to be copied into a
            // context where the sign is not already known.
            window.unsigned_abs()
        })
        .map_err(|_| {
            format!(
                "muse_glimmer KV cache specs: sliding_window {} exceeds i32::MAX ({}); \
                 the window's consumers are an i32 kernel argument and an i32 mask \
                 width, and a value past i32::MAX reaches them negative",
                text.sliding_window,
                i32::MAX
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

    // A 0-layer config is the one shape the arity check below AGREES with: empty
    // tables give `0 == 0`, the loop iterates nothing, and this function returns
    // `Ok(vec![])`. The refusal then lands two layers away, in
    // `compute_layer_kv_cache_groups`' both-kinds guard, whose message reads
    // "0 layers produced 0 full-attention group(s)" and blames grouping for a
    // config field. Config load refuses it (`config.rs`), but this is `pub` and
    // must not trust that its caller came through `from_json_str`.
    //
    // `num_attention_heads` is deliberately NOT re-checked here: this seam never
    // reads it. The physical layout comes from `num_key_value_heads` and
    // `head_dim`, which is exactly why a 0 query-head count is invisible to the
    // cache and has to be caught at config load instead.
    if text.num_hidden_layers == 0 {
        return Err(
            "muse_glimmer KV cache specs require num_hidden_layers > 0; a 0 agrees with \
             empty layer tables and would derive an empty spec set, deferring the refusal \
             to grouping where it reads as a grouping failure"
                .to_string(),
        );
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

// ─── The window-carrying contract ────────────────────────────────────────────
//
// WHY THIS EXISTS, stated once so it cannot be dropped: a confirmed, measured
// defect in the shipping gemma4 paged path is that a sliding group's window is
// silently lost on the cache-hit prefill route. Measured on the real
// gemma-4-12b-it checkpoint (window 1024, prefill chunk 512, 40 of 48 layers
// sliding): the default `Auto` route selects the dense paged-pool SDPA gather,
// which ran `scaled_dot_product_attention_causal` with NO mask; the varlen route
// passed a literal `0` in the kernel's `sliding_window` FFI slot, and `0` is the
// kernel's "disable the sliding mask" sentinel — full causal attention, not
// "inert" (`mlx_paged_ops.cpp:482`; `paged_attention.metal:2001` collapses
// `sliding_lower` to 0 when `sw <= 0`). Cost: max |delta| 0.124512 on the
// attention output against a windowed reference whose own RMS is 0.035562 (3.5x
// the signal), versus 0.000977 (1 bf16 ULP) for the one route that did pass the
// window. Real-model effect: the greedy continuation differed from the flat
// reference path at 6 of 9 prompt lengths from 1338 tokens up.
//
// Muse-Glimmer would inherit that by construction — 39 of 52 layers sliding at
// window 2048 — the moment M1 wires a sliding adapter. It has no forward pass
// yet, which is exactly why the contract is written here first: the family's
// paged dispatch has to ask this seam for its window before it can launch, and
// the seam refuses the combination that produced the defect.
//
// The split is honest and stated in both halves:
//
//   * ENFORCED TODAY (this module, model-free, tested): a windowed group paired
//     with a route that cannot express a window is an `Err`. The window value
//     itself cannot be typed by a call site — `PagedWindowSlot`'s fields are
//     private and its only constructor derives them from the group's own
//     `AttentionKind` — so there is no slot for a literal `0` to sit in.
//   * ONLY M1 CAN ENFORCE: that every dispatch it writes actually asks. The type
//     system cannot stop a forward pass from calling the FFI directly with a
//     hand-written argument. That half is a stated requirement plus the source
//     tripwire `wiring_a_sliding_adapter_without_this_seam_trips_the_tripwire`,
//     which fails the moment paged-attention wiring appears in this family
//     without a reference to `PagedWindowSlot`.

/// How one attention route can express a sliding window.
///
/// This is a property of the ROUTE, supplied by the dispatch site, because that
/// is the fact the route knows and the group does not. vLLM makes the window a
/// property of the `AttentionImpl` and passes it to every
/// `flash_attn_varlen_func` dispatch — context, query and decode
/// (`vllm/v1/attention/backends/flash_attn.py:1340/1372/1447`) — precisely
/// because a per-route argument list is where a window gets lost. This enum is
/// the same rule spelled as data: adding a route means adding a variant, and a
/// new variant is a compile error in [`paged_window_for_kind`]'s match rather
/// than a silently window-blind fifth path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowCarrier {
    /// The route hands the window to the kernel as the `sliding_window: i32` FFI
    /// argument, and the kernel derives its own per-query-row lower bound from
    /// it. Take the value from [`PagedWindowSlot::kernel_slot`].
    ///
    /// This is the varlen and legacy paged-attention dispatches, and the batched
    /// / ragged / raw-Metal decode dispatches. It is per-row correct for a
    /// multi-token chunk on a cached prefix without the caller recomputing
    /// anything: the kernel bottom-right aligns via
    /// `effective_context_len = context_len - q_len + q_pos_in_seq + 1`
    /// (`paged_attention.metal:1860`).
    KernelArgument,
    /// The route builds an explicit boolean mask and hands it to SDPA. Take the
    /// window from [`PagedWindowSlot::mask_window`] and pass it as
    /// `create_causal_mask(seq_len, Some(cached_prefix_len), window)`
    /// (`array/mask.rs:33`, whose third parameter already is the window —
    /// `linds >= rinds && linds < rinds + window`, bool `true = keep`).
    ///
    /// This is the dense paged-pool gather and the host-read path. Both were
    /// window-blind in gemma4 not because they lack the concept but because they
    /// passed `None` / no mask at all.
    ExplicitMask,
    /// The route has no way to express a window: no kernel argument, no mask
    /// parameter. Legal for a full-attention group and for nothing else.
    ///
    /// A windowed group on such a route is the defect this module refuses. It is
    /// NOT a route to "fix later" by threading a mask in — if a route gains the
    /// ability, it changes which variant it reports, and the refusal disappears
    /// on its own.
    Unsupported,
}

impl WindowCarrier {
    /// Can this route carry a non-zero window at all?
    ///
    /// Spelled as a method so a caller can ask without matching, and so the
    /// answer for a new variant has to be written down deliberately.
    pub fn can_carry_a_window(self) -> bool {
        match self {
            Self::KernelArgument | Self::ExplicitMask => true,
            Self::Unsupported => false,
        }
    }
}

/// The window argument for ONE admitted attention dispatch.
///
/// Fields are private and there is no `Default`, no public constructor and no
/// `From`. The only way to obtain one is [`paged_window_for_kind`] /
/// [`paged_window_for_group`] / [`admit_paged_dispatch_plan`], each of which
/// derives the window from the group's own `AttentionKind`. So a dispatch site
/// cannot write the literal that caused the defect: there is no parameter for it
/// to be written into. That is strictly stronger than making the literal
/// unwritable at one call site, because it also covers the next call site nobody
/// has written yet.
///
/// The two accessors are deliberately carrier-scoped and both return `Option`:
/// each answers only for the route that admitted it, and returns `None` — never
/// a plausible `0` — when asked the other route's question. A `0` returned from
/// the wrong accessor IS the defect, spelled in a different file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedWindowSlot {
    carrier: WindowCarrier,
    /// 0 means full attention. Non-zero only ever came from an
    /// `AttentionKind::SlidingWindow` payload.
    ///
    /// Stored as the `i32` both consumers actually take — the kernel's
    /// `sliding_window` FFI slot and `create_causal_mask`'s `window_size` — and
    /// range-checked once, in [`paged_window_for_kind`]. Storing the spec's `u32`
    /// and casting in the accessors is what let a config-supplied
    /// `sliding_window` in `[2^31, 2^32-1]` reach a dispatch as a NEGATIVE
    /// window: 3_000_000_000 arrives as -1_294_967_296. There is no cast in this
    /// type any more, so the invariant is structural rather than remembered,
    /// which is the same argument as the private fields above.
    window: i32,
}

impl PagedWindowSlot {
    /// The value for the kernel's `sliding_window: i32` FFI slot, or `None` when
    /// this dispatch is not a kernel-argument route and therefore has no such
    /// slot to fill.
    ///
    /// `Some(0)` appears only for a full-attention group on a kernel route,
    /// where 0 is the correct and intended "no sliding mask" sentinel. A sliding
    /// group can never produce `Some(0)`, which is the whole point.
    pub fn kernel_slot(&self) -> Option<i32> {
        match self.carrier {
            WindowCarrier::KernelArgument => Some(self.window),
            WindowCarrier::ExplicitMask | WindowCarrier::Unsupported => None,
        }
    }

    /// The `window_size` argument for `create_causal_mask`, or `None` when no
    /// mask is needed.
    ///
    /// `None` for a full-attention group is load-bearing beyond tidiness: it
    /// tells the caller to keep the unmasked causal fast path. The mask-bearing
    /// SDPA kernel has a different bf16 reduction order from the `_causal` one,
    /// so materializing an all-true mask for a global layer would move its
    /// numerics for no correctness gain.
    pub fn mask_window(&self) -> Option<i32> {
        match self.carrier {
            WindowCarrier::ExplicitMask if self.window != 0 => Some(self.window),
            _ => None,
        }
    }

    /// Does this dispatch have a window to lose?
    pub fn is_windowed(&self) -> bool {
        self.window != 0
    }

    /// The route that was admitted. Exposed so a dispatch site can assert it
    /// took the branch it thinks it took.
    pub fn carrier(&self) -> WindowCarrier {
        self.carrier
    }
}

/// Admit ONE dispatch: pair an attention kind with the route that will execute
/// it, or refuse.
///
/// Refuses exactly three things, each because it fails OPEN otherwise:
///
///   * a windowed kind on a route whose [`WindowCarrier`] cannot express a
///     window. Silently dropping the window widens 39 of this decoder's 52
///     layers to full causal attention over positions the block table has
///     already retired to the reserved null block — never-written
///     `StorageModePrivate` memory (`layer_kv_pool.rs:499`: "not zeroed ...
///     returns whatever the driver handed out"), so the read is formally
///     undefined on top of being wrong.
///   * a `SlidingWindow { sliding_window: 0 }` payload. Config load and
///     [`compute_layer_kv_cache_specs`] both refuse it already; refused a third
///     time here because this is the last point before a kernel launch, and a 0
///     arriving in a kernel slot is indistinguishable from "disable the mask".
///   * a window past `i32::MAX`. `AttentionKind` carries the window as a `u32`
///     but BOTH consumers take an `i32` — the kernel's `sliding_window` FFI
///     argument and `create_causal_mask`'s `window_size` (`array/mask.rs:35`) —
///     so a spec in `[2^31, 2^32-1]` used to arrive here and be handed on as a
///     negative number: 3_000_000_000 becomes -1_294_967_296. The two routes
///     then diverge, and one of them is silent:
///
///     - KernelArgument fails CLOSED. The C++ validator rejects a negative
///       window (`mlx_paged_ops.cpp:483`, "must be >= 0 (use 0 to disable the
///       sliding mask)"), the entry point catches it and returns nullptr, and
///       `check_handle` turns that into an `Err`. So for this route the refusal
///       below only moves the diagnosis from deep inside the FFI to the field
///       that caused it — an error-message improvement, not a new guarantee.
///     - ExplicitMask fails OPEN and SILENTLY. `create_causal_mask`'s predicate
///       is `linds >= rinds && linds < rinds + window_size`, so a negative
///       `window_size` makes the second half false for every cell and the mask
///       all-`false` (bool `true = keep`). Measured on real MLX: 0 of 96 cells
///       kept versus 68 for the causal baseline. Fed to the fused
///       `scaled_dot_product_attention` an all-false mask does NOT produce NaN
///       and does not error — every query row comes back as the uniform mean of
///       ALL V rows, finite and plausible, max|delta| 1.2485 against the
///       windowed reference. That is the same failure shape as the gemma4
///       dropped window this seam exists to prevent, so THIS is the half the
///       refusal actually closes.
///
///     The bound is `i32::MAX`, not `max_model_len` or any context-derived
///     value: a window larger than the context is legal and must stay
///     expressible — gemma4's `cache_hit_dense_window_arg` decides "the window
///     cannot bite" by comparing it against the context, which requires
///     representing it first.
///
/// A full-attention kind is admitted on every route including `Unsupported`,
/// with nothing to carry. That is not laxity: refusing it would refuse a legal
/// global layer on a legal route, and over-strictness here would push M1 to work
/// around the seam instead of through it.
pub fn paged_window_for_kind(
    attention_kind: AttentionKind,
    carrier: WindowCarrier,
) -> std::result::Result<PagedWindowSlot, String> {
    let window = match attention_kind {
        AttentionKind::Full => 0,
        AttentionKind::SlidingWindow { sliding_window: 0 } => {
            return Err(
                "muse_glimmer paged dispatch: a SlidingWindow { sliding_window: 0 } spec \
                 reached dispatch admission; 0 is the Metal kernel's \"disable the sliding \
                 mask\" sentinel (mlx_paged_ops.cpp:482), so it is full causal attention \
                 and not a window — reject it at config load, where the field has a name"
                    .to_string(),
            );
        }
        AttentionKind::SlidingWindow { sliding_window } => {
            i32::try_from(sliding_window).map_err(|_| {
                format!(
                    "muse_glimmer paged dispatch: sliding window {sliding_window} exceeds \
                     i32::MAX ({}), so it cannot reach a dispatch intact; both consumers \
                     take an i32 — the kernel's sliding_window FFI slot, which rejects a \
                     negative (mlx_paged_ops.cpp:483), and create_causal_mask's \
                     window_size, where a negative width masks EVERY key with no error \
                     and returns the uniform mean of all V rows. Reject the window at \
                     config load, where the field has a name",
                    i32::MAX
                )
            })?
        }
    };

    if window != 0 && !carrier.can_carry_a_window() {
        return Err(format!(
            "muse_glimmer paged dispatch: a sliding-window group (window {window}) was \
             routed to {carrier:?}, which cannot express a window. Dropping it makes the \
             dispatch attend every position back to 0, including positions already retired \
             to the reserved null block — never-written StorageModePrivate memory. Either \
             give the route a window (kernel sliding_window argument, or \
             create_causal_mask's window_size) and report the matching WindowCarrier, or \
             route the sliding group elsewhere"
        ));
    }

    Ok(PagedWindowSlot { carrier, window })
}

/// [`paged_window_for_kind`] for a group produced by
/// [`compute_layer_kv_cache_groups`].
///
/// Takes the group rather than the kind so the call site cannot re-type the
/// window on the way in, and so the error can name the group whose layers are
/// affected — with the `[S,S,S,F]` pattern that is 39 layers, and an error that
/// says "group 1" without saying "39 layers" understates the blast radius.
pub fn paged_window_for_group(
    group: &KVCacheGroup,
    carrier: WindowCarrier,
) -> std::result::Result<PagedWindowSlot, String> {
    paged_window_for_kind(group.attention_kind, carrier).map_err(|e| {
        format!(
            "{e} (group_id {}, {} layers: {:?})",
            group.group_id,
            group.layer_indices.len(),
            group.layer_indices
        )
    })
}

/// Admit EVERY group of one turn against the routes that turn selected, before
/// any of them launches.
///
/// This is the choke point M1's forward pass must call, and calling it once per
/// turn is the requirement — not once per layer, and not "wherever convenient".
/// The reason is the shape of the gemma4 defect rather than tidiness: there, a
/// per-route argument list meant the sliding group and the full group were
/// planned independently, one of them lost its window, and nothing compared the
/// two. `carriers` is indexed by `group_id`, so a plan that forgets a group is
/// an arity error rather than a group that quietly runs unadmitted.
///
/// Deliberately returns one slot per group in `group_id` order, so the caller
/// indexes the result the same way it indexed the input.
pub fn admit_paged_dispatch_plan(
    groups: &[KVCacheGroup],
    carriers: &[WindowCarrier],
) -> std::result::Result<Vec<PagedWindowSlot>, String> {
    if groups.is_empty() {
        return Err(
            "muse_glimmer paged dispatch: no KV cache groups to admit; this decoder is \
             hybrid by construction and must present both a full and a sliding group"
                .to_string(),
        );
    }
    if groups.len() != carriers.len() {
        return Err(format!(
            "muse_glimmer paged dispatch: {} KV cache group(s) but {} route carrier(s); \
             carriers are indexed by group_id, so a missing entry would leave a group \
             dispatching unadmitted — which is how a sliding group loses its window",
            groups.len(),
            carriers.len()
        ));
    }

    let mut slots = Vec::with_capacity(groups.len());
    for (index, group) in groups.iter().enumerate() {
        // `group_id` is the index by contract (`group_layer_kv_cache_specs`
        // enumerates in order). Checked rather than assumed because `carriers`
        // is indexed positionally: if the two ever disagree, every carrier is
        // applied to the wrong group, and pairing the FULL group's carrier with
        // the SLIDING group is precisely the defect with a different cause.
        if group.group_id != index {
            return Err(format!(
                "muse_glimmer paged dispatch: group at position {index} reports group_id \
                 {}; carriers are indexed by group_id and position, so a gap or a \
                 permutation would apply each route to the wrong group",
                group.group_id
            ));
        }
        slots.push(paged_window_for_group(group, carriers[index])?);
    }
    Ok(slots)
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

    /// A 0-layer config is the shape the arity check AGREES with (0 == 0), so
    /// without a dedicated guard this function returns `Ok(vec![])` and the
    /// refusal lands in grouping, reading as a grouping failure. Config load
    /// refuses it too (`config::tests::rejects_a_zero_num_hidden_layers_because_empty_tables_agree_with_it`);
    /// this pins the second line of defence, because the function is `pub`.
    ///
    /// Mutation caught: deleting the `num_hidden_layers == 0` guard. Pinned to
    /// this guard's own wording — the arity message also names
    /// `num_hidden_layers`, so a looser assertion would survive the deletion.
    #[test]
    fn refuses_a_zero_layer_count_that_the_arity_check_agrees_with() {
        let mut cfg = checkpoint_config();
        cfg.text_config.num_hidden_layers = 0;
        cfg.text_config.layer_kinds.clear();
        cfg.text_config.layer_rope_theta.clear();
        let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect_err("a 0-layer decoder must not derive an empty spec set");
        assert!(
            err.contains("require num_hidden_layers > 0"),
            "the error must name the layer count as a caller-supplied precondition, got: {err}"
        );
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

    // ── The window-carrying contract ──────────────────────────────────────
    //
    // These are the "sliding group x attention numerics" cell one level up from
    // the kernel: no MLX arrays, no device, no weights. They pin the ROUTING
    // decision that the gemma4 defect got wrong, at the only layer this family
    // has today. Each names the mutation it would catch, because a guard whose
    // test stays green under the mutation it exists to prevent is decoration.

    /// THE defect, refused. gemma4's measured failure was a sliding group
    /// reaching a route with no window concept (the dense paged-pool SDPA
    /// gather, running `scaled_dot_product_attention_causal` with no mask) and
    /// producing plausible, wrong output — max |delta| 0.124512 against a
    /// windowed reference whose own RMS is 0.035562.
    ///
    /// Mutation caught: making `paged_window_for_kind` return
    /// `Ok(PagedWindowSlot { carrier, window })` unconditionally, i.e. deleting
    /// the `can_carry_a_window` guard.
    #[test]
    fn a_sliding_group_on_a_window_blind_route_is_refused() {
        let err = paged_window_for_kind(
            AttentionKind::SlidingWindow {
                sliding_window: 2048,
            },
            WindowCarrier::Unsupported,
        )
        .expect_err("a windowed group must never be admitted to a window-blind route");
        assert!(
            err.contains("cannot express a window") && err.contains("2048"),
            "the error must name both the incapacity and the window it dropped, so the \
             reader knows which group and how wide, got: {err}"
        );
    }

    /// The same refusal reached through the real grouping path, on the real
    /// checkpoint-shaped config — so the guard is pinned against the groups this
    /// family actually produces, not a hand-built kind. The message must name
    /// the layer count: "group 1" understates 39 layers.
    ///
    /// Mutation caught: `paged_window_for_group` dropping its `map_err` context,
    /// or being wired to `groups[0]`'s kind for every group.
    #[test]
    fn the_thirty_nine_sliding_layers_are_named_when_their_window_would_be_dropped() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);
        let sliding = groups
            .iter()
            .find(|g| matches!(g.attention_kind, AttentionKind::SlidingWindow { .. }))
            .expect("the checkpoint shape must produce a sliding group");

        let err = paged_window_for_group(sliding, WindowCarrier::Unsupported)
            .expect_err("the sliding group must not be admitted to a window-blind route");
        assert!(
            err.contains("39 layers"),
            "the error must state how many layers lose their window, got: {err}"
        );

        // And the full group on the same route is fine: there is nothing to
        // carry, and refusing it would refuse a legal global layer.
        let full = groups
            .iter()
            .find(|g| g.attention_kind == AttentionKind::Full)
            .expect("the checkpoint shape must produce a full group");
        let slot = paged_window_for_group(full, WindowCarrier::Unsupported)
            .expect("a full-attention group carries no window and needs no route capability");
        assert!(!slot.is_windowed());
        assert_eq!(slot.kernel_slot(), None);
        assert_eq!(slot.mask_window(), None);
    }

    /// A kernel-argument route gets the window as the `sliding_window: i32` FFI
    /// value and NO mask — the kernel derives its own per-query-row lower bound.
    /// The value must be the config's 2048, and it must not be reachable as a 0.
    ///
    /// Mutation caught: `kernel_slot` returning `Some(0)`, which is the literal
    /// that caused the defect (`paged_kv_cache_adapter.rs`'s varlen call site),
    /// and is indistinguishable from a full-attention dispatch.
    #[test]
    fn a_kernel_route_carries_the_window_as_the_ffi_argument_and_no_mask() {
        let slot = paged_window_for_kind(
            AttentionKind::SlidingWindow {
                sliding_window: 2048,
            },
            WindowCarrier::KernelArgument,
        )
        .expect("a kernel-argument route can carry a window");
        assert_eq!(slot.kernel_slot(), Some(2048));
        assert_eq!(
            slot.mask_window(),
            None,
            "a kernel route must not also build a mask; double-masking is not the failure \
             mode, but a caller that builds one has misread which route it is on"
        );
        assert!(slot.is_windowed());
        assert_eq!(slot.carrier(), WindowCarrier::KernelArgument);
    }

    /// A mask route gets the window as `create_causal_mask`'s third argument and
    /// NO kernel slot. `create_causal_mask(seq_len, offset, Some(w))` computes
    /// `linds >= rinds && linds < rinds + w` (`array/mask.rs:33-74`), which is
    /// the same predicate the kernel applies; the argument already exists and
    /// gemma4's host-read path passed `None` into it.
    ///
    /// Mutation caught: `mask_window` returning `None` for a windowed
    /// mask route — the exact `None` that shipped — or `kernel_slot` answering
    /// on a mask route, which would put a silent 0 in a kernel argument.
    #[test]
    fn a_mask_route_carries_the_window_as_create_causal_mask_window_size() {
        let slot = paged_window_for_kind(
            AttentionKind::SlidingWindow {
                sliding_window: 2048,
            },
            WindowCarrier::ExplicitMask,
        )
        .expect("an explicit-mask route can carry a window");
        assert_eq!(slot.mask_window(), Some(2048));
        assert_eq!(
            slot.kernel_slot(),
            None,
            "a mask route has no kernel sliding_window slot, and answering with 0 here \
             would reproduce the defect in a second file"
        );
    }

    /// The accessors are carrier-scoped, and the property that matters is
    /// negative: NO combination of a sliding kind and any admitted route can
    /// produce a kernel slot of `Some(0)` or a mask of `Some(0)`. Enumerated
    /// rather than spot-checked, because "the one combination nobody enumerated"
    /// is how the defect survived.
    #[test]
    fn no_admitted_sliding_dispatch_can_ever_present_a_zero_window() {
        for carrier in [
            WindowCarrier::KernelArgument,
            WindowCarrier::ExplicitMask,
            WindowCarrier::Unsupported,
        ] {
            let admitted = paged_window_for_kind(
                AttentionKind::SlidingWindow {
                    sliding_window: 2048,
                },
                carrier,
            );
            let Ok(slot) = admitted else {
                // Refused — the other half of the contract, covered above.
                assert!(!carrier.can_carry_a_window());
                continue;
            };
            assert!(
                carrier.can_carry_a_window(),
                "{carrier:?} reported no window capability yet was admitted with a window"
            );
            assert_ne!(
                slot.kernel_slot(),
                Some(0),
                "{carrier:?} presented window 0"
            );
            assert_ne!(
                slot.mask_window(),
                Some(0),
                "{carrier:?} presented window 0"
            );
            assert!(
                slot.kernel_slot() == Some(2048) || slot.mask_window() == Some(2048),
                "{carrier:?} was admitted with a window but presents it nowhere; a route \
                 that carries the window through neither accessor is window-blind with \
                 extra steps"
            );
        }
    }

    /// A full-attention group must keep the UNMASKED causal fast path. This is
    /// not tidiness: the mask-bearing SDPA kernel has a different bf16 reduction
    /// order from the `_causal` one, so materializing an all-true mask for the 13
    /// global layers would move their numerics with no correctness gain — and a
    /// paged-vs-flat parity gate would then fail for a reason unrelated to
    /// windows.
    ///
    /// Mutation caught: `mask_window` returning `Some(0)` (or `Some(window)`
    /// with `window == 0`) for a full group instead of `None`.
    #[test]
    fn a_full_group_asks_for_no_mask_so_the_causal_fast_path_survives() {
        for carrier in [
            WindowCarrier::KernelArgument,
            WindowCarrier::ExplicitMask,
            WindowCarrier::Unsupported,
        ] {
            let slot = paged_window_for_kind(AttentionKind::Full, carrier)
                .expect("a full-attention group is admissible on every route");
            assert_eq!(
                slot.mask_window(),
                None,
                "{carrier:?} must not ask a full group to build a window mask"
            );
            assert!(!slot.is_windowed());
        }
        // On a kernel route the full group's FFI value is a real, intended 0 —
        // the kernel's documented "disable the sliding mask" sentinel. That is
        // the ONE place a 0 is correct, and it is reachable only from
        // `AttentionKind::Full`.
        let slot = paged_window_for_kind(AttentionKind::Full, WindowCarrier::KernelArgument)
            .expect("full attention on a kernel route");
        assert_eq!(slot.kernel_slot(), Some(0));
    }

    /// A `SlidingWindow { sliding_window: 0 }` spec is refused a third time, at
    /// the last point before a kernel launch. Config load refuses it
    /// (`config::tests::rejects_a_zero_sliding_window`) and spec derivation
    /// refuses it (`refuses_a_zero_sliding_window_even_though_config_load_already_did`),
    /// but neither runs on a group assembled by other means — and by the time a 0
    /// reaches a kernel slot it is indistinguishable from a deliberate
    /// "disable the mask".
    ///
    /// Mutation caught: deleting the `sliding_window: 0` arm, which would admit
    /// the zero window silently on a KernelArgument route as `Some(0)`.
    #[test]
    fn a_zero_window_spec_is_refused_at_dispatch_admission_too() {
        for carrier in [
            WindowCarrier::KernelArgument,
            WindowCarrier::ExplicitMask,
            WindowCarrier::Unsupported,
        ] {
            let err =
                paged_window_for_kind(AttentionKind::SlidingWindow { sliding_window: 0 }, carrier)
                    .expect_err("a zero window must not reach a dispatch");
            assert!(
                err.contains("disable the sliding mask"),
                "the error must say what a 0 means to the kernel, not merely that it is \
                 invalid, got: {err}"
            );
        }
    }

    /// The per-turn choke point admits both groups together, in `group_id`
    /// order, and hands back one slot per group. This is the call M1 must make
    /// once per turn.
    ///
    /// Mutation caught: `admit_paged_dispatch_plan` returning slots in
    /// iteration order of a set, or admitting only `groups[0]`.
    /// The negative-window analogue of
    /// `no_admitted_sliding_dispatch_can_ever_present_a_zero_window`, and the one
    /// that closes a real hole rather than restating a satisfied invariant.
    ///
    /// `AttentionKind` stores the window as a `u32`, but both consumers take an
    /// `i32`, so the band `[2^31, 2^32-1]` fits every type on the way down and
    /// still arrives at a dispatch NEGATIVE. Measured before the fix, on this
    /// exact seam: window 3_000_000_000 gave `kernel_slot() == Some(-1294967296)`
    /// and `mask_window() == Some(-1294967296)`; 2_147_483_648 gave
    /// `Some(-2147483648)`; `u32::MAX` gave `Some(-1)`.
    ///
    /// The two routes then behave differently, and only one of them is loud:
    ///
    ///   * KernelArgument fails CLOSED — the C++ validator rejects a negative
    ///     window (`mlx_paged_ops.cpp:483`), which surfaces as an `Err`. For that
    ///     route this refusal is a DIAGNOSTICS improvement: same outcome, named at
    ///     the config field instead of six frames into the FFI.
    ///   * ExplicitMask fails OPEN and SILENT — `create_causal_mask` keeps a cell
    ///     only when `linds < rinds + window_size` (`array/mask.rs:63`), so a
    ///     negative width keeps nothing. On real MLX that is 0 of 96 cells kept
    ///     versus 68 causal, and the fused SDPA fed an all-false mask returns the
    ///     uniform mean of EVERY V row: finite, plausible, no NaN, no error,
    ///     max|delta| 1.2485 against the windowed reference. This is the half the
    ///     refusal actually closes, and it is the same failure shape as the gemma4
    ///     dropped window.
    ///
    /// So the assertion is deliberately "Err", not "Some(0)" or "not negative":
    /// there is no legal thing for the seam to substitute. Refusing is the whole
    /// answer.
    ///
    /// Mutation caught: deleting the `i32::try_from` arm in
    /// `paged_window_for_kind` — every REFUSED value becomes `Ok` and presents a
    /// negative window on its carrier's accessor.
    #[test]
    fn no_admitted_dispatch_can_ever_present_a_negative_window() {
        // Every value that fits the spec's `u32` but not an `i32`.
        const REFUSED: [u32; 4] = [
            i32::MAX as u32 + 1, // 2_147_483_648 -> i32::MIN
            3_000_000_000,       // the reviewer's value -> -1_294_967_296
            u32::MAX - 1,
            u32::MAX, // -> -1
        ];
        // The real checkpoint's window, and the largest legal one. `i32::MAX` must
        // keep round-tripping: a window bigger than the context is legal, and
        // gemma4 decides "the window cannot bite" by comparing the two, which
        // needs the window representable first.
        const ACCEPTED: [u32; 3] = [2048, 1 << 30, i32::MAX as u32];

        for carrier in [
            WindowCarrier::KernelArgument,
            WindowCarrier::ExplicitMask,
            WindowCarrier::Unsupported,
        ] {
            for window in REFUSED {
                let err = paged_window_for_kind(
                    AttentionKind::SlidingWindow {
                        sliding_window: window,
                    },
                    carrier,
                )
                .expect_err(
                    "a window past i32::MAX cannot reach a dispatch intact and must be \
                     refused, not narrowed",
                );
                assert!(
                    err.contains("exceeds i32::MAX"),
                    "{carrier:?} window {window}: the refusal must name the i32 bound \
                     rather than reading as an unrelated failure, got: {err}"
                );
                assert!(
                    err.contains("masks EVERY key"),
                    "{carrier:?} window {window}: the refusal must say what a negative \
                     width does to the mask route, because that is the silent half — an \
                     error that only says \"invalid\" gets read as pedantry and deleted, \
                     got: {err}"
                );
            }

            for window in ACCEPTED {
                let slot = paged_window_for_kind(
                    AttentionKind::SlidingWindow {
                        sliding_window: window,
                    },
                    carrier,
                );
                let Ok(slot) = slot else {
                    // Refused for the OTHER reason — an incapable route. Covered by
                    // `a_sliding_group_on_a_window_blind_route_is_refused`.
                    assert!(!carrier.can_carry_a_window());
                    continue;
                };
                let expected = i32::try_from(window).expect("ACCEPTED values fit an i32");
                match carrier {
                    WindowCarrier::KernelArgument => {
                        assert_eq!(slot.kernel_slot(), Some(expected));
                        assert_eq!(slot.mask_window(), None);
                    }
                    WindowCarrier::ExplicitMask => {
                        assert_eq!(slot.mask_window(), Some(expected));
                        assert_eq!(slot.kernel_slot(), None);
                    }
                    WindowCarrier::Unsupported => unreachable!("refused above"),
                }
                assert!(
                    slot.kernel_slot().unwrap_or(1) > 0 && slot.mask_window().unwrap_or(1) > 0,
                    "{carrier:?} window {window}: an admitted sliding dispatch must \
                     present a strictly positive window wherever it presents one"
                );
            }
        }
    }

    /// The same bound on the second line of defence. `compute_layer_kv_cache_specs`
    /// is `pub`, so a config that never went through `from_json_str` — or one
    /// mutated afterwards, since the fields are `pub` — must not be able to mint a
    /// `SlidingWindow` spec that only fails later, at a dispatch. Measured before
    /// the fix: this returned `Ok` with
    /// `SlidingWindow { sliding_window: 3000000000 }` on all 39 sliding layers.
    ///
    /// Mutation caught: reverting the derivation's guard to `u32::try_from`.
    #[test]
    fn a_sliding_window_past_i32_max_never_becomes_a_spec() {
        let mut cfg = checkpoint_config();
        for window in [i32::MAX as usize + 1, 3_000_000_000, u32::MAX as usize] {
            cfg.text_config.sliding_window = window;
            let err = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
                .expect_err("a window past i32::MAX must not become 39 sliding specs");
            assert!(
                err.contains("exceeds i32::MAX") && err.contains(&window.to_string()),
                "the refusal must name the bound and print the observed window {window}, \
                 got: {err}"
            );
        }

        // And the boundary stays legal, all the way to a group.
        cfg.text_config.sliding_window = i32::MAX as usize;
        let specs = compute_layer_kv_cache_specs(&cfg, 16, KVCacheDType::BFloat16)
            .expect("i32::MAX is the largest legal window and must still derive");
        assert_eq!(
            specs[0].attention_kind,
            AttentionKind::SlidingWindow {
                sliding_window: i32::MAX as u32
            }
        );
    }

    #[test]
    fn the_dispatch_plan_admits_every_group_in_group_id_order() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);
        let slots = admit_paged_dispatch_plan(
            &groups,
            &[WindowCarrier::ExplicitMask, WindowCarrier::KernelArgument],
        )
        .expect("both groups are on window-capable routes");

        assert_eq!(slots.len(), groups.len());
        // group 0 is full, on a mask route: nothing to carry.
        assert!(!slots[0].is_windowed());
        assert_eq!(slots[0].mask_window(), None);
        // group 1 is the 39 sliding layers, on a kernel route: 2048 in the slot.
        assert!(slots[1].is_windowed());
        assert_eq!(slots[1].kernel_slot(), Some(2048));
    }

    /// A plan that forgets a group is an arity error, not a group that
    /// dispatches unadmitted. This is the structural half of the gemma4 shape:
    /// there, the sliding group and the full group were planned independently
    /// and nothing compared them.
    ///
    /// Mutation caught: replacing the arity check with
    /// `carriers.get(index).copied().unwrap_or(WindowCarrier::Unsupported)` or
    /// with `zip`, either of which silently drops the trailing group.
    #[test]
    fn a_dispatch_plan_that_forgets_a_group_is_an_error_not_a_default() {
        let cfg = checkpoint_config();
        let groups = groups(&cfg, CHUNK);
        let err = admit_paged_dispatch_plan(&groups, &[WindowCarrier::KernelArgument])
            .expect_err("one carrier for two groups must not be silently extended");
        assert!(
            err.contains("2 KV cache group(s) but 1 route carrier(s)"),
            "the error must report both counts, got: {err}"
        );

        let err = admit_paged_dispatch_plan(&[], &[])
            .expect_err("an empty group list is not a valid hybrid plan");
        assert!(err.contains("hybrid by construction"), "got: {err}");
    }

    /// Carriers are positional AND keyed on `group_id`; a permutation would pair
    /// the full group's route with the sliding group's window, which is the
    /// defect with a different cause. Checked rather than assumed because the
    /// two indexings are independent facts.
    ///
    /// Mutation caught: deleting the `group.group_id != index` check.
    #[test]
    fn a_permuted_group_list_is_refused_because_carriers_are_positional() {
        let cfg = checkpoint_config();
        let mut groups = groups(&cfg, CHUNK);
        groups.swap(0, 1);
        let err = admit_paged_dispatch_plan(
            &groups,
            &[WindowCarrier::KernelArgument, WindowCarrier::KernelArgument],
        )
        .expect_err("a permuted group list must not be admitted positionally");
        assert!(
            err.contains("reports group_id"),
            "the error must name the mismatch between position and group_id, got: {err}"
        );
    }

    /// The window a dispatch carries is the CONFIG's window, all the way
    /// through: not a constant, not the admission-block count, not
    /// `max_chunk`. 2048 is also `max_chunk * 4` and `block_size * 128` in the
    /// reference geometry, so a wrong source is locally plausible.
    ///
    /// Mutation caught: `paged_window_for_kind` sourcing the window from
    /// anything other than the `AttentionKind::SlidingWindow` payload.
    #[test]
    fn the_dispatch_window_tracks_the_config_and_is_not_a_constant() {
        let mut cfg = checkpoint_config();
        cfg.text_config.sliding_window = 777;
        let groups = groups(&cfg, CHUNK);
        let sliding = groups
            .iter()
            .find(|g| matches!(g.attention_kind, AttentionKind::SlidingWindow { .. }))
            .expect("still a hybrid");
        let slot = paged_window_for_group(sliding, WindowCarrier::KernelArgument)
            .expect("a kernel route carries any non-zero window");
        assert_eq!(slot.kernel_slot(), Some(777));
    }

    /// TRIPWIRE for the half this module cannot type-check.
    ///
    /// Stated plainly, because overstating it is how the dropped mask survived:
    /// this test does NOT prove every dispatch asks the seam for its window. The
    /// type system cannot prove that — a forward pass can always call the FFI
    /// with a hand-written argument. What it proves is narrower and still worth
    /// having: the moment paged-attention wiring appears anywhere in this family,
    /// the file that wires it must at least NAME `PagedWindowSlot`. So M1 cannot
    /// wire a sliding adapter, forget this contract entirely, and stay green —
    /// which is exactly what happened in gemma4, where no test occupied the
    /// "sliding adapter x attention numerics" cell at all.
    ///
    /// It is vacuously true today (no forward pass exists) and that is the
    /// point: it is armed, not satisfied.
    ///
    /// ## What it does NOT cover, named rather than implied
    ///
    /// 1. **Textual, not semantic.** It checks that the string
    ///    `PagedWindowSlot` appears in the same file as the wiring. A `use
    ///    super::kv_cache::PagedWindowSlot;` next to a hand-written FFI call
    ///    satisfies it. Naming the seam is the floor, not the contract.
    /// 2. **Per-file, not per-call.** A file that admits one dispatch through
    ///    the seam and launches a second one with a literal passes.
    /// 3. **This family's directory only.** Wiring placed in a shared file —
    ///    `transformer/block.rs`, a new generic paged forward — trips nothing
    ///    here. That gap is covered from the other side, in the adapter, but the
    ///    two window-blind dense readers refuse with DIFFERENT strengths and the
    ///    difference matters to whoever leans on this:
    ///
    ///    * `gather_kv_for_prefill_sdpa` refuses a sliding group
    ///      unconditionally — one `if self.sliding_window != 0 { Err }`, no
    ///      argument inspected.
    ///    * `read_kv_range` refuses on TWO grounds, neither of which is "is a
    ///      sliding group": by WIDTH, a read of more tokens than the window; and
    ///      by AGE, a read starting below the live floor
    ///      `num_tokens - sliding_window`, which is where
    ///      `prune_sliding_window_for` puts its cutoff. The age arm exists
    ///      because width alone does not imply liveness: a read whose length
    ///      equals the window slips the width check no matter how old it is, and
    ///      after a prune those positions sit on the never-written null block.
    ///      An earlier revision of this comment claimed the reader refuses a
    ///      sliding group outright, which it does not; a later one described
    ///      only the width arm, which understated it.
    ///
    ///    The narrower guarantee is still enough for the claim being made here,
    ///    and that is why the wording is worth getting exact rather than
    ///    rounding up. A window-blind dense route reads the whole context,
    ///    `read_kv_range(layer, 0, total_ctx)`, and a window can only bite when
    ///    `total_ctx > window` — which is exactly the width the reader refuses.
    ///    When `total_ctx <= window` the read is served, and being window-blind
    ///    there is numerically identical to carrying the window, because nothing
    ///    is out of range and nothing has been retired to the null block. So a
    ///    shared route cannot serve a sliding group window-blind in the regime
    ///    where that would be wrong, regardless of which family calls it.
    ///
    /// The scan itself IS recursive, so a `muse_glimmer/attention/paged.rs`
    /// subdirectory is opened — this repo already ships that layout elsewhere
    /// (`models/qwen3_5/gdn_checkpoint_store/` beside
    /// `models/qwen3_5/paged_forward.rs`), and a non-recursive `read_dir` would
    /// have gone silently vacuous the moment the family grew one.
    ///
    /// Mutations caught: adding `let w = 0i32;` next to a
    /// `mlx_paged_attention_varlen_forward` call in a new
    /// `muse_glimmer/attention.rs`, or in a new
    /// `muse_glimmer/attention/paged.rs`, that never mentions the seam;
    /// dropping the recursive descent (the file-count floor below then fails
    /// once a subdirectory exists).
    #[test]
    fn wiring_a_sliding_adapter_without_this_seam_trips_the_tripwire() {
        // Names that only real paged wiring uses. Deliberately NOT
        // `AttentionKind::SlidingWindow` or `sliding_window`, which this seam
        // itself talks about constantly.
        const WIRING_MARKERS: [&str; 6] = [
            "PagedKVCacheAdapter::new_sliding",
            "mlx_paged_attention",
            "gather_kv_for_prefill",
            "read_kv_range",
            "scaled_dot_product_attention_causal",
            "PagedKVCacheAdapter",
        ];
        const SEAM: &str = "PagedWindowSlot";

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("models")
            .join("muse_glimmer");

        // Recursive: a `muse_glimmer/attention/paged.rs` must be opened, not
        // skipped. A non-recursive scan is how this check goes silently
        // vacuous.
        let mut pending = vec![root.clone()];
        let mut sources: Vec<(std::path::PathBuf, String)> = Vec::new();
        while let Some(dir) = pending.pop() {
            let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| {
                panic!("{} must be readable: {e}", dir.display());
            });
            for entry in entries {
                let path = entry.expect("a readable directory entry").path();
                if path.is_dir() {
                    pending.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
                sources.push((path, source));
            }
        }

        let mut checked = 0usize;
        for (path, source) in &sources {
            checked += 1;

            // Strip whole-line comments so this file's own prose about the
            // defect does not trip its own tripwire. A trailing comment after
            // code would still trip it; that is a deliberate false-positive
            // bias — the failure message tells the reader what to do.
            let code: String = source
                .lines()
                .filter(|line| {
                    let t = line.trim_start();
                    !(t.starts_with("//") || t.starts_with("*") || t.starts_with("/*"))
                })
                .collect::<Vec<&str>>()
                .join("\n");

            let Some(marker) = WIRING_MARKERS.iter().find(|m| code.contains(**m)) else {
                continue;
            };
            assert!(
                code.contains(SEAM),
                "{} wires paged attention (found {marker:?}) without naming {SEAM}. Every \
                 Muse-Glimmer paged dispatch must take its sliding_window from \
                 `admit_paged_dispatch_plan` / `paged_window_for_group` in \
                 muse_glimmer/kv_cache.rs. 39 of this decoder's 52 layers are sliding at \
                 window 2048; a dispatch that fills the kernel's sliding_window slot with \
                 a literal, or runs SDPA with no mask, attends every position back to 0 \
                 including positions already retired to the never-written null block. That \
                 is the confirmed gemma4 defect, measured at 3.5x the correct signal's RMS \
                 on real weights. If this file legitimately mentions a marker in a \
                 trailing comment, move it to its own comment line.",
                path.display()
            );
        }
        // Floor equal to the family's CURRENT file count, not a loose 3. Every
        // file added to the family must raise it, which is the one moment a
        // reader is forced to look at this test — and it is the only way "the
        // scan still reaches everything" can be checked at all, since a scan
        // that silently stopped descending would otherwise just report fewer
        // files and still pass a loose floor.
        const FAMILY_FILES: usize = 14;
        assert!(
            checked >= FAMILY_FILES,
            "the tripwire scanned only {checked} file(s), fewer than the {FAMILY_FILES} this \
             family is known to have. Either the family moved and this test must be \
             repointed, or the recursive descent stopped working — in both cases the check \
             is silently vacuous. Scanned: {:?}",
            sources
                .iter()
                .map(|(p, _)| p.file_name())
                .collect::<Vec<_>>()
        );
        // And every added file must be scanned, so growing the family without
        // raising `FAMILY_FILES` is caught from the other side too.
        assert!(
            checked <= FAMILY_FILES,
            "the tripwire scanned {checked} file(s) but FAMILY_FILES says {FAMILY_FILES}. \
             The family grew: raise FAMILY_FILES, and while you are here confirm the new \
             file either names {SEAM} or wires no paged attention. Scanned: {:?}",
            sources
                .iter()
                .map(|(p, _)| p.file_name())
                .collect::<Vec<_>>()
        );
    }
}
