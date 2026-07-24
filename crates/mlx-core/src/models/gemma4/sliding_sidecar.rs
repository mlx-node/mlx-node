//! Gemma4's sliding-window state as a cold-tier [`ColdSidecar`].
//!
//! # Why gemma4 needs one at all
//!
//! Gemma4 is a HYBRID: `Gemma4Inner` sizes its paged pool over the PHYSICAL
//! FULL-ATTENTION layers only (`physical_full_attention_layer_count`, and the
//! `num_layers` it feeds into `PagedAttentionConfig`). The interleaved sliding
//! layers keep their K/V in a per-layer [`RotatingKVCache`] that the pool never
//! sees. A restored paged prefix therefore reconstructs only part of the model
//! state, and the missing half has to come from somewhere: today it is
//! RECOMPUTED by `run_sliding_only_prefill`, which re-runs the decoder over the
//! whole reused prefix. This sidecar persists it instead.
//!
//! # The payload
//!
//! One [`RotatingKVCacheSnapshot`] per PHYSICAL sliding layer — `keys` in slot
//! 0, `values` in slot 1, layer-major in decoder order. "Physical" is
//! load-bearing: a `SharedOnSliding` KV-shared alias consumes its anchor's
//! stash and never advances a slot of its own (see
//! `gemma4_sliding_caches_ready_at`), so it holds no state. Counting aliases in
//! `num_layers` would shift every later layer's tensors by the alias count and
//! restore plausible-looking garbage rather than fail.
//!
//! # Why a boundary below one window is still a real state
//!
//! A rotating cache holds `min(offset, window)` tokens, so its payload length
//! is constant only once `offset >= window`. That is a PAYLOAD-FORMAT fact,
//! not a hit rule, and the two must not be conflated: a 512-row snapshot at
//! offset 512 with `max_size` 1024 is a first-class PRE-WRAP state.
//! `RotatingKVCache::restore_snapshot` hard-checks
//! `cached_tokens == expected_cached_tokens(offset, max_size)`, and 512 is
//! exactly what that returns — it is neither a truncation nor a pad, and
//! padding is still not an escape hatch (a full-window payload at offset 512
//! would be rejected).
//!
//! So the layout is boundary-invariant in SHAPE RULE, not in byte count:
//! [`ColdSidecarPolicy::new_boundary_scaled`] declares the token axis as the
//! one that follows the boundary, and `expected_at` stamps
//! `dims[2] = min(boundary, window)` with `bytes_per_tensor` scaled to match.
//! [`layout_at`] derives the identical value on the capture side, so
//! `load_sidecar`'s layout-equality check still compares two independently
//! computed layouts. At and above the window the rule is the IDENTITY, so
//! every sidecar written before the axis existed still loads — no fingerprint
//! or on-disk format change.
//!
//! Refusing sub-window boundaries is what made this sidecar INERT: gemma4's
//! real window is 1024 and typical chat prompts are shorter, so
//! `ColdTierWalk::deepest_backed_boundary` found nothing to back and the whole
//! restore aborted while capture still wrote K/V blocks nobody could read
//! back. Zero remains unrepresentable in both directions —
//! `ColdSidecar::validate` refuses a zero boundary and `restore_snapshot`
//! refuses a zero-row cache — and that stays the fail-closed floor: below it
//! the walk reconciles down and recomputes.
//!
//! # Why the chain key, not a window-local key
//!
//! A sliding layer's K/V at position `i` is projected from a hidden state that
//! has already passed through the interleaved FULL-attention layers, which
//! attend to the ENTIRE prefix. The state at a boundary is therefore NOT a
//! function of the last `window` tokens, and two prompts agreeing only on their
//! last `window` tokens must never share a sidecar. The parent-linked chain
//! from block 0 that `ColdTierWalk::deepest_backed_boundary` builds is exactly
//! the right identity — the same per-block `extra_keys` (image-aware) and
//! `cache_salt` as the K/V chain, under the `SlidingWindow` domain tag.

use mlx_paged_attn::{ColdGroup, ColdSidecarLayout, ColdSidecarPolicy};
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::transformer::rotating_kv_cache::RotatingKVCacheSnapshot;

use super::config::Gemma4Config;

/// `keys` then `values`. Sliding layers never share K with V —
/// `attention.rs` gates `k_is_v` on `is_global`, so the count is fixed at 2.
pub(crate) const TENSORS_PER_LAYER: u32 = 2;

/// The one element type this sidecar is written and read in. The snapshot type
/// promises no dtype (see `gemma4_sliding_checkpoint_estimated_bytes`, which
/// budgets 4 bytes/elem for exactly that reason), so an f32 sliding cache must
/// SKIP capture rather than write f32 bytes under a bf16 label. `load_sidecar`
/// compares the whole layout, so a mismatch on the read side is already a miss.
const SIDECAR_DTYPE: &str = "BFloat16";

/// Geometry of gemma4's out-of-pool sliding state, derived once from the
/// config. Every field is boundary-invariant, which is what lets it back a
/// [`ColdSidecarPolicy`]; the one thing a boundary moves is how many of the
/// window's token rows a payload actually carries (see [`payload_tokens`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SlidingSidecarGeometry {
    /// `config.sliding_window` — the rotating cache's `max_size`, and the
    /// CEILING on the token count a payload carries.
    pub window: u32,
    /// Count of PHYSICAL sliding layers: `is_sliding_layer && !is_kv_shared_layer`.
    pub physical_layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl SlidingSidecarGeometry {
    /// `[batch, kv_heads, tokens, head_dim]` — the shape
    /// `RotatingKVCache::restore_snapshot` validates against, with `batch`
    /// pinned to 1 (the adapter is scoped to one request at a time). The
    /// TOKEN axis is index 2, which is the axis
    /// [`ColdSidecarPolicy::new_boundary_scaled`] is told to scale.
    fn dims(&self, tokens: u32) -> Vec<u32> {
        vec![1, self.kv_heads, tokens, self.head_dim]
    }

    fn bytes_per_tensor(&self, tokens: u32) -> Option<usize> {
        self.elements_per_tensor(tokens)?
            .checked_mul(std::mem::size_of::<u16>())
    }

    /// Elements per state tensor, i.e. the `to_uint16_native` length a capture
    /// must produce and a restore must consume.
    fn elements_per_tensor(&self, tokens: u32) -> Option<usize> {
        (self.kv_heads as usize)
            .checked_mul(tokens as usize)?
            .checked_mul(self.head_dim as usize)
    }

    /// Geometry bytes folded into the cold-tier fingerprint.
    ///
    /// [`crate::cold_tier::ColdTierGeometry`] describes the POOL, which for
    /// gemma4 covers only the full-attention layers — two configs with the same
    /// full-attention pool but a different `sliding_window` or a different
    /// `layer_types` split would produce the same pool geometry. The serialized
    /// `Gemma4Config` blob already carries those fields, but binding them
    /// explicitly makes the sidecar's identity independent of that blob's
    /// serde surface ever changing.
    pub fn fingerprint_component(&self) -> Vec<u8> {
        format!(
            "gemma4-sliding-sidecar:v1:window={}:layers={}:kv_heads={}:head_dim={}:dtype={}:tensors={}",
            self.window,
            self.physical_layers,
            self.kv_heads,
            self.head_dim,
            SIDECAR_DTYPE,
            TENSORS_PER_LAYER,
        )
        .into_bytes()
    }
}

/// Decoder indices of the layers that OWN sliding state, in order. The
/// `layer * TENSORS_PER_LAYER + slot` ordering of the payload is stated in
/// terms of this vector's positions.
pub(crate) fn physical_sliding_layers(config: &Gemma4Config) -> Vec<usize> {
    (0..config.num_hidden_layers.max(0) as usize)
        .filter(|&i| config.is_sliding_layer(i) && !config.is_kv_shared_layer(i))
        .collect()
}

/// Derive the sidecar geometry, or `None` when this checkpoint has no
/// out-of-pool sliding state to persist (an all-global `layer_types`, or a
/// nonsensical window/head geometry). `None` means "install no policy": the
/// family then behaves exactly as it did before sidecars existed.
pub(crate) fn geometry(config: &Gemma4Config) -> Option<SlidingSidecarGeometry> {
    let physical_layers = u32::try_from(physical_sliding_layers(config).len()).ok()?;
    if physical_layers == 0 {
        return None;
    }
    let window = u32::try_from(config.sliding_window).ok()?;
    let kv_heads = u32::try_from(config.effective_kv_heads(false)).ok()?;
    let head_dim = u32::try_from(config.effective_head_dim(false)).ok()?;
    if window == 0 || kv_heads == 0 || head_dim == 0 {
        return None;
    }
    let geo = SlidingSidecarGeometry {
        window,
        physical_layers,
        kv_heads,
        head_dim,
    };
    // A geometry whose LARGEST payload cannot be sized is not persistable at
    // all; every sub-window payload is a fraction of it.
    geo.bytes_per_tensor(window)?;
    Some(geo)
}

/// Token rows a sidecar anchored at `boundary_tokens` carries: exactly what a
/// live `RotatingKVCache` holds there
/// (`RotatingKVCache::expected_cached_tokens` = `min(offset, max_size)`).
///
/// Total, and deliberately the SAME `min` [`ColdSidecarPolicy::expected_at`]
/// applies to its scaled axis — including at `boundary_tokens == 0`, where
/// both yield zero rows. The policy template is built by [`template_layout`]
/// rather than by passing 0 here, so that the read and capture sides never
/// disagree about what 0 means.
fn payload_tokens(geo: &SlidingSidecarGeometry, boundary_tokens: u32) -> u32 {
    boundary_tokens.min(geo.window)
}

/// The layout a sidecar anchored at `boundary_tokens` must have. `boundary` is
/// stamped by [`ColdSidecarPolicy::expected_at`] on the read side; the capture
/// side builds the same value here so the two compare equal.
///
/// The token axis and `bytes_per_tensor` both come from [`payload_tokens`],
/// which restates the same `min(boundary, extent)` rule the policy's scaled
/// axis applies. The two are derived independently and must agree — that is
/// what makes `load_sidecar`'s layout-equality check worth anything, and it is
/// pinned by `policy_expected_at_matches_layout_at_on_every_candidate_boundary`.
pub(crate) fn layout_at(geo: &SlidingSidecarGeometry, boundary_tokens: u32) -> ColdSidecarLayout {
    layout_carrying(geo, boundary_tokens, payload_tokens(geo, boundary_tokens))
}

/// The geometry template [`policy`] declares a boundary-scaled axis on: a
/// FULL-window payload labelled `boundary_tokens: 0`.
///
/// Not `layout_at(geo, 0)`. [`ColdSidecarPolicy::new_boundary_scaled`] clamps
/// every scaled extent against the template's own, so a zero-row template
/// would freeze the token axis at zero and scale every real boundary to
/// nothing — and `validate_geometry` would reject it first. The label stays 0
/// because the policy is boundary-invariant and `expected_at` overwrites it.
fn template_layout(geo: &SlidingSidecarGeometry) -> ColdSidecarLayout {
    layout_carrying(geo, 0, geo.window)
}

/// Shared constructor: a layout LABELLED `boundary_tokens` that physically
/// carries `tokens` rows. The two differ only for [`template_layout`].
fn layout_carrying(
    geo: &SlidingSidecarGeometry,
    boundary_tokens: u32,
    tokens: u32,
) -> ColdSidecarLayout {
    ColdSidecarLayout {
        group: ColdGroup::SlidingWindow,
        boundary_tokens,
        num_layers: geo.physical_layers,
        tensors_per_layer: TENSORS_PER_LAYER,
        dtype: SIDECAR_DTYPE.to_string(),
        dims: geo.dims(tokens),
        // The full-window `bytes_per_tensor` is checked non-`None` by
        // `geometry`, and `tokens <= window`, so this cannot overflow either.
        bytes_per_tensor: geo.bytes_per_tensor(tokens).unwrap_or(0),
    }
}

/// Build the reconcile-down policy for this config, or `None` when there is no
/// sliding state to back (see [`geometry`]).
///
/// The template is [`template_layout`] — a FULL-window payload — with axis 2
/// (the token axis of `[1, kv_heads, window, head_dim]`) declared
/// boundary-scaled, so `expected_at(b)` reproduces `layout_at(&geo, b)` for
/// every `b > 0`.
pub(crate) fn policy(config: &Gemma4Config) -> Option<ColdSidecarPolicy> {
    let geo = geometry(config)?;
    ColdSidecarPolicy::new_boundary_scaled(template_layout(&geo), 2).ok()
}

/// Whether `boundary_tokens` is representable by this policy: a positive
/// multiple of `block_size`.
///
/// There is no window floor. The payload carries `min(boundary, window)` rows
/// (see [`payload_tokens`] and the module doc), which is exactly the row count
/// a live rotating cache holds at that offset, so a sub-window boundary has a
/// legal layout and a restorable state. Zero is still refused: it describes an
/// empty payload, which `ColdSidecar::validate` rejects anyway.
pub(crate) fn boundary_is_representable(
    geo: &SlidingSidecarGeometry,
    boundary_tokens: u32,
    block_size: u32,
) -> bool {
    boundary_tokens > 0
        && block_size > 0
        && boundary_tokens.is_multiple_of(block_size)
        // Already guaranteed by `geometry` (which sizes the LARGEST payload)
        // plus `boundary_tokens > 0`, but restated so representability never
        // outruns what this geometry can actually encode.
        && geo
            .bytes_per_tensor(payload_tokens(geo, boundary_tokens))
            .is_some_and(|bytes| bytes > 0)
}

/// Serialize the in-memory snapshots for one boundary into the layer-major
/// tensor payload.
///
/// `snapshots` is indexed by DECODER layer (length `num_hidden_layers`, `None`
/// at global and KV-shared slots), exactly as `snapshot_gemma4_sliding_caches`
/// produces it.
///
/// Returns `Ok(None)` — a graceful skip, never an error — when the state does
/// not match the layout this boundary implies: a missing physical layer, a
/// wrong offset, a non-bf16 element type, or a shape that is not
/// `[1, kv_heads, min(boundary, window), head_dim]`. Skipping is the
/// fail-closed direction: writing bytes that do not match the label would
/// produce a sidecar that restores wrong state instead of missing.
///
/// The caller must have materialized the snapshots first
/// (`materialize_gemma4_sliding_snapshots` = `copy()` + `eval_arrays`); the
/// `copy` is mandatory because `temporal_order` can hand back a clone that
/// ALIASES live storage which the single-token update path rewrites in place.
pub(crate) fn encode_tensors(
    config: &Gemma4Config,
    geo: &SlidingSidecarGeometry,
    snapshots: &[Option<RotatingKVCacheSnapshot>],
    boundary_tokens: u32,
) -> Result<Option<Vec<Vec<u8>>>> {
    let layers = physical_sliding_layers(config);
    if layers.len() != geo.physical_layers as usize
        || snapshots.len() != config.num_hidden_layers.max(0) as usize
    {
        return Ok(None);
    }
    let cached = payload_tokens(geo, boundary_tokens);
    let Some(elements) = geo.elements_per_tensor(cached) else {
        return Ok(None);
    };
    let expected_shape = [
        1i64,
        geo.kv_heads as i64,
        cached as i64,
        geo.head_dim as i64,
    ];

    let mut tensors = Vec::with_capacity(layers.len() * TENSORS_PER_LAYER as usize);
    for &layer_idx in &layers {
        let Some(Some(snapshot)) = snapshots.get(layer_idx) else {
            return Ok(None);
        };
        // The rotating cache's own invariants, restated against the geometry
        // this boundary's layout promises. `restore_snapshot` re-derives
        // `cached_tokens` from `offset`/`max_size` as `min(offset, max_size)`,
        // which is precisely `cached`, so these must agree here or the restored
        // cache would be rejected later — after 100s of MiB were written. The
        // `offset` check stays exact: only `cached_tokens` saturates.
        if snapshot.offset != boundary_tokens as i32
            || snapshot.max_size != geo.window as i32
            || snapshot.keep != 0
            || snapshot.cached_tokens != cached as i32
        {
            return Ok(None);
        }
        for array in [&snapshot.keys, &snapshot.values] {
            // FAIL CLOSED on dtype: reinterpreting f32 storage as bf16 would
            // be silent corruption, so refuse to capture at all.
            if array.dtype()? != DType::BFloat16 {
                return Ok(None);
            }
            if array.shape()?.as_ref() != expected_shape.as_slice() {
                return Ok(None);
            }
            let raw = array.to_uint16_native()?;
            if raw.len() != elements {
                return Ok(None);
            }
            let mut bytes = Vec::with_capacity(elements * std::mem::size_of::<u16>());
            for value in raw {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            tensors.push(bytes);
        }
    }
    Ok(Some(tensors))
}

/// Rebuild the per-decoder-layer snapshot vector from a validated sidecar
/// payload, ready for `restore_gemma4_sliding_caches`.
///
/// `tensors` must already have passed [`mlx_paged_attn::ColdSidecar`]
/// validation (tensor count and per-tensor byte length pinned by the layout);
/// this re-checks anyway so a contract slip is a graceful `None` rather than a
/// panic or a wrong-shaped array. Alias (KV-shared) and global slots are left
/// `None`, matching what `restore_gemma4_sliding_caches` expects.
pub(crate) fn decode_snapshots(
    config: &Gemma4Config,
    geo: &SlidingSidecarGeometry,
    tensors: &[Vec<u8>],
    boundary_tokens: u32,
) -> Result<Option<Vec<Option<RotatingKVCacheSnapshot>>>> {
    let layers = physical_sliding_layers(config);
    if layers.len() != geo.physical_layers as usize
        || tensors.len() != layers.len() * TENSORS_PER_LAYER as usize
    {
        return Ok(None);
    }
    // Boundary 0 carries no rows and is not a candidate — `ColdSidecar`
    // refuses to store one and `boundary_is_representable` refuses to probe
    // one, so reaching here means a caller invented the boundary. Refuse
    // before building zero-length arrays and handing back an "empty state"
    // that would look like a successful restore of nothing.
    if boundary_tokens == 0 {
        return Ok(None);
    }
    // A payload carries `min(offset, window)` rows — exactly the count
    // `restore_snapshot` re-derives from `offset`/`max_size` — so the byte
    // length below is what separates a sub-window sidecar from a full-window
    // one recorded at the same boundary. A payload whose length disagrees is a
    // miss, never a reshape.
    let cached = payload_tokens(geo, boundary_tokens);
    let Some(elements) = geo.elements_per_tensor(cached) else {
        return Ok(None);
    };
    let shape = [
        1i64,
        geo.kv_heads as i64,
        cached as i64,
        geo.head_dim as i64,
    ];

    let mut snapshots: Vec<Option<RotatingKVCacheSnapshot>> = (0..config.num_hidden_layers.max(0)
        as usize)
        .map(|_| None)
        .collect();
    for (ordinal, &layer_idx) in layers.iter().enumerate() {
        let mut arrays = Vec::with_capacity(TENSORS_PER_LAYER as usize);
        for slot in 0..TENSORS_PER_LAYER as usize {
            let Some(bytes) = tensors.get(ordinal * TENSORS_PER_LAYER as usize + slot) else {
                return Ok(None);
            };
            if bytes.len() != elements * std::mem::size_of::<u16>() {
                return Ok(None);
            }
            let raw: Vec<u16> = bytes
                .chunks_exact(std::mem::size_of::<u16>())
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            arrays.push(MxArray::from_bfloat16(&raw, &shape)?);
        }
        let values = arrays.pop().expect("values tensor");
        let keys = arrays.pop().expect("keys tensor");
        let Some(slot) = snapshots.get_mut(layer_idx) else {
            return Ok(None);
        };
        *slot = Some(RotatingKVCacheSnapshot {
            keys,
            values,
            offset: boundary_tokens as i32,
            max_size: geo.window as i32,
            // `Gemma4LayerCache::new_sliding` always builds
            // `RotatingKVCache::new(window, None)`, so `keep` is 0 and
            // `restore_snapshot` rejects any other value.
            keep: 0,
            cached_tokens: cached as i32,
        });
    }
    Ok(Some(snapshots))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 6 layers: sliding at 0,1,3,4 and global at 2,5, with the last layer
    /// KV-shared — so index 5 is an alias and the PHYSICAL sliding set is
    /// {0,1,3,4}.
    fn config() -> Gemma4Config {
        let mut cfg = Gemma4Config {
            num_hidden_layers: 6,
            sliding_window: 32,
            num_key_value_heads: 2,
            head_dim: 4,
            ..Default::default()
        };
        cfg.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        cfg
    }

    fn geo_of(cfg: &Gemma4Config) -> SlidingSidecarGeometry {
        geometry(cfg).expect("geometry")
    }

    /// Same layer split, but a realistic 1024-token window — the size that
    /// makes typical chat prompts SUB-window, which is the case the sidecar
    /// has to cover.
    fn wide_config() -> Gemma4Config {
        Gemma4Config {
            sliding_window: 1024,
            ..config()
        }
    }

    #[test]
    fn physical_sliding_layers_excludes_global_and_shared_aliases() {
        let mut cfg = config();
        assert_eq!(physical_sliding_layers(&cfg), vec![0, 1, 3, 4]);

        // Two KV-shared trailing layers: index 4 becomes an alias even though
        // `layer_types` still calls it sliding. Counting it would shift every
        // later layer's tensors and restore garbage rather than fail.
        cfg.num_kv_shared_layers = Some(2);
        assert_eq!(physical_sliding_layers(&cfg), vec![0, 1, 3]);
        assert_eq!(geo_of(&cfg).physical_layers, 3);
    }

    #[test]
    fn all_global_config_installs_no_policy() {
        let mut cfg = config();
        cfg.layer_types = vec!["full_attention".to_string(); 6];
        assert!(geometry(&cfg).is_none());
        assert!(policy(&cfg).is_none());
    }

    #[test]
    fn layout_is_boundary_invariant_except_the_boundary() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let a = layout_at(&geo, 32);
        let b = layout_at(&geo, 64);
        assert_eq!(a.boundary_tokens, 32);
        assert_eq!(b.boundary_tokens, 64);
        assert_eq!(a.dims, b.dims);
        assert_eq!(a.bytes_per_tensor, b.bytes_per_tensor);
        assert_eq!(a.num_layers, 4);
        assert_eq!(a.tensors_per_layer, 2);
        assert_eq!(a.dims, vec![1, 2, 32, 4]);
        assert_eq!(a.bytes_per_tensor, 2 * 32 * 4 * 2);

        // And the policy stamps exactly that.
        let policy = policy(&cfg).expect("policy");
        assert_eq!(policy.group(), ColdGroup::SlidingWindow);
        assert_eq!(policy.expected_at(32), a);
        assert_eq!(policy.expected_at(64), b);
    }

    /// Representability is POSITIVITY + BLOCK ALIGNMENT, and nothing else.
    ///
    /// This replaces `boundaries_below_one_window_are_not_representable`, which
    /// asserted the old window floor. That floor was a payload-format fact
    /// leaking into the hit rule: the payload now carries `min(b, window)`
    /// rows, so `b = 16` under a 32-token window is a legal pre-wrap state, not
    /// an unrepresentable one. Zero is still refused — it is the policy
    /// template and would describe an empty payload.
    #[test]
    fn representability_is_positivity_and_block_alignment_only() {
        let geo = geo_of(&config());
        // window = 32, block_size = 16.
        assert!(!boundary_is_representable(&geo, 0, 16));
        assert!(boundary_is_representable(&geo, 16, 16));
        assert!(boundary_is_representable(&geo, 32, 16));
        assert!(boundary_is_representable(&geo, 48, 16));
        // Not block aligned.
        assert!(!boundary_is_representable(&geo, 40, 16));
        // Degenerate block size never yields a boundary.
        assert!(!boundary_is_representable(&geo, 32, 0));
    }

    /// Build a materialized snapshot vector for `boundary`, filling the
    /// physical sliding slots with distinct bf16 patterns so a mis-ordered
    /// round trip is visible rather than plausible.
    ///
    /// The row count is `min(boundary, window)` — exactly what a live
    /// `RotatingKVCache` holds at that offset (`expected_cached_tokens`), so a
    /// sub-window boundary produces a genuine PRE-WRAP snapshot rather than a
    /// padded full-window one.
    fn snapshots_for(
        cfg: &Gemma4Config,
        geo: &SlidingSidecarGeometry,
        boundary: u32,
        dtype_is_bf16: bool,
    ) -> Vec<Option<RotatingKVCacheSnapshot>> {
        let cached = boundary.min(geo.window);
        let elements = (geo.kv_heads as usize) * (cached as usize) * (geo.head_dim as usize);
        let shape = [
            1i64,
            geo.kv_heads as i64,
            cached as i64,
            geo.head_dim as i64,
        ];
        let mut out: Vec<Option<RotatingKVCacheSnapshot>> =
            (0..cfg.num_hidden_layers as usize).map(|_| None).collect();
        for (ordinal, &layer) in physical_sliding_layers(cfg).iter().enumerate() {
            let make = |tag: u16| -> MxArray {
                let raw: Vec<u16> = (0..elements)
                    .map(|i| (i as u16).wrapping_mul(31).wrapping_add(tag))
                    .collect();
                if dtype_is_bf16 {
                    MxArray::from_bfloat16(&raw, &shape).unwrap()
                } else {
                    MxArray::from_float16(&raw, &shape).unwrap()
                }
            };
            out[layer] = Some(RotatingKVCacheSnapshot {
                keys: make(ordinal as u16 * 2),
                values: make(ordinal as u16 * 2 + 1),
                offset: boundary as i32,
                max_size: geo.window as i32,
                keep: 0,
                cached_tokens: cached as i32,
            });
        }
        out
    }

    #[test]
    fn payload_round_trips_layer_major_and_bit_exact() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let boundary = geo.window; // smallest representable boundary
        let snapshots = snapshots_for(&cfg, &geo, boundary, true);

        let tensors = encode_tensors(&cfg, &geo, &snapshots, boundary)
            .unwrap()
            .expect("bf16 snapshots must encode");
        let layout = layout_at(&geo, boundary);
        assert_eq!(tensors.len(), layout.tensor_count().unwrap());
        assert!(tensors.iter().all(|t| t.len() == layout.bytes_per_tensor));

        let decoded = decode_snapshots(&cfg, &geo, &tensors, boundary)
            .unwrap()
            .expect("payload must decode");
        assert_eq!(decoded.len(), snapshots.len());
        for (layer, (before, after)) in snapshots.iter().zip(decoded.iter()).enumerate() {
            match (before, after) {
                (None, None) => {}
                (Some(before), Some(after)) => {
                    assert_eq!(after.offset, boundary as i32);
                    assert_eq!(after.max_size, geo.window as i32);
                    assert_eq!(after.keep, 0);
                    assert_eq!(after.cached_tokens, geo.window as i32);
                    // Bit-exact, not approximately equal: the whole point of
                    // the u16 path is that no f32 round trip happens.
                    assert_eq!(
                        before.keys.to_uint16_native().unwrap(),
                        after.keys.to_uint16_native().unwrap(),
                        "keys diverged on layer {layer}"
                    );
                    assert_eq!(
                        before.values.to_uint16_native().unwrap(),
                        after.values.to_uint16_native().unwrap(),
                        "values diverged on layer {layer}"
                    );
                }
                _ => panic!("layer {layer} slot occupancy changed across the round trip"),
            }
        }
    }

    #[test]
    fn capture_skips_rather_than_mislabels_non_bf16_state() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let boundary = geo.window;
        // f16 has the same byte width as bf16, so a length check alone would
        // let it through and the label would lie about the element type.
        let snapshots = snapshots_for(&cfg, &geo, boundary, false);
        assert!(
            encode_tensors(&cfg, &geo, &snapshots, boundary)
                .unwrap()
                .is_none(),
            "a non-bf16 sliding cache must skip capture, not reinterpret its bytes"
        );
    }

    #[test]
    fn capture_skips_on_offset_mismatch_or_missing_physical_layer() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let boundary = geo.window;

        // Snapshot offsets describing a DIFFERENT boundary than the one the
        // layout would be stamped with.
        let stale = snapshots_for(&cfg, &geo, boundary, true);
        assert!(
            encode_tensors(&cfg, &geo, &stale, boundary * 2)
                .unwrap()
                .is_none()
        );

        // A physical sliding layer with no snapshot at all.
        let mut holed = snapshots_for(&cfg, &geo, boundary, true);
        holed[3] = None;
        assert!(
            encode_tensors(&cfg, &geo, &holed, boundary)
                .unwrap()
                .is_none()
        );
    }

    /// Decode is length-pinned, so a payload that does not match the byte
    /// length THIS boundary implies is a miss rather than a reshape.
    ///
    /// This was `decode_refuses_short_payloads_and_sub_window_boundaries`. The
    /// truncated / clipped cases are unchanged; the last case flipped meaning
    /// but not expectation: a sub-window boundary is now representable, yet a
    /// payload holding a FULL window of rows still cannot be read at one,
    /// because `min(b, window)` rows is fewer bytes than it carries.
    #[test]
    fn decode_refuses_payloads_that_do_not_match_the_boundary_length() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let boundary = geo.window;
        let tensors = encode_tensors(
            &cfg,
            &geo,
            &snapshots_for(&cfg, &geo, boundary, true),
            boundary,
        )
        .unwrap()
        .unwrap();

        // One tensor short of the layout's count.
        let mut truncated = tensors.clone();
        truncated.pop();
        assert!(
            decode_snapshots(&cfg, &geo, &truncated, boundary)
                .unwrap()
                .is_none()
        );

        // One tensor short of the layout's byte length.
        let mut clipped = tensors.clone();
        clipped[0].pop();
        assert!(
            decode_snapshots(&cfg, &geo, &clipped, boundary)
                .unwrap()
                .is_none()
        );

        // A full-window payload read at a HALF-window boundary: that boundary
        // is representable now, but it expects half the bytes.
        assert_eq!(
            layout_at(&geo, boundary - 16).bytes_per_tensor * 2,
            layout_at(&geo, boundary).bytes_per_tensor
        );
        assert!(
            decode_snapshots(&cfg, &geo, &tensors, boundary - 16)
                .unwrap()
                .is_none()
        );

        // Boundary 0 is the policy template, never a candidate.
        assert!(decode_snapshots(&cfg, &geo, &tensors, 0).unwrap().is_none());
    }

    #[test]
    fn fingerprint_component_separates_window_and_layer_split() {
        let cfg = config();
        let base = geo_of(&cfg).fingerprint_component();

        let mut wider = cfg.clone();
        wider.sliding_window = 64;
        assert_ne!(geo_of(&wider).fingerprint_component(), base);

        // Same full-attention pool (2 global layers), different sliding split:
        // the POOL geometry is identical, so only this component separates them.
        let mut resplit = cfg.clone();
        resplit.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "full_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        assert_eq!(
            physical_sliding_layers(&resplit).len(),
            physical_sliding_layers(&cfg).len(),
            "this fixture keeps the layer COUNT equal on purpose"
        );
        // Counts agree, so the component agrees — the differing ORDER is
        // carried by the serialized config blob the fingerprint also folds in.
        assert_eq!(geo_of(&resplit).fingerprint_component(), base);

        let mut heads = cfg.clone();
        heads.num_key_value_heads = 4;
        assert_ne!(geo_of(&heads).fingerprint_component(), base);
    }

    // ---------------------------------------------------------------------
    // Sub-window boundaries. A rotating cache holds `min(offset, window)`
    // tokens, and `RotatingKVCache::restore_snapshot` derives `idx` from
    // exactly that (`expected_cached_tokens`), so a 512-row payload at offset
    // 512 with `max_size` 1024 is a first-class PRE-WRAP state — not a
    // truncation and not a pad. Typical chat prompts are shorter than one
    // window, so refusing them made the whole sidecar inert.
    // ---------------------------------------------------------------------

    #[test]
    fn sub_window_boundaries_are_representable() {
        let geo = geo_of(&wide_config());
        assert_eq!(geo.window, 1024);

        // Half a window, block aligned: a real, restorable pre-wrap state.
        assert!(boundary_is_representable(&geo, 512, 16));
        assert!(boundary_is_representable(&geo, 16, 16));
        assert!(boundary_is_representable(&geo, 1024, 16));
        assert!(boundary_is_representable(&geo, 2048, 16));

        // The rules that DO still apply: positive and block aligned.
        assert!(!boundary_is_representable(&geo, 0, 16));
        assert!(!boundary_is_representable(&geo, 520, 16));
        assert!(!boundary_is_representable(&geo, 512, 0));
    }

    #[test]
    fn layout_token_axis_scales_with_a_sub_window_boundary() {
        let geo = geo_of(&wide_config());
        let half = layout_at(&geo, 512);
        let full = layout_at(&geo, 1024);

        assert_eq!(half.dims[2], 512, "the token axis must follow the boundary");
        assert_eq!(full.dims[2], 1024);
        assert_eq!(
            half.bytes_per_tensor * 2,
            full.bytes_per_tensor,
            "half the tokens is exactly half the payload"
        );

        // Only the token axis moves; everything else is still geometry.
        assert_eq!(half.group, full.group);
        assert_eq!(half.num_layers, full.num_layers);
        assert_eq!(half.tensors_per_layer, full.tensors_per_layer);
        assert_eq!(half.dtype, full.dtype);
        assert_eq!(
            [half.dims[0], half.dims[1], half.dims[3]],
            [full.dims[0], full.dims[1], full.dims[3]]
        );

        // At and above the window the rule is the IDENTITY, which is what
        // keeps already-written sidecars loadable without a format bump.
        assert_eq!(layout_at(&geo, 4096).dims, full.dims);
        assert_eq!(
            layout_at(&geo, 4096).bytes_per_tensor,
            full.bytes_per_tensor
        );
    }

    #[test]
    fn payload_round_trips_at_a_sub_window_boundary() {
        let cfg = wide_config();
        let geo = geo_of(&cfg);
        let boundary = geo.window / 2; // 512: a pre-wrap cache, half full
        let snapshots = snapshots_for(&cfg, &geo, boundary, true);

        let tensors = encode_tensors(&cfg, &geo, &snapshots, boundary)
            .unwrap()
            .expect("a sub-window pre-wrap state must encode");
        let layout = layout_at(&geo, boundary);
        assert_eq!(tensors.len(), layout.tensor_count().unwrap());
        assert!(tensors.iter().all(|t| t.len() == layout.bytes_per_tensor));

        let decoded = decode_snapshots(&cfg, &geo, &tensors, boundary)
            .unwrap()
            .expect("a sub-window payload must decode");
        assert_eq!(decoded.len(), snapshots.len());
        let mut restored = 0usize;
        for (layer, (before, after)) in snapshots.iter().zip(decoded.iter()).enumerate() {
            match (before, after) {
                (None, None) => {}
                (Some(before), Some(after)) => {
                    restored += 1;
                    // The trio `restore_snapshot` cross-checks:
                    // `cached_tokens == min(offset, max_size)`.
                    assert_eq!(after.cached_tokens, 512);
                    assert_eq!(after.offset, 512);
                    assert_eq!(after.max_size, 1024);
                    assert_eq!(after.keep, 0);
                    assert_eq!(after.keys.shape().unwrap().as_ref(), &[1, 2, 512, 4]);
                    assert_eq!(
                        before.keys.to_uint16_native().unwrap(),
                        after.keys.to_uint16_native().unwrap(),
                        "keys diverged on layer {layer}"
                    );
                    assert_eq!(
                        before.values.to_uint16_native().unwrap(),
                        after.values.to_uint16_native().unwrap(),
                        "values diverged on layer {layer}"
                    );
                }
                _ => panic!("layer {layer} slot occupancy changed across the round trip"),
            }
        }
        assert_eq!(restored, geo.physical_layers as usize);
    }

    /// The bytes the PRE-CHANGE encoder produced, recomputed from the
    /// snapshots by the old rule: layer-major, `keys` then `values`, every
    /// tensor a full `kv_heads * window * head_dim` run of little-endian
    /// `to_uint16_native` words.
    fn legacy_full_window_bytes(
        cfg: &Gemma4Config,
        geo: &SlidingSidecarGeometry,
        snapshots: &[Option<RotatingKVCacheSnapshot>],
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for &layer in physical_sliding_layers(cfg).iter() {
            let snapshot = snapshots[layer].as_ref().expect("physical layer snapshot");
            for array in [&snapshot.keys, &snapshot.values] {
                let raw = array.to_uint16_native().unwrap();
                assert_eq!(
                    raw.len(),
                    (geo.kv_heads as usize) * (geo.window as usize) * (geo.head_dim as usize),
                    "this fixture must be a FULL-window payload"
                );
                out.push(raw.iter().flat_map(|v| v.to_le_bytes()).collect());
            }
        }
        out
    }

    /// NO SILENT FORMAT DRIFT for objects already on disk.
    ///
    /// Boundary-scaling is `min(b, window)`, which at and above the window is
    /// the IDENTITY. So at `b == window` and `b == 2 * window` the encoder must
    /// emit byte-for-byte what it emitted before the token axis could move, and
    /// the layout must still be the frozen full-window one — that is what lets
    /// a gemma4 sidecar written by an older process still compare equal under
    /// `load_sidecar` with no fingerprint or on-disk format bump.
    #[test]
    fn encoding_at_and_above_the_window_is_byte_identical_to_the_legacy_format() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let full_bytes = (geo.kv_heads as usize)
            * (geo.window as usize)
            * (geo.head_dim as usize)
            * std::mem::size_of::<u16>();

        for boundary in [geo.window, geo.window * 2] {
            let snapshots = snapshots_for(&cfg, &geo, boundary, true);
            // `min(b, window) == window` here, so the fixture is full-window
            // and the legacy expectation is well defined.
            let expected = legacy_full_window_bytes(&cfg, &geo, &snapshots);

            let tensors = encode_tensors(&cfg, &geo, &snapshots, boundary)
                .unwrap()
                .expect("a full-window state must still encode");
            assert_eq!(
                tensors, expected,
                "encoded bytes drifted from the pre-change format at boundary {boundary}"
            );

            // And the layout the bytes are labelled with is unchanged too:
            // same dims, same byte length, only `boundary_tokens` varies.
            let layout = layout_at(&geo, boundary);
            assert_eq!(layout.boundary_tokens, boundary);
            assert_eq!(layout.dims, vec![1, geo.kv_heads, geo.window, geo.head_dim]);
            assert_eq!(layout.bytes_per_tensor, full_bytes);
            assert_eq!(layout.num_layers, geo.physical_layers);
            assert_eq!(layout.tensors_per_layer, TENSORS_PER_LAYER);
            assert_eq!(layout.dtype, SIDECAR_DTYPE);
            assert!(tensors.iter().all(|tensor| tensor.len() == full_bytes));

            // Round trip is still bit-exact through the unchanged bytes.
            let decoded = decode_snapshots(&cfg, &geo, &tensors, boundary)
                .unwrap()
                .expect("the legacy payload must decode");
            for (before, after) in snapshots.iter().zip(decoded.iter()) {
                if let (Some(before), Some(after)) = (before, after) {
                    assert_eq!(after.cached_tokens, geo.window as i32);
                    assert_eq!(
                        before.keys.to_uint16_native().unwrap(),
                        after.keys.to_uint16_native().unwrap()
                    );
                }
            }
        }
    }

    /// The read side and the write side must derive the SAME layout, because
    /// `load_sidecar` compares them for equality: `expected_at` (the policy's
    /// scaled axis) against `layout_at` (this module's own rule). A drift here
    /// would make capture write payloads no restore could ever match.
    #[test]
    fn policy_expected_at_matches_layout_at_on_every_candidate_boundary() {
        let cfg = wide_config();
        let geo = geo_of(&cfg);
        let policy = policy(&cfg).expect("policy");
        assert_eq!(policy.group(), ColdGroup::SlidingWindow);

        // 0 is included on purpose. It is not a candidate, but the two sides
        // used to mean OPPOSITE things by it — `payload_tokens` handed back a
        // full window while `expected_at` scaled to nothing — and only the
        // absence of a caller kept that from mattering.
        for boundary in [0u32, 16, 256, 512, 1008, 1024, 1040, 2048, 65536] {
            assert_eq!(
                policy.expected_at(boundary),
                layout_at(&geo, boundary),
                "capture and restore disagree at boundary {boundary}"
            );
        }
    }

    /// The policy template is the one layout that is deliberately NOT
    /// `layout_at` of its own label: full-window rows carrying the label 0.
    ///
    /// `new_boundary_scaled` clamps every scaled extent against the template's
    /// own, so building it from `layout_at(&geo, 0)` — zero rows — would peg
    /// the token axis at zero and scale every real boundary to an empty
    /// payload, turning the whole sidecar inert with no error anywhere.
    #[test]
    fn the_policy_template_carries_a_full_window_under_a_zero_label() {
        let cfg = wide_config();
        let geo = geo_of(&cfg);

        let template = template_layout(&geo);
        assert_eq!(template.boundary_tokens, 0);
        assert_eq!(template.dims, geo.dims(geo.window));
        assert_eq!(
            template.bytes_per_tensor,
            geo.bytes_per_tensor(geo.window).expect("window bytes")
        );

        assert_eq!(payload_tokens(&geo, 0), 0, "0 rows at the zero boundary");
        assert!(
            template.bytes_per_tensor > layout_at(&geo, 0).bytes_per_tensor,
            "the template must not be derivable by labelling it 0"
        );

        // The clamp the template feeds: a real boundary still scales to itself.
        let policy = policy(&cfg).expect("policy");
        assert_eq!(policy.expected_at(512).dims[2], 512);
    }
}
