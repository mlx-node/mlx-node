//! Qwen3.5 (dense + MoE) GDN recurrent state as a cold-tier [`ColdSidecar`].
//!
//! # Why qwen3_5 needs one at all
//!
//! Qwen3.5 is a HYBRID: `Qwen35Inner` sizes its paged pool over the
//! FULL-ATTENTION layers only (`full_attention_layer_count`, the `num_layers`
//! it feeds into `PagedAttentionConfig`). The interleaved GDN (gated delta-net)
//! layers keep their recurrent state in a per-layer
//! [`Qwen3_5LayerCache::Linear`] (`ArraysCache` with two slots: conv state and
//! recurrent state) that the pool never sees. A restored paged K/V prefix
//! therefore reconstructs only part of the model state; the missing half is
//! today RECOMPUTED by `run_gdn_only_prefill_materialized`, an O(prefix) replay
//! over the reused prefix. This sidecar persists it instead.
//!
//! # Why a recurrent state is valid ONLY at its exact prefix length
//!
//! A GDN recurrent state is a running summary of every preceding token (vLLM's
//! `MambaSpec`), so — unlike a sliding window — it is meaningful ONLY at the
//! exact block-aligned prefix length it was produced at. There is no windowed
//! truncation and no partial-block fixup: restore reconciles DOWN to a boundary
//! a validated sidecar actually backs, or to zero, and recomputes. That is the
//! fail-closed direction (vLLM `MambaManager.find_longest_cache_hit`:
//! full-block-aligned reuse only).
//!
//! # The payload — one concatenated blob per GDN layer
//!
//! [`ColdSidecarLayout`] carries a SINGLE `bytes_per_tensor` and
//! [`mlx_paged_attn::ColdSidecar::validate`] rejects any sidecar whose tensors
//! are not all that length. The two GDN tensors per layer have DIFFERENT sizes
//! (`conv_state` is `(K-1)*conv_dim`; `recurrent_state` is `Hv*Dv*Dk`, ~25x
//! larger on the 27B). Padding conv up to rec size would nearly double the
//! sidecar, and two `tensors_per_layer` is impossible. So each GDN layer stores
//! ONE blob = `conv_state` bytes followed by `recurrent_state` bytes, with
//! `tensors_per_layer = 1`. The layout's `dims` carries both shapes
//! (`[K-1, conv_dim, Hv, Dv, Dk]`) so the decoder knows the exact split offset
//! and a geometry drift is a layout mismatch (miss), not a reinterpretation.
//!
//! # Why the chain key, not a token-local key
//!
//! A GDN layer's recurrent state at position `i` is a function of the hidden
//! stream that has already passed through the interleaved FULL-attention layers
//! (which attend to the whole prefix), so it is NOT a function of any bounded
//! suffix. The parent-linked chain from block 0 that
//! `ColdTierWalk::deepest_backed_boundary` builds — same per-block `extra_keys`
//! and `cache_salt` as the K/V chain, under the `GdnState` domain tag — is the
//! right identity.
//!
//! # Bit-exactness
//!
//! No f32 round trip: 16-bit state goes out via `to_uint16_native` and back via
//! `from_bfloat16` / `from_float16`; f32 state via `to_float32` / `from_float32`.
//! The one element type this sidecar is written and read in is fixed at load
//! from the paged pool's cache dtype (`build_cold_tier_context`), which equals
//! the GDN activation dtype on every checkpoint; capture re-checks each array's
//! actual dtype and SKIPS (fail closed) on any mismatch rather than mislabel.

use mlx_paged_attn::{ColdGroup, ColdSidecarLayout, ColdSidecarPolicy};
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

use super::config::Qwen3_5Config;
use super::layer_cache::Qwen3_5LayerCache;

/// One blob per GDN layer (conv ++ recurrent), so exactly one tensor per layer.
pub(crate) const TENSORS_PER_LAYER: u32 = 1;

/// Decoder indices of the GDN (linear-attention) layers, ascending — the exact
/// order the payload's `tensors[gdn_ordinal]` is indexed by. Must be identical
/// on the capture and install sides or an install silently misaligns.
pub(crate) fn gdn_layers(config: &Qwen3_5Config) -> Vec<usize> {
    (0..config.num_layers.max(0) as usize)
        .filter(|&i| config.is_linear_layer(i))
        .collect()
}

/// Map a dtype label (`"BFloat16"` / `"Float16"` / `"Float32"`) to the concrete
/// [`DType`]; `None` for anything the sidecar codec cannot round-trip
/// bit-exactly.
fn dtype_from_label(label: &str) -> Option<DType> {
    match label {
        "BFloat16" => Some(DType::BFloat16),
        "Float16" => Some(DType::Float16),
        "Float32" => Some(DType::Float32),
        _ => None,
    }
}

fn element_size(dtype: DType) -> usize {
    match dtype {
        DType::Float32 => std::mem::size_of::<f32>(),
        _ => std::mem::size_of::<u16>(),
    }
}

/// Geometry of qwen3_5's out-of-pool GDN state, derived once from the config +
/// the pool's cache dtype. Every field is boundary-invariant, which is what
/// lets it back a [`ColdSidecarPolicy`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GdnSidecarGeometry {
    /// Count of GDN (linear-attention) layers = `gdn_layers(config).len()`.
    pub gdn_layers: u32,
    /// `linear_conv_kernel_dim - 1` — the conv-state row count (`[1, K-1, conv_dim]`).
    pub conv_rows: u32,
    /// `linear_conv_dim()` — the conv-state channel count.
    pub conv_dim: u32,
    /// `linear_num_value_heads` — recurrent state `Hv` axis.
    pub num_v_heads: u32,
    /// `linear_value_head_dim` — recurrent state `Dv` axis.
    pub v_head_dim: u32,
    /// `linear_key_head_dim` — recurrent state `Dk` axis.
    pub k_head_dim: u32,
    /// Element dtype label, fixed at load from the pool cache dtype.
    pub dtype: String,
}

impl GdnSidecarGeometry {
    fn dtype(&self) -> Option<DType> {
        dtype_from_label(&self.dtype)
    }

    /// `conv_state` element count = `(K-1) * conv_dim`.
    fn conv_elements(&self) -> Option<usize> {
        (self.conv_rows as usize).checked_mul(self.conv_dim as usize)
    }

    /// `recurrent_state` element count = `Hv * Dv * Dk`.
    fn recurrent_elements(&self) -> Option<usize> {
        (self.num_v_heads as usize)
            .checked_mul(self.v_head_dim as usize)?
            .checked_mul(self.k_head_dim as usize)
    }

    /// Byte length of the single per-layer blob = conv bytes ++ recurrent bytes.
    fn bytes_per_tensor(&self) -> Option<usize> {
        let esz = element_size(self.dtype()?);
        let elements = self
            .conv_elements()?
            .checked_add(self.recurrent_elements()?)?;
        elements.checked_mul(esz)
    }

    /// `[K-1, conv_dim, Hv, Dv, Dk]` — five entries (batch dropped; always 1),
    /// well within `MAX_SIDECAR_DIMS = 8`. Gives the decoder the split offset
    /// and both concrete shapes.
    fn dims(&self) -> Vec<u32> {
        vec![
            self.conv_rows,
            self.conv_dim,
            self.num_v_heads,
            self.v_head_dim,
            self.k_head_dim,
        ]
    }

    fn conv_shape(&self) -> [i64; 3] {
        [1, self.conv_rows as i64, self.conv_dim as i64]
    }

    fn recurrent_shape(&self) -> [i64; 4] {
        [
            1,
            self.num_v_heads as i64,
            self.v_head_dim as i64,
            self.k_head_dim as i64,
        ]
    }

    /// Geometry bytes folded into the cold-tier fingerprint.
    ///
    /// [`crate::cold_tier::ColdTierGeometry`] describes the POOL, which for
    /// qwen3_5 covers only the full-attention layers, so two configs with the
    /// same full-attention pool but a different GDN geometry would produce the
    /// same pool geometry. The serialized `Qwen3_5Config` already carries every
    /// `linear_*` field, but binding them here makes the sidecar's identity
    /// independent of that blob's serde surface ever changing.
    pub fn fingerprint_component(&self) -> Vec<u8> {
        format!(
            "qwen3_5-gdn-sidecar:v1:layers={}:conv_rows={}:conv_dim={}:v_heads={}:v_dim={}:k_dim={}:dtype={}:tensors={}",
            self.gdn_layers,
            self.conv_rows,
            self.conv_dim,
            self.num_v_heads,
            self.v_head_dim,
            self.k_head_dim,
            self.dtype,
            TENSORS_PER_LAYER,
        )
        .into_bytes()
    }
}

/// Derive the sidecar geometry, or `None` when this checkpoint has no
/// out-of-pool GDN state to persist (no linear layers, a conv kernel of 1, or a
/// nonsensical head geometry) or a dtype the codec cannot round-trip. `None`
/// means "install no policy": the family then behaves exactly as it did before
/// sidecars existed.
pub(crate) fn geometry(config: &Qwen3_5Config, cache_dtype: &str) -> Option<GdnSidecarGeometry> {
    let gdn_layers = u32::try_from(gdn_layers(config).len()).ok()?;
    if gdn_layers == 0 {
        return None;
    }
    // Codec must be able to round-trip this dtype bit-exactly.
    dtype_from_label(cache_dtype)?;
    // `[1, K-1, conv_dim]`: a kernel of 1 leaves K-1 = 0, which is not a
    // representable (positive) dim.
    let conv_rows = u32::try_from(config.linear_conv_kernel_dim - 1).ok()?;
    let conv_dim = u32::try_from(config.linear_conv_dim()).ok()?;
    let num_v_heads = u32::try_from(config.linear_num_value_heads).ok()?;
    let v_head_dim = u32::try_from(config.linear_value_head_dim).ok()?;
    let k_head_dim = u32::try_from(config.linear_key_head_dim).ok()?;
    if conv_rows == 0 || conv_dim == 0 || num_v_heads == 0 || v_head_dim == 0 || k_head_dim == 0 {
        return None;
    }
    let geo = GdnSidecarGeometry {
        gdn_layers,
        conv_rows,
        conv_dim,
        num_v_heads,
        v_head_dim,
        k_head_dim,
        dtype: cache_dtype.to_string(),
    };
    // A geometry whose payload cannot be sized is not persistable at all.
    geo.bytes_per_tensor()?;
    Some(geo)
}

/// The layout a sidecar anchored at `boundary_tokens` must have. `boundary` is
/// stamped by [`ColdSidecarPolicy::expected_at`] on the read side; the capture
/// side builds the same value here so the two compare equal.
pub(crate) fn layout_at(geo: &GdnSidecarGeometry, boundary_tokens: u32) -> ColdSidecarLayout {
    ColdSidecarLayout {
        group: ColdGroup::GdnState,
        boundary_tokens,
        num_layers: geo.gdn_layers,
        tensors_per_layer: TENSORS_PER_LAYER,
        dtype: geo.dtype.clone(),
        dims: geo.dims(),
        // `bytes_per_tensor` is checked non-`None` by `geometry`.
        bytes_per_tensor: geo.bytes_per_tensor().unwrap_or(0),
    }
}

/// Build the reconcile-down policy for this config + pool dtype, or `None` when
/// there is no GDN state to back (see [`geometry`]).
pub(crate) fn policy(config: &Qwen3_5Config, cache_dtype: &str) -> Option<ColdSidecarPolicy> {
    let geo = geometry(config, cache_dtype)?;
    ColdSidecarPolicy::new(layout_at(&geo, 0)).ok()
}

/// Serialize one GDN array (`conv_state` or `recurrent_state`) into `blob`.
///
/// Returns `Ok(false)` — a graceful skip, never an error — when the array does
/// not match the sidecar's fixed geometry (wrong dtype, wrong shape, wrong
/// element count). Skipping is the fail-closed direction: writing bytes that do
/// not match the label would restore wrong state instead of missing.
fn encode_array(
    array: &MxArray,
    expected_dtype: DType,
    expected_shape: &[i64],
    expected_elements: usize,
    blob: &mut Vec<u8>,
) -> Result<bool> {
    // FAIL CLOSED on dtype: reinterpreting f32 storage as bf16 (or vice versa)
    // would be silent corruption, so refuse to capture at all.
    if array.dtype()? != expected_dtype {
        return Ok(false);
    }
    if array.shape()?.as_ref() != expected_shape {
        return Ok(false);
    }
    match expected_dtype {
        DType::BFloat16 | DType::Float16 => {
            let raw = array.to_uint16_native()?;
            if raw.len() != expected_elements {
                return Ok(false);
            }
            for value in raw {
                blob.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::Float32 => {
            let raw = array.to_float32()?.to_vec();
            if raw.len() != expected_elements {
                return Ok(false);
            }
            for value in raw {
                blob.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Fires at most once per process: a GDN cold sidecar that can never be written
/// because the state's dtype does not match the geometry the pool fixed.
static DTYPE_MISMATCH_WARNED: std::sync::Once = std::sync::Once::new();

/// Report a permanently-unwritable sidecar exactly once.
///
/// [`encode_array`] fails closed on a dtype mismatch, which is right — it must
/// never reinterpret f32 storage as bf16. But the resulting `Ok(None)` is
/// indistinguishable from the benign structural skips around it, so a
/// misconfigured tier degrades into silence. A `dtype()` probe that itself
/// errors is left to the encode path to surface.
fn warn_once_on_dtype_mismatch(conv: &MxArray, rec: &MxArray, expected: DType) {
    let (Ok(conv_dtype), Ok(rec_dtype)) = (conv.dtype(), rec.dtype()) else {
        return;
    };
    if conv_dtype == expected && rec_dtype == expected {
        return;
    }
    DTYPE_MISMATCH_WARNED.call_once(|| {
        tracing::warn!(
            expected = ?expected,
            conv_state = ?conv_dtype,
            recurrent_state = ?rec_dtype,
            "qwen3_5 GDN cold sidecar can never be captured: the paged pool's cache dtype fixes \
             the sidecar element type, but the GDN state carries a different dtype. The cold tier \
             will keep persisting K/V blocks it can never restore from (hits stay at 0). Align the \
             paged pool cache dtype with the GDN activation dtype, or disable persistPagedCache."
        );
    });
}

/// Serialize the materialized GDN caches for one boundary into the layer-major
/// single-blob-per-layer payload.
///
/// `caches` is the full-length (`num_layers`) layer-cache vector; only the GDN
/// (linear) layers are read, in `gdn_layers` order. Returns `Ok(None)` — a
/// graceful skip — when the state does not match the sidecar's fixed geometry:
/// a missing slot, a wrong dtype, or a shape that is not
/// `[1, K-1, conv_dim]` / `[1, Hv, Dv, Dk]`.
///
/// The caller must have materialized the caches first
/// (`materialize_linear_layer_caches` = `eval_arrays`).
pub(crate) fn encode_tensors(
    config: &Qwen3_5Config,
    geo: &GdnSidecarGeometry,
    caches: &[Qwen3_5LayerCache],
    _boundary_tokens: u32,
) -> Result<Option<Vec<Vec<u8>>>> {
    let layers = gdn_layers(config);
    if layers.len() != geo.gdn_layers as usize || caches.len() != config.num_layers.max(0) as usize
    {
        return Ok(None);
    }
    let Some(dtype) = geo.dtype() else {
        return Ok(None);
    };
    let (Some(conv_elements), Some(rec_elements), Some(bytes_per_tensor)) = (
        geo.conv_elements(),
        geo.recurrent_elements(),
        geo.bytes_per_tensor(),
    ) else {
        return Ok(None);
    };
    let conv_shape = geo.conv_shape();
    let rec_shape = geo.recurrent_shape();

    let mut tensors = Vec::with_capacity(layers.len());
    for &layer_idx in &layers {
        let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get(layer_idx) else {
            return Ok(None);
        };
        let (Some(conv), Some(rec)) = (arrays.get(0), arrays.get(1)) else {
            return Ok(None);
        };
        // A dtype mismatch here is not a transient skip, it is a permanent
        // misconfiguration: the sidecar geometry fixes its element type from
        // the POOL's cache dtype at load, while these arrays carry the GDN
        // ACTIVATION dtype. If those disagree the skip below repeats on every
        // turn forever, so the tier keeps writing K/V blocks it can never
        // restore from — quota and I/O burned, hits pinned at zero, outwardly
        // indistinguishable from a cold cache. Say it once, loudly.
        warn_once_on_dtype_mismatch(conv, rec, dtype);
        let mut blob = Vec::with_capacity(bytes_per_tensor);
        if !encode_array(conv, dtype, &conv_shape, conv_elements, &mut blob)? {
            return Ok(None);
        }
        if !encode_array(rec, dtype, &rec_shape, rec_elements, &mut blob)? {
            return Ok(None);
        }
        if blob.len() != bytes_per_tensor {
            return Ok(None);
        }
        tensors.push(blob);
    }
    Ok(Some(tensors))
}

/// Rebuild one GDN array from a byte slice of the layer blob.
fn decode_array(
    bytes: &[u8],
    dtype: DType,
    shape: &[i64],
    elements: usize,
) -> Result<Option<MxArray>> {
    let esz = element_size(dtype);
    if bytes.len() != elements.saturating_mul(esz) {
        return Ok(None);
    }
    let array = match dtype {
        DType::BFloat16 | DType::Float16 => {
            let raw: Vec<u16> = bytes
                .chunks_exact(std::mem::size_of::<u16>())
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            if raw.len() != elements {
                return Ok(None);
            }
            if dtype == DType::BFloat16 {
                MxArray::from_bfloat16(&raw, shape)?
            } else {
                MxArray::from_float16(&raw, shape)?
            }
        }
        DType::Float32 => {
            let raw: Vec<f32> = bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                .collect();
            if raw.len() != elements {
                return Ok(None);
            }
            MxArray::from_float32(&raw, shape)?
        }
        _ => return Ok(None),
    };
    Ok(Some(array))
}

/// Rebuild the full-length layer-cache vector from a validated sidecar payload,
/// ready to assign to `Qwen35Inner::caches`.
///
/// `tensors` must already have passed [`mlx_paged_attn::ColdSidecar`] validation
/// (tensor count and per-tensor byte length pinned by the layout); this
/// re-checks anyway so a contract slip is a graceful `None` rather than a panic
/// or a wrong-shaped array. Full-attention layers become fresh
/// [`Qwen3_5LayerCache::FullAttention`] shells (their K/V lives in the paged
/// pool), exactly like `fresh_dense_layer_caches` produces.
pub(crate) fn decode_caches(
    config: &Qwen3_5Config,
    geo: &GdnSidecarGeometry,
    tensors: &[Vec<u8>],
    _boundary_tokens: u32,
) -> Result<Option<Vec<Qwen3_5LayerCache>>> {
    let layers = gdn_layers(config);
    if layers.len() != geo.gdn_layers as usize || tensors.len() != layers.len() {
        return Ok(None);
    }
    let Some(dtype) = geo.dtype() else {
        return Ok(None);
    };
    let (Some(conv_elements), Some(rec_elements), Some(bytes_per_tensor)) = (
        geo.conv_elements(),
        geo.recurrent_elements(),
        geo.bytes_per_tensor(),
    ) else {
        return Ok(None);
    };
    let conv_bytes = conv_elements * element_size(dtype);
    let conv_shape = geo.conv_shape();
    let rec_shape = geo.recurrent_shape();

    let mut caches: Vec<Qwen3_5LayerCache> = (0..config.num_layers.max(0) as usize)
        .map(|i| {
            if config.is_linear_layer(i) {
                Qwen3_5LayerCache::new_linear()
            } else {
                Qwen3_5LayerCache::new_full_attention()
            }
        })
        .collect();

    for (ordinal, &layer_idx) in layers.iter().enumerate() {
        let Some(blob) = tensors.get(ordinal) else {
            return Ok(None);
        };
        if blob.len() != bytes_per_tensor {
            return Ok(None);
        }
        let (conv_slice, rec_slice) = blob.split_at(conv_bytes);
        let Some(conv) = decode_array(conv_slice, dtype, &conv_shape, conv_elements)? else {
            return Ok(None);
        };
        let Some(rec) = decode_array(rec_slice, dtype, &rec_shape, rec_elements)? else {
            return Ok(None);
        };
        let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get_mut(layer_idx) else {
            return Ok(None);
        };
        arrays.set(0, conv);
        arrays.set(1, rec);
    }
    Ok(Some(caches))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8 layers, `full_attention_interval = 4`: linear at 0,1,2,4,5,6 and full
    /// attention at 3,7. Small GDN dims so a mis-ordered round trip is visible.
    fn config() -> Qwen3_5Config {
        Qwen3_5Config {
            vocab_size: 32,
            hidden_size: 16,
            num_layers: 8,
            num_heads: 2,
            num_kv_heads: 1,
            intermediate_size: 32,
            rms_norm_eps: 1e-6,
            head_dim: 8,
            tie_word_embeddings: false,
            attention_bias: false,
            max_position_embeddings: 256,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            linear_num_value_heads: 2,
            linear_num_key_heads: 1,
            linear_key_head_dim: 3,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            persist_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    fn geo_of(cfg: &Qwen3_5Config) -> GdnSidecarGeometry {
        geometry(cfg, "BFloat16").expect("geometry")
    }

    #[test]
    fn gdn_layers_excludes_full_attention() {
        let cfg = config();
        assert_eq!(gdn_layers(&cfg), vec![0, 1, 2, 4, 5, 6]);
        assert_eq!(geo_of(&cfg).gdn_layers, 6);
    }

    #[test]
    fn all_linear_or_no_layers_installs_no_policy() {
        // conv kernel of 1 leaves K-1 = 0: not representable.
        let mut cfg = config();
        cfg.linear_conv_kernel_dim = 1;
        assert!(geometry(&cfg, "BFloat16").is_none());
        // Unsupported dtype: codec cannot round-trip.
        assert!(geometry(&config(), "Int8").is_none());
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
        assert_eq!(a.num_layers, 6);
        assert_eq!(a.tensors_per_layer, 1);
        // conv_dim = key_dim*2 + value_dim = (1*3)*2 + (2*4) = 14; K-1 = 3.
        assert_eq!(a.dims, vec![3, 14, 2, 4, 3]);
        // (3*14 + 2*4*3) * 2 bytes = (42 + 24) * 2 = 132.
        assert_eq!(a.bytes_per_tensor, (3 * 14 + 2 * 4 * 3) * 2);

        let policy = policy(&cfg, "BFloat16").expect("policy");
        assert_eq!(policy.group(), ColdGroup::GdnState);
        assert_eq!(policy.expected_at(32), a);
        assert_eq!(policy.expected_at(64), b);
    }

    /// Build a full-length cache vec, filling each GDN layer's conv/recurrent
    /// slots with distinct bf16 patterns so a mis-ordered round trip is visible.
    fn caches_for(
        cfg: &Qwen3_5Config,
        geo: &GdnSidecarGeometry,
        bf16: bool,
    ) -> Vec<Qwen3_5LayerCache> {
        let conv_elems = geo.conv_elements().unwrap();
        let rec_elems = geo.recurrent_elements().unwrap();
        let conv_shape = geo.conv_shape();
        let rec_shape = geo.recurrent_shape();
        let mut caches: Vec<Qwen3_5LayerCache> = (0..cfg.num_layers as usize)
            .map(|i| {
                if cfg.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect();
        for (ordinal, &layer) in gdn_layers(cfg).iter().enumerate() {
            let make = |n: usize, tag: u16, shape: &[i64]| -> MxArray {
                let raw: Vec<u16> = (0..n)
                    .map(|i| (i as u16).wrapping_mul(31).wrapping_add(tag))
                    .collect();
                if bf16 {
                    MxArray::from_bfloat16(&raw, shape).unwrap()
                } else {
                    MxArray::from_float16(&raw, shape).unwrap()
                }
            };
            if let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get_mut(layer) {
                arrays.set(0, make(conv_elems, ordinal as u16 * 2, &conv_shape));
                arrays.set(1, make(rec_elems, ordinal as u16 * 2 + 1, &rec_shape));
            }
        }
        caches
    }

    #[test]
    fn payload_round_trips_layer_major_and_bit_exact() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let caches = caches_for(&cfg, &geo, true);

        let tensors = encode_tensors(&cfg, &geo, &caches, 32)
            .unwrap()
            .expect("bf16 caches must encode");
        let layout = layout_at(&geo, 32);
        assert_eq!(tensors.len(), layout.tensor_count().unwrap());
        assert!(tensors.iter().all(|t| t.len() == layout.bytes_per_tensor));

        let decoded = decode_caches(&cfg, &geo, &tensors, 32)
            .unwrap()
            .expect("payload must decode");
        assert_eq!(decoded.len(), caches.len());
        for (layer, (before, after)) in caches.iter().zip(decoded.iter()).enumerate() {
            match (before, after) {
                (Qwen3_5LayerCache::FullAttention(_), Qwen3_5LayerCache::FullAttention(_)) => {}
                (Qwen3_5LayerCache::Linear(b), Qwen3_5LayerCache::Linear(a)) => {
                    for slot in 0..2 {
                        assert_eq!(
                            b.get(slot).unwrap().to_uint16_native().unwrap(),
                            a.get(slot).unwrap().to_uint16_native().unwrap(),
                            "layer {layer} slot {slot} diverged",
                        );
                    }
                }
                _ => panic!("layer {layer} variant changed across the round trip"),
            }
        }
    }

    #[test]
    fn capture_skips_rather_than_mislabels_wrong_dtype() {
        let cfg = config();
        let geo = geo_of(&cfg);
        // f16 has the same byte width as bf16, so a length check alone would let
        // it through and the label would lie about the element type.
        let caches = caches_for(&cfg, &geo, false);
        assert!(
            encode_tensors(&cfg, &geo, &caches, 32).unwrap().is_none(),
            "a non-bf16 GDN cache must skip capture, not reinterpret its bytes"
        );
    }

    #[test]
    fn capture_skips_on_missing_slot() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let mut caches = caches_for(&cfg, &geo, true);
        if let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get_mut(2) {
            arrays.reset();
        }
        assert!(encode_tensors(&cfg, &geo, &caches, 32).unwrap().is_none());
    }

    #[test]
    fn decode_refuses_short_payloads() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let tensors = encode_tensors(&cfg, &geo, &caches_for(&cfg, &geo, true), 32)
            .unwrap()
            .unwrap();

        // One blob short of the layer count.
        let mut truncated = tensors.clone();
        truncated.pop();
        assert!(decode_caches(&cfg, &geo, &truncated, 32).unwrap().is_none());

        // One byte short of the blob length.
        let mut clipped = tensors.clone();
        clipped[0].pop();
        assert!(decode_caches(&cfg, &geo, &clipped, 32).unwrap().is_none());
    }

    #[test]
    fn fingerprint_component_separates_geometry_and_dtype() {
        let cfg = config();
        let base = geo_of(&cfg).fingerprint_component();

        let mut wider = cfg.clone();
        wider.linear_value_head_dim = 8;
        assert_ne!(geo_of(&wider).fingerprint_component(), base);

        assert_ne!(
            geometry(&cfg, "Float32").unwrap().fingerprint_component(),
            base
        );
    }

    /// GDN geometry of `Qwen3.6-27B` verbatim from its `config.json`
    /// (`~/.mlx-node/models/qwen3.6-27b`). Only the fields the sidecar sizes
    /// itself from are real; the rest are placeholders.
    fn qwen3_6_27b_config() -> Qwen3_5Config {
        Qwen3_5Config {
            num_layers: 64,
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 48,
            linear_value_head_dim: 128,
            ..config()
        }
    }

    /// Bytes one resident GDN checkpoint costs, sized by the same code that
    /// sizes the on-disk sidecar. `GDN_PREFIX_CHECKPOINT_LIMIT` and the
    /// checkpoint ladder are both bounded against this number, so it is pinned
    /// here rather than quoted from a comment.
    #[test]
    fn one_gdn_checkpoint_of_the_27b_costs_a_known_number_of_bytes() {
        let geo = geometry(&qwen3_6_27b_config(), "BFloat16").expect("geometry");
        assert_eq!(geo.gdn_layers, 48);
        // conv `[1, 3, 10240]` + recurrent `[1, 48, 128, 128]`, bf16.
        assert_eq!(geo.conv_elements(), Some(30_720));
        assert_eq!(geo.recurrent_elements(), Some(786_432));
        assert_eq!(geo.bytes_per_tensor(), Some(1_634_304));

        let layout = layout_at(&geo, 4096);
        let per_checkpoint = layout.bytes_per_tensor * layout.num_layers as usize;
        assert_eq!(per_checkpoint, 78_446_592);
    }

    /// The figure above is a shape calculation. This checks it against what the
    /// MLX allocator actually reserves for the same arrays. Ignored by default:
    /// it allocates ~75 MiB on the GPU and needs a Metal device.
    #[test]
    #[ignore = "allocates ~75 MiB of Metal buffers; run with --ignored"]
    fn a_27b_gdn_checkpoint_allocates_the_bytes_its_layout_claims() {
        use crate::array::{DType, MxArray, get_active_memory};

        let geo = geometry(&qwen3_6_27b_config(), "BFloat16").expect("geometry");
        let layout = layout_at(&geo, 4096);
        let claimed = layout.bytes_per_tensor * layout.num_layers as usize;

        // Warm the allocator so the delta measures the checkpoint only.
        let warmup = MxArray::zeros(&[1024], Some(DType::BFloat16)).expect("warmup");
        warmup.eval();
        drop(warmup);
        let before = get_active_memory();

        let mut checkpoint = Vec::new();
        for _ in 0..geo.gdn_layers {
            let conv = MxArray::zeros(
                &[1, i64::from(geo.conv_rows), i64::from(geo.conv_dim)],
                Some(DType::BFloat16),
            )
            .expect("conv state");
            let recurrent = MxArray::zeros(
                &[
                    1,
                    i64::from(geo.num_v_heads),
                    i64::from(geo.v_head_dim),
                    i64::from(geo.k_head_dim),
                ],
                Some(DType::BFloat16),
            )
            .expect("recurrent state");
            conv.eval();
            recurrent.eval();
            checkpoint.push((conv, recurrent));
        }
        let resident = get_active_memory() - before;
        drop(checkpoint);

        // The allocator rounds up, so it may reserve more than the shapes ask
        // for, but never less.
        assert!(
            resident >= claimed as f64,
            "resident {resident} < claimed {claimed}"
        );
        assert!(
            resident < claimed as f64 * 1.25,
            "resident {resident} is more than 25% above claimed {claimed}"
        );
        println!(
            "one 27B GDN checkpoint: claimed {claimed} B, resident {resident} B ({:.2} MiB)",
            resident / 1024.0 / 1024.0
        );
    }
}
