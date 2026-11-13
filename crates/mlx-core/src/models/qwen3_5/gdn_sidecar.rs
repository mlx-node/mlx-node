//! Qwen3.5 (dense + MoE) GDN recurrent state as bytes: the GDN cell span of
//! a DFlash2 state object (Splash `stateSpans`), and the `gdn` geometry its
//! namespace and layout name.
//!
//! Each GDN layer writes ONE blob, `conv_state` bytes then `recurrent_state`
//! bytes, in layer order. No f32 round trip: 16-bit state goes out via
//! `to_uint16_native` and back via `from_bfloat16` / `from_float16`; the f32
//! recurrent state via `to_float32` / `from_float32`. The conv history's
//! element type is the GDN activation dtype (the embedding's output dtype); the
//! recurrent state is ALWAYS f32 ([`RECURRENT_DTYPE`]). Encoding re-checks each
//! array's dtype and shape and refuses (fail closed) on any mismatch rather
//! than mislabel.

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

use super::config::Qwen3_5Config;
use super::layer_cache::Qwen3_5LayerCache;

/// One blob per GDN layer (conv ++ recurrent), so exactly one tensor per layer.
pub(crate) const TENSORS_PER_LAYER: u32 = 1;

/// The recurrent state is carried in f32 by every GDN kernel regardless of the
/// activation dtype, so its sidecar element type is fixed rather than taken
/// from the pool.
const RECURRENT_DTYPE: DType = DType::Float32;

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
pub(crate) fn dtype_from_label(label: &str) -> Option<DType> {
    match label {
        "BFloat16" => Some(DType::BFloat16),
        "Float16" => Some(DType::Float16),
        "Float32" => Some(DType::Float32),
        _ => None,
    }
}

/// The label [`dtype_from_label`] reads back for `dtype`; `None` for a dtype
/// the codec cannot round-trip bit-exactly.
pub(crate) fn sidecar_dtype_label(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::BFloat16 => Some("BFloat16"),
        DType::Float16 => Some("Float16"),
        DType::Float32 => Some("Float32"),
        _ => None,
    }
}

fn element_size(dtype: DType) -> usize {
    match dtype {
        DType::Float32 => std::mem::size_of::<f32>(),
        DType::BFloat16 | DType::Float16 => std::mem::size_of::<u16>(),
        other => unreachable!("sidecar dtype {other:?} is rejected by dtype_from_label"),
    }
}

/// Geometry of qwen3_5's out-of-pool GDN state, derived once from the config +
/// the activation dtype.
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
    /// Conv-history element dtype label: the GDN activation dtype, fixed at
    /// load. The recurrent state is always [`RECURRENT_DTYPE`].
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

    /// Byte length of the conv-history half of the per-layer blob.
    fn conv_bytes(&self) -> Option<usize> {
        self.conv_elements()?
            .checked_mul(element_size(self.dtype()?))
    }

    /// Bytes of the whole cell: every GDN layer's blob (Splash
    /// `GdnStateLayout`; 153,944,064 on the 27B).
    pub(crate) fn cell_bytes(&self) -> Option<u64> {
        u64::try_from(self.bytes_per_tensor()?)
            .ok()?
            .checked_mul(u64::from(self.gdn_layers))
    }

    /// Byte length of the single per-layer blob = conv bytes ++ recurrent bytes.
    fn bytes_per_tensor(&self) -> Option<usize> {
        self.conv_bytes()?.checked_add(
            self.recurrent_elements()?
                .checked_mul(element_size(RECURRENT_DTYPE))?,
        )
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

    /// The namespace's `gdn` line: layers, conv history, conv channels, value
    /// heads, value and key head dims (Splash `persistentCacheNamespace`).
    pub(crate) fn namespace_gdn(&self) -> [u32; 6] {
        [
            self.gdn_layers,
            self.conv_rows,
            self.conv_dim,
            self.num_v_heads,
            self.v_head_dim,
            self.k_head_dim,
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
    ///
    /// `v2`: the recurrent half is f32 (`rec_dtype`), so every `v1` sidecar
    /// (bf16 recurrent) misses instead of being misread.
    pub fn fingerprint_component(&self) -> Vec<u8> {
        format!(
            "qwen3_5-gdn-sidecar:v2:layers={}:conv_rows={}:conv_dim={}:v_heads={}:v_dim={}:k_dim={}:dtype={}:rec_dtype={:?}:tensors={}",
            self.gdn_layers,
            self.conv_rows,
            self.conv_dim,
            self.num_v_heads,
            self.v_head_dim,
            self.k_head_dim,
            self.dtype,
            RECURRENT_DTYPE,
            TENSORS_PER_LAYER,
        )
        .into_bytes()
    }
}

/// Derive the sidecar geometry, or `None` when this checkpoint has no
/// out-of-pool GDN state to persist (no linear layers, a conv kernel of 1, or a
/// nonsensical head geometry) or a dtype the codec cannot round-trip. `None`
/// means "install no policy": the restore walk then reconciles nothing and the
/// GDN state is recomputed.
pub(crate) fn geometry(config: &Qwen3_5Config, activation: DType) -> Option<GdnSidecarGeometry> {
    let gdn_layers = u32::try_from(gdn_layers(config).len()).ok()?;
    if gdn_layers == 0 {
        return None;
    }
    let label = sidecar_dtype_label(activation)?;
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
        dtype: label.to_string(),
    };
    // A geometry whose payload cannot be sized is not persistable at all.
    geo.bytes_per_tensor()?;
    Some(geo)
}

/// Writes one GDN array (`conv_state` or `recurrent_state`) into `out`, whose
/// length is the array's bytes. `Ok(false)` when the array does not match the
/// geometry (dtype, shape, element count): writing bytes that do not match the
/// label would restore wrong state instead of missing.
fn encode_array(
    array: &MxArray,
    expected_dtype: DType,
    expected_shape: &[i64],
    out: &mut [u8],
) -> Result<bool> {
    if array.dtype()? != expected_dtype || array.shape()?.as_ref() != expected_shape {
        return Ok(false);
    }
    match expected_dtype {
        DType::BFloat16 | DType::Float16 => {
            let raw = array.to_uint16_native()?;
            let (chunks, rest) = out.as_chunks_mut::<{ std::mem::size_of::<u16>() }>();
            if raw.len() != chunks.len() || !rest.is_empty() {
                return Ok(false);
            }
            for (chunk, value) in chunks.iter_mut().zip(raw.iter()) {
                *chunk = value.to_le_bytes();
            }
        }
        DType::Float32 => {
            let raw = array.to_float32()?;
            let (chunks, rest) = out.as_chunks_mut::<{ std::mem::size_of::<f32>() }>();
            if raw.len() != chunks.len() || !rest.is_empty() {
                return Ok(false);
            }
            for (chunk, value) in chunks.iter_mut().zip(raw.iter()) {
                *chunk = value.to_le_bytes();
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Writes the materialized GDN caches into `cell`, layer-major: per GDN layer
/// its conv bytes, then its recurrent bytes. `cell` must be exactly
/// [`GdnSidecarGeometry::cell_bytes`] long. `Ok(false)` when the state does
/// not match the geometry: a missing slot, a wrong dtype, or a shape that is
/// not `[1, K-1, conv_dim]` / `[1, Hv, Dv, Dk]`.
///
/// The caller must have materialized the caches first.
#[allow(dead_code)]
pub(crate) fn encode_into(
    config: &Qwen3_5Config,
    geo: &GdnSidecarGeometry,
    caches: &[Qwen3_5LayerCache],
    cell: &mut [u8],
) -> Result<bool> {
    let layers = gdn_layers(config);
    if layers.len() != geo.gdn_layers as usize || caches.len() != config.num_layers.max(0) as usize
    {
        return Ok(false);
    }
    let (Some(dtype), Some(conv_bytes), Some(bytes_per_tensor), Some(cell_bytes)) = (
        geo.dtype(),
        geo.conv_bytes(),
        geo.bytes_per_tensor(),
        geo.cell_bytes(),
    ) else {
        return Ok(false);
    };
    if cell.len() as u64 != cell_bytes {
        return Ok(false);
    }
    let conv_shape = geo.conv_shape();
    let rec_shape = geo.recurrent_shape();
    for (&layer_idx, blob) in layers.iter().zip(cell.chunks_exact_mut(bytes_per_tensor)) {
        let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get(layer_idx) else {
            return Ok(false);
        };
        let (Some(conv), Some(rec)) = (arrays.get(0), arrays.get(1)) else {
            return Ok(false);
        };
        let (conv_out, rec_out) = blob.split_at_mut(conv_bytes);
        if !encode_array(conv, dtype, &conv_shape, conv_out)?
            || !encode_array(rec, RECURRENT_DTYPE, &rec_shape, rec_out)?
        {
            return Ok(false);
        }
    }
    Ok(true)
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
                .as_chunks::<{ std::mem::size_of::<u16>() }>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes(*pair))
                .collect();
            if dtype == DType::BFloat16 {
                MxArray::from_bfloat16(&raw, shape)?
            } else {
                MxArray::from_float16(&raw, shape)?
            }
        }
        DType::Float32 => {
            let raw: Vec<f32> = bytes
                .as_chunks::<{ std::mem::size_of::<f32>() }>()
                .0
                .iter()
                .map(|quad| f32::from_le_bytes(*quad))
                .collect();
            MxArray::from_float32(&raw, shape)?
        }
        _ => return Ok(None),
    };
    Ok(Some(array))
}

/// Rebuild the full-length layer-cache vector from a GDN cell written by
/// [`encode_into`]. Full-attention layers become fresh
/// [`Qwen3_5LayerCache::FullAttention`] shells (their K/V lives in the paged
/// pool). `None` when `cell` is not this geometry's cell.
#[allow(dead_code)]
pub(crate) fn decode_caches(
    config: &Qwen3_5Config,
    geo: &GdnSidecarGeometry,
    cell: &[u8],
) -> Result<Option<Vec<Qwen3_5LayerCache>>> {
    let layers = gdn_layers(config);
    if layers.len() != geo.gdn_layers as usize {
        return Ok(None);
    }
    let (
        Some(dtype),
        Some(conv_elements),
        Some(rec_elements),
        Some(conv_bytes),
        Some(bytes_per_tensor),
        Some(cell_bytes),
    ) = (
        geo.dtype(),
        geo.conv_elements(),
        geo.recurrent_elements(),
        geo.conv_bytes(),
        geo.bytes_per_tensor(),
        geo.cell_bytes(),
    )
    else {
        return Ok(None);
    };
    if cell.len() as u64 != cell_bytes {
        return Ok(None);
    }
    let conv_shape = geo.conv_shape();
    let rec_shape = geo.recurrent_shape();

    let mut caches: Vec<Qwen3_5LayerCache> = Qwen3_5LayerCache::fresh_layer_caches(config);
    for (&layer_idx, blob) in layers.iter().zip(cell.chunks_exact(bytes_per_tensor)) {
        let (conv_slice, rec_slice) = blob.split_at(conv_bytes);
        let Some(conv) = decode_array(conv_slice, dtype, &conv_shape, conv_elements)? else {
            return Ok(None);
        };
        let Some(rec) = decode_array(rec_slice, RECURRENT_DTYPE, &rec_shape, rec_elements)? else {
            return Ok(None);
        };
        let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get_mut(layer_idx) else {
            return Ok(None);
        };
        arrays.set(0, conv)?;
        arrays.set(1, rec)?;
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
            rope_scaling: None,
            qwen35_gguf_gdn_layout: None,
            kv_format: None,
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
            n_mtp_layers: 0,
        }
    }

    fn geo_of(cfg: &Qwen3_5Config) -> GdnSidecarGeometry {
        geometry(cfg, DType::BFloat16).expect("geometry")
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
        assert!(geometry(&cfg, DType::BFloat16).is_none());
        // Unsupported dtype: codec cannot round-trip.
        assert!(geometry(&config(), DType::Int8).is_none());
    }

    /// Build a full-length cache vec, filling each GDN layer's conv (bf16, or
    /// f16 when `bf16` is false) and recurrent (`rec` dtype) slots with
    /// distinct patterns so a mis-ordered round trip is visible.
    fn caches_for(
        cfg: &Qwen3_5Config,
        geo: &GdnSidecarGeometry,
        bf16: bool,
        rec: DType,
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
            let make = |n: usize, tag: u16, shape: &[i64], dtype: DType| -> MxArray {
                let raw: Vec<u16> = (0..n)
                    .map(|i| (i as u16).wrapping_mul(31).wrapping_add(tag))
                    .collect();
                match dtype {
                    DType::BFloat16 => MxArray::from_bfloat16(&raw, shape).unwrap(),
                    DType::Float16 => MxArray::from_float16(&raw, shape).unwrap(),
                    // Low mantissa bits set (and subnormals for small `r`):
                    // finite patterns a bf16 round trip would not preserve.
                    _ => MxArray::from_float32(
                        &raw.iter()
                            .map(|&r| f32::from_bits((u32::from(r) << 8) | 0x11))
                            .collect::<Vec<_>>(),
                        shape,
                    )
                    .unwrap(),
                }
            };
            let conv_dtype = if bf16 {
                DType::BFloat16
            } else {
                DType::Float16
            };
            if let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get_mut(layer) {
                arrays
                    .set(
                        0,
                        make(conv_elems, ordinal as u16 * 2, &conv_shape, conv_dtype),
                    )
                    .expect("test cache slot 0");
                arrays
                    .set(1, make(rec_elems, ordinal as u16 * 2 + 1, &rec_shape, rec))
                    .expect("test cache slot 1");
            }
        }
        caches
    }

    #[test]
    fn payload_round_trips_layer_major_and_bit_exact() {
        use crate::models::qwen3_5::model::arrays_bits_equal_for_test;
        let cfg = config();
        let geo = geo_of(&cfg);
        let caches = caches_for(&cfg, &geo, true, DType::Float32);

        let mut cell = vec![0u8; geo.cell_bytes().unwrap() as usize];
        assert!(
            encode_into(&cfg, &geo, &caches, &mut cell).unwrap(),
            "bf16 conv + f32 recurrent caches must encode"
        );
        assert_eq!(cell.len(), 6 * (3 * 14 * 2 + 2 * 4 * 3 * 4));

        let decoded = decode_caches(&cfg, &geo, &cell)
            .unwrap()
            .expect("payload must decode");
        assert_eq!(decoded.len(), caches.len());
        for (layer, (before, after)) in caches.iter().zip(decoded.iter()).enumerate() {
            match (before, after) {
                (Qwen3_5LayerCache::FullAttention(_), Qwen3_5LayerCache::FullAttention(_)) => {}
                (Qwen3_5LayerCache::Linear(b), Qwen3_5LayerCache::Linear(a)) => {
                    assert_eq!(a.get(0).unwrap().dtype().unwrap(), DType::BFloat16);
                    assert_eq!(a.get(1).unwrap().dtype().unwrap(), DType::Float32);
                    for slot in 0..2 {
                        assert!(
                            arrays_bits_equal_for_test(b.get(slot).unwrap(), a.get(slot).unwrap())
                                .unwrap(),
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
        let caches = caches_for(&cfg, &geo, false, DType::Float32);
        assert!(
            !encode_into(
                &cfg,
                &geo,
                &caches,
                &mut vec![0u8; geo.cell_bytes().unwrap() as usize]
            )
            .unwrap(),
            "a non-bf16 conv GDN cache must skip capture, not reinterpret its bytes"
        );
        // A bf16 recurrent state (the pre-f32 carry) must never be widened
        // into an f32-labelled blob.
        let caches = caches_for(&cfg, &geo, true, DType::BFloat16);
        assert!(
            !encode_into(
                &cfg,
                &geo,
                &caches,
                &mut vec![0u8; geo.cell_bytes().unwrap() as usize]
            )
            .unwrap(),
            "a non-f32 recurrent GDN cache must skip capture"
        );
    }

    #[test]
    fn capture_skips_on_missing_slot() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let mut caches = caches_for(&cfg, &geo, true, DType::Float32);
        if let Some(Qwen3_5LayerCache::Linear(arrays)) = caches.get_mut(2) {
            arrays.reset();
        }
        assert!(
            !encode_into(
                &cfg,
                &geo,
                &caches,
                &mut vec![0u8; geo.cell_bytes().unwrap() as usize]
            )
            .unwrap()
        );
    }

    #[test]
    fn decode_refuses_short_payloads() {
        let cfg = config();
        let geo = geo_of(&cfg);
        let mut cell = vec![0u8; geo.cell_bytes().unwrap() as usize];
        assert!(
            encode_into(
                &cfg,
                &geo,
                &caches_for(&cfg, &geo, true, DType::Float32),
                &mut cell,
            )
            .unwrap()
        );
        assert!(
            decode_caches(&cfg, &geo, &cell[..cell.len() - 1])
                .unwrap()
                .is_none()
        );
        assert!(
            !encode_into(
                &cfg,
                &geo,
                &caches_for(&cfg, &geo, true, DType::Float32),
                &mut cell[1..],
            )
            .unwrap()
        );
    }

    #[test]
    fn fingerprint_component_separates_geometry_and_dtype() {
        let cfg = config();
        let base = geo_of(&cfg).fingerprint_component();

        let mut wider = cfg.clone();
        wider.linear_value_head_dim = 8;
        assert_ne!(geo_of(&wider).fingerprint_component(), base);

        assert_ne!(
            geometry(&cfg, DType::Float32)
                .unwrap()
                .fingerprint_component(),
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
        let geo = geometry(&qwen3_6_27b_config(), DType::BFloat16).expect("geometry");
        assert_eq!(geo.gdn_layers, 48);
        // conv `[1, 3, 10240]` bf16 + recurrent `[1, 48, 128, 128]` f32.
        assert_eq!(geo.conv_elements(), Some(30_720));
        assert_eq!(geo.recurrent_elements(), Some(786_432));
        assert_eq!(geo.conv_bytes(), Some(61_440));
        assert_eq!(geo.bytes_per_tensor(), Some(61_440 + 3_145_728));

        assert_eq!(geo.cell_bytes(), Some(153_944_064));
        assert_eq!(geo.namespace_gdn(), [48, 3, 10240, 48, 128, 128]);
    }

    /// Checks the shape calculation above against what the MLX allocator actually
    /// reserves. Ignored by default: needs a Metal device and allocates on the GPU.
    #[test]
    #[ignore = "allocates ~147 MiB of Metal buffers; run with --ignored"]
    fn a_27b_gdn_checkpoint_allocates_the_bytes_its_layout_claims() {
        use crate::array::{DType, MxArray, get_active_memory};

        let geo = geometry(&qwen3_6_27b_config(), DType::BFloat16).expect("geometry");
        let claimed = geo.cell_bytes().expect("cell bytes") as usize;

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
                Some(RECURRENT_DTYPE),
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
