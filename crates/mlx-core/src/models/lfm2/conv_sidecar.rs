//! Durable LFM2 short-convolution state for hybrid paged-cache restores.
//!
//! LFM2 pages only its full-attention layers. Every convolution layer carries
//! a `[1, conv_L_cache - 1, hidden_size]` recurrent tensor outside that pool,
//! so restoring K/V without this state would resume a model state that never
//! existed. The sidecar is content-addressed under [`ColdGroup::ConvState`]
//! at the same exact block boundary as the K/V chain. Restore therefore uses
//! the common reconcile-down policy: no validated sidecar means no cold hit.

use mlx_paged_attn::{ColdGroup, ColdSidecarLayout, ColdSidecarPolicy};
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};

use super::config::Lfm2Config;
use super::layer_cache::Lfm2LayerCache;

const TENSORS_PER_LAYER: u32 = 1;

pub(crate) fn conv_layers(config: &Lfm2Config) -> Vec<usize> {
    (0..config.num_hidden_layers.max(0) as usize)
        .filter(|&index| !config.is_attention_layer(index))
        .collect()
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ConvSidecarGeometry {
    pub conv_layers: u32,
    pub rows: u32,
    pub hidden_size: u32,
    pub dtype: String,
}

impl ConvSidecarGeometry {
    fn dtype(&self) -> Option<DType> {
        dtype_from_label(&self.dtype)
    }

    fn elements_per_tensor(&self) -> Option<usize> {
        (self.rows as usize).checked_mul(self.hidden_size as usize)
    }

    fn bytes_per_tensor(&self) -> Option<usize> {
        self.elements_per_tensor()?
            .checked_mul(element_size(self.dtype()?))
    }

    fn shape(&self) -> [i64; 3] {
        [1, self.rows as i64, self.hidden_size as i64]
    }

    pub(crate) fn fingerprint_component(&self) -> Vec<u8> {
        format!(
            "lfm2-conv-sidecar:v1:layers={}:rows={}:hidden={}:dtype={}:tensors={}",
            self.conv_layers, self.rows, self.hidden_size, self.dtype, TENSORS_PER_LAYER,
        )
        .into_bytes()
    }
}

pub(crate) fn geometry(config: &Lfm2Config, cache_dtype: &str) -> Option<ConvSidecarGeometry> {
    let conv_layers = u32::try_from(conv_layers(config).len()).ok()?;
    let rows = u32::try_from(config.conv_l_cache.checked_sub(1)?).ok()?;
    let hidden_size = u32::try_from(config.hidden_size).ok()?;
    dtype_from_label(cache_dtype)?;
    if conv_layers == 0 || rows == 0 || hidden_size == 0 {
        return None;
    }
    let geometry = ConvSidecarGeometry {
        conv_layers,
        rows,
        hidden_size,
        dtype: cache_dtype.to_string(),
    };
    geometry.bytes_per_tensor()?;
    Some(geometry)
}

pub(crate) fn layout_at(geometry: &ConvSidecarGeometry, boundary_tokens: u32) -> ColdSidecarLayout {
    ColdSidecarLayout {
        group: ColdGroup::ConvState,
        boundary_tokens,
        num_layers: geometry.conv_layers,
        tensors_per_layer: TENSORS_PER_LAYER,
        dtype: geometry.dtype.clone(),
        dims: vec![geometry.rows, geometry.hidden_size],
        bytes_per_tensor: geometry.bytes_per_tensor().unwrap_or(0),
    }
}

pub(crate) fn policy(config: &Lfm2Config, cache_dtype: &str) -> Option<ColdSidecarPolicy> {
    let geometry = geometry(config, cache_dtype)?;
    ColdSidecarPolicy::new(layout_at(&geometry, 0)).ok()
}

fn encode_array(
    array: &MxArray,
    dtype: DType,
    shape: &[i64],
    elements: usize,
) -> Result<Option<Vec<u8>>> {
    if array.dtype()? != dtype || array.shape()?.as_ref() != shape {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(elements.saturating_mul(element_size(dtype)));
    match dtype {
        DType::BFloat16 | DType::Float16 => {
            let raw = array.to_uint16_native()?;
            if raw.len() != elements {
                return Ok(None);
            }
            bytes.extend(raw.into_iter().flat_map(u16::to_le_bytes));
        }
        DType::Float32 => {
            let raw = array.to_float32()?.to_vec();
            if raw.len() != elements {
                return Ok(None);
            }
            bytes.extend(raw.into_iter().flat_map(f32::to_le_bytes));
        }
        _ => return Ok(None),
    }
    Ok(Some(bytes))
}

#[cfg(test)]
pub(crate) fn encode_tensors(
    config: &Lfm2Config,
    geometry: &ConvSidecarGeometry,
    caches: &[Lfm2LayerCache],
) -> Result<Option<Vec<Vec<u8>>>> {
    let layers = conv_layers(config);
    if layers.len() != geometry.conv_layers as usize
        || caches.len() != config.num_hidden_layers.max(0) as usize
    {
        return Ok(None);
    }
    let Some(dtype) = geometry.dtype() else {
        return Ok(None);
    };
    let Some(elements) = geometry.elements_per_tensor() else {
        return Ok(None);
    };
    let shape = geometry.shape();
    let mut tensors = Vec::with_capacity(layers.len());
    for layer in layers {
        let Some(Lfm2LayerCache::Conv(cache)) = caches.get(layer) else {
            return Ok(None);
        };
        let Some(state) = cache.get(0) else {
            return Ok(None);
        };
        let Some(bytes) = encode_array(state, dtype, &shape, elements)? else {
            return Ok(None);
        };
        tensors.push(bytes);
    }
    Ok(Some(tensors))
}

/// Clone the immutable MLX handles for every convolution layer at one exact
/// boundary. Later cache updates replace those handles rather than mutating
/// their values, so this is a cheap, stable checkpoint until capture.
pub(crate) fn snapshot_states(
    config: &Lfm2Config,
    caches: &[Lfm2LayerCache],
) -> Option<Vec<MxArray>> {
    let mut states = Vec::with_capacity(conv_layers(config).len());
    for layer in conv_layers(config) {
        let Lfm2LayerCache::Conv(cache) = caches.get(layer)? else {
            return None;
        };
        states.push(cache.get(0)?.clone());
    }
    Some(states)
}

pub(crate) fn encode_states(
    geometry: &ConvSidecarGeometry,
    states: &[MxArray],
) -> Result<Option<Vec<Vec<u8>>>> {
    if states.len() != geometry.conv_layers as usize {
        return Ok(None);
    }
    let Some(dtype) = geometry.dtype() else {
        return Ok(None);
    };
    let Some(elements) = geometry.elements_per_tensor() else {
        return Ok(None);
    };
    let shape = geometry.shape();
    let mut tensors = Vec::with_capacity(states.len());
    for state in states {
        let Some(bytes) = encode_array(state, dtype, &shape, elements)? else {
            return Ok(None);
        };
        tensors.push(bytes);
    }
    Ok(Some(tensors))
}

fn decode_array(
    bytes: &[u8],
    dtype: DType,
    shape: &[i64],
    elements: usize,
) -> Result<Option<MxArray>> {
    if bytes.len() != elements.saturating_mul(element_size(dtype)) {
        return Ok(None);
    }
    match dtype {
        DType::BFloat16 | DType::Float16 => {
            let raw = bytes
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect::<Vec<_>>();
            if raw.len() != elements {
                return Ok(None);
            }
            if dtype == DType::BFloat16 {
                Ok(Some(MxArray::from_bfloat16(&raw, shape)?))
            } else {
                Ok(Some(MxArray::from_float16(&raw, shape)?))
            }
        }
        DType::Float32 => {
            let raw = bytes
                .chunks_exact(4)
                .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                .collect::<Vec<_>>();
            if raw.len() != elements {
                return Ok(None);
            }
            Ok(Some(MxArray::from_float32(&raw, shape)?))
        }
        _ => Ok(None),
    }
}

pub(crate) fn decode_caches(
    config: &Lfm2Config,
    geometry: &ConvSidecarGeometry,
    tensors: &[Vec<u8>],
) -> Result<Option<Vec<Lfm2LayerCache>>> {
    let layers = conv_layers(config);
    if layers.len() != geometry.conv_layers as usize || tensors.len() != layers.len() {
        return Ok(None);
    }
    let Some(dtype) = geometry.dtype() else {
        return Ok(None);
    };
    let Some(elements) = geometry.elements_per_tensor() else {
        return Ok(None);
    };
    let shape = geometry.shape();
    let mut caches = super::model::init_caches(config);
    for (ordinal, layer) in layers.into_iter().enumerate() {
        let Some(state) = decode_array(&tensors[ordinal], dtype, &shape, elements)? else {
            return Ok(None);
        };
        let Some(cache) = caches
            .get_mut(layer)
            .and_then(Lfm2LayerCache::as_conv_cache_mut)
        else {
            return Ok(None);
        };
        cache.set(0, state);
    }
    Ok(Some(caches))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Lfm2Config {
        Lfm2Config {
            vocab_size: 32,
            hidden_size: 4,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            conv_bias: false,
            conv_l_cache: 3,
            block_dim: 4,
            block_ff_dim: 8,
            block_multiple_of: 1,
            block_ffn_dim_multiplier: 1.0,
            block_auto_adjust_ff_dim: false,
            rope_theta: 10_000.0,
            layer_types: vec![
                "conv".into(),
                "full_attention".into(),
                "conv".into(),
                "full_attention".into(),
            ],
            tie_embedding: true,
            eos_token_id: 1,
            bos_token_id: 2,
            pad_token_id: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            persist_paged_cache: None,
            intermediate_size: None,
            moe_intermediate_size: None,
            num_experts: None,
            num_experts_per_tok: None,
            num_dense_layers: None,
            norm_topk_prob: None,
            use_expert_bias: None,
        }
    }

    #[test]
    fn layout_pins_the_out_of_pool_geometry() {
        let geometry = geometry(&config(), "BFloat16").expect("geometry");
        assert_eq!(geometry.conv_layers, 2);
        assert_eq!(layout_at(&geometry, 32).dims, vec![2, 4]);
        assert_eq!(layout_at(&geometry, 32).bytes_per_tensor, 16);
        assert_eq!(
            policy(&config(), "BFloat16").unwrap().group(),
            ColdGroup::ConvState
        );
    }

    #[test]
    fn conv_state_round_trips_bit_exact() {
        let config = config();
        let geometry = geometry(&config, "BFloat16").unwrap();
        let mut caches = super::super::model::init_caches(&config);
        for (ordinal, layer) in conv_layers(&config).into_iter().enumerate() {
            let raw = (0..8)
                .map(|index| (ordinal as u16 * 100).wrapping_add(index))
                .collect::<Vec<_>>();
            caches[layer]
                .as_conv_cache_mut()
                .unwrap()
                .set(0, MxArray::from_bfloat16(&raw, &[1, 2, 4]).unwrap());
        }
        let tensors = encode_tensors(&config, &geometry, &caches)
            .unwrap()
            .expect("encode");
        let restored = decode_caches(&config, &geometry, &tensors)
            .unwrap()
            .expect("decode");
        for layer in conv_layers(&config) {
            let before = match &caches[layer] {
                Lfm2LayerCache::Conv(cache) => cache.get(0).unwrap(),
                _ => unreachable!(),
            };
            let after = match &restored[layer] {
                Lfm2LayerCache::Conv(cache) => cache.get(0).unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(
                before.to_uint16_native().unwrap(),
                after.to_uint16_native().unwrap()
            );
        }
    }

    #[test]
    fn missing_or_wrong_dtype_state_fails_closed() {
        let config = config();
        let geometry = geometry(&config, "BFloat16").unwrap();
        let mut caches = super::super::model::init_caches(&config);
        assert!(
            encode_tensors(&config, &geometry, &caches)
                .unwrap()
                .is_none()
        );
        caches[0]
            .as_conv_cache_mut()
            .unwrap()
            .set(0, MxArray::from_float16(&[0; 8], &[1, 2, 4]).unwrap());
        assert!(
            encode_tensors(&config, &geometry, &caches)
                .unwrap()
                .is_none()
        );
    }
}
