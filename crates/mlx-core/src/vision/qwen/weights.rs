//! Load the shared Qwen vision tower from normalized tensors.
use super::encoder::{QwenVisionConfig, QwenVisionEncoder};
use crate::array::MxArray;
use crate::nn::LayerNorm;
use crate::vision::encoder::{VisionAttention, VisionEncoderLayer, VisionMLP};
use crate::vision::projector::SpatialProjector;
use napi::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use tracing::info;

/// Parse vision config from JSON.
pub(crate) fn parse_vision_config(raw: &Value) -> QwenVisionConfig {
    let vision_cfg = raw.get("vision_config");

    let get = |key: &str, default: i32| -> i32 {
        vision_cfg
            .and_then(|v| v[key].as_i64())
            .unwrap_or(default as i64) as i32
    };

    QwenVisionConfig {
        hidden_size: get("hidden_size", 1152),
        intermediate_size: get("intermediate_size", 4304),
        num_heads: get("num_heads", 16),
        num_layers: vision_cfg
            .and_then(|v| {
                v["depth"]
                    .as_i64()
                    .or_else(|| v["num_hidden_layers"].as_i64())
            })
            .unwrap_or(27) as i32,
        patch_size: get("patch_size", 16),
        spatial_merge_size: get("spatial_merge_size", 2),
        image_size: get("image_size", 768),
        out_hidden_size: get("out_hidden_size", 4096),
    }
}

/// Collapse a 5D Conv3d patch-embed weight `[out, kD, kH, kW, in]` into a 2D
/// Conv2d kernel `[out, kH, kW, in]` by summing over the temporal axis.
///
/// The image processor duplicates the static frame across the temporal axis, so
/// the effective 2D kernel is the sum of the temporal slices (matches mlx-vlm
/// `qwen3_vl/vision.py`). Summing all `kD` slices keeps this robust if `kD != 2`.
fn collapse_patch_embed_conv3d(pe_weight: &MxArray) -> Result<MxArray> {
    pe_weight.sum(Some(&[1]), None)
}

/// Load vision encoder weights from params.
pub(crate) fn load_vision_weights(
    encoder: &mut QwenVisionEncoder,
    params: &HashMap<String, MxArray>,
    config: &QwenVisionConfig,
) -> Result<()> {
    let get = |key: &str| -> Result<&MxArray> {
        params
            .get(key)
            .ok_or_else(|| Error::from_reason(format!("Missing vision weight: {}", key)))
    };

    let get_opt = |key: &str| -> Option<&MxArray> { params.get(key) };

    // Patch embedding is the required load anchor: the encoder constructor
    // starts with a zero-valued placeholder, so silently accepting a missing
    // projection would install an image-blind tower. Check the already
    // normalized key used by both dense and MoE loaders. Keep both supported
    // checkpoint layouts: 4D Conv2d [out, kH, kW, in] and 5D Conv3d
    // [out, kD, kH, kW, in]. For Conv3d, the static frame is duplicated across
    // the temporal axis, so the effective 2D kernel is the sum of its slices.
    let pe_weight = get("patch_embed.proj.weight")?;
    let pe_bias = get_opt("patch_embed.proj.bias");
    let ndim = pe_weight.ndim()?;
    if ndim == 5 {
        let conv2d_weight = collapse_patch_embed_conv3d(pe_weight)?;
        encoder.set_patch_embed(&conv2d_weight, pe_bias)?;
    } else {
        encoder.set_patch_embed(pe_weight, pe_bias)?;
    }

    // Position embedding
    if let Some(pos_embed) = get_opt("pos_embed.weight") {
        encoder.set_pos_embed(pos_embed);
    }

    // Encoder layers (blocks.0..blocks.N)
    for layer_idx in 0..config.num_layers {
        let prefix = format!("blocks.{}", layer_idx);

        let qkv_w = get(&format!("{}.attn.qkv.weight", prefix))?;
        let qkv_b = get_opt(&format!("{}.attn.qkv.bias", prefix));
        let proj_w = get(&format!("{}.attn.proj.weight", prefix))?;
        let proj_b = get_opt(&format!("{}.attn.proj.bias", prefix));

        let attn = VisionAttention::new(
            config.hidden_size as u32,
            config.num_heads as u32,
            qkv_w,
            qkv_b,
            proj_w,
            proj_b,
        )?;

        let fc1_w = get(&format!("{}.mlp.linear_fc1.weight", prefix))?;
        let fc1_b = get_opt(&format!("{}.mlp.linear_fc1.bias", prefix));
        let fc2_w = get(&format!("{}.mlp.linear_fc2.weight", prefix))?;
        let fc2_b = get_opt(&format!("{}.mlp.linear_fc2.bias", prefix));

        let mlp = VisionMLP::new(fc1_w, fc1_b, fc2_w, fc2_b)?;

        let norm1_w = get(&format!("{}.norm1.weight", prefix))?;
        let norm1_b = get_opt(&format!("{}.norm1.bias", prefix));
        let norm2_w = get(&format!("{}.norm2.weight", prefix))?;
        let norm2_b = get_opt(&format!("{}.norm2.bias", prefix));

        let ln1 = LayerNorm::from_weights(norm1_w, norm1_b, Some(1e-6))?;
        let ln2 = LayerNorm::from_weights(norm2_w, norm2_b, Some(1e-6))?;

        let layer = VisionEncoderLayer::new(&ln1, &ln2, &attn, &mlp);
        encoder.add_layer(&layer);
    }

    // Merger (spatial projector)
    let ln_q_w = get("merger.norm.weight")?;
    let ln_q_b = get("merger.norm.bias")?;
    let fc1_w = get("merger.linear_fc1.weight")?;
    let fc1_b = get("merger.linear_fc1.bias")?;
    let fc2_w = get("merger.linear_fc2.weight")?;
    let fc2_b = get("merger.linear_fc2.bias")?;

    let merger = SpatialProjector::new(
        config.spatial_merge_size as u32,
        ln_q_w,
        ln_q_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
    )?;
    encoder.set_merger(merger);

    info!(
        "Loaded vision encoder: {} layers, merger ready",
        config.num_layers
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collapse_patch_embed_conv3d_sums_temporal_slices() {
        // Synthetic Conv3d weight [out=2, kD=2, kH=2, kW=2, in=3] with distinct
        // values per temporal slice. The collapse must SUM over the temporal
        // axis (slice0 + slice1), not drop slice1.
        let out_c = 2i64;
        let kd = 2i64;
        let kh = 2i64;
        let kw = 2i64;
        let in_c = 3i64;
        let n = (out_c * kd * kh * kw * in_c) as usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let pe_weight = MxArray::from_float32(&data, &[out_c, kd, kh, kw, in_c]).unwrap();

        let collapsed = collapse_patch_embed_conv3d(&pe_weight).unwrap();
        collapsed.eval();
        let shape: Vec<i64> = collapsed.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![out_c, kh, kw, in_c]);

        // Expected = slice[:,0,:,:,:] + slice[:,1,:,:,:].
        let slice0 = pe_weight
            .slice(&[0, 0, 0, 0, 0], &[out_c, 1, kh, kw, in_c])
            .unwrap()
            .squeeze(Some(&[1]))
            .unwrap();
        let slice1 = pe_weight
            .slice(&[0, 1, 0, 0, 0], &[out_c, 2, kh, kw, in_c])
            .unwrap()
            .squeeze(Some(&[1]))
            .unwrap();
        let expected = slice0.add(&slice1).unwrap();
        expected.eval();

        let got: Vec<f32> = collapsed.to_float32().unwrap().to_vec();
        let exp: Vec<f32> = expected.to_float32().unwrap().to_vec();
        assert_eq!(got.len(), exp.len());
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "element {i}: collapsed {g} != slice0+slice1 {e}"
            );
        }
    }

    #[test]
    fn vision_patch_embed_anchor_is_required_and_accepts_4d_or_5d() {
        let config = QwenVisionConfig {
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 1,
            num_layers: 0,
            patch_size: 1,
            spatial_merge_size: 1,
            image_size: 1,
            out_hidden_size: 4,
        };
        let mut missing_encoder =
            QwenVisionEncoder::new(config.clone()).expect("construct tiny vision encoder");
        let err = load_vision_weights(&mut missing_encoder, &HashMap::new(), &config)
            .expect_err("a partial vision tower must not retain the zero patch projection");
        assert_eq!(
            err.reason, "Missing vision weight: patch_embed.proj.weight",
            "the required anchor uses the normalized vision key"
        );

        let array = |shape: &[i64]| {
            let len = shape.iter().map(|dim| *dim as usize).product();
            MxArray::from_float32(&vec![0.0; len], shape).expect("construct tiny vision weight")
        };
        let patch_shapes: [&[i64]; 2] = [&[4, 1, 1, 3], &[4, 2, 1, 1, 3]];
        for patch_shape in patch_shapes {
            let mut params = HashMap::new();
            params.insert("patch_embed.proj.weight".to_string(), array(patch_shape));
            params.insert("merger.norm.weight".to_string(), array(&[4]));
            params.insert("merger.norm.bias".to_string(), array(&[4]));
            params.insert("merger.linear_fc1.weight".to_string(), array(&[4, 4]));
            params.insert("merger.linear_fc1.bias".to_string(), array(&[4]));
            params.insert("merger.linear_fc2.weight".to_string(), array(&[4, 4]));
            params.insert("merger.linear_fc2.bias".to_string(), array(&[4]));

            let mut encoder =
                QwenVisionEncoder::new(config.clone()).expect("construct tiny vision encoder");
            load_vision_weights(&mut encoder, &params, &config)
                .unwrap_or_else(|err| panic!("{patch_shape:?} patch anchor must load: {err}"));
        }
    }
}
