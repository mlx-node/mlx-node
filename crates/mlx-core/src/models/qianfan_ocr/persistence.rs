//! Weight Loading (Persistence) for Qianfan-OCR
//!
//! Transforms HuggingFace InternVL weight keys to the internal format
//! and handles Conv2d NCHW→NHWC transposition for the vision encoder's
//! patch embedding.

use crate::array::MxArray;
use std::collections::HashMap;

/// Transform a HuggingFace InternVL weight key to the internal key format.
///
/// The model has 745 weights split across three prefixes:
/// - `vision_model.*` (340 keys) → `vision.*`
/// - `language_model.*` (399 keys) → `lm.*`
/// - `mlp1.*` (6 keys) → `bridge.*`
pub(crate) fn transform_key(key: &str) -> String {
    // --- Vision encoder keys ---
    if let Some(rest) = key.strip_prefix("vision_model.") {
        // Embeddings
        if let Some(rest) = rest.strip_prefix("embeddings.") {
            // All embedding keys keep the vision.embeddings prefix
            return format!("vision.embeddings.{rest}");
        }
        // Encoder layers
        if let Some(rest) = rest.strip_prefix("encoder.layers.") {
            // rest is like "{n}.attn.proj.weight"
            // Find the layer number
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                let new_suffix = suffix;
                return format!("vision.layers.{layer_num}.{new_suffix}");
            }
        }
        // Anything else under vision_model (shouldn't happen for known weights)
        return format!("vision.{rest}");
    }

    // --- Bridge MLP keys (mlp1 is Sequential: 0=LN, 1=Linear, 3=Linear) ---
    if let Some(rest) = key.strip_prefix("mlp1.") {
        let new_rest = match rest {
            "0.weight" => "ln.weight",
            "0.bias" => "ln.bias",
            "1.weight" => "linear1.weight",
            "1.bias" => "linear1.bias",
            "3.weight" => "linear2.weight",
            "3.bias" => "linear2.bias",
            _ => rest,
        };
        return format!("bridge.{new_rest}");
    }

    // --- Language model keys ---
    if let Some(rest) = key.strip_prefix("language_model.") {
        if let Some(rest) = rest.strip_prefix("model.") {
            // embed_tokens
            if let Some(suffix) = rest.strip_prefix("embed_tokens.") {
                return format!("lm.embedding.{suffix}");
            }
            // final norm
            if let Some(suffix) = rest.strip_prefix("norm.") {
                return format!("lm.final_norm.{suffix}");
            }
            // layers: pass through self_attn/mlp/layernorm subkeys unchanged
            if let Some(rest) = rest.strip_prefix("layers.") {
                return format!("lm.layers.{rest}");
            }
        }
        // lm_head
        if let Some(suffix) = rest.strip_prefix("lm_head.") {
            return format!("lm.lm_head.{suffix}");
        }
    }

    // Fallback: return key unchanged
    key.to_string()
}

/// Check if a weight needs Conv2d NCHW→NHWC transposition.
///
/// Only the vision encoder's patch embedding Conv2d weight needs transposing.
/// PyTorch stores Conv2d as [out_channels, in_channels, kH, kW] (NCHW),
/// but MLX expects [out_channels, kH, kW, in_channels] (NHWC).
fn needs_conv2d_transpose(key: &str, weight: &MxArray) -> bool {
    if !key.contains("patch_embedding.weight") || weight.ndim().unwrap_or(0) != 4 {
        return false;
    }
    // Detect PyTorch OIHW vs MLX OHWI by checking dim layout.
    // PyTorch: [O, I, H, W] — for RGB input, dim[1]=3, dim[2]=kernel
    // MLX:     [O, H, W, I] — for RGB input, dim[1]=kernel, dim[3]=3
    // Only transpose if it looks like OIHW (dim[1] < dim[2]).
    if let Ok(shape) = weight.shape()
        && shape.len() == 4
    {
        return shape[1] < shape[2]; // OIHW: in_channels < kernel_h
    }
    false
}

/// Transpose a Conv2d weight from PyTorch NCHW to MLX NHWC format.
///
/// [O, I, H, W] → [O, H, W, I]
fn transpose_conv2d_weight(weight: &MxArray) -> napi::Result<MxArray> {
    weight.transpose(Some(&[0, 2, 3, 1]))
}

/// Load and transform Qianfan-OCR weights from HuggingFace format.
///
/// Applies key transformations and Conv2d transposition for all weights.
pub(crate) fn load_qianfan_ocr_weights(
    weights: HashMap<String, MxArray>,
) -> napi::Result<HashMap<String, MxArray>> {
    let mut result = HashMap::with_capacity(weights.len());
    for (key, value) in weights {
        let new_key = transform_key(&key);
        let new_value = if needs_conv2d_transpose(&new_key, &value) {
            transpose_conv2d_weight(&value)?
        } else {
            value
        };
        result.insert(new_key, new_value);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // transform_key: Vision embeddings
    // ========================================

    #[test]
    fn test_transform_key_vision_patch_embedding_weight() {
        assert_eq!(
            transform_key("vision_model.embeddings.patch_embedding.weight"),
            "vision.embeddings.patch_embedding.weight"
        );
    }

    #[test]
    fn test_transform_key_vision_patch_embedding_bias() {
        assert_eq!(
            transform_key("vision_model.embeddings.patch_embedding.bias"),
            "vision.embeddings.patch_embedding.bias"
        );
    }

    #[test]
    fn test_transform_key_vision_class_embedding() {
        assert_eq!(
            transform_key("vision_model.embeddings.class_embedding"),
            "vision.embeddings.class_embedding"
        );
    }

    #[test]
    fn test_transform_key_vision_position_embedding() {
        assert_eq!(
            transform_key("vision_model.embeddings.position_embedding"),
            "vision.embeddings.position_embedding"
        );
    }

    // ========================================
    // transform_key: Vision encoder layers
    // ========================================

    #[test]
    fn test_transform_key_vision_attn_qkv() {
        assert_eq!(
            transform_key("vision_model.encoder.layers.0.attn.qkv.weight"),
            "vision.layers.0.attn.qkv.weight"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.23.attn.qkv.bias"),
            "vision.layers.23.attn.qkv.bias"
        );
    }

    #[test]
    fn test_transform_key_vision_attn_proj() {
        assert_eq!(
            transform_key("vision_model.encoder.layers.5.attn.proj.weight"),
            "vision.layers.5.attn.proj.weight"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.5.attn.proj.bias"),
            "vision.layers.5.attn.proj.bias"
        );
    }

    #[test]
    fn test_transform_key_vision_mlp() {
        assert_eq!(
            transform_key("vision_model.encoder.layers.10.mlp.fc1.weight"),
            "vision.layers.10.mlp.fc1.weight"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.10.mlp.fc1.bias"),
            "vision.layers.10.mlp.fc1.bias"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.10.mlp.fc2.weight"),
            "vision.layers.10.mlp.fc2.weight"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.10.mlp.fc2.bias"),
            "vision.layers.10.mlp.fc2.bias"
        );
    }

    #[test]
    fn test_transform_key_vision_layer_scale() {
        assert_eq!(
            transform_key("vision_model.encoder.layers.3.ls1"),
            "vision.layers.3.ls1"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.3.ls2"),
            "vision.layers.3.ls2"
        );
    }

    #[test]
    fn test_transform_key_vision_layer_norm() {
        assert_eq!(
            transform_key("vision_model.encoder.layers.7.norm1.weight"),
            "vision.layers.7.norm1.weight"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.7.norm1.bias"),
            "vision.layers.7.norm1.bias"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.7.norm2.weight"),
            "vision.layers.7.norm2.weight"
        );
        assert_eq!(
            transform_key("vision_model.encoder.layers.7.norm2.bias"),
            "vision.layers.7.norm2.bias"
        );
    }

    #[test]
    fn test_transform_key_vision_layer_indices_preserved() {
        // Verify various layer indices
        for i in [0, 1, 11, 23] {
            assert_eq!(
                transform_key(&format!("vision_model.encoder.layers.{i}.attn.qkv.weight")),
                format!("vision.layers.{i}.attn.qkv.weight")
            );
        }
    }

    // ========================================
    // transform_key: Bridge MLP
    // ========================================

    #[test]
    fn test_transform_key_bridge_ln() {
        assert_eq!(transform_key("mlp1.0.weight"), "bridge.ln.weight");
        assert_eq!(transform_key("mlp1.0.bias"), "bridge.ln.bias");
    }

    #[test]
    fn test_transform_key_bridge_linear1() {
        assert_eq!(transform_key("mlp1.1.weight"), "bridge.linear1.weight");
        assert_eq!(transform_key("mlp1.1.bias"), "bridge.linear1.bias");
    }

    #[test]
    fn test_transform_key_bridge_linear2() {
        assert_eq!(transform_key("mlp1.3.weight"), "bridge.linear2.weight");
        assert_eq!(transform_key("mlp1.3.bias"), "bridge.linear2.bias");
    }

    // ========================================
    // transform_key: Language model
    // ========================================

    #[test]
    fn test_transform_key_lm_embedding() {
        assert_eq!(
            transform_key("language_model.model.embed_tokens.weight"),
            "lm.embedding.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_final_norm() {
        assert_eq!(
            transform_key("language_model.model.norm.weight"),
            "lm.final_norm.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_head() {
        assert_eq!(
            transform_key("language_model.lm_head.weight"),
            "lm.lm_head.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_self_attn() {
        assert_eq!(
            transform_key("language_model.model.layers.0.self_attn.q_proj.weight"),
            "lm.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.0.self_attn.k_proj.weight"),
            "lm.layers.0.self_attn.k_proj.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.0.self_attn.v_proj.weight"),
            "lm.layers.0.self_attn.v_proj.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.0.self_attn.o_proj.weight"),
            "lm.layers.0.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_qk_norm() {
        assert_eq!(
            transform_key("language_model.model.layers.15.self_attn.q_norm.weight"),
            "lm.layers.15.self_attn.q_norm.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.15.self_attn.k_norm.weight"),
            "lm.layers.15.self_attn.k_norm.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_mlp() {
        assert_eq!(
            transform_key("language_model.model.layers.2.mlp.gate_proj.weight"),
            "lm.layers.2.mlp.gate_proj.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.2.mlp.up_proj.weight"),
            "lm.layers.2.mlp.up_proj.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.2.mlp.down_proj.weight"),
            "lm.layers.2.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_layernorms() {
        assert_eq!(
            transform_key("language_model.model.layers.4.input_layernorm.weight"),
            "lm.layers.4.input_layernorm.weight"
        );
        assert_eq!(
            transform_key("language_model.model.layers.4.post_attention_layernorm.weight"),
            "lm.layers.4.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_transform_key_lm_layer_indices_preserved() {
        for i in [0, 1, 17, 35] {
            assert_eq!(
                transform_key(&format!(
                    "language_model.model.layers.{i}.self_attn.q_proj.weight"
                )),
                format!("lm.layers.{i}.self_attn.q_proj.weight")
            );
        }
    }

    // ========================================
    // needs_conv2d_transpose
    // ========================================

    #[test]
    fn test_needs_conv2d_transpose_patch_embedding() {
        // 4D patch conv weight should be transposed
        let weight = MxArray::from_float32(&[0.0; 1024 * 3 * 14 * 14], &[1024, 3, 14, 14])
            .expect("create array");
        assert!(needs_conv2d_transpose(
            "vision.embeddings.patch_embedding.weight",
            &weight
        ));
    }

    #[test]
    fn test_needs_conv2d_transpose_non_patch() {
        // Non-patch keys should not be transposed
        let weight = MxArray::from_float32(&[0.0; 16], &[2, 2, 2, 2]).expect("create array");
        assert!(!needs_conv2d_transpose(
            "vision.layers.0.attn.qkv.weight",
            &weight
        ));
    }

    #[test]
    fn test_needs_conv2d_transpose_non_4d() {
        // 2D weight with patch_embedding name should not be transposed
        let weight = MxArray::from_float32(&[0.0; 4], &[2, 2]).expect("create array");
        assert!(!needs_conv2d_transpose(
            "vision.embeddings.patch_embedding.weight",
            &weight
        ));
    }

    // ========================================
    // transpose_conv2d_weight
    // ========================================

    #[test]
    fn test_transpose_conv2d_weight_shape() {
        // [O, I, H, W] = [2, 3, 4, 5] → [O, H, W, I] = [2, 4, 5, 3]
        let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let weight = MxArray::from_float32(&data, &[2, 3, 4, 5]).expect("create array");
        let transposed = transpose_conv2d_weight(&weight).expect("transpose");

        let shape: Vec<i64> = transposed.shape().expect("shape").as_ref().to_vec();
        assert_eq!(shape, vec![2, 4, 5, 3]);
    }

    // ========================================
    // load_qianfan_ocr_weights: end-to-end
    // ========================================

    #[test]
    fn test_load_weights_key_transformation() {
        let mut weights = HashMap::new();

        // Add a representative weight from each category
        let small = MxArray::from_float32(&[1.0, 2.0], &[2]).expect("create array");

        weights.insert(
            "language_model.model.embed_tokens.weight".to_string(),
            small.clone(),
        );
        weights.insert("mlp1.0.weight".to_string(), small.clone());
        weights.insert(
            "vision_model.encoder.layers.0.attn.qkv.weight".to_string(),
            small.clone(),
        );
        weights.insert(
            "vision_model.embeddings.class_embedding".to_string(),
            small.clone(),
        );
        weights.insert("language_model.lm_head.weight".to_string(), small.clone());

        let result = load_qianfan_ocr_weights(weights).expect("load weights");

        assert!(result.contains_key("lm.embedding.weight"));
        assert!(result.contains_key("bridge.ln.weight"));
        assert!(result.contains_key("vision.layers.0.attn.qkv.weight"));
        assert!(result.contains_key("vision.embeddings.class_embedding"));
        assert!(result.contains_key("lm.lm_head.weight"));
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_load_weights_conv2d_transposed() {
        let mut weights = HashMap::new();

        // Patch embedding weight in PyTorch NCHW format
        let data: Vec<f32> = (0..2 * 3 * 4 * 4).map(|i| i as f32).collect();
        let patch_weight = MxArray::from_float32(&data, &[2, 3, 4, 4]).expect("create array");

        weights.insert(
            "vision_model.embeddings.patch_embedding.weight".to_string(),
            patch_weight,
        );

        let result = load_qianfan_ocr_weights(weights).expect("load weights");
        let transposed = result
            .get("vision.embeddings.patch_embedding.weight")
            .expect("patch conv weight present");

        // Should be transposed to NHWC: [2, 4, 4, 3]
        let shape: Vec<i64> = transposed.shape().expect("shape").as_ref().to_vec();
        assert_eq!(shape, vec![2, 4, 4, 3]);
    }

    #[test]
    fn test_load_weights_non_conv_untouched() {
        let mut weights = HashMap::new();

        // A regular 2D weight should pass through without transposition
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let linear_weight = MxArray::from_float32(&data, &[2, 2]).expect("create array");

        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.weight".to_string(),
            linear_weight,
        );

        let result = load_qianfan_ocr_weights(weights).expect("load weights");
        let w = result
            .get("lm.layers.0.self_attn.q_proj.weight")
            .expect("weight present");

        let shape: Vec<i64> = w.shape().expect("shape").as_ref().to_vec();
        assert_eq!(shape, vec![2, 2]);
    }

    // ========================================
    // Fallback behavior
    // ========================================

    #[test]
    fn test_transform_key_unknown_passthrough() {
        // Unknown keys should pass through unchanged
        assert_eq!(transform_key("some.unknown.key"), "some.unknown.key");
    }
}
